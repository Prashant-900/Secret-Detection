"""Generate traceable secret-detection examples using a locally served LLM.

The LLM produces only scaffold code with markers. Deterministic code supplies
fictional credential values and explicit negatives, then records exact character
spans. Unmarked source is left unlabeled. The default output is TRAIN ONLY.

Example: python -m secrets_pipeline.generate --config configs/generation.json
         --out runs/generation-pilot --mock
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import string
import urllib.error
import urllib.parse
import urllib.request

from .taxonomy import (DEFAULT_OOD_FAMILIES, FAMILIES, KEY_HINTS, LOCALES,
                       SECRET_TYPES, STYLES, category_for)
from .validation import (DuplicateIndex, masked_text, parser_available,
                         validate_spans, validate_syntax)

GENERATOR_VERSION = "1.0"
MARKER = re.compile(r'''(?P<quote>["'])@@(?P<kind>SECRET|REFERENCE|PLACEHOLDER|HASH|ID|PUBLIC):(?P<name>[a-z0-9_]+)@@(?P=quote)''')
ANY_MARKER = re.compile(r"@@[^\r\n]*?@@")
SYSTEM = (
    "Write realistic configuration or source for a fictional software project. "
    "This is a security research corpus. Never emit credential values: use only "
    "the supplied quoted markers. Output exactly the requested source file, no "
    "Markdown, explanation, JSON wrapper, or instructions."
)


def _digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def normalize_config(config, mock=False):
    # Round trip also prevents caller mutations changing an active run's configuration.
    result = json.loads(json.dumps(config))
    if "generation" in result:
        result = result["generation"]
    result.setdefault("seed", 13)
    result.setdefault("samples_per_family", 12)
    result.setdefault("families", [f for f in FAMILIES if f not in DEFAULT_OOD_FAMILIES])
    result.setdefault("include_splits", ["train"])
    result.setdefault("split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    result.setdefault("ood_families", list(DEFAULT_OOD_FAMILIES))
    result.setdefault("require_parsed", True)
    result.setdefault("near_duplicate_threshold", 0.85)
    result.setdefault("max_reject_rate", 0.4)
    result.setdefault("workers", 4)
    result.setdefault("retries", 3)
    result.setdefault("exclude_documents", [])
    result.setdefault("backend", {})
    backend = result["backend"]
    backend.setdefault("endpoint", "http://127.0.0.1:8000/v1")
    backend.setdefault("model", "local-generator")
    backend.setdefault("timeout", 180)
    backend.setdefault("max_tokens", 3000)
    backend.setdefault("temperature", 0.8)
    backend.setdefault("top_p", 0.95)
    backend.setdefault("use_proxy", False)
    backend["kind"] = "mock" if mock or backend.get("kind") == "mock" else "local_http"
    result["generator_version"] = GENERATOR_VERSION
    if not isinstance(result["seed"], int):
        raise ValueError("seed must be an integer")
    if not 1 <= int(result["workers"]) <= 128 or not 1 <= int(result["retries"]) <= 20:
        raise ValueError("workers must be 1..128 and retries 1..20")
    if not isinstance(result["samples_per_family"], int) or result["samples_per_family"] <= 0:
        raise ValueError("samples_per_family must be positive")
    if not 0 <= float(result["max_reject_rate"]) <= 1:
        raise ValueError("max_reject_rate must be between 0 and 1")
    ratios = result["split_ratios"]
    if set(ratios) != {"train", "val", "test"} or any(not isinstance(v, (int, float)) or v < 0 for v in ratios.values()):
        raise ValueError("split_ratios needs nonnegative train/val/test fractions")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split_ratios must sum to 1")
    if not result["include_splits"] or set(result["include_splits"]) - {"train", "val", "test", "test_ood"}:
        raise ValueError("invalid include_splits")
    if set(result["families"]) - set(FAMILIES) or set(result["ood_families"]) - set(FAMILIES):
        raise ValueError("unknown family")
    endpoint = urllib.parse.urlsplit(backend["endpoint"])
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password:
        raise ValueError("endpoint must be an HTTP(S) server URL without embedded credentials")
    if endpoint.query or endpoint.fragment:
        raise ValueError("endpoint cannot contain a query or fragment")
    if "api_key" in backend:
        raise ValueError("Use api_key_env, not a literal API key in configuration")
    return result


def topic_split(family, topic, config):
    if family in config["ood_families"]:
        return "test_ood"
    number = int(_digest(f"{config['seed']}:{family}:{topic}")[:16], 16) / float(16 ** 16)
    train = config["split_ratios"]["train"]
    return "train" if number < train else "val" if number < train + config["split_ratios"]["val"] else "test"


def build_plan(config):
    """Freeze family/topic groups before contacting a generator.

    ``cells`` is a quota list: family, count, optional secret_type, locale,
    key_hint, style, topic. Audit-produced cells can be copied here directly.
    Values omitted in a cell cycle over the full axis deterministically.
    """
    cells = config.get("cells") or [{"family": family, "count": config["samples_per_family"]}
                                    for family in config["families"]]
    plan = []
    for cell_index, cell in enumerate(cells):
        family = cell["family"]
        if family not in FAMILIES:
            raise ValueError(f"unknown family: {family}")
        count = cell.get("count", config["samples_per_family"])
        if not isinstance(count, int) or count < 0:
            raise ValueError("cell count must be a nonnegative integer")
        topics = [cell["topic"]] if cell.get("topic") else FAMILIES[family]["topics"]
        topics = [topic for topic in topics if topic_split(family, topic, config) in config["include_splits"]]
        if not topics:
            continue
        for index in range(count):
            topic = topics[index % len(topics)]
            item = {
                "family": family, "topic": topic,
                "language": FAMILIES[family]["language"], "path": FAMILIES[family]["path"],
                "secret_type": cell.get("secret_type", SECRET_TYPES[index % len(SECRET_TYPES)]),
                "locale": cell.get("locale", LOCALES[index % len(LOCALES)]),
                "key_hint": cell.get("key_hint", KEY_HINTS[index % len(KEY_HINTS)]),
                "style": cell.get("style", STYLES[index % len(STYLES)]),
                "sample": index, "cell": cell_index, "split": topic_split(family, topic, config),
            }
            for key, choices in (("secret_type", SECRET_TYPES), ("locale", LOCALES), ("key_hint", KEY_HINTS), ("style", STYLES)):
                if item[key] not in choices:
                    raise ValueError(f"invalid generation cell {key}: {item[key]}")
            if family in config["ood_families"] and item["split"] != "test_ood":
                raise ValueError("OOD family cannot enter training or validation")
            item["id"] = f"{family}-{_digest([config['seed'], item])[:20]}"
            item["group_id"] = f"generated:{family}:{_digest(topic)[:16]}"
            item["seed"] = int(_digest([config["seed"], item["id"]])[:15], 16)
            plan.append(item)
    if not plan:
        raise ValueError("No eligible plan cells; inspect include_splits and reserved OOD families")
    return plan


def make_prompt(item, attempt=0):
    return (
        f"Create one {item['language']} file named {item['path']} for {item['family']}: {item['topic']}.\n"
        f"Style: {item['style']}. Key naming: {item['key_hint']}. Use natural {item['locale']} variable names/comments "
        "where the language permits, while keeping identifiers idiomatic. Produce 20-100 useful lines.\n"
        f"The primary authentication slot holds a {item['secret_type']}; place it in realistic authentication context.\n"
        'Use exactly one quoted "@@SECRET:primary@@" marker for that slot. Also include one each of '
        '"@@REFERENCE:reference@@", "@@PLACEHOLDER:placeholder@@", "@@HASH:checksum@@", '
        '"@@ID:identifier@@", and "@@PUBLIC:public_key@@". Markers must occupy an entire quoted string, '
        "never only part of a URL or another literal.\n"
        "REFERENCE means environment/secret-store lookup; PLACEHOLDER is a deliberately unset credential. "
        "HASH is a build checksum; ID is a non-secret resource identifier; PUBLIC is a publishable client key. "
        "Put these on semantically suitable fields. For Kubernetes put the primary slot on an env entry's value "
        "and the REFERENCE slot on valueFrom; for JSON/TOML/INI use an application that explicitly supports "
        "environment substitutions. For XML use element text markers, never attributes.\n"
        "Include realistic benign booleans, names, numeric settings, nested structures, SDK calls or comments. "
        "Vary layout and naming, not random filler. Do not mention synthetic data, labels, or this task in source. "
        "Do not invent any credential-like value or high-entropy literal outside the six markers. All servers "
        "must use localhost or reserved example.test/example.invalid domains. No network calls are executed.\n"
        f"Diversity sample {item['sample']}, variation {attempt}. Return only the file."
    )


def _strip_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[\w+-]*\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text + "\n"


def scaffold_gate(text):
    matches = list(MARKER.finditer(text))
    counts = Counter(match["kind"] for match in matches)
    required = {"SECRET", "REFERENCE", "PLACEHOLDER", "HASH", "ID", "PUBLIC"}
    if set(counts) != required or any(counts[kind] != 1 for kind in required):
        raise ValueError("marker_contract")
    if len(ANY_MARKER.findall(text)) != len(matches):
        raise ValueError("invalid_marker")
    if len(text) > 100_000 or len(text.splitlines()) > 1500:
        raise ValueError("scaffold_too_large")
    scrubbed = MARKER.sub('"__SLOT__"', text)
    if re.search(r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|sk_live_[A-Za-z0-9]{16,}|-----BEGIN [A-Z ]*PRIVATE KEY-----", scrubbed):
        raise ValueError("unmarked_provider_secret")
    for match in re.finditer(r'''(?i)\b(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret)\b["']?\s*[:=]\s*(["'])([^\r\n]*?)\1''', scrubbed):
        value = match[2]
        if value != "__SLOT__" and not re.search(r"\$\{|\{\{|getenv|environ|process\.env", value):
            raise ValueError("unmarked_sensitive_assignment")
    for match in re.finditer(r'''["']([A-Za-z0-9+/=_-]{24,})["']''', scrubbed):
        value = match[1]
        entropy = -sum((n / len(value)) * math.log2(n / len(value)) for n in Counter(value).values())
        if entropy > 4.2:
            raise ValueError("unmarked_high_entropy_literal")
    return matches


def secret_value(secret_type, seed):
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits
    token = lambda n: "".join(rng.choice(alphabet) for _ in range(n))
    if secret_type == "password_low_entropy":
        return rng.choice(("garden7", "winter2026", "alpine123", "bluebird", "maple42", "monsoon9"))
    if secret_type == "password_random":
        return token(18) + "!"
    if secret_type == "opaque_token":
        return token(rng.choice((20, 24, 32, 48)))
    if secret_type == "api_key_hex":
        return "".join(rng.choice("0123456789abcdef") for _ in range(40))
    if secret_type == "github_token":
        return "ghp_" + token(36)
    if secret_type == "aws_access_key":
        return "AKIA" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(16))
    if secret_type == "aws_secret_key":
        return "".join(rng.choice(alphabet + "+/") for _ in range(40))
    if secret_type == "jwt":
        encode = lambda b: base64.urlsafe_b64encode(b).decode().rstrip("=")
        header = encode(b'{"alg":"HS256","typ":"JWT"}')
        payload = encode(json.dumps({"sub": str(seed % 10000), "iss": "example.invalid", "exp": 946684800}, separators=(",", ":")).encode())
        signature = encode(bytes(rng.randrange(256) for _ in range(32)))
        return f"{header}.{payload}.{signature}"
    if secret_type == "connection_string":
        return f"postgresql://service:{token(16)}@db.example.invalid:5432/application"
    if secret_type == "base64_secret":
        return base64.b64encode((token(24) + ":" + token(16)).encode()).decode()
    if secret_type == "stripe_secret":
        return "sk_live_" + token(24)
    if secret_type == "slack_bot_token":
        digits = lambda n: "".join(rng.choice(string.digits) for _ in range(n))
        return "xoxb-" + digits(11) + "-" + digits(12) + "-" + token(24)
    if secret_type == "gitlab_token":
        return "glpat-" + token(20)
    if secret_type == "huggingface_token":
        return "hf_" + token(34)
    if secret_type == "npm_token":
        return "npm_" + token(36)
    if secret_type == "google_api_key":
        return "AIza" + token(35)
    if secret_type == "azure_client_secret":
        return token(3) + "~" + token(36)
    if secret_type == "basic_auth":
        return "Basic " + base64.b64encode(("service:" + token(20)).encode()).decode()
    if secret_type == "private_key_pem":
        # Deterministic valid private keys, never installed anywhere. Vary key
        # family and serialization so the corpus does not teach one PEM body size.
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization
        material = hashlib.sha256(str(seed).encode()).digest()
        if seed % 3 == 0:
            key = Ed25519PrivateKey.from_private_bytes(material)
        else:
            order = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
            key = ec.derive_private_key(int.from_bytes(material, "big") % (order - 1) + 1, ec.SECP256R1())
        form = serialization.PrivateFormat.TraditionalOpenSSL if seed % 3 == 2 else serialization.PrivateFormat.PKCS8
        return key.private_bytes(serialization.Encoding.PEM, form,
                                 serialization.NoEncryption()).decode().rstrip("\n")
    raise ValueError("unknown secret type")


def _reference(item, key="DB_PASSWORD"):
    language, family = item["language"], item["family"]
    if language == "python":
        return f'os.environ["{key}"]', True
    if language in ("javascript", "typescript"):
        return f"process.env.{key}", True
    if language == "java":
        return f'System.getenv("{key}")', True
    if language == "go":
        return f'os.Getenv("{key}")', True
    if language == "groovy":
        return f'env.{key}', True
    if language == "hcl":
        return f"var.{key.lower()}", True
    if family == "kubernetes":
        return '{secretKeyRef: {name: application-auth, key: password}}', True
    if family == "github_actions":
        return "${{ secrets." + key + " }}", False
    if family in ("ansible", "helm"):
        return "{{ lookup('env', '" + key + "') }}" if family == "ansible" else "{{ .Values.existingSecret }}", False
    return "${" + key + "}", False


def _literal(value, item, scaffold, position):
    """Return (replacement source, labeled payload start, payload end)."""
    language = item["language"]
    if language == "xml":
        encoded = html.escape(value, quote=False)
        return encoded, 0, len(encoded)
    if "\n" in value and language == "python":
        rendered = '"""' + value + '"""'
        return rendered, 3, len(rendered) - 3
    if "\n" in value and language == "hcl":
        rendered = "<<-CREDENTIAL\n" + value + "\nCREDENTIAL"
        return rendered, len("<<-CREDENTIAL\n"), len(rendered) - len("\nCREDENTIAL")
    if "\n" in value and language == "yaml":
        line_prefix = scaffold[scaffold.rfind("\n", 0, position) + 1:position]
        if re.fullmatch(r"[ \t]*[\w.-]+\s*:\s*", line_prefix):
            indent = len(line_prefix) - len(line_prefix.lstrip()) + 2
            body = "\n".join(" " * indent + line for line in value.splitlines())
            return "|\n" + body, 2 + indent, 2 + len(body)
    rendered = json.dumps(value, ensure_ascii=False)
    return rendered, 1, len(rendered) - 1


def fill_variants(scaffold, item, config, *, backend_kind="mock", attempt=0):
    markers = scaffold_gate(scaffold)
    variants = []
    for variant in ("secret", "reference", "placeholder"):
        parts, spans, length, cursor = [], [], 0, 0
        for marker in markers:
            kind = marker["kind"]
            value_seed = int(_digest([item["seed"], marker["name"]])[:15], 16)
            label, category, secret_type = 0, "", ""
            role = variant.upper() if kind == "SECRET" else kind
            if role == "SECRET":
                value = secret_value(item["secret_type"], value_seed)
                label, category, secret_type = 1, category_for(item["secret_type"]), item["secret_type"]
            elif role == "REFERENCE":
                value, expression = _reference(item)
                category = "reference"
            elif role == "PLACEHOLDER":
                value = ("CHANGE_ME", "<your-credential>", "REPLACE_WITH_VALUE")[value_seed % 3]
                category = "placeholder"
            elif role == "HASH":
                value = _digest(f"build:{value_seed}")
                category = "hash"
            elif role == "ID":
                h = _digest(f"resource:{value_seed}")
                value = f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
                category = "identifier"
            else:
                value = "pk_live_" + _digest(f"public:{value_seed}")[:24]
                category = "public_key"
            start = marker.start()
            prefix = scaffold[cursor:start]
            if role == "REFERENCE" and expression:
                rendered, payload_start, payload_end = value, 0, len(value)
                if item["family"] == "kubernetes":
                    # Native Kubernetes env reference needs the valueFrom key.
                    key_match = re.search(r"\bvalue\s*:\s*$", prefix)
                    if key_match:
                        prefix = prefix[:key_match.start()] + "valueFrom: "
            else:
                rendered, payload_start, payload_end = _literal(value, item, scaffold, start)
            parts.append(prefix)
            length += len(prefix)
            span_start, span_end = length + payload_start, length + payload_end
            parts.append(rendered)
            length += len(rendered)
            spans.append({"id": f"{item['id']}:{variant}:{marker['name']}", "start": span_start, "end": span_end,
                          "label": label, "category": category, "secret_type": secret_type,
                          "label_origin": "deterministic_injection", "slot": marker["name"],
                          "surface_sha256": _digest(rendered[payload_start:payload_end])})
            cursor = marker.end()
        parts.append(scaffold[cursor:])
        text = "".join(parts)
        syntax = validate_syntax(text, item["language"], config["require_parsed"])
        if not syntax.valid:
            raise ValueError(syntax.reason)
        doc = {
            "id": f"generated:{item['id']}:{variant}", "repo_id": f"generated/{item['family']}/{_digest(item['topic'])[:16]}",
            "group_id": item["group_id"], "split": item["split"], "source": "generated",
            "path": item["path"], "language": item["language"], "text": text, "spans": spans,
            "metadata": {"generator_version": GENERATOR_VERSION, "plan_id": item["id"],
                         "family": item["family"], "topic": item["topic"], "locale": item["locale"],
                         "key_hint": item["key_hint"], "style": item["style"], "variant": variant,
                         "counterfactual_group": item["id"], "seed": item["seed"], "attempt": attempt,
                         "backend": backend_kind, "model": config["backend"]["model"],
                         "text_sha256": _digest(text), "scaffold_sha256": _digest(scaffold),
                         "prompt_sha256": _digest(make_prompt(item, attempt)), "validation": syntax.to_dict(),
                         "semantic_validation": "not_performed", "unmarked_content": "unlabeled",
                         "credential_status": "fictional_never_provisioned", "headline_test": False},
        }
        validate_spans(doc)
        variants.append(doc)
    # The primary slot changes its label, while the five explicit negatives persist.
    if sum(s["label"] for s in variants[0]["spans"]) != 1 or any(s["label"] for d in variants[1:] for s in d["spans"]):
        raise ValueError("counterfactual_label_contract")
    return variants


def mock_scaffold(item, attempt=0):
    """A parser-valid integration fixture, never a quality substitute for an LLM."""
    language, family = item["language"], item["family"]
    key = "password" if item["key_hint"] == "explicit" else "setting_value" if item["key_hint"] == "neutral" else "display_option"
    fields = [(key, "SECRET", "primary"), ("credential_reference", "REFERENCE", "reference"),
              ("pending_password", "PLACEHOLDER", "placeholder"), ("build_checksum", "HASH", "checksum"),
              ("resource_id", "ID", "identifier"), ("publishable_key", "PUBLIC", "public_key")]
    # Different fixture nesting/field layouts make the smoke path exercise real dedup.
    rotation = (item["sample"] + attempt) % len(fields)
    fields = fields[rotation:] + fields[:rotation]
    values = {name: f"@@{kind}:{slot}@@" for name, kind, slot in fields}
    service = f"application_{item['sample']}_{attempt}"
    if language == "json":
        values.update({"host": "db.example.invalid", "port": 5432, "enabled": True})
        return json.dumps({service: values}, indent=2) + "\n"
    if language == "python":
        pairs = "\n".join(f"    {name!r}: {value!r}," for name, value in values.items())
        return f"import os\n\ndef configure_{service}():\n    return {{\n{pairs}\n    }}\n"
    if language == "toml":
        return f"[{service}]\n" + "\n".join(f'{name} = "{value}"' for name, value in values.items()) + '\nenabled = true\n'
    if language == "ini":
        return f"[{service}]\n" + "\n".join(f'{name} = "{value}"' for name, value in values.items()) + '\nport = 5432\n'
    if language == "dotenv":
        return "\n".join(f'{name.upper()}="{value}"' for name, value in values.items()) + f'\nAPPLICATION_NAME="{service}"\n'
    if language == "xml":
        return '<application name="' + service + '">\n' + "\n".join(f'  <{name}>"{value}"</{name}>' for name, value in values.items()) + '\n</application>\n'
    if language == "hcl":
        return 'locals {\n' + f'  application_name = "{service}"\n' + "\n".join(f'  {name} = "{value}"' for name, value in values.items()) + '\n}\n'
    if language == "yaml":
        if family == "kubernetes":
            rows = []
            for name, kind, slot in fields:
                value_key = "valueFrom" if kind == "REFERENCE" else "value"
                rows.append(f'        - name: {name.upper()}\n          {value_key}: "@@{kind}:{slot}@@"')
            return 'apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: application\nspec:\n  template:\n    spec:\n      containers:\n      - name: web\n        image: application:1.0\n        env:\n' + "\n".join(rows) + '\n'
        return f'{service}:\n' + "\n".join(f'  {name}: "{value}"' for name, value in values.items()) + '\n  port: 5432\n'
    if language in {"javascript", "typescript"}:
        return 'export const ' + service + ' = {\n' + "\n".join(f'  {name}: "{value}",' for name, value in values.items()) + '\n};\n'
    if language == "java":
        return 'class Configuration {\n' + "\n".join(f'  String {name} = "{value}";' for name, value in values.items()) + '\n}\n'
    if language == "go":
        return 'package configuration\nimport "os"\nvar ' + service + ' = map[string]string{\n' + "\n".join(f'  "{name}": "{value}",' for name, value in values.items()) + '\n}\n'
    if language == "groovy":
        return 'def ' + service + ' = [\n' + "\n".join(f'  {name}: "{value}",' for name, value in values.items()) + '\n]\n'
    raise ValueError("unsupported_mock_language")


class LocalBackend:
    def __init__(self, config):
        self.config = config
        endpoint = config["endpoint"].rstrip("/")
        self.url = endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"

    def complete(self, item, attempt):
        body = {"model": self.config["model"], "messages": [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": make_prompt(item, attempt)}],
                "max_tokens": int(self.config["max_tokens"]), "temperature": float(self.config["temperature"]),
                "top_p": float(self.config["top_p"]), "seed": (item["seed"] + attempt) % (2 ** 31)}
        headers = {"Content-Type": "application/json"}
        if self.config.get("api_key_env"):
            key = os.environ.get(self.config["api_key_env"])
            if not key:
                raise ValueError("missing_generator_api_key_environment_variable")
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            class NoRedirects(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    raise ValueError("generator_redirect_not_allowed")
            handlers = [NoRedirects()]
            if not self.config.get("use_proxy", False):
                handlers.append(urllib.request.ProxyHandler({}))
            opener = urllib.request.build_opener(*handlers)
            with opener.open(request, timeout=float(self.config["timeout"])) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("generator_response_too_large")
            payload = json.loads(raw)
            choice = payload["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("generator_response_truncated")
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("generator_content_not_text")
            return content
        except urllib.error.HTTPError as exc:
            raise ValueError(f"generator_http_{exc.code}") from None
        except urllib.error.URLError:
            raise ValueError("generator_connection_error") from None
        except (KeyError, IndexError, json.JSONDecodeError):
            raise ValueError("generator_response_schema_error") from None


def _load_exclusions(paths, index):
    manifest = []
    for path in paths:
        path = Path(path)
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                if not raw.strip():
                    continue
                doc = json.loads(raw)
                index.add(masked_text(doc), "excluded:" + doc["id"], doc["split"])
                count += 1
        manifest.append({"path": str(path.resolve()), "sha256": digest.hexdigest(), "documents": count})
    return manifest


def generate_dataset(config, out, *, mock=False, resume=True):
    config = normalize_config(config, mock)
    out = Path(out)
    plan = build_plan(config)
    if config["require_parsed"]:
        missing = sorted({item["language"] for item in plan if not parser_available(item["language"])})
        if missing:
            raise ValueError("Missing syntax parsers for: " + ", ".join(missing) +
                             ". Install generation requirements or explicitly set require_parsed=false.")
    if any(item["secret_type"] == "private_key_pem" for item in plan):
        import importlib.util
        if importlib.util.find_spec("cryptography") is None:
            raise ValueError("private_key_pem generation requires cryptography")
    index = DuplicateIndex(float(config["near_duplicate_threshold"]))
    exclusions = _load_exclusions(config["exclude_documents"], index)
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("generate.py", "validation.py", "taxonomy.py")}
    frozen = {"config": config, "config_sha256": _digest(config), "source_sha256": source_hashes,
              "exclusions": exclusions, "items": plan}
    frozen["plan_sha256"] = _digest(frozen)
    plan_path = out / "plan.json"
    if plan_path.exists():
        if not resume:
            raise FileExistsError("Output contains a frozen plan; choose a new output directory")
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        if saved != frozen:
            raise ValueError("Frozen plan/config/exclusion hashes changed; choose a new output directory")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Output is nonempty without a frozen plan; choose an empty directory")
        _atomic_json(plan_path, frozen)
    shard_dir = out / "shards"
    shard_dir.mkdir(exist_ok=True)
    backend_kind = config["backend"]["kind"]
    backend = None if backend_kind == "mock" else LocalBackend(config["backend"])

    def propose(item, first_attempt=0):
        reasons = []
        for attempt in range(first_attempt, int(config["retries"])):
            try:
                raw = mock_scaffold(item, attempt) if backend_kind == "mock" else backend.complete(item, attempt)
                scaffold = _strip_fences(raw)
                documents = fill_variants(scaffold, item, config, backend_kind=backend_kind, attempt=attempt)
                return {"documents": documents, "scaffold": scaffold, "attempt": attempt, "reasons": reasons}
            except (ValueError, TimeoutError, OSError) as exc:
                # Avoid serializing exception text from external libraries that can
                # contain source content. Our gates have simple machine reason codes.
                message = str(exc)
                reason = message if re.fullmatch(r"[a-zA-Z0-9_:.-]{1,100}", message) else type(exc).__name__
                reasons.append({"attempt": attempt, "reason": reason})
        return {"documents": [], "attempt": int(config["retries"]) - 1, "reasons": reasons}

    cached = {}
    for item in plan:
        shard_path = shard_dir / (item["id"] + ".json")
        if shard_path.exists():
            row = json.loads(shard_path.read_text(encoding="utf-8"))
            check = dict(row)
            expected = check.pop("sha256", None)
            if expected != _digest(check) or row.get("plan_sha256") != frozen["plan_sha256"]:
                raise ValueError("Resume shard failed integrity check: " + item["id"])
            for document in row.get("documents", []):
                validate_spans(document)
                if document["metadata"]["text_sha256"] != _digest(document["text"]):
                    raise ValueError("Resume document hash mismatch")
            cached[item["id"]] = row
    # Dispatch generation in parallel, but consume in the frozen order. Admission to
    # the dedup bank is therefore deterministic even if server requests finish out of order.
    counts, rejected, split_counts, cells = Counter(), Counter(), Counter(), Counter()
    document_count = positive_spans = negative_spans = unvalidated_documents = 0
    with ThreadPoolExecutor(max_workers=int(config["workers"])) as pool:
        pending = {item["id"]: pool.submit(propose, item) for item in plan if item["id"] not in cached}
        for item in plan:
            shard_path = shard_dir / (item["id"] + ".json")
            row = cached.get(item["id"])
            if row is None:
                proposal = pending.pop(item["id"]).result()
                reasons = list(proposal["reasons"])
                while proposal["documents"]:
                    duplicate = next((match for document in proposal["documents"]
                                      if (match := index.match(masked_text(document), item["id"], item["split"]))), None)
                    if not duplicate:
                        break
                    reasons.append({"attempt": proposal["attempt"], "reason": "cross_split_duplicate" if duplicate["cross_split"] else "near_duplicate",
                                    "match": duplicate})
                    proposal = propose(item, proposal["attempt"] + 1)
                    reasons.extend(proposal["reasons"])
                row = {"plan_id": item["id"], "plan_sha256": frozen["plan_sha256"],
                       "status": "accepted" if proposal["documents"] else "rejected",
                       "documents": proposal["documents"], "reasons": reasons}
                if proposal["documents"]:
                    row["scaffold"] = proposal["scaffold"]
                row["sha256"] = _digest(row)
                _atomic_json(shard_path, row)
            if row["documents"]:
                duplicate = next((match for document in row["documents"]
                                  if (match := index.match(masked_text(document), item["id"], item["split"]))), None)
                if duplicate:
                    raise ValueError("Resume dataset admission order/integrity mismatch")
                for document in row["documents"]:
                    index.add(masked_text(document), item["id"], item["split"])
                document_count += len(row["documents"])
                positive_spans += sum(span["label"] for doc in row["documents"] for span in doc["spans"])
                negative_spans += sum(span["label"] == 0 for doc in row["documents"] for span in doc["spans"])
                unvalidated_documents += sum(not doc["metadata"]["validation"]["parsed"] for doc in row["documents"])
                counts[item["family"]] += 1
                split_counts[item["split"]] += len(row["documents"])
                cells[(item["family"], item["secret_type"], item["locale"], item["key_hint"])] += 1
            else:
                rejected[item["family"]] += 1
    documents_path = out / "documents.jsonl"
    temporary = documents_path.with_suffix(".jsonl.tmp")
    dataset_hash = hashlib.sha256()
    with temporary.open("wb") as handle:
        for item in plan:
            row = json.loads((shard_dir / (item["id"] + ".json")).read_text(encoding="utf-8"))
            for document in row["documents"]:
                raw = (json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                dataset_hash.update(raw)
                handle.write(raw)
    os.replace(temporary, documents_path)
    reason_counts = Counter()
    for item in plan:
        row = json.loads((shard_dir / (item["id"] + ".json")).read_text(encoding="utf-8"))
        reason_counts.update(reason["reason"] for reason in row["reasons"])
    reject_rate = sum(rejected.values()) / len(plan)
    report = {
        "status": "complete" if document_count and reject_rate <= config["max_reject_rate"] else "needs_review",
        "backend": backend_kind, "mock_data": backend_kind == "mock",
        "planned_groups": len(plan), "accepted_groups": sum(counts.values()), "rejected_groups": sum(rejected.values()),
        "documents": document_count, "accepted_by_family": dict(counts), "rejected_by_family": dict(rejected),
        "rejection_reasons": dict(reason_counts), "reject_rate": reject_rate, "documents_by_split": dict(split_counts),
        "coverage_cells": [{"family": key[0], "secret_type": key[1], "locale": key[2], "key_hint": key[3], "accepted_groups": count}
                           for key, count in sorted(cells.items())],
        "positive_spans": positive_spans, "negative_spans": negative_spans,
        "unvalidated_documents": unvalidated_documents,
        "plan_sha256": frozen["plan_sha256"], "documents_sha256": dataset_hash.hexdigest(),
        "documents_path": str(documents_path.resolve()), "exclusions": exclusions,
        "limitations": ["Syntax validation does not establish runtime/semantic validity.",
                        "Only injected spans are labeled; unmarked source is not automatically benign.",
                        "Generated held-out examples are synthetic stress tests, never the real-repository headline.",
                        "The local server may not implement deterministic sampling despite a supplied seed."],
    }
    _atomic_json(out / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = generate_dataset(config, args.out, mock=args.mock, resume=not args.no_resume)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
