#!/usr/bin/env python3
"""
IaC / source secrets scanner v2 - cascade with a learned re-ranker.

  Stage 1  CANDIDATES  where could a secret be?
             * known-format rules from rules.toml (v1 rule database), with the
               capture group of the rule used as the secret (v1 used the whole match,
               key name included)
             * extra rules: PEM key header, Basic/Bearer Authorization, credentials
               in URLs, PASETO tokens, NATS NKey seeds
             * assignment extractor: key = value / "key": "value" / setKey("value") where the
               key (camelCase / snake_case aware) contains a secret-ish word; any value
               length >= 3 (v1 needed >= 10)
             * entropy dragnet (as in v1)
  Stage 2  FILTER      placeholders (CHANGEME, <your-key>, ...) and values that are
                       themselves safe references (var.x, ${{ secrets.X }}, ...) are dropped
  Stage 2b RE-RANK     a small gradient-boosted model (model_v2.json, pure-python inference,
                       no ML dependency at scan time) scores every surviving candidate from
                       ~110 features of the value, the key, the line and the file type.
                       Known-provider rules keep a score floor of their rule confidence.
  Verdict  score >= block -> BLOCK, >= warn -> WARN, else PASS
           (thresholds come from model_v2.json: chosen on out-of-fold dev predictions;
            BLOCK = high-precision point, WARN = max-F1 point; override with --block-at/--warn-at)

Same API as v1 (load_rules, scan_file) and same CLI / exit codes (1 if anything BLOCKs).

Usage:
  python3 scan.py <dir-or-file> [--rules rules.toml] [--json out.json]
                  [--block-at X] [--warn-at X] [--all-files] [--quiet]
"""
import argparse, base64, concurrent.futures, json, math, os, re, sys

try:
    import tomllib
except ImportError:                       # Python < 3.11
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "model_v2.json"


# ----------------------------------------------------------------- basics
def shannon(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def charset_of(s: str):
    if re.fullmatch(r"[0-9a-fA-F]+", s):
        return "hex", 3.0
    if re.fullmatch(r"[A-Za-z0-9+/=_\-\.]+", s):
        return "b64ish", 4.2
    return "mixed", 3.8


TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-\.!@#\$%]{16,}")


def entropy_candidates(line: str):
    for m in TOKEN_RE.finditer(line):
        tok = m.group(0).strip(".=")
        if len(tok) < 16:
            continue
        cs, thr = charset_of(tok)
        h = shannon(tok)
        if h >= thr:
            yield m.start() + (len(m.group(0)) - len(m.group(0).lstrip(".="))), tok, h, cs


# ----------------------------------------------------------------- rules
# rules that are context-based rather than a distinctive provider format: they get no score floor
WEAK_RULE_IDS = {
    "helm-secret-value", "docker-env-secret", "docker-build-arg-secret", "terraform-default-secret",
    "github-actions-hardcoded", "ansible-vault-password", "azure-storage-key-assign",
    "redis-connection-url", "postgres-connection-url", "mysql-connection-url", "jdbc-connection-string",
    "mongodb-connection-string", "aws-secret-access-key-assign", "aws-session-token-assign",
    "firebase-url", "kubernetes-secret-yaml", "hashicorp-tf-password", "jwt", "jwt-base64"}


def load_rules(path=None):
    """Returns (rules, ctx, kw_list, no_kw_rules) - same shape as v1; ctx also carries
    the v2 extras and the re-ranker model."""
    path = path or os.path.join(HERE, "rules.toml")
    if tomllib is None:
        sys.exit("ERROR: Python 3.11+ or `pip install tomli` required")
    with open(path, "rb") as f:
        data = tomllib.load(f)

    compiled = []
    for r in data.get("rules", []):
        rx_s = r.get("regex", "")
        if not rx_s:
            continue
        try:
            rx = re.compile(rx_s)
        except re.error:
            continue
        sg = int(r.get("secret_group", 0))
        if sg == 0 and rx.groups == 1:      # v2: a single capture group IS the secret
            sg = 1
        rid = r["id"]
        generic = ("generic" in rid or "generic" in r.get("tags", []) or rid in WEAK_RULE_IDS)
        compiled.append(dict(
            id=rid, rx=rx, conf=float(r.get("confidence", 0.8)),
            kws=[k.lower() for k in r.get("keywords", [])],
            entropy_min=float(r.get("entropy_min", 0.0)), sg=sg, generic=generic))

    raw = data.get("context", {})

    def build(items, flags=0):
        return re.compile("(" + "|".join(items) + ")", flags) if items else re.compile(r"a^")

    ctx = dict(
        safe_refs=build(raw.get("safe_references", [])),
        neg_shapes=[re.compile(p) for p in raw.get("negative_shapes", [])],
        placeholders=[re.compile(p, re.I) for p in raw.get("placeholder_patterns", [])],
        model=load_model(os.path.join(os.path.dirname(os.path.abspath(path)), MODEL_NAME)),
    )
    kw_index, no_kw = {}, []
    for rule in compiled:
        if not rule["kws"]:
            no_kw.append(rule)
        for kw in rule["kws"]:
            kw_index.setdefault(kw, []).append(rule)
    return compiled, ctx, list(kw_index.items()), no_kw


def load_model(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        m = json.load(f)
    if m.get("feature_names") != FEATURE_NAMES:
        sys.exit(f"{path}: feature list does not match scan.py (retrain with training/train_v2.py)")
    return m


def predict(model, feats):
    """pure-python gradient boosting inference -> probability"""
    raw = model["init"]
    lr = model["lr"]
    for t in model["trees"]:
        left, right, feat, thr, val = t["l"], t["r"], t["f"], t["t"], t["v"]
        i = 0
        while left[i] != -1:
            i = left[i] if feats[feat[i]] <= thr[i] else right[i]
        raw += lr * val[i]
    if raw < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-raw))


# ----------------------------------------------------------------- stage 1: candidates
SECRET_TOKENS = {
    "password", "passwd", "pwd", "pass", "passphrase", "secret", "token", "key", "apikey", "auth",
    "authorization", "credential", "credentials", "cred", "creds", "bearer", "private", "signing",
    "salt", "nonce", "seed", "hmac", "jwt", "api", "access", "secretkey", "privatekey", "sas", "pat"}
KEY_TOKEN_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def key_tokens(key: str):
    return [t.lower() for t in KEY_TOKEN_RE.findall(key)]


def secretish_key(key: str) -> bool:
    return bool(key) and any(t in SECRET_TOKENS for t in key_tokens(key))


ASSIGN_RE = re.compile(
    r'''(?P<key>[A-Za-z_][\w.\-]*)(?P<q>["']?)\s*(?P<sep>=>|:=|==|!=|=|:)\s*'''
    r'''(?P<val>"(?:[^"\\]|\\.){1,300}"|'(?:[^'\\]|\\.){1,300}'|[^\s,;"'(){}\[\]<>]{3,300})''')
CALL_RE = re.compile(
    r'''(?P<key>[A-Za-z_][\w.]*)\s*\(\s*(?P<val>"(?:[^"\\]|\\.){3,300}"|'(?:[^'\\]|\\.){3,300}')''')
KEY_BEFORE_RE = re.compile(r'''([A-Za-z_][\w.\-]*)["']?\s*(?:=>|:=|==|!=|=|:|\()\s*["'(]?\s*$''')

# extra rules: (id, regex, group, confidence, specific)
EXTRA_RULES = [
    ("pem-private-key-header", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----"), 0, 0.95, True),
    ("auth-basic", re.compile(r"(?i)\bBasic\s+([A-Za-z0-9+/]{8,}={0,2})(?![A-Za-z0-9+/=])"), 1, 0.6, False),
    ("auth-bearer", re.compile(r"(?i)\bBearer\s+([A-Za-z0-9._~+/\-]{12,}=*)"), 1, 0.6, False),
    ("url-credentials", re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s:/@'\"<>]{1,64}:([^\s@'\"/<>]{3,})@"), 1, 0.7, False),
    ("paseto-token", re.compile(r"\bv[1-4]\.(?:local|public)\.[A-Za-z0-9_\-]{20,}"), 0, 0.9, True),
    ("nats-nkey-seed", re.compile(r"\bS[ACNOUX][A-Z2-7]{50,}\b"), 0, 0.8, True),
]


def key_before(line, s):
    m = KEY_BEFORE_RE.search(line[:s])
    return m.group(1) if m else ""


def _unquote(val, start):
    """strip surrounding quotes from an assignment value; returns (value, start, end, quoted)"""
    if len(val) >= 2 and val[0] in "\"'" and val[-1] == val[0]:
        return val[1:-1], start + 1, start + len(val) - 1, True
    return val, start, start + len(val), False


def extract_candidates(line, ctx, kw_list, no_kw_rules):
    cands = {}      # (s, e) -> dict

    def add(s, e, value, key, src, conf, specific=False, quoted=False, sep=""):
        if not value or len(value) < 3:
            return
        c = cands.get((s, e))
        if c is None:
            cands[(s, e)] = dict(s=s, e=e, value=value, key=key, srcs={src}, conf=conf,
                                 specific=specific, quoted=quoted, sep=sep)
        else:
            c["srcs"].add(src)
            c["conf"] = max(c["conf"], conf)
            c["specific"] = c["specific"] or specific
            c["key"] = c["key"] or key
            c["sep"] = c["sep"] or sep

    low = line.lower()
    # v1 rule database
    seen, rules = set(), list(no_kw_rules)
    seen.update(r["id"] for r in rules)
    for kw, rs in kw_list:
        if kw in low:
            for r in rs:
                if r["id"] not in seen:
                    seen.add(r["id"])
                    rules.append(r)
    for r in rules:
        for m in r["rx"].finditer(line):
            sg = r["sg"] if (r["sg"] and m.lastindex and r["sg"] <= m.lastindex) else 0
            val = m.group(sg)
            if not val or len(val) < 4:
                continue
            if r["entropy_min"] > 0 and shannon(val) < r["entropy_min"]:
                continue
            s, e = m.span(sg)
            q = s > 0 and line[s - 1] in "\"'"
            add(s, e, val, key_before(line, s), "rule:" + r["id"], r["conf"],
                specific=not r["generic"], quoted=q)
    # extra rules
    for rid, rx, g, conf, spec in EXTRA_RULES:
        for m in rx.finditer(line):
            val = m.group(g)
            s, e = m.span(g)
            add(s, e, val, key_before(line, s), "rule:" + rid, conf, specific=spec)
    # assignment extractor
    for m in ASSIGN_RE.finditer(line):
        key = m.group("key")
        if not secretish_key(key):
            continue
        val, s, e, q = _unquote(m.group("val"), m.start("val"))
        add(s, e, val, key, "assign", 0.5, quoted=q, sep=m.group("sep"))
    for m in CALL_RE.finditer(line):
        key = m.group("key")
        if not secretish_key(key):
            continue
        val, s, e, q = _unquote(m.group("val"), m.start("val"))
        add(s, e, val, key, "call", 0.45, quoted=q, sep="(")
    # entropy dragnet
    if ("=" in line or ":" in line) or any(k in low for k in ("password", "secret", "token", "key")):
        for s, tok, h, cs in entropy_candidates(line):
            key = key_before(line, s)
            add(s, s + len(tok), tok, key, f"entropy:{cs}", 0.45 + min(0.25, (h - 3.0) * 0.08))
    return list(cands.values())


# ----------------------------------------------------------------- stage 2: filters
def is_placeholder(value, ctx):
    t = value.strip().strip("\"'")
    return any(p.search(t) for p in ctx["placeholders"])


# ----------------------------------------------------------------- stage 2b: features
EXT_GROUPS = {
    "py": "py", "js": "js", "jsx": "js", "ts": "js", "tsx": "js", "mjs": "js",
    "java": "jvm", "kt": "jvm", "scala": "jvm", "groovy": "jvm", "gradle": "jvm",
    "go": "go", "c": "c", "h": "c", "cc": "c", "cpp": "c", "hpp": "c", "m": "c",
    "cs": "cs", "php": "php", "rb": "rb", "sh": "sh", "bash": "sh", "ps1": "sh", "bat": "sh",
    "yml": "yaml", "yaml": "yaml", "json": "json", "xml": "markup", "html": "markup",
    "htm": "markup", "md": "doc", "txt": "doc", "rst": "doc", "tf": "tf", "tfvars": "tf",
    "hcl": "tf", "tfstate": "tf", "properties": "cfg", "ini": "cfg", "cfg": "cfg",
    "conf": "cfg", "toml": "cfg", "env": "cfg", "sql": "sql"}
EXT_ORDER = ["py", "js", "jvm", "go", "c", "cs", "php", "rb", "sh", "yaml", "json", "markup",
             "doc", "tf", "cfg", "sql"]

KEY_FLAGS = ["password", "pass", "pwd", "passwd", "secret", "token", "key", "api", "auth", "bearer",
             "credential", "cred", "private", "signing", "salt", "nonce", "seed", "access",
             "id", "name", "version", "url", "uri", "path", "file", "type", "label", "user",
             "host", "port", "test", "example", "default", "hash", "public", "client", "db",
             "mock", "fake", "dummy", "new", "old", "confirm", "reset", "forgot", "field", "input",
             "hint", "prompt", "policy", "validator", "format", "length", "min", "max", "size",
             "fingerprint", "thumbprint", "digest", "checksum", "etag", "cert", "certificate", "sha",
             "md5", "commit", "image", "ami", "arn", "uuid", "guid", "serial", "signature", "tls"]
SRC_FLAGS = ["assign", "call", "entropy", "pem", "auth", "url", "paseto", "nkey"]
VAL_WORDS = ["example", "your", "test", "dummy", "sample", "xxx", "password", "secret", "token",
             "key", "changeme", "null", "none", "true", "false", "default", "admin", "root", "foo", "bar"]

FEATURE_NAMES = (
    ["len", "loglen", "entropy", "entropy_norm", "has_lower", "has_upper", "has_digit", "has_special",
     "digit_ratio", "upper_ratio", "special_ratio", "is_hex", "is_b64ish", "is_alnum", "is_alpha_lower",
     "is_digits", "vowel_ratio", "max_run", "alpha_run", "case_humps", "n_dot", "n_underscore",
     "n_dash", "n_slash", "is_uuid", "is_date_or_version", "has_scheme", "has_space",
     "has_brace_or_dollar", "has_paren_or_bracket", "has_angle_or_pct", "has_at", "is_dotted_ident",
     "is_identifier", "b64_userpass", "b64_pad", "val_eq_key", "val_in_key", "key_in_val",
     "quoted", "key_len", "key_tokens", "key_camel", "key_empty", "key_secretish",
     "conf", "specific_rule", "generic_rule", "n_srcs", "sep_eq", "sep_colon", "sep_compare",
     "sep_call", "sep_arrow", "sep_none", "line_comment", "line_len_log", "pos_ratio",
     "preceded_by_dot", "followed_by_paren", "followed_by_bracket", "safe_ref_line", "safe_ref_value",
     "env_access", "member_access_line", "neg_shape", "n_cands_hint", "json_style", "markup_line",
     "val_is_word_phrase", "val_starts_upper_word", "digits_only_suffix", "key_ends_id",
     "key_ends_name", "line_has_http", "line_is_import_or_def",
     "prev_name_secretish", "prev_has_secret_word"]
    + ["key_" + k for k in KEY_FLAGS]
    + ["src_" + k for k in SRC_FLAGS]
    + ["ext_" + k for k in EXT_ORDER] + ["ext_other"]
    + ["val_has_" + w for w in VAL_WORDS])

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_DATEVER_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}|v?\d+(\.\d+){1,3}([-+.\w]*)?)$")
_ENV_RE = re.compile(r"getenv|environ|process\.env|ENV\[|System\.getProperty|\$\{?[A-Z_]{3,}\}?|%[A-Z_]+%")
_MEMBER_RE = re.compile(r"\b(?:self|this|config|cfg|conf|args|opts|options|params|request|req|settings|props|env)\.")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DOTTED_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+(\(\))?$")
_VOWELS = set("aeiouAEIOU")


def _b64_userpass(v):
    if len(v) < 8 or not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", v):
        return 0.0
    try:
        d = base64.b64decode(v + "=" * (-len(v) % 4), validate=False).decode("ascii")
    except Exception:
        return 0.0
    return 1.0 if ":" in d and d.isprintable() else 0.0


def _max_run(v):
    best = run = 1
    for a, b in zip(v, v[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def _alpha_run(v):
    best = run = 0
    for ch in v:
        run = run + 1 if ch.isalpha() else 0
        best = max(best, run)
    return best


def featurize(c, line, ext, ctx, n_cands, prevf=(0.0, 0.0)):
    v = c["value"]
    key = c["key"] or ""
    kl = key.lower()
    ktoks = key_tokens(key)
    n = len(v)
    letters = [ch for ch in v if ch.isalpha()]
    digits = sum(ch.isdigit() for ch in v)
    uppers = sum(ch.isupper() for ch in v)
    specials = sum(not ch.isalnum() for ch in v)
    h = shannon(v)
    vl = v.lower()
    s, e = c["s"], c["e"]
    stripped = line.lstrip()
    after = line[e:e + 3].lstrip("\"' ")
    srcs = c["srcs"]
    sep = c.get("sep", "")
    rest_after_sep = line[max(0, s - 1):min(len(line), e + 20)]
    f = {}
    f["len"] = n
    f["loglen"] = math.log(n + 1)
    f["entropy"] = h
    f["entropy_norm"] = h / math.log2(n) if n > 1 else 0.0
    f["has_lower"] = float(any(ch.islower() for ch in v))
    f["has_upper"] = float(uppers > 0)
    f["has_digit"] = float(digits > 0)
    f["has_special"] = float(specials > 0)
    f["digit_ratio"] = digits / n
    f["upper_ratio"] = uppers / n
    f["special_ratio"] = specials / n
    f["is_hex"] = float(bool(re.fullmatch(r"[0-9a-fA-F]+", v)))
    f["is_b64ish"] = float(bool(re.fullmatch(r"[A-Za-z0-9+/=_\-]+", v)))
    f["is_alnum"] = float(v.isalnum())
    f["is_alpha_lower"] = float(v.isalpha() and v.islower())
    f["is_digits"] = float(v.isdigit())
    f["vowel_ratio"] = (sum(ch in _VOWELS for ch in letters) / len(letters)) if letters else 0.0
    f["max_run"] = _max_run(v)
    f["alpha_run"] = _alpha_run(v)
    f["case_humps"] = sum(1 for a, b in zip(v, v[1:]) if a.islower() and b.isupper())
    f["n_dot"] = v.count(".")
    f["n_underscore"] = v.count("_")
    f["n_dash"] = v.count("-")
    f["n_slash"] = v.count("/")
    f["is_uuid"] = float(bool(_UUID_RE.match(v)))
    f["is_date_or_version"] = float(bool(_DATEVER_RE.match(v)))
    f["has_scheme"] = float("://" in v)
    f["has_space"] = float(" " in v)
    f["has_brace_or_dollar"] = float(any(ch in v for ch in "{}$"))
    f["has_paren_or_bracket"] = float(any(ch in v for ch in "()[]"))
    f["has_angle_or_pct"] = float(any(ch in v for ch in "<>%"))
    f["has_at"] = float("@" in v)
    f["is_dotted_ident"] = float(bool(_DOTTED_RE.match(v)))
    f["is_identifier"] = float(bool(_IDENT_RE.match(v)))
    f["b64_userpass"] = _b64_userpass(v)
    f["b64_pad"] = float(v.endswith("="))
    f["val_eq_key"] = float(bool(kl) and vl == kl)
    f["val_in_key"] = float(bool(kl) and vl in kl)
    f["key_in_val"] = float(bool(kl) and len(kl) > 3 and kl in vl)
    f["quoted"] = float(c["quoted"])
    f["key_len"] = len(key)
    f["key_tokens"] = len(ktoks)
    f["key_camel"] = float(bool(re.search(r"[a-z][A-Z]", key)))
    f["key_empty"] = float(not key)
    f["key_secretish"] = float(secretish_key(key))
    f["conf"] = c["conf"]
    f["specific_rule"] = float(c["specific"])
    f["generic_rule"] = float(any(x.startswith("rule:") for x in srcs) and not c["specific"])
    f["n_srcs"] = len(srcs)
    f["sep_eq"] = float(sep in ("=", ":="))
    f["sep_colon"] = float(sep == ":")
    f["sep_compare"] = float(sep in ("==", "!="))
    f["sep_call"] = float(sep == "(")
    f["sep_arrow"] = float(sep == "=>")
    f["sep_none"] = float(sep == "")
    f["line_comment"] = float(stripped.startswith(("#", "//", "/*", "*", "--", ";", "<!--", "%")))
    f["line_len_log"] = math.log(len(line) + 1)
    f["pos_ratio"] = s / max(1, len(line))
    f["preceded_by_dot"] = float(s > 0 and line[s - 1] == ".")
    f["followed_by_paren"] = float(after.startswith("("))
    f["followed_by_bracket"] = float(after.startswith("["))
    f["safe_ref_line"] = float(bool(ctx["safe_refs"].search(line)))
    f["safe_ref_value"] = float(bool(ctx["safe_refs"].search(rest_after_sep)))
    f["env_access"] = float(bool(_ENV_RE.search(line)))
    f["member_access_line"] = float(bool(_MEMBER_RE.search(line)))
    f["neg_shape"] = float(any(p.match(v) for p in ctx["neg_shapes"]))
    f["n_cands_hint"] = n_cands
    f["json_style"] = float(bool(re.search(r'"\s*:\s*"', line)))
    f["markup_line"] = float(stripped.startswith("<"))
    f["val_is_word_phrase"] = float(bool(re.fullmatch(r"[A-Za-z]+([ _\-][A-Za-z]+)+", v)))
    f["val_starts_upper_word"] = float(bool(re.match(r"[A-Z][a-z]{2,}", v)))
    f["digits_only_suffix"] = float(bool(re.fullmatch(r"[A-Za-z]+\d+", v)))
    f["key_ends_id"] = float(kl.endswith("id") and len(kl) > 2)
    f["key_ends_name"] = float(kl.endswith("name"))
    f["line_has_http"] = float("http" in line)
    f["line_is_import_or_def"] = float(bool(re.match(r"\s*(import|from|def|class|function|public|private|protected|using|#include)\b", line)))
    f["prev_name_secretish"], f["prev_has_secret_word"] = prevf
    tk = set(ktoks)
    for k in KEY_FLAGS:
        f["key_" + k] = float(k in tk or (len(k) > 4 and k in kl))
    for k in SRC_FLAGS:
        f["src_" + k] = float(any(x == k or x.startswith(k) or
                                  (k == "pem" and "pem-" in x) or (k == "auth" and "auth-" in x) or
                                  (k == "url" and "url-cred" in x) or (k == "paseto" and "paseto" in x) or
                                  (k == "nkey" and "nkey" in x) for x in srcs))
    g = EXT_GROUPS.get(ext, None)
    for k in EXT_ORDER:
        f["ext_" + k] = float(g == k)
    f["ext_other"] = float(g is None)
    for w in VAL_WORDS:
        f["val_has_" + w] = float(w in vl)
    return [f[name] for name in FEATURE_NAMES]


# ----------------------------------------------------------------- scoring one candidate
def ext_of(path):
    base = os.path.basename(path).lower()
    if base in (".env", "dockerfile", "jenkinsfile", "makefile"):
        return "env" if base == ".env" else "sh"
    return base.rsplit(".", 1)[-1] if "." in base else ""


_PREV_NAME_RE = re.compile(r"""(?:^|[\s\-])(?:name|key|id)\s*[:=]\s*["']?([A-Za-z_][\w.\-]*)""")


def prev_features(prev):
    """k8s / compose / GHA style:  `- name: DB_PASSWORD` on the line before `value: ...`"""
    if not prev:
        return (0.0, 0.0)
    m = _PREV_NAME_RE.search(prev)
    return (float(bool(m and secretish_key(m.group(1)))),
            float(any(t in SECRET_TOKENS for t in key_tokens(prev[:200]))))


def candidate_features(line, ext, ctx, kw_list, no_kw_rules, prev=""):
    """-> list of (cand, features)  (placeholders / reference values already removed)"""
    out = []
    prevf = prev_features(prev)
    cands = extract_candidates(line, ctx, kw_list, no_kw_rules)
    cands = [c for c in cands if not is_placeholder(c["value"], ctx)
             and not ctx["safe_refs"].fullmatch(c["value"])]
    for c in cands:
        out.append((c, featurize(c, line, ext, ctx, len(cands), prevf)))
    return out


def legacy_score(c, feats):
    """fallback when no model file is present: simple confidence adjustments"""
    i = FEATURE_NAMES.index
    s = c["conf"]
    s += 0.35 * feats[i("key_secretish")]
    s -= 0.8 * feats[i("safe_ref_line")]
    s -= 0.7 * feats[i("neg_shape")]
    s -= 0.3 * feats[i("line_comment")]
    return max(0.0, min(1.0, s))


def score_candidate(c, feats, ctx):
    m = ctx["model"]
    p = predict(m, feats) if m else legacy_score(c, feats)
    if c["specific"] and not feats[FEATURE_NAMES.index("neg_shape")]:
        p = max(p, c["conf"])                 # known provider format keeps its floor
    return p


# ----------------------------------------------------------------- driver
def scan_file(path, root, ctx, kw_list, no_kw_rules):
    findings = []
    try:
        with open(path, errors="replace") as f:
            lines = f.read().split("\n")
    except OSError:
        return findings
    rel = os.path.relpath(path, root)
    ext = ext_of(path)
    for i, line in enumerate(lines, 1):
        if len(line) > 2000 or len(line) < 6:
            continue
        for c, feats in candidate_features(line, ext, ctx, kw_list, no_kw_rules,
                                              lines[i - 2] if i >= 2 else ""):
            score = score_candidate(c, feats, ctx)
            if score <= 0.02:
                continue
            v = c["value"]
            findings.append(dict(
                file=rel, line=i, score=round(score, 4),
                reason=sorted(c["srcs"], key=lambda x: (not x.startswith("rule:"), x))[0],
                match=(v[:6] + "..." + v[-4:]) if len(v) > 14 else v))
    return findings


def walk(target, all_files):
    if os.path.isfile(target):
        yield target
        return
    skip = (".git", "node_modules")
    for dp, dn, fns in os.walk(target):
        dn[:] = [d for d in dn if d not in skip]
        for fn in fns:
            low = fn.lower()
            if all_files or low.rsplit(".", 1)[-1] in EXT_GROUPS or low in (
                    ".env", "dockerfile", "jenkinsfile", "vagrantfile", ".npmrc", ".pypirc", ".netrc",
                    "credentials", "inventory"):
                yield os.path.join(dp, fn)


_G = {}


def _init(rules_path):
    _G["args"] = load_rules(rules_path)


def _work(a):
    path, root = a
    rules, ctx, kw, nokw = _G["args"]
    return scan_file(path, root, ctx, kw, nokw)


def main():
    ap = argparse.ArgumentParser(description="IaC secrets scanner v2 (cascade + learned re-ranker)")
    ap.add_argument("target")
    ap.add_argument("--rules", default=None)
    ap.add_argument("--json")
    ap.add_argument("--block-at", type=float, default=None)
    ap.add_argument("--warn-at", type=float, default=None)
    ap.add_argument("--all-files", action="store_true", help="scan every file type (default: config/IaC/code extensions)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rules_path = args.rules or os.path.join(HERE, "rules.toml")
    rules, ctx, kw, nokw = load_rules(rules_path)
    th = (ctx["model"] or {}).get("thresholds", {})
    block_at = args.block_at if args.block_at is not None else th.get("block", 0.7)
    warn_at = args.warn_at if args.warn_at is not None else th.get("warn", 0.4)
    if not args.quiet:
        print(f"Loaded {len(rules)} rules; model={'yes' if ctx['model'] else 'NO (legacy scoring)'}; "
              f"block>={block_at} warn>={warn_at}")
    root = args.target if os.path.isdir(args.target) else os.path.dirname(args.target) or "."
    files = [(p, root) for p in walk(args.target, args.all_files)]
    all_f = []
    with concurrent.futures.ProcessPoolExecutor(initializer=_init, initargs=(rules_path,)) as ex:
        for res in ex.map(_work, files, chunksize=16):
            all_f.extend(res)
    for f in all_f:
        f["verdict"] = "BLOCK" if f["score"] >= block_at else "WARN" if f["score"] >= warn_at else "PASS"
    blocked = [f for f in all_f if f["verdict"] == "BLOCK"]
    warned = [f for f in all_f if f["verdict"] == "WARN"]
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(all_f, fh, indent=1)
    if not args.quiet:
        for f in sorted(all_f, key=lambda x: -x["score"]):
            if f["verdict"] != "PASS":
                print(f'[{f["verdict"]}] {f["file"]}:{f["line"]}  score={f["score"]}  ({f["reason"]})  {f["match"]}')
        print(f"\n{len(blocked)} BLOCK, {len(warned)} WARN, {len(all_f) - len(blocked) - len(warned)} low-score")
    sys.exit(1 if blocked else 0)


if __name__ == "__main__":
    main()
