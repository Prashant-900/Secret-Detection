#!/usr/bin/env python3
"""gen_skeletons.py - LLM writes realistic IaC skeletons with typed SLOT markers; code fills the slots.

The model NEVER writes a secret and never decides what is a secret. It writes markers:
    @@SLOT:<type>@@   a credential of that type   (code replaces it with a synthetic value)
    @@REF@@           a pointer to a secret store (code renders a family-appropriate reference)
    @@PLACEHOLDER@@   a dummy value               (code picks one)
    @@TRAP:<type>@@   high-entropy non-secret     (sha256 / uuid / ami / digest / sha1)
Labels come from the markers we filled. A file is kept only if it passes the validation gate:
all markers filled, parses (YAML/JSON/HCL), each filled value reachable in the parsed structure,
not a near-duplicate of an accepted file. Split is decided by (family, topic) so all samples of one
topic land in the same split.

usage (GPU):  python gen_skeletons.py --out out/gen --n-per-family 300 --model Qwen/Qwen2.5-Coder-32B-Instruct
dry run (CPU): python gen_skeletons.py --out /tmp/gen --n-per-family 6 --backend mock
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import secrets_lib as S  # noqa: E402
from llm_backend import add_backend_args, make_backend  # noqa: E402
from mutate_real import contains_string, parse_ok  # noqa: E402

# family -> list of (topic, fmt, filename)
SPECS = {
    "terraform": [(t, "hcl", "main.tf") for t in [
        "AWS RDS PostgreSQL instance with parameter group and subnet group", "AWS EKS cluster with managed node group",
        "GCP Cloud SQL instance and Cloud Run service", "Azure AKS cluster with Key Vault", "AWS Lambda behind API Gateway with environment variables",
        "Datadog and PagerDuty provider configuration", "Kubernetes and Helm providers deploying an application", "AWS ElastiCache Redis replication group",
        "MongoDB Atlas cluster and database user", "GitHub provider: repository, webhook and deploy key", "Vault provider mounting secrets engines",
        "Cloudflare DNS records with API token", "AWS Elasticsearch/OpenSearch domain with master user", "Docker provider running a database container"]],
    "k8s": [(t, "yaml", "manifest.yaml") for t in [
        "Deployment with env vars for a Django app talking to Postgres", "StatefulSet for Redis with auth", "CronJob backing up a database to S3",
        "Ingress with TLS and basic-auth annotations", "Secret and ConfigMap for a Node.js API", "Deployment using an imagePullSecret for a private registry",
        "Alertmanager config with Slack webhook and PagerDuty key", "Grafana deployment with admin credentials and SMTP", "Deployment for a Java service with DB and JWT settings",
        "Job running database migrations", "ExternalSecret / SealedSecret style objects", "Keycloak deployment with admin password"]],
    "helm": [(t, "yaml", "values.yaml") for t in [
        "values.yaml for a WordPress chart", "values.yaml for a PostgreSQL chart with auth settings", "values.yaml for an Airflow chart",
        "values.yaml for Keycloak", "values.yaml for a Grafana chart with SMTP and LDAP", "values.yaml for a Jenkins chart with credentials",
        "values.yaml for a GitLab runner chart", "values.yaml for an Argo CD chart with repo credentials", "values.yaml for a Redis chart"]],
    "ansible": [(t, "yaml", "playbook.yml") for t in [
        "playbook installing and configuring PostgreSQL with users", "role defaults/main.yml for a Redis role", "playbook deploying an app using docker_container env",
        "group_vars/all.yml with service accounts and API tokens", "playbook configuring nginx with htpasswd and TLS", "vars for AWS credentials and S3 backups",
        "playbook configuring a Jenkins server and admin user", "playbook creating MySQL databases and users"]],
    "compose": [(t, "yaml", "docker-compose.yml") for t in [
        "WordPress with MySQL", "Gitea with Postgres", "Airflow stack with Postgres and Redis", "Mattermost with Postgres", "Keycloak with Postgres",
        "Nextcloud with Redis and MariaDB", "Grafana, Prometheus and Alertmanager", "a Django app with Celery and RabbitMQ"]] + [
        (t, "env", ".env") for t in [".env for a Django project", ".env for a Node app with Stripe and SendGrid keys", ".env for a Rails app with AWS and database settings",
                                    ".env for a docker-compose Postgres + app stack", ".env for a Laravel application"]],
    "gha": [(t, "yaml", "workflow.yml") for t in [
        "workflow deploying to AWS with environment variables", "workflow pushing a Docker image to a registry", "workflow publishing an npm package",
        "workflow running tests with a Postgres service container", "workflow deploying to Azure Web App", "workflow with Slack notification and Codecov token",
        "workflow building and signing a release", "workflow deploying to a Kubernetes cluster"]],
    "cfn": [(t, "yaml", "template.yaml") for t in [
        "RDS instance with parameters", "Lambda function with environment variables", "ECS task definition with secrets and environment",
        "API Gateway with API key and usage plan", "OpenSearch domain with master user", "Cognito user pool client"]] + [
        (t, "json", "template.json") for t in ["RDS instance with parameters (JSON)", "Lambda function with environment variables (JSON)",
                                              "Elastic Beanstalk environment with option settings (JSON)"]],
}
STYLES = ["minimal and terse", "heavily commented", "production-hardened", "a sloppy dev sandbox with bad defaults",
          "parameterised with variables", "copy-pasted from an old tutorial", "legacy style", "multi-environment (dev/staging/prod)"]
SLOT_TYPES = [t for t in S.SINGLE_LINE_TYPES] + ["connection_string"]  # PEM is covered by mutate_real
SYSTEM = "You write realistic infrastructure-as-code files for a security-tooling test corpus. Output only the file."
MARK = re.compile(r'(["\']?)@@(SLOT|TRAP|REF|PLACEHOLDER)(?::([a-z0-9_]+))?@@\1')
LABEL_FMT = {"hcl": "hcl", "yaml": "yaml", "json": "json", "env": "env"}


def make_prompt(topic, fmt, fname, style, n_lines):
    return (
        f"Write ONE realistic {fmt.upper()} file named `{fname}`: {topic}. Style: {style}. About {n_lines} lines.\n\n"
        "Rules:\n"
        "- Output ONLY the file contents. No markdown fences, no explanation.\n"
        "- Wherever a hard-coded credential would appear (password, token, API key, connection string), write exactly "
        "@@SLOT:<type>@@ inside double quotes. Pick <type> from: " + ", ".join(SLOT_TYPES) + ". The key name next to it must fit the type "
        "(e.g. a database password key gets db_password_random or db_password_human).\n"
        "- Use 2 to 5 such slots, with realistic key names. Some may sit under a neutral key name.\n"
        "- Include at least one @@REF@@ (a reference to a variable / secret store instead of a literal value) and at least one "
        "@@PLACEHOLDER@@ (a dummy value such as changeme) on credential-looking keys, inside double quotes.\n"
        "- Include one @@TRAP:<type>@@ with <type> from sha256, sha1, md5, uuid, digest, ami, subnet: a high-entropy value that is NOT a secret "
        "(a checksum, id or digest), on a key like checksum/commit_sha/ami_id.\n"
        "- NEVER invent any real-looking secret yourself; only the markers above."
    )


def strip_fences(t):
    t = t.strip("\n")
    t = re.sub(r"^```[\w+-]*\n", "", t)
    t = re.sub(r"\n```\s*$", "", t)
    return t


def key_before(line, start):
    m = re.search(r'([A-Za-z_][\w.\-/]*)["\']?\s*[:=]\s*$', line[:start])
    return m.group(1) if m else ""


def fill(text, fmt, family, rng):
    """Replace markers. Returns (new_text, tags_by_line_index, stats) or (None, None, reason)."""
    lines = text.split("\n")
    tags = [[] for _ in lines]
    retyped = 0
    for i, ln in enumerate(lines):
        out = []
        for m in MARK.finditer(ln):
            kind, typ = m.group(2), m.group(3)
            key = key_before(ln, m.start())
            if kind == "SLOT":
                if typ not in SLOT_TYPES:
                    return None, None, "bad_slot_type"
                ok_types = S.types_for_key(key)
                if S.is_sensitive_key(key) and typ not in ok_types:  # the CODE enforces type/key compatibility
                    typ = rng.choice([t for t in ok_types if t in SLOT_TYPES] or ["generic_token_urlsafe"])
                    retyped += 1
                g = S.generate(typ, rng)
                val = dict(text=g["text"], needle=g.get("inner") or g["text"], label="secret", category=g["category"], secret_type=typ)
                rend = f'"{g["text"]}"'
            elif kind == "REF":
                fk = "code" if fmt in ("py", "ts", "groovy") else fmt
                rtext, quoted = S.reference(fk, key or "secret", rng, family)
                val = dict(text=rtext, needle=rtext, label="benign", category="reference", secret_type="")
                rend = f'"{rtext}"' if quoted else rtext
            elif kind == "PLACEHOLDER":
                ph = S.placeholder(rng)
                val = dict(text=ph, needle=ph, label="benign", category="placeholder", secret_type="")
                rend = f'"{ph}"'
            else:  # TRAP
                _, v, tt = S.trap(rng)
                val = dict(text=v, needle=v, label="benign", category="trap", secret_type=tt)
                rend = f'"{v}"'
            out.append((m.start(), m.end(), rend, val, key))
        for s, e, rend, val, key in reversed(out):
            ln = ln[:s] + rend + ln[e:]
            tags[i].append(dict(val, key=key))
        lines[i] = ln
    new = "\n".join(lines)
    if "@@" in new:
        return None, None, "unfilled_marker"
    return new, tags, {"retyped": retyped}


def shingles(text):
    return {hashlib.md5(l.strip().encode()).hexdigest()[:10] for l in text.split("\n") if len(l.strip()) > 6}


def bucket(family, topic, seed):
    h = int(hashlib.sha1(f"{seed}:{family}:{topic}".encode()).hexdigest(), 16) % 1000 / 1000.0
    return "train" if h < 0.70 else ("val" if h < 0.80 else "test")


# ---------------------------------------------------------------- mock responder for CPU dry runs
def mock_responder(system, user):
    h = int(hashlib.sha1(user.encode()).hexdigest(), 16)
    fmt = re.search(r"realistic (\w+) file", user).group(1).lower()
    if h % 5 == 0:  # deliberately broken sample to exercise the gate
        return "this: is: not: valid: yaml: [\n  @@SLOT:db_password_random@@"
    if fmt == "hcl":
        return ('resource "aws_db_instance" "main" {\n  identifier = "app-db"\n  engine     = "postgres"\n  username   = "admin"\n'
                '  password   = "@@SLOT:db_password_random@@"\n  tags = {\n    owner = "team"\n  }\n}\n\nprovider "datadog" {\n'
                '  api_key = "@@SLOT:generic_api_key_hex@@"\n  app_key = "@@REF@@"\n}\n\nlocals {\n  checksum = "@@TRAP:sha256@@"\n'
                '  db_password = "@@PLACEHOLDER@@"\n  config_value = "@@SLOT:custom_token@@"\n}\n')
    if fmt == "env":
        return ('APP_ENV=production\nDB_HOST=db\nDB_PASSWORD="@@SLOT:db_password_human@@"\nSTRIPE_API_KEY="@@SLOT:stripe_secret@@"\n'
                'SECRET_KEY="@@PLACEHOLDER@@"\nMAIL_PASSWORD="@@REF@@"\nGIT_COMMIT="@@TRAP:sha1@@"\n')
    if fmt == "json":
        return ('{\n  "Resources": {\n    "Fn": {\n      "Type": "AWS::Lambda::Function",\n      "Properties": {\n'
                '        "Runtime": "python3.12",\n        "Environment": {\n          "Variables": {\n            "API_TOKEN": "@@SLOT:github_pat@@",\n'
                '            "DB_PASSWORD": "@@REF@@",\n            "SECRET": "@@PLACEHOLDER@@",\n            "ETAG": "@@TRAP:md5@@",\n'
                '            "LOG": "info"\n          }\n        }\n      }\n    }\n  }\n}\n')
    return ('name: app\nservices:\n  web:\n    image: nginx\n    environment:\n      DB_PASSWORD: "@@SLOT:db_password_random@@"\n'
            '      API_KEY: "@@SLOT:google_api_key@@"\n      SESSION_SECRET: "@@REF@@"\n      TOKEN: "@@PLACEHOLDER@@"\n'
            '      BUILD_UUID: "@@TRAP:uuid@@"\n      CLIENT_SECRET: "@@SLOT:aws_access_key_id@@"\n')


def main():
    ap = argparse.ArgumentParser()
    add_backend_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-family", type=int, default=200, help="accepted files wanted per family")
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--dedup-jaccard", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    be = make_backend(a, mock_responder)
    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "files"), exist_ok=True)
    label_rows, index_rows, shingle_bank = [], [], defaultdict(list)
    fail, accepted = Counter(), Counter()
    cols_meta = ["file_id", "repo", "family", "split", "relpath", "fmt"]

    for rnd in range(a.max_rounds):
        need = {f: a.n_per_family - accepted[f] for f in SPECS if accepted[f] < a.n_per_family}
        if not need:
            break
        jobs = []
        for fam, n in need.items():
            for _ in range(int(n * (1.5 if rnd == 0 else 2.0)) + 1):   # oversample: some will fail the gate
                topic, fmt, fname = rng.choice(SPECS[fam])
                style, n_lines = rng.choice(STYLES), rng.randint(25, 90)
                jobs.append((fam, topic, fmt, fname, style, make_prompt(topic, fmt, fname, style, n_lines)))
        print(f"round {rnd}: generating {len(jobs)} candidates")
        outs = be.chat([(SYSTEM, j[-1]) for j in jobs], max_tokens=2200, temperature=0.95, top_p=0.95, seed=a.seed + rnd)
        for (fam, topic, fmt, fname, style, _), raw in zip(jobs, outs):
            if accepted[fam] >= a.n_per_family:
                continue
            text = strip_fences(raw)
            new, tags, info = fill(text, fmt, fam, rng)
            if new is None:
                fail[info] += 1
                continue
            ok, parsed = parse_ok(LABEL_FMT[fmt], new)
            if not ok:
                fail["parse_fail"] += 1
                continue
            flat = [t for ts in tags for t in ts]
            if not any(t["label"] == "secret" for t in flat):
                fail["no_secret_slot"] += 1
                continue
            if parsed is not None and not all(contains_string(parsed, t["text"]) for t in flat):
                fail["value_not_reachable"] += 1
                continue
            sh = shingles(text)  # dedup on the PRE-fill skeleton: filled random values would make every line unique
            if any(len(sh & o) / max(1, len(sh | o)) > a.dedup_jaccard for o in shingle_bank[fam]):
                fail["near_duplicate"] += 1
                continue
            shingle_bank[fam].append(sh)
            split = bucket(fam, topic, a.seed)
            fid = f"gen__{fam}__{hashlib.sha1(new.encode()).hexdigest()[:10]}"
            op = os.path.join(a.out, "files", fid, fname)
            os.makedirs(os.path.dirname(op), exist_ok=True)
            open(op, "w", encoding="utf-8").write(new)
            meta = dict(file_id=fid, repo=f"generated/{fam}", family=fam, split=split, relpath=fname, fmt=fmt)
            nl = 0
            for i, ts in enumerate(tags, 1):
                for sid, t in enumerate(ts, 1):
                    ln = new.split("\n")[i - 1]
                    s = ln.find(t["needle"])
                    s, e = (s, s + len(t["needle"])) if s >= 0 else (0, len(ln))
                    label_rows.append(dict(meta, line_no=i, label=t["label"], category=t["category"], secret_type=t["secret_type"],
                                           operator="llm_skeleton", key_name=t["key"],
                                           key_hint="sensitive" if S.is_sensitive_key(t["key"]) else "neutral", encoding="plain",
                                           commented=0, span_id=len(label_rows) + 1, span_start=s, span_end=e, pair_id="", variant=""))
                    nl += 1
            index_rows.append(dict(meta, n_lines=len(new.split("\n")), n_labeled=nl, validated=int(fmt in ("yaml", "json", "hcl"))))
            accepted[fam] += 1

    cols = cols_meta + ["line_no", "label", "category", "secret_type", "operator", "key_name", "key_hint", "encoding", "commented",
                        "span_id", "span_start", "span_end", "pair_id", "variant"]
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(label_rows)
    with open(os.path.join(a.out, "files_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_meta + ["n_lines", "n_labeled", "validated"])
        w.writeheader()
        w.writerows(index_rows)
    total_fail = sum(fail.values())
    stats = dict(accepted=dict(accepted), rejected=dict(fail), reject_rate=round(total_fail / max(1, total_fail + sum(accepted.values())), 3),
                 split=dict(Counter(r["split"] for r in index_rows)))
    json.dump(stats, open(os.path.join(a.out, "gen_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    if stats["reject_rate"] > 0.4:
        print("WARNING: >40% rejected by the gate - tighten the prompt / add few-shot examples before scaling up.")


if __name__ == "__main__":
    main()
