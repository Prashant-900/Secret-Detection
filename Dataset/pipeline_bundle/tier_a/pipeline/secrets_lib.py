"""secrets_lib.py - typed-slot library of SYNTHETIC secrets and look-alike negatives.

Every value produced here is random and fake. Nothing is derived from a real credential.

Design rule (fixes the "JWT in an AWS key slot" problem): a slot is only ever filled
with a secret type that is compatible with its key name. Use `types_for_key(key)`.

Categories (what the label means):
  secret_known_format   vendor-shaped secret (AKIA..., ghp_..., PEM, JWT ...)
  unknown_format_secret high-entropy secret with no vendor prefix / custom format
  low_entropy_secret    human-style password (Summer2024!)
  reference             pointer to a secret (var.x, ${X}, secrets.X)      -> benign
  placeholder           dummy value (changeme, <YOUR_KEY>)               -> benign
  trap                  high-entropy NON-secret (sha256, uuid, ami id)    -> benign
"""
import base64
import json
import random
import re
import string
import uuid

UP, LOW, DIG = string.ascii_uppercase, string.ascii_lowercase, string.digits
ALNUM = UP + LOW + DIG
B64 = ALNUM + "+/"
B64URL = ALNUM + "-_"
HEX = "0123456789abcdef"
B32 = UP + "234567"
# symbols that are safe inside double-quoted YAML / HCL / JSON / shell strings (no $ { } \ " ' ` :)
SAFE_SYMBOLS = "!@#%^&*-_+=."


def _rs(rng, alphabet, n):
    return "".join(rng.choice(alphabet) for _ in range(n))


# ----------------------------------------------------------------------------- word lists
WORDS = ["Summer", "Winter", "Spring", "Autumn", "Welcome", "Admin", "Password", "Passw0rd", "Qwerty",
         "Letmein", "Monkey", "Dragon", "Master", "Shadow", "Sunshine", "Princess", "Football", "Baseball",
         "Secret", "Company", "Prod", "Staging", "Server", "Database", "Redis", "Mysql", "Postgres",
         "Cloud", "Docker", "Jenkins", "Ansible", "Deploy", "Service", "Backend", "Internal", "Changeit"]
COMPANIES = ["acme", "globex", "initech", "umbrella", "hooli", "stark", "wayne", "wonka", "cyberdyne", "soylent"]
KEYBOARD_WALKS = ["qwerty", "qwertyuiop", "asdfgh", "zxcvbn", "1qaz2wsx", "qazwsx", "q1w2e3r4", "1q2w3e4r"]


# ----------------------------------------------------------------------------- generators
def gen_aws_access_key_id(rng):
    return "AKIA" + _rs(rng, B32, 16)


def gen_aws_secret_access_key(rng):
    return _rs(rng, ALNUM + "+/", 40)


def gen_github_pat(rng):
    return "ghp_" + _rs(rng, ALNUM, 36)


def gen_slack_bot_token(rng):
    return f"xoxb-{_rs(rng, DIG, 12)}-{_rs(rng, DIG, 12)}-{_rs(rng, ALNUM, 24)}"


def gen_google_api_key(rng):
    return "AIza" + _rs(rng, B64URL, 35)


def gen_stripe_secret(rng):
    return "sk_live_" + _rs(rng, ALNUM, 24)


def gen_azure_client_secret(rng):
    return _rs(rng, ALNUM, 3) + "8Q~" + _rs(rng, ALNUM + "_.~-", 34)


def _b64u(b):
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def gen_jwt(rng):
    compact = dict(separators=(",", ":"))  # real JWTs are compact JSON
    header = _b64u(json.dumps({"alg": rng.choice(["HS256", "HS384", "RS256"]), "typ": "JWT"}, **compact).encode())
    payload = _b64u(json.dumps({"sub": _rs(rng, DIG, 10), "name": rng.choice(["svc", "admin", "deploy", "ci"]),
                                "iat": rng.randint(1500000000, 1800000000)}, **compact).encode())
    return f"{header}.{payload}.{_rs(rng, B64URL, 43)}"


def gen_generic_api_key_hex(rng):
    return _rs(rng, HEX, rng.choice([32, 40, 64]))


def gen_generic_token_urlsafe(rng):
    return _rs(rng, B64URL, rng.choice([32, 43, 48]))


def gen_webhook_secret(rng):
    return _rs(rng, HEX, 64)


def gen_db_password_random(rng):
    n = rng.randint(14, 24)
    core = _rs(rng, ALNUM, n)
    # sprinkle 1-3 safe symbols so it looks like a generated password
    chars = list(core)
    for _ in range(rng.randint(1, 3)):
        chars[rng.randrange(n)] = rng.choice(SAFE_SYMBOLS)
    return "".join(chars)


def gen_db_password_human(rng):
    """Low-entropy, human style password. Scanners that rely on entropy tend to miss these."""
    style = rng.randrange(6)
    w = rng.choice(WORDS)
    if style == 0:
        return f"{w}{rng.randint(2015, 2026)}{rng.choice('!@#.')}"
    if style == 1:
        return f"{w}{rng.randint(1, 999)}{rng.choice('!@#_-')}"
    if style == 2:
        return f"{rng.choice(COMPANIES)}-{rng.choice(['prod', 'dev', 'db', 'admin', 'root'])}-{rng.randint(1, 99)}"
    if style == 3:
        return rng.choice(KEYBOARD_WALKS) + rng.choice(["", "!", "1", "123", "@2024"])
    if style == 4:
        return w.lower().replace("o", "0").replace("a", "@") + str(rng.randint(10, 99))
    return f"{w}{rng.choice(WORDS)}{rng.randint(1, 99)}"


def gen_custom_token(rng):
    """Unknown vendor format: random prefix, separator, body with varied charset."""
    prefix = rng.choice(COMPANIES)[:rng.randint(2, 5)]
    kind = rng.choice(["pat", "key", "tok", "sec", "svc"])
    sep = rng.choice(["_", "-", "."])
    body_alpha = rng.choice([ALNUM, LOW + DIG, UP + DIG, HEX, B64URL])
    return f"{prefix}{sep}{kind}{sep}{_rs(rng, body_alpha, rng.randint(24, 48))}"


def gen_connection_string(rng, password=None):
    scheme, port = rng.choice([("postgres", 5432), ("postgresql", 5432), ("mysql", 3306),
                               ("mongodb", 27017), ("redis", 6379), ("amqp", 5672)])
    user = rng.choice(["admin", "app", "svc_api", "root", "dbuser", "service"])
    pw = password or _rs(rng, ALNUM, rng.randint(12, 20))  # URL-safe: no symbols
    host = rng.choice(["db.internal", "10.0.%d.%d" % (rng.randint(0, 255), rng.randint(1, 254)),
                       "mydb.cluster-%s.us-east-1.rds.amazonaws.com" % _rs(rng, LOW + DIG, 12),
                       "redis-master.cache.svc", "postgres"])
    name = rng.choice(["app", "prod", "orders", "users", "main", "analytics"])
    if scheme == "redis":
        return f"{scheme}://:{pw}@{host}:{port}/0", pw
    return f"{scheme}://{user}:{pw}@{host}:{port}/{name}", pw


def gen_private_key_pem(rng, kind=None):
    """FAKE key material: random base64 body with a real header/footer. Not a valid key."""
    kind = kind or rng.choice(["RSA PRIVATE KEY", "PRIVATE KEY", "EC PRIVATE KEY", "OPENSSH PRIVATE KEY"])
    n_lines = rng.randint(8, 24)
    body = [_rs(rng, B64, 64) for _ in range(n_lines)]
    body.append(_rs(rng, B64, rng.randint(8, 60)) + "=")
    return [f"-----BEGIN {kind}-----"] + body + [f"-----END {kind}-----"]


# type -> (generator, category, key-name candidates)
SECRET_TYPES = {
    "aws_access_key_id": (gen_aws_access_key_id, "secret_known_format",
                          ["access_key", "aws_access_key_id", "AWS_ACCESS_KEY_ID", "accessKeyId", "access_key_id"]),
    "aws_secret_access_key": (gen_aws_secret_access_key, "secret_known_format",
                              ["secret_key", "aws_secret_access_key", "AWS_SECRET_ACCESS_KEY", "secretAccessKey", "secret_access_key"]),
    "github_pat": (gen_github_pat, "secret_known_format",
                   ["github_token", "GITHUB_TOKEN", "gh_token", "GH_PAT", "token"]),
    "slack_bot_token": (gen_slack_bot_token, "secret_known_format",
                        ["slack_token", "SLACK_BOT_TOKEN", "slack_bot_token", "SLACK_TOKEN"]),
    "google_api_key": (gen_google_api_key, "secret_known_format",
                       ["google_api_key", "GOOGLE_API_KEY", "maps_api_key", "api_key", "gcp_api_key"]),
    "stripe_secret": (gen_stripe_secret, "secret_known_format",
                      ["stripe_secret_key", "STRIPE_API_KEY", "stripe_key", "STRIPE_SECRET_KEY"]),
    "azure_client_secret": (gen_azure_client_secret, "secret_known_format",
                            ["client_secret", "ARM_CLIENT_SECRET", "AZURE_CLIENT_SECRET", "clientSecret"]),
    "jwt": (gen_jwt, "secret_known_format",
            ["jwt_token", "auth_token", "bearer_token", "JWT_TOKEN", "access_token", "id_token"]),
    "generic_api_key_hex": (gen_generic_api_key_hex, "unknown_format_secret",
                            ["api_key", "API_KEY", "apikey", "x-api-key", "service_api_key", "datadog_api_key"]),
    "generic_token_urlsafe": (gen_generic_token_urlsafe, "unknown_format_secret",
                              ["secret", "auth_token", "access_token", "client_secret", "SECRET_KEY", "app_secret", "token"]),
    "webhook_secret": (gen_webhook_secret, "unknown_format_secret",
                       ["webhook_secret", "signing_key", "encryption_key", "hmac_secret", "WEBHOOK_SECRET"]),
    "custom_token": (gen_custom_token, "unknown_format_secret",
                     ["token", "api_token", "service_token", "deploy_token", "registry_token", "API_TOKEN"]),
    "db_password_random": (gen_db_password_random, "unknown_format_secret",
                           ["password", "db_password", "DB_PASSWORD", "POSTGRES_PASSWORD", "MYSQL_ROOT_PASSWORD",
                            "master_password", "administrator_login_password", "admin_password", "REDIS_PASSWORD"]),
    "db_password_human": (gen_db_password_human, "low_entropy_secret",
                          ["password", "db_password", "DB_PASSWORD", "POSTGRES_PASSWORD", "MYSQL_ROOT_PASSWORD",
                           "master_password", "admin_password", "ansible_password", "ansible_become_pass"]),
    # structured types handled specially by operators (multi-line / embedded):
    "connection_string": (None, "unknown_format_secret",
                          ["database_url", "DATABASE_URL", "connection_string", "redis_url", "MONGO_URI", "SQLALCHEMY_DATABASE_URI"]),
    "private_key_pem": (None, "secret_known_format",
                        ["private_key", "ssh_private_key", "tls_key", "privateKey", "tls.key", "SSH_PRIVATE_KEY"]),
}
SINGLE_LINE_TYPES = [t for t, (g, _, _) in SECRET_TYPES.items() if g is not None]

SENSITIVE_KEY_RE = re.compile(
    r"(pass(word|wd|phrase)?|pwd|secret|token|api[_\-]?key|apikey|access[_\-]?key|private[_\-]?key|"
    r"credential|auth(?!or)|client[_\-]?secret|connection[_\-]?string|dsn|signing|webhook|bearer|passwd|become_pass|"
    r"hmac|encrypt)",
    re.I)

NEUTRAL_KEYS = ["value", "data", "config", "param", "setting", "content", "payload", "entry", "opt", "item",
                "v", "x", "blob", "extra", "custom", "field", "attr", "prop", "arg", "input"]

# key-name fragments -> compatible types (first match wins; most specific first)
_KEY_RULES = [
    (re.compile(r"aws.*secret|secret.?access", re.I), ["aws_secret_access_key"]),
    (re.compile(r"aws.*access|access.?key.?id|accesskeyid", re.I), ["aws_access_key_id"]),
    (re.compile(r"github|gh_", re.I), ["github_pat"]),
    (re.compile(r"slack", re.I), ["slack_bot_token"]),
    (re.compile(r"stripe", re.I), ["stripe_secret"]),
    (re.compile(r"google|gcp|maps", re.I), ["google_api_key"]),
    (re.compile(r"client.?secret|arm_client|azure", re.I), ["azure_client_secret", "generic_token_urlsafe"]),
    (re.compile(r"jwt|bearer|id_token", re.I), ["jwt"]),
    (re.compile(r"private.?key|ssh_key|tls.?key", re.I), ["private_key_pem"]),
    (re.compile(r"url|uri|dsn|connection", re.I), ["connection_string"]),
    (re.compile(r"webhook|signing|hmac|encrypt", re.I), ["webhook_secret"]),
    (re.compile(r"pass|pwd", re.I), ["db_password_random", "db_password_random", "db_password_human"]),
    (re.compile(r"api.?key|apikey", re.I), ["generic_api_key_hex", "google_api_key"]),
    (re.compile(r"token", re.I), ["custom_token", "generic_token_urlsafe", "github_pat", "jwt"]),
    (re.compile(r"secret", re.I), ["generic_token_urlsafe", "generic_api_key_hex", "webhook_secret"]),
]


def types_for_key(key):
    """Compatible secret types for a key name. Falls back to generic tokens."""
    for rx, types in _KEY_RULES:
        if rx.search(key or ""):
            return list(types)
    return ["generic_token_urlsafe", "generic_api_key_hex", "custom_token"]


def key_for_type(rng, secret_type):
    return rng.choice(SECRET_TYPES[secret_type][2])


def is_sensitive_key(key):
    return bool(SENSITIVE_KEY_RE.search(key or ""))


def generate(secret_type, rng):
    """Return dict(text, secret_type, category, lines) for single-line types.
    For connection_string the password substring is returned as `inner`.
    For private_key_pem `lines` holds the multi-line value."""
    gen, category, _ = SECRET_TYPES[secret_type]
    if secret_type == "connection_string":
        text, inner = gen_connection_string(rng)
        return dict(text=text, inner=inner, secret_type=secret_type, category=category)
    if secret_type == "private_key_pem":
        lines = gen_private_key_pem(rng)
        return dict(text="\n".join(lines), lines=lines, secret_type=secret_type, category=category)
    return dict(text=gen(rng), secret_type=secret_type, category=category)


# ----------------------------------------------------------------------------- negatives
PLACEHOLDERS = ["changeme", "CHANGEME", "change-me", "<password>", "<YOUR_API_KEY>", "<your-token-here>",
                "your-password-here", "xxxxxxxxxxxxxxxx", "REPLACE_ME", "REPLACE_WITH_YOUR_TOKEN", "********",
                "example", "TODO", "dummy", "placeholder", "<secret>", "password", "secret", "${PASSWORD}-placeholder",
                "INSERT_KEY_HERE", "0000000000000000", "aaaaaaaaaaaaaaaaaaaa", "<REDACTED>", "YOUR_SECRET_KEY"]


def placeholder(rng):
    return rng.choice(PLACEHOLDERS)


YAML_REFS_BY_FAMILY = {
    "gha": lambda low, up: [(f"${{{{ secrets.{up} }}}}", True), (f"${{{{ vars.{up} }}}}", True)],
    "helm": lambda low, up: [(f"{{{{ .Values.{low} }}}}", True), (f"{{{{ .Values.global.{low} }}}}", True),
                             (f"{{{{ required \"{low} is required\" .Values.{low} }}}}", True)],
    "ansible": lambda low, up: [(f"{{{{ vault_{low} }}}}", True), (f"{{{{ lookup('env', '{up}') }}}}", True),
                                (f"{{{{ {low} }}}}", True)],
    "compose": lambda low, up: [(f"${{{up}}}", True), (f"${{{up}:?err}}", True), (f"${{{up}:-}}", True)],
    "k8s": lambda low, up: [(f"$({up})", True), (f"${{{up}}}", True), (f"<path:secret/data/app#{low}>", True)],
    "cfn": lambda low, up: [(f"{{{{resolve:secretsmanager:{low}:SecretString:password}}}}", True),
                            (f"{{{{resolve:ssm-secure:/app/{low}:1}}}}", True)],
}


def reference(fmt, key, rng, family=None):
    """A pointer to a secret, rendered for the file format (and IaC family). Returns (text, needs_quotes)."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_") or "SECRET"
    low, up = name.lower(), name.upper()
    if fmt == "yaml" and family in YAML_REFS_BY_FAMILY:
        return rng.choice(YAML_REFS_BY_FAMILY[family](low, up))
    opts = {
        "hcl": [(f"var.{low}", False), (f"data.aws_secretsmanager_secret_version.{low}.secret_string", False),
                (f"random_password.{low}.result", False), (f"data.vault_generic_secret.app.data[\"{low}\"]", False),
                (f"${{var.{low}}}", True)],
        "yaml": [(f"${{{up}}}", True), (f"{{{{ .Values.{low} }}}}", True), (f"{{{{ lookup('env', '{up}') }}}}", True),
                 (f"${{{{ secrets.{up} }}}}", True), (f"{{{{ vault_{low} }}}}", True), (f"$({up})", True)],
        "json": [(f"{{{{resolve:secretsmanager:{low}:SecretString:password}}}}", True),
                 (f"{{{{resolve:ssm-secure:/app/{low}:1}}}}", True), (f"${{{up}}}", True)],
        "env": [(f"${{{up}}}", True), (f"$(cat /run/secrets/{low})", True), (f"${up}", False)],
        "code": [(f"os.environ['{up}']", False), (f"process.env.{up}", False), (f"credentials('{low}')", False),
                 (f"System.getenv('{up}')", False), (f"config.requireSecret('{low}')", False)],
    }
    text, quoted = rng.choice(opts.get(fmt, opts["yaml"]))
    return text, quoted


TRAP_KEYS = ["checksum", "sha256", "digest", "commit_sha", "ami_id", "image_digest", "etag", "uuid",
             "resource_id", "kms_key_id", "build_id", "revision", "fingerprint", "content_hash", "snapshot_id"]


def trap(rng, key=None):
    """High-entropy non-secret. Returns (key, value, trap_type)."""
    kind = rng.choice(["sha256", "sha1", "md5", "uuid", "digest", "ami", "account", "subnet"])
    if kind == "sha256":
        return key or rng.choice(["checksum", "sha256", "content_hash"]), _rs(rng, HEX, 64), kind
    if kind == "sha1":
        return key or rng.choice(["commit_sha", "revision", "git_ref"]), _rs(rng, HEX, 40), kind
    if kind == "md5":
        return key or rng.choice(["etag", "md5", "fingerprint"]), _rs(rng, HEX, 32), kind
    if kind == "uuid":
        return key or rng.choice(["uuid", "resource_id", "kms_key_id", "build_id"]), str(uuid.UUID(int=rng.getrandbits(128), version=4)), kind
    if kind == "digest":
        return key or "image_digest", "sha256:" + _rs(rng, HEX, 64), kind
    if kind == "ami":
        return key or "ami_id", "ami-" + _rs(rng, HEX, 17), kind
    if kind == "account":
        return key or "account_id", _rs(rng, DIG, 12), kind
    return key or "subnet_id", "subnet-" + _rs(rng, HEX, 17), kind


if __name__ == "__main__":  # smoke test
    r = random.Random(0)
    for t in SECRET_TYPES:
        v = generate(t, r)
        print(f"{t:24s} {v['category']:22s} {v['text'][:70].splitlines()[0]}")
    print(placeholder(r), reference("hcl", "db_password", r), trap(r))
