#!/usr/bin/env python3
"""mutate_real.py - Tier B: inject labeled secrets / traps / references into REAL IaC files.

Labels come ONLY from strings this script injected (never from a model or a scanner).
Every mutated file must still parse (YAML / JSON / HCL / Python) and each injected value
must be reachable in the parsed structure, otherwise the file is retried, then dropped.

Operators: inject_secret, to_reference, to_placeholder, rename_key, base64, pem,
           conn_string, commented, trap, k8s_secret_doc, plus --pair-frac contrastive triplets
           (same slot rendered as secret / reference / placeholder).

usage:
  python mutate_real.py --repos repos --manifest manifest_v2.csv --splits splits.csv --out out/tierB \
         [--exclude-hits preexisting_hits.csv] [--max-files-per-repo 20] [--seed 13]
"""
import argparse
import ast
import base64
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

import yaml

try:
    import hcl2
except ImportError:  # HCL validation degrades to "unchanged brackets"
    hcl2 = None

import secrets_lib as S

SKIP_DIRS = {".git", "node_modules", "vendor", ".terraform", "dist", "build", "__pycache__", ".venv", "venv",
             "site-packages", ".idea", ".vscode", "bower_components"}
FAMILY_FMTS = {
    "terraform": ["hcl"], "helm": ["yaml"], "k8s": ["yaml"], "ansible": ["yaml"], "compose": ["yaml", "env"],
    "gha": ["yaml"], "jenkins": ["groovy", "yaml"], "pulumi": ["py", "ts", "yaml"], "cfn": ["json", "yaml"],
}
CODE_FMTS = ("py", "ts", "groovy")
MAX_BYTES, MIN_LINES = 200_000, 5


# ============================================================================ format helpers
def detect_fmt(relpath, text=None):
    low = os.path.basename(relpath).lower()
    if low.endswith((".tf", ".tfvars")):
        return "hcl"
    if low.endswith((".yaml", ".yml")):
        return "yaml"
    if low.endswith(".json"):
        return "json"
    if low.endswith(".template") and text:
        return "json" if text.lstrip().startswith("{") else "yaml"
    if re.fullmatch(r"(\.?env(\.[\w.\-]+)?|[\w.\-]+\.env(\.[\w.\-]+)?)", low):
        return "env"
    if low == "jenkinsfile" or low.endswith((".groovy", ".jenkinsfile")):
        return "groovy"
    if low.endswith(".py"):
        return "py"
    if low.endswith((".ts", ".js")) and not low.endswith((".min.js", ".d.ts")):
        return "ts"
    return None


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that accepts unknown tags (CloudFormation !Ref/!Sub, Ansible !vault ...)."""


def _any_tag(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_mapping(node, deep=True)


_TolerantLoader.add_multi_constructor("!", _any_tag)


def parse_ok(fmt, text):
    """Return (ok, parsed_or_None). Formats we cannot really parse return (True, None)."""
    try:
        if fmt == "yaml":
            return True, list(yaml.load_all(text, Loader=_TolerantLoader))
        if fmt == "json":
            return True, json.loads(text)
        if fmt == "hcl":
            return (True, hcl2.loads(text)) if hcl2 else (True, None)
        if fmt == "py":
            ast.parse(text)
            return True, None
        return True, None
    except Exception:
        return False, None


def contains_string(obj, needle):
    if isinstance(obj, str):
        return needle in obj
    if isinstance(obj, dict):
        return any(contains_string(k, needle) or contains_string(v, needle) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return any(contains_string(v, needle) for v in obj)
    return False


ASSIGN_RE = {
    "yaml": re.compile(r'^(?P<indent>[ ]*)(?P<dash>-[ ]+)?(?P<key>[A-Za-z_][\w.\-/]*)[ ]*:(?:[ ]+(?P<val>.*))?$'),
    "hcl": re.compile(r'^(?P<indent>[ \t]*)(?P<key>[A-Za-z_][\w.\-]*)[ \t]*=[ \t]*(?P<val>.+)$'),
    "env": re.compile(r'^(?P<indent>[ \t]*)(?P<export>export[ ]+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<val>.*)$'),
    "json": re.compile(r'^(?P<indent>[ \t]*)"(?P<key>[^"\\]+)"[ \t]*:[ \t]*(?P<val>.+)$'),
    "code": re.compile(r'^(?P<indent>[ \t]*)(?P<decl>(?:(?:const|let|var|val|def|final|static|String|export)[ ]+)*)'
                       r'(?P<key>[A-Za-z_]\w*)[ \t]*(?P<sep>[=:])[ \t]*(?P<val>.+)$'),
}
ENV_LIST_RE = re.compile(r'^(?P<indent>[ ]*)-[ ]+(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<val>.*)$')
NAME_RE = re.compile(r'^(?P<indent>[ ]*)-[ ]+name:[ ]*(?P<name>\S.*?)[ ]*$')


def value_region(fmt, val):
    """(start, end, quote) of a simple one-line scalar inside `val`, else None."""
    v = val.rstrip()
    if not v:
        return None
    q = v[0]
    if q in "\"'":
        j = v.find(q, 1)
        while j != -1 and v[j - 1] == "\\":
            j = v.find(q, j + 1)
        if j == -1:
            return None
        rest = v[j + 1:].strip()
        if rest and not re.match(r'^[,;]?\s*(#.*|//.*)?$', rest):
            return None
        return 0, j + 1, q
    end = len(v)
    m = re.search(r'\s(#|//)', v)
    if m:
        end = m.start()
    seg = v[:end].rstrip()
    while seg.endswith((",", ";")):
        seg = seg[:-1].rstrip()
    if not seg:
        return None
    if fmt != "env" and seg[0] in "{[|>&*!<%@`":
        return None
    if seg.count("(") != seg.count(")") or seg.count("[") != seg.count("]") or seg.count("{") != seg.count("}"):
        return None
    return 0, len(seg), ""


def parse_line(fmt, line):
    s = line.strip()
    if not s or s.startswith(("#", "//", "/*", "*")) or line.rstrip().endswith("\\"):
        return None
    m = ASSIGN_RE["code" if fmt in CODE_FMTS else fmt].match(line)
    if not m:
        return None
    val = m.group("val")
    if val is None or (fmt == "hcl" and val.startswith("=")):
        return None
    reg = value_region(fmt, val)
    if not reg:
        return None
    off = m.start("val")
    q = reg[2]
    if fmt in CODE_FMTS and not q:
        return None
    gd = m.groupdict()
    return dict(indent=gd["indent"], dash=gd.get("dash") or "", key=gd["key"], vs=off + reg[0], ve=off + reg[1],
                quote=q, sep=gd.get("sep"), export=gd.get("export") or "", decl=gd.get("decl") or "",
                tail=line[off + reg[1]:])


# ============================================================================ editing context
class Ctx:
    """Lines + parallel tags. Tags travel with their line, so insertions never invalidate labels."""

    def __init__(self, fmt, text, rng, family):
        self.fmt, self.rng, self.family = fmt, rng, family
        self.lines = text.split("\n")
        self.tags = [None] * len(self.lines)
        self.n_span = 0

    def text(self):
        return "\n".join(self.lines)

    def splice(self, start, remove, lines, tags):
        self.lines[start:start + remove] = lines
        self.tags[start:start + remove] = tags

    def new_span(self):
        self.n_span += 1
        return self.n_span

    def analyze(self):
        """blocked[i]: inside YAML block scalar / HCL heredoc. root[i]: enclosing top-level HCL block type."""
        n = len(self.lines)
        blocked, root = [False] * n, [None] * n
        self.parents = [None] * n
        if self.fmt in ("yaml", "json"):
            stack = []
            for i, ln in enumerate(self.lines):
                if not ln.strip() or ln.lstrip().startswith("#"):
                    continue
                ind = len(ln) - len(ln.lstrip(" "))
                while stack and stack[-1][0] >= ind:
                    stack.pop()
                self.parents[i] = stack[-1][1] if stack else None
                if self.fmt == "yaml":
                    m = re.match(r'^( *)(- +)?([A-Za-z_][\w.\-/]*) *:\s*(#.*)?$', ln)
                    if m:
                        stack.append((ind + len(m.group(2) or ""), m.group(3)))
                else:
                    m = re.match(r'^ *"([^"]+)" *: *[{\[]\s*$', ln)
                    if m:
                        stack.append((ind, m.group(1)))
        if self.fmt == "yaml":
            parent = None
            for i, ln in enumerate(self.lines):
                ind = len(ln) - len(ln.lstrip(" "))
                if parent is not None:
                    if ln.strip() == "" or ind > parent:
                        blocked[i] = True
                        continue
                    parent = None
                if re.search(r':[ ]+[|>][+\-0-9]*[ ]*(#.*)?$', ln) or re.search(r'-[ ]+[|>][+\-0-9]*[ ]*$', ln):
                    parent = ind
        elif self.fmt == "hcl":
            stack, term = [], None
            for i, ln in enumerate(self.lines):
                if term is not None:
                    blocked[i] = True
                    if ln.strip() == term:
                        term = None
                    continue
                m = re.search(r'<<-?\s*([A-Za-z_][A-Za-z0-9_]*)\s*$', ln)
                if m:
                    term = m.group(1)
                root[i] = stack[0] if stack else None
                s = ln.strip()
                if s.endswith("{") and not s.startswith(("#", "//")):
                    stack.append(s.split()[0] if s.split() else "")
                    root[i] = stack[0]
                elif s.startswith("}") and stack:
                    stack.pop()
        return blocked, root


def enumerate_sites(ctx):
    """All places where a value can be replaced or a sibling line inserted."""
    fmt = ctx.fmt
    blocked, root = ctx.analyze()
    parents = ctx.parents
    sites = []
    for i, line in enumerate(ctx.lines):
        if ctx.tags[i] or blocked[i]:
            continue
        if fmt == "yaml":
            m = ENV_LIST_RE.match(line)
            if m and not line.lstrip().startswith("- name:"):
                val = m.group("val")
                reg = value_region("env", val) if val else None
                if reg:
                    off = m.start("val")
                    p = dict(indent=m.group("indent"), dash="- ", key=m.group("key"), vs=off + reg[0],
                             ve=off + reg[1], quote="", sep="=", export="", tail=line[off + reg[1]:])
                    sites.append(dict(kind="env_list", idx=i, key=p["key"], p=p, insertable=True, replaceable=True,
                                      parent=parents[i]))
                    continue
            nm = NAME_RE.match(line)
            if nm and i + 1 < len(ctx.lines) and not ctx.tags[i + 1] and not blocked[i + 1]:
                p2 = parse_line("yaml", ctx.lines[i + 1])
                if p2 and p2["key"] == "value" and not p2["dash"] and len(p2["indent"]) == len(nm.group("indent")) + 2:
                    key = nm.group("name").strip("\"'")
                    sites.append(dict(kind="name_value", idx=i + 1, key=key, p=p2, dash_indent=nm.group("indent"),
                                      insertable=True, replaceable=True, parent=parents[i]))
                    continue
        p = parse_line(fmt, line)
        if not p:
            continue
        if fmt == "yaml" and p["key"] in ("name", "value") and p["dash"] == "" and False:
            continue
        ok_hcl = fmt != "hcl" or root[i] in (None, "resource", "provider", "locals", "module", "data")
        ins = ok_hcl
        if fmt == "json":
            ins = line.rstrip().endswith(",")
        sites.append(dict(kind="assign", idx=i, key=p["key"], p=p, insertable=ins, replaceable=True,
                          parent=parents[i] if fmt in ("yaml", "json") else root[i]))
    for s in sites:
        s["sensitive"] = S.is_sensitive_key(s["key"])
    return sites


PREFERRED_PARENTS = {"env", "environment", "data", "stringdata", "config", "configmap", "vars", "variables",
                     "parameters", "properties", "with", "secrets", "settings", "extravars", "defaults", "values",
                     "locals", "provider", "module", "global", "auth", "credentials", "app", "application"}


def site_weight(ctx, s):
    """Prefer places where a credential would naturally live; avoid top-level keys of a document."""
    if ctx.fmt in ("env",) or ctx.fmt in CODE_FMTS:
        return 1.0
    par = s.get("parent")
    if par is None:
        return 0.15 if ctx.fmt in ("yaml", "json") else 1.0
    if str(par).lower() in PREFERRED_PARENTS:
        return 6.0
    return 1.0


def pick_site(ctx, which):
    sites = enumerate_sites(ctx)
    if which == "sensitive":
        c = [s for s in sites if s["sensitive"] and s["replaceable"] and s["key"] not in ("name",)]
        return ctx.rng.choice(c) if c else None
    c = [s for s in sites if s["insertable"]]
    if not c:
        return None
    return ctx.rng.choices(c, weights=[site_weight(ctx, s) for s in c], k=1)[0]


# ============================================================================ rendering
def quote_for(fmt, site=None):
    if fmt in ("json", "hcl", "env", "yaml"):
        if fmt == "yaml" and site and site["p"]["quote"] == "'":
            return "'"
        return '"'
    return (site["p"]["quote"] if site and site["p"]["quote"] else "'")


def render_value(fmt, val, site):
    if site["kind"] == "env_list" or not val["quoted"]:
        return val["text"]
    q = quote_for(fmt, site)
    return f"{q}{val['text']}{q}"


def sibling_indent(site):
    p = site["p"]
    if site["kind"] == "env_list":
        return p["indent"]
    return p["indent"] + " " * len(p["dash"])


def build_insert(ctx, site, key, val):
    """Return (new_lines, index_of_value_line_within_new_lines)."""
    fmt, p, kind = ctx.fmt, site["p"], site["kind"]
    r = render_value(fmt, val, site)
    if fmt == "yaml":
        if kind == "env_list":
            return [f"{p['indent']}- {key}={r}"], 0
        if kind == "name_value":
            di = site["dash_indent"]
            return [f"{di}- name: {key}", f"{di}  value: {r}"], 1
        return [f"{sibling_indent(site)}{key}: {r}"], 0
    ind = sibling_indent(site)
    if fmt == "hcl":
        return [f"{ind}{key.replace('.', '_')} = {r}"], 0
    if fmt == "json":
        return [f'{ind}"{key}": {r},'], 0
    if fmt == "env":
        return [f"{ind}{p['export']}{key.upper()}={r}"], 0
    tail = p["tail"].strip()
    if p["sep"] == ":":
        return [f"{ind}{key}: {r}{',' if tail.startswith(',') else ''}"], 0
    return [f"{ind}{p.get('decl', '')}{key} = {r}{';' if tail.startswith(';') else ''}"], 0


def make_tag(val, op, key, span_id, commented=False, pair_id="", variant=""):
    return dict(label=val["label"], category=val["category"], secret_type=val.get("secret_type", ""), operator=op,
                key_name=key, key_hint="sensitive" if S.is_sensitive_key(key) else "neutral",
                encoding=val.get("encoding", "plain"), commented=int(commented), span_id=span_id,
                needle=val.get("needle") or val["text"], text=val["text"], pair_id=pair_id, variant=variant)


def place(ctx, site, mode, key, val, op, pair_id="", variant=""):
    """Replace the value on site['idx'] or insert a sibling line after it. Returns True."""
    fmt, p = ctx.fmt, site["p"]
    sid = ctx.new_span()
    if mode == "replace":
        i = site["idx"]
        line = ctx.lines[i]
        ctx.lines[i] = line[:p["vs"]] + render_value(fmt, val, site) + line[p["ve"]:]
        ctx.tags[i] = make_tag(val, op, site["key"], sid, pair_id=pair_id, variant=variant)
        return True
    new, vi = build_insert(ctx, site, key, val)
    tags = [None] * len(new)
    tags[vi] = make_tag(val, op, key, sid, pair_id=pair_id, variant=variant)
    ctx.splice(site["idx"] + 1, 0, new, tags)
    return True


def single_val(g, quoted=True, encoding="plain"):
    return dict(text=g["text"], needle=g.get("inner") or g["text"], quoted=quoted, label="secret",
                category=g["category"], secret_type=g["secret_type"], encoding=encoding)


def neg_val(text, category, quoted, stype=""):
    return dict(text=text, needle=text, quoted=quoted, label="benign", category=category, secret_type=stype)


def key_style(ctx, key):
    return key.upper() if ctx.fmt == "env" else key


def choose_site_and_key(ctx, rng, prefer_sensitive=0.6):
    """Returns (site, mode, key) using an existing sensitive line when possible."""
    site = pick_site(ctx, "sensitive") if rng.random() < prefer_sensitive else None
    if site:
        return site, "replace", site["key"]
    site = pick_site(ctx, "insert")
    return (site, "insert", None) if site else (None, None, None)


# ============================================================================ operators
def op_inject_secret(ctx):
    rng = ctx.rng
    site, mode, key = choose_site_and_key(ctx, rng)
    if not site:
        return False
    if mode == "replace":
        t = rng.choice(S.types_for_key(key))
    else:
        t = rng.choice(S.SINGLE_LINE_TYPES + ["connection_string", "private_key_pem"])
        key = key_style(ctx, S.key_for_type(rng, t))
    if t == "private_key_pem":
        return op_pem(ctx, site=site, mode=mode, key=key)
    g = S.generate(t, rng)
    return place(ctx, site, mode, key, single_val(g), "inject_secret")


def _neg_op(ctx, kind, op):
    rng = ctx.rng
    site, mode, key = choose_site_and_key(ctx, rng, prefer_sensitive=0.7)
    if not site:
        return False
    if mode == "insert":
        key = key_style(ctx, S.key_for_type(rng, rng.choice(S.SINGLE_LINE_TYPES)))
    if kind == "reference":
        fmt_key = "code" if ctx.fmt in CODE_FMTS else ctx.fmt
        text, quoted = S.reference(fmt_key, key, rng, ctx.family)
        val = neg_val(text, "reference", quoted)
    else:
        val = neg_val(S.placeholder(rng), "placeholder", True)
    return place(ctx, site, mode, key, val, op)


def op_to_reference(ctx):
    return _neg_op(ctx, "reference", "to_reference")


def op_to_placeholder(ctx):
    return _neg_op(ctx, "placeholder", "to_placeholder")


def op_rename_key(ctx):
    """Secret under a neutral key name: tests how much a scanner leans on key-name context."""
    rng = ctx.rng
    site = pick_site(ctx, "insert")
    if not site:
        return False
    t = rng.choice(S.SINGLE_LINE_TYPES + ["connection_string"])
    key = key_style(ctx, rng.choice(S.NEUTRAL_KEYS))
    return place(ctx, site, "insert", key, single_val(S.generate(t, rng)), "rename_key")


def op_base64(ctx):
    if ctx.fmt not in ("yaml", "json", "env"):
        return False
    rng = ctx.rng
    site = pick_site(ctx, "insert")
    if not site:
        return False
    t = rng.choice(["db_password_random", "generic_token_urlsafe", "generic_api_key_hex", "aws_secret_access_key",
                    "custom_token", "github_pat", "db_password_human"])
    g = S.generate(t, rng)
    enc = base64.b64encode(g["text"].encode()).decode()
    val = dict(text=enc, needle=enc, quoted=True, label="secret", category=g["category"], secret_type=t,
               encoding="base64")
    key = key_style(ctx, rng.choice(["password", "token", "api_key", "secret", "db-password", "DB_PASSWORD", "tls_secret"]))
    return place(ctx, site, "insert", key, val, "base64")


def op_pem(ctx, site=None, mode=None, key=None):
    fmt, rng = ctx.fmt, ctx.rng
    if fmt in CODE_FMTS:
        return False
    if site is None:
        site, mode, key = choose_site_and_key(ctx, rng, prefer_sensitive=0.4)
        if not site:
            return False
        if mode == "insert":
            key = key_style(ctx, rng.choice(S.SECRET_TYPES["private_key_pem"][2]))
    lines = S.gen_private_key_pem(rng)
    if fmt in ("json", "env"):
        text = "\\n".join(lines) + "\\n"
        val = dict(text=text, needle=text, quoted=True, label="secret", category="secret_known_format",
                   secret_type="private_key_pem", encoding="escaped_newlines")
        return place(ctx, site, mode, key, val, "pem")
    sid = ctx.new_span()
    mk = lambda ln: make_tag(dict(text=ln, needle=ln, label="secret", category="secret_known_format",
                                  secret_type="private_key_pem"), "pem", key, sid)
    p = site["p"]
    if fmt == "yaml" and site["kind"] == "assign":
        ind = sibling_indent(site) if mode == "insert" else p["indent"] + " " * len(p["dash"])
        head = f"{ind}{key}: |"
        body = [f"{ind}  {ln}" for ln in lines]
    elif fmt == "hcl":
        ind = sibling_indent(site) if mode == "insert" else p["indent"]
        k = key.replace(".", "_")
        head = f"{ind}{k} = <<-EOT"
        body = [f"{ind}  {ln}" for ln in lines] + [f"{ind}EOT"]
    else:
        return False
    new = [head] + body
    tags = [None] + [mk(ln) for ln in lines] + ([None] if fmt == "hcl" else [])
    if mode == "replace":
        ctx.splice(site["idx"], 1, new, tags)
    else:
        ctx.splice(site["idx"] + 1, 0, new, tags)
    return True


def op_conn_string(ctx):
    rng = ctx.rng
    sites = enumerate_sites(ctx)
    cand = [s for s in sites if s["replaceable"] and S.types_for_key(s["key"]) == ["connection_string"]]
    if cand and rng.random() < 0.7:
        site, mode, key = rng.choice(cand), "replace", None
        key = site["key"]
    else:
        site = pick_site(ctx, "insert")
        if not site:
            return False
        mode, key = "insert", key_style(ctx, S.key_for_type(rng, "connection_string"))
    return place(ctx, site, mode, key, single_val(S.generate("connection_string", rng)), "conn_string")


def op_commented(ctx):
    fmt, rng = ctx.fmt, ctx.rng
    if fmt == "json":
        return False
    site = pick_site(ctx, "insert")
    if not site:
        return False
    t = rng.choice(S.SINGLE_LINE_TYPES)
    g = S.generate(t, rng)
    key = key_style(ctx, S.key_for_type(rng, t))
    ind = sibling_indent(site)
    cc = "//" if fmt in ("groovy", "ts") else "#"
    if fmt == "yaml":
        line = f'{ind}{cc} {key}: "{g["text"]}"'
    elif fmt == "hcl":
        line = f'{ind}{cc} {key.replace(".", "_")} = "{g["text"]}"'
    elif fmt == "env":
        line = f"{ind}{cc} {key}={g['text']}"
    else:
        line = f"{ind}{cc} {key} = '{g['text']}'"
    ctx.splice(site["idx"] + 1, 0, [line], [make_tag(single_val(g), "commented", key, ctx.new_span(), commented=True)])
    return True


def op_trap(ctx):
    rng = ctx.rng
    site = pick_site(ctx, "insert")
    if not site:
        return False
    key, value, ttype = S.trap(rng)
    return place(ctx, site, "insert", key_style(ctx, key), neg_val(value, "trap", True, ttype), "trap")


def op_k8s_secret_doc(ctx):
    """Append a Kubernetes Secret document (base64 `data:` + plain `stringData:`)."""
    if ctx.fmt != "yaml" or ctx.family != "k8s":
        return False
    txt = ctx.text()
    if "apiVersion:" not in txt or "kind:" not in txt:
        return False
    rng = ctx.rng
    t1 = rng.choice(["db_password_random", "generic_token_urlsafe", "aws_secret_access_key", "custom_token"])
    t2 = rng.choice(S.SINGLE_LINE_TYPES)
    g1, g2 = S.generate(t1, rng), S.generate(t2, rng)
    enc = base64.b64encode(g1["text"].encode()).decode()
    k1 = rng.choice(["password", "token", "api-key", "DB_PASSWORD", "secret-key"])
    k2 = S.key_for_type(rng, t2)
    name = rng.choice(["app", "db", "api", "registry", "backend"]) + "-" + rng.choice(["credentials", "secrets", "creds"])
    v1 = dict(text=enc, needle=enc, label="secret", category=g1["category"], secret_type=t1, encoding="base64")
    new = ["---", "apiVersion: v1", "kind: Secret", "metadata:", f"  name: {name}", "type: Opaque", "data:",
           f"  {k1}: {enc}", "stringData:", f'  {k2}: "{g2["text"]}"']
    tags = [None] * 7 + [make_tag(v1, "k8s_secret_doc", k1, ctx.new_span()), None,
                         make_tag(single_val(g2), "k8s_secret_doc", k2, ctx.new_span())]
    at = len(ctx.lines) - 1 if ctx.lines and ctx.lines[-1] == "" else len(ctx.lines)
    ctx.splice(at, 0, new, tags)
    return True


# name -> (fn, weight, fmts or None)
OPERATORS = {
    "inject_secret": (op_inject_secret, 3.0, None),
    "to_reference": (op_to_reference, 1.5, None),
    "to_placeholder": (op_to_placeholder, 1.5, None),
    "rename_key": (op_rename_key, 1.5, None),
    "base64": (op_base64, 1.0, ("yaml", "json", "env")),
    "pem": (op_pem, 0.7, ("yaml", "hcl", "json", "env")),
    "conn_string": (op_conn_string, 1.0, None),
    "commented": (op_commented, 1.0, None),
    "trap": (op_trap, 2.0, None),
    "k8s_secret_doc": (op_k8s_secret_doc, 0.8, ("yaml",)),
}


# ============================================================================ validation gate
def validate(fmt, before_unbalanced, new_text, ctx):
    ok, parsed = parse_ok(fmt, new_text)
    if not ok:
        return False, "parse_fail_after"
    if fmt in ("yaml", "json", "hcl") and parsed is not None:
        for t in ctx.tags:
            if t and not t["commented"] and "\n" not in t["text"]:
                if not contains_string(parsed, t["text"]):
                    return False, "value_not_reachable"
    return True, ""


def finalize_rows(ctx, meta):
    """Compute (line_no, span) rows from the final line order."""
    rows = []
    for i, t in enumerate(ctx.tags):
        if not t:
            continue
        ln = ctx.lines[i]
        s = ln.find(t["needle"])
        s, e = (s, s + len(t["needle"])) if s >= 0 else (0, len(ln))
        rows.append(dict(meta, line_no=i + 1, label=t["label"], category=t["category"], secret_type=t["secret_type"],
                         operator=t["operator"], key_name=t["key_name"], key_hint=t["key_hint"],
                         encoding=t["encoding"], commented=t["commented"], span_id=t["span_id"],
                         span_start=s, span_end=e, pair_id=t["pair_id"], variant=t["variant"]))
    return rows


# ============================================================================ per-file processing
def pick_ops(rng, fmt, family):
    names = [n for n, (_, w, f) in OPERATORS.items() if (f is None or fmt in f) and (n != "k8s_secret_doc" or family == "k8s")]
    weights = [OPERATORS[n][1] for n in names]
    k = rng.choices([1, 2, 3, 4], weights=[2, 4, 3, 1])[0]
    return rng.choices(names, weights=weights, k=k)


def mutate_normal(text, fmt, family, rng):
    reason = "no_site"
    for _ in range(3):
        ctx = Ctx(fmt, text, rng, family)
        applied = sum(1 for n in pick_ops(rng, fmt, family) if OPERATORS[n][0](ctx))
        if not applied:
            continue
        ok, reason = validate(fmt, False, ctx.text(), ctx)
        if ok:
            return [("", ctx)], ""
    return None, reason


def mutate_pair(text, fmt, family, rng):
    """Contrastive triplet: the SAME slot rendered as secret / reference / placeholder."""
    base = Ctx(fmt, text, rng, family)
    site, mode, key = choose_site_and_key(base, rng, prefer_sensitive=0.5)
    if not site:
        return None, "no_site"
    if mode == "insert":
        t = rng.choice([x for x in S.SINGLE_LINE_TYPES if x != "db_password_human"] + ["connection_string"])
        key = key_style(base, S.key_for_type(rng, t))
    else:
        types = [x for x in S.types_for_key(key) if x != "private_key_pem"] or ["generic_token_urlsafe"]
        t = rng.choice(types)
    fmt_key = "code" if fmt in CODE_FMTS else fmt
    rtext, rq = S.reference(fmt_key, key, rng, family)
    vals = {"secret": single_val(S.generate(t, rng)), "reference": neg_val(rtext, "reference", rq),
            "placeholder": neg_val(S.placeholder(rng), "placeholder", True)}
    out = []
    for var, val in vals.items():
        c = Ctx(fmt, text, random.Random(rng.random()), family)
        place(c, site, mode, key, val, "pair_" + var, pair_id="P", variant=var)
        ok, reason = validate(fmt, False, c.text(), c)
        if not ok:
            return None, "pair_" + reason
        out.append((var, c))
    return out, ""


# ============================================================================ main
def walk_files(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            yield os.path.join(dp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-hits")
    ap.add_argument("--max-files-per-repo", type=int, default=20)
    ap.add_argument("--cap-override", default="", help="family=N,family=N (e.g. jenkins=40,pulumi=60)")
    ap.add_argument("--pair-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    caps = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.cap_override.split(",") if "=" in kv}
    split_of = {r["repo"]: r["split"] for r in csv.DictReader(open(a.splits))}
    manifest = list(csv.DictReader(open(a.manifest)))
    excluded = set()
    if a.exclude_hits and os.path.exists(a.exclude_hits):
        excluded = {(r["repo"], r["relpath"]) for r in csv.DictReader(open(a.exclude_hits))}

    os.makedirs(os.path.join(a.out, "files"), exist_ok=True)
    label_rows, index_rows = [], []
    drops, stats = Counter(), Counter()

    for m in manifest:
        repo, family, sha = m["repo"], m["family"], m["commit_sha"]
        rdir = os.path.join(a.repos, repo.replace("/", "__"))
        if not os.path.isdir(rdir):
            drops["repo_missing"] += 1
            continue
        if repo not in split_of:
            sys.exit(f"repo {repo} missing from splits file")
        split = split_of[repo]
        rng_repo = random.Random(f"{a.seed}:{repo}")
        elig = []
        for path in walk_files(rdir):
            rel = os.path.relpath(path, rdir)
            try:
                size = os.path.getsize(path)
                if size > MAX_BYTES or size < 100:
                    continue
                text = open(path, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            fmt = detect_fmt(rel, text)
            if fmt not in FAMILY_FMTS.get(family, []) or "\r" in text or text.count("\n") < MIN_LINES:
                continue
            if (repo, rel) in excluded:
                drops["excluded_preexisting_hit"] += 1
                continue
            if fmt in ("yaml", "json", "hcl"):
                ok, _ = parse_ok(fmt, text)
                if not ok:
                    drops["unparseable_before"] += 1
                    continue
            w = 3.0 if S.SENSITIVE_KEY_RE.search(text) else 1.0
            elig.append((rng_repo.random() ** (1.0 / w), rel, fmt, text))
        elig.sort(reverse=True)
        cap = caps.get(family, a.max_files_per_repo) if split == "test_ood" else a.max_files_per_repo
        done = 0
        for _, rel, fmt, text in elig:
            if done >= cap:
                break
            rng = random.Random(f"{a.seed}:{repo}:{rel}")
            if rng.random() < a.pair_frac:
                variants, reason = mutate_pair(text, fmt, family, rng)
            else:
                variants, reason = mutate_normal(text, fmt, family, rng)
            if not variants:
                drops[reason or "unknown"] += 1
                continue
            base_id = f"{repo.replace('/', '__')}__{sha[:8]}__{hashlib.sha1(rel.encode()).hexdigest()[:8]}"
            for var, ctx in variants:
                fid = base_id + (f"__{var}" if var else "")
                outp = os.path.join(a.out, "files", fid, rel)
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                open(outp, "w", encoding="utf-8").write(ctx.text())
                meta = dict(file_id=fid, repo=repo, family=family, split=split, relpath=rel, fmt=fmt)
                rows = finalize_rows(ctx, meta)
                for r in rows:
                    if r["pair_id"]:
                        r["pair_id"] = base_id
                label_rows += rows
                index_rows.append(dict(meta, n_lines=len(ctx.lines), n_labeled=len(rows),
                                       validated=int(fmt in ("yaml", "json", "hcl", "py") and (fmt != "hcl" or hcl2 is not None))))
                stats[(split, family)] += 1
            done += 1

    cols = ["file_id", "repo", "family", "split", "relpath", "fmt", "line_no", "label", "category", "secret_type",
            "operator", "key_name", "key_hint", "encoding", "commented", "span_id", "span_start", "span_end",
            "pair_id", "variant"]
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(label_rows)
    icols = ["file_id", "repo", "family", "split", "relpath", "fmt", "n_lines", "n_labeled", "validated"]
    with open(os.path.join(a.out, "files_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=icols)
        w.writeheader()
        w.writerows(index_rows)

    print(f"files written: {len(index_rows)}   labeled lines: {len(label_rows)}")
    print("files by split:", dict(Counter(r["split"] for r in index_rows)))
    print("files by family:", dict(Counter(r["family"] for r in index_rows)))
    print("labels by category:", dict(Counter(r["category"] for r in label_rows)))
    print("labels by operator:", dict(Counter(r["operator"] for r in label_rows)))
    print("drops:", dict(drops))


if __name__ == "__main__":
    main()
