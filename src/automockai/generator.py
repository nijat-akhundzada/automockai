import os
import re
from typing import Dict, Any, List, Optional

try:
    import ollama
except Exception:
    ollama = None

try:
    from automockai.fallback import generate_fake_rows
except Exception:
    generate_fake_rows = None

FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\n?|```", re.MULTILINE)
# regex for ISO8601 timestamp inside single quotes
ISO_TS_RE = re.compile(
    r"'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2}(?:\.\d+)?)(Z?)'")

# regex to find string literals
STRING_RE = re.compile(r"'([^']*)'")

# 'left' || 'right'
CONCAT_QQ_RE = re.compile(r"('(?:[^']|'')*')\s*\|\|\s*('(?:[^']|'')*')")

# 'left' || identifier (e.g., O''Reilly, SomeWord123)  -> we wrap RHS as a literal and merge
CONCAT_QI_RE = re.compile(
    r"('(?:[^']|'')*')\s*\|\|\s*([A-Za-z][A-Za-z0-9_]*(?:''[A-Za-z0-9_]+)?)")

# ======================================================
#                 PROMPT CONSTRUCTION
# ======================================================


def build_prompt(table: str, tinfo: Dict[str, Any], rows: int) -> str:
    cols = [c["name"] + ":" + str(c["type"]).upper() for c in tinfo["columns"]]
    pk_cols = tinfo.get("primary_key") or []

    auto_pk_omit, must_include = [], []
    for c in tinfo["columns"]:
        cname, ctype = c["name"], str(c["type"]).upper()
        if cname in pk_cols and any(k in ctype for k in ["INT", "BIGINT", "SMALLINT", "SERIAL", "IDENTITY"]):
            auto_pk_omit.append(cname)
        elif not c.get("nullable", True) and cname not in pk_cols:
            must_include.append(f"{cname} ({ctype})")

    return f"""
You are a senior data engineer. Output ONLY raw SQL INSERT statements for PostgreSQL.
Target table: "{table}"
Columns (with types): {cols}
Primary key(s): {pk_cols}
Required NOT NULL columns: {must_include}
Columns to OMIT (likely autoincrement/serial): {auto_pk_omit}
Rules:
- Generate ~{rows} rows as multi-row INSERT statements (≤200 VALUES per statement).
- STRICTLY respect column types:
  - INTEGER/BIGINT/SMALLINT → plain integers only.
  - BOOLEAN → TRUE or FALSE.
  - TIMESTAMP/DATE → NOW() or realistic ISO dates.
  - VARCHAR/TEXT → quoted strings.
- Escape single quotes with two (O''Reilly).
- Never add BEGIN/COMMIT or comments.
- Only list valid columns; number of values must match columns.
Return: ONLY valid SQL INSERTs for "{table}", each ending with semicolon.
""".strip()


# ======================================================
#                 OUTPUT SANITIZATION
# ======================================================

def _merge_two_quoted(m: re.Match) -> str:
    left = m.group(1)[1:-1]   # content without the outer quotes
    right = m.group(2)[1:-1]
    merged = left + right
    return "'" + merged + "'"


def _merge_quoted_identifier(m: re.Match) -> str:
    left = m.group(1)[1:-1]
    ident = m.group(2)
    # merge with a space between, and re-escape
    merged = (left + " " + ident).replace("'", "''")
    return "'" + merged + "'"


def sanitize_sql_output(raw: str) -> str:
    if not raw:
        return ""
    # 1) strip fences
    s = FENCE_RE.sub("", raw).strip()

    # 2) cut to first INSERT
    idx = s.upper().find("INSERT INTO")
    if idx > 0:
        s = s[idx:]

    inserts, buf, capturing = [], [], False
    for line in s.splitlines():
        if "INSERT INTO" in line.upper():
            capturing = True
        if capturing:
            buf.append(line)
            if line.strip().endswith(";"):
                stmt = "\n".join(buf).strip()
                buf, capturing = [], False

                # fix timestamps
                stmt = ISO_TS_RE.sub(r"'\1 \2'", stmt)

                # escape apostrophes inside literals (double any single quotes)
                def _escape_quotes(m):
                    inner = m.group(1).replace("'", "''")
                    return f"'{inner}'"
                stmt = STRING_RE.sub(_escape_quotes, stmt)

                # merge 'left' || 'right'
                while CONCAT_QQ_RE.search(stmt):
                    stmt = CONCAT_QQ_RE.sub(_merge_two_quoted, stmt)

                # merge 'left' || IDENT  (wrap RHS as literal and merge)
                while CONCAT_QI_RE.search(stmt):
                    stmt = CONCAT_QI_RE.sub(_merge_quoted_identifier, stmt)

                inserts.append(stmt)

    return "\n\n".join(inserts).strip()
# ======================================================
#             FAKER → SQL FALLBACK HELPERS
# ======================================================


def _faker_sql(table: str, tinfo: Dict[str, Any], rows: int) -> str:
    if generate_fake_rows is None:
        return ""
    try:
        data = generate_fake_rows(table, tinfo, rows)
    except Exception:
        return ""
    if not data:
        return ""
    cols = sorted({k for r in data for k in r.keys()})
    cols_sql = ", ".join(f'"{c}"' for c in cols)

    values = []
    for r in data:
        vals = []
        for c in cols:
            v = r.get(c)
            if v is None:
                vals.append("NULL")
            elif isinstance(v, bool):
                vals.append("TRUE" if v else "FALSE")
            elif isinstance(v, (int, float)):
                vals.append(str(v))
            else:
                s = str(v).replace("'", "''")   # ✅ fix
                vals.append(f"'{s}'")
        values.append("(" + ", ".join(vals) + ")")

    return f'INSERT INTO "{table}" ({cols_sql}) VALUES\n  ' + ",\n  ".join(values) + ";"


# ======================================================
#                 OLLAMA PROVIDER
# ======================================================

def generate_ollama_sql(
    table: str,
    tinfo: Dict[str, Any],
    rows: int,
    model: str = "mistral",
    host: Optional[str] = None,
) -> str:
    if ollama is None:
        return ""
    prompt = build_prompt(table, tinfo, rows)
    try:
        client = ollama.Client(host=host) if host else ollama.Client()
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        raw = (resp["message"]["content"] or "").strip()
        if os.getenv("AUTOMOCKAI_DEBUG") == "1":
            print("\n===== DEBUG PROMPT =====")
            print(prompt[:1500])
            print("\n===== DEBUG RAW OUTPUT =====")
            print(raw[:1500])
        return sanitize_sql_output(raw)
    except Exception as e:
        if os.getenv("AUTOMOCKAI_DEBUG") == "1":
            print(f"⚠️ Ollama failed: {e}")
        return ""


# ======================================================
#                 PUBLIC ENTRYPOINT
# ======================================================

def generate_data(
    table: str,
    tinfo: Dict[str, Any],
    rows: int = 10,
    model: str = "mistral",
    ollama_host: Optional[str] = None,
) -> str:
    """
    Try Ollama first, fallback to Faker if unavailable or invalid.
    Always returns a SQL string.
    """
    sql = ""
    if ollama is not None:
        sql = generate_ollama_sql(
            table, tinfo, rows, model=model, host=ollama_host)
    if not sql.strip():
        sql = _faker_sql(table, tinfo, rows)
    return sql
