import os
import typer
from typing import Optional, List, Dict
from fnmatch import fnmatch
from sqlalchemy import text

from automockai.schema import make_engine, get_schema
from automockai.generator import generate_data

app = typer.Typer(
    help="🚀 AutoMockAI – Schema-aware AI-powered mock data generator")

DEFAULT_DJANGO_EXCLUDES = [
    "django_*", "auth_*", "admin_*", "sqlite_*", "contenttypes_*",
    "django_content_type", "django_admin_log", "django_migrations",
    "django_session", "sessions_*",
]

# ----------------- Filters -----------------


def _split_csv_patterns(val: Optional[str]) -> List[str]:
    return [p.strip() for p in val.split(",") if p.strip()] if val else []


def _should_exclude(name: str, patterns: List[str]) -> bool:
    parts = [name, name.split(".")[-1]]
    return any(fnmatch(c, pat) for pat in patterns for c in parts)


def _filter_tables(full_schema, include_patterns, exclude_patterns, only_prefix, skip_django):
    tables = list(full_schema.keys())
    if only_prefix:
        tables = [t for t in tables if t.split(
            ".")[-1].startswith(only_prefix)]
    if skip_django:
        tables = [t for t in tables if not _should_exclude(
            t, DEFAULT_DJANGO_EXCLUDES)]
    if include_patterns:
        tables = [t for t in tables if any(fnmatch(t, p) or fnmatch(
            t.split(".")[-1], p) for p in include_patterns)]
    if exclude_patterns:
        tables = [t for t in tables if not _should_exclude(
            t, exclude_patterns)]
    return tables


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

# ----------------- C) FK-safe link tables -----------------


def is_link_table(tinfo: dict) -> bool:
    fks = tinfo.get("foreign_keys") or []
    cols = tinfo.get("columns") or []
    pk = set(tinfo.get("primary_key") or [])
    fk_cols = {c for fk in fks for c in fk.get("constrained_columns", [])}
    non_pk_non_fk = [c["name"] for c in cols if c["name"]
                     not in fk_cols and c["name"] not in pk]
    return len(fk_cols) >= 2 and len(non_pk_non_fk) == 0


def build_fk_link_insert(table: str, tinfo: dict, rows: int) -> str:
    fks = tinfo.get("foreign_keys") or []
    if not fks:
        return ""

    def fk_triplet(fk: dict):
        return fk["constrained_columns"][0], fk["referred_table"], fk["referred_columns"][0]
    if len(fks) >= 2:
        c1, rt1, rk1 = fk_triplet(fks[0])
        c2, rt2, rk2 = fk_triplet(fks[1])
        return (
            f'INSERT INTO {_q(table)} ({_q(c1)}, {_q(c2)})\n'
            f'SELECT p.{_q(rk1)}, e.{_q(rk2)}\n'
            f'FROM {_q(rt1)} p CROSS JOIN {_q(rt2)} e\n'
            f'ORDER BY random()\nLIMIT {rows};'
        )
    else:
        c1, rt1, rk1 = fk_triplet(fks[0])
        return (
            f'INSERT INTO {_q(table)} ({_q(c1)})\n'
            f'SELECT {_q(rk1)} FROM {_q(rt1)} ORDER BY random() LIMIT {rows};'
        )

# ----------------- D) Parent-first topo sort -----------------


def topo_sort_tables(full_schema: Dict[str, dict], selected: List[str]) -> List[str]:
    graph = {t: set() for t in selected}
    for t in selected:
        for fk in full_schema[t].get("foreign_keys") or []:
            parent = fk.get("referred_table")
            if parent and parent in graph and parent != t:
                graph[t].add(parent)
    incoming = {t: set(parents) for t, parents in graph.items()}
    no_incoming = [t for t, parents in incoming.items() if not parents]
    order = []
    while no_incoming:
        n = no_incoming.pop()
        order.append(n)
        for m, parents in incoming.items():
            if n in parents:
                parents.remove(n)
                if not parents:
                    no_incoming.append(m)
    return order + [t for t in selected if t not in order]

# ----------------- CLI command -----------------


@app.command()
def generate(
    dsn: str = typer.Option(..., help="SQLAlchemy DSN"),
    rows: int = typer.Option(20, help="Rows per table"),
    model: str = typer.Option("mistral", help="Ollama model name"),
    dry_run: bool = typer.Option(True, help="Preview only"),
    execute: bool = typer.Option(False, help="Actually execute inserts"),
    include: Optional[str] = typer.Option(None, help="Include patterns"),
    exclude: Optional[str] = typer.Option(None, help="Exclude patterns"),
    only_prefix: Optional[str] = typer.Option(
        None, help="Only tables with prefix"),
    skip_django: bool = typer.Option(True, help="Skip Django internal tables"),
    ollama_host: Optional[str] = typer.Option(
        None, help="Ollama host, e.g. http://localhost:11434"),
    debug_llm: bool = typer.Option(False, help="Print raw LLM I/O"),
):
    typer.echo("🚀 AutoMockAI starting...")
    if debug_llm:
        os.environ["AUTOMOCKAI_DEBUG"] = "1"

    engine = make_engine(dsn)
    full_schema = get_schema(engine)

    typer.echo(f"✅ Connected to DB, found {len(full_schema)} tables:")
    for t, info in full_schema.items():
        typer.echo(f"  • {t} ({len(info['columns'])} columns)")

    selected = _filter_tables(full_schema,
                              _split_csv_patterns(include),
                              _split_csv_patterns(exclude),
                              only_prefix,
                              skip_django)

    if not selected:
        typer.echo("⚠️  No tables selected after filters.")
        raise typer.Exit(code=1)

    selected = topo_sort_tables(full_schema, selected)
    typer.echo("\n🧩 Tables selected for generation:")
    for t in selected:
        typer.echo(f"  • {t}")

    for table in selected:
        typer.echo(f"\n📌 Generating data for: {table} (rows≈{rows})")
        tinfo = full_schema[table]

        if is_link_table(tinfo):
            typer.echo("ℹ️  Detected link table, generating FK-safe inserts.")
            sql = build_fk_link_insert(table, tinfo, rows)
        else:
            sql = generate_data(table, tinfo, rows,
                                model=model, ollama_host=ollama_host)

        if dry_run and not execute:
            typer.echo("✅ SQL preview:")
            preview = sql[:1600] + \
                "\n-- ...truncated..." if len(sql) > 1600 else sql
            typer.echo(preview if preview else "-- (no SQL) --")
            continue

        if execute:
            if not sql.strip():
                typer.echo("⚠️  No SQL to execute.")
                continue
            try:
                with engine.begin() as conn:
                    for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                        conn.execute(text(stmt))
                typer.echo("✅ Committed inserts for table.")
            except Exception as e:
                typer.echo(f"❌ Failed executing inserts for {table}: {e}")


def main():
    app()


if __name__ == "__main__":
    main()
