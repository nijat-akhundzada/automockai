from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from typing import Dict, Any

def make_engine(dsn: str) -> Engine:
    """
    Create a SQLAlchemy engine from a DSN and test connectivity.
    """
    engine = create_engine(dsn, future=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine

def get_schema(engine: Engine) -> Dict[str, Any]:
    """
    Reflect the schema and return a dict with tables and columns.
    """
    insp = inspect(engine)
    schema_info = {}

    for table_name in insp.get_table_names():
        cols = []
        for col in insp.get_columns(table_name):
            cols.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col["nullable"],
                "default": str(col["default"]) if col["default"] else None,
            })

        schema_info[table_name] = {
            "columns": cols,
            "primary_key": insp.get_pk_constraint(table_name).get("constrained_columns"),
            "foreign_keys": insp.get_foreign_keys(table_name),
        }

    return schema_info
