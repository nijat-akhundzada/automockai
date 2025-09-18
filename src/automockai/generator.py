import os
import re
import json
import random
from typing import Dict, Any, List, Optional, Tuple, Set
from sqlalchemy.engine import Engine
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__) 

try:
    import ollama
except ImportError:
    ollama = None

try:
    from automockai.fallback import FallbackGenerator
except ImportError:
    FallbackGenerator = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    AutoTokenizer, AutoModelForCausalLM, pipeline = None, None, None

# --- Constants ---
FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\n?|```", re.MULTILINE)
ISO_TS_RE = re.compile(r"'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2}(?:\.\d+)?)(Z?)")
STRING_RE = re.compile(r"'([^']*)'")
CONCAT_QQ_RE = re.compile(r"('(?:[^']|'')*')\s*\|\|\s*('(?:[^']|'')*')")
CONCAT_QI_RE = re.compile(r"('(?:[^']|'')*')\s*\|\|\s*([A-Za-z][A-Za-z0-9_]*(?:''[A-Za-z0-9_]+)?)")
LOCAL_MODEL_DIR = "model/trained_model"


class DataGenerator:
    """
    Data Generator Agent - Generates rows based on schema.
    - If column is a foreign key → reference already inserted rows
    - If field is semantic (name, email, salary, date) → call AI model
    - Otherwise → fallback to Faker
    - Must enforce constraints (uniqueness, business rules)
    """
    
    def __init__(self, engine: Engine, schema_info: Dict[str, Any],
                 use_local_model: bool = True, use_ollama_fallback: bool = True):
        self.engine = engine
        self.schema_info = schema_info
        self.existing_data_cache = {}
        self.generated_values_cache = {}
        self.fallback_generator = FallbackGenerator() if FallbackGenerator else None
        self.use_ollama_fallback = use_ollama_fallback # Use the passed argument
        
        # AI model placeholders
        self.local_model = None
        self.local_tokenizer = None
        self.text_generator = None
        
        if use_local_model: # Only load local model if explicitly requested
            self._load_local_model()

    def _load_local_model(self):
        """Load the fine-tuned model from the local directory if it exists."""
        if pipeline is None or not os.path.exists(LOCAL_MODEL_DIR):
            logger.info("Local model not found or transformers library not installed. Using fallback.")
            return
            
        try:
            logger.info(f"Loading local model from {LOCAL_MODEL_DIR}...")
            self.local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
            self.local_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR)
            self.text_generator = pipeline(
                "text-generation",
                model=self.local_model,
                tokenizer=self.local_tokenizer,
                max_new_tokens=50,
            )
            logger.info("✅ Local model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self.text_generator = None

    def get_existing_foreign_key_values(self, table_name: str, column_name: str) -> List[Any]:
        """Get existing values from a foreign key referenced table."""
        cache_key = f"{table_name}.{column_name}"
        if cache_key in self.existing_data_cache:
            return self.existing_data_cache[cache_key]
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f'SELECT DISTINCT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL'))
                values = [row[0] for row in result.fetchall()]
                self.existing_data_cache[cache_key] = values
                return values
        except Exception as e:
            logger.warning(f"Failed to get existing FK values for {table_name}.{column_name}: {e}")
            return []
    
    def generate_foreign_key_value(self, fk_info: Dict[str, Any]) -> Optional[Any]:
        """Generate a valid foreign key value by selecting from referenced table."""
        referred_table = fk_info["referred_table"]
        referred_column = fk_info["referred_columns"][0] if fk_info["referred_columns"] else "id"
        
        # 1. Check newly generated values first
        cache_key = f"{referred_table}.{referred_column}"
        if cache_key in self.generated_values_cache and self.generated_values_cache[cache_key]:
            return random.choice(self.generated_values_cache[cache_key])

        # 2. Fallback to existing values in the DB
        existing_values = self.get_existing_foreign_key_values(referred_table, referred_column)
        if existing_values:
            return random.choice(existing_values)
        return None
    
    def check_unique_constraint(self, table_name: str, column_name: str, value: Any) -> bool:
        """Check if a value violates unique constraints."""
        cache_key = f"{table_name}.{column_name}.unique"
        if cache_key not in self.existing_data_cache:
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(f'SELECT DISTINCT "{column_name}" FROM "{table_name}"'))
                    existing_values = {row[0] for row in result.fetchall()}
                    self.existing_data_cache[cache_key] = existing_values
            except Exception:
                self.existing_data_cache[cache_key] = set()
        
        return value not in self.existing_data_cache[cache_key]

    def generate_semantic_value(self, column_info: Dict[str, Any], semantic_type: str, 
                              constraints: Dict[str, Any] = None) -> Any:
        """Generate semantically appropriate values using a tiered AI approach."""
        # 1. Try local fine-tuned model first
        if self.text_generator:
            try:
                prompt = f"field: {column_info['name']} type: {column_info['type']} value:"
                generated_text = self.text_generator(prompt)[0]['generated_text']
                
                # Extract the value part
                value_part = generated_text.split("value:")[-1].strip()
                if value_part:
                    parsed_value = self._parse_semantic_response(value_part, column_info, constraints)
                    if parsed_value is not None:
                        return parsed_value
            except Exception as e:
                logger.warning(f"Local model generation failed: {e}")

        # 2. Fallback to Ollama if available
        if self.use_ollama_fallback and ollama:
            try:
                prompt = self._build_ollama_prompt(column_info, semantic_type, constraints)
                client = ollama.Client()
                resp = client.chat(
                    model="mistral",
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.7},
                )
                raw_response = resp["message"]["content"].strip()
                parsed_value = self._parse_semantic_response(raw_response, column_info, constraints)
                if parsed_value is not None:
                    return parsed_value
            except Exception as e:
                logger.warning(f"Ollama generation failed: {e}")

        # 3. Final fallback to Faker
        return self._fallback_semantic_value(column_info, semantic_type, constraints)

    def _build_ollama_prompt(self, column_info: Dict[str, Any], semantic_type: str, 
                             constraints: Dict[str, Any] = None) -> str:
        """Build AI prompt for Ollama semantic data generation."""
        column_name = column_info["name"]
        column_type = column_info["type"]
        
        base_prompt = f'''Generate a realistic {semantic_type} value for a database column.
Column: "{column_name}"
SQL Type: {column_type}

Return ONLY the raw value, without quotes or explanations.'''

        if constraints and "length" in constraints:
            base_prompt += f"\n- Max length: {constraints['length']} characters"
        
        return base_prompt

    def _parse_semantic_response(self, response: str, column_info: Dict[str, Any], constraints: Dict[str, Any] = None) -> Any:
        """Parse AI response into appropriate Python type."""
        # Clean the response, remove extra text
        response = response.split('\n')[0].strip().strip('"\'')
        
        column_type_lower = column_info["type"].lower()
        
        if 'int' in column_type_lower:
            try:
                return int(float(re.search(r'-?\d+', response).group(0)))
            except (ValueError, AttributeError):
                return None
        elif any(t in column_type_lower for t in ['float', 'decimal', 'numeric']):
            try:
                val = float(re.search(r'-?\d*\.?\d+', response).group(0))
                precision = column_info.get("precision")
                scale = column_info.get("scale")
                if precision and scale is not None:
                    max_val = (10 ** (precision - scale))
                    if abs(val) >= max_val:
                        return None # Out of range
                return val
            except (ValueError, AttributeError):
                return None
        elif 'bool' in column_type_lower:
            return response.lower() in ('true', '1', 'yes', 'on')
        elif 'date' in column_type_lower or 'time' in column_type_lower:
            try:
                # Try to parse a date/time from the beginning of the string
                match = re.search(r'\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?', response)
                if match:
                    return match.group(0)
                else:
                    return None
            except (ValueError, AttributeError):
                return None
        else:
            if constraints and constraints.get("length"):
                if len(response) > constraints.get("length"):
                    return None # Too long
            return response
    
    def _fallback_semantic_value(self, column_info: Dict[str, Any], semantic_type: str, 
                                constraints: Dict[str, Any] = None) -> Any:
        """Generate semantic value using fallback generator."""
        if self.fallback_generator:
            return self.fallback_generator.generate_semantic_value(
                column_info, semantic_type, constraints
            )
        
        # Basic fallback without Faker
        if semantic_type == "email":
            return f"user{random.randint(1000, 9999)}@example.com"
        elif semantic_type == "phone":
            return f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        elif semantic_type == "name":
            names = ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown"]
            return random.choice(names)
        elif semantic_type == "money":
            return round(random.uniform(10.0, 1000.0), 2)
        elif semantic_type == "datetime":
            return "2024-01-01 12:00:00"
        elif semantic_type == "address":
            return f"{random.randint(100, 999)} Main St"
        elif semantic_type == "url":
            return f"https://example{random.randint(1, 100)}.com"
        else:
            return f"sample_{semantic_type}_{random.randint(1, 1000)}"
    
    def generate_row_data(self, table_name: str, count: int = 1) -> List[Dict[str, Any]]:
        """Generate row data for a table respecting all constraints."""
        table_info = self.schema_info["tables"][table_name]
        columns = table_info["columns"]
        foreign_keys = {fk["constrained_columns"][0]: fk for fk in table_info.get("foreign_keys", [])}
        semantic_columns = table_info.get("semantic_columns", {})
        unique_constraints = table_info.get("unique_constraints", [])
        pk_columns = set(table_info.get("primary_key", {}).get("constrained_columns", []))
        
        generated_rows = []
        
        for _ in range(count):
            row_data = {}
            
            for column in columns:
                col_name = column["name"]
                col_type = column["type"]
                
                # Skip auto-increment primary keys
                if (
                    col_name in pk_columns
                    and column.get("autoincrement", False)
                    and any(kw in col_type.lower() for kw in ['serial', 'identity', 'autoincrement'])
                ):
                    continue
                
                # Handle foreign keys
                if col_name in foreign_keys:
                    fk_value = self.generate_foreign_key_value(foreign_keys[col_name])
                    if fk_value is not None:
                        row_data[col_name] = fk_value
                    elif not column["nullable"]:
                        logger.warning(f"No FK values available for required column {col_name}")
                        pass
                
                # Handle semantic columns
                elif col_name in semantic_columns:
                    semantic_type = semantic_columns[col_name]
                    constraints = {
                        "length": column.get("length"),
                        "check_constraints": [cc["sqltext"] for cc in table_info.get("check_constraints", [])]
                    }
                    
                    value = self.generate_semantic_value(column, semantic_type, constraints)
                    
                    # Ensure uniqueness if required
                    if any(col_name in uc["column_names"] for uc in unique_constraints):
                        attempts = 0
                        while not self.check_unique_constraint(table_name, col_name, value) and attempts < 10:
                            value = self.generate_semantic_value(column, semantic_type, constraints)
                            attempts += 1
                    
                    row_data[col_name] = value
                
                # Handle regular columns with fallback
                else:
                    if self.fallback_generator:
                        value = self.fallback_generator.generate_column_value(column)
                    else:
                        value = self._basic_fallback_value(column)
                    
                    # Ensure uniqueness if required
                    if any(col_name in uc["column_names"] for uc in unique_constraints):
                        attempts = 0
                        while not self.check_unique_constraint(table_name, col_name, value) and attempts < 10:
                            if self.fallback_generator:
                                value = self.fallback_generator.generate_column_value(column)
                            else:
                                value = self._basic_fallback_value(column)
                            attempts += 1
                    
                    row_data[col_name] = value
            
            generated_rows.append(row_data)

            # Cache generated primary key values
            if pk_columns:
                for pk_col in pk_columns:
                    if pk_col in row_data:
                        cache_key = f"{table_name}.{pk_col}"
                        if cache_key not in self.generated_values_cache:
                            self.generated_values_cache[cache_key] = []
                        self.generated_values_cache[cache_key].append(row_data[pk_col])

        return generated_rows
    
    def _basic_fallback_value(self, column: Dict[str, Any]) -> Any:
        """Basic fallback value generation without external libraries."""
        col_type = column["type"].lower()
        
        if 'int' in col_type or 'serial' in col_type:
            return random.randint(1, 1000)
        elif 'float' in col_type or 'decimal' in col_type or 'numeric' in col_type:
            return round(random.uniform(1.0, 100.0), 2)
        elif 'bool' in col_type:
            return random.choice([True, False])
        elif 'date' in col_type or 'time' in col_type:
            return "2024-01-01 12:00:00"
        else:
            return f"sample_text_{random.randint(1, 1000)}"


# ======================================================
#                 PROMPT CONSTRUCTION (Legacy)
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

    return f'''
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
'''


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
    cols_sql = ", ".join(f'\"{c}\"' for c in cols)

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

    return f'INSERT INTO \"{table}\" ({cols_sql}) VALUES\n  ' + ",\n  ".join(values) + ";"


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