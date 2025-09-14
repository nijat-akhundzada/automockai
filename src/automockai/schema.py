from sqlalchemy import create_engine, inspect, text, MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Any, List, Optional, Set
import logging

logger = logging.getLogger(__name__)

class SchemaAnalyzer:
    """
    Schema Analyzer Agent - Uses SQLAlchemy reflection to extract tables, columns, 
    PKs, FKs, enums, and constraints. Normalizes schema into a JSON/dict structure.
    Supports PostgreSQL, MySQL, and SQLite.
    """
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.inspector = inspect(engine)
        self.metadata = MetaData()
        self.db_type = self._detect_db_type()
        
    def _detect_db_type(self) -> str:
        """Detect database type from engine dialect."""
        dialect_name = self.engine.dialect.name.lower()
        if 'postgresql' in dialect_name or 'postgres' in dialect_name:
            return 'postgresql'
        elif 'mysql' in dialect_name:
            return 'mysql'
        elif 'sqlite' in dialect_name:
            return 'sqlite'
        else:
            return 'unknown'
    
    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get all table names in the database."""
        try:
            return self.inspector.get_table_names(schema=schema)
        except SQLAlchemyError as e:
            logger.error(f"Failed to get table names: {e}")
            return []
    
    def get_column_info(self, table_name: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract detailed column information including constraints."""
        try:
            columns = self.inspector.get_columns(table_name, schema=schema)
            column_info = []
            
            for col in columns:
                col_data = {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "python_type": col["type"].python_type.__name__ if hasattr(col["type"], 'python_type') else None,
                    "nullable": col.get("nullable", True),
                    "default": self._format_default(col.get("default")),
                    "autoincrement": col.get("autoincrement", False),
                    "comment": col.get("comment"),
                }
                
                # Add database-specific type information
                if hasattr(col["type"], 'length'):
                    col_data["length"] = col["type"].length
                if hasattr(col["type"], 'precision'):
                    col_data["precision"] = col["type"].precision
                if hasattr(col["type"], 'scale'):
                    col_data["scale"] = col["type"].scale
                    
                column_info.append(col_data)
                
            return column_info
        except SQLAlchemyError as e:
            logger.error(f"Failed to get columns for table {table_name}: {e}")
            return []
    
    def _format_default(self, default) -> Optional[str]:
        """Format default value for serialization."""
        if default is None:
            return None
        if hasattr(default, 'arg'):
            return str(default.arg)
        return str(default)
    
    def get_primary_key(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Get primary key constraint information."""
        try:
            pk_constraint = self.inspector.get_pk_constraint(table_name, schema=schema)
            return {
                "name": pk_constraint.get("name"),
                "constrained_columns": pk_constraint.get("constrained_columns", []),
            }
        except SQLAlchemyError as e:
            logger.error(f"Failed to get primary key for table {table_name}: {e}")
            return {"name": None, "constrained_columns": []}
    
    def get_foreign_keys(self, table_name: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get foreign key constraints with detailed information."""
        try:
            fks = self.inspector.get_foreign_keys(table_name, schema=schema)
            foreign_keys = []
            
            for fk in fks:
                fk_info = {
                    "name": fk.get("name"),
                    "constrained_columns": fk.get("constrained_columns", []),
                    "referred_table": fk.get("referred_table"),
                    "referred_schema": fk.get("referred_schema"),
                    "referred_columns": fk.get("referred_columns", []),
                    "options": fk.get("options", {}),
                }
                foreign_keys.append(fk_info)
                
            return foreign_keys
        except SQLAlchemyError as e:
            logger.error(f"Failed to get foreign keys for table {table_name}: {e}")
            return []
    
    def get_unique_constraints(self, table_name: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get unique constraints for a table."""
        try:
            constraints = self.inspector.get_unique_constraints(table_name, schema=schema)
            unique_constraints = []
            
            for constraint in constraints:
                constraint_info = {
                    "name": constraint.get("name"),
                    "column_names": constraint.get("column_names", []),
                }
                unique_constraints.append(constraint_info)
                
            return unique_constraints
        except SQLAlchemyError as e:
            logger.error(f"Failed to get unique constraints for table {table_name}: {e}")
            return []
    
    def get_check_constraints(self, table_name: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get check constraints for a table (PostgreSQL and MySQL)."""
        if self.db_type == 'sqlite':
            return []  # SQLite check constraints are not easily introspectable
            
        try:
            constraints = self.inspector.get_check_constraints(table_name, schema=schema)
            check_constraints = []
            
            for constraint in constraints:
                constraint_info = {
                    "name": constraint.get("name"),
                    "sqltext": str(constraint.get("sqltext", "")),
                }
                check_constraints.append(constraint_info)
                
            return check_constraints
        except (SQLAlchemyError, AttributeError):
            # Some database versions might not support this
            return []
    
    def get_indexes(self, table_name: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get index information for a table."""
        try:
            indexes = self.inspector.get_indexes(table_name, schema=schema)
            index_info = []
            
            for index in indexes:
                idx_info = {
                    "name": index.get("name"),
                    "column_names": index.get("column_names", []),
                    "unique": index.get("unique", False),
                    "type": index.get("type"),
                }
                index_info.append(idx_info)
                
            return index_info
        except SQLAlchemyError as e:
            logger.error(f"Failed to get indexes for table {table_name}: {e}")
            return []
    
    def analyze_table_dependencies(self, tables: List[str]) -> Dict[str, Set[str]]:
        """Analyze foreign key dependencies between tables."""
        dependencies = {table: set() for table in tables}
        
        for table in tables:
            fks = self.get_foreign_keys(table)
            for fk in fks:
                referred_table = fk["referred_table"]
                if referred_table in tables and referred_table != table:
                    dependencies[table].add(referred_table)
        
        return dependencies
    
    def topological_sort(self, tables: List[str]) -> List[str]:
        """Sort tables in dependency order (parents first)."""
        dependencies = self.analyze_table_dependencies(tables)
        
        # Kahn's algorithm for topological sorting
        in_degree = {table: 0 for table in tables}
        for table in tables:
            for dep in dependencies[table]:
                in_degree[table] += 1
        
        queue = [table for table in tables if in_degree[table] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Remove edges from current to its dependents
            for table in tables:
                if current in dependencies[table]:
                    dependencies[table].remove(current)
                    in_degree[table] -= 1
                    if in_degree[table] == 0:
                        queue.append(table)
        
        # Add any remaining tables (circular dependencies)
        remaining = [table for table in tables if table not in result]
        result.extend(remaining)
        
        return result
    
    def get_full_schema(self, schema: Optional[str] = None, tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract complete schema information for all tables or specified tables.
        Returns normalized schema structure.
        """
        if tables is None:
            tables = self.get_table_names(schema)
        
        schema_info = {
            "database_type": self.db_type,
            "schema_name": schema,
            "tables": {},
            "table_dependencies": {},
            "insertion_order": []
        }
        
        # Collect table information
        for table_name in tables:
            table_info = {
                "name": table_name,
                "columns": self.get_column_info(table_name, schema),
                "primary_key": self.get_primary_key(table_name, schema),
                "foreign_keys": self.get_foreign_keys(table_name, schema),
                "unique_constraints": self.get_unique_constraints(table_name, schema),
                "check_constraints": self.get_check_constraints(table_name, schema),
                "indexes": self.get_indexes(table_name, schema),
            }
            
            # Add semantic analysis
            table_info["semantic_columns"] = self._analyze_semantic_columns(table_info["columns"])
            table_info["is_junction_table"] = self._is_junction_table(table_info)
            
            schema_info["tables"][table_name] = table_info
        
        # Analyze dependencies and insertion order
        schema_info["table_dependencies"] = self.analyze_table_dependencies(tables)
        schema_info["insertion_order"] = self.topological_sort(tables)
        
        return schema_info
    
    def _analyze_semantic_columns(self, columns: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze columns for semantic meaning (name, email, date, etc.)."""
        semantic_map = {}
        
        for col in columns:
            col_name = col["name"].lower()
            col_type = col["type"].lower()
            
            # Email detection
            if 'email' in col_name:
                semantic_map[col["name"]] = "email"
            # Name detection
            elif any(name_part in col_name for name_part in ['name', 'title', 'label']):
                semantic_map[col["name"]] = "name"
            # Date/time detection
            elif any(date_part in col_type for date_part in ['date', 'time', 'timestamp']):
                semantic_map[col["name"]] = "datetime"
            # Phone detection
            elif 'phone' in col_name or 'mobile' in col_name:
                semantic_map[col["name"]] = "phone"
            # Address detection
            elif any(addr_part in col_name for addr_part in ['address', 'street', 'city', 'country']):
                semantic_map[col["name"]] = "address"
            # Price/money detection
            elif any(money_part in col_name for money_part in ['price', 'cost', 'amount', 'salary', 'wage']):
                semantic_map[col["name"]] = "money"
            # URL detection
            elif 'url' in col_name or 'link' in col_name:
                semantic_map[col["name"]] = "url"
            # Description/text detection
            elif any(text_part in col_name for text_part in ['description', 'comment', 'note', 'bio']):
                semantic_map[col["name"]] = "text"
        
        return semantic_map
    
    def _is_junction_table(self, table_info: Dict[str, Any]) -> bool:
        """Determine if a table is a junction/link table (many-to-many relationship)."""
        fks = table_info.get("foreign_keys", [])
        columns = table_info.get("columns", [])
        pk_cols = set(table_info.get("primary_key", {}).get("constrained_columns", []))
        
        # Junction table criteria:
        # 1. Has at least 2 foreign keys
        # 2. Most columns are part of foreign keys or primary key
        # 3. Few or no additional columns beyond FKs
        
        if len(fks) < 2:
            return False
        
        fk_cols = set()
        for fk in fks:
            fk_cols.update(fk.get("constrained_columns", []))
        
        total_cols = len(columns)
        fk_and_pk_cols = len(fk_cols | pk_cols)
        
        # If most columns are FK or PK columns, likely a junction table
        return fk_and_pk_cols >= total_cols - 1


def make_engine(dsn: str) -> Engine:
    """
    Create a SQLAlchemy engine from a DSN and test connectivity.
    """
    engine = create_engine(dsn, future=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def get_schema(engine: Engine, schema: Optional[str] = None, tables: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Reflect the schema and return a dict with tables and columns.
    Enhanced version using SchemaAnalyzer.
    """
    analyzer = SchemaAnalyzer(engine)
    return analyzer.get_full_schema(schema, tables)
