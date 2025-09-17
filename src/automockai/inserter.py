from sqlalchemy.engine import Engine
from sqlalchemy import text, MetaData, Table, Column, inspect
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from typing import Dict, Any, List, Optional, Tuple
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DataInserter:
    """
    Inserter Agent - Inserts rows into the DB in dependency order.
    Uses transactions, batch inserts, rollback on error.
    Must validate referential integrity after insertion.
    """
    
    def __init__(self, engine: Engine, schema_info: Dict[str, Any]):
        self.engine = engine
        self.schema_info = schema_info
        self.batch_size = 1000
        self.inserted_counts = {}
        
    @contextmanager
    def transaction(self):
        """Context manager for database transactions with rollback on error."""
        conn = self.engine.connect()
        trans = conn.begin()
        try:
            yield conn
            trans.commit()
            logger.info("Transaction committed successfully")
        except Exception as e:
            trans.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            conn.close()
    
    def build_insert_sql(self, table_name: str, rows: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Build parameterized INSERT SQL for batch insertion."""
        if not rows:
            return "", []
        
        # Get column names from the first row
        columns = list(rows[0].keys())
        if not columns:
            return "", []
        
        # Build the INSERT statement
        columns_sql = ", ".join(f'"{col}"' for col in columns)
        placeholders = ", ".join(f":{col}" for col in columns)
        
        sql = f'INSERT INTO "{table_name}" ({columns_sql}) VALUES ({placeholders})'
        
        return sql, rows
    
    def insert_batch(self, conn, table_name: str, rows: List[Dict[str, Any]]) -> int:
        """Insert a batch of rows into a table."""
        if not rows:
            return 0
        
        sql, params = self.build_insert_sql(table_name, rows)
        if not sql:
            return 0
        
        try:
            result = conn.execute(text(sql), params)
            
            inserted_count = result.rowcount if hasattr(result, 'rowcount') else len(params)
            logger.info(f"Inserted {inserted_count} rows into {table_name}")
            return inserted_count
            
        except IntegrityError as e:
            logger.error(f"Integrity constraint violation in {table_name}: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"SQL error inserting into {table_name}: {e}")
            raise
    
    def insert_table_data(self, table_name: str, rows: List[Dict[str, Any]], 
                         use_transaction: bool = True) -> int:
        """Insert data into a single table with batching."""
        if not rows:
            logger.warning(f"No data to insert for table {table_name}")
            return 0
        
        total_inserted = 0
        
        def _insert_batches(conn):
            nonlocal total_inserted
            # Process in batches
            for i in range(0, len(rows), self.batch_size):
                batch = rows[i:i + self.batch_size]
                inserted = self.insert_batch(conn, table_name, batch)
                total_inserted += inserted
        
        if use_transaction:
            with self.transaction() as conn:
                _insert_batches(conn)
        else:
            # Use existing connection/transaction
            with self.engine.connect() as conn:
                _insert_batches(conn)
        
        self.inserted_counts[table_name] = total_inserted
        return total_inserted
    
    def insert_all_tables(self, table_data: Dict[str, List[Dict[str, Any]]], 
                         validate_integrity: bool = True) -> Dict[str, int]:
        """Insert data into all tables in dependency order."""
        insertion_order = self.schema_info.get("insertion_order", list(table_data.keys()))
        results = {}
        
        # Filter to only tables that have data
        tables_to_insert = [table for table in insertion_order if table in table_data and table_data[table]]
        
        logger.info(f"Inserting data into {len(tables_to_insert)} tables in dependency order")
        
        with self.transaction() as conn:
            for table_name in tables_to_insert:
                rows = table_data[table_name]
                logger.info(f"Inserting {len(rows)} rows into {table_name}")
                
                try:
                    # Insert batches for this table
                    total_inserted = 0
                    for i in range(0, len(rows), self.batch_size):
                        batch = rows[i:i + self.batch_size]
                        inserted = self.insert_batch(conn, table_name, batch)
                        total_inserted += inserted
                    
                    results[table_name] = total_inserted
                    self.inserted_counts[table_name] = total_inserted
                    
                except Exception as e:
                    logger.error(f"Failed to insert data into {table_name}: {e}")
                    raise
        
        # Validate referential integrity after all insertions
        if validate_integrity:
            self.validate_referential_integrity()
        
        return results
    
    def validate_referential_integrity(self) -> Dict[str, List[str]]:
        """Validate referential integrity across all tables."""
        logger.info("Validating referential integrity...")
        violations = {}
        
        with self.engine.connect() as conn:
            for table_name, table_info in self.schema_info["tables"].items():
                table_violations = []
                
                for fk in table_info.get("foreign_keys", []):
                    violation = self._check_foreign_key_constraint(conn, table_name, fk)
                    if violation:
                        table_violations.append(violation)
                
                if table_violations:
                    violations[table_name] = table_violations
        
        if violations:
            logger.warning(f"Found referential integrity violations: {violations}")
        else:
            logger.info("All referential integrity constraints satisfied")
        
        return violations
    
    def _check_foreign_key_constraint(self, conn, table_name: str, fk_info: Dict[str, Any]) -> Optional[str]:
        """Check a specific foreign key constraint."""
        try:
            constrained_col = fk_info["constrained_columns"][0]
            referred_table = fk_info["referred_table"]
            referred_col = fk_info["referred_columns"][0]
            
            # Check for orphaned foreign key values
            sql = f"""
            SELECT COUNT(*) as violation_count 
            FROM "{table_name}" t1 
            WHERE t1."{constrained_col}" IS NOT NULL 
            AND NOT EXISTS (
                SELECT 1 FROM "{referred_table}" t2 
                WHERE t2."{referred_col}" = t1."{constrained_col}"
            )
            """
            
            result = conn.execute(text(sql))
            violation_count = result.scalar()
            
            if violation_count > 0:
                return f"FK constraint violation: {violation_count} orphaned references in {table_name}.{constrained_col} -> {referred_table}.{referred_col}"
            
        except Exception as e:
            logger.warning(f"Could not validate FK constraint {table_name}.{constrained_col}: {e}")
        
        return None
    
    def get_table_row_count(self, table_name: str) -> int:
        """Get the current row count for a table."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                return result.scalar()
        except Exception as e:
            logger.error(f"Failed to get row count for {table_name}: {e}")
            return 0
    
    def clear_table_data(self, table_names: List[str], cascade: bool = False) -> Dict[str, int]:
        """Clear data from specified tables in reverse dependency order."""
        # Reverse the insertion order for deletion
        insertion_order = self.schema_info.get("insertion_order", table_names)
        deletion_order = [table for table in reversed(insertion_order) if table in table_names]
        
        deleted_counts = {}
        
        with self.transaction() as conn:
            for table_name in deletion_order:
                try:
                    # Get count before deletion
                    count_before = self.get_table_row_count(table_name)
                    
                    # Delete all rows
                    if cascade and self.schema_info["database_type"] == "postgresql":
                        sql = f'DELETE FROM "{table_name}" CASCADE'
                    else:
                        sql = f'DELETE FROM "{table_name}"'
                    
                    conn.execute(text(sql))
                    deleted_counts[table_name] = count_before
                    
                    logger.info(f"Deleted {count_before} rows from {table_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to clear table {table_name}: {e}")
                    raise
        
        return deleted_counts
    
    def insert_with_retry(self, table_name: str, rows: List[Dict[str, Any]], 
                         max_retries: int = 3) -> int:
        """Insert data with retry logic for handling temporary failures."""
        for attempt in range(max_retries + 1):
            try:
                return self.insert_table_data(table_name, rows)
            except IntegrityError as e:
                if attempt == max_retries:
                    logger.error(f"Max retries exceeded for {table_name}: {e}")
                    raise
                logger.warning(f"Integrity error on attempt {attempt + 1} for {table_name}, retrying...")
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Max retries exceeded for {table_name}: {e}")
                    raise
                logger.warning(f"Error on attempt {attempt + 1} for {table_name}, retrying: {e}")
        
        return 0
    
    def get_insertion_summary(self) -> Dict[str, Any]:
        """Get a summary of insertion results."""
        total_rows = sum(self.inserted_counts.values())
        return {
            "total_rows_inserted": total_rows,
            "tables_processed": len(self.inserted_counts),
            "per_table_counts": self.inserted_counts.copy(),
            "insertion_order": self.schema_info.get("insertion_order", [])
        }


def insert_data_safely(engine: Engine, schema_info: Dict[str, Any], 
                      table_data: Dict[str, List[Dict[str, Any]]], 
                      validate: bool = True) -> Dict[str, int]:
    """
    Convenience function to safely insert data with proper error handling.
    """
    inserter = DataInserter(engine, schema_info)
    try:
        return inserter.insert_all_tables(table_data, validate_integrity=validate)
    except Exception as e:
        logger.error(f"Data insertion failed: {e}")
        raise