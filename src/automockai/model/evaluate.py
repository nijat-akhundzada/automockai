from sqlalchemy.engine import Engine
from sqlalchemy import text, inspect
from typing import Dict, Any, List, Optional, Tuple, Set
import logging
import re
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validation Agent - Evaluates generated data with metrics:
    - Field Accuracy
    - Referential Integrity
    - Realism Score
    - Uniqueness
    Reports issues back to generator if constraints fail.
    """
    
    def __init__(self, engine: Engine, schema_info: Dict[str, Any]):
        self.engine = engine
        self.schema_info = schema_info
        self.validation_results = {}
        
    def validate_all_tables(self, tables: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Validate data quality across all specified tables."""
        if tables is None:
            tables = list(self.schema_info["tables"].keys())
        
        results = {}
        
        for table_name in tables:
            logger.info(f"Validating data quality for table: {table_name}")
            table_results = self.validate_table(table_name)
            results[table_name] = table_results
        
        self.validation_results = results
        return results
    
    def validate_table(self, table_name: str) -> Dict[str, Any]:
        """Validate data quality for a specific table."""
        table_info = self.schema_info["tables"][table_name]
        
        validation_results = {
            "table_name": table_name,
            "row_count": self._get_row_count(table_name),
            "field_accuracy": self._validate_field_accuracy(table_name, table_info),
            "referential_integrity": self._validate_referential_integrity(table_name, table_info),
            "uniqueness_constraints": self._validate_uniqueness(table_name, table_info),
            "realism_score": self._calculate_realism_score(table_name, table_info),
            "constraint_violations": self._check_constraint_violations(table_name, table_info),
            "data_distribution": self._analyze_data_distribution(table_name, table_info),
        }
        
        # Calculate overall quality score
        validation_results["overall_quality_score"] = self._calculate_overall_score(validation_results)
        
        return validation_results
    
    def _get_row_count(self, table_name: str) -> int:
        """Get total row count for the table."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                return result.scalar()
        except Exception as e:
            logger.error(f"Failed to get row count for {table_name}: {e}")
            return 0
    
    def _validate_field_accuracy(self, table_name: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate field-level data accuracy and type correctness."""
        accuracy_results = {
            "type_correctness": {},
            "null_constraint_violations": [],
            "format_violations": {},
            "overall_accuracy": 0.0
        }
        
        columns = table_info["columns"]
        total_checks = 0
        passed_checks = 0
        
        with self.engine.connect() as conn:
            for column in columns:
                col_name = column["name"]
                col_type = column["type"].lower()
                nullable = column.get("nullable", True)
                
                # Check null constraint violations
                if not nullable:
                    null_count = conn.execute(
                        text(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" IS NULL')
                    ).scalar()
                    
                    if null_count > 0:
                        accuracy_results["null_constraint_violations"].append({
                            "column": col_name,
                            "null_count": null_count
                        })
                    
                    total_checks += 1
                    if null_count == 0:
                        passed_checks += 1
                
                # Check type-specific format violations
                format_violations = self._check_column_format(conn, table_name, col_name, col_type)
                if format_violations:
                    accuracy_results["format_violations"][col_name] = format_violations
                    total_checks += 1
                else:
                    total_checks += 1
                    passed_checks += 1
        
        accuracy_results["overall_accuracy"] = passed_checks / total_checks if total_checks > 0 else 1.0
        return accuracy_results
    
    def _check_column_format(self, conn, table_name: str, col_name: str, col_type: str) -> List[str]:
        """Check format violations for specific column types."""
        violations = []
        
        try:
            # Email format validation
            if 'email' in col_name.lower():
                invalid_emails = conn.execute(text(f"""
                    SELECT COUNT(*) FROM "{table_name}" 
                    WHERE "{col_name}" IS NOT NULL 
                    AND "{col_name}" !~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{{2,}}$'
                """)).scalar()
                
                if invalid_emails > 0:
                    violations.append(f"Invalid email format: {invalid_emails} rows")
            
            # Phone format validation
            elif 'phone' in col_name.lower():
                # Basic phone validation - should contain digits
                invalid_phones = conn.execute(text(f"""
                    SELECT COUNT(*) FROM "{table_name}" 
                    WHERE "{col_name}" IS NOT NULL 
                    AND LENGTH(REGEXP_REPLACE("{col_name}", '[^0-9]', '', 'g')) < 10
                """)).scalar()
                
                if invalid_phones > 0:
                    violations.append(f"Invalid phone format: {invalid_phones} rows")
            
            # Date format validation
            elif any(date_type in col_type for date_type in ['date', 'timestamp', 'datetime']):
                # Check for invalid dates (this is database-specific)
                if self.schema_info["database_type"] == "postgresql":
                    invalid_dates = conn.execute(text(f"""
                        SELECT COUNT(*) FROM "{table_name}" 
                        WHERE "{col_name}" IS NOT NULL 
                        AND "{col_name}"::text !~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}'
                    """)).scalar()
                    
                    if invalid_dates > 0:
                        violations.append(f"Invalid date format: {invalid_dates} rows")
        
        except Exception as e:
            logger.warning(f"Could not validate format for {col_name}: {e}")
        
        return violations
    
    def _validate_referential_integrity(self, table_name: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate foreign key referential integrity."""
        integrity_results = {
            "foreign_key_violations": [],
            "orphaned_records": {},
            "integrity_score": 1.0
        }
        
        foreign_keys = table_info.get("foreign_keys", [])
        total_fk_checks = len(foreign_keys)
        passed_fk_checks = 0
        
        with self.engine.connect() as conn:
            for fk in foreign_keys:
                constrained_col = fk["constrained_columns"][0]
                referred_table = fk["referred_table"]
                referred_col = fk["referred_columns"][0]
                
                # Check for orphaned foreign key values
                orphaned_count = conn.execute(text(f"""
                    SELECT COUNT(*) FROM "{table_name}" t1 
                    WHERE t1."{constrained_col}" IS NOT NULL 
                    AND NOT EXISTS (
                        SELECT 1 FROM "{referred_table}" t2 
                        WHERE t2."{referred_col}" = t1."{constrained_col}"
                    )
                """)).scalar()
                
                if orphaned_count > 0:
                    violation = {
                        "table": table_name,
                        "column": constrained_col,
                        "referred_table": referred_table,
                        "referred_column": referred_col,
                        "orphaned_count": orphaned_count
                    }
                    integrity_results["foreign_key_violations"].append(violation)
                    integrity_results["orphaned_records"][constrained_col] = orphaned_count
                else:
                    passed_fk_checks += 1
        
        if total_fk_checks > 0:
            integrity_results["integrity_score"] = passed_fk_checks / total_fk_checks
        
        return integrity_results
    
    def _validate_uniqueness(self, table_name: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate uniqueness constraints."""
        uniqueness_results = {
            "unique_violations": [],
            "primary_key_violations": [],
            "uniqueness_score": 1.0
        }
        
        total_unique_checks = 0
        passed_unique_checks = 0
        
        with self.engine.connect() as conn:
            # Check primary key uniqueness
            pk_columns = table_info["primary_key"]["constrained_columns"]
            if pk_columns:
                pk_cols_sql = ", ".join(f'"{col}"' for col in pk_columns)
                duplicate_pks = conn.execute(text(f"""
                    SELECT COUNT(*) FROM (
                        SELECT {pk_cols_sql}, COUNT(*) as cnt 
                        FROM "{table_name}" 
                        GROUP BY {pk_cols_sql} 
                        HAVING COUNT(*) > 1
                    ) duplicates
                """)).scalar()
                
                total_unique_checks += 1
                if duplicate_pks > 0:
                    uniqueness_results["primary_key_violations"].append({
                        "columns": pk_columns,
                        "duplicate_count": duplicate_pks
                    })
                else:
                    passed_unique_checks += 1
            
            # Check unique constraints
            unique_constraints = table_info.get("unique_constraints", [])
            for constraint in unique_constraints:
                constraint_cols = constraint["column_names"]
                cols_sql = ", ".join(f'"{col}"' for col in constraint_cols)
                
                duplicate_count = conn.execute(text(f"""
                    SELECT COUNT(*) FROM (
                        SELECT {cols_sql}, COUNT(*) as cnt 
                        FROM "{table_name}" 
                        WHERE {" AND ".join(f'"{col}" IS NOT NULL' for col in constraint_cols)}
                        GROUP BY {cols_sql} 
                        HAVING COUNT(*) > 1
                    ) duplicates
                """)).scalar()
                
                total_unique_checks += 1
                if duplicate_count > 0:
                    uniqueness_results["unique_violations"].append({
                        "constraint_name": constraint.get("name"),
                        "columns": constraint_cols,
                        "duplicate_count": duplicate_count
                    })
                else:
                    passed_unique_checks += 1
        
        if total_unique_checks > 0:
            uniqueness_results["uniqueness_score"] = passed_unique_checks / total_unique_checks
        
        return uniqueness_results
    
    def _calculate_realism_score(self, table_name: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate realism score based on data patterns and semantic correctness."""
        realism_results = {
            "semantic_accuracy": {},
            "data_variety": {},
            "pattern_consistency": {},
            "overall_realism": 0.0
        }
        
        semantic_columns = table_info.get("semantic_columns", {})
        total_semantic_checks = 0
        passed_semantic_checks = 0
        
        with self.engine.connect() as conn:
            for col_name, semantic_type in semantic_columns.items():
                score = self._evaluate_semantic_realism(conn, table_name, col_name, semantic_type)
                realism_results["semantic_accuracy"][col_name] = {
                    "type": semantic_type,
                    "score": score
                }
                total_semantic_checks += 1
                if score >= 0.7:  # Threshold for acceptable realism
                    passed_semantic_checks += 1
            
            # Evaluate data variety (avoid too much repetition)
            for column in table_info["columns"]:
                col_name = column["name"]
                variety_score = self._calculate_data_variety(conn, table_name, col_name)
                realism_results["data_variety"][col_name] = variety_score
        
        if total_semantic_checks > 0:
            realism_results["overall_realism"] = passed_semantic_checks / total_semantic_checks
        else:
            realism_results["overall_realism"] = 1.0  # No semantic columns to check
        
        return realism_results
    
    def _evaluate_semantic_realism(self, conn, table_name: str, col_name: str, semantic_type: str) -> float:
        """Evaluate realism for a specific semantic column type."""
        try:
            if semantic_type == "email":
                # Check for realistic email patterns
                valid_pattern = conn.execute(text(f"""
                    SELECT COUNT(*) FROM "{table_name}" 
                    WHERE "{col_name}" IS NOT NULL 
                    AND "{col_name}" ~ '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,4}}$'
                """)).scalar()
                
                total_non_null = conn.execute(text(f"""
                    SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" IS NOT NULL
                """)).scalar()
                
                return valid_pattern / total_non_null if total_non_null > 0 else 0.0
            
            elif semantic_type == "name":
                # Check for realistic name patterns (contains letters, reasonable length)
                realistic_names = conn.execute(text(f"""
                    SELECT COUNT(*) FROM "{table_name}" 
                    WHERE "{col_name}" IS NOT NULL 
                    AND LENGTH("{col_name}") BETWEEN 2 AND 50
                    AND "{col_name}" ~ '^[A-Za-z\\s\\-\\.]+$'
                """)).scalar()
                
                total_non_null = conn.execute(text(f"""
                    SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" IS NOT NULL
                """)).scalar()
                
                return realistic_names / total_non_null if total_non_null > 0 else 0.0
            
            elif semantic_type == "money":
                # Check for realistic monetary values (positive, reasonable precision)
                realistic_money = conn.execute(text(f"""
                    SELECT COUNT(*) FROM "{table_name}" 
                    WHERE "{col_name}" IS NOT NULL 
                    AND CAST("{col_name}" AS DECIMAL) > 0 
                    AND CAST("{col_name}" AS DECIMAL) < 1000000
                """)).scalar()
                
                total_non_null = conn.execute(text(f"""
                    SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" IS NOT NULL
                """)).scalar()
                
                return realistic_money / total_non_null if total_non_null > 0 else 0.0
            
            else:
                return 0.8  # Default score for other semantic types
                
        except Exception as e:
            logger.warning(f"Could not evaluate realism for {col_name}: {e}")
            return 0.5  # Neutral score on error
    
    def _calculate_data_variety(self, conn, table_name: str, col_name: str) -> float:
        """Calculate data variety score (1.0 = high variety, 0.0 = all same values)."""
        try:
            total_rows = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
            if total_rows == 0:
                return 1.0
            
            distinct_values = conn.execute(text(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{table_name}"')).scalar()
            
            return min(distinct_values / total_rows, 1.0)
            
        except Exception as e:
            logger.warning(f"Could not calculate variety for {col_name}: {e}")
            return 0.5
    
    def _check_constraint_violations(self, table_name: str, table_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for check constraint violations."""
        violations = []
        
        check_constraints = table_info.get("check_constraints", [])
        
        with self.engine.connect() as conn:
            for constraint in check_constraints:
                constraint_name = constraint.get("name", "unnamed")
                constraint_sql = constraint.get("sqltext", "")
                
                if not constraint_sql:
                    continue
                
                try:
                    # This is database-specific and might need adaptation
                    violation_count = conn.execute(text(f"""
                        SELECT COUNT(*) FROM "{table_name}" 
                        WHERE NOT ({constraint_sql})
                    """)).scalar()
                    
                    if violation_count > 0:
                        violations.append({
                            "constraint_name": constraint_name,
                            "constraint_sql": constraint_sql,
                            "violation_count": violation_count
                        })
                        
                except Exception as e:
                    logger.warning(f"Could not check constraint {constraint_name}: {e}")
        
        return violations
    
    def _analyze_data_distribution(self, table_name: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data distribution patterns."""
        distribution_results = {
            "null_percentages": {},
            "value_distributions": {},
            "outliers": {}
        }
        
        with self.engine.connect() as conn:
            for column in table_info["columns"]:
                col_name = column["name"]
                col_type = column["type"].lower()
                
                # Calculate null percentage
                total_rows = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
                null_count = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" IS NULL')).scalar()
                
                null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0
                distribution_results["null_percentages"][col_name] = null_percentage
                
                # Analyze numeric distributions
                if any(num_type in col_type for num_type in ['int', 'float', 'decimal', 'numeric']):
                    try:
                        stats = conn.execute(text(f"""
                            SELECT 
                                MIN(CAST("{col_name}" AS DECIMAL)) as min_val,
                                MAX(CAST("{col_name}" AS DECIMAL)) as max_val,
                                AVG(CAST("{col_name}" AS DECIMAL)) as avg_val,
                                COUNT(DISTINCT "{col_name}") as distinct_count
                            FROM "{table_name}" 
                            WHERE "{col_name}" IS NOT NULL
                        """)).fetchone()
                        
                        if stats:
                            distribution_results["value_distributions"][col_name] = {
                                "min": float(stats[0]) if stats[0] is not None else None,
                                "max": float(stats[1]) if stats[1] is not None else None,
                                "avg": float(stats[2]) if stats[2] is not None else None,
                                "distinct_count": stats[3]
                            }
                    except Exception as e:
                        logger.warning(f"Could not analyze distribution for numeric column {col_name}: {e}")
        
        return distribution_results
    
    def _calculate_overall_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score from individual metrics."""
        scores = []
        
        # Field accuracy score
        field_accuracy = validation_results["field_accuracy"]["overall_accuracy"]
        scores.append(field_accuracy)
        
        # Referential integrity score
        integrity_score = validation_results["referential_integrity"]["integrity_score"]
        scores.append(integrity_score)
        
        # Uniqueness score
        uniqueness_score = validation_results["uniqueness_constraints"]["uniqueness_score"]
        scores.append(uniqueness_score)
        
        # Realism score
        realism_score = validation_results["realism_score"]["overall_realism"]
        scores.append(realism_score)
        
        # Constraint violations (inverse score)
        constraint_violations = len(validation_results["constraint_violations"])
        constraint_score = 1.0 if constraint_violations == 0 else max(0.0, 1.0 - constraint_violations * 0.1)
        scores.append(constraint_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def generate_validation_report(self, tables: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        if not self.validation_results:
            self.validate_all_tables(tables)
        
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "database_type": self.schema_info["database_type"],
            "tables_validated": len(self.validation_results),
            "overall_summary": {},
            "table_results": self.validation_results,
            "recommendations": []
        }
        
        # Calculate overall summary
        all_scores = [result["overall_quality_score"] for result in self.validation_results.values()]
        report["overall_summary"] = {
            "average_quality_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "total_rows_validated": sum(result["row_count"] for result in self.validation_results.values()),
            "tables_with_issues": len([r for r in self.validation_results.values() if r["overall_quality_score"] < 0.8])
        }
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for table_name, results in self.validation_results.items():
            if results["overall_quality_score"] < 0.8:
                recommendations.append(f"Table '{table_name}' has quality issues (score: {results['overall_quality_score']:.2f})")
            
            if results["referential_integrity"]["foreign_key_violations"]:
                recommendations.append(f"Fix foreign key violations in table '{table_name}'")
            
            if results["uniqueness_constraints"]["unique_violations"]:
                recommendations.append(f"Address uniqueness constraint violations in table '{table_name}'")
            
            if results["field_accuracy"]["overall_accuracy"] < 0.9:
                recommendations.append(f"Improve field accuracy in table '{table_name}' (current: {results['field_accuracy']['overall_accuracy']:.2f})")
        
        return recommendations


def validate_generated_data(engine: Engine, schema_info: Dict[str, Any], 
                          tables: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to validate generated data and return a comprehensive report.
    """
    validator = DataValidator(engine, schema_info)
    return validator.generate_validation_report(tables)