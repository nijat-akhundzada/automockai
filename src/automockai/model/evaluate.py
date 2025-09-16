
import logging
from sqlalchemy.engine import Engine
from sqlalchemy import text
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Validates the integrity and quality of data inserted into the database.
    """
    def __init__(self, engine: Engine, schema_info: Dict[str, Any]):
        self.engine = engine
        self.schema_info = schema_info

    def generate_validation_report(self, table_names: List[str]) -> Dict[str, Any]:
        """
        Generates a comprehensive report on data quality for the specified tables.
        """
        logger.info("Starting data validation...")
        report = {
            "tables": {},
            "overall_summary": {
                "average_quality_score": 0.0,
                "tables_with_issues": 0,
                "total_tables_validated": len(table_names),
            },
            "recommendations": [],
        }
        total_score = 0.0

        with self.engine.connect() as connection:
            for table_name in table_names:
                table_report = self._validate_table(connection, table_name)
                report["tables"][table_name] = table_report
                total_score += table_report["quality_score"]
                if not table_report["is_valid"]:
                    report["overall_summary"]["tables_with_issues"] += 1
                    report["recommendations"].append(
                        f"Review validation issues for table '{table_name}'."
                    )

        if table_names:
            report["overall_summary"]["average_quality_score"] = total_score / len(table_names)

        logger.info("Validation complete.")
        return report

    def _validate_table(self, connection, table_name: str) -> Dict[str, Any]:
        """
        Performs all validation checks for a single table.
        """
        table_schema = self.schema_info['tables'][table_name]
        columns = table_schema['columns']
        
        total_checks = 0
        passed_checks = 0
        issues = []

        # 1. Check for nulls in NOT NULL columns
        for col_name, col_props in columns.items():
            if not col_props['nullable']:
                total_checks += 1
                query = text(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col_name}" IS NULL')
                null_count = connection.execute(query).scalar_one()
                if null_count == 0:
                    passed_checks += 1
                else:
                    issues.append({
                        "check": "NOT NULL",
                        "column": col_name,
                        "details": f"{null_count} null values found in a NOT NULL column."
                    })
        
        # 2. Check for uniqueness in UNIQUE columns
        for col_name, col_props in columns.items():
            if col_props['unique']:
                total_checks += 1
                query = text(f'SELECT COUNT("{col_name}") - COUNT(DISTINCT "{col_name}") FROM "{table_name}"')
                duplicate_count = connection.execute(query).scalar_one()
                if duplicate_count == 0:
                    passed_checks += 1
                else:
                    issues.append({
                        "check": "UNIQUE",
                        "column": col_name,
                        "details": f"{duplicate_count} duplicate values found in a UNIQUE column."
                    })

        # 3. Check foreign key integrity
        for fk in self.schema_info.get('foreign_keys', []):
            if fk['constrained_table'] == table_name:
                total_checks += 1
                constrained_col = fk['constrained_columns'][0] # Assuming single-column FKs for simplicity
                referred_table = fk['referred_table']
                referred_col = fk['referred_columns'][0]
                
                query = text(f'''
                    SELECT COUNT(t1."{constrained_col}")
                    FROM "{table_name}" AS t1
                    LEFT JOIN "{referred_table}" AS t2 ON t1."{constrained_col}" = t2."{referred_col}"
                    WHERE t2."{referred_col}" IS NULL AND t1."{constrained_col}" IS NOT NULL
                ''')
                dangling_refs = connection.execute(query).scalar_one()

                if dangling_refs == 0:
                    passed_checks += 1
                else:
                    issues.append({
                        "check": "FOREIGN KEY",
                        "column": constrained_col,
                        "details": f"{dangling_refs} references to non-existent rows in table '{referred_table}'."
                    })

        quality_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 100.0
        
        return {
            "is_valid": not issues,
            "quality_score": quality_score,
            "issues": issues,
        }
