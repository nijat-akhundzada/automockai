import os
import typer
import logging
from typing import Optional, List, Dict, Any
from fnmatch import fnmatch
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from automockai.schema import SchemaAnalyzer, make_engine
from automockai.generator import DataGenerator
from automockai.inserter import DataInserter
from automockai.model.evaluate import DataValidator
from automockai.schema import SchemaAnalyzer, make_engine
from automockai.generator import DataGenerator
from automockai.inserter import DataInserter
from automockai.model.evaluate import DataValidator
from automockai.model.preprocessing import create_training_data
from automockai.model.train import train_model

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
    """Check if a table name matches any of the exclude patterns."""
    return any(fnmatch(name, pat) for pat in patterns)

def _filter_tables(all_tables: List[str], include: List[str], exclude: List[str], 
                   skip_django: bool) -> List[str]:
    """Filter tables based on include/exclude patterns and other flags."""
    tables = all_tables
    
    if skip_django:
        exclude.extend(DEFAULT_DJANGO_EXCLUDES)
    
    if include:
        tables = [t for t in tables if any(fnmatch(t, p) for p in include)]
    
    if exclude:
        tables = [t for t in tables if not any(fnmatch(t, p) for p in exclude)]
        
    return tables


@app.command()
def generate(
    dsn: str = typer.Option(..., help="Database connection string (DSN)"),
    count: int = typer.Option(10, "--count", "-c", help="Number of rows to generate per table"),
    include: Optional[str] = typer.Option(None, "--include", "-i", help="Comma-separated glob patterns of tables to include"),
    exclude: Optional[str] = typer.Option(None, "--exclude", "-e", help="Comma-separated glob patterns of tables to exclude"),
    skip_django: bool = typer.Option(True, help="Exclude default Django tables"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate data but do not insert into the database"),
    use_local_model: bool = typer.Option(True, help="Use the local fine-tuned model for generation."),
    fallback_only: bool = typer.Option(False, "--fallback-only", help="Use only the fallback generator (Faker), skipping AI"),
    validate: bool = typer.Option(False, "--validate", help="Run validation checks on the generated data after insertion"),
    seed: Optional[int] = typer.Option(None, help="Seed for deterministic data generation in fallback mode"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Path to save generated data as JSON (for dry runs)"),
):
    """
    Main CLI command to orchestrate the data generation process.
    """
    logger.info("🚀 AutoMockAI is starting the data generation process...")

    # --- 1. Schema Analyzer ---
    logger.info("Connecting to the database and analyzing schema...")
    try:
        engine = make_engine(dsn)
        schema_analyzer = SchemaAnalyzer(engine)
        schema_info = schema_analyzer.get_full_schema()
        logger.info(f"✅ Schema analysis complete. Found {len(schema_info['tables'])} tables.")
    except Exception as e:
        logger.error(f"❌ Failed to connect to the database or analyze schema: {e}")
        raise typer.Exit(code=1)

    # --- 2. Table Filtering ---
    all_tables = list(schema_info["tables"].keys())
    include_patterns = [p.strip() for p in include.split(',')]
    exclude_patterns = [p.strip() for p in exclude.split(',')]
    
    selected_tables = _filter_tables(all_tables, include_patterns, exclude_patterns, skip_django)
    
    if not selected_tables:
        logger.warning("⚠️ No tables selected after applying filters. Exiting.")
        raise typer.Exit()
        
    logger.info(f"Selected {len(selected_tables)} tables for data generation.")
    
    # --- 3. Data Generator ---
    logger.info("Generating mock data...")
    data_generator = DataGenerator(engine, schema_info)
    if not use_local_model:
        data_generator.text_generator = None # Disable local model if not requested

    generated_data = {}
    
    insertion_order = schema_info.get("insertion_order", selected_tables)
    
    for table_name in insertion_order:
        if table_name in selected_tables:
            logger.info(f"  - Generating {count} rows for table: {table_name}")
            try:
                rows = data_generator.generate_row_data(table_name, count)
                generated_data[table_name] = rows
            except Exception as e:
                logger.error(f"❌ Failed to generate data for table {table_name}: {e}")
    
    logger.info("✅ Data generation complete.")
    
    # --- 4. Dry Run Handling ---
    if dry_run:
        logger.info("Dry run enabled. Displaying generated data instead of inserting.")
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(generated_data, f, indent=2, default=str)
                logger.info(f"Generated data saved to {output_file}")
            except Exception as e:
                logger.error(f"❌ Failed to write to output file: {e}")
        else:
            # Print a sample of the data
            for table_name, rows in generated_data.items():
                print(f"\n--- Table: {table_name} ---")
                print(json.dumps(rows[:3], indent=2, default=str)) # Print first 3 rows
                if len(rows) > 3:
                    print(f"  (... and {len(rows) - 3} more rows)")
        
        raise typer.Exit()
    
    # --- 5. Inserter ---
    logger.info("Inserting generated data into the database...")
    inserter = DataInserter(engine, schema_info)
    try:
        insertion_summary = inserter.insert_all_tables(generated_data, validate_integrity=False)
        logger.info("✅ Data insertion complete.")
        for table, num_rows in insertion_summary.items():
            logger.info(f"  - Inserted {num_rows} rows into {table}")
    except Exception as e:
        logger.error(f"❌ Data insertion failed: {e}")
        raise typer.Exit(code=1)
    
    # --- 6. Validation ---
    if validate:
        logger.info("Running validation checks on the inserted data...")
        validator = DataValidator(engine, schema_info)
        try:
            validation_report = validator.generate_validation_report(selected_tables)
            logger.info("✅ Validation complete.")
            
            # Print validation summary
            summary = validation_report['overall_summary']
            print("\n--- Validation Summary ---")
            print(f"Average Quality Score: {summary['average_quality_score']:.2f}")
            print(f"Tables with Issues: {summary['tables_with_issues']}")
            
            if validation_report['recommendations']:
                print("\nRecommendations:")
                for rec in validation_report['recommendations']:
                    print(f"- {rec}")
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
    
    logger.info("🎉 AutoMockAI process finished successfully!")

@app.command()
def preprocess(output_path: str = typer.Option("training_data.csv", help="Path to save the processed training data.")):
    """
    Downloads and preprocesses the training datasets.
    """
    logger.info("Starting data preprocessing...")
    create_training_data(output_path)

@app.command()
def train(
    training_data_path: str = typer.Option("training_data.csv", help="Path to the training data CSV file."),
    epochs: int = typer.Option(3, help="Number of training epochs."),
    batch_size: int = typer.Option(8, help="Training batch size."),
    learning_rate: float = typer.Option(5e-5, help="Learning rate for the optimizer."),
):
    """
    Trains the AI model on the preprocessed data.
    """
    logger.info("Starting model training...")
    train_model(
        training_data_path=training_data_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

def main():
    app()


if __name__ == "__main__":
    main()
