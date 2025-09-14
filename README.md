# AutoMockAI 🚀

**Schema-aware AI-powered mock data generator for relational databases**

AutoMockAI connects to your database, introspects the schema, and generates realistic mock data directly with SQL `INSERT` statements. It uses a **local LLM (via Ollama)** when available, with a **Faker fallback** for guaranteed execution.

---

## ✨ Features

-   **Advanced Schema Analysis**: Automatically detects tables, columns, types, primary keys, foreign keys, and constraints for PostgreSQL, MySQL, and SQLite.
-   **AI-Powered Data Generation**: Uses local LLMs like Mistral or Llama 3 via [Ollama](https://ollama.com/) to generate context-aware, realistic data for semantic columns (e.g., names, emails, addresses).
-   **Robust Fallback System**: Uses [Faker](https://faker.readthedocs.io/) to generate type-correct data when AI generation is disabled or fails, ensuring reliability.
-   **Relational Integrity**:
    -   **Dependency-Aware Insertion**: Inserts data in a topologically sorted order to respect foreign key constraints.
    -   **Valid Foreign Key References**: Populates foreign key columns with values that exist in the referenced parent table.
-   **Constraint Enforcement**: Honors `NOT NULL`, `UNIQUE`, and `CHECK` constraints during data generation.
-   **Transactional & Batched Inserts**: Inserts data in batches within a single transaction for improved performance and safety. All insertions are rolled back if an error occurs.
-   **Data Validation**: Includes a validation agent to assess the quality of the generated data based on:
    -   Field Accuracy (type correctness, nullability)
    -   Referential Integrity
    -   Uniqueness
    -   Realism Score
-   **Flexible CLI**: A powerful and easy-to-use command-line interface built with Typer, offering fine-grained control over the generation process.

---

## 🏗️ Architecture

AutoMockAI operates through a sequence of specialized agents, ensuring a modular and robust workflow:

1.  **CLI Orchestrator**: The user entrypoint that parses commands and manages the overall process.
2.  **Schema Analyzer**: Connects to the database and builds a detailed, normalized schema map.
3.  **Data Generator**: Generates mock data row-by-row, using AI for semantic columns and respecting relational constraints.
4.  **Fallback Generator**: Provides type-correct data using Faker when AI is not used.
5.  **Data Inserter**: Inserts the generated data into the database transactionally and in the correct dependency order.
6.  **Data Validator**: (Optional) Evaluates the quality and integrity of the inserted data.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/automockai.git
cd automockai

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## 🔧 Usage

### Command

```bash
automockai generate --dsn YOUR_DATABASE_DSN [OPTIONS]
```

### Options

| Option              | Short | Description                                                              |
| ------------------- | ----- | ------------------------------------------------------------------------ |
| `--dsn`             |       | **(Required)** Database connection string (DSN).                         |
| `--count`           | `-c`  | Number of rows to generate per table (default: 10).                      |
| `--include`         | `-i`  | Comma-separated glob patterns of tables to include (e.g., `"public.users*"`). |
| `--exclude`         | `-e`  | Comma-separated glob patterns of tables to exclude (e.g., `"*temp*"`).       |
| `--skip-django`     |       | Exclude default Django internal tables (default: True).                  |
| `--dry-run`         |       | Generate data but do not insert it. Prints a sample to the console.      |
| `--fallback-only`   |       | Use only the Faker-based fallback generator, skipping the AI model.      |
| `--validate`        |       | Run validation checks on the generated data after insertion.             |
| `--seed`            |       | An integer seed for deterministic data generation in fallback mode.      |
| `--output`          | `-o`  | Path to save generated data as JSON (only works with `--dry-run`).       |

### Examples

**1. Basic Dry Run (PostgreSQL)**

Preview the data that would be generated for a PostgreSQL database without inserting it.

```bash
automockai generate \
  --dsn "postgresql+psycopg://user:pass@localhost:5432/mydb" \
  --count 20 \
  --dry-run
```

**2. Insert Data and Validate (MySQL)**

Generate 50 rows for all non-Django tables, insert them, and then run a validation report.

```bash
automockai generate \
  --dsn "mysql+mysqlconnector://user:pass@localhost:3306/mydb" \
  --count 50 \
  --validate
```

**3. Filter Tables and Save to File (SQLite)**

Generate data only for `users` and `products` tables, and save the output to a JSON file.

```bash
automockai generate \
  --dsn "sqlite:///./my_database.db" \
  --include "users,products" \
  --dry-run \
  --output "mock_data.json"
```

**4. Fallback-Only Mode with a Seed**

Generate deterministic data using only Faker, which is fast and useful for testing.

```bash
automockai generate \
  --dsn "postgresql+psycopg://user:pass@localhost:5432/mydb" \
  --count 100 \
  --fallback-only \
  --seed 42
```

### Ollama Setup

For AI-powered data generation, you need a running Ollama instance.

```bash
# Run Ollama in Docker
docker compose -f docker-compose.ollama.yml up -d

# Pull a model
docker exec -it ollama ollama pull mistral
```

AutoMockAI will automatically try to connect to Ollama if it's running.

---

## 📊 Example

```sql
INSERT INTO injuria_add (title, link, image, rank, created_at) VALUES
  ('O''Reilly Data Science', 'https://www.oreilly.com/data', 'data-science.png', 1, NOW()),
  ('The Hitchhiker''s Guide to Python', 'https://docs.python-guide.org/', 'hitchhikers-python.png', 2, NOW());
```

---

## 📜 License

MIT License – see [LICENSE](./LICENSE).

---

## 👥 Contributing

Contributions are welcome! Planned areas:

- Improving LLM prompt engineering
- Adding new dataset trainers
- Validation metrics and realism scoring
- Expanding database support

---

⚡ **AutoMockAI = AI + Schema Awareness + Faker Safety**

> Generate realistic mock data, safely and locally.
