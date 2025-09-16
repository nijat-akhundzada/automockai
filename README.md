# AutoMockAI 🚀

**A schema-aware, AI-powered mock data generator for relational databases.**

AutoMockAI connects to your database, analyzes its schema, and generates realistic, context-aware mock data. It features a full AI pipeline, allowing you to train a custom model on sample datasets for high-quality data generation.

---

## ✨ Features

-   **Custom AI Model Training**: Includes scripts to preprocess datasets and train a lightweight transformer model (`distilgpt2`) to generate context-aware data.
-   **Advanced Schema Analysis**: Automatically detects tables, columns, types, primary keys, foreign keys, and constraints for PostgreSQL, MySQL, and SQLite.
-   **Tiered Generation Strategy**:
    1.  **Custom AI Model**: Prioritizes the fine-tuned local model for the most accurate and context-aware data.
    2.  **Ollama Fallback**: Can use a local LLM (via Ollama) if the custom model is unavailable.
    3.  **Faker Fallback**: Guarantees type-correct data generation using [Faker](https://faker.readthedocs.io/) as a final fallback.
-   **Relational Integrity**:
    -   **Dependency-Aware Insertion**: Inserts data in a topologically sorted order to respect foreign key constraints.
    -   **Valid Foreign Key References**: Populates foreign key columns with values that exist in the referenced parent table.
-   **Constraint Enforcement**: Honors `NOT NULL`, `UNIQUE`, and `CHECK` constraints during data generation.
-   **Transactional & Batched Inserts**: Inserts data in batches within a single transaction for performance and safety.
-   **Data Validation**: Includes a validation agent to assess the quality and integrity of the inserted data.
-   **Flexible CLI**: A powerful Typer-based CLI for fine-grained control over the entire workflow.

---

## 🏗️ Architecture

AutoMockAI operates through a sequence of specialized agents:

1.  **CLI Orchestrator**: The user entrypoint that manages the `preprocess`, `train`, and `generate` commands.
2.  **Preprocessing Agent**: Downloads and transforms sample datasets into a unified format for training.
3.  **Training Agent**: Fine-tunes the transformer model on the preprocessed data.
4.  **Schema Analyzer**: Connects to the database and builds a detailed, normalized schema map.
5.  **Data Generator**: Generates mock data using the tiered AI strategy (Custom Model → Ollama → Faker).
6.  **Data Inserter**: Inserts the generated data into the database transactionally.
7.  **Data Validator**: (Optional) Evaluates the quality of the inserted data.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/automockai.git
cd automockai

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e '.[torch,transformers,pytest]'
```

---

## 🔧 Demonstration Workflow

Follow these steps to preprocess data, train the model, and generate mock data.

### 1. Preprocess the Data

This command downloads the required datasets and creates a `training_data.csv` file.

```bash
automockai preprocess
```

*Note: The Kaggle datasets require authentication. Please ensure your `kaggle.json` is placed in `~/.kaggle/` or follow the CLI prompts to download the files manually.*

### 2. Train the AI Model

This command fine-tunes the `distilgpt2` model on the data from the previous step. The trained model will be saved to `model/trained_model/`.

```bash
automockai train
```

### 3. Generate Mock Data

Now you can use the `generate` command. It will automatically load and use your custom-trained model.

```bash
automockai generate \
  --dsn "postgresql+psycopg://user:pass@localhost:5432/mydb" \
  --count 50 \
  --include "users,products"
```

---

## ⚙️ CLI Reference

### `generate` Command Options

| Option                | Short | Description                                                              |
| --------------------- | ----- | ------------------------------------------------------------------------ |
| `--dsn`               |       | **(Required)** Database connection string (DSN).                         |
| `--count`             | `-c`  | Number of rows to generate per table (default: 10).                      |
| `--include`           | `-i`  | Comma-separated glob patterns of tables to include.                      |
| `--exclude`           | `-e`  | Comma-separated glob patterns of tables to exclude.                      |
| `--use-local-model`   |       | Use the local fine-tuned model for generation (default: True).           |
| `--fallback-only`     |       | Use only the Faker-based fallback, skipping all AI models.               |
| `--dry-run`           |       | Generate data but do not insert it.                                      |
| `--validate`          |       | Run validation checks on the generated data after insertion.             |
| `--output`            | `-o`  | Path to save generated data as JSON (requires `--dry-run`).            |

### Other Commands

-   `automockai preprocess`: See step 1.
-   `automockai train`: See step 2.

---

## 📜 License

MIT License – see [LICENSE](./LICENSE).