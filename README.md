# AutoMockAI 🚀

**Schema-aware AI-powered mock data generator for relational databases**

AutoMockAI connects to your database, introspects the schema, and generates realistic mock data directly with SQL `INSERT` statements. It uses a **local LLM (via Ollama)** when available, with a **Faker fallback** for guaranteed execution.

---

## ✨ Features (Current MVP)

- **Schema introspection** → Automatically detects tables, columns, types, primary keys, and foreign keys.
- **AI-powered data generation** → Uses [Ollama](https://ollama.com/) (Mistral, LLaMA 3, etc.) to generate context-aware SQL inserts.
- **Faker fallback** → Ensures data is always populated even if AI fails.
- **Foreign key safety** → Link/junction tables handled with valid references (C).
- **Parent-first ordering** → Tables sorted by dependencies before population (D).
- **Text-heavy detection** → Pushes description/content-like tables to the AI (E).
- **Transaction safety** → Each table commits independently, so errors don’t block the whole run.
- **Sanitization & Validation** → Automatic fixes for:
  - Escaping apostrophes (`O'Reilly` → `O''Reilly`)
  - ISO 8601 timestamps (`2022-01-01T12:34:56Z` → `2022-01-01 12:34:56`)
  - Removal of unwanted concatenation (`||`).

---

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/yourusername/automockai.git
cd automockai

# Setup environment
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## 🔧 Usage

### Run against PostgreSQL

```bash
automockai \
  --dsn postgresql+psycopg://user:pass@127.0.0.1:5432/mydb \
  --rows 10 \
  --execute \
  --only-prefix myapp_
```

### Options

- `--dsn` → SQLAlchemy DSN (Postgres/MySQL/etc.)
- `--rows` → Rows per table (default: 20)
- `--execute` → Actually insert data (otherwise dry-run preview)
- `--only-prefix` → Restrict to app-specific tables (skip Django internals)
- `--skip-django` → Skip `auth_*`, `django_*`, etc.

### Ollama Setup

```bash
docker compose -f docker-compose.ollama.yml up
docker exec -it ollama ollama pull mistral
export OLLAMA_HOST=http://localhost:11434
```

---

## 🛠️ Roadmap (Planned Features)

According to the [project documentation](./AutoMockAI.docx), these are the next milestones:

- **Training on public datasets** (Northwind, HR Employee, Financial Transactions) to learn `(field_name, type, value)` distributions.
- **Hybrid model design** → lightweight transformer + statistical learners.
- **Validation framework**:

  - 80/20 train/validation split
  - Referential integrity checks
  - Generalization tests on unseen fields
  - Expert review-based realism score

- **Edge-case handling**:

  - Custom field mapping (semantic similarity)
  - JSON/geospatial support
  - Business rules (e.g. `price > 0`, `end_date > start_date`)

- **Expanded CLI UX** → support `--db-type`, `--host`, `--user`, `--password`, `--database` instead of requiring a full DSN.
- **Realism Score output** → per-table quality metric.
- **GitHub Release & MIT License** packaging.

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
