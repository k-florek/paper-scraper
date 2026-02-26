# paper-scraper

An automated academic paper pipeline that scrapes papers from PubMed (and optionally Google Scholar), stores them in a local SQLite database, embeds them into a ChromaDB vector store using an Ollama embedding model, and provides an interactive AI chat assistant for exploring the collection.

---

## Table of Contents

- [Quickstart](#quickstart)
- [Installation](#installation)
  - [1. Python environment](#1-python-environment)
  - [2. Python dependencies](#2-python-dependencies)
  - [3. Install Ollama](#3-install-ollama)
    - [macOS](#macos)
    - [Linux](#linux)
  - [4. Pull Ollama models](#4-pull-ollama-models)
  - [5. Configure the application](#5-configure-the-application)
- [Usage](#usage)
  - [Scrape & build the vector database](#scrape--build-the-vector-database)
  - [Interactive paper discovery chat](#interactive-paper-discovery-chat)
  - [Running subcommands individually](#running-subcommands-individually)
- [Configuration reference](#configuration-reference)
  - [`search`](#search)
  - [`sources`](#sources)
  - [`output`](#output)
  - [`filters`](#filters)
  - [`vectorize`](#vectorize)
  - [`chat`](#chat)
- [Project structure](#project-structure)

---

## Quickstart

```bash
# 1. clone and enter the project
git clone <repo-url> paper-scraper
cd paper-scraper

# 2. create a Python virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. install Ollama and pull the required models (see Installation section)
ollama pull nomic-embed-text
ollama pull llama3

# 4. copy the example config and edit it with your settings
cp config.example.json config.json
# → edit config.json: set your PubMed email, keywords, desired models, etc.

# 5. scrape papers and build the vector database
./paper-scraper-build

# 6. start the interactive AI chat
./paper-scraper
```

---

## Installation

### 1. Python environment

Python **3.11** or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

The core dependencies are:

| Package | Purpose |
|---|---|
| `biopython` | PubMed / NCBI Entrez API |
| `scholarly` | Google Scholar scraping |
| `chromadb` | Local vector database |
| `ollama` | Python client for the Ollama server |

### 3. Install Ollama

Ollama runs a local inference server that provides both the embedding and chat models.

#### macOS

The simplest method is the official installer:

```bash
# using Homebrew
brew install ollama
```

Or download the `.app` directly from [ollama.com/download](https://ollama.com/download) and drag it to Applications, then start it once to register the CLI.

Start the server (it runs in the background by default after installation):

```bash
ollama serve   # only needed if not already running as a service
```

#### Linux

Run the official install script (works on Debian/Ubuntu, Fedora, Arch, and most others):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This installs the `ollama` binary and registers a `systemd` service that starts automatically on boot. Verify it is running:

```bash
systemctl status ollama
# or start it manually if needed:
sudo systemctl start ollama
```

For systems **without** `systemd`, start the server manually in a terminal:

```bash
ollama serve
```

> **GPU support** — Ollama will automatically use NVIDIA or AMD GPUs when the appropriate drivers are present. No extra configuration is required for CPU-only operation.

### 4. Pull Ollama models

You need two models: one for generating embeddings and one for chat. The defaults in `config.example.json` use:

```bash
# embedding model (used by both `vectorize` and `chat`)
ollama pull nomic-embed-text

# chat / language model
ollama pull llama3
```

You can substitute any model available from [ollama.com/library](https://ollama.com/library). Some alternatives:

| Use case | Model |
|---|---|
| Lighter chat (4 GB RAM) | `ollama pull llama3.2` |
| Larger / more capable chat | `ollama pull llama3.1:70b` |
| Alternative embeddings | `ollama pull mxbai-embed-large` |

Verify models are available:

```bash
ollama list
```

### 5. Configure the application

Copy the example config and edit it for your needs:

```bash
cp config.example.json config.json
```

At a minimum, set your **PubMed email address** and choose your **search keywords** (see the [Configuration reference](#configuration-reference) below).

---

## Usage

### Scrape & build the vector database

```bash
./paper-scraper-build
```

This script runs two steps in sequence:

1. **`scrape`** — queries all enabled sources (PubMed, Google Scholar) for each keyword, deduplicates the results, and stores them in the SQLite database (`databases/papers.db` by default).
2. **`vectorize`** — reads papers from the SQLite database, generates embeddings via Ollama, and stores them in the ChromaDB vector database (`databases/vectors` by default). Re-runs are safe — papers already in ChromaDB are skipped.

Pass a custom config file to both steps:

```bash
./paper-scraper-build --config my_config.json
```

### Interactive paper discovery chat

```bash
./paper-scraper
```

This starts a REPL session where you can ask free-form questions about topics you are interested in. The assistant will retrieve the most similar papers from the vector database and present them with a brief explanation of their relevance.

**In-session commands:**

| Command | Description |
|---|---|
| `mark read <n>` | Mark paper #n from the last batch as read — it will not be suggested again |
| `status` | Print all papers that have been suggested or marked as read |
| `quit` or `exit` | End the session |

Paper interaction history (suggested / read) is persisted in the SQLite database across sessions, so the model never re-suggests a paper from a previous chat.

### Running subcommands individually

The convenience scripts delegate to `main.py`. You can also call subcommands directly:

```bash
# scrape only
python main.py scrape

# vectorize only (e.g., after adding new keywords and re-scraping)
python main.py vectorize

# chat
python main.py chat

# use a custom config file with any subcommand
python main.py scrape --config my_config.json
python main.py vectorize --config my_config.json --db path/to/papers.db
python main.py chat --config my_config.json
```

---

## Configuration reference

The application is driven by a single JSON file (default: `config.json`). A fully annotated example is provided in `config.example.json`.

### `search`

Controls what is searched for across all enabled sources.

```json
"search": {
  "keywords": [
    "infectious disease genomics",
    "bacterial whole genome sequencing"
  ],
  "boolean_operator": "AND",
  "date_range": {
    "start": "2020/01/01",
    "end":   "2026/01/01"
  },
  "max_results": 50
}
```

| Key | Type | Description |
|---|---|---|
| `keywords` | `string[]` | List of keyword phrases. Each phrase is searched as a separate query. |
| `boolean_operator` | `string` | How individual words inside a phrase are joined (`"AND"` or `"OR"`). |
| `date_range.start` | `string` | Earliest publication date to include (`YYYY/MM/DD`). |
| `date_range.end` | `string` | Latest publication date to include (`YYYY/MM/DD`). |
| `max_results` | `integer` | Maximum number of papers to retrieve **per keyword** from each source. |

### `sources`

Enables or disables individual scraping sources.

```json
"sources": {
  "pubmed": {
    "enabled": true,
    "email":   "you@example.com",
    "api_key": ""
  },
  "google_scholar": {
    "enabled":   false,
    "use_proxy": false,
    "proxy":     ""
  }
}
```

#### PubMed

| Key | Description |
|---|---|
| `enabled` | Set to `true` to scrape PubMed. |
| `email` | **Required.** A valid email address as required by the NCBI Entrez API terms of service. |
| `api_key` | Optional. A free NCBI API key raises the rate limit from 3 to 10 requests/second. Register at [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/). |

#### Google Scholar

| Key | Description |
|---|---|
| `enabled` | Set to `true` to scrape Google Scholar. Note: Google Scholar rate-limits aggressive crawling; use sparingly or with a proxy. |
| `use_proxy` | Set to `true` to route requests through the proxy below. |
| `proxy` | Proxy URL (e.g. `"http://user:pass@host:port"`). |

### `output`

```json
"output": {
  "db_path": "databases/papers.db"
}
```

| Key | Description |
|---|---|
| `db_path` | Path to the SQLite database file. The directory is created automatically if it does not exist. |

### `filters`

Post-scrape filters applied before papers are stored.

```json
"filters": {
  "language":     "english",
  "article_types": ["journal article", "review"],
  "min_citations": 0
}
```

| Key | Description |
|---|---|
| `language` | Keep only papers in this language (case-insensitive). |
| `article_types` | List of publication types to include. Common values: `"journal article"`, `"review"`, `"clinical trial"`, `"meta-analysis"`. |
| `min_citations` | Discard papers with fewer citations than this value. Set to `0` to keep all. |

### `vectorize`

Settings for embedding papers into ChromaDB.

```json
"vectorize": {
  "ollama_host":     "http://localhost:11434",
  "embedding_model": "nomic-embed-text",
  "chroma_path":     "databases/vectors",
  "collection_name": "papers",
  "batch_size":      10
}
```

| Key | Default | Description |
|---|---|---|
| `ollama_host` | `http://localhost:11434` | Base URL of the Ollama server. Change this if Ollama is running on a remote host or a different port. |
| `embedding_model` | `nomic-embed-text` | Ollama model to use for generating embeddings. Must be pulled with `ollama pull <model>` first. |
| `chroma_path` | `databases/vectors` | Directory where ChromaDB persists its vector index. |
| `collection_name` | `papers` | Name of the ChromaDB collection. Changing this creates a fresh collection. |
| `batch_size` | `10` | Number of papers to embed per Ollama API call batch. Decrease if you hit memory limits. |

### `chat`

Settings for the interactive AI chat assistant.

```json
"chat": {
  "ollama_host":     "http://localhost:11434",
  "chat_model":      "llama3",
  "embedding_model": "nomic-embed-text",
  "top_k":           5,
}
```

| Key | Default | Description |
|---|---|---|
| `ollama_host` | `http://localhost:11434` | Base URL of the Ollama server (can differ from the vectorize host). |
| `chat_model` | `llama3` | Ollama chat model to use. Any instruction-tuned model works; larger models give richer responses. |
| `embedding_model` | `nomic-embed-text` | Embedding model used to encode the user's query for vector search. Should match the model used during `vectorize`. |
| `top_k` | `5` | Number of papers to retrieve from ChromaDB per query and present to the model as context. Increase for broader coverage; decrease for more focused answers. |

---

## Project structure

```
paper-scraper/
├── main.py                  # CLI entry point (scrape / vectorize / chat)
├── paper-scraper-build      # Shell script: runs scrape then vectorize
├── paper-scraper            # Shell script: starts the chat session
├── config.json              # Your local configuration (not committed)
├── config.example.json      # Template configuration
├── requirements.txt         # Python dependencies
├── scraper/
│   ├── base.py              # Paper dataclass and BaseScraper ABC
│   ├── pubmed.py            # PubMed / NCBI Entrez scraper
│   └── google_scholar.py    # Google Scholar scraper
├── utils/
│   ├── exporter.py          # SQLite persistence layer
│   ├── vectorizer.py        # ChromaDB embedding pipeline
│   └── chat.py              # Interactive Ollama chat session
└── databases/
    ├── papers.db            # SQLite database (created at runtime)
    └── vectors/             # ChromaDB vector index (created at runtime)
```
