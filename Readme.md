Here is the complete **`README.md`** file documenting everything created for the **Declarative Medallion MLOps Engine**, including architecture details, Windows setup steps, and all complete code files.

---

```markdown
# Declarative Medallion MLOps Engine

A lightweight, declarative data pipeline engine built with **PySpark**, **Delta Lake**, **MLflow**, and **Pydantic**. 

Instead of writing hardcoded ETL scripts for every dataset, this engine uses strongly-typed **YAML configuration files** to dynamically ingest, clean, aggregate, and track data across Bronze, Silver, and Gold Medallion architecture layers.

---

## 🏗️ System Architecture

```text
               ┌───────────────────────────────┐
               │    config/pipeline.yaml       │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │     engine/config_parser.py   │ (Pydantic Schema Validation)
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │       engine/runner.py        │ (PySpark + Delta Engine)
               └───────┬───────┬───────┬───────┘
                       │       │       │
       ┌───────────────┘       │       └───────────────┐
       ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│  🥉 Bronze   │ ───>  │  🥈 Silver   │ ───>  │   🥇 Gold    │
│  (Ingestion) │       │ (Cleansing)  │       │(Aggregation) │
└──────────────┘       └──────────────┘       └──────────────┘
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               ▼
                       ┌──────────────┐
                       │  📈 MLflow   │ (Run & Metric Logging)
                       └──────────────┘

```

---

## 📁 Directory Structure

```text
dbx_mlops/
├── config/
│   └── sample_pipeline.yaml      # Declarative Medallion pipeline spec
├── data/
│   ├── lakehouse/                 # Delta Lake storage location (git-ignored)
│   └── raw/
│       └── customers.csv          # Sample raw source data
├── engine/
│   ├── __init__.py
│   ├── config_parser.py           # Pydantic configuration parser
│   └── runner.py                  # Medallion execution engine
├── hadoop/
│   └── bin/                       # Local Windows native Hadoop binaries
│       ├── winutils.exe
│       └── hadoop.dll
├── run_pipeline.py                # Main pipeline entrypoint
├── sanity_check.py                # Environment verification script
└── README.md

```

---

## ⚙️ Prerequisites & Setup (Windows)

### 1. Python & Dependencies

Install required libraries inside your virtual environment:

```bash
pip install pyspark delta-spark mlflow pydantic pyyaml

```

### 2. Java Development Kit (JDK 17)

PySpark requires JDK 8, 11, or 17. Verify your installation path (e.g., `C:\Users\<User>\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.20.8-hotspot`).

### 3. Hadoop Native Binaries for Windows (`winutils`)

Local Spark execution on Windows requires `winutils.exe` and `hadoop.dll` to manage local filesystems and permissions:

1. Create directory: `hadoop\bin` inside project root.
2. Download `winutils.exe` and `hadoop.dll` (Hadoop version `3.3.5`) from the `cdarlint/winutils` repository into `hadoop/bin/`.
3. The engine automatically injects `HADOOP_HOME` dynamically into the runtime context.

---
s
## 🚀 Execution & UI Inspection

### 1. Execute Pipeline Run

```bash
python run_pipeline.py

```

### 2. View MLflow Dashboard

Launch local MLflow UI to inspect pipeline metrics and parameters:

```bash
mlflow ui

```

Navigate to `http://127.0.0.1:5000` in your web browser.
