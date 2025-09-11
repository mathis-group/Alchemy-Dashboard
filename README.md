# AlChemy Chemistry

**AlChemy Chemistry** is a reimplementation of Walter Fontana’s *Alchemy* artificial chemistry, using lambda-calculus expressions as the “molecules” in a chemical soup. The core engine repeatedly “collides” lambda terms and applies reaction rules to produce new expressions, enabling studies of emergent behavior like expression diversity and population entropy over time. The system couples a high‑performance simulation core with an interactive Bokeh dashboard for configuring experiments and visualizing results.

![System Overview](system_overview.png)

---

## Key Components

- **Core Simulation (Rust, exposed to Python):**  
  Generates lambda expressions, applies reaction rules during random collisions, tracks outcomes, and writes results to disk.
- **Visualization Dashboard (Python + Bokeh):**  
  Lets users configure experiments, run/monitor simulations, and explore interactive plots of system dynamics (e.g., diversity, entropy).


---

## Repository Layout

```
alchemy_dashboard/
├── __init__.py
├── config.py
├── db_utils.py
├── main.py
├── models.py
├── plotting.py
├── simulation.py
├── static/
├── templates/
└── uploaded_configs/
```

> `uploaded_configs/` is a convenience folder for JSON experiment configs that the dashboard can load.  
> `db_utils.py` handles lightweight persistence (SQLite) for experiment metadata/results.

---

## Prerequisites

- **Rust toolchain** (stable) with `cargo` installed: <https://rustup.rs>
- **Python 3.7+** (3.8+ recommended)
- **Pip / build tools** (the core uses **maturin** to build the Rust library as a Python package)
- **Python libraries:** `bokeh`, `numpy` (plus standard library `sqlite3` for SQLite).


---

## Quickstart

### 1) Clone

```bash
git clone https://github.com/jjoseph12/AlChemy_Chemistry.git
cd AlChemy_Chemistry
```

### 2) Create & activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3) Build & install the Rust core (via maturin)

Install maturin and build from the Rust crate directory (adjust `-m` if your `Cargo.toml` lives elsewhere).

```bash
pip install maturin
# from the crate root (where Cargo.toml is located):
maturin develop --release
```

This compiles the Rust code into a Python extension module and installs it into your active environment so the dashboard can import it (e.g., `import alchemy` or your chosen package name).

### 4) Install Python runtime deps

 minimally:
```bash
pip install bokeh numpy
```

### 5) Run the dashboard

Depending on how your app is structured, use one of the following patterns:

**Option A: Bokeh server app directory**
```bash
bokeh serve alchemy_dashboard --show
```

**Option B: Python entrypoint**
```bash
python -m alchemy_dashboard.main
# or, if main.py is runnable directly:
python alchemy_dashboard/main.py
```

The app will open in your browser and allow you to select or upload configs (e.g., from `uploaded_configs/`), launch simulations, and explore interactive plots.

---

## Using Experiment Configs

- Place JSON config files in `uploaded_configs/` or use the UI to upload.  
- Each config typically defines parameters for generating initial lambda terms, reaction rules, runtime (number of collisions/steps), and output options.
- Results are saved in JSON or SQLite (lightweight) form, which the dashboard loads for visualization.

---

## Data & Storage

- **SQLite:** Used for simple experiment cataloging and result indexing (via Python’s built‑in `sqlite3`).  
- **JSON:** Simulation outputs can also be written as JSON blobs which the dashboard can load for ad‑hoc analyses.

---

## Troubleshooting

- **Can’t import the core package?** Re‑run `maturin develop --release` from the Rust crate root inside your active virtualenv.  
- **Bokeh doesn’t open automatically?** Visit the printed local URL (e.g., http://localhost:5006/alchemy_dashboard) manually.  
- **Version mismatch:** Verify `python --version` and `rustc --version` are sane; upgrade `pip`.

---

## Acknowledgments

- Inspired by Walter Fontana’s *Alchemy* artificial chemistry.  
- Built with Rust, PyO3/maturin, and Bokeh.

---

## License
