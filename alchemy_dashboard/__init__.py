# alchemy_dashboard/__init__.py

"""
Alchemy Dashboard: a Flask web app for running and exploring AlChemy
(lambda-calculus "soup") experiments.

This file marks the 'alchemy_dashboard' folder as a Python package, which is
what lets the modules import each other with relative imports such as
`from .config import DB_NAME`.

Run the app from the repository root with:
    uv run python -m alchemy_dashboard.main
or, inside an activated virtual environment:
    python -m alchemy_dashboard.main

Modules:
    main              Flask app: all web pages and JSON API routes.
    simulation        Runs simulations with the `alchemy` library.
    models            Creates the SQLite tables and saves/deletes experiments.
    db_utils          Reads experiments, populations and metrics from the database.
    config            Settings such as DB_NAME (path to the SQLite database).
    plotting          Bokeh plots: metrics, comparisons, expression trees,
                      multi-experiment dendrograms.
    comparison_plots  Stability plots (Jaccard / Bray-Curtis) for one experiment.
    lineage_plots     Lineage dendrograms for experiments.
    ASTGen            Parses lambda expressions into syntax trees and picks
                      colors for drawing them.

Also in this folder:
    templates/        HTML pages (Jinja templates) rendered by main.py.
    static/           Stylesheet (styles.css) shared by the pages.
"""
