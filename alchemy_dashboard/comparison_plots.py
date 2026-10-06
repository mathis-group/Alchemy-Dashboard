#alchemy comparison_plots.py
"""
Stability plots: how much an experiment's population changes over time.

Each saved snapshot is compared with the snapshot just before it, using two
similarity scores from ecology (both range from 0 to 1, where 1 = no change):

    Jaccard      Abundance-weighted (multiset) Jaccard, the same definition
                 Modern-AlChemy uses: for each expression take the smaller and
                 the larger of its two counts, then
                 = sum(smaller counts) / sum(larger counts).
    Bray-Curtis  Also looks at HOW MANY copies of each expression there are.
                 = 1 - sum(|count difference|) / sum(both counts).
                 Drops when the counts shift, even if the same expressions
                 are present.

main.py imports calculate_distance() as `run_ordination` for the
/get_distance_analysis route. Despite the names, it plots similarity between
consecutive snapshots; it does not do an ordination (MDS) plot.
"""
import pandas as pd
import numpy as np
import sqlite3
from bokeh.embed import components
from scipy.spatial.distance import pdist,squareform
from sklearn.manifold import MDS
from bokeh.plotting import figure
from bokeh.layouts import gridplot, column
from bokeh.models import ColumnDataSource, HoverTool, Slider, CustomJS
from .plotting import create_styled_figure, PRIMARY_COLOR, ACCENT_COLOR
from .config import DB_NAME
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper



def calculate_distance(config_id):
    """Plot Jaccard and Bray-Curtis similarity between consecutive snapshots.

    Uses the full population at every saved snapshot, so the values do not
    depend on run length or on how many expressions there are.

    Args:
        config_id (int): The experiment.

    Returns:
        tuple: (script, div) Bokeh components for two side-by-side line
            plots that share the same x-axis (zooming one zooms both).
            (None, None) if the experiment has no saved data.
    """
    conn = sqlite3.connect(DB_NAME)
    try:
        # Every saved expression count, at every saved snapshot.
        # Imported experiments also store the final state as collision -1,
        # which duplicates the last snapshot, so it is skipped.
        df = pd.read_sql_query(
            "SELECT collision_number, expression, count FROM Experiment "
            "WHERE config_id=? AND collision_number >= 0",
            conn, params=(config_id,))
    finally:
        conn.close()
    if df.empty: return None, None

    # one {expression: count} dict per snapshot, in collision order
    snapshots = [
        (collision, dict(zip(group["expression"], group["count"])))
        for collision, group in df.groupby("collision_number", sort=True)
    ]

    jaccard_indices = []
    bray_indices = []
    time_points = []

    # Compare each snapshot with the one before it. The first snapshot has
    # nothing to compare with, so the plot starts at the second one.
    # If both snapshots are empty they count as identical (1), as in Rust.
    for (_, previous), (collision, current) in zip(snapshots, snapshots[1:]):
        expressions = previous.keys() | current.keys()
        smaller = sum(min(previous.get(e, 0), current.get(e, 0)) for e in expressions)
        larger = sum(max(previous.get(e, 0), current.get(e, 0)) for e in expressions)
        total = sum(previous.values()) + sum(current.values())

        # weighted Jaccard: sum of smaller counts / sum of larger counts
        j_sim = smaller / larger if larger else 1.0
        # Bray-Curtis similarity: 1 - sum|a - b| / sum(a + b) = 2 * smaller / total
        b_sim = 2 * smaller / total if total else 1.0

        jaccard_indices.append(j_sim)
        bray_indices.append(b_sim)
        time_points.append(collision)

    source = ColumnDataSource(data={
        'x': time_points,
        'jacc': jaccard_indices,
        'bray': bray_indices
    })

   
    #plot for jaccard
    p1 = figure(title="Jaccard (abundance-weighted)",
                x_axis_label="Collision", y_axis_label="Similarity to previous snapshot",
                width=420, height=350, y_range=(0, 1.05))
    p1.line('x', 'jacc', source=source, line_width=2, color="#4F46E5", legend_label="Jaccard")

    # plot for bray curtis 
    p2 = figure(title="Bray-Curtis",
                x_axis_label="Collision", y_axis_label="Similarity to previous snapshot",
                width=420, height=350, y_range=(0, 1.05),
                x_range=p1.x_range)
    p2.line('x', 'bray', source=source, line_width=2, color="#10B981", legend_label="Bray-Curtis")

    
    # One hover tool shared by both plots; "Stability" is the y value
    # under the mouse
    hover = HoverTool(tooltips=[("Collision", "@x"), ("Stability", "$y{0.000}")])
    p1.add_tools(hover)
    p2.add_tools(hover)

    
    for p in [p1, p2]:
        p.legend.location = "bottom_right"
        p.background_fill_color = "#f8fafc"
        p.grid.grid_line_color = "white"

    layout = gridplot([[p1, p2]], sizing_mode='scale_width')
    
    script, div = components(layout)
    return script, div