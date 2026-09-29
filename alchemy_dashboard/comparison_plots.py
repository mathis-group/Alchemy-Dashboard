#alchemy comparison_plots.py
"""
Stability plots: how much an experiment's population changes over time.

Each saved snapshot is compared with the snapshot just before it, using two
similarity scores from ecology (both range from 0 to 1, where 1 = no change):

    Jaccard      Only looks at WHICH expressions are present.
                 = (expressions in both snapshots) / (expressions in either).
                 Drops when expressions appear or disappear.
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
from .db_utils import get_comparison_data
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper



def calculate_distance(config_id):
    """Plot Jaccard and Bray-Curtis similarity between consecutive snapshots.

    Only the experiment's 100 most common expressions are included (from
    db_utils.get_comparison_data), and at most about 120 snapshots are used.

    Args:
        config_id (int): The experiment.

    Returns:
        tuple: (script, div) Bokeh components for two side-by-side line
            plots that share the same x-axis (zooming one zooms both).
            (None, None) if the experiment has no saved data.
    """
    df = get_comparison_data(config_id, most=100)
    if df.empty: return None, None

    #create matrix: one row per collision, one column per expression,
    # each cell = how many copies existed (0 if none)
    matrix = df.pivot_table(index="collision_number", 
                            columns="expression", 
                            values="count", aggfunc='sum').fillna(0)
    
    # fix points: keep every Nth snapshot so there are at most ~120.
    # Note that this means "previous snapshot" below may be several saved
    # snapshots back for long experiments.
    snapshot_rate = max(1, len(matrix) // 120) 
    matrix = matrix.iloc[::snapshot_rate]
    
    collisions = matrix.index.tolist()
    counts = matrix.values
    #yes / no count: 1 if the expression is present at all, 0 if not (for Jaccard)
    binary = (counts > 0).astype(int)
    
    jaccard_indices = []
    bray_indices = []
    time_points = []

    #compare snapshot to previous one. The first snapshot has nothing to
    # compare with, so the plot starts at the second one.
    # If both snapshots are empty, the score is set to 0.
    for i in range(1, len(matrix)):
        #calculate jaccard index
        intersection = np.logical_and(binary[i-1], binary[i]).sum()
        union = np.logical_or(binary[i-1], binary[i]).sum()
        j_sim = intersection / union if union != 0 else 0
        
        #bray curtis
        num = np.abs(counts[i-1] - counts[i]).sum()
        den = (counts[i-1] + counts[i]).sum()
        b_sim = 1 - (num / den) if den != 0 else 0
        
        jaccard_indices.append(j_sim)
        bray_indices.append(b_sim)
        time_points.append(collisions[i])

    source = ColumnDataSource(data={
        'x': time_points,
        'jacc': jaccard_indices,
        'bray': bray_indices
    })

   
    #plot for jaccard
    p1 = figure(title="Jaccard Plot",
                x_axis_label="Collision", y_axis_label="Index",
                width=420, height=350, y_range=(0, 1.05))
    p1.line('x', 'jacc', source=source, line_width=2, color="#4F46E5", legend_label="Jaccard")

    # plot for bray curtis 
    p2 = figure(title="Bray-Curtis Plot",
                x_axis_label="Collision", y_axis_label="Index",
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