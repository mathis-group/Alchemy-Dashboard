#alchemy comparison_plots.py
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
    df = get_comparison_data(config_id, most=100)
    if df.empty: return None, None

    #ccreate matrix
    matrix = df.pivot_table(index="collision_number", 
                            columns="expression", 
                            values="count", aggfunc='sum').fillna(0)
    
    # fix points
    snapshot_rate = max(1, len(matrix) // 120) 
    matrix = matrix.iloc[::snapshot_rate]
    
    collisions = matrix.index.tolist()
    counts = matrix.values
    #yes / no count
    binary = (counts > 0).astype(int)
    
    jaccard_indices = []
    bray_indices = []
    time_points = []

    #compare snapshot to previous one 
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