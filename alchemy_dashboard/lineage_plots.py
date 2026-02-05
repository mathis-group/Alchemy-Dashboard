import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.embed import components
from .db_utils import get_comparison_data

def create_dendrogram(config_id):
    df = get_comparison_data(config_id, most=100)
    if df.empty: return None, None

    #data table
    matrix = df.pivot_table(index="collision_number", 
                            columns="expression", 
                            values="count", 
                            aggfunc='sum').fillna(0)
    
    # filter how many time frames will appear
    snapshot_rate = max(1, len(matrix) // 30)
    matrix = matrix.iloc[::snapshot_rate]
    
    #find most dominnat expression in each snapshot
    dominant_exprs = matrix.idxmax(axis=1).tolist()
    #index each time stamp to each expression
    collisions = matrix.index.astype(str).tolist()

    # group similar expressions
    Z = linkage(matrix.values, method='ward')

    #draw dendrogram
    data = dendrogram(Z, no_plot=True)

    source = ColumnDataSource(data={'xs': data['icoord'], 'ys': data['dcoord']})

    # draw dendrogram
    num_leaves = len(collisions)
    #center leaves
    x_positions = [i * 10 + 5 for i in range(num_leaves)]

    leaf_source = ColumnDataSource(data={
        'x': x_positions,
        'y': [0] * num_leaves,
        'collision': collisions,
        'molecule': dominant_exprs
    })

    p = figure(title="Dendrogram",
               x_axis_label="Snapshots (Hover for Molecule)", y_axis_label="Ward Distance",
               width=850, height=450, tools="pan,wheel_zoom,reset,save")


    p.multi_line('xs', 'ys', source=source, color="#4F46E5", line_width=2, alpha=0.6)

    leaf_renderer = p.circle('x', 'y', source=leaf_source, size=15, 
                             fill_alpha=0, line_alpha=0, hover_fill_alpha=0.3, hover_fill_color="red")

  
    hover = HoverTool(renderers=[leaf_renderer], tooltips=[
        ("Collision", "@collision"),
        ("Dominant Molecule", "@molecule")
    ])
    p.add_tools(hover)

    #center x axis
    p.xaxis.ticker = [i*10 + 5 for i in range(len(collisions))]
    #center lables
    p.xaxis.major_label_overrides = {i*10 + 5: collisions[i] for i in range(len(collisions))}
    p.xaxis.major_label_orientation = 0.8
    p.grid.grid_line_color = None
    p.background_fill_color = "white"

    return components(p)