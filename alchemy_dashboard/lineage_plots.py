import numpy as np
import pandas as pd
import Levenshtein
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.embed import components
from .db_utils import get_comparison_data

def create_dendrogram(config_id, mode='ward'):
    df = get_comparison_data(config_id, most=100)
    if df.empty: return None, None

    # pivot data
    matrix = df.pivot_table(index="collision_number", columns="expression", 
                            values="count", aggfunc='sum').fillna(0)
    
    if mode == 'ward':
        # ward distance logic
        snapshot_rate = max(1, len(matrix) // 40)
        matrix = matrix.iloc[::snapshot_rate]
        
        Z = linkage(matrix.values, method='ward')
        labels = matrix.index.astype(str).tolist()
        # hover functionality
        hover_data = matrix.idxmax(axis=1).tolist() 
        hover_label = "Dominant Molecule"
        title = "Ward Distance"
    else:
        # edit distance logic
        unique_molecules = matrix.columns.tolist()
        
        # calculate pairwise distance
        dist_matrix = pairwise_distances(
            np.array(unique_molecules).reshape(-1, 1), 
            metric=lambda x, y: Levenshtein.distance(str(x[0]), str(y[0]))
        )
        
        # edit distance drawing
        Z = linkage(squareform(dist_matrix), method='average')
        labels = unique_molecules
        hover_data = unique_molecules 
        hover_label = "Molecule Structure"
        title = "Edit Distance"

    # drawing the dendrograms
    ddata = dendrogram(Z, no_plot=True)
    
    # branch coordinates
    source = ColumnDataSource(data={'xs': ddata['icoord'], 'ys': ddata['dcoord']})

    # leaf coordiantes
    leaf_source = ColumnDataSource(data={
        'x': [i*10 + 5 for i in range(len(labels))],
        'y': [0] * len(labels),
        'label': labels,
        'detail': hover_data
    })

    p = figure(title=title, width=850, height=450, 
               tools="pan,wheel_zoom,reset,save", background_fill_color="#f8fafc")

    # branches
    p.multi_line('xs', 'ys', source=source, color="#4F46E5", line_width=2, alpha=0.6)

    # hover function
    leaf_renderer = p.circle('x', 'y', source=leaf_source, size=15, 
                             fill_alpha=0, line_alpha=0, hover_fill_alpha=0.3, hover_fill_color="red")

    hover = HoverTool(renderers=[leaf_renderer], tooltips=[
        ("Name", "@label"),
        (hover_label, "@detail")
    ])
    p.add_tools(hover)

    p.xaxis.ticker = [i*10 + 5 for i in range(len(labels))]

    if mode == 'edit':
       
        p.xaxis.major_label_text_color = None
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None
    else:
        p.xaxis.major_label_overrides = {i*10 + 5: str(label) for i, label in enumerate(labels)}
        p.xaxis.major_label_orientation = "vertical"
        p.xaxis.major_label_text_font_size = "9pt"

    return components(p)