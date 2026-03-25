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

#create dendrogram for multiple experiments and compare
def create_multi_experiment_dendrogram(config_ids, mode='ward'):
    from scipy.cluster.hierarchy import linkage, dendrogram
    import pandas as pd
    import numpy as np
    from bokeh.plotting import figure
    from bokeh.embed import components
    from .db_utils import get_expressions_for_collision, get_experiment_details, get_comparison_data

    if mode == 'ward':
        # --- MACRO ECOSYSTEM COMPARISON ---
        final_states = {}
        all_unique_expressions = set()
        experiment_labels = []

        for cid in config_ids:
            config, _, _ = get_experiment_details(cid)
            if not config: continue
            label = f"Exp {cid} (Seed: {config[1]})"
            experiment_labels.append(label)

            state = get_expressions_for_collision(cid, -1)
            if not state: continue
            
            state_dict = dict(state) 
            final_states[label] = state_dict
            all_unique_expressions.update(state_dict.keys())

        if not final_states:
            raise ValueError("No valid final state data found.")

        records = []
        for label in experiment_labels:
            if label not in final_states: continue
            row = {'Label': label}
            for expr in all_unique_expressions:
                row[expr] = final_states[label].get(expr, 0)
            records.append(row)

        df = pd.DataFrame(records).set_index('Label')
        Z = linkage(df.values, method='ward')
        ddata = dendrogram(Z, no_plot=True)

        p = figure(title="Meta-Ecosystem Comparison (Ward Distance)", 
                   height=500, sizing_mode="stretch_width",
                   toolbar_location="above", tools="pan,wheel_zoom,box_zoom,reset,save")
        
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            p.line(i, d, line_color="#4F46E5", line_width=2)

        leaves = ddata['leaves']
        labels = [df.index[leaf] for leaf in leaves]
        tick_locs = [(i * 10) + 5 for i in range(len(leaves))] 
        
        p.xaxis.ticker = tick_locs
        p.xaxis.major_label_overrides = {loc: label for loc, label in zip(tick_locs, labels)}
        p.xaxis.major_label_orientation = 0.8 
        p.yaxis.axis_label = "Population Variance"

        return components(p)

    elif mode == 'edit':
        # --- MICRO GENETICS COMPARISON (LEVENSHTEIN) ---
        from scipy.spatial.distance import squareform
        from sklearn.metrics import pairwise_distances
        from bokeh.models import ColumnDataSource, HoverTool

        all_unique_expressions = set()
        
        # Pool the top 50 survivors from EVERY selected experiment
        for cid in config_ids:
            df = get_comparison_data(cid, most=50) 
            if not df.empty:
                all_unique_expressions.update(df['expression'].unique())
        
        unique_molecules = list(all_unique_expressions)
        
        if not unique_molecules:
            raise ValueError("No expressions found to compare.")
            
        # Calculate Levenshtein typos across the giant pooled bucket
        dist_matrix = pairwise_distances(
            np.array(unique_molecules).reshape(-1, 1), 
            metric=lambda x, y: Levenshtein.distance(str(x[0]), str(y[0]))
        )
        
        Z = linkage(squareform(dist_matrix), method='average')
        ddata = dendrogram(Z, no_plot=True)
        
        p = figure(title="Cross-Experiment Structural Similarity (Edit Distance)", 
                   height=500, sizing_mode="stretch_width",
                   toolbar_location="above", tools="pan,wheel_zoom,box_zoom,reset,save")
        
        # Draw the branches in a different color to distinguish modes
        source = ColumnDataSource(data={'xs': ddata['icoord'], 'ys': ddata['dcoord']})
        p.multi_line('xs', 'ys', source=source, color="#10B981", line_width=2, alpha=0.8)
        
        # Add the HoverTool so you can see the molecules on the X-axis
        labels = unique_molecules
        leaves = ddata['leaves']
        ordered_labels = [labels[leaf] for leaf in leaves]
        
        leaf_source = ColumnDataSource(data={
            'x': [(i * 10) + 5 for i in range(len(ordered_labels))],
            'y': [0] * len(ordered_labels),
            'detail': ordered_labels
        })
        
        leaf_renderer = p.circle('x', 'y', source=leaf_source, size=15, 
                                 fill_alpha=0, line_alpha=0, hover_fill_alpha=0.5, hover_fill_color="red")
        
        hover = HoverTool(renderers=[leaf_renderer], tooltips=[("Molecule", "@detail")])
        p.add_tools(hover)
        
        # Hide the text on the X-axis because Lambda strings are too long
        p.xaxis.ticker = [(i * 10) + 5 for i in range(len(ordered_labels))]
        p.xaxis.major_label_text_color = None
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None
        p.yaxis.axis_label = "Mutation Count (Edit Distance)"
        
        return components(p)