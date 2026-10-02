"""
Dendrogram (tree) plots that show how an experiment's population changes and
how its expressions relate to each other.

A dendrogram groups similar things together: items that join low in the tree
are very similar, items that only join near the top are very different.

Two modes are used throughout:
    "ward"  Compares whole populations (how many copies of each expression
            there are). Uses Ward's method, which groups populations so that
            each group stays as uniform as possible.
    "edit"  Compares expressions by their text, using Levenshtein (edit)
            distance: the number of single-character changes needed to turn
            one expression into another.

Functions:
    create_dendrogram                   One experiment. Used by main.py
                                        (/database and /get_lineage_analysis).
    create_multi_experiment_dendrogram  Several experiments. Not currently used;
                                        main.py uses the version in plotting.py.

Both return Bokeh (script, div) components. Leaves are placed at
x = 5, 15, 25, ... because that is where scipy's dendrogram() puts them.
"""

import numpy as np
import pandas as pd
import Levenshtein
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar
from bokeh.transform import transform
from bokeh.palettes import Viridis256
from bokeh.embed import components
from .db_utils import get_comparison_data

def create_dendrogram(config_id, mode='ward'):
    """Build a dendrogram for one experiment.

    Uses the experiment's 100 most common expressions (from
    db_utils.get_comparison_data).

    Args:
        config_id (int): The experiment.
        mode (str): "ward" or "edit" (any value other than "ward" is
            treated as "edit").
            "ward": each leaf is a saved collision (snapshot in time).
                Snapshots whose populations look alike are grouped together,
                so you can see phases in the experiment's history. At most
                about 40 snapshots are used. Hovering shows the most common
                expression at that snapshot.
            "edit": each leaf is an expression. Expressions with similar text
                are grouped together. Hovering shows the expression.

    Returns:
        tuple: (script, div) Bokeh components, or (None, None) if the
            experiment has no saved data.
    """
    df = get_comparison_data(config_id, most=100)
    if df.empty: return None, None

    # pivot data into a table: one row per collision, one column per
    # expression, each cell = how many copies existed (0 if none)
    matrix = df.pivot_table(index="collision_number", columns="expression", 
                            values="count", aggfunc='sum').fillna(0)
    
    if mode == 'ward':
        # ward distance logic.
        # Keep every Nth row so there are at most ~40 leaves (keeps it readable)
        snapshot_rate = max(1, len(matrix) // 40)
        matrix = matrix.iloc[::snapshot_rate]
        
        Z = linkage(matrix.values, method='ward')
        # Store numeric collision values for the color mapper
        collision_numbers = matrix.index.tolist()
        labels = [str(c) for c in collision_numbers]
        # hover functionality: the most common expression in each snapshot
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
        
        # edit distance drawing. squareform() converts the square distance
        # table into the condensed form linkage() expects.
        Z = linkage(squareform(dist_matrix), method='average')
        collision_numbers = None
        labels = unique_molecules
        hover_data = unique_molecules 
        hover_label = "Molecule Structure"
        title = "Edit Distance"

    # drawing the dendrograms. no_plot=True: scipy only calculates the
    # layout, and Bokeh does the drawing below.
    ddata = dendrogram(Z, no_plot=True)

    # scipy reorders the leaves so branches don't cross; put the labels and
    # hover text in that same order so each one sits under the right branch
    leaves = ddata['leaves']
    labels = [labels[i] for i in leaves]
    hover_data = [hover_data[i] for i in leaves]

    # branch coordinates (icoord = x values, dcoord = heights of each branch)
    source = ColumnDataSource(data={'xs': ddata['icoord'], 'ys': ddata['dcoord']})

    # leaf coordiantes
    leaf_data = {
        'x': [i*10 + 5 for i in range(len(labels))],
        'y': [0] * len(labels),
        'label': labels,
        'detail': hover_data
    }

    # Reorder numeric collision numbers according to dendrogram leaf order
    if mode == 'ward':
        ordered_collisions = [collision_numbers[i] for i in leaves]
        leaf_data['collision'] = ordered_collisions

    leaf_source = ColumnDataSource(leaf_data)

    p = figure(title=title, width=850, height=450, 
               tools="pan,wheel_zoom,reset,save", background_fill_color="#f8fafc")

    # branches
    p.multi_line('xs', 'ys', source=source, color="#4F46E5", line_width=2, alpha=0.6)

    p.xaxis.ticker = [i*10 + 5 for i in range(len(labels))]

    # Edit mode hides the x-axis labels because lambda expressions are too long to fit
    # Ward mode labels each leaf with its collision number and adds color bar
    if mode == 'ward':
        # Setup continuous linear color mapper across snapshot collision range
        min_col, max_col = min(ordered_collisions), max(ordered_collisions)
        mapper = LinearColorMapper(palette=Viridis256, low=min_col, high=max_col)

        # hover function: invisible circles on each leaf that turn red when the
        # mouse is over them and show the tooltip
        # added colors according to LinearColorMapper
        leaf_renderer = p.circle(
            'x', 'y', source=leaf_source, size=10, 
            fill_color=transform('collision', mapper),
            line_color="#4F46E5", line_width=1,
            hover_fill_alpha=0.3, hover_fill_color="red"
        )

        # Add the ColorBar below the figure
        color_bar = ColorBar(
            color_mapper=mapper,
            title="Collision Number",
            title_text_font_size="9pt",
            title_text_font_style="bold",
            location=(0, 0),
            height=12
        )
        p.add_layout(color_bar, 'below')

        p.xaxis.major_label_overrides = {i*10 + 5: str(label) for i, label in enumerate(labels)}
        p.xaxis.major_label_orientation = "vertical"
        p.xaxis.major_label_text_font_size = "9pt"

    else:
        # hover function: invisible circles on each leaf that turn red when the
        # mouse is over them and show the tooltip
        leaf_renderer = p.circle('x', 'y', source=leaf_source, size=15, 
                                 fill_alpha=0, line_alpha=0, hover_fill_alpha=0.3, hover_fill_color="red")
        
        p.xaxis.major_label_text_color = None
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None

    # Add HoverTool
    hover = HoverTool(renderers=[leaf_renderer], tooltips=[
        ("Collision" if mode == 'ward' else "Name", "@label"),
        (hover_label, "@detail")
    ])
    p.add_tools(hover)

    p.xaxis.ticker = [i*10 + 5 for i in range(len(labels))]

    return components(p)

#create dendrogram for multiple experiments and compare
def create_multi_experiment_dendrogram(config_ids, mode='ward'):
    """Build one dendrogram comparing several experiments.

    Not currently used: main.py calls plotting.create_multi_experiment_dendrogram
    instead, which works like the "edit" mode here but colors leaves by
    experiment.

    Args:
        config_ids (list[int]): Experiments to compare.
        mode (str):
            "ward": each leaf is a whole experiment, compared by its final
                population. Experiments that ended up with similar
                populations are grouped together.
            "edit": each leaf is an expression, taken from the 50 most common
                expressions of every experiment and pooled together.
                Expressions with similar text are grouped together.

    Returns:
        tuple: (script, div) Bokeh components. Returns None if mode is
            neither "ward" nor "edit".

    Raises:
        ValueError: if none of the experiments have data to compare.
    """
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

        # Build a table: one row per experiment, one column per expression
        # seen in any experiment, each cell = its count (0 if absent)
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
        #Levenstein 
        from scipy.spatial.distance import squareform
        from sklearn.metrics import pairwise_distances
        from bokeh.models import ColumnDataSource, HoverTool

        all_unique_expressions = set()
        
        # Top 50 survivors from every experiment (shared expressions only appear once)
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
        p.yaxis.axis_label = "Edit Distance"
        
        return components(p)