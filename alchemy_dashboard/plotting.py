# alchemy_dashboard/plotting.py
"""
Bokeh plots for the Alchemy Dashboard.

Every function here builds a Bokeh figure (or the (script, div) HTML pieces
for one) that main.py sends to the browser.

Main groups of functions:
    Shared styling        create_styled_figure, color constants
    Metric plots          plot_experiment_metrics, create_bokeh_plots_from_metrics,
                          plot_comparison_metrics
    Plots from JSON files plot_simulation_metrics, get_simulation_components,
                          create_bokeh_from_data
    Plots from the DB     query_df_by_config_id, generate_bokeh_components
    Expression trees      ASTvisualizer, ASTErr
    Multi-experiment      create_multi_experiment_dendrogram

"Entropy" measures how evenly the population is spread across different
expressions; "unique expressions" is how many distinct expressions exist.
Both are recorded at each sampled collision.
"""

from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, Legend, Circle, LabelSet
from bokeh.embed import components
from .ASTGen import LambdaParser, VariableNode, LambdaNode, getColors
import pandas as pd
import json
# new imports for jaccard
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
# New imports for non-overlapping AST
import networkx as nx
from collections import defaultdict

# Define colors using CSS variables.
# These are copies of the website's CSS colors so plots match the page;
# if the CSS theme changes, update them here too.
PRIMARY_COLOR = "#4F46E5"  # var(--primary)
SECONDARY_COLOR = "#0EA5E9"  # var(--secondary)
ACCENT_COLOR = "#F59E0B"  # var(--accent)
GRID_COLOR = "#E2E8F0"  # var(--border)
TEXT_COLOR = "#1E293B"  # var(--text-primary)

# Define plot colors: one per line when several experiments share a plot
# (repeats after 6)
PLOT_COLORS = [
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    ACCENT_COLOR,
    "#10B981",  # var(--success)
    "#EF4444",  # var(--error)
    "#F59E0B",  # var(--warning)
]

def create_styled_figure(title, x_label, y_label, width=800, height=300):
    """Create a styled Bokeh figure with consistent formatting.

    Use this instead of bokeh's figure() so all plots share the same fonts,
    colors, grid, and toolbar.

    Args:
        title (str): Title shown above the plot.
        x_label (str): X-axis label.
        y_label (str): Y-axis label.
        width (int): Width in pixels.
        height (int): Height in pixels.

    Returns:
        Figure: An empty Bokeh figure ready for lines/points to be added.
    """
    fig = figure(
        title=title,
        x_axis_label=x_label,
        y_axis_label=y_label,
        width=width,
        height=height,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        background_fill_color="#FFFFFF",
        border_fill_color="#FFFFFF",
        outline_line_color=None,
        min_border=20
    )

    fig.xgrid.grid_line_color = GRID_COLOR
    fig.ygrid.grid_line_color = GRID_COLOR
    fig.xgrid.grid_line_alpha = 0.5
    fig.ygrid.grid_line_alpha = 0.5

    fig.axis.axis_line_color = "#BDBDBD"
    fig.axis.major_tick_line_color = "#BDBDBD"
    fig.axis.major_label_text_font = "Poppins"
    fig.axis.major_label_text_color = TEXT_COLOR
    fig.axis.major_label_text_font_size = "11px"

    fig.title.text_font = "Poppins"
    fig.title.text_font_size = "16px"
    fig.title.text_font_style = "bold"
    fig.title.text_color = TEXT_COLOR
    fig.title.align = "center"
    fig.title.text_alpha = 0.85

    fig.xaxis.axis_label_text_font = "Poppins"
    fig.xaxis.axis_label_text_font_size = "13px"
    fig.xaxis.axis_label_text_font_style = "normal"
    fig.xaxis.axis_label_text_color = TEXT_COLOR
    fig.xaxis.axis_label_text_alpha = 0.75

    fig.yaxis.axis_label_text_font = "Poppins"
    fig.yaxis.axis_label_text_font_size = "13px"
    fig.yaxis.axis_label_text_font_style = "normal"
    fig.yaxis.axis_label_text_color = TEXT_COLOR
    fig.yaxis.axis_label_text_alpha = 0.75

    fig.min_border_left = 40
    fig.min_border_right = 40
    fig.min_border_top = 30
    fig.min_border_bottom = 40

    return fig





from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS

def plot_experiment_metrics(df):
    """
    Create plots for a single experiment's metrics.

    Clicking a point on the entropy plot loads a histogram of the expressions
    at that collision (see the JavaScript callback below).

    Args:
        df (pandas.DataFrame): DataFrame containing metrics data, with columns
            collision_number, entropy, and unique_expressions_count.
            "unique_expressions" or "len_unique_expressions" are accepted
            and renamed.

    Returns:
        dict: Dictionary of Bokeh figure objects, keyed by plot type
            ("entropy_plot" and "unique_expressions_plot")
    """
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, LassoSelectTool
    from bokeh.layouts import column
    
    # Debug: Print DataFrame info to understand the data structure
    print(f"[DEBUG] DataFrame shape: {df.shape}")
    print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")
    print(f"[DEBUG] DataFrame head:\n{df.head()}")
    
    # Check if required columns exist
    required_columns = ['collision_number', 'entropy', 'unique_expressions_count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"[ERROR] Missing columns: {missing_columns}")
        # Try to find alternative column names
        if 'unique_expressions' in df.columns:
            df = df.rename(columns={'unique_expressions': 'unique_expressions_count'})
            print("[INFO] Renamed 'unique_expressions' to 'unique_expressions_count'")
        elif 'len_unique_expressions' in df.columns:
            df = df.rename(columns={'len_unique_expressions': 'unique_expressions_count'})
            print("[INFO] Renamed 'len_unique_expressions' to 'unique_expressions_count'")
    
    # Create separate ColumnDataSources for each plot
    entropy_source = ColumnDataSource(df)
    unique_expressions_source = ColumnDataSource(df)
    
    # Create entropy plot
    entropy_plot = create_styled_figure("Entropy Over Time", "Collision Number", "Entropy", width=800, height=400)
    
    # Add line and scatter for entropy, using its own source
    entropy_plot.line('collision_number', 'entropy', source=entropy_source, line_width=2, color=PRIMARY_COLOR)
    entropy_plot.scatter('collision_number', 'entropy', source=entropy_source, 
                        size=8, color=PRIMARY_COLOR, alpha=0.6, selection_color='red',
                        nonselection_alpha=0.1)
    
    # Add hover tool for entropy
    entropy_hover = HoverTool(
        tooltips=[
            ("Collision", "@collision_number"),
            ("Entropy", "@entropy{0.0000}")
        ]
    )
    entropy_plot.add_tools(entropy_hover)
    
    # Create unique expressions plot
    unique_expressions_plot = create_styled_figure("Unique Expressions Over Time", "Collision Number", "Unique Expressions Count", width=800, height=400)
    
    # Debug: Check if unique_expressions_count column exists and has data
    if 'unique_expressions_count' in df.columns:
        print(f"[DEBUG] Unique expressions data: {df['unique_expressions_count'].tolist()}")
        
        # Add line and scatter for unique expressions, using its own source
        unique_expressions_plot.line('collision_number', 'unique_expressions_count', source=unique_expressions_source, 
                                     line_width=2, color=SECONDARY_COLOR)
        unique_expressions_plot.scatter('collision_number', 'unique_expressions_count', source=unique_expressions_source, 
                                        size=8, color=SECONDARY_COLOR, alpha=0.6, selection_color='red',
                                        nonselection_alpha=0.1)
        
        # Add hover tool for unique expressions
        unique_expressions_hover = HoverTool(
            tooltips=[
                ("Collision", "@collision_number"),
                ("Unique Expressions", "@unique_expressions_count")
            ]
        )
        unique_expressions_plot.add_tools(unique_expressions_hover)
    else:
        print(f"[ERROR] 'unique_expressions_count' column not found. Available columns: {df.columns.tolist()}")
        # Create an empty plot with error message
        unique_expressions_plot.text(x=[400], y=[200], text=["No unique expressions data available"], 
                                   text_font_size="16px", text_color="red")
    
    # Add TapTool to entropy plot
    entropy_plot.add_tools(TapTool())

    # Add JS callback for tap. This runs in the browser: it fetches
    # /get_entropy_detail for the clicked collision and puts the result into
    # the element with id "histogram-content". The page must set
    # window.currentConfigID to the experiment being shown.
    tap_callback = CustomJS(args=dict(source=entropy_source), code="""
        const selected_index = source.selected.indices[0];
        if (selected_index != null) {
            const collision = source.data['collision_number'][selected_index];
            const configId = window.currentConfigID;
            if (!configId || configId === 'undefined') {
                console.error('No valid config ID available');
                return;
            }
            fetch(`/get_entropy_detail/${collision}?config_id=${configId}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.text();
                })
                .then(html => {
                    const target = document.getElementById("histogram-content");
                    if (target) {
                        target.innerHTML = html;
                        target.scrollIntoView({ behavior: "smooth" });
                    }
                })
                .catch(error => {
                    console.error('Error fetching histogram:', error);
                    const target = document.getElementById("histogram-content");
                    if (target) {
                        target.innerHTML = '<p style="color: red;">Error loading histogram. Please try again.</p>';
                    }
                });
        }
    """)
    entropy_source.selected.js_on_change('indices', tap_callback)

    return {
        'entropy_plot': entropy_plot,
        'unique_expressions_plot': unique_expressions_plot
    }









def plot_comparison_metrics(metric_data, metric_name):
    """
    Create a comparison plot for multiple experiments.
    
    Draws one line per experiment. Clicking an entry in the legend hides or
    shows that experiment's line.

    Args:
        metric_data (dict): Dictionary mapping config_id to experiment info, as
            returned by db_utils.get_experiment_metrics(). Each value is a dict
            with "data" (a DataFrame with collision_number and the metric
            column), plus "generator_type" and "random_seed" for the legend.
        metric_name (str): Name of the metric to plot ('entropy' or 'unique_expressions')
        
    Returns:
        Figure: Bokeh figure with comparison plot
    """
    if metric_name == 'entropy':
        title = "Entropy Comparison"
        y_label = "Entropy"
    elif metric_name == 'unique_expressions':
        title = "Unique Expressions Comparison"
        y_label = "Count"
    else:
        title = "Metric Comparison"
        y_label = "Value"
        
    comparison_plot = create_styled_figure(title, "Collision Number", y_label, width=800, height=500)
    
    legend_items = []
    for i, (config_id, experiment) in enumerate(metric_data.items()):
        df = experiment['data']
        generator_type = experiment.get('generator_type', 'Unknown')
        random_seed = experiment.get('random_seed', 'Unknown')
        
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        x_values = df['collision_number']
        y_values = df[metric_name]
        
        line = comparison_plot.line(
            x=x_values, 
            y=y_values, 
            line_width=2.5, 
            color=color, 
            alpha=0.8
        )
        
        scatter = comparison_plot.scatter(
            x=x_values, 
            y=y_values, 
            size=6, 
            color=color, 
            alpha=0.6
        )
        
        legend_items.append((f"{generator_type} (ID: {config_id}, Seed: {random_seed})", [line, scatter]))
        
    legend = Legend(items=legend_items, location="top_right")
    legend.click_policy = "hide"
    comparison_plot.add_layout(legend)
    
    return comparison_plot

def plot_simulation_metrics(results):
    """
    Generate plots from simulation results data.

    Args:
        results (dict): Dictionary with simulation results. Its
            "collisions_data" can be either:
              - old format: a dict keyed like "collision_100", where each
                value has "entropy" and a list of "unique_expressions"
              - new format: a list of dicts with "collision_number",
                "entropy", and "unique_expressions" (a count)
        
    Returns:
        list: List of Bokeh figure objects
    """
    # Extract data from results dictionary
    collisions_data = results.get("collisions_data", {})
    
    x = []
    entropy_y = []
    unique_expressions_y = []
    
    # Handle both new database format and older JSON format
    if isinstance(collisions_data, dict):  # Old JSON format
        for key in sorted(collisions_data.keys(), key=lambda k: int(k.split("_")[1]) if "_" in k else 0):
            collision_number = int(key.split("_")[1]) if "_" in key else 0
            entry = collisions_data[key]
            
            x.append(collision_number)
            entropy_y.append(entry.get("entropy", 0))
            unique_expressions_y.append(len(entry.get("unique_expressions", [])))
    else:  # New format (list of dicts)
        for entry in sorted(collisions_data, key=lambda e: e.get("collision_number", 0)):
            x.append(entry.get("collision_number", 0))
            entropy_y.append(entry.get("entropy", 0))
            unique_expressions_y.append(entry.get("unique_expressions", 0))
    
    # Create plots
    entropy_plot = create_styled_figure("Entropy Over Time", "Collision Number", "Entropy")
    entropy_plot.line(x, entropy_y, line_width=2.5, color=PRIMARY_COLOR, line_alpha=0.8)
    entropy_plot.scatter(x, entropy_y, size=6, color=PRIMARY_COLOR, alpha=0.6)
    
    unique_plot = create_styled_figure("Unique Expressions Over Time", "Collision Number", "Count")
    unique_plot.line(x, unique_expressions_y, line_width=2.5, color=SECONDARY_COLOR, line_alpha=0.8)
    unique_plot.scatter(x, unique_expressions_y, size=6, color=SECONDARY_COLOR, alpha=0.6)
    
    return [entropy_plot, unique_plot]

def create_bokeh_from_data(data):
    """
    Create Bokeh components from uploaded JSON data.

    Used by the /generate_visuals route. Unlike the other plots, these use
    Bokeh's default styling.

    Args:
        data (dict): Parsed JSON data from uploaded file. Must have a
            "collisions_data" dict keyed like "collision_100"; keys without
            an underscore are treated as collision 0.
        
    Returns:
        tuple: (script, div) tuple for Bokeh components
    """
    # Extract data
    collisions_data = data.get("collisions_data", {})
    
    x = []
    entropy_y = []
    unique_expr_y = []
    
    for key in sorted(collisions_data.keys(), key=lambda x: int(x.split("_")[1]) if "_" in x else 0):
        collision_number = int(key.split("_")[1]) if "_" in key else 0
        entry = collisions_data[key]
        
        x.append(collision_number)
        entropy_y.append(entry.get("entropy", 0))
        
        # Handle different ways unique expressions might be stored
        if "unique_expressions" in entry:
            if isinstance(entry["unique_expressions"], list):
                unique_expr_y.append(len(entry["unique_expressions"]))
            else:
                unique_expr_y.append(entry["unique_expressions"])
        elif "len_unique_expressions" in entry:
            unique_expr_y.append(entry.get("len_unique_expressions", 0))
        else:
            unique_expr_y.append(0)
    
    # Create plots
    p1 = figure(title="Entropy Over Time", x_axis_label="Collisions", y_axis_label="Entropy", width=600, height=300)
    p1.line(x, entropy_y, line_width=2, legend_label="Entropy")

    p2 = figure(title="Unique Expressions Over Time", x_axis_label="Collisions", y_axis_label="Unique Count", width=600, height=300)
    p2.line(x, unique_expr_y, line_width=2, color="green", legend_label="Unique Expressions")

    script1, div1 = components(p1)
    script2, div2 = components(p2)

    combined_script = script1 + "\n" + script2
    combined_div = div1 + "\n" + div2

    return combined_script, combined_div



def create_bokeh_plots_from_metrics(metrics_data, title_prefix=""):
    """
    Create Bokeh plots from a list of metrics data.
    
    Args:
        metrics_data (list): List of tuples containing metrics data, each
            (collision_number, entropy, unique_expressions)
        title_prefix (str): Optional prefix for plot titles (currently unused)

    Returns:
        dict: Same as plot_experiment_metrics(): {"entropy_plot",
            "unique_expressions_plot"}
    """
    # Process data into DataFrame
    data = []
    for metric in metrics_data:
        collision_number, entropy, unique_expressions = metric
        data.append({
            'collision_number': collision_number,
            'entropy': entropy,
            'unique_expressions_count': unique_expressions
        })
    
    df = pd.DataFrame(data)
    
    # Create plots
    return plot_experiment_metrics(df)



# === Imports ===
import pandas as pd
import sqlite3
from bokeh.plotting import figure
from bokeh.embed import components
from .config import DB_NAME

# === Data Query Function ===
def query_df_by_config_id(config_id):
    """
    Query Averages table for a config_id and return pandas DataFrame.

    Note: the columns are named collision_num, entropy, and
    len_unique_expressions, which differ from the names used by
    plot_experiment_metrics().
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT collision_number, entropy, unique_expressions
        FROM Averages
        WHERE config_id = ?
        ORDER BY collision_number
    ''', (config_id,))

    rows = cursor.fetchall()
    conn.close()

    df = pd.DataFrame(rows, columns=["collision_num", "entropy", "len_unique_expressions"])
    return df

# === Plot Functions ===
def generate_entropy_plot(df):
    """Simple entropy line plot from a query_df_by_config_id() DataFrame."""
    p1 = figure(
        title="Entropy Over Time",
        x_axis_label="Collision #",
        y_axis_label="Entropy",
        height = 500,
        width=800
    )
    p1.line(df["collision_num"], df["entropy"], line_width=2)
    return p1

def generate_unique_expr_plot(df):
    """Simple unique-expression line plot from a query_df_by_config_id() DataFrame."""
    p2 = figure(
        title="Unique Expressions Over Time",
        x_axis_label="Collision #",
        y_axis_label="# Unique Expressions",
        height=500,
        width=800
    )
    p2.line(df["collision_num"], df["len_unique_expressions"], line_width=2)
    return p2

# === Combine and Return Components ===
def generate_bokeh_components(config_id):
    """
    Pulls dataframe for given config ID, creates both plots, returns script + divs

    Returns:
        tuple: (script, entropy_div, unique_expressions_div)
    """
    df = query_df_by_config_id(config_id)

    entropy_plot = generate_entropy_plot(df)
    unique_expr_plot = generate_unique_expr_plot(df)

    script1, div1 = components(entropy_plot)
    script2, div2 = components(unique_expr_plot)

    full_script = script1 + "\n" + script2

    return full_script, div1, div2

def get_simulation_components(results_path: str):
    """
    Load results from JSON, generate Bokeh layout, return script and div.

    Used by the home page. The JSON file must be in a format that
    plot_simulation_metrics() understands. The two plots are stacked
    vertically.
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    layout = plot_simulation_metrics(results)
    script, div = components(column(layout))
    return script, div

#=== Plot AST Tree ====
def ASTvisualizer(expression):
    """Draw a lambda expression as a tree (its abstract syntax tree).

    The root is at the top and each level of the tree is one row, centered
    horizontally. Nodes are colored by variable (colors come from
    ASTGen.getColors), so the same variable has the same color everywhere.

    Args:
        expression (str): A lambda expression, e.g. "\\x.x".

    Returns:
        Figure: The tree plot, or an ASTErr() figure with an error message if
        the expression can't be parsed. This function does not raise.
    """
    try:
        #use lambda parser to translate expression into a tree
        parser = LambdaParser(expression)
        Atree = parser.parse()

        #if it cannot parse, send err
        if not Atree:
            return ASTErr("Invalid expression")
        
        #get variable colors
        colors = getColors(Atree)

        # Build NetworkX DiGraph
        G = nx.DiGraph()

        def build_graph(node, depth=0):
            """Add `node` and all its children to G; returns the node's ID.

            Node colors: variables and lambdas use their variable's color
            (yellow/green if none); anything else (e.g. applications) is red.
            """
            node_id = len(G) # Guarantees unique ID based on current length

            if isinstance(node, VariableNode):
                node_color = colors.get(node.name, "yellow")
            elif isinstance(node, LambdaNode):
                node_color = colors.get(node.var, "green")
            else:
                node_color = "red"

            # Add current node
            node_name = getattr(node, 'name', None) or getattr(node, 'var', None) or str(node)
            G.add_node(node_id, name=str(node_name), color=node_color, depth=depth)

            # Recurse over children
            children = getattr(node, 'children', [])
            for child in children:
                child_id = build_graph(child, depth=depth + 1)
                G.add_edge(node_id, child_id)

            return node_id

        build_graph(Atree)

        # Calculate coordinates based on depth level, preventing overlap
        levels = defaultdict(list)
        for n, data in G.nodes(data=True):
            levels[data['depth']].append(n) # Add nodes to each level

        pos = {}
        y_spacing = 2.0
        x_spacing = 2.0

        # Each row is centered on x = 0: e.g. 3 nodes go at x = -2, 0, 2
        for depth, nodes_in_level in levels.items():
            total_nodes = len(nodes_in_level)
            for idx, node_id in enumerate(nodes_in_level):
                x = (idx - (total_nodes - 1) / 2.0) * x_spacing
                y = -depth * y_spacing
                pos[node_id] = (x, y)

        # Bokeh figure
        p = create_styled_figure(
            f"AST: {expression}", "", "",
            width=800, height=450
        )

        # Draw Edges
        for u, v in G.edges():
            x_start, y_start = pos[u]
            x_end, y_end = pos[v]
            p.line([x_start, x_end], [y_start, y_end], line_width=2, color='#000000', line_alpha=0.6)

        # Extract node coordinates from DiGraph
        nodeXn = [pos[n][0] for n in G.nodes()]
        nodeYn = [pos[n][1] for n in G.nodes()]
        nodeNames = [G.nodes[n]['name'] for n in G.nodes()]
        nodeColors = [G.nodes[n]['color'] for n in G.nodes()]

        # Plot
        p.scatter(nodeXn, nodeYn, size=25, color=nodeColors, line_color=TEXT_COLOR, line_width=1, alpha=0.8)

        make_annotations = ColumnDataSource(data={'x': nodeXn, 'y': nodeYn, 'text': nodeNames})
        annotations = LabelSet(
            x='x', y='y', text='text', source=make_annotations,
            text_color='white', text_align='center', text_baseline='middle',
            text_font_style='bold', text_font_size='11px'
        )
        p.add_layout(annotations)

        p.background_fill_color = "#FFFFFF"
        p.border_fill_color = "#FFFFFF"

        return p
        
    except Exception as e:
        return ASTErr(f"Error: {str(e)}")

def ASTErr(message):
    """Return a small placeholder figure that just shows an error message."""
    p = figure(width=600, height=200, title="AST Error")
    p.text(x=[0], y=[0], text=[message], text_align='center', text_baseline='middle')
    p.xaxis.visible = False
    p.yaxis.visible = False
    return p


# multiple experiment dendrogram

def create_multi_experiment_dendrogram(config_ids, limit=20):
    """Build one dendrogram (family tree) of expressions from several experiments.

    Takes the most common expressions from each experiment and groups them by
    how similar they are, measured by Levenshtein (edit) distance: the number
    of single-character changes needed to turn one expression into another.
    Similar expressions join low in the tree.

    Each leaf is colored by the experiment it came from; expressions found in
    more than one experiment are dark grey ("Convergent (Shared)"). Hovering
    over a leaf shows the expression and where it came from.

    Args:
        config_ids (list[int]): Experiments to compare.
        limit (int): How many of the most common expressions to take from
            each experiment.

    Returns:
        tuple: (script, div) Bokeh components.

    Note: needs at least 2 distinct expressions in total, or scipy's
    linkage() will raise an error.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    import pandas as pd
    import numpy as np
    import Levenshtein
    from scipy.spatial.distance import squareform
    from sklearn.metrics import pairwise_distances
    from bokeh.plotting import figure
    from bokeh.embed import components
    from bokeh.models import ColumnDataSource, HoverTool
    from .db_utils import get_comparison_data

    PALETTE = ["#4F46E5", "#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#06B6D4"]
    
    molecule_metadata = []
    seen_expressions = {} 

    #gather data using user defined molecules.
    # seen_expressions maps each expression -> which experiments it appeared in
    for i, cid in enumerate(config_ids):
        df = get_comparison_data(cid, most=limit) 
        if not df.empty:
            color = PALETTE[i % len(PALETTE)]
            for expr in df['expression'].unique():
                if expr not in seen_expressions:
                    seen_expressions[expr] = {'origins': [], 'colors': []}
                seen_expressions[expr]['origins'].append(f"Exp {cid}")
                seen_expressions[expr]['colors'].append(color)

    unique_list = list(seen_expressions.keys())
    
    # matrix math: edit distance between every pair of expressions, then
    # cluster them with average linkage (scipy only computes the layout here;
    # no_plot=True means Bokeh does the drawing)
    dist_matrix = pairwise_distances(
        np.array(unique_list).reshape(-1, 1), 
        metric=lambda x, y: Levenshtein.distance(str(x[0]), str(y[0]))
    )
    Z = linkage(squareform(dist_matrix), method='average')
    ddata = dendrogram(Z, no_plot=True)
    
    p = figure(title=f"Dendrogram: (Top {limit} Survivors)", 
               height=600, sizing_mode="stretch_width",
               toolbar_location="above", tools="pan,wheel_zoom,reset,save")
    
    # draw branches
    source = ColumnDataSource(data={'xs': ddata['icoord'], 'ys': ddata['dcoord']})
    p.multi_line('xs', 'ys', source=source, color="black", line_width=1.5)


    # If a molecule is shared color grey
    # If it's unique use specific experiment color
    leaves = ddata['leaves']
    ordered_labels = [unique_list[leaf] for leaf in leaves]
    
    final_colors = []
    final_origins = []
    
    for label in ordered_labels:
        data = seen_expressions[label]
        if len(data['origins']) > 1:
            final_colors.append("#1e293b")
            final_origins.append(f"SHARED: {', '.join(data['origins'])}")
        else:
            final_colors.append(data['colors'][0]) 
            final_origins.append(data['origins'][0])

    # scipy places leaves at x = 5, 15, 25, ... so the dots line up with the
    # ends of the branches
    leaf_source = ColumnDataSource(data={
        'x': [(i * 10) + 5 for i in range(len(ordered_labels))],
        'y': [0] * len(ordered_labels),
        'detail': ordered_labels,
        'origin': final_origins,
        'color': final_colors
    })
    
    leaf_renderer = p.circle('x', 'y', source=leaf_source, size=12, 
                             color='color', line_color="white", line_width=1)

    #legend logic 
    legend_items = []
    
    # add key item from each experiment. The dots are drawn at NaN so they
    # never appear on the plot; they only exist to give the legend a color swatch.
    for i, cid in enumerate(config_ids):
        color = PALETTE[i % len(PALETTE)]
        dummy_glyph = p.circle(x=[float('nan')], y=[float('nan')], size=10, color=color, line_color="white")
        legend_items.append((f"Exp {cid} (Unique)", [dummy_glyph]))
        
    # key for shared molecules
    dummy_shared = p.circle(x=[float('nan')], y=[float('nan')], size=10, color="#1e293b", line_color="white")
    legend_items.append(("Convergent (Shared)", [dummy_shared]))
    
    # assemble legend
    from bokeh.models import Legend 
    
    legend = Legend(
        items=legend_items, 
        title="Origin Key",
        title_text_font_style="bold",
        background_fill_color="#f8fafc",
        border_line_color="#cbd5e1",
        padding=10
    )
    p.add_layout(legend, 'right')

    # tooltip with expression and origin info
    hover_html = """
        <div style="
            padding: 10px; 
            background-color: #1e293b; 
            color: white; 
            border-radius: 8px; 
            max-width: 300px; 
            word-wrap: break-word; 
            font-family: 'Courier New', monospace;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        ">
            <div style="font-size: 10px; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px; font-weight: bold;">
                Source: @origin
            </div>
            <div style="font-size: 13px; line-height: 1.4; color: #38bdf8;">
                @detail
            </div>
        </div>
    """

    p.add_tools(HoverTool(
        renderers=[leaf_renderer], 
        tooltips=hover_html,
        attachment="vertical",
        point_policy="follow_mouse"
    ))

    p.xaxis.major_label_text_color = None 
    p.yaxis.axis_label = "Edit Distance"
    
    return components(p)
