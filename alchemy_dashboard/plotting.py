# alchemy_dashboard/plotting.py

from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, Legend, Circle
from bokeh.embed import components
import pandas as pd
import json

# Define colors using CSS variables
PRIMARY_COLOR = "#4F46E5"  # var(--primary)
SECONDARY_COLOR = "#0EA5E9"  # var(--secondary)
ACCENT_COLOR = "#F59E0B"  # var(--accent)
GRID_COLOR = "#E2E8F0"  # var(--border)
TEXT_COLOR = "#1E293B"  # var(--text-primary)

# Define plot colors
PLOT_COLORS = [
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    ACCENT_COLOR,
    "#10B981",  # var(--success)
    "#EF4444",  # var(--error)
    "#F59E0B",  # var(--warning)
]

def create_styled_figure(title, x_label, y_label, width=800, height=300):
    """Create a styled Bokeh figure with consistent formatting."""
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

# from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS

# def plot_experiment_metrics(df):
#     """
#     Create plots for a single experiment's metrics.
    
#     Args:
#         df (DataFrame): DataFrame with collision_number, entropy, unique_expressions_count
        
#     Returns:
#         list: List of Bokeh figure objects
#     """
#     source = ColumnDataSource(data=dict(
#         x=df['collision_number'],
#         entropy=df['entropy'],
#         unique=df['unique_expressions_count'],
#         total=df.get('total_expressions', [0] * len(df))
#     ))

#     # === Entropy plot ===
#     entropy_plot = create_styled_figure("Entropy Over Time", "Collision Number", "Entropy")
#     entropy_plot.line('x', 'entropy', source=source, line_width=2.5, color=PRIMARY_COLOR, line_alpha=0.8)

#     # TapTool requires a selectable glyph — use scatter with selection color
#     entropy_plot.scatter('x', 'entropy', source=source, size=8, color=PRIMARY_COLOR,
#                          alpha=0.6, selection_color="red", nonselection_alpha=0.4)

#     entropy_plot.add_tools(HoverTool(tooltips=[
#         ("Collision", "@x"),
#         ("Entropy", "@entropy{0.0000}")
#     ], mode='vline'))

#     # === Unique expressions plot ===
#     unique_plot = create_styled_figure("Unique Expressions Over Time", "Collision Number", "Count")
#     unique_plot.line('x', 'unique', source=source, line_width=2.5, color=SECONDARY_COLOR, line_alpha=0.8)
#     unique_plot.scatter('x', 'unique', source=source, size=6, color=SECONDARY_COLOR, alpha=0.6)
#     unique_plot.add_tools(HoverTool(tooltips=[
#         ("Collision", "@x"),
#         ("Unique Expressions", "@unique")
#     ], mode='vline'))

#     plots = [entropy_plot, unique_plot]

#     # === Optional: Total expressions plot ===
#     if 'total_expressions' in df.columns:
#         total_plot = create_styled_figure("Total Expressions Over Time", "Collision Number", "Count")
#         total_plot.line('x', 'total', source=source, line_width=2.5, color=ACCENT_COLOR, line_alpha=0.8)
#         total_plot.scatter('x', 'total', source=source, size=6, color=ACCENT_COLOR, alpha=0.6)
#         total_plot.add_tools(HoverTool(tooltips=[
#             ("Collision", "@x"),
#             ("Total Expressions", "@total")
#         ], mode='vline'))
#         plots.append(total_plot)

#     # === Enable TapTool and attach JS callback ===
#     entropy_plot.add_tools(TapTool())

#     tap_callback = CustomJS(args=dict(source=source), code="""
#         const selected_index = source.selected.indices[0];
#         if (selected_index != null) {
#             const collision = source.data['x'][selected_index];
#             fetch(`/get_entropy_detail/${collision}?config_id=${window.currentConfigID}`)
#                 .then(response => response.text())
#                 .then(html => {
#                     const target = document.getElementById("entropy-details");
#                     if (target) {
#                         target.innerHTML = html;
#                         target.scrollIntoView({ behavior: "smooth" });
#                     }
#                 });
#         }
#     """)
#     source.selected.js_on_change('indices', tap_callback)

#     return plots







from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS

def plot_experiment_metrics(df):
    """
    Create plots for a single experiment's metrics.
    
    Args:
        df (pandas.DataFrame): DataFrame containing metrics data
        
    Returns:
        dict: Dictionary of Bokeh figure objects, keyed by plot type
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

    # Add JS callback for tap
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
    
    Args:
        metric_data (dict): Dictionary mapping config_id to DataFrame with metric data
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
        results (dict): Dictionary with simulation results
        
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
    
    Args:
        data (dict): Parsed JSON data from uploaded file
        
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
        metrics_data (list): List of tuples containing metrics data
        title_prefix (str): Optional prefix for plot titles
        
    Returns:
        list: List of Bokeh figure objects
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
from config import DB_NAME

# === Data Query Function ===
def query_df_by_config_id(config_id):
    """
    Query Averages table for a config_id and return pandas DataFrame.
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
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    layout = plot_simulation_metrics(results)
    script, div = components(column(layout))
    return script, div
