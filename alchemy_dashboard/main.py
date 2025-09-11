import os
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from bokeh.resources import CDN
from simulation import run_experiment
from plotting import get_simulation_components, plot_experiment_metrics
from models import init_database, save_configuration, save_experiment_state, save_averages, get_experiment_configs
from db_utils import (
    get_experiment_details, 
    process_collision_data,
    get_experiment_metrics,
    get_expressions_for_collision
)
from plotting import create_bokeh_plots_from_metrics
import re
from werkzeug.utils import secure_filename
from bokeh.embed import components

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded_configs'
app.config['EXPERIMENT_FOLDER'] = 'experiments'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXPERIMENT_FOLDER'], exist_ok=True)

# Initialize database on startup
init_database()

latest_json_path = None

@app.route('/')
def index():
    global latest_json_path
    if latest_json_path is not None:
        try:
            script, div = get_simulation_components(latest_json_path)
        except Exception as e:
            print("Error rendering Bokeh components:", e)
            script, div = "", ""
    else:
        script, div = "", ""
    return render_template('home.html', active_page='home', bokeh_script=script, bokeh_div=div)

@app.route('/database')
def database_view():
    """View database contents and experiment details."""
    configs = get_experiment_configs()
    
    # Create a default experiment if no configs exist
    if not configs:
        default_experiment = {
            'config_id': 0,
            'name': 'No experiments available',
            'generator_type': '',
            'total_collisions': 0,
            'polling_frequency': 0,
            'timestamp': '',
            'generator_params': {},
            'initial_expressions': []
        }
        return render_template('database_view.html',
                             experiment=default_experiment,
                             initial_expressions=[],
                             bokeh_script='',
                             bokeh_div='',
                             active_page='database')
    
    # Get the most recent experiment's details
    latest_config = configs[0]  # Most recent config is first due to ORDER BY timestamp DESC
    
    # Debug print to see the structure
    print("Latest config structure:", latest_config)
    
    # Get experiment details
    config, metrics, initial_expressions = get_experiment_details(latest_config['config_id'])
    
    if not config:
        return "Experiment not found", 404
    
    # Format experiment details
    experiment = {
        'config_id': config[0],
        'name': config[8] or f'Experiment {config[0]}',  # name is the 9th element (index 8)
        'generator_type': config[2],
        'total_collisions': config[3],
        'polling_frequency': config[4],
        'timestamp': config[7],
        'generator_params': {
            'freevar_generation_probability': config[6] if config[6] is not None else 0.5,
            'probability_range': json.loads(config[5]) if config[5] else [0.0, 1.0]
        }
    }
    
    # Format initial expressions
    formatted_expressions = [expr[0] for expr in initial_expressions]
    
    # Generate Bokeh components for the plots
    if metrics:
        # Process metrics data into a DataFrame
        df = process_collision_data(metrics)
        plots = plot_experiment_metrics(df)
        entropy_script, entropy_div = components(plots['entropy_plot'])
        unique_script, unique_div = components(plots['unique_expressions_plot'])
    else:
        entropy_script, entropy_div = '', ''
        unique_script, unique_div = '', ''
    
    return render_template('database_view.html',
                         experiment=experiment,
                         initial_expressions=formatted_expressions,
                         bokeh_script=entropy_script,
                         bokeh_div=entropy_div,
                         unique_expressions_script=unique_script,
                         unique_expressions_div=unique_div,
                         active_page='database')

@app.route('/simulation')
def simulation():
    return render_template('simulation.html', active_page='simulation')

@app.route('/run_simulation_form', methods=['POST'])
def run_simulation_form():
    try:
        # Get form data
        generator_type = request.form.get('generator_type', 'BTree')
        total_collisions = int(request.form.get('total_collisions', 1000))
        polling_frequency = int(request.form.get('polling_frequency', 10))
        random_seed = int(request.form.get('random_seed', 42))
        experiment_name = request.form.get('experiment_name', '')
        
        # Build config dictionary
        config = {
            'generator_type': generator_type,
            'total_collisions': total_collisions,
            'polling_frequency': polling_frequency,
            'random_seed': random_seed,
            'experiment_name': experiment_name
        }
        
        # Add generator-specific parameters
        if generator_type == 'BTree':
            config.update({
                'size': int(request.form.get('btree_size', 5)),
                'freevar_probability': float(request.form.get('freevar_probability', 0.5)),
                'max_free_vars': int(request.form.get('max_free_vars', 3)),
                'standardization': request.form.get('standardization', 'prefix'),
                'num_expressions': int(request.form.get('num_expressions', 10))
            })
        elif generator_type == 'Fontana':
            config.update({
                'abs_low': float(request.form.get('abs_low', 0.1)),
                'abs_high': float(request.form.get('abs_high', 0.5)),
                'app_low': float(request.form.get('app_low', 0.2)),
                'app_high': float(request.form.get('app_high', 0.6)),
                'max_depth': int(request.form.get('max_depth', 5)),
                'max_free_vars': int(request.form.get('fontana_max_fv', 2))
            })
        elif generator_type == 'from_file':
            if 'expressions_file' in request.files:
                file = request.files['expressions_file']
                if file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    config['file_path'] = filepath
            elif request.form.get('direct_input'):
                config['expressions'] = request.form.get('direct_input').split('\n')

        # Run the experiment
        result = run_experiment(config)
        
        # Save to database
        config_id = save_configuration(
            random_seed=random_seed,
            generator_type=generator_type,
            total_collisions=total_collisions,
            polling_frequency=polling_frequency,
            probability_range=None,  # We'll handle this later if needed
            freevar_generation_probability=config.get('freevar_probability') if generator_type == 'BTree' else None,
            name=experiment_name
        )
        
        # Save initial expressions
        initial_expressions = result.get('initial_expressions', [])
        for expr in initial_expressions:
            save_experiment_state(config_id, 0, expr, 1)  # Initial state has count 1
        
        # Save metrics and full state data
        metrics = result.get('metrics', [])
        for metric in metrics:
            save_averages(
                config_id,
                metric['collision_number'],
                metric['entropy'],
                metric['unique_expressions']
            )
            
            # Save full state expressions if available
            if 'expressions' in metric:
                collision_number = metric['collision_number']
                expressions = metric['expressions']
                
                # Count frequency of each expression
                from collections import Counter
                expression_counts = Counter(expressions)
                
                # Save each expression with its count
                for expr, count in expression_counts.items():
                    save_experiment_state(config_id, collision_number, expr, count)
        
        # Generate charts
        if metrics:
            df = process_collision_data(metrics)
            
            # Generate charts
            from bokeh.embed import components
            from plotting import plot_experiment_metrics
            
            plots = plot_experiment_metrics(df)
            
            # Get components for individual plots
            entropy_script, entropy_div = components(plots['entropy_plot'])
            unique_script, unique_div = components(plots['unique_expressions_plot'])

            bokeh_script = entropy_script + unique_script # Combine for embedding if needed, or pass separately
            bokeh_div = entropy_div + unique_div # Combine for embedding
        else:
            bokeh_script, bokeh_div = "", ""
        
        return jsonify({
            'status': 'success',
            'config_id': config_id,
            'experiment_name': experiment_name or f"Experiment {config_id}",
            'initial_expressions': initial_expressions,
            'bokeh_div': bokeh_div,  # This will contain both divs concatenated
            'script': bokeh_script, # This will contain both scripts concatenated
            'message': f"Simulation completed successfully! You can view the results in the database view."
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Error running simulation: {str(e)}"
        }), 500

@app.route('/debug_db')
def debug_db():
    try:
        import sqlite3
        from config import DB_NAME
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Configurations")
        configs = cursor.fetchall()
        conn.close()
        return jsonify({
            "status": "success",
            "message": f"Found {len(configs)} configurations",
            "data": configs
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# Update view_experiment route to include experiment name
@app.route('/view_experiment/<int:config_id>')
def view_experiment(config_id):
    """View a single experiment's details and visualizations."""
    config, metrics, initial_expressions = get_experiment_details(config_id)
    
    if not config:
        return "Experiment not found", 404
    
    # Process metrics data for visualization
    df = process_collision_data(metrics)
    
    # Create plots
    from plotting import plot_experiment_metrics
    from bokeh.embed import components
    
    plots = plot_experiment_metrics(df)
    entropy_script, entropy_div = components(plots['entropy_plot'])
    unique_script, unique_div = components(plots['unique_expressions_plot'])
    
    # Format experiment details
    experiment_details = {
        'config_id': config[0],
        'random_seed': config[1],
        'generator_type': config[2],
        'total_collisions': config[3],
        'polling_frequency': config[4],
        'generator_params': {
            'freevar_generation_probability': config[6] if config[6] is not None else 0.5,
            'probability_range': json.loads(config[5]) if config[5] else {}
        },
        'timestamp': config[7],
        'name': config[8] or f"Experiment {config_id}"
    }
    
    # Format initial expressions
    formatted_expressions = [expr[0] for expr in initial_expressions]
    
    return render_template(
        'database_view.html',
        experiment=experiment_details,
        initial_expressions=formatted_expressions,
        bokeh_script=entropy_script,
        bokeh_div=entropy_div,
        unique_expressions_script=unique_script,
        unique_expressions_div=unique_div,
        active_page='database'
    )


@app.route('/upload_json', methods=['POST'])
def upload_json():
    print("✅ upload_json route registered")
    if 'json_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded.'})
    file = request.files['json_file']
    filename = file.filename
    if not filename.endswith('.json'):
        return jsonify({'status': 'error', 'message': 'Invalid file type.'})
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"[UPLOAD] File saved to {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "collisions_data" not in data:
                return jsonify({'status': 'error', 'message': "Missing 'collisions_data' in file."})
            metrics = list(data["collisions_data"][next(iter(data["collisions_data"]))].keys())
            return jsonify({'status': 'success', 'filename': filename, 'metrics': list(metrics)})
    except json.JSONDecodeError as je:
        return jsonify({'status': 'error', 'message': f'JSON decode error: {str(je)}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generate_visuals/<filename>', methods=['GET'])
def generate_visuals(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"[VISUALIZE] Looking for file: {filepath}")
    if not os.path.exists(filepath):
        return jsonify({'status': 'error', 'message': f'File not found: {filepath}'})
    if os.path.getsize(filepath) == 0:
        return jsonify({'status': 'error', 'message': f'File is empty: {filepath}'})
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "collisions_data" not in data:
            return jsonify({'status': 'error', 'message': "Missing 'collisions_data' in file."})
        from plotting import create_bokeh_from_data
        script, div = create_bokeh_from_data(data)
        return jsonify({'status': 'success', 'script': script, 'div': div})
    except json.JSONDecodeError as je:
        return jsonify({'status': 'error', 'message': f'JSON decode error: {str(je)}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Unexpected error: {str(e)}'})
    

@app.route('/continue_experiment', methods=['POST'])
def continue_experiment():
    config_id = request.form.get('config_id')
    if not config_id:
        return "No experiment selected", 400
    
    # Get the last experiment state
    last_expressions = get_expressions_for_collision(config_id, -1)  # -1 for last collision
    
    if not last_expressions:
        return "No expressions found for experiment", 400
    
    # Create a temp file with expressions
    temp_file = os.path.join(app.config['UPLOAD_FOLDER'], f"continue_exp_{config_id}.txt")
    with open(temp_file, 'w') as f:
        for expr, count in last_expressions:
            for _ in range(count):
                f.write(f"{expr}\n")
    
    # Get original config
    config_details, _, _ = get_experiment_details(config_id)
    
    # Prepare new config
    config = {
        "random_seed": config_details[1],
        "total_collisions": config_details[3],
        "polling_frequency": config_details[4],
        "input_expressions": {
            "generator": "from_file",
            "params": {"filename": temp_file}
        }
    }
    
    # Run new experiment
    result = run_experiment(config)
    
    return redirect(url_for('view_experiment', config_id=result['config_id']))
from models import update_experiment_name

# Add this route to update experiment names
@app.route('/update_experiment_name', methods=['POST'])
def update_name():
    """Update the name of an experiment."""
    try:
        config_id = int(request.form.get('config_id'))
        new_name = request.form.get('name')
        
        if not new_name or not new_name.strip():
            return jsonify({
                "status": "error",
                "message": "Name cannot be empty"
            })
            
        success = update_experiment_name(config_id, new_name)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Experiment name updated successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to update experiment name"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


@app.route('/perturb_and_run', methods=['POST'])
def perturb_and_run():
    return "Perturbation feature not yet implemented", 501

@app.route('/list_experiments')
def list_experiments():
    """List all experiments in the database."""
    experiments = get_experiment_configs()
    result = {'experiments': []}
    
    for exp in experiments:
        # Check if exp is a tuple/list or a dictionary
        if isinstance(exp, (list, tuple)):
            result['experiments'].append({
                'config_id': exp[0],
                'random_seed': exp[1],
                'generator_type': exp[2], 
                'total_collisions': exp[3],
                'polling_frequency': exp[4],
                'timestamp': exp[5]
            })
        elif isinstance(exp, dict):
            # If it's already a dictionary, just add it
            result['experiments'].append(exp)
    
    return jsonify(result)

@app.route('/compare_experiments', methods=['GET', 'POST'])
def compare_experiments():
    """Compare multiple experiments."""
    if request.method == 'POST':
        # Get selected experiments and metric from form
        selected_ids = request.form.getlist('experiment_ids')
        metric = request.form.get('metric', 'entropy')
        
        if not selected_ids:
            return redirect(url_for('compare_experiments'))
        
        # Convert to integers
        config_ids = [int(id) for id in selected_ids]
        
        # Get metric data for selected experiments
        metric_data = get_experiment_metrics(config_ids, metric)
        
        # Create comparison plot
        from plotting import plot_comparison_metrics
        from bokeh.embed import components
        
        comparison_plot = plot_comparison_metrics(metric_data, metric)
        script, div = components(comparison_plot)
        
        # Get all experiments for the form
        all_experiments = get_experiment_configs()
        
        return render_template(
            'compare_experiments.html',
            experiments=all_experiments,
            selected_ids=selected_ids,
            selected_metric=metric,
            bokeh_script=script,
            bokeh_div=div
        )
    else:
        # Display form to select experiments for comparison
        experiments = get_experiment_configs()
        return render_template('compare_experiments.html', experiments=experiments)

from flask import jsonify
from plotting import generate_bokeh_components  # or whatever file you use

@app.route('/get_experiment_plot/<int:config_id>')
def get_experiment_plot(config_id):
    try:
        print(f"[DEBUG] Plot request for config_id={config_id}")
        # Get metrics data for the experiment - we need both entropy and unique_expressions
        config, metrics, initial_expressions = get_experiment_details(config_id)
        if not config or not metrics:
            return jsonify({"status": "error", "message": "No metrics found for this experiment"}), 404
            
        # Process the metrics data
        df = process_collision_data(metrics)
        
        # Generate plots
        from plotting import plot_experiment_metrics
        from bokeh.embed import components
        
        plots = plot_experiment_metrics(df)
        
        entropy_script, entropy_div = components(plots['entropy_plot'])
        unique_script, unique_div = components(plots['unique_expressions_plot'])
        
        # Clean up script tags
        clean_entropy_script = re.sub(r'<script[^>]*>', '', entropy_script)
        clean_entropy_script = clean_entropy_script.replace("</script>", "")

        clean_unique_script = re.sub(r'<script[^>]*>', '', unique_script)
        clean_unique_script = clean_unique_script.replace("</script>", "")
        
        return jsonify({
            "status": "success",
            "entropy_script": clean_entropy_script,
            "entropy_div": entropy_div,
            "unique_expressions_script": clean_unique_script,
            "unique_expressions_div": unique_div
        })
    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


from db_utils import get_experiment_details_and_expressions  # or equivalent

@app.route('/get_experiment_metadata/<int:config_id>')
def get_experiment_metadata(config_id):
    try:
        config, metrics, initial_expressions = get_experiment_details(config_id)
        if not config:
            return jsonify({"status": "error", "message": "Experiment not found"}), 404
            
        # Format experiment details
        details = {
            'config_id': config[0],
            'random_seed': config[1],
            'generator_type': config[2],
            'total_collisions': config[3],
            'polling_frequency': config[4],
            'generator_params': {
                'freevar_generation_probability': config[6] if config[6] is not None else 0.5,
                'probability_range': json.loads(config[5]) if config[5] else {}
            },
            'timestamp': config[7],
            'name': config[8] or f"Experiment {config_id}"
        }
        
        # Format expressions
        expressions = [expr[0] for expr in initial_expressions]
        
        return jsonify({
            "status": "success",
            "details": details,
            "expressions": expressions
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


from db_utils import get_entropy_and_histogram
from bokeh.plotting import figure
from bokeh.embed import components

def create_histogram_html(histogram):
    """Create a simple HTML histogram without Bokeh dependencies."""
    if not histogram:
        return "<p>No data available for this collision.</p>"
    
    # Take top 20 expressions
    top_expressions = histogram[:20]
    
    # Find max count for scaling
    max_count = max(h["count"] for h in top_expressions) if top_expressions else 1
    
    html = """
    <div style="margin: 20px 0;">
        <h4>Top 20 Expressions by Frequency</h4>
        <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;">
    """
    
    for i, item in enumerate(top_expressions):
        expression = item["expression"]
        count = item["count"]
        percentage = (count / max_count) * 100
        
        # Truncate long expressions for display
        display_expr = expression[:50] + "..." if len(expression) > 50 else expression
        
        html += f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <div style="width: 200px; font-family: monospace; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{expression}">
                    {display_expr}
                </div>
                <div style="margin-left: 10px; font-weight: bold; min-width: 30px;">{count}</div>
            </div>
            <div style="background: #e2e8f0; height: 20px; border-radius: 3px; overflow: hidden;">
                <div style="background: #4F46E5; height: 100%; width: {percentage}%; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html

@app.route('/get_entropy_detail/<int:collision_number>')
def get_entropy_detail(collision_number):
    try:
        config_id_param = request.args.get("config_id")
        if not config_id_param or config_id_param == "undefined":
            return "Error: No valid config_id provided", 400
        
        config_id = int(config_id_param)
        print(f"[DEBUG] Fetching histogram for config_id={config_id}, collision={collision_number}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'alchemy_experiments.db'))
        print(f"[DEBUG] Database path: {db_path}")
        print(f"[DEBUG] Database exists: {os.path.exists(db_path)}")
        
        result = get_entropy_and_histogram(config_id, collision_number)
        entropy = result["entropy"]
        histogram = result["histogram"]
        
        print(f"[DEBUG] Found {len(histogram)} expressions in histogram")
        print(f"[DEBUG] Entropy value: {entropy}")
        print(f"[DEBUG] Histogram data: {histogram[:3] if histogram else 'Empty'}")

        histogram_html = create_histogram_html(histogram)

        return f"""
        <div class="card-header">
            <h3 class="card-title">Details for Collision {collision_number}</h3>
        </div>
        <div class="card-body">
            <p><strong>Entropy:</strong> {entropy:.4f}</p>
            {histogram_html}
        </div>
        """
    except ValueError as e:
        return f"Error: Invalid config_id or collision_number - {str(e)}", 400
    except Exception as e:
        print(f"[ERROR] Exception in get_entropy_detail: {str(e)}")
        return f"Error: {str(e)}", 500


@app.route('/dashboard')
def dashboard():
    """Main dashboard with experiment list and overview statistics."""
    # Get latest 5 experiments
    experiments = get_experiment_configs()[:5]
    
    # Convert to more readable format
    experiment_list = [
        {
            'config_id': exp[0],
            'random_seed': exp[1],
            'generator_type': exp[2],
            'total_collisions': exp[3],
            'polling_frequency': exp[4],
            'timestamp': exp[5]
        } for exp in experiments
    ]
    
    # Count by generator type
    generator_counts = {}
    for exp in experiments:
        generator_type = exp[2]
        if generator_type in generator_counts:
            generator_counts[generator_type] += 1
        else:
            generator_counts[generator_type] = 1
    
    return render_template(
        'dashboard.html',
        experiments=experiment_list,
        generator_counts=generator_counts,
        total_experiments=len(experiments)
    )

if __name__ == '__main__':
    app.run(debug=True)