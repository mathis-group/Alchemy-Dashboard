import os
import json
import io
import re
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from bokeh.resources import CDN
from bokeh.embed import components, json_item
from werkzeug.utils import secure_filename

from .simulation import run_experiment
from .plotting import get_simulation_components, plot_experiment_metrics, ASTvisualizer, ASTErr, create_bokeh_plots_from_metrics
from .models import (
    init_database, save_configuration, save_experiment_state, save_averages,
    get_experiment_configs, save_continuation_metadata, get_continuation_metadata,
    update_experiment_name, delete_experiment
)
from .db_utils import (
    get_experiment_details, process_collision_data, get_experiment_metrics,
    get_expressions_for_collision, get_entropy_and_histogram
)
from .ASTGen import LambdaParser, VariableNode, LambdaNode, getColors

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded_configs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

@app.route('/simulation')
def simulation():
    return render_template('simulation.html', active_page='simulation')

@app.route('/database')
def database_view():
    """View database contents and experiment details."""
    configs = get_experiment_configs()
    mode = request.args.get('tree_mode', 'ward')
    
    if not configs:
        default_experiment = {
            'config_id': 0, 'name': 'No experiments available', 'generator_type': '',
            'total_collisions': 0, 'polling_frequency': 0, 'timestamp': '',
            'generator_params': {}, 'initial_expressions': []
        }
        return render_template('database_view.html', experiment=default_experiment,
                               initial_expressions=[], bokeh_script='', bokeh_div='',
                               active_page='database', mode=mode)
    
    latest_config = configs[0]
    config, metrics, initial_expressions = get_experiment_details(latest_config['config_id'])
    
    if not config:
        return "Experiment not found", 404
    
    try:
        stored_params = json.loads(config[5]) if config[5] else {}
    except json.JSONDecodeError:
        stored_params = {}

    if config[6] is not None:
        stored_params.setdefault('freevar_probability', config[6])

    continuation_meta = get_continuation_metadata(config[0])

    experiment = {
        'config_id': config[0],
        'name': config[8] or f'Experiment {config[0]}',
        'generator_type': config[2],
        'total_collisions': config[3],
        'polling_frequency': config[4],
        'timestamp': config[7],
        'generator_params': stored_params,
        'continuation': continuation_meta
    }
    
    lineage_script, lineage_div = "", ""
    try:
        from .lineage_plots import create_dendrogram
        l_script, l_div = create_dendrogram(latest_config['config_id'], mode=mode)
        lineage_script = re.sub(r'<script[^>]*>', '', l_script).replace("</script>", "")
        lineage_div = l_div
    except Exception as e:
        print(f"Dendrogram Error: {e}")

    formatted_expressions = [expr[0] for expr in initial_expressions]
    
    if metrics:
        df = process_collision_data(metrics)
        plots = plot_experiment_metrics(df)
        entropy_script, entropy_div = components(plots['entropy_plot'])
        unique_script, unique_div = components(plots['unique_expressions_plot'])
        combined_script = entropy_script + unique_script
    else:
        entropy_div, unique_div, combined_script, unique_script = '', '', '', ''
    
    return render_template('database_view.html',
                           experiment=experiment,
                           initial_expressions=formatted_expressions,
                           bokeh_script=combined_script,
                           bokeh_div=entropy_div,
                           unique_expressions_script=unique_script,
                           unique_expressions_div=unique_div,
                           lineage_script=lineage_script,
                           lineage_div=lineage_div,
                           mode=mode,
                           active_page='database')

@app.route('/visualize_ast', methods=['POST'])
def visualize_ast():
    try:
        expression = request.form.get('expression')
        if not expression:
            return jsonify({'status': 'error', 'message': 'No expression provided.'}), 400

        plot_object = ASTvisualizer(expression)
        script, div = components(plot_object)
        clean_script = re.sub(r'<script[^>]*>', '', script)
        clean_script = clean_script.replace("</script>", "")

        return jsonify({'status': 'success', 'div': div, 'script': clean_script})
    except Exception as e:
        error_message = f"Error generating visualization: {str(e)}"
        print(f"[ERROR] {error_message}")
        return jsonify({ 'status': 'error', 'message': error_message }), 500

from .comparison_plots import calculate_distance as run_ordination

@app.route('/get_distance_analysis/<int:config_id>')
def get_distance_analysis_route(config_id):
    print(f"DEBUG: Ordination request received for ID {config_id}")
    try:
        script, div = run_ordination(config_id)
        if script is None:
            return jsonify({"status": "error", "message": "No data available"})
        
        clean_script = re.sub(r'<script[^>]*>', '', script).replace("</script>", "")
        return jsonify({"status": "success", "script": clean_script, "div": div})
    except Exception as e:
        print(f"Ordination Route Error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_lineage_analysis/<int:config_id>')
def get_lineage_analysis_route(config_id):
    mode = request.args.get('tree_mode', 'ward')
    try:
        from .lineage_plots import create_dendrogram
        script, div = create_dendrogram(config_id, mode=mode)
        clean_script = re.sub(r'<script[^>]*>', '', script).replace("</script>", "")
        return jsonify({"status": "success", "script": clean_script, "div": div, "mode": mode})
    except Exception as e:
        print(f"Dendrogram Error: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/continuation_config/<int:config_id>')
def continuation_config(config_id):
    try:
        config, metrics, _ = get_experiment_details(config_id)
        if not config:
            return jsonify({'status': 'error', 'message': 'Experiment not found'}), 404

        try:
            generator_params = json.loads(config[5]) if config[5] else {}
        except json.JSONDecodeError:
            generator_params = {}

        if config[6] is not None:
            generator_params.setdefault('freevar_probability', config[6])

        final_state = get_expressions_for_collision(config_id, -1)
        final_population = sum(count for _, count in final_state) if final_state else 0
        last_collision_number = metrics[-1][0] if metrics else None

        payload = {
            'status': 'success',
            'config_id': config_id,
            'name': config[8] or f'Experiment {config_id}',
            'generator_type': config[2],
            'total_collisions': config[3],
            'polling_frequency': config[4],
            'random_seed': config[1],
            'generator_params': generator_params,
            'default_fraction': 0.5,
            'final_population': final_population,
            'last_collision_number': last_collision_number
        }
        return jsonify(payload)
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500

@app.route('/download_initial_state/<int:config_id>')
def download_initial_state(config_id):
    try:
        config, _, initial_expressions = get_experiment_details(config_id)
        if not config:
            return "Experiment not found", 404

        continuation = get_continuation_metadata(config_id)
        payload = {
            'config_id': config_id,
            'name': config[8] or f'Experiment {config_id}',
            'generator_type': config[2],
            'random_seed': config[1],
            'total_collisions': config[3],
            'polling_frequency': config[4],
            'timestamp': config[7],
            'continuation': continuation,
            'initial_expression_counts': [{'expression': expr, 'count': count} for expr, count in initial_expressions]
        }

        if not payload['initial_expression_counts']:
            return "No initial expressions recorded for this experiment", 404

        buffer = io.BytesIO()
        buffer.write(json.dumps(payload, indent=2).encode('utf-8'))
        buffer.seek(0)
        filename = f"experiment_{config_id}_initial_state.json"
        return send_file(buffer, mimetype='application/json', as_attachment=True, download_name=filename)
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500

@app.route('/download_final_state/<int:config_id>')
def download_final_state(config_id):
    try:
        config, metrics, _ = get_experiment_details(config_id)
        if not config:
            return "Experiment not found", 404

        final_state = get_expressions_for_collision(config_id, -1)
        if not final_state:
            return "No final state data available for this experiment", 404

        last_collision = metrics[-1][0] if metrics else None
        payload = {
            'config_id': config_id,
            'name': config[8] or f'Experiment {config_id}',
            'generator_type': config[2],
            'timestamp': config[7],
            'last_collision_number': last_collision,
            'final_state_counts': [{'expression': expr, 'count': count} for expr, count in final_state]
        }

        buffer = io.BytesIO()
        buffer.write(json.dumps(payload, indent=2).encode('utf-8'))
        buffer.seek(0)
        filename = f"experiment_{config_id}_final_state.json"
        return send_file(buffer, mimetype='application/json', as_attachment=True, download_name=filename)
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500

@app.route('/download_initial_expressions/<int:config_id>')
def download_initial_expressions(config_id):
    try:
        config, _, initial_expressions = get_experiment_details(config_id)
        if not config:
            return "Experiment not found", 404

        if not initial_expressions:
            return "No initial expressions recorded for this experiment", 404

        lines = []
        for entry in initial_expressions:
            if isinstance(entry, (list, tuple)):
                lines.append(str(entry[0]))
            else:
                lines.append(str(entry))

        buffer = io.BytesIO()
        buffer.write("\n".join(lines).encode('utf-8'))
        buffer.seek(0)
        filename = f"experiment_{config_id}_initial_expressions.txt"
        return send_file(buffer, mimetype='text/plain', as_attachment=True, download_name=filename)
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500

@app.route('/delete_experiment', methods=['POST'])
def delete_experiment_route():
    try:
        payload = request.get_json(silent=True) or {}
        config_id = payload.get('config_id')
        if not config_id:
            return jsonify({'status': 'error', 'message': 'Missing config_id'}), 400

        success = delete_experiment(int(config_id))
        if success:
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'Failed to delete experiment'}), 500
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500

@app.route('/run_simulation_form', methods=['POST'])
def run_simulation_form():
    try:
        generator_type = request.form.get('generator_type', 'BTree')
        total_collisions = int(request.form.get('total_collisions', 1000))
        polling_frequency = int(request.form.get('polling_frequency', 10))
        random_seed = int(request.form.get('random_seed', 42))
        experiment_name = request.form.get('experiment_name', '')

        continuation_parent_id = request.form.get('continuation_parent_id')
        continuation_fraction_raw = request.form.get('continuation_fraction')
        continuation_info = None

        if continuation_parent_id:
            parent_id = int(continuation_parent_id)
            try:
                fraction_value = float(continuation_fraction_raw or 0)
            except ValueError:
                fraction_value = 0.0
            if fraction_value > 1: fraction_value = fraction_value / 100.0
            fraction_value = max(0.0, min(1.0, fraction_value))
            continuation_info = {'parent_config_id': parent_id, 'fraction': fraction_value}

        config = {
            'generator_type': generator_type, 'total_collisions': total_collisions,
            'polling_frequency': polling_frequency, 'random_seed': random_seed, 'experiment_name': experiment_name
        }
        if continuation_info: config['continuation'] = continuation_info

        generator_params_payload = {}
        if generator_type == 'BTree':
            size = int(request.form.get('btree_size', 5))
            freevar_probability = float(request.form.get('freevar_probability', 0.5))
            max_free_vars = int(request.form.get('max_free_vars', 3))
            standardization = request.form.get('standardization', 'prefix')
            num_expressions = int(request.form.get('num_expressions', 10))
            config.update({'size': size, 'freevar_probability': freevar_probability, 'max_free_vars': max_free_vars, 'standardization': standardization, 'num_expressions': num_expressions})
            generator_params_payload = {'size': size, 'freevar_probability': freevar_probability, 'max_free_vars': max_free_vars, 'standardization': standardization, 'num_expressions': num_expressions}
        elif generator_type == 'Fontana':
            abs_low = float(request.form.get('abs_low', 0.1))
            abs_high = float(request.form.get('abs_high', 0.5))
            app_low = float(request.form.get('app_low', 0.2))
            app_high = float(request.form.get('app_high', 0.6))
            min_depth = int(request.form.get('min_depth', 1))
            max_depth = int(request.form.get('max_depth', 5))
            max_free_vars = int(request.form.get('fontana_max_fv', 2))
            expression_count = int(request.form.get('fontana_expression_count', 10))
            free_variable_probability = float(request.form.get('free_variable_probability', 0.5))
            config.update({'abs_low': abs_low, 'abs_high': abs_high, 'app_low': app_low, 'app_high': app_high, 'min_depth': min_depth, 'max_depth': max_depth, 'max_free_vars': max_free_vars, 'initial_expression_count': expression_count, 'free_variable_probability': free_variable_probability})
            generator_params_payload = {'abs_low': abs_low, 'abs_high': abs_high, 'app_low': app_low, 'app_high': app_high, 'min_depth': min_depth, 'max_depth': max_depth, 'max_free_vars': max_free_vars, 'initial_expression_count': expression_count, 'free_variable_probability': free_variable_probability}
        elif generator_type == 'from_file':
            if 'expressions_file' in request.files:
                file = request.files['expressions_file']
                if file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    config['file_path'] = filepath
                    generator_params_payload = {'input_mode': 'file', 'filename': filename}
            elif request.form.get('direct_input'):
                expressions = [expr for expr in request.form.get('direct_input').split('\n') if expr.strip()]
                config['expressions'] = expressions
                generator_params_payload = {'input_mode': 'direct_input', 'expression_count': len(expressions)}

        result = run_experiment(config)

        config_id = save_configuration(
            random_seed=random_seed, generator_type=generator_type, total_collisions=total_collisions,
            polling_frequency=polling_frequency, probability_range=json.dumps(generator_params_payload) if generator_params_payload else None,
            freevar_generation_probability=config.get('freevar_probability') if generator_type == 'BTree' else None, name=experiment_name
        )

        initial_expressions = result.get('initial_expressions', [])
        for expr, count in Counter(initial_expressions).items():
            save_experiment_state(config_id, 0, expr, count)

        metrics = result.get('metrics', [])
        for metric in metrics:
            save_averages(config_id, metric['collision_number'], metric['entropy'], metric['unique_expressions'])
            if 'expressions' in metric:
                for expr, count in Counter(metric['expressions']).items():
                    save_experiment_state(config_id, metric['collision_number'], expr, count)

        continuation_summary = result.get('continuation_summary', {})
        if continuation_summary and continuation_summary.get('parent_config_id'):
            save_continuation_metadata(config_id, continuation_summary['parent_config_id'], continuation_summary.get('fraction_used', 0.0), continuation_summary.get('continued_expression_count', 0), continuation_summary.get('new_expression_count', 0))

        if metrics:
            df = process_collision_data(metrics)
            plots = plot_experiment_metrics(df)
            entropy_script, entropy_div = components(plots['entropy_plot'])
            unique_script, unique_div = components(plots['unique_expressions_plot'])
            bokeh_script, bokeh_div = entropy_script + unique_script, entropy_div + unique_div
        else:
            bokeh_script, bokeh_div = "", ""
        
        return jsonify({
            'status': 'success', 'config_id': config_id, 'experiment_name': experiment_name or f"Experiment {config_id}",
            'initial_expressions': initial_expressions, 'initial_expression_count': len(initial_expressions),
            'continuation_summary': continuation_summary, 'download_initial_state_url': url_for('download_initial_state', config_id=config_id) if continuation_summary and continuation_summary.get('parent_config_id') else None,
            'bokeh_div': bokeh_div, 'script': bokeh_script, 'message': "Simulation completed successfully!"
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Error running simulation: {str(e)}"}), 500

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
        return jsonify({"status": "success", "message": f"Found {len(configs)} configurations", "data": configs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/view_experiment/<int:config_id>')
def view_experiment(config_id):
    config, metrics, initial_expressions = get_experiment_details(config_id)
    if not config: return "Experiment not found", 404
    
    df = process_collision_data(metrics)
    plots = plot_experiment_metrics(df)
    entropy_script, entropy_div = components(plots['entropy_plot'])
    unique_script, unique_div = components(plots['unique_expressions_plot'])
    
    experiment_details = {
        'config_id': config[0], 'random_seed': config[1], 'generator_type': config[2],
        'total_collisions': config[3], 'polling_frequency': config[4],
        'generator_params': {'freevar_generation_probability': config[6] if config[6] is not None else 0.5, 'probability_range': json.loads(config[5]) if config[5] else {}},
        'timestamp': config[7], 'name': config[8] or f"Experiment {config_id}"
    }
    
    return render_template('database_view.html', experiment=experiment_details, initial_expressions=[expr[0] for expr in initial_expressions], bokeh_script=entropy_script + unique_script, bokeh_div=entropy_div, unique_expressions_script=unique_script, unique_expressions_div=unique_div, active_page='database')

@app.route('/upload_json', methods=['POST'])
def upload_json():
    if 'json_file' not in request.files: return jsonify({'status': 'error', 'message': 'No file uploaded.'})
    file = request.files['json_file']
    if not file.filename.endswith('.json'): return jsonify({'status': 'error', 'message': 'Invalid file type.'})
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "collisions_data" not in data: return jsonify({'status': 'error', 'message': "Missing 'collisions_data' in file."})
            return jsonify({'status': 'success', 'filename': file.filename, 'metrics': list(data["collisions_data"][next(iter(data["collisions_data"]))].keys())})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generate_visuals/<filename>', methods=['GET'])
def generate_visuals(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return jsonify({'status': 'error', 'message': f'File not found: {filepath}'})
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        from .plotting import create_bokeh_from_data
        script, div = create_bokeh_from_data(data)
        return jsonify({'status': 'success', 'script': script, 'div': div})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/update_experiment_name', methods=['POST'])
def update_name():
    try:
        config_id = int(request.form.get('config_id'))
        new_name = request.form.get('name')
        if not new_name or not new_name.strip(): return jsonify({"status": "error", "message": "Name cannot be empty"})
        success = update_experiment_name(config_id, new_name)
        if success: return jsonify({"status": "success", "message": "Experiment name updated"})
        return jsonify({"status": "error", "message": "Failed to update"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/perturb_and_run', methods=['POST'])
def perturb_and_run():
    return "Perturbation feature not yet implemented", 501

@app.route('/list_experiments')
def list_experiments():
    experiments = get_experiment_configs()
    result = {'experiments': []}
    for exp in experiments:
        if isinstance(exp, dict):
            entry = {'config_id': exp.get('config_id'), 'name': exp.get('name'), 'generator_type': exp.get('generator_type'), 'total_collisions': exp.get('total_collisions'), 'polling_frequency': exp.get('polling_frequency'), 'timestamp': exp.get('timestamp')}
        else:
            entry = {'config_id': exp[0], 'random_seed': exp[1], 'generator_type': exp[2], 'total_collisions': exp[3], 'polling_frequency': exp[4], 'timestamp': exp[5], 'name': exp[8] if len(exp) > 8 else None}
        result['experiments'].append(entry)
    return jsonify(result)

@app.route('/compare_experiments', methods=['GET', 'POST'])
def compare_experiments():
    if request.method == 'POST':
        selected_ids = request.form.getlist('experiment_ids')
        metric = request.form.get('metric', 'entropy')
        if not selected_ids: return redirect(url_for('compare_experiments'))
        config_ids = [int(id) for id in selected_ids]
        metric_data = get_experiment_metrics(config_ids, metric)
        from .plotting import plot_comparison_metrics
        script, div = components(plot_comparison_metrics(metric_data, metric))
        return render_template('compare_experiments.html', experiments=get_experiment_configs(), selected_ids=selected_ids, selected_metric=metric, bokeh_script=script, bokeh_div=div)
    return render_template('compare_experiments.html', experiments=get_experiment_configs())

@app.route('/get_experiment_plot/<int:config_id>')
def get_experiment_plot(config_id):
    try:
        config, metrics, _ = get_experiment_details(config_id)
        if not config or not metrics: return jsonify({"status": "error", "message": "No metrics found"}), 404
        plots = plot_experiment_metrics(process_collision_data(metrics))
        entropy_script, entropy_div = components(plots['entropy_plot'])
        unique_script, unique_div = components(plots['unique_expressions_plot'])
        return jsonify({
            "status": "success", "entropy_script": re.sub(r'<script[^>]*>', '', entropy_script).replace("</script>", ""),
            "entropy_div": entropy_div, "unique_expressions_script": re.sub(r'<script[^>]*>', '', unique_script).replace("</script>", ""),
            "unique_expressions_div": unique_div
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_experiment_metadata/<int:config_id>')
def get_experiment_metadata(config_id):
    try:
        config, metrics, initial_expressions = get_experiment_details(config_id)
        if not config: return jsonify({"status": "error", "message": "Experiment not found"}), 404
        try: generator_params = json.loads(config[5]) if config[5] else {}
        except: generator_params = {}
        if config[6] is not None: generator_params.setdefault('freevar_probability', config[6])
        return jsonify({
            "status": "success",
            "details": {'config_id': config[0], 'random_seed': config[1], 'generator_type': config[2], 'total_collisions': config[3], 'polling_frequency': config[4], 'generator_params': generator_params, 'timestamp': config[7], 'name': config[8] or f"Experiment {config[0]}", 'continuation': get_continuation_metadata(config_id)},
            "expressions": [expr[0] for expr in initial_expressions]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def create_histogram_html(histogram):
    if not histogram: return "<p>No data available for this collision.</p>"
    top_expressions = histogram[:20]
    max_count = max(h["count"] for h in top_expressions) if top_expressions else 1
    html = '<div style="margin: 20px 0;"><h4>Top 20 Expressions by Frequency</h4><div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;">'
    for item in top_expressions:
        expr, count = item["expression"], item["count"]
        html += f'<div style="margin-bottom: 8px;"><div style="display: flex; align-items: center; margin-bottom: 4px;"><div style="width: 200px; font-family: monospace; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{expr}">{expr[:50] + "..." if len(expr) > 50 else expr}</div><div style="margin-left: 10px; font-weight: bold; min-width: 30px;">{count}</div></div><div style="background: #e2e8f0; height: 20px; border-radius: 3px; overflow: hidden;"><div style="background: #4F46E5; height: 100%; width: {(count / max_count) * 100}%; transition: width 0.3s ease;"></div></div></div>'
    return html + '</div></div>'

@app.route('/get_entropy_detail/<int:collision_number>')
def get_entropy_detail(collision_number):
    try:
        config_id = int(request.args.get("config_id"))
        result = get_entropy_and_histogram(config_id, collision_number)
        return f'<div class="card-header"><h3 class="card-title">Details for Collision {collision_number}</h3></div><div class="card-body"><p><strong>Entropy:</strong> {result["entropy"]:.4f}</p>{create_histogram_html(result["histogram"])}</div>'
    except Exception as e:
        return f"Error: {str(e)}", 500

#sequence alignment route for comparing expressions using Levenshtein distance and percentage identity

@app.route('/api/sequence_alignment/<int:config_id>', methods=['POST'])
def sequence_alignment(config_id):
    try:
        import Levenshtein
        from .db_utils import get_comparison_data
        
        data = request.get_json()
        target_expr = data.get('expression')
        
        if not target_expr:
            return jsonify({'status': 'error', 'message': 'No target expression provided.'}), 400

        # obtain most abundant molecules
        df = get_comparison_data(config_id, most=100)
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data found for this experiment.'}), 404
            
        unique_molecules = df['expression'].unique().tolist()
        
        results = []
        for expr in unique_molecules:
            #molcule is not compared to self
            if expr == target_expr:
                continue 
            #use levenshtein distance to calculate conversion from target expr to expr
            dist = Levenshtein.distance(str(target_expr), str(expr))
            max_len = max(len(str(target_expr)), len(str(expr)))
            
            # percentage identity forumla
            identity = round((1 - (dist / max_len)) * 100, 2) if max_len > 0 else 100.0
            
            results.append({
                'expression': expr,
                'distance': dist,
                'identity': identity
            })
            
        # sort results
        results = sorted(results, key=lambda x: x['identity'], reverse=True)
        
        # return top 10 matches
        return jsonify({'status': 'success', 'target': target_expr, 'results': results[:10]})
        
    except Exception as e:
        print(f"Alignment Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

#function to simulate extinction event 
@app.route('/trigger_extinction', methods=['POST'])
def trigger_extinction():
    try:
        
        data = request.get_json()
        parent_config_id = data.get('config_id')
        
        if not parent_config_id:
            return jsonify({'status': 'error', 'message': 'Missing config_id'}), 400

        # fetch the final state
        final_state = get_expressions_for_collision(parent_config_id, -1)
        if not final_state:
            return jsonify({'status': 'error', 'message': 'No final data found.'}), 404

        # sort population count from highest to lowest
        sorted_population = sorted(final_state, key=lambda x: x[1], reverse=True)
        
        # remove the molecule with the highest count
        apex_predator = sorted_population[0][0]
        apex_count = sorted_population[0][1]
        survivors = sorted_population[1:] 

        # reconstruct survivor list for rust
        survivor_expressions = []
        for expr, count in survivors:
            survivor_expressions.extend([expr] * count) 

        if not survivor_expressions:
            return jsonify({'status': 'error', 'message': 'No surivors left!'}), 400

        # send to rust 
        config = {
            'generator_type': 'from_file', 
            'expressions': survivor_expressions,
            'total_collisions': 1000, 
            'polling_frequency': 10,
            'random_seed': 42,
            'experiment_name': f"Post-Extinction (Parent: {parent_config_id})",
            'continuation': {
                'parent_config_id': parent_config_id,
                'fraction': 1.0 
            }
        }

        # run experiment with survivors
        result = run_experiment(config)
        
        # save new configuration 
        new_config_id = save_configuration(
            random_seed=config['random_seed'], 
            generator_type='from_file', 
            total_collisions=config['total_collisions'],
            polling_frequency=config['polling_frequency'], 
            probability_range=json.dumps({'input_mode': 'extinction_event', 'culled_species': apex_predator}),
            name=config['experiment_name']
        )

        # save initial state, metrics and record timeline
        for expr, count in Counter(survivor_expressions).items():
            save_experiment_state(new_config_id, 0, expr, count)

        metrics = result.get('metrics', [])
        for metric in metrics:
            save_averages(new_config_id, metric['collision_number'], metric['entropy'], metric['unique_expressions'])
            if 'expressions' in metric:
                for expr, count in Counter(metric['expressions']).items():
                    save_experiment_state(new_config_id, metric['collision_number'], expr, count)

        save_continuation_metadata(new_config_id, parent_config_id, 1.0, len(survivors), 0)

        return jsonify({
            'status': 'success', 
            'new_config_id': new_config_id,
            'message': f"The most abundant expression '{apex_predator[:15]}...' ({apex_count} copies) was eradicated. A new timeline has started"
        })

    except Exception as e:
        print(f"Extinction Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    experiments = get_experiment_configs()[:5]
    return render_template('dashboard.html', experiments=[{'config_id': exp[0], 'random_seed': exp[1], 'generator_type': exp[2], 'total_collisions': exp[3], 'polling_frequency': exp[4], 'timestamp': exp[5]} for exp in experiments], generator_counts={exp[2]: sum(1 for e in experiments if e[2] == exp[2]) for exp in experiments}, total_experiments=len(experiments))

if __name__ == '__main__':
    app.run(debug=True)