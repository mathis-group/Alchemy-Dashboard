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
        config, metrics, initial_expressions = get_experiment_details(config_id)
        if not config:
            return "Experiment not found", 404

        continuation = get_continuation_metadata(config_id)
        
        # Build the payload
        payload = {
            'config_id': config_id,
            'name': config[8] or f'Experiment {config_id}',
            'generator_type': config[2],
            'random_seed': config[1],
            'total_collisions': config[3],
            'polling_frequency': config[4],
            'timestamp': config[7],
            'continuation': continuation,
            'initial_expression_counts': [{'expression': expr, 'count': count} for expr, count in initial_expressions],
            
            # Graph data so it can be re-uploaded to the UI
            'collisions_data': {
                "experiment_history": {
                    str(m[0]): {
                        "entropy": m[1],
                        "unique_expressions": m[2]
                    } for m in metrics
                }
            }
        }

        if not payload['initial_expression_counts']:
            return "No initial expressions recorded", 404

        buffer = io.BytesIO()
        buffer.write(json.dumps(payload, indent=2).encode('utf-8'))
        buffer.seek(0)
        return send_file(buffer, mimetype='application/json', as_attachment=True, download_name=f"experiment_{config_id}_initial_full.json")
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
            return "No final state data available", 404

        last_collision = metrics[-1][0] if metrics else None
        
        # Build the payload
        payload = {
            'config_id': config_id,
            'name': config[8] or f'Experiment {config_id}',
            'generator_type': config[2],
            'timestamp': config[7],
            'last_collision_number': last_collision,
            'final_state_counts': [{'expression': expr, 'count': count} for expr, count in final_state],
            
            'collisions_data': {
                "experiment_history": {
                    str(m[0]): {
                        "entropy": m[1],
                        "unique_expressions": m[2]
                    } for m in metrics
                }
            }
        }

        buffer = io.BytesIO()
        buffer.write(json.dumps(payload, indent=2).encode('utf-8'))
        buffer.seek(0)
        return send_file(buffer, mimetype='application/json', as_attachment=True, download_name=f"experiment_{config_id}_final_full.json")
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

@app.route('/upload_and_import', methods=['POST'])
def upload_and_import():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'status': 'error', 'message': 'No file received'}), 400
        
        data = json.load(file)
        if 'collisions_data' not in data:
            return jsonify({'status': 'error', 'message': 'Missing timeline data.'}), 400

        # 1. Save Main Configuration
        # Probability range needs to be a string for the DB
        prob_range = json.dumps(data.get('generator_params', {}))
        
        new_config_id = save_configuration(
            data.get('random_seed', 0),
            data.get('generator_type', 'Imported'),
            data.get('total_collisions', 1000),
            data.get('polling_frequency', 10),
            prob_range,
            f"{data.get('name', 'Experiment')} (Imported)"
        )

        # 2. Re-hydrate Graphs (The Timeline)
        # FIX: Using positional arguments to avoid 'new_id' naming errors
        history = data['collisions_data'].get('experiment_history', {})
        for col_num, metrics in history.items():
            save_averages(
                new_config_id, 
                int(col_num), 
                metrics['entropy'], 
                metrics['unique_expressions']
            )

        # 3. Handle Molecular Population (AST & Histogram Fix)
        final_pop = data.get('final_state_counts', [])
        initial_pop = data.get('initial_expression_counts', [])
        last_col = data.get('last_collision_number', -1)

        # Map molecules to Collision 0 so the Sidebar/AST dropdown works
        startup_data = initial_pop if initial_pop else final_pop
        for item in startup_data:
            save_experiment_state(new_config_id, 0, item['expression'], item['count'])

        # Map molecules to Final State (-1) and the specific last collision (e.g., 990)
        for item in final_pop:
            save_experiment_state(new_config_id, -1, item['expression'], item['count'])
            if last_col != -1:
                save_experiment_state(new_config_id, last_col, item['expression'], item['count'])

        return jsonify({'status': 'success', 'new_config_id': new_config_id})

    except Exception as e:
        print(f"Import Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


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

@app.route('/delete_all_experiments', methods=['POST'])
def delete_all_experiments():
    try:
        # Get all current experiments
        experiments = get_experiment_configs()
        
        deleted_count = 0
        for exp in experiments:
            config_id = exp['config_id'] if isinstance(exp, dict) else exp[0]
            if delete_experiment(config_id):
                deleted_count += 1
            
        # Reset the ID counter
        import sqlite3
        from .config import DB_NAME
        conn = sqlite3.connect(DB_NAME) 
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM sqlite_sequence") 
            conn.commit()
        except sqlite3.OperationalError as e:
            if "no such table: sqlite_sequence" not in str(e):
                raise e
        finally:
            conn.close()
        
        return jsonify({
            'status': 'success', 
            'message': f'{deleted_count} experiments deleted and counters reset.'
        })

    except Exception as e:
        print(f"Error during mass delete: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500



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
            entry = {
                'config_id': exp.get('config_id'), 
                'name': exp.get('name'), 
                'random_seed': exp.get('random_seed'), 
                'generator_type': exp.get('generator_type'), 
                'total_collisions': exp.get('total_collisions'), 
                'polling_frequency': exp.get('polling_frequency'), 
                'timestamp': exp.get('timestamp')
            }
        else:
            entry = {
                'config_id': exp[0], 
                'random_seed': exp[1], 
                'generator_type': exp[2], 
                'total_collisions': exp[3], 
                'polling_frequency': exp[4], 
                'timestamp': exp[5], 
                'name': exp[8] if len(exp) > 8 else None
            }
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
        target_expr = data.get('target_expression')

        if not parent_config_id or not target_expr:
            return jsonify({'status': 'error', 'message': 'Missing config_id or target_expression'}), 400

        parent_data = get_experiment_details(parent_config_id)
        if not parent_data or not parent_data[0]:
            return jsonify({'status': 'error', 'message': 'Parent data not found.'}), 404
        
        parent_config = parent_data[0]
        
        final_state = get_expressions_for_collision(parent_config_id, -1)
        if not final_state:
            return jsonify({'status': 'error', 'message': 'Final population state not found.'}), 404

        survivors = [item for item in final_state if item[0].strip() != target_expr.strip()]

        if not survivors:
            return jsonify({'status': 'error', 'message': 'Extinction killed everything! No survivors.'}), 400

        survivor_pool = []
        for expr, count in survivors:
            survivor_pool.extend([expr] * count)

        new_seed = parent_config[1] or 42
        
        config = {
            'generator_type': 'from_file', 
            'expressions': survivor_pool,
            'total_collisions': parent_config[3], 
            'polling_frequency': parent_config[4], 
            'random_seed': new_seed,
            'experiment_name': f"Post-Extinction: {target_expr[:15]}... (From: {parent_config_id})",
        }

        result = run_experiment(config)
        
        new_id = save_configuration(
            random_seed=config['random_seed'], 
            generator_type='from_file', 
            total_collisions=config['total_collisions'],
            polling_frequency=config['polling_frequency'], 
            probability_range=json.dumps({
                'event': 'targeted_extinction', 
                'removed': target_expr[:50] + "..." if len(target_expr) > 50 else target_expr
            }),
            name=config['experiment_name']
        )

        for expr, count in Counter(survivor_pool).items():
            save_experiment_state(new_id, 0, expr, count)

        metrics = result.get('metrics', [])
        for metric in metrics:
            save_averages(new_id, metric['collision_number'], metric['entropy'], metric['unique_expressions'])
            if 'expressions' in metric:
                for expr, count in Counter(metric['expressions']).items():
                    save_experiment_state(new_id, metric['collision_number'], expr, count)

        save_continuation_metadata(new_id, parent_config_id, 1.0, len(survivor_pool), 0)

        return jsonify({
            'status': 'success', 
            'new_config_id': new_id,
            'message': f"Successfully purged target. Soup continues with {len(survivor_pool)} surviving particles."
        })

    except Exception as e:
        print(f"CRITICAL ERROR IN EXTINCTION: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


#route for comparison dendrograms 
@app.route('/multi_compare')
def multi_compare_page():
    # Fetch all experiments so the user can select them from a list
    experiments = get_experiment_configs()
    return render_template('multi_compare.html', experiments=experiments, active_page='multi_compare')

# multiple dendrogram api route

@app.route('/api/generate_multi_dendrogram', methods=['POST'])
def generate_multi_dendrogram():
    try:
        data = request.get_json()
        ids = data.get('experiment_ids', [])
        
      
        user_limit = data.get('limit', 20) 
        
        if len(ids) < 2:
            return jsonify({'status': 'error', 'message': 'Please select at least two experiments to compare.'}), 400

        from .plotting import create_multi_experiment_dendrogram 
        
        
        script, div = create_multi_experiment_dendrogram(ids, limit=user_limit)
        
        import re
        clean_script = re.sub(r'<script[^>]*>', '', script).replace("</script>", "")
        
        return jsonify({
            'status': 'success', 
            'script': clean_script, 
            'div': div
        })
    except Exception as e:
        print(f"Error generating dendrogram: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/run_simulation_form', methods=['POST'])
def run_simulation_form():
    try:
        total_collisions = int(request.form.get('total_collisions', 1000))
        polling_frequency = int(request.form.get('polling_frequency', 10))
        random_seed = int(request.form.get('random_seed', 42))
        base_name = request.form.get('experiment_name', 'Auto-Evo')
        if not base_name.strip():
            base_name = 'Auto-Evo'
            
        num_generations = int(request.form.get('num_generations', 1))
        
        recursive_parent_id = request.form.get('recursive_parent_id')
        generator_type = request.form.get('generator_type', 'Fontana')
        
        current_pool = None
        last_config_id = int(recursive_parent_id) if recursive_parent_id else None

        # If starting from an existing parent, fetch its survivors once
        if last_config_id:
            from .db_utils import get_experiment_details, get_expressions_for_collision
            parent_config, _, _ = get_experiment_details(last_config_id)
            random_seed = parent_config[1]
    
            final_state = get_expressions_for_collision(last_config_id, -1)
            if not final_state:
                return jsonify({'status': 'error', 'message': f"Parent ID {last_config_id} has no collision data to inherit."}), 400

            current_pool = []
            for expr, count in sorted(final_state, key=lambda x: x[1], reverse=True)[:15]:
                current_pool.extend([expr] * count)

            generator_type = 'from_file'

        # --- MULTI-GENERATION LOOP ---
        for gen in range(num_generations):
            gen_idx = gen + 1
            exp_name = f"{base_name} [Gen {gen_idx}]" if num_generations > 1 else base_name
            
            config = {
                'generator_type': generator_type,
                'total_collisions': total_collisions,
                'polling_frequency': polling_frequency,
                'random_seed': random_seed,
                'experiment_name': exp_name,
                'expressions': current_pool
            }

            # Apply parameters for Gen 1 if it is a FRESH start
            if gen == 0 and not recursive_parent_id:
                if generator_type == 'from_file':
                    if 'expressions_file' in request.files and request.files['expressions_file'].filename:
                        file = request.files['expressions_file']
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(filepath)
                        config['file_path'] = filepath
                    elif request.form.get('direct_input'):
                        config['expressions'] = [e.strip() for e in request.form.get('direct_input').split('\n') if e.strip()]
                    
                    if not config.get('expressions') and not config.get('file_path'):
                        return jsonify({'status': 'error', 'message': "Selected 'From File' but provided no file or text."}), 400
                        
                elif generator_type == 'BTree':
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
                        'min_depth': int(request.form.get('min_depth', 1)),
                        'max_depth': int(request.form.get('max_depth', 5)),
                        'max_free_vars': int(request.form.get('fontana_max_fv', 2)),
                        'initial_expression_count': int(request.form.get('fontana_expression_count', 10)),
                        'free_variable_probability': float(request.form.get('free_variable_probability', 0.5))
                    })

            # Fire Engine
            result = run_experiment(config)

            # Validate engine output
            if not result or 'metrics' not in result:
                raise ValueError(f"Simulation engine failed to return metrics for {exp_name}.")

            # Save DB
            new_id = save_configuration(
                random_seed=random_seed, 
                generator_type=generator_type,
                total_collisions=total_collisions,
                polling_frequency=polling_frequency,
                probability_range=json.dumps(config.get('generator_params', {})),
                name=exp_name
            )

            # Save population/metrics
            initial_expressions = result.get('initial_expressions', [])
            for expr, count in Counter(initial_expressions).items():
                save_experiment_state(new_id, 0, expr, count)

            metrics = result.get('metrics', [])
            for metric in metrics:
                save_averages(new_id, metric['collision_number'], metric['entropy'], metric['unique_expressions'])
                if 'expressions' in metric:
                    for expr, count in Counter(metric['expressions']).items():
                        save_experiment_state(new_id, metric['collision_number'], expr, count)

            # Link Lineage
            if last_config_id:
                save_continuation_metadata(new_id, last_config_id, 1.0, len(current_pool or initial_expressions), 0)

            # SETUP FOR NEXT GENERATION IN THE LOOP
            last_config_id = new_id
            generator_type = 'from_file'
            
            # Extract survivors directly from memory for the next loop (Fast & Safe)
            if metrics and 'expressions' in metrics[-1]:
                survivors = Counter(metrics[-1]['expressions']).items()
                current_pool = []
                for expr, count in sorted(survivors, key=lambda x: x[1], reverse=True)[:15]:
                    current_pool.extend([expr] * count)
            else:
                raise ValueError(f"{exp_name} produced no surviving expressions to pass on.")

        return jsonify({
            'status': 'success', 
            'config_id': last_config_id, 
            'experiment_name': exp_name,
            'message': f"Successfully ran {num_generations} generation(s)."
        })

    except Exception as e:
        print(f"Recursion Loop Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/trigger_invasive_species', methods=['POST'])
def trigger_invasive_species():
    try:
        data = request.get_json()
        parent_config_id = data.get('config_id')
        invasive_expr = data.get('expression', '\\x.x')
        invasive_count = int(data.get('count', 50))

        if not parent_config_id:
            return jsonify({'status': 'error', 'message': 'Missing config_id'}), 400

        final_state = get_expressions_for_collision(parent_config_id, -1)
        if not final_state:
            return jsonify({'status': 'error', 'message': 'No final data found.'}), 404

        survivor_expressions = []
        for expr, count in final_state:
            survivor_expressions.extend([expr] * count)

        # Inject the invasive molecules
        survivor_expressions.extend([invasive_expr] * invasive_count)

        config = {
            'generator_type': 'from_file', 
            'expressions': survivor_expressions,
            'total_collisions': 1000, 
            'polling_frequency': 10,
            'random_seed': 42,
            'experiment_name': f"Invasion: {invasive_expr[:20]} (Parent: {parent_config_id})"
        }

        result = run_experiment(config)
        new_id = save_configuration(
            random_seed=42, generator_type='from_file', 
            total_collisions=1000, polling_frequency=10, 
            name=config['experiment_name']
        )

        for expr, count in Counter(survivor_expressions).items():
            save_experiment_state(new_id, 0, expr, count)

        metrics = result.get('metrics', [])
        for metric in metrics:
            save_averages(new_id, metric['collision_number'], metric['entropy'], metric['unique_expressions'])
            if 'expressions' in metric:
                for expr, count in Counter(metric['expressions']).items():
                    save_experiment_state(new_id, metric['collision_number'], expr, count)

        save_continuation_metadata(new_id, parent_config_id, 1.0, len(survivor_expressions) - invasive_count, invasive_count)

        return jsonify({'status': 'success', 'new_config_id': new_id})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500



@app.route('/dashboard')
def dashboard():
    experiments = get_experiment_configs()[:5]
    return render_template('dashboard.html', experiments=[{'config_id': exp[0], 'random_seed': exp[1], 'generator_type': exp[2], 'total_collisions': exp[3], 'polling_frequency': exp[4], 'timestamp': exp[5]} for exp in experiments], generator_counts={exp[2]: sum(1 for e in experiments if e[2] == exp[2]) for exp in experiments}, total_experiments=len(experiments))

if __name__ == '__main__':
    app.run(debug=True)