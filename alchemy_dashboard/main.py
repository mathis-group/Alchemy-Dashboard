
#main.py -- This is the main entry point and router of everything
# It creates a flask app, an uploaded_configs/folder for uploaded JSOn and calls init_database() to create tables if missing 


#import libraries 
import os                                          # filesystem ops (paths, makedirs for upload folder)
import json                                        # parse/serialize JSON (configs, export/import payloads)
import io                                          # in-memory byte buffers for file downloads
import re                                          # regex, used to strip <script> tags from Bokeh output
from collections import Counter                    # tally expressions into (expression, count) pairs
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
                                                    # Flask, template rendering, request data, redirects,
                                                    # URL building, JSON responses, file downloads
from bokeh.resources import CDN                    # Bokeh CDN resource loader (currently unused/legacy)
from bokeh.embed import components, json_item      # embed Bokeh plots as (script, div) or JSON item
from werkzeug.utils import secure_filename         # sanitize uploaded filenames
import sqlite3                                     # direct SQLite access for ad-hoc queries in main.py
from .config import DB_NAME                        # shared path to the SQLite database file

from .simulation import run_experiment             # runs the Rust alchemy engine for N collisions

from .plotting import (
    get_simulation_components,                     # load results JSON -> Bokeh (script, div) for home page
    plot_experiment_metrics,                       # build entropy / unique-expressions line charts
    ASTvisualizer,                                 # render a lambda expression as an AST tree plot
    ASTErr,                                        # fallback "error" plot when AST parsing fails
    create_bokeh_plots_from_metrics                # (currently unused/legacy)
)

from .models import (
    init_database,                                 # create SQLite tables on startup if missing
    save_configuration,                             # insert a new row into Configurations
    save_experiment_state,                          # insert a population snapshot into Experiment
    save_averages,                                  # insert entropy/unique_expressions into Averages
    get_experiment_configs,                         # fetch all saved experiment configs
    save_continuation_metadata,                     # link a child experiment to its parent
    get_continuation_metadata,                      # fetch parent/child linkage for an experiment
    update_experiment_name,                         # rename an experiment
    delete_experiment                               # cascading delete of a config + related rows
)

from .db_utils import (
    get_experiment_details,                         # fetch config + metrics + initial expressions
    process_collision_data,                         # convert raw metrics rows into a pandas DataFrame
    get_experiment_metrics,                         # pull one metric across multiple experiments
    get_expressions_for_collision,                  # get population at a given collision (-1 = final state)
    get_entropy_and_histogram                       # entropy + expression histogram for one collision
)

from .ASTGen import LambdaParser, VariableNode, LambdaNode, getColors
                                                    # lambda expression parser, AST node types,
                                                    # and per-variable color assignment for visualization





app = Flask(__name__)                              # create the Flask application instance

app.config['UPLOAD_FOLDER'] = 'uploaded_configs'   # folder where uploaded JSON configs get saved
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                                                    # create that folder if it doesn't already exist

# Initialize database on startup
init_database()                                    # create SQLite tables if they don't exist yet

latest_json_path = None                            # tracks the most recently loaded results file (if any),
                                                    # used to auto-plot on the home page

@app.route('/')                                    # route for the home page
def index():
    global latest_json_path                        # read the module-level "last loaded file" pointer
    if latest_json_path is not None:                # if a results file has been loaded this session...
        try:
            script, div = get_simulation_components(latest_json_path)
                                                    # build Bokeh (script, div) from that results file
        except Exception as e:
            print("Error rendering Bokeh components:", e)
                                                    # log failure instead of crashing the page
            script, div = "", ""                   # fall back to empty plot content
    else:
        script, div = "", ""                       # no file loaded yet -> render page with no plot

    return render_template('home.html', active_page='home', bokeh_script=script, bokeh_div=div)
                                                    # render the home template, passing plot HTML/JS in

@app.route('/simulation')                          # route for the "run a simulation" page
def simulation():
    return render_template('simulation.html', active_page='simulation')
                                                    # just render the form, no data needed yet

@app.route('/database')                            # route for viewing a single experiment's data
def database_view():
    """View database contents and experiment details."""
    configs = get_experiment_configs()              # fetch list of all saved experiments
    mode = request.args.get('tree_mode', 'ward')    # dendrogram mode from query string, default 'ward'
    requested_id = request.args.get('config_id', type=int)
                                                    # which experiment the user asked for (if any)

    if not configs:                                 # no experiments exist in the DB at all
        default_experiment = {
            'config_id': 0, 'name': 'No experiments available', 'generator_type': '',
            'total_collisions': 0, 'polling_frequency': 0, 'timestamp': '',
            'generator_params': {}, 'initial_expressions': []
        }                                           # placeholder data so the template doesn't break
        return render_template('database_view.html', experiment=default_experiment,
                               initial_expressions=[], bokeh_script='', bokeh_div='',
                               active_page='database', mode=mode)
                                                    # render an empty-state version of the page

    if requested_id is not None:                    # a specific config_id was requested via URL
        config, metrics, initial_expressions = get_experiment_details(requested_id)
        if not config:                              # requested ID doesn't exist in the DB
            # Invalid ID - fall back to latest
            config, metrics, initial_expressions = get_experiment_details(configs[0]['config_id'])
            selected_config_id = configs[0]['config_id']
                                                    # fall back to the most recent experiment instead
        else:
            selected_config_id = requested_id       # requested ID was valid, use it
    else:
        # No ID provided - use latest
        config, metrics, initial_expressions = get_experiment_details(configs[0]['config_id'])
        selected_config_id = configs[0]['config_id']
                                                    # default to the most recently created experiment

    if not config:                                  # still nothing found (shouldn't normally happen)
        return "Experiment not found", 404

    try:
        stored_params = json.loads(config[5]) if config[5] else {}
                                                    # decode the JSON-encoded generator params column
    except json.JSONDecodeError:
        stored_params = {}                          # malformed JSON -> just use empty params

    if config[6] is not None:                       # legacy standalone freevar_probability column
        stored_params.setdefault('freevar_probability', config[6])
                                                    # fold it into the params dict if not already present

    continuation_meta = get_continuation_metadata(config[0])
                                                    # check if this experiment is a child of another one

    experiment = {                                  # build a clean dict for the template
        'config_id': config[0],
        'name': config[8] or f'Experiment {config[0]}',
                                                    # fall back to a generated name if none was saved
        'generator_type': config[2],
        'total_collisions': config[3],
        'polling_frequency': config[4],
        'timestamp': config[7],
        'generator_params': stored_params,
        'continuation': continuation_meta
    }

    lineage_script, lineage_div = "", ""            # default to no lineage plot
    try:
        from .lineage_plots import create_dendrogram
                                                    # local import to avoid circular import / lazy load
        l_script, l_div = create_dendrogram(selected_config_id, mode=mode)
                                                    # build the dendrogram plot for this experiment
        lineage_script = re.sub(r'<script[^>]*>', '', l_script).replace("</script>", "")
                                                    # strip the <script> wrapper tags so it can be
                                                    # injected inline into the template's own <script>
        lineage_div = l_div
    except Exception as e:
        print(f"Dendrogram Error: {e}")             # log failure, leave lineage plot empty

    formatted_expressions = [expr[0] for expr in initial_expressions]
                                                    # pull just the expression strings out of (expr, count) tuples

    if metrics:                                     # only build charts if there's metric data to show
        df = process_collision_data(metrics)        # convert raw metrics rows into a DataFrame
        plots = plot_experiment_metrics(df)          # build entropy + unique-expressions Bokeh figures
        entropy_script, entropy_div = components(plots['entropy_plot'])
        unique_script, unique_div = components(plots['unique_expressions_plot'])
                                                    # split each figure into embeddable (script, div)
        combined_script = entropy_script + unique_script
                                                    # merge both scripts to inject together
    else:
        entropy_div, unique_div, combined_script, unique_script = '', '', '', ''
                                                    # no metrics -> empty plot content

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
                                                    # render the full experiment detail page


@app.route('/visualize_ast', methods=['POST'])      # AJAX endpoint: render one lambda expression as an AST tree
def visualize_ast():
    try:
        expression = request.form.get('expression') # get the expression string from the POST form data
        if not expression:                          # no expression sent
            return jsonify({'status': 'error', 'message': 'No expression provided.'}), 400

        plot_object = ASTvisualizer(expression)      # parse the expression and build the Bokeh AST figure
        script, div = components(plot_object)        # split figure into embeddable (script, div)
        clean_script = re.sub(r'<script[^>]*>', '', script)
        clean_script = clean_script.replace("</script>", "")
                                                    # strip <script> wrapper tags so the frontend can
                                                    # inject the raw JS itself

        return jsonify({'status': 'success', 'div': div, 'script': clean_script})
                                                    # send plot HTML/JS back to the frontend
    except Exception as e:
        error_message = f"Error generating visualization: {str(e)}"
        print(f"[ERROR] {error_message}")           # log the failure server-side
        return jsonify({ 'status': 'error', 'message': error_message }), 500

from .comparison_plots import calculate_distance as run_ordination
                                                    # local import, aliased for clarity in this route's context

@app.route('/get_distance_analysis/<int:config_id>')
                                                    # AJAX endpoint: Jaccard/Bray-Curtis stability plot for one experiment
def get_distance_analysis_route(config_id):
    print(f"DEBUG: Ordination request received for ID {config_id}")
                                                    # debug log of the incoming request
    try:
        script, div = run_ordination(config_id)     # compute similarity metrics + build comparison plot
        if script is None:                          # no data existed for this experiment
            return jsonify({"status": "error", "message": "No data available"})

        clean_script = re.sub(r'<script[^>]*>', '', script).replace("</script>", "")
                                                    # strip <script> wrapper tags before returning
        return jsonify({"status": "success", "script": clean_script, "div": div})
    except Exception as e:
        print(f"Ordination Route Error: {e}")       # log the failure server-side
        return jsonify({"status": "error", "message": str(e)})






@app.route('/get_lineage_analysis/<int:config_id>')
                                                    # AJAX endpoint: dendrogram plot for one experiment
def get_lineage_analysis_route(config_id):
    mode = request.args.get('tree_mode', 'ward')    # clustering mode from query string, default 'ward'
    try:
        from .lineage_plots import create_dendrogram
                                                    # local import, lazy-loaded
        script, div = create_dendrogram(config_id, mode=mode)
                                                    # build the dendrogram figure for this experiment
        clean_script = re.sub(r'<script[^>]*>', '', script).replace("</script>", "")
                                                    # strip <script> wrapper tags before returning
        return jsonify({"status": "success", "script": clean_script, "div": div, "mode": mode})
                                                    # send plot HTML/JS + which mode was used back to frontend
    except Exception as e:
        print(f"Dendrogram Error: {e}")             # log the failure server-side
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/continuation_config/<int:config_id>')
                                                    # AJAX endpoint: metadata needed to set up a "continue this
                                                    # experiment" (recursive) run based on a parent experiment
def continuation_config(config_id):
    try:
        config, metrics, _ = get_experiment_details(config_id)
                                                    # fetch parent experiment's config + metrics
        if not config:                              # parent experiment doesn't exist
            return jsonify({'status': 'error', 'message': 'Experiment not found'}), 404

        try:
            generator_params = json.loads(config[5]) if config[5] else {}
                                                    # decode the JSON-encoded generator params column
        except json.JSONDecodeError:
            generator_params = {}                  # malformed JSON -> empty params

        if config[6] is not None:                   # legacy standalone freevar_probability column
            generator_params.setdefault('freevar_probability', config[6])
                                                    # fold it into the params dict if not already present

        final_state = get_expressions_for_collision(config_id, -1)
                                                    # get the parent's final population (-1 = last collision)
        final_population = sum(count for _, count in final_state) if final_state else 0
                                                    # total molecule count in that final population
        last_collision_number = metrics[-1][0] if metrics else None
                                                    # the last recorded collision number, if any

        payload = {
            'status': 'success',
            'config_id': config_id,
            'name': config[8] or f'Experiment {config_id}',
                                                    # fall back to a generated name if none was saved
            'generator_type': config[2],
            'total_collisions': config[3],
            'polling_frequency': config[4],
            'random_seed': config[1],
            'generator_params': generator_params,
            'default_fraction': 0.5,                # suggested default fraction of population to carry over
            'final_population': final_population,
            'last_collision_number': last_collision_number
        }
        return jsonify(payload)                     # send all continuation setup info back to the frontend
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500






@app.route('/download_initial_state/<int:config_id>')
                                                    # export the starting population of one experiment as JSON
def download_initial_state(config_id):
    try:
        config, _, initial_expressions = get_experiment_details(config_id)
                                                    # fetch config + initial expressions (metrics unused here)
        if not config:
            return "Experiment not found", 404

        continuation = get_continuation_metadata(config_id)
                                                    # check if this experiment was continued from a parent
        payload = {
            'config_id': config_id,
            'name': config[8] or f'Experiment {config_id}',
                                                    # fall back to a generated name if none was saved
            'generator_type': config[2],
            'random_seed': config[1],
            'total_collisions': config[3],
            'polling_frequency': config[4],
            'timestamp': config[7],
            'continuation': continuation,
            'initial_expression_counts': [{'expression': expr, 'count': count} for expr, count in initial_expressions]
                                             ) tuples into a list of dicts
        }

        buffer = io.BytesIO()                       # build the JSON file in memory rather than on disk
        buffer.write(json.dumps(payload, indent=2).encode('utf-8'))
                                                    # serialize payload to pretty-printed JSON bytes
        buffer.seek(0)                       d_file reads from the start
        return send_file(buffer, mimetype='application/json', as_attachment=True,
                        download_name=f"experiment_{config_id}_initial_state.json")
                                             downloadable .json file
    except Exception as exc:
        return jsonify({'status': 'error', 'm


@app.route('/download_final_state/<int:config_id>')
                                             nt results (final + sampled states) as JSON
def download_final_state(config_id):
    try:
        config, metrics, initial_expressions = get_experiment_details(config_id)
                                                    # fetch config, all metrics, and initial expressions
        if not config:
            return "Experiment not found", 404

        final_state = get_expressions_for_collision(config_id, -1)
                                                    # get the population at the last collision (-1 = final)
        if not final_state:
            return "No data found", 404

        last_collision = metrics[-1][0] if metrics else None
                                                    # the last recorded collision number, if any

        # Get expression state at every sampled collision
        sampled_collisions = {}
        for metric in metrics:
            col_num = metric[0]
            expr_data = get_expressions_for_collision(config_id, col_num)
                                                    # population snapshot at this specific collision
            sampled_collisions[str(col_num)] = [
                {'expression': expr, 'count':pr_data
            ]                                       # convert to list-of-dicts, keyed by collision number as string

        payload = {
            'config_id': config_id,
            'name': config[8] or f'Experiment {config_id}',
                                                    # fall back to a generated name if none was saved
            'generator_type': config[2],
            'random_seed': config[1],
            'total_collisions': config[3],
            'polling_frequency': config[4],
            'timestamp': config[7],
            'last_collision_number': last_collision,
            'final_state_counts': [{'expression': expr, 'count': count} for expr, count in final_state],
            'initial_expression_counts': [{'expression': expr, 'count': count} for expr, count in initial_expressions],

            # Graph data for home page import
            'collisions_data': {
                "experiment_history": {
                    str(m[0]): {
                        "entropy": m[1],
                        "unique_expressions": m[2]
                    } for m in metrics
                }                                   # rebuild the entropy/unique-expressions history by collision
            },

            # Add sampled collisions for distance analysis
            'sampled_collisions': sampled_collisions
                                                    # full population snapshots, used to rebuild dendrograms/
                                                    # ordination plots after re-importing this file
        }

        buffer = io.BytesIO()                       # build the JSON file in memory rather than on disk
        buffer.write(json.dumps(payload, inde
                                                    # serialize payload to pretty-printed JSON bytes
        buffer.seek(0)                              # rewind buffer so send_file reads from the start
        return send_file(buffer, mimetype='application/json', as_attachment=True,
                        download_name=f"experiment_{config_id}_final_state.json")
                                                    # stream it back as a downloadable .json file
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500







@app.route('/download_current_expressions/<int:config_id>')
                                                    # export the final population as a flat JSON list ("soup")
def download_current_expressions(config_id):
    """
    This is the primary export for the "Soup" (expressions + counts).
    """
    try:
        state_data = get_expressions_for_collision(config_id, -1)
                                                    # get the population at the last collision (-1 = final)
        if not state_data:
            return "No data found", 404

        # Flatten counts so [('x', 2)] becomes ['x', 'x']
        expression_pool = []
        for expr, count in state_data:
            expression_pool.extend([expr] * count)
                                                    # repeat each expression `count` times so the pool
                                                    # matches the actual population, not just unique entries

        buffer = io.BytesIO()                       # build the JSON file in memory rather than on disk
        buffer.write(json.dumps(expression_pool, indent=2).encode('utf-8'))
                                                    # serialize the flat list to pretty-printed JSON bytes
        buffer.seek(0)                              # rewind buffer so send_file reads from the start

        return send_file(
            buffer,
            mimetype='application/json',
            as_attachment=True,
            download_name=f"experiment_{config_id}_soup.json"
        )                                           # stream it back as a downloadable .json file
    except Exception as e:
        return str(e), 500

@app.route('/download_initial_expressions/<int:config_id>')
                                                    # export just the starting expressions as a plain text file
def download_initial_expressions(config_id):
    try:
        config, _, initial_expressions = get_experiment_details(config_id)
                                                    # fetch config + initial expressions (metrics unused here)
        if not config:
            return "Experiment not found", 40

        if not initial_expressions:
            return "No initial expressions recorded for this experiment", 404

        lines = []
        for entry in initial_expressions:
            if isinstance(entry, (list, tuple)):
                lines.append(str(entry[0]))          # entry is (expression, count) -> take just the expression
            else:
                lines.append(str(entry))      plain value -> use it as-is

        buffer = io.BytesIO()                        # build the text file in memory rather than on disk
        buffer.write("\n".join(lines).encode(
                                                    # one expression per line
        buffer.seek(0)                              # rewind buffer so send_file reads from the start
        filename = f"experiment_{config_id}_initial_expressions.txt"
        return send_file(buffer, mimetype='text/plain', as_attachment=True, download_name=filename)
                                                    # stream it back as a downloadable .txt file
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500










    @app.route('/upload_and_import', methods=['POST'])
                                                    # rebuild a full experiment in the DB from an exported JSON file
def upload_and_import():
    try:
        file = request.files.get('file')            # get the uploaded file from the multipart form
        if not file:
            return jsonify({'status': 'error', 'message': 'No file received'}), 400

        data = json.load(file)                       # parse the uploaded file directly as JSON

        # 1. Save Main Configuration
        prob_range = json.dumps(data.get('generator_params', {}))
                                                    # re-encode generator params for storage
        original_name = data.get('name', 'Imported Experiment')
                                                    # remember the name from the original export

        new_config_id = save_configuration(
            data.get('random_seed', 0),
            data.get('generator_type', 'Imported'),
            data.get('total_collisions', 1000),
            data.get('polling_frequency', 10),
            prob_range,
            f"{original_name} (Imported)"
        )                                           # insert a new Configurations row, marking it as imported

        # 2. Restore Averages (entropy & unique_expressions)
        if 'collisions_data' in data:                # the export included a metrics history
            history = data['collisions_data'].get('experiment_history', {})
            for col_num, metrics in history.items():
                save_averages(
                    new_config_id,
                    int(col_num),
                    metrics['entropy'],
                    metrics['unique_expressions']
                )                            n's entropy/unique-count into Averages

        # 3. Handle Molecular Population
        final_pop = data.get('final_state_counts', [])
        initial_pop = data.get('initial_expression_counts', [])
        last_col = data.get('last_collision_number', -1)

        # Save initial state (collision 0)
        startup_data = initial_pop if initial_pop else final_pop
                                                    # prefer the recorded initial state; if the export
                                                    # only had a final state, seed with that instead
        for item in startup_data:
            save_experiment_state(new_config_id, 0, item['expression'], item['count'])
                                                    # write as collision 0 (the starting population)

        # Save final state (collision -1) and last collision
        for item in final_pop:
            save_experiment_state(new_config_id, -1, item['expression'], item['count'])
                                                    # write as collision -1 (sentinel for "final state")
            if last_col != -1:
                save_experiment_state(new_config_id, last_col, item['expression'], item['count'])
                                                    # also write it under its real collision number if known

        # Also restore expression state at every sampled collision if available
        if 'sampled_collisions' in data:
            for col_num, expressions in data['sampled_collisions'].items():
                for item in expressions:
                    save_experiment_state(new_config_id, int(col_num), item['expression'], item['count'])
                                                    # replay every intermediate snapshot, needed for
                                             on plots to work after re-import

        # Store original name in generator_params for display
        cursor = sqlite3.connect(DB_NAME).curassing the models.py helpers
        cursor.execute(
            "UPDATE Configurations SET probabid = ?",
            (json.dumps({'original_name': original_name, 'imported': True}), new_config_id)
        )                                           # tag this row as imported and preserve the original name
        cursor.connection.commit()
        cursor.connection.close()

        return jsonify({'status': 'success', 'new_config_id': new_config_id, 'original_name': original_name})
                                                    # report the new experiment's ID back to the frontend

    except Exception as e:
        print(f"Import Error: {e}")                 # log the failure server-side
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/delete_experiment', methods=['POST'])
                                                    # delete a single experiment and all its related rows
def delete_experiment_route():
    try:
        payload = request.get_json(silent=True) or {}
                                                    # parse JSON body, default to empty dict if missing/invalid
        config_id = payload.get('config_id')
        if not config_id:
            return jsonify({'status': 'error', 'message': 'Missing config_id'}), 400

        success = delete_experiment(int(config_id))  # cascading delete via models.py
        if success:
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'Failed to delete experiment'}), 500
    except Exception as exc:
        return jsonify({'status': 'error', 'm

@app.route('/delete_all_experiments', methods=['POST'])
                                                    # wipe every experiment from the database and reset IDs
def delete_all_experiments():
    try:
        # Get all current experiments
        experiments = get_experiment_configs()

        deleted_count = 0
        for exp in experiments:
            config_id = exp['config_id'] if isinstance(exp, dict) else exp[0]
                                                    # handle either dict-row or tuple-row format
            if delete_experiment(config_id):
                deleted_count += 1

        # Reset the ID counter
        import sqlite3
        from .config import DB_NAME                 # re-imported locally (redundant with top-level imports)
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM sqlite_sequence")
                                                    # clear SQLite's autoincrement tracking so new
                                                    # experiments start back at config_id 1
            conn.commit()
        except sqlite3.OperationalError as e:
            if "no such table: sqlite_sequence" not in str(e):
                raise e                              # only swallow the "table doesn't exist yet" case
        finally:
            conn.close()

        return jsonify({
            'status': 'success',
            'message': f'{deleted_count} experiments deleted and counters reset.'
        })

    except Exception as e:
        print(f"Error during mass delete: {e}ver-side
        return jsonify({'status': 'error', 'message': str(e)}), 500












@app.route('/debug_db')                            # quick diagnostic endpoint to dump raw Configurations rows
def debug_db():
    try:
        import sqlite3                              # local import (redundant with top-level import)
        from config import DB_NAME                  # NOTE: missing the leading dot (should be .config) —
                                                    # this import will likely fail unless run in a context
                                                    # where "config" resolves as a top-level module
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Configurations")
                                                    # pull every column of every experiment config, unfiltered
        configs = cursor.fetchall()
        conn.close()
        return jsonify({"status": "success", "message": f"Found {len(configs)} configurations", "data": configs})
                                                    # dump raw rows back as JSON for debugging
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/view_experiment/<int:config_id>')
                                                    # alternate route to render an experiment's detail page
                                                    # (simpler variant of database_view, no lineage/mode handling)
def view_experiment(config_id):
    config, metrics, initial_expressions = get_experiment_details(config_id)
                                                    # fetch config, metrics, and initial expressions
    if not config: return "Experiment not found", 404

    df = process_collision_data(metrics)             # convert raw metrics rows into a DataFrame
    plots = plot_experiment_metrics(df)              # build entropy + unique-expressions Bokeh figures
    entropy_script, entropy_div = components(
    unique_script, unique_div = components(plots['unique_expressions_plot'])
                                             to embeddable (script, div)

    experiment_details = {
        'config_id': config[0], 'random_seed': config[1], 'generator_type': config[2],
        'total_collisions': config[3], 'polling_frequency': config[4],
        'generator_params': {'freevar_generation_probability': config[6] if config[6] is not None else 0.5, 'probability_range': json.loads(config[5]) if
                                                    # rebuild generator params dict, defaulting freevar
                                                    # probability to 0.5 and decoding stored probability_range JSON
        'timestamp': config[7], 'name': confiid}"
                                                    # fall back to a generated name if none was saved
    }

    return render_template('database_view.html', experiment=experiment_details, initial_expressions=[expr[0] for expr in initial_expressions], bokeh_script=entropy_script + unique_script, bokeh_div=entropy_div, unique_expressions_script=unique_script, unique_expressions_div=unique_div, active_page='database')
                                                    # render the same template used by /database, but without
                                             ge_div or mode (those default to
                                                    # undefined in the template context)









@app.route('/upload_json', methods=['POST'])
                                                    # save an uploaded results JSON to disk and report its
                                                    # available metric keys (legacy/simple upload flow)
def upload_json():
    if 'json_file' not in request.files: return jsonify({'status': 'error', 'message': 'No file uploaded.'})
                                                    # no file field present in the form
    file = request.files['json_file']
    if not file.filename.endswith('.json'): return jsonify({'status': 'error', 'message': 'Invalid file type.'})
                                                    # only accept .json files
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)                             # persist the file to uploaded_configs/ (unlike
                                                    # upload_and_import, this keeps the raw file on disk)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "collisions_data" not in data: return jsonify({'status': 'error', 'message': "Missing 'collisions_data' in file."})
                                                    # basic shape check on the uploaded file
            return jsonify({'status': 'success', 'filename': file.filename, 'metrics': list(data["collisions_data"][next(iter(data["collisions_data"]))].keys())})
                                                    # peek at the first collision entry's keys to report
                                                    # which metrics are available for plotting
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/generate_visuals/<filename>', methods=['GET'])
                                                    # build Bokeh plots from a previously uploaded JSON file
def generate_visuals(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath): return jsonify({'status': 'error', 'message': f'File not found: {filepath}'})
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        from .plotting import create_bokeh_from_data
                                             oaded
        script, div = create_bokeh_from_data(data)   # build entropy/unique-expressions plots from raw JSON
        return jsonify({'status': 'success', 'script': script, 'div': div})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/update_experiment_name', methods=['POST'])
                                             xperiment
def update_name():
    try:
        config_id = int(request.form.get('config_id'))
        new_name = request.form.get('name')
        if not new_name or not new_name.strip(): return jsonify({"status": "error", "message": "Name cannot be empty"})
                                                    # reject blank/whitespace-only names
        success = update_experiment_name(config_id, new_name)
        if success: return jsonify({"status": "success", "message": "Experiment name updated"})
        return jsonify({"status": "error", "message": "Failed to update"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/perturb_and_run', methods=['POST'])
                                                    # placeholder route — feature not built yet
def perturb_and_run():
    return "Perturbation feature not yet implemented", 501
                                                    # 501 Not Implemented

@app.route('/list_experiments')
                                                    # list all experiments in a normalized JSON shape
                                             opdowns in the frontend)
def list_experiments():
    experiments = get_experiment_configs()
    result = {'experiments': []}
    for exp in experiments:
        if isinstance(exp, dict):            ict (sqlite3.Row-based query)
tamp': exp.get('timestamp')
            }
        else:                                       # row came back as a plain tuple
            entry = {
                'config_id': exp[0],
                'random_seed': exp[1],
                'generator_type': exp[2],
                'total_collisions': exp[3],
                'polling_frequency': exp[4],
                'timestamp': exp[5],
                'name': exp[8] if len(exp) > 8 else None                                                                                      r tuples that don't include name
xperiments', methods=['GET', 'POST'])
                                                    # page for comparing one metric across several experiments
def compare_experiments():
    if request.method == 'POST':                    # user submitted a comparison request
        selected_ids = request.form.getlist('experiment_ids')
                                                    # multiple checkbox values with the same fiel        metric = request.form.get('metric', '
                                                    # which metric to compare, default entropy
        if not selected_ids: return redirect(url_for('compare_experiments'))                                                                  just reload the empty form
        config_ids = [int(id) for id in selected_ids]
        metric_data = get_experiment_metrics(config_ids, metric)
                                                    # pull that metric's history for each selected experiment
        from .plotting import plot_comparison_metrics
                                                    # local import, lazy-loaded                          script, div = components(plot_compariic))
                                                    # build the overlaid comparison chart
        return render_template('compare_experiments.html', experiments=get_experiment_configs(), selected_ids=selected_ids, selected_metric=mekeh_div=div)
                                                    # re-render page with results + previous selections preserve    return render_template('compare_experimeneriment_configs())
                                                    # GET request -> just show the selection form

@app.route('/get_experiment_plot/<int:config_id>')
                                                    # AJAX endpoint: entropy + unique-expressions plots for one experiment
entropy_div = components(plots['entropy_plot'])
        unique_script, unique_div = components(plots['unique_expressions_plot'])
                                                    # split each figure into embeddable (script, div)
        return jsonify({
            "status": "success", "entropy_script": re.sub(r'<script[^>]*>', '', entropy_script).replace("</script>", ""),
            "entropy_div": entropy_div, "unique_expressions_script": re.sub(r'<script[^>]*>', '', unique_script).replace("</script>", ""),
            "unique_expressions_div": unique_div
        })                                          # strip <script> wrapper tags before returning both plots         except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500                                               
@app.route('/get_experiment_metadata/<int:config_id>')
                                                    # AJAX endpoint: config + initial expressions for one experiment
def get_experiment_metadata(config_id):
    try:                                                                                                                  config, metrics, initial_expressions ig_id)
success",
            "details": {'config_id': config[0], 'random_seed': config[1], 'generator_type': config[2], 'total_collisions': config[3], 'polling_frequency': config[4], 'generator_params': generator_params, 'timestamp': config[7], 'name': config[8] or f"Experiment {config[0]}", 'continuation': get_continuation_metadata(config_id)},
                                                    # bundle all config metadata, falling back to a generated
                                                    # name, plus parent/child linkage info
            "expressions": [expr[0] for expr in initial_expressions]
                                                    # just the expression strings from (expr, count) tuples
        })
    except Exception as e:                                                                                                return jsonify({"status": "error", "m

def create_histogram_html(histogram):
                                                    # build a simple HTML bar-chart of expression frequencies
                                                    # (used by the entropy-plot click-to-drill-down feature)
    if not histogram: return "<p>No data available for this collision.</p>"
    top_expressions = histogram[:20]                # only show the top 20 most frequent expressions
    max_count = max(h["count"] for h in top_expressions) if top_expressions else 1
                                                    # used to scale each bar's width as a percentage
    html = '<div style="margin: 20px 0;"><h4>Top 20 Expressions by Frequency</h4><div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;">'                                                           for item in top_expressions:
        expr, count = item["expression"], item["count"]
        html += f'<div style="margin-bottom: 8px;"><div style="display: flex; align-items: center; margin-bottom: 4px;"><div style="width: 200px; font-family: verflow: hidden; text-overflow: ellipsis;white-space: nowrap;" title="{expr}">{expr[:50] + "..." if len(expr) > 50 else expr}</div><div style="margin-left: 10px; font-weight: bold; min-width: 30px;">{count}</div></div><div style="background: #e2e8f0; height: 20px; border-radius: 3px; overflow: hidden;"><div style="background: #4F46E5; height: 100%; width: {(count / max_count) * 100}%; transition: width 0.3s ease;"></div></div></div>'
                                                    # one row per expression: truncated label (title = full
                                                    # text on hover), count, and a bar sized relative to max_count
    return html + '</div></div>'
















@app.route('/get_entropy_detail/<int:collision_number>')
                                                    # AJAX endpoint: returns an HTML fragment showing entropy +
                                                    # expression histogram for one clicked collision point
def get_entropy_detail(collision_number):
    try:
        config_id = int(request.args.get("config_id"))
                                                    # which experiment this collision belongs to
        result = get_entropy_and_histogram(config_id, collision_number)
                                                    # fetch entropy value + expression frequency list
        return f'<div class="card-header"><h3 class="card-title">Details for Collision {collision_number}</h3></div><div class="card-body"><p><strong>Entropy:</strong> {result["entropy"]:.4f}</p>{create_histogram_html(result["histogram"])}</div>'
                                                    # return raw HTML directly (not JSON) — the frontend
                                                    # injects this straight into a div's innerHTML
    except Exception as e:
        return f"Error: {str(e)}", 500

#sequence alignment route for comparing expressions using Levenshtein distance and percentage identity

@app.route('/api/sequence_alignment/<int:config_id>', methods=['POST'])
                                                    # find the molecules most structurally similar to a target
                                                    # expression, using Levenshtein edit distance
def sequence_alignment(config_id):
    try:
        import Levenshtein                          # local import, lazy-loaded
        from .db_utils import get_comparison_data    # local import, lazy-loaded

        data = request.get_json()
        target_expr = data.get('expression')         # the expression to compare everything else against

        if not target_expr:
            return jsonify({'status': 'error', 'message': 'No target expression provided.'}), 400

        # obtain most abundant molecules
        df = get_comparison_data(config_id, m
                                                    # top 100 most abundant expressions across the run
        if df.empty:
            return jsonify({'status': 'error', 'message': 'No data found for this experiment.'}), 404

        unique_molecules = df['expression'].unique().tolist()

        results = []
        for expr in unique_molecules:
            #molcule is not compared to self
            if expr == target_expr:
                continue
            #use levenshtein distance to calculate conversion from target expr to expr
            dist = Levenshtein.distance(str(t
                                                    # number of single-character edits to turn one into the other
            max_len = max(len(str(target_expr)), len(str(expr)))
                                                    # normalize by the longer string's length

            # percentage identity forumla
            identity = round((1 - (dist / max 0 else 100.0
                                                    # convert edit distance into a 0-100% similarity score

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
        print(f"Alignment Error: {e}")              # log the failure server-side
        return jsonify({'status': 'error', 'message': str(e)}), 500

#function to simulate extinction event
@app.route('/trigger_extinction', methods=['POST'])
                                                    # remove one expression from a parent experiment's final
                                                    # population, optionally refill to original size, then
                                                    # re-run the simulation on the survivors
def trigger_extinction():
    try:
        data = request.get_json()
        parent_id = data.get('config_id')            # which experiment to branch off of
        target_expr = data.get('target_expres
                                                    # the expression to wipe out
        should_refill = data.get('refill', False)     # whether to top the population back up after removal

        if not parent_id or not target_expr:
            return jsonify({'status': 'error', 'message': 'Missing data'}), 400

        parent_data = get_experiment_details(parent_id)
        parent_config = parent_data[0]               # just need the config row, not metrics/expressions

        final_state = get_expressions_for_collision(parent_id, -1)
                                             at its last collision

        # Identify survivors
        survivors = [item for item in final_srget_expr]
                                                    # everything except the targeted expression
        if not survivors:
            return jsonify({'status': 'error', 'message': 'Extinction wiped out everyone!'}), 400
                                                    # target was the entire population

        # Calculate Original and Current Population
        original_n = sum(count for _, count i
                                                    # total population size before extinction
        survivor_pool = []

        mode_label = "Refill" if should_refil

        if should_refill:
            # add actual survivors to pool
            for expr, count in survivors:
                survivor_pool.extend([expr] * count)


            x_to_add = original_n - len(survivor_pool)
                                                    # how many slots need to be refilled to restore
                                                    # the original population size

            # find top survivors by count to add extra copies of
            top_performers = sorted(survivors, key=lambda x: x[1], reverse=True)
                                                    # most abundant survivors get boosted first


            for i in range(x_to_add):

                boost_target = top_performers[i % len(top_performers)][0]
                                                    # cycle through top performers, adding one copy each
                survivor_pool.append(boost_target)
        else:

            for expr, count in survivors:
                survivor_pool.extend([expr] * count)
                                             ation just shrinks, no refill

        short_target = (target_expr[:12] + "..") if len(target_expr) > 12 else target_expr
                                                    # truncate long expressions for display purposes
        temp_name = f"Extinction ({mode_label}) - Removed: {short_target}"

_collisions': parent_config[3],    # inherit collision count from parent
            'polling_frequency': parent_config[4],   # inherit polling frequency from parent
            'random_seed': parent_config[1],         # inherit random seed from parent
            'experiment_name': temp_name
        }
        result = run_experiment(config)              # re-run the simulation on the modified population

        new_id = save_configuration(
            config['random_seed'], 'from_file', config['total_collisions'],
            config['polling_frequency'],
            json.dumps({'event': 'extinction'urged': target_expr}),
            temp_name
        )                                           # save the new experiment, recording what was purged

        update_experiment_name(new_id, f"Expe({mode_label}) - Removed: {short_target}")
                                  # rename with the new ID now that it's known

        for expr, count in Counter(survivor_pool).items():
            save_experiment_state(new_id, 0, expr, count)
                                                    # save the post-extinction pool as the initial state (collision 0)

        metrics = result.get('metrics', [])
        for metric in metrics:
            save_averages(new_id, metric['collision_number'], metric['entropy'], metric['unique_expressions'])
                                                    # save entropy/unique-count for each sampled collision
            if 'expressions' in metric:
                for expr, count in Counter(metric['expressions']).items():
                    save_experiment_state(new_id, metric['collision_number'], expr, count)
                                             tion snapshot at that collision

        save_continuation_metadata(new_id, parent_id, 1.0, len(survivor_pool), 0)
                                                    # record the parent/child link (100% of survivors reused,
                                                    # no brand-new expressions added)

        return jsonify({'status': 'success',

    except Exception as e:
        return jsonify({'status': 'error', 'm


#route for comparison dendrograms
@app.route('/multi_compare')
                                             ultiple experiments to compare
ts so the user can select them from a list
    experiments = get_experiment_configs()
    return render_template('multi_compare.html', experiments=experiments, active_page='multi_compare')

# multiple dendrogram api route

@app.route('/api/generate_multi_dendrogram', methods=['POST'])
                                                    # AJAX endpoint: build a dendrogram comparing several experiments
def generate_multi_dendrogram():
    try:
        data = request.get_json()
        ids = data.get('experiment_ids', [])         # which experiments to include


        user_limit = data.get('limit', 20)   expressions to include per experiment

        if len(ids) < 2:
            return jsonify({'status': 'error', 'message': 'Please select at least two experiments to compare.'}), 400
                                             periments to compare

        from .plotting import create_multi_experiment_dendrogram
                                                    # local import, lazy-loaded


        script, div = create_multi_experiment_dendrogram(ids, limit=user_limit)
                                                    # build the cross-experiment clustering plot

        import re                                   # local import (redundant with top-level import)
        clean_script = re.sub(r'<script[^>]*>', '', script).replace("</script>", "")
                                                    # strip <script> wrapper tags before returning

        return jsonify({
            'status': 'success',
            'script': clean_script,
            'div': div
        })
    except Exception as e:
        print(f"Error generating dendrogram: ver-side
        return jsonify({'status': 'error', 'message': str(e)}), 500







@app.route('/run_simulation_form', methods=['POST'])
                                                    # main entry point for launching a simulation from the UI —
                                                    # handles fresh runs, recursive continuations, and
                                                    # multi-generation chains all in one route
def run_simulation_form():
    try:
        total_collisions = int(request.form.get('total_collisions', 1000))
        polling_frequency = int(request.form.get('polling_frequency', 10))
        random_seed = int(request.form.get('random_seed', 42))
        base_name = request.form.get('experiment_name', 'Auto-Evo')
        if not base_name.strip():
            base_name = 'Auto-Evo'                  # fall back to a default if name is blank/whitespace

        num_generations = int(request.form.get('num_generations', 1))
                                                    # how many chained generations to run in this request

        recursive_parent_id = request.form.get('recursive_parent_id')
                                                    # set only when continuing from an existing experiment
        generator_type = request.form.get('generator_type', 'Fontana')

        current_pool = None                         # holds the expression pool carried between generations
        last_config_id = int(recursive_parent_id) if recursive_parent_id else None
                                                    # tracks the most recently created config_id in the chain

        # If starting from an existing parent, fetch its survivors once
        if last_config_id:
            from .db_utils import get_experiment_details, get_expressions_for_collision
                                                    # local import, lazy-loaded (shadows the top-level import
                                                    # of get_expressions_for_collision within this scope)
            parent_config, _, _ = get_experiment_details(last_config_id)
            random_seed = parent_config[1]           # inherit the parent's random seed

            final_state = get_expressions_for_collision(last_config_id, -1)
                                                    # parent's population at its last collision
            if not final_state:
                return jsonify({'status': 'error', 'message': f"Parent ID {last_config_id} has no collision data to inherit."}), 400

            current_pool = []
  def dashboard():
      experiments = get_experiment_configs()[:5
      return render_template('dashboard.html', experiments=[{'config_id': exp[0], 'random_seed': exp[1], 'generator_type': exp[2], 'total_collisions': exp[3], 'polling_frequency': exp[4], 'timestamp': exp[5]} for exp in experiments], generator_counts={exp[2]: sum(1 for e in experiments if e[2] == exp[2]) for exp in experiments}, total_experiments=len(experiments))

  if __name__ == '__main__':
      app.run(debug=True)

@app.route('/run_simulation_form', methods=['POST'])
                                                    # main entry point for launching a simulation from the UI —
                                                    # handles fresh runs, recursive continuations, and
                                                    # multi-generation chains all in one route
def run_simulation_form():
    try:
        total_collisions = int(request.form.get('total_collisions', 1000))
        polling_frequency = int(request.form.get('polling_frequency', 10))
        random_seed = int(request.form.get('random_seed', 42))
        base_name = request.form.get('experiment_name', 'Auto-Evo')
        if not base_name.strip():
            base_name = 'Auto-Evo'                  # fall back to a default if name is blank/whitespace

        num_generations = int(request.form.get('num_generations', 1))
                                                    # how many chained generations to run in this request

        recursive_parent_id = request.form.get('recursive_parent_id')
                                                    # set only when continuing from an existing experiment
        generator_type = request.form.get('ge

        current_pool = None                   pool carried between generations
        last_config_id = int(recursive_parent_id) if recursive_parent_id else None
                                                    # tracks the most recently created config_id in the chain

        # If starting from an existing parent, fetch its survivors once
        if last_config_id:
            from .db_utils import get_experiment_details, get_expressions_for_collision
                                                    # local import, lazy-loaded (shadows the top-level import
                                                    # of get_expressions_for_collision within this scope)
            parent_config, _, _ = get_experiment_details(last_config_id)
            random_seed = parent_config[1]           # inherit the parent's random seed

            final_state = get_expressions_for_collision(last_config_id, -1)
                                                    # parent's population at its last collision
            if not final_state:
                return jsonify({'status': 'error', 'message': f"Parent ID {last_config_id} has no collision data to inherit."}), 400

            current_pool = []
            for expr, count in sorted(final_state, key=lambda x: x[1], reverse=True)[:15]:
                current_pool.extend([expr] * count)
                                                    # seed the next generation with only the top 15 most
                                                    # abundant survivor expressions

            generator_type = 'from_file'             # subsequent generations always feed expressions directly

        # --- MULTI-GENERATION LOOP ---
        for gen in range(num_generations):
            gen_idx = gen + 1
            exp_name = f"{base_name} [Gen {gen_idx}]" if num_generations > 1 else base_name
                                                    # only append a generation tag if there's more than one

            config = {
                'generator_type': generator_type,
                'total_collisions': total_collisions,
                'polling_frequency': polling_frequency,
                'random_seed': random_seed,
                'experiment_name': exp_name,
                'expressions': current_pool          # None on gen 0 of a fresh run; set for later generations
            }

            # Apply parameters for Gen 1 if it is a FRESH start
            if gen == 0 and not recursive_parent_id:
                                                    # only the very first generation of a brand-new run
                                                    # needs generator-specific params filled in
                if generator_type == 'from_file':
                    # Check for uploaded file
                    if 'expressions_file' in request.files and request.files['expressions_file'].filename:
                        file = request.files['expressions_file']
                        filename = secure_filename(file.filename)
                                             fore use

                        if filename.endswith('.json'):
                            file_content = file.read().decode('utf-8')
                            data = json.loads(file_content)
                            config['expressions'] = []

                            if isinstance(data, dict):
                                # PRIORITY 1: Final state counts
                                if 'final_state_counts' in data:
                                    for item in data['final_state_counts']:
                                        config['expressions'].extend([item['expression']] * item['count'])
                                    print(f"Ls'])} expressions from FINAL STATE")
                                                    # prefer a previously exported final population

                                # PRIORITY 2:
                                elif 'initial_expression_counts' in data:
                                    for item in data['initial_expression_counts']:
                                        config['expressions'].extend([item['expression']] * item['count'])
                                    print(f"Loaded {len(config['expressions'])} expressions from INITIAL STATE")
                                                    # fall back to an initial-state export

                                else:
                                    return jsonify({'status': 'error', 'message': 'JSON missing expression counts. Use
Final State or Initial State JSON.'}), 400
                                                    # dict didn't match either expected shape

                            elif isinstance(data, list):
                                config['expressions'] = data
                                print(f"Loaded {len(config['expressions'])} expressions from flat list")
                                             sion strings

                            else:
                                return jsonify({'status': 'error', 'message': 'JSON format not recognized'}), 400

                        else:
                            # Text file: read line by line
                            content = file.read().decode('utf-8')
                            config['expressions'] = [line.strip() for line in content.split('\n') if line.strip()]
                                                    # one expression per non-blank line

                    # Check for direct input
                    elif request.form.get('direct_input'):
                        config['expressions'] = [e.strip() for e in request.form.get('direct_input').split('\n') if e.strip()]
                                             ns directly into a textarea

                    # Validate we have expressions
                    if not config.get('expressions'):
                        return jsonify({'status': 'error', 'message': 'No expressions provided.'}), 400
                                             nput, nothing to run

                elif generator_type == 'BTree':
                    config.update({
                        'size': int(request.form.get('btree_size', 5)),
                        'freevar_probability': float(request.form.get('freevar_probability', 0.5)),
                        'max_free_vars': int(request.form.get('max_free_vars', 3)),
                        'standardization': request.form.get('standardization', 'prefix'),
                        'num_expressions': int(request.form.get('num_expressions', 10))
                    })                              # pull BTree generator params from the form
                elif generator_type == 'Fontana':
                    config.update({
                        'abs_low': float(request.form.get('abs_low', 0.1)),
                        'abs_high': float(request.form.get('abs_high', 0.5)),
                        'app_low': float(requ,
                        'app_high': float(request.form.get('app_high', 0.6)),
                        'min_depth': int(request.form.get('min_depth', 1)),
                        'max_depth': int(request.form.get('max_depth', 5)),
                        'max_free_vars': int(request.form.get('fontana_max_fv', 2)),
                        'initial_expression_count': int(request.form.get('fontana_expression_count', 10)),
                        'free_variable_probabt('free_variable_probability', 0.5))
                    })                              # pull Fontana generator params from the form

            # Fire Engine
            result = run_experiment(config)          # actually run the simulation for this generation

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
            )                                       # create the Configurations row for this generation

            # Save population/metrics
            initial_expressions = result.get('initial_expressions', [])
            for expr, count in Counter(initial_expressions).items():
                save_experiment_state(new_id, 0, expr, count)
                                                    # save starting population as collision 0

            metrics = result.get('metrics', [])
            for metric in metrics:
                save_averages(new_id, metric['collision_number'], metric['entropy'], metric['unique_expressions'])
                                                    # save entropy/unique-count per sampled collision
                if 'expressions' in metric:
                    for expr, count in Counter(metric['expressions']).items():
                        save_experiment_statember'], expr, count)
                                                    # save full population snapshot at that collision

            # Link Lineage
            if last_config_id:
                save_continuation_metadata(new_id, last_config_id, 1.0, len(current_pool or initial_expressions), 0)
                                                    # record parent -> child link for this generation

            # SETUP FOR NEXT GENERATION IN THE LOOP
            last_config_id = new_id                  # this generation becomes the parent for the next
            generator_type = 'from_file'             # every subsequent generation feeds expressions directly

            # Extract survivors directly from memory for the next loop
            if metrics and 'expressions' in metrics[-1]:
                survivors = Counter(metrics[-1]['expressions']).items()
                                                    # tally the final sampled population in memory
                                                    # (avoids re-querying the DB)
                current_pool = []
                for expr, count in sorted(survivors, key=lambda x: x[1], reverse=True)[:15]:
                    current_pool.extend([expr] * count)
                                                    # carry only the top 15 most abundant survivors forward
            else:
                raise ValueError(f"{exp_name} produced no surviving expressions to pass on.")
                                                    # can't continue the chain with nothing to inherit

        return jsonify({
            'status': 'success',
            'config_id': last_config_id,             # the final generation's config_id
            'experiment_name': exp_name,
            'message': f"Successfully ran {num_generations} generation(s)."
        })

    except Exception as e:
        print(f"Recursion Loop Error: {e}")          # log the failure server-side
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/trigger_invasive_species', methods=['POST'])
                                                    # inject a new "invasive" expression into a parent
                                                    # experiment's final population and re-run the simulation
def trigger_invasive_species():
    try:
        data = request.get_json()
        parent_config_id = data.get('config_id')     # which experiment to branch off of
        invasive_expr = data.get('expression', '\\x.x')
                                                    # the expression being introduced, default identity function
        invasive_count = int(data.get('count' inject

        if not parent_config_id:
            return jsonify({'status': 'error'id'}), 400

        #fetch parent data
        parent_data = get_experiment_details(parent_config_id)
        parent_config = parent_data[0]               # just need the config row

        final_state = get_expressions_for_collision(parent_config_id, -1)
                                                    # parent's population at its last collision
        if not final_state:
            return jsonify({'status': 'error', 'message': 'No final data found.'}), 404

        survivor_expressions = []
        for expr, count in final_state:
            survivor_expressions.extend([expr] * count)
                                                    # flatten (expr, count) pairs into a repeated list

        # Inject the invasive molecules
        survivor_expressions.extend([invasive_expr] * invasive_count)
                                                    # add the invasive expression copies into the pool

        config = {
            'generator_type': 'from_file',
            'expressions': survivor_expressions,
            'total_collisions': 1000,        ot configurable from the UI here
            'polling_frequency': 10,
            'random_seed': parent_config[1],         # inherit parent's random seed
            'experiment_name': f"Invasion: {invasive_expr[:20]} (Parent: {parent_config_id})"
                                                    # truncate long expressions in the display name
        }

        result = run_experiment(config)              # re-run the simulation on the invaded population
        new_id = save_configuration(
            random_seed=parent_config[1], generator_type='from_file',
            total_collisions=1000, polling_frequency=10,
            name=config['experiment_name']
        )                                           # create the new Configurations row

        for expr, count in Counter(survivor_e
            save_experiment_state(new_id, 0, expr, count)
                                                    # save the post-invasion pool as the initial state

        metrics = result.get('metrics', [])
        for metric in metrics:
            save_averages(new_id, metric['collision_number'], metric['entropy'], metric['unique_expressions'])
                                             count per sampled collision
            if 'expressions' in metric:
                for expr, count in Counter(metric['expressions']).items():
                    save_experiment_state(new'], expr, count)
                                              snapshot at that collision

        save_continuation_metadata(new_id, parent_config_id, 1.0, len(survivor_expressions) - invasive_count, invasive_count)
                                             essions were reused vs. newly added

        return jsonify({'status': 'success', 'new_config_id': new_id})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/final_state/<int:config_id>')
                                                    # AJAX endpoint: just the final population counts for one experiment
def api_final_state(config_id):
    try:
        final_state = get_expressions_for_collision(config_id, -1)
                                                    # population at the last collision (-1 = final)
        if not final_state:
            return jsonify({'status': 'error', 'message': 'No final state found'}), 404
        return jsonify({
            'status': 'success',
            'final_state_counts': [{'expression': expr, 'count': count} for expr, count in final_state]
                                                    # convert (expr, count) tuples into a list of dicts
        })
    except Exception as exc:
        return jsonify({'status': 'error', 'message': str(exc)}), 500



@app.route('/dashboard')
                                                    # small summary view showing the 5 most recent experiments
def dashboard():
    experiments = get_experiment_configs()[:5]       # only the 5 most recent (list is already newest-first)
    return render_template('dashboard.html', experiments=[{'config_id': exp[0], 'random_seed': exp[1], 'generator_type': exp[2], 'total_collisions': exp[3], 'polling_p': exp[5]} for exp in experiments],generator_counts={exp[2]: sum(1 for e in experiments if e[2] == exp[2]) for exp in experiments}, total_experiments=len(experiments))
                                             lies how many of these 5 use each
                                                    # generator type; total_experiments is just len(experiments)
                                                    # NOTE: both stats are computed only over this 5-item slice,
                                                    # not the full experiment history

if __name__ == '__main__':
    app.run(debug=True)                      erver with debug mode (auto-reload,
                                                    # interactive tracebacks) when executed directly





