# alchemy_dashboard/simulation.py
"""
Runs lambda-calculus "soup" simulations using the `alchemy` library.

A simulation starts with a population of lambda expressions. At each
collision, expressions are combined and reduced, so the population changes
over time. Every `polling_frequency` collisions we record a snapshot:
the entropy, the number of unique expressions, and the full population.

Main entry point:
    run_experiment(config)  builds the starting population, runs the
                            simulation, and returns the recorded snapshots.
                            It does not save anything; main.py saves the
                            results to the database.

The starting population comes from a generator:
    "BTree"      random expressions of a fixed size (alchemy.PyBTreeGen)
    "Fontana"    random expressions within a depth range (alchemy.PyFontanaGen)
    "from_file"  a list of expressions supplied by the user
"""

import json
import os
import alchemy
import random
from collections import Counter
from. db_utils import get_expressions_for_collision

def _seed_hex(seed):
    """Convert an integer seed to the 64-char hex string the alchemy generators expect.

    The Rust side takes a 32-byte seed as hex; passing None makes it pick a
    random seed, so runs would not be reproducible.
    """
    if seed is None:
        return None
    return f"{int(seed) % (1 << 256):064x}"


# Defaults for the reactor and soup settings, matching Modern-AlChemy dev
# (src/config/reactors/untyped_lambda.rs and src/config/soup.rs). A value in
# the dashboard config overrides each one.
REACTOR_DEFAULTS = {
    "rules": ["\\x.\\y.x y"],
    "discard_copy_actions": True,
    "discard_identity": True,
    "discard_free_variable_expressions": True,
    "reduction_cutoff": 500,
    "size_cutoff": 500,
}
SOUP_DEFAULTS = {
    # With discard_parents on, Modern-AlChemy replaces the two parents with
    # two new expressions from the config's generator.
    "discard_parents": False,
    "maintain_constant_population_size": True,
}


def settings_from_rust_config(rust_config_json):
    """Read the reactor and soup settings back out of a saved config.

    The reverse of build_rust_config_json, for child runs (recursive,
    extinction, invasive species) that should keep their parent's settings.

    Args:
        rust_config_json (str or None): a config saved by an earlier run.

    Returns:
        dict: the REACTOR_DEFAULTS and SOUP_DEFAULTS keys found in the
            config, plus "parent_generator" (the parent's generator block).
            Empty for experiments saved before configs were stored, so those
            children use the defaults.
    """
    if not rust_config_json:
        return {}
    rust_config = json.loads(rust_config_json)
    expression = rust_config["expression"]["UntypedLambda"]
    reactor = expression["reactor"]
    soup = rust_config["soup_config"]
    settings = {key: reactor[key] for key in REACTOR_DEFAULTS if key in reactor}
    settings.update({key: soup[key] for key in SOUP_DEFAULTS if key in soup})
    settings["parent_generator"] = expression["generator"]
    return settings


def form_values_from_rust_config(rust_config):
    """Turn an uploaded config into values for the Run Simulation form.

    Accepts the files from Download Config JSON and Modern-AlChemy's own
    config files. The config does not include starting expressions, so
    this only fills the form for a fresh BTree or Fontana run.

    Args:
        rust_config (dict): the parsed config.

    Returns:
        tuple: (values, warnings)
            values: dict of form field name -> value
            warnings: list of str, for settings that could not be loaded

    Raises:
        ValueError: if the config is missing a section or has a setting the
            dashboard cannot run with.
    """
    try:
        expression = rust_config["expression"]["UntypedLambda"]
        generator = expression["generator"]
        reactor = expression["reactor"]
        soup = rust_config["soup_config"]
    except (KeyError, TypeError):
        raise ValueError("Not a Modern-AlChemy config: expected expression.UntypedLambda "
                         "with generator and reactor, and soup_config")

    if reactor.get("rules", REACTOR_DEFAULTS["rules"]) != REACTOR_DEFAULTS["rules"]:
        raise ValueError(f"The dashboard only runs the rule {REACTOR_DEFAULTS['rules'][0]}")

    values = {}
    warnings = []

    if rust_config.get("run_limit") is not None:
        values["total_collisions"] = int(rust_config["run_limit"])
    if rust_config.get("polling_interval") is not None:
        values["polling_frequency"] = int(rust_config["polling_interval"])
    else:
        warnings.append("No polling_interval in the file; kept the form's value.")
    if soup.get("seed") is not None:
        # Saved as 64 hex characters (see _seed_hex)
        values["random_seed"] = int(soup["seed"], 16)
    else:
        warnings.append("No seed in the file; kept the form's value.")

    sample_size = rust_config.get("sample_size")
    if "BTree" in generator:
        params = generator["BTree"]
        values.update({
            "generator_type": "BTree",
            "btree_size": int(params["size"]),
            "freevar_probability": float(params["freevar_generation_probability"]),
            "max_free_vars": int(params["n_max_free_vars"]),
            "standardization": str(params["standardization"]).lower(),
        })
        if sample_size is not None:
            values["num_expressions"] = int(sample_size)
    elif "Fontana" in generator:
        params = generator["Fontana"]
        values.update({
            "generator_type": "Fontana",
            "abs_low": float(params["abstraction_prob_range"][0]),
            "abs_high": float(params["abstraction_prob_range"][1]),
            "app_low": float(params["application_prob_range"][0]),
            "app_high": float(params["application_prob_range"][1]),
            "min_depth": int(params.get("min_depth", 1)),
            "max_depth": int(params["max_depth"]),
        })
        if sample_size is not None:
            values["fontana_expression_count"] = int(sample_size)
    else:
        raise ValueError(f"Unknown generator: {', '.join(generator) or 'none'}")

    for key in REACTOR_DEFAULTS:
        if key != "rules" and key in reactor:
            values[key] = reactor[key]
    for key in SOUP_DEFAULTS:
        if key in soup:
            values[key] = soup[key]

    recursive = rust_config.get("recursive_config") or {}
    if recursive.get("n_generations", 1) != 1:
        warnings.append("recursive_config was ignored; use Recursive Simulation for generations.")

    return values, warnings


def build_rust_config_json(config, sample_size=None):
    """Build the JSON string that alchemy.PyConfig.from_json reads.

    This is the only place that knows the shape of Modern-AlChemy's config
    (dev branch, default/config.json). If that shape changes, change it here.

    Args:
        config (dict): the dashboard config passed to run_experiment().
        sample_size (int, optional): size of the starting population. Pass
            the real count once the soup is filled; otherwise the number of
            expressions the generator was asked for is used.

    Returns:
        str: the config as JSON.

    Raises:
        ValueError: if a value is missing or invalid. Rust panics on a bad
            config instead of raising an error, so it is checked here first.
    """
    seed = _seed_hex(config.get("random_seed"))
    if seed is None:
        # A null seed makes Rust pick one we can never read back.
        raise ValueError("random_seed is required")

    generator_type = config["generator_type"]
    if generator_type == "BTree":
        standardization = config["standardization"].lower()
        if standardization not in ("prefix", "postfix", "none"):
            raise ValueError(f"Unknown standardization: {standardization}")
        generator = {"BTree": {
            "seed": seed,
            "size": int(config["size"]),
            "freevar_generation_probability": float(config["freevar_probability"]),
            "n_max_free_vars": int(config["max_free_vars"]),
            # The form sends "prefix"; the JSON wants "Prefix"
            "standardization": standardization.capitalize(),
        }}
        requested = config.get("num_expressions", 10)
    elif generator_type == "Fontana":
        generator = {"Fontana": {
            "seed": seed,
            "min_depth": int(config.get("min_depth", 1)),
            "max_depth": int(config["max_depth"]),
            "abstraction_prob_range": [float(config["abs_low"]), float(config["abs_high"])],
            "application_prob_range": [float(config["app_low"]), float(config["app_high"])],
        }}
        requested = config.get("initial_expression_count", 10)
    elif generator_type == "from_file":
        # The starting expressions are given, but the generator still makes
        # replacements when discard_parents is on. A child run reuses its
        # parent's generator (with this run's seed) so replacements match
        # the parent's settings. Uploads and children of experiments saved
        # before configs were stored fall back to a fixed BTree generator.
        parent_generator = config.get("parent_generator")
        if parent_generator:
            (kind, params), = parent_generator.items()
            generator = {kind: {**params, "seed": seed}}
        else:
            generator = {"BTree": {
                "seed": seed,
                "size": 20,
                "freevar_generation_probability": 0.2,
                "n_max_free_vars": 6,
                "standardization": "Prefix",
            }}
        requested = len(config.get("expressions") or [])
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")

    reactor = {key: config.get(key, default) for key, default in REACTOR_DEFAULTS.items()}
    soup = {key: config.get(key, default) for key, default in SOUP_DEFAULTS.items()}
    soup["seed"] = seed

    rust_config = {
        "expression": {
            "UntypedLambda": {
                "generator": generator,
                "reactor": reactor,
            }
        },
        "run_limit": int(config["total_collisions"]),
        "sample_size": int(sample_size if sample_size is not None else requested),
        "polling_interval": int(config["polling_frequency"]),
        "verbose_logging": False,
        "soup_config": soup,
        # The dashboard runs its own generations, one config each, so Rust's
        # recursive settings stay at their single-generation defaults.
        "recursive_config": {
            "n_generations": 1,
            "wipeout_percent": 0,
            "refill_type": "ConfigGenerator",
            "repopulation_expression": "\\x.\\y.x y",
        },
    }
    return json.dumps(rust_config)


def canonical_expression(expr):
    """Return the engine's canonical form of one expression, e.g. "\\x.x" -> "λa.a".

    Expressions are stored the way the engine writes them, so user input must
    be converted before it is stored or compared. Returns None if the engine
    cannot parse the expression.
    """
    soup = alchemy.PySoup()
    soup.perturb([expr])
    parsed = soup.expressions()
    return parsed[0] if parsed else None


def load_input_expressions(generator_type, gen_params):
    """
    Load initial expressions based on generator type and parameters.

    Note: run_experiment() does not use this function; it has its own
    generator code. The "Fontana" option here only returns 3 fixed
    placeholder expressions, and unknown generator types return 3 identity
    functions instead of raising an error.

    Args:
        generator_type (str): Type of generator to use
            ("from_file", "BTree", or "Fontana")
        gen_params (dict): Generator parameters. For "from_file": "filename"
            (a text file with one expression per line). For "BTree": "size",
            "freevar_generation_probability", "max_free_vars",
            "standardization", "num_expressions".

    Returns:
        list: List of initial expressions (empty if the file is missing)
    """
    if generator_type == "from_file":
        filename = gen_params.get("filename")
        if filename and os.path.exists(filename):
            with open(filename, "r") as f:
                return [line.strip() for line in f if line.strip()]
        return []
    
    elif generator_type == "BTree":
        size = gen_params.get("size", 5)
        fvp = gen_params.get("freevar_generation_probability", 0.5)
        max_fv = gen_params.get("max_free_vars", 3)
        std_type = gen_params.get("standardization", "prefix")
        num_expr = gen_params.get("num_expressions", 10)
        
        btree_gen = alchemy.PyBTreeGen.from_config(
            size, fvp, max_fv, alchemy.PyStandardization(std_type)
        )
        return btree_gen.generate_n(num_expr)
    
    elif generator_type == "Fontana":
        # Implement Fontana generator if available
        # This is a placeholder - implement based on your alchemy library
        return ["(λx.x)", "(λx.λy.x y)", "(λx.x x)"]
    
    else:
        # Default to some basic lambda expressions
        return ["(λx.x)", "(λy.y)", "(λz.z)"]


# Continuation helpers
def _build_continuation_expressions(parent_config_id, fraction):
    """Return expressions drawn from the parent's last recorded state.

    Takes roughly `fraction` of each expression's count from the parent's
    latest recorded population, so the mix of expressions stays about the
    same. Every expression gets at least one copy while there is room, and
    the total is exactly round(total_population * fraction) (at least 1).

    Example: parent has {"A": 10, "B": 4} and fraction = 0.5
             -> 5 copies of "A" and 2 copies of "B".

    Args:
        parent_config_id (int or None): Experiment to take expressions from.
        fraction (float): Share of the parent's population to take; values
            outside 0.0-1.0 are clamped.

    Returns:
        list[str]: Flat list with one entry per copy (e.g. ["A", "A", "B"]).
            Empty if there is no parent, fraction <= 0, or no saved data.
    """
    if parent_config_id is None or fraction <= 0:
        return []

    last_state = get_expressions_for_collision(parent_config_id, -1)
    if not last_state:
        return []

    # Clamp fraction to [0.0, 1.0]
    fraction = max(0.0, min(1.0, fraction))
    if fraction == 0:
        return []

    total_population = sum(count for _, count in last_state)
    if total_population == 0:
        return []

    target_total = max(1, int(round(total_population * fraction)))

    sampled = []
    sampled_counter = Counter()
    remaining = target_total

    for expression, count in last_state:
        if remaining <= 0:
            break

        take = int(round(count * fraction))
        if take <= 0 and count > 0:
            # Guarantee at least one instance if we still need samples
            take = 1

        take = min(take, count, remaining)
        if take <= 0:
            continue

        sampled.extend([expression] * take)
        sampled_counter[expression] += take
        remaining -= take

    # If rounding undershot, top up greedily with available counts
    # (in the order the parent's expressions were returned)
    if remaining > 0:
        for expression, count in last_state:
            if remaining <= 0:
                break
            already_taken = sampled_counter.get(expression, 0)
            available = count - already_taken
            if available <= 0:
                continue
            take = min(available, remaining)
            sampled.extend([expression] * take)
            sampled_counter[expression] += take
            remaining -= take

    return sampled


def run_experiment(config):
    """
    Run an experiment with the given configuration.

    The starting population is: (expressions copied from a parent experiment,
    if "continuation" is given) + (new expressions from the generator).

    Args:
        config (dict): Configuration dictionary containing:
            - generator_type: Type of generator to use ('BTree', 'Fontana', 'from_file')
            - total_collisions: Number of collisions to simulate
            - polling_frequency: How often to record metrics
            - random_seed: Random seed for reproducibility
            - experiment_name: Optional name for the experiment (not used
              here; main.py uses it when saving)
            - continuation (optional): {"parent_config_id": int,
              "fraction": float} to start with part of a parent
              experiment's population
            - Additional parameters based on generator_type:
                BTree:     size, freevar_probability, max_free_vars,
                           standardization, num_expressions
                Fontana:   abs_low, abs_high, app_low, app_high, max_depth,
                           min_depth (default 1),
                           initial_expression_count (default 10)
                from_file: file_path (text file, one expression per line)
                           or expressions (list of strings)

    Returns:
        dict: Results containing metrics and initial expressions:
            - metrics: list of snapshots, one every polling_frequency
              collisions, each with collision_number, entropy,
              unique_expressions (a count), and expressions (the full
              population at that point)
            - initial_expressions: the starting population, in the engine's
              canonical form (unparseable expressions removed)
            - dropped_expression_count: how many starting expressions the
              engine could not parse
            - continuation_summary: how many expressions came from the
              parent vs. were newly generated

    Raises:
        ValueError: if generator_type is unknown or there are no starting
            expressions.
    """
    # Set random seed for reproducibility
    random.seed(config['random_seed'])
    
    # Initialize metrics collection
    metrics = []
    
    # Configure generator based on type
    generator_type = config['generator_type']
    continuation_info = config.get('continuation') or {}
    parent_config_id = continuation_info.get('parent_config_id')
    fraction_used = continuation_info.get('fraction', 0.0)

    continuation_expressions = _build_continuation_expressions(parent_config_id, fraction_used)
    new_expressions = []
    
    if generator_type == 'BTree':
        # Configure BTree generator
        std = alchemy.PyStandardization(config['standardization'])
        generator = alchemy.PyBTreeGen.from_config(
            size=config['size'],
            freevar_generation_probability=config['freevar_probability'],
            max_free_vars=config['max_free_vars'],
            std=std,
            seed=_seed_hex(config['random_seed'])
        )
        # Generate initial expressions
        new_expressions = generator.generate_n(config['num_expressions'])
        
    elif generator_type == 'Fontana':
        # Configure Fontana generator.
        # abs_range / app_range are probability ranges for creating
        # abstractions (\x.body) and applications (f x) while building a tree.
        # Note: max_free_vars and free_variable_probability sent by the form
        # are not passed to the generator.
        generator = alchemy.PyFontanaGen.from_config(
            abs_range=(config['abs_low'], config['abs_high']),
            app_range=(config['app_low'], config['app_high']),
            min_depth=config.get('min_depth', 1),
            max_depth=config['max_depth'],
            seed=_seed_hex(config['random_seed'])
        )
        # Generate initial expressions. generate() can return nothing, and
        # those attempts are skipped, so the result may have fewer than `desired`.
        desired = config.get('initial_expression_count', 10)
        new_expressions = []
        for _ in range(desired):
            expr = generator.generate()
            if expr:
                new_expressions.append(expr)
        
    elif generator_type == 'from_file':
        # Handle file-based input
        if 'file_path' in config:
            with open(config['file_path'], 'r') as f:
                new_expressions = [line.strip() for line in f if line.strip()]
        elif 'expressions' in config:
            new_expressions = config['expressions']
        else:
            # No new expressions is fine if we're continuing from a parent
            if not continuation_expressions:
                raise ValueError("No expressions provided for 'from_file' generator")
            new_expressions = []
    
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")

    initial_expressions = continuation_expressions + new_expressions

    if not initial_expressions:
        raise ValueError("No initial expressions available to start the simulation")
    
    # Initialize simulation: an empty soup, then add the starting population
    rust_config_json = build_rust_config_json(config, sample_size=len(initial_expressions))
    print(rust_config_json) 
    rust_config = alchemy.PyConfig.from_json(rust_config_json)
    simulation = alchemy.PySoup.from_config(rust_config)
    simulation.perturb(initial_expressions)

    # The engine converts expressions to its canonical form and silently
    # drops any it cannot parse, so report what was actually simulated.
    requested_count = len(initial_expressions)
    initial_expressions = simulation.expressions()
    dropped_count = requested_count - len(initial_expressions)
    if dropped_count:
        print(f"Warning: {dropped_count} starting expressions could not be parsed and were dropped")

    def record_snapshot(collision_number):
        metrics.append({
            'collision_number': collision_number,
            'entropy': simulation.population_entropy(),
            'unique_expressions': len(simulation.unique_expressions()),
            'expressions': simulation.expressions()  # Add full state data
        })

    # Collision 0 is the starting population, before any reaction
    record_snapshot(0)

    # Run simulation. Snapshot n is the state after n collisions, taken every
    # polling_frequency collisions, and the final collision is always recorded.
    total_collisions = config['total_collisions']
    for n in range(1, total_collisions + 1):
        simulation.simulate_for(1, log=False)
        if n % config['polling_frequency'] == 0 or n == total_collisions:
            record_snapshot(n)
    
    return {
        'metrics': metrics,
        'rust_config': rust_config_json,
        'initial_expressions': initial_expressions,
        'dropped_expression_count': dropped_count,
        'continuation_summary': {
            'parent_config_id': parent_config_id,
            'fraction_used': fraction_used,
            'continued_expression_count': len(continuation_expressions),
            'new_expression_count': len(new_expressions)
        }
    }
