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

import os
import alchemy
import random
from collections import Counter
from. db_utils import get_expressions_for_collision

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
            - initial_expressions: the starting population
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
            std=std
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
    simulation = alchemy.PySoup()
    simulation.perturb(initial_expressions)
    
    # Run simulation
    for i in range(config['total_collisions']):
        simulation.simulate_for(1, log=False)
        
        # Record metrics at specified intervals.
        # Snapshots are taken after collision i runs, at i = 0,
        # polling_frequency, 2 * polling_frequency, ... so the very last
        # collision is only recorded if it lands on one of those numbers.
        if i % config['polling_frequency'] == 0:
            # Get current state expressions
            current_expressions = simulation.expressions()
            
            metrics.append({
                'collision_number': i,
                'entropy': simulation.population_entropy(),
                'unique_expressions': len(simulation.unique_expressions()),
                'expressions': current_expressions  # Add full state data
            })
    
    return {
        'metrics': metrics,
        'initial_expressions': initial_expressions,
        'continuation_summary': {
            'parent_config_id': parent_config_id,
            'fraction_used': fraction_used,
            'continued_expression_count': len(continuation_expressions),
            'new_expression_count': len(new_expressions)
        }
    }
