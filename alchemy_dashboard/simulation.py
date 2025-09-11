# alchemy_dashboard/simulation.py

import os
import alchemy
import random
import json
from collections import Counter
from models import (
    save_configuration, 
    save_experiment_state, 
    save_averages, 
    get_last_config_id
)

def load_input_expressions(generator_type, gen_params):
    """
    Load initial expressions based on generator type and parameters.
    
    Args:
        generator_type (str): Type of generator to use
        gen_params (dict): Generator parameters
    
    Returns:
        list: List of initial expressions
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


# Update run_experiment function to handle experiment naming
def run_experiment(config):
    """
    Run an experiment with the given configuration.
    
    Args:
        config (dict): Configuration dictionary containing:
            - generator_type: Type of generator to use ('BTree', 'Fontana', 'from_file')
            - total_collisions: Number of collisions to simulate
            - polling_frequency: How often to record metrics
            - random_seed: Random seed for reproducibility
            - experiment_name: Optional name for the experiment
            - Additional parameters based on generator_type
    
    Returns:
        dict: Results containing metrics and initial expressions
    """
    # Set random seed for reproducibility
    random.seed(config['random_seed'])
    
    # Initialize metrics collection
    metrics = []
    initial_expressions = []
    
    # Configure generator based on type
    generator_type = config['generator_type']
    
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
        initial_expressions = generator.generate_n(config['num_expressions'])
        
    elif generator_type == 'Fontana':
        # Configure Fontana generator
        generator = alchemy.PyFontanaGen.from_config(
            abs_range=(config['abs_low'], config['abs_high']),
            app_range=(config['app_low'], config['app_high']),
            max_depth=config['max_depth'],
            max_free_vars=config['max_free_vars']
        )
        # Generate initial expressions
        initial_expressions = []
        for _ in range(10):  # Default to 10 expressions
            expr = generator.generate()
            if expr:
                initial_expressions.append(expr)
        
    elif generator_type == 'from_file':
        # Handle file-based input
        if 'file_path' in config:
            with open(config['file_path'], 'r') as f:
                initial_expressions = [line.strip() for line in f if line.strip()]
        elif 'expressions' in config:
            initial_expressions = config['expressions']
        else:
            raise ValueError("No expressions provided for 'from_file' generator")
    
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
    
    # Initialize simulation
    simulation = alchemy.PySoup()
    simulation.perturb(initial_expressions)
    
    # Run simulation
    for i in range(config['total_collisions']):
        simulation.simulate_for(1, log=False)
        
        # Record metrics at specified intervals
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
        'initial_expressions': initial_expressions
    }