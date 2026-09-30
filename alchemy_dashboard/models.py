# alchemy_dashboard/models.py
"""
Database layer for the Alchemy Dashboard.

Creates the SQLite tables and provides small helper functions to save, read,
rename, and delete experiments. Every function opens its own connection to
DB_NAME (from config.py) and closes it before returning.

Tables (created by init_database):
    Configurations        One row per experiment: its settings and name.
    Experiment            The population: how many copies of each expression
                          existed at a given collision. Collision 0 is the
                          initial population. (db_utils.get_expressions_for_collision
                          treats -1 as "the latest recorded collision".)
    Averages              Summary metrics (entropy, number of unique
                          expressions) recorded at each sampled collision.
    ContinuationMetadata  Links an experiment to the parent experiment it was
                          started from (multi-generation runs, extinction,
                          invasive species).
"""

import sqlite3
import json
import os
from datetime import datetime
from .config import DB_NAME

from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime, create_engine

def init_database():
    """Initialize the database and create tables if they don't exist.

    Safe to call every time the app starts: existing tables and data are kept,
    and columns added in later versions are added to older databases.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create Configurations table (one row per experiment).
    # probability_range holds the generator parameters as a JSON string.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Configurations (
        config_id INTEGER PRIMARY KEY AUTOINCREMENT,
        random_seed INTEGER,
        generator_type TEXT NOT NULL,
        total_collisions INTEGER NOT NULL,
        polling_frequency INTEGER NOT NULL,
        probability_range TEXT,
        freevar_generation_probability REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        name TEXT
    )
    ''')
    
    # Check if name column exists, add it if it doesn't
    # This handles existing databases that were created before the name column was added
    try:
        cursor.execute("SELECT name FROM pragma_table_info('Configurations') WHERE name='name'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE Configurations ADD COLUMN name TEXT")
            cursor.execute("UPDATE Configurations SET name = 'Experiment ' || config_id WHERE name IS NULL")
    except Exception as e:
        print(f"Error adding name column: {e}")
    
    # Create Experiment table: one row per (experiment, collision, expression)
    # with how many copies of that expression were in the population
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Experiment (
        experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_id INTEGER,
        collision_number INTEGER NOT NULL,
        expression TEXT NOT NULL,
        count INTEGER NOT NULL,
        FOREIGN KEY (config_id) REFERENCES Configurations(config_id)
    )
    ''')
    
    # Create Averages table: entropy and unique-expression count per sampled collision
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Averages (
        average_id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_id INTEGER,
        collision_number INTEGER NOT NULL,
        entropy REAL,
        unique_expressions INTEGER,
        FOREIGN KEY (config_id) REFERENCES Configurations(config_id)
    )
    ''')

    # Track recursive experiments / continuations.
    # Each child experiment has at most one parent (child_config_id is the key).
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ContinuationMetadata (
        child_config_id INTEGER PRIMARY KEY,
        parent_config_id INTEGER NOT NULL,
        fraction_used REAL NOT NULL,
        reused_expression_count INTEGER NOT NULL DEFAULT 0,
        additional_expression_count INTEGER NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (child_config_id) REFERENCES Configurations(config_id),
        FOREIGN KEY (parent_config_id) REFERENCES Configurations(config_id)
    )
    ''')

    # Ensure reused_expression_count column exists for older databases
    try:
        cursor.execute("SELECT reused_expression_count FROM ContinuationMetadata LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE ContinuationMetadata ADD COLUMN reused_expression_count INTEGER NOT NULL DEFAULT 0")
    
    conn.commit()
    conn.close()
    
def save_configuration(random_seed, generator_type, total_collisions, polling_frequency, 
                      probability_range=None, freevar_generation_probability=None, name=None):
    """
    Save experiment configuration to the database.
    
    Args:
        random_seed (int): Random seed for reproducibility
        generator_type (str): Type of generator used (e.g., "Fontana")
        total_collisions (int): Total number of collisions to run
        polling_frequency (int): Frequency at which data is collected
        probability_range (str): JSON representation of probability ranges
        freevar_generation_probability (float): Probability of generating free variables
        name (str): User-specified name for the experiment (optional).
            If omitted, the experiment is named "Experiment <id>".

    Returns:
        int: The ID of the newly created configuration
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # If name is not provided, create a default one that will be updated after we know the ID
    default_name = name or "Unnamed Experiment"
    
    cursor.execute('''
    INSERT INTO Configurations 
    (random_seed, generator_type, total_collisions, polling_frequency, 
     probability_range, freevar_generation_probability, name)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        random_seed,
        generator_type,
        total_collisions,
        polling_frequency,
        probability_range,
        freevar_generation_probability,
        default_name
    ))
    
    config_id = cursor.lastrowid
    
    # If no name was provided, update with a default name that includes the ID
    if name is None:
        cursor.execute('''
        UPDATE Configurations
        SET name = ? 
        WHERE config_id = ?
        ''', (f"Experiment {config_id}", config_id))
    
    conn.commit()
    conn.close()
    
    return config_id


def save_experiment_bundle(
    random_seed,
    generator_type,
    total_collisions,
    polling_frequency,
    probability_range=None,
    freevar_generation_probability=None,
    name=None,
    population_rows=(),
    averages_rows=(),
    continuation_metadata=None,
    configuration_update=None,
):
    """Save one experiment and all associated rows in a single transaction.

    ``population_rows`` contains ``(collision_number, expression, count)``
    tuples and ``averages_rows`` contains ``(collision_number, entropy,
    unique_expressions)`` tuples. Optional continuation metadata is a tuple
    of ``(parent_config_id, fraction_used, reused_count, additional_count)``.
    ``configuration_update`` may be a mapping of configuration column names
    to values, used by imports that replace metadata after restoring data.
    """
    conn = sqlite3.connect(DB_NAME)
    try:
        cursor = conn.cursor()
        experiment_name = name if isinstance(name, str) and name else "Unnamed Experiment"
        cursor.execute(
            """INSERT INTO Configurations
            (random_seed, generator_type, total_collisions, polling_frequency,
             probability_range, freevar_generation_probability, name)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (random_seed, generator_type, total_collisions, polling_frequency,
             probability_range, freevar_generation_probability, experiment_name),
        )
        config_id = cursor.lastrowid
        if callable(name):
            cursor.execute(
                "UPDATE Configurations SET name = ? WHERE config_id = ?",
                (name(config_id), config_id),
            )
        elif name is None:
            cursor.execute(
                "UPDATE Configurations SET name = ? WHERE config_id = ?",
                (f"Experiment {config_id}", config_id),
            )

        cursor.executemany(
            """INSERT INTO Experiment
            (config_id, collision_number, expression, count)
            VALUES (?, ?, ?, ?)""",
            ((config_id, collision, expression, count)
             for collision, expression, count in population_rows),
        )
        cursor.executemany(
            """INSERT INTO Averages
            (config_id, collision_number, entropy, unique_expressions)
            VALUES (?, ?, ?, ?)""",
            ((config_id, collision, entropy, unique)
             for collision, entropy, unique in averages_rows),
        )

        if continuation_metadata is not None:
            parent_id, fraction, reused_count, additional_count = continuation_metadata
            cursor.execute(
                """INSERT OR REPLACE INTO ContinuationMetadata
                (child_config_id, parent_config_id, fraction_used,
                 reused_expression_count, additional_expression_count)
                VALUES (?, ?, ?, ?, ?)""",
                (config_id, parent_id, fraction, reused_count, additional_count),
            )

        if configuration_update:
            allowed_columns = {"probability_range", "freevar_generation_probability", "name"}
            if not set(configuration_update).issubset(allowed_columns):
                raise ValueError("Unsupported configuration update column")
            assignments = ", ".join(f"{column} = ?" for column in configuration_update)
            cursor.execute(
                f"UPDATE Configurations SET {assignments} WHERE config_id = ?",
                (*configuration_update.values(), config_id),
            )

        conn.commit()
        return config_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_experiment_name(config_id, new_name):
    """
    Update the name of an existing experiment.
    
    Args:
        config_id (int): ID of the experiment to update
        new_name (str): New name for the experiment
        
    Returns:
        bool: True if successful, False otherwise (including when no
            experiment has that ID)
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute('''
        UPDATE Configurations
        SET name = ?
        WHERE config_id = ?
        ''', (new_name, config_id))
        
        conn.commit()
        success = cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating experiment name: {e}")
        conn.rollback()
        success = False
    
    conn.close()
    return success

def save_experiment_state(config_id, collision_number, expression, count):
    """
    Save experiment state to the database.

    Stores how many copies of one expression were in the population at one
    collision. Call it once per distinct expression.

    Args:
        config_id (int): ID of the configuration
        collision_number (int): Current collision number
            (0 = initial population)
        expression (str): The lambda expression
        count (int): Count/frequency of this expression
    
    Returns:
        int: The ID of the newly created experiment state entry
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO Experiment 
    (config_id, collision_number, expression, count)
    VALUES (?, ?, ?, ?)
    ''', (
        config_id,
        collision_number,
        expression,
        count
    ))
    
    experiment_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return experiment_id

def save_averages(config_id, collision_number, entropy, unique_expressions):
    """
    Save averages/metrics to the database.

    Called once per sampled collision; these rows are what the entropy and
    unique-expression plots are drawn from.

    Args:
        config_id (int): ID of the configuration
        collision_number (int): Collision number
        entropy (float): Entropy value (how evenly spread the population is
            across different expressions)
        unique_expressions (int): Count of unique expressions
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO Averages 
    (config_id, collision_number, entropy, unique_expressions)
    VALUES (?, ?, ?, ?)
    ''', (
        config_id,
        collision_number,
        entropy,
        unique_expressions
    ))

    conn.commit()
    conn.close()


def save_continuation_metadata(child_config_id, parent_config_id, fraction_used, reused_expression_count, additional_expression_count):
    """Record metadata for a continuation experiment.

    Links a child experiment to the parent it was started from. If the child
    already has a record, it is replaced.

    Args:
        child_config_id (int): The new experiment.
        parent_config_id (int): The experiment it continued from.
        fraction_used (float): Fraction of the parent's population carried
            over (0.0 to 1.0).
        reused_expression_count (int): Number of expressions taken from the
            parent.
        additional_expression_count (int): Number of new expressions added
            on top (e.g. invasive species copies).
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO ContinuationMetadata
        (child_config_id, parent_config_id, fraction_used, reused_expression_count, additional_expression_count)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        child_config_id,
        parent_config_id,
        fraction_used,
        reused_expression_count,
        additional_expression_count
    ))

    conn.commit()
    conn.close()


def get_continuation_metadata(child_config_id):
    """Fetch continuation metadata for a given experiment.

    Args:
        child_config_id (int): The experiment to look up.

    Returns:
        dict or None: Keys parent_config_id, fraction_used,
        reused_expression_count, additional_expression_count, created_at.
        None if the experiment was not continued from another one.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT parent_config_id, fraction_used, reused_expression_count, additional_expression_count, created_at
        FROM ContinuationMetadata
        WHERE child_config_id = ?
    ''', (child_config_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        'parent_config_id': row[0],
        'fraction_used': row[1],
        'reused_expression_count': row[2],
        'additional_expression_count': row[3],
        'created_at': row[4]
    }


def get_last_config_id():
    """Get the ID of the most recently added configuration.

    Returns:
        int: The highest config_id, or 1 if the table is empty.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT MAX(config_id) FROM Configurations")
    last_id = cursor.fetchone()[0]
    
    conn.close()
    return last_id if last_id else 1  # Default to 1 if no configurations exist

def get_experiment_configs():
    """Get all experiment configurations.

    Returns:
        list[dict]: One dict per experiment, newest first. Keys: config_id,
        random_seed, generator_type, total_collisions, polling_frequency,
        timestamp, probability_range, freevar_generation_probability, name.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT config_id, random_seed, generator_type, total_collisions, polling_frequency, timestamp, probability_range, freevar_generation_probability, name
    FROM Configurations
    ORDER BY timestamp DESC
    ''')

    configs = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return configs

def get_experiment_data(config_id):
    """Get experiment data for a specific configuration.

    Args:
        config_id (int): The experiment to look up.

    Returns:
        dict: {"config": <all Configurations columns as a dict>,
               "averages": [{"collision_number", "entropy",
                             "unique_expressions"}, ...] in collision order}
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get configuration details
    cursor.execute('''
    SELECT * FROM Configurations WHERE config_id = ?
    ''', (config_id,))
    config = dict(cursor.fetchone())
    
    # Get averages data
    cursor.execute('''
    SELECT collision_number, entropy, unique_expressions
    FROM Averages
    WHERE config_id = ?
    ORDER BY collision_number
    ''', (config_id,))
    averages = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        'config': config,
        'averages': averages
    }

def get_experiment_expressions(config_id, collision_number):
    """Get expressions for a specific configuration and collision number.

    Args:
        config_id (int): The experiment.
        collision_number (int): Which collision (0 = initial). Unlike
            db_utils.get_expressions_for_collision, -1 is not treated as
            "latest" here; it only matches rows actually saved as -1.

    Returns:
        list[dict]: [{"expression", "count"}, ...], most common first.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT expression, count
    FROM Experiment
    WHERE config_id = ? AND collision_number = ?
    ORDER BY count DESC
    ''', (config_id, collision_number))
    
    expressions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return expressions


def delete_experiment(config_id):
    """Remove an experiment and all associated records from the database.

    Also removes lineage links where this experiment is the parent, so its
    child experiments are kept but no longer show where they came from.

    Args:
        config_id (int): The experiment to delete.

    Returns:
        bool: True if deleted, False if an error occurred (nothing is
        deleted in that case).
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        cursor.execute('DELETE FROM ContinuationMetadata WHERE child_config_id = ? OR parent_config_id = ?', (config_id, config_id))
        cursor.execute('DELETE FROM Experiment WHERE config_id = ?', (config_id,))
        cursor.execute('DELETE FROM Averages WHERE config_id = ?', (config_id,))
        cursor.execute('DELETE FROM Configurations WHERE config_id = ?', (config_id,))
        conn.commit()
        return True
    except Exception as exc:
        print(f"Error deleting experiment {config_id}: {exc}")
        conn.rollback()
        return False
    finally:
        conn.close()


def reset_database_counters():
    """Wipes the SQLite memory of old IDs so the next experiment starts at 1.

    Only makes sense after every experiment has been deleted; otherwise new
    IDs could collide with existing ones.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        # This clears the internal SQLite tracker for all tables
        cursor.execute("DELETE FROM sqlite_sequence")
        conn.commit()
    except sqlite3.OperationalError:
        # This happens if no AUTOINCREMENT tables have been used yet; safe to ignore
        pass
    finally:
        conn.close()


# --- SQLAlchemy session (not used by the functions above) ---
# Note: this connects to "alchemy_experiments.db" relative to the folder the
# app is started from, which may not be the same file as DB_NAME.
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///alchemy_experiments.db", echo=False)

Session = sessionmaker(bind=engine)
db_session = Session()

__all__ = ["Base", "ExperimentConfiguration", "ExperimentResult", "db_session"]
from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
