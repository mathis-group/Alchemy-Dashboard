import sqlite3

import pytest

from alchemy_dashboard import models


class ConnectionTracker:
    def __init__(self, connection, fail_on_executemany=False):
        self.connection = connection
        self.fail_on_executemany = fail_on_executemany
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.executemany_calls = []

    def cursor(self):
        return CursorTracker(self, self.connection.cursor())

    def commit(self):
        self.commits += 1
        return self.connection.commit()

    def rollback(self):
        self.rollbacks += 1
        return self.connection.rollback()

    def close(self):
        self.closed = True
        return self.connection.close()


class CursorTracker:
    def __init__(self, tracker, cursor):
        self.tracker = tracker
        self.cursor = cursor

    @property
    def lastrowid(self):
        return self.cursor.lastrowid

    def execute(self, *args, **kwargs):
        return self.cursor.execute(*args, **kwargs)

    def executemany(self, sql, rows):
        rows = list(rows)
        self.tracker.executemany_calls.append((sql, rows))
        result = self.cursor.executemany(sql, rows)
        if self.tracker.fail_on_executemany:
            raise RuntimeError("injected batch failure")
        return result


@pytest.fixture
def database(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.sqlite"
    monkeypatch.setattr(models, "DB_NAME", str(db_path))
    models.init_database()
    return db_path


def _bundle(**overrides):
    values = {
        "random_seed": 7,
        "generator_type": "test",
        "total_collisions": 20,
        "polling_frequency": 5,
        "population_rows": [(0, "x", 3), (5, "y", 2)],
        "averages_rows": [(0, 0.1, 2), (5, 0.2, 3)],
        "continuation_metadata": (1, 1.0, 3, 0),
        "configuration_update": {"probability_range": '{"imported": true}'},
    }
    values.update(overrides)
    return values


def test_experiment_uses_one_connection_batches_and_commits_once(
    database, monkeypatch
):
    real_connect = sqlite3.connect
    connections = []

    def tracked_connect(*args, **kwargs):
        tracker = ConnectionTracker(real_connect(*args, **kwargs))
        connections.append(tracker)
        return tracker

    monkeypatch.setattr(models.sqlite3, "connect", tracked_connect)
    config_id = models.save_experiment_bundle(**_bundle())

    assert len(connections) == 1
    assert connections[0].commits == 1
    assert connections[0].rollbacks == 0
    assert connections[0].closed
    assert [len(rows) for _, rows in connections[0].executemany_calls] == [2, 2]

    with real_connect(database) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM Experiment WHERE config_id = ?", (config_id,)
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM Averages WHERE config_id = ?", (config_id,)
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM ContinuationMetadata WHERE child_config_id = ?",
            (config_id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT probability_range FROM Configurations WHERE config_id = ?",
            (config_id,),
        ).fetchone()[0] == '{"imported": true}'


def test_exception_rolls_back_and_closes_connection(database, monkeypatch):
    real_connect = sqlite3.connect
    connections = []

    def failing_connect(*args, **kwargs):
        tracker = ConnectionTracker(
            real_connect(*args, **kwargs), fail_on_executemany=True
        )
        connections.append(tracker)
        return tracker

    monkeypatch.setattr(models.sqlite3, "connect", failing_connect)
    with pytest.raises(RuntimeError, match="injected batch failure"):
        models.save_experiment_bundle(**_bundle())

    assert len(connections) == 1
    assert connections[0].commits == 0
    assert connections[0].rollbacks == 1
    assert connections[0].closed

    with real_connect(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM Configurations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM Experiment").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM Averages").fetchone()[0] == 0


def test_imported_configuration_keeps_import_name_out_of_freevar_probability(database):
    original_name = "Imported source"
    config_id = models.save_experiment_bundle(
        random_seed=7,
        generator_type="test",
        total_collisions=20,
        polling_frequency=5,
        probability_range='{"freevar_probability": 0.25}',
        name=f"{original_name} (Imported)",
        configuration_update={
            "probability_range": '{"original_name": "Imported source", "imported": true}'
        },
    )

    with sqlite3.connect(database) as conn:
        imported_config = conn.execute(
            """SELECT name, freevar_generation_probability, probability_range
               FROM Configurations WHERE config_id = ?""",
            (config_id,),
        ).fetchone()

    assert imported_config == (
        "Imported source (Imported)",
        None,
        '{"original_name": "Imported source", "imported": true}',
    )
