"""Real HTTP/Flask/native/SQLite checks; no engine or persistence doubles."""
import unittest
from urllib.parse import urlencode

from isolated_dashboard import IsolatedDashboard


IDENTITY = r"\x.x"
LEFT = r"\x.\y.x"


class NativeHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = IsolatedDashboard().__enter__()
        cls.addClassCleanup(cls.server.__exit__, None, None, None)

    def submit(self, **values):
        form = dict(generator_type="from_file", direct_input="\n".join([IDENTITY] * 4),
                    total_collisions=0, polling_frequency=2, random_seed=42,
                    num_generations=1, wipeout_percent=0)
        form.update(values)
        return self.server.request("/run_simulation_form", urlencode(form).encode(),
                                   {"Content-Type": "application/x-www-form-urlencoded"})

    def test_exact_completed_polls_and_population_counts_are_saved(self):
        status, result = self.submit(total_collisions=5, experiment_name="Exact polls")
        self.assertEqual(status, 200, result)
        config_id = result["config_id"]
        self.assertEqual(self.server.query(
            "SELECT collision_number, SUM(count) FROM Experiment WHERE config_id=? "
            "GROUP BY collision_number ORDER BY collision_number", (config_id,)),
            [(0, 4), (2, 4), (4, 4), (5, 4)])
        status, exported = self.server.request(f"/download_final_state/{config_id}")
        self.assertEqual(status, 200)
        self.assertEqual(exported["last_collision_number"], 5)
        self.assertEqual(sum(row["count"] for row in exported["final_state_counts"]), 4)

    def test_native_wipeout_creates_generation_lineage_and_refill_population(self):
        status, result = self.submit(num_generations=2, wipeout_percent=100,
                                     repopulation_expression=LEFT, experiment_name="Wipeout")
        self.assertEqual(status, 200, result)
        child = result["config_id"]
        parent, fraction, reused, added = self.server.query(
            "SELECT parent_config_id, fraction_used, reused_expression_count, "
            "additional_expression_count FROM ContinuationMetadata WHERE child_config_id=?",
            (child,))[0]
        self.assertEqual((fraction, reused, added), (0, 0, 4))
        _, first = self.server.request(f"/download_initial_state/{parent}")
        _, second = self.server.request(f"/download_initial_state/{child}")
        self.assertNotEqual(first["initial_expression_counts"], second["initial_expression_counts"])
        self.assertEqual(sum(row["count"] for row in second["initial_expression_counts"]), 4)

    def test_parent_supplies_full_population_and_seed_without_fresh_fields(self):
        status, parent = self.submit(random_seed=93)
        self.assertEqual(status, 200, parent)
        form = dict(recursive_parent_id=parent["config_id"], total_collisions=0,
                    polling_frequency=2, num_generations=1, wipeout_percent=0)
        status, child = self.server.request("/run_simulation_form", urlencode(form).encode(),
                                            {"Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(status, 200, child)
        _, exported = self.server.request(f"/download_initial_state/{child['config_id']}")
        self.assertEqual(exported["random_seed"], 93)
        self.assertEqual(sum(row["count"] for row in exported["initial_expression_counts"]), 4)
        self.assertEqual(exported["continuation"]["parent_config_id"], parent["config_id"])

    def test_invalid_requests_do_not_persist_any_generation(self):
        for values in ({"polling_frequency": 0}, {"num_generations": 0},
                       {"wipeout_percent": 101}, {"total_collisions": -1},
                       {"total_collisions": "not a number"}, {"recursive_parent_id": 999999},
                       {"num_generations": 2, "wipeout_percent": 25},
                       {"direct_input": "("}):
            with self.subTest(values=values):
                before = self.server.query("SELECT COUNT(*) FROM Configurations")
                status, result = self.submit(**values)
                self.assertEqual(status, 400, result)
                self.assertEqual(result["status"], "error")
                self.assertEqual(self.server.query("SELECT COUNT(*) FROM Configurations"), before)


if __name__ == "__main__":
    unittest.main()
