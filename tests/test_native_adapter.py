"""Dashboard mapping against the real native engine; no Flask or engine stubs."""
import unittest
from collections import Counter

import alchemy
from alchemy_dashboard import simulation


IDENTITY = r"\x.x"
LEFT = r"\x.\y.x"


def supplied(**overrides):
    config = dict(generator_type="from_file", expressions=[IDENTITY] * 4,
                  total_collisions=0, polling_frequency=2, random_seed=42,
                  num_generations=1, wipeout_percent=0)
    config.update(overrides)
    return config


def canonical(expression):
    soup = alchemy.PySoup()
    soup.perturb([expression])
    return soup.expressions()[0]


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(simulation, "run_recursive_experiment"),
                        "Dashboard needs a real native recursive adapter")

    def run_native(self, config):
        return simulation.run_recursive_experiment(config)

    def test_full_supplied_population_is_not_truncated_and_zero_run_has_one_snapshot(self):
        expressions = [r"\x." + " ".join(["x"] * n) for n in range(1, 21)] * 2
        result = self.run_native(supplied(expressions=expressions))[0]
        self.assertEqual(len(result["initial_expressions"]), 40)
        self.assertEqual(len(set(result["initial_expressions"])), 20)
        self.assertEqual([m["collision_number"] for m in result["metrics"]], [0])
        self.assertEqual(Counter(result["final_expressions"]), Counter(result["initial_expressions"]))

    def test_completed_cadence_and_final_population_are_persistable(self):
        generations = self.run_native(supplied(total_collisions=5, num_generations=2))
        self.assertEqual(len(generations), 2)
        for result in generations:
            self.assertEqual([m["collision_number"] for m in result["metrics"]], [0, 2, 4, 5])
            self.assertEqual(result["metrics"][0]["expressions"], result["initial_expressions"])
            self.assertEqual(result["metrics"][-1]["expressions"], result["final_expressions"])
            self.assertTrue(all(m["unique_expressions"] == 1 and m["entropy"] == 0 for m in result["metrics"]))

    def test_manual_wipeout_uses_explicit_custom_expression_and_retains_policy(self):
        first, second = self.run_native(supplied(num_generations=2, wipeout_percent=100,
                                                repopulation_expression=LEFT))
        self.assertEqual(Counter(first["final_expressions"]), Counter({canonical(IDENTITY): 4}))
        self.assertEqual(Counter(second["initial_expressions"]), Counter({canonical(LEFT): 4}))
        policy = second["native_config"]["recursive_config"]
        self.assertEqual(policy["refill_type"], "CustomExpression")
        self.assertEqual(policy["wipeout_percent"], 100)
        self.assertEqual(policy["repopulation_expression"], canonical(LEFT))

    def test_manual_wipeout_requires_valid_explicit_expression(self):
        for expression in (None, "", "("):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                self.run_native(supplied(num_generations=2, wipeout_percent=25,
                                         repopulation_expression=expression))

    def test_zero_wipeout_needs_no_refill_choice(self):
        self.assertEqual(len(self.run_native(supplied(num_generations=2, repopulation_expression=""))), 2)

    def test_btree_config_and_both_seeds_map_without_using_seed_constructor(self):
        config = supplied(generator_type="BTree", expressions=None, size=3,
                          freevar_probability=0, max_free_vars=2, standardization="prefix",
                          num_expressions=4, num_generations=2, wipeout_percent=25)
        first = self.run_native(config)
        second = self.run_native(config)
        native = first[0]["native_config"]
        generator = native["expression"]["UntypedLambda"]["generator"]["BTree"]
        self.assertEqual(generator["n_max_free_vars"], 2)
        self.assertEqual(generator["standardization"], "Prefix")
        self.assertEqual(generator["seed"], f"{42:064x}")
        self.assertEqual(native["soup_config"]["seed"], generator["seed"])
        self.assertEqual(native["recursive_config"]["refill_type"], "ConfigGenerator")
        self.assertEqual([r["initial_expressions"] for r in first], [r["initial_expressions"] for r in second])

    def test_unsafe_btree_inputs_are_rejected_before_native_generation(self):
        config = supplied(generator_type="BTree", expressions=None, size=3,
                          freevar_probability=0, max_free_vars=2,
                          standardization="prefix", num_expressions=4)
        for field, value in (("size", 0), ("max_free_vars", 0),
                             ("freevar_probability", -0.1), ("freevar_probability", 1.1),
                             ("freevar_probability", float("nan")),
                             ("freevar_probability", float("inf"))):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.run_native({**config, field: value})

    def test_fontana_ranges_depth_and_count_map_to_native_config(self):
        result = self.run_native(supplied(generator_type="Fontana", expressions=None,
                                         abs_low=.2, abs_high=.4, app_low=.2, app_high=.3,
                                         min_depth=1, max_depth=3, initial_expression_count=4))[0]
        generator = result["native_config"]["expression"]["UntypedLambda"]["generator"]["Fontana"]
        self.assertEqual(generator["abstraction_prob_range"], [.2, .4])
        self.assertEqual(generator["application_prob_range"], [.2, .3])
        self.assertEqual((generator["min_depth"], generator["max_depth"]), (1, 3))
        self.assertEqual(len(result["initial_expressions"]), 4)

    def test_invalid_supplied_expressions_are_reported_before_native_run(self):
        result = self.run_native(supplied(expressions=["("] + [IDENTITY] * 4))[0]
        self.assertEqual(result["dropped_expression_count"], 1)
        self.assertEqual(len(result["initial_expressions"]), 4)
        with self.assertRaises(ValueError):
            self.run_native(supplied(expressions=["(", IDENTITY]))

    def test_invalid_run_settings_are_rejected(self):
        for field, value in (("polling_frequency", 0), ("num_generations", 0),
                             ("wipeout_percent", 101), ("total_collisions", -1),
                             ("generator_type", "unknown")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_native(supplied(**{field: value}))


if __name__ == "__main__":
    unittest.main()
