"""Source-level route contract only; not Flask or database execution."""
import ast
from pathlib import Path
import unittest


class NativeRouteSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (Path(__file__).resolve().parents[1] / "alchemy_dashboard/main.py").read_text()
        cls.route = next(node for node in ast.parse(source).body
                         if isinstance(node, ast.FunctionDef) and node.name == "run_simulation_form")

    def calls(self, name, scope=None):
        return [node for node in ast.walk(scope or self.route)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name]

    def test_one_native_call_outside_python_loops_precedes_persistence(self):
        calls = self.calls("run_recursive_experiment")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.calls("run_experiment"), [])
        for loop in (node for node in ast.walk(self.route) if isinstance(node, (ast.For, ast.While))):
            self.assertEqual(self.calls("run_recursive_experiment", loop), [])
        self.assertLess(calls[0].lineno, min(node.lineno for node in self.calls("save_configuration")))

    def test_lineage_fraction_matches_actual_integer_rounded_retention(self):
        # Execute only the production arithmetic AST, never the route or DB calls.
        call = self.calls("save_continuation_metadata")[0]
        assignments = sorted(
            (node for node in ast.walk(self.route) if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id in ("removed", "fraction_used")
                     for target in node.targets)), key=lambda node: node.lineno,
        )
        cases = ((3, 25, 2, (1.0, 3, 0)), (4, 25, 2, (.75, 3, 1)),
                 (4, 100, 2, (0.0, 0, 4)), (3, 25, 1, (1.0, 3, 0)))
        for population, percent, generation, expected in cases:
            with self.subTest(population=population, percent=percent, generation=generation):
                values = dict(previous_population_size=population,
                              wipeout_percent=percent, gen_idx=generation)
                for assignment in assignments:
                    value = eval(compile(ast.Expression(assignment.value), "route arithmetic", "eval"),
                                 {"__builtins__": {}}, values)
                    values[assignment.targets[0].id] = value
                actual = tuple(eval(compile(ast.Expression(arg), "lineage argument", "eval"),
                                    {"__builtins__": {}}, values) for arg in call.args[2:5])
                self.assertEqual(actual, expected)

    def test_route_has_validation_error_response_and_no_previous_poll_as_final(self):
        handlers = [node for node in ast.walk(self.route) if isinstance(node, ast.ExceptHandler)]
        self.assertTrue(any(isinstance(node.type, ast.Name) and node.type.id == "ValueError" for node in handlers))
        for node in ast.walk(self.route):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "metrics":
                self.assertFalse(isinstance(node.slice, ast.UnaryOp), "last poll must not stand in for exact final state")


if __name__ == "__main__":
    unittest.main()
