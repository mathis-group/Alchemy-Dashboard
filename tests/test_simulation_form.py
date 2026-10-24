"""Template contracts, not a Flask/native or browser integration test.

Run with: python -m unittest discover -s tests -v
Uses only Python's standard library and Node.js for the script unit tests.
"""

import json
import os
from pathlib import Path
import subprocess
import unittest
from html.parser import HTMLParser


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(os.environ.get(
    "SIMULATION_TEMPLATE", ROOT / "alchemy_dashboard/templates/simulation.html"
))


class Elements(HTMLParser):
    """Read control ownership and defaults from the actual HTML template."""

    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.stack = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        element = {"tag": tag, "attrs": dict(attrs), "parents": list(self.stack)}
        self.elements.append(element)
        if tag not in {"input", "br", "hr", "img", "meta", "link"}:
            self.stack.append(element)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if self.stack and self.stack[-1]["tag"] == tag:
            self.stack.pop()

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["tag"] == tag:
                del self.stack[index:]
                break


class SimulationFormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TEMPLATE.read_text()
        cls.elements = Elements(cls.source).elements

    def controls(self, name):
        return [e for e in self.elements if e["attrs"].get("name") == name]

    def test_one_multipart_form_owns_every_payload_control(self):
        forms = [e for e in self.elements if e["tag"] == "form"]
        self.assertEqual(len(forms), 1)
        form = forms[0]
        self.assertEqual(form["attrs"]["action"], "/run_simulation_form")
        self.assertEqual(form["attrs"]["method"].upper(), "POST")
        self.assertEqual(form["attrs"]["enctype"], "multipart/form-data")
        names = [e["attrs"]["name"] for e in self.elements if "name" in e["attrs"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), {
            "experiment_name", "total_collisions", "polling_frequency",
            "num_generations", "recursive_parent_id", "random_seed",
            "wipeout_percent", "repopulation_expression",
            "generator_type", "btree_size", "freevar_probability", "max_free_vars",
            "standardization", "num_expressions", "abs_low", "abs_high",
            "app_low", "app_high", "min_depth", "max_depth",
            "fontana_expression_count", "expressions_file", "direct_input",
        })
        for element in self.elements:
            if "name" in element["attrs"]:
                self.assertIn(form, element["parents"])

    def test_generations_are_editable_and_default_to_one(self):
        generations = self.controls("num_generations")
        self.assertEqual(len(generations), 1)
        self.assertEqual(generations[0]["attrs"]["type"], "number")
        self.assertEqual(generations[0]["attrs"]["value"], "1")
        self.assertEqual(generations[0]["attrs"]["min"], "1")

    def test_previous_experiment_is_optional_with_a_fresh_choice(self):
        parent = self.controls("recursive_parent_id")[0]
        self.assertNotIn("required", parent["attrs"])
        blank_options = [e for e in self.elements if e["tag"] == "option"
                         and parent in e["parents"] and e["attrs"].get("value") == ""]
        self.assertEqual(len(blank_options), 1)

    def test_fresh_only_inputs_share_a_disableable_group(self):
        fieldsets = [e for e in self.elements if e["tag"] == "fieldset"]
        self.assertEqual(len(fieldsets), 1)
        group = fieldsets[0]
        self.assertNotIn("disabled", group["attrs"])
        for name in ("generator_type", "random_seed", "expressions_file", "direct_input",
                     "btree_size", "abs_low", "fontana_expression_count"):
            self.assertIn(group, self.controls(name)[0]["parents"])
        for name in ("num_generations", "recursive_parent_id", "total_collisions",
                     "polling_frequency", "experiment_name", "wipeout_percent", "repopulation_expression"):
            self.assertNotIn(group, self.controls(name)[0]["parents"])

    def test_original_generators_and_file_controls_remain_available(self):
        generator = self.controls("generator_type")[0]
        options = [e["attrs"]["value"] for e in self.elements
                   if e["tag"] == "option" and generator in e["parents"]]
        self.assertEqual(options, ["from_file", "BTree", "Fontana"])
        self.assertEqual(self.controls("expressions_file")[0]["attrs"]["type"], "file")
        self.assertEqual(self.controls("direct_input")[0]["tag"], "textarea")
        for name, value in {"random_seed": "42", "total_collisions": "1000",
                            "polling_frequency": "10", "btree_size": "5",
                            "fontana_expression_count": "10"}.items():
            self.assertEqual(self.controls(name)[0]["attrs"]["value"], value)

    def test_wipeout_and_explicit_refill_have_safe_defaults(self):
        wipeout = self.controls("wipeout_percent")
        self.assertEqual(len(wipeout), 1)
        self.assertEqual(wipeout[0]["attrs"]["value"], "0")
        self.assertEqual(wipeout[0]["attrs"]["min"], "0")
        self.assertEqual(wipeout[0]["attrs"]["max"], "100")
        refill = self.controls("repopulation_expression")
        self.assertEqual(len(refill), 1)
        self.assertEqual(refill[0]["attrs"].get("value", ""), "")
        self.assertIn("placeholder", refill[0]["attrs"])

    def test_script_contracts(self):
        # This runs the real inline script in Node with explicit DOM/fetch doubles.
        # It does not prove DOM rendering, browser FormData, or server behavior.
        result = subprocess.run(
            ["node", str(ROOT / "tests/simulation_form_script.cjs"), str(TEMPLATE)],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["passed"], 8)


if __name__ == "__main__":
    unittest.main()
