"""Optional real-browser checks. Run directly; requires Playwright and Chromium."""
import json
import os
from pathlib import Path
import unittest

from playwright.sync_api import expect, sync_playwright

from isolated_dashboard import IsolatedDashboard


class BrowserSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = IsolatedDashboard().__enter__()
        cls.addClassCleanup(cls.server.__exit__, None, None, None)
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        options = {"headless": True, "chromium_sandbox": True}
        if os.environ.get("CHROMIUM_PATH"):
            options["executable_path"] = os.environ["CHROMIUM_PATH"]
        cls.browser = cls.playwright.chromium.launch(**options)
        cls.addClassCleanup(cls.browser.close)
        cls.artifacts = Path(os.environ["BROWSER_ARTIFACTS"]) if os.environ.get("BROWSER_ARTIFACTS") else None
        if cls.artifacts:
            cls.artifacts.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if cls.artifacts:
            cls.server.save_evidence(cls.artifacts)

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 1000})
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(self.server.url + "/simulation", wait_until="domcontentloaded")
        expect(self.page.locator("#parent-load-status")).not_to_have_text("Loading previous experiments...")

    def tearDown(self):
        if self.artifacts:
            self.page.screenshot(path=str(self.artifacts / (self._testMethodName + ".png")), full_page=True)
        self.assertEqual(self.errors, [])

    def manual(self):
        self.page.locator("#generator-type").select_option("from_file")
        self.page.locator('[name="direct_input"]').fill("\n".join([r"\x.x"] * 4))
        self.page.locator('[name="total_collisions"]').fill("0")

    def submit(self):
        with self.page.expect_response(lambda response: response.url.endswith("/run_simulation_form")) as pending:
            self.page.locator('button[type="submit"]').click()
        response = pending.value
        self.assertEqual(response.status, 200, response.text())
        expect(self.page.locator("#success-message")).to_be_visible()
        return response.json()

    def test_defaults_real_file_upload_and_parent_switching(self):
        self.assertEqual(self.page.locator("form").count(), 1)
        expect(self.page.locator("#num-generations")).to_have_value("1")
        expect(self.page.locator("#refill-options")).to_be_hidden()
        self.manual()
        self.page.locator('[name="direct_input"]').fill("")
        self.page.locator('[name="expressions_file"]').set_input_files({
            "name": "population.json", "mimeType": "application/json",
            "buffer": json.dumps([r"\x.x"] * 4).encode(),
        })
        result = self.submit()
        _, initial = self.server.request(f"/download_initial_state/{result['config_id']}")
        self.assertEqual(sum(row["count"] for row in initial["initial_expression_counts"]), 4)

        self.page.reload(wait_until="domcontentloaded")
        expect(self.page.locator(f'#recursive-parent-id option[value="{result["config_id"]}"]')).to_have_count(1)
        self.page.locator("#random-seed").fill("73")
        self.page.locator("#recursive-parent-id").select_option(str(result["config_id"]))
        expect(self.page.locator("#fresh-options")).to_be_hidden()
        fields = self.page.locator("form").evaluate("form => [...new FormData(form).keys()]")
        self.assertNotIn("generator_type", fields)
        self.assertNotIn("random_seed", fields)
        self.page.locator('[name="total_collisions"]').fill("0")
        child = self.submit()
        _, initial = self.server.request(f"/download_initial_state/{child['config_id']}")
        self.assertEqual(initial["random_seed"], 42)
        self.assertEqual(initial["continuation"]["parent_config_id"], result["config_id"])
        self.page.locator("#recursive-parent-id").select_option("")
        expect(self.page.locator("#fresh-options")).to_be_visible()
        expect(self.page.locator("#random-seed")).to_have_value("73")

    def test_manual_wipeout_uses_native_generations_and_browser_validation(self):
        self.manual()
        self.page.locator("#num-generations").fill("2")
        self.page.locator("#wipeout-percent").fill("100")
        expect(self.page.locator("#refill-expression")).to_be_visible()
        self.assertFalse(self.page.locator("form").evaluate("form => form.checkValidity()"))
        self.page.locator("#refill-expression").fill(r"\x.\y.x")
        result = self.submit()
        expect(self.page.locator("#summary-generator")).to_have_text("Supplied population")
        expect(self.page.locator("#summary-collisions")).to_have_text("0")
        expect(self.page.locator("#summary-expressions")).to_have_text("4")
        self.assertEqual(self.server.query(
            "SELECT fraction_used, reused_expression_count, additional_expression_count "
            "FROM ContinuationMetadata WHERE child_config_id=?", (result["config_id"],)), [(0, 0, 4)])

    def test_same_page_overlapping_submits_persist_only_one_experiment(self):
        self.manual()
        before = self.server.query("SELECT COUNT(*) FROM Configurations")[0][0]
        self.page.locator("form").evaluate("form => { form.requestSubmit(); form.requestSubmit(); }")
        expect(self.page.locator("#success-message")).to_be_visible()
        # Drain same-page requests before reading persisted results.
        self.page.wait_for_load_state("networkidle")
        after = self.server.query("SELECT COUNT(*) FROM Configurations")[0][0]
        self.assertEqual(after - before, 1)
        expect(self.page.locator('button[type="submit"]')).to_be_enabled()
        self.submit()
        self.assertEqual(self.server.query("SELECT COUNT(*) FROM Configurations")[0][0], after + 1)

    def test_server_rejection_allows_corrected_resubmission(self):
        self.manual()
        self.page.on("dialog", lambda dialog: dialog.accept())
        self.page.locator('[name="direct_input"]').fill("(")
        with self.page.expect_response(lambda response: response.url.endswith("/run_simulation_form")) as pending:
            self.page.locator('button[type="submit"]').click()
        self.assertEqual(pending.value.status, 400)
        expect(self.page.locator('button[type="submit"]')).to_be_enabled()
        self.manual()
        self.submit()


if __name__ == "__main__":
    unittest.main()
