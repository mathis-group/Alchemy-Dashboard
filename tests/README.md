# Simulation form checks

Run `python -m unittest discover -s tests -p test_simulation_form.py -v` with
Python 3.10+ and Node.js available on PATH. These UI checks need no test packages.

The Python tests parse the actual template to check form ownership, field names,
defaults, file encoding, and which controls belong to the fresh-only fieldset.
The Node test executes the actual inline script with narrow DOM, fetch, and
FormData doubles. It checks generator/parent switching, option population,
submission wiring, repeated fresh/seeded/fresh submissions, and empty/failed
parent lists.

These are template-contract and script-unit tests. They do not validate browser
rendering, native constraint validation, real FormData/file serialization, Jinja
inheritance, Flask routes, SQLite persistence, or the Rust engine.

To run against a separate template snapshot, set `SIMULATION_TEMPLATE` to its
absolute path. This supports regression and negative-control checks without
modifying the working template.


## Native adapter and route source checks

`python -m unittest discover -s tests -p test_native_adapter.py -v` imports the
real `alchemy` extension and the simulation module (including its existing pandas
dependency). It requires the recursive binding with `polling_mode="completed"`;
no engine stubs or automatic fallback are used. The default dashboard dependency
still tracks upstream main; use the explicit review override described below.
An older/missing extension
must fail, not count as a passing native run.

These adapter tests cover full population counts, zero/divisor/non-divisor cadence,
exact boundaries, explicit wipeout/refill, invalid input, reproducible seeds, and
BTree/Fontana JSON mapping. They do not launch Flask or write the database.

`python -m unittest discover -s tests -p test_native_route_contract.py -v` only
checks the route AST for one native call before persistence and validation shape.
It is not execution evidence for Flask, SQLite persistence, or uploaded-file parsing.
The new UI tests likewise retain their explicitly doubled DOM/network boundary.

Fresh generated starts refill using their selected generator. Parent/manual starts
with nonzero wipeout must supply an explicit refill expression (empty by default);
zero wipeout requires no choice. The form route stores the native configuration and
refill policy beside existing generator metadata. Existing perturbation endpoints
still use the old single-run adapter and are outside this form-path increment.

## HTTP and browser integration

Use the explicit [native review setup](REVIEW_NATIVE.md) while the native API is
unmerged. `python -m unittest discover -s tests -v` includes `test_native_http.py`:
real HTTP requests run the Flask route, native extension, and SQLite persistence.
The checks cover exact collision coordinates, full population counts, generation
lineage/refill, parent seed inheritance, and validation rejection without writes.

`python tests/browser_simulation.py -v` additionally requires Playwright/Chromium.
It renders the actual Jinja page, uploads a real JSON file through browser FormData,
switches between fresh and parent populations, checks required refill validation,
runs native wipeout generations, verifies result summaries, prevents overlapping
submits, and permits corrected resubmission after server rejection. No browser,
network, native engine, or database doubles are used in these integration checks.

The fixture copies application files byte for byte into a temporary directory with
a disposable database and uploads directory. Browser launch uses the Chromium
sandbox; `CHROMIUM_PATH` optionally selects an existing executable. Set
`BROWSER_ARTIFACTS` to retain screenshots, server output, database, and source hashes.
This is focused form-path coverage, not a full accessibility audit, browser matrix,
plotting-endpoint review, or proof of rollback after a database write failure.
