"""Run byte-identical application files with a disposable database and uploads."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SERVER = """
from pathlib import Path
from werkzeug.serving import make_server
from alchemy_dashboard.main import app
server = make_server('127.0.0.1', 0, app, threaded=True)
Path('port').write_text(str(server.server_port))
server.serve_forever()
"""


class IsolatedDashboard:
    def __enter__(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="alchemy-dashboard-test-")
        self.root = Path(self.temporary.name)
        source = ROOT / "alchemy_dashboard"
        destination = self.root / source.name
        shutil.copytree(source, destination,
                        ignore=shutil.ignore_patterns("__pycache__", "*.db"))
        self.source_hashes = {}
        for path in destination.rglob("*"):
            if path.is_file():
                relative = path.relative_to(destination)
                assert path.read_bytes() == (source / relative).read_bytes()
                self.source_hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.log = (self.root / "server.log").open("w+")
        self.process = subprocess.Popen([sys.executable, "-u", "-c", SERVER],
                                        cwd=self.root, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 30
        while not (self.root / "port").exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                self.log.seek(0)
                failure = self.log.read()
                self.__exit__(None, None, None)
                raise RuntimeError("Dashboard did not start:\n" + failure)
            time.sleep(.05)
        self.url = "http://127.0.0.1:" + (self.root / "port").read_text()
        return self

    def __exit__(self, *_):
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.log.close()
        self.temporary.cleanup()

    def query(self, sql, parameters=()):
        with sqlite3.connect(self.root / "alchemy_experiments.db") as connection:
            return connection.execute(sql, parameters).fetchall()

    def request(self, path, data=None, headers=None):
        request = Request(self.url + path, data=data, headers=headers or {})
        try:
            response = urlopen(request, timeout=30)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def save_evidence(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / "server.log", destination / "server.log")
        shutil.copyfile(self.root / "alchemy_experiments.db", destination / "experiments.db")
        (destination / "application-sha256.json").write_text(json.dumps(self.source_hashes, indent=2))
