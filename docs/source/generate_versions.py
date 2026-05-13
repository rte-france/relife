import json
import os
import subprocess
from pathlib import Path

BASE_URL = os.environ.get("DOCS_BASE_URL", "https://opensource.rte-france.com/relife/")
OUTPUT = Path("docs/source/_static/versions.json")


def get_versions():
    out = subprocess.check_output(["python", "/tmp/select_versions.py"], text=True)
    versions = []
    for line in out.splitlines():
        tag, folder = line.split()
        versions.append(
            {
                "name": tag,
                "version": folder,
                "url": f"{BASE_URL}{folder}/",
            }
        )
    return versions


def main():
    versions = get_versions()
    entries = [
        {
            "name": "latest (dev)",
            "version": "latest",
            "url": f"{BASE_URL}latest/",
        }
    ] + versions

    # Most recent will be marked "stable"
    if versions:
        entries[1]["preferred"] = True

    OUTPUT.write_text(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
