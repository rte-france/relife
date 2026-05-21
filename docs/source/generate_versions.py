import json
import os
import subprocess
from pathlib import Path

# Environment variable defined in your Github repository to build docs on your fork
BASE_URL = os.environ.get("DOCS_BASE_URL", "https://opensource.rte-france.com/relife/")

# Output location of versions.json file (mandatory for the versions dropdown)
OUTPUT = Path("docs/source/_static/versions.json")


def get_versions():
    """
    Builds the Python dictionary with all tag versions.
    """
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

    # Add dev version (docs build on main)
    entries = [
        {
            "name": "latest (dev)",
            "version": "latest",
            "url": f"{BASE_URL}latest/",
        }
    ] + versions

    OUTPUT.write_text(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
