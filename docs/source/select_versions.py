"""
Select the last 3 major.minor versions for the docs.
If multiple options: select the latest tag.
This function only print results (for bash or other python scripts)
"""

import re
import subprocess
from collections import defaultdict

# Change this parameter to add more older versions
N_VERSIONS = 3

def list_tags():
    out = subprocess.check_output(["git", "tag", "-l", "v*"], text=True)
    return [t.strip() for t in out.splitlines() if t.strip()]


def main():
    families = defaultdict(list)  # (major, minor) -> [(patch, tag), ...]
    for tag in list_tags():
        m = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$").match(tag)
        if not m:
            continue
        major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
        families[(major, minor)].append((patch, tag))

    latest_per_family = {}
    for fam, items in families.items():
        items.sort(reverse=True)
        latest_per_family[fam] = items[0][1]  # tag ex: "v2.8.1"

    sorted_families = sorted(latest_per_family.keys(), reverse=True)[:N_VERSIONS]

    for fam in sorted_families:
        tag = latest_per_family[fam]
        folder = f"v{fam[0]}.{fam[1]}"
        print(f"{tag} {folder}")


if __name__ == "__main__":
    main()
