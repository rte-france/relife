#!/usr/bin/env bash
# Build the multi-version Sphinx documentation site.
#
# Expects to be run from the root of the relife repository, in a clean git
# checkout with all tags fetched (fetch-depth: 0 in the GH Actions checkout).
# 
# Outputs the full site under ./_site/.
#
# IMPORTANT NOTES : 
# - This script might need to be updated when the hierarchy of docs/ folder is changed
# - This script relies on two Python scripts : 
#   * generate_versions.py : script that generates the versions.json file at the site root.
#     This JSON file is mandatory to build the documentation versions.
#     See : https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html 
#   * select_versions.py : utility function for this script to print tags and versions.
#     This script defines the number of older versions of the documentation we want to build.


# -----------------------------------------------------------------------------
# You can define a repository environment variable on your Github repository to 
# deploy the documentation on your own Github Pages (useful for debugging 
# before creating a PR).
# -----------------------------------------------------------------------------

echo "=========================================="
echo "Deploying with URL: $DOCS_BASE_URL"
echo "=========================================="

# -----------------------------------------------------------------------------
# Helper function for the source doc folder (will become obsolete when building 
# documentation versions from 2.8.0 and after)
# -----------------------------------------------------------------------------

# Detect the doc source dir (docs/source or doc/source, depending on the tag).
detect_doc_src() {
  if [ -d "docs/source" ]; then
    echo "docs/source"
  elif [ -d "doc/source" ]; then
    echo "doc/source"
  else
    echo "ERROR: no source file found"
    exit 1
  fi
}

# -----------------------------------------------------------------------------
# Helper function to build one version of the documentation.
# Args:
#   $1 - version name (e.g. "latest" or "v2.8"), used both as DOCS_VERSION
#        and as the output folder name under _site/
# -----------------------------------------------------------------------------

build_version() {
    local version="$1"

    # Find the source documentation folder
    DOC_SRC=$(detect_doc_src)

    # Copy main conf file (most recent build logic)
    cp /tmp/conf_main.py "$DOC_SRC/conf.py"

    # Install dependencies from current branch/tag
    python -m pip install . --group dev

    # Create the environment variable DOCS_VERSION on the go (for conf.py file)
    # when building the documentation with sphinx
    # -M option of sphinx-build calls the make command
    # -Ea options are safety nets to tell sphinx to rebuild everything (ignore cache)
    # since we are building multiple times on the same machine
    DOCS_VERSION="$version" sphinx-build -M html "./$DOC_SRC" ./build_tmp -Ea

    # Create destination directory
    mkdir -p "_site/$version"

    # Copy build HTML files only to _site folder and delete them in the build_tmp/
    # This step is necessary to avoid keeping unused sphinx-generated files
    cp -r ./build_tmp/html/. "_site/$version/"
    rm -rf ./build_tmp
}

# -----------------------------------------------------------------------------
# Step 1: Prepare site and scripts folders
# Files from main branch are copied to implement the new logic to older tags
# -----------------------------------------------------------------------------

# Create _site/ folder even if it exists
mkdir -p _site

# Copy recent scripts from main into tmp/ (for older tags without them)
cp docs/source/select_versions.py /tmp/select_versions.py
cp docs/source/generate_versions.py /tmp/generate_versions.py
cp docs/source/conf.py /tmp/conf_main.py

# -----------------------------------------------------------------------------
# Step 2: Build "latest" from main
# -----------------------------------------------------------------------------

echo "=== Building main into latest folder ==="

# Force checkout to drop eventual changes made in versioned git files
git checkout --force main

# Call helper function to build this version
build_version "latest"

# -----------------------------------------------------------------------------
# Step 3: Create "versions.json" file from main
# -----------------------------------------------------------------------------

python /tmp/generate_versions.py
cp "$DOC_SRC/_static/versions.json" _site/versions.json

# -----------------------------------------------------------------------------
# Step 4: Build each selected tag
# -----------------------------------------------------------------------------

# List of all tags and versions
SELECTION=$(python /tmp/select_versions.py)

# Build docs for each version of type major.minor
while read -r TAG FOLDER; do

    echo "=== Building $TAG into $FOLDER ==="

    # Make sure previous builds do not interfere with the current one
    rm -rf docs/ doc/ build_tmp/

    # Force checkout to drop eventual changes made in versioned git files
    git checkout --force "$TAG"

    # Call helper function to build this version
    build_version "$FOLDER"

done <<< "$SELECTION"

# -----------------------------------------------------------------------------
# Step 5: root index (redirects to latest) + nojekyll
# -----------------------------------------------------------------------------

# index.html redirects to /latest/
cat > _site/index.html <<'EOF'
<!DOCTYPE html>
<meta http-equiv="refresh" content="0; url=./latest/">
EOF

# Jekyll by default ignore folders starting with _
touch _site/.nojekyll

echo "=========================================="
echo "Build completed"
echo "=========================================="
