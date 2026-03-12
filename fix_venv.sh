#!/bin/bash
# Fix for iCloud paths with spaces breaking Python .pth editable installs.
# Run after `uv sync` if `uv run python -m anomaly_system` fails with ModuleNotFoundError.
#
# Root cause: Python's .pth file processor splits lines on spaces, so paths
# like "/Users/.../Mobile Documents/..." are truncated.

SITE_PACKAGES="$(dirname "$0")/.venv/lib/python3.12/site-packages"
cat > "$SITE_PACKAGES/sitecustomize.py" << 'PYEOF'
import sys, os
_src = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__))))),
    "src",
)
if os.path.isdir(_src) and _src not in sys.path:
    sys.path.insert(0, _src)
PYEOF
echo "Installed sitecustomize.py fix in $SITE_PACKAGES"
