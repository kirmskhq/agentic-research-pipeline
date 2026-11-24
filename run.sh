#!/bin/bash
# Generate one report. Usage: ./run.sh "ваша сфера" > report.md
# The client report goes to stdout; report.md, review.md (notes for the editor) and
# verdict.json are also written to reports/<timestamp>/.
# Exit code: 0 = nothing flagged, 3 = HOLD (editor should read it closely), 1 = crash.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$HERE/reports"
exec docker run --rm   --env-file "$HERE/.env"   -v "$HERE:/app:ro"   -v "$HERE/reports:/reports"   research-pipeline:latest   python /app/pipeline.py "$1" --out "/reports/$RUN_ID"
