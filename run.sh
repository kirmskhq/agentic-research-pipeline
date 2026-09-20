#!/bin/bash
# Generate one report. Usage: ./run.sh "ваша сфера" > report.md
# Exit code: 0 = report is fit to deliver, 3 = HOLD (needs a human), 1 = crash.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec docker run --rm   --env-file "$HERE/.env"   -v "$HERE:/app:ro"   research-pipeline:latest   python /app/pipeline.py "$1"
