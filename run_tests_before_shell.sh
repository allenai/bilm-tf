#!/usr/bin/env bash

# Verify environment by running tests
# before dropping into a shell.
set -e
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover tests/
echo "Tests passed!"

/bin/bash