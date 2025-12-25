#!/bin/bash
set -e
git add .
git commit -m "wip: snapshot $(date +'%Y-%m-%d %H:%M')"
git push origin HEAD
echo "Work saved to remote."
