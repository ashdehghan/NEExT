#!/bin/sh
# Start a new research session: copies _template/ to sessions/YYYY-MM-DD-<topic-slug>.
set -eu

if [ $# -ne 1 ]; then
    echo "usage: $0 <topic-slug>   (e.g. $0 khop-sampling)" >&2
    exit 1
fi

cd "$(dirname "$0")"

slug=$1
case "$slug" in
    *[!a-z0-9-]*)
        echo "error: topic-slug must be lowercase letters, digits, and hyphens only" >&2
        exit 1
        ;;
esac

target="sessions/$(date +%F)-$slug"
if [ -e "$target" ]; then
    echo "error: $target already exists" >&2
    exit 1
fi

cp -r _template "$target"
echo "Created $target"
echo "Next: fill in $target/README.md and add a row to INDEX.md"
