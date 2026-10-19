#!/bin/sh
# Make a variant of the package's lib/ to try a change in - build/perf/variants/NAME/lib/, a copy of lib/ (or of another
# variant's, FROM) with the generated models linked, not copied. The lab's timings take its lib as
# /build/perf/variants/NAME/lib; tools/perf/compare.js compares it with lib/.
#
#     tools/perf/variant.sh NAME [FROM]
set -eu
cd "$(dirname "$0")/../.."
NAME="$1"
FROM="${2:-lib}"
case "$FROM" in lib) SOURCE=lib;; *) SOURCE="build/perf/variants/$FROM/lib";; esac
DEST="build/perf/variants/$NAME/lib"
if [ -e "$DEST" ]; then
    echo "variant.sh: $DEST exists" >&2
    exit 1
fi
mkdir -p "$DEST"
for FILE in "$SOURCE"/*.js; do
    cp "$FILE" "$DEST/"
done
ln -s "$(pwd)/lib/models" "$DEST/models"
echo "$DEST"
