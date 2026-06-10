#!/usr/bin/env bash
# Export all PDF figures from draw_pictures/ into release/ and zip them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DRAW_DIR="$PROJECT_ROOT/draw_pictures"
RELEASE_DIR="$PROJECT_ROOT/release"
ZIP_NAME="$RELEASE_DIR/figures.zip"

rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

for ext in pdf md; do
    find "$DRAW_DIR" -name "*.$ext" | while read -r src; do
        rel="${src#"$DRAW_DIR"/}"
        dest="$RELEASE_DIR/$rel"
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
        echo "  copied: $rel"
    done
done

count=$(find "$RELEASE_DIR" -type f | wc -l)
echo "Exported $count files to $RELEASE_DIR/"

# Create zip via Python (no external dependency)
python3 -c "
import zipfile, os, sys
release = sys.argv[1]
zip_path = sys.argv[2]
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(release):
        for f in files:
            if f.endswith('.pdf') or f.endswith('.md'):
                full = os.path.join(root, f)
                arc = os.path.relpath(full, release)
                zf.write(full, arc)
                print(f'  zipped: {arc}')
size = os.path.getsize(zip_path) / (1024*1024)
print(f'Created {os.path.basename(zip_path)} ({size:.2f} MB)')
" "$RELEASE_DIR" "$ZIP_NAME"
