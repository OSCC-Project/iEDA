#!/bin/bash
# Patch IHP SG13G2 Liberty files for iEDA parser compatibility.
#
# The IHP PDK Liberty files contain constructs that iEDA's parser cannot handle:
#   1. Unquoted wire_load_from_area model names starting with digits (e.g. 0_1k)
#   2. Paired dont_touch/dont_use attributes on cells (unsupported by iEDA)
#   3. delay_template_2x2 lookup table template (unused after removing IOPadAnalog)
#   4. sg13g2_IOPadAnalog cell definition (references deleted template)
#
# Usage: ./patch_liberty.sh [FOUNDRY_DIR]
#   FOUNDRY_DIR defaults to ../../foundry/ihp130

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FOUNDRY_DIR="${1:-${FOUNDRY_DIR:-$SCRIPT_DIR/../../foundry/ihp130}}"
OUTPUT_DIR="$SCRIPT_DIR/lib_patched"

STDCELL_SRC="$FOUNDRY_DIR/ihp-sg13g2/libs.ref/sg13g2_stdcell/lib/sg13g2_stdcell_typ_1p20V_25C.lib"
IO_SRC="$FOUNDRY_DIR/ihp-sg13g2/libs.ref/sg13g2_io/lib/sg13g2_io_typ_1p2V_3p3V_25C.lib"

for f in "$STDCELL_SRC" "$IO_SRC"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: source file not found: $f" >&2
        echo "Make sure the IHP PDK submodule is initialized:" >&2
        echo "  git submodule update --init scripts/foundry/ihp130" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

# Sed expression to quote wire_load_from_area model names
QUOTE_WLA='s/wire_load_from_area\s*\(([^,]+),\s*([^,]+),\s*([^")][^)]*)\)/wire_load_from_area (\1, \2, "\3")/'

# --- Patch stdcell Liberty ---
# Only needs wire_load_from_area quoting
sed -E "$QUOTE_WLA" "$STDCELL_SRC" > "$OUTPUT_DIR/sg13g2_stdcell_typ_1p20V_25C.lib"
echo "Patched: $OUTPUT_DIR/sg13g2_stdcell_typ_1p20V_25C.lib"

# --- Patch IO Liberty ---
# Use awk for structural changes, then sed for quoting
awk '
    # --- Remove delay_template_2x2 block ---
    /lu_table_template[[:space:]]*\(delay_template_2x2\)/ {
        depth = 0
        while (1) {
            if ($0 ~ /\{/) depth++
            if ($0 ~ /\}/) { depth--; if (depth <= 0) break }
            if (getline <= 0) break
        }
        next
    }

    # --- Remove sg13g2_IOPadAnalog cell block ---
    /cell[[:space:]]*\(sg13g2_IOPadAnalog\)/ {
        depth = 0
        while (1) {
            if ($0 ~ /\{/) depth++
            if ($0 ~ /\}/) { depth--; if (depth <= 0) break }
            if (getline <= 0) break
        }
        next
    }

    # --- Remove dont_touch + dont_use pairs ---
    /^[[:space:]]*dont_touch[[:space:]]*:[[:space:]]*true[[:space:]]*;/ {
        saved = $0
        if (getline nextline > 0) {
            if (nextline ~ /^[[:space:]]*dont_use[[:space:]]*:[[:space:]]*true[[:space:]]*;/) {
                next  # skip both lines
            } else {
                print saved
                $0 = nextline
            }
        }
    }

    { print }
' "$IO_SRC" | sed -E "$QUOTE_WLA" > "$OUTPUT_DIR/sg13g2_io_typ_1p2V_3p3V_25C.lib"

echo "Patched: $OUTPUT_DIR/sg13g2_io_typ_1p2V_3p3V_25C.lib"
echo "Done. Patched files in: $OUTPUT_DIR"
