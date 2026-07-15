#!/usr/bin/env bash
#
# build_pdfs.sh — ONE combined PDF (design_document.pdf) with every Mermaid
# diagram embedded as an inline base64 PNG (no external image refs, so the
# PDF renderer's file:// blocking can't break them).
#
# SETUP (one time):
#   npm install -g @mermaid-js/mermaid-cli   # mmdc
#   npm install -g md-to-pdf                  # md -> pdf
#   sudo apt-get install -y chromium-browser  # or chromium (Linux)
#
# USAGE:  ./build_pdfs.sh            (uses ORDER below)
#         ./build_pdfs.sh a b c      (explicit order)
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_PDF="$SCRIPT_DIR/design_document.pdf"
WORK="$(mktemp -d)"
ASSEMBLED="$WORK/assembled.md"; : > "$ASSEMBLED"

ORDER=(
  "system_context.mmd"
  "deployment.mmd"
  "node_graph.mmd"
  "tf_tree.mmd"
  "node_interfaces.md"
  "data_flow.md"
  "state_machine.md"
  "conventions.md"
)
if [ "$#" -gt 0 ]; then FILES=("$@"); else
  FILES=(); for f in "${ORDER[@]}"; do [ -f "$f" ] && FILES+=("$f"); done
fi
[ "${#FILES[@]}" -eq 0 ] && { echo "No input files."; exit 1; }

title_of(){ basename "$1" | sed -E 's/\.(mmd|md)$//; s/_/ /g'; }

# Per-file render scale override (default 2). Bigger = larger diagram.
# Match by the base filename. Edit to taste.
scale_for(){
  case "$(basename "$1")" in
    tf_tree.mmd)    echo 1 ;;   # tall -> render smaller so it fits
    *)              echo 2 ;;
  esac
}

# Per-file <img> style. Tall diagrams get a max-height so they shrink to fit
# one page (A4 portrait usable height ~= 245mm after margins + title).
style_for(){
  case "$(basename "$1")" in
    tf_tree.mmd) echo "max-width:100%; max-height:240mm; width:auto;" ;;
    *)           echo "max-width:100%;" ;;
  esac
}

# render mermaid file -> inline <img> data-URI (echoed to stdout).
emit_img(){
  local src="$1" png="$WORK/x.png" s st
  s="$(scale_for "$src")"; st="$(style_for "$src")"
  if mmdc -i "$src" -o "$png" -b white -s "$s" >/dev/null 2>&1 && [ -s "$png" ]; then
    local b64; b64="$(base64 -w0 "$png" 2>/dev/null || base64 "$png" | tr -d '\n')"
    printf '<img style="%s" src="data:image/png;base64,%s" />\n' "$st" "$b64"
  else
    printf '_(diagram failed to render: %s)_\n' "$(basename "$src")"
  fi
}

n=0
for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue
  n=$((n+1)); title="$(title_of "$f")"; echo ">> $f"
  if [[ "$f" == *.mmd ]]; then
    { echo "# ${title}"; echo; emit_img "$f"; echo;
      echo '<div style="page-break-after: always;"></div>'; echo; } >> "$ASSEMBLED"
  else
    # split ```mermaid blocks out, render each inline, keep the rest verbatim
    awk -v w="$WORK" -v ix="$n" '
      BEGIN{inm=0;k=0}
      /^```mermaid[[:space:]]*$/{inm=1;k++;blk="";next}
      inm==1 && /^```[[:space:]]*$/{
        inm=0; mmf=sprintf("%s/b_%d_%d.mmd",w,ix,k);
        printf "%s",blk > mmf; close(mmf);
        printf "@@IMG:%s@@\n",mmf; next }
      inm==1{blk=blk $0 "\n"; next}
      {print}
    ' "$f" > "$WORK/pre_$n.md"
    while IFS= read -r line; do
      case "$line" in
        @@IMG:*@@) mmf="${line#@@IMG:}"; mmf="${mmf%@@}"; emit_img "$mmf" ;;
        *) printf '%s\n' "$line" ;;
      esac
    done < "$WORK/pre_$n.md" >> "$ASSEMBLED"
    printf '\n<div style="page-break-after: always;"></div>\n\n' >> "$ASSEMBLED"
  fi
done

echo ">> building PDF"
# Stylesheet: rotate the node-graph page to landscape so wide diagrams fit.
CSS="$WORK/style.css"
cat > "$CSS" << 'CSSEOF'
img { display: block; margin: 0 auto; max-width: 100%; }
CSSEOF
# md-to-pdf reads a config file for stylesheet + pdf options.
CFG="$WORK/cfg.js"
cat > "$CFG" << CFGEOF
module.exports = {
  stylesheet: ["$CSS"],
  pdf_options: { format: "A4", margin: "15mm" },
  launch_options: { args: ["--no-sandbox"] }
};
CFGEOF
md-to-pdf --config-file "$CFG" "$ASSEMBLED" >/dev/null
mv "${ASSEMBLED%.md}.pdf" "$OUT_PDF"
rm -rf "$WORK"
echo "Done -> $OUT_PDF"
