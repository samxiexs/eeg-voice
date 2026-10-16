#!/usr/bin/env bash
# Download the KaraOne imagined/overt speech EEG dataset (Zhao & Rudzicz 2015,
# University of Toronto) into this bundle's data directory.
#
#   bash scripts/download_karaone.sh                 # all 14 participants (~24.8 GB of archives)
#   SUBJECTS="MM05 P02" bash scripts/download_karaone.sh
#   bash scripts/download_karaone.sh verify          # re-check archives/extractions only
#   META_ONLY=1 bash scripts/download_karaone.sh     # only the authors' code and paper (no EEG)
#
# Layout after extraction:
#   data/karaone/<PARTICIPANT>/...      one folder per participant archive
#   data/karaone/archives/<P>.tar.bz2   kept only with KEEP_ARCHIVES=1
#   data/karaone/_meta/                 src.zip (authors' code), paper, download_manifest.json
#
# Downloads resume (curl -C -), every archive is size-checked against the
# server and integrity-checked with tar before extraction.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$BUNDLE_DIR/data/karaone}"
BASE_URL="${KARAONE_BASE_URL:-https://www.cs.toronto.edu/~complingweb/data/karaOne}"
ALL_SUBJECTS="MM05 MM08 MM09 MM10 MM11 MM12 MM14 MM15 MM16 MM18 MM19 MM20 MM21 P02"
SUBJECTS="${SUBJECTS:-$ALL_SUBJECTS}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"
META_ONLY="${META_ONLY:-0}"
MODE="${1:-download}"
[[ "$META_ONLY" == "1" ]] && SUBJECTS=""

command -v curl >/dev/null 2>&1 || { echo "ERROR: curl is required" >&2; exit 127; }
command -v tar >/dev/null 2>&1 || { echo "ERROR: tar is required" >&2; exit 127; }
for s in $SUBJECTS; do
  case " $ALL_SUBJECTS " in *" $s "*) ;; *) echo "ERROR: unknown participant '$s' (expected one of: $ALL_SUBJECTS)" >&2; exit 2;; esac
done

ARCHIVES="$DATA_DIR/archives"; META="$DATA_DIR/_meta"
mkdir -p "$ARCHIVES" "$META"
echo "[karaone] destination: $DATA_DIR"
echo "[karaone] participants: $SUBJECTS"

remote_size() { curl -sIL --max-time 60 "$1" | awk 'tolower($1)=="content-length:"{v=$2} END{gsub("\r","",v); print v+0}'; }
local_size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1" 2>/dev/null || echo 0; }

fetch() {  # fetch <url> <destination>
  local url="$1" dest="$2" expected actual
  expected="$(remote_size "$url")"
  if [[ -z "$expected" || "$expected" == "0" ]]; then echo "ERROR: cannot read size of $url (offline or URL changed?)" >&2; exit 1; fi
  actual="$(local_size "$dest")"
  if [[ "$actual" == "$expected" ]]; then echo "[karaone] complete: $(basename "$dest") ($expected bytes)"; return; fi
  echo "[karaone] downloading $(basename "$dest") ($expected bytes; resuming from $actual)"
  curl -L --fail --retry 5 --retry-delay 10 -C - -o "$dest" "$url"
  actual="$(local_size "$dest")"
  [[ "$actual" == "$expected" ]] || { echo "ERROR: size mismatch for $dest ($actual != $expected)" >&2; exit 1; }
}

verify_archive() { tar -tjf "$1" >/dev/null 2>&1; }

manifest_entry() {  # manifest_entry <participant> <archive> <status>
  printf '  {"participant": "%s", "archive": "%s", "bytes": %s, "status": "%s"}' "$1" "$(basename "$2")" "$(local_size "$2")" "$3"
}

entries=()
if [[ "$MODE" == "download" ]]; then
  need=0; for s in $SUBJECTS; do need=$((need + $(remote_size "$BASE_URL/$s.tar.bz2"))); done
  avail_kb="$(df -k "$DATA_DIR" | awk 'NR==2{print $4}')"
  echo "[karaone] archives to fetch: $((need / 1000000)) MB; free space: $((avail_kb / 1000)) MB (extraction needs roughly the same again)"
  if (( avail_kb * 1024 < need * 2 )); then echo "WARNING: free space may be insufficient for archives + extraction" >&2; fi
  fetch "$BASE_URL/src.zip" "$META/src.zip"
  fetch "$BASE_URL/ZhaoRudzicz15.pdf" "$META/ZhaoRudzicz15.pdf"
fi

for s in $SUBJECTS; do
  archive="$ARCHIVES/$s.tar.bz2"; target="$DATA_DIR/$s"
  if [[ "$MODE" == "download" ]]; then
    if [[ -d "$target" && -f "$target/.extracted" && ! -f "$archive" ]]; then
      echo "[karaone] already extracted: $s"; entries+=("$(manifest_entry "$s" "$archive" extracted)"); continue
    fi
    fetch "$BASE_URL/$s.tar.bz2" "$archive"
    echo "[karaone] verifying $s.tar.bz2"
    verify_archive "$archive" || { echo "ERROR: corrupt archive $archive; delete it and rerun" >&2; exit 1; }
    echo "[karaone] extracting $s"
    rm -rf "$target.partial" "$target"; mkdir -p "$target.partial"
    tar -xjf "$archive" -C "$target.partial"
    # Archives carry the authors' absolute path prefix (p/spoclab/users/szhao/EEG/data/<P>/...);
    # keep only the participant folder itself.
    inner="$(find "$target.partial" -type d -name "$s" | head -1)"
    if [[ -n "$inner" ]]; then mv "$inner" "$target" && rm -rf "$target.partial"; else mv "$target.partial" "$target"; fi
    date -u +%Y-%m-%dT%H:%M:%SZ > "$target/.extracted"
    [[ "$KEEP_ARCHIVES" == "1" ]] || rm -f "$archive"
    entries+=("$(manifest_entry "$s" "$archive" extracted)")
  else
    if [[ -f "$archive" ]]; then
      verify_archive "$archive" && status=archive_ok || status=archive_corrupt
    elif [[ -f "$target/.extracted" ]]; then status=extracted
    else status=missing; fi
    echo "[karaone] $s: $status"; entries+=("$(manifest_entry "$s" "$archive" "$status")")
  fi
done

{
  echo '{'
  echo "  \"source\": \"$BASE_URL\","
  echo "  \"written_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo '  "participants": ['
  if (( ${#entries[@]} )); then (IFS=$',\n'; printf '%s\n' "${entries[*]}"); fi
  echo '  ]'
  echo '}'
} > "$META/download_manifest.json"
echo "[karaone] done; manifest: $META/download_manifest.json"
