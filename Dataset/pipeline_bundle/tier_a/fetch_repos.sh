#!/bin/bash
# Fetch every repo in manifest.csv at its pinned commit (shallow, no history).
# usage: ./fetch_repos.sh [manifest.csv] [outdir]
set -u
manifest="${1:-manifest_v2.csv}"; out="${2:-repos}"
mkdir -p "$out"
fetch_one() {
  repo="$1"; sha="$2"; out="$3"
  d="$out/${repo//\//__}"
  [ -d "$d/.git" ] && { echo "skip $repo"; return; }
  mkdir -p "$d" && cd "$d" || return
  git init -q && git remote add origin "https://github.com/$repo.git"
  if git fetch -q --depth 1 origin "$sha" 2>/dev/null && git checkout -q FETCH_HEAD; then
    echo "ok   $repo @ ${sha:0:8}"
  else
    echo "FAIL $repo"
  fi
}
export -f fetch_one
tail -n +2 "$manifest" | awk -F, '{print $1, $3}' | xargs -P 8 -L 1 bash -c 'fetch_one "$0" "$1" "'"$out"'"'
