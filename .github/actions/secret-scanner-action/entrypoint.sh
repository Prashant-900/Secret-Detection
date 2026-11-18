#!/usr/bin/env bash
# Secret Detection Action — entrypoint
# Runs either the fast GBM scanner or the llm encoder bundle,
# then normalises output, emits annotations, and writes SARIF.
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
ACTION_DIR="${ACTION_PATH}"
FAST_SCANNER="${ACTION_DIR}/scanner/scan.py"
NORMALISE="${ACTION_DIR}/normalise.py"
TO_SARIF="${ACTION_DIR}/to_sarif.py"

RAW_JSON=$(mktemp /tmp/scan-raw-XXXXXX.json)
NORM_JSON=$(mktemp /tmp/scan-norm-XXXXXX.json)

echo "::group::Secret Detection Scanner (mode=${INPUT_MODE})"

# ── Mode: fast ────────────────────────────────────────────────────────────────
if [[ "${INPUT_MODE}" == "fast" ]]; then
    echo ">> Mode: fast (GBM re-ranker, no ML deps)"

    TARGET_PATHS=(${INPUT_TARGET})

    if [[ ${#TARGET_PATHS[@]} -gt 1 ]]; then
        echo ">> Scanning ${#TARGET_PATHS[@]} individual changed files…"
        MERGED_JSON=$(mktemp /tmp/merged-XXXXXX.json)
        echo "[]" > "${MERGED_JSON}"
        for f in "${TARGET_PATHS[@]}"; do
            if [[ -f "$f" ]]; then
                FILE_RAW=$(mktemp /tmp/file-raw-XXXXXX.json)
                ARGS=("$f" --json "${FILE_RAW}")
                [[ -n "${INPUT_BLOCK_AT:-}" ]] && ARGS+=(--block-at "${INPUT_BLOCK_AT}")
                [[ -n "${INPUT_WARN_AT:-}"  ]] && ARGS+=(--warn-at  "${INPUT_WARN_AT}")
                [[ "${INPUT_ALL_FILES:-false}" == "true" ]] && ARGS+=(--all-files)
                python3 "${FAST_SCANNER}" "${ARGS[@]}" || true
                python3 -c "
import json
try:
    m = json.load(open('${MERGED_JSON}'))
    r = json.load(open('${FILE_RAW}'))
    if isinstance(r, list):
        m.extend(r)
    json.dump(m, open('${MERGED_JSON}', 'w'))
except Exception:
    pass
"
                rm -f "${FILE_RAW}"
            fi
        done
        mv "${MERGED_JSON}" "${RAW_JSON}"
        BLOCKED_COUNT=$(python3 -c "import json; print(sum(1 for x in json.load(open('${RAW_JSON}')) if x.get('verdict')=='BLOCK'))")
        SCAN_EXIT=$(( BLOCKED_COUNT > 0 ? 1 : 0 ))
    else
        ARGS=("${INPUT_TARGET}" --json "${RAW_JSON}")
        [[ -n "${INPUT_BLOCK_AT:-}" ]] && ARGS+=(--block-at "${INPUT_BLOCK_AT}")
        [[ -n "${INPUT_WARN_AT:-}"  ]] && ARGS+=(--warn-at  "${INPUT_WARN_AT}")
        [[ "${INPUT_ALL_FILES:-false}" == "true" ]] && ARGS+=(--all-files)

        set +e
        python3 "${FAST_SCANNER}" "${ARGS[@]}"
        SCAN_EXIT=$?
        set -e
    fi

# ── Mode: llm ─────────────────────────────────────────────────────────────────
elif [[ "${INPUT_MODE}" == "llm" ]]; then
    echo ">> Mode: llm (CodeBERT + UniXcoder ensemble)"

    # --- Cache-aligned Model bundle directory (${HOME}/.secret-scanner-model) ---
    BUNDLE_DIR="${HOME}/.secret-scanner-model"
    mkdir -p "${BUNDLE_DIR}"

    find_selected_json() {
        if [[ -f "${BUNDLE_DIR}/selected.json" ]]; then
            echo "${BUNDLE_DIR}/selected.json"
        elif [[ -f "${BUNDLE_DIR}/artifacts/experiment/selected.json" ]]; then
            echo "${BUNDLE_DIR}/artifacts/experiment/selected.json"
        else
            echo ""
        fi
    }

    SELECTED_JSON=$(find_selected_json)

    if [[ -z "${SELECTED_JSON}" ]]; then
        # Check if custom download URL (e.g. Hugging Face) is specified
        if [[ -n "${INPUT_BUNDLE_URL:-}" ]]; then
            echo ">> Downloading model bundle from custom URL: ${INPUT_BUNDLE_URL}…"
            if [[ -n "${HF_TOKEN:-}" ]]; then
                curl -sSL -f -H "Authorization: Bearer ${HF_TOKEN}" \
                    -o "${BUNDLE_DIR}/model_bundle.tar.gz" \
                    "${INPUT_BUNDLE_URL}"
            else
                curl -sSL -f \
                    -o "${BUNDLE_DIR}/model_bundle.tar.gz" \
                    "${INPUT_BUNDLE_URL}"
            fi
        else
            echo ">> Downloading private model bundle from GitHub Releases into ${BUNDLE_DIR}…"
            gh release download v1 \
                --pattern "model_bundle.tar.gz" \
                --dir "${BUNDLE_DIR}" \
                --repo "${GITHUB_REPOSITORY}"
        fi

        tar -xzf "${BUNDLE_DIR}/model_bundle.tar.gz" -C "${BUNDLE_DIR}"
        rm -f "${BUNDLE_DIR}/model_bundle.tar.gz"

        SELECTED_JSON=$(find_selected_json)
        if [[ -z "${SELECTED_JSON}" ]]; then
            echo "::error::model_bundle.tar.gz did not contain selected.json"
            exit 2
        fi
        echo ">> Model manifest located at: ${SELECTED_JSON}"
    else
        echo ">> Model bundle already present in cache: ${SELECTED_JSON}"
    fi

    # Verify model weight directories
    BUNDLE_ROOT="$(dirname "${SELECTED_JSON}")"
    CB="${BUNDLE_ROOT}/runs/codebert-augmented-plain-seed37/best"
    UX="${BUNDLE_ROOT}/runs/unixcoder-augmented-plain-seed37/best"
    for d in "${CB}" "${UX}"; do
        if [[ ! -d "${d}" ]]; then
            echo "::error::Expected model directory not found: ${d}"
            exit 2
        fi
    done
    echo ">> Both ensemble models verified (${CB}, ${UX})"

    # Ensure secrets_pipeline can be imported from target workspace
    export PYTHONPATH="${TARGET_WORKSPACE:-$PWD}:${PYTHONPATH:-}"

    TARGET_PATHS=(${INPUT_TARGET})

    if [[ ${#TARGET_PATHS[@]} -gt 1 ]]; then
        echo ">> Running deep LLM scan across ${#TARGET_PATHS[@]} individual changed files…"
        MERGED_JSON=$(mktemp /tmp/merged-llm-XXXXXX.json)
        echo '{"status":"PASS","findings":[]}' > "${MERGED_JSON}"
        for f in "${TARGET_PATHS[@]}"; do
            if [[ -f "$f" ]]; then
                FILE_RAW=$(mktemp /tmp/file-llm-raw-XXXXXX.json)
                F_ABS="$(realpath "$f")"
                SEL_ABS="$(realpath "${SELECTED_JSON}")"
                (cd "${TARGET_WORKSPACE:-$PWD}" && python3 -m secrets_pipeline scan "${F_ABS}" --bundle "${SEL_ABS}" --out "${FILE_RAW}") || true
                python3 -c "
import json
try:
    m = json.load(open('${MERGED_JSON}'))
    r = json.load(open('${FILE_RAW}'))
    if isinstance(r, dict) and 'findings' in r:
        m['findings'].extend(r['findings'])
        if r.get('status') == 'BLOCK':
            m['status'] = 'BLOCK'
        elif r.get('status') == 'WARN' and m.get('status') != 'BLOCK':
            m['status'] = 'WARN'
    json.dump(m, open('${MERGED_JSON}', 'w'))
except Exception:
    pass
"
                rm -f "${FILE_RAW}"
            fi
        done
        mv "${MERGED_JSON}" "${RAW_JSON}"
        BLOCKED_COUNT=$(python3 -c "import json; print(sum(1 for x in json.load(open('${RAW_JSON}')).get('findings',[]) if x.get('verdict')=='BLOCK'))")
        SCAN_EXIT=$(( BLOCKED_COUNT > 0 ? 1 : 0 ))
    else
        TARGET_ABS="$(realpath "${INPUT_TARGET}")"
        RAW_JSON_ABS="$(realpath "${RAW_JSON}")"
        SELECTED_JSON_ABS="$(realpath "${SELECTED_JSON}")"

        ARGS=("${TARGET_ABS}" --bundle "${SELECTED_JSON_ABS}" --out "${RAW_JSON_ABS}")
        [[ "${INPUT_ALL_FILES:-false}" == "true" ]] && ARGS+=(--all-files)

        set +e
        (cd "${TARGET_WORKSPACE:-$PWD}" && python3 -m secrets_pipeline scan "${ARGS[@]}")
        SCAN_EXIT=$?
        set -e
    fi

else
    echo "::error::Unknown mode '${INPUT_MODE}'. Use 'fast' or 'llm'."
    exit 2
fi

echo "::endgroup::"

# ── Normalise output ──────────────────────────────────────────────────────────
echo ">> Normalising scanner output (mode=${INPUT_MODE})…"
python3 "${NORMALISE}" \
    --raw  "${RAW_JSON}" \
    --out  "${NORM_JSON}" \
    --mode "${INPUT_MODE}"

# ── Enforce exit code 1 if any finding has verdict == BLOCK ───────────────────
BLOCKED_COUNT=$(python3 -c "
import json
d = json.load(open('${NORM_JSON}'))
print(sum(1 for f in d if f.get('verdict')=='BLOCK'))
")

if [[ "${BLOCKED_COUNT}" -gt 0 ]]; then
    SCAN_EXIT=1
fi

# ── GitHub Annotations (inline PR comments) ───────────────────────────────────
export NORM_JSON
python3 - <<'PYEOF'
import json, os, sys

norm_json = os.environ.get("NORM_JSON")
if not norm_json or not os.path.exists(norm_json):
    sys.exit(0)

findings = json.load(open(norm_json))
printed = 0
for f in findings:
    if f.get("verdict") not in ("BLOCK", "WARN"):
        continue
    level = "error" if f["verdict"] == "BLOCK" else "warning"
    reason = f.get("reason", "ml-scorer")
    msg = (f'[{f["verdict"]}] score={f["score"]:.4f}  '
           f'via {reason}  —  {f.get("match", "[REDACTED]")}')
    print(f'::{level} file={f["file"]},line={f["line"]}::{msg}')
    printed += 1

if printed == 0:
    print("No BLOCK or WARN findings.")
PYEOF

# ── SARIF ─────────────────────────────────────────────────────────────────────
python3 "${TO_SARIF}" "${NORM_JSON}" "${INPUT_SARIF}"
echo ">> SARIF written to ${INPUT_SARIF}"

# ── Outputs ───────────────────────────────────────────────────────────────────
BLOCKED=$(python3 -c "
import json, sys
d = json.load(open('${NORM_JSON}'))
print(sum(1 for f in d if f.get('verdict')=='BLOCK'))
")
WARNED=$(python3 -c "
import json, sys
d = json.load(open('${NORM_JSON}'))
print(sum(1 for f in d if f.get('verdict')=='WARN'))
")
STATUS="PASS"
[[ "${SCAN_EXIT}" -eq 1 ]] && STATUS="BLOCK"
[[ "${SCAN_EXIT}" -eq 2 ]] && STATUS="ERROR"
[[ "${SCAN_EXIT}" -eq 0 && "${WARNED}" -gt 0 ]] && STATUS="WARN"

{
  echo "blocked=${BLOCKED}"
  echo "warned=${WARNED}"
  echo "status=${STATUS}"
} >> "${GITHUB_OUTPUT}"

echo ">> Summary: BLOCK=${BLOCKED}  WARN=${WARNED}  STATUS=${STATUS}"

# Clean up
rm -f "${RAW_JSON}" "${NORM_JSON}"

exit "${SCAN_EXIT}"