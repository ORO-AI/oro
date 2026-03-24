#!/usr/bin/env bash
# Run the default one-shot agent against all 750 problems (250 per category).
#
# Usage:
#   bash scripts/run_full_evaluation.sh
#
# Prerequisites:
#   docker compose up -d search-server proxy inference-gateway
#
# Output files land in logs/ via Docker volume mount.
# Estimated runtime: ~40 min with 8 parallel workers (~25s per problem).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MAX_WORKERS="${SANDBOX_MAX_WORKERS:-8}"
TIMEOUT="${SANDBOX_TIMEOUT:-300.0}"

CATEGORIES=(
  "product:data/synthesize_product_test.jsonl:logs/full_eval_product.jsonl"
  "shop:data/synthesize_shop_test.jsonl:logs/full_eval_shop.jsonl"
  "voucher:data/synthesize_voucher_test.jsonl:logs/full_eval_voucher.jsonl"
)

echo "=== Full Evaluation Run ==="
echo "Started at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Max workers: $MAX_WORKERS"
echo "Timeout per problem: ${TIMEOUT}s"
echo ""

failed=0

for entry in "${CATEGORIES[@]}"; do
  IFS=':' read -r category problem_file output_file <<< "$entry"

  echo "--- Running $category ($(wc -l < "$problem_file") problems) ---"
  echo "  Problem file: $problem_file"
  echo "  Output file:  $output_file"
  start_time=$(date +%s)

  # Run sandbox; allow non-zero exit (agent failures cause exit code 1)
  if docker compose run --rm \
    -e "SANDBOX_MAX_WORKERS=$MAX_WORKERS" \
    -e "SANDBOX_TIMEOUT=$TIMEOUT" \
    sandbox \
    python -m src.agent.run_sandbox \
      --problem-file "$problem_file" \
      --max-workers "$MAX_WORKERS" \
      --timeout "$TIMEOUT" \
      --output "/app/$output_file"; then
    echo "  Completed successfully."
  else
    echo "  WARNING: $category exited with non-zero status (some problems may have failed)."
    failed=$((failed + 1))
  fi

  end_time=$(date +%s)
  echo "  Duration: $((end_time - start_time))s"
  echo ""
done

echo "=== Evaluation Complete ==="
echo "Finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

if [ "$failed" -gt 0 ]; then
  echo "WARNING: $failed category run(s) had failures. Check output files for details."
fi

# Verify output files exist
echo ""
echo "Output files:"
for entry in "${CATEGORIES[@]}"; do
  IFS=':' read -r category _ output_file <<< "$entry"
  if [ -f "$output_file" ]; then
    lines=$(wc -l < "$output_file")
    echo "  $output_file: $lines problems"
  else
    echo "  $output_file: MISSING"
  fi
done
