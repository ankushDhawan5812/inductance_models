#!/bin/bash
# Run ablation v2 experiments sequentially.
# Each run takes ~5-15 min depending on unroll steps.
#
# Usage: bash run_ablation_v2.sh
#   or:  bash run_ablation_v2.sh warmup10 warmup15   (run specific configs only)

set -e
cd "$(dirname "$0")"

CONFIGS_DIR="configs/ablation_v2"
EXPERIMENTS=(
    "baseline"
    "window15"
    "window20"
    "unroll30"
    "unroll40"
    "warmup10"
    "warmup15"
    "large_model"
    "combined_best"
)

# If args provided, use those instead
if [ $# -gt 0 ]; then
    EXPERIMENTS=("$@")
fi

echo "============================================"
echo "  Ablation v2: ${#EXPERIMENTS[@]} experiments"
echo "============================================"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    config="$CONFIGS_DIR/${exp}.yaml"
    if [ ! -f "$config" ]; then
        echo "SKIP: $config not found"
        continue
    fi
    echo ">>> Running: $exp"
    echo "    Config: $config"
    python main.py "$config"
    echo ""
    echo ">>> Done: $exp"
    echo "--------------------------------------------"
    echo ""
done

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "============================================"

# Print summary table
echo ""
echo "=== Results Summary ==="
echo ""
printf "%-20s %8s %8s %8s %8s %8s\n" "Experiment" "Mean" "Median" "Max" "P95" "@0.1mm"
echo "------------------------------------------------------------------------"
for dir in output/2026-*_causal_transformer_ar/; do
    if [ -f "$dir/metrics.json" ]; then
        name=$(basename "$dir")
        python3 -c "
import json
m = json.load(open('$dir/metrics.json'))
print(f'${name:<20s} {m[\"euclidean_mean\"]:8.4f} {m[\"euclidean_median\"]:8.4f} {m[\"euclidean_max\"]:8.2f} {m[\"euclidean_p95\"]:8.4f} {m[\"acc_0p1mm\"]:7.1f}%')
" 2>/dev/null || true
    fi
done
