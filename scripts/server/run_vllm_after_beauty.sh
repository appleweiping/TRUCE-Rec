#!/bin/bash
# Wait for beauty observation to finish, then switch to vLLM for remaining domains
set -euo pipefail

PROJECT_DIR="$HOME/projects/TRUCE-Rec"
cd "$PROJECT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm
export PYTHONPATH="$PROJECT_DIR/src"

echo "=== TRUCE-Rec vLLM Observation Pipeline ==="
echo "Start: $(date)"

# Wait for beauty to finish (the transformers-based run)
echo "Waiting for beauty observation to complete..."
while ! [ -f "outputs/server_observations/qwen3_8b/beauty/manifest.json" ]; do
    sleep 60
    echo "  $(date '+%H:%M') waiting..."
done
echo "Beauty observation complete!"

# Kill any remaining transformers-based observation processes
pkill -f "run_qwen3_observation" 2>/dev/null || true
sleep 5

# Wait for GPU to be mostly free (need at least 35GB for vLLM)
echo "Waiting for GPU memory to free up..."
while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    if [ "$FREE" -gt 35000 ]; then
        echo "  GPU has ${FREE}MiB free. Starting vLLM."
        break
    fi
    echo "  $(date '+%H:%M') GPU free: ${FREE}MiB (need 35000+). Waiting 60s..."
    sleep 60
done

# Run vLLM observation for remaining domains
for domain in books electronics movies sports toys; do
    OUTPUT="outputs/server_observations/qwen3_8b/${domain}"
    INPUT="outputs/observation_inputs/${domain}_test_forced_json.jsonl"

    if [ -f "${OUTPUT}/manifest.json" ]; then
        echo "$domain: already complete, skipping."
        continue
    fi

    echo ""
    echo "=== Starting vLLM observation: $domain ==="
    echo "Time: $(date)"
    rm -rf "$OUTPUT"
    mkdir -p "$OUTPUT"

    CUDA_VISIBLE_DEVICES=0 python scripts/server/run_vllm_observation.py \
        --config configs/server/qwen3_8b_observation.yaml \
        --input-jsonl "$INPUT" \
        --output-dir "$OUTPUT" \
        --gpu-memory-utilization 0.85

    if [ -f "${OUTPUT}/manifest.json" ]; then
        GROUNDED=$(python3 -c "import json; m=json.load(open('${OUTPUT}/manifest.json')); print(m.get('total_grounded_count',0))")
        echo "$domain: COMPLETE ($GROUNDED grounded predictions)"
    else
        echo "$domain: FAILED"
    fi
done

echo ""
echo "=== ALL OBSERVATIONS COMPLETE ==="
echo "End: $(date)"
