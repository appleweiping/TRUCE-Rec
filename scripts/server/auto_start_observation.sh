#!/bin/bash
# Auto-start TRUCE-Rec observation when GPU is free
# Waits for TGL-Rec LoRA to finish, then runs observation on all domains
set -euo pipefail

PROJECT_DIR="$HOME/projects/TRUCE-Rec"
cd "$PROJECT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vllm
export PYTHONPATH="$PROJECT_DIR/src"

echo "=== TRUCE-Rec Auto-Start Observation ==="
echo "Waiting for GPU to be free (TGL-Rec LoRA to finish)..."
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Wait for TGL-Rec LoRA to finish
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; do
    echo "  $(date '+%H:%M:%S') GPU still occupied. Waiting 60s..."
    sleep 60
done

echo ""
echo "GPU is FREE! Starting observations..."
echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Run observations sequentially (each loads model, runs, unloads)
DOMAINS="beauty books electronics movies"

for domain in $DOMAINS; do
    INPUT="outputs/observation_inputs/${domain}_test_forced_json.jsonl"
    OUTPUT="outputs/server_observations/qwen3_8b/${domain}"

    if [ -f "${OUTPUT}/manifest.json" ]; then
        echo "  $domain: already complete, skipping."
        continue
    fi

    if [ ! -f "$INPUT" ]; then
        echo "  $domain: observation input not found, skipping."
        continue
    fi

    echo ""
    echo "=== Starting observation: $domain ==="
    echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    rm -rf "$OUTPUT"
    mkdir -p "$OUTPUT"

    CUDA_VISIBLE_DEVICES=0 python scripts/server/run_qwen3_observation.py \
        --config configs/server/qwen3_8b_observation.yaml \
        --input-jsonl "$INPUT" \
        --output-dir "$OUTPUT" \
        --execute-server

    if [ -f "${OUTPUT}/grounded_predictions.jsonl" ]; then
        COUNT=$(wc -l < "${OUTPUT}/grounded_predictions.jsonl")
        echo "  $domain: COMPLETE ($COUNT predictions)"
    else
        echo "  $domain: FAILED (no grounded_predictions.jsonl)"
    fi
done

# Check if sports/toys are ready
for domain in sports toys; do
    INPUT="outputs/observation_inputs/${domain}_test_forced_json.jsonl"
    OUTPUT="outputs/server_observations/qwen3_8b/${domain}"

    if [ -f "${OUTPUT}/manifest.json" ]; then
        echo "  $domain: already complete, skipping."
        continue
    fi

    if [ ! -f "$INPUT" ]; then
        # Try to build observation inputs if preprocessing is done
        PROCESSED="data/processed/amazon_reviews_2023_${domain}/full"
        if [ -f "${PROCESSED}/observation_examples.jsonl" ]; then
            echo "  $domain: building observation inputs..."
            mkdir -p outputs/observation_inputs
            python scripts/build_observation_inputs.py \
                --dataset "amazon_reviews_2023_${domain}" \
                --processed-suffix full \
                --split test \
                --prompt-template forced_json \
                --output-jsonl "$INPUT" || {
                echo "  $domain: failed to build inputs, skipping."
                continue
            }
        else
            echo "  $domain: preprocessing not complete, skipping."
            continue
        fi
    fi

    echo ""
    echo "=== Starting observation: $domain ==="
    echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    rm -rf "$OUTPUT"
    mkdir -p "$OUTPUT"

    CUDA_VISIBLE_DEVICES=0 python scripts/server/run_qwen3_observation.py \
        --config configs/server/qwen3_8b_observation.yaml \
        --input-jsonl "$INPUT" \
        --output-dir "$OUTPUT" \
        --execute-server

    if [ -f "${OUTPUT}/grounded_predictions.jsonl" ]; then
        COUNT=$(wc -l < "${OUTPUT}/grounded_predictions.jsonl")
        echo "  $domain: COMPLETE ($COUNT predictions)"
    else
        echo "  $domain: FAILED"
    fi
done

echo ""
echo "=== All observations complete ==="
echo "End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
