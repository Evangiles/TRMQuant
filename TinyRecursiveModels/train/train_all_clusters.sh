#!/bin/bash
# Train all clusters sequentially with epoch 10

echo "================================================================================"
echo "TRAINING ALL CLUSTERS (0-6)"
echo "================================================================================"
echo ""

for cluster_id in {0..6}
do
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "Training Cluster ${cluster_id}"
    echo "────────────────────────────────────────────────────────────────────────────"

    uv run TinyRecursiveModels/train/train_group_denoiser.py \
        --cluster_id ${cluster_id} \
        --epochs 10 \
        --batch_size 64

    if [ $? -ne 0 ]; then
        echo "❌ Cluster ${cluster_id} training failed!"
        exit 1
    fi

    echo "✅ Cluster ${cluster_id} completed"
    echo ""
done

echo "================================================================================"
echo "ALL CLUSTERS TRAINED SUCCESSFULLY"
echo "================================================================================"
