#!/usr/bin/env bash
# Stage 5: TP=1 reference, then TP=2 comparison.
set -e
echo "=============== TP=1 (reference) ==============="
python test_stage5.py
echo
echo "=============== TP=2 ==============="
torchrun --nproc_per_node=2 --master_port=29517 test_stage5.py
