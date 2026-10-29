#!/usr/bin/env bash
# 批量跑随机种子。
# 用法：
#   bash scripts/run_all_seeds.sh B 环保 42 43 44 45 46            # CE-only 基线
#   LOSS_CONFIG=dg bash scripts/run_all_seeds.sh B 环保 42 43 44 45 46  # 完整 DG
#   bash scripts/run_all_seeds.sh A "" 42 43 44 45 46               # 兼容 A（当前实验不用）
set -euo pipefail

SUBTASK=${1:-B}
FOLD=${2:-}
shift 2 2>/dev/null || shift $#   # 兼容空 fold
SEEDS="${*:-42 43 44 45 46}"
LOSS_CONFIG="${LOSS_CONFIG:-baseline}"

if [ "$SUBTASK" = "B" ] && [ -z "$FOLD" ]; then
  echo "Subtask B 必须指定留出域，例如：bash scripts/run_all_seeds.sh B 环保 42 43 44 45 46"
  exit 1
fi

echo "=== subtask=$SUBTASK fold=${FOLD:-all} loss=$LOSS_CONFIG seeds=[$SEEDS] ==="
for seed in $SEEDS; do
  echo ""
  echo ">>> seed=$seed"
  if [ -n "$FOLD" ]; then
    python -m src.train.run_fold --subtask "$SUBTASK" --fold "$FOLD" \
      --seed "$seed" --loss-config "$LOSS_CONFIG"
  else
    python -m src.train.run_fold --subtask "$SUBTASK" \
      --seed "$seed" --loss-config "$LOSS_CONFIG"
  fi
done
echo ""
echo "=== 完成 ==="
