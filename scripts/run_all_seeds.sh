#!/usr/bin/env bash
# 批量跑随机种子。
# 用法：
#   bash scripts/run_all_seeds.sh B 环保 42 43 44 45 46    # subtask B 的某个留出域
#   bash scripts/run_all_seeds.sh A "" 42 43 44 45 46      # subtask A（无 fold）
set -euo pipefail

SUBTASK=${1:-B}
FOLD=${2:-}
shift 2 2>/dev/null || shift $#   # 兼容空 fold
SEEDS="${*:-42 43 44 45 46}"

echo "=== subtask=$SUBTASK fold=${FOLD:-all} seeds=[$SEEDS] ==="
for seed in $SEEDS; do
  echo ""
  echo ">>> seed=$seed"
  if [ -n "$FOLD" ]; then
    python -m src.train.run_fold --subtask "$SUBTASK" --fold "$FOLD" --seed "$seed"
  else
    python -m src.train.run_fold --subtask "$SUBTASK" --seed "$seed"
  fi
done
echo ""
echo "=== 完成 ==="
