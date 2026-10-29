#!/usr/bin/env bash
# =============================================================================
# 云端 GPU 环境安装 / 自检（Linux，RTX 3080 等）
#
# 用法（在项目根目录）：
#   bash scripts/setup_cloud.sh
#
# 可选环境变量：
#   PYTHON=python3.10                 指定解释器
#   TORCH_VERSION=2.4.1               需要安装 torch 时使用的版本
#   TORCH_INDEX_URL=.../whl/cu121     对应镜像 CUDA 的 wheel 源
#   DOWNLOAD_MODEL=0                  跳过 hfl 模型下载
#   HF_ENDPOINT=https://hf-mirror.com 国内镜像（默认已设置）
#
# 行为：
#   1) 检查 Python 3.8-3.11（requirements 的 numpy/pandas/scipy pin 限制）
#   2) 复用镜像自带 GPU torch；不可用时按 TORCH_INDEX_URL 安装
#   3) 安装 requirements-cloud.txt（不覆盖 torch）
#   4) 按需生成 interim/splits，运行协议检查与测试
#   5) 按需下载 hfl/chinese-roberta-wwm-ext
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "==> 项目根目录：$PROJECT_ROOT"
echo "==> Python：$("$PYTHON_BIN" -V 2>&1)"

"$PYTHON_BIN" - <<'PY'
import sys
v = sys.version_info[:2]
if not ((3, 8) <= v <= (3, 11)):
    raise SystemExit(
        f"需要 Python 3.8-3.11，当前 {sys.version.split()[0]}。"
        "请新建环境（如 conda create -n stance python=3.10），"
        "或放宽 requirements-cloud.txt 中 numpy/pandas/scipy 的版本。"
    )
print("Python 版本检查通过")
PY

echo ""
echo "==> 检查 GPU / torch"
if "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA 不可用"
print(f"复用镜像 torch: {torch.__version__} / CUDA {torch.version.cuda} / "
      f"{torch.cuda.get_device_name(0)}")
PY
then
    echo "==> 复用现有 GPU torch"
else
    echo "==> 未检测到可用 GPU torch，安装 torch==${TORCH_VERSION}"
    echo "    index: ${TORCH_INDEX_URL}"
    "$PYTHON_BIN" -m pip install --timeout 300 --retries 20 \
        "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
fi

echo ""
echo "==> 安装训练依赖（requirements-cloud.txt，不含 torch）"
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install --timeout 300 --retries 20 -r requirements-cloud.txt
"$PYTHON_BIN" -m pip check

echo ""
echo "==> 准备数据（interim / splits 不进 git，需要本地重建）"
if [ ! -f data/interim/all.parquet ]; then
    "$PYTHON_BIN" -m src.data.prepare
else
    echo "    已存在 data/interim/all.parquet，跳过 prepare"
fi
if [ ! -f data/processed/split_manifest.json ]; then
    "$PYTHON_BIN" -m src.data.split
else
    echo "    已存在 data/processed/split_manifest.json，跳过 split"
fi

echo ""
echo "==> 协议一致性检查"
"$PYTHON_BIN" scripts/check_protocol.py

echo ""
echo "==> 运行测试"
"$PYTHON_BIN" -m pytest -q

if [ "$DOWNLOAD_MODEL" = "1" ]; then
    echo ""
    echo "==> 下载底座模型 hfl/chinese-roberta-wwm-ext（HF_ENDPOINT=$HF_ENDPOINT）"
    "$PYTHON_BIN" - <<'PY'
from transformers import AutoModel, AutoTokenizer
name = "hfl/chinese-roberta-wwm-ext"
AutoTokenizer.from_pretrained(name)
AutoModel.from_pretrained(name)
print(f"模型已就绪：{name}")
PY
else
    echo ""
    echo "==> DOWNLOAD_MODEL=0，跳过模型下载"
fi

echo ""
echo "==> 环境自检"
"$PYTHON_BIN" - <<'PY'
import torch, transformers, pandas, pyarrow
from src.common.config import Config
cfg = Config.load(dg="baseline.yaml")
print(f"torch={torch.__version__} cuda={torch.version.cuda} "
      f"available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__} pyarrow={pyarrow.__version__}")
print(f"backbone={cfg.backbone} max_length={cfg.max_length} "
      f"batch_size={cfg.base['data']['batch_size']} fp16={cfg.base['training']['fp16']}")
print(f"baseline loss={cfg.dg['loss']}")
PY

cat <<'EOF'

=====================================================================
环境就绪。建议先跑单折基线（后台执行，避免 SSH 断连）：
  nohup python -m src.train.run_fold --subtask B --fold 环保 --seed 42 \
      --loss-config baseline > outputs/logs/B_环保_seed42.log 2>&1 &

产物：
  outputs/ckpt/B_环保_seed42.pt
  outputs/tables/B_环保_seed42_metrics.json
  outputs/predictions/B_环保_seed42_predictions.parquet

多种子：
  bash scripts/run_all_seeds.sh B 环保 42 43 44 45 46
  LOSS_CONFIG=dg bash scripts/run_all_seeds.sh B 环保 42 43 44 45 46
=====================================================================
EOF
