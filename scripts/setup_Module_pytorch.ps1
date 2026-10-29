# Module_pytorch 训练环境安装脚本
# 用法：在项目根目录执行 powershell -ExecutionPolicy Bypass -File .\scripts\setup_Module_pytorch.ps1
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvName = "Module_pytorch"
$CondaRoot = Split-Path (Split-Path (Get-Command conda).Source -Parent) -Parent
$Python = Join-Path $CondaRoot "envs\$EnvName\python.exe"
$TorchWheel = Join-Path $env:TEMP "torch-2.4.1+cu121-cp38-cp38-win_amd64.whl"
$TorchUrl = "https://mirrors.aliyun.com/pytorch-wheels/cu121/torch-2.4.1%2Bcu121-cp38-cp38-win_amd64.whl"
$TorchBytes = 2444894201

Set-Location $ProjectRoot

# 创建 Python 3.8.19 环境；--no-plugins 规避本机 Conda CUDA 插件权限错误
if (-not (Test-Path -LiteralPath $Python)) {
    conda --no-plugins create --solver classic -n $EnvName python=3.8.19 pip -y
    if ($LASTEXITCODE -ne 0) { throw "Conda 环境创建失败" }
}

& $Python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip 基础工具安装失败" }

# 仅当 CUDA 版 torch 尚不可用时下载，分片可跨网络中断续传
& $Python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    if (Test-Path -LiteralPath $TorchWheel) {
        $existingBytes = (Get-Item -LiteralPath $TorchWheel).Length
        if ($existingBytes -gt $TorchBytes) {
            throw "临时 torch 分片大于预期，请删除后重试：$TorchWheel"
        }
    }
    if (-not (Test-Path -LiteralPath $TorchWheel) -or
        (Get-Item -LiteralPath $TorchWheel).Length -lt $TorchBytes) {
        curl.exe -L --fail --retry 50 --retry-delay 5 --retry-all-errors `
            --connect-timeout 30 --continue-at - --output $TorchWheel $TorchUrl
        if ($LASTEXITCODE -ne 0) { throw "PyTorch CUDA 轮子下载失败，可重跑本脚本续传" }
    }
    if ((Get-Item -LiteralPath $TorchWheel).Length -ne $TorchBytes) {
        throw "PyTorch CUDA 轮子大小不完整：$TorchWheel"
    }
    & $Python -m pip install --timeout 300 --retries 50 $TorchWheel
    if ($LASTEXITCODE -ne 0) { throw "PyTorch CUDA 轮子安装失败" }
}

# 项目锁定依赖 + Parquet 读写所需的 pyarrow
& $Python -m pip install --timeout 300 --retries 50 -r requirements.txt pyarrow==17.0.0
if ($LASTEXITCODE -ne 0) { throw "项目依赖安装失败" }

& $Python -m pip check
if ($LASTEXITCODE -ne 0) { throw "依赖一致性检查失败" }

# 验证 CUDA、Parquet 读取与项目配置导入
& $Python -c "import torch, transformers, pyarrow, pandas; assert torch.cuda.is_available(), 'CUDA 不可用'; print('torch', torch.__version__); print('CUDA', torch.version.cuda); print('GPU', torch.cuda.get_device_name(0)); print('transformers', transformers.__version__); print('pyarrow', pyarrow.__version__); print('rows', len(pandas.read_parquet('data/interim/all.parquet')))"
if ($LASTEXITCODE -ne 0) { throw "训练环境验证失败" }

& $Python -c "from src.common.config import Config; cfg = Config.load(); print('backbone', cfg.backbone); print('max_length', cfg.max_length); print('batch_size', cfg.base['data']['batch_size'])"
if ($LASTEXITCODE -ne 0) { throw "项目配置导入失败" }

Write-Host "Module_pytorch 安装完成。激活命令：conda activate Module_pytorch" -ForegroundColor Green
