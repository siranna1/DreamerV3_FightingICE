#!/bin/sh
set -e

echo "--- DreamerV3 Setup for FightingICE ---"

# デバッグ情報: GPUとJAXのバージョン確認
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Detected:"
    nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv || true
else
    echo "Mode: CPU Only (or nvidia-smi not found)"
fi

echo "JAX Version:"
pip freeze | grep -E "jax|nvidia" || echo "JAX/Nvidia packages not found in pip."
echo "---------------------------------------"

# 【重要】Xvfb (仮想ディスプレイ) 経由でコマンドを実行
# DreamerV3がリプレイ動画（学習の様子）を生成する際に画面出力が必要になるため、これが必要です。
# "$@" は Dockerfile の CMD で指定したコマンド (python train_dreamer.py) に置き換わります。
#exec xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@"
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@"

