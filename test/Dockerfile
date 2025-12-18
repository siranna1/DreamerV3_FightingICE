# ベースイメージ
FROM python:3.11-slim

# 作業ディレクトリ
WORKDIR /app

# 必要なライブラリをインストール
RUN pip install --no-cache "pyftg>=2.0"

# スクリプトをコピー
COPY test_connect.py .

# 実行コマンド
# ここでは直接スクリプトを動かします
#CMD ["python", "test_connect.py"]
