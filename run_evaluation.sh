#!/bin/bash

# ==========================================
# 対戦相手のリスト (ここを自由に変えてください)
# スペース区切りで並べるか、配列形式で記述
# ==========================================
OPPONENTS=("BlackMamba" "MctsAi23i" "IBM_AI" "EggTart")

# ==========================================
# メインループ
# ==========================================
for ai in "${OPPONENTS[@]}"; do
    echo ""
    echo "#########################################################"
    echo " Starting 10 games against: $ai"
    echo "#########################################################"
    echo ""

    # ユーザー指定のDockerコマンド
    # --ai2 の部分だけ変数 $ai に置き換えています
    docker run --rm -it \
        --name j2130dreamer \
        --network fighting-net \
        --gpus '"device=1"' \
        --shm-size=4g \
        -e FIGHTINGICE_HOST=fightingice \
        -e FIGHTINGIOCE_PORT=31415 \
        -v ~/research/dreamerv3:/app/dreamerv3 \
        -v ~/research/dreamerv3_client/log:/app/log \
        -v ~/research/dreamerv3_client:/app \
        j2130/dreamerv3 \
        python play_dreamer.py \
        --host fightingice \
        --logdir /app/log/003 \
        --port 31415 \
        --ai2 "$ai" \
        --games 10

    echo "Finished session against $ai."
    
    # 連続実行によるトラブル防止のため少し待機（任意）
    sleep 5
done

echo "All evaluation games finished!"