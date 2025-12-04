#FROM ubuntu/jre:17-22.04_edge
From ubuntu:22.04

# Java 17 (OpenJDK) をインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Xvfbと必要なライブラリをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        xvfb \
        libgl1 \
        libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
USER root

COPY ./FightingICE.jar .
COPY ./data/ai ./data/ai
COPY ./data/characters/ZEN/gSetting.txt ./data/characters/ZEN/gSetting.txt
COPY ./data/characters/ZEN/Motion.csv ./data/characters/ZEN/Motion.csv
COPY ./lib/*.jar ./lib/
COPY ./lib/lwjgl/*.jar ./lib/

EXPOSE 31415/tcp
# Javaの実行引数を定義
ENV JAVA_ARGS "-cp FightingICE.jar:./lib/* Main --lightweight-mode"

#ENTRYPOINT [ "/opt/java/bin/java", "-cp", "FightingICE.jar:./lib/*", "Main", "--lightweight-mode" ]
#CMD [ "--limithp", "400", "400", "--pyftg-mode" ]

# ENTRYPOINTでxvfb-runを設定し、Javaアプリケーションを仮想ディスプレイで起動
# -a: 自動的に利用可能なディスプレイ番号を選択
# -s "-screen 0 1024x768x24": 仮想スクリーンの設定
ENTRYPOINT [ "xvfb-run", "-a", "-s", "-screen 0 1024x768x24", "/opt/java/bin/java" ]

# CMDでJavaの引数（FightingICEのクラスパスとオプション）を渡す
# --disable-window はウィンドウ表示を抑制するオプションとして追加
CMD [ "-Xmx1G", "-Xms512M", "-Xmn256M", "-Djava.library.path=./lib/lwjgl", "Main", "--lightweight-mode", "--disable-window", "--limithp", "400", "400", "--pyftg-mode" ]
