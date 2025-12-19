#FROM ubuntu/jre:17-22.04_edge
From ubuntu:22.04

WORKDIR /app
USER root
ENV DEBIAN_FRONTEND=noninteractive

# Java 17 (OpenJDK) をインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jdk && \
    rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME /usr/lib/jvm/java-21-openjdk-amd64
ENV PATH $PATH:$JAVA_HOME/bin

# Xvfbと必要なライブラリをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        xvfb \
        libgl1 \
        libsm6 libxext6 libxrender1 \
        fontconfig \
    && rm -rf /var/lib/apt/lists/*

#Fightingice.jarのコンパイル
COPY ./src ./src

#COPY ./FightingICE.jar .
#COPY ./data/ai ./data/ai
#COPY ./data/characters/ZEN/gSetting.txt ./data/characters/ZEN/gSetting.txt
#COPY ./data/characters/ZEN/Motion.csv ./data/characters/ZEN/Motion.csv
COPY ./data ./data
COPY ./lib/*.jar ./lib/
COPY ./lib/lwjgl/*.jar ./lib/
COPY ./lib/lwjgl/natives/linux/amd64 /app/lib/
#COPY ./run_ice.sh /usr/local/bin/run_ice.sh
#RUN chmod +x /usr/local/bin/run_ice.sh

#コンパイルの実行
RUN mkdir bin && \
    javac -encoding UTF-8 -d bin -cp "lib/*" $(find src -name "*.java") 

RUN echo '#!/bin/sh\n\
rm -f /tmp/.X1-lock\n\
xvfb-run -a -s "-screen 0 1024x768x24" java -cp "bin:lib/*" Main "$@"\n"\
' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--limithp", "400", "400", "--pyftg-mode", "--gray-bg"]


EXPOSE 31415/tcp
EXPOSE 4242/tcp
#音声ドライバーをnullにする
ENV ALSOFT_DRIVERS=null
 # Javaの実行引数を定義
# ENV JAVA_ARGS "-cp FightingICE.jar:./lib/* Main --lightweight-mode"

#ENTRYPOINT [ "/opt/java/bin/java", "-cp", "FightingICE.jar:./lib/*", "Main", "--lightweight-mode" ]
#CMD [ "--limithp", "400", "400", "--pyftg-mode" ]

# ENTRYPOINTでxvfb-runを設定し、Javaアプリケーションを仮想ディスプレイで起動
# -a: 自動的に利用可能なディスプレイ番号を選択
# -s "-screen 0 1024x768x24": 仮想スクリーンの設定
#ENTRYPOINT [ "xvfb-run", "-a", "-s", "-screen 0 1024x768x24"]
#xvfb-run -a java -cp FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/linux/amd64/*:./lib/grpc/* Main --limithp 400 400 --grey-bg --pyftg-mode

# CMDでJavaの引数（FightingICEのクラスパスとオプション）を渡す
# --disable-window はウィンドウ表示を抑制するオプションとして追加
#CMD ["java", "-Xmx1G", "-Xms512M", "-Xmn256M","-cp", "FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/linux/amd64/*:./lib/grpc/*", "Main", "--disable-window", "--limithp", "400", "400", "--pyftg-mode", "--fastmode", "--mute" ]
#CMD ["sh", "-c", "java -Xmx1G -Xms512M -Xmn256M -cp 'FightingICE.jar:./lib/*:./lib/lwjgl/*:./lib/lwjgl/natives/linux/amd64/*:./lib/grpc/*' Main --disable-window --limithp 400 400 --pyftg-mode --fastmode --mute"]

