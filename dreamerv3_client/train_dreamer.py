import sys
import os
import gym
import gym_fightingice
import numpy as np
import warnings
from functools import partial
import pathlib
# DreamerV3のパスが通っていない場合のための処理
sys.path.append('/app/dreamer')
try:
    import dreamer
    from dreamer import embodied
except ImportError:
    print("Error: dreamerv3 module not found. Make sure you are in the correct directory.")
    sys.exit(1)

# ----------------------------------------------------------------
# 1. 通信先の強制書き換え (Monkey Patch)
# ----------------------------------------------------------------
# gym-fightingiceは通常 localhost 固定なので、Gatewayを書き換えて
# Dockerのホスト名 (fightingice) に繋ぐようにします。
from pyftg.socket.aio.gateway import Gateway
original_init = Gateway.__init__

def patched_gateway_init(self, host="127.0.0.1", port=31415):
    # 環境変数があればそれを優先、なければ引数、それもなければデフォルト
    target_host = os.environ.get("FIGHTINGICE_HOST", "fightingice")
    target_port = int(os.environ.get("FIGHTINGICE_PORT", 31415))
    print(f"[Wrapper] Connecting to FightingICE Server at {target_host}:{target_port}")
    original_init(self, host=target_host, port=target_port)

Gateway.__init__ = patched_gateway_init

# ----------------------------------------------------------------
# 2. 環境作成関数 (DreamerV3用アダプター)
# ----------------------------------------------------------------
def make_env(task, config, idx):
    # Javaを自分で起動しない設定 (java_env_path="")
    env = gym.make(
        task, 
        java_env_path="", 
        port=31415, 
        freq_restart_java=0
    )
    
    # Gymのラッパーを使ってDreamerV3形式に変換
    # obs_key='vector' : FightingICEの数値データを 'vector' という名前でDreamerに渡す
    env = embodied.envs.from_gym.FromGym(env, obs_key='vector')
    
    # DreamerV3に必要なメタデータを追加
    env = embodied.wrappers.ConvertTo32Bit(env)
    return env

# ----------------------------------------------------------------
# 3. メイン学習ループ
# ----------------------------------------------------------------
def main():
    # 警告の抑制
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # 設定の読み込み (サイズは 'small', 'medium', 'large' から選択)
    # CPUのみで動かす場合は 'debug' や 'small' を推奨
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['small'])
    
    # ログ保存場所
    logdir = embodied.Path('./log/dreamer_fightingice')
    
    # 設定の上書き (必要に応じて)
    config = config.update({
        'logdir': str(logdir),
        'run.train_ratio': 64,   # 環境1ステップあたりの学習回数
        'run.log_every': 60,     # ログ出力頻度(秒)
        'batch_size': 16,        # バッチサイズ
       # 'jax.policy_devices': ['cpu'], # GPUがない場合
       # 'jax.train_devices': ['cpu'],  # GPUがない場合
    })

    print("Logdir:", logdir)
    logdir.mkdirs()
    
    # ロガーの設定
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        # TensorBoardを使いたい場合は以下を有効化
        embodied.logger.TensorBoardOutput(logdir),
    ])

    # 環境ID (画像なし・数値データのみ版を使用)
    task = "FightingiceDataNoFrameskip-v0"

    # DreamerV3の学習開始
    # env_fns: 環境を作成する関数のリスト（並列化のためリストで渡す）
    env_fns = [partial(make_env, task, config, i) for i in range(config.envs)]
    
    # リプレイバッファ
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay')

    # エージェント初期化のためのダミー環境作成
    tmp_env = env_fns[0]()
    agent = dreamerv3.Agent(tmp_env.obs_space, tmp_env.act_space, step, config)
    tmp_env.close()

    # 学習実行
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length
    )
    
    embodied.run.train(agent, env_fns, replay, logger, args)

if __name__ == '__main__':
    main()