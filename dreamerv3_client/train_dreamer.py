import sys
import os
import pathlib
import warnings
from functools import partial as bind

# ----------------------------------------------------------------
# 1. パスとインポート設定 (main.py の構成に合わせる)
# ----------------------------------------------------------------
# リポジトリのルート (/app/dreamerv3) をパスに追加して
# local の embodied, dreamerv3 パッケージを読み込めるようにする
repo_root = pathlib.Path('/app/dreamerv3')
sys.path.insert(0, str(repo_root))

try:
    import elements
    import embodied
    import dreamerv3
    from dreamerv3 import agent as agent_module
    import ruamel.yaml as yaml
    import gym
    import gym_fightingice
    from pyftg.socket.aio.gateway import Gateway
except ImportError as e:
    print(f"Import Error: {e}")
    print("必要なライブラリ (elements, gym, gym-fightingice, pyftg, ruamel.yaml) がインストールされているか確認してください。")
    sys.exit(1)

# ----------------------------------------------------------------
# 2. FightingICE 接続用モンキーパッチ
# ----------------------------------------------------------------
original_init = Gateway.__init__

def patched_gateway_init(self, host="127.0.0.1", port=31415):
    target_host = os.environ.get("FIGHTINGICE_HOST", "fightingice")
    target_port = int(os.environ.get("FIGHTINGICE_PORT", 31415))
    print(f"[Wrapper] Connecting to FightingICE Server at {target_host}:{target_port}")
    original_init(self, host=target_host, port=target_port)

Gateway.__init__ = patched_gateway_init

# ----------------------------------------------------------------
# 3. 各種 Factory 関数の定義 (embodied.run.train が要求する形式)
# ----------------------------------------------------------------

def make_env(config, index, **overrides):
    """環境を作成する関数"""
    # FightingICE環境の作成
    # java_env_path="" によりJavaを起動せず、既存のサーバーに接続
    env = gym.make(
        "FightingiceDataNoFrameskip-v0", 
        java_env_path="", 
        port=31415, 
        freq_restart_java=0
    )
    
    # Gymラッパー: DreamerV3形式に変換 (obs_key='vector' で数値データを扱う)
    env = embodied.envs.from_gym.FromGym(env, obs_key='vector')
    env = embodied.wrappers.ConvertTo32Bit(env)
    return env

def make_agent(config):
    """エージェントを作成する関数"""
    # エージェントの初期化に必要な空間情報を取得するために、一時的に環境を作る
    env = make_env(config, 0)
    obs_space = env.obs_space
    act_space = env.act_space
    env.close()

    # dreamerv3.agent.Agent の初期化
    return agent_module.Agent(obs_space, act_space, config)

def make_replay(config, folder='replay', mode='train'):
    """リプレイバッファを作成する関数 (main.py のロジックを移植)"""
    logdir = elements.Path(config.logdir)
    directory = logdir / folder
    
    # メモリ容量などの計算
    capacity = config.replay.size
    length = config.batch_length * config.consec_train + config.replay_context
    
    # embodied.replay.Replay の作成
    return embodied.replay.Replay(
        length=length, 
        capacity=int(capacity), 
        online=config.replay.online,
        chunksize=config.replay.chunksize, 
        directory=directory
    )

def make_logger(config):
    """ロガーを作成する関数"""
    logdir = elements.Path(config.logdir)
    step = elements.Counter()
    
    outputs = [
        elements.logger.TerminalOutput(),
        elements.logger.JSONLOutput(logdir, 'metrics.jsonl'),
    ]
    # TensorBoardを使いたい場合はコメントアウトを外す
    # outputs.append(elements.logger.TensorBoardOutput(logdir))

    return elements.Logger(step, outputs, multiplier=1)

# ----------------------------------------------------------------
# 4. メイン処理
# ----------------------------------------------------------------
def main():
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # 設定ファイルの読み込み
    # configs.yaml は repo_root/dreamerv3/configs.yaml にあります
    config_path = repo_root / 'dreamerv3' / 'configs.yaml'
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    print(f"Loading configs from: {config_path}")
    configs_text = elements.Path(config_path).read()
    configs = yaml.YAML(typ='safe').load(configs_text)
    
    # 'defaults' 設定をベースにする
    config = elements.Config(configs['defaults'])
    
    # 必要に応じて他の設定 (debugなど) をマージ
    # config = config.update(configs['debug']) # デバッグ時はこれを有効化
    
    # FightingICE用の設定上書き
    logdir = elements.Path('./log/dreamer_fightingice')
    config = config.update({
        'logdir': str(logdir),
        'task': 'fightingice_custom',
        'run.train_ratio': 64,   # 学習の頻度
        'run.log_every': 60,     # ログ出力間隔(秒)
        'batch_size': 16,
        'batch_length': 64,
        # GPU設定 (自動検出されるため明示的に指定しなくても動くことが多いですが念のため)
        # 'jax.policy_devices': [0],
    })

    print("Logdir:", logdir)
    logdir.mkdir()
    
    # ストリーム作成関数 (Replayからバッチを取り出す処理)
    def make_stream(config, replay, mode):
        return embodied.run.train.make_stream(config, replay, mode) #

    # 学習実行用引数の準備
    args = elements.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length
    )

    print("Starting training...")
    
    # embodied.run.train の呼び出し
    # ここでは関数(factory)を渡すのがポイントです
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args
    )

if __name__ == '__main__':
    main()
