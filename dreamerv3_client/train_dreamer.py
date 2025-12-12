import sys
import os
import pathlib
import warnings
import threading
import queue
import time
import numpy as np
import gym
from gym import spaces
import ruamel.yaml as yaml
from functools import partial as bind

# --- PyFTG 関連のインポート ---
from pyftg import AIInterface, FrameData, AudioData, RoundResult, ScreenData, Key, GameData, CommandCenter
from pyftg.socket.aio.gateway import Gateway
import asyncio

# ----------------------------------------------------------------
# 1. パス設定
# ----------------------------------------------------------------
repo_root = pathlib.Path('/app/dreamerv3')
sys.path.insert(0, str(repo_root))

try:
    import elements
    import embodied
    from embodied.envs import from_gym
    import dreamerv3
    from dreamerv3 import agent as agent_module
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# ----------------------------------------------------------------
# 2. カスタム Gym 環境 (PyFTG ラッパー)
# ----------------------------------------------------------------

ACTION_MAP = [
    "STAND_B", "CROUCH_B", "STAND_A", "CROUCH_A", 
    "FORWARD_WALK", "BACK_STEP", "JUMP", "CROUCH", "STAND",
    "STAND_FA", "STAND_FB", "CROUCH_FA", "CROUCH_FB",
    "THROW_A", "THROW_B"
]

class PyFTGAgent(AIInterface):
    def __init__(self, obs_queue, act_queue, name="DreamerAI"):
        super().__init__()
        self.obs_queue = obs_queue
        self.act_queue = act_queue
        self.cc = CommandCenter()
        self.key = Key()
        self.player = True
        self.game_started = False
        self.frame_data = None
        self.agent_name = name

    def name(self) -> str: return self.agent_name
    def is_blind(self) -> bool: return False

    def initialize(self, game_data: GameData, player: bool):
        print(f"[{self.agent_name}] Initialize called. PlayerID: {player}")
        self.player = player
        self.cc = CommandCenter()
        self.game_started = True
        self.key = Key()

    def close(self): pass
    def get_audio_data(self, audio_data: AudioData): pass
    def get_non_delay_frame_data(self, frame_data: FrameData): pass
    def get_screen_data(self, screen_data: ScreenData): pass

    def get_information(self, frame_data: FrameData, is_control: bool):
        self.frame_data = frame_data
        self.cc.set_frame_data(frame_data, self.player)

    def processing(self):
        if not self.game_started or self.frame_data is None or self.frame_data.empty_flag or self.frame_data.current_frame_number < 0:
            return

        obs = self._extract_observation(self.frame_data)
        
        while not self.obs_queue.empty():
            try: self.obs_queue.get_nowait()
            except queue.Empty: pass
        self.obs_queue.put(obs)

        try:
            action_idx = self.act_queue.get(timeout=0.001)
            command = ACTION_MAP[action_idx]
            self.cc.command_call(command)
        except queue.Empty:
            pass

        self.key = self.cc.get_skill_key()

    def input(self) -> Key:
        return self.key

    def round_end(self, result: RoundResult): print(f"[{self.agent_name}] Round End")
    def game_end(self): print(f"[{self.agent_name}] Game End")

    def _extract_observation(self, fd: FrameData):
        me = fd.get_character(self.player)
        opp = fd.get_character(not self.player)
        
        me_hp = me.hp if me else 0
        me_energy = me.energy if me else 0
        me_x = me.x if me else 0
        me_y = me.y if me else 0
        me_sx = me.speed_x if me else 0
        me_sy = me.speed_y if me else 0
        me_air = 1.0 if me and me.state == "AIR" else 0.0

        opp_hp = opp.hp if opp else 0
        opp_energy = opp.energy if opp else 0
        opp_x = opp.x if opp else 0
        opp_y = opp.y if opp else 0
        opp_sx = opp.speed_x if opp else 0
        opp_sy = opp.speed_y if opp else 0
        
        obs = np.array([
            me_hp / 400.0,
            me_energy / 300.0,
            me_x / 960.0,
            me_y / 640.0,
            me_sx / 20.0,
            me_sy / 20.0,
            me_air,
            opp_hp / 400.0,
            opp_energy / 300.0,
            opp_x / 960.0,
            opp_y / 640.0,
            opp_sx / 20.0,
            opp_sy / 20.0,
            (me_x - opp_x) / 960.0,
            (me_y - opp_y) / 640.0
        ], dtype=np.float32)
        return obs

class FightingIceGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTION_MAP))
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(15,), dtype=np.float32)
        
        self.obs_queue = queue.Queue()
        self.act_queue = queue.Queue()
        
        self.thread = threading.Thread(target=self._run_pyftg, daemon=True)
        self.thread.start()
        
        print("Waiting for FightingICE connection and Game Start...")
        try:
            self.current_obs = self.obs_queue.get(timeout=30) 
            print("FightingICE Game Started!")
        except queue.Empty:
            print("Error: Connection timed out. The game did not start in 30 seconds.")
            self.current_obs = np.zeros(15, dtype=np.float32)

    def _run_pyftg(self):
        async def main_loop():
            host = os.environ.get("FIGHTINGICE_HOST", "fightingice")
            port = int(os.environ.get("FIGHTINGICE_PORT", 31415))
            
            print(f"Connecting to {host}:{port}...")
            gateway = Gateway(host=host, port=port)
            
            agent1 = PyFTGAgent(self.obs_queue, self.act_queue, "DreamerAI")
            gateway.register_ai("DreamerAI", agent1)
            try:
                # サーバー側の MctsAi23i と対戦
                print("Requesting Game Start: DreamerAI vs MctsAi23i")
                await gateway.run_game(["ZEN", "ZEN"], ["DreamerAI", "MctsAi23i"], 1000)
            except Exception as e:
                print(f"PyFTG Error: {e}")
            finally:
                await gateway.close()

        asyncio.run(main_loop())

    def reset(self, seed=None, options=None):
        if not self.obs_queue.empty():
            self.current_obs = self.obs_queue.get()
        return self.current_obs, {}

    def step(self, action):
        self.act_queue.put(action)
        try:
            self.current_obs = self.obs_queue.get(timeout=2.0)
            reward = 0.0 
            return self.current_obs, reward, False, False, {}
        except queue.Empty:
            return self.current_obs, 0.0, True, False, {}

# ----------------------------------------------------------------
# 3. Factory 関数
# ----------------------------------------------------------------

def make_env(config, index, **overrides):
    env = FightingIceGymEnv()
    env = from_gym.FromGym(env, obs_key='vector')
    env = embodied.wrappers.UnifyDtypes(env)
    return env

def make_agent(config):
    # ダミー環境を作成して空間定義を取得
    class DummyEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.Discrete(len(ACTION_MAP))
            self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(15,), dtype=np.float32)
    
    env = DummyEnv()
    env = from_gym.FromGym(env, obs_key='vector')
    env = embodied.wrappers.UnifyDtypes(env)
    
    # 'reset' を除外したアクション空間を作成
    notlog = lambda k: not k.startswith('log/')
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}

    # エージェント用設定の再構築
    agent_config = elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    )
    
    # 【重要】除外済みの act_space を渡す
    return agent_module.Agent(obs_space, act_space, agent_config)

def make_replay(config, folder='replay', mode='train'):
    logdir = elements.Path(config.logdir)
    directory = logdir / folder
    capacity = config.replay.size
    length = config.batch_length * config.consec_train + config.replay_context
    return embodied.replay.Replay(
        length=length, 
        capacity=int(capacity), 
        online=config.replay.online,
        chunksize=config.replay.chunksize, 
        directory=directory
    )

def make_logger(config):
    logdir = elements.Path(config.logdir)
    step = elements.Counter()
    outputs = [
        elements.logger.TerminalOutput(),
        elements.logger.JSONLOutput(logdir, 'metrics.jsonl'),
    #tensorboardの設定頑張って
    #    elements.logger.TensorBoardOutput(logdir),
    ]
    return elements.Logger(step, outputs, multiplier=1)

# 【修正】make_stream を正しく実装
def make_stream(config, replay, mode):
    fn = bind(replay.sample, config.batch_size, mode)
    stream = embodied.streams.Stateless(fn)
    stream = embodied.streams.Consec(
        stream,
        length=config.batch_length if mode == 'train' else config.report_length,
        consec=config.consec_train if mode == 'train' else config.consec_report,
        prefix=config.replay_context,
        strict=(mode == 'train'),
        contiguous=True)
    return stream

# ----------------------------------------------------------------
# 4. メイン処理
# ----------------------------------------------------------------
def main():
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    config_path = repo_root / 'dreamerv3' / 'configs.yaml'
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    print(f"Loading configs from: {config_path}")
    configs_text = elements.Path(config_path).read()
    configs = yaml.YAML(typ='safe').load(configs_text)
    
    config = elements.Config(configs['defaults'])
    
    # FightingICE用のデフォルト設定
    fightingice_defaults = {
        'logdir': './log/dreamer_fightingice',
        'task': 'fightingice_custom',
        'run.train_ratio': 64,
        'run.log_every': 60,
        'batch_size': 16,
        'batch_length': 64,
    }
    config = config.update(fightingice_defaults)

    # 【修正】コマンドライン引数で設定を上書き (dreamer/main.py と同じ挙動)
    # これにより --logdir や --batch_size などが使えます
    config = elements.Flags(config).parse(sys.argv[1:])

    logdir = elements.Path(config.logdir)
    print("Logdir:", logdir)
    logdir.mkdir()
    
    # 学習実行用引数の準備（batch_sizeなどを明示的にコピー）
    args = elements.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,      # 【追加】
        batch_length=config.batch_length,  # 【追加】
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )

    print("Starting training...")
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