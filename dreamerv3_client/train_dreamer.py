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

# 画像処理用
try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: opencv-python not found. Using slow numpy resizing.")

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
# 2. カスタム Gym 環境 (画像対応版)
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
        self.last_hp = 0
        self.last_opp_hp = 0

    def name(self) -> str: return self.agent_name
    def is_blind(self) -> bool: return False 

    def initialize(self, game_data: GameData, player: bool):
        print(f"[{self.agent_name}] Initialize (Game Start). PlayerID: {player}")
        self.player = player
        self.cc = CommandCenter()
        self.game_started = True
        self.key = Key()
        self.last_hp = 400
        self.last_opp_hp = 400

    def close(self): pass
    def get_audio_data(self, audio_data: AudioData): pass
    def get_non_delay_frame_data(self, frame_data: FrameData): pass
    
    # 【変更】画像データを受け取る
    def get_screen_data(self, screen_data: ScreenData):
        self.screen_data = screen_data

    def get_information(self, frame_data: FrameData, is_control: bool):
        self.frame_data = frame_data
        self.cc.set_frame_data(frame_data, self.player)

    def processing(self):
        if not self.game_started or self.frame_data is None or self.frame_data.empty_flag or self.frame_data.current_frame_number < 0:
            return

        # 1. 観測データの作成 (画像 + 数値)
        obs = {}
        
        # --- 画像処理 ---
        if hasattr(self, 'screen_data') and self.screen_data is not None:
            # PyFTGの仕様: display_bytes は bytes型
            # FightingICEの画像は 960x640, RGB (またはBGRA)
            raw_data = self.screen_data.display_bytes
            try:
                # バイト列をnumpy配列に変換
                img = np.frombuffer(raw_data, dtype=np.uint8)
                img = img.reshape((640, 960, 3)) # 高, 幅, ch
                
                # リサイズ (64x64)
                if cv2:
                    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
                else:
                    # CV2がない場合の簡易リサイズ (間引き)
                    img = img[::10, ::15] # 64x64に近いサイズに間引く
                    # 必要ならパディング等調整
                    
                obs['image'] = img
            except Exception as e:
                # 画像処理失敗時は黒画像
                print(f"Image Error: {e}")
                obs['image'] = np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            print("No screen data available.")
            print(f"hasattr(self, 'screen_data'): {hasattr(self, 'screen_data')}")
            print(f"self.screen_data is None: {getattr(self, 'screen_data', None) is None}")
            obs['image'] = np.zeros((64, 64, 3), dtype=np.uint8)

        # --- 報酬計算 ---
        me = self.frame_data.get_character(self.player)
        opp = self.frame_data.get_character(not self.player)
        current_hp = me.hp if me else 0
        current_opp_hp = opp.hp if opp else 0
        reward = (current_hp - self.last_hp) - (current_opp_hp - self.last_opp_hp)
        self.last_hp = current_hp
        self.last_opp_hp = current_opp_hp

        # キュー送信
        while not self.obs_queue.empty():
            try: self.obs_queue.get_nowait()
            except queue.Empty: pass
        
        self.obs_queue.put((obs, reward, False))

        # アクション実行
        try:
            action_idx = self.act_queue.get(timeout=0.001)
            command = ACTION_MAP[action_idx]
            self.cc.command_call(command)
        except queue.Empty:
            pass
        self.key = self.cc.get_skill_key()

    def input(self) -> Key: return self.key

    def round_end(self, result: RoundResult):
        print(f"[{self.agent_name}] Round End")
        dummy_obs = {'image': np.zeros((64, 64, 3), dtype=np.uint8)}
        self.obs_queue.put((dummy_obs, 0.0, True))
        self.last_hp = 400
        self.last_opp_hp = 400

    def game_end(self):
        print(f"[{self.agent_name}] Game End")
        dummy_obs = {'image': np.zeros((64, 64, 3), dtype=np.uint8)}
        self.obs_queue.put((dummy_obs, 0.0, True))

class FightingIceGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(ACTION_MAP))
        # 【変更】観測空間を画像に変更 (64x64 RGB)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        })
        
        self.obs_queue = queue.Queue()
        self.act_queue = queue.Queue()
        self.current_obs = {'image': np.zeros((64, 64, 3), dtype=np.uint8)}
        
        self.thread = threading.Thread(target=self._run_pyftg, daemon=True)
        self.thread.start()
        print("Waiting for FightingICE connection...")

    def _run_pyftg(self):
        async def main_loop():
            host = os.environ.get("FIGHTINGICE_HOST", "fightingice")
            port = int(os.environ.get("FIGHTINGICE_PORT", 31415))
            
            while True:
                print(f"Connecting to {host}:{port}...")
                gateway = Gateway(host=host, port=port)
                agent1 = PyFTGAgent(self.obs_queue, self.act_queue, "DreamerAI")
                gateway.register_ai("DreamerAI", agent1)
                try:
                    print("Requesting Game Start: DreamerAI vs MctsAi")
                    await gateway.run_game(["ZEN", "ZEN"], ["DreamerAI", "MctsAi23i"], 1000)
                except Exception as e:
                    print(f"PyFTG Disconnected: {e}")
                finally:
                    await gateway.close()
                print("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
        asyncio.run(main_loop())

    def reset(self, seed=None, options=None):
        while not self.obs_queue.empty():
            try: self.obs_queue.get_nowait()
            except queue.Empty: pass

        print("Reset: Waiting for new observation...")
        try:
            obs, reward, done = self.obs_queue.get(timeout=30.0)
            self.current_obs = obs
        except queue.Empty:
            print("Reset Timeout.")
            self.current_obs = {'image': np.zeros((64, 64, 3), dtype=np.uint8)}

        return self.current_obs

    def step(self, action):
        self.act_queue.put(action)
        try:
            obs, reward, done = self.obs_queue.get(timeout=5.0)
            self.current_obs = obs
            return obs, float(reward), done, {}
        except queue.Empty:
            print("Step Timeout.")
            return self.current_obs, 0.0, True, {}

# ----------------------------------------------------------------
# Factory 関数
# ----------------------------------------------------------------
def make_env(config, index, **overrides):
    env = FightingIceGymEnv()
    # 【変更】obs_key は指定しない (Dictのまま渡すため) または 'image' をキーとして使う設定を確認
    # DreamerV3 の FromGym は obs_key='image' を指定するとそのキーの値だけを取り出してしまうが、
    # Dict space の場合は obs_key を指定せず、そのまま渡して UnifyDtypes で処理させるのが一般的
    # ここでは obs_key を指定せず、Dict全体を流す設定にします
    env = from_gym.FromGym(env) 
    env = embodied.wrappers.UnifyDtypes(env)
    return env

def make_agent(config):
    # ダミー環境も画像対応
    class DummyEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.Discrete(len(ACTION_MAP))
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            })
            
    env = DummyEnv()
    env = from_gym.FromGym(env)
    env = embodied.wrappers.UnifyDtypes(env)
    
    notlog = lambda k: not k.startswith('log/')
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}

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
    return agent_module.Agent(obs_space, act_space, agent_config)

def make_replay(config, folder='replay', mode='train'):
    logdir = elements.Path(config.logdir)
    return embodied.replay.Replay(
        length=config.batch_length * config.consec_train + config.replay_context,
        capacity=int(config.replay.size),
        online=config.replay.online,
        chunksize=config.replay.chunksize,
        directory=logdir / folder
    )

def make_logger(config):
    logdir = elements.Path(config.logdir)
    outputs = [
        elements.logger.TerminalOutput(),
        elements.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        elements.logger.TensorBoardOutput(logdir), 
    ]
    return elements.Logger(elements.Counter(), outputs, multiplier=1)

def make_stream(config, replay, mode):
    fn = bind(replay.sample, config.batch_size, mode)
    return embodied.streams.Consec(
        embodied.streams.Stateless(fn),
        length=config.batch_length if mode == 'train' else config.report_length,
        consec=config.consec_train if mode == 'train' else config.consec_report,
        prefix=config.replay_context,
        strict=(mode == 'train'),
        contiguous=True)

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')
    
    config_path = repo_root / 'dreamerv3' / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(elements.Path(config_path).read())
    
    config = elements.Config(configs['defaults'])
    fightingice_defaults = {
        'logdir': './log/dreamer_fightingice',
        'task': 'fightingice_custom',
        'run.train_ratio': 64,
        'run.log_every': 60,
        'batch_size': 16,
        'batch_length': 64,
        'run.envs': 1, # シングル環境
    }
    config = config.update(fightingice_defaults)
    config = elements.Flags(config).parse(sys.argv[1:])

    logdir = elements.Path(config.logdir)
    print("Logdir:", logdir)
    logdir.mkdir()
    
    args = elements.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )

    print("Starting training (Visual Input)...")
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