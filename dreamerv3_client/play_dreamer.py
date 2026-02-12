import sys
import os
import pathlib
import warnings
import queue
import numpy as np
import ruamel.yaml as yaml
from functools import partial as bind
import argparse
import traceback

# --- Gym 関連 ---
import gym
from gym import spaces

# --- PyFTG 関連 ---
from pyftg import AIInterface, FrameData, AudioData, RoundResult, ScreenData, Key, GameData, CommandCenter
from pyftg.socket.aio.gateway import Gateway
import asyncio

# 画像処理用
try:
    import cv2
except ImportError:
    cv2 = None

# ----------------------------------------------------------------
# 1. パスとライブラリ設定
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

# 【重要】チェックポイント読み込み用のパッチ (元のコードから維持)
try:
    from elements import checkpoint as elements_ckpt
    elements_ckpt.exists = lambda p: os.path.exists(str(p))
    print("Patched elements.checkpoint.exists to use os.path.exists")
except ImportError:
    pass

ACTION_MAP = [
    "STAND_B", "CROUCH_B", "STAND_A", "CROUCH_A", 
    "FORWARD_WALK", "BACK_STEP", "JUMP", "CROUCH", "STAND",
    "STAND_FA", "STAND_FB", "CROUCH_FA", "CROUCH_FB",
    "THROW_A", "THROW_B"
]

# ----------------------------------------------------------------
# 2. 推論実行用エージェント (PyFTG -> Dreamer Policy)
# ----------------------------------------------------------------
class DreamerInferenceAgent(AIInterface):
    def __init__(self, policy_fn, initial_state, name="DreamerAI", blind=False):
        super().__init__()
        self.policy_fn = policy_fn
        self.initial_state = initial_state 
        self.cc = CommandCenter()
        self.key = Key()
        self.player = True 
        self.agent_name = name
        self.blind = blind
        self.agent_state = initial_state 

    def name(self) -> str: return self.agent_name
    def is_blind(self) -> bool: return self.blind
    def isBlind(self) -> bool: return self.blind

    def initialize(self, game_data: GameData, player: bool):
        print(f"[{self.agent_name}] Initialize. PlayerID: {'P1' if player else 'P2'} ({player})")
        self.player = player
        self.cc = CommandCenter()
        self.key = Key()
        self.agent_state = self.initial_state

    def close(self):
        pass

    def get_information(self, frame_data: FrameData, is_control: bool):
        self.frame_data = frame_data
        self.cc.set_frame_data(frame_data, self.player)

    def get_screen_data(self, screen_data: ScreenData):
        self.screen_data = screen_data
    def getScreenData(self, screen_data: ScreenData):
        self.get_screen_data(screen_data)

    def get_audio_data(self, audio_data: AudioData): pass
    def get_non_delay_frame_data(self, frame_data: FrameData): pass

    def processing(self):
        try:
            if self.frame_data is None or self.frame_data.empty_flag or self.frame_data.current_frame_number < 0:
                return


            # 1. 観測データの作成
            obs = {}
            
            if hasattr(self, 'screen_data') and self.screen_data is not None:
                raw_data = self.screen_data.display_bytes
                try:
                    img = np.frombuffer(raw_data, dtype=np.uint8).reshape((640, 960, 3))
                    if cv2:
                        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
                    else:
                        img = img[::10, ::15]
                    # if not hasattr(self, 'debug_saved'):
                    #  # RGBをBGRに変換しないとcv2.imwriteで色が変になるが、形確認ならそのままでOK
                    #     if cv2:
                    #         cv2.imwrite('debug_vision.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    #         print("Saved debug_vision.png! Check this file.")
                    #         self.debug_saved = True
                    obs['image'] = img
                except Exception as e:
                    # print(f"Image processing error: {e}")
                    obs['image'] = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                obs['image'] = np.zeros((64, 64, 3), dtype=np.uint8)

            # 2. DreamerV3 ポリシーの実行
            obs = {k: v[None] for k, v in obs.items()}
            
            is_first = (self.agent_state is self.initial_state)
            obs['reward'] = np.array([0.0], dtype=np.float32)
            obs['is_first'] = np.array([is_first], dtype=bool)
            obs['is_last'] = np.array([False], dtype=bool)
            obs['is_terminal'] = np.array([False], dtype=bool)

            result = self.policy_fn(self.agent_state, obs)
            
            if result is None:
                print("Error: policy_fn returned None!")
                return
            
            self.agent_state, outs, _ = result
            
            if 'action' in outs:
                action_idx = int(outs['action'][0])
                if 0 <= action_idx < len(ACTION_MAP):
                    command = ACTION_MAP[action_idx]
                    self.cc.command_call(command)
                else:
                    print(f"Invalid action index: {action_idx}")
            
            self.key = self.cc.get_skill_key()

        except Exception as e:
            print(f"Error inside processing: {e}")
            traceback.print_exc()

    def input(self) -> Key:
        return self.key

    def round_end(self, result: RoundResult):
        print(f"[{self.agent_name}] Round End")
        self.agent_state = self.initial_state

    def game_end(self):
        print(f"[{self.agent_name}] Game End")

# ----------------------------------------------------------------
# 3. 設定とメイン処理
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='/app/log/001') 
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=31415)
    
    # 対戦設定
    parser.add_argument('--char1', type=str, default='ZEN', help='Character for Player 1')
    parser.add_argument('--char2', type=str, default='ZEN', help='Character for Player 2')
    parser.add_argument('--ai1', type=str, default='DreamerAI', help='AI name for Player 1')
    parser.add_argument('--ai2', type=str, default='MctsAi23i', help='AI name for Player 2')
    
    # 【追加】ゲーム回数指定
    parser.add_argument('--games', type=int, default=1, help='Number of games to play')
    
    args_cli = parser.parse_args()

    logdir_path = pathlib.Path(args_cli.logdir)
    print(f"Target Logdir: {logdir_path}")

    # --- Config ---
    config_path = repo_root / 'dreamerv3' / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(elements.Path(config_path).read())
    config = elements.Config(configs['defaults'])
    
    fightingice_defaults = {
        'logdir': args_cli.logdir,
        'task': 'fightingice_custom',
        'run.train_ratio': 64,
        'run.log_every': 60,
        'batch_size': 1,   # 推論用に1
        'batch_length': 64, 
        'run.envs': 1,
        'replay.size': 2e6,
    }
    config = config.update(fightingice_defaults)
    
    # --- Space 定義 ---
    class DummyEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.Discrete(len(ACTION_MAP))
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            })
        def reset(self): return self.observation_space.sample()
        def step(self, a): return self.observation_space.sample(), 0, False, {}

    dummy_env = DummyEnv()
    env = from_gym.FromGym(dummy_env)
    
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
    
    # --- Agent ---
    agent_config = elements.Config(
        **config.agent,
        jax=config.jax,
        seed=config.seed,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
        batch_size=config.batch_size, 
        batch_length=config.batch_length, 
        logdir=config.logdir,
    )
    
    print("Initializing Agent...")
    agent = agent_module.Agent(obs_space, act_space, agent_config)
    
    # --- Checkpoint (元のコードから維持: 堅牢な読み込み) ---
    ckpt_dir = logdir_path / 'ckpt'
    print(f"Looking for checkpoints in: {ckpt_dir}")
    
    agent_loaded = False

    if ckpt_dir.exists():
        # 日付順にソートし、新しい順に並べる (reverse=True)
        files = [f for f in ckpt_dir.iterdir() if f.name != 'latest' and not f.name.startswith('.')]
        files = sorted(files, key=lambda x: x.name, reverse=True)
        
        for ckpt_candidate in files:
            try:
                print(f"Attempting to load checkpoint: {ckpt_candidate.name}")
                
                load_path = str(ckpt_candidate)
                
                checkpoint = elements.Checkpoint()
                checkpoint.agent = agent
                
                # 読み込み試行
                checkpoint.load(load_path, keys=['agent'])
                
                print(f"Successfully loaded checkpoint: {ckpt_candidate.name}")
                agent_loaded = True
                break  # 成功したらループを抜ける

            except Exception as e:
                print(f"Warning: Failed to load {ckpt_candidate.name}.")
                print(f"Reason: {e}")
                print("Trying the next older checkpoint...")
                continue
    
    if not agent_loaded:
        print(f"Error: No valid checkpoint found in {ckpt_dir}. Exiting.")
        sys.exit(1)
       
    print("Generating initial state...")
    initial_state = agent.init_policy(batch_size=1)
    
    policy_fn = bind(agent.policy, mode='train')
    # policy_fn = bind(agent.policy, mode='train')

    # Gateway Loop (ここが今回の変更点)
    async def run_games_loop():
        for i in range(args_cli.games):
            print(f"\n{'='*20} Game {i+1} / {args_cli.games} {'='*20}")
            
            gateway = Gateway(host=args_cli.host, port=args_cli.port)
            
            ai_agent = DreamerInferenceAgent(policy_fn, initial_state, name=args_cli.ai1)
            gateway.register_ai(args_cli.ai1, ai_agent)
            
            print(f"Connected to {args_cli.host}:{args_cli.port}")
            
            try:
                print(f"Starting Game {i+1}: {args_cli.ai1}({args_cli.char1}) vs {args_cli.ai2}({args_cli.char2})")
                await gateway.run_game([args_cli.char1, args_cli.char2], [args_cli.ai1, args_cli.ai2], 1)
            except Exception as e:
                print(f"Game finished or error in game {i+1}: {e}")
            finally:
                await gateway.close()
            
            print(f"Game {i+1} Finished. Waiting 2 seconds for next game...")
            await asyncio.sleep(2)

    asyncio.run(run_games_loop())

if __name__ == '__main__':
    main()