import asyncio
import os
import sys
import traceback

# 必要なクラスをインポート
from pyftg import AIInterface, FrameData, AudioData, RoundResult, ScreenData, Key, GameData, CommandCenter
from pyftg.socket.aio.gateway import Gateway

class BaseCombatAI(AIInterface):
    def __init__(self, mode):
        super().__init__()
        self.cc = CommandCenter()
        self.input_key = Key()
        self.frame_data = None
        self.player = True
        self.mode = mode  # "PUNCH" or "KICK"

    def name(self) -> str:
        return f"{self.mode}AI"

    def is_blind(self) -> bool:
        return False

    def initialize(self, game_data: GameData, player: bool):
        self.player = player
        self.input_key = Key()
        self.frame_data = FrameData()
        self.cc.set_frame_data(self.frame_data, self.player)

    def close(self): pass
    def input(self) -> Key: return self.input_key
    def get_non_delay_frame_data(self, frame_data: FrameData): pass

    def get_information(self, frame_data: FrameData, is_control: bool):
        self.frame_data = frame_data
        self.cc.set_frame_data(frame_data, self.player)

    def get_screen_data(self, screen_data: ScreenData): pass
    def get_audio_data(self, audio_data: AudioData): pass
    def round_end(self, round_result: RoundResult): print(f"Round End: {self.mode}AI")
    def game_end(self): print("Game End")

    def processing(self):
        try:
            if self.frame_data is None or self.frame_data.empty_flag or self.frame_data.current_frame_number < 0:
                return


            # 1. まずCommandCenterから現在のフレームのキーを取得
            self.input_key = self.cc.get_skill_key()

            # 2. キー入力が有効か（何らかのボタンが押されているか）判定する関数
            def is_key_active(key):
                # 攻撃ボタン(A,B,C) または 方向キー(U,D,L,R) が押されていれば「実行中」とみなす
                return key.A or key.B or key.C or key.U or key.D or key.L or key.R

            # 3. アクション実行中なら、新しい判断をせずにリターン（今の動作を継続）
            if is_key_active(self.input_key):
                return
                
            self.input_key = Key() # 入力リセット
            
            # 自分のキャラと相手のキャラの位置を取得
            me = self.frame_data.get_character(self.player)
            opp = self.frame_data.get_character(not self.player)
            
            # 距離を計算 (X座標の差の絶対値)
            distance = abs(me.x - opp.x)
            
            # ロジック: 距離が100より遠ければ近づく、近ければ攻撃
            if distance > 100:
                self.cc.command_call("FORWARD_WALK")
            else:
                if self.mode == "PUNCH":
                    self.cc.command_call("STAND_A") # Aボタン（パンチ）
                elif self.mode == "KICK":
                    self.cc.command_call("STAND_B") # Bボタン（キック）

            # 生成されたコマンドをキー入力に反映
            self.input_key = self.cc.get_skill_key()

        except Exception as e:
            traceback.print_exc()
            sys.exit(1)

# 個別のクラスとして定義（登録名のために使い分ける）
class PunchAI(BaseCombatAI):
    def __init__(self):
        super().__init__("PUNCH")

class KickAI(BaseCombatAI):
    def __init__(self):
        super().__init__("KICK")

async def main():
    host = os.environ.get("FIGHTINGICE_HOST", "127.0.0.1")
    port = int(os.environ.get("FIGHTINGICE_PORT", 31415))

    print(f"Connecting to FightingICE server at {host}:{port} ...")
    
    gateway = Gateway(host=host, port=port)
    agent1 = KickAI()
    agent2 = PunchAI()

    gateway.register_ai("KickAI_1", agent1)
    gateway.register_ai("KickAI_2", agent2)

    print("AI registered. Starting game...")

    try:
        await gateway.run_game(["ZEN", "ZEN"], ["KickAI_1", "KickAI_2"], 1)
    except Exception as e:
        print(f"Python Side Error: {e}")
        import traceback
        traceback.print_exc() # 詳細なエラースタックトレースを表示
    finally:
        await gateway.close()
        print("Disconnected.")

if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
