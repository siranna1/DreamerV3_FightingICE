import asyncio
import os
import sys
import traceback

# 必要なクラスをインポート
from pyftg import AIInterface, FrameData, AudioData, RoundResult, ScreenData, Key, GameData, CommandCenter
from pyftg.socket.aio.gateway import Gateway

class TestKickAI(AIInterface):
    def __init__(self):
        super().__init__()
        self.cc = CommandCenter()
        self.input_key = Key()
        self.frame_data = None
        self.blind = False
        self.player = True 

    def name(self) -> str:
        return "TestKickAI"

    def is_blind(self) -> bool:
        return self.blind

    def initialize(self, game_data: GameData, player: bool):
        self.player = player
        self.input_key = Key()
        self.frame_data = FrameData()
        self.cc.set_frame_data(self.frame_data, self.player)

    def close(self):
        pass

    def input(self) -> Key:
        return self.input_key

    def get_non_delay_frame_data(self, frame_data: FrameData):
        pass

    def get_information(self, frame_data: FrameData, is_control: bool):
        self.frame_data = frame_data
        self.cc.set_frame_data(frame_data, self.player)
    
    def get_screen_data(self, screen_data: ScreenData):
        pass

    def get_audio_data(self, audio_data: AudioData):
        pass

    def processing(self):
        try:
            # メンバ変数 frame_data が None の場合はスキップ
            if self.frame_data is None:
                return

            # ★重要修正: メソッド() ではなく プロパティ変数 としてアクセスする
            # (get_empty_flag() -> empty_flag, get_remaining_frames_number() -> remaining_frames_number)
            if self.frame_data.empty_flag: 
                return
            if self.frame_data.current_frame_number < 0:
                return

            if self.cc.get_skill_key():
                self.cc.command_call("B")
                self.input_key = self.cc.get_skill_key()
            else:
                self.input_key = Key()

        except Exception as e:
            print(f"Processing Error: {e}")
            traceback.print_exc() # 詳細なエラースタックトレースを表示
            sys.exit(1)

    def round_end(self, round_result: RoundResult):
        # RoundResult も同様にプロパティアクセスの可能性があるため修正
        # (get_remaining_hps() -> remaining_hps)
        try:
            print(f"Round End: Result {round_result.remaining_hps}")
        except:
            # 万が一メソッドだった場合のフォールバック（デバッグ用）
            print(f"Round End: Result (Unknown format)")

    def game_end(self):
        print("Game End")

async def main():
    host = os.environ.get("FIGHTINGICE_HOST", "127.0.0.1")
    port = int(os.environ.get("FIGHTINGICE_PORT", 31415))

    print(f"Connecting to FightingICE server at {host}:{port} ...")
    
    gateway = Gateway(host=host, port=port)
    agent1 = TestKickAI()
    agent2 = TestKickAI()

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
