import asyncio
import os
import sys

# ★修正1: CommandCenter を追加インポート
from pyftg import AIInterface, FrameData, AudioData, RoundResult, ScreenData, Key, GameData, CommandCenter
from pyftg.socket.aio.gateway import Gateway

class TestKickAI(AIInterface):
    def __init__(self):
        super().__init__()
        # ★修正2: ここでCommandCenterを自分で初期化する
        self.cc = CommandCenter()
        
        self.blind = False

    def name(self) -> str:
        return "TestKickAI"

    def is_blind(self) -> bool:
        return self.blind

    def initialize(self, game_data: GameData, player: bool):
        # ゲーム開始時に呼ばれる初期化処理
        self.input_key = Key()
        self.cc.set_frame_data(FrameData(), player)

    def close(self):
        pass

    def input(self) -> Key:
        return self.input_key

    def get_non_delay_frame_data(self, frame_data: FrameData):
        pass

    def get_information(self, frame_data: FrameData, is_control: bool):
        self.frame_data = frame_data
        # CommandCenterにフレームデータをセット
        self.cc.set_frame_data(frame_data, self.player)
    
    def get_screen_data(self, screen_data: ScreenData):
        pass

    def get_audio_data(self, audio_data: AudioData):
        pass

    def processing(self):
        if self.frame_data.get_empty_flag() or self.frame_data.get_remaining_frames_number() <= 0:
            return

        # キック連打
        if self.cc.get_skill_key():
            self.cc.command_call("B")
            self.input_key = self.cc.get_skill_key()

    def round_end(self, round_result: RoundResult):
        print(f"Round End: Result {round_result.get_remaining_hps()}")

    def game_end(self):
        print("Game End")

async def main():
    # ホスト名とポートの取得
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
        print(f"Error occurred: {e}")
    finally:
        await gateway.close()
        print("Disconnected.")

if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
