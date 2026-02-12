import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def analyze_battles():
    # ==========================================
    # 1. データの読み込みと結合
    # ==========================================
    files = glob.glob("HPMode_DreamerAI_*.csv")
    
    if not files:
        print("CSVファイルが見つかりません。")
        return

    all_data = []
    print(f"{len(files)} 個のファイルを処理中...")

    for file in files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        
        # 名前抽出 (DreamerAI_と日時の間の部分)
        if len(parts) >= 4:
            opponent_name = "_".join(parts[2:-1])
        else:
            opponent_name = "Unknown"

        try:
            # ヘッダーなしCSVとして読み込み
            df = pd.read_csv(file, header=None, names=['Match', 'MyHP', 'OpponentHP', 'Time'])
            df['OpponentAI'] = opponent_name
            all_data.append(df)
        except Exception as e:
            print(f"スキップしました ({filename}): {e}")

    if not all_data:
        print("有効なデータがありませんでした。")
        return

    merged_df = pd.concat(all_data, ignore_index=True)

    # ==========================================
    # 2. 勝敗判定と集計
    # ==========================================
    # 勝敗条件: 自分のHP > 相手のHP
    merged_df['IsWin'] = merged_df['MyHP'] > merged_df['OpponentHP']
    
    # AIごとの集計
    summary = merged_df.groupby('OpponentAI').agg(
        Matches=('Match', 'count'),
        Wins=('IsWin', 'sum'),
        AvgMyHP=('MyHP', 'mean'),
        AvgOppHP=('OpponentHP', 'mean'),
        AvgTime=('Time', 'mean')
    ).reset_index()

    # 勝率(%)を計算
    summary['WinRate'] = (summary['Wins'] / summary['Matches']) * 100

    # 結果をCSVに保存
    summary.to_csv("Battle_Summary.csv", index=False, float_format='%.2f')
    print("\n【集計結果】 Battle_Summary.csv に保存しました。")
    print(summary[['OpponentAI', 'Matches', 'Wins', 'WinRate']])

    # ==========================================
    # 3. グラフの作成 (matplotlib)
    # ==========================================
    try:
        # 【ここを変更】全体の文字サイズを一括で大きくする
        plt.rcParams.update({'font.size': 14}) 

        # グラフのサイズも少し大きくする (横12インチ, 縦8インチ)
        plt.figure(figsize=(12, 8))
        
        # 棒グラフを描画
        bars = plt.bar(summary['OpponentAI'], summary['WinRate'], color='skyblue', edgecolor='navy')
        
        # 【ここを変更】各項目のフォントサイズをさらに個別に指定
        plt.title('', fontsize=24, fontweight='bold') # タイトル
        plt.xlabel('', fontsize=18)      # 下の軸名
        plt.ylabel('', fontsize=18)     # 左の軸名
        
        # 【ここを追加】AIの名前（X軸の目盛り）が重ならないように回転させたりサイズ調整
        plt.xticks(rotation=45, fontsize=14) 
        plt.yticks(fontsize=14)

        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # レイアウトを自動調整（文字切れ防止）
        plt.tight_layout()

        # 棒の上に数値を表示
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 1,
                     f'{height:.1f}%', 
                     ha='center', va='bottom', fontweight='bold', 
                     fontsize=16) # 【ここを変更】数値のサイズ

        # 画像として保存
        plt.savefig("WinRate_Graph.png")
        print("【グラフ】 WinRate_Graph.png に保存しました（文字サイズ大）。")
        
    except Exception as e:
        print(f"グラフ作成エラー: {e}") 
if __name__ == "__main__":
    analyze_battles()