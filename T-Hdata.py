import streamlit as st
import pandas as pd
import numpy as np

st.title("温湿度データ整理アプリ（30分丸め & ロガー名統一版）")

uploaded_files = st.file_uploader(
    "月ごとのエクセルファイルを複数選択してください",
    type=["xlsx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

all_merged = []

for file in uploaded_files:
    st.write(f"---\n### 📄 読み込み：{file.name}")

    # Excel 読み込み（ヘッダー2行スキップ）
    df = pd.read_excel(file, header=1)

    # ===== データ構造 =====
    # 1列目   → 不要 or 文字
    # 2列目   → Date/Time（湿度）
    # 3~15    → ロガー湿度（13個）
    # 次の列  → Date/Time（温度）
    # その後  → ロガー温度（13個）
    # ======================

    # 列名を確認
    cols = df.columns.tolist()
    st.write("列名:", cols)

    # 湿度ブロック（2〜14列目）
    hum_cols = cols[1:14+1]
    hum_block = df[hum_cols].copy()
    hum_block.columns = ["Time"] + [f"Logger{i+1}" for i in range(13)]

    # 温度ブロック（16〜28列目）
    tem_cols = cols[15:28+1]
    tem_block = df[tem_cols].copy()
    tem_block.columns = ["Time"] + [f"Logger{i+1}" for i in range(13)]

    st.write(f"湿度ブロック shape: {hum_block.shape}")
    st.write(f"温度ブロック shape: {tem_block.shape}")

    # long形式へ変換
    hum_long = hum_block.melt(id_vars=["Time"], var_name="Logger", value_name="Hum")
    tem_long = tem_block.melt(id_vars=["Time"], var_name="Logger", value_name="Temp")

    # 欠損除去
    hum_long = hum_long.dropna(subset=["Hum"])
    tem_long = tem_long.dropna(subset=["Temp"])

    # 時刻を datetime に変換
    hum_long["Time"] = pd.to_datetime(hum_long["Time"], errors="coerce")
    tem_long["Time"] = pd.to_datetime(tem_long["Time"], errors="coerce")
    hum_long = hum_long.dropna(subset=["Time"])
    tem_long = tem_long.dropna(subset=["Time"])

    # Logger 名を正規化（大小・空白・_ などを補正）
    def normalize(x):
        return str(x).strip().lower().replace(" ", "").replace("_", "")

    hum_long["Logger_norm"] = hum_long["Logger"].apply(normalize)
    tem_long["Logger_norm"] = tem_long["Logger"].apply(normalize)

    # 30分に丸め
    hum_long["Time30"] = hum_long["Time"].dt.floor("30min")
    tem_long["Time30"] = tem_long["Time"].dt.floor("30min")

    # 時間 × logger ごとに平均化
    hum_grp = hum_long.groupby(["Logger_norm", "Time30"], as_index=False)["Hum"].mean()
    tem_grp = tem_long.groupby(["Logger_norm", "Time30"], as_index=False)["Temp"].mean()

    # マージ
    merged = pd.merge(hum_grp, tem_grp, on=["Logger_norm", "Time30"], how="inner")
    merged["SourceFile"] = file.name  # どの月ファイルか記録

    st.write("マージ結果 shape:", merged.shape)

    all_merged.append(merged)

# ===== 最終結合 =====
final_df = pd.concat(all_merged, ignore_index=True)

st.write("### 🎉 全ファイル統合結果")
st.write(final_df)

# ===== CSV ダウンロード =====
csv = final_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="📥 CSV をダウンロード",
    data=csv,
    file_name="merged_THdata.csv",
    mime="text/csv"
)
