import streamlit as st
import pandas as pd
import numpy as np

st.title("温湿度データ整理アプリ（30分丸め & ロガー名自動判定）")

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

    df = pd.read_excel(file, header=1)
    cols = df.columns.tolist()
    st.write("列名:", cols)

    # 湿度ブロック抽出
    hum_cols = cols[1: ]    # まず全体から湿度部分の開始点だけ指定

    # 湿度ブロックの Time 列の次に温度側 Time が来るので境界を自動検出
    # 2つめの "Date/Time" の位置を探す
    time_positions = [i for i, c in enumerate(cols) if "Date" in str(c) or "Time" in str(c)]

    if len(time_positions) < 2:
        st.error("2つ目の Date/Time 列が見つかりません。ファイル形式を確認してください。")
        st.stop()

    hum_start = time_positions[0]
    tem_start = time_positions[1]

    hum_cols = cols[hum_start : tem_start]
    tem_cols = cols[tem_start : ]

    hum_block = df[hum_cols].copy()
    tem_block = df[tem_cols].copy()

    st.write(f"湿度ブロック shape: {hum_block.shape}")
    st.write(f"温度ブロック shape: {tem_block.shape}")

    # 列数からロガー数を自動判定
    hum_logger_n = hum_block.shape[1] - 1
    tem_logger_n = tem_block.shape[1] - 1

    hum_block.columns = ["Time"] + [f"Logger{i+1}" for i in range(hum_logger_n)]
    tem_block.columns = ["Time"] + [f"Logger{i+1}" for i in range(tem_logger_n)]

    # long化
    hum_long = hum_block.melt(id_vars=["Time"], var_name="Logger", value_name="Hum")
    tem_long = tem_block.melt(id_vars=["Time"], var_name="Logger", value_name="Temp")

    hum_long = hum_long.dropna(subset=["Hum"])
    tem_long = tem_long.dropna(subset=["Temp"])

    hum_long["Time"] = pd.to_datetime(hum_long["Time"], errors="coerce")
    tem_long["Time"] = pd.to_datetime(tem_long["Time"], errors="coerce")

    hum_long = hum_long.dropna(subset=["Time"])
    tem_long = tem_long.dropna(subset=["Time"])

    # ロガー名正規化
    def normalize(x):
        return str(x).strip().lower().replace(" ", "").replace("_", "")

    hum_long["Logger_norm"] = hum_long["Logger"].apply(normalize)
    tem_long["Logger_norm"] = tem_long["Logger"].apply(normalize)

    # 30分丸め
    hum_long["Time30"] = hum_long["Time"].dt.floor("30min")
    tem_long["Time30"] = tem_long["Time"].dt.floor("30min")

    hum_grp = hum_long.groupby(["Logger_norm", "Time30"], as_index=False)["Hum"].mean()
    tem_grp = tem_long.groupby(["Logger_norm", "Time30"], as_index=False)["Temp"].mean()

    merged = pd.merge(hum_grp, tem_grp, on=["Logger_norm", "Time30"], how="inner")
    merged["SourceFile"] = file.name

    st.write("マージ結果 shape:", merged.shape)
    all_merged.append(merged)

# 全結合
final_df = pd.concat(all_merged, ignore_index=True)
st.write("### 🎉 全ファイル統合結果")
st.write(final_df)

csv = final_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="📥 CSV をダウンロード",
    data=csv,
    file_name="merged_THdata.csv",
    mime="text/csv"
)
