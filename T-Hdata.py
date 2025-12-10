import streamlit as st
import pandas as pd
import numpy as np

st.title("温湿度ロガー データ整理アプリ（30分丸め版）")

st.write("月ごとの Excel ファイルを複数選択してアップロードしてください。")

uploaded_files = st.file_uploader("Excel ファイルを選ぶ", type=["xlsx", "xls"], accept_multiple_files=True)

if uploaded_files:
    all_data = []

    for file in uploaded_files:
        st.write(f"読み込み中: {file.name}")

        # エクセル読み込み（1行目不要 → header=1）
        df = pd.read_excel(file, header=1)

        # 湿度ブロック（左側）
        hum_block = df.iloc[:, :14].copy()
        hum_block.columns = ["Time"] + list(hum_block.columns[1:])

        # 温度ブロック（右側）
        tem_block = df.iloc[:, 14:].copy()
        tem_block.columns = ["Time"] + list(tem_block.columns[1:])

        # 時刻を datetime に変換
        hum_block["Time"] = pd.to_datetime(hum_block["Time"], errors="coerce")
        tem_block["Time"] = pd.to_datetime(tem_block["Time"], errors="coerce")

        # ロング形式へ変換
        hum_long = hum_block.melt(id_vars="Time", var_name="Logger", value_name="Humidity")
        tem_long = tem_block.melt(id_vars="Time", var_name="Logger", value_name="Temperature")

        # Logger 名 正規化
        hum_long["Logger"] = hum_long["Logger"].astype(str).str.strip()
        tem_long["Logger"] = tem_long["Logger"].astype(str).str.strip()

        # 時刻を30分単位に丸める
        hum_long["Time"] = hum_long["Time"].dt.floor("30min")
        tem_long["Time"] = tem_long["Time"].dt.floor("30min")

        # 温度・湿度をマージ
        merged = pd.merge(hum_long, tem_long, on=["Time", "Logger"], how="inner")

        # 湿度・温度が欠けている行を削除
        merged = merged.dropna(subset=["Humidity", "Temperature"])

        all_data.append(merged)

    # 全月を結合
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)

        st.write("### 🔍 整理されたデータ（プレビュー）")
        st.dataframe(final_df.head(50))

        # CSV 保存
        csv_data = final_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="📥 CSV をダウンロード",
            data=csv_data,
            file_name="T-H_merged_30min.csv",
            mime="text/csv"
        )

    else:
        st.error("データがありません。Excel ファイルの形式を確認してください。")
