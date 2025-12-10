import streamlit as st
import pandas as pd

st.title("温湿度データ統合＆クレンジングツール")

uploaded_files = st.file_uploader(
    "温湿度データ（複数可: Excel or CSV）をアップロード",
    accept_multiple_files=True
)

if uploaded_files:
    dfs = []

    for uploaded_file in uploaded_files:

        # ファイル形式判定
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        # 列名候補（あなたのデータから推測）
        temp_cols = ["温度", "Temperature", "Temp"]
        hum_cols = ["湿度", "Humidity", "Hum"]

        # 実際の列名を探す
        temp_col = next((c for c in temp_cols if c in df.columns), None)
        hum_col = next((c for c in hum_cols if c in df.columns), None)

        if temp_col is None or hum_col is None:
            st.error(f"{uploaded_file.name} に温度/湿度の列が見つかりません。")
            continue

        # 数値に変換（変な文字が来たら NaN にする）
        df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
        df[hum_col] = pd.to_numeric(df[hum_col], errors="coerce")

        # ✔ 温湿度のどちらかが欠損なら、そのロガーの行だけ削除
        df_clean = df.dropna(subset=[temp_col, hum_col])

        dfs.append(df_clean)

    # ファイルの統合
    if len(dfs) > 0:
        df_all = pd.concat(dfs, ignore_index=True)

        st.success("データを統合し、欠損のあるロガーの行を削除しました。")
        st.write(df_all.head())

        # ダウンロードリンク作成
        csv = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            "統合データをCSVでダウンロード",
            csv,
            "merged_TH_data.csv",
            "text/csv"
        )
