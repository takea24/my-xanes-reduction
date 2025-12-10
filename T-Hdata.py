import streamlit as st
import pandas as pd

st.title("Temperature & Humidity Data")

uploaded_file = st.file_uploader("ファイルをアップロード (Excel or CSV)")

if uploaded_file:
    # ファイル読み込み
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    st.subheader("元データ（先頭）")
    st.write(df.head())

    # ---- ここから追加部分 ----
    # 列名の候補（あなたのデータに合わせる）
    temp_cols = ["温度", "Temperature", "Temp"]
    hum_cols  = ["湿度", "Humidity", "Hum"]

    # 実際に存在する列を探す
    temp_col = next((c for c in temp_cols if c in df.columns), None)
    hum_col  = next((c for c in hum_cols if c in df.columns), None)

    if temp_col is None or hum_col is None:
        st.error("温度または湿度の列が見つかりません。列名を確認してください。")
    else:
        # 数値化（文字列があれば NaN に）
        df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
        df[hum_col]  = pd.to_numeric(df[hum_col], errors="coerce")

        # ✔ 欠損値がある行だけ除去（ロガーの行を飛ばす）
        df_clean = df.dropna(subset=[temp_col, hum_col])

        st.subheader("欠損行を除外したデータ")
        st.write(df_clean.head())

        # ダウンロードボタン（必要なら）
        csv = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            "欠損を除外したCSVをダウンロード",
            csv,
            "cleaned_data.csv",
            "text/csv"
        )
