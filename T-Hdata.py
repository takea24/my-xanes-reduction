import streamlit as st
import pandas as pd

st.title("📊 温湿度ロガー データ変換アプリ（Excel → tidy CSV）")

uploaded_files = st.file_uploader(
    "月ごとの Excel ファイルをまとめてアップロード", 
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

if uploaded_files:
    all_data = []

    for file in uploaded_files:
        st.write(f"処理中: {file.name}")

        # 1. Excel 読み込み（1行目スキップ）
        df = pd.read_excel(file, header=1)   # 1行目スキップ → ヘッダは2行目

        # 2. 不要な空列を削除
        df = df.dropna(axis=1, how="all")

        # 3. 列構造を把握して分割
        # 湿度パート：左半分
        # 温度パート：右半分
        n_cols = df.shape[1]
        half = n_cols // 2

        hum = df.iloc[:, :half]
        tem = df.iloc[:, half:]

        # 4. 列名取得
        hum_cols = hum.columns[1:]  # ロガー名（Date/Time を除く）
        tem_cols = tem.columns[1:]  # 同じ順

        # 5. reshape: wide → long
        hum_long = hum.melt(id_vars=[hum.columns[0]], 
                            value_vars=hum_cols,
                            var_name="Logger",
                            value_name="Humidity")

        tem_long = tem.melt(id_vars=[tem.columns[0]],
                            value_vars=tem_cols,
                            var_name="Logger",
                            value_name="Temperature")

        # 6. 時間＋ロガーで結合
        hum_long = hum_long.rename(columns={hum.columns[0]: "Time"})
        tem_long = tem_long.rename(columns={tem.columns[0]: "Time"})

        merged = pd.merge(hum_long, tem_long, on=["Time", "Logger"], how="inner")
        all_data.append(merged)

    # 7. すべての月を結合
    result = pd.concat(all_data, ignore_index=True)

    st.subheader("📄 整形後データ（プレビュー）")
    st.dataframe(result)

    # 8. CSV ダウンロード
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 CSV をダウンロード",
        data=csv,
        file_name="logger_year_data.csv",
        mime="text/csv"
    )
