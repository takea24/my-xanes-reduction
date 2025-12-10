import streamlit as st
import pandas as pd

st.title("📊 温湿度ロガー データ変換アプリ（Excel → tidy CSV）")

uploaded_files = st.file_uploader(
    "月ごとの Excel ファイルをまとめてアップロードしてください",
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

if uploaded_files:
    all_data = []

    for file in uploaded_files:
        st.write(f"処理中: {file.name}")

        # 1. Excel 読み込み（1行目: 単位行を除外）
        df = pd.read_excel(file, header=1)

        # 2. 空列の除去
        df = df.dropna(axis=1, how="all")

        # 3. 湿度と温度のブロックを分割（半分ずつ）
        n_cols = df.shape[1]
        half = n_cols // 2

        hum = df.iloc[:, :half]
        tem = df.iloc[:, half:]

        # 4. 湿度/温度のロガー名（1列目は Date/Time）
        hum_cols = hum.columns[1:]
        tem_cols = tem.columns[1:]

        # 5. wide → long（縦長に変換）
        hum_long = hum.melt(
            id_vars=[hum.columns[0]],
            value_vars=hum_cols,
            var_name="Logger",
            value_name="Humidity"
        )

        tem_long = tem.melt(
            id_vars=[tem.columns[0]],
            value_vars=tem_cols,
            var_name="Logger",
            value_name="Temperature"
        )

        # 6. 列名統一
        hum_long = hum_long.rename(columns={hum.columns[0]: "Time"})
        tem_long = tem_long.rename(columns={tem.columns[0]: "Time"})

        # ---- 🔧 型統一（重要） ----
        hum_long["Time"] = pd.to_datetime(hum_long["Time"], errors="coerce")
        tem_long["Time"] = pd.to_datetime(tem_long["Time"], errors="coerce")

        hum_long["Logger"] = hum_long["Logger"].astype(str).str.strip()
        tem_long["Logger"] = tem_long["Logger"].astype(str).str.strip()

        hum_long["Humidity"] = pd.to_numeric(hum_long["Humidity"], errors="coerce")
        tem_long["Temperature"] = pd.to_numeric(tem_long["Temperature"], errors="coerce")

        # ---- 結合 ----
        merged = pd.merge(
            hum_long,
            tem_long,
            on=["Time", "Logger"],
            how="inner"
        )

        # ---- 欠損削除（Humidity or Temperature が欠けている行）----
        merged = merged.dropna(subset=["Humidity", "Temperature"])

        all_data.append(merged)

    # 7. 全ファイル結合
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
