import streamlit as st
import pandas as pd

st.title("温湿度ロガー データ整理アプリ（30分丸め・列自動判定版）")

uploaded_files = st.file_uploader("Excel ファイルを選択", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    all_data = []

    for file in uploaded_files:
        st.write(f"処理中 → {file.name}")

        # ヘッダー2行目を読み込む
        df = pd.read_excel(file, header=1)

        # ---- ポイント：Date/Time 列を探す ----
        dt_cols = [i for i, c in enumerate(df.columns) if "Date" in str(c) or "Time" in str(c)]

        if len(dt_cols) != 2:
            st.error("Date/Time 列が2つ見つかりません。ファイル形式が違います。")
            st.write(df.head())
            continue

        dt1, dt2 = dt_cols  # 左ブロックの Date/Time と右ブロックの Date/Time

        # 湿度ブロックは dt1 から dt2 - 1
        hum_block = df.iloc[:, dt1:dt2].copy()
        hum_block.columns = ["Time"] + list(hum_block.columns[1:])

        # 温度ブロックは dt2 から最後まで
        tem_block = df.iloc[:, dt2:].copy()
        tem_block.columns = ["Time"] + list(tem_block.columns[1:])

        # datetime 化
        hum_block["Time"] = pd.to_datetime(hum_block["Time"], errors="coerce")
        tem_block["Time"] = pd.to_datetime(tem_block["Time"], errors="coerce")

        # ロング形式
        hum_long = hum_block.melt(id_vars="Time", var_name="Logger", value_name="Humidity")
        tem_long = tem_block.melt(id_vars="Time", var_name="Logger", value_name="Temperature")

        # Logger 名前整形
        hum_long["Logger"] = hum_long["Logger"].astype(str).str.strip()
        tem_long["Logger"] = tem_long["Logger"].astype(str).str.strip()

        # 30分単位で丸め
        hum_long["Time"] = hum_long["Time"].dt.floor("30min")
        tem_long["Time"] = tem_long["Time"].dt.floor("30min")

        # merge
        merged = pd.merge(hum_long, tem_long, on=["Time", "Logger"], how="inner")

        # 欠損値除去
        merged = merged.dropna(subset=["Humidity", "Temperature"])

        all_data.append(merged)

    # 全ファイル結合
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        st.write("### 🔍 プレビュー")
        st.dataframe(final_df.head(50))

        # ダウンロード
        csv = final_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 CSV ダウンロード", data=csv, file_name="merged_TH.csv", mime="text/csv")
