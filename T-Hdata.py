import streamlit as st
import pandas as pd

st.title("📊 温湿度ロガー データ変換アプリ（Excel → tidy CSV）")

uploaded_files = st.file_uploader(
    "Excel ファイルをアップロード（複数可）",
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

if uploaded_files:
    all_data = []

    for file in uploaded_files:
        st.write(f"処理中: {file.name}")

        # ヘッダーは2行目（= header=1）
        df = pd.read_excel(file, header=1)

        # 全空列消す
        df = df.dropna(axis=1, how="all")

        # ---- 🔍 左側の Date/Time の列位置を検出 ----
        dt_cols = [c for c in df.columns if "Date" in str(c)]
        if len(dt_cols) != 2:
            st.error(f"{file.name}: Date/Time 列が2つ検出できません。")
            continue

        dt_hum = dt_cols[0]   # 湿度側
        dt_tem = dt_cols[1]   # 温度側

        # ---- 湿度ブロック ----
        hum_start = df.columns.get_loc(dt_hum)
        tem_start = df.columns.get_loc(dt_tem)

        hum = df.iloc[:, hum_start:tem_start]     # 湿度ブロック
        tem = df.iloc[:, tem_start:]              # 温度ブロック

        # 列名
        hum_cols = hum.columns[1:]
        tem_cols = tem.columns[1:]

        # ---- long 形式へ ----
        hum_long = hum.melt(id_vars=[dt_hum], value_vars=hum_cols,
                            var_name="Logger", value_name="Humidity")
        tem_long = tem.melt(id_vars=[dt_tem], value_vars=tem_cols,
                            var_name="Logger", value_name="Temperature")

        hum_long = hum_long.rename(columns={dt_hum: "Time"})
        tem_long = tem_long.rename(columns={dt_tem: "Time"})

        # ---- 型そろえる ----
        hum_long["Time"] = pd.to_datetime(hum_long["Time"], errors="coerce")
        tem_long["Time"] = pd.to_datetime(tem_long["Time"], errors="coerce")

        hum_long["Logger"] = hum_long["Logger"].astype(str).strip()
        tem_long["Logger"] = tem_long["Logger"].astype(str).strip()

        hum_long["Humidity"] = pd.to_numeric(hum_long["Humidity"], errors="coerce")
        tem_long["Temperature"] = pd.to_numeric(tem_long["Temperature"], errors="coerce")

        # ---- 結合 ----
        merged = pd.merge(hum_long, tem_long, on=["Time", "Logger"], how="inner")

        # ---- 欠損削除 ----
        merged = merged.dropna(subset=["Humidity", "Temperature"])

        all_data.append(merged)

    # ---- 全部結合 ----
    if all_data:
        result = pd.concat(all_data, ignore_index=True)

        st.subheader("📄 整形後データ（プレビュー）")
        st.dataframe(result)

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button("📥 CSV ダウンロード", csv, "logger_year.csv", "text/csv")
    else:
        st.warning("処理できたデータがありませんでした。")
