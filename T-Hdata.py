import streamlit as st
import pandas as pd

st.title("博物館 温湿度データ 統合ツール")
st.write("湿度ブロック → 温度ブロックの順で並んでいる Excel を自動解析して縦長データに統合します。")

uploaded_files = st.file_uploader(
    "Excel ファイルをアップロード（複数可）",
    type=["xlsx"],
    accept_multiple_files=True
)

def process_file(uploaded_file):
    """1つのファイルを処理して tidy データにして返す"""

    # まず全体を読み込む（1行目を無視し、2行目をヘッダーにする）
    df = pd.read_excel(uploaded_file, header=1)

    # 列名リスト
    cols = list(df.columns)

    # ---- 湿度ブロックと温度ブロックの境界を探す ----
    # Date/Time が2つあるはず：最初 → 湿度, 次 → 温度
    date_cols = [i for i, c in enumerate(cols) if "Date" in str(c)]
    if len(date_cols) < 2:
        st.error(f"{uploaded_file.name}：Date/Time 列が2つ必要です。")
        return None

    hum_start = date_cols[0]              # 湿度 Date/Time
    temp_start = date_cols[1]             # 温度 Date/Time

    hum_cols = cols[hum_start:temp_start]     # 湿度ブロック列
    temp_cols = cols[temp_start:]             # 温度ブロック列

    # ---- 湿度ブロック処理 ----
    df_hum = df[hum_cols].copy()
    df_hum = df_hum.rename(columns={df_hum.columns[0]: "datetime"})
    df_hum["datetime"] = pd.to_datetime(df_hum["datetime"], errors="coerce")

    # ---- 温度ブロック処理 ----
    df_temp = df[temp_cols].copy()
    df_temp = df_temp.rename(columns={df_temp.columns[0]: "datetime"})
    df_temp["datetime"] = pd.to_datetime(df_temp["datetime"], errors="coerce")

    # ---- センサー名（可変） ----
    hum_sensors = df_hum.columns[1:]
    temp_sensors = df_temp.columns[1:]

    # ---- 縦長化 ----
    hum_long = df_hum.melt(
        id_vars="datetime",
        value_vars=hum_sensors,
        var_name="location",
        value_name="humidity_RH"
    )

    temp_long = df_temp.melt(
        id_vars="datetime",
        value_vars=temp_sensors,
        var_name="location",
        value_name="temperature_C"
    )

    # ---- 湿度と温度を結合 ----
    df_merge = pd.merge(hum_long, temp_long, on=["datetime", "location"], how="outer")

    # ---- 欠損（片側だけ NaN）の行を除外 ----
    df_merge = df_merge.dropna(subset=["humidity_RH", "temperature_C"], how="any")

    return df_merge


if uploaded_files:
    all_data = []

    for f in uploaded_files:
        st.write(f"処理中: {f.name}")
        df_single = process_file(f)
        if df_single is not None:
            all_data.append(df_single)

    if len(all_data) > 0:
        df_final = pd.concat(all_data, ignore_index=True)
        df_final = df_final.sort_values("datetime")

        st.success("統合完了！")
        st.dataframe(df_final.head(30))

        csv = df_final.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="統合データ CSV ダウンロード",
            data=csv,
            file_name="museum_env_merged.csv",
            mime="text/csv"
        )
