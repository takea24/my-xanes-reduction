import streamlit as st
import pandas as pd
import io

st.title("博物館 温湿度データ 統合ツール（Streamlit版）")
st.write("月ごとの Excel ファイルを複数アップロードすると、年間通した縦長データに統合します。")

uploaded_files = st.file_uploader("Excel ファイルを月ごとにアップロード", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    all_data = []

    for uploaded_file in uploaded_files:
        st.write(f"処理中: {uploaded_file.name}")

        # Excel 読み込み
        df = pd.read_excel(uploaded_file)

        # datetime 列を設定
        if "Unnamed: 1" in df.columns:
            df = df.rename(columns={"Unnamed: 1": "datetime"})
        else:
            st.error(f"{uploaded_file.name}: datetime 列（Unnamed: 1）が見つかりません")
            continue

        # datetime を変換
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        # センサー名が含まれる 1 行目を抽出
        header = df.iloc[0]

        # データ本体（1行目を削除）
        df = df.drop(0)

        # 不要列（Unnamed 系）を削除
        df = df.loc[:, ~df.columns.str.contains("Unnamed")]

        # 湿度列（例：%.1, %.2 ...）
        hum_cols = [col for col in df.columns if col.startswith("%")]
        # 温度列（例：°C.1, °C.2 ...）
        temp_cols = [col for col in df.columns if col.startswith("°C")]

        # 列名をセンサー名に置き換え
        hum_map = {col: header[col] for col in hum_cols}
        temp_map = {col: header[col] for col in temp_cols}

        df = df.rename(columns=hum_map)
        df = df.rename(columns=temp_map)

        # wide → long
        df_hum = df.melt(
            id_vars="datetime",
            value_vars=list(hum_map.values()),
            var_name="location",
            value_name="humidity_RH"
        )

        df_temp = df.melt(
            id_vars="datetime",
            value_vars=list(temp_map.values()),
            var_name="location",
            value_name="temperature_C"
        )

        # 温湿度を結合
        df_merge = pd.merge(df_hum, df_temp, on=["datetime", "location"])

        all_data.append(df_merge)

    # 全月の結合
    df_final = pd.concat(all_data, ignore_index=True)
    df_final = df_final.sort_values("datetime")

    st.success("統合が完了しました！")

    st.dataframe(df_final.head(20))

    # CSV としてダウンロード
    csv = df_final.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="統合データを CSV でダウンロード",
        data=csv,
        file_name="museum_env_all.csv",
        mime="text/csv"
    )
