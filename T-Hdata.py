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

        # ----------------------------
        # ① datetime列の自動生成処理
        # ----------------------------

        # 元の位置に datetime があればそれを使う
        if "datetime" in df.columns:
            pass

        # Unnamed: 1 が datetime の場合
        elif "Unnamed: 1" in df.columns:
            df = df.rename(columns={"Unnamed: 1": "datetime"})

        else:
            # Date / Time 列を探す
            date_candidates = [c for c in df.columns if "date" in c.lower()]
            time_candidates = [c for c in df.columns if "time" in c.lower()]

            if date_candidates and time_candidates:
                date_col = date_candidates[0]
                time_col = time_candidates[0]
                st.write(f"→ {date_col} と {time_col} から datetime を生成しました")

                df["datetime"] = pd.to_datetime(
                    df[date_col].astype(str) + " " + df[time_col].astype(str),
                    errors="coerce"
                )
            else:
                st.error(f"{uploaded_file.name}: datetime 列が見つからず、Date/Time 列もありません")
                continue

        # datetime を変換
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        # ----------------------------
        # ② センサー名抽出
        # ----------------------------
        header = df.iloc[0]
        df = df.drop(0)

        df = df.loc[:, ~df.columns.str.contains("Unnamed")]

        hum_cols = [col for col in df.columns if col.startswith("%")]
        temp_cols = [col for col in df.columns if col.startswith("°C")]

        hum_map = {col: header[col] for col in hum_cols}
        temp_map = {col: header[col] for col in temp_cols}

        df = df.rename(columns=hum_map)
        df = df.rename(columns=temp_map)

        # ----------------------------
        # ③ wide → long に変換
        # ----------------------------
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

        df_merge = pd.merge(df_hum, df_temp, on=["datetime", "location"])

        # ----------------------------
        # ④ 欠損ロガーを除外
        # ----------------------------
        df_merge["humidity_RH"] = pd.to_numeric(df_merge["humidity_RH"], errors="coerce")
        df_merge["temperature_C"] = pd.to_numeric(df_merge["temperature_C"], errors="coerce")

        df_merge = df_merge.dropna(subset=["humidity_RH", "temperature_C"])

        all_data.append(df_merge)

    # ----------------------------
    # ⑤ 全月のデータを統合
    # ----------------------------
    df_final = pd.concat(all_data, ignore_index=True)
    df_final = df_final.sort_values("datetime")

    st.success("統合が完了しました！（datetime 自動生成＋欠損ロガー除外）")

    st.dataframe(df_final.head(20))

    # CSV ダウンロード
    csv = df_final.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="統合データを CSV でダウンロード",
        data=csv,
        file_name="museum_env_all.csv",
        mime="text/csv"
    )
