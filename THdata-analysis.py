# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# meteostat はオプション扱い
try:
    from meteostat import Point, Hourly
    METEOSTAT_AVAILABLE = True
except:
    METEOSTAT_AVAILABLE = False

st.title("館内温湿度モニタリング（外気比較付き）")

# ----------------------------
# 1. CSV アップロード
# ----------------------------
uploaded = st.file_uploader("館内温湿度データ CSV をアップロードしてください", type=["csv"])

if uploaded:

    # --- CSV 読み込み ---
    df = pd.read_csv(uploaded)

    # あなたの CSV 向けに列名を統一
    rename_map = {
        "Time30": "datetime",
        "Temp": "temperature_C",
        "Hum": "humidity_RH",
        "Logger_norm": "location"
    }
    df = df.rename(columns=rename_map)

    # datetime を変換
    try:
        df["datetime"] = pd.to_datetime(df["datetime"])
    except:
        st.error("日時のパースに失敗しました（Time30 列が正しくない可能性）")
        st.stop()

    st.success("CSV を読み込みました")
    st.write(df.head())

    # ----------------------------
    # 2. 外気データ取得（京都 左京区付近）
    # ----------------------------
    st.subheader("外気データ取得（京都市左京区付近）")

    start = df["datetime"].min()
    end = df["datetime"].max()

    st.markdown(f"期間: **{start} 〜 {end}** の外気データ")

    outdoor = None

    if METEOSTAT_AVAILABLE:
        kyoto_point = Point(35.03, 135.78, 50)

        with st.spinner("外気データ取得中..."):
            try:
                outdoor = Hourly(kyoto_point, start, end).fetch()
            except Exception as e:
                st.error(f"Meteostat 取得エラー: {e}")
                outdoor = None
        outdoor.index = outdoor.index.tz_localize("UTC").tz_convert("Asia/Tokyo")
        outdoor.index = outdoor.index.tz_localize(None)
        outdoor.reset_index(inplace=True)

        if outdoor is not None and len(outdoor) > 0:
            outdoor = outdoor.rename(columns={
                "temp": "outdoor_temp",
                "rhum": "outdoor_rh"
            })
            outdoor.index.name = "datetime"
            outdoor.reset_index(inplace=True)
            st.success("外気データ取得完了")
        else:
            st.warning("外気データが取得できなかったため、外気比較はスキップします。")
            outdoor = None

    else:
        st.warning("Meteostat がインストールされていません。外気比較はスキップします。")

    if outdoor is not None:
        st.subheader("外気データ（先頭5行）")
        st.write(outdoor.head())
    else:
        st.warning("外気データは取得されていません。")

    st.write("df datetime dtype:", df["datetime"].dtype)
    st.write("df_out datetime dtype:", df_out["datetime"].dtype)

    st.write("df head:", df.head())
    st.write("df_out head:", df_out.head())


    # ----------------------------
    # 3. データ結合
    # ----------------------------
    if outdoor is not None:
        df_merged = pd.merge(
            df,
            outdoor[["datetime", "outdoor_temp", "outdoor_rh"]],
            on="datetime",
            how="left"
        )
    else:
        df_merged = df.copy()
        df_merged["outdoor_temp"] = np.nan
        df_merged["outdoor_rh"] = np.nan

        st.subheader("外気データ結合チェック")
        st.write(df_merged[["datetime", "temperature_C", "outdoor_temp", "outdoor_rh"]].head(20))

        st.write("外気温 NaN 数:", df_merged["outdoor_temp"].isna().sum())
        st.write("外気湿度 NaN 数:", df_merged["outdoor_rh"].isna().sum())


    # ----------------------------
    # 4. ロガー選択
    # ----------------------------
    st.subheader("ロガー選択")

    locations = df_merged["location"].unique()
    selected_loc = st.selectbox("表示するロガーを選んでください", locations)

    df_loc = df_merged[df_merged["location"] == selected_loc]

    # ----------------------------
    # 5. 温度：館内 vs 外気
    # ----------------------------
    st.subheader("温度の比較（館内 vs 外気）")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_loc["datetime"], df_loc["temperature_C"], label=f"{selected_loc}（館内）")

    if outdoor is not None:
        ax.plot(df_loc["datetime"], df_loc["outdoor_temp"], label="京都（外気）", alpha=0.6)

    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)

    # ----------------------------
    # 6. 湿度：館内 vs 外気
    # ----------------------------
    st.subheader("湿度の比較（館内 vs 外気）")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_loc["datetime"], df_loc["humidity_RH"], label=f"{selected_loc}（館内）")

    if outdoor is not None:
        ax.plot(df_loc["datetime"], df_loc["outdoor_rh"], label="京都（外気）", alpha=0.6)

    ax.set_ylabel("Relative Humidity (%)")
    ax.legend()
    st.pyplot(fig)

    # ----------------------------
    # 7. 月別クリモグラフ（館内平均）
    # ----------------------------
    st.subheader("月別クリモグラフ（館内平均）")

    df_merged["month"] = df_merged["datetime"].dt.to_period("M")

    monthly = df_merged.groupby("month").agg({
        "temperature_C": "mean",
        "humidity_RH": "mean"
    })

    fig, ax1 = plt.subplots(figsize=(10,5))
    months = monthly.index.to_timestamp()

    ax1.plot(months, monthly["temperature_C"], color="red", label="Temp (°C)")
    ax1.set_ylabel("Temperature (°C)", color="red")

    ax2 = ax1.twinx()
    ax2.bar(months, monthly["humidity_RH"], alpha=0.3, color="blue", label="RH (%)")
    ax2.set_ylabel("Humidity (%)", color="blue")

    plt.title("Monthly Climatogram (館内平均)")
    st.pyplot(fig)

    # ----------------------------
    # 8. ロガー間比較（最新1週間）
    # ----------------------------
    st.subheader("ロガー間比較：最新 1 週間の温度")

    latest_week = df_merged[df_merged["datetime"] >
                            df_merged["datetime"].max() - pd.Timedelta(days=7)]

    fig, ax = plt.subplots(figsize=(10,5))

    for loc in latest_week["location"].unique():
        ax.plot(
            latest_week[latest_week["location"] == loc]["datetime"],
            latest_week[latest_week["location"] == loc]["temperature_C"],
            label=loc
        )

    ax.legend()
    ax.set_ylabel("Temperature (°C)")
    st.pyplot(fig)

else:
    st.info("館内データ CSV をアップロードしてください。")
