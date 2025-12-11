# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from meteostat import Point, Hourly
from datetime import datetime

st.title("館内温湿度モニタリング（外気比較付き）")

# ----------------------------
# 1. CSV アップロード
# ----------------------------
uploaded = st.file_uploader("館内温湿度データ CSV をアップロードしてください", type=["csv"])

if uploaded:
    # ヘッダーを読む
    df_head = pd.read_csv(uploaded, nrows=5)
    uploaded.seek(0)

    # datetime の候補
    datetime_candidates = ["datetime", "date", "date/time", "time", "time30", "timestamp"]

    lower_cols = [c.lower() for c in df_head.columns]

    dt_col = None
    for cand in datetime_candidates:
        if cand in lower_cols:
            dt_col = df_head.columns[lower_cols.index(cand)]
            break

    if dt_col is None:
        st.error("日時列が見つかりませんでした（datetime, time, time30 など）。")
    else:
        st.success(f"日時列として '{dt_col}' を使用します。")

        df = pd.read_csv(uploaded, parse_dates=[dt_col])
        df = df.rename(columns={dt_col: "datetime"})

        st.write(df.head())

    
    # ----------------------------
    # 2. 外気データの取得（京都 左京区付近）
    # ----------------------------
    st.subheader("外気データ取得（京都市左京区付近）")

    start = df["datetime"].min()
    end = df["datetime"].max()

    st.markdown(f"期間: **{start} 〜 {end}** の外気データを取得します")

    # 京都市左京区に最も近い気象庁観測点（Meteostat: Kyoto）
    kyoto_point = Point(35.03, 135.78, 50)

    with st.spinner("外気データ取得中..."):
        outdoor = Hourly(kyoto_point, start, end)
        outdoor = outdoor.fetch()

    outdoor = outdoor.rename(columns={"temp": "outdoor_temp", "rhum": "outdoor_rh"})
    outdoor.index.name = "datetime"
    outdoor.reset_index(inplace=True)

    st.success("外気データを取得しました！")

    # ----------------------------
    # 3. データ結合
    # ----------------------------
    df_merged = pd.merge(
        df,
        outdoor[["datetime", "outdoor_temp", "outdoor_rh"]],
        on="datetime",
        how="left"
    )

    # ----------------------------
    # 4. ロガー選択
    # ----------------------------
    st.subheader("ロガー選択")
    locations = df["location"].unique()
    selected_loc = st.selectbox("表示するロガーを選んでください", locations)

    df_loc = df_merged[df_merged["location"] == selected_loc]

    # ----------------------------
    # 5. グラフ：館内 vs 外気（温度）
    # ----------------------------
    st.subheader("温度の比較（館内 vs 外気）")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_loc["datetime"], df_loc["temperature_C"], label=f"{selected_loc}（館内）")
    ax.plot(df_loc["datetime"], df_loc["outdoor_temp"], label="京都（外気）", alpha=0.6)
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)

    # ----------------------------
    # 6. グラフ：湿度（館内 vs 外気）
    # ----------------------------
    st.subheader("湿度の比較（館内 vs 外気）")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_loc["datetime"], df_loc["humidity_RH"], label=f"{selected_loc}（館内）")
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

    ax1.plot(months, monthly["temperature_C"], label="Temp (°C)", color="red")
    ax1.set_ylabel("Temperature (°C)", color="red")

    ax2 = ax1.twinx()
    ax2.bar(months, monthly["humidity_RH"], alpha=0.3, label="RH (%)", color="blue")
    ax2.set_ylabel("Humidity (%)", color="blue")

    plt.title("Monthly Climatogram (館内平均)")
    st.pyplot(fig)

    # ----------------------------
    # 8. ロガー間比較
    # ----------------------------
    st.subheader("ロガー間比較：最新 1 週間の温度")

    latest_week = df_merged[df_merged["datetime"] > df_merged["datetime"].max() - pd.Timedelta(days=7)]

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
