# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import matplotlib.font_manager as fm


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

    start = df["datetime"].min() - pd.Timedelta(hours=9)
    end = df["datetime"].max()

    st.markdown(f"期間: **{start} 〜 {end}** の外気データ")

    outdoor = None

    if METEOSTAT_AVAILABLE:
        # 左京区（吉田本町付近）
        kyoto_point = Point(35.03, 135.78, 50)

        with st.spinner("外気データ取得中..."):
            try:
                outdoor = Hourly(kyoto_point, start, end).fetch()
            except Exception as e:
                st.error(f"Meteostat 取得エラー: {e}")
                outdoor = None

        if outdoor is not None and len(outdoor) > 0:

            # タイムゾーン処理
            outdoor.index = outdoor.index.tz_localize("UTC").tz_convert("Asia/Tokyo")
            outdoor.index = outdoor.index.tz_localize(None)

            # 列に戻す
            outdoor = outdoor.reset_index()

            # 列名統一
            outdoor = outdoor.rename(columns={
                "time": "datetime",
                "temp": "outdoor_temp",
                "rhum": "outdoor_rh"
            })

            # ------------------------------
            # ここで 30分刻みに補間する
            # ------------------------------
            outdoor = outdoor.set_index("datetime")
            outdoor = outdoor.resample("30T").interpolate()
            outdoor = outdoor.reset_index()

            st.success("外気データ取得 + 30分補間 完了")
            st.write(outdoor.head())

        else:
            st.warning("外気データが取得できなかったため、外気比較はスキップします。")
            outdoor = None

    else:
        st.warning("Meteostat がインストールされていません。外気比較はスキップします。")

    
    # ----------------------------
    # 3. データ結合（datetime で結合）
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
    st.write(df_merged[["datetime", "temperature_C", "outdoor_temp", "outdoor_rh"]].head())
    st.write("外気温 NaN 数:", df_merged["outdoor_temp"].isna().sum())

    # ----------------------------
    # 4. ロガー選択
    # ----------------------------
    st.subheader("ロガー選択")

    locations = df_merged["location"].unique()
    selected_loc = st.selectbox("表示するロガーを選んでください", locations)

    df_loc = df_merged[df_merged["location"] == selected_loc]

    # ----------------------------
    # 期間選択（横軸の幅）
    # ----------------------------
    st.subheader("表示期間の選択")

    min_time = pd.to_datetime(df_loc["datetime"].min()).to_pydatetime()
    max_time = pd.to_datetime(df_loc["datetime"].max()).to_pydatetime()

    start_time, end_time = st.slider(
        "表示する期間を選択してください",
        min_value=min_time,
        max_value=max_time,
        value=(min_time, max_time),
        format="YYYY-MM-DD HH:mm"
    )

    # 選択期間でフィルタ
    df_view = df_loc[(df_loc["datetime"] >= start_time) & (df_loc["datetime"] <= end_time)]


    # ----------------------------
    # 5. 温度：館内 vs 外気
    # ----------------------------
    st.subheader("温度の比較（館内 vs 外気）")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_view["datetime"], df_view["temperature_C"], label=f"{selected_loc}(KUM)")

    if outdoor is not None:
        ax.plot(df_view["datetime"], df_view["outdoor_temp"], label="Kyoto Meteostat", alpha=0.6)

    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    st.pyplot(fig)

    # ----------------------------
    # 6. 湿度：館内 vs 外気
    # ----------------------------
    st.subheader("湿度の比較（館内 vs 外気）")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_view["datetime"], df_view["humidity_RH"], label=f"{selected_loc}(KUM)")

    if outdoor is not None:
        ax.plot(df_view["datetime"], df_view["outdoor_rh"], label="Kyoto Meteostat", alpha=0.6)

    ax.set_ylabel("Relative Humidity (%)")
    ax.legend()
    st.pyplot(fig)

    # ----------------------------
    # 7. 月別クリモグラフ（ロガー別 Temp–RH）
    # ----------------------------
    st.subheader("月別クリモグラフ（Temp–RH、ロガー別選択）")

    # ★ 年 + 月 の Period を使う
    df_merged["ym"] = df_merged["datetime"].dt.to_period("M")

    # ロガー一覧
    logger_list = sorted(df_merged["location"].unique().tolist())

    # 複数ロガー選択
    selected_loggers = st.multiselect(
        "プロットするロガーを選択してください：",
        logger_list,
        default=[selected_loc] # デフォルトは現在表示中のロガーだけ
    )

    import plotly.graph_objects as go

    fig = go.Figure()
    # ---------------------------------------------------------
    # 境界線の関数を定義
    # ---------------------------------------------------------

    T = np.linspace(0, 40, 400)

    # (25,60) - (15,80)
    H1 = 60 + (80 - 60) / (15 - 25) * (T - 25)

    # (15,80) - (10,100)
    H2 = 80 + (100 - 80) / (10 - 15) * (T - 15)

    # (25,80) - (10,90)
    H3 = 80 + (90 - 80) / (10 - 25) * (T - 25)

    # =========================================================
    # ① 淡赤（ライトレッド）領域
    # =========================================================

    # A: 温度 ≥25 & 湿度 ≥ 60
    Ta = np.array([25, 40, 40, 25])
    Ha = np.array([60, 60, 100, 100])

    # B: 湿度60–80 & H>=H1
    mask1 = (T <= 25) & (T >= 15)
    Tb1 = T[mask1]
    Hb1 = H1[mask1]

    # C: 湿度80–100 & H>=H2
    mask2 = (T <= 15) & (T >= 10)
    Tc1 = T[mask2]
    Hc1 = H2[mask2]

    # 多角形として結合
    T_light = np.concatenate([
        Ta,
        Tb1[::-1],
        Tc1[::-1]
    ])
    H_light = np.concatenate([
        Ha,
        Hb1[::-1],
        Hc1[::-1]
    ])

    fig.add_trace(go.Scatter(
        x=T_light,
        y=H_light,
        fill="toself",
        fillcolor="rgba(255,150,150,0.4)",
        line=dict(color="rgba(0,0,0,0)"),
        name="light red region"
    ))

    # =========================================================
    # ② 赤（レッド）領域
    # =========================================================

    # A: 温度≥25 & 湿度≥80
    Ta2 = np.array([25, 40, 40, 25])
    Ha2 = np.array([80, 80, 100, 100])

    # B: 湿度80–100 & H>=H3
    mask3 = (T <= 25) & (T >= 10)
    Tb2 = T[mask3]
    Hb2 = H3[mask3]

    T_red = np.concatenate([
        Ta2,
        Tb2[::-1]
    ])
    H_red = np.concatenate([
        Ha2,
        Hb2[::-1]
    ])

    fig.add_trace(go.Scatter(
        x=T_red,
        y=H_red,
        fill="toself",
        fillcolor="rgba(255,0,0,0.5)",
        line=dict(color="rgba(0,0,0,0)"),
        name="red region"
    ))

    # 選ばれたロガーごとに作図
    for lg in selected_loggers:

        # ★ 年月ごとに平均
        monthly = (
            df_merged[df_merged["location"] == lg]
            .groupby("ym")
            .agg(
                temperature=("temperature_C", "mean"),
                humidity=("humidity_RH", "mean")
            )
            .reset_index()
        )

        # ★ プロットの順序を chronological に
        monthly = monthly.sort_values("ym")

        # ★ ホバーとラベル用（2024-01 形式）
        monthly["label"] = monthly["ym"].astype(str)

        fig.add_trace(
            go.Scatter(
                x=monthly["humidity"],
                y=monthly["temperature"],
                mode="lines+markers+text",
                name=lg,
                text=monthly["label"],          # ← 年月ラベル表示
                textposition="middle right",
                hovertemplate=(
                    "年月: %{text}<br>"
                    "湿度: %{x:.1f}%<br>"
                    "温度: %{y:.1f}℃<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title="月別クリモグラフ（温度 vs 湿度）",
        xaxis_title="湿度 (%)",
        yaxis_title="温度 (°C)",
        width=800,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # 8. ロガー間比較（任意期間）
    # ----------------------------
    st.subheader("ロガー間比較：任意期間の温度")

    # --- 日付範囲指定 ---
    min_date = df_merged["datetime"].min().date()
    max_date = df_merged["datetime"].max().date()

    start_date, end_date = st.date_input(
        "表示する期間を選んでください",
        value=(max_date - pd.Timedelta(days=7), max_date),
        min_value=min_date,
        max_value=max_date
    )

    # 入力された日付を datetime に変換
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # 当日分を含めるため

    # --- データ抽出 ---
    selected_period = df_merged[
        (df_merged["datetime"] >= start_dt) &
        (df_merged["datetime"] < end_dt)
    ]

    # --- プロット ---
    fig, ax = plt.subplots(figsize=(10,5))

    for loc in selected_period["location"].unique():
        ax.plot(
            selected_period[selected_period["location"] == loc]["datetime"],
            selected_period[selected_period["location"] == loc]["temperature_C"],
            label=loc
        )

    ax.legend()
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Period: {start_date} ~ {end_date}")
    st.pyplot(fig)


else:
    st.info("館内データ CSV をアップロードしてください。")
