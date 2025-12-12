# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import io
import zipfile
import matplotlib.colors as mcolors

from datetime import datetime

import matplotlib.font_manager as fm


# meteostat はオプション扱い
try:
    from meteostat import Point, Hourly
    METEOSTAT_AVAILABLE = True
except:
    METEOSTAT_AVAILABLE = False

st.title("館内温湿度ロガー解析（外気比較付き）")

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

        # 🔥 外気を location='outdoor' として追加する処理
        outdoor_long = outdoor.rename(columns={
            "outdoor_temp": "temperature_C",
            "outdoor_rh": "humidity_RH"
        }).copy()
        outdoor_long["location"] = "meteostat(Kyoto)"

        df_merged = pd.concat([df_merged, outdoor_long[["datetime", "location", "temperature_C", "humidity_RH"]]])
        df_merged = df_merged.sort_values("datetime").reset_index(drop=True)

    else:
        df_merged = df.copy()
        df_merged["outdoor_temp"] = np.nan
        df_merged["outdoor_rh"] = np.nan


    st.markdown("<hr style='border:1.5px solid #bbb;'>", unsafe_allow_html=True)


    # ----------------------------
    # 8. ロガー間比較 （任意期間）
    # ----------------------------
    st.subheader("ロガー間比較(重ねて表示)：任意期間の温度・湿度")

    # ----------------------------
    # 表示するロガー選択（複数可）
    # ----------------------------
    logger_list = sorted(df_merged["location"].unique().tolist())
    selected_compare_locs = st.multiselect(
        "表示するロガーを選んでください（複数可）",
        logger_list,
        default=logger_list  # 初期は全部
    )



    # ----------------------------
    # 日付範囲指定
    # ----------------------------
    min_date = df_merged["datetime"].min().date()
    max_date = df_merged["datetime"].max().date()

    start_date, end_date = st.date_input(
        "表示する期間を選んでください",
        value=(max_date - pd.Timedelta(days=7), max_date),
        min_value=min_date,
        max_value=max_date
    )

    # datetime に変換
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)

    # ----------------------------
    # データ抽出：ロガー選択 + 日付範囲
    # ----------------------------
    selected_period = df_merged[
        (df_merged["location"].isin(selected_compare_locs)) &
        (df_merged["datetime"] >= start_dt) &
        (df_merged["datetime"] < end_dt)
    ]

    # ----------------------------
    # プロット：温度
    # ----------------------------
    fig, ax = plt.subplots(figsize=(10,5))

    # ロガー別にプロット
    for loc in selected_compare_locs:
        sub = selected_period[selected_period["location"] == loc]
        ax.plot(sub["datetime"], sub["temperature_C"], label=loc)

    ax.legend()
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Period: {start_date} ~ {end_date}")
    st.pyplot(fig)

    # ----------------------------
    # プロット：湿度
    # ----------------------------
    fig_h, ax_h = plt.subplots(figsize=(10,5))

    for loc in selected_compare_locs:
        sub = selected_period[selected_period["location"] == loc]
        ax_h.plot(sub["datetime"], sub["humidity_RH"], label=loc)

    ax_h.legend()
    ax_h.set_ylabel("Relative Humidity (%)")
    ax_h.set_title(f"Period: {start_date} ~ {end_date}")
    st.pyplot(fig_h)

    st.markdown("<hr style='border:1.5px solid #bbb;'>", unsafe_allow_html=True)

    # ----------------------------
    # 4. ロガー選択（複数選択）
    # ----------------------------
    st.subheader("ロガー別グラフ（個別保存用）")

    locations = sorted(df_merged["location"].unique())
    selected_locs = st.multiselect(
        "表示するロガーを選んでください（複数可）",
        options=locations,
        default=[locations[0]]  # 初期値
    )

    # ----------------------------
    # 期間選択（横軸の幅）
    # ----------------------------
    st.subheader("表示期間の選択")

    df_selected = df_merged[df_merged["location"].isin(selected_locs)]
    min_time = pd.to_datetime(df_selected["datetime"].min()).to_pydatetime()
    max_time = pd.to_datetime(df_selected["datetime"].max()).to_pydatetime()

    start_time, end_time = st.slider(
        "表示する期間を選択してください",
        min_value=min_time,
        max_value=max_time,
        value=(min_time, max_time),
        format="YYYY-MM-DD HH:mm"
    )

    df_view = df_selected[(df_selected["datetime"] >= start_time) & (df_selected["datetime"] <= end_time)]

    # ----------------------------
    # 各ロガーごとに個別グラフを作成
    # ----------------------------
    for loc in selected_locs:
        df_loc = df_view[df_view["location"] == loc]

        # 温度グラフ
        st.subheader(f"{loc} の温度の比較（館内 vs 外気）")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_loc["datetime"], df_loc["temperature_C"], label=f"{loc}(KUM)")
        if outdoor is not None:
            ax.plot(df_loc["datetime"], df_loc["outdoor_temp"], label="Kyoto Meteostat", alpha=0.6)
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        st.pyplot(fig)

        # 湿度グラフ
        st.subheader(f"{loc} の湿度の比較（館内 vs 外気）")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_loc["datetime"], df_loc["humidity_RH"], label=f"{loc}(KUM)")
        if outdoor is not None:
            ax.plot(df_loc["datetime"], df_loc["outdoor_rh"], label="Kyoto Meteostat", alpha=0.6)
        ax.set_ylabel("Relative Humidity (%)")
        ax.legend()
        st.pyplot(fig)

    if st.button("📥 選択ロガーの全グラフを一括保存（PNG）"):

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zip_file:

            for loc in selected_locs:
                df_loc = df_view[df_view["location"] == loc]

                # --- 温度グラフ ---
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(df_loc["datetime"], df_loc["temperature_C"], label=f"{loc}(KUM)")
                if outdoor is not None:
                    ax.plot(df_loc["datetime"], df_loc["outdoor_temp"], label="Kyoto Meteostat", alpha=0.6)
                ax.set_ylabel("Temperature (°C)")
                ax.legend()
                ax.set_title(f"{loc} 温度")
                
                png_bytes = io.BytesIO()
                fig.savefig(png_bytes, format="png", bbox_inches="tight")
                plt.close(fig)
                zip_file.writestr(f"{loc}_temperature.png", png_bytes.getvalue())

                # --- 湿度グラフ ---
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(df_loc["datetime"], df_loc["humidity_RH"], label=f"{loc}(KUM)")
                if outdoor is not None:
                    ax.plot(df_loc["datetime"], df_loc["outdoor_rh"], label="Kyoto Meteostat", alpha=0.6)
                ax.set_ylabel("Relative Humidity (%)")
                ax.legend()
                ax.set_title(f"{loc} 湿度")
                
                png_bytes = io.BytesIO()
                fig.savefig(png_bytes, format="png", bbox_inches="tight")
                plt.close(fig)
                zip_file.writestr(f"{loc}_humidity.png", png_bytes.getvalue())

        # ZIP をダウンロード
        st.download_button(
            label="📥 ZIP をダウンロード",
            data=zip_buffer.getvalue(),
            file_name="logger_graphs.zip",
            mime="application/zip"
        )

    st.markdown("<hr style='border:1.5px solid #bbb;'>", unsafe_allow_html=True)




    # ----------------------------
    # 7. 月別クリモグラフ（ロガー別 Temp–RH）
    # ----------------------------
    st.subheader("月別クリモグラフ（Temp–RH、ロガー別選択）")

    # ★ 年・月・年月を作る（全てここで作る）
    df_merged["year"]  = df_merged["datetime"].dt.year
    df_merged["month"] = df_merged["datetime"].dt.month
    df_merged["ym"]    = df_merged["datetime"].dt.to_period("M")

    # ロガー一覧
    logger_list = sorted(df_merged["location"].unique().tolist())

    selected_loggers = st.multiselect(
        "プロットするロガーを選択してください：",
        options=logger_list,
        default=[logger_list[0]] if len(logger_list) > 0 else []  # ここを修正
    )



    import plotly.graph_objects as go

    fig = go.Figure()

    # ==========================================
    # クリモグラフ表示用データ作成
    # ==========================================
    all_monthly = []

    for lg in selected_loggers:

        # ★ 年月単位で平均
        monthly = (
            df_merged[df_merged["location"] == lg]
            .groupby(["year", "month", "ym"])
            .agg(
                temperature=("temperature_C", "mean"),
                humidity=("humidity_RH", "mean")
            )
            .reset_index()
            .assign(logger=lg)
        )

        all_monthly.append(monthly)

        # 年月の昇順
        monthly = monthly.sort_values(["year", "month"])

        # 表示用ラベル
        monthly["label"] = monthly["ym"].astype(str)

        fig.add_trace(
            go.Scatter(
                x=monthly["humidity"],
                y=monthly["temperature"],
                mode="lines+markers+text",
                name=lg,
                text=monthly["label"],   # ← 年月表示
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

    # ==========================================
    # クリモグラフに使用したデータを結合
    # ==========================================
    df_monthly_all = pd.concat(all_monthly, ignore_index=True)

    # ==========================================
    # データ表示
    # ==========================================
    st.subheader("クリモグラフで使用した月別平均データ")
    st.dataframe(df_monthly_all)

    # ==========================================
    # CSV ダウンロード
    # ==========================================
    csv = df_monthly_all.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 月別平均データを CSV でダウンロード",
        data=csv,
        file_name="climograph_monthly_data.csv",
        mime="text/csv"
    )

    st.markdown("<hr style='border:1.5px solid #bbb;'>", unsafe_allow_html=True)

    # ================================
    # ① 月別箱ひげ図（ロガー別の季節変動）
    # ================================
    st.subheader("月別箱ひげ図（年別）")

    # 年・月を抽出（なければ追加）
    df_merged["year"] = df_merged["datetime"].dt.year
    df_merged["month"] = df_merged["datetime"].dt.month

    # ロガー選択
    logger_for_box = st.selectbox(
        "箱ひげ図を表示するロガーを選択してください",
        sorted(df_merged["location"].unique())
    )

    df_box = df_merged[df_merged["location"] == logger_for_box]

    st.write("箱：50%のデータ範囲(ばらつきの指標)+中央値")
    st.write("ヒゲ：箱外の最大/最小値の1.5倍までの範囲（通常のデータ範囲）")
    st.write("それ以外はハズレ値")


    # 温度の箱ひげ図
    fig_temp = px.box(
        df_box,
        x="month",
        y="temperature_C",
        color="year",
        points="outliers",
        title=f"{logger_for_box} の月別温度（年別）箱ひげ図",
        labels={"month": "月", "temperature_C": "温度 (°C)", "year": "年"},
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    # 湿度の箱ひげ図
    fig_hum = px.box(
        df_box,
        x="month",
        y="humidity_RH",
        color="year",
        points="outliers",
        title=f"{logger_for_box} の月別湿度（年別）箱ひげ図",
        labels={"month": "月", "humidity_RH": "湿度 (%)", "year": "年"},
    )
    st.plotly_chart(fig_hum, use_container_width=True)


    st.header("📦 各ロガーの箱ひげ図を ZIP で一括ダウンロード")
    if st.button("ZIP を生成してダウンロード（HTML）"):

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zip_file:

            for logger in df_merged["location"].unique():
                dlog = df_merged[df_merged["location"] == logger]

                # 温度の箱ひげ図
                fig_temp = px.box(
                    dlog,
                    x="month",
                    y="temperature_C",
                    color="year",
                    points="outliers",
                    title=f"{logger} の月別温度（年別）箱ひげ図"
                )

                # 湿度の箱ひげ図
                fig_hum = px.box(
                    dlog,
                    x="month",
                    y="humidity_RH",
                    color="year",
                    points="outliers",
                    title=f"{logger} の月別湿度（年別）箱ひげ図"
                )

                # HTML 文字列として ZIP に追加
                zip_file.writestr(f"{logger}_temperature_boxplot.html", fig_temp.to_html(full_html=True))
                zip_file.writestr(f"{logger}_humidity_boxplot.html", fig_hum.to_html(full_html=True))

        st.download_button(
            label="📥 ZIP をダウンロード",
            data=zip_buffer.getvalue(),
            file_name="logger_boxplots.zip",
            mime="application/zip",
        )

    if st.button("ZIP を生成してダウンロード"):

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zip_file:

            for logger in df_merged["location"].unique():
                dlog = df_merged[df_merged["location"] == logger]

                for var, ylabel in [("temperature_C", "Temperature (°C)"),
                                    ("humidity_RH", "Relative Humidity (%)")]:

                    # 年ごとに色を決める
                    years = sorted(dlog["year"].unique())
                    colors = plt.cm.tab10.colors  # 10色まで

                    fig, ax = plt.subplots(figsize=(10,6))

                    positions = []
                    labels = []

                    for i, year in enumerate(years):
                        for month in range(1, 13):
                            data = dlog[(dlog["year"] == year) & (dlog["month"] == month)][var]
                            if len(data) == 0:
                                continue
                            pos = len(positions) + 1
                            positions.append(pos)
                            labels.append(f"{month}-{year}")
                            bplot = ax.boxplot(data,
                                               positions=[pos],
                                               widths=0.6,
                                               patch_artist=True,
                                               boxprops=dict(facecolor=colors[i % len(colors)], alpha=0.6),
                                               flierprops=dict(marker='o', color=colors[i % len(colors)], alpha=0.6)
                                               )

                    ax.set_xticks(range(1, len(positions)+1))
                    ax.set_xticklabels(labels, rotation=45, ha="right")
                    ax.set_ylabel(ylabel)
                    ax.set_title(f"{logger} - {ylabel} Boxplot (Month-Year)")

                    fig.tight_layout()

                    # PNG 保存
                    png_bytes = io.BytesIO()
                    fig.savefig(png_bytes, format="png", bbox_inches="tight")
                    plt.close(fig)

                    zip_file.writestr(f"{logger}_{var}_boxplot.png", png_bytes.getvalue())

        st.download_button(
            label="📥 ZIP をダウンロード",
            data=zip_buffer.getvalue(),
            file_name="logger_boxplots_matplotlib.zip",
            mime="application/zip",
        )

    st.markdown("<hr style='border:1.5px solid #bbb;'>", unsafe_allow_html=True)
            
    # ================================
    # ② ロガー間の相関マトリクス
    # ================================

    st.subheader("ロガー間の相関マトリクス（温度・湿度）")

    # --- ロガー×時間 の pivot（温度）
    temp_pivot = df_merged.pivot_table(
        index="datetime",
        columns="location",
        values="temperature_C"
    )

    # --- ロガー×時間 の pivot（湿度）
    rh_pivot = df_merged.pivot_table(
        index="datetime",
        columns="location",
        values="humidity_RH"
    )

    # 相関計算
    temp_corr = temp_pivot.corr()
    rh_corr = rh_pivot.corr()

    # カラーマップは相関係数が高いほど濃い色にする（Blues系）
    color_map = "Blues"

    # 温度相関マトリクス
    fig_temp_corr = px.imshow(
        temp_corr,
        text_auto=".2f",         # 有効数字2桁
        aspect="auto",
        color_continuous_scale=color_map,
        title="ロガー・外気間の相関（温度）"
    )
    st.plotly_chart(fig_temp_corr, use_container_width=True)

    # 湿度相関マトリクス
    fig_rh_corr = px.imshow(
        rh_corr,
        text_auto=".2f",         # 有効数字2桁
        aspect="auto",
        color_continuous_scale=color_map,
        title="ロガー・外気間の相関（湿度）"
    )
    st.plotly_chart(fig_rh_corr, use_container_width=True)

    st.caption("相関係数 1.0 に近いほど、温度/湿度の変動パターンが似ているロガーです。")

    st.markdown("<hr style='border:1.5px solid #bbb;'>", unsafe_allow_html=True)

    # ================================
    # ③ 保存基準との比較（達成率）
    # ================================
    st.subheader("保存基準との比較（ロガー別診断）")

    # --- 基準値をユーザー入力で設定
    # 温度基準 (float)
    TEMP_LOW = st.number_input("最低温度", value=18.0, step=0.1, format="%.1f")
    TEMP_HIGH = st.number_input("最高温度", value=22.0, step=0.1, format="%.1f")

    # 湿度基準 (整数で良ければ int に統一)
    RH_LOW = st.number_input("最低湿度", value=40, step=1, format="%d")
    RH_HIGH = st.number_input("最高湿度", value=55, step=1, format="%d")


    # --- ロガー別診断
    logger_summary = []

    for lg in sorted(df_merged["location"].unique()):
        sub = df_merged[df_merged["location"] == lg]
        total = len(sub)

        temp_good = ((sub["temperature_C"] >= TEMP_LOW) & (sub["temperature_C"] <= TEMP_HIGH)).sum()
        rh_good = ((sub["humidity_RH"] >= RH_LOW) & (sub["humidity_RH"] <= RH_HIGH)).sum()

        logger_summary.append({
            "location": lg,
            "総サンプル数": total,
            "温度基準範囲": f"{TEMP_LOW}–{TEMP_HIGH} °C",
            "温度が基準内 (%)": round(temp_good / total * 100, 1),
            "湿度基準範囲": f"{RH_LOW}–{RH_HIGH} %",
            "湿度が基準内 (%)": round(rh_good / total * 100, 1),
        })

    df_criteria = pd.DataFrame(logger_summary)

    st.dataframe(df_criteria, use_container_width=True)

    # --- CSV ダウンロード
    csv_criteria = df_criteria.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 保存基準比較の結果を CSV ダウンロード",
        data=csv_criteria,
        file_name="environment_criteria_report.csv",
        mime="text/csv"
    )


else:
    st.info("館内データ CSV をアップロードしてください。")
