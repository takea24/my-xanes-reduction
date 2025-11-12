# multi-peak-finder.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
import io
import zipfile
import csv
import re

st.set_page_config(page_title="Multi Peak Finder", layout="wide")
st.title("📈 Multi Peak Finder App")

st.markdown("""
複数のスペクトルデータ（txt, csv）をまとめて処理します。  
タブ・カンマ・スペース区切りや、余計な先頭行があるファイルにも対応します。
""")

uploaded_files = st.file_uploader("数値データを選択（複数可）", type=["txt", "csv"], accept_multiple_files=True)

st.sidebar.header("ピーク検出パラメータ")
height = st.sidebar.number_input("最小高さ (height)", value=0.0)
distance = st.sidebar.number_input("最小距離 (distance)", value=5)
prominence = st.sidebar.number_input("顕著さ (prominence)", value=0.0)


def read_numeric_data(file) -> pd.DataFrame:
    """最初に不要な行をスキップし、区切り文字を自動判定して2列の数値データを読む"""
    text = file.read().decode("utf-8", errors="ignore")
    lines = text.splitlines()

    # --- データ行の開始位置を自動検出 ---
    start_idx = 0
    for i, line in enumerate(lines):
        # 数値っぽい行を探す（例: 123.4 456.7）
        if re.match(r"^\s*[-+]?\d", line):
            start_idx = i
            break

    data_text = "\n".join(lines[start_idx:])

    # --- 区切り文字を推定 ---
    try:
        dialect = csv.Sniffer().sniff(data_text[:1000], delimiters="\t,; ")
        sep = dialect.delimiter
    except Exception:
        sep = r"\s+"

    # --- 読み込み ---
    df = pd.read_csv(io.StringIO(data_text), sep=sep, engine="python", comment="#", header=None)
    # 数値列のみ残す
    df = df.select_dtypes(include=[np.number])
    # 最初の2列を x, y として扱う
    df = df.iloc[:, :2]
    df.columns = ["x", "y"]
    return df


if uploaded_files:
    st.info(f"{len(uploaded_files)} 個のファイルを処理します。")

    zip_buffer = io.BytesIO()
    zip_archive = zipfile.ZipFile(zip_buffer, "w")
    results_summary = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        st.subheader(f"📄 {filename}")

        try:
            df = read_numeric_data(uploaded_file)
        except Exception as e:
            st.error(f"⚠️ {filename} の読み込みに失敗しました: {e}")
            continue

        # ピーク検出
        peaks, properties = find_peaks(df["y"], height=height, distance=distance, prominence=prominence)
        peaks_df = pd.DataFrame({
            "x": df["x"].iloc[peaks],
            "y": df["y"].iloc[peaks],
            "prominence": properties.get("prominences", np.nan)
        })
        peaks_df["filename"] = filename

        st.write(f"検出ピーク数: {len(peaks_df)}")

        # Plotlyプロット
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["x"], y=df["y"], mode="lines", name="Data"))
        fig.add_trace(go.Scatter(
            x=peaks_df["x"], y=peaks_df["y"],
            mode="markers+text",
            text=[f"{x:.2f}" for x in peaks_df["x"]],
            textposition="top center",
            name="Peaks",
            marker=dict(color="red", size=8, symbol="x")
        ))
        fig.update_layout(title=f"{filename}", xaxis_title="X", yaxis_title="Y")
        st.plotly_chart(fig, use_container_width=True)

        # 結果保存
        results_summary.append(peaks_df)
        img_bytes = fig.to_image(format="png")
        zip_archive.writestr(f"{filename}.png", img_bytes)

    zip_archive.close()

    if results_summary:
        summary_df = pd.concat(results_summary, ignore_index=True)
        st.subheader("📊 すべてのピーク検出結果")
        st.dataframe(summary_df)

        csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("ピーク一覧をCSVでダウンロード", csv_bytes, "all_peaks.csv", "text/csv")

        st.download_button(
            "全グラフをZIPでダウンロード (PNG)",
            data=zip_buffer.getvalue(),
            file_name="all_peak_plots.zip",
            mime="application/zip"
        )
