import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import zipfile

st.title("Multi-file Peak Finder App")

# --- ファイルアップロード ---
uploaded_files = st.file_uploader(
    "TXTファイルを複数選択",
    type=["txt"],
    accept_multiple_files=True
)

# --- スムージングオプション ---
smooth_method = st.selectbox("Smoothing method", ["None", "Gaussian", "Savitzky-Golay"])

if smooth_method == "Gaussian":
    sigma = st.slider("Gaussian sigma", 0.0, 5.0, 1.0, 0.1)
elif smooth_method == "Savitzky-Golay":
    window = st.number_input("SG window length (odd)", min_value=3, max_value=101, value=11, step=2)
    poly = st.number_input("SG polyorder", min_value=1, max_value=5, value=2)

# --- ピーク検出パラメータ ---
distance = st.slider("Minimum peak distance", 1, 50, 5)
height = st.number_input("Minimum peak height (optional, leave 0 to ignore)", value=0.0)
prominence = st.number_input("Minimum peak prominence (optional, leave 0 to ignore)", value=0.0)

if uploaded_files:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for uploaded_file in uploaded_files:
            # --- データ読み込み ---
            try:
                df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None, engine='python')
                if df.shape[1] < 2:
                    df = pd.read_csv(uploaded_file, sep=',', header=None, engine='python')
            except Exception as e:
                st.error(f"Cannot read {uploaded_file.name}: {e}")
                continue

            x = df.iloc[:,0].values
            y = df.iloc[:,1].values

            # --- スムージング ---
            if smooth_method == "Gaussian":
                y_smooth = gaussian_filter1d(y, sigma=sigma)
            elif smooth_method == "Savitzky-Golay":
                if window % 2 == 0:
                    window += 1  # 奇数に調整
                y_smooth = savgol_filter(y, window_length=window, polyorder=poly)
            else:
                y_smooth = y

            # --- ピーク検出 ---
            peak_kwargs = {"distance": distance}
            if height > 0: peak_kwargs["height"] = height
            if prominence > 0: peak_kwargs["prominence"] = prominence

            peaks_idx, properties = find_peaks(y_smooth, **peak_kwargs)
            peaks_x = x[peaks_idx]
            peaks_y = y_smooth[peaks_idx]

            # --- Plotlyで表示 ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Data"))
            fig.add_trace(go.Scatter(
                x=peaks_x,
                y=peaks_y,
                mode="markers+text",
                text=[f"{px:.2f}" for px in peaks_x],
                textposition="top center",
                name="Peaks",
                marker=dict(color="red", size=8, symbol="x")
            ))
            fig.update_layout(title=uploaded_file.name, xaxis_title="X", yaxis_title="Y")
            st.plotly_chart(fig)

            # --- MatplotlibでPNG保存 ---
            fig_mat, ax = plt.subplots()
            ax.plot(x, y, label="Data")
            ax.plot(peaks_x, peaks_y, "rx", label="Peaks")
            for px, py in zip(peaks_x, peaks_y):
                ax.text(px, py, f"{px:.2f}", fontsize=8, ha="center", va="bottom")
            ax.set_title(uploaded_file.name)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()

            img_buffer = io.BytesIO()
            fig_mat.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            zip_file.writestr(f"{uploaded_file.name}.png", img_buffer.read())
            plt.close(fig_mat)

    zip_buffer.seek(0)
    st.download_button("全グラフをZIPでダウンロード", zip_buffer, "plots.zip", "application/zip")
