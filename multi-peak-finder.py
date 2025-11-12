import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d, grey_opening
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import zipfile

st.title("Multi-file Peak Finder App with Background Subtraction and Selectable Display")

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

# --- 背景差し引き ---
do_bg_sub = st.checkbox("Background subtraction (rolling ball)")
if do_bg_sub:
    r = st.slider("Background subtraction radius", 1, 100, 10)

# --- ピーク検出パラメータ ---
distance = st.slider("Minimum peak distance", 1, 50, 5)
height = st.number_input("Minimum peak height (optional, leave 0 to ignore)", value=0.01)
prominence = st.number_input("Minimum peak prominence (optional, leave 0 to ignore)", value=0.01)

# --- 表示する要素を選択 ---
show_original = st.checkbox("Show original", value=True)
show_background = st.checkbox("Show background", value=True)
show_corrected = st.checkbox("Show corrected", value=True)
show_peaks = st.checkbox("Show peaks", value=True)

# --- メイン処理 ---
if uploaded_files:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for uploaded_file in uploaded_files:
            # --- データ読み込み（文字列行を自動スキップ） ---
            try:
                df_raw = pd.read_csv(uploaded_file, sep=None, header=None, engine='python', dtype=str)
                def is_numeric_row(row):
                    try:
                        float(row[0])
                        float(row[1])
                        return True
                    except:
                        return False
                df_numeric = df_raw[df_raw.apply(is_numeric_row, axis=1)]
                if df_numeric.empty:
                    st.warning(f"{uploaded_file.name} に有効なデータがありません")
                    continue
                df = df_numeric.iloc[:, :2].astype(float)
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
                    window += 1
                y_smooth = savgol_filter(y, window_length=window, polyorder=poly)
            else:
                y_smooth = y

            # --- 背景差し引き ---
            if do_bg_sub:
                y_bg = grey_opening(y_smooth, size=r)
                y_corrected = y_smooth - y_bg
            else:
                y_bg = np.zeros_like(y_smooth)
                y_corrected = y_smooth

            # --- ピーク検出 ---
            peak_kwargs = {"distance": distance}
            if height > 0: peak_kwargs["height"] = height
            if prominence > 0: peak_kwargs["prominence"] = prominence

            peaks_idx, properties = find_peaks(y_corrected, **peak_kwargs)
            peaks_x = x[peaks_idx]
            peaks_y = y_corrected[peaks_idx]

            # --- Plotly表示 ---
            fig = go.Figure()
            if show_original:
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Original"))
            if do_bg_sub and show_background:
                fig.add_trace(go.Scatter(x=x, y=y_bg, mode="lines", name="Background", line=dict(color="green", dash="dash")))
            if show_corrected:
                fig.add_trace(go.Scatter(x=x, y=y_corrected, mode="lines", name="Corrected", line=dict(color="orange")))
            if show_peaks:
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
            if show_original:
                ax.plot(x, y, label="Original")
            if do_bg_sub and show_background:
                ax.plot(x, y_bg, "g--", label="Background")
            if show_corrected:
                ax.plot(x, y_corrected, "orange", label="Corrected")
            if show_peaks:
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
