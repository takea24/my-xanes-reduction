import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import io
import os

# -----------------------------
# 定数
# -----------------------------
HC = 12398.419      # eV*Å
D_SI111 = 3.1356    # Å
PULSES_PER_DEG = 36000
DEG2RAD = np.pi / 180.0
E0_FE = 7111.08     # 基準鉄foil第一変曲点

# -----------------------------
# パルス→エネルギー変換
# -----------------------------
def pulse_to_energy(pulse, pulse_e0):
    theta0 = np.arcsin(HC / (2.0 * D_SI111 * E0_FE))
    dtheta = (pulse - pulse_e0) / PULSES_PER_DEG * DEG2RAD
    theta = theta0 + dtheta
    E = HC / (2.0 * D_SI111 * np.sin(theta))
    return E

# -----------------------------
# ガウス関数
# -----------------------------
def gaussian(E, A, mu, sigma):
    return np.abs(A) * np.exp(-(E - mu)**2 / (2*sigma**2))

def two_gauss(E, A1, mu1, sigma1, A2, mu2, sigma2):
    return gaussian(E, A1, mu1, sigma1) + gaussian(E, A2, mu2, sigma2)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("XANES Multiple File Fitting App")

uploaded_files = st.file_uploader(
    "Select multiple .dat files",
    type=["dat", "txt"],
    accept_multiple_files=True
)

use_previous = st.checkbox("Use previous pulse reference from first app", value=False)
pulse_ref = None
if not use_previous:
    pulse_ref = st.number_input("Enter pulse reference", min_value=0.0, value=581650.0)

if uploaded_files:
    for uploaded_file in uploaded_files:
        basename = os.path.splitext(uploaded_file.name)[0]
        # CSV 読み込み
        try:
            df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')), skiprows=3, header=None)
        except Exception as e:
            st.error(f"Failed to read {uploaded_file.name}: {e}")
            continue

        pulse = df[0].values
        I0 = df[1].values
        Fe_Ka = df[2].values
        Fe_Ka_norm = Fe_Ka / I0

        # エネルギー変換
        pulse_used = pulse_ref if pulse_ref is not None else 581650
        energy = pulse_to_energy(pulse, pulse_used)
        Fe_Ka_smooth = gaussian_filter1d(Fe_Ka_norm, sigma=1)

        # 昇順ソート
        sort_idx = np.argsort(energy)
        energy = energy[sort_idx]
        Fe_Ka_norm = Fe_Ka_norm[sort_idx]
        Fe_Ka_smooth = Fe_Ka_smooth[sort_idx]

        # 自動直線ベースライン
        mask_low = energy <= 7109
        mask_high = energy >= 7114

        E_low = energy[mask_low][np.argmax(Fe_Ka_smooth[mask_low])]
        I_low = Fe_Ka_smooth[mask_low][np.argmax(Fe_Ka_smooth[mask_low])]

        E_high = energy[mask_high][np.argmin(Fe_Ka_smooth[mask_high])]
        I_high = Fe_Ka_smooth[mask_high][np.argmin(Fe_Ka_smooth[mask_high])]

        m_lin = (I_high - I_low) / (E_high - E_low)
        c_lin = I_low - m_lin * E_low
        baseline = m_lin * energy + c_lin

        # プレエッジ2ガウスフィット（7110-7115 eV）
        mask_gauss = (energy >= 7110) & (energy <= 7115)
        E_gauss = energy[mask_gauss]
        I_gauss = Fe_Ka_smooth[mask_gauss] - baseline[mask_gauss]

        p0_gauss = [0.1, 7111.8, 0.5, 0.1, 7113.7, 0.5]
        lower_gauss = [0, 7110, 0, 0, 7112, 0]
        upper_gauss = [np.inf, 7112, 2, np.inf, 7115, 2]

        try:
            popt_gauss, _ = curve_fit(two_gauss, E_gauss, I_gauss,
                                      p0=p0_gauss, bounds=(lower_gauss, upper_gauss),
                                      maxfev=5000)
        except Exception as e:
            st.warning(f"Gaussian fit failed for {basename}: {e}")
            continue

        gauss_fit = two_gauss(E_gauss, *popt_gauss)

        # 面積加重重心
        area1 = popt_gauss[0] * popt_gauss[2] * np.sqrt(2*np.pi)
        area2 = popt_gauss[3] * popt_gauss[5] * np.sqrt(2*np.pi)
        centroid = (popt_gauss[1]*area1 + popt_gauss[4]*area2) / (area1 + area2)

        # -----------------------------
        # Plotly でインタラクティブ表示
        # -----------------------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=energy, y=Fe_Ka_norm, mode='markers', name='raw', marker=dict(color='black', opacity=0.5)))
        fig.add_trace(go.Scatter(x=energy, y=Fe_Ka_smooth, mode='lines', name='smoothed', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=energy, y=baseline, mode='lines', name='baseline', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=E_gauss, y=gauss_fit + baseline[mask_gauss], mode='lines', name='total fit', line=dict(color='blue')))
        fig.add_vline(x=centroid, line=dict(color='blue', dash='dot'), annotation_text=f"Centroid={centroid:.2f}", annotation_position="top right")

        fig.update_layout(
            title=basename,
            xaxis_title="Energy (eV)",
            yaxis_title="Normalized intensity",
            xaxis_range=[7108, 7116]
        )

        st.plotly_chart(fig, use_container_width=True, key=f"fig_{basename}")

        # -----------------------------
        # Matplotlib で PNG 保存
        # -----------------------------
        fig_mpl, ax = plt.subplots(figsize=(10,6))
        ax.plot(energy, Fe_Ka_norm, 'ko', alpha=0.5, label='raw data')
        ax.plot(energy, Fe_Ka_smooth, 'k-', alpha=0.8, label='smoothed')
        ax.plot(energy, baseline, 'r--', linewidth=2, label='baseline')
        ax.plot(E_gauss, gauss_fit + baseline[mask_gauss], 'b-', linewidth=2, label='total fit')
        ax.axvline(centroid, color='blue', linestyle=':', label=f'Centroid={centroid:.2f}')
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Normalized intensity")
        ax.set_xlim(7108, 7116)
        # 縦軸自動調整
        mask_ylim = (energy >= 7114) & (energy <= 7116)
        ylim_max = np.ceil(Fe_Ka_smooth[mask_ylim].max() / 0.01) * 0.01
        ax.set_ylim(0, ylim_max)
        ax.legend()
        plt.tight_layout()

        png_buffer = io.BytesIO()
        fig_mpl.savefig(png_buffer, format='png', dpi=300)
        plt.close(fig_mpl)
        png_buffer.seek(0)

        st.download_button(
            f"Download {basename} PNG",
            data=png_buffer,
            file_name=f"{basename}_fitting.png",
            mime="image/png",
            key=f"download_{basename}"
        )

        # -----------------------------
        # テキスト結果出力
        # -----------------------------
        txt_lines = [
            f"Linear baseline: slope={m_lin:.6f}, intercept={c_lin:.6f}",
            f"Gaussian peaks:",
            f"  Peak1: mu={popt_gauss[1]:.3f}, sigma={popt_gauss[2]:.3f}, A={popt_gauss[0]:.3f}",
            f"  Peak2: mu={popt_gauss[4]:.3f}, sigma={popt_gauss[5]:.3f}, A={popt_gauss[3]:.3f}",
            f"Centroid (area-weighted) = {centroid:.3f} eV"
        ]
        txt_content = "\n".join(txt_lines)
        st.text_area(f"Fitting results: {basename}", value=txt_content, height=200)
        st.download_button(
            f"Download {basename} TXT",
            data=txt_content,
            file_name=f"{basename}_fitting.txt",
            mime="text/plain",
            key=f"txt_{basename}"
        )
