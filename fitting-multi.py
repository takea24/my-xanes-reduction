import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import io
import zipfile
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
# 関数
# -----------------------------
def pulse_to_energy(pulse, pulse_ref):
    theta0 = np.arcsin(HC / (2.0 * D_SI111 * E0_FE))
    dtheta = (pulse - pulse_ref) / PULSES_PER_DEG * DEG2RAD
    theta = theta0 + dtheta
    return HC / (2.0 * D_SI111 * np.sin(theta))

def gaussian(E, A, mu, sigma):
    return np.abs(A) * np.exp(-(E - mu)**2 / (2*sigma**2))

def two_gauss(E, A1, mu1, sigma1, A2, mu2, sigma2):
    return gaussian(E, A1, mu1, sigma1) + gaussian(E, A2, mu2, sigma2)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("XANES Pre-edge Gaussian Fitting (Fe foil) - Interactive (HTML Download)")

uploaded_files = st.file_uploader(
    "Select multiple .dat/.txt files for analysis",
    type=['dat', 'txt'],
    accept_multiple_files=True
)

pulse_ref = st.number_input("Enter reference pulse", value=581650.0)

if uploaded_files and pulse_ref:

    # ZIP作成用バッファ（テキストまとめ用）
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:

        for uploaded_file in uploaded_files:
            basename = os.path.splitext(uploaded_file.name)[0]

            st.subheader(f"Processing: {basename}")

            # -----------------------------
            # データ読み込み
            # -----------------------------
            data = pd.read_csv(uploaded_file, skiprows=3, header=None)
            pulse = data[0].values
            I0 = data[1].values
            Fe_Ka = data[2].values
            Fe_Ka_norm = Fe_Ka / I0
            energy = pulse_to_energy(pulse, pulse_ref)
            Fe_Ka_smooth = gaussian_filter1d(Fe_Ka_norm, sigma=1)

            # 昇順ソート
            sort_idx = np.argsort(energy)
            energy = energy[sort_idx]
            Fe_Ka_norm = Fe_Ka_norm[sort_idx]
            Fe_Ka_smooth = Fe_Ka_smooth[sort_idx]

            # -----------------------------
            # 自動直線ベースライン
            # -----------------------------
            mask_low = energy <= 7109
            mask_high = energy >= 7114
            E_low = energy[mask_low][np.argmax(Fe_Ka_smooth[mask_low])]
            I_low = Fe_Ka_smooth[mask_low][np.argmax(Fe_Ka_smooth[mask_low])]
            E_high = energy[mask_high][np.argmin(Fe_Ka_smooth[mask_high])]
            I_high = Fe_Ka_smooth[mask_high][np.argmin(Fe_Ka_smooth[mask_high])]
            m_lin = (I_high - I_low) / (E_high - E_low)
            c_lin = I_low - m_lin * E_low
            baseline = m_lin * energy + c_lin

            # -----------------------------
            # プレエッジ2ガウスフィット（7110-7115 eV）
            # -----------------------------
            mask_gauss = (energy >= 7110) & (energy <= 7115)
            E_gauss = energy[mask_gauss]
            I_gauss = Fe_Ka_smooth[mask_gauss] - baseline[mask_gauss]

            p0_gauss = [0.1, 7111.8, 0.5, 0.1, 7113.7, 0.5]
            lower_gauss = [0, 7110, 0, 0, 7112, 0]
            upper_gauss = [np.inf, 7112, 2, np.inf, 7115, 2]

            popt_gauss, _ = curve_fit(two_gauss, E_gauss, I_gauss,
                                      p0=p0_gauss, bounds=(lower_gauss, upper_gauss),
                                      maxfev=5000)
            gauss_fit = two_gauss(E_gauss, *popt_gauss)

            area1 = popt_gauss[0] * popt_gauss[2] * np.sqrt(2*np.pi)
            area2 = popt_gauss[3] * popt_gauss[5] * np.sqrt(2*np.pi)
            centroid = (popt_gauss[1]*area1 + popt_gauss[4]*area2) / (area1 + area2)

            # -----------------------------
            # テキスト出力
            # -----------------------------
            txt_lines = []
            txt_lines.append("Linear baseline parameters (auto max/min):")
            txt_lines.append(f"slope = {m_lin:.6f}, intercept = {c_lin:.6f}")
            txt_lines.append("\nGaussian peaks:")
            txt_lines.append(f"Peak1: mu={popt_gauss[1]:.3f}, sigma={popt_gauss[2]:.3f}, A={popt_gauss[0]:.3f}")
            txt_lines.append(f"Peak2: mu={popt_gauss[4]:.3f}, sigma={popt_gauss[5]:.3f}, A={popt_gauss[3]:.3f}")
            txt_lines.append(f"Centroid (area-weighted) = {centroid:.3f} eV")
            txt_content = "\n".join(txt_lines)

            txt_filename = f"{basename}_fitting.txt"
            zipf.writestr(txt_filename, txt_content)

            # -----------------------------
            # Plotlyでインタラクティブ表示
            # -----------------------------
            gauss1 = gaussian(E_gauss, popt_gauss[0], popt_gauss[1], popt_gauss[2])
            gauss2 = gaussian(E_gauss, popt_gauss[3], popt_gauss[4], popt_gauss[5])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=energy, y=Fe_Ka_norm, mode='markers', name='Raw'))
            fig.add_trace(go.Scatter(x=energy, y=Fe_Ka_smooth, mode='lines', name='Smoothed'))
            fig.add_trace(go.Scatter(x=energy, y=baseline, mode='lines', name='Baseline', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=E_gauss, y=gauss1 + baseline[mask_gauss], mode='lines', name='Gaussian 1'))
            fig.add_trace(go.Scatter(x=E_gauss, y=gauss2 + baseline[mask_gauss], mode='lines', name='Gaussian 2'))
            fig.add_trace(go.Scatter(x=E_gauss, y=gauss_fit + baseline[mask_gauss], mode='lines', name='Total fit'))
            fig.add_trace(go.Scatter(x=[centroid, centroid], y=[0, max(Fe_Ka_smooth)], mode='lines',
                                     name='Centroid', line=dict(dash='dot')))
            fig.update_layout(
                title=f"{basename} Fitting",
                xaxis_title="Energy (eV)",
                yaxis_title="Normalized intensity",
                xaxis=dict(range=[7108, 7116]),
                yaxis=dict(range=[0, np.ceil(max(Fe_Ka_smooth)/0.01)*0.01])
            )

            st.plotly_chart(fig, use_container_width=True)

            # -----------------------------
            # HTMLダウンロード
            # -----------------------------
            html_buffer = io.StringIO()
            fig.write_html(html_buffer)
            html_buffer.seek(0)
            st.download_button(f"Download {basename} graph (HTML)", html_buffer,
                               file_name=f"{basename}_fitting.html")

    # -----------------------------
    # ZIPダウンロード（テキストまとめ）
    # -----------------------------
    zip_buffer.seek(0)
    st.download_button("Download all results (txt files)", data=zip_buffer, file_name="results.zip")
