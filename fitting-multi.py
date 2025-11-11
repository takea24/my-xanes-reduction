import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import io
import zipfile
import os
import plotly.graph_objects as go

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
st.title("XANES Multiple File Fitting")

uploaded_files = st.file_uploader("解析するdatファイルを複数選択", accept_multiple_files=True, type=['dat','txt'])
pulse_ref_input = st.number_input("基準パルス (pulse reference)", value=36000.0, step=1.0)

if uploaded_files:
    st.write(f"{len(uploaded_files)} ファイルが選択されました。")
    
    # PNG保存用のメモリストリーム
    png_buffers = []
    
    for uploaded_file in uploaded_files:
        try:
            # ファイル読み込み
            data = pd.read_csv(uploaded_file, skiprows=3, header=None)
            pulse = data[0].values
            I0 = data[1].values
            Fe_Ka = data[2].values
            Fe_Ka_norm = Fe_Ka / I0
            energy = pulse_to_energy(pulse, pulse_ref_input)
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

            # プレエッジ2ガウスフィット
            mask_gauss = (energy >= 7110) & (energy <= 7115)
            E_gauss = energy[mask_gauss]
            I_gauss = Fe_Ka_smooth[mask_gauss] - baseline[mask_gauss]
            p0_gauss = [0.1, 7111.8, 0.5, 0.1, 7113.7, 0.5]
            lower_gauss = [0, 7110, 0, 0, 7112, 0]
            upper_gauss = [np.inf, 7112, 2, np.inf, 7115, 2]
            popt_gauss, _ = curve_fit(two_gauss, E_gauss, I_gauss, p0=p0_gauss,
                                      bounds=(lower_gauss, upper_gauss), maxfev=5000)
            gauss_fit = two_gauss(E_gauss, *popt_gauss)

            # 面積加重重心
            area1 = popt_gauss[0] * popt_gauss[2] * np.sqrt(2*np.pi)
            area2 = popt_gauss[3] * popt_gauss[5] * np.sqrt(2*np.pi)
            centroid = (popt_gauss[1]*area1 + popt_gauss[4]*area2) / (area1 + area2)

            # -----------------------------
            # テキスト出力
            # -----------------------------
            txt_lines = [
                "Linear baseline parameters (auto max/min):",
                f"slope = {m_lin:.6f}, intercept = {c_lin:.6f}",
                "",
                "Gaussian peaks:",
                f"Peak1: mu={popt_gauss[1]:.3f}, sigma={popt_gauss[2]:.3f}, A={popt_gauss[0]:.3f}",
                f"Peak2: mu={popt_gauss[4]:.3f}, sigma={popt_gauss[5]:.3f}, A={popt_gauss[3]:.3f}",
                f"Centroid (area-weighted) = {centroid:.3f} eV"
            ]
            st.text("\n".join(txt_lines))

            # -----------------------------
            # Matplotlib PNG生成
            # -----------------------------
            fig_mpl, ax = plt.subplots(figsize=(10,6), constrained_layout=True)
            ax.plot(energy, Fe_Ka_norm, 'ko', alpha=0.5, label='raw')
            ax.plot(energy, Fe_Ka_smooth, 'k-', alpha=0.8, label='smoothed')
            ax.plot(energy, baseline, 'r--', linewidth=2, label='baseline')
            gauss1 = gaussian(E_gauss, popt_gauss[0], popt_gauss[1], popt_gauss[2])
            gauss2 = gaussian(E_gauss, popt_gauss[3], popt_gauss[4], popt_gauss[5])
            ax.plot(E_gauss, gauss1+baseline[mask_gauss], 'g--', linewidth=2, label='Gaussian1')
            ax.plot(E_gauss, gauss2+baseline[mask_gauss], 'm--', linewidth=2, label='Gaussian2')
            ax.plot(E_gauss, gauss_fit+baseline[mask_gauss], 'b-', linewidth=2, label='Total fit')
            ax.axvline(centroid, color='blue', linestyle=':', label=f'Centroid={centroid:.2f}')
            
            # 縦軸固定
            mask_ylim = (energy >= 7114) & (energy <= 7116)
            ylim_max = np.ceil(Fe_Ka_smooth[mask_ylim].max() / 0.01) * 0.01
            ax.set_xlim(7108, 7116)
            ax.set_ylim(0, ylim_max)
            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("Normalized intensity")
            ax.legend()

            png_buffer = io.BytesIO()
            fig_mpl.savefig(png_buffer, dpi=300)
            png_buffer.seek(0)
            png_buffers.append((uploaded_file.name, png_buffer))
            plt.close(fig_mpl)

            # 個別PNGダウンロード
            st.download_button(f"Download {uploaded_file.name} PNG", png_buffer, file_name=f"{uploaded_file.name}_fitting.png")
            
            # -----------------------------
            # Plotlyインタラクティブ
            # -----------------------------
            fig_plotly = go.Figure()
            fig_plotly.add_trace(go.Scatter(x=energy, y=Fe_Ka_norm, mode='markers', name='raw', marker=dict(color='black', opacity=0.5)))
            fig_plotly.add_trace(go.Scatter(x=energy, y=Fe_Ka_smooth, mode='lines', name='smoothed', line=dict(color='gray')))
            fig_plotly.add_trace(go.Scatter(x=energy, y=baseline, mode='lines', name='baseline', line=dict(color='red', dash='dash')))
            fig_plotly.add_trace(go.Scatter(x=E_gauss, y=gauss_fit+baseline[mask_gauss], mode='lines', name='Total fit', line=dict(color='blue')))
            fig_plotly.add_vline(x=centroid, line=dict(color='blue', dash='dot'), annotation_text=f"Centroid={centroid:.2f}", annotation_position="top right")
            
            # 縦軸固定
            fig_plotly.update_layout(
                xaxis=dict(range=[7108, 7116]),
                yaxis=dict(range=[0, ylim_max]),
                title=uploaded_file.name,
                xaxis_title="Energy (eV)",
                yaxis_title="Normalized intensity"
            )
            st.plotly_chart(fig_plotly, use_container_width=True, key=f"fig_{uploaded_file.name}")

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    # -----------------------------
    # 一括PNGダウンロード（ワンクリック）
    # -----------------------------
    if png_buffers:  # ファイルがある場合のみ
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for name, buf in png_buffers:
                zf.writestr(f"{name}_fitting.png", buf.getvalue())
        zip_buffer.seek(0)
        st.download_button("Download all PNGs as ZIP", zip_buffer, file_name="all_fittings.zip")

