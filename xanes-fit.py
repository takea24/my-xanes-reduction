# xanes-fit-with-pulseref.py (描画範囲拡大版)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import io
import zipfile
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
def pulse_to_energy(pulse, pulse_ref):
    theta0 = np.arcsin(HC / (2.0 * D_SI111 * E0_FE))
    dtheta = (pulse - pulse_ref)/PULSES_PER_DEG * DEG2RAD
    theta = theta0 + dtheta
    E = HC / (2.0 * D_SI111 * np.sin(theta))
    return E

# -----------------------------
# ガウス関数
# -----------------------------
def gaussian(E, A, mu, sigma):
    return np.abs(A) * np.exp(-(E-mu)**2/(2*sigma**2))

def two_gauss(E, A1, mu1, sigma1, A2, mu2, sigma2):
    return gaussian(E,A1,mu1,sigma1) + gaussian(E,A2,mu2,sigma2)

# -----------------------------
# Fe-foil解析関数
# -----------------------------
SKIP_HEADER = 3
SG_WINDOW = 31
SG_POLY = 5
DEFAULT_SEARCH_MIN = 581650

def load_xanes_file(file):
    pulse_list=[]
    mu_list=[]
    lines=file.read().decode('utf-8', errors='ignore').splitlines()
    lines = lines[SKIP_HEADER:]
    for line in lines:
        s=line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split(',') if ',' in s else s.split()
        if len(parts)<3: continue
        try:
            p=float(parts[0])
            I0=float(parts[1])
            FeKa=float(parts[2])
        except:
            continue
        if I0==0: continue
        pulse_list.append(p)
        mu_list.append(FeKa/I0)
    if len(pulse_list)==0:
        raise RuntimeError("No valid numeric data found.")
    return np.array(pulse_list), np.array(mu_list)

def compute_smoothed_d2(pulse, mu):
    if SG_WINDOW >= len(mu):
        lw = len(mu)-1 if len(mu)%2==0 else len(mu)
        window = max(5,lw)
        if window%2==0: window-=1
    else:
        window=SG_WINDOW
    mu_s = savgol_filter(mu, window_length=window, polyorder=SG_POLY, mode='interp')
    d1 = np.gradient(mu_s, pulse)
    d2 = np.gradient(d1, pulse)
    return mu_s, d2

def find_zero_crossing(p, d2, search_min):
    mask = p >= search_min
    idxs = np.where(mask)[0]
    if len(idxs)<2: return None
    for i0 in range(idxs[0], len(p)-1):
        y1=d2[i0]
        y2=d2[i0+1]
        if y1==0 or y1*y2<0:
            x0 = p[i0] - y1*(p[i0+1]-p[i0])/(y2-y1)
            return x0
    return None

# -----------------------------
# Step 1: Fe-foil解析
# -----------------------------
st.title("XANES Multiple File Fitting with Pulse Reference ver1.0")

if "step1_done" not in st.session_state:
    st.session_state.step1_done = False

st.subheader("Step 1: Pulse Reference Selection (Fe-foil)")

method = st.radio("Choose pulse reference method:", ["Input manually", "Analyze Fe-foil file"])
pulse_ref = None

if method=="Input manually":
    pulse_ref = st.number_input("鉄の第一変曲点のパルス位置", value=581700.0, step=1.0)
    if st.button("Confirm pulse reference"):
        st.session_state.step1_done = True
        st.success(f"Confirmed pulse reference: {pulse_ref}")

elif method=="Analyze Fe-foil file":
    uploaded_file = st.file_uploader("Select Fe foil .dat file", type=['dat','txt'])
    search_min = st.number_input("第一変曲点探索範囲指定（これより大きい値で変曲点を再探索）", value=DEFAULT_SEARCH_MIN, step=1)

    if uploaded_file is not None:
        pulse, mu = load_xanes_file(uploaded_file)
        mu_s, d2 = compute_smoothed_d2(pulse, mu)
        guess = find_zero_crossing(pulse, d2, search_min)
        initial_pulse = int(guess) if guess else int(pulse.min())

        col1, col2 = st.columns([3,1])
        with col1:
            chosen_slider = st.slider("Adjust pulse", min_value=int(pulse.min()), max_value=int(pulse.max()), value=initial_pulse, step=1)
        with col2:
            chosen_input = st.number_input("Or enter manually", min_value=int(pulse.min()), max_value=int(pulse.max()), value=chosen_slider, step=1)

        pulse_ref = chosen_input

        # Fe-foil Plotly
        mask = pulse >= search_min
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pulse[mask], y=mu[mask], mode='lines+markers', name='mu', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=pulse[mask], y=mu_s[mask], mode='lines', name='smoothed', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=pulse[mask], y=d2[mask], mode='lines', name='d2', line=dict(dash='dash'), yaxis='y2'))

        fig.add_shape(type="line", x0=pulse[mask].min(), x1=pulse[mask].max(), y0=0, y1=0, yref="y2", line=dict(color="blue", dash="dot"))
        fig.add_vline(x=pulse_ref, line=dict(color='red', dash='dash'))

        fig.update_layout(
            xaxis_title="Pulse",
            yaxis=dict(title="mu"),
            yaxis2=dict(title="d2", overlaying='y', side='right'),
            xaxis=dict(tickformat="06d"),
            width=800, height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Confirm pulse reference"):
            st.session_state.step1_done = True
            st.success(f"Confirmed pulse reference: {pulse_ref}")

# -----------------------------
# Step 2: XANES Fitting
# -----------------------------
if st.session_state.step1_done:
    st.subheader("Step 2: Multiple File Fitting")
    st.write("※誤差はfittingパラメータから得られる最低限の誤差で測定誤差を含まない")

    uploaded_files = st.file_uploader("Select dat files for fitting", accept_multiple_files=True, type=['dat','txt'])
    if uploaded_files:
        st.write(f"{len(uploaded_files)} files selected.")
        st.subheader("Baseline selection (manual input)")
        bg_low = st.number_input("Lower background limit (eV, ≤)", value=7110.0, step=0.01)
        bg_high = st.number_input("Upper background limit (eV, ≥)", value=7114.0, step=0.01)

        png_buffers = []
        all_params = []

        st.subheader("Gaussian fitting")

        for uploaded_file in uploaded_files:
            try:
                data=pd.read_csv(uploaded_file, skiprows=3, header=None)
                pulse = data[0].values
                I0 = data[1].values
                FeKa = data[2].values
                FeKa_norm = FeKa/I0
                energy = pulse_to_energy(pulse, pulse_ref)
                FeKa_smooth = gaussian_filter1d(FeKa_norm, sigma=1)
                sort_idx=np.argsort(energy)
                energy=energy[sort_idx]
                FeKa_norm=FeKa_norm[sort_idx]
                FeKa_smooth=FeKa_smooth[sort_idx]

                # Baseline
                mask_low = energy <= bg_low
                mask_high=energy >= bg_high
                E_low = energy[mask_low][np.argmax(FeKa_smooth[mask_low])]
                I_low = FeKa_smooth[mask_low][np.argmax(FeKa_smooth[mask_low])]
                E_high = energy[mask_high][np.argmin(FeKa_smooth[mask_high])]
                I_high = FeKa_smooth[mask_high][np.argmin(FeKa_smooth[mask_high])]
                m_lin=(I_high-I_low)/(E_high-E_low)
                c_lin=I_low - m_lin*E_low
                baseline = m_lin*energy+c_lin

                # --- ここからガウシアン範囲を個別指定 ---
                st.markdown(f"> File: {uploaded_file.name}")
                st.write("- ガウシアンのfitting範囲を指定（option）")
                col1, col2 = st.columns(2)
                with col1:
                    gauss_min = st.number_input(f"{uploaded_file.name} Gaussian fit min (eV)", value=7110.0, step=0.01, key=f"gauss_min_{uploaded_file.name}")
                with col2:
                    gauss_max = st.number_input(f"{uploaded_file.name} Gaussian fit max (eV)", value=7115.0, step=0.01, key=f"gauss_max_{uploaded_file.name}")

                mask_gauss = (energy >= gauss_min) & (energy <= gauss_max)
                E_gauss = energy[mask_gauss]
                I_gauss = FeKa_smooth[mask_gauss] - baseline[mask_gauss]
                # 初期値・範囲はデフォルトのまま
                p0_gauss=[0.001,7111.8,0.5,0.001,7113.7,0.5]
                lower=[0,7110,0,0,7112,0]
                upper=[np.inf,7112,1,np.inf,7114,1]
                popt, pcov = curve_fit(two_gauss, E_gauss, I_gauss,
                       p0=p0_gauss, bounds=(lower, upper), maxfev=5000)

                A1, mu1, sigma1, A2, mu2, sigma2 = popt

                if pcov is not None:
                    perr = np.sqrt(np.diag(pcov))
                else:
                    perr = np.zeros_like(popt)
                # パラメータ誤差
                perr = np.sqrt(np.diag(pcov))  # [A1_err, mu1_err, sigma1_err, A2_err, mu2_err, sigma2_err]

                # 描画用マスクは固定で7108-7116
                mask_plot = (energy>=7108)&(energy<=7116)
                E_plot = energy[mask_plot]
                g1_plot = gaussian(E_plot,popt[0],popt[1],popt[2])
                g2_plot = gaussian(E_plot,popt[3],popt[4],popt[5])
                gauss_fit_plot = g1_plot + g2_plot

                # Matplotlib
                fig_mpl, ax = plt.subplots(figsize=(10,6), constrained_layout=True)
                ax.plot(energy, FeKa_norm, 'ko', alpha=0.5, label='raw')
                ax.plot(energy, FeKa_smooth, 'k-', alpha=0.8, label='smoothed')
                ax.plot(energy, baseline, 'r--', linewidth=2, label='baseline')
                ax.plot(E_plot, g1_plot + baseline[mask_plot], 'g--', linewidth=3, label='Gaussian1')
                ax.plot(E_plot, g2_plot + baseline[mask_plot], 'm--', linewidth=3, label='Gaussian2')
                ax.plot(E_plot, gauss_fit_plot + baseline[mask_plot], 'b-', linewidth=1, label='Total fit')

                # Centroid
                area1 = popt[0]*popt[2]*np.sqrt(2*np.pi)
                area2 = popt[3]*popt[5]*np.sqrt(2*np.pi)
                centroid = (popt[1]*area1 + popt[4]*area2)/(area1+area2)
                ax.axvline(centroid,color='blue',linestyle=':',label=f'Centroid={centroid:.2f}')

                # Centroid 誤差伝播（簡易）
                if pcov is not None:
                    dA1, dmu1, dA2, dmu2 = perr[0], perr[1], perr[3], perr[4]
                    numerator = A1*mu1 + A2*mu2
                    denominator = A1 + A2
                    d_centroid = np.sqrt(
                        ((mu1*denominator - numerator)/(denominator**2) * dA1)**2 +
                        (A1/denominator * dmu1)**2 +
                        ((mu2*denominator - numerator)/(denominator**2) * dA2)**2 +
                        (A2/denominator * dmu2)**2
                    )
                else:
                    d_centroid = np.nan

                # 軸設定
                ax.set_xlim(7108,7116)
                mask_ylim=(energy>=7112)&(energy<=7116)
                ylim_default = np.ceil(FeKa_smooth[mask_ylim].max()/0.01)*0.01
                st.write("- 描画範囲(y軸)を指定（option）")
                ylim_max_input = st.number_input(
                    f"{uploaded_file.name} Y-axis max", 
                    value=ylim_default, step=0.01, key=f"ymax_{uploaded_file.name}"
                )

                ax.set_ylim(0, ylim_max_input)
                ax.set_xlabel("Energy (eV)")
                ax.set_ylabel("Normalized intensity")
                ax.legend()

                png_buffer=io.BytesIO()
                fig_mpl.savefig(png_buffer,dpi=300)
                png_buffer.seek(0)
                png_buffers.append((uploaded_file.name,png_buffer))
                plt.close(fig_mpl)

                # Plotly描画（省略せず元コードと同じ）
                fig_plotly=go.Figure()
                fig_plotly.add_trace(go.Scatter(x=energy,y=FeKa_norm,mode='markers',name='raw',marker=dict(color='black',opacity=0.5)))
                fig_plotly.add_trace(go.Scatter(x=energy,y=FeKa_smooth,mode='lines',name='smoothed',line=dict(color='gray')))
                fig_plotly.add_trace(go.Scatter(x=energy,y=baseline,mode='lines',name='baseline',line=dict(color='red',dash='dash')))
                fig_plotly.add_trace(go.Scatter(x=E_plot,y=g1_plot+baseline[mask_plot],mode='lines',name='Gaussian1',line=dict(color='green',dash='dash', width=3)))
                fig_plotly.add_trace(go.Scatter(x=E_plot,y=g2_plot+baseline[mask_plot],mode='lines',name='Gaussian2',line=dict(color='magenta',dash='dash', width=3)))
                fig_plotly.add_trace(go.Scatter(x=E_plot,y=gauss_fit_plot+baseline[mask_plot],mode='lines',name='Total fit',line=dict(color='blue', width=1)))
                fig_plotly.add_vline(x=centroid,line=dict(color='blue',dash='dot'),annotation_text=f"Centroid={centroid:.2f}",annotation_position="top right")
                fig_plotly.update_layout(
                    xaxis=dict(range=[7108,7116]),
                    yaxis=dict(range=[0, ylim_max_input]),
                    title=uploaded_file.name,
                    xaxis_title="Energy (eV)",
                    yaxis_title="Normalized intensity"
                )
                st.plotly_chart(fig_plotly,use_container_width=True)

                st.download_button(f"グラフ個別Download {uploaded_file.name} PNG", png_buffer, file_name=f"{uploaded_file.name}_fitting.png")

                # パラメータ計算
                FWHM1 = 2.35482 * sigma1
                FWHM2 = 2.35482 * sigma2
                df = pd.DataFrame(
                    {
                        "Gaussian 1": [A1, mu1, sigma1, FWHM1, area1],
                        "Gaussian 2": [A2, mu2, sigma2, FWHM2, area2],
                    },
                    index=["Height A", "Center μ (eV)", "σ", "FWHM", "Area"]
                )
                st.dataframe(df.style.format("{:.5g}"))
                st.write(f"**Centroid peak position** = {centroid:.2f} ± {d_centroid:.2f} eV")                      

                all_params.append({
                    "File": uploaded_file.name,
                    "Centroid": centroid,
                    "Error": d_centroid,
                    "Gaussian1_A": A1,
                    "Gaussian1_mu": mu1,
                    "Gaussian1_sigma": sigma1,
                    "Gaussian1_FWHM": FWHM1,
                    "Gaussian2_A": A2,
                    "Gaussian2_mu": mu2,
                    "Gaussian2_sigma": sigma2,
                    "Gaussian2_FWHM": FWHM2
                })

                # 区切り線を挿入
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

        # Summary & ZIPは元コードと同じ

        # Summary
        if all_params:
            df_all = pd.DataFrame(all_params)
            st.subheader("Summary of all fittings")
            st.dataframe(df_all)
            csv_buffer = io.StringIO()
            df_all.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download all fitting parameters (CSV/TXT)",
                data=csv_buffer.getvalue(),
                file_name="all_fitting_parameters.csv",
                mime="text/csv"
            )

        # ZIP一括
        if png_buffers:
            zip_buffer=io.BytesIO()
            with zipfile.ZipFile(zip_buffer,"w") as zf:
                for name, buf in png_buffers:
                    zf.writestr(f"{name}_fitting.png", buf.getvalue())
            zip_buffer.seek(0)
            st.download_button("Download all PNGgraphs as ZIP", zip_buffer, file_name="all_fittings.zip")
