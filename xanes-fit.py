# xanes-fit-with-pulseref.py (完全版改良)
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
st.title("XANES Multiple File Fitting with Pulse Reference")
st.subheader("Step 1: Pulse Reference Selection (Fe-foil)")

method = st.radio("Choose pulse reference method:", ["Input manually", "Analyze Fe-foil file"])
pulse_ref = None

search_min = st.number_input("Fe-foil analysis: minimum pulse to analyze (SEARCH_MIN)", value=DEFAULT_SEARCH_MIN, step=1)

if method=="Input manually":
    pulse_ref = st.number_input("Enter pulse reference", value=36000.0, step=1.0)
    if st.button("Confirm pulse reference"):
        st.success(f"Confirmed pulse reference: {pulse_ref}")

elif method=="Analyze Fe-foil file":
    uploaded_file = st.file_uploader("Select Fe foil .dat file", type=['dat','txt'])
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

        # 第二軸 y=0 水平線
        fig.add_shape(
            type="line",
            x0=pulse[mask].min(),
            x1=pulse[mask].max(),
            y0=0,
            y1=0,
            yref="y2",
            line=dict(color="blue", dash="dot")
        )

        # パルスリファレンス縦線
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
            st.success(f"Confirmed pulse reference: {pulse_ref}")

# -----------------------------
# Step 2: XANES Fitting
# -----------------------------
if pulse_ref is not None:
    st.subheader("Step 2: Multiple File Fitting")
    uploaded_files = st.file_uploader("Select dat files for fitting", accept_multiple_files=True, type=['dat','txt'])
    if uploaded_files:
        st.write(f"{len(uploaded_files)} files selected.")

        # Step2のみベースライン入力フォーム
        st.subheader("Baseline selection (manual input)")
        bg_low = st.number_input("Lower background limit (eV, ≤)", value=7110.0, step=0.01)
        bg_high = st.number_input("Upper background limit (eV, ≥)", value=7114.0, step=0.01)

        png_buffers = []

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

                # Gauss fitting
                mask_gauss=(energy>=7110)&(energy<=7115)
                E_gauss=energy[mask_gauss]
                I_gauss=FeKa_smooth[mask_gauss]-baseline[mask_gauss]
                p0_gauss=[0.1,7111.8,0.5,0.1,7113.7,0.5]
                lower=[0,7110,0,0,7112,0]
                upper=[np.inf,7112,2,np.inf,7115,2]
                popt,_=curve_fit(two_gauss,E_gauss,I_gauss,p0=p0_gauss,bounds=(lower,upper),maxfev=5000)
                gauss_fit=two_gauss(E_gauss,*popt)

                area1 = popt[0]*popt[2]*np.sqrt(2*np.pi)
                area2 = popt[3]*popt[5]*np.sqrt(2*np.pi)
                centroid = (popt[1]*area1 + popt[4]*area2)/(area1+area2)

                # Matplotlib
                fig_mpl, ax = plt.subplots(figsize=(10,6), constrained_layout=True)
                ax.plot(energy, FeKa_norm, 'ko', alpha=0.5, label='raw')
                ax.plot(energy, FeKa_smooth, 'k-', alpha=0.8, label='smoothed')
                ax.plot(energy, baseline, 'r--', linewidth=2, label='baseline')
                g1 = gaussian(E_gauss,popt[0],popt[1],popt[2])
                g2 = gaussian(E_gauss,popt[3],popt[4],popt[5])
                ax.plot(E_gauss,g1+baseline[mask_gauss],'g--',linewidth=2,label='Gaussian1')
                ax.plot(E_gauss,g2+baseline[mask_gauss],'m--',linewidth=2,label='Gaussian2')
                ax.plot(E_gauss,gauss_fit+baseline[mask_gauss],'b-',linewidth=2,label='Total fit')
                ax.axvline(centroid,color='blue',linestyle=':',label=f'Centroid={centroid:.2f}')
                mask_ylim=(energy>=7114)&(energy<=7116)
                ylim_max=np.ceil(FeKa_smooth[mask_ylim].max()/0.01)*0.01
                ax.set_xlim(7108,7116)
                ax.set_ylim(0,ylim_max)
                ax.set_xlabel("Energy (eV)")
                ax.set_ylabel("Normalized intensity")
                ax.legend()

                png_buffer=io.BytesIO()
                fig_mpl.savefig(png_buffer,dpi=300)
                png_buffer.seek(0)
                png_buffers.append((uploaded_file.name,png_buffer))
                plt.close(fig_mpl)
                st.download_button(f"Download {uploaded_file.name} PNG", png_buffer, file_name=f"{uploaded_file.name}_fitting.png")

                # Plotly
                fig_plotly=go.Figure()
                fig_plotly.add_trace(go.Scatter(x=energy,y=FeKa_norm,mode='markers',name='raw',marker=dict(color='black',opacity=0.5)))
                fig_plotly.add_trace(go.Scatter(x=energy,y=FeKa_smooth,mode='lines',name='smoothed',line=dict(color='gray')))
                fig_plotly.add_trace(go.Scatter(x=energy,y=baseline,mode='lines',name='baseline',line=dict(color='red',dash='dash')))
                # Gaussians
                fig_plotly.add_trace(go.Scatter(x=E_gauss,y=g1+baseline[mask_gauss],mode='lines',name='Gaussian1',line=dict(color='green',dash='dash')))
                fig_plotly.add_trace(go.Scatter(x=E_gauss,y=g2+baseline[mask_gauss],mode='lines',name='Gaussian2',line=dict(color='magenta',dash='dash')))
                fig_plotly.add_trace(go.Scatter(x=E_gauss,y=gauss_fit+baseline[mask_gauss],mode='lines',name='Total fit',line=dict(color='blue')))
                # パルスリファレンス表示など不要
                fig_plotly.add_vline(x=centroid,line=dict(color='blue',dash='dot'),annotation_text=f"Centroid={centroid:.2f}",annotation_position="top right")
                fig_plotly.update_layout(
                    xaxis=dict(range=[7108,7116]),
                    yaxis=dict(range=[0,ylim_max]),
                    title=uploaded_file.name,
                    xaxis_title="Energy (eV)",
                    yaxis_title="Normalized intensity"
                )
                st.plotly_chart(fig_plotly,use_container_width=True)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")

        # ZIP一括ダウンロード
        if png_buffers:
            zip_buffer=io.BytesIO()
            with zipfile.ZipFile(zip_buffer,"w") as zf:
                for name, buf in png_buffers:
                    zf.writestr(f"{name}_fitting.png", buf.getvalue())
            zip_buffer.seek(0)
            st.download_button("Download all PNGs as ZIP", zip_buffer, file_name="all_fittings.zip")
