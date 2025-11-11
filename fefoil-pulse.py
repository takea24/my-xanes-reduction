# fefoil-pulse.py (Streamlit完全版)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import plotly.graph_objects as go

# ---------- user-configurable ----------
SKIP_HEADER = 3
SG_WINDOW = 31
SG_POLY = 5
SEARCH_MIN = 581650
OUT_DECIMALS = 1
SLIDER_RANGE = 1000   # ±範囲
# -------------------------------------

def load_xanes_file(file, skip_header=SKIP_HEADER):
    pulse_list = []
    mu_list = []
    lines = file.read().decode('utf-8', errors='ignore').splitlines()
    lines = lines[skip_header:]

    for line in lines:
        s = line.strip()
        if not s: continue
        if s.startswith('#'): continue
        if ',' in s:
            parts = [p.strip() for p in s.split(',') if p.strip()!='']
        else:
            parts = s.split()
        if len(parts) < 3:
            continue
        try:
            p = float(parts[0])
            I0 = float(parts[1])
            FeKa = float(parts[2])
        except:
            continue
        if I0 == 0:
            continue
        pulse_list.append(p)
        mu_list.append(FeKa / I0)

    if len(pulse_list) == 0:
        raise RuntimeError("No valid numeric data found.")
    return np.array(pulse_list), np.array(mu_list)

def compute_smoothed_d2(pulse, mu, window=SG_WINDOW, poly=SG_POLY):
    if window >= len(mu):
        lw = len(mu) - 1 if (len(mu)%2==0) else len(mu)
        window = max(5, lw)
        if window % 2 == 0: window -= 1
    mu_s = savgol_filter(mu, window_length=window, polyorder=poly, mode='interp')
    d1 = np.gradient(mu_s, pulse)
    d2 = np.gradient(d1, pulse)
    return mu_s, d2

def find_zero_crossing(p, d2):
    mask = p >= SEARCH_MIN
    idxs = np.where(mask)[0]
    if len(idxs) < 2:
        return None, None
    for i0 in range(idxs[0], len(p) - 1):
        y1 = d2[i0]
        y2 = d2[i0+1]
        if y1 == 0:
            return p[i0], i0
        if y1 * y2 < 0:
            x1 = p[i0]
            x2 = p[i0+1]
            x0 = x1 - y1 * (x2 - x1) / (y2 - y1)
            return x0, i0
    return None, None

def save_result_txt(chosen_pulse, filename, decimals=OUT_DECIMALS):
    base = os.path.splitext(filename)[0]
    fname = f"{base}_{chosen_pulse:.{decimals}f}.txt"
    with open(fname,'w') as f:
        f.write(f"file: {filename}\n")
        f.write(f"pulse: {chosen_pulse:.{decimals}f}\n")
    return fname

def save_static_png(pulse, mu, mu_s, d2, chosen_pulse, filename):
    mask = pulse >= SEARCH_MIN
    p_plot = pulse[mask]
    mu_plot = mu[mask]
    mu_s_plot = mu_s[mask]
    d2_plot = d2[mask]

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax2 = ax1.twinx()
    ax1.plot(p_plot, mu_plot, label="mu", color='black')
    ax1.plot(p_plot, mu_s_plot, label="mu_smooth", color='gray')
    ax2.plot(p_plot, d2_plot, label="d2", linestyle='--')
    ax2.axhline(0, color='black', linewidth=1)
    ax1.axvline(chosen_pulse, color='red', linestyle='--', label=f'selected={chosen_pulse:.1f}')
    ax1.set_xlabel("Pulse")
    ax1.set_ylabel("mu")
    ax2.set_ylabel("d2")
    ax1.legend(loc='best')
    plt.tight_layout()
    base = os.path.splitext(filename)[0]
    png_file = f"{base}_{chosen_pulse:.1f}.png"
    fig.savefig(png_file, dpi=300)
    plt.close(fig)
    return png_file

# ---------- Streamlit UI ----------
st.title("Fe Foil E0 Pulse Determination (XANES)")

uploaded_file = st.file_uploader("Select Fe foil .dat file", type=['dat','txt'])

if uploaded_file is not None:
    try:
        pulse, mu = load_xanes_file(uploaded_file)
        mu_s, d2 = compute_smoothed_d2(pulse, mu)
        guess_pulse, idx = find_zero_crossing(pulse, d2)

        st.subheader("Pulse Data & Smoothed 2nd Derivative")

        # スライダー
        slider_min = int(guess_pulse - SLIDER_RANGE)
        slider_max = int(guess_pulse + SLIDER_RANGE)
        selected_pulse = st.slider(
            "Adjust pulse manually",
            min_value=slider_min,
            max_value=slider_max,
            value=int(guess_pulse),
            step=1
        )

        # Plotlyでインタラクティブグラフ
        mask = pulse >= SEARCH_MIN
        p_plot = pulse[mask]
        mu_plot = mu[mask]
        mu_s_plot = mu_s[mask]
        d2_plot = d2[mask]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_plot, y=mu_plot, mode='lines+markers', name='mu'))
        fig.add_trace(go.Scatter(x=p_plot, y=mu_s_plot, mode='lines', name='mu_smooth'))
        fig.add_trace(go.Scatter(x=p_plot, y=d2_plot, mode='lines', name='d2', yaxis='y2'))

        # スライダー追従縦線
        fig.add_vline(x=selected_pulse, line=dict(color='red', dash='dash'))

        fig.update_layout(
            yaxis=dict(title='mu'),
            yaxis2=dict(title='d2', overlaying='y', side='right')
        )

        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Selected pulse: {selected_pulse}")

        # 結果保存
        out_file = save_result_txt(selected_pulse, uploaded_file.name)
        st.download_button("Download result txt", data=open(out_file,'r').read(), file_name=out_file)

        # PNG出力
        png_file = save_static_png(pulse, mu, mu_s, d2, selected_pulse, uploaded_file.name)
        with open(png_file,'rb') as f:
            st.download_button("Download PNG graph", data=f, file_name=png_file)

    except Exception as e:
        st.error(f"Error: {e}")
