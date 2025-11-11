# app_interactive.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import os

# ---------- user-configurable ----------
SKIP_HEADER = 3
SG_WINDOW = 31
SG_POLY = 5
SEARCH_MIN = 581650
OUT_DECIMALS = 1
# -------------------------------------

def load_xanes_file(file, skip_header=SKIP_HEADER):
    pulse_list = []
    mu_list = []
    lines = file.read().decode('utf-8', errors='ignore').splitlines()
    lines = lines[skip_header:]

    for line in lines:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if ',' in s:
            parts = [p.strip() for p in s.split(',') if p.strip() != '']
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
        lw = len(mu) - 1 if (len(mu) % 2 == 0) else len(mu)
        window = max(5, lw)
        if window % 2 == 0:
            window -= 1
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
        y2 = d2[i0 + 1]
        if y1 == 0:
            return p[i0], i0
        if y1 * y2 < 0:
            x1 = p[i0]
            x2 = p[i0 + 1]
            x0 = x1 - y1 * (x2 - x1) / (y2 - y1)
            return x0, i0
    return None, None

def save_result_txt(chosen_pulse, filename, decimals=OUT_DECIMALS):
    base = os.path.splitext(filename)[0]
    fname = f"{base}_{chosen_pulse:.{decimals}f}.txt"
    with open(fname, 'w') as f:
        f.write(f"file: {filename}\n")
        f.write(f"pulse: {chosen_pulse:.{decimals}f}\n")
    return fname

# ---------- Streamlit UI ----------
st.title("Fe Foil E0 Pulse Determination (XANES) - Interactive")

uploaded_file = st.file_uploader("Select Fe foil .dat file", type=['dat', 'txt'])
if uploaded_file is not None:
    try:
        pulse, mu = load_xanes_file(uploaded_file)
        mu_s, d2 = compute_smoothed_d2(pulse, mu)
        guess_pulse, idx = find_zero_crossing(pulse, d2)

        # Plotly グラフ表示
        mask = pulse >= SEARCH_MIN
        p_plot = pulse[mask]
        mu_plot = mu[mask]
        mu_s_plot = mu_s[mask]
        d2_plot = d2[mask]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_plot, y=mu_plot, mode='lines+markers', name='mu (FeKa/I0)'))
        fig.add_trace(go.Scatter(x=p_plot, y=mu_s_plot, mode='lines', name='mu_smooth'))
        fig.add_trace(go.Scatter(x=p_plot, y=d2_plot, mode='lines', name='d2', yaxis='y2'))
        if guess_pulse is not None:
            fig.add_vline(x=guess_pulse, line=dict(color='red', dash='dash'), annotation_text=f'guess={guess_pulse:.1f}')

        # 右軸追加
        fig.update_layout(
            yaxis=dict(title='mu'),
            yaxis2=dict(title='d2', overlaying='y', side='right'),
            title='Pulse Data & 2nd Derivative',
            width=800, height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # ユーザー選択
        chosen = guess_pulse
        if guess_pulse is not None:
            accept = st.radio(f"Use guessed pulse = {guess_pulse:.1f} ?", ["Yes", "No"])
            if accept == "No":
                st.info("Click on the graph to select pulse manually")
                selected_points = st.plotly_events(fig, click_event=True, hover_event=False)
                if selected_points:
                    chosen = selected_points[0]['x']

        st.success(f"Selected pulse: {chosen:.1f}")
        out_file = save_result_txt(chosen, uploaded_file.name)
        st.download_button("Download result txt", data=open(out_file, 'r').read(), file_name=out_file)

    except Exception as e:
        st.error(f"Error: {e}")
