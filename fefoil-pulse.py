# fefoil-pulse-plotly-final.py
import streamlit as st
import numpy as np
from scipy.signal import savgol_filter
import plotly.graph_objects as go

# ---------- 設定 ----------
SKIP_HEADER = 3
SG_WINDOW = 31
SG_POLY = 5
SEARCH_MIN = 581650

# ---------- データ読み込み ----------
def load_xanes_file(file):
    pulse_list = []
    mu_list = []
    lines = file.read().decode('utf-8', errors='ignore').splitlines()
    lines = lines[SKIP_HEADER:]

    for line in lines:
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split(',') if ',' in s else s.split()
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

def compute_smoothed_d2(pulse, mu):
    if SG_WINDOW >= len(mu):
        lw = len(mu) - 1 if len(mu) % 2 == 0 else len(mu)
        window = max(5, lw)
        if window % 2 == 0: window -= 1
    else:
        window = SG_WINDOW
    mu_s = savgol_filter(mu, window_length=window, polyorder=SG_POLY, mode='interp')
    d1 = np.gradient(mu_s, pulse)
    d2 = np.gradient(d1, pulse)
    return mu_s, d2

def find_zero_crossing(p, d2):
    mask = p >= SEARCH_MIN
    idxs = np.where(mask)[0]
    if len(idxs) < 2:
        return None
    for i0 in range(idxs[0], len(p) - 1):
        y1 = d2[i0]
        y2 = d2[i0 + 1]
        if y1 == 0 or y1 * y2 < 0:
            x0 = p[i0] - y1 * (p[i0+1] - p[i0]) / (y2 - y1)
            return x0
    return None

# ---------- Streamlit UI ----------
st.title("Fe Foil E0 Pulse Determination (XANES)")

uploaded_file = st.file_uploader("Select Fe foil .dat file", type=['dat','txt'])
if uploaded_file is not None:
    try:
        pulse, mu = load_xanes_file(uploaded_file)
        mu_s, d2 = compute_smoothed_d2(pulse, mu)
        guess_pulse = find_zero_crossing(pulse, d2)

        min_p, max_p = int(pulse.min()), int(pulse.max())
        initial_pulse = int(guess_pulse) if guess_pulse else min_p

        # --- UI: スライダーと数値入力 ---
        st.subheader("Select Pulse")
        col1, col2 = st.columns([3,1])
        with col1:
            chosen_slider = st.slider(
                "Adjust pulse",
                min_value=min_p,
                max_value=max_p,
                value=initial_pulse,
                step=1
            )
        with col2:
            chosen_input = st.number_input(
                "Or enter pulse manually",
                min_value=min_p,
                max_value=max_p,
                value=chosen_slider,
                step=1
            )

        # 入力値を優先
        chosen = chosen_input

        # --- Plotly グラフ ---
        mask = pulse >= SEARCH_MIN
        p_plot = pulse[mask]
        mu_plot = mu[mask]
        mu_s_plot = mu_s[mask]
        d2_plot = d2[mask]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_plot, y=mu_plot, mode='lines+markers',
                                 name='mu (FeKa/I0)', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=p_plot, y=mu_s_plot, mode='lines',
                                 name='smoothed', line=dict(color='gray')))
        fig.add_trace(go.Scatter(x=p_plot, y=d2_plot, mode='lines',
                                 name='d2', line=dict(dash='dash'), yaxis='y2'))

        # 縦線
        fig.add_vline(x=chosen, line=dict(color='red', dash='dash'))

        fig.update_layout(
            xaxis_title="Pulse",
            yaxis=dict(title="mu"),
            yaxis2=dict(title="d2", overlaying='y', side='right'),
            width=800,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Selected pulse: {chosen:.1f}")

    except Exception as e:
        st.error(f"Error: {e}")
