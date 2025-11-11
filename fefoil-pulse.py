# fefoil-pulse-slider-aligned.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# ---------- user-configurable ----------
SKIP_HEADER = 3
SG_WINDOW = 31
SG_POLY = 5
SEARCH_MIN = 581650
# -------------------------------------

def load_xanes_file(file, skip_header=SKIP_HEADER):
    pulse_list = []
    mu_list = []
    lines = file.read().decode('utf-8', errors='ignore').splitlines()
    lines = lines[skip_header:]
    for line in lines:
        s = line.strip()
        if not s or s.startswith('#'): continue
        parts = s.split(',') if ',' in s else s.split()
        if len(parts) < 3: continue
        try:
            p = float(parts[0])
            I0 = float(parts[1])
            FeKa = float(parts[2])
        except:
            continue
        if I0 == 0: continue
        pulse_list.append(p)
        mu_list.append(FeKa / I0)
    if len(pulse_list) == 0:
        raise RuntimeError("No valid numeric data found.")
    return np.array(pulse_list), np.array(mu_list)

def compute_smoothed_d2(pulse, mu, window=SG_WINDOW, poly=SG_POLY):
    if window >= len(mu):
        lw = len(mu) - 1 if len(mu) % 2 == 0 else len(mu)
        window = max(5, lw)
        if window % 2 == 0: window -= 1
    mu_s = savgol_filter(mu, window_length=window, polyorder=poly, mode='interp')
    d1 = np.gradient(mu_s, pulse)
    d2 = np.gradient(d1, pulse)
    return mu_s, d2

def find_zero_crossing(p, d2, search_min=SEARCH_MIN):
    mask = p >= search_min
    idxs = np.where(mask)[0]
    if len(idxs) < 2: return None, None
    for i0 in range(idxs[0], len(p) - 1):
        y1, y2 = d2[i0], d2[i0+1]
        if y1 == 0: return p[i0], i0
        if y1 * y2 < 0:
            x1, x2 = p[i0], p[i0+1]
            x0 = x1 - y1 * (x2 - x1) / (y2 - y1)
            return x0, i0
    return None, None

# ---------- Streamlit UI ----------
st.title("Fe Foil E0 Pulse Determination (XANES)")

uploaded_file = st.file_uploader("Select Fe foil .dat file", type=['dat','txt'])
if uploaded_file is not None:
    try:
        pulse, mu = load_xanes_file(uploaded_file)
        mu_s, d2 = compute_smoothed_d2(pulse, mu)
        guess_pulse, _ = find_zero_crossing(pulse, d2)

        st.subheader("Pulse Data & Smoothed 2nd Derivative")

        # グラフの横軸範囲に合わせてスライダーの範囲を設定
        mask = pulse >= SEARCH_MIN
        p_plot = pulse[mask]
        mu_plot = mu[mask]
        mu_s_plot = mu_s[mask]
        d2_plot = d2[mask]
        min_p, max_p = int(p_plot.min()), int(p_plot.max())

        selected_pulse = st.slider(
            "Adjust pulse manually",
            min_value=min_p,
            max_value=max_p,
            value=int(guess_pulse if guess_pulse is not None else min_p),
            step=1
        )

        # 描画
        fig, ax1 = plt.subplots(figsize=(8,4))
        ax2 = ax1.twinx()
        ax1.plot(p_plot, mu_plot, label="mu (FeKa/I0)", color='black')
        ax1.plot(p_plot, mu_s_plot, label="mu_smooth", color='gray', linewidth=1)
        ax2.plot(p_plot, d2_plot, label="d2", linestyle='--')
        ax2.axhline(0, color='black', linewidth=1)
        ax1.axvline(selected_pulse, color='red', linestyle='--', label=f"selected={selected_pulse}")
        ax1.set_xlabel("Pulse")
        ax1.set_ylabel("mu")
        ax2.set_ylabel("d2")
        ax1.set_xlim(min_p, max_p)
        ax1.legend(loc='best')
        st.pyplot(fig)

        st.success(f"Selected pulse: {selected_pulse:.1f}")

    except Exception as e:
        st.error(f"Error: {e}")
