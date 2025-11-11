#!/usr/bin/env python3
# determine_fe_e0.py
# Determine Fe foil first inflection (E0) pulse via GUI-confirmed 2nd derivative zero-crossing.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# try tkinter; if not available, fall back to CLI prompts
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TK_OK = True
except Exception:
    TK_OK = False

# ---------- user-configurable ----------
SKIP_HEADER = 3         # lines to skip
SG_WINDOW = 31          # smoothing window (odd)
SG_POLY = 5             # smoothing polynomial
SEARCH_MIN = 581650     # search pulse ≥ this value
OUT_DECIMALS = 1
# -------------------------------------

def load_xanes_file(filepath, skip_header=SKIP_HEADER):
    pulse_list = []
    mu_list = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

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
    """
    Find first zero-cross (sign change) in d2 for p >= SEARCH_MIN
    Linear interpolation for more accurate zero location.
    """
    mask = p >= SEARCH_MIN
    idxs = np.where(mask)[0]
    if len(idxs) < 2:
        return None, None

    for i0 in range(idxs[0], len(p) - 1):
        y1 = d2[i0]
        y2 = d2[i0+1]
        if y1 == 0:
            return p[i0], i0
        if y1 * y2 < 0:  # sign flips
            x1 = p[i0]
            x2 = p[i0+1]
            # linear interpolation
            x0 = x1 - y1 * (x2 - x1) / (y2 - y1)
            return x0, i0
    return None, None


def ask_file_via_dialog():
    root = tk.Tk(); root.withdraw()
    fname = filedialog.askopenfilename(
        title="Select Fe foil .dat file",
        filetypes=[("dat files","*.dat"), ("text files","*.txt"), ("all","*.*")]
    )
    root.destroy()
    return fname


def confirm_and_choose(pulse, mu, mu_s, d2, guess_pulse, input_filepath):
    mask = pulse >= SEARCH_MIN
    p_plot = pulse[mask]
    mu_plot = mu[mask]
    mu_s_plot = mu_s[mask]
    d2_plot = d2[mask]

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax2 = ax1.twinx()

    ax1.plot(p_plot, mu_plot, label="mu (FeKa/I0)", color='black')
    ax1.plot(p_plot, mu_s_plot, label="mu_smooth", color='gray', linewidth=1)

    ax2.plot(p_plot, d2_plot, label="d2", linestyle='--')

    ax2.axhline(0, color='black', linewidth=1)

    if guess_pulse is not None:
        ax1.axvline(guess_pulse, color='r', linestyle='--',
                    label=f"guess={guess_pulse:.1f}")

    ax1.set_xlabel("Pulse")
    ax1.set_ylabel("mu")
    ax2.set_ylabel("d2")
    ax1.set_title(os.path.basename(input_filepath))

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, loc='best')
    plt.tight_layout()
    plt.show(block=False)

    if TK_OK:
        root = tk.Tk(); root.withdraw()
        ans = messagebox.askyesno("Confirm",
                                  f"Use guessed pulse = {guess_pulse:.1f} ? (Yes = accept, No = pick manually)")
        root.destroy()
    else:
        t = input(f"Use guessed pulse = {guess_pulse:.1f}? (y/N): ")
        ans = t.lower().startswith("y")

    chosen = guess_pulse
    if not ans:
        print("Click new pulse on the figure (1 click).")
        pts = plt.ginput(1, timeout=0)
        plt.close(fig)
        if len(pts) == 0:
            raise RuntimeError("No click detected.")
        chosen = float(pts[0][0])
    else:
        plt.close(fig)

    return chosen


def save_result_txt(chosen_pulse, input_filepath, decimals=OUT_DECIMALS):
    out_dir = os.path.dirname(os.path.abspath(input_filepath)) or "."
    fname = f"{chosen_pulse:.{decimals}f}.txt"
    outpath = os.path.join(out_dir, fname)
    with open(outpath,'w') as f:
        f.write(f"file: {input_filepath}\n")
        f.write(f"pulse: {chosen_pulse:.{decimals}f}\n")
    return outpath


def main():
    if TK_OK:
        filepath = ask_file_via_dialog()
    else:
        if len(sys.argv) >= 2:
            filepath = sys.argv[1]
        else:
            filepath = input("path: ")
    if not filepath:
        print("No file selected.")
        return

    pulse, mu = load_xanes_file(filepath)
    mu_s, d2 = compute_smoothed_d2(pulse, mu)

    guess_pulse, idx = find_zero_crossing(pulse, d2)
    if guess_pulse is None:
        print("No zero-crossing found ≥ SEARCH_MIN")
        guess_pulse = pulse[np.argmin(np.abs(pulse - SEARCH_MIN))]

    chosen = confirm_and_choose(pulse, mu, mu_s, d2, guess_pulse, filepath)
    out = save_result_txt(chosen, filepath)

    if TK_OK:
        messagebox.showinfo("Saved", f"Saved {out}")
    else:
        print("Saved:", out)


if __name__ == "__main__":
    main()
