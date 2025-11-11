import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
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
# GUIで複数ファイル選択
# -----------------------------
Tk().withdraw()
filenames = askopenfilenames(title="解析するdatファイルを複数選択")
pulse_ref = float(input("基準パルスを入力: "))

for filename in filenames:
    basename = os.path.splitext(os.path.basename(filename))[0]
    print(f"\n--- Processing {basename} ---")

    # -----------------------------
    # データ読み込み
    # -----------------------------
    data = pd.read_csv(filename, skiprows=3, header=None)
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
    # 低エネルギー側最大値、 高エネルギー側最小値
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

    # 初期値と境界
    p0_gauss = [0.1, 7111.8, 0.5, 0.1, 7113.7, 0.5]
    lower_gauss = [0, 7110, 0, 0, 7112, 0]
    upper_gauss = [np.inf, 7112, 2, np.inf, 7115, 2]

    popt_gauss, _ = curve_fit(two_gauss, E_gauss, I_gauss,
                              p0=p0_gauss, bounds=(lower_gauss, upper_gauss),
                              maxfev=5000)
    gauss_fit = two_gauss(E_gauss, *popt_gauss)

    # 面積加重重心
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

    txt_filename = f"{basename}_fitting.txt"
    with open(txt_filename, 'w') as f:
        f.write("\n".join(txt_lines))
    print(f"結果を保存: {txt_filename}")

    # -----------------------------
    # グラフ出力（7108-7116 eV）
    # -----------------------------
    gauss1 = gaussian(E_gauss, popt_gauss[0], popt_gauss[1], popt_gauss[2])
    gauss2 = gaussian(E_gauss, popt_gauss[3], popt_gauss[4], popt_gauss[5])

    plt.figure(figsize=(10,6), constrained_layout=True)
    plt.plot(energy, Fe_Ka_norm, 'ko', alpha=0.5, label=f'{basename} raw data')
    plt.plot(energy, Fe_Ka_smooth, 'k-', alpha=0.8, label='smoothed data')
    plt.plot(energy, baseline, 'r--', linewidth=2, label='auto linear baseline')
    plt.plot(E_gauss, gauss1 + baseline[mask_gauss], 'g--', linewidth=2, label='Gaussian 1')
    plt.plot(E_gauss, gauss2 + baseline[mask_gauss], 'm--', linewidth=2, label='Gaussian 2')
    plt.plot(E_gauss, gauss_fit + baseline[mask_gauss], 'b-', linewidth=2, label='Total fit')

    # Centroid表示
    plt.axvline(centroid, color='blue', linestyle=':', label=f'Centroid = {centroid:.2f} eV')

    # ガウス中心値をグラフ上に明記
    plt.text(popt_gauss[1], gauss1.max() + baseline[mask_gauss].max()*0.05,
             f'{popt_gauss[1]:.2f} eV', color='green', ha='center')
    plt.text(popt_gauss[4], gauss2.max() + baseline[mask_gauss].max()*0.05,
             f'{popt_gauss[4]:.2f} eV', color='magenta', ha='center')

    plt.xlabel("Energy (eV)")
    plt.ylabel("Normalized intensity")
    plt.xlim(7108, 7116)

    mask_ylim = (energy >= 7114) & (energy <= 7116)
    ylim_max = np.ceil(Fe_Ka_smooth[mask_ylim].max() / 0.01) * 0.01
    plt.ylim(0, ylim_max)

    plt.legend()
    png_filename = f"{basename}_fitting.png"
    plt.savefig(png_filename, dpi=300)
    plt.close()
    print(f"グラフを保存: {png_filename}")
