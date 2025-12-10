import streamlit as st
import pandas as pd
import numpy as np

st.title("温湿度マージツール（30分丸め対応・ロガー名自動正規化）")

uploaded = st.file_uploader("CSV をアップロード（温度・湿度 両方含む）", type=["csv"])
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded)

# ---- 1. 湿度・温度ブロックを自動判定 ----
hum_cols = [c for c in df.columns if "湿" in c or "Hum" in c or "RH" in c]
tem_cols = [c for c in df.columns if "温" in c or "Temp" in c or "T" == c]

# 行方向に Hum / Temp を含む列だけ抽出（DateTime も含める）
base_cols = ["Date/Time", "DateTime", "Time", "日時", "date", "time"]
base_cols = [c for c in base_cols if c in df.columns]

hum_block = df[base_cols + hum_cols].copy()
tem_block = df[base_cols + tem_cols].copy()

st.write(f"湿度ブロック shape: {hum_block.shape}")
st.write(f"温度ブロック shape: {tem_block.shape}")

# ---- 2. 長い形式に変換（列名 → Logger） ----
hum_long = hum_block.melt(
    id_vars=base_cols,
    var_name="Logger",
    value_name="Hum"
).dropna(subset=["Hum"])

tem_long = tem_block.melt(
    id_vars=base_cols,
    var_name="Logger",
    value_name="Temp"
).dropna(subset=["Temp"])

# ---- 3. 日時列を統一 ----
time_col = base_cols[0]  # どれでも良いが最初を採用

hum_long["Time"] = pd.to_datetime(hum_long[time_col], errors="coerce")
tem_long["Time"] = pd.to_datetime(tem_long[time_col], errors="coerce")

hum_long = hum_long.dropna(subset=["Time"])
tem_long = tem_long.dropna(subset=["Time"])

# ---- 4. ロガー名を正規化（大小文字無視・前後スペース除去） ----
def normalize(s):
    return str(s).strip().lower().replace(" ", "").replace("_", "")

hum_long["Logger_norm"] = hum_long["Logger"].apply(normalize)
tem_long["Logger_norm"] = tem_long["Logger"].apply(normalize)

# ---- 5. 時間を30分単位に丸め ----
hum_long["Time30"] = hum_long["Time"].dt.floor("30min")
tem_long["Time30"] = tem_long["Time"].dt.floor("30min")

# ---- 6. 重複を平均化（ロガー×丸め時刻で）----
hum_grp = hum_long.groupby(["Logger_norm", "Time30"], as_index=False)["Hum"].mean()
tem_grp = tem_long.groupby(["Logger_norm", "Time30"], as_index=False)["Temp"].mean()

# ---- 7. マージ ----
merged = pd.merge(hum_grp, tem_grp, on=["Logger_norm", "Time30"], how="inner")

# ---- 8. 結果が empty の確認 ----
st.write("マージ後 shape:", merged.shape)

if merged.empty:
    st.error("⚠ マージ結果が empty です。Logger 名が一致しない可能性があります。")
    st.write("ロガー名（湿度 側）:", hum_grp["Logger_norm"].unique())
    st.write("ロガー名（温度 側）:", tem_grp["Logger_norm"].unique())
else:
    st.success("マージ成功！")
    st.dataframe(merged)
