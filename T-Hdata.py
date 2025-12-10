import streamlit as st
import pandas as pd

st.title("温湿度ロガー データ整理アプリ（30分丸め・列自動判定版）")

uploaded_files = st.file_uploader("Excel ファイルを選択", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    all_data = []

    for file in uploaded_files:
        st.write(f"処理中 → {file.name}")

        # ヘッダー2行目を読み込む
        df = pd.read_excel(file, header=1)

        # ---- ポイント：Date/Time 列を探す ----
        dt_cols = [i for i, c in enumerate(df.columns) if "Date" in str(c) or "Time" in str(c)]

        if len(dt_cols) != 2:
            st.error("Date/Time 列が2つ見つかりません。ファイル形式が違います。")
            st.write(df.head())
            continue

        dt1, dt2 = dt_cols  # 左ブロックの Date/Time と右ブロックの Date/Time

        # 湿度ブロックは dt1 から dt2 - 1
        hum_block = df.iloc[:, dt1:dt2].copy()
        hum_block.columns = ["Time"] + list(hum_block.columns[1:])

        # 温度ブロックは dt2 から最後まで
        tem_block = df.iloc[:, dt2:].copy()
        tem_block.columns = ["Time"] + list(tem_block.columns[1:])

        # datetime 化
        hum_block["Time"] = pd.to_datetime(hum_block["Time"], errors="coerce")
        tem_block["Time"] = pd.to_datetime(tem_block["Time"], errors="coerce")
# --- ここから下をそのまま置き換えれば OK ---

st.write(f"湿度ブロック shape: {hum_block.shape}")
st.write(f"温度ブロック shape: {tem_block.shape}")

# 1. long 形式へ
hum_long = hum_block.melt(id_vars=["Time"], var_name="Logger", value_name="Hum")
tem_long = tem_block.melt(id_vars=["Time"], var_name="Logger", value_name="Temp")

# 欠損除去
hum_long = hum_long.dropna(subset=["Hum"])
tem_long = tem_long.dropna(subset=["Temp"])

# 2. Logger を正規化
def normalize(x):
    return str(x).strip().lower().replace(" ", "").replace("_", "")

hum_long["Logger_norm"] = hum_long["Logger"].apply(normalize)
tem_long["Logger_norm"] = tem_long["Logger"].apply(normalize)

# 3. Time → datetime
hum_long["Time"] = pd.to_datetime(hum_long["Time"], errors="coerce")
tem_long["Time"] = pd.to_datetime(tem_long["Time"], errors="coerce")

hum_long = hum_long.dropna(subset=["Time"])
tem_long = tem_long.dropna(subset=["Time"])

# 4. 30分に丸める
hum_long["Time30"] = hum_long["Time"].dt.floor("30min")
tem_long["Time30"] = tem_long["Time"].dt.floor("30min")

# 5. logger × 時間で平均化
hum_grp = hum_long.groupby(["Logger_norm", "Time30"], as_index=False)["Hum"].mean()
tem_grp = tem_long.groupby(["Logger_norm", "Time30"], as_index=False)["Temp"].mean()

# 6. マージ
merged = pd.merge(hum_grp, tem_grp, on=["Logger_norm", "Time30"], how="inner")

st.write("merged shape:", merged.shape)
st.dataframe(merged)



