import streamlit as st
import pandas as pd
import numpy as np
import io

st.title("温湿度データ整理アプリ（30分丸め & ロガー名元列名保持）")

uploaded_files = st.file_uploader(
    "月ごとのエクセルファイルを複数選択してください",
    type=["xlsx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

all_merged = []

for file in uploaded_files:
    st.write(f"---\n### 📄 読み込み：{file.name}")

    # 列名の自動変更を防ぐ
    df = pd.read_excel(io.BytesIO(file.read()), header=1)
    cols = df.columns.tolist()
    st.write("列名:", cols)

    # Date/Time 列の位置を検出
    time_positions = [i for i, c in enumerate(cols) if "Date" in str(c) or "Time" in str(c)]
    if len(time_positions) < 2:
        st.error("2つ目の Date/Time 列が見つかりません。ファイル形式を確認してください。")
        continue  # 問題のあるファイルはスキップ

    hum_start = time_positions[0]
    tem_start = time_positions[1]

    hum_cols = cols[hum_start:tem_start]
    tem_cols = cols[tem_start:]

    hum_block = df[hum_cols].copy()
    tem_block = df[tem_cols].copy()

    st.write(f"湿度ブロック shape: {hum_block.shape}")
    st.write(f"温度ブロック shape: {tem_block.shape}")

    # long化（Logger名は元の列名を使用）
    hum_long = hum_block.melt(id_vars=[hum_cols[0]], var_name="Logger", value_name="Hum")
    tem_long = tem_block.melt(id_vars=[tem_cols[0]], var_name="Logger", value_name="Temp")

    # Time列名を統一
    hum_long = hum_long.rename(columns={hum_cols[0]: "Time"})
    tem_long = tem_long.rename(columns={tem_cols[0]: "Time"})

    # NaN削除
    hum_long = hum_long.dropna(subset=["Hum"])
    tem_long = tem_long.dropna(subset=["Temp"])

    hum_long["Time"] = pd.to_datetime(hum_long["Time"], errors="coerce")
    tem_long["Time"] = pd.to_datetime(tem_long["Time"], errors="coerce")

    hum_long = hum_long.dropna(subset=["Time"])
    tem_long = tem_long.dropna(subset=["Time"])

    # Logger名正規化（比較用）
    def normalize(x):
        return str(x).strip().lower().replace(" ", "").replace("_", "")

    hum_long["Logger_norm"] = hum_long["Logger"].apply(normalize)
    tem_long["Logger_norm"] = tem_long["Logger"].apply(normalize)

    # 30分丸め
    hum_long["Time30"] = hum_long["Time"].dt.floor("30min")
    tem_long["Time30"] = tem_long["Time"].dt.floor("30min")

    # 平均化
    hum_grp = hum_long.groupby(["Logger_norm", "Time30"], as_index=False)["Hum"].mean()
    tem_grp = tem_long.groupby(["Logger_norm", "Time30"], as_index=False)["Temp"].mean()

    # 湿度と温度をマージ
    merged = pd.merge(hum_grp, tem_grp, on=["Logger_norm", "Time30"], how="inner")
    merged["SourceFile"] = file.name

    # 元のLogger名も残す（必要に応じて）
    merged["Logger"] = merged["Logger_norm"]

    st.write("マージ結果 shape:", merged.shape)
    all_merged.append(merged)

# 全結合
if all_merged:
    final_df = pd.concat(all_merged, ignore_index=True)
    st.write("### 🎉 全ファイル統合結果")
    st.write(final_df)

    csv = final_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 CSV をダウンロード",
        data=csv,
        file_name="merged_THdata.csv",
        mime="text/csv"
    )
else:
    st.warning("有効なデータがありませんでした。")
