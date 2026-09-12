import streamlit as st
import pandas as pd
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Nowcasting PDB Sektoral", page_icon="📈", layout="wide")

st.title("📈 Sistem Nowcasting PDB 17 Sektor Indonesia")
st.caption("Aplikasi Estimasi Cepat PDB Sektoral Berbasis Machine Learning")
st.markdown("---")

# Mengambil Katalog Model dari Memory/File
# (Di Colab, pastikan dictionary best_models_catalog sudah running sebelumnya)
try:
    catalog = best_models_catalog
except NameError:
    st.error("⚠️ Variable 'best_models_catalog' tidak ditemukan. Jalankan cell pembuatan katalog model terlebih dahulu!")
    st.stop()

# Sidebar Pilihan Sektor
st.sidebar.header("🎯 Parameter Sektor")
sektor_pilihan = st.sidebar.selectbox("Pilih Sektor PDB KBLI:", options=list(catalog.keys()))

model_info = catalog[sektor_pilihan]
model = model_info['model']
scaler = model_info['scaler']
fitur_semua = model_info['fitur']
fitur_aktif = model_info['fitur_aktif']
metode_terpakai = model_info['nama_metode']

# Display Info Sektor
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Sektor Terpilih", value=sektor_pilihan)
with col2:
    st.metric(label="Metode Terbaik", value=metode_terpakai)

st.markdown("---")
st.subheader("📥 Input Nilai Indikator Prediktor")

input_data = {}
form_col1, form_col2 = st.columns(2)

for idx, fitur in enumerate(fitur_semua):
    is_active = fitur in fitur_aktif
    label_text = f"{fitur} {'🟢 [Fitur Utama]' if is_active else '⚪ [Sensitivitas Rendah]'}"

    target_col = form_col1 if idx % 2 == 0 else form_col2
    with target_col:
        val = st.number_input(label=label_text, value=0.0, step=0.01, key=f"input_{fitur}")
        input_data[fitur] = val

st.markdown("---")

if st.button("🚀 Jalankan Nowcasting PDB", type="primary", use_container_width=True):
    df_single_input = pd.DataFrame([input_data])[fitur_semua]
    X_scaled = scaler.transform(df_single_input)
    prediksi_pdb = model.predict(X_scaled)[0]

    st.success("✅ Process Nowcasting Selesai!")
    st.subheader("📊 Hasil Estimasi PDB Sektoral:")
    st.title(f"Rp {prediksi_pdb:,.4f}")
