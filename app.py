# =============================================
# --- APLIKASI WEB PREDIKSI HARGA TIKET PESAWAT ---
# =============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- FUNGSI UNTUK MEMUAT MODEL DAN KOLOM ---
@st.cache_data
def load_model_and_columns():
    """Memuat model dan daftar kolom yang tersimpan."""
    model = joblib.load('flight_price_model.pkl')
    columns = joblib.load('flight_price_columns.pkl')
    return model, columns

# Memuat model dan kolom saat aplikasi pertama kali dijalankan
try:
    model, model_columns = load_model_and_columns()
except FileNotFoundError:
    st.error("File model atau kolom tidak ditemukan. Pastikan 'flight_price_model.pkl' dan 'flight_price_columns.pkl' ada di repositori GitHub Anda.")
    st.stop()


# --- FUNGSI UNTUK MEMBUAT FITUR DARI INPUT PENGGUNA ---
def preprocess_input(airline, source_city, destination_city, class_type, stops, departure_date):
    """Memproses input pengguna menjadi format yang bisa diterima model."""
    # Menghitung sisa hari
    scrape_date = pd.to_datetime(datetime.now().date())
    days_left = (departure_date - scrape_date).days

    # Membuat DataFrame dari input
    input_data = {
        'stops': [stops],
        'duration': [12.22], # Menggunakan rata-rata sebagai placeholder
        'days_left': [days_left],
        'class': [1 if class_type == 'Business' else 0]
    }
    input_df = pd.DataFrame(input_data)

    # --- Rekayasa Fitur (Harus SAMA PERSIS dengan di notebook) ---
    temp_date = pd.to_datetime(departure_date)
    
    # Skor Musiman Bertingkat
    is_weekend = 1 if temp_date.dayofweek >= 4 else 0
    current_year = temp_date.year
    summer_peak = (pd.to_datetime(f'{current_year}-08-10') <= temp_date <= pd.to_datetime(f'{current_year}-08-25'))
    winter_peak = (pd.to_datetime(f'{current_year}-12-20') <= temp_date <= pd.to_datetime(f'{current_year}-12-31'))
    peak_season = 1 if summer_peak or winter_peak else 0
    
    holiday_dates_str = [f'{current_year}-08-15', f'{current_year}-10-02', f'{current_year}-10-20', f'{current_year}-12-25']
    holiday_dates = [pd.to_datetime(date) for date in holiday_dates_str]
    holiday_window = {date + pd.Timedelta(days=i) for date in holiday_dates for i in range(-2, 2)}
    is_national_holiday = 1 if temp_date in holiday_window else 0

    if is_national_holiday == 1:
        input_df['seasonality_score'] = 3
    elif is_weekend == 1:
        input_df['seasonality_score'] = 2
    elif peak_season == 1:
        input_df['seasonality_score'] = 1
    else:
        input_df['seasonality_score'] = 0

    # Fitur Route dan Booking Window (dibuat sebagai kolom sementara untuk encoding)
    input_df['route'] = source_city + '_' + destination_city
    
    booking_window_val = ''
    if days_left <= 2: booking_window_val = 'Last_Minute'
    elif days_left <= 7: booking_window_val = 'One_Week_Out'
    elif days_left <= 15: booking_window_val = 'Two_Weeks_Out'
    elif days_left <= 30: booking_window_val = 'One_Month_Out'
    else: booking_window_val = 'More_Than_Month'
    
    input_df['booking_window'] = booking_window_val

    input_df['airline'] = airline
    
    # Melakukan One-Hot Encoding
    input_encoded = pd.get_dummies(input_df)
    
    # Menyelaraskan kolom dengan kolom saat training (CARA YANG LEBIH AMAN)
    final_input = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    return final_input

# =================
# --- UI APLIKASI ---
# =================
st.set_page_config(page_title="Prediksi Harga Tiket Pesawat", layout="wide")
st.title("✈️ Aplikasi Prediksi Harga Tiket Pesawat Dinamis")
st.markdown("Masukkan detail penerbangan Anda untuk mendapatkan estimasi harga tiket.")

col1, col2 = st.columns([2, 1])
with col1:
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        st.header("Detail Rute")
        airline = st.selectbox("Pilih Maskapai", ('Vistara', 'Air_India', 'Indigo', 'GO_FIRST', 'AirAsia', 'SpiceJet'))
        source_city = st.selectbox("Kota Keberangkatan", ('Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'))
        destination_city = st.selectbox("Kota Tujuan", ('Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'))
    with sub_col2:
        st.header("Detail Penerbangan")
        class_type = st.radio("Kelas Penerbangan", ('Economy', 'Business'))
        stops = st.selectbox("Jumlah Transit", options=[0, 1, 2], index=0)
        departure_date = st.date_input("Tanggal Keberangkatan", min_value=datetime.now())

st.write("")
if st.button("Prediksi Harga", use_container_width=True):
    if source_city == destination_city:
        st.error("Kota keberangkatan dan tujuan tidak boleh sama.")
    else:
        processed_input = preprocess_input(airline, source_city, destination_city, class_type, stops, pd.to_datetime(departure_date))
        with st.spinner('Memprediksi harga...'):
            prediction = model.predict(processed_input)
            predicted_price_inr = prediction[0]
            INR_TO_IDR_RATE = 196 
            predicted_price_idr = predicted_price_inr * INR_TO_IDR_RATE
        st.success("Estimasi Harga Tiket Anda:")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Harga dalam Rupee India (INR)", value=f"₹ {predicted_price_inr:,.0f}")
        with res_col2:
            st.metric(label="Harga dalam Rupiah (IDR)", value=f"Rp {predicted_price_idr:,.0f}")
        st.info("Catatan: Harga ini adalah estimasi dan dapat berubah sewaktu-waktu. Kurs 1 INR = 196 IDR.")

st.markdown("---")
st.markdown("Dibuat dengan Streamlit | Model: Random Forest")
