import streamlit as st
import pandas as pd
import joblib
import os
import json

# --- Konfigurasi Path (Menggunakan String Relatif) ---
# Path ini mengasumsikan Anda menjalankan 'streamlit run' dari folder utama proyek ('uas')
MODELS_BASE_DIR = "model-v2"  # Nama folder models Anda
DATASET_PATH = "Thyroid_Diff.csv" # Nama file dataset Anda
ACCURACY_PATH = os.path.join(MODELS_BASE_DIR, "accuracies.json") # Path ke file akurasi

# --- Pemetaan untuk UI ---
DATASET_MAPPING = {
    "Dataset Murni (Original)": "murni",
    "Dataset Oversampling (SMOTE)": "oversampling",
    "Dataset Undersampling": "undersampling"
}
MODEL_MAPPING = {
    "Random Forest": "model_rf",
    "Logistic Regression": "model_lr",
    "K-NN (K-Nearest Neighbors)": "model_knn"
}

# --- Fungsi-fungsi untuk Memuat Aset (dengan Caching) ---
@st.cache_data
def load_raw_data(file_path):
    """Memuat dataframe mentah untuk UI."""
    try:
        return pd.read_csv(file_path, delimiter=',')
    except FileNotFoundError:
        st.error(f"Error: Dataset mentah tidak ditemukan di '{file_path}'.")
        st.error("Pastikan Anda menjalankan aplikasi dari folder utama proyek ('uas').")
        st.stop()

@st.cache_resource
def load_preprocessors(base_dir):
    """Memuat scaler dan label encoders."""
    try:
        scaler = joblib.load(os.path.join(base_dir, "scaler.joblib"))
        label_encoders = joblib.load(os.path.join(base_dir, "label_encoders.joblib"))
        return scaler, label_encoders
    except FileNotFoundError:
        st.error(f"Error: 'scaler.joblib' atau 'label_encoders.joblib' tidak ditemukan di '{base_dir}'.")
        st.error("Pastikan file-file ini ada di dalam folder models Anda.")
        st.stop()

@st.cache_data
def load_accuracies(file_path):
    """Memuat file akurasi."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Peringatan: File '{file_path}' tidak ditemukan. Akurasi tidak dapat ditampilkan.")
        return None

@st.cache_resource
def load_model(dataset_folder, model_file):
    """Memuat model machine learning."""
    model_path = os.path.join(MODELS_BASE_DIR, dataset_folder, f"{model_file}.joblib")
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Error: File model tidak ditemukan di '{model_path}'.")
        st.stop()

# --- Logika Utama Aplikasi Streamlit ---
st.title("Prediksi Rekurensi Tiroid")

# Muat semua aset yang diperlukan
df_original = load_raw_data(DATASET_PATH)
scaler, label_encoders = load_preprocessors(MODELS_BASE_DIR)
accuracies = load_accuracies(ACCURACY_PATH)

# Dapatkan informasi fitur dari data mentah
categorical_cols_original = df_original.drop(columns=['Recurred']).select_dtypes(include='object').columns.tolist()
feature_names = [col for col in df_original.columns if col != 'Recurred']
recurred_classes = label_encoders.get('Recurred', {}).classes_

# --- Sidebar untuk Pilihan Pengguna ---
st.sidebar.header("Pilih Konfigurasi Model")
selected_dataset_name_ui = st.sidebar.selectbox("Pilih Jenis Dataset Training:", list(DATASET_MAPPING.keys()))
selected_model_name_ui = st.sidebar.selectbox("Pilih Model Machine Learning:", list(MODEL_MAPPING.keys()))

dataset_key = DATASET_MAPPING[selected_dataset_name_ui]
model_file_key = MODEL_MAPPING[selected_model_name_ui]

# --- Tampilkan Akurasi di Sidebar ---
if accuracies:
    try:
        accuracy_value = accuracies[dataset_key][selected_model_name_ui]
        st.sidebar.metric(
            label="Akurasi pada Test Set",
            value=f"{accuracy_value:.2%}" # Tampilkan dalam format persentase
        )
    except KeyError:
        st.sidebar.warning("Data akurasi untuk model ini tidak ditemukan.")

# Muat model yang sesuai dengan pilihan pengguna
model_to_use = load_model(dataset_key, model_file_key)

# --- Input Fitur dari Pengguna di Halaman Utama ---
st.header("Masukkan Fitur Pasien untuk Prediksi")
user_inputs = {}
for feature in feature_names:
    if feature in categorical_cols_original:
        options = df_original[feature].unique().tolist()
        user_inputs[feature] = st.selectbox(f"Masukkan {feature}:", options=options, key=f"input_{feature}")
    else:
        if pd.api.types.is_numeric_dtype(df_original[feature]):
            min_val = float(df_original[feature].min())
            max_val = float(df_original[feature].max())
            default_val = float(df_original[feature].mean())
            user_inputs[feature] = st.number_input(f"Masukkan {feature}:", min_value=min_val, max_value=max_val, value=default_val, key=f"input_{feature}")

# --- Tombol dan Logika Prediksi ---
if st.button("Prediksi Rekurensi"):
    input_df = pd.DataFrame([user_inputs])[feature_names]
    processed_input = input_df.copy()

    for col_name in categorical_cols_original:
        if col_name in processed_input.columns:
            user_str_input = processed_input[col_name].iloc[0]
            processed_input[col_name] = label_encoders[col_name].transform([user_str_input])[0]

    try:
        scaled_user_input = scaler.transform(processed_input)
        prediction_numeric = model_to_use.predict(scaled_user_input)
        prediction_proba = model_to_use.predict_proba(scaled_user_input)

        st.subheader(f"Hasil Prediksi dari Model '{selected_model_name_ui}'")
        if 'Recurred' in label_encoders and recurred_classes is not None:
            predicted_label = label_encoders['Recurred'].inverse_transform(prediction_numeric)[0]
        else:
            predicted_label = "Rekuren" if prediction_numeric[0] == 1 else "Tidak Rekuren"
        
        st.success(f"Prediksi Kondisi Pasien: **{predicted_label}**")

        st.write("Probabilitas Prediksi:")
        if recurred_classes is not None:
            st.dataframe(pd.DataFrame(prediction_proba, columns=recurred_classes))
        else:
            st.write(f"- Probabilitas Tidak Rekuren (Kelas 0): {prediction_proba[0][0]:.2%}")
            st.write(f"- Probabilitas Rekuren (Kelas 1): {prediction_proba[0][1]:.2%}")

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

# --- Tampilan Data Sampel ---
st.markdown("---")
st.subheader("Sampel Data Asli")
st.dataframe(df_original.head())