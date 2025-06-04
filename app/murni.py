import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import joblib 
import os

# --- Fungsi untuk Memuat dan Melatih Model ---
@st.cache_data
def load_data_and_get_preprocessors(file_path):
    try:
        df_orig = pd.read_csv(file_path, delimiter=',')
    except FileNotFoundError:
        st.error(f"Error: File dataset tidak ditemukan di path: {file_path}")
        st.error(f"Pastikan file dataset Anda berada di direktori 'uas' dan skrip Streamlit Anda berada di 'uas/app/'.")
        st.stop() 
    except Exception as e:
        st.error(f"Error saat membaca file CSV: {e}")
        st.stop()

    df = df_orig.copy()
    label_encoders = {}
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if 'Recurred' in df.columns and df['Recurred'].dtype == 'object' and 'Recurred' not in categorical_cols:
         categorical_cols.append('Recurred')


    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    if 'Recurred' in label_encoders:
        recurred_classes = label_encoders['Recurred'].classes_
    else:
        recurred_classes = None

    X = df.drop(columns=['Recurred'])
    y = df['Recurred']
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    return df_orig, X_scaled_df, y, label_encoders, scaler, feature_names, recurred_classes, categorical_cols

@st.cache_resource
def train_models(X_train, y_train):
    models = {}
    accuracies = {}

    # Random Forest
    model_RF = RandomForestClassifier(n_estimators=100, random_state=42)
    model_RF.fit(X_train, y_train)
    models['Random Forest'] = model_RF

    # Logistic Regression
    model_LR = LogisticRegression(max_iter=1000, random_state=42)
    model_LR.fit(X_train, y_train)
    models['Logistic Regression'] = model_LR

    # KNN
    model_KNN = KNeighborsClassifier(n_neighbors=5)
    model_KNN.fit(X_train, y_train)
    models['KNN'] = model_KNN

    return models

# --- Load Data dan Latih Model (Hanya sekali atau saat cache invalid) ---
url = "https://drive.google.com/uc?export=download&id=11lMuTGycjsA7i4soyqBjYTC-64pUdyxV"
dataset_path = "../Thyroid_Diff.csv"
df_original, X_scaled_df, y, label_encoders, scaler, feature_names, recurred_classes, categorical_cols_original = load_data_and_get_preprocessors(url)

# Split data (menggunakan data yang sudah di-scale untuk X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Latih model
trained_models = train_models(X_train, y_train)

# Hitung akurasi
accuracies = {}
for model_name, model in trained_models.items():
    y_pred = model.predict(X_test)
    accuracies[model_name] = accuracy_score(y_test, y_pred)


# --- Frontend Streamlit ---
st.title("Prediksi Rekurensi dengan Model Machine Learning")

st.sidebar.header("Pilih Model dan Masukkan Fitur")
selected_model_name = st.sidebar.selectbox("Pilih Model:", list(trained_models.keys()))

st.sidebar.subheader("Akurasi Model (pada Test Set):")
for name, acc in accuracies.items():
    st.sidebar.write(f"{name}: {acc:.4f}")

st.header(f"Input Fitur untuk Model: {selected_model_name}")

# Kumpulkan input dari pengguna
user_inputs = {}

original_categorical_values = {}
for col in categorical_cols_original:
    if col in df_original.columns and col != 'Recurred':
        original_categorical_values[col] = df_original[col].unique().tolist()

input_df_cols = [col for col in feature_names if col != 'Recurred']

for feature in input_df_cols:
    # Jika fitur adalah kategorikal (berdasarkan daftar dari df_original)
    if feature in categorical_cols_original:
        # Ambil kelas/kategori asli dari label encoder yang sesuai
        # Periksa apakah feature ada di label_encoders (untuk menghindari error jika ada kolom object yang tidak di-encode)
        if feature in label_encoders:
            # Gunakan nilai unik dari df_original untuk pilihan dropdown
            options = original_categorical_values.get(feature, ["Tidak ada data"]) # df_original[feature].unique().tolist()
            user_inputs[feature] = st.selectbox(f"Masukkan {feature}:", options=options, key=f"input_{feature}")
        else: # Jika tidak ada di label_encoders, mungkin sudah numerik atau kesalahan logika
             user_inputs[feature] = st.number_input(f"Masukkan {feature} (numerik):", value=0.0, format="%.2f", key=f"input_{feature}")

    # Jika fitur adalah numerik
    else:
        # Cari nilai min dan max dari kolom di X_scaled_df (setelah scaling) untuk panduan,
        # atau lebih baik dari df_original jika ingin input dalam skala asli.
        # Untuk input pengguna, lebih intuitif menggunakan skala asli jika memungkinkan.
        # Tapi karena model dilatih dengan data yang di-scale, input juga harus di-scale.
        # Jadi, kita minta input numerik, lalu kita akan scale.
        # Untuk contoh ini, kita akan menggunakan number_input sederhana.
        # Anda bisa menambahkan min_value dan max_value berdasarkan data asli Anda (df_original[feature].min())
        min_val = float(df_original[feature].min()) if pd.api.types.is_numeric_dtype(df_original[feature]) else 0.0
        max_val = float(df_original[feature].max()) if pd.api.types.is_numeric_dtype(df_original[feature]) else 100.0
        default_val = float(df_original[feature].mean()) if pd.api.types.is_numeric_dtype(df_original[feature]) else 0.0
        user_inputs[feature] = st.number_input(f"Masukkan {feature}:", min_value=min_val, max_value=max_val, value=default_val, key=f"input_{feature}")


if st.button("Prediksi Rekurensi"):
    # Buat DataFrame dari input pengguna
    input_list = []
    # Pastikan urutan kolom sesuai dengan feature_names yang digunakan saat training scaler
    for feature_col_name in feature_names: # feature_names adalah urutan kolom X sebelum scaling
        input_list.append(user_inputs[feature_col_name])

    input_df_for_processing = pd.DataFrame([input_list], columns=feature_names)
    processed_input = input_df_for_processing.copy() # Salin untuk diproses

    # 1. Label Encode input kategorikal pengguna
    for col_name in feature_names: # Iterasi sesuai urutan kolom X
        if col_name in label_encoders: # Cek apakah kolom ini punya label encoder
            # Dapatkan nilai string dari user_inputs
            user_str_input = user_inputs[col_name]
            try:
                # Transform string input ke bentuk numerik menggunakan encoder yang sudah di-fit
                processed_input[col_name] = label_encoders[col_name].transform([user_str_input])[0]
            except ValueError as e:
                st.error(f"Nilai '{user_str_input}' untuk fitur '{col_name}' tidak dikenali. Pilih dari opsi yang tersedia.")
                st.stop() # Hentikan eksekusi jika ada error
        else: # Jika kolom numerik, pastikan tipenya benar (float untuk scaler)
            processed_input[col_name] = float(user_inputs[col_name])


    # 2. Scale input numerik pengguna (semua fitur karena X_scaled_df adalah input model)
    try:
        scaled_user_input = scaler.transform(processed_input) # scaler di-fit pada X (sebelum train_test_split)
    except Exception as e:
        st.error(f"Error saat scaling input: {e}")
        st.error(f"Data yang akan di-scale: {processed_input.to_dict()}")
        st.error(f"Tipe data di processed_input: {processed_input.dtypes}")
        st.stop()

    # 3. Lakukan prediksi
    model_to_use = trained_models[selected_model_name]
    prediction_numeric = model_to_use.predict(scaled_user_input)
    prediction_proba = model_to_use.predict_proba(scaled_user_input)

    # 4. Tampilkan hasil
    st.subheader("Hasil Prediksi:")

    # Jika 'Recurred' di-encode, inverse transform untuk mendapatkan label asli
    if 'Recurred' in label_encoders and recurred_classes is not None:
        predicted_label = label_encoders['Recurred'].inverse_transform(prediction_numeric)[0]
    else:
        # Asumsikan 0 = Tidak Rekuren, 1 = Rekuren jika tidak ada info encoding
        # Atau sesuaikan dengan makna numerik 'Recurred' Anda
        predicted_label = "Rekuren" if prediction_numeric[0] == 1 else "Tidak Rekuren"


    st.write(f"Prediksi {selected_model_name}: **{predicted_label}**")

    # Tampilkan probabilitas jika model mendukung
    if hasattr(model_to_use, "predict_proba"):
        st.write("Probabilitas:")
        if recurred_classes is not None:
             # Tampilkan probabilitas dengan nama kelas asli jika 'Recurred' di-encode
            for i, class_label in enumerate(recurred_classes):
                st.write(f"  - {class_label}: {prediction_proba[0][i]:.4f}")
        else:
            # Jika 'Recurred' tidak di-encode atau numerik (misal 0 dan 1)
            st.write(f"  - Probabilitas Tidak Rekuren (Kelas 0): {prediction_proba[0][0]:.4f}")
            st.write(f"  - Probabilitas Rekuren (Kelas 1): {prediction_proba[0][1]:.4f}")

st.markdown("---")
st.subheader("Data Asli (Sampel):")
st.dataframe(df_original.head())

st.subheader("Data Setelah Label Encoding dan Sebelum Scaling (Digunakan untuk melatih Scaler dan Model):")
# Kita butuh X yang sudah di-encode tapi belum di-scale untuk dilihat,
# namun kode asli langsung scale X, jadi kita ambil dari `load_data_and_get_preprocessors`
# Mari kita modifikasi sedikit `load_data_and_get_preprocessors` untuk juga return df yang hanya diencode

# Jika Anda ingin melihat data yang sudah di-encode tapi belum di-scale:
# Anda perlu memodifikasi fungsi `load_data_and_get_preprocessors` untuk juga mengembalikan
# dataframe yang hanya di-labelencode sebelum di-scale.
# Untuk saat ini, kita tampilkan X_scaled_df (yang sudah di-scale)
# atau lebih baik X_train yang sudah di-scale (karena ini yang dipakai training)
st.dataframe(X_train.head())