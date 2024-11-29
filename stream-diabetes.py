import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pre-trained model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Load dataset globally
diabetes_dataset = pd.read_csv('diabetes_data_upload.csv')

# Sidebar - Main menu
st.sidebar.title("Menu Utama")
menu = st.sidebar.radio("Pilih Menu", ["Deskripsi", "Dataset", "Visualisasi", "Prediksi"])

# Deskripsi
if menu == "Deskripsi":
    st.title('Prediksi Diabetes Menggunakan Data Mining')
    st.markdown("""
    Ini adalah model prediktif untuk mendeteksi diabetes berdasarkan berbagai indikator kesehatan. 
    Model ini menggunakan **Support Vector Machine (SVM)** sebagai algoritma klasifikasi untuk memprediksi apakah seseorang menderita diabetes atau tidak.

    #### Sumber Dataset:
    Dataset ini berdasarkan data dari **https://www.kaggle.com/datasets/himanshu86503/diabetes**, yang mencakup berbagai fitur kesehatan dan variabel target `class` (1: Diabetes, 0: Tidak Diabetes).

    #### Gambaran Model:
    Model ini dilatih menggunakan algoritma **SVM**, yang dikenal efektif dalam tugas klasifikasi. Fitur yang digunakan dalam prediksi meliputi informasi seperti usia, jenis kelamin, polyuria, polidipsia, dan gejala lainnya yang terkait dengan diabetes.

    #### Pembuat:
    Proyek ini dibuat oleh **Fahrul Roziqin Akbar** sebagai bagian dari proyek Data Mining.
    """)


# Dataset
elif menu == "Dataset":
    st.title("View Dataset")
    st.write(diabetes_dataset.head())

# Visualisasi
elif menu == "Visualisasi":
    st.title("Dataset Visualizations")

    # Plotting a bar chart for the distribution of 'class'
    fig, ax = plt.subplots()
    diabetes_dataset['class'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'], ax=ax)
    plt.title('Distribusi Kelas (Diabetes vs Non-Diabetes)')
    plt.xlabel('Kelas')
    plt.ylabel('Jumlah')
    st.pyplot(fig)

    # Plotting a correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = diabetes_dataset.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Matriks Korelasi Fitur')
    st.pyplot(fig)

# Prediksi
elif menu == "Prediksi":
    st.title("Input Data untuk Prediksi Diabetes")

    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input('Umur', min_value=0)

    with col2:
        Gender = st.selectbox('Jenis Kelamin', ['Pria', 'Wanita'])

    with col1:
        Polyuria = st.selectbox('Apakah Anda sering buang air kecil?', ['Ya', 'Tidak'])

    with col2:
        Polydipsia = st.selectbox('Apakah Anda sering merasa haus?', ['Ya', 'Tidak'])

    with col1:
        Suddenweightloss = st.selectbox('Apakah Anda mengalami penurunan berat badan tiba-tiba?', ['Ya', 'Tidak'])

    with col2:
        Weakness = st.selectbox('Apakah Anda merasa lemah?', ['Ya', 'Tidak'])

    with col1:
        Polyphagia = st.selectbox('Apakah Anda merasa lapar berlebihan?', ['Ya', 'Tidak'])

    with col2:
        Genitalthrush = st.selectbox('Apakah Anda mengalami infeksi genital?', ['Ya', 'Tidak'])

    with col1:
        Visualblurring = st.selectbox('Apakah Anda mengalami gangguan penglihatan?', ['Ya', 'Tidak'])

    with col2:
        Itching = st.selectbox('Apakah Anda merasa gatal?', ['Ya', 'Tidak'])

    with col1:
        Irritability = st.selectbox('Apakah Anda merasa mudah marah?', ['Ya', 'Tidak'])

    with col2:
        Delayedhealing = st.selectbox('Apakah luka Anda lambat sembuh?', ['Ya', 'Tidak'])

    with col1:
        Partialparesis = st.selectbox('Apakah Anda mengalami kelumpuhan sebagian?', ['Ya', 'Tidak'])

    with col2:
        Musclestiffness = st.selectbox('Apakah Anda mengalami kekakuan otot?', ['Ya', 'Tidak'])

    with col1:
        Alopecia = st.selectbox('Apakah Anda mengalami kerontokan rambut?', ['Ya', 'Tidak'])

    with col2:
        Obesity = st.selectbox('Apakah Anda obesitas?', ['Ya', 'Tidak'])

    # Handle prediction
    diab_diagnosis = ''

    if st.button('Test Prediksi Diabetes'):
        # Map inputs to numeric values
        Gender = 1 if Gender == 'Pria' else 0
        Polyuria = 1 if Polyuria == 'Ya' else 0
        Polydipsia = 1 if Polydipsia == 'Ya' else 0
        Suddenweightloss = 1 if Suddenweightloss == 'Ya' else 0
        Weakness = 1 if Weakness == 'Ya' else 0
        Polyphagia = 1 if Polyphagia == 'Ya' else 0
        Genitalthrush = 1 if Genitalthrush == 'Ya' else 0
        Visualblurring = 1 if Visualblurring == 'Ya' else 0
        Itching = 1 if Itching == 'Ya' else 0
        Irritability = 1 if Irritability == 'Ya' else 0
        Delayedhealing = 1 if Delayedhealing == 'Ya' else 0
        Partialparesis = 1 if Partialparesis == 'Ya' else 0
        Musclestiffness = 1 if Musclestiffness == 'Ya' else 0
        Alopecia = 1 if Alopecia == 'Ya' else 0
        Obesity = 1 if Obesity == 'Ya' else 0

        # Make prediction
        input_data = [Age, Gender, Polyuria, Polydipsia, Suddenweightloss, Weakness, Polyphagia, Genitalthrush, Visualblurring, Itching, Irritability, Delayedhealing, Partialparesis, Musclestiffness, Alopecia, Obesity]
        
        diab_prediction = diabetes_model.predict([input_data])

        if(diab_prediction[0] == 1):
            diab_diagnosis = 'Pasien terkena Diabetes'
        else:
            diab_diagnosis = 'Pasien tidak terkena Diabetes'

        st.success(diab_diagnosis)
