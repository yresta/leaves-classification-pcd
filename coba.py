import streamlit as st
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import numpy as np
import cv2
import joblib

# Load pipeline yang sudah disimpan
pipeline = joblib.load('model_pipeline.pkl')

def preprocess_image(img):
    img = img.convert('RGB').resize((128, 128))
    img_np = np.array(img)

    # Ekstraksi color moments dari HSV
    hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    def color_moments(channel):
        mean = np.mean(channel)
        var = np.var(channel)
        skew = np.mean((channel - mean) ** 3) / (np.std(channel) ** 3 + 1e-8)
        return [mean, var, skew]

    color_features = []
    for i in range(3):
        color_features.extend(color_moments(hsv_img[:, :, i]))

    # Ekstraksi GLCM dari grayscale
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    texture_features = [
        graycoprops(glcm, prop)[0, 0] for prop in ['contrast', 'correlation', 'energy', 'homogeneity']
    ]

    features = np.array(color_features + texture_features).reshape(1, -1)
    return features

def main():
    st.set_page_config(page_title="Klasifikasi Daun Anggur", layout="centered")
    st.sidebar.title("Navigasi")
    menu = st.sidebar.selectbox("Pilih Halaman", ["Penjelasan", "Prediksi", "Tentang"])

    if menu == "Penjelasan":
        st.title("🧠 Penjelasan Aplikasi Klasifikasi Daun Anggur")

        st.subheader("🌱 Latar Belakang Singkat")
        st.markdown("""
        Klasifikasi daun tanaman, khususnya daun anggur (*grapevine*), merupakan aspek penting dalam botani dan pertanian, baik untuk identifikasi spesies maupun deteksi penyakit tanaman. 
        Daun mengandung informasi morfologis seperti bentuk, warna, dan tekstur yang dapat dimanfaatkan dalam sistem identifikasi otomatis. 
        Tanaman anggur sebagai komoditas bernilai ekonomi tinggi membutuhkan identifikasi varietas yang akurat, karena perbedaan varietas memengaruhi kebutuhan pemupukan, irigasi, dan pengendalian hama.

        Proses identifikasi manual cenderung lambat dan subjektif, sehingga diperlukan sistem otomatis berbasis citra digital. Dengan bantuan teknologi machine learning, proses klasifikasi dapat dilakukan secara cepat dan akurat. 
        Penelitian ini memanfaatkan Grapevine Leaves Image Dataset dari Kaggle untuk membangun sistem klasifikasi berbasis SVM yang menggabungkan ekstraksi fitur dengan Color Moments dan Gray Level Co-occurrence Matrix (GLCM).
        Dataset yang kami gunakan bersumber dari [Kaggle Grapevine Leaves Dataset](https://www.kaggle.com/datasets/muratkokludataset/grapevine-leaves-image-dataset), dengan melakukan klasifikasi terhadap lima kelas varietas daun anggur, yaitu **Ak, Ala_Idris, Buzgulu, Dimnit**, dan **Nazli**.
        """)

        st.subheader("🎯 Tujuan")
        st.markdown("""
        Aplikasi ini dikembangkan dengan tujuan:
        1. Membangun model klasifikasi daun anggur yang akurat berbasis ekstraksi fitur warna dan tekstur.
        2. Mengidentifikasi fitur yang paling berpengaruh dalam membedakan varietas daun anggur.
        3. Mengukur performa algoritma **Support Vector Machine (SVM)** dalam klasifikasi daun anggur menggunakan dataset dari Kaggle.
        """)

        st.subheader("🧪 Metode")
        st.markdown("""
        Ekstraksi fitur citra dilakukan menggunakan dua pendekatan utama:
        - **Color Moments** dari ruang warna HSV untuk fitur warna (mean, varian, skewness).
        - **Gray Level Co-occurrence Matrix (GLCM)** untuk fitur tekstur (contrast, correlation, energy, homogeneity).

        Model SVM dilatih dalam pipeline machine learning dan disimpan dalam bentuk file `.pkl` untuk digunakan dalam aplikasi prediksi ini.
        """)

    elif menu == "Prediksi":
        st.title("🌿 Prediksi Varietas Daun Anggur")
        st.markdown("""
        **Upload gambar daun anggur Anda di sini!** 🚀  
        Aplikasi akan menganalisis dan memprediksi varietas daun anggur secara otomatis.
        """)

        uploaded_file = st.file_uploader(
            "📤 Drag & drop gambar daun anggur Anda (format PNG/JPG/JPEG) di sini atau klik untuk memilih file", 
            type=["png", "jpg", "jpeg"],
            help="Pastikan gambar daun jelas dengan background yang kontras"
        )
        
        if uploaded_file is not None:
            st.success("✔️ Gambar berhasil diupload!")
            image = Image.open(uploaded_file)
            st.image(image, caption='Gambar daun yang diupload', use_column_width=True)
            
            st.markdown("""
            **Langkah selanjutnya:**  
            Klik tombol prediksi di bawah ini untuk mengetahui klasifikasi daun anggur Anda!
            """)

            if st.button('🔍 Analisis & Prediksi Kelas', help="Proses mungkin memakan waktu beberapa detik"):
                with st.spinner('🔄 Sedang menganalisis gambar...'):
                    features = preprocess_image(image)
                    prediction = pipeline.predict(features)
                
                st.balloons()
                st.success(f"""
                **🎉 Hasil Prediksi:**  
                Jenis daun anggur Anda adalah **{prediction[0]}**  
                
                *Hasil analisis berdasarkan fitur morfologi daun menggunakan model SVM*
                """)

    elif menu == "Tentang":
        st.title(" 🖥️ Tentang Aplikasi")
        st.markdown("""
        **Aplikasi Klasifikasi Daun Anggur** ini dikembangkan sebagai bagian dari proyek pembelajaran 
        dalam bidang klasifikasi citra berbasis machine learning.

        - **Framework**: Streamlit
        - **Ekstraksi Fitur**: OpenCV, scikit-image
        - **Model**: Support Vector Machine (SVM) dalam pipeline scikit-learn
        - **Dataset**: Grapevine Leaves Image Dataset dari Kaggle
        - **Pengembang**: 
            - Nailah Masruroh 		    (23031554100)
            - Tutik Hidayah Hardiyanti 	(23031554156)
            - Yulia Eka Restania 	    (23031554199)

        Aplikasi ini diharapkan dapat membantu proses identifikasi varietas daun anggur secara otomatis, efisien, dan akurat.
        """)

if __name__ == "__main__":
    main()
