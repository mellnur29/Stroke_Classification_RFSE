import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import plot_tree
import itertools



st.set_page_config(
    page_title="Klasifikasi Stroke dengan RFSE",
    page_icon='https://raw.githubusercontent.com/Amel039/Stroke-Classification/main/stroke.jpg',
    
    initial_sidebar_state="expanded",
    layout="wide"
   
)

st.write("""<h1 style="text-align: center;">ALGORITMA HYBRID SAMPLING DENGAN KOMBINASI SYNTHETIC MINORITY OVERSAMPLING TECHNIQUE EDITED NEAREST NEIGHBOR PADA KLASIFIKASI RANDOM FOREST UNTUK KETIDAKSEIMBANGAN DATA STROKE </h1><br>""", unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
            st.write("""<h3 style="text-align: center;"><img src="https://raw.githubusercontent.com/Amel039/Stroke-Classification/main/stroke.jpg" width="90" height="90"><br> AMELIA NUR SEPTIYASARI <p>200411100039</p></h3>""", unsafe_allow_html=True),
            ["Home", "Description", "Dataset", "Prepocessing" ,"Modeling", "Implementation"],
            icons=['house', 'file-earmark-font', 'bar-chart', 'gear', 'arrow-down-square', 'check2-square'],
            menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#6495ED"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "white"},
                "nav-link-selected": {"background-color": "#00008B"}
            }
        )
        st.write("""
        <div style = "position: fixed; left:40px; bottom: 10px;">
            <center><a href=""><span><img src="https://cdns.iconmonstr.com/wp-content/releases/preview/2012/240/iconmonstr-github-1.png" width="40px" height="40px"></span></a><a style = "margin-left: 20px;" href="http://hanifsantoso05.github.io/datamining/intro.html"><span><img src="https://friconix.com/png/fi-stluxx-jupyter-notebook.png" width="40px" height="40px"></span></a> <a style = "margin-left: 20px;" href="mailto: mellnur2901@gmail.com"><span><img src="https://cdn-icons-png.flaticon.com/512/60/60543.png" width="40px" height="40px"></span></a></center>
        </div> 
        """, unsafe_allow_html=True)

    if selected == "Home":
        st.write("""<div style="text-align: center;">
        Stroke merupakan permasalahan kesehatan global yang signifikan dan terus berkembang. Secara global, stroke menjadi penyebab utama kecacatan fisik yang dialami oleh orang dewasa, dan menempati peringkat kedua sebagai penyebab kematian di negara-negara dengan tingkat pendapatan menengah hingga tinggi.
        Diperlukan pendekatan untuk deteksi dini stroke. Untuk membantu penderita dan masyarakat umum diperlukan suatu model yang dapat memprediksi penyakit stroke agar dapat terhindar dari stroke. Salah satunya dengan menggunakan suatu model prediksi penyakit stroke. Data mining dapat dimanfaatkan untuk mendiagnosis penyakit. 
        Dalam penerapannya teknik data mining dapat digunakan untuk mengklasifikasikan atau memprediksi suatu penyakit berdasarkan data medis yang ada.<br>
      </div>
      """, unsafe_allow_html=True)


        st.write("""
                <div style="display: flex; justify-content: center;">
                    <div style="text-align: center;">
                        <h3>Dosen Pembimbing 1: Achmad Jauhari, ST.,M.Kom</h3>
                        <img src="https://raw.githubusercontent.com/Amel039/Stroke-Classification/main/pakjau.jpeg" width="200" height="200">
                    </div>
                    <div style="text-align: center;">
                        <h3>Dosen Pembimbing 2: Fifin Ayu Mufarroha, S.Kom., M.Kom</h3><br>
                        <img src="https://raw.githubusercontent.com/Amel039/Stroke-Classification/main/bufifin.jpeg" width="200" height="200">
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.write("""<br>
        <div style="display: flex; justify-content: center;">
            <div style="text-align: center;">
                <h3>Dosen Penguji 1: Dr. Cucun Very Angkoso, S.T., MT</h3><br>
                <img src="https://raw.githubusercontent.com/Amel039/Stroke-Classification/main/bapakcucun.jpg" width="200" height="200">
            </div>
            <div style="text-align: center;">
                <h3>Dosen Penguji 2: Arik Kurniawati, S.Kom., M.T.</h3><br>
                <img src="https://raw.githubusercontent.com/Amel039/Stroke-Classification/main/ibuarik.jpg" width="200" height="200">
            </div>
            <div style="text-align: center;">
                <h3>Dosen Penguji 3: Yudha Dwi Putra Negara, S.Kom., M.Kom</h3>
                <img src="https://raw.githubusercontent.com/Amel039/Stroke-Classification/main/pakyudha.jpeg" width="200" height="200">
            </div>
        </div>
        """, unsafe_allow_html=True)

        
    elif selected == "Description":
        st.subheader("""Pengertian""")
        st.write(""" 
        Dataset yang digunakan pada penelitian ini merupakan data Prediksi Stroke. Dataset ini berjumlah 5110 data dengan 11 fitur dan 1 label berisi 2 kelas yakni 1 (stroke) dan 0 (tidak stroke). Dataset ini berisi jumlah diagnosis stroke 249 orang dan tidak stroke sebanyak 4861 orang. Dan di dalam dataset terdapat nilai kosong (missing value) sebanyak 201 data.
Dataset stroke ini berasal dari penelitian thesis Oluwafemi Emmanuel Zachariah tentang “Prediksi Penyakit Stroke dengan Data Demografis Dan Perilaku Menggunakan Algoritma Random Forest” dari Sheffield Hallam University yang diambil dari catatan kesehatan dari berbagai rumah sakit di Bangladesh yang dapat diakses melalui link berikut  ini https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/.
        </div>""")
        

        st.subheader("""Kegunaan Dataset""")
        st.write("""
        Data ini akan digunakan untuk melakukan prediksi atau klasifikasi penderita Stroke </div>
        """)

        st.subheader("""Fitur""")
        st.markdown(
            """
            Fitur-fitur yang terdapat pada dataset:
            -   ID	Merupakan id unik tiap pasien bertipe Numerik
            -   Gender	Merupakan fitur jenis kelamin pasien. berisi dari kategori yaitu female (perempuan), male (laki-laki)  dan other (lainnya) bertipe	Kategorik
            -   Age	Merupakan fitur umur pasien	Numerik
            -   Hypertension,	Merupakan fitur yang berisi apakah pasien tersebut memiliki penyakit jantung atau tidak. Berisi kategori 0 dan kategori 1 bertipe Kategorik
            -   heart_disease	Fitur  apakah pasien tersebut memiliki penyakit hipertensi (1) atau tidak (0) bertipe Kategorik
            -   Ever_married	Fitur berisi apakah pasien sudah pernah menikah atau tidak, terdiri dari 2 kategori yaitu kategori Yes dan No bertipe Kategorik
            -   Work_type	Fitur tipe pekerjaan berisi 5 fitur yakni Private (pribadi) , self- employed (bekerja sendiri), children (anak-anak) , Govt_job (pekerjaan pemerintahan) , Never_worked (tidak pernah bekerja) bertipeKategorik
            -   Residence_type	Fitur jenis tempat tinggal berisi berisi 2 kategori Urban (Perkotaan) dan Rural (Pedesaan) bertipe	Kategorik
            -   avg_glucose_level	Fitur berisi rata rata tingkat kadar glukosa dalam darah pada pasien bertipe Numerik 
            -   bmi	Fitur bmi (body mass index) pasien bertipe Numerik
            -   smoking_status	fitur 	berisi status merokok pasien, Never smoke (tidak pernah merokok), Unknown (tidak diketahui), formery smoked (sebelumnya merokok), smoked (merokok) bertipe Kategorik
            -   stroke	Fitur berisi label diagnosis stroke berisi 2 kategori 1 (stroke) orang dan 0 (tidak) bertipe Kategorik

            """
        )

        st.subheader("""Sumber Dataset""")
        st.write("""
        Sumber data di dapatkan melalui website kaggle.com, Berikut merupakan link untuk mengakses sumber dataset.
        <br>
        <a href="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/"> Kunjungi Sumber Data di Kaggle</a>
        <br>
        <a href="https://raw.githubusercontent.com/Amel039/Stroke-Classification/main/healthcare-dataset-stroke-data.csv"> Lihat Dataset Stroke</a>""", unsafe_allow_html=True)
      
        st.subheader("""Tipe Data""")
        st.write("""
        Tipe data yang di gunakan pada dataset ini adalah numerik dan kategorikal.
        """)

    elif selected == "Dataset":
        # Memuat data
        df = pd.read_csv('https://raw.githubusercontent.com/Amel039/Stroke-Classification/main/healthcare-dataset-stroke-data.csv')

        # Menampilkan header dan dataset
        st.header("Exploratory Data Analysis (EDA) Data Stroke")
        st.subheader("Dataset Stroke")
        st.dataframe(df, width=1000)

        # Tombol untuk menampilkan deskripsi dataset
        if st.button('Tampilkan Deskripsi Dataset Stroke'):
            des = df.describe()
            st.subheader("Deskripsi Dataset Stroke")
            st.dataframe(des, width=800)

        # Tombol untuk menampilkan tipe dataset
        if st.button('Tampilkan Tipe Dataset Stroke'):
            dt = df.dtypes.reset_index()
            dt.columns = ['Fitur', 'Tipe Data']
            st.subheader("Tipe Dataset Stroke")
            st.dataframe(dt, width=800)
        
        if st.button('Tampilkan Korelasi Dataset Stroke'):
            df_encode = pd.get_dummies(df, columns=['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
            df_en = pd.DataFrame(df_encode, columns=df_encode.columns)
            st.subheader("""Cek Korelasi Fitur dalam Data""")
            corr=df_en.corr()
            # Tampilkan heatmap korelasi
            plt.figure(figsize=(16, 12))
            # sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f",
            sns.heatmap(corr, annot=True, cmap='plasma', linewidths=0.5, fmt=".2f")
            heatmap_fig = plt.gcf()  # Mengambil objek gambar dari heatmap
            st.pyplot(heatmap_fig)

        # Tombol untuk menampilkan jumlah missing values
        if st.button('Melihat Missing Value dalam Dataset Stroke'):
            mis = df.isnull().sum().reset_index()
            mis.columns = ['Fitur', 'Jumlah Missing Values']
            st.subheader("Cek Missing Value dalam Dataset Stroke")
            st.dataframe(mis, width=800)

        st.subheader("Cek Persebaran dalam Dataset Stroke")
        selected_features = st.multiselect('Pilih fitur untuk dianalisis', df.columns.tolist(), default=[])
        if st.button('Cek Persebaran data dalam Dataset Stroke'):
            if selected_features:
                for selected_feature in selected_features:
                    if selected_feature in ['age', 'bmi', 'avg_glucose_level']:
                            # Plot box plot for the selected feature
                            plt.figure(figsize=(8, 6))
                            sns.boxplot(x=df[selected_feature])
                            plt.xlabel(selected_feature)
                            plt.title('Box Plot untuk fitur {}'.format(selected_feature))
                            st.pyplot(plt)
                    else:
                            # Plot count plot for the selected feature
                            plt.figure(figsize=(6, 6))
                            sns.countplot(x=df[selected_feature])
                            plt.xlabel(selected_feature)
                            plt.title('Count Plot untuk fitur {}'.format(selected_feature))
                            st.pyplot(plt)
            else:
                st.write("Silakan pilih setidaknya satu fitur untuk dianalisis.")

        
        # Menampilkan persebaran data dari fitur yang dipilih
        # Pilih fitur untuk dianalisis
       # Pilih fitur untuk dianalisis
     
# Menampilkan persebaran data dari fitur yang dipilih
# Pilih fitur untuk dianalisis
       
    elif selected == "Prepocessing":
        df = pd.read_csv('https://raw.githubusercontent.com/Amel039/Stroke-Classification/main/healthcare-dataset-stroke-data.csv')
        st.subheader("""Data Selection""")
        df = df.drop('id', axis=1)
        # Membersihkan nilai di kolom 'gender' dari spasi atau karakter tambahan
        df['gender'] = df['gender'].str.strip()
        # Menggunakan metode drop untuk menghapus baris dengan nilai 'gender' sama dengan 'Other'
        df = df.drop(df[df['gender'] == 'Other'].index)
        st.subheader('Hasil Penghapusan Kolom ID dan gender=other')
        st.dataframe(df, width=600)
        
        st.subheader("""One Hot Encoding Data""")
        st.write("""Langkah ini dilakukan dengan one hot Encoding untuk mengubah fitur dengan nilai kategorikal menjadi biner 1 dan 0. Contohnya misal fitur gender ada 2 yakni Male dan female maka fitur gender dipecah menjadi 2 kolom fitur. Berikut merupakan hasil one hot Encoding dari sampel dataset yang digunakan. :""")
        original_columns = df.columns.tolist()
        df_encode = pd.get_dummies(df, columns=['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
        df = pd.DataFrame(df_encode, columns=df_encode.columns)
        st.dataframe(df)
        df.to_csv("datas_encoded.csv")
        data_clean_nans = df.dropna()
        data_clean_nans.to_csv("data_clean_nans.csv")
        # Imputasi Missing Values dengan KNN (k=5)
        st.subheader("""Imputasi Data""")
        imputer = KNNImputer(n_neighbors=5)
        X = df.drop('stroke', axis=1)
        y = df['stroke']
        df_imputed = imputer.fit_transform(X)
        df_imputed = pd.DataFrame(df_imputed, columns=X.columns)
        df_imputed['stroke'] = y.values
        st.dataframe(df_imputed)
        df_imputed.to_csv("datas_imputed.csv")

       # Menghapus outlier dari kolom BMI
        q1_bmi = df_imputed['bmi'].quantile(0.25)
        q3_bmi = df_imputed['bmi'].quantile(0.75)
        iqr_bmi = q3_bmi - q1_bmi
        upper_bmi = q3_bmi + 1.5 * iqr_bmi
        lower_bmi = q1_bmi - 1.5 * iqr_bmi
        st.write(lower_bmi)
        st.write(upper_bmi)
        outliers_bmi = df_imputed[(df_imputed['bmi'] < lower_bmi) | (df_imputed['bmi'] > upper_bmi)]
        st.write('Outlier Bmi')
        st.dataframe(outliers_bmi)
        # Menghapus outlier BMI dari DataFrame
        cleaned_df = df_imputed.drop(outliers_bmi.index, axis=0)

        # Menghitung outlier pada kolom avg_glucose_level setelah menghapus outlier pada BMI
        q1_avg_glucose = cleaned_df['avg_glucose_level'].quantile(0.25)
        q3_avg_glucose = cleaned_df['avg_glucose_level'].quantile(0.75)
        iqr_avg_glucose = q3_avg_glucose - q1_avg_glucose
        upper_avg_glucose = q3_avg_glucose + 1.5 * iqr_avg_glucose
        lower_avg_glucose = q1_avg_glucose - 1.5 * iqr_avg_glucose
        st.write(lower_avg_glucose)
        st.write(upper_avg_glucose)
        outliers_avg_glucose = cleaned_df[(cleaned_df['avg_glucose_level'] < lower_avg_glucose) | (cleaned_df['avg_glucose_level'] > upper_avg_glucose)]
        st.write('Outlier Average Glucose Level')
        st.dataframe(outliers_avg_glucose)
        # Menghapus outlier avg_glucose_level dari DataFrame
        cleaned_df = cleaned_df.drop(outliers_avg_glucose.index, axis=0)
        cleaned_df.to_csv('data_clean.csv', index=False)
        st.subheader('Hasil IQR')
        st.dataframe(cleaned_df)
        # Normalisasi
        columns_to_normalize = ['age', 'bmi', 'avg_glucose_level']
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(cleaned_df[columns_to_normalize])
        cleaned_df[columns_to_normalize] = scaled
        st.subheader('Hasil Normalisasi')
        st.dataframe(cleaned_df)
        cleaned_df.to_csv('data_cleaned.csv', index=False)
        # Simpan DataFrame setelah semua operasi pemrosesan
        # cleaned_df.to_csv('data_cleaned.csv', index=False)
        # # Menampilkan hasil imputasi
        # st.dataframe(df_imputed, width=600)
        k_values = [3, 5, 7]
        skenario_sampling=[]
        X = cleaned_df.drop('stroke', axis=1)
        y = cleaned_df['stroke']
        # Data asli sebelum sampling
        total_asli = cleaned_df.shape[0]
        jumlah_stroke_asli = (cleaned_df['stroke'] == 1).sum()
        jumlah_non_stroke_asli = (cleaned_df['stroke'] == 0).sum()
        skenario_sampling.append({'Skenario': 'Data asli','Jumlah Total':total_asli, 'Jumlah Stroke': jumlah_stroke_asli, 'Jumlah Tidak Stroke': jumlah_non_stroke_asli})


        st.subheader('Resampling Data')
        # Membuat subplot
        fig, axes = plt.subplots(2, len(k_values), figsize=(15, 10))
        fitur1 = st.selectbox("Pilih fitur 1:", X.columns.tolist())
        fitur2 = st.selectbox("Pilih fitur 2:", X.columns.tolist())
        for i, k in enumerate(k_values):
            # Sampling data dengan nilai k yang dipilih
            smote = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=42)
            X_resampled_smote, y_resampled_smote = smote.fit_resample(X, y)

            enn = EditedNearestNeighbours(n_neighbors=k)
            X_resampled_enn, y_resampled_enn = enn.fit_resample(X, y)

            smote_enn = SMOTEENN(sampling_strategy='auto', smote=smote, enn=enn, random_state=42)
            X_resampled_smote_enn, y_resampled_smote_enn = smote_enn.fit_resample(X, y)
            
            df_smote = pd.DataFrame(X_resampled_smote, columns=X.columns)
            df_smote['stroke'] = y_resampled_smote
            df_smote.to_csv(f'data_sampling_smote_k{k}.csv', index=False)

            df_enn = pd.DataFrame(X_resampled_enn, columns=X.columns)
            df_enn['stroke'] = y_resampled_enn
            df_enn.to_csv(f'data_sampling_enn_k{k}.csv', index=False)

            df_smote_enn = pd.DataFrame(X_resampled_smote_enn, columns=X.columns)
            df_smote_enn['stroke'] = y_resampled_smote_enn
            df_smote_enn.to_csv(f'data_sampling_smote_enn_k{k}.csv', index=False)
            
            sns.scatterplot(x=fitur1, y=fitur2, hue=y, data=pd.concat([pd.DataFrame(X), pd.DataFrame({'stroke': y})], axis=1), ax=axes[0, i])
            axes[0, i].set_title(f'Data asli')
            
            # sns.scatterplot(x=fitur1, y=fitur2, hue=y_resampled_smote, data=pd.concat([pd.DataFrame(X_resampled_smote), pd.DataFrame({'stroke': y_resampled_smote})], axis=1), ax=axes[1, i])
            # axes[1, i].set_title(f'SMOTE (k={k})')

            # sns.scatterplot(x=fitur1, y=fitur2, hue=y_resampled_enn, data=pd.concat([pd.DataFrame(X_resampled_enn), pd.DataFrame({'stroke': y_resampled_enn})], axis=1), ax=axes[2, i])
            # axes[2, i].set_title(f'ENN (k={k})')

            sns.scatterplot(x=fitur1, y=fitur2, hue=y_resampled_smote_enn, data=pd.concat([pd.DataFrame(X_resampled_smote_enn), pd.DataFrame({'stroke': y_resampled_smote_enn})], axis=1), ax=axes[1, i])
            axes[1, i].set_title(f'SMOTE-ENN (k={k})')
           
            total_asli_sm = df_smote.shape[0]
            jumlah_stroke_sm = (df_smote['stroke'] == 1).sum()
            jumlah_non_stroke_sm = (df_smote['stroke'] == 0).sum()
            # jumlah_stroke_enn = (df_enn['stroke'] == 1).sum()
            # jumlah_non_stroke_enn = (df_enn['stroke'] == 0).sum()
            # st.write(jumlah_non_stroke)
            # st.write(jumlah_stroke)
            
            total_asli_smenn = df_smote_enn.shape[0]
            jumlah_stroke = (df_smote_enn['stroke'] == 1).sum()
            jumlah_non_stroke = (df_smote_enn['stroke'] == 0).sum()
            skenario_sampling.append({'Skenario':  f'SMOTE (k={k})','Jumlah Total':total_asli_sm, 'Jumlah Stroke': jumlah_stroke_sm, 'Jumlah Tidak Stroke': jumlah_non_stroke_sm})
            # skenario_sampling.append({'Skenario':  f'ENN (k={k})', 'Jumlah Stroke': jumlah_stroke_enn, 'Jumlah Tidak Stroke': jumlah_non_stroke_enn})
            skenario_sampling.append({'Skenario':  f'SMOTE-ENN (k={k})','Jumlah Total':total_asli_smenn, 'Jumlah Stroke': jumlah_stroke, 'Jumlah Tidak Stroke': jumlah_non_stroke})
            
            # Tampilkan hasil samping
        sken = pd.DataFrame(skenario_sampling)
        st.write("Ringkasan Hasil Sampling SMOTE-ENN")
        st.dataframe(sken) 
        plt.tight_layout()
        st.pyplot(fig)
                
    elif selected == "Modeling":
        with st.form("modeling"):
            st.subheader('Modeling')
            st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
            RF = st.checkbox('Random Forest')
            RFS = st.checkbox('Random Forest SMOTE ')
            RFSE = st.checkbox('Random Forest Smote ENN')
            submitted = st.form_submit_button("Submit")
            if submitted:
                if RF:
                    data= pd.read_csv('data_clean_nans.csv')
                    data=data.drop('Unnamed: 0',axis=1)
                    X= data.drop('stroke',axis=1)
                    y= data['stroke']
                    n_test=0.2
                    train= (1-n_test)*100
                    test= n_test*100
                    st.subheader(f'Hasil Pengujian data testing {test}% ')
                    # X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=n_test,random_state=42)
                    # training= pd.concat([X_train,y_train],axis=1)
                    # training.to_csv(f'data_train_{train}_RF.csv', index=False)
                    # testing= pd.concat([X_test,y_test],axis=1)
                    # testing.to_csv(f'data_test_{test}_RF.csv', index=False)
                    training= pd.read_csv(f'data_train_{train}_RF.csv')
                    testing= pd.read_csv(f'data_test_{test}_RF.csv')
                    n_estimators_values = [5,15]
                    max_depth_values = [4, 5, None]
                    criterion_values = ['entropy']
                    
                    # untuk menyimpan label setiap kombinasi
                    train= (1-n_test)*100
                    test= n_test*100
                    # X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=n_test,random_state=42)
                    # training= pd.concat([X_train,y_train],axis=1)
                    
                    # training.to_csv(f'data_train_{train}_RF.csv', index=False)
                    # testing= pd.concat([X_test,y_test],axis=1)
                    # testing.to_csv(f'data_test_{test}_RF.csv', index=False)
                    
                    train_data = pd.read_csv(f'data_train_{train}_RF.csv', index_col=0)
                    
                    X_train = train_data.drop(columns=['stroke'])
                    feature=X_train
                    y_train = train_data['stroke']
                    test_data= pd.read_csv(f'data_test_{test}_RF.csv',index_col=0)
                    X_test = test_data.drop(columns=['stroke'])
                    y_test= test_data['stroke']
                    st.write(f'Jumlah data training Random Forest : {len(train_data)}')
                    st.write(f'Jumlah data test Random Forest : {len(test_data)}')
                    # Loop melalui semua kombinasi parameter
                    # Inisialisasi list untuk menyimpan hasil evaluasi
                    results = {
                        'Parameter': [],
                        'Accuracy': [],
                        'TP': [],
                        'FP': [],
                        'TN': [],
                        'FN': []
                    }
                    # Loop melalui semua kombinasi parameter
                    for est, depth, crit in itertools.product(n_estimators_values, max_depth_values, criterion_values):
                        # Membuat model dengan parameter yang dipilih
                        RF = RandomForestClassifier(n_estimators=est, max_depth=depth, criterion=crit, random_state=42)
                        # model_name = f"RF_n_estimators={est}_max_depth={depth}_criterion={crit}.joblib"
                        # RF = joblib.load(model_name)
                        # Melakukan prediksi
                        
                        # Melakukan prediksi
                        RF.fit(X_train, y_train)
                        # model_name = f"RF_n_estimators={est}_max_depth={depth}_criterion={crit}.joblib"
                        # joblib.dump(RF, model_name)
                        class_names = ["Tidak Stroke", "Stroke"]
                        feature_names = X_train.columns.tolist()
                        # Membuat plot pohon keputusan
                        
                        plt.title(f"Decision Tree Random Forest RF n_estimators={est}_max_depth={depth}_criterion={crit}", fontsize=16)

                        # Menampilkan gambar pohon keputusan dalam Streamlit
                                
                        # # Melakukan prediksi
                        RF_pred = RF.predict(X_test)
                        # Menghitung metrik evaluasi
                        
                        precision = precision_score(y_test, RF_pred)
                        recall = recall_score(y_test, RF_pred)
                        f1 = f1_score(y_test, RF_pred)
                        cm = confusion_matrix(y_test, RF_pred)
                        TP = cm[1, 1]
                        FP = cm[0, 1]
                        TN = cm[0, 0]
                        FN = cm[1, 0]
                        accuracy = round((TP + TN) / (TP + TN + FP + FN),2)
                        df_result = pd.DataFrame({'y_test': y_test, 'RF_pred': RF_pred})
    
                        # Tampilkan hasil
                        st.subheader(f"DataFrame hasil prediksi untuk model: n_estimators={est}, max_depth={depth}, criterion={crit}")
                        st.dataframe(df_result)
                        # Tampilkan hasil
                        st.markdown(f"""
                        <h3>Confusion matrix hasil prediksi RFSE untuk model:</h3>
                        <p>n_estimators={est}, max_depth={depth}, criterion={crit}</p>
                        <br><br>
                        """, unsafe_allow_html=True)
                        
                        fig, ax = plt.subplots(figsize=(25, 15), dpi=200)  # Ukuran gambar lebih besar
                        plot_tree(
                            RF.estimators_[0], 
                            filled=True, 
                            feature_names=feature_names,  # Menggunakan nama fitur asli dari X_train
                            class_names=class_names,  # Menyertakan nama kelas
                            ax=ax , # Ukuran font lebih besar
                            fontsize=10,
                            
                        )
                        
                        # Menampilkan gambar pohon keputusan dalam Streamlit
                        # st.pyplot(fig)
                        # plt.close(fig) 
                        # Menyimpan hasil evaluasi
                        
                        results['Parameter'].append(f"n_estimators={est}, max_depth={depth}, criterion={crit}")
                        results['Accuracy'].append(accuracy)
                        # results['Precision'].append(precision)
                        # results['Recall'].append(recall)
                        # results['F1 Score'].append(f1)
                        results['TP'].append(TP)
                        results['FP'].append(FP)
                        results['TN'].append(TN)
                        results['FN'].append(FN)
                    # Membuat DataFrame dari hasil evaluasi
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    # Menampilkan parameter terbaik
                    best_idx = results_df['Accuracy'].idxmax()
                    best_params = results_df.loc[best_idx, 'Parameter']
                    st.subheader("Parameter Terbaik:")
                    st.write(best_params)

                    # Menampilkan grafik untuk setiap metrik dan confusion matrix
                    st.subheader('Hasil Evaluasi Tiap Kombinasi Parameter')

                    parameter_values = results_df['Parameter']
                    # accuracy_values = results_df['Accuracy'].round(4)
                    # precision_values = results_df['Precision'].round(4)
                    # recall_values = results_df['Recall'].round(4)
                    # f1_values = results_df['F1 Score'].round(4)
                    # Membuat overlapped bar chart untuk setiap metrik
                    # results_df['Accuracy'] = results_df['Accuracy'].round(4)
                    def plot_accuracy_graph(results_df):
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        ax.plot(results_df['Parameter'], results_df['Accuracy'], marker='o', linestyle='-', color='b')
                        ax.set_ylabel('Accuracy', fontsize=14)
                        ax.set_xlabel('Parameter', fontsize=14)
                        ax.tick_params(axis='x', rotation=90)
                        ax.grid(True)
                        
                        # Menambahkan nilai pada tiap titik
                        for i, nilai in enumerate(results_df['Accuracy']):
                            ax.text(i, nilai, f'{nilai:.2f}', color='black', ha='center', va='bottom', fontsize=12)
                        
                        plt.tight_layout()
                        st.pyplot(fig)

                    # Gunakan fungsi di atas dengan melewatkan parameter yang sesuai
                    plot_accuracy_graph(results_df)

                
                elif RFS:
    
                    df_imputed=pd.read_csv('data_cleaned.csv')
                    # st.dataframe(df_imputed)
                    k_sampling = st.number_input("Masukkan k untuk Sampling Data SMOTE :", min_value=3, max_value=7, value=5, step=2, key='k_smp')
                    
                    st.write(f'Penanganan Sampling SMOTE  data dengan k-{k_sampling}')
                    df_smote=pd.read_csv(f'data_sampling_smote_k{k_sampling}.csv')
                    st.dataframe(df_smote)
                    X_resampled_smote= df_smote.drop('stroke',axis=1)
                    y_resampled_smote = df_smote['stroke']  
                    
                   
                    clf = DecisionTreeClassifier(random_state=42,criterion='entropy')
                    clf.fit(X_resampled_smote, y_resampled_smote)

                    # Get the feature importances (which include the gain ratio)
                    importances = clf.feature_importances_

                    gain_ratios = {}
                    # Hitung gain ratio untuk setiap fitur
                    for i, feature in enumerate(X_resampled_smote.columns):
                        gain = importances[i]
                        split = clf.tree_.impurity[i]
                        
                        # Periksa apakah nilai impurity sangat kecil atau nol
                        if split < 1e-7:
                            # Jika nilai impurity sangat kecil, set gain ratio ke  0)
                            gain_ratio = 0 #jika impurity lebih kecil dari 0.0000001, maka gain_ratio akan diatur ke 0.
                        else:
                            # Hitung gain ratio
                            gain_ratio = gain / split
                            
                        gain_ratios[feature] = gain_ratio
                    
                    # st.write(X_resampled_smote_enn.columns)
                    #Membuat DataFrame untuk menampilkan impurity untuk setiap fitur
                
                    # Mengurutkan gain ratios secara menurun
                    sorted_gain_ratios = sorted(gain_ratios.items(), key=lambda x: x[1], reverse=True)
                    st.subheader('Hasil Seleksi Fitur Keseluruhan')
                    st.dataframe(sorted_gain_ratios)
                    # Mengambil fitur-fitur dengan gain ratio lebih dari 0
                    selected_features = [feature for feature, gain_ratio in sorted_gain_ratios if gain_ratio > 0]
                    st.write(f'Hasil Seleksi Fitur dengan k sampling-{k_sampling}')
                    st.dataframe(selected_features)
                    # Mengambil data baru menggunakan fitur-fitur terpilih
                    X_new = X_resampled_smote[selected_features]

                    # Menampilkan DataFrame yang berisi data baru
                    st.subheader("Data dengan fitur-fitur terpilih:")
                    st.dataframe(X_new, width=600)

                    # DataFrame untuk variabel target
                    y_resampled_smote_df = pd.DataFrame(y_resampled_smote, columns=['stroke'])
                                            # Gabungkan variabel target dengan X_new
                    new_data = pd.concat([X_new, y_resampled_smote_df], axis=1)
                    n_fitur = len(selected_features)
                    # data=new_data.to_csv(f'data_sampling_smote_{k_sampling}_{n_fitur}_fitur.csv', index=False)
                    X_fitur=pd.read_csv(f'data_sampling_smote_{k_sampling}_{n_fitur}_fitur.csv')
                    X_new = new_data.drop('stroke',axis=1)
                    st.subheader(f'Data berdasarkan {n_fitur} Fitur terbaik')
                            
                    st.dataframe(new_data,width=600)
                    st.session_state.selected_features = X_new.columns
                   
                    threshold=0.5
                    n_estimators_values = [5,15]
                    max_depth_values = [4, 5, None]
                    criterion_values = ['entropy']
                    n_test=0.2
                      # untuk menyimpan akurasi dari setiap kombinasi
                    # untuk menyimpan label setiap kombinasi
                    train= (1-n_test)*100
                    test= n_test*100
                   
                    # data= pd.read_csv(f'data_sampling_smote_{k_sampling}_{n_fitur}_fitur.csv')
                    # X= data.drop('stroke',axis=1)
                    # y=data['stroke']
                    # X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=n_test,random_state=42)
                    # training= pd.concat([X_train,y_train],axis=1)
                    # st.dataframe(y_train)
                    # training.to_csv(f'data_train_{train}_RFS_k{k_sampling}.csv', index=False)
                    # testing= pd.concat([X_test,y_test],axis=1)
                    # testing.to_csv(f'data_test_{test}_RFS_k{k_sampling}.csv', index=False)
                    
                    train_data = pd.read_csv(f'data_train_{train}_RFS_k{k_sampling}.csv')
                    st.dataframe(train_data)
                    X_train = train_data.drop(columns=['stroke'])
                    y_train = train_data['stroke']
                    test_data= pd.read_csv(f'data_test_{test}_RFS_k{k_sampling}.csv')
                    X_test = test_data.drop(columns=['stroke'])
                    y_test= test_data['stroke']
                    st.write(f'Jumlah data training Random SMOTE  : {len(train_data)}')
                    st.write(f'Jumlah data test Random Forest SMOTE  : {len(test_data)}')
                    
                    # Loop melalui semua kombinasi parameter
                    # Inisialisasi list untuk menyimpan hasil evaluasi
                    results = {
                        'Parameter': [],
                        'Accuracy': [],
                        'TP': [],
                        'FP': [],
                        'TN': [],
                        'FN': []
                    }
                    # Loop melalui semua kombinasi parameter
                    
                    # Loop melalui semua kombinasi parameter
                    # Loop melalui semua kombinasi parameter
                    for est, depth, crit in itertools.product(n_estimators_values, max_depth_values, criterion_values):
                        # Membuat model dengan parameter yang dipilih
                        RF = RandomForestClassifier(n_estimators=est, max_depth=depth, criterion=crit, random_state=42)
                        model_filename = f"random_forest_smote_{k_sampling}_n_estimator={est}_max_depth={depth}_criterion={crit}.joblib"
                        RF = joblib.load(model_filename)
                        # Melakukan prediksi
                        RF.fit(X_train, y_train)
                        
                        # Melakukan prediksi
                        class_names = ["Tidak Stroke", "Stroke"]
                        feature_names = X_train.columns.tolist()
                        # Membuat plot pohon keputusan
                        fig, ax = plt.subplots(figsize=(25, 15), dpi=200)  # Ukuran gambar lebih besar
                        plot_tree(
                            RF.estimators_[0], 
                            filled=True, 
                            feature_names=feature_names,  # Menggunakan nama fitur asli dari X_train
                            class_names=class_names,  # Menyertakan nama kelas
                            ax=ax , # Ukuran font lebih besar
                            fontsize=11,
                            
                        )
                        
                        plt.title(F"Decision Tree RFS_k_{k_sampling}_n_estimators={est}_max_depth={depth}_criterion={crit}", fontsize=16)
                       
                        # Menampilkan gambar pohon keputusan dalam Streamlit
                        plt.tight_layout()

                        # Menampilkan gambar pohon keputusan dalam Streamlit
                        st.pyplot(fig, use_container_width=True)

                        # # Menutup plot untuk menghindari gambar ganda
                        plt.close(fig)
                        RF_pred = RF.predict(X_test)

                        # Menghitung metrik evaluasi
                        accuracy = round(accuracy_score(y_test, RF_pred),2)
                        precision = precision_score(y_test, RF_pred)
                        recall = recall_score(y_test, RF_pred)
                        f1 = f1_score(y_test, RF_pred)
                        cm = confusion_matrix(y_test, RF_pred)
                        df_result = pd.DataFrame({'y_test': y_test, 'RF_pred': RF_pred}, index=X_test.index)
    
                        # Tampilkan hasil
                        st.markdown(f"""
                        <h3>Confusion matrix hasil prediksi RFS untuk model:</h3>
                        <p>n_estimators={est}, max_depth={depth}, criterion={crit}</p>
                        <br><br>
                        """, unsafe_allow_html=True)
                        
                        # st.dataframe(df_result)
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Tidak Stroke', 'Stroke'], yticklabels=['Tidak Stroke', 'Stroke'], ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        # Menyesuaikan tata letak plot
                        fig.tight_layout()

                        # Menampilkan plot menggunakan Streamlit
                        # st.pyplot(fig)
                        # Menampilkan plot menggunakan Streamlit
                       

                        # Ekstrak nilai TP, FP, TN, FN
                        TP = cm[1, 1]
                        FP = cm[0, 1]
                        TN = cm[0, 0]
                        FN = cm[1, 0]

                    
                        # st.write(f"True Positive (TP): {TP}")
                        # st.write(f"False Positive (FP): {FP}")
                        # st.write(f"True Negative (TN): {TN}")
                        # st.write(f"False Negative (FN): {FN}")
                        # Membuat gambar pohon keputusan
                        
                        # Menyimpan hasil evaluasi
                        results['Parameter'].append(f"n_estimators={est}, max_depth={depth}, criterion={crit}")
                        results['Accuracy'].append(accuracy)
                        # results['Precision'].append(precision)
                        # results['Recall'].append(recall)
                        # results['F1 Score'].append(f1)
                        results['TP'].append(TP)
                        results['FP'].append(FP)
                        results['TN'].append(TN)
                        results['FN'].append(FN)
                    
                    # Membuat DataFrame dari hasil evaluasi
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    # Menampilkan parameter terbaik
                    best_idx = results_df['Accuracy'].idxmax()
                    best_params = results_df.loc[best_idx, 'Parameter']
                    st.subheader("Parameter Terbaik:")
                    st.write(best_params)


                    # Menampilkan grafik untuk setiap metrik dan confusion matrix
                    st.subheader('Hasil Evaluasi Tiap Kombinasi Parameter')

                    # parameter_values = results_df['Parameter']
                    # accuracy_values = results_df['Accuracy'].round(4)
                    # precision_values = results_df['Precision'].round(4)
                    # recall_values = results_df['Recall'].round(4)
                    # f1_values = results_df['F1 Score'].round(4)
                 
                    
                    def plot_accuracy_graph(results_df):
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        ax.plot(results_df['Parameter'], results_df['Accuracy'], marker='o', linestyle='-', color='b')
                        ax.set_ylabel('Accuracy', fontsize=16)
                        ax.set_xlabel('Parameter', fontsize=16)
                        ax.tick_params(axis='x', rotation=90)
                        ax.grid(True)
                        
                        # Menambahkan nilai pada tiap titik
                        for i, nilai in enumerate(results_df['Accuracy']):
                            ax.text(i, nilai, f'{nilai:.2f}', color='black', ha='center', va='bottom', fontsize=16)
                        
                        plt.tight_layout()
                        return fig
                    
                    fig=plot_accuracy_graph(results_df)
                    st.pyplot(fig)
                    
                # elif RFE:
                #     k_sampling = st.number_input("Masukkan k untuk Sampling Data  ENN:", min_value=3, max_value=7, value=5, step=2, key='k_smp')
                    
                #     st.write(f'Penanganan Sampling  ENN data dengan k-{k_sampling}')
                #     df_enn=pd.read_csv(f'data_sampling_enn_k{k_sampling}.csv')
                #     st.dataframe(df_enn)
                #     X_resampled_enn= df_enn.drop('stroke',axis=1)
                #     y_resampled_enn = df_enn['stroke']  
                    
                   
                #     clf = DecisionTreeClassifier(random_state=42,criterion='entropy')
                #     clf.fit(X_resampled_enn, y_resampled_enn)

                #     # Get the feature importances (which include the gain ratio)
                #     importances = clf.feature_importances_

                #     gain_ratios = {}
                #     # Hitung gain ratio untuk setiap fitur
                #     for i, feature in enumerate(X_resampled_enn.columns):
                #         gain = importances[i]
                #         split = clf.tree_.impurity[i]
                        
                #         # Periksa apakah nilai impurity sangat kecil atau nol
                #         if split < 1e-7:
                #             # Jika nilai impurity sangat kecil, set gain ratio ke  0)
                #             gain_ratio = 0 #jika impurity lebih kecil dari 0.0000001, maka gain_ratio akan diatur ke 0.
                #         else:
                #             # Hitung gain ratio
                #             gain_ratio = gain / split
                            
                #         gain_ratios[feature] = gain_ratio
                    
                #     # st.write(X_resampled_smote_enn.columns)
                #     #Membuat DataFrame untuk menampilkan impurity untuk setiap fitur
                
                #     # Mengurutkan gain ratios secara menurun
                #     sorted_gain_ratios = sorted(gain_ratios.items(), key=lambda x: x[1], reverse=True)
                #     st.subheader('Hasil Seleksi Fitur Keseluruhan')
                #     st.dataframe(sorted_gain_ratios)
                #     # Mengambil fitur-fitur dengan gain ratio lebih dari 0
                #     selected_features = [feature for feature, gain_ratio in sorted_gain_ratios if gain_ratio > 0]
                #     st.write(f'Hasil Seleksi Fitur dengan k sampling-{k_sampling}')
                #     st.dataframe(selected_features)
                #     # Mengambil data baru menggunakan fitur-fitur terpilih
                #     X_new = X_resampled_enn[selected_features]

                #     # Menampilkan DataFrame yang berisi data baru
                #     st.subheader("Data dengan fitur-fitur terpilih:")
                #     st.dataframe(X_new, width=600)

                #     # DataFrame untuk variabel target
                #     y_resampled_smote_df = pd.DataFrame(y_resampled_enn, columns=['stroke'])
                #                             # Gabungkan variabel target dengan X_new
                #     new_data = pd.concat([X_new, y_resampled_smote_df], axis=1)
                #     n_fitur = len(selected_features)
                #     data=new_data.to_csv(f'data_sampling_enn_{k_sampling}_{n_fitur}_fitur.csv', index=False)
                #     X_fitur=pd.read_csv(f'data_sampling_enn_{k_sampling}_{n_fitur}_fitur.csv')
                #     X_new = new_data.drop('stroke',axis=1)
                #     st.subheader(f'Data berdasarkan {n_fitur} Fitur terbaik')
                            
                #     st.dataframe(new_data,width=600)
                #     st.session_state.selected_features = X_new.columns
                   
                #     threshold=0.5
                #     n_estimators_values = [5,10,15]
                #     max_depth_values = [4, 5, None]
                #     criterion_values = ['entropy']
                #     n_test=0.2
                #       # untuk menyimpan akurasi dari setiap kombinasi
                #     # untuk menyimpan label setiap kombinasi
                #     train= (1-n_test)*100
                #     test= n_test*100
                   
                #     # data= pd.read_csv(f'data_sampling_enn_{k_sampling}_{n_fitur}_fitur.csv')
                #     # X= data.drop('stroke',axis=1)
                #     # y=data['stroke']
                #     # X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=n_test,random_state=42)
                #     # training= pd.concat([X_train,y_train],axis=1)
                #     # st.dataframe(y_train)
                #     # training.to_csv(f'data_train_{train}_RFE_k{k_sampling}.csv', index=False)
                #     # testing= pd.concat([X_test,y_test],axis=1)
                #     # testing.to_csv(f'data_test_{test}_RFE_k{k_sampling}.csv', index=False)
                    
                #     train_data = pd.read_csv(f'data_train_{train}_RFE_k{k_sampling}.csv')
                #     st.dataframe(train_data)
                #     X_train = train_data.drop(columns=['stroke'])
                #     y_train = train_data['stroke']
                #     test_data= pd.read_csv(f'data_test_{test}_RFE_k{k_sampling}.csv')
                #     X_test = test_data.drop(columns=['stroke'])
                #     y_test= test_data['stroke']
                #     st.write(f'Jumlah data training Random ENN  : {len(train_data)}')
                #     st.write(f'Jumlah data test Random Forest ENN  : {len(test_data)}')
                    
                #     # Loop melalui semua kombinasi parameter
                #     # Inisialisasi list untuk menyimpan hasil evaluasi
                #     results = {
                #         'Parameter': [],
                #         'Accuracy': [],
                #         'TP': [],
                #         'FP': [],
                #         'TN': [],
                #         'FN': []
                #     }
                #     # Loop melalui semua kombinasi parameter
                    
                #     # Loop melalui semua kombinasi parameter
                #     # Loop melalui semua kombinasi parameter
                #     for est, depth, crit in itertools.product(n_estimators_values, max_depth_values, criterion_values):
                #         # Membuat model dengan parameter yang dipilih
                #         #model_name = f"RFSE_k_{k_sampling}_n_estimators={est}_max_depth={depth}_criterion={crit}.joblib"
                #         RF = RandomForestClassifier(n_estimators=est, max_depth=depth, criterion=crit, random_state=42)
                        
                #         # Melakukan prediksi
                #         RF.fit(X_train, y_train)
                        
                #         # Melakukan prediksi
                #         class_names = ["Tidak Stroke", "Stroke"]
                #         feature_names = X_train.columns.tolist()
                #         # Membuat plot pohon keputusan
                         
                #         # Menampilkan gambar pohon keputusan dalam Streamlit
                #         # st.pyplot(fig)
                #         RF_pred = RF.predict(X_test)

                #         # Menghitung metrik evaluasi
                #         accuracy = round(accuracy_score(y_test, RF_pred),2)
                #         precision = precision_score(y_test, RF_pred)
                #         recall = recall_score(y_test, RF_pred)
                #         f1 = f1_score(y_test, RF_pred)
                #         cm = confusion_matrix(y_test, RF_pred)
                #         df_result = pd.DataFrame({'y_test': y_test, 'RF_pred': RF_pred}, index=X_test.index)
    
                #         # Tampilkan hasil
                #         st.subheader(f"DataFrame hasil prediksi RFE untuk model: n_estimators={est}, max_depth={depth}, criterion={crit}")
                #         # st.dataframe(df_result)
                #         fig, ax = plt.subplots(figsize=(8, 6))
                #         sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Tidak Stroke', 'Stroke'], yticklabels=['Tidak Stroke', 'Stroke'], ax=ax)
                #         ax.set_xlabel('Predicted')
                #         ax.set_ylabel('Actual')
                #         ax.set_title('Confusion Matrix')

                #         # Menampilkan plot menggunakan Streamlit
                #         st.pyplot(fig)

                #         # Ekstrak nilai TP, FP, TN, FN
                #         TP = cm[1, 1]
                #         FP = cm[0, 1]
                #         TN = cm[0, 0]
                #         FN = cm[1, 0]

                    
                #         # st.write(f"True Positive (TP): {TP}")
                #         # st.write(f"False Positive (FP): {FP}")
                #         # st.write(f"True Negative (TN): {TN}")
                #         # st.write(f"False Negative (FN): {FN}")
                #         # Membuat gambar pohon keputusan
                        
                #         # Menyimpan hasil evaluasi
                #         results['Parameter'].append(f"n_estimators={est}, max_depth={depth}, criterion={crit}")
                #         results['Accuracy'].append(accuracy)
                #         # results['Precision'].append(precision)
                #         # results['Recall'].append(recall)
                #         # results['F1 Score'].append(f1)
                #         results['TP'].append(TP)
                #         results['FP'].append(FP)
                #         results['TN'].append(TN)
                #         results['FN'].append(FN)
                #     model_filename = f"random_forest_enn_{k_sampling}_n_estimator={est}_max_depth={depth}_criterion={crit}.joblib"
                #     joblib.dump(RF, model_filename)
                #     # Membuat DataFrame dari hasil evaluasi
                #     results_df = pd.DataFrame(results)
                #     st.dataframe(results_df)
                #     # Menampilkan parameter terbaik
                #     best_idx = results_df['Accuracy'].idxmax()
                #     best_params = results_df.loc[best_idx, 'Parameter']
                #     st.subheader("Parameter Terbaik:")
                #     st.write(best_params)



                #     # Menampilkan grafik untuk setiap metrik dan confusion matrix
                #     st.subheader('Hasil Evaluasi Tiap Kombinasi Parameter')

                #     # parameter_values = results_df['Parameter']
                #     # accuracy_values = results_df['Accuracy'].round(4)
                #     # precision_values = results_df['Precision'].round(4)
                #     # recall_values = results_df['Recall'].round(4)
                #     # f1_values = results_df['F1 Score'].round(4)
                 
                    
                #     def plot_accuracy_graph(results_df):
                #         fig, ax = plt.subplots(figsize=(12, 8))
                        
                #         ax.plot(results_df['Parameter'], results_df['Accuracy'], marker='o', linestyle='-', color='b')
                #         ax.set_ylabel('Accuracy', fontsize=14)
                #         ax.set_xlabel('Parameter', fontsize=14)
                #         ax.tick_params(axis='x', rotation=90)
                #         ax.grid(True)
                        
                #         # Menambahkan nilai pada tiap titik
                #         for i, nilai in enumerate(results_df['Accuracy']):
                #             ax.text(i, nilai, f'{nilai:.2f}', color='black', ha='center', va='bottom', fontsize=12)
                        
                #         plt.tight_layout()
                #         return fig
                    
                #     fig=plot_accuracy_graph(results_df)
                #     st.pyplot(fig)
                    
                elif RFSE:
    
                    df_imputed=pd.read_csv('data_cleaned.csv')
                    # st.dataframe(df_imputed)
                    k_sampling = st.number_input("Masukkan k untuk Sampling Data SMOTE ENN:", min_value=3, max_value=7, value=5, step=2, key='k_smp')
                    if k_sampling==3:
                            st.write(f'Penanganan Sampling SMOTE ENN data dengan k-{k_sampling}')
                            df_smote_enn=pd.read_csv('data_sampling_smote_enn_k3.csv')
                            st.dataframe(df_smote_enn)
                            X_resampled_smote_enn= df_smote_enn.drop('stroke',axis=1)
                            y_resampled_smote_enn = df_smote_enn['stroke']  
                    elif k_sampling==5:
                            st.write(f'Penanganan Sampling SMOTE ENN data dengan k-{k_sampling}')
                            df_smote_enn=pd.read_csv('data_sampling_smote_enn_k5.csv')
                            st.dataframe(df_smote_enn)
                            X_resampled_smote_enn= df_smote_enn.drop('stroke',axis=1)
                            y_resampled_smote_enn = df_smote_enn['stroke']
                    else:
                            st.write(f'Penanganan Sampling SMOTE ENN data dengan k-{k_sampling}')
                            df_smote_enn=pd.read_csv('data_sampling_smote_enn_k7.csv')
                            st.dataframe(df_smote_enn)
                            X_resampled_smote_enn= df_smote_enn.drop('stroke',axis=1)
                            y_resampled_smote_enn = df_smote_enn['stroke']
                   
                    clf = DecisionTreeClassifier(random_state=42,criterion='entropy')
                    clf.fit(X_resampled_smote_enn, y_resampled_smote_enn)

                    # Get the feature importances (which include the gain ratio)
                    importances = clf.feature_importances_

                    gain_ratios = {}
                    # Hitung gain ratio untuk setiap fitur
                    for i, feature in enumerate(X_resampled_smote_enn.columns):
                        gain = importances[i]
                        split = clf.tree_.impurity[i]
                        
                        # Periksa apakah nilai impurity sangat kecil atau nol
                        if split < 1e-7:
                            # Jika nilai impurity sangat kecil, set gain ratio ke  0)
                            gain_ratio = 0 #jika impurity lebih kecil dari 0.0000001, maka gain_ratio akan diatur ke 0.
                        else:
                            # Hitung gain ratio
                            gain_ratio = gain / split
                            
                        gain_ratios[feature] = gain_ratio
                    
                    # st.write(X_resampled_smote_enn.columns)
                    #Membuat DataFrame untuk menampilkan impurity untuk setiap fitur
                
                    # Mengurutkan gain ratios secara menurun
                    sorted_gain_ratios = sorted(gain_ratios.items(), key=lambda x: x[1], reverse=True)
                    st.subheader('Hasil Seleksi Fitur Keseluruhan')
                    st.dataframe(sorted_gain_ratios)
                    # Mengambil fitur-fitur dengan gain ratio lebih dari 0
                    selected_features = [feature for feature, gain_ratio in sorted_gain_ratios if gain_ratio > 0]
                    st.write(f'Hasil Seleksi Fitur dengan k sampling-{k_sampling}')
                    st.dataframe(selected_features)
                    # Mengambil data baru menggunakan fitur-fitur terpilih
                    X_new = X_resampled_smote_enn[selected_features]

                    # Menampilkan DataFrame yang berisi data baru
                    st.subheader("Data dengan fitur-fitur terpilih:")
                    st.dataframe(X_new, width=600)

                    # DataFrame untuk variabel target
                    y_resampled_smote_enn_df = pd.DataFrame(y_resampled_smote_enn, columns=['stroke'])
                                            # Gabungkan variabel target dengan X_new
                    new_data = pd.concat([X_new, y_resampled_smote_enn_df], axis=1)
                    n_fitur = len(selected_features)
                    data=new_data.to_csv(f'data_sampling_smote_enn{k_sampling}_{n_fitur}_fitur.csv', index=False)
                    #     # X_fitur=pd.read_csv(f'data_imputer{k_mis}_sampling_smote_enn{k_sampling}_{n_fitur}_fitur.csv')
                        # X_new = X_resampled_smote_enn[selected_features].drop('stroke',axis=1)
                    st.subheader(f'Data berdasarkan {n_fitur} Fitur terbaik')
                            
                    st.dataframe(new_data,width=600)
                    st.session_state.selected_features = X_new.columns
                   
                    n_estimators_values = [5,15]
                    max_depth_values = [4, 5, None]
                    criterion_values = ['entropy']
                    n_test=0.2
                      # untuk menyimpan akurasi dari setiap kombinasi
                    # untuk menyimpan label setiap kombinasi
                    train= (1-n_test)*100
                    test= n_test*100
                   
                    data= pd.read_csv(f'data_sampling_smote_enn{k_sampling}_{n_fitur}_fitur.csv')
                    # X= data.drop('stroke',axis=1)
                    # y=data['stroke']
                    # X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=n_test,random_state=42)
                    # training= pd.concat([X_train,y_train],axis=1)
                    # st.dataframe(y_train)
                    # training.to_csv(f'data_train_{train}_RFSE_k{k_sampling}.csv', index=False)
                    # testing= pd.concat([X_test,y_test],axis=1)
                    # testing.to_csv(f'data_test_{test}_RFSE_k{k_sampling}.csv', index=False)
                    
                    train_data = pd.read_csv(f'data_train_{train}_RFSE_k{k_sampling}.csv')
                    # st.dataframe(train_data)
                    X_train = train_data.drop(columns=['stroke'])
                    y_train = train_data['stroke']
                    test_data= pd.read_csv(f'data_test_{test}_RFSE_k{k_sampling}.csv')
                    X_test = test_data.drop(columns=['stroke'])
                    y_test= test_data['stroke']
                    st.write(f'Jumlah data training Random SMOTE ENN : {len(train_data)}')
                    st.write(f'Jumlah data test Random Forest SMOTE ENN : {len(test_data)}')
                    
                    # Loop melalui semua kombinasi parameter
                    # Inisialisasi list untuk menyimpan hasil evaluasi
                    results = {
                        'Parameter': [],
                        'Accuracy': [],
                        'TP': [],
                        'FP': [],
                        'TN': [],
                        'FN': []
                    }
                    # Loop melalui semua kombinasi parameter
                    
                    # Loop melalui semua kombinasi parameter
                    # Loop melalui semua kombinasi parameter
                    for est, depth, crit in itertools.product(n_estimators_values, max_depth_values, criterion_values):
                        # Membuat model dengan parameter yang dipilih
                        #model_name = f"RFSE_k_{k_sampling}_n_estimators={est}_max_depth={depth}_criterion={crit}.joblib"
                        model_name = f"random_forest_smoteenn_{k_sampling}_n_estimator={est}_max_depth={depth}_criterion={crit}.joblib"
                        RF = RandomForestClassifier(n_estimators=est, max_depth=depth, criterion=crit, random_state=42)
                        RF = joblib.load(model_name)
                        # Melakukan prediksi
                        RF.fit(X_train, y_train)
                        # joblib.dump(RF, model_name)
                        # Melakukan prediksi
                        class_names = ["Tidak Stroke", "Stroke"]
                        feature_names = X_train.columns.tolist()
                        # Membuat plot pohon keputusan
                        fig, ax = plt.subplots(figsize=(20, 15), dpi=200)  # Ukuran gambar lebih besar
                        plot_tree(
                            RF.estimators_[0], 
                            filled=True, 
                            feature_names=feature_names,  # Menggunakan nama fitur asli dari X_train
                            class_names=class_names,  # Menyertakan nama kelas
                            ax=ax , # Ukuran font lebih besar
                            fontsize=10,
                            
                        )
                        plt.title(F"Decision Tree RFSE_k_{k_sampling}_n_estimators={est}_max_depth={depth}_criterion={crit}", fontsize=16)

                        # Menampilkan gambar pohon keputusan dalam Streamlit
                        st.pyplot(fig)
                        RF_pred = RF.predict(X_test)

                        # Menghitung metrik evaluasi
                        accuracy = round(accuracy_score(y_test, RF_pred),2)
                        # precision = precision_score(y_test, RF_pred)
                        # recall = recall_score(y_test, RF_pred)
                        # f1 = f1_score(y_test, RF_pred)
                        cm = confusion_matrix(y_test, RF_pred)
                        df_result = pd.DataFrame({'y_test': y_test, 'RF_pred': RF_pred}, index=X_test.index)
    
                        # Tampilkan hasil
                        # Tampilkan hasil
                        st.markdown(f"""
                        <h3>Confusion matrix hasil prediksi RFSE untuk model:</h3>
                        <p>n_estimators={est}, max_depth={depth}, criterion={crit}</p>
                        <br><br>
                        """, unsafe_allow_html=True)
                        
                        # st.dataframe(df_result)
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Tidak Stroke', 'Stroke'], yticklabels=['Tidak Stroke', 'Stroke'], ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        # Menyesuaikan tata letak plot
                        fig.tight_layout()

                        # Menampilkan plot menggunakan Streamlit
                        # st.pyplot(fig)

                        # Ekstrak nilai TP, FP, TN, FN
                        TP = cm[1, 1]
                        FP = cm[0, 1]
                        TN = cm[0, 0]
                        FN = cm[1, 0]

                    
                        # st.write(f"True Positive (TP): {TP}")
                        # st.write(f"False Positive (FP): {FP}")
                        # st.write(f"True Negative (TN): {TN}")
                        # st.write(f"False Negative (FN): {FN}")
                        # Membuat gambar pohon keputusan
                        
                        # Menyimpan hasil evaluasi
                        results['Parameter'].append(f"n_estimators={est}, max_depth={depth}, criterion={crit}")
                        results['Accuracy'].append(accuracy)
                        # results['Precision'].append(precision)
                        # results['Recall'].append(recall)
                        # results['F1 Score'].append(f1)
                        results['TP'].append(TP)
                        results['FP'].append(FP)
                        results['TN'].append(TN)
                        results['FN'].append(FN)
                    
                    # Membuat DataFrame dari hasil evaluasi
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    # Menampilkan parameter terbaik
                    best_idx = results_df['Accuracy'].idxmax()
                    best_params = results_df.loc[best_idx, 'Parameter']
                    st.subheader("Parameter Terbaik:")
                    st.write(best_params)

                    # Menampilkan grafik untuk setiap metrik dan confusion matrix
                    st.subheader('Hasil Evaluasi Tiap Kombinasi Parameter')

                    # parameter_values = results_df['Parameter']
                    # accuracy_values = results_df['Accuracy'].round(4)
                    # precision_values = results_df['Precision'].round(4)
                    # recall_values = results_df['Recall'].round(4)
                    # f1_values = results_df['F1 Score'].round(4)
                    
                    # def plot_confusion_matrix(results_df):
                    #     fig, ax = plt.subplots(figsize=(12, 8))
                    #     bar_width = 0.2
                    #     index = np.arange(len(results_df['Parameter']))

                    #     rects1 = ax.bar(index, results_df['TP'], bar_width, label='True Positive', color='g')
                    #     rects2 = ax.bar(index + bar_width, results_df['FP'], bar_width, label='False Positive', color='r')
                    #     rects3 = ax.bar(index + 2 * bar_width, results_df['TN'], bar_width, label='True Negative', color='b')
                    #     rects4 = ax.bar(index + 3 * bar_width, results_df['FN'], bar_width, label='False Negative', color='y')

                    #     ax.set_xlabel('Parameters')
                    #     ax.set_ylabel('Count')
                    #     ax.set_title('Evaluation Metrics by Parameter Settings')
                    #     ax.set_xticks(index + 1.5 * bar_width)
                    #     ax.set_xticklabels(results_df['Parameter'], rotation=45, ha='right')
                    #     ax.legend()

                    #     plt.tight_layout()
                    #     return fig

                    # fig = plot_confusion_matrix(results_df)
                    # st.pyplot(fig)
                    
                    def plot_accuracy_graph(results_df):
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        ax.plot(results_df['Parameter'], results_df['Accuracy'], marker='o', linestyle='-', color='b')
                        ax.set_ylabel('Accuracy', fontsize=16)
                        ax.set_xlabel('Parameter', fontsize=16)
                        ax.tick_params(axis='x', rotation=90)
                        ax.grid(True)
                        
                        # Menambahkan nilai pada tiap titik
                        for i, nilai in enumerate(results_df['Accuracy']):
                            ax.text(i, nilai, f'{nilai:.2f}', color='black', ha='center', va='bottom', fontsize=16)
                        
                        plt.tight_layout()
                        return fig
                    
                    fig=plot_accuracy_graph(results_df)
                    st.pyplot(fig)
                    # Loop melalui semua kombinasi parameter
                #     for est, depth, crit in itertools.product(n_estimators_values, max_depth_values, criterion_values):
                #         # Buat model dengan parameter yang dipilih
                #         RF = RandomForestClassifier(n_estimators=est, max_depth=depth, criterion=crit, random_state=42)
                #         RF.fit(X_train, y_train)

                #         # Lakukan prediksi dan evaluasi model
                #         RF_pred = RF.predict(X_test)
                #         accuracy = accuracy_score(y_test, RF_pred)
                        
                #         # Tambahkan akurasi dan label ke dalam list
                #         accuracies.append(accuracy)
                #         labels.append(f"n_estimators={est}, max_depth={depth}, criterion={crit}")
                #     st.subheader(f'Hasil Pengujian data testing {test}% ,Parameter n_estimator={est}, max_depth={depth},criterion={crit}')
                #     # Buat grafik batang
                #     plt.figure(figsize=(10, 6))
                #     plt.bar(labels, accuracies, color='blue')
                #     plt.xlabel('Parameter Kombinasi')
                #     plt.ylabel('Akurasi')
                #     plt.title('Akurasi untuk Setiap Kombinasi Parameter')
                #     plt.xticks(rotation=90)  # agar label sumbu x tidak bertumpuk
                #     plt.tight_layout()        # agar layout lebih rapi
                #     plt.show()

                # #     # Menyimpan model
                # #     model_filename = f"random_forest_model_k_imp{k_mis}_ksampling{k_sampling}_{n_fitur}_fitur_n_estimator={est}_max_depth={depth}_criterion={crit}.joblib"
                # #     joblib.dump(RF, model_filename)
                #     prediksi = pd.DataFrame(np.array(RF_pred).reshape(-1),columns=['hasil prediksi y test'])
                   
                #     aktual=pd.DataFrame(np.array(y_test).reshape(-1),columns=['y test asli'])
                #     # st.write(len(RF_pred))
                #     # st.write(len(testing_with_features))
                #     # Menambahkan prediksi dan y_test asli ke DataFrame
                #     st.dataframe(X_test)
                #     st.write(len(X_test))
                #     # Menambahkan 
                #     # prediksi dan y_test asli ke DataFrame
                #     # Membuat DataFrame baru dengan hasil prediksi dan nilai aktual
                #     result_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True), pd.Series(RF_pred, name='Predicted')], axis=1)

                    
                #     st.subheader("Data Test dengan Prediksi dan Nilai Aktual")
                #     st.dataframe(result_df)

                #     # Menampilkan grafik
                #    # Buat plot
                #     plt.figure(figsize=(6,4))
                #     plt.scatter(y_test.index, y_test, color='blue', label='Actual')
                #     plt.scatter(y_test.index, RF_pred, color='red', label='Predicted')
                #     plt.title('Perbandingan Nilai Aktual dan Prediksi')
                #     plt.xlabel('Index Data Uji')
                #     plt.ylabel('Nilai')
                #     plt.legend()

                    # Simpan plot ke dalam variabel
                    # plot = plt.gcf()

                    # # Tampilkan plot menggunakan st.pyplot()
                    # st.pyplot(plot)
                    # for i in range(1,est+1):
                    #     estimator = RF.estimators_[i-1]
                    #     fig, ax = plt.subplots(figsize=(10, 8))
                    #     plot_tree(estimator, filled=True, feature_names=X_train.columns, class_names=["No Stroke", "Stroke"], ax=ax)
                    #     plt.title(f"Random Forest Trees Estimator {i}")  # Menggunakan nilai i+1 karena i dimulai dari 0
                    #     plt.savefig(f"random_forest_estimator_{i}.png", dpi=300)
                    #     plt.close(fig)
                    #     st.image(f"random_forest_estimator_{i}.png")
                    
                    
                    # for i in range(1, est + 1):
                    #     estimator = RF.estimators_[i - 1]
                    #     fig, ax = plt.subplots(figsize=(10, 8))
                    #     plot_tree(estimator, filled=True, feature_names=X_train.columns, class_names=["No Stroke", "Stroke"], ax=ax)
                    #     plt.title(f"Random Forest Trees Estimator {i}")  # Menggunakan nilai i+1 karena i dimulai dari 0
                    # fig, axes = plt.subplots(nrows=1, ncols=est, figsize=(20, 8))  # Mengatur ukuran gambar besar
                    # for i, estimator in enumerate(RF.estimators_):
                    #     plot_tree(estimator, filled=True, feature_names=X_train.columns, class_names=["No Stroke", "Stroke"], ax=axes[i])
                    #     axes[i].set_title(f"Tree {i+1}")

                    
                    # folder_name = "random_forest_plots"
                    # os.makedirs(folder_name, exist_ok=True)

                    # # Menyusun path lengkap untuk menyimpan gambar
                    # plot_path = os.path.join(folder_name, f"random_forest_all_estimators_test_{test}_k_mis_5_k_sampling_{k_sampling}_n_fitur_{n_fitur}.png")

                    # # Simpan gambar besar ke dalam folder baru
                    # plt.savefig(plot_path, dpi=300)

                    # st.image(plot_path)
                    # st.image(f"random_forest_estimator_{i}.png")  # Menampilkan gambar
                # # Simpan model ke dalam file dengan nama yang dinamis
                # model_filename = f"random_forest_model_k_imp5_ksampling{k_sampling}_{n_fitur}_fitur_n_estimator={est}_max_depth={depth}_criterion={crit}.joblib"
                # joblib.dump(RF, model_filename)

                    # Inisialisasi KFold
            
                #     # Melakukan KFold cross-validation
                # for fold, (train_idx, test_idx) in enumerate(kf.split(X_new, y_resampled_smote_enn), 1):
                #     X_train, X_test = X_new.iloc[train_idx], X_new.iloc[test_idx]
                #     y_train, y_test = y_resampled_smote_enn[train_idx], y_resampled_smote_enn[test_idx]

                #         # Membuat dan melatih model Random Forest
                #     clf = RandomForestClassifier(n_estimators=est, max_depth=depth, criterion=crit, random_state=42)
                #     clf.fit(X_train, y_train)
                        

                #         # Memprediksi kelas untuk setiap sampel data di set pengujian
                #     fold_predictions = clf.predict(X_test)

                #         # Menghitung matriks kebingungan untuk fold ini
                #     fold_confusion_matrix = confusion_matrix(y_test, fold_predictions)
                #     accuracy = accuracy_score(y_test, fold_predictions)
                #     precision = precision_score(y_test, fold_predictions)
                #     recall = recall_score(y_test, fold_predictions)
                #     f1 = f1_score(y_test, fold_predictions)

                #         # Menambahkan nilai-nilai ke dalam list
                #     accuracy_scores.append(accuracy)
                #     precision_scores.append(precision)
                #     recall_scores.append(recall)
                #     f1_scores.append(f1)
                #         # Menampilkan matriks kebingungan dari fold ini
                #     st.write(f"Confusion Matrix untuk Fold {fold}:")
                #     st.write(fold_confusion_matrix)
                #     st.write(f"Akurasi untuk Fold {fold}: {accuracy}")
                #     st.write(f"Presisi untuk Fold {fold}: {precision}")
                #     st.write(f"Recall untuk Fold {fold}: {recall}")
                #     st.write(f"F1-score untuk Fold {fold}: {f1}")

                    
                #     # Menampilkan matriks kebingungan dari fold ini
                #     st.write(f"Confusion Matrix untuk Fold {fold}:")
                #     st.write(fold_confusion_matrix)

                #         # Visualisasi pohon keputusan untuk fold ini
                    # for i, estimator in enumerate(clf.estimators_, 1):
                    #     fig, ax = plt.subplots(figsize=(10, 8))
                    #     plot_tree(estimator, filled=True, feature_names=X_train.columns, class_names=["No Stroke", "Stroke"], ax=ax)
                    #     plt.title(f"Random Forest Trees for Fold {fold}, Estimator {i}")
                    #     plt.savefig(f"random_forest_fold_{fold}_estimator_{i}.png", dpi=300)  # Menyimpan gambar
                    #     plt.close(fig)  # Menutup gambar agar tidak ditampilkan di Streamlit

                    #     st.image(f"random_forest_fold_{fold}_estimator_{i}.png")  # Menampilkan gambar

                #     # Menampilkan rata-rata dari semua fold
                # st.write(f"Rata-rata Akurasi: {np.mean(accuracy_scores)}")
                # st.write(f"Rata-rata Presisi: {np.mean(precision_scores)}")
                # st.write(f"Rata-rata Recall: {np.mean(recall_scores)}")
                # st.write(f"Rata-rata F1-score: {np.mean(f1_scores)}")
                        
                # st.subheader('Confusion Matrix')
                # def plot_confusion_matrix(y_true, y_pred, classes, title=None):
                #     cm = confusion_matrix(y_true, y_pred)
                #     fig,ax=plt.subplots(figsize=(6, 6))
                #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                #                     xticklabels=classes, yticklabels=classes)
                #     ax.set_xlabel('Predicted labels')
                #     ax.set_ylabel('True labels')
                #     ax.set_title(title or 'Confusion Matrix')
                #     plt.show()
                #     st.pyplot(fig)    
                # destree = DecisionTreeClassifier()
                # destree.fit(X_train, y_train)
                # # prediction
                # destree.score(X_test, y_test)
                # y_pred = destree.predict(X_test)
                # #Accuracy
                # accuracy = round(100 * accuracy_score(y_test,y_pred))
                # joblib.dump(f'model{n_test}', f'model{n_test}.joblib')       
                    
        
    elif selected == "Implementation":
            st.write("Pilihlah model untuk melakukan klasifikasi:")
            option = st.radio("Pilih Metode:", ['Random Forest', 'Random Forest SMOTE','Random Forest SMOTE ENN+ Seleksi Fitur'])
            
            if option == 'Random Forest':
                with st.form(key='random_forest_form'):
                    st.subheader('Masukkan Data Anda')
                    age = st.number_input('Masukkan Umur Pasien')
                    gender = st.radio("Gender",('Male', 'Female'))
                    if gender == "Male":
                        gen_Female = 0
                        gen_Male = 1
                        
                    elif gender == "Female" :
                        gen_Female = 1
                        gen_Male = 0

                    # HYPERTENSION
                    hypertension = st.radio("Hypertency",('No', 'Yes'))
                    if hypertension == "Yes":
                        
                        hypertension_0 = 0
                        hypertension_1 = 1
                    elif hypertension == "No":
                        hypertension_1 = 0
                        hypertension_0 = 1
                    
                    # HEART
                    heart_disease = st.radio("Heart Disease",('No', 'Yes'))
                    if heart_disease == "Yes":
                        heart_disease_1 = 1
                        heart_disease_0 = 0
                    elif heart_disease == "No":
                        heart_disease_1 = 0
                        heart_disease_0 = 1

                    # MARRIED
                    ever_married = st.radio("Ever Married",('No', 'Yes'))
                    if ever_married == "Yes":
                        ever_married_Y = 1
                        ever_married_N = 0
                    elif ever_married == "No":
                        ever_married_Y = 0
                        ever_married_N = 1

                    # WORK
                    work_type = st.selectbox(
                    'Select a Work Type',
                    options=['Govt_job', 'Never_worked','Private', 'Self_employed', 'childern'])
                    if work_type == "Govt_job":
                        work_type_G = 1
                        work_type_Never = 0
                        work_type_P = 0
                        work_type_S = 0
                        work_type_C = 0
                    elif work_type == "Never_worked":
                        work_type_G = 0
                        work_type_Never = 1
                        work_type_P = 0
                        work_type_S = 0
                        work_type_C = 0
                    elif work_type == "Private":
                        work_type_G = 0
                        work_type_Never = 0
                        work_type_P = 1
                        work_type_S = 0
                        work_type_C = 0
                    elif work_type == "Self_employed":
                        work_type_G = 0
                        work_type_Never = 0
                        work_type_P = 0
                        work_type_S = 1
                        work_type_C = 0
                    elif work_type == "childern":
                        work_type_G = 0
                        work_type_Never = 0
                        work_type_P = 0
                        work_type_S = 0
                        work_type_C = 1

                    # RESIDENCE
                    residence_type = st.radio("Residence Type",('Rural', 'Urban'))
                    if residence_type == "Rural":
                        residence_type_R = 1
                        residence_type_U = 0
                    elif residence_type == "Urban":
                        residence_type_R = 0
                        residence_type_U = 1

                    # GLUCOSE
                    avg_glucose_level = st.number_input('Average Glucose Level')
                    
                    # SMOKE
                    smoking_status = st.selectbox(
                    'Select a smoking status',
                    options=['Unknown', 'Formerly smoked', 'never smoked', 'smokes'])

                    if smoking_status == "Unknown":
                        smoking_status_U = 1
                        smoking_status_F = 0
                        smoking_status_N = 0
                        smoking_status_S = 0
                    elif smoking_status == "Formerly smoked":
                        smoking_status_U = 0
                        smoking_status_F = 1
                        smoking_status_N = 0
                        smoking_status_S = 0
                    elif smoking_status == "never smoked":
                        smoking_status_U = 0
                        smoking_status_F = 0
                        smoking_status_N = 1
                        smoking_status_S = 0
                    elif smoking_status == "smokes":
                        smoking_status_U = 0
                        smoking_status_F = 0
                        smoking_status_N = 0
                        smoking_status_S = 1
                        
                    bmi = st.number_input('BMI')
                  

                    
                    inputs = np.array([[
                            avg_glucose_level,bmi, gen_Female, gen_Male,
                            hypertension_0,hypertension_1,
                            heart_disease_0,heart_disease_1,
                            ever_married_N, ever_married_Y,
                            work_type_G, work_type_Never, work_type_P, work_type_S, work_type_C,
                            residence_type_R, residence_type_U,
                            smoking_status_U, smoking_status_F, smoking_status_N, smoking_status_S]])
                
                    
                    st.subheader('Hasil Klasifikasi dengan Random Forest')
                    cek_hasil = st.form_submit_button("Cek Hasil Klasifikasi")
                    if cek_hasil :
                        st.write(inputs)
                        # Memeriksa apakah ada nilai yang hilang dalam DataFrame
                        
                        use_model = joblib.load("RF_n_estimators=5_max_depth=4_criterion=entropy.joblib")
                        

                        # Now you can pass inputs_reshaped to the predict() method
                        input_pred = use_model.predict(inputs)
                        # input_pred = use_model.predict([inputs])[0]
                        st.subheader('Hasil Prediksi')
                        st.write(input_pred)
                        if input_pred == 1:
                            st.error('Anda  Terkena Stroke')
                        else:
                            st.success('Anda tidak terkena Stroke')
                    
            elif option == 'Random Forest SMOTE':
                with st.form(key='RFS_form'):
                           
                    
                            st.subheader('Masukkan Data Anda')
                            age = st.number_input('Masukkan Umur Pasien')
                            

            # GENDER
                            gender = st.radio("Gender",('Male', 'Female', 'Other'))
                            if gender == "Male":
                                gen_Female = 0
                                gen_Male = 1
                                
                            elif gender == "Female" :
                                gen_Female = 1
                                gen_Male = 0
                                gen_Other = 0
                            

                            # HYPERTENSION
                            hypertension = st.radio("Hypertency",('No', 'Yes'))
                            if hypertension == "Yes":
                                
                                hypertension_0 = 0
                                hypertension_1 = 1
                            elif hypertension == "No":
                                hypertension_1 = 0
                                hypertension_0 = 1
                            
                            # HEART
                            heart_disease = st.radio("Heart Disease",('No', 'Yes'))
                            if heart_disease == "Yes":
                                heart_disease_1 = 1
                                heart_disease_0 = 0
                            elif heart_disease == "No":
                                heart_disease_1 = 0
                                heart_disease_0 = 1

                            # MARRIED
                            ever_married = st.radio("Ever Married",('No', 'Yes'))
                            if ever_married == "Yes":
                                ever_married_Y = 1
                                ever_married_N = 0
                            elif ever_married == "No":
                                ever_married_Y = 0
                                ever_married_N = 1

                            # WORK
                            work_type = st.selectbox(
                            'Select a Work Type',
                            options=['Govt_job', 'Never_worked','Private', 'Self_employed', 'childern'])
                            if work_type == "Govt_job":
                                work_type_G = 1
                                work_type_Never = 0
                                work_type_P = 0
                                work_type_S = 0
                                work_type_C = 0
                            elif work_type == "Never_worked":
                                work_type_G = 0
                                work_type_Never = 1
                                work_type_P = 0
                                work_type_S = 0
                                work_type_C = 0
                            elif work_type == "Private":
                                work_type_G = 0
                                work_type_Never = 0
                                work_type_P = 1
                                work_type_S = 0
                                work_type_C = 0
                            elif work_type == "Self_employed":
                                work_type_G = 0
                                work_type_Never = 0
                                work_type_P = 0
                                work_type_S = 1
                                work_type_C = 0
                            elif work_type == "childern":
                                work_type_G = 0
                                work_type_Never = 0
                                work_type_P = 0
                                work_type_S = 0
                                work_type_C = 1

                            # RESIDENCE
                            residence_type = st.radio("Residence Type",('Rural', 'Urban'))
                            if residence_type == "Rural":
                                residence_type_R = 1
                                residence_type_U = 0
                            elif residence_type == "Urban":
                                residence_type_R = 0
                                residence_type_U = 1

                            # GLUCOSE
                            avg_glucose_level = st.number_input('Average Glucose Level')
                            
                            
                             
                            # if avg_glucose_level <=100 :
                            #     avg_glucose_level=0
                            # elif 101 <= avg_glucose_level <= 200:
                            #     avg_glucose_level=1
                            # else:
                            #     avg_glucose_level=2
                            # SMOKE
                            smoking_status = st.selectbox(
                            'Select a smoking status',
                            options=['Unknown', 'Formerly smoked', 'never smoked', 'smokes'])

                            if smoking_status == "Unknown":
                                smoking_status_U = 1
                                smoking_status_F = 0
                                smoking_status_N = 0
                                smoking_status_S = 0
                            elif smoking_status == "Formerly smoked":
                                smoking_status_U = 0
                                smoking_status_F = 1
                                smoking_status_N = 0
                                smoking_status_S = 0
                            elif smoking_status == "never smoked":
                                smoking_status_U = 0
                                smoking_status_F = 0
                                smoking_status_N = 1
                                smoking_status_S = 0
                            elif smoking_status == "smokes":
                                smoking_status_U = 0
                                smoking_status_F = 0
                                smoking_status_N = 0
                                smoking_status_S = 1
                                
                            bmi = st.number_input('BMI')
                           
                            df = pd.read_csv('data_clean.csv')
                            # st.dataframe(df)
                            x = df.drop(columns=['stroke'])
                            x_normalized = x.copy() 
                            
                            #Normalisasi data input
                            df_min_bmi = x_normalized['bmi'].min().reshape(-1, 1)
                            df_max_bmi =  x_normalized['bmi'].max().reshape(-1, 1)
                            
                            df_min_age = x_normalized['age'].min().reshape(-1, 1)
                            df_max_age = x_normalized['age'].max().reshape(-1, 1)
                                    
                            df_min_avg = x_normalized['avg_glucose_level'].min().reshape(-1, 1)
                            df_max_avg = x_normalized['avg_glucose_level'] .max().reshape(-1, 1)
                            
                             # Make a copy of x to keep the original data
                            age_norm = float((age - df_min_age) / (df_max_age - df_min_age))
                            avg_norm = float((avg_glucose_level - df_min_avg) / (df_max_avg - df_min_avg))
                            bmi_norm = float((bmi - df_min_bmi) / (df_max_bmi - df_min_bmi))
                            
                            inputs = np.array([[bmi_norm,age_norm,avg_norm,work_type_G,gen_Male,smoking_status_N, gen_Female,work_type_P,hypertension_0,heart_disease_0,smoking_status_U,residence_type_R,heart_disease_1,
                                                ever_married_Y
                                                
                                         ]])
                  
                           
                            
                            
                            st.subheader('Hasil Klasifikasi dengan Random Forest SMOTE ')
                            cek_hasil = st.form_submit_button("Cek Hasil Klasifikasi")
                            if cek_hasil :
                                st.dataframe(inputs)

                                
                                use_model = joblib.load("random_forest_smote_3_n_estimator=15_max_depth=None_criterion=entropy.joblib")
                                #Normalisasi data input
                                              # Now you can pass inputs_reshaped to the predict() method
                                input_pred = use_model.predict(inputs)
                                # input_pred = use_model.predict([inputs])[0]
                            
                            
                                st.subheader('Hasil Prediksi')
                                st.dataframe(input_pred)
                                if input_pred == 1:
                                    st.error('Anda  Terkena Stroke')
                                else:
                                    st.success('Anda tidak terkena Stroke')                            
            else:
                with st.form(key='RFSE_form'):
                           
                    
                            st.subheader('Masukkan Data Anda')
                            age = st.number_input('Masukkan Umur Pasien')
                            

            # GENDER
                            gender = st.radio("Gender",('Male', 'Female', 'Other'))
                            if gender == "Male":
                                gen_Female = 0
                                gen_Male = 1
                                
                            elif gender == "Female" :
                                gen_Female = 1
                                gen_Male = 0
                                gen_Other = 0
                            

                            # HYPERTENSION
                            hypertension = st.radio("Hypertency",('No', 'Yes'))
                            if hypertension == "Yes":
                                
                                hypertension_0 = 0
                                hypertension_1 = 1
                            elif hypertension == "No":
                                hypertension_1 = 0
                                hypertension_0 = 1
                            
                            # HEART
                            heart_disease = st.radio("Heart Disease",('No', 'Yes'))
                            if heart_disease == "Yes":
                                heart_disease_1 = 1
                                heart_disease_0 = 0
                            elif heart_disease == "No":
                                heart_disease_1 = 0
                                heart_disease_0 = 1

                            # MARRIED
                            ever_married = st.radio("Ever Married",('No', 'Yes'))
                            if ever_married == "Yes":
                                ever_married_Y = 1
                                ever_married_N = 0
                            elif ever_married == "No":
                                ever_married_Y = 0
                                ever_married_N = 1

                            # WORK
                            work_type = st.selectbox(
                            'Select a Work Type',
                            options=['Govt_job', 'Never_worked','Private', 'Self_employed', 'childern'])
                            if work_type == "Govt_job":
                                work_type_G = 1
                                work_type_Never = 0
                                work_type_P = 0
                                work_type_S = 0
                                work_type_C = 0
                            elif work_type == "Never_worked":
                                work_type_G = 0
                                work_type_Never = 1
                                work_type_P = 0
                                work_type_S = 0
                                work_type_C = 0
                            elif work_type == "Private":
                                work_type_G = 0
                                work_type_Never = 0
                                work_type_P = 1
                                work_type_S = 0
                                work_type_C = 0
                            elif work_type == "Self_employed":
                                work_type_G = 0
                                work_type_Never = 0
                                work_type_P = 0
                                work_type_S = 1
                                work_type_C = 0
                            elif work_type == "childern":
                                work_type_G = 0
                                work_type_Never = 0
                                work_type_P = 0
                                work_type_S = 0
                                work_type_C = 1

                            # RESIDENCE
                            residence_type = st.radio("Residence Type",('Rural', 'Urban'))
                            if residence_type == "Rural":
                                residence_type_R = 1
                                residence_type_U = 0
                            elif residence_type == "Urban":
                                residence_type_R = 0
                                residence_type_U = 1

                            # GLUCOSE
                            avg_glucose_level = st.number_input('Average Glucose Level')
                            
                          
                            # SMOKE
                            smoking_status = st.selectbox(
                            'Select a smoking status',
                            options=['Unknown', 'Formerly smoked', 'never smoked', 'smokes'])

                            if smoking_status == "Unknown":
                                smoking_status_U = 1
                                smoking_status_F = 0
                                smoking_status_N = 0
                                smoking_status_S = 0
                            elif smoking_status == "Formerly smoked":
                                smoking_status_U = 0
                                smoking_status_F = 1
                                smoking_status_N = 0
                                smoking_status_S = 0
                            elif smoking_status == "never smoked":
                                smoking_status_U = 0
                                smoking_status_F = 0
                                smoking_status_N = 1
                                smoking_status_S = 0
                            elif smoking_status == "smokes":
                                smoking_status_U = 0
                                smoking_status_F = 0
                                smoking_status_N = 0
                                smoking_status_S = 1
                                
                            bmi = st.number_input('BMI')
                           
                            df = pd.read_csv('data_clean.csv')
                            st.dataframe(df)
                            x = df.drop(columns=['stroke'])
                            x_normalized = x.copy() 
                            
                            #Normalisasi data input
                            df_min_bmi = x_normalized['bmi'].min().reshape(-1, 1)
                            df_max_bmi =  x_normalized['bmi'].max().reshape(-1, 1)
                            
                            df_min_age = x_normalized['age'].min().reshape(-1, 1)
                            df_max_age = x_normalized['age'].max().reshape(-1, 1)
                                    
                            df_min_avg = x_normalized['avg_glucose_level'].min().reshape(-1, 1)
                            df_max_avg = x_normalized['avg_glucose_level'] .max().reshape(-1, 1)
                            
                             # Make a copy of x to keep the original data
                            age_norm = float((age - df_min_age) / (df_max_age - df_min_age))
                            avg_norm = float((avg_glucose_level - df_min_avg) / (df_max_avg - df_min_avg))
                            bmi_norm = float((bmi - df_min_bmi) / (df_max_bmi - df_min_bmi))
                            
                            inputs = np.array([[age_norm,bmi_norm,avg_norm,work_type_G,work_type_P,gen_Female,smoking_status_F,smoking_status_U ,gen_Male,heart_disease_0,ever_married_Y, residence_type_R,hypertension_0,heart_disease_1
                                                
                                         ]])
                  
                           
                            
                            
                            st.subheader('Hasil Klasifikasi dengan Random Forest SMOTE ENN')
                            cek_hasil = st.form_submit_button("Cek Hasil Klasifikasi")
                            if cek_hasil :
                                st.dataframe(inputs)

                                
                                use_model = joblib.load("random_forest_smoteenn_3_n_estimator=15_max_depth=None_criterion=entropy.joblib")
                                #Normalisasi data input
                                              # Now you can pass inputs_reshaped to the predict() method
                                input_pred = use_model.predict(inputs)
                                # input_pred = use_model.predict([inputs])[0]
                            
                            
                                st.subheader('Hasil Prediksi')
                                st.dataframe(input_pred)
                                if input_pred == 1:
                                    st.error('Anda  Terkena Stroke')
                                else:
                                    st.success('Anda tidak terkena Stroke')
                            
                    