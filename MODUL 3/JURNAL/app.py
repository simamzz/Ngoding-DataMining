
#Lanjutkan dibawah

# Import Library yang akan dipakai (JANGAN DI HAPUS)
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
# Tambahkan Import Library untuk SMOTE Oversampling (JANGAN DI HAPUS, ISI DIBAWAH INI)
from imblearn.over_sampling import SMOTE


# Inisiasi variabel untuk memuat model yang sudah dilatih Gunakan Pickel untuk load model(JANGAN DI HAPUS, ISI DIBAWAH INI)
pickle_in = open('E:/Kuliah/Semester 5/Ngoding-DataMining/MODUL 3/JURNAL/model.pkl', 'rb')
classsifier = pickle.load(pickle_in)

# Buat Fungsi yang dapat mengeluarkan hasil prediksi dari model berdasarkan input dari user(JANGAN DI HAPUS, ISI DIBAWAH INI)
def prediction(Gender, Partner, Internet_Service, Streaming_TV, Contract):
    input_data = np.array([[Gender, Partner, Internet_Service, Streaming_TV, Contract]]).astype(np.float64)

    pred = classsifier.predict(input_data)
    proba = classsifier.predict_proba(input_data)

    return pred, proba


# Buat fungsi yang dapat mengeluarkan metrik evaluasi model (JANGAN DI HAPUS, ISI DIBAWAH INI)
def evaluate_model(X_test, y_test):
    y_pred = classsifier.predict(X_test)
    y_pred_proba = classsifier.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, precision, recall, f1, roc_auc, y_pred, y_pred_proba


# Buat fungsi untuk membuat visualisasi plot kurva ROC (JANGAN DI HAPUS, ISI DIBAWAH INI)
def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(plt)


# Buat fungsi untuk membuat visualisasi confusion matrix (JANGAN DI HAPUS, ISI DIBAWAH INI)
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    st.pyplot(plt)


# Buat fungsi utama yang akan memuat semua fungsi di atas dan ditampilkan pada Streamlit (JANGAN DI HAPUS)
# Ganti st.title dengan nama masing masing, dan data balancing ganti menjadi Oversampling SMOTE yang udah di buat di praktikum (JANGAN DI HAPUS, ISI DIBAWAH INI)
def main():
    st.title("SYARIF IMAM MUSLIM")
    
    html_temp = """
    <div style ="background-color:darkblue;padding:13px; border-radius:15px; margin-bottom:20px;">
    <h1 style ="color:white; text-align:center;">Telco customer churn Classifier ML App </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    df = pd.read_csv('E:/Kuliah/Semester 5/Ngoding-DataMining/MODUL 3/JURNAL/dataset_jurnal.csv')

    majority = df[df['Churn Label'] == 0]
    minority = df[df['Churn Label'] == 1]
    
    minority_undersampled = minority.sample(len(minority), random_state=42)
    df = pd.concat([majority, minority_undersampled, minority])

    X = df.drop(['Churn Label'], axis=1)
    y = df['Churn Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    accuracy, precision, recall, f1, roc_auc, y_pred, y_pred_proba = evaluate_model(X_test, y_test)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.success(f"Accuracy: {accuracy:.2f}")

    with col2:
        st.info(f"Precision: {precision:.2f}")

    with col3:
        st.warning(f"Recall: {recall:.2f}")

    with col4:
        st.error(f"F1 Score: {f1:.2f}")

    plot_option = st.selectbox("Select the plot to display:", ["Select", "ROC AUC Curve", "Confusion Matrix"])

    if plot_option == "ROC AUC Curve":
        fpr, tpr, roc_auc = roc_curve(y_test, y_pred_proba)
        plot_roc_curve(fpr, tpr, roc_auc)

    elif plot_option == "Confusion Matrix":
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm)

    Gender = st.text_input("Gender")
    Partner = st.text_input("Partner")
    Internet_Service = st.text_input("Internet Service")
    Streaming_TV = st.text_input("Streaming TV")
    Contract = st.text_input("Contract")

    result = ""
    prob_result = ""

    if st.button("Predict"):
        result, proba = prediction(Gender, Partner, Internet_Service, Streaming_TV, Contract)
        prob_result = f"{proba[0]:.2f}"
        result = 'Yes' if result[0] == 1 else 'No'

    st.success(f"Prediksi Churn: {result}")
    st.success(f"Probabilitas Churn: {prob_result}")


# __main__
if __name__=='__main__':
    main()
