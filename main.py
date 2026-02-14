import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_assets():
    model = joblib.load('model/hr_model_rf.joblib')
    columns = joblib.load('model/model_columns.joblib')
    return model, columns


model, model_columns = load_assets()

st.title("İK İstifa Analizi")

st.sidebar.header("Çalışan Parametreleri")

age = st.sidebar.slider("Yaş", 18, 60, 35)
monthly_income = st.sidebar.number_input("Aylık Maaş ($)", 1000, 25000, 5000)
distance = st.sidebar.slider("Evden Uzaklık", 1, 30, 5)
total_working_years = st.sidebar.slider("Toplam Deneyim", 0, 40, 10)
years_at_company = st.sidebar.slider("Şirketteki Yıl", 0, 40, 5)

env_sat = st.sidebar.selectbox("Ortam Memnuniyeti (1-4)", [1, 2, 3, 4], index=2)
job_inv = st.sidebar.selectbox("İşe Bağlılık (1-4)", [1, 2, 3, 4], index=2)
job_sat = st.sidebar.selectbox("İş Memnuniyeti (1-4)", [1, 2, 3, 4], index=2)
work_life = st.sidebar.selectbox("İş-Yaşam Dengesi (1-4)", [1, 2, 3, 4], index=2)

overtime = st.sidebar.selectbox("Fazla Mesai", ["Yes", "No"], index=1)
marital = st.sidebar.selectbox("Medeni Durum", ["Married", "Single", "Divorced"])

if st.button("Analiz Et"):
    data = {
        'Age': age,
        'MonthlyIncome': monthly_income,
        'DistanceFromHome': distance,
        'TotalWorkingYears': total_working_years,
        'YearsAtCompany': years_at_company,
        'EnvironmentSatisfaction': env_sat,
        'JobInvolvement': job_inv,
        'JobSatisfaction': job_sat,
        'WorkLifeBalance': work_life,
        'OverTime': 1 if overtime == "Yes" else 0,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel': 1
    }

    df_input = pd.DataFrame(0, index=[0], columns=model_columns)

    for key, value in data.items():
        if key in df_input.columns:
            df_input[key] = value

    marital_col = f"MaritalStatus_{marital}"
    if marital_col in df_input.columns:
        df_input[marital_col] = 1

    prob = model.predict_proba(df_input)[0]
    risk_score = prob[1]

    st.divider()
    col_a, col_b = st.columns(2)
    col_a.metric("Ayrılma Riski", f"%{risk_score * 100:.1f}")
    col_b.metric("Bağlılık Puanı", f"%{prob[0] * 100:.1f}")

    if risk_score > 0.5:
        st.error("Dikkat: Bu çalışanın istifa etme ihtimali yüksek.")
    else:
        st.success("Güvenli: Çalışan şirkete bağlı görünüyor.")

    st.write(df_input)