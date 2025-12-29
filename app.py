import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data and Model components
@st.cache_data
def load_data():
    return pd.read_csv('Student_performance_data _.csv')

df = load_data()

# Try to load model and scaler (Ensure you have saved them using joblib first)
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
except:
    st.warning("Model or Scaler not found. Please train and save them first.")

st.title("🎓 Student Performance Predictor (GPA)")
st.markdown("This app uses Machine Learning to predict student GPA based on various factors.")

# Sidebar for Navigation
page = st.sidebar.selectbox("Navigate", ["Exploratory Data Analysis", "GPA Prediction"])

if page == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    st.write("Review the correlation between student habits and their GPA.")
    
    # Visualization 1: GPA vs Absences
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Absences', y='GPA', ax=ax, color='red')
    st.pyplot(fig)
    
    # Visualization 2: Study Time Distribution
    fig2, ax2 = plt.subplots()
    sns.histplot(df['StudyTimeWeekly'], kde=True, ax=ax2)
    st.pyplot(fig2)

elif page == "GPA Prediction":
    st.header("🔮 Predict Student GPA")
    
    # Input Fields
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 15, 18, 17)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female")
        ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3])
        parent_edu = st.selectbox("Parental Education Level", [0, 1, 2, 3, 4])
    with col2:
        study_time = st.number_input("Weekly Study Time (Hours)", 0.0, 20.0, 10.0)
        absences = st.number_input("Absences", 0, 30, 5)
        tutoring = st.radio("Tutoring", [0, 1])
        parent_support = st.selectbox("Parental Support", [0, 1, 2, 3, 4])

    # Prediction Logic
    if st.button("Predict GPA"):
        # [cite_start]Arrange inputs as they were in the CSV [cite: 1]
        input_data = [[age, gender, ethnicity, parent_edu, study_time, absences, 
                       tutoring, parent_support, 0, 0, 0, 0]] # Placeholder for binary features
        
        # Scaling is essential for SVR
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        
        st.success(f"Estimated GPA: {prediction[0]:.2f}")
