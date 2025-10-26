# ============================================
# HoneVitals - Health Dashboard (Streamlit App)
# ============================================

import streamlit as st
import joblib
import numpy as np
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import HexColor, white
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Load model and scaler
# ------------------------
model = joblib.load('hone_vitals_model.pkl')
scaler = joblib.load('hone_scaler.pkl')

# ------------------------
# Page Config
# ------------------------
st.set_page_config(
    page_title="HoneVitals - Health Dashboard",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------
# Dark Theme CSS
# ------------------------
st.markdown('''
<style>
body {background-color: #0E1117; color: #E4E6EB; font-family: 'Inter', sans-serif;}
.main-title {font-size: 32px; font-weight: 700; color: #00BFFF; margin-bottom: 5px;}
.subtitle {font-size: 16px; color: #B0B3B8; margin-bottom: 25px;}
.stButton button {background-color: #00BFFF; color: white; border-radius: 8px; padding: 10px 25px; border: none; font-weight: bold; transition: 0.3s;}
.stButton button:hover {background-color: #009ACD;}
.result-box {background-color: #1C1F26; border-radius: 10px; padding: 20px; margin-top: 20px; border: 1px solid #333;}
.success {color: #3CB371;}
.warning {color: #FFD700;}
.error {color: #FF6347;}
.css-18e3th9 {background-color: #0E1117;}
.css-1d391kg {background-color: #0E1117;}
</style>
''', unsafe_allow_html=True)

# ------------------------
# Header
# ------------------------
st.markdown("<div class='main-title'>HoneVitals: Predictive Health & Lifestyle Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-driven cardiovascular & lifestyle risk screening with modern dark UI.</div>", unsafe_allow_html=True)
st.write("---")

# ------------------------
# Patient Details
# ------------------------
st.subheader("Patient Information & Health Parameters")
patient_name = st.text_input("Patient Name", "")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 20, 80, 35)
    cigsPerDay = st.number_input("Cigarettes per Day", 0, 50, 0)
    BMI = st.number_input("Body Mass Index (BMI)", 10.0, 50.0, 24.0)
    heartRate = st.number_input("Heart Rate (bpm)", 40, 120, 72)
    sleep_hours = st.number_input("Average Sleep Hours per Day", 3, 12, 7)
with col2:
    totChol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200)
    sysBP = st.number_input("Systolic BP (mmHg)", 90, 200, 120)
    diaBP = st.number_input("Diastolic BP (mmHg)", 60, 140, 80)
    glucose = st.number_input("Glucose (mg/dL)", 50, 200, 90)
    steps_walk = st.number_input("Average Steps Walked per Day", 0, 50000, 5000)
    kcal_burn = st.number_input("Total Calories Burned per Day", 0, 10000, 2000)

st.write("---")

# ------------------------
# Run Analysis
# ------------------------
if st.button("Run Health & Lifestyle Analysis") and patient_name:
    # Prediction
    input_data = np.array([[age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
    scaled_data = scaler.transform(input_data)
    prob = model.predict_proba(scaled_data)[0][1]
    hone_score = round((1 - prob) * 100, 2)

    # Risk Interpretation
    if prob < 0.1:
        risk_label = "Low Risk"
        risk_style = "success"
        message = "You have a low cardiovascular risk. Maintain your healthy lifestyle!"
    elif 0.1 <= prob <= 0.2:
        risk_label = "Moderate Risk"
        risk_style = "warning"
        message = "Your risk is moderate. Improve nutrition and exercise habits."
    else:
        risk_label = "High Risk"
        risk_style = "error"
        message = "High cardiovascular risk detected. Seek medical advice soon."

    # Display Result
    st.markdown(
        f"<div class='result-box'><h4 style='color:#00BFFF;'>{risk_label}</h4>"
        f"<p class='{risk_style}'>HoneScore: {hone_score} | 10-Year CHD Risk: {prob*100:.1f}%</p>"
        f"<p>{message}</p></div>",
        unsafe_allow_html=True
    )
    st.caption(f"Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ------------------------
    # Medical Chart
    # ------------------------
    metrics = ['BMI','Systolic BP','Diastolic BP','Cholesterol','Heart Rate','Glucose']
    values = [BMI, sysBP, diaBP, totChol, heartRate, glucose]

    colors = [
        "#3CB371" if BMI < 25 else "#FFD700" if BMI < 30 else "#FF6347",
        "#3CB371" if sysBP <= 120 else "#FFD700" if sysBP < 140 else "#FF6347",
        "#3CB371" if diaBP <= 80 else "#FFD700" if diaBP < 90 else "#FF6347",
        "#3CB371" if totChol < 200 else "#FFD700" if totChol < 240 else "#FF6347",
        "#3CB371" if 60 <= heartRate <= 100 else "#FFD700" if heartRate <= 110 else "#FF6347",
        "#3CB371" if glucose < 100 else "#FFD700" if glucose < 126 else "#FF6347"
    ]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(metrics, values, color=colors)
    ax.set_xlabel("Value")
    ax.set_title("Health Parameters Overview", color="#00BFFF")
    for i, v in enumerate(values):
        ax.text(v + 2, i, str(v), color='white', va='center', fontweight='bold')
    st.pyplot(fig)

    # ------------------------
    # Personalized Report
    # ------------------------
    report = f"Patient Name: {patient_name}\n\nUser Inputs:\n- Age: {age}\n- Cigarettes/Day: {cigsPerDay}\n- BMI: {BMI}\n- Heart Rate: {heartRate}\n"
    report += f"- Total Cholesterol: {totChol}\n- Systolic BP: {sysBP}\n- Diastolic BP: {diaBP}\n- Glucose: {glucose}\n"
    report += f"- Sleep Hours: {sleep_hours}\n- Steps Walked: {steps_walk}\n- Calories Burned: {kcal_burn}\n\n"
    report += f"Analysis Outcome:\n- Risk Level: {risk_label}\n- HoneScore: {hone_score}\n- CHD Risk: {prob*100:.1f}%\n\n"

    report += "Lifestyle & Diet Recommendations:\n"
    if age > 50: report += "- Schedule regular health checkups.\n"
    if cigsPerDay > 0: report += "- Quit smoking immediately.\n"
    if totChol > 240: report += "- Avoid fried/fatty foods; increase fiber intake.\n"
    if sysBP > 130 or diaBP > 85: report += "- Engage in regular aerobic activity (150 min/week).\n"
    if BMI > 25: report += "- Focus on a balanced diet and weight management.\n"
    if glucose > 120: report += "- Limit sugar and processed foods.\n"
    if sleep_hours < 7: report += "- Improve sleep hygiene; target 7-8 hours/night.\n"
    if steps_walk < 8000: report += "- Increase daily physical activity; aim for 8-10k steps.\n"
    if kcal_burn < 1800: report += "- Increase calorie burn via exercise.\n"

    report += "\nGeneral Health Tips:\n- Eat fruits, vegetables, whole grains, lean proteins.\n- Avoid smoking & excess alcohol.\n- Stay hydrated & manage stress.\n"

    # ------------------------
    # PDF Generation
    # ------------------------
    pdf_path = f"{patient_name.replace(' ','_')}_HealthReport.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 60

    c.setFillColor(HexColor("#0E1117"))
    c.rect(0, 0, width, height, fill=1, stroke=0)
    c.setFillColor(white)

    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(HexColor("#00BFFF"))
    c.drawString(50, y, "HoneVitals Health & Lifestyle Report")
    y -= 30

    c.setFont("Helvetica", 12)
    c.setFillColor(white)
    for line in report.split("\n"):
        c.drawString(50, y, line)
        y -= 15
        if y < 60:
            c.showPage()
            c.setFillColor(HexColor("#0E1117"))
            c.rect(0, 0, width, height, fill=1, stroke=0)
            c.setFillColor(white)
            y = height - 60
    c.save()

    # Download Button
    with open(pdf_path, "rb") as file:
        st.download_button(
            label="📄 Download Health & Lifestyle Report (PDF)",
            data=file,
            file_name=f"{patient_name.replace(' ','_')}_HealthReport.pdf",
            mime="application/pdf"
        )

st.write("---")
st.markdown("<p style='color:#B0B3B8;font-size:12px;'>Disclaimer: This dashboard is for educational and lifestyle guidance only.</p>", unsafe_allow_html=True)
