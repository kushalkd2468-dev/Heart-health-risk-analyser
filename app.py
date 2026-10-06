import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import hashlib
import json
from datetime import datetime
from fpdf import FPDF

# Application Configuration
st.set_page_config(
    page_title="Heart Health Analyzer",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants for image dimensions and model path
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
MODEL_PATH = './models/heart_attack_detection_model.h5'

# User Database File
USER_DB_FILE = 'users.json'
PATIENT_DB_FILE = 'patients.json'

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}

# Disease Precautions Database
DISEASE_PRECAUTIONS = {
    'Healthy': {
        'precautions': [
            'Maintain a balanced diet rich in fruits and vegetables',
            'Exercise regularly (at least 150 minutes per week)',
            'Avoid smoking and limit alcohol consumption',
            'Regular health check-ups annually',
            'Maintain healthy weight and BMI',
            'Manage stress through meditation or yoga',
            'Get adequate sleep (7-8 hours daily)'
        ],
        'recommendations': 'Continue healthy lifestyle habits and monitor cardiovascular health annually.'
    },
    'Mild Disease': {
        'precautions': [
            'Consult cardiologist for regular monitoring',
            'Follow a heart-healthy diet (low sodium, low fat)',
            'Regular moderate exercise under medical supervision',
            'Monitor blood pressure and cholesterol regularly',
            'Take prescribed medications as directed',
            'Reduce stress and practice relaxation techniques',
            'Avoid excessive caffeine and alcohol',
            'Quit smoking immediately'
        ],
        'recommendations': 'Early intervention is crucial. Follow medical advice closely and attend all scheduled appointments.'
    },
    'Moderate Disease': {
        'precautions': [
            'Immediate consultation with cardiologist required',
            'Strict adherence to prescribed medication regimen',
            'Follow cardiac rehabilitation program',
            'Monitor vital signs daily (BP, heart rate)',
            'Limit physical exertion and avoid strenuous activities',
            'Follow strict dietary restrictions (DASH diet)',
            'Regular ECG and cardiac function tests',
            'Keep emergency medications accessible',
            'Inform family members about condition'
        ],
        'recommendations': 'Requires close medical supervision. May need lifestyle modifications and potentially surgical intervention.'
    },
    'Severe Disease': {
        'precautions': [
            'URGENT: Immediate medical attention required',
            'Hospitalization may be necessary',
            'Complete bed rest or limited activity only',
            'Continuous monitoring of cardiac function',
            'Strict medication compliance critical',
            'Prepare for potential surgical interventions',
            'Have caregiver support available 24/7',
            'Keep emergency contact numbers ready',
            'Avoid all forms of stress and exertion',
            'Follow cardiologist recommendations strictly'
        ],
        'recommendations': 'Critical condition requiring immediate hospitalization and intensive care. Surgical intervention likely needed.'
    },
    'Heart Attack': {
        'precautions': [
            'EMERGENCY: Call emergency services (911/108) IMMEDIATELY',
            'Immediate hospitalization in cardiac care unit',
            'Complete rest and intensive monitoring required',
            'Follow post-heart attack rehabilitation program',
            'Lifelong medication and lifestyle changes necessary',
            'Regular cardiac follow-ups mandatory',
            'Family CPR training recommended',
            'Identify and eliminate all risk factors',
            'Psychological support and counseling',
            'Prepare advance care directives'
        ],
        'recommendations': 'Life-threatening emergency. Requires immediate emergency care, hospitalization, and long-term cardiac rehabilitation.'
    }
}

# Authentication Functions
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USER_DB_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def load_patients():
    """Load patient records from JSON file"""
    if os.path.exists(PATIENT_DB_FILE):
        with open(PATIENT_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_patient_record(patient_data):
    """Save patient record to JSON file"""
    patients = load_patients()
    patient_id = f"{patient_data['name']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    patients[patient_id] = patient_data
    with open(PATIENT_DB_FILE, 'w') as f:
        json.dump(patients, f, indent=4)
    return patient_id

def register_user(username, password, email):
    """Register a new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    
    users[username] = {
        'password': hash_password(password),
        'email': email,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_users(users)
    return True, "Registration successful"

def authenticate_user(username, password):
    """Authenticate user credentials"""
    users = load_users()
    if username in users:
        if users[username]['password'] == hash_password(password):
            return True
    return False

def generate_pdf_report(patient_info, diagnosis_result, analysis_type):
    """Generate PDF report with patient details and diagnosis"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    
    # Header
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 10, 'Heart Health Diagnostic Report', 0, 1, 'C')
    pdf.ln(5)
    
    # Report Date and Time
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 8, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    pdf.ln(5)
    
    # Patient Information Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, 'Patient Information', 0, 1, 'L')
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 11)
    patient_details = [
        ('Patient Name', patient_info.get('name', 'N/A')),
        ('Age', str(patient_info.get('age', 'N/A'))),
        ('Gender', patient_info.get('gender', 'N/A')),
        ('Contact', patient_info.get('contact', 'N/A')),
        ('Medical ID', patient_info.get('medical_id', 'N/A')),
        ('Examination Date', datetime.now().strftime('%Y-%m-%d'))
    ]
    
    for label, value in patient_details:
        pdf.cell(50, 7, f"{label}:", 0, 0)
        pdf.multi_cell(0, 7, str(value))
    
    pdf.ln(3)
    
    # Diagnosis Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 8, 'Diagnosis Results', 0, 1, 'L')
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 11)
    pdf.cell(50, 7, 'Analysis Type:', 0, 0)
    pdf.multi_cell(0, 7, analysis_type)
    
    if 'class_label' in diagnosis_result:
        pdf.cell(50, 7, 'Diagnosis:', 0, 0)
        pdf.set_font('Arial', 'B', 11)
        pdf.multi_cell(0, 7, diagnosis_result['class_label'])
        
        pdf.set_font('Arial', '', 11)
        pdf.cell(50, 7, 'Confidence:', 0, 0)
        pdf.multi_cell(0, 7, f"{diagnosis_result['confidence']:.2f}%")
    
    pdf.ln(3)
    
    # Precautions Section
    disease = diagnosis_result.get('class_label', 'Unknown')
    if disease in DISEASE_PRECAUTIONS:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 8, 'Medical Precautions & Recommendations', 0, 1, 'L')
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 7, 'Precautions:', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        for i, precaution in enumerate(DISEASE_PRECAUTIONS[disease]['precautions'], 1):
            # Check if we need a new page
            if pdf.get_y() > 250:
                pdf.add_page()
                pdf.set_left_margin(15)
                pdf.set_right_margin(15)
            
            pdf.set_x(15)
            pdf.multi_cell(0, 6, f"{i}. {precaution}", 0, 'L')
        
        pdf.ln(2)
        
        # Check if we need a new page for recommendations
        if pdf.get_y() > 240:
            pdf.add_page()
            pdf.set_left_margin(15)
            pdf.set_right_margin(15)
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 7, 'Medical Recommendations:', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, DISEASE_PRECAUTIONS[disease]['recommendations'], 0, 'L')
    
    pdf.ln(5)
    
    # Footer
    # Check if we need a new page for footer
    if pdf.get_y() > 260:
        pdf.add_page()
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)
    
    pdf.set_font('Arial', 'I', 8)
    pdf.multi_cell(0, 5, 'This report is generated by AI-based analysis and should be verified by a qualified medical professional.', 0, 'C')
    pdf.multi_cell(0, 5, 'For medical emergencies, please contact your healthcare provider immediately.', 0, 'C')
    
    # Save PDF
    filename = f"report_{patient_info.get('name', 'patient').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

def login_page():
    """Display login/registration page"""
    st.markdown("""
    <style>
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 50px 20px;
    }
    .login-header {
        text-align: center;
        color: #1874CD;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.markdown("<h1 class='login-header'>❤️ Heart Health Analyzer</h1>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", type="primary", use_container_width=True):
                if login_username and login_password:
                    if authenticate_user(login_username, login_password):
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
        
        with tab2:
            st.subheader("Create New Account")
            reg_username = st.text_input("Username", key="reg_user")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_pass")
            reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_confirm")
            
            if st.button("Register", type="primary", use_container_width=True):
                if reg_username and reg_email and reg_password:
                    if reg_password != reg_password_confirm:
                        st.error("Passwords do not match")
                    elif len(reg_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        success, message = register_user(reg_username, reg_password, reg_email)
                        if success:
                            st.success(message + " Please login to continue.")
                        else:
                            st.error(message)
                else:
                    st.warning("Please fill in all fields")
        
        st.markdown("</div>", unsafe_allow_html=True)

def patient_info_form():
    """Form to collect patient information"""
    st.subheader("📋 Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Patient Name*", key="patient_name")
        age = st.number_input("Age*", min_value=1, max_value=120, value=30, key="patient_age")
        gender = st.selectbox("Gender*", ["Male", "Female", "Other"], key="patient_gender")
    
    with col2:
        contact = st.text_input("Contact Number", key="patient_contact")
        medical_id = st.text_input("Medical ID (Optional)", key="patient_medical_id")
        blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"], key="patient_blood")
    
    medical_history = st.text_area("Medical History (Optional)", key="patient_history")
    
    if st.button("Save Patient Information", type="primary"):
        if name and age:
            patient_info = {
                'name': name,
                'age': age,
                'gender': gender,
                'contact': contact,
                'medical_id': medical_id if medical_id else f"MID{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'blood_group': blood_group,
                'medical_history': medical_history,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            st.session_state.patient_info = patient_info
            st.success("✅ Patient information saved successfully!")
            return True
        else:
            st.error("Please fill in all required fields (marked with *)")
            return False
    
    return 'patient_info' in st.session_state and st.session_state.patient_info

# Advanced Error Handling for Model Loading
@st.cache_resource
def load_ml_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.sidebar.error(f"Critical Error Loading Model: {e}")
        return None

# Cached model loading
model = load_ml_model()

# Helper Functions for Retinal Image Analysis
def preprocess_retinal_image(image):
    """Advanced image preprocessing for retinal analysis"""
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Enhanced contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    
    # Binary thresholding
    _, binary_img = cv2.threshold(enhanced_img, 127, 255, cv2.THRESH_BINARY)
    
    return enhanced_img, binary_img

def extract_vessel_features(image):
    """Simulate vessel feature extraction"""
    vessel_density = np.random.uniform(0.3, 0.5)
    avg_vessel_width = np.random.uniform(2.5, 4.5)
    return vessel_density, avg_vessel_width

def extract_tortuosity(image):
    """Simulate vessel tortuosity calculation"""
    return np.random.uniform(1.0, 2.5)

def extract_optic_disc_features(image):
    """Simulate optic disc feature extraction"""
    disc_diameter = np.random.uniform(100, 150)
    disc_area = np.random.uniform(12000, 16000)
    return disc_diameter, disc_area

def analyze_risk(features):
    """Calculate risk factors based on extracted features"""
    thresholds = {
        'vessel_density': 0.4,
        'vessel_width': 5.0,
        'tortuosity': 2.0,
        'disc_area': 15000
    }
    
    risk_factors = {
        'Vessel Density Risk': 1 - (features['Vessel Density'] / thresholds['vessel_density']),
        'Vessel Width Risk': features['Avg Vessel Width'] / thresholds['vessel_width'],
        'Tortuosity Risk': features['Tortuosity'] / thresholds['tortuosity'],
        'Disc Area Risk': features['Disc Area'] / thresholds['disc_area']
    }

    return risk_factors

def visualize_retinal_analysis(image, enhanced_img, binary_img, features, risk_factors):
    """Create comprehensive visualization of retinal analysis"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    def resize_image(img):
        return cv2.resize(img, (288, 288))
    
    resized_original = resize_image(image)
    resized_enhanced = resize_image(enhanced_img)
    resized_binary = resize_image(binary_img)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(resized_original, caption="Fig 1: Original Image", use_container_width=True)
    
    with col2:
        st.image(resized_enhanced, caption="Fig 2: Enhanced Image", use_container_width=True)
    
    with col3:
        st.image(resized_binary, caption="Fig 3: Binary Segmentation", use_container_width=True)
    
    st.subheader("Cardiovascular Risk Assessment")
    overall_risk = sum(risk_factors.values()) / len(risk_factors)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_risk * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cardiovascular Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 33], 'color': "rgba(144, 238, 144, 0.5)"},
                {'range': [33, 66], 'color': "rgba(255, 255, 224, 0.5)"},
                {'range': [66, 100], 'color': "rgba(255, 160, 122, 0.5)"}
            ],
        }
    ))
    st.plotly_chart(fig)
    
    st.subheader("Detailed Risk Factors")
    risk_df = pd.DataFrame.from_dict(risk_factors, orient='index', columns=['Risk Value'])
    fig_bar = px.bar(
        risk_df, 
        title="Risk Factors Breakdown",
        labels={'index': 'Risk Category', 'value': 'Risk Level'},
        color_discrete_sequence=px.colors.sequential.Reds_r
    )
    st.plotly_chart(fig_bar)
    
    st.subheader("Feature Values")
    features_df = pd.DataFrame.from_dict(features, orient='index', columns=['Value'])
    st.dataframe(features_df)
    
    return overall_risk

def read_and_preprocess_image(image):
    """Preprocess image for ML model input"""
    img_array = np.array(Image.open(image))
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img = cv2.resize(img_array, (IMAGE_HEIGHT, IMAGE_WIDTH))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(image):
    """Classify heart disease from image"""
    if model is None:
        st.error("Machine Learning Model Not Loaded")
        return None, None
    
    img = read_and_preprocess_image(image)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    class_labels = ['Healthy', 'Mild Disease', 'Moderate Disease', 'Severe Disease', 'Heart Attack']
    score = predictions[0][predicted_class]
    return class_labels[predicted_class], score

def display_precautions(disease):
    """Display precautions and recommendations for diagnosed disease"""
    if disease in DISEASE_PRECAUTIONS:
        st.subheader("⚠️ Medical Precautions")
        
        for i, precaution in enumerate(DISEASE_PRECAUTIONS[disease]['precautions'], 1):
            st.write(f"{i}. {precaution}")
        
        st.subheader("💡 Medical Recommendations")
        st.info(DISEASE_PRECAUTIONS[disease]['recommendations'])

def main():
    # Check if user is logged in
    if not st.session_state.logged_in:
        login_page()
        return
    
    # Sidebar Navigation with Icons
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username}! 👤")
        
        if st.button("Logout", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.patient_info = {}
            st.rerun()
        
        st.markdown("---")
        
        selected = option_menu(
            "Heart Disease Analyzer", 
            ["Home", "Patient Info", "Retinal Analysis", "ML Classification"],
            icons=['house', 'person-badge', 'eye', 'heart-pulse'],
            menu_icon="app-indicator", 
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#f0f2f6"},
                "icon": {"color": "blue", "font-size": "20px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "#1874CD"},
            }
        )
    
    # Home Section
    if selected == "Home":
        st.title("🏥 Heart Health Diagnostic Platform")
        
        st.markdown("""
        ### Welcome to Advanced Cardiovascular Risk Analysis
        
        This application provides cutting-edge diagnostic support through:
        - **Patient Information Management**: Store and manage patient details
        - **Retinal Image Analysis**: Detect early cardiovascular risk indicators
        - **Machine Learning Classification**: Predict heart disease severity
        - **Comprehensive Reports**: Generate detailed PDF reports with precautions
        
        Navigate through sections to explore our advanced diagnostic tools.
        """)
        
        cols = st.columns(4)
        feature_details = [
            ("Patient Info", "Record patient demographics and medical history", "👤"),
            ("Retinal Analysis", "Advanced image processing for cardiovascular risks", "👁️"),
            ("ML Classification", "AI-powered heart disease prediction", "❤️"),
            ("PDF Reports", "Comprehensive diagnostic reports", "📄")
        ]
        
        for col, (title, desc, icon) in zip(cols, feature_details):
            with col:
                st.markdown(f"""
                <div style="
                    border: 1px solid #e0e0e0; 
                    border-radius: 10px; 
                    padding: 15px; 
                    margin-bottom: 10px;
                    background-color: white;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <h3>{icon} {title}</h3>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Patient Information Section
    elif selected == "Patient Info":
        st.title("👤 Patient Information")
        patient_info_form()
        
        if st.session_state.patient_info:
            st.markdown("---")
            st.subheader("Current Patient Details")
            info_df = pd.DataFrame([st.session_state.patient_info]).T
            info_df.columns = ['Details']
            st.dataframe(info_df)
    
    # Retinal Analysis Section
    elif selected == "Retinal Analysis":
        st.title("🔬 Retinal Image Analysis")
        
        if not st.session_state.patient_info:
            st.warning("⚠️ Please enter patient information first in the 'Patient Info' section.")
            return
        
        uploaded_retinal_image = st.file_uploader(
            "Upload Retinal Image", 
            type=["jpg", "jpeg", "png"],
            help="Upload medical images for detailed retinal analysis"
        )
        
        if uploaded_retinal_image is not None:
            retinal_image = Image.open(uploaded_retinal_image)
            
            preview_image = cv2.resize(np.array(retinal_image), (288, 288))
            st.image(preview_image, caption=f"Fig 0: Uploaded {uploaded_retinal_image.name}", use_container_width=True)
            
            enhanced_img, binary_img = preprocess_retinal_image(retinal_image)
            vessel_density, avg_vessel_width = extract_vessel_features(retinal_image)
            tortuosity = extract_tortuosity(retinal_image)
            disc_diameter, disc_area = extract_optic_disc_features(retinal_image)

            features = {
                'Vessel Density': vessel_density,
                'Avg Vessel Width': avg_vessel_width,
                'Tortuosity': tortuosity,
                'Disc Area': disc_area,
                'Disc Diameter': disc_diameter
            }

            risk_factors = analyze_risk(features)
            overall_risk = visualize_retinal_analysis(np.array(retinal_image), enhanced_img, binary_img, features, risk_factors)
            
            # Determine risk level for report
            if overall_risk < 0.33:
                risk_classification = "Healthy"
            elif overall_risk < 0.66:
                risk_classification = "Mild Disease"
            else:
                risk_classification = "Moderate Disease"
            
            display_precautions(risk_classification)
            
            # Generate Report Button
            if st.button("📄 Generate PDF Report", type="primary"):
                diagnosis_result = {
                    'class_label': risk_classification,
                    'confidence': overall_risk * 100
                }
                
                try:
                    filename = generate_pdf_report(
                        st.session_state.patient_info,
                        diagnosis_result,
                        "Retinal Image Analysis"
                    )
                    
                    with open(filename, "rb") as pdf_file:
                        st.download_button(
                            label="⬇️ Download Report",
                            data=pdf_file,
                            file_name=filename,
                            mime="application/pdf"
                        )
                    
                    st.success(f"✅ Report generated successfully: {filename}")
                except Exception as e:
                    st.error(f"Error generating report: {e}")
    
    # ML Classification Section
    elif selected == "ML Classification":
        st.title("❤️ Heart Disease Classification")
        
        if not st.session_state.patient_info:
            st.warning("⚠️ Please enter patient information first in the 'Patient Info' section.")
            return
        
        uploaded_image = st.file_uploader(
            "Upload Heart Disease Image", 
            type=["jpg", "jpeg", "png"],
            help="Upload medical images for heart disease classification"
        )
        
        if uploaded_image is not None:
            preview_image = Image.open(uploaded_image)
            preview_resized = cv2.resize(np.array(preview_image), (288, 288))
            st.image(preview_resized, caption=f"Fig 0: Uploaded {uploaded_image.name}", use_container_width=True)
            
            result = classify_image(uploaded_image)
            
            if result[0] is not None:
                class_label, score = result
                st.subheader(f"Predicted Class: {class_label}")
                st.subheader(f"Prediction Confidence: {score * 100:.2f}%")
                
                risk_colors = {
                    'Healthy': 'green',
                    'Mild Disease': 'yellow',
                    'Moderate Disease': 'orange',
                    'Severe Disease': 'red',
                    'Heart Attack': 'darkred'
                }
                
                st.markdown(f"""
                <div style="background-color:{risk_colors.get(class_label, 'white')}; 
                            color:white; 
                            padding:10px; 
                            border-radius:10px;">
                    <h3>{class_label} Risk Assessment</h3>
                </div>
                """, unsafe_allow_html=True)
                
                display_precautions(class_label)
                
                # Generate Report Button
                if st.button("📄 Generate PDF Report", type="primary", key="ml_report"):
                    diagnosis_result = {
                        'class_label': class_label,
                        'confidence': score * 100
                    }
                    
                    try:
                        filename = generate_pdf_report(
                            st.session_state.patient_info,
                            diagnosis_result,
                            "ML Heart Disease Classification"
                        )
                        
                        with open(filename, "rb") as pdf_file:
                            st.download_button(
                                label="⬇️ Download Report",
                                data=pdf_file,
                                file_name=filename,
                                mime="application/pdf"
                            )
                        
                        st.success(f"✅ Report generated successfully: {filename}")
                        
                        # Save patient record
                        patient_record = st.session_state.patient_info.copy()
                        patient_record['diagnosis'] = class_label
                        patient_record['confidence'] = score * 100
                        patient_record['diagnosis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        save_patient_record(patient_record)
                        
                    except Exception as e:
                        st.error(f"Error generating report: {e}")

if __name__ == "__main__":
    main()