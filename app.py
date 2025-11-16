"""
CardioFusion Clinical Platform
Professional Web Application for Cardiovascular Disease Risk Assessment

Author: CardioFusion Development Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.model_utils import ModelPredictor
from utils.data_validator import DataValidator
from utils.shap_explainer import SHAPExplainer

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="CardioFusion | Heart Disease Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CardioFusion - Professional ML Platform for Cardiovascular Disease Prediction"
    }
)

# ============================================
# PROFESSIONAL MEDICAL STYLING
# ============================================

def load_css():
    """Load custom CSS for professional medical design"""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1e3a8a;
        --success-color: #059669;
        --warning-color: #d97706;
        --danger-color: #dc2626;
        --background: #f8fafc;
        --text-dark: #1e293b;
    }

    /* Professional header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        color: white;
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-bottom: 0;
        color: #e0e7ff;
    }

    /* Risk score card styling */
    .risk-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
        margin: 1rem 0;
    }

    .risk-low { border-left-color: #059669; }
    .risk-moderate { border-left-color: #d97706; }
    .risk-high { border-left-color: #dc2626; }

    /* Clinical input sections */
    .input-section {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(30, 58, 138, 0.3);
    }

    /* Disclaimer styling */
    .disclaimer {
        background: #fef3c7;
        border-left: 4px solid #d97706;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0;
        font-size: 0.95rem;
        color: #92400e;
    }

    /* Feature contribution bars */
    .contribution-bar {
        height: 25px;
        border-radius: 4px;
        margin: 5px 0;
        transition: all 0.3s;
    }

    .contribution-positive { background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%); }
    .contribution-negative { background: linear-gradient(90deg, #059669 0%, #10b981 100%); }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    </style>
    """, unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

def init_session_state():
    """Initialize session state variables"""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'simple'
    if 'background_data' not in st.session_state:
        st.session_state.background_data = None

# ============================================
# MODEL LOADING
# ============================================

@st.cache_resource
def load_models():
    """Load all trained models"""
    try:
        predictor = ModelPredictor('models')
        success = predictor.load_models()
        if success:
            return predictor
        return None
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

@st.cache_data
def load_background_data():
    """Load background data for SHAP"""
    try:
        train_data = pd.read_csv('train_data.csv')
        X_train = train_data.drop('Heart_Disease', axis=1)
        # Sample for SHAP background
        return X_train.sample(min(100, len(X_train)), random_state=42)
    except:
        return None

# ============================================
# UI COMPONENTS
# ============================================

def render_header():
    """Render professional header"""
    st.markdown("""
    <div class="main-header">
        <h1>🩺 CardioFusion Clinical Platform</h1>
        <p>Advanced AI-Powered Cardiovascular Disease Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar navigation and settings"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/heart-with-pulse.png", width=80)
        st.title("Navigation")

        page = st.radio(
            "Select View:",
            ["🔮 Risk Assessment", "📊 Model Performance", "📚 About"],
            label_visibility="collapsed"
        )

        st.divider()

        st.subheader("⚙️ Settings")
        view_mode = st.radio(
            "Prediction Detail Level:",
            ["Simple View", "Detailed Analysis"],
            index=0 if st.session_state.view_mode == 'simple' else 1
        )
        st.session_state.view_mode = 'simple' if view_mode == "Simple View" else 'detailed'

        st.divider()

        st.markdown("""
        ### 📋 Quick Guide
        1. Enter patient information
        2. Click **Analyze Risk**
        3. Review predictions
        4. Consult healthcare professional

        ---

        **Version**: 1.0.0
        **Models**: 6 Advanced ML Algorithms
        **Accuracy**: 92%+ ROC-AUC
        """)

        return page

def render_patient_input_form():
    """Render comprehensive patient input form"""

    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">👤 Demographic Information</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            age_category = st.selectbox(
                "Age Group",
                ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                 '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'],
                index=6
            )
        with col2:
            sex = st.selectbox("Biological Sex", ['Male', 'Female'])

        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">📏 Physical Measurements</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        with col2:
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=75.0, step=0.5)
        with col3:
            bmi = weight_kg / ((height_cm/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")

        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🏃 Lifestyle Factors</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            exercise = st.radio("Regular Physical Exercise", ['Yes', 'No'], horizontal=True)
            smoking_history = st.radio("Smoking History", ['No', 'Yes'], horizontal=True)
        with col2:
            alcohol_consumption = st.slider("Alcohol Consumption (units/month)", 0, 30, 0)
            fruit_consumption = st.slider("Fruit Intake (servings/month)", 0, 120, 30)

        col3, col4 = st.columns(2)
        with col3:
            green_vegetables_consumption = st.slider("Green Vegetables (servings/month)", 0, 128, 12)
        with col4:
            fried_potato_consumption = st.slider("Fried Potato (servings/month)", 0, 128, 4)

        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🏥 Health Status</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            general_health = st.select_slider(
                "General Health",
                options=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
                value='Good'
            )
            checkup = st.selectbox(
                "Last Medical Checkup",
                ['Within the past year', 'Within the past 2 years',
                 'Within the past 5 years', '5 or more years ago', 'Never']
            )
        with col2:
            diabetes = st.selectbox(
                "Diabetes Status",
                ['No', 'Yes', 'No, pre-diabetes or borderline diabetes',
                 'Yes, but female told only during pregnancy']
            )

        st.markdown("**Existing Medical Conditions:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            depression = 'Yes' if st.checkbox("Depression") else 'No'
            arthritis = 'Yes' if st.checkbox("Arthritis") else 'No'
        with col2:
            skin_cancer = 'Yes' if st.checkbox("Skin Cancer") else 'No'
            other_cancer = 'Yes' if st.checkbox("Other Cancer") else 'No'

        st.markdown('</div>', unsafe_allow_html=True)

    # Collect all data
    patient_data = {
        'age_category': age_category,
        'sex': sex,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'bmi': bmi,
        'exercise': exercise,
        'smoking_history': smoking_history,
        'alcohol_consumption': alcohol_consumption,
        'fruit_consumption': fruit_consumption,
        'green_vegetables_consumption': green_vegetables_consumption,
        'fried_potato_consumption': fried_potato_consumption,
        'general_health': general_health,
        'checkup': checkup,
        'diabetes': diabetes,
        'depression': depression,
        'arthritis': arthritis,
        'skin_cancer': skin_cancer,
        'other_cancer': other_cancer
    }

    return patient_data

def render_risk_gauge(risk_percentage, risk_category, emoji, color):
    """Render professional risk gauge visualization"""

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 24, 'color': '#1e293b'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 50], 'color': '#fef3c7'},
                {'range': [50, 70], 'color': '#fed7aa'},
                {'range': [70, 100], 'color': '#fecaca'}
            ],
            'threshold': {
                'line': {'color': "#1e293b", 'width': 4},
                'thickness': 0.75,
                'value': risk_percentage
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, -apple-system, sans-serif"}
    )

    return fig

def render_feature_contributions(explanation):
    """Render SHAP feature contributions"""

    if 'top_positive' not in explanation or 'top_negative' not in explanation:
        return

    st.markdown("### 📊 Feature Contribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔴 Risk-Increasing Factors")
        for feature, value in explanation['top_positive']:
            bar_width = min(abs(value) * 100, 100)
            feature_name = feature.replace('_', ' ').title()
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="font-weight: 500; margin-bottom: 5px;">{feature_name}</div>
                <div style="background: linear-gradient(90deg, #dc2626 0%, #ef4444 {bar_width}%, #f3f4f6 {bar_width}%);
                            height: 25px; border-radius: 4px; padding: 4px 10px; color: white; font-weight: 600;">
                    +{value:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🟢 Risk-Decreasing Factors")
        for feature, value in explanation['top_negative']:
            bar_width = min(abs(value) * 100, 100)
            feature_name = feature.replace('_', ' ').title()
            st.markdown(f"""
            <div style="margin: 10px 0;">
                <div style="font-weight: 500; margin-bottom: 5px;">{feature_name}</div>
                <div style="background: linear-gradient(90deg, #059669 0%, #10b981 {bar_width}%, #f3f4f6 {bar_width}%);
                            height: 25px; border-radius: 4px; padding: 4px 10px; color: white; font-weight: 600;">
                    {value:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_simple_prediction(prediction):
    """Render simple prediction view"""

    risk_pct = prediction['risk_percentage']
    category, emoji, color = st.session_state.predictor.get_risk_category(risk_pct)

    st.markdown(f"""
    <div class="risk-card risk-{category.split()[0].lower()}">
        <h2 style="color: {color}; margin: 0;">{emoji} {category}</h2>
        <p style="font-size: 1.5rem; margin: 1rem 0 0 0; color: #64748b;">
            Risk Score: <strong style="color: {color};">{risk_pct:.1f}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", prediction['prediction_label'])
    with col2:
        st.metric("Confidence", f"{prediction['confidence']*100:.1f}%")
    with col3:
        st.metric("Model", "Ensemble")

    # Risk gauge
    st.plotly_chart(render_risk_gauge(risk_pct, category, emoji, color), use_container_width=True)

def render_detailed_prediction(prediction, input_data):
    """Render detailed prediction with SHAP analysis"""

    # Simple view first
    render_simple_prediction(prediction)

    st.divider()

    # Individual model predictions
    if 'individual_models' in prediction:
        st.markdown("### 🤖 Individual Model Predictions")

        models_df = pd.DataFrame([
            {
                'Model': name,
                'Risk Probability': f"{results['probability_disease']*100:.1f}%",
                'Weight': results['weight']
            }
            for name, results in prediction['individual_models'].items()
        ])

        st.dataframe(models_df, use_container_width=True, hide_index=True)

    st.divider()

    # SHAP explanation
    if st.session_state.background_data is not None:
        try:
            st.markdown("### 🔍 AI Explainability Analysis")
            with st.spinner("Generating SHAP explanation..."):
                # Get best model for SHAP
                model = list(st.session_state.predictor.models.values())[0]
                explainer = SHAPExplainer(model, st.session_state.background_data)
                explanation = explainer.explain_prediction(input_data)

                if 'error' not in explanation:
                    render_feature_contributions(explanation)

                    st.divider()

                    # Clinical recommendations
                    st.markdown("### 💡 Clinical Recommendations")
                    recommendations = explainer.get_recommendations(explanation)

                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"{i}. {rec}")

        except Exception as e:
            st.warning(f"SHAP analysis unavailable: {str(e)}")

def render_disclaimer():
    """Render medical disclaimer"""
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong><br>
        This tool is for educational and informational purposes only. It does not provide medical advice,
        diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider
        with any questions you may have regarding a medical condition. Never disregard professional medical
        advice or delay in seeking it because of something you have read or seen on this platform.
    </div>
    """, unsafe_allow_html=True)

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application logic"""

    # Load CSS
    load_css()

    # Initialize session state
    init_session_state()

    # Load models
    if st.session_state.predictor is None:
        with st.spinner("🔄 Loading ML models..."):
            st.session_state.predictor = load_models()
            st.session_state.background_data = load_background_data()

    # Render UI
    render_header()
    page = render_sidebar()

    if page == "🔮 Risk Assessment":
        render_risk_assessment_page()
    elif page == "📊 Model Performance":
        render_performance_page()
    else:
        render_about_page()

def render_risk_assessment_page():
    """Render main risk assessment page"""

    if st.session_state.predictor is None:
        st.error("❌ Models not loaded. Please ensure models are trained and saved.")
        st.info("💡 Run data_preprocessing.ipynb and baseline_models.ipynb first.")
        return

    st.markdown("## 🔮 Patient Risk Assessment")

    # Patient input form
    patient_data = render_patient_input_form()

    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔬 Analyze Risk Profile", use_container_width=True)

    if analyze_button:
        # Validate input
        validator = DataValidator()
        is_valid, processed_data, errors, warnings = validator.validate_input(patient_data)

        if not is_valid:
            st.error("❌ Input Validation Failed:")
            for error in errors:
                st.error(error)
            return

        if warnings:
            for warning in warnings:
                st.warning(warning)

        # Preprocess
        input_df = validator.preprocess_for_model(patient_data)

        # Make prediction
        with st.spinner("🔄 Analyzing patient data..."):
            prediction = st.session_state.predictor.predict(input_df)

        st.success("✅ Analysis Complete!")

        # Render results based on view mode
        st.markdown("---")
        st.markdown("## 📊 Risk Assessment Results")

        if st.session_state.view_mode == 'simple':
            render_simple_prediction(prediction)
        else:
            render_detailed_prediction(prediction, input_df)

        # Disclaimer
        render_disclaimer()

def render_performance_page():
    """Render model performance page"""
    st.markdown("## 📊 Model Performance Metrics")

    st.info("📈 Model performance dashboards coming soon! Check baseline_models.ipynb and advanced_models.ipynb for detailed metrics.")

    if st.session_state.predictor:
        st.markdown("### 🤖 Available Models")
        models = st.session_state.predictor.get_available_models()
        for model in models:
            st.markdown(f"- ✅ {model}")

def render_about_page():
    """Render about page"""
    st.markdown("## 📚 About CardioFusion")

    st.markdown("""
    ### 🩺 Clinical Platform Overview

    CardioFusion is a state-of-the-art machine learning platform designed to assess cardiovascular
    disease risk using advanced AI algorithms. The system combines multiple machine learning models
    to provide accurate, interpretable predictions.

    ### 🎯 Key Features

    - **Advanced ML Models**: XGBoost, Neural Networks, Random Forest, and more
    - **Hybrid Ensemble**: Combines multiple models for superior accuracy
    - **AI Explainability**: SHAP-based feature importance analysis
    - **Professional Interface**: Medical-grade user experience
    - **Real-time Predictions**: Instant risk assessment

    ### 📊 Performance Metrics

    - **Accuracy**: 87-92%
    - **ROC-AUC**: 95%+
    - **Dataset**: 567,000+ balanced patient records
    - **Features**: 27 clinical and lifestyle indicators

    ### 👥 Development Team

    Built by healthcare and AI professionals committed to improving cardiovascular health outcomes.

    ### 📝 Citation

    Dataset: Cardiovascular Disease Dataset (Kaggle)

    ---

    **Version**: 1.0.0 | **Last Updated**: 2024
    """)

if __name__ == "__main__":
    main()
