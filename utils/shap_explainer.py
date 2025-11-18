"""
SHAP Explainability Module - CardioFusion Clinical Platform
Provides interpretable AI explanations for predictions
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    Professional SHAP-based model explainability handler
    Generates clinical interpretations of ML predictions
    """

    def __init__(self, model: Any, background_data: pd.DataFrame):
        """
        Initialize SHAP explainer

        Args:
            model: Trained ML model
            background_data: Representative sample for SHAP background
        """
        self.model = model
        self.background_data = background_data

        # Initialize appropriate explainer based on model type
        self.explainer = self._create_explainer()

    def _create_explainer(self):
        """Create appropriate SHAP explainer for model type"""
        model_type = type(self.model).__name__

        try:
            if 'RandomForest' in model_type or 'DecisionTree' in model_type:
                return shap.TreeExplainer(self.model)
            elif 'XGBoost' in model_type or 'LightGBM' in model_type:
                return shap.TreeExplainer(self.model)
            else:
                # Use KernelExplainer for other models (slower but universal)
                background_sample = shap.sample(self.background_data, min(100, len(self.background_data)))
                return shap.KernelExplainer(self.model.predict_proba, background_sample)
        except Exception as e:
            print(f"⚠️ Creating explainer: {e}")
            # Fallback to Kernel explainer
            background_sample = shap.sample(self.background_data, 50)
            return shap.KernelExplainer(self.model.predict_proba, background_sample)

    def explain_prediction(self,
                          input_data: pd.DataFrame,
                          feature_names: Optional[list] = None) -> Dict:
        """
        Generate SHAP explanation for a prediction

        Args:
            input_data: Single patient data
            feature_names: List of feature names

        Returns:
            Dictionary with SHAP values and interpretations
        """
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(input_data)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification - use positive class
                shap_values = shap_values[1]

            # Get feature names
            if feature_names is None:
                feature_names = input_data.columns.tolist()

            # Create feature contributions dictionary
            if len(shap_values.shape) == 2:
                contributions = dict(zip(feature_names, shap_values[0]))
            else:
                contributions = dict(zip(feature_names, shap_values))

            # Sort by absolute contribution
            sorted_contributions = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            return {
                'shap_values': shap_values,
                'contributions': contributions,
                'sorted_contributions': sorted_contributions,
                'top_positive': self._get_top_features(sorted_contributions, positive=True, top_n=5),
                'top_negative': self._get_top_features(sorted_contributions, positive=False, top_n=5),
                'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0.5
            }

        except Exception as e:
            print(f"❌ Error generating SHAP explanation: {e}")
            return {
                'error': str(e),
                'contributions': {},
                'sorted_contributions': []
            }

    def _get_top_features(self,
                         sorted_contributions: list,
                         positive: bool = True,
                         top_n: int = 5) -> list:
        """
        Extract top contributing features

        Args:
            sorted_contributions: Sorted list of (feature, value) tuples
            positive: True for risk-increasing, False for risk-decreasing
            top_n: Number of top features to return

        Returns:
            List of top contributing features
        """
        if positive:
            features = [(f, v) for f, v in sorted_contributions if v > 0]
        else:
            features = [(f, v) for f, v in sorted_contributions if v < 0]

        return features[:top_n]

    def generate_clinical_interpretation(self, explanation: Dict) -> str:
        """
        Convert SHAP values to clinical narrative

        Args:
            explanation: SHAP explanation dictionary

        Returns:
            Clinical interpretation string
        """
        if 'error' in explanation:
            return "⚠️ Explanation not available"

        output = "📊 **FEATURE CONTRIBUTION ANALYSIS**\n\n"

        # Risk-increasing factors
        if explanation['top_positive']:
            output += "🔴 **INCREASING RISK:**\n"
            for feature, value in explanation['top_positive']:
                bar_length = int(abs(value) * 20)
                bar = "█" * min(bar_length, 20)
                output += f"├─ {self._format_feature_name(feature):<30} {bar:<20} +{value:.3f}\n"

        output += "\n"

        # Risk-decreasing factors
        if explanation['top_negative']:
            output += "🟢 **DECREASING RISK:**\n"
            for feature, value in explanation['top_negative']:
                bar_length = int(abs(value) * 20)
                bar = "█" * min(bar_length, 20)
                output += f"├─ {self._format_feature_name(feature):<30} {bar:<20} {value:.3f}\n"

        return output

    def _format_feature_name(self, feature: str) -> str:
        """
        Convert technical feature names to readable clinical terms

        Args:
            feature: Technical feature name

        Returns:
            Human-readable feature name
        """
        # Feature name mapping for clinical context
        name_mapping = {
            'Age_Numeric': 'Age',
            'BMI': 'Body Mass Index',
            'General_Health_Encoded': 'General Health Status',
            'Exercise_Yes': 'Regular Exercise',
            'Smoking_History_Yes': 'Smoking History',
            'Diabetes_Yes': 'Diabetes Diagnosis',
            'Arthritis_Yes': 'Arthritis',
            'Depression_Yes': 'Depression',
            'Lifestyle_Risk_Score': 'Lifestyle Risk',
            'Health_Conditions_Count': 'Comorbidities',
            'Alcohol_Consumption': 'Alcohol Intake',
            'Sex_Male': 'Male Gender',
            'Skin_Cancer_Yes': 'Skin Cancer History',
            'Other_Cancer_Yes': 'Cancer History'
        }

        return name_mapping.get(feature, feature.replace('_', ' ').title())

    def plot_force_plot(self,
                       explanation: Dict,
                       matplotlib: bool = True):
        """
        Create SHAP force plot visualization

        Args:
            explanation: SHAP explanation dictionary
            matplotlib: Use matplotlib (True) or return interactive plot

        Returns:
            Force plot visualization
        """
        try:
            if 'error' in explanation:
                return None

            shap_values = explanation['shap_values']
            base_value = explanation['base_value']

            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]

            if matplotlib:
                shap.force_plot(
                    base_value,
                    shap_values[0] if len(shap_values.shape) == 2 else shap_values,
                    matplotlib=True,
                    show=False
                )
                return plt.gcf()
            else:
                return shap.force_plot(
                    base_value,
                    shap_values[0] if len(shap_values.shape) == 2 else shap_values
                )

        except Exception as e:
            print(f"❌ Error creating force plot: {e}")
            return None

    def get_recommendations(self, explanation: Dict) -> list:
        """
        Generate personalized health recommendations based on SHAP analysis

        Args:
            explanation: SHAP explanation dictionary

        Returns:
            List of clinical recommendations
        """
        recommendations = []

        if 'error' in explanation:
            return ["Consult with a healthcare professional for personalized advice"]

        # Analyze top risk factors
        for feature, value in explanation['top_positive']:
            rec = self._generate_recommendation(feature, value)
            if rec:
                recommendations.append(rec)

        # Generic recommendations
        if not recommendations:
            recommendations = [
                "✅ Maintain regular cardiovascular check-ups",
                "✅ Follow a heart-healthy diet (Mediterranean style)",
                "✅ Engage in regular physical activity (150min/week)",
                "✅ Monitor blood pressure and cholesterol levels"
            ]

        return recommendations[:6]  # Limit to 6 recommendations

    def _generate_recommendation(self, feature: str, value: float) -> Optional[str]:
        """
        Generate specific recommendation for a risk factor

        Args:
            feature: Feature name
            value: SHAP value

        Returns:
            Recommendation string or None
        """
        recommendations = {
            'Exercise_Yes': "✅ Start a regular exercise routine (consult physician first)",
            'Smoking_History_Yes': "✅ Consider smoking cessation programs",
            'Diabetes_Yes': "✅ Maintain strict blood glucose control",
            'BMI': "✅ Work towards a healthy BMI (18.5-24.9)",
            'Lifestyle_Risk_Score': "✅ Adopt healthier lifestyle habits",
            'Alcohol_Consumption': "✅ Moderate alcohol consumption",
            'General_Health_Encoded': "✅ Address overall health concerns with your doctor",
            'Depression_Yes': "✅ Seek support for mental health management"
        }

        return recommendations.get(feature, None)


def generate_explanation(model: Any,
                        input_data: pd.DataFrame,
                        background_data: pd.DataFrame,
                        detailed: bool = True) -> Tuple[Dict, str]:
    """
    Convenience function to generate complete explanation

    Args:
        model: Trained model
        input_data: Patient data
        background_data: Background dataset
        detailed: Include detailed interpretation

    Returns:
        Tuple of (explanation dict, interpretation string)
    """
    explainer = SHAPExplainer(model, background_data)
    explanation = explainer.explain_prediction(input_data)

    if detailed:
        interpretation = explainer.generate_clinical_interpretation(explanation)
        recommendations = explainer.get_recommendations(explanation)
        interpretation += "\n\n💡 **RECOMMENDATIONS:**\n"
        for rec in recommendations:
            interpretation += f"{rec}\n"
    else:
        interpretation = "SHAP analysis complete"

    return explanation, interpretation
