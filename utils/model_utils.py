"""
Model Utilities - CardioFusion Clinical Platform
Handles model loading, prediction, and ensemble operations
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class ModelPredictor:
    """
    Professional-grade model prediction handler
    Manages multiple models and ensemble predictions
    """

    def __init__(self, models_dir: str = 'models'):
        """
        Initialize the predictor with model directory

        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_names = None
        self.scaler = None
        self.label_encoder = None

    def load_models(self) -> bool:
        """
        Load all available trained models

        Returns:
            bool: Success status
        """
        try:
            # Load preprocessing components
            self._load_preprocessing_components()

            # Load baseline models
            self._load_baseline_models()

            # Load advanced models (if available)
            self._load_advanced_models()

            print(f"✅ Successfully loaded {len(self.models)} models")
            return True

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

    def _load_preprocessing_components(self):
        """Load scaler and label encoder"""
        try:
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
            print("✅ Preprocessing components loaded")
        except FileNotFoundError:
            print("⚠️ Preprocessing components not found")

    def _load_baseline_models(self):
        """Load baseline machine learning models"""
        baseline_dir = self.models_dir / 'baseline_models'

        model_files = {
            'Logistic Regression': 'logistic_regression_model.pkl',
            'Decision Tree': 'decision_tree_model.pkl',
            'Random Forest': 'random_forest_model.pkl'
        }

        for name, filename in model_files.items():
            model_path = baseline_dir / filename
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                print(f"  📊 Loaded: {name}")

    def _load_advanced_models(self):
        """Load advanced models (XGBoost, Neural Network, Ensemble)"""
        advanced_dir = self.models_dir / 'advanced_models'

        if not advanced_dir.exists():
            print("ℹ️ Advanced models not yet trained")
            return

        model_files = {
            'XGBoost': 'xgboost_model.pkl',
            'Neural Network': 'neural_network_model.pkl',
            'Hybrid Ensemble': 'hybrid_ensemble_model.pkl'
        }

        for name, filename in model_files.items():
            model_path = advanced_dir / filename
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                print(f"  🚀 Loaded: {name}")

    def predict(self,
                input_data: pd.DataFrame,
                model_name: Optional[str] = None) -> Dict:
        """
        Make prediction on input data

        Args:
            input_data: Patient data as DataFrame
            model_name: Specific model to use (None = ensemble)

        Returns:
            Dictionary with prediction results
        """
        if model_name and model_name in self.models:
            return self._single_model_prediction(input_data, model_name)
        else:
            return self._ensemble_prediction(input_data)

    def _single_model_prediction(self,
                                   data: pd.DataFrame,
                                   model_name: str) -> Dict:
        """
        Prediction from a single model

        Args:
            data: Input features
            model_name: Name of model to use

        Returns:
            Prediction results dictionary
        """
        model = self.models[model_name]

        # Get probability predictions
        proba = model.predict_proba(data)[0]
        prediction = model.predict(data)[0]

        return {
            'model': model_name,
            'prediction': int(prediction),
            'prediction_label': self.label_encoder.inverse_transform([prediction])[0],
            'probability_no_disease': float(proba[0]),
            'probability_disease': float(proba[1]),
            'confidence': float(max(proba)),
            'risk_percentage': float(proba[1] * 100)
        }

    def _ensemble_prediction(self, data: pd.DataFrame) -> Dict:
        """
        Ensemble prediction using all available models

        Args:
            data: Input features

        Returns:
            Aggregated prediction results
        """
        predictions = []
        probabilities = []

        # Define model weights (higher for better performers)
        weights = {
            'Decision Tree': 0.30,
            'Random Forest': 0.25,
            'Logistic Regression': 0.15,
            'XGBoost': 0.35,  # If available
            'Neural Network': 0.10,  # If available
            'Hybrid Ensemble': 0.50  # Highest weight if available
        }

        # Collect predictions from all models
        model_results = {}
        total_weight = 0

        for name, model in self.models.items():
            weight = weights.get(name, 0.15)
            proba = model.predict_proba(data)[0]

            model_results[name] = {
                'probability_disease': float(proba[1]),
                'weight': weight
            }

            probabilities.append(proba[1] * weight)
            total_weight += weight

        # Calculate weighted average
        ensemble_prob = sum(probabilities) / total_weight if total_weight > 0 else 0.5
        ensemble_prediction = 1 if ensemble_prob >= 0.5 else 0

        return {
            'model': 'Ensemble (Weighted Average)',
            'prediction': int(ensemble_prediction),
            'prediction_label': self.label_encoder.inverse_transform([ensemble_prediction])[0],
            'probability_no_disease': float(1 - ensemble_prob),
            'probability_disease': float(ensemble_prob),
            'confidence': float(max(ensemble_prob, 1 - ensemble_prob)),
            'risk_percentage': float(ensemble_prob * 100),
            'individual_models': model_results
        }

    def get_risk_category(self, risk_percentage: float) -> Tuple[str, str, str]:
        """
        Categorize risk level with clinical interpretation

        Args:
            risk_percentage: Risk score (0-100)

        Returns:
            Tuple of (category, emoji, color)
        """
        if risk_percentage < 30:
            return ("Low Risk", "✅", "#059669")
        elif risk_percentage < 50:
            return ("Moderate-Low Risk", "⚠️", "#f59e0b")
        elif risk_percentage < 70:
            return ("Moderate-High Risk", "⚠️", "#d97706")
        else:
            return ("High Risk", "🚨", "#dc2626")

    def get_available_models(self) -> List[str]:
        """Get list of loaded model names"""
        return list(self.models.keys())


def load_all_models(models_dir: str = 'models') -> ModelPredictor:
    """
    Convenience function to create and load ModelPredictor

    Args:
        models_dir: Directory containing models

    Returns:
        Initialized ModelPredictor
    """
    predictor = ModelPredictor(models_dir)
    predictor.load_models()
    return predictor


def format_prediction_output(prediction: Dict, detailed: bool = False) -> str:
    """
    Format prediction results for display

    Args:
        prediction: Prediction dictionary
        detailed: Show detailed output

    Returns:
        Formatted string output
    """
    risk_pct = prediction['risk_percentage']
    category, emoji, _ = ModelPredictor(None).get_risk_category(risk_pct)

    if not detailed:
        # Simple output
        return f"""
{emoji} **{category}**
Risk Score: {risk_pct:.1f}%
Prediction: {prediction['prediction_label']}
        """
    else:
        # Detailed output
        output = f"""
╔═══════════════════════════════════════╗
║  🩺 CARDIOVASCULAR RISK ASSESSMENT    ║
╠═══════════════════════════════════════╣

📊 RISK LEVEL: {emoji} {category.upper()}
├─ No Disease Probability: {prediction['probability_no_disease']*100:.1f}%
└─ Heart Disease Probability: {prediction['probability_disease']*100:.1f}%

🎯 PREDICTION: {prediction['prediction_label']}
📈 Confidence: {prediction['confidence']*100:.1f}%
🤖 Model: {prediction['model']}
        """

        if 'individual_models' in prediction:
            output += "\n\n📊 INDIVIDUAL MODEL PREDICTIONS:\n"
            for model, results in prediction['individual_models'].items():
                output += f"├─ {model}: {results['probability_disease']*100:.1f}% (weight: {results['weight']:.2f})\n"

        output += "\n╚═══════════════════════════════════════╝"
        return output
