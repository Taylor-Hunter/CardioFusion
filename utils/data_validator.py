"""
Data Validation Module - CardioFusion Clinical Platform
Validates and preprocesses user input data
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class DataValidator:
    """
    Professional data validation and preprocessing handler
    Ensures input data matches model requirements
    """

    # Clinical value ranges and categories
    VALID_RANGES = {
        'age': (18, 120),
        'height_cm': (100, 250),
        'weight_kg': (30, 300),
        'bmi': (10, 100),
        'alcohol_consumption': (0, 30),
        'fruit_consumption': (0, 120),
        'green_vegetables_consumption': (0, 128),
        'fried_potato_consumption': (0, 128)
    }

    VALID_CATEGORIES = {
        'general_health': ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
        'checkup': ['Never', '5 or more years ago', 'Within the past 5 years',
                   'Within the past 2 years', 'Within the past year'],
        'exercise': ['Yes', 'No'],
        'sex': ['Male', 'Female'],
        'age_category': ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                        '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'],
        'smoking_history': ['Yes', 'No'],
        'diabetes': ['No', 'Yes', 'No, pre-diabetes or borderline diabetes',
                    'Yes, but female told only during pregnancy'],
        'binary': ['Yes', 'No']
    }

    # Feature engineering mappings
    AGE_MAPPING = {
        '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37, '40-44': 42,
        '45-49': 47, '50-54': 52, '55-59': 57, '60-64': 62, '65-69': 67,
        '70-74': 72, '75-79': 77, '80+': 82
    }

    ORDINAL_MAPPINGS = {
        'general_health': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4},
        'age_category': {cat: idx for idx, cat in enumerate(VALID_CATEGORIES['age_category'])},
        'bmi_category': {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
    }

    def __init__(self):
        """Initialize the validator"""
        self.validation_errors = []
        self.validation_warnings = []

    def validate_input(self, user_data: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate user input data

        Args:
            user_data: Dictionary of user inputs

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.validation_errors = []
        self.validation_warnings = []

        # Validate numerical ranges
        self._validate_numerical_values(user_data)

        # Validate categorical values
        self._validate_categorical_values(user_data)

        # Clinical validation
        self._validate_clinical_logic(user_data)

        is_valid = len(self.validation_errors) == 0

        return is_valid, self.validation_errors, self.validation_warnings

    def _validate_numerical_values(self, data: Dict):
        """Validate numerical inputs are within acceptable ranges"""

        # Height
        if 'height_cm' in data:
            height = data['height_cm']
            if not (self.VALID_RANGES['height_cm'][0] <= height <= self.VALID_RANGES['height_cm'][1]):
                self.validation_errors.append(
                    f"❌ Height must be between {self.VALID_RANGES['height_cm'][0]} and {self.VALID_RANGES['height_cm'][1]} cm"
                )

        # Weight
        if 'weight_kg' in data:
            weight = data['weight_kg']
            if not (self.VALID_RANGES['weight_kg'][0] <= weight <= self.VALID_RANGES['weight_kg'][1]):
                self.validation_errors.append(
                    f"❌ Weight must be between {self.VALID_RANGES['weight_kg'][0]} and {self.VALID_RANGES['weight_kg'][1]} kg"
                )

        # BMI (if calculated)
        if 'bmi' in data:
            bmi = data['bmi']
            if bmi < 10 or bmi > 100:
                self.validation_errors.append("❌ BMI value seems incorrect")

        # Alcohol consumption
        if 'alcohol_consumption' in data:
            alcohol = data['alcohol_consumption']
            if alcohol > 30:
                self.validation_warnings.append("⚠️ High alcohol consumption reported")

    def _validate_categorical_values(self, data: Dict):
        """Validate categorical inputs are from valid options"""

        categorical_fields = {
            'general_health': 'general_health',
            'checkup': 'checkup',
            'exercise': 'binary',
            'sex': 'sex',
            'age_category': 'age_category',
            'smoking_history': 'binary',
            'diabetes': 'diabetes'
        }

        for field, category_type in categorical_fields.items():
            if field in data:
                value = data[field]
                valid_values = self.VALID_CATEGORIES[category_type]
                if value not in valid_values:
                    self.validation_errors.append(
                        f"❌ Invalid {field.replace('_', ' ')}: '{value}'. Must be one of {valid_values}"
                    )

    def _validate_clinical_logic(self, data: Dict):
        """Validate clinical logic and consistency"""

        # BMI consistency check
        if 'height_cm' in data and 'weight_kg' in data:
            height_m = data['height_cm'] / 100
            calculated_bmi = data['weight_kg'] / (height_m ** 2)

            if 'bmi' in data:
                if abs(calculated_bmi - data['bmi']) > 2:
                    self.validation_warnings.append(
                        f"⚠️ BMI mismatch: Calculated {calculated_bmi:.1f}, Provided {data['bmi']:.1f}"
                    )

        # Age-related validations
        if 'age_category' in data:
            age_cat = data['age_category']

            # Pregnancy diabetes check
            if data.get('diabetes') == 'Yes, but female told only during pregnancy':
                if data.get('sex') != 'Female':
                    self.validation_errors.append(
                        "❌ Gestational diabetes can only be selected for females"
                    )

    def preprocess_for_model(self, user_data: Dict) -> pd.DataFrame:
        """
        Convert user input to model-ready format

        Args:
            user_data: Validated user input dictionary

        Returns:
            DataFrame ready for model prediction
        """
        # Create base dataframe
        processed_data = {}

        # Calculate BMI if not provided
        if 'bmi' not in user_data and 'height_cm' in user_data and 'weight_kg' in user_data:
            height_m = user_data['height_cm'] / 100
            user_data['bmi'] = user_data['weight_kg'] / (height_m ** 2)

        # Add numerical features
        numerical_features = [
            'height_cm', 'weight_kg', 'bmi', 'alcohol_consumption',
            'fruit_consumption', 'green_vegetables_consumption',
            'fried_potato_consumption'
        ]

        for feat in numerical_features:
            processed_data[feat.replace('_cm', '_(cm)').replace('_kg', '_(kg)').title()] = [
                user_data.get(feat, 0)
            ]

        # Feature engineering
        processed_data['Age_Numeric'] = [self.AGE_MAPPING.get(user_data.get('age_category', '50-54'), 52)]
        processed_data['BMI_Category_Encoded'] = [self._categorize_bmi(user_data.get('bmi', 25))]
        processed_data['Lifestyle_Risk_Score'] = [self._calculate_lifestyle_risk(user_data)]
        processed_data['Health_Conditions_Count'] = [self._count_health_conditions(user_data)]

        # Ordinal encoding
        processed_data['General_Health_Encoded'] = [
            self.ORDINAL_MAPPINGS['general_health'].get(user_data.get('general_health', 'Good'), 2)
        ]
        processed_data['Age_Category_Encoded'] = [
            self.ORDINAL_MAPPINGS['age_category'].get(user_data.get('age_category', '50-54'), 6)
        ]

        # One-hot encoding for categorical features
        self._add_one_hot_features(processed_data, user_data)

        # Create DataFrame
        df = pd.DataFrame(processed_data)

        # Ensure all required features are present (match training data)
        df = self._ensure_feature_completeness(df)

        return df

    def _categorize_bmi(self, bmi: float) -> int:
        """Categorize BMI into standard health categories"""
        if bmi < 18.5:
            return 0  # Underweight
        elif 18.5 <= bmi < 25:
            return 1  # Normal
        elif 25 <= bmi < 30:
            return 2  # Overweight
        else:
            return 3  # Obese

    def _calculate_lifestyle_risk(self, data: Dict) -> int:
        """Calculate lifestyle risk score"""
        risk_score = 0

        if data.get('smoking_history') == 'Yes':
            risk_score += 2
        if data.get('exercise') == 'No':
            risk_score += 2
        if data.get('alcohol_consumption', 0) > 14:
            risk_score += 1
        if data.get('fruit_consumption', 0) < 8:
            risk_score += 1
        if data.get('green_vegetables_consumption', 0) < 8:
            risk_score += 1
        if data.get('fried_potato_consumption', 0) > 8:
            risk_score += 1

        return risk_score

    def _count_health_conditions(self, data: Dict) -> int:
        """Count existing health conditions"""
        conditions = ['skin_cancer', 'other_cancer', 'depression', 'diabetes', 'arthritis']
        count = 0

        for condition in conditions:
            if data.get(condition) == 'Yes':
                count += 1
            elif condition == 'diabetes' and 'diabetes' in data.get('diabetes', ''):
                count += 1

        return count

    def _add_one_hot_features(self, processed_data: Dict, user_data: Dict):
        """Add one-hot encoded features"""

        # Checkup features
        checkup_value = user_data.get('checkup', 'Within the past year')
        processed_data['Checkup_Never'] = [1 if checkup_value == 'Never' else 0]
        processed_data['Checkup_Within the past 2 years'] = [
            1 if checkup_value == 'Within the past 2 years' else 0
        ]
        processed_data['Checkup_Within the past 5 years'] = [
            1 if checkup_value == 'Within the past 5 years' else 0
        ]
        processed_data['Checkup_Within the past year'] = [
            1 if checkup_value == 'Within the past year' else 0
        ]

        # Binary features
        processed_data['Exercise_Yes'] = [1 if user_data.get('exercise') == 'Yes' else 0]
        processed_data['Skin_Cancer_Yes'] = [1 if user_data.get('skin_cancer') == 'Yes' else 0]
        processed_data['Other_Cancer_Yes'] = [1 if user_data.get('other_cancer') == 'Yes' else 0]
        processed_data['Depression_Yes'] = [1 if user_data.get('depression') == 'Yes' else 0]
        processed_data['Arthritis_Yes'] = [1 if user_data.get('arthritis') == 'Yes' else 0]
        processed_data['Sex_Male'] = [1 if user_data.get('sex') == 'Male' else 0]
        processed_data['Smoking_History_Yes'] = [1 if user_data.get('smoking_history') == 'Yes' else 0]

        # Diabetes features
        diabetes_value = user_data.get('diabetes', 'No')
        processed_data['Diabetes_No, pre-diabetes or borderline diabetes'] = [
            1 if diabetes_value == 'No, pre-diabetes or borderline diabetes' else 0
        ]
        processed_data['Diabetes_Yes'] = [1 if diabetes_value == 'Yes' else 0]
        processed_data['Diabetes_Yes, but female told only during pregnancy'] = [
            1 if diabetes_value == 'Yes, but female told only during pregnancy' else 0
        ]

    def _ensure_feature_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all expected features are present"""

        expected_features = [
            'Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption',
            'Fruit_Consumption', 'Green_Vegetables_Consumption',
            'FriedPotato_Consumption', 'Age_Numeric', 'Lifestyle_Risk_Score',
            'Health_Conditions_Count', 'General_Health_Encoded',
            'Age_Category_Encoded', 'BMI_Category_Encoded',
            'Checkup_Never', 'Checkup_Within the past 2 years',
            'Checkup_Within the past 5 years', 'Checkup_Within the past year',
            'Exercise_Yes', 'Skin_Cancer_Yes', 'Other_Cancer_Yes',
            'Depression_Yes', 'Diabetes_No, pre-diabetes or borderline diabetes',
            'Diabetes_Yes', 'Diabetes_Yes, but female told only during pregnancy',
            'Arthritis_Yes', 'Sex_Male', 'Smoking_History_Yes'
        ]

        # Add missing features with default value 0
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0

        # Reorder columns to match expected order
        df = df[expected_features]

        return df


def validate_user_input(user_data: Dict) -> Tuple[bool, pd.DataFrame, List[str], List[str]]:
    """
    Convenience function to validate and preprocess user input

    Args:
        user_data: User input dictionary

    Returns:
        Tuple of (is_valid, processed_df, errors, warnings)
    """
    validator = DataValidator()

    # Validate
    is_valid, errors, warnings = validator.validate_input(user_data)

    # Preprocess if valid
    if is_valid:
        processed_df = validator.preprocess_for_model(user_data)
    else:
        processed_df = pd.DataFrame()

    return is_valid, processed_df, errors, warnings
