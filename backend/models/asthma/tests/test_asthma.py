"""Unit Tests for Asthma Preprocessing, Model Artifacts, and Prediction Pipeline.

Covers:
1. Required serialized artifacts and schema existence.
2. Artifact loading and compatibility.
3. Feature validation against the 14-feature schema.
4. Valid sample prediction execution.
5. Prediction output format and dictionary keys.
6. Probability range constraint: 0.0 <= probability <= 1.0.
7. Missing required feature rejection (ValueError).
8. Unexpected / excluded feature handling (Patient_ID, Asthma_Control_Level omitted safely).
9. Repeated prediction determinism.
10. Batch prediction functionality with DataFrame input.
"""

import json
import os
import sys
import unittest
import joblib
import pandas as pd

# Add the asthma model directory to sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from predict import predict
from preprocessing import FEATURE_COLUMNS, format_single_input


class TestAsthmaPipeline(unittest.TestCase):
    """Test suite for Asthma risk-screening model pipeline."""

    def setUp(self):
        """Prepare valid reference patient profiles."""
        self.valid_sample_high_risk = {
            "Age": 55,
            "Gender": "Female",
            "BMI": 29.5,
            "Smoking_Status": "Current",
            "Family_History": 1,
            "Allergies": "Pollen",
            "Air_Pollution_Level": "High",
            "Physical_Activity_Level": "Sedentary",
            "Occupation_Type": "Indoor",
            "Comorbidities": "Both",
            "Medication_Adherence": 0.50,
            "Number_of_ER_Visits": 2,
            "Peak_Expiratory_Flow": 320.0,
            "FeNO_Level": 45.0,
        }
        self.valid_sample_low_risk = {
            "Age": 30,
            "Gender": "Male",
            "BMI": 22.0,
            "Smoking_Status": "Never",
            "Family_History": 0,
            "Allergies": "None",
            "Air_Pollution_Level": "Low",
            "Physical_Activity_Level": "Active",
            "Occupation_Type": "Outdoor",
            "Comorbidities": "None",
            "Medication_Adherence": 0.90,
            "Number_of_ER_Visits": 0,
            "Peak_Expiratory_Flow": 550.0,
            "FeNO_Level": 12.0,
        }

    def test_01_artifacts_exist(self):
        """1. Verify that all required serialized artifacts and schema files exist."""
        model_path = os.path.join(MODEL_DIR, "model.joblib")
        preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")
        schema_path = os.path.join(MODEL_DIR, "feature_schema.json")
        metrics_path = os.path.join(MODEL_DIR, "metrics.json")

        self.assertTrue(os.path.exists(model_path), f"Missing model artifact: {model_path}")
        self.assertTrue(os.path.exists(preprocessor_path), f"Missing preprocessor artifact: {preprocessor_path}")
        self.assertTrue(os.path.exists(schema_path), f"Missing schema file: {schema_path}")
        self.assertTrue(os.path.exists(metrics_path), f"Missing metrics file: {metrics_path}")

    def test_02_artifacts_load(self):
        """2. Verify that preprocessor and model artifacts load correctly via joblib."""
        model_path = os.path.join(MODEL_DIR, "model.joblib")
        preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        self.assertIsNotNone(model)
        self.assertIsNotNone(preprocessor)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "predict_proba"))
        self.assertTrue(hasattr(preprocessor, "transform"))

    def test_03_expected_feature_validation(self):
        """3. Verify input formatting and validation for valid 14-feature inputs."""
        df_formatted = format_single_input(self.valid_sample_high_risk)
        self.assertEqual(df_formatted.shape, (1, 14))
        self.assertEqual(list(df_formatted.columns), FEATURE_COLUMNS)
        self.assertEqual(df_formatted["Age"].iloc[0], 55.0)
        self.assertEqual(df_formatted["Gender"].iloc[0], "Female")
        self.assertEqual(df_formatted["Family_History"].iloc[0], 1)

    def test_04_valid_sample_prediction(self):
        """4. Verify prediction runs successfully on valid patient profiles."""
        result = predict(self.valid_sample_high_risk)
        self.assertIsInstance(result, dict)
        self.assertIn("prediction", result)
        self.assertIn("class", result)
        self.assertIn("probability", result)
        self.assertIn("risk_level", result)
        self.assertIn("disclaimer", result)

    def test_05_output_format_is_correct(self):
        """5. Verify exact output types and categorical range."""
        result = predict(self.valid_sample_high_risk)
        self.assertIn(result["prediction"], ["Has Asthma", "No Asthma"])
        self.assertIn(result["class"], [0, 1])
        self.assertIsInstance(result["probability"], float)
        self.assertIn(result["risk_level"], ["Low", "Moderate", "High"])
        self.assertIsInstance(result["disclaimer"], str)

    def test_06_probability_range_is_valid(self):
        """6. Verify that probability is strictly bounded in [0.0, 1.0]."""
        res_high = predict(self.valid_sample_high_risk)
        res_low = predict(self.valid_sample_low_risk)

        self.assertGreaterEqual(res_high["probability"], 0.0)
        self.assertLessEqual(res_high["probability"], 1.0)
        self.assertGreaterEqual(res_low["probability"], 0.0)
        self.assertLessEqual(res_low["probability"], 1.0)

    def test_07_missing_feature_rejected(self):
        """7. Verify ValueError is raised when any required feature is omitted."""
        incomplete = self.valid_sample_high_risk.copy()
        del incomplete["Family_History"]
        with self.assertRaises(ValueError) as ctx:
            predict(incomplete)
        self.assertIn("Missing required feature", str(ctx.exception))

    def test_08_unexpected_and_excluded_features_handled(self):
        """8. Verify prohibited leakage columns (Patient_ID, Asthma_Control_Level) are excluded safely."""
        with_leakage = self.valid_sample_high_risk.copy()
        with_leakage["Patient_ID"] = "ASTH999999"
        with_leakage["Asthma_Control_Level"] = "Poorly Controlled"

        # The inference function must strip them and predict without error or leakage
        result = predict(with_leakage)
        self.assertIn(result["prediction"], ["Has Asthma", "No Asthma"])

        # Invalid category should be rejected
        invalid_cat = self.valid_sample_high_risk.copy()
        invalid_cat["Smoking_Status"] = "VapingOnly"
        with self.assertRaises(ValueError) as ctx_cat:
            predict(invalid_cat)
        self.assertIn("Invalid category", str(ctx_cat.exception))

    def test_09_repeated_prediction_is_deterministic(self):
        """9. Verify deterministic inference for identical patient inputs."""
        res1 = predict(self.valid_sample_high_risk)
        res2 = predict(self.valid_sample_high_risk)
        self.assertEqual(res1["class"], res2["class"])
        self.assertEqual(res1["probability"], res2["probability"])
        self.assertEqual(res1["risk_level"], res2["risk_level"])

    def test_10_batch_prediction_works(self):
        """10. Verify batch inference over a pandas DataFrame."""
        df_batch = pd.DataFrame([self.valid_sample_high_risk, self.valid_sample_low_risk])
        results = predict(df_batch)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], dict)
        self.assertIsInstance(results[1], dict)


if __name__ == "__main__":
    unittest.main()
