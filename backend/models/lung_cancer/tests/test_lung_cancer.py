"""Unit Tests for Lung Cancer Preprocessing, Model Artifacts, and Prediction Pipeline.

Covers:
1. Model and preprocessor artifact existence.
2. Artifact loading and compatibility.
3. Correct feature validation against schema.
4. Valid sample prediction execution.
5. Prediction output format and schema adherence.
6. Probability range constraint: 0.0 <= probability <= 1.0.
7. Missing and invalid feature handling (raising ValueError).
8. Repeated prediction consistency (determinism).
9. Batch prediction functionality.
"""

import json
import os
import sys
import unittest
import joblib
import pandas as pd

# Add the lung_cancer directory to sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from predict import predict
from preprocessing import FEATURE_COLUMNS, format_single_input


class TestLungCancerPipeline(unittest.TestCase):
    """Test suite for Lung Cancer screening model pipeline."""

    def setUp(self):
        """Prepare valid reference patient data."""
        self.valid_sample_high_risk = {
            "gender": "MALE",
            "age": 68,
            "smoking": 1,
            "yellow_fingers": 1,
            "anxiety": 1,
            "peer_pressure": 0,
            "chronic_disease": 1,
            "fatigue": 1,
            "allergy": 1,
            "wheezing": 1,
            "alcohol_consuming": 1,
            "coughing": 1,
            "shortness_of_breath": 1,
            "swallowing_difficulty": 1,
            "chest_pain": 1,
        }
        self.valid_sample_low_risk = {
            "gender": "FEMALE",
            "age": 30,
            "smoking": 0,
            "yellow_fingers": 0,
            "anxiety": 0,
            "peer_pressure": 0,
            "chronic_disease": 0,
            "fatigue": 0,
            "allergy": 0,
            "wheezing": 0,
            "alcohol_consuming": 0,
            "coughing": 0,
            "shortness_of_breath": 0,
            "swallowing_difficulty": 0,
            "chest_pain": 0,
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

    def test_03_correct_feature_validation(self):
        """3. Verify input formatting and validation for valid inputs."""
        df_formatted = format_single_input(self.valid_sample_high_risk)
        self.assertEqual(df_formatted.shape, (1, 15))
        self.assertEqual(list(df_formatted.columns), FEATURE_COLUMNS)
        self.assertEqual(df_formatted["gender"].iloc[0], 1)
        self.assertEqual(df_formatted["age"].iloc[0], 68.0)

    def test_04_valid_sample_prediction(self):
        """4. Verify prediction runs on valid patient profiles."""
        result = predict(self.valid_sample_high_risk)
        self.assertIsInstance(result, dict)
        self.assertIn("prediction", result)
        self.assertIn("class", result)
        self.assertIn("probability", result)
        self.assertIn("risk_level", result)
        self.assertIn("disclaimer", result)

    def test_05_prediction_output_format(self):
        """5. Verify the exact types and allowed values of prediction output."""
        result = predict(self.valid_sample_high_risk)
        self.assertIn(result["prediction"], ["YES", "NO"])
        self.assertIn(result["class"], [0, 1])
        self.assertIsInstance(result["probability"], float)
        self.assertIn(result["risk_level"], ["Low", "Moderate", "High"])
        self.assertIsInstance(result["disclaimer"], str)

    def test_06_probability_range(self):
        """6. Verify that predicted probability is strictly bounded in [0.0, 1.0]."""
        result_high = predict(self.valid_sample_high_risk)
        result_low = predict(self.valid_sample_low_risk)

        self.assertGreaterEqual(result_high["probability"], 0.0)
        self.assertLessEqual(result_high["probability"], 1.0)

        self.assertGreaterEqual(result_low["probability"], 0.0)
        self.assertLessEqual(result_low["probability"], 1.0)

    def test_07_invalid_or_missing_feature_handling(self):
        """7. Verify ValueError is raised when features are missing or values are out-of-range."""
        # Missing feature
        incomplete = self.valid_sample_high_risk.copy()
        del incomplete["smoking"]
        with self.assertRaises(ValueError) as ctx_missing:
            predict(incomplete)
        self.assertIn("Missing required feature", str(ctx_missing.exception))

        # Invalid age (out of realistic range)
        invalid_age = self.valid_sample_high_risk.copy()
        invalid_age["age"] = -5
        with self.assertRaises(ValueError) as ctx_age:
            predict(invalid_age)
        self.assertIn("out of realistic clinical range", str(ctx_age.exception))

        # Invalid binary feature (not 0 or 1)
        invalid_binary = self.valid_sample_high_risk.copy()
        invalid_binary["coughing"] = 5
        with self.assertRaises(ValueError) as ctx_bin:
            predict(invalid_binary)
        self.assertIn("must be 0 or 1", str(ctx_bin.exception))

        # Invalid gender string
        invalid_gender = self.valid_sample_high_risk.copy()
        invalid_gender["gender"] = "UNKNOWN"
        with self.assertRaises(ValueError) as ctx_gen:
            predict(invalid_gender)
        self.assertIn("Unrecognized gender value", str(ctx_gen.exception))

    def test_08_repeated_prediction_consistency(self):
        """8. Verify deterministic prediction output across multiple consecutive calls."""
        res1 = predict(self.valid_sample_high_risk)
        res2 = predict(self.valid_sample_high_risk)
        self.assertEqual(res1["class"], res2["class"])
        self.assertEqual(res1["probability"], res2["probability"])
        self.assertEqual(res1["risk_level"], res2["risk_level"])

    def test_09_batch_prediction(self):
        """9. Verify batch prediction using DataFrame input."""
        df_batch = pd.DataFrame([self.valid_sample_high_risk, self.valid_sample_low_risk])
        results = predict(df_batch)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], dict)
        self.assertIsInstance(results[1], dict)


if __name__ == "__main__":
    unittest.main()
