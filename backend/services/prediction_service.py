"""Prediction Service Module.

Provides a centralized service layer that orchestrates inference across
completed disease risk screening models without duplicating ML preprocessing,
retraining models, or modifying serialized artifacts.
"""

import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))


class PredictionService:
    """Centralized service for invoking disease prediction modules."""

    SUPPORTED_DISEASES: Dict[str, str] = {
        "lung_cancer": "lung_cancer",
        "lung-cancer": "lung_cancer",
        "asthma": "asthma",
        "parkinsons": "parkinsons",
        "parkinson": "parkinsons",
    }

    def __init__(self, models_root: str = MODELS_ROOT):
        self.models_root = models_root
        self._loaded_modules: Dict[str, Any] = {}

    def _normalize_disease_key(self, disease_key: str) -> str:
        """Map URL or alias keys to canonical disease directory names."""
        clean_key = str(disease_key).strip().lower().replace("-", "_")
        canonical = self.SUPPORTED_DISEASES.get(clean_key)
        if not canonical:
            raise ValueError(
                f"Unsupported disease: '{disease_key}'. Supported diseases: ['lung_cancer', 'asthma', 'parkinsons']."
            )
        return canonical

    def _get_module(self, canonical_disease: str) -> Any:
        """Load and cache a disease's predict module dynamically with import isolation."""
        if canonical_disease in self._loaded_modules:
            return self._loaded_modules[canonical_disease]

        disease_dir = os.path.join(self.models_root, canonical_disease)
        predict_file = os.path.join(disease_dir, "predict.py")
        prep_file = os.path.join(disease_dir, "preprocessing.py")

        if not os.path.exists(predict_file):
            raise FileNotFoundError(
                f"Prediction module for '{canonical_disease}' not found at: {predict_file}"
            )

        # Load local disease preprocessing first to ensure correct sys.modules binding
        if os.path.exists(prep_file):
            spec_prep = importlib.util.spec_from_file_location(
                f"{canonical_disease}_preprocessing", prep_file
            )
            if spec_prep and spec_prep.loader:
                mod_prep = importlib.util.module_from_spec(spec_prep)
                sys.modules["preprocessing"] = mod_prep
                sys.modules[f"{canonical_disease}_preprocessing"] = mod_prep
                spec_prep.loader.exec_module(mod_prep)

        # Load predict module
        spec_pred = importlib.util.spec_from_file_location(
            f"{canonical_disease}_predict", predict_file
        )
        if not spec_pred or not spec_pred.loader:
            raise ImportError(f"Failed to load module spec for '{canonical_disease}'")

        mod_pred = importlib.util.module_from_spec(spec_pred)
        sys.modules[f"{canonical_disease}_predict"] = mod_pred
        spec_pred.loader.exec_module(mod_pred)

        self._loaded_modules[canonical_disease] = mod_pred
        return mod_pred

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Verify artifact availability and return status for supported models."""
        checked_diseases = ["lung_cancer", "asthma", "parkinsons"]
        results: List[Dict[str, Any]] = []

        for disease_id in checked_diseases:
            disease_dir = os.path.join(self.models_root, disease_id)
            model_path = os.path.join(disease_dir, "model.joblib")
            preprocessor_path = os.path.join(disease_dir, "preprocessor.joblib")
            predict_path = os.path.join(disease_dir, "predict.py")

            is_available = (
                os.path.exists(model_path)
                and os.path.exists(preprocessor_path)
                and os.path.exists(predict_path)
            )

            if is_available:
                try:
                    self._get_module(disease_id)
                except Exception:
                    is_available = False

            results.append({
                "disease": disease_id,
                "available": is_available,
            })

        return results

    def predict(self, disease_key: str, payload: Any) -> Dict[str, Any]:
        """Execute risk screening prediction for a specified disease.

        Args:
            disease_key: Canonical name or alias of the disease.
            payload: Feature payload (single dictionary or list of dicts).

        Returns:
            Dictionary containing disease identifier and prediction output.

        Raises:
            ValueError: If disease is unsupported or payload validation fails.
            FileNotFoundError: If model artifacts are missing.
        """
        canonical_key = self._normalize_disease_key(disease_key)

        if not isinstance(payload, (dict, list)):
            raise ValueError(f"Payload must be a JSON object or list of objects, got: {type(payload).__name__}")

        if isinstance(payload, dict) and not payload:
            raise ValueError("Payload dictionary cannot be empty.")

        predictor_module = self._get_module(canonical_key)

        # Call existing disease prediction function directly (never retrains or modifies artifacts)
        prediction_result = predictor_module.predict(payload)

        return {
            "disease": canonical_key,
            "prediction": prediction_result,
        }


# Singleton service instance
_SERVICE_INSTANCE: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Access the singleton PredictionService instance."""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = PredictionService()
    return _SERVICE_INSTANCE
