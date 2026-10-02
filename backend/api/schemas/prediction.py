"""API Request Schemas and Validation for Disease Predictions.

Provides formal Pydantic schemas reflecting each disease's feature contract
established in their respective feature_schema.json files.
"""

from typing import Any, Dict, List, Union
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class LungCancerInputSchema(BaseModel):
    """Lung Cancer feature input schema."""

    model_config = ConfigDict(extra="ignore")

    gender: str = Field(..., description="Patient gender (e.g., 'MALE' or 'FEMALE')")
    age: float = Field(..., ge=1, le=120, description="Patient age in years")
    smoking: int = Field(..., description="Smoking status (1=No, 2=Yes or 0/1)")
    yellow_fingers: int = Field(..., description="Yellow fingers symptom indicator")
    anxiety: int = Field(..., description="Anxiety indicator")
    peer_pressure: int = Field(..., description="Peer pressure indicator")
    chronic_disease: int = Field(..., description="Chronic disease indicator")
    fatigue: int = Field(..., description="Fatigue indicator")
    allergy: int = Field(..., description="Allergy indicator")
    wheezing: int = Field(..., description="Wheezing indicator")
    alcohol_consuming: int = Field(..., description="Alcohol consuming indicator")
    coughing: int = Field(..., description="Coughing indicator")
    shortness_of_breath: int = Field(..., description="Shortness of breath indicator")
    swallowing_difficulty: int = Field(..., description="Swallowing difficulty indicator")
    chest_pain: int = Field(..., description="Chest pain indicator")


class AsthmaInputSchema(BaseModel):
    """Asthma feature input schema."""

    model_config = ConfigDict(extra="ignore")

    Age: float = Field(..., ge=1.0, le=120.0, description="Age in years")
    Gender: str = Field(..., description="Biological sex ('Female', 'Male', 'Other')")
    BMI: float = Field(..., ge=10.0, le=65.0, description="Body Mass Index")
    Smoking_Status: str = Field(..., description="Smoking status ('Never', 'Former', 'Current')")
    Family_History: int = Field(..., ge=0, le=1, description="Family history of asthma (0 or 1)")
    Allergies: str = Field(..., description="Allergy status ('None', 'Dust', 'Pollen', 'Pets', 'Multiple')")
    Air_Pollution_Level: str = Field(..., description="Air pollution level ('Low', 'Moderate', 'High')")
    Physical_Activity_Level: str = Field(..., description="Physical activity ('Sedentary', 'Moderate', 'Active')")
    Occupation_Type: str = Field(..., description="Occupation ('Indoor', 'Outdoor')")
    Comorbidities: str = Field(..., description="Comorbidities ('None', 'Diabetes', 'Hypertension', 'Both')")
    Medication_Adherence: float = Field(..., ge=0.0, le=1.0, description="Medication adherence ratio [0.0, 1.0]")
    Number_of_ER_Visits: float = Field(..., ge=0.0, le=20.0, description="Number of ER visits in past year")
    Peak_Expiratory_Flow: float = Field(..., ge=50.0, le=900.0, description="Peak expiratory flow rate (L/min)")
    FeNO_Level: float = Field(..., ge=0.0, le=150.0, description="Fractional exhaled nitric oxide (ppb)")


class ParkinsonsInputSchema(BaseModel):
    """Parkinson's disease 22 acoustic features input schema."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    fo_hz: float = Field(..., alias="MDVP:Fo(Hz)", ge=50.0, le=400.0)
    fhi_hz: float = Field(..., alias="MDVP:Fhi(Hz)", ge=50.0, le=700.0)
    flo_hz: float = Field(..., alias="MDVP:Flo(Hz)", ge=40.0, le=350.0)
    jitter_pct: float = Field(..., alias="MDVP:Jitter(%)", ge=0.0, le=0.1)
    jitter_abs: float = Field(..., alias="MDVP:Jitter(Abs)", ge=0.0, le=0.001)
    rap: float = Field(..., alias="MDVP:RAP", ge=0.0, le=0.1)
    ppq: float = Field(..., alias="MDVP:PPQ", ge=0.0, le=0.1)
    ddp: float = Field(..., alias="Jitter:DDP", ge=0.0, le=0.3)
    shimmer: float = Field(..., alias="MDVP:Shimmer", ge=0.0, le=0.5)
    shimmer_db: float = Field(..., alias="MDVP:Shimmer(dB)", ge=0.0, le=5.0)
    apq3: float = Field(..., alias="Shimmer:APQ3", ge=0.0, le=0.3)
    apq5: float = Field(..., alias="Shimmer:APQ5", ge=0.0, le=0.3)
    apq: float = Field(..., alias="MDVP:APQ", ge=0.0, le=0.5)
    dda: float = Field(..., alias="Shimmer:DDA", ge=0.0, le=1.0)
    nhr: float = Field(..., alias="NHR", ge=0.0, le=1.0)
    hnr: float = Field(..., alias="HNR", ge=0.0, le=60.0)
    rpde: float = Field(..., alias="RPDE", ge=0.0, le=1.0)
    dfa: float = Field(..., alias="DFA", ge=0.0, le=1.5)
    spread1: float = Field(..., alias="spread1", ge=-15.0, le=0.0)
    spread2: float = Field(..., alias="spread2", ge=0.0, le=1.0)
    d2: float = Field(..., alias="D2", ge=0.5, le=5.0)
    ppe: float = Field(..., alias="PPE", ge=0.0, le=1.0)


SCHEMA_REGISTRY: Dict[str, Any] = {
    "lung_cancer": LungCancerInputSchema,
    "asthma": AsthmaInputSchema,
    "parkinsons": ParkinsonsInputSchema,
}


def validate_payload_for_disease(disease_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate raw payload dictionary using the appropriate Pydantic schema.

    Args:
        disease_key: Canonical disease name.
        payload: Dictionary of input features.

    Returns:
        Validated dictionary matching raw input keys.

    Raises:
        ValueError: If required features are missing or invalid.
    """
    schema_cls = SCHEMA_REGISTRY.get(disease_key)
    if not schema_cls:
        raise ValueError(f"No validation schema registered for disease: '{disease_key}'")

    try:
        # Validate against schema
        validated_instance = schema_cls.model_validate(payload)
        # Return original payload keys cleaned of extra fields
        return payload
    except ValidationError as err:
        error_msgs = []
        for e in err.errors():
            loc = " -> ".join(str(p) for p in e["loc"])
            msg = e["msg"]
            error_msgs.append(f"{loc}: {msg}")
        raise ValueError(f"Invalid input features: {'; '.join(error_msgs)}")
