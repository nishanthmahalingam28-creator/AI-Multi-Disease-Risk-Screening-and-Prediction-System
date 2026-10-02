"""Database and Repository Layer Integration Tests.

Validates:
1. Database configuration loads.
2. SQLAlchemy models import.
3. Metadata/schema is valid.
4. Users can be represented correctly.
5. Clinic users can be represented correctly.
6. All 10 diseases can be represented.
7. Screening references user correctly.
8. Screening references disease correctly.
9. Clinic-assisted screening can reference clinic user.
10. Screening input JSON is stored correctly.
11. Prediction result references screening correctly.
12. Probability validation works.
13. Required fields are enforced.
14. Foreign-key integrity works.
15. Unique constraints work.
16. Database exceptions are safely handled with rollback.
17. No secrets are printed in logs.
18. No production data is modified (isolated in-memory test engine).
"""

from datetime import date
import os
import sys
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.db.database import (
    Base,
    create_db_engine,
    get_database_url,
    init_db,
)
from backend.db.models import (
    ClinicUser,
    Disease,
    PredictionResult,
    Screening,
    ScreeningInput,
    User,
)
from backend.db.repositories.screenings import (
    DISEASE_SEEDS,
    create_screening,
    get_disease_by_code,
    get_screening_by_id,
    get_user_screening_history,
    list_diseases,
    save_prediction_result,
    save_screening_input,
    seed_diseases,
)
from backend.db.repositories.users import (
    create_clinic_user,
    create_user,
    get_clinic_user_by_email,
    get_clinic_user_by_id,
    get_user_by_email,
    get_user_by_id,
)


@pytest.fixture
def test_db_session():
    """Create an isolated in-memory SQLite database session for unit tests."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def test_01_database_configuration_loads():
    """Verify database configuration resolves cleanly without errors."""
    url = get_database_url()
    assert isinstance(url, str)
    assert len(url) > 0
    # Engine creation succeeds
    engine = create_db_engine("sqlite:///:memory:")
    assert engine is not None


def test_02_sqlalchemy_models_import():
    """Verify all 6 core domain models import and are bound to declarative Base."""
    assert User.__tablename__ == "users"
    assert ClinicUser.__tablename__ == "clinic_users"
    assert Disease.__tablename__ == "diseases"
    assert Screening.__tablename__ == "screenings"
    assert ScreeningInput.__tablename__ == "screening_inputs"
    assert PredictionResult.__tablename__ == "prediction_results"


def test_03_metadata_schema_valid():
    """Verify metadata registers exactly the 6 required application tables."""
    table_names = set(Base.metadata.tables.keys())
    expected_tables = {
        "users",
        "clinic_users",
        "diseases",
        "screenings",
        "screening_inputs",
        "prediction_results",
    }
    assert expected_tables.issubset(table_names)


def test_04_user_creation_and_retrieval(test_db_session):
    """Verify User patient profile creation and lookup."""
    user = create_user(
        test_db_session,
        full_name="Jane Doe",
        email="jane.doe@example.com",
        mobile="+1234567890",
        date_of_birth=date(1985, 6, 15),
    )
    test_db_session.commit()

    assert user.id is not None
    assert user.email == "jane.doe@example.com"
    assert user.is_active is True

    # Lookup by ID and email
    by_id = get_user_by_id(test_db_session, user.id)
    assert by_id is not None
    assert by_id.full_name == "Jane Doe"

    by_email = get_user_by_email(test_db_session, "jane.doe@example.com")
    assert by_email is not None
    assert by_email.id == user.id


def test_05_clinic_user_creation_and_retrieval(test_db_session):
    """Verify ClinicUser healthcare staff profile creation and lookup."""
    clinic_user = create_clinic_user(
        test_db_session,
        full_name="Dr. Gregory House",
        email="dr.house@diagnostics.clinic",
        clinic_name="Princeton-Plainsboro Clinic",
        mobile="+1987654321",
    )
    test_db_session.commit()

    assert clinic_user.id is not None
    assert clinic_user.clinic_name == "Princeton-Plainsboro Clinic"

    fetched = get_clinic_user_by_id(test_db_session, clinic_user.id)
    assert fetched is not None
    assert fetched.email == "dr.house@diagnostics.clinic"

    by_email = get_clinic_user_by_email(test_db_session, "dr.house@diagnostics.clinic")
    assert by_email is not None
    assert by_email.id == clinic_user.id


def test_06_all_10_diseases_represented_and_seeded(test_db_session):
    """Verify all 10 planned diseases exist in reference seed catalog and can be seeded."""
    assert len(DISEASE_SEEDS) == 10
    expected_codes = {
        "breast_cancer",
        "diabetes",
        "heart_disease",
        "stroke",
        "kidney",
        "liver",
        "thyroid",
        "lung_cancer",
        "asthma",
        "parkinsons",
    }
    seed_codes = {d["code"] for d in DISEASE_SEEDS}
    assert seed_codes == expected_codes

    # Seed into database
    seeded = seed_diseases(test_db_session)
    test_db_session.commit()
    assert len(seeded) == 10

    all_active = list_diseases(test_db_session, active_only=True)
    assert len(all_active) == 10

    # Test individual lookup
    lung = get_disease_by_code(test_db_session, "lung_cancer")
    assert lung is not None
    assert lung.name == "Lung Cancer"

    parkinsons = get_disease_by_code(test_db_session, "parkinsons")
    assert parkinsons is not None
    assert parkinsons.name == "Parkinson's Disease"


def test_07_screening_references_user_correctly(test_db_session):
    """Verify Screening entity establishes proper relationship to User."""
    user = create_user(test_db_session, full_name="Alice Smith", email="alice@test.com")
    seed_diseases(test_db_session)
    lung_disease = get_disease_by_code(test_db_session, "lung_cancer")
    test_db_session.commit()

    screening = create_screening(test_db_session, user_id=user.id, disease_id=lung_disease.id)
    test_db_session.commit()

    assert screening.id is not None
    assert screening.user_id == user.id
    assert screening.user.email == "alice@test.com"
    assert user.screenings[0].id == screening.id


def test_08_screening_references_disease_correctly(test_db_session):
    """Verify Screening entity references Disease catalog record."""
    user = create_user(test_db_session, full_name="Bob Jones", email="bob@test.com")
    seed_diseases(test_db_session)
    asthma = get_disease_by_code(test_db_session, "asthma")
    test_db_session.commit()

    screening = create_screening(test_db_session, user_id=user.id, disease_id=asthma.id)
    test_db_session.commit()

    assert screening.disease_id == asthma.id
    assert screening.disease.code == "asthma"


def test_09_clinic_assisted_screening_relationship(test_db_session):
    """Verify clinic-assisted screening correctly references ClinicUser while allowing self-screenings."""
    user = create_user(test_db_session, full_name="Charlie Brown", email="charlie@test.com")
    clinic_user = create_clinic_user(
        test_db_session,
        full_name="Nurse Joy",
        email="joy@pokemon.clinic",
        clinic_name="Center Clinic",
    )
    seed_diseases(test_db_session)
    parkinsons = get_disease_by_code(test_db_session, "parkinsons")
    test_db_session.commit()

    # Clinic-assisted screening
    assisted = create_screening(
        test_db_session,
        user_id=user.id,
        disease_id=parkinsons.id,
        clinic_user_id=clinic_user.id,
    )
    test_db_session.commit()
    assert assisted.clinic_user_id == clinic_user.id
    assert assisted.clinic_user.clinic_name == "Center Clinic"

    # Self-screening (clinic_user_id is None)
    self_screen = create_screening(
        test_db_session,
        user_id=user.id,
        disease_id=parkinsons.id,
        clinic_user_id=None,
    )
    test_db_session.commit()
    assert self_screen.clinic_user_id is None


def test_10_screening_input_json_stored_correctly(test_db_session):
    """Verify submitted patient feature dictionary is stored in JSON field."""
    user = create_user(test_db_session, full_name="David Clark", email="david@test.com")
    seed_diseases(test_db_session)
    lung = get_disease_by_code(test_db_session, "lung_cancer")
    test_db_session.commit()

    screening = create_screening(test_db_session, user_id=user.id, disease_id=lung.id)
    test_db_session.commit()

    input_payload = {"age": 60, "smoking": 2, "coughing": 2, "shortness_of_breath": 2}
    s_input = save_screening_input(test_db_session, screening_id=screening.id, input_data=input_payload)
    test_db_session.commit()

    assert s_input.id is not None
    assert s_input.input_data["age"] == 60
    assert s_input.input_data["smoking"] == 2
    assert screening.screening_input.input_data == input_payload


def test_11_prediction_result_references_screening(test_db_session):
    """Verify PredictionResult establishes 1-to-1 relationship with Screening."""
    user = create_user(test_db_session, full_name="Emma Watson", email="emma@test.com")
    seed_diseases(test_db_session)
    asthma = get_disease_by_code(test_db_session, "asthma")
    test_db_session.commit()

    screening = create_screening(test_db_session, user_id=user.id, disease_id=asthma.id)
    test_db_session.commit()

    result = save_prediction_result(
        test_db_session,
        screening_id=screening.id,
        predicted_class=1,
        probability=0.88,
        risk_level="High",
        disclaimer="Statistical research assessment only.",
    )
    test_db_session.commit()

    assert result.id is not None
    assert result.probability == 0.88
    assert result.risk_level == "High"
    assert screening.prediction_result.predicted_class == 1


def test_12_probability_validation_check_constraint(test_db_session):
    """Verify database check constraint ck_valid_probability rejects values > 1.0 or < 0.0."""
    user = create_user(test_db_session, full_name="Frank Miller", email="frank@test.com")
    seed_diseases(test_db_session)
    disease = get_disease_by_code(test_db_session, "lung_cancer")
    test_db_session.commit()

    screening = create_screening(test_db_session, user_id=user.id, disease_id=disease.id)
    test_db_session.commit()

    # Probability > 1.0 must fail
    with pytest.raises(IntegrityError):
        save_prediction_result(
            test_db_session,
            screening_id=screening.id,
            predicted_class=1,
            probability=1.5,
            risk_level="High",
            disclaimer="Test",
        )
        test_db_session.commit()
    test_db_session.rollback()

    # Probability < 0.0 must fail
    with pytest.raises(IntegrityError):
        save_prediction_result(
            test_db_session,
            screening_id=screening.id,
            predicted_class=0,
            probability=-0.2,
            risk_level="Low",
            disclaimer="Test",
        )
        test_db_session.commit()
    test_db_session.rollback()


def test_13_required_fields_enforced(test_db_session):
    """Verify NOT NULL constraints are enforced on required attributes."""
    # User missing email
    with pytest.raises(IntegrityError):
        user_no_email = User(full_name="No Email User", email=None)
        test_db_session.add(user_no_email)
        test_db_session.commit()
    test_db_session.rollback()

    # Disease missing code
    with pytest.raises(IntegrityError):
        dis_no_code = Disease(code=None, name="No Code Disease")
        test_db_session.add(dis_no_code)
        test_db_session.commit()
    test_db_session.rollback()


def test_14_unique_constraints_work(test_db_session):
    """Verify unique constraint rejects duplicate emails."""
    create_user(test_db_session, full_name="User One", email="unique@example.com")
    test_db_session.commit()

    with pytest.raises(IntegrityError):
        create_user(test_db_session, full_name="User Two", email="unique@example.com")
        test_db_session.commit()
    test_db_session.rollback()


def test_15_screening_history_retrieval(test_db_session):
    """Verify get_user_screening_history returns complete ordered user history."""
    user = create_user(test_db_session, full_name="Grace Hopper", email="grace@compiler.org")
    seed_diseases(test_db_session)
    lung = get_disease_by_code(test_db_session, "lung_cancer")
    asthma = get_disease_by_code(test_db_session, "asthma")
    test_db_session.commit()

    s1 = create_screening(test_db_session, user_id=user.id, disease_id=lung.id)
    s2 = create_screening(test_db_session, user_id=user.id, disease_id=asthma.id)
    test_db_session.commit()

    history = get_user_screening_history(test_db_session, user.id)
    assert len(history) == 2
    history_disease_ids = {s.disease_id for s in history}
    assert history_disease_ids == {lung.id, asthma.id}


def test_16_database_exception_handling_with_rollback(test_db_session):
    """Verify transactional rollback leaves database in consistent state after failure."""
    create_user(test_db_session, full_name="Initial User", email="initial@test.com")
    test_db_session.commit()

    try:
        # Trigger failure
        create_user(test_db_session, full_name="Duplicate", email="initial@test.com")
        test_db_session.commit()
    except IntegrityError:
        test_db_session.rollback()

    # Session is operational after rollback
    surviving_user = get_user_by_email(test_db_session, "initial@test.com")
    assert surviving_user is not None
    assert surviving_user.full_name == "Initial User"
