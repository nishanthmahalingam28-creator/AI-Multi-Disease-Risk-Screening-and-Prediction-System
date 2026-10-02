"""SQLAlchemy ORM Data Models for Supabase PostgreSQL Database.

Implements core domain entities:
1. User (Patients)
2. ClinicUser (Healthcare Staff)
3. Disease (Reference Catalog for all 10 planned diseases)
4. Screening (Screening Sessions: Patient self-screening or clinic-assisted)
5. ScreeningInput (PostgreSQL JSON/JSONB input payloads)
6. PredictionResult (Model inference outputs with probability boundary constraints)
"""

import uuid
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from backend.db.database import Base


class User(Base):
    """Registered patient user profile.

    Designed to associate with Supabase Auth (auth.users.id) in subsequent steps.
    Does NOT store plaintext passwords.
    """

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    mobile = Column(String(50), nullable=True)
    date_of_birth = Column(Date, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    screenings = relationship("Screening", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id='{self.id}', email='{self.email}')>"


class ClinicUser(Base):
    """Clinical staff user profile assisting patients during in-clinic screening."""

    __tablename__ = "clinic_users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    mobile = Column(String(50), nullable=True)
    clinic_name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    assisted_screenings = relationship("Screening", back_populates="clinic_user")

    def __repr__(self) -> str:
        return f"<ClinicUser(id='{self.id}', clinic='{self.clinic_name}')>"


class Disease(Base):
    """Reference catalog table supporting all 10 planned multi-disease categories.

    Note: A disease existing in this reference catalog does not imply its prediction
    model is currently active in the API. Active models are governed by API availability.
    """

    __tablename__ = "diseases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    # Relationships
    screenings = relationship("Screening", back_populates="disease")

    def __repr__(self) -> str:
        return f"<Disease(id={self.id}, code='{self.code}')>"


class Screening(Base):
    """Unified screening record supporting both patient self-screening and clinic-assisted screening."""

    __tablename__ = "screenings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    disease_id = Column(Integer, ForeignKey("diseases.id", ondelete="RESTRICT"), nullable=False, index=True)
    clinic_user_id = Column(
        String(36), ForeignKey("clinic_users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status = Column(String(50), default="COMPLETED", nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="screenings")
    disease = relationship("Disease", back_populates="screenings")
    clinic_user = relationship("ClinicUser", back_populates="assisted_screenings")
    screening_input = relationship(
        "ScreeningInput", back_populates="screening", uselist=False, cascade="all, delete-orphan"
    )
    prediction_result = relationship(
        "PredictionResult", back_populates="screening", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Screening(id='{self.id}', user='{self.user_id}', disease={self.disease_id})>"


class ScreeningInput(Base):
    """Submitted patient clinical and biometric input features preserved as JSON/JSONB."""

    __tablename__ = "screening_inputs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    screening_id = Column(
        String(36), ForeignKey("screenings.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    input_data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    # Relationships
    screening = relationship("Screening", back_populates="screening_input")

    def __repr__(self) -> str:
        return f"<ScreeningInput(id='{self.id}', screening='{self.screening_id}')>"


class PredictionResult(Base):
    """Statistical inference result associated with a completed screening."""

    __tablename__ = "prediction_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    screening_id = Column(
        String(36), ForeignKey("screenings.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    predicted_class = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    risk_level = Column(String(50), nullable=False)
    model_version = Column(String(50), default="1.0.0", nullable=True)
    disclaimer = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    # Database-level probability constraint: 0.0 <= probability <= 1.0
    __table_args__ = (
        CheckConstraint(
            "probability >= 0.0 AND probability <= 1.0",
            name="ck_valid_probability",
        ),
    )

    # Relationships
    screening = relationship("Screening", back_populates="prediction_result")

    def __repr__(self) -> str:
        return f"<PredictionResult(id='{self.id}', risk='{self.risk_level}', prob={self.probability})>"
