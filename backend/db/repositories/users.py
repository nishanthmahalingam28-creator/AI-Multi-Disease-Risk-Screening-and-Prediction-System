"""User and Clinic User Data Access Repositories."""

from datetime import date
from typing import Optional
from sqlalchemy.orm import Session

from backend.db.models import ClinicUser, User


def create_user(
    session: Session,
    full_name: str,
    email: str,
    mobile: Optional[str] = None,
    date_of_birth: Optional[date] = None,
) -> User:
    """Create a new registered patient user record."""
    user = User(
        full_name=full_name.strip(),
        email=email.strip().lower(),
        mobile=mobile.strip() if mobile else None,
        date_of_birth=date_of_birth,
    )
    session.add(user)
    session.flush()
    return user


def get_user_by_id(session: Session, user_id: str) -> Optional[User]:
    """Retrieve a user by their UUID primary key."""
    return session.query(User).filter(User.id == str(user_id)).first()


def get_user_by_email(session: Session, email: str) -> Optional[User]:
    """Retrieve a user by their unique email address."""
    return session.query(User).filter(User.email == email.strip().lower()).first()


def create_clinic_user(
    session: Session,
    full_name: str,
    email: str,
    clinic_name: str,
    mobile: Optional[str] = None,
) -> ClinicUser:
    """Create a new healthcare clinician/staff profile."""
    clinic_user = ClinicUser(
        full_name=full_name.strip(),
        email=email.strip().lower(),
        clinic_name=clinic_name.strip(),
        mobile=mobile.strip() if mobile else None,
    )
    session.add(clinic_user)
    session.flush()
    return clinic_user


def get_clinic_user_by_id(session: Session, clinic_user_id: str) -> Optional[ClinicUser]:
    """Retrieve a clinic user by their UUID primary key."""
    return session.query(ClinicUser).filter(ClinicUser.id == str(clinic_user_id)).first()


def get_clinic_user_by_email(session: Session, email: str) -> Optional[ClinicUser]:
    """Retrieve a clinic user by their unique email address."""
    return session.query(ClinicUser).filter(ClinicUser.email == email.strip().lower()).first()
