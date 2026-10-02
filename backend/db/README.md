# Supabase Database Architecture & SQLAlchemy Data Layer

## 1. Overview
This module implements the database persistence foundation for the **AI Multi-Disease Risk Screening and Prediction System**.

- **Database Platform**: **Supabase**
- **Database Engine**: **PostgreSQL 15+**
- **Backend ORM / Data Layer**: **SQLAlchemy 2.0+**
- **Architectural Policy**: **Firebase is NOT used**. The system uses Supabase PostgreSQL as its single source of truth for persistent clinical data.

> **CRITICAL ARCHITECTURAL BOUNDARY**:
> **DATABASE PERSISTENCE IS NOT YET CONNECTED TO THE STEP 10 PREDICTION ENDPOINTS.**
> The Step 10 Flask prediction endpoints (`/api/predict/*`) remain completely stateless and independent of live database connections. Database persistence will be integrated in subsequent user/clinic session management workflows.

---

## 2. Environment Configuration
The database connection string is supplied via standard environment variables:

```bash
DATABASE_URL=postgresql+psycopg://postgres:<PASSWORD>@db.<PROJECT_REF>.supabase.co:5432/postgres
```

### Safety & Secret Protection
- **No Hardcoded Credentials**: Passwords, project references, and JWT secrets are strictly read from environment variables and never checked into source control.
- **Git Protection**: `.gitignore` explicitly protects `.env`, `.env.local`, and credential files.
- **Local / Test Fallback**: In automated testing and local development where a live Supabase instance is unavailable, `get_database_url()` provides a safe isolated fallback, preventing accidental mutations to production data.

---

## 3. Database Schema & Tables

The schema defines six core relational tables:

```text
users (Patients)
  │
  └── screenings (Screening Sessions)
          ├── diseases (Reference Catalog: 10 Planned Diseases)
          ├── clinic_users (Healthcare Staff / Clinicians)
          ├── screening_inputs (Submitted Feature JSON/JSONB)
          └── prediction_results (Model Inference Outputs)
```

| Table Name | Description | Key Columns |
| :--- | :--- | :--- |
| `users` | Registered patient user profiles (associates with Supabase Auth `auth.users.id`). | `id` (UUID, PK), `full_name`, `email` (Unique), `mobile`, `date_of_birth`, `is_active`, `created_at`, `updated_at` |
| `clinic_users` | Healthcare staff and clinic professionals assisting patients during on-site screenings. | `id` (UUID, PK), `full_name`, `email` (Unique), `mobile`, `clinic_name`, `is_active`, `created_at`, `updated_at` |
| `diseases` | Reference catalog for all 10 planned diseases. | `id` (Serial, PK), `code` (Unique), `name`, `description`, `is_active`, `created_at` |
| `screenings` | Individual screening event (supports both self-screening and clinic-assisted screenings). | `id` (UUID, PK), `user_id` (FK $\to$ `users`), `disease_id` (FK $\to$ `diseases`), `clinic_user_id` (FK $\to$ `clinic_users`, Nullable), `status`, `created_at`, `updated_at` |
| `screening_inputs`| Complete submitted feature payload preserved in native PostgreSQL `JSONB`. | `id` (UUID, PK), `screening_id` (FK $\to$ `screenings`, Unique), `input_data` (JSONB), `created_at` |
| `prediction_results`| Statistical inference result with probability bounds constraint. | `id` (UUID, PK), `screening_id` (FK $\to$ `screenings`, Unique), `predicted_class`, `probability` ($0.0 \le p \le 1.0$), `risk_level`, `model_version`, `disclaimer`, `created_at` |

---

## 4. Foreign Key Constraints & Data Integrity
- **Cascade Deletion**: When a user record is purged, child `screenings`, `screening_inputs`, and `prediction_results` cascade cleanly.
- **Disease Protection**: `disease_id` foreign key is restricted (`ON DELETE RESTRICT`) to prevent accidental deletion of disease categories with active screening records.
- **Clinic Decoupling**: If a `clinic_user` is removed, past screening history is preserved with `clinic_user_id` set to `NULL` (`ON DELETE SET NULL`).
- **Probability Boundary Constraint**: `ck_valid_probability` database-level check constraint strictly enforces $0.0 \le \text{probability} \le 1.0$.

---

## 5. Reference Seed Data: 10 Planned Diseases
All 10 multi-disease categories are seeded in the reference database:
1. `breast_cancer`: Breast Cancer
2. `diabetes`: Diabetes
3. `heart_disease`: Heart Disease
4. `stroke`: Stroke
5. `kidney`: Chronic Kidney Disease
6. `liver`: Liver Disease
7. `thyroid`: Thyroid Disease
8. `lung_cancer`: Lung Cancer
9. `asthma`: Asthma
10. `parkinsons`: Parkinson's Disease

*Distinction*: An entry in `diseases` indicates that the database schema supports the condition. Active API screening availability is independently determined by Member 3 model implementation status (`lung_cancer`, `asthma`, `parkinsons`).

---

## 6. Supabase Security & Row Level Security (RLS)
- **RLS Enforced by Default**: All application tables have RLS enabled via `ALTER TABLE ... ENABLE ROW LEVEL SECURITY;`.
- **Public Read on Reference Catalog**: Public read is permitted strictly on active entries in the `diseases` reference table.
- **Medical Privacy by Default**: Sensitive patient profiles, screening inputs, and risk predictions are protected under a deny-all posture for unauthenticated public requests. No insecure `USING (true)` wildcard policies exist. Dedicated user-scoped policies (`auth.uid() = user_id`) will be applied upon completion of the authentication integration step.

---

## 7. Migration Structure
Migrations are located in `supabase/migrations/` using timestamped SQL files:
- `supabase/migrations/20261002_000001_initial_schema.sql`: Complete DDL, indexes, constraints, RLS enablement, and seed data.

---

## 8. Automated Testing Strategy
Automated testing uses an in-memory SQLite engine to evaluate models, schema constraints, relationships, and repository CRUD logic in complete isolation from the production Supabase database. Zero live credentials or destructive production actions are required for testing.
