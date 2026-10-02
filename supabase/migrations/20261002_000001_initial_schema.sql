-- =============================================================================
-- Migration: 20261002_000001_initial_schema.sql
-- Description: Initial Supabase PostgreSQL Schema for Multi-Disease Screening
-- Engine: PostgreSQL 15+ (Supabase)
-- =============================================================================

-- Enable pgcrypto extension for UUID generation if not already active
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- -----------------------------------------------------------------------------
-- 1. Table: users (Patients)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    mobile VARCHAR(50),
    date_of_birth DATE,
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT clock_timestamp() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT clock_timestamp() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- -----------------------------------------------------------------------------
-- 2. Table: clinic_users (Healthcare Staff / Clinicians)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clinic_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    mobile VARCHAR(50),
    clinic_name VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT clock_timestamp() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT clock_timestamp() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_clinic_users_email ON clinic_users(email);

-- -----------------------------------------------------------------------------
-- 3. Table: diseases (Reference Catalog for all 10 planned diseases)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS diseases (
    id SERIAL PRIMARY KEY,
    code VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT clock_timestamp() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_diseases_code ON diseases(code);

-- -----------------------------------------------------------------------------
-- 4. Table: screenings (Screening Sessions)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS screenings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    disease_id INTEGER NOT NULL REFERENCES diseases(id) ON DELETE RESTRICT,
    clinic_user_id UUID REFERENCES clinic_users(id) ON DELETE SET NULL,
    status VARCHAR(50) DEFAULT 'COMPLETED' NOT NULL,
    created_at TIMESTAMPTZ DEFAULT clock_timestamp() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT clock_timestamp() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_screenings_user_id ON screenings(user_id);
CREATE INDEX IF NOT EXISTS idx_screenings_disease_id ON screenings(disease_id);
CREATE INDEX IF NOT EXISTS idx_screenings_clinic_user_id ON screenings(clinic_user_id);

-- -----------------------------------------------------------------------------
-- 5. Table: screening_inputs (Patient Feature Payload)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS screening_inputs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    screening_id UUID UNIQUE NOT NULL REFERENCES screenings(id) ON DELETE CASCADE,
    input_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT clock_timestamp() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_screening_inputs_screening_id ON screening_inputs(screening_id);

-- -----------------------------------------------------------------------------
-- 6. Table: prediction_results (Inference Output)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    screening_id UUID UNIQUE NOT NULL REFERENCES screenings(id) ON DELETE CASCADE,
    predicted_class INTEGER NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    risk_level VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) DEFAULT '1.0.0',
    disclaimer TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT clock_timestamp() NOT NULL,
    CONSTRAINT ck_valid_probability CHECK (probability >= 0.0 AND probability <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_prediction_results_screening_id ON prediction_results(screening_id);

-- =============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =============================================================================
-- Enable RLS across all application tables by default
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE clinic_users ENABLE ROW LEVEL SECURITY;
ALTER TABLE diseases ENABLE ROW LEVEL SECURITY;
ALTER TABLE screenings ENABLE ROW LEVEL SECURITY;
ALTER TABLE screening_inputs ENABLE ROW LEVEL SECURITY;
ALTER TABLE prediction_results ENABLE ROW LEVEL SECURITY;

-- 1. Disease Catalog: Publicly readable for active categories
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies WHERE tablename = 'diseases' AND policyname = 'Public read for active diseases'
    ) THEN
        CREATE POLICY "Public read for active diseases" ON diseases
            FOR SELECT USING (is_active = true);
    END IF;
END $$;

-- 2. Sensitive Patient & Screening Data:
-- Secure by default (deny-all to unauthenticated public access).
-- Dedicated user-scoped policies (e.g., auth.uid() = user_id) will be attached
-- once Supabase Auth identity federation is established in subsequent steps.
-- Unsafe wildcard policies (e.g. USING (true)) are strictly prohibited.

-- =============================================================================
-- REFERENCE SEED DATA: 10 Planned Disease Categories
-- =============================================================================
INSERT INTO diseases (code, name, description, is_active)
VALUES
    ('breast_cancer', 'Breast Cancer', 'Biomarker risk screening based on tissue biopsy cell nuclei features.', TRUE),
    ('diabetes', 'Diabetes', 'Type 2 diabetes risk screening based on metabolic, glucose, and biometric indices.', TRUE),
    ('heart_disease', 'Heart Disease', 'Cardiovascular risk screening based on clinical vitals, cholesterol, and cardiac metrics.', TRUE),
    ('stroke', 'Stroke', 'Cerebrovascular risk assessment based on physiological factors, hypertension, and medical history.', TRUE),
    ('kidney', 'Chronic Kidney Disease', 'Renal function screening assessing filtration, electrolytes, and urinalysis profiles.', TRUE),
    ('liver', 'Liver Disease', 'Hepatic risk assessment utilizing liver enzyme ratios and metabolic indicators.', TRUE),
    ('thyroid', 'Thyroid Disease', 'Endocrine screening for hypothyroidism and thyroid dysfunction risk.', TRUE),
    ('lung_cancer', 'Lung Cancer', 'Pulmonary risk screening analyzing demographic risks, respiratory symptoms, and exposures.', TRUE),
    ('asthma', 'Asthma', 'Respiratory risk screening based on inflammatory biomarkers, spirometry, and clinical triggers.', TRUE),
    ('parkinsons', 'Parkinson''s Disease', 'Acoustic dysphonia analysis of sustained phonation to detect neurodegenerative vocal tremors.', TRUE)
ON CONFLICT (code) DO NOTHING;
