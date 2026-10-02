/**
 * Constants and reference definitions for the AI Multi-Disease Screening System.
 * 
 * Defines all 10 planned diseases, distinguishing available prediction models
 * from upcoming models, and provides medical screening disclaimers.
 */

export const PLANNED_DISEASES = [
  {
    code: 'lung_cancer',
    name: 'Lung Cancer Risk Screening',
    description: 'AI-assisted clinical risk assessment based on demographic, lifestyle, and symptomatic indicators.',
    available: true,
    endpoint: '/predict/lung-cancer',
    icon: '🫁',
    category: 'Pulmonary & Oncology',
    featuresCount: 15,
  },
  {
    code: 'asthma',
    name: 'Asthma Risk Screening',
    description: 'AI screening using physiological biomarkers, spirometry (PEF, FeNO), and clinical history.',
    available: true,
    endpoint: '/predict/asthma',
    icon: '💨',
    category: 'Respiratory Health',
    featuresCount: 14,
  },
  {
    code: 'parkinsons',
    name: "Parkinson's Disease Risk Screening",
    description: 'Non-invasive acoustic voice phonation analysis assessing frequency, jitter, shimmer, and entropy.',
    available: true,
    endpoint: '/predict/parkinsons',
    icon: '🧠',
    category: 'Neurology',
    featuresCount: 22,
  },
  {
    code: 'breast_cancer',
    name: 'Breast Cancer Risk Screening',
    description: 'Cell nucleus morphological characteristics for diagnostic risk prediction.',
    available: false,
    endpoint: null,
    icon: '🎗️',
    category: 'Oncology',
    featuresCount: 30,
  },
  {
    code: 'diabetes',
    name: 'Diabetes Risk Screening',
    description: 'Metabolic markers, glycemic indices, insulin levels, and BMI assessment.',
    available: false,
    endpoint: null,
    icon: '🩸',
    category: 'Endocrinology',
    featuresCount: 8,
  },
  {
    code: 'heart_disease',
    name: 'Heart Disease Risk Screening',
    description: 'Cardiovascular hemodynamic profiling, ECG findings, and coronary risk factors.',
    available: false,
    endpoint: null,
    icon: '❤️',
    category: 'Cardiology',
    featuresCount: 13,
  },
  {
    code: 'stroke',
    name: 'Stroke Risk Screening',
    description: 'Cerebrovascular risk factor screening including hypertension, glucose, and smoking history.',
    available: false,
    endpoint: null,
    icon: '⚡',
    category: 'Neurology',
    featuresCount: 10,
  },
  {
    code: 'kidney',
    name: 'Chronic Kidney Disease Risk Screening',
    description: 'Renal function screening measuring serum creatinine, albumin, urea, and specific gravity.',
    available: false,
    endpoint: null,
    icon: '🫧',
    category: 'Nephrology',
    featuresCount: 24,
  },
  {
    code: 'liver',
    name: 'Liver Disease Risk Screening',
    description: 'Hepatic biochemical screening measuring bilirubin, enzymes (ALT, AST), and proteins.',
    available: false,
    endpoint: null,
    icon: '🧪',
    category: 'Hepatology',
    featuresCount: 10,
  },
  {
    code: 'thyroid',
    name: 'Thyroid Disease Risk Screening',
    description: 'Endocrine screening assessing TSH, T3, T4 hormone levels and clinical thyroid symptoms.',
    available: false,
    endpoint: null,
    icon: '🦋',
    category: 'Endocrinology',
    featuresCount: 21,
  },
];

export const MANDATORY_DISCLAIMER =
  'This result is an AI-based screening/risk estimate and is not a medical diagnosis. Please consult a qualified healthcare professional for medical advice, diagnosis, or treatment.';

export const RISK_TIERS = {
  LOW: {
    label: 'Low Risk',
    class: 'risk-low',
    color: '#10b981',
    description: 'Low statistical likelihood of disease based on provided screening indicators.',
  },
  MODERATE: {
    label: 'Moderate Risk',
    class: 'risk-moderate',
    color: '#f59e0b',
    description: 'Moderate statistical likelihood. Observational monitoring or follow-up evaluation is advisable.',
  },
  HIGH: {
    label: 'High Risk',
    class: 'risk-high',
    color: '#ef4444',
    description: 'High statistical likelihood. Priority clinical consultation with a healthcare specialist is recommended.',
  },
};
