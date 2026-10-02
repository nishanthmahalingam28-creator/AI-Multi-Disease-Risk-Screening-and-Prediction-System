/**
 * Clinic Screening Result Card Component.
 * 
 * Displays model prediction outcome, probability gauge, risk level badge,
 * patient identity metadata, assisting practitioner details, and mandatory clinical disclaimer.
 */

import { MANDATORY_CLINICAL_DISCLAIMER, RISK_TIERS } from '../types/constants.js';
import { clinicStore } from '../hooks/store.js';

export function renderClinicResultCard(container, { disease, result, patientInfo, clinicUser }) {
  if (!result) {
    container.innerHTML = `
      <div class="empty-state">
        <p>No clinical prediction result available.</p>
        <button class="btn btn-primary" id="btn-clinic-back-dash">Return to Dashboard</button>
      </div>
    `;
    container.querySelector('#btn-clinic-back-dash')?.addEventListener('click', () => {
      clinicStore.navigate('dashboard');
    });
    return;
  }

  const data = result.prediction || result;
  const rawLevel = (data.risk_level || 'LOW').toUpperCase();
  const tierInfo = RISK_TIERS[rawLevel] || RISK_TIERS.LOW;
  const probabilityPct = data.probability !== undefined ? (data.probability * 100).toFixed(1) : '0.0';
  const isPositive = data.predicted_class === 1;

  const patientName = patientInfo ? patientInfo.fullName : 'Confidential Patient';
  const patientEmail = patientInfo ? patientInfo.email : 'N/A';
  const doctorName = clinicUser ? clinicUser.fullName : 'Authorized Provider';
  const clinicName = clinicUser ? clinicUser.clinicName : 'Healthcare Facility';

  container.innerHTML = `
    <div class="result-card clinic-result-card" role="region" aria-label="Clinical Screening Result">
      <div class="result-header">
        <div class="result-disease-meta">
          <span class="result-icon">${disease.icon}</span>
          <div>
            <span class="result-category">${disease.category}</span>
            <h2 class="result-disease-title">${disease.name}</h2>
          </div>
        </div>
        <div class="risk-badge-wrapper">
          <span class="badge ${tierInfo.class} badge-lg">
            ● ${tierInfo.label}
          </span>
        </div>
      </div>

      <!-- Clinical Session Context -->
      <div class="clinical-context-panel" style="background: rgba(2, 132, 199, 0.08); border: 1px solid rgba(2, 132, 199, 0.25); border-radius: var(--radius-md); padding: 16px 20px; margin-bottom: 22px;">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; font-size: 0.88rem;">
          <div>
            <span style="color: var(--text-dim); display: block; font-size: 0.75rem; text-transform: uppercase;">Patient Reference:</span>
            <strong>${patientName}</strong> (${patientEmail})
          </div>
          <div>
            <span style="color: var(--text-dim); display: block; font-size: 0.75rem; text-transform: uppercase;">Assisting Clinician:</span>
            <strong>${doctorName}</strong>
          </div>
          <div>
            <span style="color: var(--text-dim); display: block; font-size: 0.75rem; text-transform: uppercase;">Clinical Center:</span>
            <strong>${clinicName}</strong>
          </div>
        </div>
      </div>

      <div class="result-score-panel">
        <div class="score-meta">
          <span class="score-label">Clinical Risk Likelihood</span>
          <span class="score-value" style="color: ${tierInfo.color}">${probabilityPct}%</span>
        </div>
        <div class="probability-track">
          <div class="probability-fill" style="width: ${probabilityPct}%; background-color: ${tierInfo.color}"></div>
        </div>
        <div class="probability-scale">
          <span>0% (Low Risk)</span>
          <span>50% (Moderate Risk)</span>
          <span>100% (Elevated Risk)</span>
        </div>
      </div>

      <div class="result-details-grid">
        <div class="detail-box">
          <span class="detail-label">Model Classification</span>
          <span class="detail-val ${isPositive ? 'text-danger' : 'text-success'}">
            ${isPositive ? 'Screening Positive (Elevated Statistical Likelihood)' : 'Screening Negative (Low Statistical Likelihood)'}
          </span>
        </div>

        <div class="detail-box">
          <span class="detail-label">Model Version</span>
          <span class="detail-val font-mono">${data.model_version || '1.0.0'}</span>
        </div>

        <div class="detail-box full-width">
          <span class="detail-label">Clinical Interpretation</span>
          <p class="detail-explanation">
            ${tierInfo.description} The verified machine learning screening pipeline computed an estimated risk probability of ${probabilityPct}%.
          </p>
        </div>
      </div>

      <div class="disclaimer-banner" role="note">
        <div class="disclaimer-icon">⚠️</div>
        <div class="disclaimer-text">
          <strong>Mandatory Medical Notice:</strong>
          ${data.disclaimer || MANDATORY_CLINICAL_DISCLAIMER}
        </div>
      </div>

      <div class="result-actions">
        <button class="btn btn-outline" id="btn-screen-another-patient">
          🔄 Screen Another Patient
        </button>
        <button class="btn btn-secondary" id="btn-view-clinic-report">
          🖨️ View Clinical Report
        </button>
        <button class="btn btn-primary" id="btn-return-clinic-dash">
          📊 Return to Clinic Dashboard
        </button>
      </div>
    </div>
  `;

  container.querySelector('#btn-screen-another-patient')?.addEventListener('click', () => {
    clinicStore.navigate('dashboard');
  });

  container.querySelector('#btn-view-clinic-report')?.addEventListener('click', () => {
    clinicStore.navigate('report', { disease, result, patientInfo, clinicUser, screeningId: result.screening_id });
  });

  container.querySelector('#btn-return-clinic-dash')?.addEventListener('click', () => {
    clinicStore.navigate('dashboard');
  });
}

