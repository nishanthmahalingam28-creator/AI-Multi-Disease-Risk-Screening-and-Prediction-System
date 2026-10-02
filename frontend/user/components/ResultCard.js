/**
 * Prediction Screening Result Card Component.
 * 
 * Presents model inference outcome, risk level badge, probability gauge,
 * model metadata, and prominent clinical screening disclaimer.
 */

import { MANDATORY_DISCLAIMER, RISK_TIERS } from '../types/constants.js';
import { store } from '../hooks/store.js';

export function renderResultCard(container, { disease, result }) {
  if (!result) {
    container.innerHTML = `
      <div class="empty-state">
        <p>No screening result available to display.</p>
        <button class="btn btn-primary" id="btn-back-dashboard">Return to Dashboard</button>
      </div>
    `;
    container.querySelector('#btn-back-dashboard')?.addEventListener('click', () => {
      store.navigate('dashboard');
    });
    return;
  }

  const rawLevel = (result.risk_level || 'LOW').toUpperCase();
  const tierInfo = RISK_TIERS[rawLevel] || RISK_TIERS.LOW;
  const probabilityPct = (result.probability * 100).toFixed(1);
  const isPositive = result.predicted_class === 1;

  container.innerHTML = `
    <div class="result-card" role="region" aria-label="Screening Result">
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

      <div class="result-score-panel">
        <div class="score-meta">
          <span class="score-label">Estimated Risk Probability</span>
          <span class="score-value" style="color: ${tierInfo.color}">${probabilityPct}%</span>
        </div>
        <div class="probability-track">
          <div class="probability-fill" style="width: ${probabilityPct}%; background-color: ${tierInfo.color}"></div>
        </div>
        <div class="probability-scale">
          <span>0% (Low)</span>
          <span>50% (Moderate)</span>
          <span>100% (High)</span>
        </div>
      </div>

      <div class="result-details-grid">
        <div class="detail-box">
          <span class="detail-label">Screening Classification</span>
          <span class="detail-val ${isPositive ? 'text-danger' : 'text-success'}">
            ${isPositive ? 'Screening Positive (High Risk Indicator)' : 'Screening Negative (Low Risk Indicator)'}
          </span>
        </div>

        <div class="detail-box">
          <span class="detail-label">Model Version</span>
          <span class="detail-val font-mono">${result.model_version || '1.0.0'}</span>
        </div>

        <div class="detail-box full-width">
          <span class="detail-label">Clinical Interpretation Summary</span>
          <p class="detail-explanation">
            ${tierInfo.description} The machine learning screening pipeline processed all validated clinical features and computed a statistical disease risk probability of ${probabilityPct}%.
          </p>
        </div>
      </div>

      <div class="disclaimer-banner" role="note">
        <div class="disclaimer-icon">⚠️</div>
        <div class="disclaimer-text">
          <strong>Mandatory Medical Notice:</strong>
          ${result.disclaimer || MANDATORY_DISCLAIMER}
        </div>
      </div>

      <div class="result-actions">
        <button class="btn btn-outline" id="btn-screen-again">
          🔄 Screen Another Disease
        </button>
        <button class="btn btn-secondary" id="btn-view-user-report">
          🖨️ View Printable Report
        </button>
        <button class="btn btn-primary" id="btn-return-dash">
          📊 Return to Dashboard
        </button>
      </div>
    </div>
  `;

  container.querySelector('#btn-screen-again')?.addEventListener('click', () => {
    store.navigate('dashboard');
  });

  container.querySelector('#btn-view-user-report')?.addEventListener('click', () => {
    store.navigate('report', { disease, result, screeningId: result.screening_id });
  });

  container.querySelector('#btn-return-dash')?.addEventListener('click', () => {
    store.navigate('dashboard');
  });
}

