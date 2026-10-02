/**
 * User Dashboard Page.
 * 
 * Displays personalized welcome banner, system explanation, and cards for
 * all 10 planned diseases (3 active and operational, 7 tagged Coming Soon).
 */

import { PLANNED_DISEASES } from '../types/constants.js';
import { createDiseaseCard } from '../components/DiseaseCard.js';
import { renderHeader } from '../components/Header.js';
import { store } from '../hooks/store.js';

export function renderDashboardPage(container) {
  const { user } = store.state;

  container.innerHTML = `
    <div class="dashboard-page">
      <div id="dashboard-header-container"></div>

      <div class="welcome-card">
        <div class="welcome-text">
          <h2>Hello, ${user ? user.fullName : 'Valued Patient'} 👋</h2>
          <p>
            Welcome to the <strong>AI-Powered Multi-Disease Risk Screening Platform</strong>.
            This system evaluates standardized clinical features, demographic risk factors,
            and specialized physiological/acoustic biomarkers through trained machine learning models
            to assist in early risk detection and clinical vigilance.
          </p>
          <div class="welcome-notice">
            <span>🛡️ <strong>Note on Scope:</strong> These screenings provide preventive statistical risk indications. They do not replace formal clinical examinations or diagnostic lab tests.</span>
          </div>
        </div>
      </div>

      <div class="disease-section-header">
        <div class="section-title-wrap">
          <h2 class="section-title">Multi-Disease Screening Catalog</h2>
          <span class="section-subtitle">Select an available condition to launch a clinical risk questionnaire</span>
        </div>
        <div class="catalog-legend">
          <span class="legend-item"><span class="badge badge-success">3 Operational</span> Ready for Inference</span>
          <span class="legend-item"><span class="badge badge-soon">7 In Development</span> Coming Soon</span>
        </div>
      </div>

      <div class="disease-grid" id="disease-catalog-grid"></div>
    </div>
  `;

  // Render header
  const headerContainer = container.querySelector('#dashboard-header-container');
  if (headerContainer) {
    renderHeader(headerContainer, {
      title: 'Patient Screening Dashboard',
      subtitle: 'Early detection AI models across ten chronic and critical health conditions.',
      userName: user ? user.fullName : '',
    });
  }

  // Render disease cards
  const grid = container.querySelector('#disease-catalog-grid');
  if (grid) {
    PLANNED_DISEASES.forEach((disease) => {
      const card = createDiseaseCard(disease);
      grid.appendChild(card);
    });
  }
}
