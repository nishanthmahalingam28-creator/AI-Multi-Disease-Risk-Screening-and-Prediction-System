/**
 * Clinic Dashboard Page.
 * 
 * Displays clinician metadata, clinic center affiliation, workflow instructions,
 * and 10 disease cards (3 operational, 7 tagged Coming Soon).
 */

import { PLANNED_DISEASES } from '../types/constants.js';
import { createClinicDiseaseCard } from '../components/ClinicCard.js';
import { renderClinicHeader } from '../components/Header.js';
import { clinicStore } from '../hooks/store.js';

export function renderClinicDashboardPage(container) {
  const { clinicUser } = clinicStore.state;

  container.innerHTML = `
    <div class="dashboard-page clinic-dashboard">
      <div id="clinic-dashboard-header-container"></div>

      <div class="welcome-card clinic-welcome-card" style="border-left: 4px solid var(--accent-teal);">
        <div class="welcome-text">
          <h2>Clinical Consultation & Screening Console</h2>
          <p>
            Welcome, <strong>${clinicUser ? clinicUser.fullName : 'Practitioner'}</strong> (${clinicUser ? clinicUser.clinicName : 'Healthcare Center'}).
            This console enables authorized healthcare providers to conduct patient-assisted risk screening across standardized clinical parameters.
          </p>
          <div class="welcome-notice" style="background: rgba(20, 184, 166, 0.1); border-left-color: var(--accent-teal); color: #5eead4;">
            <span>📋 <strong>Clinical Workflow:</strong> Select an available condition below, enter the patient's demographic information, input verified clinical measurements, and generate an AI-assisted risk screening estimation.</span>
          </div>
        </div>
      </div>

      <div class="disease-section-header">
        <div class="section-title-wrap">
          <h2 class="section-title">Clinical Screening Capabilities</h2>
          <span class="section-subtitle">Select an operational disease model to begin a patient assessment session</span>
        </div>
        <div class="catalog-legend">
          <span class="legend-item"><span class="badge badge-success">3 Operational Models</span></span>
          <span class="legend-item"><span class="badge badge-soon">7 Clinical Trial Phase</span></span>
        </div>
      </div>

      <div class="disease-grid" id="clinic-disease-catalog-grid"></div>
    </div>
  `;

  // Render header
  const headerContainer = container.querySelector('#clinic-dashboard-header-container');
  if (headerContainer) {
    renderClinicHeader(headerContainer, {
      title: `Clinic Dashboard — ${clinicUser ? clinicUser.clinicName : 'Healthcare Facility'}`,
      subtitle: `Authorized Practitioner: ${clinicUser ? clinicUser.fullName : 'Provider'} | Active Shift`,
      clinicUser,
    });
  }

  // Render cards
  const grid = container.querySelector('#clinic-disease-catalog-grid');
  if (grid) {
    PLANNED_DISEASES.forEach((disease) => {
      const card = createClinicDiseaseCard(disease);
      grid.appendChild(card);
    });
  }
}
