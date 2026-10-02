/**
 * Clinic Disease Card Component.
 * 
 * Renders an interactive screening trigger card for each condition in the clinic catalog.
 */

import { clinicStore } from '../hooks/store.js';

export function createClinicDiseaseCard(disease) {
  const isAvailable = disease.available;

  const card = document.createElement('div');
  card.className = `disease-card clinic-disease-card ${isAvailable ? 'card-available' : 'card-unavailable'}`;
  card.setAttribute('data-disease-code', disease.code);
  card.setAttribute('role', 'region');
  card.setAttribute('aria-label', disease.name);

  card.innerHTML = `
    <div class="card-header">
      <div class="card-icon-wrapper">
        <span class="card-icon">${disease.icon}</span>
      </div>
      <div class="card-status-badge">
        ${isAvailable 
          ? `<span class="badge badge-success">Available Now</span>` 
          : `<span class="badge badge-soon">Coming Soon</span>`
        }
      </div>
    </div>

    <div class="card-body">
      <span class="card-category">${disease.category}</span>
      <h3 class="card-title">${disease.name}</h3>
      <p class="card-description">${disease.description}</p>
    </div>

    <div class="card-footer">
      <div class="card-features-info">
        <span class="info-label">Clinical Inputs:</span>
        <span class="info-val">${disease.featuresCount} validated features</span>
      </div>
      ${isAvailable ? `
        <button class="btn btn-primary btn-block btn-clinic-start" data-code="${disease.code}">
          Start Patient Screening →
        </button>
      ` : `
        <button class="btn btn-disabled btn-block" disabled title="Inference model under validation">
          🔒 Coming Soon
        </button>
      `}
    </div>
  `;

  if (isAvailable) {
    const btn = card.querySelector('.btn-clinic-start');
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      clinicStore.navigate('screening', { disease: disease.code });
    });
  }

  return card;
}
