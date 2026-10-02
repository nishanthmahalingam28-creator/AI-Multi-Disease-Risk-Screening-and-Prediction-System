/**
 * Disease Card Component.
 * 
 * Renders an interactive card for each of the 10 planned diseases.
 * Available models are clearly distinguished with an action button;
 * unavailable models are tagged with a prominent "Coming Soon" badge.
 */

import { store } from '../hooks/store.js';

export function createDiseaseCard(disease) {
  const isAvailable = disease.available;

  const card = document.createElement('div');
  card.className = `disease-card ${isAvailable ? 'card-available' : 'card-unavailable'}`;
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
        <span class="info-label">Inputs:</span>
        <span class="info-val">${disease.featuresCount} clinical features</span>
      </div>
      ${isAvailable ? `
        <button class="btn btn-primary btn-block btn-start-screening" data-code="${disease.code}">
          Start Screening →
        </button>
      ` : `
        <button class="btn btn-disabled btn-block" disabled title="Model training and validation in progress">
          🔒 Coming Soon
        </button>
      `}
    </div>
  `;

  if (isAvailable) {
    const btn = card.querySelector('.btn-start-screening');
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      store.navigate('screening', { disease: disease.code });
    });
  }

  return card;
}
