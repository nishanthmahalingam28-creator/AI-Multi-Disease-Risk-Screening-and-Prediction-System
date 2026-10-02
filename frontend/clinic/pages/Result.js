/**
 * Clinic Screening Result Page.
 * 
 * Embeds renderClinicResultCard displaying model prediction, patient reference,
 * practitioner attribution, probability gauge, risk level, and medical disclaimer.
 */

import { renderClinicResultCard } from '../components/ResultCard.js';
import { renderClinicHeader } from '../components/Header.js';
import { clinicStore } from '../hooks/store.js';

export function renderClinicResultPage(container) {
  const { selectedDisease, predictionResult, patientInfo, clinicUser } = clinicStore.state;

  container.innerHTML = `
    <div class="result-page clinic-result-page">
      <div id="clinic-result-header-container"></div>
      <div id="clinic-result-card-container"></div>
    </div>
  `;

  // Render header
  const headerContainer = container.querySelector('#clinic-result-header-container');
  if (headerContainer) {
    renderClinicHeader(headerContainer, {
      title: 'Clinical Screening Assessment Report',
      subtitle: `Statistical risk evaluation completed for ${patientInfo ? patientInfo.fullName : 'Patient'}.`,
      clinicUser,
    });
  }

  const resultContainer = container.querySelector('#clinic-result-card-container');
  if (resultContainer) {
    renderClinicResultCard(resultContainer, {
      disease: selectedDisease,
      result: predictionResult,
      patientInfo,
      clinicUser,
    });
  }
}
