/**
 * Clinic Disease-Specific Screening Input Forms.
 * 
 * Reuses validated clinical feature fields for Lung Cancer (15), Asthma (14),
 * and Parkinson's (22 acoustic features), ensuring strict alignment with feature_schema.json.
 */

import { 
  renderLungCancerForm as renderUserLCForm, 
  renderAsthmaForm as renderUserAsthmaForm, 
  renderParkinsonsForm as renderUserParkinsonsForm 
} from '../../user/components/ScreeningForms.js';

export function renderClinicDiseaseForm(container, diseaseCode, onSubmit) {
  if (diseaseCode === 'lung_cancer') {
    renderUserLCForm(container, onSubmit);
  } else if (diseaseCode === 'asthma') {
    renderUserAsthmaForm(container, onSubmit);
  } else if (diseaseCode === 'parkinsons') {
    renderUserParkinsonsForm(container, onSubmit);
  } else {
    container.innerHTML = `
      <div class="empty-state">
        <p>This disease model is currently unavailable for clinical inference.</p>
      </div>
    `;
  }
}
