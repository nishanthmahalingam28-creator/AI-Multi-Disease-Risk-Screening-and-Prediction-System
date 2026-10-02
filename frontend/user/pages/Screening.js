/**
 * Disease Screening Execution Page.
 * 
 * Orchestrates disease-specific form presentation, input validation,
 * API dispatch, loading states, and result navigation.
 */

import { PLANNED_DISEASES } from '../types/constants.js';
import { 
  renderLungCancerForm, 
  renderAsthmaForm, 
  renderParkinsonsForm 
} from '../components/ScreeningForms.js';
import { renderHeader } from '../components/Header.js';
import { apiService } from '../services/api.js';
import { store } from '../hooks/store.js';

export function renderScreeningPage(container) {
  const { selectedDisease, user } = store.state;

  const disease = PLANNED_DISEASES.find(d => d.code === selectedDisease);

  if (!disease) {
    container.innerHTML = `
      <div class="empty-state">
        <p>No screening disease selected. Please choose a condition from the catalog.</p>
        <button class="btn btn-primary" id="btn-return-catalog">Browse Catalog</button>
      </div>
    `;
    container.querySelector('#btn-return-catalog')?.addEventListener('click', () => {
      store.navigate('dashboard');
    });
    return;
  }

  if (!disease.available) {
    container.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">${disease.icon}</span>
        <h2>${disease.name}</h2>
        <p class="badge badge-soon">Model Under Development — Coming Soon</p>
        <p>The statistical prediction model for this condition is currently undergoing training and rigorous clinical validation.</p>
        <button class="btn btn-primary" id="btn-back-to-dash">Return to Dashboard</button>
      </div>
    `;
    container.querySelector('#btn-back-to-dash')?.addEventListener('click', () => {
      store.navigate('dashboard');
    });
    return;
  }

  container.innerHTML = `
    <div class="screening-page">
      <div id="screening-header-container"></div>

      <div class="screening-workflow-card">
        <div class="workflow-header">
          <div class="disease-badge-title">
            <span class="disease-icon-large">${disease.icon}</span>
            <div>
              <span class="workflow-category">${disease.category}</span>
              <h2 class="workflow-disease-name">${disease.name}</h2>
            </div>
          </div>
          <button class="btn btn-outline btn-sm" id="btn-change-disease">
            ← Change Condition
          </button>
        </div>

        <div class="screening-instructions">
          <p>
            Please provide accurate observations or laboratory values for the required parameters.
            All measurements will be validated against the model's standardized feature contract.
          </p>
        </div>

        <div id="screening-form-mount" class="form-container"></div>
      </div>
    </div>
  `;

  // Render header
  const headerContainer = container.querySelector('#screening-header-container');
  if (headerContainer) {
    renderHeader(headerContainer, {
      title: `${disease.name}`,
      subtitle: `Clinical screening questionnaire for ${disease.name.toLowerCase()}.`,
      userName: user ? user.fullName : '',
    });
  }

  // Change disease button
  container.querySelector('#btn-change-disease')?.addEventListener('click', () => {
    store.navigate('dashboard');
  });

  // Mount appropriate form
  const formMount = container.querySelector('#screening-form-mount');
  if (formMount) {
    const handleSubmit = async (payload) => {
      store.clearAlert();
      store.setLoading(true, `Running AI inference for ${disease.name}...`);

      const submitBtns = formMount.querySelectorAll('.btn-submit');
      submitBtns.forEach(btn => {
        btn.disabled = true;
        btn.textContent = 'Processing Prediction...';
      });

      try {
        const response = await apiService.predict(disease.code, payload);

        // Auto-persist screening record if user is authenticated
        try {
          const persistResp = await apiService.saveScreening({
            diseaseCode: disease.code,
            inputData: payload,
            predictionResult: response,
            user,
          });
          if (persistResp && persistResp.id) {
            response.screening_id = persistResp.id;
          }
        } catch (persistErr) {
          console.warn('Screening persistence warning:', persistErr);
        }

        store.setLoading(false);
        store.navigate('result', { disease, result: response });
      } catch (err) {
        store.setLoading(false);
        const errorMsg = err.message || 'An error occurred while evaluating the screening model.';
        store.setAlert('error', `Prediction Failed: ${errorMsg}`);
        submitBtns.forEach(btn => {
          btn.disabled = false;
          btn.textContent = 'Retry Screening Submission →';
        });
      }
    };

    if (disease.code === 'lung_cancer') {
      renderLungCancerForm(formMount, handleSubmit);
    } else if (disease.code === 'asthma') {
      renderAsthmaForm(formMount, handleSubmit);
    } else if (disease.code === 'parkinsons') {
      renderParkinsonsForm(formMount, handleSubmit);
    }
  }
}
