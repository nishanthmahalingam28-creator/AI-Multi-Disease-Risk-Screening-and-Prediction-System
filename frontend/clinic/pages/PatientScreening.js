/**
 * Clinic Patient Screening Execution Page.
 * 
 * Orchestrates patient demographic intake (compatible with users table)
 * alongside disease-specific clinical feature inputs, input validation,
 * API dispatch, and navigation to clinic screening results.
 */

import { PLANNED_DISEASES } from '../types/constants.js';
import { renderClinicHeader } from '../components/Header.js';
import { renderPatientForm, validateAndExtractPatientInfo } from '../components/PatientForm.js';
import { renderClinicDiseaseForm } from '../components/ScreeningForm.js';
import { clinicApiService } from '../services/api.js';
import { clinicStore } from '../hooks/store.js';

export function renderPatientScreeningPage(container) {
  const { selectedDisease, clinicUser } = clinicStore.state;

  const disease = PLANNED_DISEASES.find(d => d.code === selectedDisease);

  if (!disease) {
    container.innerHTML = `
      <div class="empty-state">
        <p>No screening disease selected. Please choose a condition from the catalog.</p>
        <button class="btn btn-primary" id="btn-clinic-return-catalog">Browse Catalog</button>
      </div>
    `;
    container.querySelector('#btn-clinic-return-catalog')?.addEventListener('click', () => {
      clinicStore.navigate('dashboard');
    });
    return;
  }

  if (!disease.available) {
    container.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">${disease.icon}</span>
        <h2>${disease.name}</h2>
        <p class="badge badge-soon">Model Under Development — Coming Soon</p>
        <p>This condition is currently in clinical trial validation and cannot be evaluated.</p>
        <button class="btn btn-primary" id="btn-clinic-back-dash">Return to Dashboard</button>
      </div>
    `;
    container.querySelector('#btn-clinic-back-dash')?.addEventListener('click', () => {
      clinicStore.navigate('dashboard');
    });
    return;
  }

  container.innerHTML = `
    <div class="screening-page clinic-screening-page">
      <div id="clinic-screening-header-container"></div>

      <div class="screening-workflow-card">
        <div class="workflow-header">
          <div class="disease-badge-title">
            <span class="disease-icon-large">${disease.icon}</span>
            <div>
              <span class="workflow-category">${disease.category}</span>
              <h2 class="workflow-disease-name">${disease.name} (Clinic Assisted)</h2>
            </div>
          </div>
          <button class="btn btn-outline btn-sm" id="btn-clinic-change-disease">
            ← Change Condition
          </button>
        </div>

        <div class="screening-instructions">
          <p>
            <strong>Practitioner Instructions:</strong> Complete the Patient Identification section below,
            then input verified clinical/laboratory observations into the feature form.
            Patient demographic details will be linked to this clinic consultation.
          </p>
        </div>

        <!-- 1. Patient Intake Section -->
        <div id="patient-intake-mount" style="margin-bottom: 24px;"></div>

        <!-- 2. Disease Inputs Section -->
        <div id="disease-inputs-mount" class="form-container"></div>
      </div>
    </div>
  `;

  // Render header
  const headerContainer = container.querySelector('#clinic-screening-header-container');
  if (headerContainer) {
    renderClinicHeader(headerContainer, {
      title: `Patient Screening — ${disease.name}`,
      subtitle: `Clinical consultation session conducted by ${clinicUser ? clinicUser.fullName : 'Provider'}.`,
      clinicUser,
    });
  }

  container.querySelector('#btn-clinic-change-disease')?.addEventListener('click', () => {
    clinicStore.navigate('dashboard');
  });

  // Mount Patient Intake Form
  const patientMount = container.querySelector('#patient-intake-mount');
  if (patientMount) {
    renderPatientForm(patientMount, clinicStore.state.patientInfo || {});
  }

  // Mount Disease Form
  const diseaseMount = container.querySelector('#disease-inputs-mount');
  if (diseaseMount) {
    const handleClinicSubmit = async (modelPayload) => {
      clinicStore.clearAlert();

      // 1. Validate Patient Info first
      let patientInfo = null;
      try {
        patientInfo = validateAndExtractPatientInfo(patientMount);
      } catch (err) {
        clinicStore.setAlert('error', `Patient Validation Error: ${err.message}`);
        window.scrollTo({ top: 200, behavior: 'smooth' });
        return;
      }

      // 2. Dispatch API call
      clinicStore.setLoading(true, `Computing statistical risk estimation for ${patientInfo.fullName}...`);

      const submitBtns = diseaseMount.querySelectorAll('.btn-submit');
      submitBtns.forEach(btn => {
        btn.disabled = true;
        btn.textContent = 'Processing Clinical Prediction...';
      });

      try {
        const response = await clinicApiService.predict(disease.code, modelPayload);

        // Auto-persist clinical consultation record
        try {
          const persistResp = await clinicApiService.saveScreening({
            diseaseCode: disease.code,
            inputData: modelPayload,
            predictionResult: response,
            clinicUser,
            patientInfo,
          });
          if (persistResp && persistResp.id) {
            response.screening_id = persistResp.id;
          }
        } catch (persistErr) {
          console.warn('Clinic screening persistence warning:', persistErr);
        }

        clinicStore.setLoading(false);
        clinicStore.navigate('result', {
          disease,
          result: response,
          patientInfo,
        });
      } catch (err) {
        clinicStore.setLoading(false);
        clinicStore.setAlert('error', `Prediction Failed: ${err.message}`);
        submitBtns.forEach(btn => {
          btn.disabled = false;
          btn.textContent = 'Retry Clinical Submission →';
        });
      }
    };

    renderClinicDiseaseForm(diseaseMount, disease.code, handleClinicSubmit);
  }
}
