/**
 * Clinic Screening Report Page View.
 * 
 * Fetches clinical consultation screening record and renders the reusable
 * ScreeningReport with practitioner attribution and patient identification.
 */

import { clinicApiService } from '../services/api.js';
import { clinicStore } from '../hooks/store.js';
import { renderScreeningReport } from '../../components/ScreeningReport.js';

export function renderClinicReportPage(container) {
  const { clinicUser, selectedScreeningId, activeReportScreening, selectedDisease, predictionResult, patientInfo } = clinicStore.state;

  if (activeReportScreening) {
    renderScreeningReport(container, {
      screening: activeReportScreening,
      onBack: () => clinicStore.navigate('history'),
      onDashboard: () => clinicStore.navigate('dashboard'),
      isClinicContext: true,
    });
    return;
  }

  // Navigate straight from screening result
  if (!selectedScreeningId && predictionResult && selectedDisease) {
    const data = predictionResult.prediction || predictionResult;
    const syntheticRecord = {
      id: predictionResult.screening_id || 'CLINIC-LOCAL-' + Date.now().toString(36).toUpperCase(),
      disease: selectedDisease,
      status: 'COMPLETED',
      created_at: new Date().toISOString(),
      prediction: data,
      patient: {
        full_name: patientInfo ? patientInfo.fullName : 'Confidential Patient',
        email: patientInfo ? patientInfo.email : 'N/A',
        mobile: patientInfo ? patientInfo.mobile : '',
        date_of_birth: patientInfo ? patientInfo.dateOfBirth : '',
      },
      clinic_user: clinicUser ? {
        full_name: clinicUser.fullName,
        clinic_name: clinicUser.clinicName,
      } : null,
      input_data: {},
    };

    renderScreeningReport(container, {
      screening: syntheticRecord,
      onBack: () => clinicStore.navigate('result', { disease: selectedDisease, result: predictionResult, patientInfo }),
      onDashboard: () => clinicStore.navigate('dashboard'),
      isClinicContext: true,
    });
    return;
  }

  if (!selectedScreeningId) {
    container.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">⚠️</span>
        <h2>Screening Record Not Selected</h2>
        <p>Please select an authorized clinical record from the audit log.</p>
        <button class="btn btn-primary" id="btn-err-clinic-history">Return to Audit Log</button>
      </div>
    `;
    container.querySelector('#btn-err-clinic-history')?.addEventListener('click', () => {
      clinicStore.navigate('history');
    });
    return;
  }

  container.innerHTML = `
    <div class="empty-state" role="status" aria-live="polite">
      <div class="loading-spinner" style="margin: 0 auto 16px;"></div>
      <p>Loading clinical consultation report...</p>
    </div>
  `;

  clinicApiService.getScreeningDetail(selectedScreeningId, clinicUser)
    .then(data => {
      renderScreeningReport(container, {
        screening: data,
        onBack: () => clinicStore.navigate('history'),
        onDashboard: () => clinicStore.navigate('dashboard'),
        isClinicContext: true,
      });
    })
    .catch(err => {
      let errorMsg = 'Screening record not found.';
      if (err.status === 403 || err.code === 'FORBIDDEN') {
        errorMsg = 'You are not authorized to view this screening record.';
      } else if (err.status === 404 || err.code === 'NOT_FOUND') {
        errorMsg = 'Screening record not found.';
      } else if (err.message) {
        errorMsg = err.message;
      }

      container.innerHTML = `
        <div class="empty-state">
          <span class="empty-icon">🔒</span>
          <h2>Unable to Load Clinical Report</h2>
          <p style="color: var(--danger); margin-bottom: 20px;">${errorMsg}</p>
          <div style="display: flex; gap: 12px; justify-content: center;">
            <button class="btn btn-outline" id="btn-err-clinic-history">← Return to Records</button>
            <button class="btn btn-primary" id="btn-err-clinic-dashboard">📊 Clinic Dashboard</button>
          </div>
        </div>
      `;

      container.querySelector('#btn-err-clinic-history')?.addEventListener('click', () => {
        clinicStore.navigate('history');
      });
      container.querySelector('#btn-err-clinic-dashboard')?.addEventListener('click', () => {
        clinicStore.navigate('dashboard');
      });
    });
}
