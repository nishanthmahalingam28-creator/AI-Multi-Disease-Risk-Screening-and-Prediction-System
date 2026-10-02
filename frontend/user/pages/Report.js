/**
 * User Screening Report Page View.
 * 
 * Retrieves detailed screening record from the backend and delegates
 * rendering to the reusable ScreeningReport component with print support.
 */

import { apiService } from '../services/api.js';
import { store } from '../hooks/store.js';
import { renderScreeningReport } from '../../components/ScreeningReport.js';

export function renderUserReportPage(container) {
  const { user, selectedScreeningId, activeReportScreening, selectedDisease, predictionResult } = store.state;

  // If already loaded in memory (e.g. immediately post-screening), render directly
  if (activeReportScreening) {
    renderScreeningReport(container, {
      screening: activeReportScreening,
      onBack: () => store.navigate('history'),
      onDashboard: () => store.navigate('dashboard'),
      isClinicContext: false,
    });
    return;
  }

  // If we navigated straight from screening result with existing result state
  if (!selectedScreeningId && predictionResult && selectedDisease) {
    const syntheticRecord = {
      id: predictionResult.screening_id || 'LOCAL-DRAFT-' + Date.now().toString(36).toUpperCase(),
      disease: selectedDisease,
      status: 'COMPLETED',
      created_at: new Date().toISOString(),
      prediction: predictionResult.prediction || predictionResult,
      patient: {
        full_name: user ? user.fullName : 'Direct User',
        email: user ? user.email : 'N/A',
      },
      clinic_user: null,
      input_data: {},
    };

    renderScreeningReport(container, {
      screening: syntheticRecord,
      onBack: () => store.navigate('result', { disease: selectedDisease, result: predictionResult }),
      onDashboard: () => store.navigate('dashboard'),
      isClinicContext: false,
    });
    return;
  }

  if (!selectedScreeningId) {
    container.innerHTML = `
      <div class="empty-state">
        <span class="empty-icon">⚠️</span>
        <h2>Screening Record Not Selected</h2>
        <p>Please select a valid screening record from your history.</p>
        <button class="btn btn-primary" id="btn-err-history">Return to History</button>
      </div>
    `;
    container.querySelector('#btn-err-history')?.addEventListener('click', () => {
      store.navigate('history');
    });
    return;
  }

  // Loading state
  container.innerHTML = `
    <div class="empty-state" role="status" aria-live="polite">
      <div class="loading-spinner" style="margin: 0 auto 16px;"></div>
      <p>Loading clinical screening report...</p>
    </div>
  `;

  // Fetch from backend
  apiService.getScreeningDetail(selectedScreeningId, user)
    .then(data => {
      renderScreeningReport(container, {
        screening: data,
        onBack: () => store.navigate('history'),
        onDashboard: () => store.navigate('dashboard'),
        isClinicContext: false,
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
          <h2>Unable to Load Report</h2>
          <p style="color: var(--danger); margin-bottom: 20px;">${errorMsg}</p>
          <div style="display: flex; gap: 12px; justify-content: center;">
            <button class="btn btn-outline" id="btn-err-history">← Return to History</button>
            <button class="btn btn-primary" id="btn-err-dashboard">📊 Dashboard</button>
          </div>
        </div>
      `;

      container.querySelector('#btn-err-history')?.addEventListener('click', () => {
        store.navigate('history');
      });
      container.querySelector('#btn-err-dashboard')?.addEventListener('click', () => {
        store.navigate('dashboard');
      });
    });
}
