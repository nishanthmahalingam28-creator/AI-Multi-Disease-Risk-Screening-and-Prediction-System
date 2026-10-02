/**
 * Clinic Screening History Page.
 * 
 * Displays historical clinical screenings conducted by or associated with
 * the active practitioner and healthcare facility, including patient references.
 */

import { renderClinicHeader } from '../components/Header.js';
import { clinicApiService } from '../services/api.js';
import { clinicStore } from '../hooks/store.js';
import { PLANNED_DISEASES, RISK_TIERS } from '../types/constants.js';

export function renderClinicHistoryPage(container) {
  const { clinicUser } = clinicStore.state;

  let currentFilterDisease = '';
  let currentFilterRisk = '';
  let currentPage = 1;
  const pageSize = 10;
  let totalRecords = 0;
  let historyItems = [];
  let isLoading = false;
  let loadError = null;

  container.innerHTML = `
    <div class="history-page clinic-history-page">
      <div id="clinic-history-header-container"></div>
      <div id="clinic-history-content-mount"></div>
    </div>
  `;

  // Render header
  const headerContainer = container.querySelector('#clinic-history-header-container');
  if (headerContainer) {
    renderClinicHeader(headerContainer, {
      title: 'Clinical Screening Records',
      subtitle: `Audit log of disease risk consultations conducted at ${clinicUser ? clinicUser.clinicName : 'this clinic'}.`,
      clinicUser,
    });
  }

  const contentMount = container.querySelector('#clinic-history-content-mount');

  async function loadHistory() {
    isLoading = true;
    loadError = null;
    renderContent();

    try {
      const resp = await clinicApiService.getClinicHistory({
        clinicUser,
        page: currentPage,
        pageSize,
        disease: currentFilterDisease || null,
        riskLevel: currentFilterRisk || null,
      });

      historyItems = resp.items || [];
      totalRecords = resp.total !== undefined ? resp.total : historyItems.length;
      currentPage = resp.page || 1;
      isLoading = false;
      renderContent();
    } catch (err) {
      isLoading = false;
      loadError = err.message || 'Unable to load screening history. Please try again.';
      renderContent();
    }
  }

  function renderContent() {
    if (!contentMount) return;

    if (isLoading) {
      contentMount.innerHTML = `
        <div class="empty-state" role="status" aria-live="polite">
          <div class="loading-spinner" style="margin: 0 auto 16px;"></div>
          <p>Loading screening history...</p>
        </div>
      `;
      return;
    }

    if (loadError) {
      contentMount.innerHTML = `
        <div class="empty-state">
          <span class="empty-icon">⚠️</span>
          <h2>Unable to Load History</h2>
          <p style="color: var(--danger); margin-bottom: 16px;">Unable to load screening history. Please try again.</p>
          <button class="btn btn-outline" id="btn-retry-clinic-history">Retry Loading</button>
        </div>
      `;
      contentMount.querySelector('#btn-retry-clinic-history')?.addEventListener('click', loadHistory);
      return;
    }

    const totalPages = Math.ceil(totalRecords / pageSize) || 1;

    contentMount.innerHTML = `
      <div class="history-toolbar">
        <div class="history-filters">
          <div class="filter-group">
            <label for="clinic-filter-disease" class="filter-label">Condition:</label>
            <select id="clinic-filter-disease" class="filter-select">
              <option value="">All Supported Conditions</option>
              <option value="lung_cancer" ${currentFilterDisease === 'lung_cancer' ? 'selected' : ''}>Lung Cancer</option>
              <option value="asthma" ${currentFilterDisease === 'asthma' ? 'selected' : ''}>Asthma</option>
              <option value="parkinsons" ${currentFilterDisease === 'parkinsons' ? 'selected' : ''}>Parkinson's Disease</option>
            </select>
          </div>

          <div class="filter-group">
            <label for="clinic-filter-risk" class="filter-label">Risk Level:</label>
            <select id="clinic-filter-risk" class="filter-select">
              <option value="">All Risk Tiers</option>
              <option value="LOW" ${currentFilterRisk === 'LOW' ? 'selected' : ''}>Low Risk</option>
              <option value="MODERATE" ${currentFilterRisk === 'MODERATE' ? 'selected' : ''}>Moderate Risk</option>
              <option value="HIGH" ${currentFilterRisk === 'HIGH' ? 'selected' : ''}>High Risk</option>
            </select>
          </div>
        </div>

        <div class="history-count-badge">
          Showing ${historyItems.length} of ${totalRecords} record${totalRecords === 1 ? '' : 's'}
        </div>
      </div>

      ${historyItems.length === 0 ? `
        <div class="empty-state">
          <span class="empty-icon">📋</span>
          <h2>No screening history available.</h2>
          <p>No clinical consultations matching the current criteria were found for your practitioner account.</p>
          <button class="btn btn-primary" id="btn-start-patient-screening">Initiate Patient Screening</button>
        </div>
      ` : `
        <!-- Table View for Desktop -->
        <div class="history-table-container">
          <table class="history-table" aria-label="Clinical Screening Records">
            <thead>
              <tr>
                <th>Screening ID</th>
                <th>Patient Reference</th>
                <th>Condition</th>
                <th>Risk Tier</th>
                <th>Probability</th>
                <th>Model</th>
                <th>Timestamp</th>
                <th>Status</th>
                <th style="text-align: right;">Report</th>
              </tr>
            </thead>
            <tbody>
              ${historyItems.map(item => {
                const disease = PLANNED_DISEASES.find(d => d.code === item.disease_code) || {
                  name: item.disease_name || item.disease_code,
                  icon: '⚕️',
                };
                const rawRisk = (item.risk_level || 'LOW').toUpperCase();
                let riskBadgeClass = 'risk-low';
                if (rawRisk === 'HIGH') riskBadgeClass = 'risk-high';
                else if (rawRisk === 'MODERATE') riskBadgeClass = 'risk-moderate';

                const probPct = item.probability !== undefined && item.probability !== null
                  ? (item.probability * 100).toFixed(1) + '%'
                  : 'N/A';

                const dateStr = item.created_at ? new Date(item.created_at).toLocaleDateString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                }) : 'N/A';

                const patientDisplay = item.patient_name || item.patient_email || 'Confidential Patient';

                return `
                  <tr>
                    <td><span class="font-mono" style="font-size: 0.8rem; color: var(--text-dim);">${item.id.slice(0, 8)}...</span></td>
                    <td>
                      <div style="display: flex; flex-direction: column;">
                        <strong>${patientDisplay}</strong>
                        ${item.patient_email && item.patient_name ? `<span style="font-size: 0.75rem; color: var(--text-dim);">${item.patient_email}</span>` : ''}
                      </div>
                    </td>
                    <td>
                      <div style="display: flex; align-items: center; gap: 8px;">
                        <span>${disease.icon}</span>
                        <span>${disease.name}</span>
                      </div>
                    </td>
                    <td>
                      <span class="badge ${riskBadgeClass}">
                        ● ${rawRisk}
                      </span>
                    </td>
                    <td><strong>${probPct}</strong></td>
                    <td><span class="font-mono" style="font-size: 0.8rem;">${item.model_version || '1.0.0'}</span></td>
                    <td style="color: var(--text-muted); font-size: 0.85rem;">${dateStr}</td>
                    <td><span class="badge badge-success" style="font-size: 0.75rem;">${item.status || 'COMPLETED'}</span></td>
                    <td style="text-align: right;">
                      <button class="btn btn-outline btn-sm btn-view-clinic-report" data-id="${item.id}">
                        📄 View Report
                      </button>
                    </td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>

        <!-- Mobile Responsive Card View -->
        <div class="history-card-list">
          ${historyItems.map(item => {
            const disease = PLANNED_DISEASES.find(d => d.code === item.disease_code) || {
              name: item.disease_name || item.disease_code,
              icon: '⚕️',
            };
            const rawRisk = (item.risk_level || 'LOW').toUpperCase();
            let riskBadgeClass = 'risk-low';
            if (rawRisk === 'HIGH') riskBadgeClass = 'risk-high';
            else if (rawRisk === 'MODERATE') riskBadgeClass = 'risk-moderate';

            const probPct = item.probability !== undefined && item.probability !== null
              ? (item.probability * 100).toFixed(1) + '%'
              : 'N/A';

            const dateStr = item.created_at ? new Date(item.created_at).toLocaleDateString('en-US', {
              month: 'short',
              day: 'numeric',
              year: 'numeric',
            }) : 'N/A';

            const patientDisplay = item.patient_name || item.patient_email || 'Confidential Patient';

            return `
              <div class="history-record-card">
                <div class="history-card-header">
                  <div>
                    <span style="font-size: 0.75rem; color: var(--accent-teal); text-transform: uppercase;">Patient:</span>
                    <strong style="display: block; font-size: 1rem;">${patientDisplay}</strong>
                    <span style="font-size: 0.75rem; color: var(--text-dim); font-family: var(--font-mono);">${item.id.slice(0, 12)}</span>
                  </div>
                  <span class="badge ${riskBadgeClass}">● ${rawRisk}</span>
                </div>

                <div class="history-card-body">
                  <div>
                    <span style="color: var(--text-dim); display: block;">Condition:</span>
                    <span>${disease.icon} ${disease.name}</span>
                  </div>
                  <div>
                    <span style="color: var(--text-dim); display: block;">Probability:</span>
                    <strong>${probPct}</strong>
                  </div>
                  <div>
                    <span style="color: var(--text-dim); display: block;">Date:</span>
                    <span>${dateStr}</span>
                  </div>
                  <div>
                    <span style="color: var(--text-dim); display: block;">Status:</span>
                    <span class="badge badge-success" style="font-size: 0.72rem;">${item.status || 'COMPLETED'}</span>
                  </div>
                </div>

                <div class="history-card-footer">
                  <button class="btn btn-primary btn-sm btn-view-clinic-report" data-id="${item.id}" style="width: 100%;">
                    📄 View Consultation Report
                  </button>
                </div>
              </div>
            `;
          }).join('')}
        </div>

        ${totalPages > 1 ? `
          <div class="pagination-container">
            <div>Page <strong>${currentPage}</strong> of <strong>${totalPages}</strong></div>
            <div class="pagination-controls">
              <button class="btn btn-outline btn-sm" id="btn-prev-clinic-page" ${currentPage <= 1 ? 'disabled' : ''}>
                ← Previous
              </button>
              <button class="btn btn-outline btn-sm" id="btn-next-clinic-page" ${currentPage >= totalPages ? 'disabled' : ''}>
                Next →
              </button>
            </div>
          </div>
        ` : ''}
      `}
    `;

    contentMount.querySelector('#btn-start-patient-screening')?.addEventListener('click', () => {
      clinicStore.navigate('dashboard');
    });

    const diseaseSelect = contentMount.querySelector('#clinic-filter-disease');
    if (diseaseSelect) {
      diseaseSelect.addEventListener('change', (e) => {
        currentFilterDisease = e.target.value;
        currentPage = 1;
        loadHistory();
      });
    }

    const riskSelect = contentMount.querySelector('#clinic-filter-risk');
    if (riskSelect) {
      riskSelect.addEventListener('change', (e) => {
        currentFilterRisk = e.target.value;
        currentPage = 1;
        loadHistory();
      });
    }

    contentMount.querySelector('#btn-prev-clinic-page')?.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        loadHistory();
      }
    });

    contentMount.querySelector('#btn-next-clinic-page')?.addEventListener('click', () => {
      if (currentPage < totalPages) {
        currentPage++;
        loadHistory();
      }
    });

    contentMount.querySelectorAll('.btn-view-clinic-report').forEach(btn => {
      btn.addEventListener('click', () => {
        const screeningId = btn.dataset.id;
        clinicStore.navigate('report', { screeningId });
      });
    });
  }

  loadHistory();
}
