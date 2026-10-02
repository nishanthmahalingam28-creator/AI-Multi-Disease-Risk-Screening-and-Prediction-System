/**
 * User Screening History Page.
 * 
 * Displays historical disease risk screening evaluations for the authenticated
 * patient user with filtering by disease, risk tier, and pagination.
 */

import { renderHeader } from '../components/Header.js';
import { apiService } from '../services/api.js';
import { store } from '../hooks/store.js';
import { PLANNED_DISEASES, RISK_TIERS } from '../types/constants.js';

export function renderUserHistoryPage(container) {
  const { user } = store.state;

  let currentFilterDisease = '';
  let currentFilterRisk = '';
  let currentPage = 1;
  const pageSize = 10;
  let totalRecords = 0;
  let historyItems = [];
  let isLoading = false;
  let loadError = null;

  container.innerHTML = `
    <div class="history-page">
      <div id="history-header-container"></div>
      <div id="history-content-mount"></div>
    </div>
  `;

  // Render header
  const headerContainer = container.querySelector('#history-header-container');
  if (headerContainer) {
    renderHeader(headerContainer, {
      title: 'Screening History',
      subtitle: 'Review and verify your historical AI disease screening records and printable clinical reports.',
      userName: user ? user.fullName : '',
    });
  }

  const contentMount = container.querySelector('#history-content-mount');

  async function loadHistory() {
    isLoading = true;
    loadError = null;
    renderContent();

    try {
      const resp = await apiService.getUserHistory({
        user,
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
          <button class="btn btn-outline" id="btn-retry-history">Retry Loading</button>
        </div>
      `;
      contentMount.querySelector('#btn-retry-history')?.addEventListener('click', loadHistory);
      return;
    }

    const totalPages = Math.ceil(totalRecords / pageSize) || 1;

    // Build filter toolbar and table/cards
    contentMount.innerHTML = `
      <div class="history-toolbar">
        <div class="history-filters">
          <div class="filter-group">
            <label for="filter-disease" class="filter-label">Condition:</label>
            <select id="filter-disease" class="filter-select">
              <option value="">All Supported Conditions</option>
              <option value="lung_cancer" ${currentFilterDisease === 'lung_cancer' ? 'selected' : ''}>Lung Cancer</option>
              <option value="asthma" ${currentFilterDisease === 'asthma' ? 'selected' : ''}>Asthma</option>
              <option value="parkinsons" ${currentFilterDisease === 'parkinsons' ? 'selected' : ''}>Parkinson's Disease</option>
            </select>
          </div>

          <div class="filter-group">
            <label for="filter-risk" class="filter-label">Risk Tier:</label>
            <select id="filter-risk" class="filter-select">
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
          <p>You have not conducted any disease risk assessments matching the selected criteria.</p>
          <button class="btn btn-primary" id="btn-start-screening">Start a Health Screening</button>
        </div>
      ` : `
        <!-- Desktop Table View -->
        <div class="history-table-container">
          <table class="history-table" aria-label="Screening History Records">
            <thead>
              <tr>
                <th>Screening ID</th>
                <th>Disease Condition</th>
                <th>Risk Level</th>
                <th>Probability</th>
                <th>Model Version</th>
                <th>Timestamp</th>
                <th>Status</th>
                <th style="text-align: right;">Action</th>
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

                return `
                  <tr>
                    <td><span class="font-mono" style="font-size: 0.8rem; color: var(--text-dim);">${item.id.slice(0, 8)}...</span></td>
                    <td>
                      <div style="display: flex; align-items: center; gap: 8px;">
                        <span>${disease.icon}</span>
                        <strong>${disease.name}</strong>
                      </div>
                    </td>
                    <td>
                      <span class="badge ${riskBadgeClass}">
                        ● ${rawRisk}
                      </span>
                    </td>
                    <td><strong>${probPct}</strong></td>
                    <td><span class="font-mono" style="font-size: 0.82rem;">${item.model_version || '1.0.0'}</span></td>
                    <td style="color: var(--text-muted); font-size: 0.85rem;">${dateStr}</td>
                    <td><span class="badge badge-success" style="font-size: 0.75rem;">${item.status || 'COMPLETED'}</span></td>
                    <td style="text-align: right;">
                      <button class="btn btn-outline btn-sm btn-view-report" data-id="${item.id}">
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

            return `
              <div class="history-record-card">
                <div class="history-card-header">
                  <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.4rem;">${disease.icon}</span>
                    <div>
                      <strong>${disease.name}</strong>
                      <div style="font-size: 0.75rem; color: var(--text-dim); font-family: var(--font-mono);">${item.id.slice(0, 12)}</div>
                    </div>
                  </div>
                  <span class="badge ${riskBadgeClass}">● ${rawRisk}</span>
                </div>

                <div class="history-card-body">
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
                  <div>
                    <span style="color: var(--text-dim); display: block;">Version:</span>
                    <span class="font-mono">${item.model_version || '1.0.0'}</span>
                  </div>
                </div>

                <div class="history-card-footer">
                  <button class="btn btn-primary btn-sm btn-view-report" data-id="${item.id}" style="width: 100%;">
                    📄 View Full Report
                  </button>
                </div>
              </div>
            `;
          }).join('')}
        </div>

        <!-- Pagination Controls -->
        ${totalPages > 1 ? `
          <div class="pagination-container">
            <div>Page <strong>${currentPage}</strong> of <strong>${totalPages}</strong></div>
            <div class="pagination-controls">
              <button class="btn btn-outline btn-sm" id="btn-prev-page" ${currentPage <= 1 ? 'disabled' : ''}>
                ← Previous
              </button>
              <button class="btn btn-outline btn-sm" id="btn-next-page" ${currentPage >= totalPages ? 'disabled' : ''}>
                Next →
              </button>
            </div>
          </div>
        ` : ''}
      `}
    `;

    // Attach event listeners
    contentMount.querySelector('#btn-start-screening')?.addEventListener('click', () => {
      store.navigate('dashboard');
    });

    const diseaseSelect = contentMount.querySelector('#filter-disease');
    if (diseaseSelect) {
      diseaseSelect.addEventListener('change', (e) => {
        currentFilterDisease = e.target.value;
        currentPage = 1;
        loadHistory();
      });
    }

    const riskSelect = contentMount.querySelector('#filter-risk');
    if (riskSelect) {
      riskSelect.addEventListener('change', (e) => {
        currentFilterRisk = e.target.value;
        currentPage = 1;
        loadHistory();
      });
    }

    contentMount.querySelector('#btn-prev-page')?.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        loadHistory();
      }
    });

    contentMount.querySelector('#btn-next-page')?.addEventListener('click', () => {
      if (currentPage < totalPages) {
        currentPage++;
        loadHistory();
      }
    });

    contentMount.querySelectorAll('.btn-view-report').forEach(btn => {
      btn.addEventListener('click', () => {
        const screeningId = btn.dataset.id;
        store.navigate('report', { screeningId });
      });
    });
  }

  // Initial load
  loadHistory();
}
