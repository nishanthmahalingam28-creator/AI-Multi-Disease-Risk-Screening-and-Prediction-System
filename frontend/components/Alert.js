/**
 * Global Dismissible Alert Component.
 * 
 * Displays contextual feedback messages (error, success, warning, info)
 * with accessible dismiss capability.
 */

import { store } from '../user/hooks/store.js';

export function renderAlert(container) {
  const { alert } = store.state;

  if (!alert) {
    container.innerHTML = '';
    return;
  }

  const icons = {
    error: '⚠️',
    warning: '⚡',
    success: '✅',
    info: 'ℹ️',
  };

  const alertClass = `alert alert-${alert.type || 'info'}`;
  const icon = icons[alert.type] || 'ℹ️';

  container.innerHTML = `
    <div class="${alertClass}" role="alert" id="global-alert">
      <div class="alert-content">
        <span class="alert-icon">${icon}</span>
        <span class="alert-message">${alert.message}</span>
      </div>
      <button class="alert-dismiss" id="alert-dismiss-btn" aria-label="Dismiss alert">&times;</button>
    </div>
  `;

  const dismissBtn = container.querySelector('#alert-dismiss-btn');
  if (dismissBtn) {
    dismissBtn.addEventListener('click', () => {
      store.clearAlert();
    });
  }
}
