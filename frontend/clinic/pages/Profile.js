/**
 * Clinic User Profile Page.
 * 
 * Displays and allows editing of practitioner information:
 * Full Name, Mobile, and Clinic Name (institutional facility).
 */

import { clinicStore } from '../hooks/store.js';
import { clinicAuthService } from '../services/auth.js';
import { renderClinicHeader } from '../components/Header.js';

export function renderClinicProfilePage(container) {
  const { clinicUser } = clinicStore.state;

  if (!clinicUser) {
    clinicStore.navigate('login');
    return;
  }

  container.innerHTML = `
    <div class="profile-page clinic-profile-page">
      <div id="clinic-profile-header-container"></div>

      <div class="profile-card">
        <div class="profile-avatar-row">
          <div class="profile-avatar-circle" style="background: linear-gradient(135deg, var(--accent-teal), var(--primary));">
            ⚕️
          </div>
          <div class="profile-meta">
            <h2 class="profile-name">${clinicUser.fullName}</h2>
            <span class="profile-email">${clinicUser.email}</span>
            <div class="profile-badges">
              <span class="badge badge-success">● Provider Active</span>
              <span class="badge badge-pulse">${clinicUser.clinicName}</span>
            </div>
          </div>
        </div>

        <hr class="divider" />

        <form id="clinic-profile-form" class="profile-form">
          <div class="form-grid">
            <div class="form-group">
              <label for="clinic-prof-name" class="form-label">Practitioner Full Name <span class="required">*</span></label>
              <input 
                type="text" 
                id="clinic-prof-name" 
                name="fullName" 
                class="form-control" 
                value="${clinicUser.fullName || ''}" 
                required 
              />
            </div>

            <div class="form-group">
              <label for="clinic-prof-email" class="form-label">Institutional Email (Login Identity)</label>
              <input 
                type="email" 
                id="clinic-prof-email" 
                class="form-control" 
                value="${clinicUser.email || ''}" 
                disabled 
                title="Institutional email is your immutable authentication identity" 
              />
              <small class="form-hint">Email serves as your primary institutional identity</small>
            </div>

            <div class="form-group">
              <label for="clinic-prof-facility" class="form-label">Clinic / Hospital Center <span class="required">*</span></label>
              <input 
                type="text" 
                id="clinic-prof-facility" 
                name="clinicName" 
                class="form-control" 
                value="${clinicUser.clinicName || ''}" 
                required 
              />
            </div>

            <div class="form-group">
              <label for="clinic-prof-mobile" class="form-label">Practitioner Mobile Contact</label>
              <input 
                type="tel" 
                id="clinic-prof-mobile" 
                name="mobile" 
                class="form-control" 
                value="${clinicUser.mobile || ''}" 
                placeholder="+1 555-0344" 
              />
            </div>
          </div>

          <div class="profile-actions">
            <button type="button" class="btn btn-outline" id="btn-cancel-clinic-profile">
              Cancel
            </button>
            <button type="submit" class="btn btn-primary" id="btn-save-clinic-profile">
              Save Provider Changes
            </button>
          </div>
        </form>
      </div>
    </div>
  `;

  // Render header
  const headerContainer = container.querySelector('#clinic-profile-header-container');
  if (headerContainer) {
    renderClinicHeader(headerContainer, {
      title: 'Practitioner Profile & Center Affiliation',
      subtitle: 'Manage practitioner credentials and clinical institutional details.',
      clinicUser,
    });
  }

  const form = container.querySelector('#clinic-profile-form');
  const btnSave = container.querySelector('#btn-save-clinic-profile');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    clinicStore.clearAlert();

    const fullName = form.fullName.value.trim();
    const clinicName = form.clinicName.value.trim();
    const mobile = form.mobile.value.trim();

    try {
      btnSave.disabled = true;
      btnSave.textContent = 'Saving...';

      await clinicAuthService.updateProfile({ fullName, clinicName, mobile });
      clinicStore.setAlert('success', 'Practitioner profile details successfully updated.');
      renderClinicProfilePage(container);
    } catch (err) {
      clinicStore.setAlert('error', err.message || 'Failed to update practitioner profile.');
    } finally {
      btnSave.disabled = false;
      btnSave.textContent = 'Save Provider Changes';
    }
  });

  container.querySelector('#btn-cancel-clinic-profile')?.addEventListener('click', () => {
    clinicStore.navigate('dashboard');
  });
}
