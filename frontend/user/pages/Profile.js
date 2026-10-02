/**
 * User Profile Page.
 * 
 * Allows users to view and update their personal profile data
 * (Full Name, Mobile, Date of Birth).
 */

import { store } from '../hooks/store.js';
import { authService } from '../services/auth.js';
import { renderHeader } from '../components/Header.js';

export function renderProfilePage(container) {
  const { user } = store.state;

  if (!user) {
    store.navigate('login');
    return;
  }

  container.innerHTML = `
    <div class="profile-page">
      <div id="profile-header-container"></div>

      <div class="profile-card">
        <div class="profile-avatar-row">
          <div class="profile-avatar-circle">
            ${user.fullName ? user.fullName[0].toUpperCase() : 'U'}
          </div>
          <div class="profile-meta">
            <h2 class="profile-name">${user.fullName}</h2>
            <span class="profile-email">${user.email}</span>
            <div class="profile-badges">
              <span class="badge badge-success">● Account Active</span>
              <span class="badge badge-neutral">Patient Role</span>
            </div>
          </div>
        </div>

        <hr class="divider" />

        <form id="profile-form" class="profile-form">
          <div class="form-grid">
            <div class="form-group">
              <label for="profile-name" class="form-label">Full Name <span class="required">*</span></label>
              <input 
                type="text" 
                id="profile-name" 
                name="fullName" 
                class="form-control" 
                value="${user.fullName || ''}" 
                required 
              />
            </div>

            <div class="form-group">
              <label for="profile-email" class="form-label">Email Address (Identity)</label>
              <input 
                type="email" 
                id="profile-email" 
                class="form-control" 
                value="${user.email || ''}" 
                disabled 
                title="Email is your immutable login identity" 
              />
              <small class="form-hint">Email serves as your unique authentication identifier</small>
            </div>

            <div class="form-group">
              <label for="profile-mobile" class="form-label">Mobile Contact Number</label>
              <input 
                type="tel" 
                id="profile-mobile" 
                name="mobile" 
                class="form-control" 
                value="${user.mobile || ''}" 
                placeholder="+1 555-0199" 
              />
            </div>

            <div class="form-group">
              <label for="profile-dob" class="form-label">Date of Birth</label>
              <input 
                type="date" 
                id="profile-dob" 
                name="dateOfBirth" 
                class="form-control" 
                value="${user.dateOfBirth || ''}" 
              />
            </div>
          </div>

          <div class="profile-actions">
            <button type="button" class="btn btn-outline" id="btn-cancel-profile">
              Cancel
            </button>
            <button type="submit" class="btn btn-primary" id="btn-save-profile">
              Save Profile Changes
            </button>
          </div>
        </form>
      </div>
    </div>
  `;

  // Render header
  const headerContainer = container.querySelector('#profile-header-container');
  if (headerContainer) {
    renderHeader(headerContainer, {
      title: 'User Profile & Settings',
      subtitle: 'Manage your contact information and demographic details.',
      userName: user.fullName,
    });
  }

  const form = container.querySelector('#profile-form');
  const btnSave = container.querySelector('#btn-save-profile');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    store.clearAlert();

    const fullName = form.fullName.value.trim();
    const mobile = form.mobile.value.trim();
    const dateOfBirth = form.dateOfBirth.value;

    try {
      btnSave.disabled = true;
      btnSave.textContent = 'Saving...';

      await authService.updateProfile({ fullName, mobile, dateOfBirth });
      store.setAlert('success', 'Your profile details have been successfully updated.');
      renderProfilePage(container); // Re-render with updated values
    } catch (err) {
      store.setAlert('error', err.message || 'Failed to update profile.');
    } finally {
      btnSave.disabled = false;
      btnSave.textContent = 'Save Profile Changes';
    }
  });

  container.querySelector('#btn-cancel-profile')?.addEventListener('click', () => {
    store.navigate('dashboard');
  });
}
