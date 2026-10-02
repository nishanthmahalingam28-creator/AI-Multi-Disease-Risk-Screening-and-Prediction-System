/**
 * Global Navigation Bar Component.
 * 
 * Displays brand identity, current route links, user profile badge, and logout.
 */

import { store } from '../user/hooks/store.js';
import { authService } from '../user/services/auth.js';

export function renderNavbar(container) {
  const { user, currentRoute } = store.state;

  const html = `
    <header class="navbar">
      <div class="nav-container">
        <div class="nav-brand" id="nav-brand" role="button" tabindex="0">
          <span class="brand-icon">⚕️</span>
          <div class="brand-text">
            <span class="brand-title">AI Multi-Disease Risk</span>
            <span class="brand-subtitle">Screening Platform</span>
          </div>
        </div>

        <nav class="nav-links">
          ${user ? `
            <button class="nav-link ${currentRoute === 'dashboard' ? 'active' : ''}" id="nav-dashboard">
              📊 Dashboard
            </button>
            <button class="nav-link ${currentRoute === 'history' ? 'active' : ''}" id="nav-history">
              📋 History
            </button>
            <button class="nav-link ${currentRoute === 'profile' ? 'active' : ''}" id="nav-profile">
              👤 Profile
            </button>
            <div class="user-badge" id="user-badge" title="${user.email}">
              <span class="user-avatar">${user.fullName ? user.fullName[0].toUpperCase() : 'U'}</span>
              <span class="user-name">${user.fullName}</span>
            </div>
            <button class="btn btn-outline-danger btn-sm" id="btn-logout" title="Log out">
              🚪 Logout
            </button>
          ` : `
            <button class="nav-link ${currentRoute === 'login' ? 'active' : ''}" id="nav-login">
              Sign In
            </button>
            <button class="btn btn-primary btn-sm" id="nav-signup">
              Create Account
            </button>
          `}
          <a href="./clinic.html" class="nav-link" id="nav-clinic-portal" title="Open Clinical Provider Console">
            🏥 Clinic Portal
          </a>
        </nav>
      </div>
    </header>
  `;

  container.innerHTML = html;

  // Attach event handlers
  const brand = container.querySelector('#nav-brand');
  if (brand) {
    brand.addEventListener('click', () => {
      store.navigate(authService.isAuthenticated() ? 'dashboard' : 'login');
    });
  }

  const navDashboard = container.querySelector('#nav-dashboard');
  if (navDashboard) {
    navDashboard.addEventListener('click', () => store.navigate('dashboard'));
  }

  const navHistory = container.querySelector('#nav-history');
  if (navHistory) {
    navHistory.addEventListener('click', () => store.navigate('history'));
  }

  const navProfile = container.querySelector('#nav-profile');
  if (navProfile) {
    navProfile.addEventListener('click', () => store.navigate('profile'));
  }


  const btnLogout = container.querySelector('#btn-logout');
  if (btnLogout) {
    btnLogout.addEventListener('click', () => {
      authService.logout();
      store.setAlert('info', 'You have been successfully logged out.');
      store.navigate('login');
    });
  }

  const navLogin = container.querySelector('#nav-login');
  if (navLogin) {
    navLogin.addEventListener('click', () => store.navigate('login'));
  }

  const navSignup = container.querySelector('#nav-signup');
  if (navSignup) {
    navSignup.addEventListener('click', () => store.navigate('signup'));
  }
}
