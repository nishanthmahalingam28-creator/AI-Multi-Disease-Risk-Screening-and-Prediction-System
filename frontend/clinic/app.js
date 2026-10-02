/**
 * Clinic Module Orchestrator and Client Router.
 * 
 * Manages clinic navigation, role-based views, patient screening workflow,
 * and session state.
 */

import { clinicStore } from './hooks/store.js';
import { clinicAuthService } from './services/auth.js';
import { renderAlert } from '../components/Alert.js';
import { renderClinicLoginPage } from './pages/Login.js';
import { renderClinicDashboardPage } from './pages/Dashboard.js';
import { renderClinicProfilePage } from './pages/Profile.js';
import { renderPatientScreeningPage } from './pages/PatientScreening.js';
import { renderClinicResultPage } from './pages/Result.js';
import { renderClinicHistoryPage } from './pages/History.js';
import { renderClinicReportPage } from './pages/Report.js';

export function renderClinicNavbar(container) {
  const { clinicUser, currentRoute } = clinicStore.state;

  container.innerHTML = `
    <header class="navbar clinic-navbar" style="border-bottom: 2px solid var(--accent-teal);">
      <div class="nav-container">
        <div class="nav-brand" id="clinic-nav-brand" role="button" tabindex="0">
          <span class="brand-icon" style="background: rgba(20, 184, 166, 0.15); border-color: rgba(20, 184, 166, 0.3);">🏥</span>
          <div class="brand-text">
            <span class="brand-title">AI Multi-Disease Risk</span>
            <span class="brand-subtitle" style="color: var(--accent-teal);">Clinical Provider Portal</span>
          </div>
        </div>

        <nav class="nav-links">
          ${clinicUser ? `
            <button class="nav-link ${currentRoute === 'dashboard' ? 'active' : ''}" id="clinic-nav-dashboard">
              📊 Clinic Dashboard
            </button>
            <button class="nav-link ${currentRoute === 'history' ? 'active' : ''}" id="clinic-nav-history">
              📋 Screening Records
            </button>
            <button class="nav-link ${currentRoute === 'profile' ? 'active' : ''}" id="clinic-nav-profile">
              👨‍⚕️ Provider Profile
            </button>
            <div class="user-badge" id="clinic-user-badge" title="${clinicUser.email}">
              <span class="user-avatar" style="background: var(--accent-teal);">⚕️</span>
              <div style="display: flex; flex-direction: column; text-align: left;">
                <span class="user-name" style="line-height: 1.1;">${clinicUser.fullName}</span>
                <span style="font-size: 0.7rem; color: var(--accent-teal);">${clinicUser.clinicName}</span>
              </div>
            </div>
            <button class="btn btn-outline-danger btn-sm" id="btn-clinic-logout" title="Log out of clinical shift">
              🚪 Logout
            </button>
          ` : `
            <a href="./index.html" class="nav-link">
              ← Patient Portal
            </a>
            <button class="btn btn-primary btn-sm" id="clinic-nav-login">
              Provider Sign In
            </button>
          `}
        </nav>
      </div>
    </header>
  `;

  const brand = container.querySelector('#clinic-nav-brand');
  if (brand) {
    brand.addEventListener('click', () => {
      clinicStore.navigate(clinicAuthService.isAuthenticated() ? 'dashboard' : 'login');
    });
  }

  const navDashboard = container.querySelector('#clinic-nav-dashboard');
  if (navDashboard) {
    navDashboard.addEventListener('click', () => clinicStore.navigate('dashboard'));
  }

  const navHistory = container.querySelector('#clinic-nav-history');
  if (navHistory) {
    navHistory.addEventListener('click', () => clinicStore.navigate('history'));
  }

  const navProfile = container.querySelector('#clinic-nav-profile');
  if (navProfile) {
    navProfile.addEventListener('click', () => clinicStore.navigate('profile'));
  }

  const btnLogout = container.querySelector('#btn-clinic-logout');
  if (btnLogout) {
    btnLogout.addEventListener('click', () => {
      clinicAuthService.logout();
      clinicStore.setAlert('info', 'Clinic provider session closed successfully.');
      clinicStore.navigate('login');
    });
  }

  const navLogin = container.querySelector('#clinic-nav-login');
  if (navLogin) {
    navLogin.addEventListener('click', () => clinicStore.navigate('login'));
  }
}

export class ClinicApp {
  constructor() {
    this.navbarContainer = document.getElementById('navbar-root');
    this.alertContainer = document.getElementById('alert-root');
    this.mainContainer = document.getElementById('app-root');
    this.loadingOverlay = document.getElementById('loading-overlay');
    this.loadingText = document.getElementById('loading-text');

    this.init();
  }

  init() {
    clinicStore.subscribe((state) => {
      this.render(state);
    });
  }

  render(state) {
    // 1. Render Clinic Navbar
    if (this.navbarContainer) {
      renderClinicNavbar(this.navbarContainer);
    }

    // 2. Render Alert
    if (this.alertContainer) {
      renderAlert(this.alertContainer);
    }

    // 3. Render Loading Overlay
    if (this.loadingOverlay) {
      if (state.isLoading) {
        this.loadingOverlay.classList.remove('hidden');
        if (this.loadingText) {
          this.loadingText.textContent = state.loadingMessage || 'Evaluating statistical model...';
        }
      } else {
        this.loadingOverlay.classList.add('hidden');
      }
    }

    // 4. Render Active Clinic View
    if (!this.mainContainer) return;

    switch (state.currentRoute) {
      case 'login':
        renderClinicLoginPage(this.mainContainer);
        break;
      case 'dashboard':
        renderClinicDashboardPage(this.mainContainer);
        break;
      case 'profile':
        renderClinicProfilePage(this.mainContainer);
        break;
      case 'screening':
        renderPatientScreeningPage(this.mainContainer);
        break;
      case 'result':
        renderClinicResultPage(this.mainContainer);
        break;
      case 'history':
        renderClinicHistoryPage(this.mainContainer);
        break;
      case 'report':
        renderClinicReportPage(this.mainContainer);
        break;
      default:
        renderClinicDashboardPage(this.mainContainer);
        break;
    }
  }
}


if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    new ClinicApp();
  });
}
