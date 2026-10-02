/**
 * Main Application Orchestrator and Client Router.
 * 
 * Subscribes to reactive store state, renders navbar, alerts, active page views,
 * and loading indicators.
 */

import { store } from './hooks/store.js';
import { renderNavbar } from '../components/Navbar.js';
import { renderAlert } from '../components/Alert.js';
import { renderLoginPage } from './pages/Login.js';
import { renderSignupPage } from './pages/Signup.js';
import { renderDashboardPage } from './pages/Dashboard.js';
import { renderProfilePage } from './pages/Profile.js';
import { renderScreeningPage } from './pages/Screening.js';
import { renderResultPage } from './pages/Result.js';
import { renderUserHistoryPage } from './pages/History.js';
import { renderUserReportPage } from './pages/Report.js';

export class App {
  constructor() {
    this.navbarContainer = document.getElementById('navbar-root');
    this.alertContainer = document.getElementById('alert-root');
    this.mainContainer = document.getElementById('app-root');
    this.loadingOverlay = document.getElementById('loading-overlay');
    this.loadingText = document.getElementById('loading-text');

    this.init();
  }

  init() {
    // Subscribe to state updates
    store.subscribe((state) => {
      this.render(state);
    });
  }

  render(state) {
    // 1. Render Navbar
    if (this.navbarContainer) {
      renderNavbar(this.navbarContainer);
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

    // 4. Render Active View
    if (!this.mainContainer) return;

    switch (state.currentRoute) {
      case 'login':
        renderLoginPage(this.mainContainer);
        break;
      case 'signup':
        renderSignupPage(this.mainContainer);
        break;
      case 'dashboard':
        renderDashboardPage(this.mainContainer);
        break;
      case 'profile':
        renderProfilePage(this.mainContainer);
        break;
      case 'screening':
        renderScreeningPage(this.mainContainer);
        break;
      case 'result':
        renderResultPage(this.mainContainer);
        break;
      case 'history':
        renderUserHistoryPage(this.mainContainer);
        break;
      case 'report':
        renderUserReportPage(this.mainContainer);
        break;
      default:
        renderDashboardPage(this.mainContainer);
        break;
    }
  }
}


// Auto-bootstrap once DOM is loaded
if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    new App();
  });
}
