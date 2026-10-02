/**
 * Lightweight Reactive State Store for AI Multi-Disease Screening Frontend.
 * 
 * Manages active route, user authentication state, current disease selection,
 * active screening input state, prediction results, and notification alerts.
 */

import { authService } from '../services/auth.js';

class AppStore {
  constructor() {
    this.state = {
      user: authService.getUser(),
      currentRoute: authService.isAuthenticated() ? 'dashboard' : 'login',
      selectedDisease: null,
      predictionResult: null,
      selectedScreeningId: null,
      activeReportScreening: null,
      isLoading: false,
      loadingMessage: '',
      alert: null, // { type: 'success' | 'error' | 'warning' | 'info', message: string }
    };

    this.subscribers = [];

    // Keep auth in sync
    authService.onAuthStateChange(user => {
      this.setState({ user });
      if (!user && ['dashboard', 'profile', 'screening', 'result', 'history', 'report'].includes(this.state.currentRoute)) {
        this.navigate('login');
      }
    });
  }

  /**
   * Subscribe to state modifications.
   */
  subscribe(callback) {
    this.subscribers.push(callback);
    callback(this.state);
    return () => {
      this.subscribers = this.subscribers.filter(cb => cb !== callback);
    };
  }

  _notify() {
    for (const callback of this.subscribers) {
      try {
        callback(this.state);
      } catch (err) {
        console.error('Store subscriber error:', err);
      }
    }
  }

  /**
   * Update partial state and notify subscribers.
   */
  setState(partialState) {
    this.state = { ...this.state, ...partialState };
    this._notify();
  }

  /**
   * Navigate between application views.
   */
  navigate(route, params = {}) {
    // Route guard for protected views
    const protectedRoutes = ['dashboard', 'profile', 'screening', 'result', 'history', 'report'];
    if (protectedRoutes.includes(route) && !authService.isAuthenticated()) {
      this.setAlert('warning', 'Please sign in to access the screening platform.');
      this.setState({ currentRoute: 'login' });
      return;
    }

    this.setState({
      currentRoute: route,
      selectedDisease: params.disease || this.state.selectedDisease,
      predictionResult: params.result !== undefined ? params.result : this.state.predictionResult,
      selectedScreeningId: params.screeningId !== undefined ? params.screeningId : this.state.selectedScreeningId,
      activeReportScreening: params.screening !== undefined ? params.screening : this.state.activeReportScreening,
      alert: params.alert || null,
    });
    
    if (typeof window !== 'undefined') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }


  /**
   * Set global user alert.
   */
  setAlert(type, message) {
    this.setState({ alert: { type, message } });
  }

  /**
   * Clear global user alert.
   */
  clearAlert() {
    this.setState({ alert: null });
  }

  /**
   * Set global loading state.
   */
  setLoading(isLoading, loadingMessage = 'Processing...') {
    this.setState({ isLoading, loadingMessage });
  }
}

export const store = new AppStore();
