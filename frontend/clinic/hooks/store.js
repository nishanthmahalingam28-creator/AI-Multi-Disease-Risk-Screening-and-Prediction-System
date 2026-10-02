/**
 * Reactive State Store for Clinic Provider Module.
 * 
 * Manages clinic routing, clinic authentication state, selected disease,
 * patient identity intake state, prediction result, and clinical alerts.
 */

import { clinicAuthService } from '../services/auth.js';

class ClinicStore {
  constructor() {
    this.state = {
      clinicUser: clinicAuthService.getClinicUser(),
      currentRoute: clinicAuthService.isAuthenticated() ? 'dashboard' : 'login',
      selectedDisease: null,
      patientInfo: null, // { fullName, email, mobile, dateOfBirth }
      predictionResult: null,
      selectedScreeningId: null,
      activeReportScreening: null,
      isLoading: false,
      loadingMessage: '',
      alert: null,
    };

    this.subscribers = [];

    // Keep clinic auth in sync
    clinicAuthService.onAuthStateChange(clinicUser => {
      this.setState({ clinicUser });
      if (!clinicUser && ['dashboard', 'profile', 'screening', 'result', 'history', 'report'].includes(this.state.currentRoute)) {
        this.navigate('login');
      }
    });
  }

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
        console.error('Clinic store subscriber error:', err);
      }
    }
  }

  setState(partialState) {
    this.state = { ...this.state, ...partialState };
    this._notify();
  }

  /**
   * Navigate between clinic views with role enforcement.
   */
  navigate(route, params = {}) {
    const protectedRoutes = ['dashboard', 'profile', 'screening', 'result', 'history', 'report'];
    if (protectedRoutes.includes(route) && !clinicAuthService.isAuthenticated()) {
      this.setAlert('warning', 'Please sign in with authorized clinic credentials to access the clinical provider portal.');
      this.setState({ currentRoute: 'login' });
      return;
    }

    this.setState({
      currentRoute: route,
      selectedDisease: params.disease !== undefined ? params.disease : this.state.selectedDisease,
      patientInfo: params.patientInfo !== undefined ? params.patientInfo : this.state.patientInfo,
      predictionResult: params.result !== undefined ? params.result : this.state.predictionResult,
      selectedScreeningId: params.screeningId !== undefined ? params.screeningId : this.state.selectedScreeningId,
      activeReportScreening: params.screening !== undefined ? params.screening : this.state.activeReportScreening,
      alert: params.alert || null,
    });

    if (typeof window !== 'undefined') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }


  setAlert(type, message) {
    this.setState({ alert: { type, message } });
  }

  clearAlert() {
    this.setState({ alert: null });
  }

  setLoading(isLoading, loadingMessage = 'Evaluating statistical model...') {
    this.setState({ isLoading, loadingMessage });
  }
}

export const clinicStore = new ClinicStore();
