/**
 * Screening Result View Page.
 * 
 * Embeds ResultCard component and provides back navigation.
 */

import { renderResultCard } from '../components/ResultCard.js';
import { renderHeader } from '../components/Header.js';
import { store } from '../hooks/store.js';

export function renderResultPage(container) {
  const { selectedDisease, predictionResult, user } = store.state;

  container.innerHTML = `
    <div class="result-page">
      <div id="result-header-container"></div>
      <div id="result-card-container"></div>
    </div>
  `;

  // Render header
  const headerContainer = container.querySelector('#result-header-container');
  if (headerContainer) {
    renderHeader(headerContainer, {
      title: 'Screening Risk Evaluation',
      subtitle: 'Model-derived likelihood computation based on provided clinical factors.',
      userName: user ? user.fullName : '',
    });
  }

  const resultContainer = container.querySelector('#result-card-container');
  if (resultContainer) {
    renderResultCard(resultContainer, {
      disease: selectedDisease,
      result: predictionResult,
    });
  }
}
