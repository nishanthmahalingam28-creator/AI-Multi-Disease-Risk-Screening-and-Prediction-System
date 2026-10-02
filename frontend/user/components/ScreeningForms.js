/**
 * Disease-Specific Screening Input Forms.
 * 
 * Implements accessible, validated form controls strictly conforming to the
 * feature schemas established in backend/models/{disease}/feature_schema.json.
 */

export function renderLungCancerForm(container, onSubmit) {
  container.innerHTML = `
    <form id="form-lung-cancer" class="screening-form">
      <div class="form-section-title">
        <span>👤 Demographics & Habits</span>
      </div>

      <div class="form-grid">
        <div class="form-group">
          <label for="gender" class="form-label">Biological Sex <span class="required">*</span></label>
          <select id="gender" name="gender" class="form-control" required>
            <option value="">Select Gender...</option>
            <option value="MALE">Male</option>
            <option value="FEMALE">Female</option>
          </select>
        </div>

        <div class="form-group">
          <label for="age" class="form-label">Age (Years) <span class="required">*</span></label>
          <input type="number" id="age" name="age" class="form-control" min="1" max="120" placeholder="e.g. 55" required />
          <small class="form-hint">Valid range: 1 – 120</small>
        </div>

        <div class="form-group">
          <label for="smoking" class="form-label">Smoking History <span class="required">*</span></label>
          <select id="smoking" name="smoking" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No (Non-Smoker)</option>
            <option value="1">Yes (Regular / Heavy Smoker)</option>
          </select>
        </div>

        <div class="form-group">
          <label for="alcohol_consuming" class="form-label">Alcohol Consumption <span class="required">*</span></label>
          <select id="alcohol_consuming" name="alcohol_consuming" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes (Regular Consumption)</option>
          </select>
        </div>

        <div class="form-group">
          <label for="peer_pressure" class="form-label">Peer Pressure Influencing Habits <span class="required">*</span></label>
          <select id="peer_pressure" name="peer_pressure" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div class="form-group">
          <label for="yellow_fingers" class="form-label">Yellow Nicotine-Stained Fingers <span class="required">*</span></label>
          <select id="yellow_fingers" name="yellow_fingers" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>
      </div>

      <div class="form-section-title">
        <span>🩺 Symptoms & Clinical Factors</span>
      </div>

      <div class="form-grid">
        <div class="form-group">
          <label for="coughing" class="form-label">Chronic or Persistent Cough <span class="required">*</span></label>
          <select id="coughing" name="coughing" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div class="form-group">
          <label for="shortness_of_breath" class="form-label">Shortness of Breath (Dyspnea) <span class="required">*</span></label>
          <select id="shortness_of_breath" name="shortness_of_breath" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div class="form-group">
          <label for="chest_pain" class="form-label">Recurrent Chest Pain <span class="required">*</span></label>
          <select id="chest_pain" name="chest_pain" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div class="form-group">
          <label for="wheezing" class="form-label">Wheezing or Whistling Sound <span class="required">*</span></label>
          <select id="wheezing" name="wheezing" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div class="form-group">
          <label for="swallowing_difficulty" class="form-label">Difficulty Swallowing (Dysphagia) <span class="required">*</span></label>
          <select id="swallowing_difficulty" name="swallowing_difficulty" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div class="form-group">
          <label for="fatigue" class="form-label">Persistent Unexplained Fatigue <span class="required">*</span></label>
          <select id="fatigue" name="fatigue" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div class="form-group">
          <label for="allergy" class="form-label">History of Allergies <span class="required">*</span></label>
          <select id="allergy" name="allergy" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div class="form-group">
          <label for="chronic_disease" class="form-label">History of Chronic Disease <span class="required">*</span></label>
          <select id="chronic_disease" name="chronic_disease" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div class="form-group">
          <label for="anxiety" class="form-label">Frequent Anxiety / Nervousness <span class="required">*</span></label>
          <select id="anxiety" name="anxiety" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>
      </div>

      <div class="form-actions">
        <button type="button" class="btn btn-outline" id="btn-prefill-lung">Fill Example Data</button>
        <button type="submit" class="btn btn-primary btn-submit">Submit Lung Cancer Screening →</button>
      </div>
    </form>
  `;

  const form = container.querySelector('#form-lung-cancer');

  // Prefill helper
  container.querySelector('#btn-prefill-lung')?.addEventListener('click', () => {
    form.gender.value = 'MALE';
    form.age.value = '62';
    form.smoking.value = '1';
    form.alcohol_consuming.value = '1';
    form.peer_pressure.value = '0';
    form.yellow_fingers.value = '1';
    form.coughing.value = '1';
    form.shortness_of_breath.value = '1';
    form.chest_pain.value = '1';
    form.wheezing.value = '1';
    form.swallowing_difficulty.value = '0';
    form.fatigue.value = '1';
    form.allergy.value = '0';
    form.chronic_disease.value = '1';
    form.anxiety.value = '0';
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const payload = {
      gender: form.gender.value,
      age: parseFloat(form.age.value),
      smoking: parseInt(form.smoking.value, 10),
      yellow_fingers: parseInt(form.yellow_fingers.value, 10),
      anxiety: parseInt(form.anxiety.value, 10),
      peer_pressure: parseInt(form.peer_pressure.value, 10),
      chronic_disease: parseInt(form.chronic_disease.value, 10),
      fatigue: parseInt(form.fatigue.value, 10),
      allergy: parseInt(form.allergy.value, 10),
      wheezing: parseInt(form.wheezing.value, 10),
      alcohol_consuming: parseInt(form.alcohol_consuming.value, 10),
      coughing: parseInt(form.coughing.value, 10),
      shortness_of_breath: parseInt(form.shortness_of_breath.value, 10),
      swallowing_difficulty: parseInt(form.swallowing_difficulty.value, 10),
      chest_pain: parseInt(form.chest_pain.value, 10),
    };
    onSubmit(payload);
  });
}

export function renderAsthmaForm(container, onSubmit) {
  container.innerHTML = `
    <form id="form-asthma" class="screening-form">
      <div class="form-section-title">
        <span>👤 Demographics & Clinical History</span>
      </div>

      <div class="form-grid">
        <div class="form-group">
          <label for="asthma-age" class="form-label">Age (Years) <span class="required">*</span></label>
          <input type="number" id="asthma-age" name="Age" class="form-control" min="1" max="120" placeholder="e.g. 34" required />
          <small class="form-hint">Valid range: 1 – 120</small>
        </div>

        <div class="form-group">
          <label for="asthma-gender" class="form-label">Gender <span class="required">*</span></label>
          <select id="asthma-gender" name="Gender" class="form-control" required>
            <option value="">Select Gender...</option>
            <option value="Female">Female</option>
            <option value="Male">Male</option>
            <option value="Other">Other</option>
          </select>
        </div>

        <div class="form-group">
          <label for="asthma-bmi" class="form-label">BMI (kg/m²) <span class="required">*</span></label>
          <input type="number" id="asthma-bmi" name="BMI" class="form-control" min="10" max="65" step="0.1" placeholder="e.g. 24.5" required />
          <small class="form-hint">Valid range: 10.0 – 65.0</small>
        </div>

        <div class="form-group">
          <label for="asthma-smoking" class="form-label">Smoking Status <span class="required">*</span></label>
          <select id="asthma-smoking" name="Smoking_Status" class="form-control" required>
            <option value="">Select...</option>
            <option value="Never">Never Smoked</option>
            <option value="Former">Former Smoker</option>
            <option value="Current">Current Smoker</option>
          </select>
        </div>

        <div class="form-group">
          <label for="asthma-family" class="form-label">Family History of Asthma <span class="required">*</span></label>
          <select id="asthma-family" name="Family_History" class="form-control" required>
            <option value="">Select...</option>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div class="form-group">
          <label for="asthma-allergies" class="form-label">Known Allergies <span class="required">*</span></label>
          <select id="asthma-allergies" name="Allergies" class="form-control" required>
            <option value="">Select...</option>
            <option value="None">None</option>
            <option value="Dust">Dust</option>
            <option value="Pollen">Pollen</option>
            <option value="Pets">Pets</option>
            <option value="Multiple">Multiple Triggers</option>
          </select>
        </div>
      </div>

      <div class="form-section-title">
        <span>🌍 Environment & Comorbidities</span>
      </div>

      <div class="form-grid">
        <div class="form-group">
          <label for="asthma-pollution" class="form-label">Air Pollution Level <span class="required">*</span></label>
          <select id="asthma-pollution" name="Air_Pollution_Level" class="form-control" required>
            <option value="">Select...</option>
            <option value="Low">Low</option>
            <option value="Moderate">Moderate</option>
            <option value="High">High</option>
          </select>
        </div>

        <div class="form-group">
          <label for="asthma-activity" class="form-label">Physical Activity Level <span class="required">*</span></label>
          <select id="asthma-activity" name="Physical_Activity_Level" class="form-control" required>
            <option value="">Select...</option>
            <option value="Sedentary">Sedentary</option>
            <option value="Moderate">Moderate</option>
            <option value="Active">Active</option>
          </select>
        </div>

        <div class="form-group">
          <label for="asthma-occupation" class="form-label">Primary Occupation Environment <span class="required">*</span></label>
          <select id="asthma-occupation" name="Occupation_Type" class="form-control" required>
            <option value="">Select...</option>
            <option value="Indoor">Indoor</option>
            <option value="Outdoor">Outdoor</option>
          </select>
        </div>

        <div class="form-group">
          <label for="asthma-comorbidities" class="form-label">Comorbidities <span class="required">*</span></label>
          <select id="asthma-comorbidities" name="Comorbidities" class="form-control" required>
            <option value="">Select...</option>
            <option value="None">None</option>
            <option value="Diabetes">Diabetes</option>
            <option value="Hypertension">Hypertension</option>
            <option value="Both">Both (Diabetes & Hypertension)</option>
          </select>
        </div>
      </div>

      <div class="form-section-title">
        <span>🧪 Spirometry & Physiological Measures</span>
      </div>

      <div class="form-grid">
        <div class="form-group">
          <label for="asthma-pef" class="form-label">Peak Expiratory Flow (PEF, L/min) <span class="required">*</span></label>
          <input type="number" id="asthma-pef" name="Peak_Expiratory_Flow" class="form-control" min="50" max="900" step="1" placeholder="e.g. 420" required />
          <small class="form-hint">Valid range: 50 – 900 L/min</small>
        </div>

        <div class="form-group">
          <label for="asthma-feno" class="form-label">FeNO Level (ppb) <span class="required">*</span></label>
          <input type="number" id="asthma-feno" name="FeNO_Level" class="form-control" min="0" max="150" step="0.1" placeholder="e.g. 28.5" required />
          <small class="form-hint">Fractional exhaled Nitric Oxide [0 – 150]</small>
        </div>

        <div class="form-group">
          <label for="asthma-adherence" class="form-label">Medication Adherence Ratio <span class="required">*</span></label>
          <input type="number" id="asthma-adherence" name="Medication_Adherence" class="form-control" min="0" max="1" step="0.01" placeholder="e.g. 0.85" required />
          <small class="form-hint">Compliance ratio: 0.0 (none) – 1.0 (perfect)</small>
        </div>

        <div class="form-group">
          <label for="asthma-er" class="form-label">Emergency Room (ER) Visits (Past Year) <span class="required">*</span></label>
          <input type="number" id="asthma-er" name="Number_of_ER_Visits" class="form-control" min="0" max="20" placeholder="e.g. 1" required />
          <small class="form-hint">Valid range: 0 – 20</small>
        </div>
      </div>

      <div class="form-actions">
        <button type="button" class="btn btn-outline" id="btn-prefill-asthma">Fill Example Data</button>
        <button type="submit" class="btn btn-primary btn-submit">Submit Asthma Screening →</button>
      </div>
    </form>
  `;

  const form = container.querySelector('#form-asthma');

  // Prefill helper
  container.querySelector('#btn-prefill-asthma')?.addEventListener('click', () => {
    form.Age.value = '42';
    form.Gender.value = 'Female';
    form.BMI.value = '26.4';
    form.Smoking_Status.value = 'Former';
    form.Family_History.value = '1';
    form.Allergies.value = 'Pollen';
    form.Air_Pollution_Level.value = 'Moderate';
    form.Physical_Activity_Level.value = 'Moderate';
    form.Occupation_Type.value = 'Indoor';
    form.Comorbidities.value = 'None';
    form.Peak_Expiratory_Flow.value = '380';
    form.FeNO_Level.value = '35.0';
    form.Medication_Adherence.value = '0.90';
    form.Number_of_ER_Visits.value = '1';
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const payload = {
      Age: parseFloat(form.Age.value),
      Gender: form.Gender.value,
      BMI: parseFloat(form.BMI.value),
      Smoking_Status: form.Smoking_Status.value,
      Family_History: parseInt(form.Family_History.value, 10),
      Allergies: form.Allergies.value,
      Air_Pollution_Level: form.Air_Pollution_Level.value,
      Physical_Activity_Level: form.Physical_Activity_Level.value,
      Occupation_Type: form.Occupation_Type.value,
      Comorbidities: form.Comorbidities.value,
      Medication_Adherence: parseFloat(form.Medication_Adherence.value),
      Number_of_ER_Visits: parseFloat(form.Number_of_ER_Visits.value),
      Peak_Expiratory_Flow: parseFloat(form.Peak_Expiratory_Flow.value),
      FeNO_Level: parseFloat(form.FeNO_Level.value),
    };
    onSubmit(payload);
  });
}

export function renderParkinsonsForm(container, onSubmit) {
  container.innerHTML = `
    <form id="form-parkinsons" class="screening-form">
      <div class="preset-banner">
        <div>
          <strong>Acoustic Phonation Input:</strong> 22 acoustic voice measures required.
          Use voice biomarker test values or prefill using clinically validated reference samples.
        </div>
        <div class="preset-buttons">
          <button type="button" class="btn btn-sm btn-outline-success" id="btn-prefill-parkinsons-healthy">
            Load Healthy Sample
          </button>
          <button type="button" class="btn btn-sm btn-outline-danger" id="btn-prefill-parkinsons-risk">
            Load At-Risk Sample
          </button>
        </div>
      </div>

      <div class="form-section-title">
        <span>🎙️ Vocal Fundamental Frequencies (Hz)</span>
      </div>

      <div class="form-grid">
        <div class="form-group">
          <label for="p-fo" class="form-label">MDVP:Fo(Hz) (Average Frequency) <span class="required">*</span></label>
          <input type="number" id="p-fo" name="MDVP:Fo(Hz)" class="form-control" min="50" max="400" step="0.001" placeholder="e.g. 119.992" required />
          <small class="form-hint">Range: 50.0 – 400.0 Hz</small>
        </div>

        <div class="form-group">
          <label for="p-fhi" class="form-label">MDVP:Fhi(Hz) (Max Frequency) <span class="required">*</span></label>
          <input type="number" id="p-fhi" name="MDVP:Fhi(Hz)" class="form-control" min="50" max="700" step="0.001" placeholder="e.g. 157.302" required />
          <small class="form-hint">Range: 50.0 – 700.0 Hz</small>
        </div>

        <div class="form-group">
          <label for="p-flo" class="form-label">MDVP:Flo(Hz) (Min Frequency) <span class="required">*</span></label>
          <input type="number" id="p-flo" name="MDVP:Flo(Hz)" class="form-control" min="40" max="350" step="0.001" placeholder="e.g. 74.997" required />
          <small class="form-hint">Range: 40.0 – 350.0 Hz</small>
        </div>
      </div>

      <div class="form-section-title">
        <span>📈 Vocal Frequency Jitter Measures</span>
      </div>

      <div class="form-grid">
        <div class="form-group">
          <label for="p-jitter-pct" class="form-label">MDVP:Jitter(%) <span class="required">*</span></label>
          <input type="number" id="p-jitter-pct" name="MDVP:Jitter(%)" class="form-control" min="0" max="0.1" step="0.00001" placeholder="e.g. 0.00784" required />
          <small class="form-hint">Range: 0.0 – 0.1</small>
        </div>

        <div class="form-group">
          <label for="p-jitter-abs" class="form-label">MDVP:Jitter(Abs) (Seconds) <span class="required">*</span></label>
          <input type="number" id="p-jitter-abs" name="MDVP:Jitter(Abs)" class="form-control" min="0" max="0.001" step="0.000001" placeholder="e.g. 0.00007" required />
          <small class="form-hint">Range: 0.0 – 0.001 s</small>
        </div>

        <div class="form-group">
          <label for="p-rap" class="form-label">MDVP:RAP <span class="required">*</span></label>
          <input type="number" id="p-rap" name="MDVP:RAP" class="form-control" min="0" max="0.1" step="0.00001" placeholder="e.g. 0.00370" required />
          <small class="form-hint">Range: 0.0 – 0.1</small>
        </div>

        <div class="form-group">
          <label for="p-ppq" class="form-label">MDVP:PPQ <span class="required">*</span></label>
          <input type="number" id="p-ppq" name="MDVP:PPQ" class="form-control" min="0" max="0.1" step="0.00001" placeholder="e.g. 0.00554" required />
          <small class="form-hint">Range: 0.0 – 0.1</small>
        </div>

        <div class="form-group">
          <label for="p-ddp" class="form-label">Jitter:DDP <span class="required">*</span></label>
          <input type="number" id="p-ddp" name="Jitter:DDP" class="form-control" min="0" max="0.3" step="0.00001" placeholder="e.g. 0.01109" required />
          <small class="form-hint">Range: 0.0 – 0.3</small>
        </div>
      </div>

      <div class="form-section-title">
        <span>🌊 Vocal Amplitude Shimmer Measures</span>
      </div>

      <div class="form-grid">
        <div class="form-group">
          <label for="p-shimmer" class="form-label">MDVP:Shimmer <span class="required">*</span></label>
          <input type="number" id="p-shimmer" name="MDVP:Shimmer" class="form-control" min="0" max="0.5" step="0.0001" placeholder="e.g. 0.04374" required />
          <small class="form-hint">Range: 0.0 – 0.5</small>
        </div>

        <div class="form-group">
          <label for="p-shimmer-db" class="form-label">MDVP:Shimmer(dB) <span class="required">*</span></label>
          <input type="number" id="p-shimmer-db" name="MDVP:Shimmer(dB)" class="form-control" min="0" max="5" step="0.001" placeholder="e.g. 0.426" required />
          <small class="form-hint">Range: 0.0 – 5.0 dB</small>
        </div>

        <div class="form-group">
          <label for="p-apq3" class="form-label">Shimmer:APQ3 <span class="required">*</span></label>
          <input type="number" id="p-apq3" name="Shimmer:APQ3" class="form-control" min="0" max="0.3" step="0.00001" placeholder="e.g. 0.02182" required />
          <small class="form-hint">Range: 0.0 – 0.3</small>
        </div>

        <div class="form-group">
          <label for="p-apq5" class="form-label">Shimmer:APQ5 <span class="required">*</span></label>
          <input type="number" id="p-apq5" name="Shimmer:APQ5" class="form-control" min="0" max="0.3" step="0.00001" placeholder="e.g. 0.03130" required />
          <small class="form-hint">Range: 0.0 – 0.3</small>
        </div>

        <div class="form-group">
          <label for="p-apq" class="form-label">MDVP:APQ (11-pt) <span class="required">*</span></label>
          <input type="number" id="p-apq" name="MDVP:APQ" class="form-control" min="0" max="0.5" step="0.00001" placeholder="e.g. 0.02971" required />
          <small class="form-hint">Range: 0.0 – 0.5</small>
        </div>

        <div class="form-group">
          <label for="p-dda" class="form-label">Shimmer:DDA <span class="required">*</span></label>
          <input type="number" id="p-dda" name="Shimmer:DDA" class="form-control" min="0" max="1" step="0.0001" placeholder="e.g. 0.06545" required />
          <small class="form-hint">Range: 0.0 – 1.0</small>
        </div>
      </div>

      <div class="form-section-title">
        <span>🎚️ Harmonicity & Noise Ratios</span>
      </div>

      <div class="form-grid">
        <div class="form-group">
          <label for="p-nhr" class="form-label">NHR (Noise-to-Harmonics) <span class="required">*</span></label>
          <input type="number" id="p-nhr" name="NHR" class="form-control" min="0" max="1" step="0.0001" placeholder="e.g. 0.02211" required />
          <small class="form-hint">Range: 0.0 – 1.0</small>
        </div>

        <div class="form-group">
          <label for="p-hnr" class="form-label">HNR (Harmonics-to-Noise, dB) <span class="required">*</span></label>
          <input type="number" id="p-hnr" name="HNR" class="form-control" min="0" max="60" step="0.01" placeholder="e.g. 21.033" required />
          <small class="form-hint">Range: 0.0 – 60.0 dB</small>
        </div>
      </div>

      <div class="form-section-title">
        <span>🌀 Nonlinear Complexity & Dynamical Measures</span>
      </div>

      <div class="form-grid">
        <div class="form-group">
          <label for="p-rpde" class="form-label">RPDE (Recurrence Period Entropy) <span class="required">*</span></label>
          <input type="number" id="p-rpde" name="RPDE" class="form-control" min="0" max="1" step="0.0001" placeholder="e.g. 0.414783" required />
          <small class="form-hint">Range: 0.0 – 1.0</small>
        </div>

        <div class="form-group">
          <label for="p-dfa" class="form-label">DFA (Fractal Scaling Exponent) <span class="required">*</span></label>
          <input type="number" id="p-dfa" name="DFA" class="form-control" min="0" max="1.5" step="0.0001" placeholder="e.g. 0.815285" required />
          <small class="form-hint">Range: 0.0 – 1.5</small>
        </div>

        <div class="form-group">
          <label for="p-spread1" class="form-label">spread1 (Log-Frequency Variation) <span class="required">*</span></label>
          <input type="number" id="p-spread1" name="spread1" class="form-control" min="-15" max="0" step="0.0001" placeholder="e.g. -4.813031" required />
          <small class="form-hint">Range: -15.0 – 0.0</small>
        </div>

        <div class="form-group">
          <label for="p-spread2" class="form-label">spread2 (Frequency Variation) <span class="required">*</span></label>
          <input type="number" id="p-spread2" name="spread2" class="form-control" min="0" max="1" step="0.0001" placeholder="e.g. 0.266482" required />
          <small class="form-hint">Range: 0.0 – 1.0</small>
        </div>

        <div class="form-group">
          <label for="p-d2" class="form-label">D2 (Correlation Dimension) <span class="required">*</span></label>
          <input type="number" id="p-d2" name="D2" class="form-control" min="0.5" max="5" step="0.0001" placeholder="e.g. 2.301442" required />
          <small class="form-hint">Range: 0.5 – 5.0</small>
        </div>

        <div class="form-group">
          <label for="p-ppe" class="form-label">PPE (Pitch Period Entropy) <span class="required">*</span></label>
          <input type="number" id="p-ppe" name="PPE" class="form-control" min="0" max="1" step="0.0001" placeholder="e.g. 0.284654" required />
          <small class="form-hint">Range: 0.0 – 1.0</small>
        </div>
      </div>

      <div class="form-actions">
        <button type="submit" class="btn btn-primary btn-submit">Submit Parkinson's Acoustic Screening →</button>
      </div>
    </form>
  `;

  const form = container.querySelector('#form-parkinsons');

  const populateFields = (data) => {
    for (const [k, v] of Object.entries(data)) {
      const field = form.elements[k];
      if (field) field.value = v;
    }
  };

  container.querySelector('#btn-prefill-parkinsons-healthy')?.addEventListener('click', () => {
    populateFields({
      'MDVP:Fo(Hz)': 197.076,
      'MDVP:Fhi(Hz)': 206.896,
      'MDVP:Flo(Hz)': 192.055,
      'MDVP:Jitter(%)': 0.00289,
      'MDVP:Jitter(Abs)': 0.00001,
      'MDVP:RAP': 0.00166,
      'MDVP:PPQ': 0.00168,
      'Jitter:DDP': 0.00498,
      'MDVP:Shimmer': 0.01098,
      'MDVP:Shimmer(dB)': 0.097,
      'Shimmer:APQ3': 0.00563,
      'Shimmer:APQ5': 0.00680,
      'MDVP:APQ': 0.00802,
      'Shimmer:DDA': 0.01689,
      'NHR': 0.00339,
      'HNR': 26.775,
      'RPDE': 0.422229,
      'DFA': 0.741367,
      'spread1': -7.348300,
      'spread2': 0.177551,
      'D2': 1.743867,
      'PPE': 0.085569,
    });
  });

  container.querySelector('#btn-prefill-parkinsons-risk')?.addEventListener('click', () => {
    populateFields({
      'MDVP:Fo(Hz)': 119.992,
      'MDVP:Fhi(Hz)': 157.302,
      'MDVP:Flo(Hz)': 74.997,
      'MDVP:Jitter(%)': 0.00784,
      'MDVP:Jitter(Abs)': 0.00007,
      'MDVP:RAP': 0.00370,
      'MDVP:PPQ': 0.00554,
      'Jitter:DDP': 0.01109,
      'MDVP:Shimmer': 0.04374,
      'MDVP:Shimmer(dB)': 0.426,
      'Shimmer:APQ3': 0.02182,
      'Shimmer:APQ5': 0.03130,
      'MDVP:APQ': 0.02971,
      'Shimmer:DDA': 0.06545,
      'NHR': 0.02211,
      'HNR': 21.033,
      'RPDE': 0.414783,
      'DFA': 0.815285,
      'spread1': -4.813031,
      'spread2': 0.266482,
      'D2': 2.301442,
      'PPE': 0.284654,
    });
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const payload = {
      'MDVP:Fo(Hz)': parseFloat(form['MDVP:Fo(Hz)'].value),
      'MDVP:Fhi(Hz)': parseFloat(form['MDVP:Fhi(Hz)'].value),
      'MDVP:Flo(Hz)': parseFloat(form['MDVP:Flo(Hz)'].value),
      'MDVP:Jitter(%)': parseFloat(form['MDVP:Jitter(%)'].value),
      'MDVP:Jitter(Abs)': parseFloat(form['MDVP:Jitter(Abs)'].value),
      'MDVP:RAP': parseFloat(form['MDVP:RAP'].value),
      'MDVP:PPQ': parseFloat(form['MDVP:PPQ'].value),
      'Jitter:DDP': parseFloat(form['Jitter:DDP'].value),
      'MDVP:Shimmer': parseFloat(form['MDVP:Shimmer'].value),
      'MDVP:Shimmer(dB)': parseFloat(form['MDVP:Shimmer(dB)'].value),
      'Shimmer:APQ3': parseFloat(form['Shimmer:APQ3'].value),
      'Shimmer:APQ5': parseFloat(form['Shimmer:APQ5'].value),
      'MDVP:APQ': parseFloat(form['MDVP:APQ'].value),
      'Shimmer:DDA': parseFloat(form['Shimmer:DDA'].value),
      'NHR': parseFloat(form['NHR'].value),
      'HNR': parseFloat(form['HNR'].value),
      'RPDE': parseFloat(form['RPDE'].value),
      'DFA': parseFloat(form['DFA'].value),
      'spread1': parseFloat(form['spread1'].value),
      'spread2': parseFloat(form['spread2'].value),
      'D2': parseFloat(form['D2'].value),
      'PPE': parseFloat(form['PPE'].value),
    };
    onSubmit(payload);
  });
}
