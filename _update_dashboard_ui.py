from pathlib import Path
import re
p=Path('index.html')
s=p.read_text(encoding='utf-8-sig')

def replace_block(start, end, new):
    global s
    pattern = re.compile(re.escape(start) + r'.*?(?=' + re.escape(end) + r')', re.S)
    s, n = pattern.subn(new, s, count=1)
    if n != 1:
        raise RuntimeError(f'block not found: {start[:60]}')

def replace_once(old, new):
    global s
    if old not in s:
        raise RuntimeError(f'text not found: {old[:100]}')
    s=s.replace(old,new,1)

# Make the global shell and project identity match the root simulator.
s=s.replace('<span class="brand-badge">V2.4</span>', '<span class="brand-badge">ROOT</span>')
replace_once('''        <div class="env-badge-group">
          <button class="env-btn active" id="btnEnvV2" onclick="setEnv(\'v2\')" title="V2 Production: 16x16 Grid, 8 Vehicles">V2 (16x16 / 8 AVs)</button>
          <button class="env-btn" id="btnEnvV1" onclick="setEnv(\'v1\')" title="V1 Baseline Checkpoint: 10x10 Grid, 4 Vehicles">V1 (10x10 / 4 AVs)</button>
        </div>''','''        <div class="env-badge-group"><span class="env-btn active">10×10 GRID · 4 VEHICLES</span></div>''')
s=s.replace("    let currentEnv = 'v2'; // 'v1' or 'v2'\n", '')
replace_block('    function setEnv(env) {', '    function toggleRouteVectors(checked) {', '')

# Benchmark table: only policies that are actually implemented in the root project.
replace_block('    function renderBenchmarkTable() {', '    // ==========================================================================\n    // CHART.JS INITIALIZATION', '''    function renderBenchmarkTable() {
      const tbody = document.getElementById('benchmarkTableBody');
      const rd = getReplayData();
      if (!tbody || !rd) return;
      tbody.innerHTML = '';
      const res = rd.results || {};
      const b = res.baseline || {};
      const q = res.qlearning || {};
      const metrics = [
        { key: 'avg_wait', label: 'Overall Average Wait Time (steps)', unit: 'steps', lowerBetter: true },
        { key: 'surge_wait', label: 'Surge-Phase Wait Time (steps)', unit: 'steps', lowerBetter: true },
        { key: 'normal_wait', label: 'Normal-Phase Wait Time (steps)', unit: 'steps', lowerBetter: true },
        { key: 'served', label: 'Completed Passenger Trips', unit: 'served', lowerBetter: false },
        { key: 'dropped', label: 'Dropped Passenger Requests', unit: 'dropped', lowerBetter: true },
        { key: 'total_dist', label: 'Fleet Total Distance Travelled', unit: 'units', lowerBetter: true }
      ];
      metrics.forEach(m => {
        const bVal = b[m.key]?.mean ?? 0;
        const bStd = b[m.key]?.std ?? 0;
        const qVal = q[m.key]?.mean ?? 0;
        const qStd = q[m.key]?.std ?? 0;
        const delta = qVal - bVal;
        const better = m.lowerBetter ? delta < 0 : delta > 0;
        const tr = document.createElement('tr');
        tr.innerHTML = `<td><strong>${m.label}</strong></td>
          <td style="font-family:var(--font-mono)">${bVal.toFixed(2)} &plusmn; ${bStd.toFixed(2)}</td>
          <td style="font-family:var(--font-mono);font-weight:700;color:var(--accent-blue)">${qVal.toFixed(2)} &plusmn; ${qStd.toFixed(2)}</td>
          <td><span class="kpi-delta ${better ? 'good' : 'warn'}">${delta > 0 ? '+' : ''}${delta.toFixed(2)} ${m.unit}</span></td>`;
        tbody.appendChild(tr);
      });
    }

    // ==========================================================================\n    // CHART.JS INITIALIZATION''')

# Two-policy analytics; remove the nonexistent forecast-aware policy.
replace_block('    function initSurgeComparisonChart() {', '    function initWaitByPhaseChart() {', '''    function initSurgeComparisonChart() {
      const ctx = document.getElementById('chartSurgeComparison');
      if (!ctx) return;
      const res = getReplayData()?.results || {};
      const b = res.baseline?.surge_wait?.mean ?? 0;
      const q = res.qlearning?.surge_wait?.mean ?? 0;
      chartSurgeComparison = new Chart(ctx, {
        type: 'bar', data: { labels: ['Nearest Baseline', 'Tabular Q-Learning'], datasets: [{
          label: 'Surge Wait Time (steps)', data: [b, q], backgroundColor: ['#64748b', '#3b82f6'], borderRadius: 6
        }] }, options: { responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, title: { display: true, text: 'Wait Time (steps)' } } } }
      });
    }

    function initWaitByPhaseChart() {'''.replace('\n+','\n'))
replace_block('    function initWaitByPhaseChart() {', '    function initServedDroppedChart() {', '''    function initWaitByPhaseChart() {
      const ctx = document.getElementById('chartWaitByPhase');
      if (!ctx) return;
      const res = getReplayData()?.results || {};
      const b = res.baseline || {}, q = res.qlearning || {};
      chartWaitByPhase = new Chart(ctx, { type: 'bar',
        data: { labels: ['Overall Wait', 'Surge Phase Wait', 'Normal Phase Wait'], datasets: [
          { label: 'Nearest Baseline', data: [b.avg_wait?.mean ?? 0, b.surge_wait?.mean ?? 0, b.normal_wait?.mean ?? 0], backgroundColor: '#64748b' },
          { label: 'Tabular Q-Learning', data: [q.avg_wait?.mean ?? 0, q.surge_wait?.mean ?? 0, q.normal_wait?.mean ?? 0], backgroundColor: '#3b82f6' }
        ] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } },
          scales: { y: { beginAtZero: true, title: { display: true, text: 'Wait Time (steps)' } } } }
      });
    }

    function initServedDroppedChart() {'''.replace('\n+','\n'))
replace_block('    function initServedDroppedChart() {', '    function initTrainingCurves() {', '''    function initServedDroppedChart() {
      const ctx = document.getElementById('chartServedDropped');
      if (!ctx) return;
      const res = getReplayData()?.results || {};
      const b = res.baseline || {}, q = res.qlearning || {};
      chartServedDropped = new Chart(ctx, { type: 'bar',
        data: { labels: ['Served (Dropoffs)', 'Dropped (Canceled)'], datasets: [
          { label: 'Nearest Baseline', data: [b.served?.mean ?? 0, b.dropped?.mean ?? 0], backgroundColor: '#64748b' },
          { label: 'Tabular Q-Learning', data: [q.served?.mean ?? 0, q.dropped?.mean ?? 0], backgroundColor: '#3b82f6' }
        ] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } },
          scales: { y: { beginAtZero: true, title: { display: true, text: 'Passengers' } } }
        }
      });
    }

    function initTrainingCurves() {'''.replace('\n+','\n'))

# Make the Q-table inspector's index formula match env.encode_state().
replace_block('    function applyPreset(vq, ph, hq, bo, bh) {', '    function updatePolicyInspector() {', '''    function applyPreset(vq, ph, hq, b0, b1, b2, b3) {
      document.getElementById('selVehQuad').value = vq;
      document.getElementById('selPhase').value = ph;
      document.getElementById('selHotspotQuad').value = hq;
      ['selBucket0','selBucket1','selBucket2','selBucket3'].forEach((id, i) => {
        document.getElementById(id).value = [b0,b1,b2,b3][i];
      });
      updatePolicyInspector();
    }

    function calculateStateIndex(vq, ph, hq, buckets) {
      const bucketIndex = buckets[0] * 27 + buckets[1] * 9 + buckets[2] * 3 + buckets[3];
      return (((ph * 4 + hq) * 4 + vq) * 81 + bucketIndex);
    }

    function updatePolicyInspector() {'''.replace('\n+','\n'))

replace_once('''      const vq = parseInt(document.getElementById('selVehQuad').value, 10);
      const ph = parseInt(document.getElementById('selPhase').value, 10);
      const hq = parseInt(document.getElementById('selHotspotQuad').value, 10);
      const bo = parseInt(document.getElementById('selOwnBucket').value, 10);
      const bh = parseInt(document.getElementById('selHotspotBucket').value, 10);

      const stateIndex = calculateStateIndex(vq, ph, hq, bo, bh);'''.replace('\n+','\n'), '''      const vq = parseInt(document.getElementById('selVehQuad').value, 10);
      const ph = parseInt(document.getElementById('selPhase').value, 10);
      const hq = parseInt(document.getElementById('selHotspotQuad').value, 10);
      const buckets = ['selBucket0','selBucket1','selBucket2','selBucket3'].map(id => parseInt(document.getElementById(id).value, 10));

      const stateIndex = calculateStateIndex(vq, ph, hq, buckets);'''.replace('\n+','\n'))
replace_once('const hasRequests = (bo > 0 || bh > 0);', 'const hasRequests = buckets.some(bucket => bucket > 0);')

# Replace two old queue controls with four per-quadrant buckets.
replace_once('<label class="form-label">4. Own Quadrant Queue Bucket</label>', '<label class="form-label">4. Q0 Pending-Request Bucket</label>')
replace_once('id="selOwnBucket"', 'id="selBucket0"')
replace_once('<label class="form-label">5. Hotspot Quadrant Queue Bucket</label>', '<label class="form-label">5. Q1 Pending-Request Bucket</label>')
replace_once('id="selHotspotBucket"', 'id="selBucket1"')
replace_once('''            <!-- State Index Calculation Box -->''', '''            <div class="form-group">
              <label class="form-label">6. Q2 Pending-Request Bucket</label>
              <select class="inspector-select" id="selBucket2" onchange="updatePolicyInspector()">
                <option value="0" selected>0 pending</option><option value="1">1–2 pending</option><option value="2">3+ pending</option>
              </select>
            </div>
            <div class="form-group">
              <label class="form-label">7. Q3 Pending-Request Bucket</label>
              <select class="inspector-select" id="selBucket3" onchange="updatePolicyInspector()">
                <option value="0" selected>0 pending</option><option value="1">1–2 pending</option><option value="2">3+ pending</option>
              </select>
            </div>

            <!-- State Index Calculation Box -->'''.replace('\n+','\n'))
replace_once('Formula: (((vq&middot;4 + ph)&middot;4 + hq)&middot;3 + bo)&middot;3 + bh', 'Formula: (((phase&middot;4 + hotspot)&middot;4 + vehicle quadrant)&middot;81) + B')
replace_once('onclick="applyPreset(1, 1, 2, 0, 0)"', 'onclick="applyPreset(1, 1, 2, 0, 0, 0, 0)"')
replace_once('onclick="applyPreset(2, 2, 2, 2, 2)"', 'onclick="applyPreset(2, 2, 2, 0, 0, 2, 2)"')
replace_once('onclick="applyPreset(0, 0, 1, 1, 0)"', 'onclick="applyPreset(0, 0, 1, 1, 0, 0, 0)"')
replace_once('Matrix Size: <strong>576 States &times; 6 Actions</strong>', 'Matrix Size: <strong>5,184 States &times; 6 Actions</strong>')
replace_once('actual 576 &times; 6 Q-table matrix trained on 10,000 episodes', 'actual 5,184 &times; 6 Q-table matrix trained on 10,000 episodes')
replace_once('If both own-quadrant and hotspot-quadrant request buckets are zero, dispatch actions are masked out, forcing the vehicle to choose between repositioning to an anticipated high-demand zone or remaining in standby.', 'When all four quadrant buckets are empty, dispatch actions are masked out and the vehicle may reposition to a different quadrant center.')

# The experiment log belongs to fleet-rl and is replaced with the root project notes.
replace_block('      <section class="view-section" id="sec-experiments">', '    </div>\n  </main>', '''      <section class="view-section" id="sec-experiments">
        <div class="card">
          <div class="card-header"><div><span class="card-title">Root Project Run Artifacts</span>
            <span class="card-desc">Generated by run_all.py from the 10×10, four-vehicle simulator.</span></div></div>
          <div style="padding:16px;color:var(--text-secondary);line-height:1.7">
            <p><strong>Training:</strong> 10,000 tabular Q-learning episodes.</p>
            <p><strong>Evaluation:</strong> 50 paired seeded episodes against the nearest-vehicle baseline.</p>
            <p><strong>Files:</strong> q_table.npy, training_history.json, simulation_results.json, replay_data.js, and figures/.</p>
            <p>Use the benchmark table for the measured comparison. The learned policy is not guaranteed to outperform the baseline.</p>
          </div>
        </div>
      </section>

    </div>
  </main>
'''.replace('\n+','\n'))

# Align report and policy labels, remove V1/V2 framing, and correct demand values.
s=s.replace('V2.4','ROOT')
s=s.replace('Three-Policy Evaluation Benchmark', 'Two-Policy Evaluation Benchmark')
s=s.replace('<th>Forecast-Aware Rule</th>\n', '')
s=s.replace('Forecast-Aware Rule', 'Tabular Q-Learning')
s=s.replace('V2 Environment (16&times;16 Grid, 8 Autonomous Vehicles)', 'Root Environment (10&times;10 Grid, 4 Autonomous Vehicles)')
s=s.replace('16&times;16', '10&times;10').replace('16x16','10x10').replace('8 autonomous vehicles','4 autonomous vehicles').replace('All 8 autonomous vehicles','All 4 autonomous vehicles')
s=s.replace('Normal Poisson demand arrivals (p=0.25)', 'Normal demand arrivals (p=0.20)')
s=s.replace('p=0.55 (2.2x increase)', 'p=0.60 (3x increase)').replace('p=0.55 (2.2× increase)', 'p=0.60 (3× increase)')
s=s.replace('data: [0.25, 0.25, 0.55, 0.25]', 'data: [0.20, 0.20, 0.60, 0.20]')
# Root dashboard has no second environment or forecast-rule evaluation.
s=s.replace("const res = currentEnv === 'v2' ? (rd.results || {}) : (rd.results_v1 || {});", "const res = rd.results || {};")
# Replace only the table header's forecast policy with the RL policy columns.
s=s.replace('<span class="card-title">Two-Policy Evaluation Benchmark (50 Held-out Test Episodes)</span>', '<span class="card-title">Two-Policy Benchmark (50 Paired Seeded Episodes)</span>')
s=s.replace('<th>Nearest-Vehicle Baseline</th>\n                  <th>Tabular Q-Learning</th>\n                  <th>Tabular Q-Learning (RL)</th>', '<th>Nearest-Vehicle Baseline</th>\n                  <th>Tabular Q-Learning (RL)</th>')
# Remove stale environment selector behavior; the current UI now shows one root environment.
s=s.replace("          <div class=\"card-title\">Three-Policy Evaluation Benchmark (50 Held-out Test Episodes)</div>", "          <div class=\"card-title\">Two-Policy Benchmark</div>")

p.write_text(s, encoding='utf-8')
