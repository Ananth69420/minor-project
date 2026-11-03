from pathlib import Path
p=Path('README.md'); s=p.read_text(encoding='utf-8-sig')
s=s.replace('├── run_all.py                # Master execution runner', '├── dashboard_adapter.py      # Converts root outputs into dashboard data\n├── replay_data.js            # Generated payload consumed by index.html\n├── run_all.py                # Master execution runner')
s=s.replace('├── index.html                # Separate fleet-rl dashboard (not refreshed by root runner)', '├── index.html                # Interactive dashboard for the root 10x10 fleet simulator')
s=s.replace('4. Check the console notice about the separate dashboard data source; the root benchmark outputs and figures are refreshed, but the existing dashboard is not.', '4. Adapt benchmark results, replay frames, training history, and the Q-table into `replay_data.js` for the dashboard.')
p.write_text(s,encoding='utf-8')

p=Path('index.html'); s=p.read_text(encoding='utf-8-sig')
s=s.replace('<span class="card-desc">Highlighting the greedy valid policy action argmax Q(s, a)</span>', '<span class="card-desc">Q-values for the selected state; reposition validity depends on the vehicle position.</span>')
s=s.replace('Pre-Surge Anticipation:</strong> Vehicle in Q1, warning for Q2, own queue empty &rarr; Agent repositions to Q2', 'Pre-Surge State:</strong> Vehicle in Q1, hotspot Q2, all queue buckets empty')
s=s.replace('Surge Hotspot Overload:</strong> Vehicle in Q2, surge active, 3+ pending &rarr; Agent serves nearest', 'Surge State:</strong> Vehicle in Q2, active surge, 3+ requests in Q2')
s=s.replace('Normal Operation:</strong> Vehicle in Q0, phase 0, 1-2 pending in own zone &rarr; Agent serves nearest', 'Normal State:</strong> Vehicle in Q0, early phase, 1–2 requests in Q0')
p.write_text(s,encoding='utf-8')

p=Path('run_all.py'); s=p.read_text(encoding='utf-8-sig').replace('import sys\n',''); p.write_text(s,encoding='utf-8')
