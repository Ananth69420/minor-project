from pathlib import Path
import re
p=Path('index.html'); s=p.read_text(encoding='utf-8-sig')
repls={
'Benchmark Advantage (V2)':'Measured Benchmark Results',
'Tabular Q-learning pre-positions idle vehicles during the 20-step warning phase into Hotspot Q2, reducing surge passenger wait times from <strong>17.15 steps</strong> (Nearest Baseline) to <strong>16.32 steps</strong> (RL) across 50 held-out test episodes.':'The paired test benchmark compares the learned policy with the nearest-vehicle baseline. See Performance Analytics for the measured results; the learned policy does not currently outperform the baseline.',
'<span class="kpi-value" id="kpiFleetSize">8</span>':'<span class="kpi-value" id="kpiFleetSize">4</span>',
'0 busy / 8 idle':'0 busy / 4 idle',
'Benchmark: 97.96 &plusmn; 6.69':'Benchmark: 85.10 &plusmn; 6.34',
'Benchmark: 3.04 &plusmn; 2.93':'Benchmark: 12.04 &plusmn; 5.89',
'<span class="kpi-value" id="kpiSurgeWait">16.32</span>':'<span class="kpi-value" id="kpiSurgeWait">22.73</span>',
'<span class="kpi-delta good">-0.83 steps</span> vs Baseline (17.15)':'<span class="kpi-delta warn">+2.32 steps</span> vs Baseline (20.41)',
'Autonomous Fleet Live Grid (16 &times; 16)':'Autonomous Fleet Live Grid (10 &times; 10)',
'Parameter-shared across 8 AVs':'Parameter-shared across 4 vehicles',
'<span class="kpi-value">576</span>':'<span class="kpi-value">5,184</span>',
'4 &times; 4 &times; 4 &times; 3 &times; 3 factorization':'4 &times; 4 &times; 4 &times; 81 factorization',
'Seeds 2000 &ndash; 4999 (245s training)':'10,000 seeded training episodes',
'Normal demand arrivals (p=0.25)':'Normal demand arrivals (p=0.20)',
'Spawn probability returns to normal uniform rate (p=0.25)':'Spawn probability returns to normal uniform rate (p=0.20)',
'Demand arrival rate surged to p=0.55 (2.2x increase)':'Demand arrival rate surged to p=0.60 (3x increase)',
'Demand arrival rate surged to p=0.55 (2.2× increase)':'Demand arrival rate surged to p=0.60 (3× increase)',
'16 &times; 16':'10 &times; 10',
'V2 Production':'Root Project',
'V2 Environment':'Root Environment',
'16x16':'10x10',
'8 autonomous vehicles':'4 autonomous vehicles',
'All 8 autonomous vehicles':'All 4 autonomous vehicles',
}
for old,new in repls.items():
    s=s.replace(old,new)
# Drop the copied V1/V2 experiment comparison; it reports a different simulator.
s,n=re.subn(r'        <!-- V1 vs V2 Structural Comparison Card -->.*?(?=      </section>)','',s, count=1, flags=re.S)
if n != 1: raise RuntimeError('legacy V1/V2 comparison card not found')
# Remove stale wording in the overview header and replace historical experiment claims.
s=s.replace('Seed 5000','Seed 12345')
s=s.replace('Seeds 5000 &ndash; 5049', '50 paired episodes; fixed evaluation seed 12345')
s=s.replace('>VERIFIED REPRODUCIBLE<', '>SEEDED EVALUATION<')
s=s.replace('The policy masks service actions when there are no unassigned requests and masks reposition actions that would leave a vehicle at its current center.', 'The policy masks service actions when no unassigned requests exist and masks reposition actions that would leave the vehicle at its current center.')
p.write_text(s,encoding='utf-8')
