import csv, json, sys, statistics as st, re
sys.argv=['x']
exec(open('calib2.py').read())
D='/home/claude/deliv/matlab/data/'
# 1 reliability by range
with open(D+'reliability_by_range.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['cue','range_lo_m','range_hi_m','reliability'])
    for k in R.W:
        for b in range(4):
            v=REL[(k,b)]; w.writerow([k,BINS[b],min(BINS[b+1],6.0),'' if v is None else f'{v:.3f}'])
# 2 held-out results
res=json.load(open('calib2.json'))
names={'strict':'Strict ReID (original FollowMe)','fixed':'Fixed-weight fusion','gated':'Gated, a-priori reliabilities','gated_cal':'Gated, calibrated','cascade':'Deployed cascade'}
with open(D+'heldout_results.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['rule','parameter','owner_accept_select','owner_accept_test','wrong_person_test','volunteer_accept_test','volunteer_accepted_frames'])
    for k,v in res.items(): w.writerow([names[k],v[0] if v[0] is not None else '',f'{v[1]:.4f}',f'{v[2]:.4f}',f'{v[3]:.4f}',f'{v[4]:.4f}',v[5]])
# 3 acceptance by range on test
bins=[(0,1.5),(1.5,2.5),(2.5,3.5),(3.5,99)]
with open(D+'acceptance_by_range.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['range_lo_m','range_hi_m','frames']+list(names.keys()))
    for lo,hi in bins:
        sel=[c for c in TEST_OWN if lo <= (F(c[oidx(c)]['range_m']) or -1) < hi]
        w.writerow([lo,min(hi,6),len(sel)]+[f"{sum(pick(c,k,res[k][0])==oidx(c) for c in sel)/max(1,len(sel)):.3f}" for k in names])
# 4 timings per frame (IDENTIFYING)
with open(D+'timing_per_frame.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['t_mono','t_segment','t_reid','t_identify','t_localize','t_gesture','t_total'])
    for r in rows:
        if r['mode']=='IDENTIFYING' and r['cand'] in ('0','-1'): w.writerow([r['t_mono'],r['t_segment'],r['t_reid'],r['t_identify'],r['t_localize'],r['t_gesture'],r['t_total']])
# 5 tegrastats
with open(D+'tegrastats_15W.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['cpu_mean_pct','gpu_pct','vdd_in_mw','ram_mb','tj_c'])
    for l in open('../followme_logs/tegrastats_15W.log'):
        m=re.search(r'CPU \[([^\]]+)\]',l); g=re.search(r'GR3D_FREQ (\d+)%',l); v=re.search(r'VDD_IN (\d+)mW',l); ram=re.search(r'RAM (\d+)/',l); tj=re.search(r'tj@([\d.]+)C',l)
        if m and g and v: w.writerow([f"{st.mean(int(x.split('%')[0]) for x in m.group(1).split(',') if '%' in x):.1f}",g.group(1),v.group(1),ram.group(1) if ram else '',tj.group(1) if tj else ''])
print(open(D+'heldout_results.csv').read()); print(open(D+'acceptance_by_range.csv').read())
