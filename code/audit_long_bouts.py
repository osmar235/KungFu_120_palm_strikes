"""Long-bout event review (manuscript v6). Applies the frozen visual decisions in event_decisions.csv
(record, decision, sample0 at 1000 Hz, original_event starting at 1, reason) to the automated ipsilateral
event list of each 120 strike series, checks that every decision coincides with a discrepancy between the
automated detector and the template search (template_check.py), and recomputes the early-to-late bout
statistics for the original list, the reviewed list, the first 60 events and the first 56 events.
Usage: python audit_long_bouts.py ../data ../results/audit_long_bouts"""
from pathlib import Path
from io import BytesIO
import sys, zipfile, hashlib
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pipeline_core as core
from template_check import detect

HERE=Path(__file__).parent
MET=('rms','mdf_hz','onset_ms')
def save(df,path):df.to_csv(path,index=False)
def wp(a,b):
    if len(a)<2:return np.nan
    if np.all(np.array(a)==np.array(b)):return 1.
    return float(wilcoxon(a,b).pvalue)
def measures(y,E,ev,pid,goal):
    out=[]
    for i,k in enumerate(ev):
        on=core.onsets(E,k)
        for j,m in enumerate(core.MUS):
            if pid=='P5' and m=='DP':continue
            seg=y[k-250:k+150,j]
            out.append((pid,goal,m,i+1,k,i/(len(ev)-1),np.sqrt(np.mean(seg**2)),core.mdf(seg),*on[j]))
    return pd.DataFrame(out,columns=['pid','cond','muscle','event','sample0','frac','rms','mdf_hz','onset_ms','peak_ms'])

def summaries(frames,out):
    summaries=[];individual=[];contrasts=[]
    for tag,d in frames.items():
        el=d[(d.frac<=.2)|(d.frac>=.8)].copy()
        el['phase']=np.where(el.frac<=.2,'early','late')
        tab=el.groupby(['pid','cond','muscle','phase'])[list(MET)].median().unstack()
        for v in MET:
            sub=tab[v].copy()
            sub['delta']=sub.late-sub.early if v=='onset_ms' else 100*(sub.late/sub.early-1)
            sub=sub.reset_index()
            individual.append(sub.assign(method=tag,metric=v))
            for (goal,mus),g in sub.groupby(['cond','muscle']):
                summaries.append(dict(method=tag,cond=goal,muscle=mus,metric=v,n=len(g),early=g.early.median(),late=g.late.median(),change=g.delta.median(),n_up=(g.delta>0).sum(),n_down=(g.delta<0).sum(),p=wp(g.late,g.early)))
            for mus,g in sub.groupby('muscle'):
                p=g.pivot(index='pid',columns='cond',values='delta').reindex(columns=['impact','no_impact']).dropna()
                contrasts.append(dict(method=tag,muscle=mus,metric=v,n=len(p),shield=p.impact.median(),air=p.no_impact.median(),paired=(p.impact-p.no_impact).median(),p=wp(p.impact,p.no_impact)))
    save(pd.DataFrame(summaries),out/'within.csv')
    save(pd.concat(individual,ignore_index=True),out/'individual.csv')
    save(pd.DataFrame(contrasts),out/'contrasts.csv')

def secondary(d,ret,out):
    d=d.copy();d['phase']=np.where(d.frac<=.2,'early',np.where(d.frac>=.8,'late','mid'))
    e=d[d.phase!='mid']
    w=e.pivot(index=['pid','cond','phase','event'],columns='muscle',values='onset_ms')
    p=e.pivot(index=['pid','cond','phase','event'],columns='muscle',values='peak_ms')
    stats=[]
    for k,s in [('DA-TB',w.DA-w.TB),('BR-TB',w.BR-w.TB),('BR-TB no edge',(w.BR-w.TB).where((w.BR!=-400)&(w.TB!=-400))),('DP-TB',w.DP-w.TB),('TB peak',p.TB)]:
        for goal in ['impact','no_impact']:
            v=s.xs(goal,level='cond').groupby(['pid','phase']).median().unstack().dropna()
            stats.append((k,goal,len(v),v.early.median(),v.late.median(),(v.late-v.early).median(),wp(v.late,v.early)))
    lag=(p.TB-p.DA).xs('early',level='phase')
    for k,f in [('SD','std'),('MAD',lambda x:1.4826*np.median(abs(x-np.median(x)))),('IQR',lambda x:x.quantile(.75)-x.quantile(.25))]:
        v=lag.groupby(['pid','cond']).agg(f).unstack()
        stats.append((k,'shield vs air',len(v),v.impact.median(),v.no_impact.median(),int((v.impact<v.no_impact).sum()),wp(v.impact,v.no_impact)))
    v=pd.DataFrame(ret).groupby(['pid','cond'])[['middle','strike']].median()
    w=(v.middle/v.strike).unstack()
    stats.append(('DP retraction','shield vs air',len(w),w.impact.median(),w.no_impact.median(),(w.impact-w.no_impact).median(),wp(w.impact,w.no_impact)))
    save(pd.DataFrame(stats,columns=['metric','comparison','n','early_or_shield','late_or_air','change_or_number_lower','p']),out/'secondary.csv')
def figure(d,out):
    fig,axes=plt.subplots(2,2,figsize=(10,6.6),sharex=True,sharey='row',layout='constrained')
    for col,cond in enumerate(['impact','no_impact']):
        for row,metric in enumerate(['rms','mdf_hz']):
            ax=axes[row,col]
            for m,label,color,mark in zip(['TB','BR','DA','DP'],['Triceps brachii','Brachioradialis','Anterior deltoid','Posterior deltoid'],['#ad452b','#167653','#23628d','#99711e'],['o','^','s','D']):
                g=d[(d.cond==cond)&(d.muscle==m)].copy()
                g['bin']=np.minimum((g.frac*10).astype(int),9)
                b=g.groupby(['pid','bin'])[metric].median().unstack()
                b=100*b.div(g[g.frac<=.2].groupby('pid')[metric].median(),axis=0)
                x=np.arange(5,100,10)
                ax.plot(x,b.median(),label=label,color=color,lw=2 if m=='TB' else 1.25,marker=mark,ms=4)
                if m=='TB':ax.fill_between(x,b.quantile(.25),b.quantile(.75),color=color,alpha=.13,lw=0)
            ax.axhline(100,ls=':',color='0.45',lw=.7)
            if not col:ax.set_ylabel(('RMS amplitude' if not row else 'Median frequency')+'\n(% of early-bout median)')
            if not row:ax.set_title(['Shield, maximal force','Air, maximal speed'][col])
            else:ax.set_xlabel('Progress through the recorded bout (%)')
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.savefig(out/'fig_bout_amplitude_and_median_frequency.png',dpi=220);plt.close(fig)

def main(archive,out):
    out.mkdir(exist_ok=True,parents=True)
    dec=pd.read_csv(HERE/'event_decisions.csv')
    counts=[];events=[];olds=[];ret=[];selected=[]
    import glob
    files=sorted(glob.glob(str(archive)+'/P*/*_120.csv.gz'))
    assert len(files)==18
    if True:
        for f in files:
            pid=Path(f).parent.name;cond=Path(f).name.replace('_120.csv.gz','');name=f"{pid}_{cond}"
            x=pd.read_csv(f).iloc[:,1:5].to_numpy()
            y=core.bandpass(x);E=core.envelope(y);old,z=core.strike_events(E,120);old=np.array(old)
            alt,*_=detect(z,old);grp=dec[dec.record==name]
            remove=set(grp.loc[grp.decision=='exclude','sample0']);add=set(grp.loc[grp.decision=='insert','sample0'])
            assert remove<=set(old), (name,'remove')
            assert not (add&set(old))
            assert set(alt[np.min(abs(alt[:,None]-old),axis=1)>220])==add, (name,'add')
            assert set(old[np.min(abs(old[:,None]-alt),axis=1)>220])==remove|set(grp.loc[grp.decision=='retain','sample0']), (name,'disc')
            ev=np.array(sorted((set(old)-remove)|add));counts.append(dict(pid=pid,cond=cond,original=len(old),reviewed=len(ev),excluded=len(remove),inserted=len(add),median_interval_ms=np.median(np.diff(ev)),first_s=ev[0]/1000,last_s=ev[-1]/1000))
            selected.extend(dict(pid=pid,cond=cond,sample0=k,original=k in old,reviewed=k in ev) for k in sorted(set(ev)|set(old)))
            if pid!='P5':
                for i,j in zip(ev[:-1],ev[1:]):
                    ret.append(dict(pid=pid,cond=cond,middle=z[i+int(.2*(j-i)):i+int(.8*(j-i)),3].max(),strike=z[i-100:i+150,3].max()))
            events.append(measures(y,E,ev,pid,cond));olds.append(measures(y,E,old,pid,cond))
    counts=pd.DataFrame(counts);save(counts,out/'counts.csv');save(pd.DataFrame(selected),out/'timestamps.csv')
    d=pd.concat(events,ignore_index=True);old=pd.concat(olds,ignore_index=True)
    frames={'original':old,'reviewed_full':d}
    for tag,b in [('original',old),('reviewed',d)]:
        cap=b[b.event<=60].copy()
        cap['frac']=(cap.event-1)/(cap.groupby(['pid','cond']).event.transform('max')-1)
        frames[tag+'_cap60']=cap
        complete=b.groupby(['pid','cond']).event.max().unstack().min(axis=1)>=60
        frames[tag+'_cap60_complete_pairs']=cap[cap.pid.isin(complete[complete].index)]
    # Common prefix: largest identical number available for all 18 reviewed recordings.
    common=int(counts.reviewed.min())
    equal=d[d.event<=common].copy()
    equal['frac']=(equal.event-1)/(common-1)
    frames[f'reviewed_equal_{common}']=equal
    summaries(frames,out);secondary(d,ret,out);figure(d,out)
    d.to_csv(out/'reviewed_event_metrics.csv.gz',index=False)
if __name__=='__main__':
    main(Path(sys.argv[1]) if len(sys.argv)>1 else HERE.parent/'data', Path(sys.argv[2]) if len(sys.argv)>2 else HERE.parent/'results'/'audit_long_bouts')
