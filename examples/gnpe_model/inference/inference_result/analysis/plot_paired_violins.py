"""Overlaid marginal posteriors for the same paired injections used in pp_plot.py."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from batch_read import load_injection_truth, load_injection_posterior
from plot_posteriors import load_bilby, align_time_convention
from dingo.gw.result import Result
from dingo.gw.derived_parameters import with_derived_parameters

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--outdir',required=True)
    args=ap.parse_args()
    pairs=pd.read_csv(args.manifest).sort_values('seed')
    assert len(pairs)==100 and pairs.seed.nunique()==100
    params=['chirp_mass','mass_ratio','chi_eff','luminosity_distance','a_1','a_2','tilt_1','tilt_2','geocent_time','ra','dec','psi']
    labels={'chirp_mass':'Chirp mass [solar masses]','luminosity_distance':'Luminosity distance [Mpc]','geocent_time':'Geocentric time offset [s]'}
    records=[]; summary=[]
    for row in pairs.itertuples():
        with h5py.File(row.dingo) as h:
            assert int(h['injection_parameters'].attrs['injection_seed'])==row.seed
        truth=with_derived_parameters(load_injection_truth(row.dingo),names=['chi_eff'])
        d=load_injection_posterior(row.dingo)
        b=align_time_convention(load_bilby(row.bilby),Result(file_name=row.dingo).t_ref)
        for method,frame in [('Dingo',d),('Bilby',b)]:
            frame['chi_eff']=with_derived_parameters(frame,names=['chi_eff'])['chi_eff']
            for p in params:
                v=frame[p].to_numpy(dtype=float)
                assert len(v)>1 and np.isfinite(v).all(),(row.seed,method,p)
                q=np.quantile(v,[.05,.5,.95])
                summary.append(dict(seed=row.seed,method=method,parameter=p,truth=truth[p],q05=q[0],median=q[1],q95=q[2],n_samples=len(v)))
        records.append((row.seed,truth,d,b))
        print('Loaded',row.seed,flush=True)
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summary).to_csv(out/'posterior_summary.csv',index=False)
    pairs.to_csv(out/'injection_pairs.csv',index=False)
    handles=[Patch(facecolor='#1b7f79',alpha=.45,label='Dingo'),Patch(facecolor='#d1622b',alpha=.45,label='Bilby'),Line2D([],[],color='black',marker='x',ls='',label='Injected truth')]
    with PdfPages(out/'violin_dingo_bilby_100.pdf') as pdf:
        for group in range(3):
            for start in range(0,100,25):
                chunk=records[start:start+25]; x=np.arange(len(chunk))
                fig,axes=plt.subplots(4,1,figsize=(18,13),sharex=True)
                for ax,p in zip(axes,params[group*4:group*4+4]):
                    for idx,color in [(2,'#1b7f79'),(3,'#d1622b')]:
                        vals=[rec[idx][p].to_numpy() for rec in chunk]
                        parts=ax.violinplot(vals,positions=x,widths=.85,showextrema=False,showmedians=False,points=120,bw_method='scott')
                        for body in parts['bodies']:
                            body.set_facecolor(color);body.set_edgecolor(color);body.set_alpha(.42);body.set_linewidth(.8)
                        qs=np.array([np.quantile(v,[.05,.5,.95]) for v in vals])
                        offset=-.075 if idx==2 else .075
                        ax.vlines(x+offset,qs[:,0],qs[:,2],color=color,lw=1.3)
                        ax.scatter(x+offset,qs[:,1],color=color,s=9,zorder=4)
                    ax.scatter(x,[rec[1][p] for rec in chunk],marker='x',color='black',s=23,linewidths=1,zorder=5)
                    ax.set_ylabel(labels.get(p,p));ax.grid(axis='y',alpha=.2)
                    ax.set_xlim(-.7,len(chunk)-.3)
                axes[-1].set_xticks(x,[str(rec[0]) for rec in chunk],rotation=60,ha='right')
                axes[-1].set_xlabel('Injection seed')
                fig.suptitle(f'Dingo 5M complete best vs Bilby (nlive=250) | seeds {chunk[0][0]}–{chunk[-1][0]}',y=.995,fontsize=15)
                fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.977),ncol=3)
                fig.text(.5,.012,'Overlaid KDEs, equal maximum width per violin; dots: medians; bars: 5–95% quantiles. Seed 10026 excluded.',ha='center',fontsize=10)
                fig.tight_layout(rect=(0,.03,1,.94))
                name=f'violin_group{group+1}_seeds{chunk[0][0]}_{chunk[-1][0]}.png'
                fig.savefig(out/name,dpi=150);pdf.savefig(fig);plt.close(fig)
                print('Saved',name,flush=True)
    (out/'README.txt').write_text('100 paired injections: seeds 10000–10100 except 10026.\n12 PNG pages and one multipage PDF; four parameters and 25 injections per page.\nGreen=Dingo, orange=Bilby, black cross=injected truth. Full posterior samples used, no subsampling.\nViolin KDE: Scott bandwidth, equal maximum width; kernels may extend beyond parameter bounds.\nDots=median, vertical bars=5th–95th quantiles (slightly offset for visibility).\nTime is relative to t_ref; chi_eff derived from joint spins and mass ratio.\nAngles retain original coordinate ordering; boundary wrapping can split angular distributions.\n')
if __name__=='__main__':main()
