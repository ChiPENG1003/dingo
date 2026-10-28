#!/bin/bash
#SBATCH --job-name=diag10026
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --partition=private-dpnc-gpu,shared-gpu
#SBATCH --constraint=COMPUTE_TYPE_AMPERE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
set -eo pipefail
cd /srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference
module load GCCcore/12.3.0 Python/3.11.3
source "$HOME/venvs/dingo_env/bin/activate"
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python -u - <<'PY'
import inference as inf
import numpy as np, torch, traceback, copy
main, init=inf._load_pair('../training/main_train_dir_ET_5M_complete/best_model.pt','../training/init_train_dir_ET/model_latest.pt')
gen=inf.build_injection_generator(main)
for seed in (10025,10026):
 sampler=inf.build_sampler(main,init,num_iterations=30)
 theta=inf._draw_reproducible_truth(gen,seed)
 print('TRUTH',seed,theta,flush=True)
 context,pure=inf._build_injection(gen,theta)
 for det in pure:
  h=pure[det]; asd=context['asds'][det]; start=gen.data_domain.min_idx
  print('SIGNAL',seed,det,'finite',np.isfinite(h).all(),'max',np.max(np.abs(h)),'snr',np.sqrt(np.sum(np.abs(h[start:]/asd[start:])**2)/gen.data_domain.noise_std**2),flush=True)
 for attempt in range(3 if seed==10026 else 1):
  if attempt: torch.manual_seed(seed+attempt*1000000); torch.cuda.manual_seed_all(seed+attempt*1000000)
  sampler.context=copy.deepcopy(context)
  try:
   sampler.run_sampler(num_samples=500,batch_size=500)
   print('SAMPLING_OK',seed,attempt,flush=True)
  except Exception as e:
   traceback.print_exc()
   tb=e.__traceback__
   while tb:
    if 'rational_quadratic' in tb.tb_frame.f_code.co_filename:
     for key in ('discriminant','inputs','a','b','c','unnormalized_widths','unnormalized_heights','unnormalized_derivatives'):
      x=tb.tb_frame.f_locals.get(key)
      if torch.is_tensor(x):
       v=x.detach(); finite=v[torch.isfinite(v)]
       print('SPLINE',seed,attempt,key,'shape',tuple(v.shape),'nonfinite',int((~torch.isfinite(v)).sum()),'negative',int((v<0).sum()),'minmax',(float(finite.min()),float(finite.max())) if finite.numel() else None,flush=True)
    tb=tb.tb_next
   print('SAMPLING_FAILED',seed,attempt,type(e).__name__,flush=True)
PY
