import json
import sys
import numpy as np
sys.path.insert(0, 'bilby_baseline')
from dingo.gw.result import Result
from builders import build_ifos, build_waveform_generator
from joint_builders import build_joint_prior, build_joint_likelihood, read_joint_injection_parameters, split_parameters

source = 'examples/gnpe_model/inference/inference_result/posteriors/ET_5M_complete_best140/overlap/seed10051_dprior_shift0.100s_bilby_joint.hdf5'
result_file = 'bilby_baseline/bilby_injections/ET5M_best140_joint_seed10051_shift0p1_nlive250/bilby_joint_result.json'
res = Result(file_name=source)
ifos, duration, fs = build_ifos(res)
wfg = build_waveform_generator(res, duration, fs)
like = build_joint_likelihood(ifos, wfg)
prior = build_joint_prior(res, prior='ET')
truth = read_joint_injection_parameters(source, res.t_ref)
post = json.load(open(result_file))['posterior']['content']
idx = int(np.argmax(post['log_likelihood']))
sample = {key: values[idx] for key, values in post.items() if key in prior}
for label, params in [('truth', truth), ('best posterior', sample)]:
    print('\n', label, flush=True)
    print('logLR', like.log_likelihood_ratio(params), flush=True)
    print('prior prob', prior.prob(params), flush=True)
    for p in ('chirp_mass','mass_ratio','luminosity_distance','geocent_time','ra','dec','chirp_mass_2','mass_ratio_2','luminosity_distance_2','delta_tc_2','ra_2','dec_2'):
        print(p, params.get(p), flush=True)
print('posterior max logL', max(post['log_likelihood']), 'noise', like.noise_log_likelihood(), flush=True)
from scipy.optimize import minimize
first, second = split_parameters(truth)
model1 = wfg.frequency_domain_strain(first)
model2 = wfg.frequency_domain_strain({**second, 'geocent_time': first['geocent_time']})
snr1sq = snr2sq = cross = 0.
for ifo in ifos:
    h1 = ifo.get_detector_response(model1, first)
    h2 = like._shift_second_earlier(ifo.get_detector_response(model2, {**second, 'geocent_time': first['geocent_time']}), truth['delta_tc_2'])
    a = np.real(ifo.optimal_snr_squared(h1))
    b = np.real(ifo.optimal_snr_squared(h2))
    c = 4 / ifo.duration * np.real(np.sum((np.conj(h1) * h2 / ifo.power_spectral_density_array)[ifo.frequency_mask]))
    print('IFO SNR', ifo.name, np.sqrt(a), np.sqrt(b), 'cross', c, flush=True)
    snr1sq += a; snr2sq += b; cross += c
print('network SNR1', np.sqrt(snr1sq), 'SNR2', np.sqrt(snr2sq), 'cross', cross, flush=True)
for label, params in [('truth',truth),('best posterior',sample)]:
    a,b = split_parameters(params)
    pol1 = wfg.frequency_domain_strain(a)
    h1 = {ifo.name:ifo.get_detector_response(pol1,a) for ifo in ifos}
    one = sum(np.real(ifo.inner_product(h1[ifo.name]))-0.5*np.real(ifo.optimal_snr_squared(h1[ifo.name])) for ifo in ifos)
    print(label,'only-first logLR',one,'increment from second',like.log_likelihood_ratio(params)-one,flush=True)
for key in prior:
    if key in truth and hasattr(prior[key], 'prob'):
        v = prior[key].prob(truth[key])
        if np.any(np.asarray(v) == 0): print('zero prior component', key, truth[key], prior[key], flush=True)
