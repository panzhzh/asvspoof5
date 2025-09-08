import os
import numpy as np

from .calculate_modules import *


def calculate_minDCF_EER_CLLR(cm_scores_file,
                               output_file,
                               printout=True):
    Pspoof = 0.05
    dcf_cost_model = {
        'Pspoof': Pspoof,
        'Cmiss': 1,
        'Cfa': 10,
    }

    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    cm_keys = cm_data[:, 3]
    cm_scores = cm_data[:, 2].astype(np.float64)

    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    eer_cm, frr, far, thresholds = compute_eer(bona_cm, spoof_cm)
    cllr_cm = calculate_CLLR(bona_cm, spoof_cm)
    minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])
    actDCF_cm, _ = compute_actDCF(bona_cm, spoof_cm, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tminDCF\t\t= {:8.6f} (min DCF)\n'.format(minDCF_cm))
            f_res.write('\tactDCF\t\t= {:8.6f} (actual DCF)\n'.format(actDCF_cm))
            f_res.write('\tEER\t\t= {:8.6f} % (EER)\n'.format(eer_cm * 100))
            f_res.write('\tCLLR\t\t= {:8.6f} bits (CLLR)\n'.format(cllr_cm))
        os.system(f"cat {output_file}")

    return minDCF_cm, eer_cm, cllr_cm, actDCF_cm


def calculate_aDCF_tdcf_tEER(cm_scores_file,
                              asv_scores_file,
                              output_file,
                              printout=True):
    from a_dcf import a_dcf
    adcf = a_dcf.calculate_a_dcf(asv_scores_file)['min_a_dcf']

    Pspoof = 0.05
    tdcf_cost_model = {
        'Pspoof': Pspoof,
        'Ptar': (1 - Pspoof) * 0.99,
        'Pnon': (1 - Pspoof) * 0.01,
        'Cmiss': 1,
        'Cfa': 10,
        'Cmiss_asv': 1,
        'Cfa_asv': 10,
        'Cmiss_cm': 1,
        'Cfa_cm': 10,
    }

    asv_data = np.genfromtxt(asv_scores_file, dtype=str)
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float64)

    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    cm_keys = cm_data[:, 3]
    cm_scores = cm_data[:, 2].astype(np.float64)

    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    X_tar = np.array([asv_scores[asv_keys == 'target'], cm_scores[asv_keys == 'target']], dtype=object)
    X_non = np.array([asv_scores[asv_keys == 'nontarget'], cm_scores[asv_keys == 'nontarget']], dtype=object)
    X_spf = np.array([asv_scores[asv_keys == 'spoof'], cm_scores[asv_keys == 'spoof']], dtype=object)

    Pfa_non_ASV = 0.01881016557566423
    Pmiss_ASV = 0.01880141010575793
    Pfa_spf_ASV = 0.4607082907604729
    _, _, _, tau_ASV = compute_Pmiss_Pfa_Pspoof_curves(X_tar[0], X_non[0], X_spf[0])

    Pmiss_CM, Pfa_CM, tau_CM = compute_det_curve(np.concatenate([X_tar[1], X_non[1]]), X_spf[1])

    eer_asv, _, _, asv_threshold = compute_eer(tar_asv, non_asv)
    eer_cm, frr, far, thresholds = compute_eer(bona_cm, spoof_cm)
    minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, tdcf_cost_model['Cmiss'], tdcf_cost_model['Cfa'])

    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv] = obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold
    )

    tDCF_curve, _ = compute_tDCF(
        bona_cm,
        spoof_cm,
        Pfa_asv,
        Pmiss_asv,
        Pmiss_spoof_asv,
        tdcf_cost_model,
        print_cost=False,
    )

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    teer = compute_teer(Pmiss_CM, Pfa_CM, tau_CM, Pmiss_asv, Pfa_non_ASV, Pfa_spf_ASV, tau_ASV)

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nSASV RESULT\n')
            f_res.write('\tEER\t\t= {:8.9f} % (Equal error rate for countermeasure)\n'.format(teer))
            f_res.write('\nTANDEM\n')
            f_res.write('\tmin-tDCF\t\t= {:8.9f}\n'.format(min_tDCF))
        os.system(f"cat {output_file}")

    return adcf, min_tDCF, teer
