import stim
from math import prod
from utilities.general_utils import DEM_to_dictionary
from sims.color_code_bare_ancilla.utilities_for_color_code import *
from utilities.general_utils import bulk_prob_formula

def estimate_all_edges_for_defect_type(sols_for_defect_type, obj, num_rounds: int, vi_mean, vivj_mean, stims_DEM: stim.DetectorErrorModel, defects_type):
    '''Estimate the edges and hyperedges for the X-, or Z-DEM.

    Input:
        sols_for_defect_type: dictionary with solutions containing 3-point events
        obj: color code object
        num_rounds: # of QEC rounds (int)
        vi_mean:  average counts of individual detectors  (dims: (num_rounds+1) x # of detectors per round)
        vivj_mean: average 2-point coincidences of detector pairs  (dims: (num_rounds+1) x (num_rounds+1) x # of detectors per round x # of detectors per round)
        stims_DEM: the X-, or Z-DEM of stim (noise-aware)
        defects_type: "X" or "Z"
    Output:
        pij_bulk: dictionary of bulk probabilities
        pij_time: dictionary of time probabilities
        pij_bd:   dictionary of boundary probabilities
        p3:       dictionary of 3-point probabilities
        '''
    
    Z_dets,X_dets = get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds)

    if defects_type=="X":
        dets = X_dets
    elif defects_type=="Z":
        dets = Z_dets     
    else:
        raise Exception("defects_type can be only X or Z")

    errors_in_DEM = DEM_to_dictionary(stims_DEM)
    
    pij_bulk = {}
    pij_time = {}

    all_keys = list(dets.keys())

    #Get all the 3 pnt events
    p3 = {}
    for key in sols_for_defect_type.keys():
        if key in errors_in_DEM.keys():
            p3[key] = sols_for_defect_type[key][key]

    for m in range(len(all_keys)):
        
        det1     = all_keys[m]
        rd1,anc1 = dets[det1]

        for k in range(m+1,len(all_keys)):

            det2     = all_keys[k]
            rd2,anc2 = dets[det2]

            if tuple([det1,det2]) not in errors_in_DEM.keys() and tuple([det1,det2,"L0"]) not in errors_in_DEM.keys():
                continue

            if rd1!=rd2 and anc1==anc2: #time edges

                v1 = vi_mean[rd1,anc1]
                v2 = vi_mean[rd2,anc2]
                v1v2 = vivj_mean[rd1,rd2,anc1,anc2]

                if bulk_prob_formula(v1,v2,v1v2)<=0:
                    continue

                pij_time[tuple([det1,det2])] = bulk_prob_formula(v1,v2,v1v2)

                for key in p3.keys():

                    if det1 in key and det2 in key:
                        p = pij_time[tuple([det1,det2])]
                        pij_time[tuple([det1,det2])] = (p-p3[key])/(1-2*p3[key])            

            else: #space-bulk or space-time bulk

                v1   = vi_mean[rd1,anc1]
                v2   = vi_mean[rd2,anc2]
                v1v2 = vivj_mean[rd1,rd2,anc1,anc2]

                if bulk_prob_formula(v1,v2,v1v2)<=0:
                    continue
                
                pij_bulk[tuple([det1,det2])] = bulk_prob_formula(v1,v2,v1v2)

                for key in p3.keys():

                    if det1 in key and det2 in key:
                        p                            = pij_bulk[tuple([det1,det2])]
                        pij_bulk[tuple([det1,det2])] = (p-p3[key])/(1-2*p3[key])


    for key in pij_bulk.keys():
        if pij_bulk[key]<0:
            pij_bulk[key]=0

    for key in pij_time.keys():
        if pij_time[key]<0:
            pij_time[key]=0


    pij_bd = {}
    #Finally, get the boundary edges:
    
    for q in range(len(all_keys)):
    
        det1     = all_keys[q]

        if (det1,) not in errors_in_DEM.keys() and (det1,"L0") not in errors_in_DEM.keys():
            continue

        rd1,anc1 = dets[det1]
        v1       = vi_mean[rd1,anc1]

        DENOM  = 1
        DENOM *= prod(1 - 2 * p for k, p in pij_bulk.items() if det1 in k)
        DENOM *= prod(1 - 2 * p for k, p in pij_time.items() if det1 in k)
        DENOM *= prod(1 - 2 * p for k, p in p3.items() if det1 in k)
        
        pij_bd[det1] = 1/2 + (v1-1/2)/DENOM

    for key in pij_bd.keys():
        if pij_bd[key]<=0:
            pij_bd[key]=0

    return pij_bulk,pij_time,pij_bd,p3




