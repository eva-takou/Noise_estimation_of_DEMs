import stim
import numpy as np
import xarray as xr
from numba import njit




def DEM_to_dictionary(DEM: stim.DetectorErrorModel):
    '''
    Convert all the error instructions of stim's DEM
    into a dictionary.
    
    Input: 
        DEM: Detector error model obtained by stim
    Output:
        error_dict: Dictionary with keys the detectors & logical observables
                    and values the error probabilities.

    '''

    error_dict ={}
    for instruction in DEM:

        if instruction.type=="error":

            targets = instruction.targets_copy()
            prob    = instruction.args_copy()[0]

            dets = [f"D{t.val}" if t.is_relative_detector_id() else "L0" for t in targets]
            error_dict[tuple(dets)]=prob
            
    return error_dict

def DEM_to_dictionary_drop_logicals(DEM: stim.DetectorErrorModel):
    '''
    Convert all the error instructions of stim's DEM
    into a dictionary, but drop the logicals from the name of the dictionary.
    
    Input: 
        DEM: Detector error model obtained by stim
    Output:
        error_dict: Dictionary with keys the detectors & logical observables
                    and values the error probabilities.

    '''

    error_dict ={}
    for instruction in DEM:

        if instruction.type=="error":

            targets = instruction.targets_copy()
            prob    = instruction.args_copy()[0]

            dets = [f"D{t.val}" for t in targets if t.is_relative_detector_id()]
            error_dict[tuple(dets)]=prob
            
    return error_dict


def avg_vi(defect_matrix: xr.DataArray):
    '''
    Get the <vi> of detection events, across many runs of the circuit.

    Input:
        defect_matrix: an xArray of dims # of shots x # of qec_rounds +1 x # of ancilla qubits.

    Output:
        vi_mean: averages of single-clicks for any detector in defects matrix [xArray of dims # of qec rounds x # of ancilla qubits].

    '''
    num_shots = np.size(defect_matrix.data,axis=0)
    vi_mean   = np.sum(defect_matrix.data,axis=0)/num_shots
    
    return vi_mean


@njit
def avg_vivj(defect_matrix_data):
    num_shots, num_rounds, num_anc = defect_matrix_data.shape
    result = np.zeros((num_rounds, num_rounds, num_anc, num_anc))

    for shot in range(num_shots):
        mat = defect_matrix_data[shot]
        for i in range(num_rounds):
            row_i = mat[i]
            for j in range(num_rounds):
                row_j = mat[j]
                for k in range(num_anc):
                    for l in range(num_anc):
                        result[i, j, k, l] += row_i[k] * row_j[l]

    result /= num_shots
    return result


def bulk_prob_formula(v1: float,v2: float, v1v2:float):
    '''
    Estimate the bulk edge probability based on the formula:

    1/2 - sqrt{  1/4 - numer/denom}
    
    numer =  <vi vj> - <vi><vj>
    denom =  1-2<vi \oplus vj> = 1 - 2<vi> - 2<vj> + 4<vi vj>

    where <vi> (vj) is the total # of clicks we get at a detector 'vi' divided by # of shots.
    The <vi vj> average is the # of coincident clicks we observe, divided by the # of shots.

    Input:
        v1 (v2): the mean # of times detector 1 (2) fires  [# of detection events/num_shots].
        v1v2: the mean # of times detector 1 and detector 2 fire [# of coincidences/num_shots].
    
    Output:
        p: The probability of error for edge described by the input detection events.

    '''
    numer  = v1v2 - v1*v2
    denom  = 1    - 2*(v1+v2) + 4*v1v2
    p      = 1/2  - np.sqrt(1/4- numer/denom)

    if p<0:
        p=0

    return p



