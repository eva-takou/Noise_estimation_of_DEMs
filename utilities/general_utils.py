import stim
import numpy as np
import xarray as xr
from numba import njit




def stims_DEM_to_dictionary(DEM: stim.DetectorErrorModel):
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
            
            dets = []
            for target in targets:
                if target.is_relative_detector_id():
                    ind = target.val
                    dets.append("D"+str(ind))
                else:
                    dets.append("L0")

            key = tuple(dets)
            error_dict[key]=prob
            
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
    '''
    Get the <vivj> of detection events, across many runs of the circuit.

    Input:
        defect_matrix: an xArray of dims # of shots x # of qec_rounds +1 x # of ancilla qubits.

    Output:
        avg_vivj: <vivj> array of dims # of qec rounds x # of qec rounds  x # of ancilla qubits x # of ancilla qubits. 
    '''
   
    num_shots, num_rounds, num_anc = defect_matrix_data.shape
    
    avg_vivj = np.zeros((num_rounds * num_rounds, num_anc * num_anc))

    for shot in range(num_shots):
        mat = defect_matrix_data[shot]
        for i in range(num_rounds):
            for j in range(num_rounds):
                for k in range(num_anc):
                    for l in range(num_anc):
                        idx1 = i * num_rounds + j
                        idx2 = k * num_anc + l
                        avg_vivj[idx1, idx2] += mat[i, k] * mat[j, l]

    avg_vivj /= num_shots
    return avg_vivj.reshape((num_rounds, num_rounds, num_anc, num_anc))

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



