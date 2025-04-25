import numpy as np
import xarray as xr
import stim
import pymatching
from pymatching import Matching
from utilities.defects_matrix_utils import *


def project_data_meas(data_meas, num_rounds: int, anc_qubits: list):
    '''Project the last data qubit measurement onto stabilizer values.
    
    Input: 
        data_meas: measurement data obtained from stim (# of shots x num of data qubits)
        num_rounds: # of QEC rounds
        anc_qubits: list of names of anc_qubits
    Output:
        syndrome_proj: syndrome projection (xarray of dims # of shots x # of anc_qubits)
    '''

    num_shots = np.shape(data_meas)[0]
    shots     = np.arange(num_shots) 
    
    syndrome_proj = data_meas.data[:,:-1] ^ data_meas.data[:,1:]  #dims: # of shots x # of ancilla
    #the 0th and 1st entries are xored, to give the 1st ancilla measurement equivalent
    #the 1st and 2nd entries are xored, to give the 2nd ancilla measurement equivalent

    syndrome_proj = xr.DataArray(data=syndrome_proj,
                                 dims=["shot","anc_qubit"],
                                 coords = dict(shot=shots,anc_qubit=anc_qubits))

    syndrome_proj["qec_round"]  = num_rounds    

    return syndrome_proj


def get_defects_matrix(distance: int, num_rounds: int, num_shots: int, circuit: stim.Circuit):
    '''Construct the defects matrix from the detection outcomes, for a repetition code
       under Steane style syndrome extraction.

    Input:
        distance:   distance of the code
        num_rounds: # of QEC rounds
        num_shots:  # of shots to sample the circuit
        circuit: the stim circuit
    Output:
        defects_matrix: the detector matrix of dimensions (# of shots x # of QEC rounds +1 x # of detectors per round)

    '''
    
    d             = distance    
    data_qubits   = np.arange(0,d).tolist()
    anc_qubits    = (np.arange(d)+d).tolist()        
    SHOTS         = np.arange(num_shots)
    QEC_ROUNDS    = np.arange(num_rounds)

    sampler   = circuit.compile_sampler()
    samples   = sampler.sample(shots=num_shots)

    NUM_ANC = len(anc_qubits)
    NN      = NUM_ANC*num_rounds

    anc_qubit_samples  = samples[:,:NN]
    data_qubit_samples = samples[:,NN:]
    anc_qubit_samples  = anc_qubit_samples.reshape(num_shots,num_rounds,NUM_ANC) 

    #2 consecutive ancilla measurements create the syndrome
    anc_qubit_samples = anc_qubit_samples[:,:,:-1] ^ anc_qubit_samples[:,:,1:]

    #Now redefine the anc qubits as detectors:
    anc_qubits    = (np.arange(d-1)+d).tolist()      

    anc_meas = xr.DataArray(data = anc_qubit_samples, 
                        dims=["shot","qec_round","anc_qubit"],
                        coords = dict(shot=SHOTS,qec_round=QEC_ROUNDS,anc_qubit=anc_qubits),)


    data_meas = xr.DataArray(data=data_qubit_samples, 
                        dims=["shot","data_qubits"],
                        coords = dict(shot=SHOTS,data_qubits=data_qubits))    

    initial_state = get_initial_state(anc_qubits)

    syndrome_proj                     = project_data_meas(data_meas,num_rounds,anc_qubits)
    syndromes                         = xr.concat([anc_meas,syndrome_proj],"qec_round")


    syndrome_matrix_copy              = syndromes.copy()
    syndrome_matrix_copy.data[:,-1,:] = initial_state
    defects_matrix                    = syndromes ^ syndrome_matrix_copy.roll(qec_round=1)

    return defects_matrix



def construct_dem(pij_bulk: dict, pij_bd: dict, pij_time: dict, p4_cnts: dict):
    '''Contruct the detector error model given the estimated values of error probabilities.
    
    Input:
        pij_bulk: dictionary with keys the detector names of space-bulk errors and values the error probabilities
        pij_bd: dictionary with keys the detector names of boundary errors and values the error probabilities
        pij_time: dictionary with keys the detector names of time errors and values the error probabilities
        p4_cnts: dictionary with keys the detector names of 4-point errors and values the error probabilities

    Output:
        reconstructed_DEM: our estimated detector error model
    '''
    reconstructed_DEM = stim.DetectorErrorModel()

    for key in pij_bulk.keys():
        det_list = []
        for det in key:
            ind = int(det[1:])
            det_list.append(stim.target_relative_detector_id(ind))
        
        det_list.append(stim.target_logical_observable_id(0))

        reconstructed_DEM.append("error",pij_bulk[key],det_list)

    for key in pij_bd.keys():
        
        ind      = int(key[1:])
        det_list = [stim.target_relative_detector_id(ind),stim.target_logical_observable_id(0)]

        reconstructed_DEM.append("error",pij_bd[key],det_list)

    for key in pij_time.keys():
        
        det_list = []
        for det in key:
            ind = int(det[1:])
            det_list.append(stim.target_relative_detector_id(ind))
        
        reconstructed_DEM.append("error",pij_time[key],det_list)
    
    for key in p4_cnts.keys():
        det_list=[]
        for det in key:
            ind=int(det[1:])
            det_list.append(stim.target_relative_detector_id(ind))
        
        reconstructed_DEM.append("error",p4_cnts[key],det_list)

    return reconstructed_DEM

