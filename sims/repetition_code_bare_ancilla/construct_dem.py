import numpy as np
import xarray as xr
import stim
from utilities.general_utils import *
from estimation_funcs_rep_code import *



def get_measurement_data(samples,DATA_QUBITS: list,ANC_QUBITS: list,NUM_ROUNDS: int,NUM_SHOTS: int):
    '''
    Extract the measurement outcomes as ancilla and data qubit outcomes, given the samples (from sampler) produced by stim.

    Input
    ------
    samples: an array of the total samples collected from sampler.samples of stim (corresponding to data and ancilla qubits)
    DATA_QUBITS: a list of str containing the data qubit names
    ANC_QUBITS: a list of str containing the ancilla qubit names
    NUM_ROUNDS: the total # of QEC rounds
    NUM_SHOTS: the total # of shots


    Output
    ------
    anc_qubit_samples:  the ancilla qubit outcomes per shot and per QEC round. (xArray of size # of shots x # of rounds  x # of ancilla)
    data_qubit_samples: the data qubit outcomes (last measurements). (xArray of size # shots x # of data qubits)

    '''
    NUM_ANC            = len(ANC_QUBITS)
    NN                 = NUM_ANC*NUM_ROUNDS
    anc_qubit_samples  = samples[:,:NN]
    data_qubit_samples = samples[:,NN:]
    anc_qubit_samples  = anc_qubit_samples.reshape(NUM_SHOTS,NUM_ROUNDS,NUM_ANC) 
    
    SHOTS      = np.arange(NUM_SHOTS)
    QEC_ROUNDS = np.arange(NUM_ROUNDS)

    anc_meas = xr.DataArray(data = anc_qubit_samples, 
                            dims=["shot","qec_round","anc_qubit"],
                            coords = dict(shot=SHOTS,
                                        qec_round=QEC_ROUNDS,
                                        anc_qubit=ANC_QUBITS),)


    data_meas = xr.DataArray(data=data_qubit_samples, 
                            dims=["shot","data_qubits"],
                            coords = dict(shot=SHOTS,data_qubits=DATA_QUBITS))


    return anc_meas, data_meas 


def get_initial_state(anc_qubits):

    #Initial ancilla state
    initial_state = np.zeros(len(anc_qubits), dtype=int)
    initial_state = initial_state==True
    initial_state = xr.DataArray ( data=initial_state,dims=[ "anc_qubit" ] ,coords=dict ( anc_qubit=anc_qubits) , ) 
    
    return initial_state

def project_data_meas(data_meas, num_shots,num_rounds,d,anc_qubits):

    shots = np.arange(num_shots) 
    
    syndrome_proj = data_meas.data[:,:-1] ^ data_meas.data[:,1:]  #dims: # of shots x # of ancilla
    #the 0th and 1st entries are xored, to give the 1st ancilla measurement equivalent
    #the 1st and 2nd entries are xored, to give the 2nd ancilla measurement equivalent

    syndrome_proj = xr.DataArray(data=syndrome_proj,
                                 dims=["shot","anc_qubit"],
                                 coords = dict(shot=shots,anc_qubit=anc_qubits))

    syndrome_proj["qec_round"]  = num_rounds    

    return syndrome_proj


def get_defects(num_shots, circuit,d, num_rounds):

    data_qubits   = np.arange(0,d).tolist()
    anc_qubits    = np.arange(d,d+d-1).tolist()

    initial_state = get_initial_state(anc_qubits)

    sampler   = circuit.compile_sampler()
    samples   = sampler.sample(shots=num_shots)

    anc_meas,data_meas = get_measurement_data(samples,data_qubits,anc_qubits,num_rounds,num_shots)

    syndrome_proj      = project_data_meas(data_meas,num_shots,num_rounds,d,anc_qubits)
    syndromes          = xr.concat([anc_meas,syndrome_proj],"qec_round")

    syndrome_matrix_copy              = syndromes.copy()
    syndrome_matrix_copy.data[:,-1,:] = initial_state
    defects_matrix                    = syndromes ^ syndrome_matrix_copy.roll(qec_round=1)


    return defects_matrix 


#Get the estimated error probabilities 
def extract_error_probs(num_rounds,defects_matrix):

    num_rounds  = np.shape(defects_matrix)[1]-1
    num_ancilla = np.shape(defects_matrix)[2]

    vi_mean   = avg_vi(defects_matrix)
    vivj_mean = avg_vivj(defects_matrix)

    #Estimate the different probs: dictionaries where fields have the form ((rd1,rd2),(anc1,anc2))
    pij_time = estimate_time_edge_probs(num_rounds,num_ancilla, vi_mean,vivj_mean)
    pij_bd   = estimate_data_edge_bd_probs(num_rounds,num_ancilla,vi_mean,vivj_mean,
                                           include_spacetime_edges=True,dictionary_format=True)
    
    pij_bulk = estimate_data_edge_bulk_probs_all(num_rounds,num_ancilla,vi_mean,vivj_mean)   

    return pij_time,pij_bd,pij_bulk 


def space_bulk_edge_names(num_rounds,num_ancilla):
    '''Return the name of space edges in the bulk as a dictionary.'''
    
    bulk_edge_names = []

    for rd1 in range(num_rounds+1):
        rd2 = rd1
        for anc1 in range(num_ancilla-1):
            anc2 = anc1 +1

            indx1 = anc1 + num_ancilla*rd1
            indx2 = anc2 + num_ancilla*rd2

            name = ("D"+str(indx1),"D"+str(indx2))

            bulk_edge_names.append(name)

    #Add diagonal edges
    for rd1 in range(num_rounds):
        rd2 = rd1+1
        for anc1 in range(1,num_ancilla):
            anc2 = anc1 -1

            indx1 = anc1 + num_ancilla*rd1
            indx2 = anc2 + num_ancilla*rd2

            name = ("D"+str(indx1),"D"+str(indx2))

            bulk_edge_names.append(name)

    return bulk_edge_names


def construct_estimated_DEM(pij_time,pij_bd,pij_bulk,space_edge_names,stims_DEM):

    my_DEM = stim.DetectorErrorModel()

    for key in pij_bd.keys():

        det_id  = int(key[1:])
        targets = [stim.target_relative_detector_id(det_id),stim.target_logical_observable_id(0)]
        my_DEM.append("error",pij_bd[key],targets)
    
    for key in pij_time.keys():

        det_id0 = int(key[0][1:])
        det_id1 = int(key[1][1:])

        targets = [stim.target_relative_detector_id(det_id0),stim.target_relative_detector_id(det_id1)]
        my_DEM.append("error",pij_time[key],targets)

    for name in space_edge_names:

        det_id0 = int(name[0][1:])
        det_id1 = int(name[1][1:])
        targets = [stim.target_relative_detector_id(det_id0),stim.target_relative_detector_id(det_id1)]
        targets.append(stim.target_logical_observable_id(0))
        
        my_DEM.append("error",pij_bulk[name],targets)


    for instruction in stims_DEM:

        if instruction.type=="error":
            continue
        else: #detector annotations
            my_DEM.append(instruction)

    return my_DEM


