import numpy as np
import xarray as xr


def get_measurement_data(samples,DATA_QUBITS: list,ANC_QUBITS: list,NUM_ROUNDS: int,NUM_SHOTS: int):
    '''
    Extract the measurement outcomes as ancilla and data qubit outcomes, given the samples (from sampler) produced by stim.

    Input:
        samples: an array of the total samples collected from sampler.samples of stim (corresponding to data and ancilla qubits)
        DATA_QUBITS: a list of str containing the data qubit names
        ANC_QUBITS: a list of str containing the ancilla qubit names
        NUM_ROUNDS: the total # of QEC rounds
        NUM_SHOTS: the total # of shots

    Output:
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


def get_initial_state(anc_qubits: list):
    '''Get the initial state for ancilla qubits. Assumed to be the all 0 state.
    
    Input: 
        anc_qubits: list of integers denoting the ancilla qubit names
    Output:
        initial_state: xArray of all zeros, with dimensions the ancilla qubit names
    '''
    
    # initial_state = np.zeros(len(anc_qubits), dtype=int)
    initial_state = np.zeros(len(anc_qubits), dtype=np.uint8) 
    # initial_state = initial_state==True
    initial_state = xr.DataArray ( data=initial_state,dims=[ "anc_qubit" ] ,coords=dict ( anc_qubit=anc_qubits) , ) 
    
    return initial_state
