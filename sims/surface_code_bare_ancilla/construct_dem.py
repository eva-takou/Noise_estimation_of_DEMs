import numpy as np
import xarray as xr
from sims.surface_code_bare_ancilla.surface_code_lattice import surface_code_star_stabs
import stim
from utilities.defects_matrix_utils import *

from numba import njit

# def project_data_meas(data_qubit_samples, L: int, NUM_ROUNDS: int, NUM_SHOTS: int, ANC_QUBITS: list):
#     '''Perform stabilizer reconstruction for the surface code, using the last data qubit measurements.

#     Input: 
#         data_qubit_samples: the measurement outcomes of data qubits (xArray of dimensions # of shots x # of data qubits)
#         L: distance of surface code (int)
#         NUM_ROUNDS: # of QEC rounds (int)
#         NUM_SHOTS: # of shots we sample the stim circuit (int)
#         ANC_QUBITS: ancilla qubit names (list)
#     Output:
#         syndrome_proj: syndrome projection onto stabilizers (xArray of dimensions # of shots x # of ancilla qubits)
        
#     '''
#     shots    = np.arange(NUM_SHOTS)
#     HX       = surface_code_star_stabs(L).todense()
#     num_rows = np.shape(HX)[0]
#     s_proj   = []
    

#     for k in range(num_rows):

#         locs = np.nonzero(HX[k,:])[1] 
#         temp = data_qubit_samples.data[:,locs[0]]
        
#         for m in range(1,len(locs)):
            
#             # if m==0:
#             #     continue
#             # else:
#             temp = temp ^ data_qubit_samples.data[:,locs[m]]

#         s_proj.append(temp)


#     s_proj  = np.vstack(s_proj).T

#     syndrome_proj = xr.DataArray(data=s_proj,
#                 dims=["shot","anc_qubit"],
#                 coords=dict(shot=shots,anc_qubit=ANC_QUBITS))
#     syndrome_proj["qec_round"]  = NUM_ROUNDS

#     return syndrome_proj


def project_data_meas(data_qubit_samples, L: int, NUM_ROUNDS: int, NUM_SHOTS: int, ANC_QUBITS: list):
    '''Perform stabilizer reconstruction for the surface code, using the last data qubit measurements.

    Input: 
        data_qubit_samples: the measurement outcomes of data qubits (xArray of dimensions # of shots x # of data qubits)
        L: distance of surface code (int)
        NUM_ROUNDS: # of QEC rounds (int)
        NUM_SHOTS: # of shots we sample the stim circuit (int)
        ANC_QUBITS: ancilla qubit names (list)
    Output:
        syndrome_proj: syndrome projection onto stabilizers (xArray of dimensions # of shots x # of ancilla qubits)
        
    '''
 
    HX       = surface_code_star_stabs(L).todense()
    s_proj   = project_data_fast(data_qubit_samples.data.astype(np.uint8), HX.astype(np.uint8))


    syndrome_proj = xr.DataArray(data=s_proj,
                dims=["shot","anc_qubit"],
                coords=dict(shot=np.arange(NUM_SHOTS),anc_qubit=ANC_QUBITS))
    syndrome_proj["qec_round"]  = NUM_ROUNDS

    return syndrome_proj


@njit
def project_data_fast(data, HX):
    num_shots = data.shape[0]
    num_stabilizers = HX.shape[0]
    s_proj = np.empty((num_shots, num_stabilizers), dtype=np.uint8)

    for k in range(num_stabilizers):
        # Get indices of qubits involved in stabilizer k
        locs = np.nonzero(HX[k])[0]
        temp = data[:, locs[0]].copy()
        for m in range(1, len(locs)):
            temp ^= data[:, locs[m]]
        s_proj[:, k] = temp

    return s_proj


def get_defects(circuit: stim.Circuit, distance: int, num_shots: int, num_rounds: int):
    '''Reconstruct the defects matrix of the surface code circuit.
    
    Input: 
        circuit: the stim circuit (stim.Circuit)
        distance: distance of surface code (int)
        num_shots: # of times to sample the raw measurements of the stim circuit (int)
        num_rounds: # of QEC rounds (int)
    
    Output:
        defects_matrix: detector/defects matrix of dimesnsions # of shots x # of QEC rounds +1 x # of detectors (xArray)
                       
    '''

    L             = distance
    all_qubits    = circuit.get_final_qubit_coordinates()
    qubit_indices = list(all_qubits.keys())

    ANC_QUBITS  = np.arange(L**2+(L-1)**2+1,len(qubit_indices)+1)
    DATA_QUBITS = list( set(ANC_QUBITS) ^ set(qubit_indices) )


    initial_state = get_initial_state(ANC_QUBITS)

    sampler   = circuit.compile_sampler()
    # samples   = sampler.sample(shots=num_shots)
    samples   = sampler.sample(shots=num_shots).astype(np.uint8)

    anc_qubit_samples,data_qubit_samples = get_measurement_data(samples,DATA_QUBITS,ANC_QUBITS,num_rounds,num_shots)

    #Syndrome projection
    syndrome_proj = project_data_meas(data_qubit_samples,L,num_rounds,num_shots,ANC_QUBITS)

   
    #Construct the defect matrix
    syndromes      = xr.concat([anc_qubit_samples,syndrome_proj],"qec_round")
    
    syndrome_matrix_copy              = syndromes.copy()
    syndrome_matrix_copy.data[:,-1,:] = initial_state
    defects_matrix                    = syndromes ^ syndrome_matrix_copy.roll(qec_round=1)


    return defects_matrix, data_qubit_samples


def surface_code_DEM(pij_bulk: dict,pij_bd: dict,pij_time: dict, stims_DEM: stim.DetectorErrorModel):
    '''Construct the detector error model of the surface code based on the probabilities we estimate.
    
    Input:
        pij_bulk: dictionary containing the bulk edge probabilities
        pij_bd:   dictionary containing the boundary edge probabilities
        pij_time: dictionary containing the time edge probabilities
        stims_DEM: stims detector error model

    Output:
        reconstructed_DEM: the detector error model we estimate

    '''
    reconstructed_DEM = stim.DetectorErrorModel()

    # for key in pij_bd.keys():

    #     det_indx = int(key[1:])

    #     if pij_bd[key]>0:

    #         reconstructed_DEM.append("error",pij_bd[key], #
    #                             [stim.target_relative_detector_id(det_indx), ])

    for key,prob in pij_bd.items():

        if prob>0:
            det_indx = int(key[1:])
            reconstructed_DEM.append("error",prob, #
                                [stim.target_relative_detector_id(det_indx), ])


    # for key in pij_time.keys():

    #     d0,d1 = key

    #     det_indx1 = int(d0[1:])
    #     det_indx2 = int(d1[1:])

    #     if pij_time[key]>0:

    #         reconstructed_DEM.append("error",pij_time[key], #
    #                             [stim.target_relative_detector_id(det_indx1),stim.target_relative_detector_id(det_indx2) ])        

    for (d0,d1),prob in pij_time.items():

        if prob>0:

            det_indx1 = int(d0[1:])
            det_indx2 = int(d1[1:])
            reconstructed_DEM.append("error",prob, #
                                [stim.target_relative_detector_id(det_indx1),stim.target_relative_detector_id(det_indx2) ])        



    # for key in pij_bulk.keys():

    #     d0 = key[0]
    #     d1 = key[1]
    #     det_indx1 = int(d0[1:])
    #     det_indx2 = int(d1[1:])
    #     logic = key[2]

    #     if logic=='': #No error
            
    #         if pij_bulk[key]>0:
    #             reconstructed_DEM.append("error",pij_bulk[key], #
    #                                 [stim.target_relative_detector_id(det_indx1),stim.target_relative_detector_id(det_indx2) ])        
    #     else:

    #         if pij_bulk[key]>0:
    #             reconstructed_DEM.append("error",pij_bulk[key], #
    #                                 [stim.target_relative_detector_id(det_indx1),stim.target_relative_detector_id(det_indx2),
    #                                 stim.target_logical_observable_id(0)])        

    for (d0,d1,logic), prob in pij_bulk.items():

        if prob>0:
            det_indx1 = int(d0[1:])
            det_indx2 = int(d1[1:])
            targets = [stim.target_relative_detector_id(det_indx1),stim.target_relative_detector_id(det_indx2) ]

            if logic != '':
                targets.append(stim.target_logical_observable_id(0))

            reconstructed_DEM.append("error",prob,targets)

     

    for instruction in stims_DEM:

        if instruction.type != "error":
            reconstructed_DEM.append(instruction)

        # if instruction.type=="error":
        #     continue
        # else:   #detector annotations
        #     reconstructed_DEM.append(instruction)


    return reconstructed_DEM
