import numpy as np
from noise_est_funcs_for_color_code import *
from utilities.general_utils import *

def get_observable_flips(data_qubit_samples, distance: int):
    '''Find whether the logical observable is flipped in each experiment.

    Input:
        data_qubit_samples: an array of the data qubit measurements (last QEC round) of dims # of shots x # of data qubits
        distance:           distance of the color code
    Output:
        obs_flips: an array of True or False (True: flipped, False: not flipped) of length # of shots
    '''
    
    restricted_data = data_qubit_samples.data[:,:distance]
    obs_flips       = np.logical_xor.reduce(restricted_data, axis=1)

    return obs_flips

def get_all_det_nodes(obj):
    '''Return all detector node names of the color code circuit.

    Input:
        obj: the color code object
    Output:
        det_nodes: list of detector node names of the form "Dj" 
        '''
    all_nodes   = collect_color_of_nodes(obj)
    det_nodes = all_nodes['r']+all_nodes['g']+all_nodes['b']
    
    inds      = [int(node[1:]) for node in det_nodes]    
    inds      = np.sort(inds)
    det_nodes = [f"D{ind}" for ind in inds]

    return det_nodes 

def get_Z_X_det_nodes(obj, num_rounds: int):
    '''Get the detector node of the Z and X dems.

    Input:
        obj: the color code object
        num_rounds: the total # of QEC rounds (int)
    Output:
        Z_det_nodes: set of names of Z-type detectors
        Z_det_nodes: set of names of X-type detectors'''

    num_ancilla   = len(obj.qubit_groups['anc'])
    det_nodes     = get_all_det_nodes(obj)
    num_Z_ancilla = num_ancilla//2
    num_X_ancilla = num_ancilla//2
    
    Z_det_nodes = []
    X_det_nodes = []

    for rd in range(num_rounds+1):

        if rd==0 or rd==1:

            for anc in range(num_Z_ancilla):
                temp = det_nodes.pop(0)
                Z_det_nodes.append(temp)
            
        else:
            
            while True:
                
                for anc in range(num_X_ancilla):
                    temp=det_nodes.pop(0)
                    X_det_nodes.append(temp)

                for anc in range(num_Z_ancilla):
                    temp = det_nodes.pop(0)
                    Z_det_nodes.append(temp)

                if len(det_nodes)==0:
                    return Z_det_nodes,X_det_nodes
                elif len(det_nodes)==num_Z_ancilla:

                    Z_det_nodes += det_nodes
                    return Z_det_nodes,X_det_nodes
               
    return Z_det_nodes,X_det_nodes

def get_Z_X_det_nodes_as_rd_anc_pairs(obj, num_rounds: int):
    '''Get the (rd,anc) pair for each detector that exists in the X- or Z-DEM.
    
    Input:
        obj: the color code object
        num_rounds: the total # of QEC rounds (int)
        
    Output:
        Z_det_nodes: list of tuples (rd,anc) which are names of Z-type detectors
        Z_det_nodes: list of tuples (rd,anc) which are names of X-type detectors
        '''

    num_ancilla = len(obj.qubit_groups['anc'])

    num_Z_ancilla = num_ancilla//2
    num_X_ancilla = num_ancilla//2
    
    Z_det_nodes = []
    X_det_nodes = []
               
    for rd in range(num_rounds+1):

        for anc in range(num_Z_ancilla):
            Z_det_nodes.append((rd,anc))

    for rd in range(num_rounds-1):
        for anc in range(num_X_ancilla):
            X_det_nodes.append((rd,anc))

    return Z_det_nodes,X_det_nodes

def get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj, num_rounds: int):
    '''Get the (rd,anc) pair for each detector that exists in the X- or Z-DEM, in dictionary format.
    
    Input:
        obj: the color code object
        num_rounds: the total # of QEC rounds (int)
    Output:
        Z_dets_dict: dictionary with keys the detector names "Dj" and values the (rd,anc) corresponding to each detector in the Z-DEM
        X_dets_dict: dictionary with keys the detector names "Dj" and values the (rd,anc) corresponding to each detector in the X-DEM
        '''
    Z_dets,X_dets = get_Z_X_det_nodes_as_rd_anc_pairs(obj,num_rounds)

    Z_det_names,X_det_names = get_Z_X_det_nodes(obj,num_rounds)

    Z_dets_dict = {}
    X_dets_dict = {}

    for k in range(len(Z_det_names)):

        name        = Z_det_names[k]
        rd_anc_pair = Z_dets[k]

        Z_dets_dict[name]=rd_anc_pair

    for k in range(len(X_det_names)):
        name        = X_det_names[k]
        rd_anc_pair = X_dets[k]

        X_dets_dict[name]=rd_anc_pair

    return Z_dets_dict,X_dets_dict

def collect_color_of_nodes(obj):
    '''Get all the detectors of the color code object distinguished by their color.

    Input:
        obj: the color code object
    Output:
        nodes: dictionary with keys 'r', 'b', 'g' and values the "Dj" detectors corresponding to each color
    '''

    nodes = {key: ["D" + str(val) for val in vals] for key, vals in obj.detector_ids.items()}

    return nodes
