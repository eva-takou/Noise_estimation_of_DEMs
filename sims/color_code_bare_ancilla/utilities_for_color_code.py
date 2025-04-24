import numpy as np
from noise_est_funcs_for_color_code import *
from utilities.general_utils import *


def get_all_det_nodes(obj):
    '''Returns all the detector nodes of the circuit. For rd>1 it includes both detectors of Z and X checks.'''
    all_nodes   = collect_color_of_nodes(obj)
    det_nodes = all_nodes['r']+all_nodes['g']+all_nodes['b']
    #now order them
    inds=[]
    for node in det_nodes:
        inds.append(int(node[1:]))
    
    inds = np.sort(inds)
    det_nodes=[]
    for ind in inds:
        det_nodes.append("D"+str(ind))

    return det_nodes 

def get_Z_X_det_nodes(obj,num_rounds):
    '''
    Input:
        obj: the color code object
        num_rounds: the total # of QEC rounds (int)
        num_ancilla: total # of ancilla qubits (i.e., both Z and X type) (int)
    Output:
        Z_det_nodes: list of names of Z-type detectors
        Z_det_nodes: list of names of X-type detectors'''

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

def get_Z_X_det_nodes_as_rd_anc_pairs(obj,num_rounds):
    '''
    Input:
        obj: the color code object
        num_rounds: the total # of QEC rounds (int)
        
    Output:
        Z_det_nodes: list of tuples (rd,anc) which are names of Z-type detectors
        Z_det_nodes: list of tuples (rd,anc) which are names of X-type detectors'''

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

def get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds):

    Z_dets,X_dets = get_Z_X_det_nodes_as_rd_anc_pairs(obj,num_rounds)

    Z_det_names,X_det_names = get_Z_X_det_nodes(obj,num_rounds)

    Z_dets_dict = {}
    X_dets_dict = {}

    for k in range(len(Z_det_names)):

        name        = Z_det_names[k]
        rd_anc_pair = Z_dets[k]

        Z_dets_dict[name]=rd_anc_pair

    for k in range(len(X_det_names)):
        name = X_det_names[k]
        rd_anc_pair = X_dets[k]

        X_dets_dict[name]=rd_anc_pair

    return Z_dets_dict,X_dets_dict

def collect_color_of_nodes(obj):
    '''These are all the detector ids. For rd=1 we have only Z-type ancilla.
    For rd>1 we have both Z and X type ancilla.'''

    nodes = {key: ["D" + str(val) for val in vals] for key, vals in obj.detector_ids.items()}

    return nodes
