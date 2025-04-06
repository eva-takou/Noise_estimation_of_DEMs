from utilities.general_utils import avg_vi, avg_vivj
import xarray as xr
import numpy as np
import stim
import math
import time


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

    # det_nodes     = get_all_det_nodes(obj)
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


#Get the dictionary: Note that for the X-part we "rename" the first appearance of X-type detectors as the 0-th round
#for X-type detectors.
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


# def collect_color_of_nodes(obj):
#     '''These are all the detector ids. For rd=1 we have only Z-type ancilla.
#     For rd>1 we have both Z and X type ancilla.'''
#     dets  = obj.detector_ids
#     nodes = {}
#     for key in dets.keys():
#         vals = dets[key]
#         temp = []
#         for val in vals:
#             temp.append("D"+str(val))
#         nodes[key] =temp

#     return nodes


def collect_color_of_nodes(obj):
    '''These are all the detector ids. For rd=1 we have only Z-type ancilla.
    For rd>1 we have both Z and X type ancilla.'''

    nodes = {key: ["D" + str(val) for val in vals] for key, vals in obj.detector_ids.items()}

    return nodes



def get_dets_from_stim_circuit(circuit):
    
    DEM = circuit.detector_error_model()
    cnt = 0
    
    for instruction in DEM:

        if instruction.type=="detector":
            
            det_annotations = DEM[cnt+1:]
            break
        cnt+=1

    return det_annotations


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
    NUM_ANC = len(ANC_QUBITS)
    NN      = NUM_ANC*NUM_ROUNDS
    #Anc samples go as: first Z ancilla, then X ancilla, then Z ancilla and so on (alternating)
    anc_qubit_samples  = samples[:,:NN]
    data_qubit_samples = samples[:,NN:]
    anc_qubit_samples  = anc_qubit_samples.reshape(NUM_SHOTS,NUM_ROUNDS,NUM_ANC)  #First ancilla are Z next ancilla are X
    
    SHOTS      = np.arange(NUM_SHOTS)
    QEC_ROUNDS = np.arange(NUM_ROUNDS)

    anc_meas = xr.DataArray(data   = anc_qubit_samples, 
                            dims   = ["shot","qec_round","anc_qubit"],
                            coords = dict(shot=SHOTS,
                                        qec_round=QEC_ROUNDS,
                                        anc_qubit=ANC_QUBITS),)


    data_meas = xr.DataArray(data   = data_qubit_samples, 
                             dims   = ["shot","data_qubits"],
                             coords = dict(shot=SHOTS,data_qubits=DATA_QUBITS))


    return anc_meas, data_meas 


def get_events_that_cause_L0_flips(circuit):
    '''Use the circuit instructions to find which errors cause logical flips.
       These error events are output as a list of tuple ("Dj","Dk"), or ("Dj","").
    '''
    dem    = circuit.detector_error_model()
    events = []

    for instruction in dem:
        if instruction.type=="error":
            targets=instruction.targets_copy()
                
            if targets[-1].is_logical_observable_id():
                temp = []
                for k in range(len(targets)-1):
                    
                    temp.append(str(targets[k]))
                events.append(tuple(temp))

    return events

def get_qubit_names(obj):
    '''Get all qubit names for the circuit, including data, ancilla, anc_X, anc_Z etc'''

    num_data  = len(obj.qubit_groups['data'])
    num_anc   = len(obj.qubit_groups['anc'])
    num_anc_X = len(obj.qubit_groups['anc_X'])
    num_anc_Z = len(obj.qubit_groups['anc_Z'])

    num_anc_red   = len(obj.qubit_groups['anc_red'])
    num_anc_blue  = len(obj.qubit_groups['anc_blue'])
    num_anc_green = len(obj.qubit_groups['anc_green'])

    data_qubit_names   = []
    anc_qubit_names    = []
    X_anc_qubit_names  = []
    Z_anc_qubit_names  = []
    anc_red            = []
    anc_green          = []
    anc_blue           = []

    for k in range(num_data):
        data_qubit_names.append(obj.qubit_groups['data'][k]['qid'])

    for k in range(num_anc):
        anc_qubit_names.append(obj.qubit_groups['anc'][k]['qid'])

    for k in range(num_anc_X):
        X_anc_qubit_names.append(obj.qubit_groups['anc_X'][k]['qid'])

    for k in range(num_anc_Z):
        Z_anc_qubit_names.append(obj.qubit_groups['anc_Z'][k]['qid'])    
    
    for k in range(num_anc_red):
        anc_red.append(obj.qubit_groups['anc_red'][k]['qid'])    

    for k in range(num_anc_blue):
        anc_blue.append(obj.qubit_groups['anc_blue'][k]['qid'])    

    for k in range(num_anc_green):
        anc_green.append(obj.qubit_groups['anc_green'][k]['qid'])    

    qubit_names            = {}
    qubit_names['data']    = data_qubit_names
    qubit_names['anc']     = anc_qubit_names
    qubit_names['anc_X']   = X_anc_qubit_names
    qubit_names['anc_Z']   = Z_anc_qubit_names

    qubit_names['anc_red']   = anc_red
    qubit_names['anc_blue']  = anc_blue
    qubit_names['anc_green'] = anc_green

    return qubit_names


def get_inds_for_syndrome_projection(circuit):
    '''Get the indices for which data qubits need to be combined to form the 
    stabilizer projection'''
    cnt      = 0
    data_Q   = []
    all_inds = []
    L_C      = len(circuit)
    all_qubits_per_stab=[]
    for instruction in circuit:

        if instruction.name=="M": #Last data qubit measurement
            temp_Q=instruction.targets_copy()
            for Q in temp_Q:
                
                data_Q.append(Q.qubit_value)

            for l in range(cnt+1,L_C-1): #Ignore last one which is observable include
                
                current_instruction  = circuit[l]
                recs = current_instruction.targets_copy()
                qubits_per_stab = []
                inds=[]
                for rec in recs:
                    id = rec.value
                    
                    
                    if abs(id)<=len(data_Q):
                        id=id+len(data_Q)
                        qubits_per_stab.append(data_Q[id])
                        inds.append(id)

                all_inds.append(inds)
                all_qubits_per_stab.append(qubits_per_stab)

        cnt=cnt+1

    return all_inds



#TODO: Make the part of syndrome projection faster..
def get_defects(circuit,num_rounds,num_shots,obj):
    '''Get the defects (i.e., detector) matrix of the circuit, by composing raw circuit measurements.

    Input:
        circuit: the stim.Circuit
        num_rounds: the # of QEC rounds (excluding the last data qubit measurement) (int) 
        num_shots: the # of shots to sample from the circuit
        obj: the color code object

    Output:
        defects_matrix_Z: the defects matrix originating from Z-stab measurements (xArray: num_shots x (# of QEC rds + 1) x num_Z_ancilla)
        defects_matrix_X: the defects matrix originating from X-stab measurements (xArray: num_shots x (# of QEC rds + 1) x num_X_ancilla)
        data_qubit_samples: the last data qubit measurements (xArray: num_shots x num_data_qubits)
        Z_ANC_QUBITS: list of names of Z-ancilla qubits
        X_ANC_QUBITS: list of names of X-ancilla qubits
        
    '''
    qubit_names  = get_qubit_names(obj)
    DATA_QUBITS  = qubit_names['data']
    Z_ANC_QUBITS = qubit_names['anc_Z']
    X_ANC_QUBITS = qubit_names['anc_X']

    shots   = np.arange(num_shots)    
    sampler = circuit.compile_sampler()
    samples = sampler.sample(shots=num_shots)

    ANC_QUBITS = Z_ANC_QUBITS+X_ANC_QUBITS #So that we order the measurement data as first Z, then X ancilla outcomes
    
    
    anc_qubit_samples,data_qubit_samples = get_measurement_data(samples=samples,
                                                                DATA_QUBITS=DATA_QUBITS,
                                                                ANC_QUBITS=ANC_QUBITS,
                                                                NUM_ROUNDS=num_rounds,
                                                                NUM_SHOTS=num_shots) #First ancilla are Z, last ancilla are X

    Z_anc_qubit_samples = anc_qubit_samples.loc[:,:,Z_ANC_QUBITS] 
    X_anc_qubit_samples = anc_qubit_samples.loc[:,:,X_ANC_QUBITS] 


    #---------- Projection of data qubit measurements onto Z-stabs ---------------
    
    all_inds = get_inds_for_syndrome_projection(circuit)

    # s_proj   = []

    

    # for inds in all_inds:

        # temp = data_qubit_samples.data[:,inds[0]]
        
        # for m in range(len(inds)):
            
        #     if m==0:
        #         continue
        #     else:
        #         temp = temp ^ data_qubit_samples.data[:,inds[m]]
        
        # s_proj.append(temp)
    

    # if np.array_equal(s_proj,s_proj_alt):
    #     print("All good")
    # else:
    #     raise Exception("This is incorrect.")
    
    
    #This takes too long....
    s_proj = [np.logical_xor.reduce(data_qubit_samples.data[:, inds], axis=1) for inds in all_inds]

    s_proj  = np.vstack(s_proj).T

    syndrome_proj               = xr.DataArray(data   = s_proj,
                                               dims   = ["shot","anc_qubit"],
                                               coords = dict(shot=shots,anc_qubit=Z_ANC_QUBITS))
    syndrome_proj["qec_round"]  = num_rounds
    syndromes_Z                 = xr.concat([Z_anc_qubit_samples,syndrome_proj],"qec_round")

    #---------- Defects matrix from Z checks (X defects) --------------------
    initial_state = xr.DataArray ( data=np.zeros(len(Z_ANC_QUBITS), dtype=int),
                                        dims=[ "anc_qubit" ] ,
                                        coords=dict ( anc_qubit=Z_ANC_QUBITS) , )      

    syndrome_matrix_copy              = syndromes_Z.copy()
    syndrome_matrix_copy.data[:,-1,:] = initial_state

    defects_matrix_Z                  = syndromes_Z ^ syndrome_matrix_copy.roll(qec_round=1)



    #------ Repeat for the X checks (Z defects) -----------------
   
    if num_rounds>1:
        syndromes_X                       = X_anc_qubit_samples[:,1:,:] #Start from 1st rd.
        syndrome_matrix_copy              = syndromes_X.copy()
        initial_state                     = X_anc_qubit_samples[:,0,:].copy()   
        syndrome_matrix_copy.data[:,-1,:] = initial_state              
        defects_matrix_X                  = syndromes_X ^ syndrome_matrix_copy.roll(qec_round=1)
    else:
        X_ANC_QUBITS=[]
        defects_matrix_X=[]


    return defects_matrix_Z,defects_matrix_X,data_qubit_samples,Z_ANC_QUBITS,X_ANC_QUBITS #defects_matrix,data_qubit_samples,Z_ANC_QUBITS





#In this case we directly use the circuit compiler of Stim.
def get_defects_V3(circuit,num_rounds,num_shots,obj):
    '''Get the defects (i.e., detector) matrix of the circuit, by composing raw circuit measurements.

    Input:
        circuit: the stim.Circuit
        num_rounds: the # of QEC rounds (excluding the last data qubit measurement) (int) 
        num_shots: the # of shots to sample from the circuit
        obj: the color code object

    Output:
        defects_matrix_Z: the defects matrix originating from Z-stab measurements (xArray: num_shots x (# of QEC rds + 1) x num_Z_ancilla)
        defects_matrix_X: the defects matrix originating from X-stab measurements (xArray: num_shots x (# of QEC rds + 1) x num_X_ancilla)
        data_qubit_samples: the last data qubit measurements (xArray: num_shots x num_data_qubits)
        Z_ANC_QUBITS: list of names of Z-ancilla qubits
        X_ANC_QUBITS: list of names of X-ancilla qubits
        
    '''
    qubit_names  = get_qubit_names(obj)
    # DATA_QUBITS  = qubit_names['data']
    Z_ANC_QUBITS = qubit_names['anc_Z']
    X_ANC_QUBITS = qubit_names['anc_X']


    Z_dets,X_dets = get_Z_X_det_nodes(obj,num_rounds)

    Z_anc_inds=[]
    for det in Z_dets:
        Z_anc_inds.append(int(det[1:]))

    X_anc_inds=[]
    for det in X_dets:
        X_anc_inds.append(int(det[1:]))

    shots   = np.arange(num_shots)    
    sampler = circuit.compile_detector_sampler()
    samples = sampler.sample(shots=num_shots)

    #First we have Z_ANC_QUBITS then X_ANC_QUBITS

    Z_anc_samples = samples[:,Z_anc_inds]
    Z_anc_samples = Z_anc_samples.reshape(num_shots,num_rounds+1,len(Z_ANC_QUBITS))

    if X_anc_inds!=[]:
        X_anc_samples = samples[:,X_anc_inds]
        X_anc_samples = X_anc_samples.reshape(num_shots,num_rounds,len(X_ANC_QUBITS))
        defects_matrix_X = xr.DataArray(data = X_anc_samples, dims=["shot","qec_round","anc_qubit"],
                                                                    coords=dict(shot=shots,
                                                                                qec_round=np.arange(num_rounds+1),
                                                                                anc_qubit=X_ANC_QUBITS))
    else:
        defects_matrix_X=[]

    
    defects_matrix_Z = xr.DataArray(data = Z_anc_samples, dims=["shot","qec_round","anc_qubit"],
                                                                coords=dict(shot=shots,
                                                                            qec_round=np.arange(num_rounds+1),
                                                                            anc_qubit=Z_ANC_QUBITS))




    return defects_matrix_Z,defects_matrix_X



#This is slower
def get_defects_V2(circuit,num_rounds,num_shots,obj):
    '''Get the defects (i.e., detector) matrix of the circuit, by composing raw circuit measurements.

    Input:
        circuit: the stim.Circuit
        num_rounds: the # of QEC rounds (excluding the last data qubit measurement) (int) 
        num_shots: the # of shots to sample from the circuit
        obj: the color code object

    Output:
        defects_matrix_Z: the defects matrix originating from Z-stab measurements (xArray: num_shots x (# of QEC rds + 1) x num_Z_ancilla)
        defects_matrix_X: the defects matrix originating from X-stab measurements (xArray: num_shots x (# of QEC rds + 1) x num_X_ancilla)
        data_qubit_samples: the last data qubit measurements (xArray: num_shots x num_data_qubits)
        Z_ANC_QUBITS: list of names of Z-ancilla qubits
        X_ANC_QUBITS: list of names of X-ancilla qubits
        
    '''
    qubit_names  = get_qubit_names(obj)
    DATA_QUBITS  = qubit_names['data']
    Z_ANC_QUBITS = qubit_names['anc_Z']
    X_ANC_QUBITS = qubit_names['anc_X']

    shots   = np.arange(num_shots)    
    sampler = circuit.compile_sampler()
    samples = sampler.sample(shots=num_shots)

    ANC_QUBITS = Z_ANC_QUBITS+X_ANC_QUBITS #So that we order the measurement data as first Z, then X ancilla outcomes
    
    
    anc_qubit_samples,data_qubit_samples = get_measurement_data(samples=samples,
                                                              DATA_QUBITS=DATA_QUBITS,
                                                              ANC_QUBITS=ANC_QUBITS,
                                                              NUM_ROUNDS=num_rounds,
                                                              NUM_SHOTS=num_shots) #First ancilla are Z, last ancilla are X

    Z_anc_qubit_samples = anc_qubit_samples.loc[:,:,Z_ANC_QUBITS] 
    X_anc_qubit_samples = anc_qubit_samples.loc[:,:,X_ANC_QUBITS] 


    #---------- Projection of data qubit measurements onto Z-stabs ---------------
    
    all_inds = get_inds_for_syndrome_projection(circuit)

    s_proj   = []

    

    for inds in all_inds:

        temp = data_qubit_samples.data[:,inds[0]]
        
        for m in range(len(inds)):
            
            if m==0:
                continue
            else:
                temp = temp ^ data_qubit_samples.data[:,inds[m]]
        
        s_proj.append(temp)
    

    # if np.array_equal(s_proj,s_proj_alt):
    #     print("All good")
    # else:
    #     raise Exception("This is incorrect.")
    
    
    #This takes too long....
    # s_proj = [np.logical_xor.reduce(data_qubit_samples.data[:, inds], axis=1) for inds in all_inds]

    s_proj  = np.vstack(s_proj).T

    syndrome_proj               = xr.DataArray(data   = s_proj,
                                               dims   = ["shot","anc_qubit"],
                                               coords = dict(shot=shots,anc_qubit=Z_ANC_QUBITS))
    syndrome_proj["qec_round"]  = num_rounds
    syndromes_Z                 = xr.concat([Z_anc_qubit_samples,syndrome_proj],"qec_round")

    #---------- Defects matrix from Z checks (X defects) --------------------
    initial_state = xr.DataArray ( data=np.zeros(len(Z_ANC_QUBITS), dtype=int),
                                        dims=[ "anc_qubit" ] ,
                                        coords=dict ( anc_qubit=Z_ANC_QUBITS) , )      

    syndrome_matrix_copy              = syndromes_Z.copy()
    syndrome_matrix_copy.data[:,-1,:] = initial_state

    defects_matrix_Z                  = syndromes_Z ^ syndrome_matrix_copy.roll(qec_round=1)



    #------ Repeat for the X checks (Z defects) -----------------
   
    if num_rounds>1:
        syndromes_X                       = X_anc_qubit_samples[:,1:,:] #Start from 1st rd.
        syndrome_matrix_copy              = syndromes_X.copy()
        initial_state                     = X_anc_qubit_samples[:,0,:].copy()   
        syndrome_matrix_copy.data[:,-1,:] = initial_state              
        defects_matrix_X                  = syndromes_X ^ syndrome_matrix_copy.roll(qec_round=1)
    else:
        X_ANC_QUBITS=[]
        defects_matrix_X=[]


    return defects_matrix_Z,defects_matrix_X,data_qubit_samples,Z_ANC_QUBITS,X_ANC_QUBITS #defects_matrix,data_qubit_samples,Z_ANC_QUBITS




def bulk_prob_formula(v1: float,v2: float, v1v2:float):
    '''
    Estimate the bulk edge probability (for a data qubit error) based on the formula:

    1/2 - sqrt{  1/4 - numer/denom}
    
    numer =  <vi vj> - <vi><vj>
    denom =  1-2<vi \oplus vj> = 1 - 2<vi> - 2<vj> + 4<vi vj>

    where <vi> is the total # of clicks we get at a detector 'vi', and <vj> is the total # of clicks we get at a detector 'vj', 
    where 'vi' and 'vj' are connected by an edge. These are the clicks unconditional on any other events.
    The <vi vj> average is the # of coincident clicks we observe, and the <vi \oplus vj> is the number of times where one detector clicks but not the other
    and vice-versa (i.e., XOR).

    The statistics are collected across some # of QEC rounds.
    
    Input
    ---------------
    v1 (v2): the mean # of times detector 1 (2) fires  [# of detection events/ num_shots]
    v1v2: the mean # of times detector 1 and detector 2 fire
    
    Output
    ---------------
    The probability of error for edge described by the input detection events.

    '''
    numer  = v1v2 - v1*v2
    denom  = 1    - 2*(v1+v2) + 4*v1v2
    p      = 1/2  - np.sqrt(1/4- numer/denom)

    if p<0:
        p=0
    elif math.isnan(p):
        print("numer/denom:",1/4-numer/denom)
        raise Exception("Encountered nan value")

    return p


def bd_prob_formula(pij_bulk,vi_mean,anc,rd,dets_to_exclude):
    '''
    Function to calculate the boundary probability as 
    p_ii = 1/2 + (<d_i>-1/2)/\prod_{j\neq i}(1-2p_{ij}), where i-j edges are nearest to node i.
    Input:
        pij_bulk: dictionary with keys ("Dj","Dk") and values the probabilities
        vi_mean: dictionary with keys 'X' and 'Z' and values the mean <di> counts for X-type or Z-type stabs
                 (# of QEC rds x # of X/Z ancilla)
        anc: ancilla index (int)
        rd:  index of QEC round (int)
        det_to_exclude: list of detector tuples ("Dj","Dk") to use to calculate the boundary probability.
    Output:
        probability p_ii
        '''
    #Dets to exclude: list of tuples
    v0    = vi_mean[rd,anc]
    DENOM = 1
    for dets in dets_to_exclude:

        #They are tuples
        p = pij_bulk[dets]

        DENOM *=1-2*p

    pii = 1/2+(v0-1/2)/DENOM

    if pii<0:
        pii=0


    return pii


def estimate_bulk_edges(num_ancilla,num_rounds,vi_mean,vivj_mean,Z_dets,X_dets):
    '''Estimate the bulk edge probabilities of edges between Z-type nodes and edges between X-type nodes.

    Input: 
        num_ancilla: # of X-ancilla qubits = # of Z-ancilla qubits
        num_rounds:  # of QEC rds
        vi_mean: Dictionary for average <di> cnts taken over multiple shots extracted from defects matrix.
                 The keys are ['X'] (['Z']) for defects extracted from X-type (Z-type) stabs
                 size of value of dict: # of QEC rds x # of X-ancilla
        vivj_mean:
        Z_dets:

    Output:
        pij_bulk: a dictionary with keys ("Dj","Dk") and values the probabilities of the various edges
       '''
    
    pij_bulk = {}

    #Start with Z-type stabs

    for anc1 in range(num_ancilla):

        for anc2 in range(num_ancilla):

            for rd1 in range(num_rounds+1):

                for rd2 in range(num_rounds+1):

                    INDX1 = anc1 + (num_ancilla)*rd1
                    INDX2 = anc2 + (num_ancilla)*rd2

                    indx1 = min([INDX1,INDX2])
                    indx2 = max([INDX1,INDX2])

                    # det1 = "D"+str(indx1)
                    # det2 = "D"+str(indx2)

                    det1 = Z_dets[indx1]
                    det2 = Z_dets[indx2]

                    v1   = vi_mean['Z'][rd1,anc1]
                    v2   = vi_mean['Z'][rd2,anc2]
                    v1v2 = vivj_mean['Z'][rd1,rd2,anc1,anc2]

                    if det1==det2:
                        continue
                    
                    val                   = bulk_prob_formula(v1,v2,v1v2)
                    pij_bulk[(det1,det2)] = val

    #X-type stabs
    for anc1 in range(num_ancilla):

        for anc2 in range(num_ancilla):

            for rd1 in range(num_rounds-1):

                for rd2 in range(num_rounds-1):

                    INDX1 = anc1 + (num_ancilla)*rd1
                    INDX2 = anc2 + (num_ancilla)*rd2

                    indx1 = min([INDX1,INDX2])
                    indx2 = max([INDX1,INDX2])

                    # det1 = "D"+str(indx1)
                    # det2 = "D"+str(indx2)

                    det1 = X_dets[indx1]
                    det2 = X_dets[indx2]

                    v1   = vi_mean['X'][rd1,anc1]
                    v2   = vi_mean['X'][rd2,anc2]
                    v1v2 = vivj_mean['X'][rd1,rd2,anc1,anc2]

                    if det1==det2:
                        continue
                    
                    val                   = bulk_prob_formula(v1,v2,v1v2)
                    pij_bulk[(det1,det2)] = val

    return pij_bulk


def get_bd_probs_for_each_restricted_color(pij_bulk,vi_mean,nodes,num_rounds,num_ancilla,obj):
    '''nodes is a dictironary with keys the colors 'r', 'g', 'b' and values a list
     of detectors ["Dj", "Dk", ...]. 

     To find the boundary probabilities for a specific color restricted lattice,
     we need to exlcude nodes that are of the particular color.
     For example, in the red-restricted lattice, we ignore any edges D_j - D_2 or D_j - D_5

     num_ancilla: # of X ancilla =  # of Z ancilla (int)
     '''
    
    rd_anc_pair_Z,rd_anc_pair_X = get_Z_X_det_nodes_as_rd_anc_pairs(obj,num_rounds,num_ancilla*2)
    Z_dets,X_dets               = get_Z_X_det_nodes(obj,num_rounds,num_ancilla*2)    
    
    pij_bd_Z={}

    #Start with Z-type nodes
    for color in nodes.keys():
        
        pij_bd_per_color={}

        if color=='r': #do not use the red nodes
            remaining_nodes = nodes['g']+nodes['b']
        elif color=='g': #do not use the green nodes
            remaining_nodes = nodes['r']+nodes['b']
        else:            #do not use the blue nodes
            remaining_nodes = nodes['r']+nodes['g']

        remaining_nodes = [x for x in remaining_nodes if x not in X_dets]

        for detector_node in remaining_nodes:

            INDX1     = int(detector_node[1:])
            loc       = Z_dets.index(detector_node)
            (rd,anc)  = rd_anc_pair_Z[loc]

            #Form all other combinations of this detector node with remaining nodes
            dets_to_exclude=[]
            
            for other_node in remaining_nodes:

                if other_node!=detector_node:
                    
                    INDX2 = int(other_node[1:])

                    indx1 = min([INDX1,INDX2])
                    indx2 = max([INDX1,INDX2])
                    temp  = ("D"+str(indx1),"D"+str(indx2))
                    dets_to_exclude.append(temp)

            val =  bd_prob_formula(pij_bulk,vi_mean['Z'],anc,rd,dets_to_exclude)
            
            if val<0:
                val=0

            pij_bd_per_color[("D"+str(INDX1))] = val

        pij_bd_Z[color]=pij_bd_per_color

    pij_bd_X={}
    #Continue with X-type nodes
    if len(X_dets)>0:
        for color in nodes.keys():
            
            pij_bd_per_color={}

            if color=='r': #do not use the red nodes
                remaining_nodes = nodes['g']+nodes['b']
            elif color=='g': #do not use the green nodes
                remaining_nodes = nodes['r']+nodes['b']
            else:            #do not use the blue nodes
                remaining_nodes = nodes['r']+nodes['g']

            remaining_nodes = [x for x in remaining_nodes if x not in Z_dets]

            for detector_node in remaining_nodes:

                INDX1     = int(detector_node[1:])
                loc       = X_dets.index(detector_node)
                (rd,anc)  = rd_anc_pair_X[loc]

                #Form all other combinations of this detector node with remaining nodes
                dets_to_exclude=[]
                
                for other_node in remaining_nodes:

                    if other_node!=detector_node:
                        
                        INDX2 = int(other_node[1:])

                        indx1 = min([INDX1,INDX2])
                        indx2 = max([INDX1,INDX2])
                        temp  = ("D"+str(indx1),"D"+str(indx2))
                        dets_to_exclude.append(temp)

                val =  bd_prob_formula(pij_bulk,vi_mean['X'],anc,rd,dets_to_exclude)
                
                if val<0:
                    val=0

                pij_bd_per_color[("D"+str(INDX1))] = val

            pij_bd_X[color]=pij_bd_per_color


        pij_bd={}
        pij_bd['r'] = pij_bd_X['r'] | pij_bd_Z['r']
        pij_bd['g'] = pij_bd_X['g'] | pij_bd_Z['g']
        pij_bd['b'] = pij_bd_X['b'] | pij_bd_Z['b']
    else:
        pij_bd = pij_bd_Z

    return pij_bd


def get_observable_flips(data_qubit_samples,distance):
    '''The observable include in the circuit is on the first d-final data qubit measurements.
    Thus, we have to XOR the first d data_qubit_samples.'''
    
    restricted_data  = data_qubit_samples.data[:,:distance]
    # cols             = np.shape(restricted_data)[1]

    # obs_flips_alt = restricted_data[:,0]
    # for l in range(1,cols):
    #     obs_flips_alt = np.logical_xor(obs_flips_alt,restricted_data[:,l])

    obs_flips = np.logical_xor.reduce(restricted_data, axis=1)

    # if not (obs_flips_alt==obs_flips_vect).all():
    #     raise Exception("Vectorized code is wrong.")
    # else:
    #     print("All good.")


    return obs_flips

def extract_single_counts_w_and_wo_L0_excluding_nodes(num_shots,obs_flips,defects_matrix,target_node,exclude_nodes):

    rd1                = target_node[0]
    anc1               = target_node[1]
    single_count_w_L0  = 0
    single_count_wo_L0 = 0

    for k in range(num_shots):

        if defects_matrix.data[k,rd1,anc1]==True and obs_flips[k]==True:

            if exclude_nodes is not None:
                for node in exclude_nodes:
                    rd  = node[0]
                    anc = node[1]
                    
                    if defects_matrix.data[k,rd,anc]==False:
                        Flag=True
                    else:
                        Flag=False
                        break
            else:
                Flag=True

            if Flag==True:
                single_count_w_L0 += 1/num_shots

        elif defects_matrix.data[k,rd1,anc1]==True and obs_flips[k]==False:
        
            if exclude_nodes is not None:
                for node in exclude_nodes:
                    rd  = node[0]
                    anc = node[1]
                    
                    if defects_matrix.data[k,rd,anc]==False:
                        Flag=True
                    else:
                        Flag=False
                        break

            else:
                Flag=True

            if Flag==True:
                single_count_wo_L0 += 1/num_shots        
   

    return single_count_wo_L0, single_count_w_L0


def get_3Point_prob(defects_matrix,num_shots,triplets):
    #triplets: list of tuples [(rd1,anc1),(rd2,anc2),(rd3,anc3)]

    p3 = 0

    rd1 = triplets[0][0]
    anc1  = triplets[0][1]

    rd2 = triplets[1][0]
    anc2  = triplets[1][1]

    rd3 = triplets[2][0]
    anc3  = triplets[2][1]

    # for k in range(num_shots):

    #     this_shot = defects_matrix.data[k,:,:]

    #     if this_shot[rd1,anc1] and this_shot[rd2,anc2] and this_shot[rd3,anc3]:

    #         p3+=1

    #Faster alternative method:
    p3 = np.logical_and(defects_matrix.data[:,rd1,anc1],defects_matrix.data[:,rd2,anc2])
    p3 = np.logical_and(p3,defects_matrix.data[:,rd3,anc3])
    p3 = sum(p3)
    # print("alt:",p3_alt)
    # if p3==sum(p3_alt):
    #     print("True!")
    # else:
    #     raise Exception("Alternative method is wrong")
              
    
    return p3/num_shots


def get_3Point_prob_exclude(defects_matrix,num_shots,include_pairs,exclude_pairs):
    #Include_pairs: list of tuples [(rd1,anc1),(rd2,anc2),...]
    #Exclude_pairs: list of tuples [(rd1,anc1),(rd2,anc2),...]

    p3 = 0
    # rds  = np.arange(np.shape(defects_matrix)[1])
    # ancs = np.arange(np.shape(defects_matrix)[2])

    for k in range(num_shots):

        this_shot = defects_matrix.data[k,:,:]
        Flag      = True

        if this_shot[include_pairs[0][0],include_pairs[0][1]]==True and  \
           this_shot[include_pairs[1][0],include_pairs[1][1]]==True and \
           this_shot[include_pairs[2][0],include_pairs[2][1]]==True:
            
            Flag=True
        else:
            Flag=False

        if Flag==False:
            continue

        if exclude_pairs is not None:
            # flags_all = []
            for other_pair in exclude_pairs:

                if this_shot[other_pair[0],other_pair[1]]==True: #This excludes any case where at least 1 node fires
                    Flag = False
                    break
                
                #We might want to exclude cases where all nodes in exclude pairs fire instead
                #The one below seems to be worse
            #     if this_shot[other_pair[0],other_pair[1]]==True:
            #         flags_all.append(True)
            #     else:
            #         flags_all.append(False)
            
            # if False in flags_all:
            #     Flag=True
            # else: #all True
            #     Flag=False

        if Flag==True: #True so we can increase

            p3 +=1

    return p3/num_shots

#TODO: Check which is more accurate... do we need to exlcude some nodes when we estimate some 3-point events?
#Seems like for different rds I need to exclude certain stuff.
#Maybe i need to exclude the following nodes:
#nodes between the detectors of the triplet
#nodes before?
#Can probably leave the nodes after, since there could be other errors that cause those events
#Still very very non-trivial to keep track of which errors flip what...
#Either we have to do it more brute-force, or we have to think of sth more clever.


def get_all_3_point_probs_new(defects_matrix,nodes):
    '''Consider only 3 point events where 
    node1 \in red
    node2 \in blue and 
    node3 \in green'''
    p_3cnts     = {}
    num_shots   = np.shape(defects_matrix)[0]
    num_rounds  = np.shape(defects_matrix)[1]
    num_ancilla = np.shape(defects_matrix)[2]

    #Get all 3 point events:
    #If all detectors are in same rd, then calculate w/o exclusion
    #If detectors are across rds, then calculate by excluding all other nodes

    all_nodes = [] #list of tuples

    for rd in range(num_rounds):
        for anc in range(num_ancilla):
            all_nodes.append((rd,anc))

    red_nodes   = []
    blue_nodes  = []
    green_nodes = []

    for node in nodes['r']:
        red_nodes.append(int(node[1:]))
    for node in nodes['g']:
        green_nodes.append(int(node[1:]))
    for node in nodes['b']:
        blue_nodes.append(int(node[1:]))

    print("red:",red_nodes)
    print("blue:",blue_nodes)
    print("green:",green_nodes)

    for m in range(len(all_nodes)):
        
        pair1   = all_nodes[m]
        rd1     = pair1[0]
        anc1    = pair1[1]
        indx1   = anc1 + num_ancilla * rd1

        if indx1 in red_nodes:
            
            for l in range(len(all_nodes)):

                pair2   = all_nodes[l]
                rd2     = pair2[0]
                anc2    = pair2[1]
                indx2   = anc2 + num_ancilla * rd2

                if indx2 in blue_nodes:
                    
                    for k in range(len(all_nodes)):

                        pair3   = all_nodes[k]
                        rd3     = pair3[0]
                        anc3    = pair3[1]
                        indx3   = anc3 + num_ancilla * rd3

                        if indx3 in green_nodes:
                            
                            triplet = [pair1,pair2,pair3]

                            inds              = np.sort([indx1,indx2,indx3])
                            INDX1,INDX2,INDX3 = inds
                            # inds              = np.sort([indx1,indx2,indx3])
                            # indx1,indx2,indx3 = inds

                            if rd1==rd2 and rd2==rd3: #All in same rd, calculate w/o exclusion
                                p_3cnts[("D"+str(INDX1),"D"+str(INDX2),"D"+str(INDX3))] = get_3Point_prob(defects_matrix,num_shots,triplet)
                            else:
                                exclude_pairs=[]

                                for pair in all_nodes:

                                    if pair!=pair1 and pair!=pair2 and pair!=pair3:
                                        exclude_pairs.append(pair)

                                p_3cnts[("D"+str(INDX1),"D"+str(INDX2),"D"+str(INDX3))] = get_3Point_prob_exclude(defects_matrix,num_shots,triplet,exclude_pairs)
                else:
                    continue


        else:
            continue

    return p_3cnts

def get_all_3_point_probs(defects_matrix):
    
    p_3cnts     = {}
    num_shots   = np.shape(defects_matrix)[0]
    num_rounds  = np.shape(defects_matrix)[1]
    num_ancilla = np.shape(defects_matrix)[2]

    #Get all 3 point events:
    #If all detectors are in same rd, then calculate w/o exclusion
    #If detectors are across rds, then calculate by excluding all other nodes
    
    all_nodes = [] #list of tuples

    for rd in range(num_rounds):
        for anc in range(num_ancilla):
            all_nodes.append((rd,anc))

    for m  in range(len(all_nodes)):
        
        pair1 = all_nodes[m]
        rd1   = pair1[0]
        anc1  = pair1[1]
        

        for l in range(m+1,len(all_nodes)):

            pair2 = all_nodes[l]
            rd2   = pair2[0]
            anc2  = pair2[1]

            for k in range(l+1,len(all_nodes)):

                pair3 = all_nodes[k]

                if pair1!=pair2 and pair2!=pair3 and pair1!=pair3:
                    
                    triplet = [pair1,pair2,pair3]
                    rd3     = pair3[0]
                    anc3    = pair3[1]

                    indx1   = anc1 + num_ancilla * rd1
                    indx2   = anc2 + num_ancilla * rd2
                    indx3   = anc3 + num_ancilla * rd3

                    inds              = np.sort([indx1,indx2,indx3])
                    indx1,indx2,indx3 = inds

                    
                    if rd1==rd2 and rd2==rd3: #All in same rd, calculate w/o exclusion
                        p_3cnts[("D"+str(indx1),"D"+str(indx2),"D"+str(indx3))] = get_3Point_prob(defects_matrix,num_shots,triplet)
                    else: #calculate by exlcusion

                        exclude_pairs=[]

                        for pair in all_nodes:

                            if pair!=pair1 and pair!=pair2 and pair!=pair3:
                                exclude_pairs.append(pair)

                        p_3cnts[("D"+str(indx1),"D"+str(indx2),"D"+str(indx3))] = get_3Point_prob_exclude(defects_matrix,num_shots,triplet,exclude_pairs)




    return p_3cnts




def get_all_single_cnts_w_and_wo_L0(defects_matrix,obs_flips):

    num_shots   = np.shape(defects_matrix)[0]
    num_rounds  = np.shape(defects_matrix)[1]
    num_ancilla = np.shape(defects_matrix)[2]
    
    p1_cnts   = {}
    all_nodes = []
    for rd in range(num_rounds):
        for anc in range(num_ancilla):
            all_nodes.append((rd,anc))
            

    for rd in range(num_rounds):
        for anc in range(num_ancilla):
            
            target_node   = (rd,anc)
            indx          = anc + num_ancilla * rd
            exclude_nodes = [x for x in all_nodes if x != target_node]

            cnts_wo_L0,cnts_w_L0 = extract_single_counts_w_and_wo_L0_excluding_nodes(num_shots,obs_flips,defects_matrix,target_node,exclude_nodes)
            
            p1_cnts[("D"+str(indx),"L0")] = cnts_w_L0
            p1_cnts[("D"+str(indx),"")]  = cnts_wo_L0


    return p1_cnts

#---------------- For bd probs --------------------------------------



def map_pairs_to_detectors(anc,rd,num_ancilla):
    
    name_of_det = "D" + str(anc + num_ancilla * rd)

    return name_of_det


def detector_name_to_rd_anc_name(name_of_det,num_ancilla,num_rounds):
    '''output the (round_indx,ancilla_indx) conversion'''
    Flag=False
    for rd in range(num_rounds+1):

        for anc in range(num_ancilla):

            indx = anc+num_ancilla*rd

            if int(name_of_det[1:])==indx:
                Flag=True
                pair=(rd,anc) 
                break

    if Flag==False:
        raise Exception("Conversion from detector name to (rd,anc) has failed.")

    return pair




#TODO: Consider for more rds and distances
def get_color_restricted_DEMs(pij_bulk,pij_bd,nodes):
    #nodes: a dictionary with keys 'r','b','g'
    #       and values the node names "Dj"
    #DEMs_1: a dictionary that includes all color-restricted lattices.

    dems_1 = {}
    colors = ['r','g','b']

    for color in colors:

        temp_dem    = stim.DetectorErrorModel()

        if color=='r':
            other_nodes = nodes['g']+nodes['b']
        elif color=='b':
            other_nodes = nodes['r']+nodes['g']
        else:
            other_nodes = nodes['r']+nodes['b']
        
        other_nodes = list(np.sort(other_nodes))

        for l in range(len(other_nodes)):

            for m in range(l+1,len(other_nodes)):

                v1 = other_nodes[l]
                v2 = other_nodes[m]

                INDX1 = int(v1[1:])
                INDX2 = int(v2[1:])

                indx1 = min([INDX1,INDX2])
                indx2 = max([INDX1,INDX2])

                p = pij_bulk[("D"+str(indx1),"D"+str(indx2))]
                if p>0:

                    temp_dem.append("error",p,
                                        [stim.target_relative_detector_id(indx1),
                                        stim.target_relative_detector_id(indx2)])


        #Add also the bd edges:
        for v in other_nodes:
            indx1 = int(v[1:])
            p = pij_bd[color][(v)]
            if p>0:
                temp_dem.append("error",p,
                                [stim.target_relative_detector_id(indx1)] )

     
        dems_1[color]=temp_dem



    return dems_1




def get_updated_probs_of_two_point_events(pij,p_3cnts,nodes):
    '''  pij: a dictionary with bulk-type probabilities
     p_3cnts: a dictionary that has the probabilities of each 3 point event
       nodes: a dictionary which with keys r,g,b and the respective nodes.'''
    
    all_nodes = nodes['r']+nodes['g']+nodes['b']

    #Create the pair D_i - D_j and check where this pair is included
    #in the 3-point cnts
    p2_updated = {}

    for m in range(len(all_nodes)):
                   
        v = all_nodes[m]

        for k in range(m+1,len(all_nodes)):
            w = all_nodes[k]

            if v!=w:
                
                numer = 0
                denom = 1
                #Loop through all the keys:
                
                for key in p_3cnts.keys():

                    if v in key and w in key:
                        
                        numer -= p_3cnts[key]
                        denom *= 1-2*p_3cnts[key]

                if (v,w) in pij.keys():
                # INDX1 = int(v[1:])
                # INDX2 = int(w[1:])
                # indx1 = min([INDX1,INDX2])
                # indx2 = max([INDX1,INDX2])
                    val = (pij[(v,w)]+numer)/denom
                    if val<0:
                        val = 0
                    
                    p2_updated[(v,w)]=val
                elif (w,v) in pij.keys():
                    val=(pij[(w,v)]+numer)/denom
                    if val<0:
                        val=0
                    p2_updated[(w,v)]=val

    return p2_updated








#------- OLD --------------------

#----- This is for red-restricted only. Should be deleted --------------
def get_all_bd_probs(pij_bulk,vi_mean):
    '''I am sure about 0,1,3,4 nodes (which are blue/green)
       Not sure about D2 and D5 which are the red nodes.'''
    pij_bd = {}

    #--------- D0 L1: ignore D0-D2 or D0-D5 [2 and 5 are in red lattice] --------------
    rd,anc             = 0,0
    dets_to_exclude    = [("D0","D1"),("D0","D3"),("D0","D4")] #Do not bother with D2 or D5
    pij_bd[("D0","L")] = extract_bd_prob(pij_bulk,vi_mean,anc,rd,dets_to_exclude)

    #------------ D1 L: ignore D1-D2 or D1-D5 -------------------------
    rd,anc           = 0,1
    dets_to_exclude    = [("D0","D1"),("D1","D3"),("D1","D4")] #Do not bother with D2 or D5
    pij_bd[("D1","L")] = extract_bd_prob(pij_bulk,vi_mean,anc,rd,dets_to_exclude)

    #----- D3 L: ignore D2-D3 or D2-D5 [red nodes] ---------------
    anc,rd             = 0,1
    dets_to_exclude    = [("D0","D3"),("D1","D3"),("D3","D4")] #Do not bother with D2 or D5
    pij_bd[("D3","L")] = extract_bd_prob(pij_bulk,vi_mean,anc,rd,dets_to_exclude)

    #------- D4 L: ignore D2-D4 or D4-D5 [red nodes]  -------------
    anc,rd             = 1,1
    dets_to_exclude    = [("D1","D4"),("D3","D4"),("D0","D4")] #Do not bother with D2 or D5
    pij_bd[("D4","L")] = extract_bd_prob(pij_bulk,vi_mean,anc,rd,dets_to_exclude)

    #Can we also find D5? Is this a coincidence?
    anc,rd             = 2,1
    dets_to_exclude    = [("D1","D5"),("D3","D5"),("D2","D5")] #Do not bother with D2 or D5
    pij_bd[("D5","")] = extract_bd_prob(pij_bulk,vi_mean,anc,rd,dets_to_exclude)

    anc,rd             = 1,0
    dets_to_exclude    = [("D0","D1"),("D1","D3")] #Do not bother with D2 or D5
    pij_bd[("D1","")] = extract_bd_prob(pij_bulk,vi_mean,anc,rd,dets_to_exclude)

    return pij_bd




