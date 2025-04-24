import xarray as xr
import stim
from utilities.defects_matrix_utils import *

def get_qubit_names(obj):
    '''Get all qubit names for the circuit, including their type (data or ancilla),
       and the color (blue, red, green)

    Input:
        obj: the color code object
    Output:
        qubit_names: dictionary of the qubit names with keys: data,anc,anc_X,anc_Z, anc_red,anc_blue,anc_green
        '''

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


def get_inds_for_syndrome_projection(circuit: stim.Circuit):
    '''Get the indices for which data qubits need to be combined to form the 
       stabilizer projection of the last QEC round.
       
       Input:
            circuit: the stim Circuit of the color code memory
       Output:
            all_inds: list of lists containing the indices per Z-type stabilizer.
       '''
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



#TODO: Make the part of syndrome projection faster.
def get_defects(circuit: stim.Circuit, num_rounds: int, num_shots: int,obj):
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
