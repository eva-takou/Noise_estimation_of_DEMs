import stim
from sims.surface_code_bare_ancilla.surface_code_lattice import *
import numpy as np
from utilities.circuit_control import *

#TODO: Make this more readable, cleanup, 

def construct_Z_type_check_coordinates_for_planar_SC(data_edge, HZ):
    '''
    The Z-type stabilizers in this surface code construction are loops.
    We start from the inner loops, and then move to the loops that touch the boundaries.
    '''

    #HZ = surface_code_loop_stabs(L) #In each row, we have which qubit participates. So we want to put the detector coordinate 
                                    #in the middle of the loop.

    num_rows         = np.shape(HZ)[0]
    qubits_per_check = []

    for k in range(num_rows):
        temp = np.nonzero(HZ[k,:])[1]
        qubits_per_check.append(temp)  
    
    check_coords=[]
    for check_qubits in qubits_per_check:
        
        coords = []
        for k in check_qubits:
           
            coords.append(data_edge[k])
       
        #find the distinct y coords and the distinct x-coords
        y_coords = []
        x_coords = []
        for coord in coords:

            if coord[0][1] not in y_coords:
                y_coords.append(coord[0][1])
            if coord[1][1] not in y_coords:
                y_coords.append(coord[1][1])
            if coord[0][0] not in x_coords:
                x_coords.append(coord[0][0])
            if coord[1][0] not in x_coords:
                x_coords.append(coord[1][0])

        x = np.sum(x_coords)/2
        y = np.sum(y_coords)/2
        check_coords.append([x,y])

    return check_coords




def construct_repeating_block(X_stab_Check, num_ancilla, num_rounds: int, include_both: bool):
    #This is fine if we set all detectors (both on X and Z stabs)


    #prefactor       = 2  #THIS IS CORRECT WHEN WE HAVE ONLY 1 TYPE OF CHECKS
    if include_both==True:
        prefactor = 3
    elif include_both==False:
        prefactor = 2

    new_num_rounds  = num_rounds-1


    str_round = get_str(X_stab_Check)

    #We need to modify the above str and now put besides rec[-1] other previous measurements:

    kstart = 0
    cnt    = 0
    cnt_L = 0
    while True:

        Flag=False
        

        for k in range(kstart,len(str_round)): 

            if str_round[k]=="r" and str_round[k+1]=="e" and str_round[k+2]=="c":

                Flag = True

                for m in range(k+3,len(str_round)):

                    if str_round[m]=="]":

                        #num_det = -prefactor*len(anc_qubits) + cnt
                        #num_det = -prefactor*num_ancilla + cnt  #THIS IS CORRECT COUNTER WHEN WE HAVE ONLY 1 TYPE OF CHECKS

                        num_det = -prefactor*(num_ancilla) + cnt
                        
                        cnt_L +=1
                        if cnt_L == num_ancilla: #restart
                            cnt=-1

                        str_round = str_round[:m+1] + " rec[" + str(num_det) + "]"  + str_round[m+1:]
                        kstart    =  m + len(" rec[") + len(str(num_det)) + len("]")
                        cnt      +=1
                        break
                break

        if Flag==False:
            break

    str_round       = """REPEAT """ + str(new_num_rounds) + """{ """ + str_round + """\n }"""
    repeating_block = stim.Circuit(str_round)

    

    return repeating_block



def find_qubits_per_X_stab_meas(L,include_both=True):

    data_qubits = np.arange(1,L**2+(L-1)**2 +1).astype(int)
    n_data      = len(data_qubits)
    
    HX           = surface_code_star_stabs(L)
    n_anc        = np.shape(HX)[0]

    past_indices = []

    for k in range(n_anc):

        locs         = np.nonzero(HX[k,:])[1]
        locs         = locs + 1
        temp_indices = []
        
        for loc in locs:

            temp_indices.append(stim.target_rec(-n_data+loc-1))
        if include_both:
            temp_indices.append(stim.target_rec(-n_data - 2*n_anc  +k))    #This is the previous detector by 1 rd
        else:
            temp_indices.append(stim.target_rec(-n_data - n_anc  +k)) #This is the previous detector by 1 rd
        #temp_indices.append(stim.target_rec(-n_data - 2*n_anc  +k)) #This is the previous detector by 2 rd

        past_indices.append(temp_indices)
        

        
    return past_indices

#Need to shift sth here
def find_qubits_per_Z_stab_meas(L):

    data_qubits = np.arange(1,L**2+(L-1)**2 +1).astype(int)
    n_data      = len(data_qubits)
    
    HZ           = surface_code_loop_stabs(L)
    n_anc        = np.shape(HZ)[0]
    past_indices = []

    for k in range(n_anc):

        locs         = np.nonzero(HZ[k,:])[1]
        locs         = locs + 1
        temp_indices = []
        
        for loc in locs:

            temp_indices.append(stim.target_rec(-n_data+loc-1))
        
        temp_indices.append(stim.target_rec(-n_data - n_anc  +k))    #This is the previous detector by 1 rd
        #temp_indices.append(stim.target_rec(-n_data - 2*n_anc  +k)) #This is the previous detector by 2 rd

        past_indices.append(temp_indices)
        

    return past_indices



def assign_coordinates(circuit, data_edge, meas_nodes, HZ, anc_type: str):
    
    data_qubit_coords = []
    anc_coords_X      = []
    cnt               = 1

    for edge in data_edge:  #Assign the data qubit coordinates in the circuit
        
        P1 = edge[0]
        P2 = edge[1]
        
        x_coord = (P1[0]+P2[0])/2
        y_coord = (P1[1]+P2[1])/2
        circuit.append("QUBIT_COORDS",cnt,[x_coord,y_coord])
        data_qubit_coords.append([x_coord,y_coord])
        
        cnt+=1
   
    #Coordinates of X-check ancilla qubits:
    if anc_type=='X':
        
        for k in range(len(meas_nodes)):

            node = meas_nodes[k]          #The first 1/2 coordinates are correct, and they correspond to the star-check coordinates. (The last half coordinates are the virtual detectors which we do not need)
            circuit.append("QUBIT_COORDS",cnt,node)
            anc_coords_X.append(node)
            cnt+=1

        return circuit, data_qubit_coords, anc_coords_X

    elif anc_type=='Z':

        #Coordinates of Z-check ancilla qubits:
        anc_coords_Z = construct_Z_type_check_coordinates_for_planar_SC(data_edge, HZ)

        for k in range(len(anc_coords_Z)):
            
            circuit.append("QUBIT_COORDS",cnt,anc_coords_Z[k])            
            cnt +=1
        
        return circuit, data_qubit_coords, anc_coords_Z

    else:

        for k in range(len(meas_nodes)):

            node = meas_nodes[k]          #The first 1/2 coordinates are correct, and they correspond to the star-check coordinates. (The last half coordinates are the virtual detectors which we do not need)
            circuit.append("QUBIT_COORDS",cnt,node)
            anc_coords_X.append(node)
            cnt+=1
        
        #Coordinates of Z-check ancilla qubits:
        anc_coords_Z = construct_Z_type_check_coordinates_for_planar_SC(data_edge, HZ)

        for k in range(len(anc_coords_Z)):
            
            circuit.append("QUBIT_COORDS",cnt,anc_coords_Z[k])
            
            cnt +=1        


        return circuit, data_qubit_coords, anc_coords_X, anc_coords_Z
        



    return


def input_depol_channel(qubits,p_depol_input):

    depol_channel = stim.Circuit()

    depol_channel.append("DEPOLARIZE1",qubits,p_depol_input)

    return depol_channel



def X_stab_Check_surface_code(HX, anc_qubits, anc_coords):
    '''Construct the circuit for the X-checks in a planar (unrotated) surface code. The X-checks form star stabilizers.
    An X-check is performed by applying Hadamard on the ancilla, then perform CNOTs with control the ancilla and targets
    the qubits it checks, and then Hadamard again on the ancilla and measurement in the computational basis.
    '''

    n_anc        = np.shape(HX)[0]
    X_stab_Check = stim.Circuit()

    cnt = 0

    for k in anc_qubits:

        qubits_to_check = np.nonzero(HX[cnt,:])[1]
        qubits_to_check = [x+1 for x in qubits_to_check]
        
        X_stab_Check.append("H",k)

        for Q in qubits_to_check:

            X_stab_Check.append("CX",[k,Q])

            
        X_stab_Check.append("H",k)    
        X_stab_Check.append("TICK",[])
        
        cnt+=1
    
    X_stab_Check.append("MR",anc_qubits)
    
    #Put all the detectors here    
    for k in range(n_anc):
        xy_coords = anc_coords[k]
        X_stab_Check.append("DETECTOR",[stim.target_rec(-n_anc+k)],[xy_coords[0],xy_coords[1],0]) #stim.target_rec(-2*len(anc_qubits)+k)
        
    X_stab_Check.append("SHIFT_COORDS",[],[0,0,1])

    return X_stab_Check



def sample_from_lognorm(p,std):
    
    error_prob = np.random.lognormal(mean=p,sigma=std)
    error_prob = np.log(error_prob)

    if error_prob<0:
        error_prob=sample_from_lognorm(p,std)

    return error_prob

def sample_from_lognorm_data_q(p,std,len_data_qubits):

    np.random.seed(0)
    probs = []

    for k in range(len_data_qubits):
        probs.append(sample_from_lognorm(p,std))

    return probs

def sample_from_lognorm_anc_q(p,std,len_anc_qubits):

    np.random.seed(10)

    probs = []

    for k in range(len_anc_qubits):
        probs.append(sample_from_lognorm(p,std))

    return probs


def planar_surface_code_circuit_X_memory(L: int, num_rounds: int, p_depol_input: float, p_depol_anc:float, std: float):
    #For X-memory, if we check the X-stabs, we need to use the Z-logical operator and not the X-logical operator 
    #as an observable (this is because it is conjugated by Hadamards). 
    #The surface code structure is created as follows (example for d=3):
    #         A3   Q7   A6
    #Q10 ------x--------x-------- Q13
    #          |        |
    #        Q2|   Q6   | Q4
    #Q9  ------x--------x-------- Q12
    #          | A2     | A5
    #        Q1|        | Q3
    #Q8  ------x--------x--------- Q11
    #         A1   Q5  A4 
    #
    #Similarly, the loop checks are ordered as follows (example for d=3):
    #              Q7   
    # Q10-------x--------x-------- Q13
    #     z    |   z    |     z
    #    A4  Q2|   A2   | Q4  A6
    #          |   Q6   | 
    # Q9-------x--------x-------- Q12
    #     z    |    z   |     z
    #     A3 Q1|   A1   | Q3  A5
    #          |        |      
    #Q8 -------x--------x--------- Q11
    #              Q5     

    #Edges are the data qubits. "x"'s are the X-type (star) checks that check for Z-type errors.
    #"o"'s are the z-type checks.
    #The data qubit coordinate enumerations starts with the vertical edges, and then we move to the inner horizontal edges, and finally to the outer horizontal edges
    #The ancilla qubit enumeration for star checks starts from the bottom left, and then we move along the vertical direction, and then we finaly move to the next column to 
    #continue the enumeration (starting from bottom and going to the top).
    #The ancilla qubit enumeration for loop checks starts from the middle blocks moves to the right, once we consider all weight-4 stabs.
    #Then we move to the left boundary and finally to the right boundary.

    circuit           = stim.Circuit()
    
    HX                = surface_code_star_stabs(L)   #Checks the X-parity of the qubits (detect Z-type errors)
    HZ                = surface_code_loop_stabs(L)   #Checks the Z-parity of the qubits (detect X-type errors)
    XL                = surface_code_z_logical(L)    #Z-Logical operator
    
    n_anc             = np.shape(HX)[0]  #Number of X-type ancilla qubits
    n_data            = np.shape(HX)[1]  #Numner of data qubits  

    data_qubits       = np.arange(1 , n_data + 1).astype(int)                             #1 till L^2+(L-1)^2
    X_anc_qubits      = np.arange(n_data + 1 , n_anc + (n_data+1)).astype(int)            #X-ancilla qubits

    data_edge,all_meas_nodes, meas_nodes, bd_nodes = surface_code_planar(L)
    anc_type = 'X'

    circuit, data_qubit_coords, anc_coords_X = assign_coordinates(circuit, data_edge, meas_nodes, HZ,anc_type)

    
    circuit.append("RX",data_qubits)               #X-memory experiment (preserve the |+>_L state)
    circuit.append("R",X_anc_qubits)  
    circuit.append("TICK",[])

    depol_channel = stim.Circuit()

    probs_dataQ = sample_from_lognorm_data_q(p_depol_input,std,len(data_qubits))
    cnt         = 0
    for qubit in data_qubits:

        # error_prob = sample_from_lognorm(p_depol_input,std)

        depol_channel.append("DEPOLARIZE1",qubit,probs_dataQ[cnt])
        cnt+=1

    probs_ancQ = sample_from_lognorm_anc_q(p_depol_anc,std,len(X_anc_qubits))
    cnt        = 0

    for qubit in X_anc_qubits:

        # error_prob = sample_from_lognorm(p_depol_anc,std)

        depol_channel.append("DEPOLARIZE1",qubit,probs_ancQ[cnt])
        cnt+=1

    # print("errors for data_q:",probs_dataQ)
    # print("errors for anc_q:",probs_ancQ)


    # total_depol_channel = input_depol_channel(data_qubits,p_depol_input) + input_depol_channel(X_anc_qubits,p_depol_anc) 
    
    X_stab_Check        = X_stab_Check_surface_code(HX, X_anc_qubits, anc_coords_X)

    block = depol_channel + X_stab_Check 

    circuit += block

    circuit.append("TICK",[])

    include_both = False

    #--------- Repeat the block by sampling again from lognormal distribution ---------------------------
    if num_rounds-1>0:
        # depol_channel = stim.Circuit()

        # probs_dataQ = sample_from_lognorm_data_q(p_depol_input,std,len(data_qubits))
        # cnt         = 0
        # for qubit in data_qubits:

        #     # error_prob = sample_from_lognorm(p_depol_input,std)

        #     depol_channel.append("X_ERROR",qubit,probs_dataQ[cnt])
        #     cnt+=1

        # probs_ancQ = sample_from_lognorm_anc_q(p_depol_anc,std,len(X_anc_qubits))
        # cnt        = 0

        # for qubit in X_anc_qubits:

        #     # error_prob = sample_from_lognorm(p_depol_anc,std)

        #     depol_channel.append("X_ERROR",qubit,probs_ancQ[cnt])
        #     cnt+=1

        # circuit += depol_channel + X_stab_Check
        
        circuit+= construct_repeating_block(depol_channel + X_stab_Check ,len(X_anc_qubits),num_rounds,include_both)

        # circuit+= construct_repeating_block(depol_channel + X_stab_Check ,len(X_anc_qubits),num_rounds,include_both)





    # if num_rounds-1>0:
    #     repeating_block = construct_repeating_block(block, n_anc, num_rounds,include_both)
    #     circuit+= repeating_block


    #Finally, measure all the data qubits, and project the final measurement results on the stabilizers
    #Check if we should project 2 rounds before, or a single round.


    circuit.append("TICK",[])
    circuit.append("MX",data_qubits) 

    #The # of detectors we need is equal to the # of stabilizers

    target_recs = find_qubits_per_X_stab_meas(L,include_both)

    circuit.append("SHIFT_COORDS",[],[0,0,1])
    #These are detectors on last qubit measurements
    cnt=0
    for rec in target_recs:
        xy_coords = anc_coords_X[cnt]
        circuit.append("DETECTOR",rec,[xy_coords[0],xy_coords[1],0])
        cnt+=1


    #Add the observable too:

    locs = np.nonzero(XL)[1]
    locs = locs           #Indices of data qubits in the logical observable

    #Count from the end -1 to see when we encounter each element of locs. This tells us how
    #many steps in the past we need to go to define the target_recs of the logical observable.
    locs = locs-n_data

    past_indices=[]
    for k in locs:
        past_indices.append(stim.target_rec(k))

    circuit.append("OBSERVABLE_INCLUDE",past_indices,0)

    return circuit








