import stim
import numpy as np


#with __get_item__(0) we get the 1st element in stim circuit [this returns circuit instruction]
#with __get_item__(slice(start,end,step)) we get some elements of the stim circuit [this returns circuit]

def get_str(circuit):

    return circuit.__str__()


def add_two_qubit_depol_error_procedure(new_circuit: stim.Circuit, instruction: stim.CircuitInstruction, p_2Q_Gate: float):
    '''Main tool to DEPOLARIZE2 error after a CNOT gate'''
    
    Qubits = instruction.targets_copy()
    L      = len(Qubits)

    if L==2:
        #We provided the CX using 2 qubits of the form [ctrl, target]
        new_circuit.append(instruction)
        new_circuit.append("DEPOLARIZE2",Qubits,p_2Q_Gate)
    
    elif L>2:

        #We provided the CX using multiple targets of the form [ctrl1, target1, ctrl2, target2 ,...]
        k=0
        while k<L:

            Q1 = Qubits[k]
            Q2 = Qubits[k+1]
            new_circuit.append("CX",[Q1,Q2])
            new_circuit.append("DEPOLARIZE2",[Q1,Q2],p_2Q_Gate)

            k +=2 

            if k==L-1:
                break



    return new_circuit


def add_single_qubit_depol_error_procedure(new_circuit: stim.Circuit, instruction: stim.CircuitInstruction, p_2Q_Gate: float):
    '''Main tool to DEPOLARIZE1 error after a CNOT gate'''
    
    Qubits = instruction.targets_copy()
    L      = len(Qubits)

    if L==2:
        #We provided the CX using 2 qubits of the form [ctrl, target]
        new_circuit.append(instruction)
        new_circuit.append("DEPOLARIZE1",Qubits,p_2Q_Gate)
    
    elif L>2:

        #We provided the CX using multiple targets of the form [ctrl1, target1, ctrl2, target2 ,...]
        k=0
        while k<L:

            Q1 = Qubits[k]
            Q2 = Qubits[k+1]
            new_circuit.append("CX",[Q1,Q2])
            new_circuit.append("DEPOLARIZE1",[Q1,Q2],p_2Q_Gate)

            k +=2 

            if k==L-1:
                break



    return new_circuit


def append_two_qubit_depol_error_after_2Q_Gates(circuit:stim.Circuit, p_2Q_Gate: float):
    '''Add 2 qubit depol error on an ideal circuit, after every CNOT. This works also in the case where the
    circuit has a repeat block'''
    new_circuit = stim.Circuit()

    for instruction in circuit:

        instruction_str = get_str(instruction)
        
        

        if instruction_str[0]=="C" and instruction_str[1]=="X":

            new_circuit = add_two_qubit_depol_error_procedure(new_circuit, instruction, p_2Q_Gate)

        else:

            #Is it a repeat block?
            if instruction.name=="REPEAT":

                num_rounds = instruction.repeat_count

                temp_circuit = stim.Circuit()
                body         = instruction.body_copy()
                
                for new_instruction in body:

                    new_instruction_str = get_str(new_instruction)
                    
                    if new_instruction_str[0]=="C" and new_instruction_str[1]=="X":

                       temp_circuit = add_two_qubit_depol_error_procedure(temp_circuit, new_instruction, p_2Q_Gate)

                    else:

                        temp_circuit.append(new_instruction)

                new_circuit.append_from_stim_program_text("REPEAT " + str(num_rounds) +"{" + get_str(temp_circuit) + "\n}")
                        

            else:

                new_circuit.append(instruction)


    return new_circuit


def append_input_error(circuit: stim.Circuit, pdepol: float, ErrorType: str):
    #To append in the beginning of the circuit and outside of a repeat block,
    #first check if there is a reset. If there is a reset operation, append the depolarizing error rate
    #right after the reset. Apply a similar idea for anything that is inside a repeat block.

    new_circuit = stim.Circuit()
    all_qubits  = np.arange(1,circuit.num_qubits).astype(int) #drop the 0-th qubit, if counting starts from 1
    #all_qubits = np.arange(9,circuit.num_qubits).astype(int)
    if ErrorType=="X":
        str_error = "X_ERROR"
    elif ErrorType=="Z":
        str_error = "Z_ERROR"
    elif ErrorType=="DEPOLARIZING":
        str_error = "DEPOLARIZE1"
    else:
        raise Exception("Can only choose X, Z, or DEPOLARIZING.")

    #Find where the reset instructions end:
    cnt=0
    for instruction in circuit:

        if instruction.name=="R" or instruction.name=="RX":
            cnt=cnt+1
            continue
        else:
            break #we terminated with the resets (hopefully the user put all of them in the beginning before any other gates)
    
    cnt    -= 1
    cnt_new = 0

    #Add the resets in the new circuit
    for instruction in circuit:

        new_circuit.append(instruction)
        
        if cnt_new==cnt:
            
            break
        cnt_new +=1
    
    new_circuit.append(str_error,all_qubits,pdepol)


    CNT_start = 0
    
    for instruction in circuit:

        if CNT_start > cnt_new:

            if instruction.name=="REPEAT":

                num_rounds   = instruction.repeat_count
                temp_circuit = stim.Circuit()
                body         = instruction.body_copy()

                temp_circuit.append(str_error,all_qubits,pdepol)
                for new_instruction in body:

                    temp_circuit.append(new_instruction)

                new_circuit.append_from_stim_program_text("REPEAT " + str(num_rounds) +"{" + get_str(temp_circuit) + "\n}")
            
            else:


                new_circuit.append(instruction)

        CNT_start +=1    

    return new_circuit


def append_single_qubit_depol_error_after_2Q_Gates(circuit:stim.Circuit, p_2Q_Gate: float):
    '''Add 2 qubit depol error on an ideal circuit, after every CNOT. This works also in the case where the
    circuit has a repeat block'''
    new_circuit = stim.Circuit()

    for instruction in circuit:

        instruction_str = get_str(instruction)
        
        

        if instruction_str[0]=="C" and instruction_str[1]=="X":

            new_circuit = add_single_qubit_depol_error_procedure(new_circuit, instruction, p_2Q_Gate)

        else:

            #Is it a repeat block?
            if instruction.name=="REPEAT":

                num_rounds = instruction.repeat_count

                temp_circuit = stim.Circuit()
                body         = instruction.body_copy()
                
                for new_instruction in body:

                    new_instruction_str = get_str(new_instruction)
                    
                    if new_instruction_str[0]=="C" and new_instruction_str[1]=="X":

                       temp_circuit = add_single_qubit_depol_error_procedure(temp_circuit, new_instruction, p_2Q_Gate)

                    else:

                        temp_circuit.append(new_instruction)

                new_circuit.append_from_stim_program_text("REPEAT " + str(num_rounds) +"{" + get_str(temp_circuit) + "\n}")
                        

            else:

                new_circuit.append(instruction)


    return new_circuit




def append_single_qubit_depol_error_after_gates(circuit: stim.Circuit, p_1Q_Gate: float):
    '''Append a single qubit depolarizing error after every H gate in an error-free circuit.
    Works also if the circuit has repeat blocks.'''
    new_circuit = stim.Circuit()

    for instruction in circuit:

        instruction_str = get_str(instruction)

        if instruction_str[0]=="H":

            Qubits = instruction.targets_copy()

            new_circuit.append("H",Qubits)
            new_circuit.append("DEPOLARIZE1",Qubits,p_1Q_Gate)


        else:
             
            if instruction.name=="REPEAT":
                
                num_rounds = instruction.repeat_count

                body = instruction.body_copy()
                temp_circuit = stim.Circuit()

                for new_instruction in body:

                    new_instruction_str = get_str(new_instruction)

                    if new_instruction_str[0]=="H":

                        Qubits = new_instruction.targets_copy()

                        temp_circuit.append("H",Qubits)
                        temp_circuit.append("DEPOLARIZE1",Qubits,p_1Q_Gate)
                    else:
                        temp_circuit.append(new_instruction)


                new_circuit.append_from_stim_program_text("REPEAT " + str(num_rounds) +"{" + get_str(temp_circuit) + "\n}")

            else:
                
                new_circuit.append(instruction)



    return new_circuit


def append_single_qubit_Z_error_after_gates(circuit: stim.Circuit, p_1Q_Gate: float):
    '''Append a single qubit depolarizing error after every H gate in an error-free circuit.
    Works also if the circuit has repeat blocks.'''
    new_circuit = stim.Circuit()

    for instruction in circuit:

        instruction_str = get_str(instruction)

        if instruction_str[0]=="H":

            Qubits = instruction.targets_copy()

            new_circuit.append("H",Qubits)
            new_circuit.append("Z_ERROR",Qubits,p_1Q_Gate)


        else:
             
            if instruction.name=="REPEAT":
                
                num_rounds = instruction.repeat_count

                body = instruction.body_copy()
                temp_circuit = stim.Circuit()

                for new_instruction in body:

                    new_instruction_str = get_str(new_instruction)

                    if new_instruction_str[0]=="H":

                        Qubits = new_instruction.targets_copy()

                        temp_circuit.append("H",Qubits)
                        temp_circuit.append("Z_ERROR",Qubits,p_1Q_Gate)
                    else:
                        temp_circuit.append(new_instruction)


                new_circuit.append_from_stim_program_text("REPEAT " + str(num_rounds) +"{" + get_str(temp_circuit) + "\n}")

            else:
                
                new_circuit.append(instruction)



    return new_circuit


def append_single_qubit_X_error_after_gates(circuit: stim.Circuit, p_1Q_Gate: float):
    '''Append a single qubit depolarizing error after every H gate in an error-free circuit.
    Works also if the circuit has repeat blocks.'''
    new_circuit = stim.Circuit()

    for instruction in circuit:

        instruction_str = get_str(instruction)

        if instruction_str[0]=="H":

            Qubits = instruction.targets_copy()

            new_circuit.append("H",Qubits)
            new_circuit.append("X_ERROR",Qubits,p_1Q_Gate)


        else:
             
            if instruction.name=="REPEAT":
                
                num_rounds = instruction.repeat_count

                body = instruction.body_copy()
                temp_circuit = stim.Circuit()

                for new_instruction in body:

                    new_instruction_str = get_str(new_instruction)

                    if new_instruction_str[0]=="H":

                        Qubits = new_instruction.targets_copy()

                        temp_circuit.append("H",Qubits)
                        temp_circuit.append("X_ERROR",Qubits,p_1Q_Gate)
                    else:
                        temp_circuit.append(new_instruction)


                new_circuit.append_from_stim_program_text("REPEAT " + str(num_rounds) +"{" + get_str(temp_circuit) + "\n}")

            else:
                
                new_circuit.append(instruction)



    return new_circuit



def get_ranges_for_det_removal(circuit):

    str_circ = get_str(circuit)
    L       = len(str_circ)-2
    kstart  = 0
    pair_k = []

    while True:
        FLAG1 =False
        for k in range(kstart,L):

            if str_circ[k]=="D" and str_circ[k+1]=="E" and str_circ[k+2]=="T":
                FLAG1 = True
                for l in range(k+2,L+2):

                    if str_circ[l]=="\n":
                        
                        pair_k.append((k,l))
                        kstart = l+1

                        break
        
        if FLAG1==False:
            break


    return pair_k


def get_ranges_for_obs_removal(circuit):

    str_circuit = get_str(circuit)
    L           = len(str_circuit)-1
    kstart      = 0
    pair_k      = []
    while True: #Repeat for all observables we find
        Flag = False
        for k in range(kstart,L):

            if str_circuit[k]=="O" and str_circuit[k+1]=="B": #Obs_include

                Flag = True
                for l in range(k+1,L+1):
                    if str_circuit[l]=="\n" or l==len(str_circuit)-1:

                        pair_k.append((k,l+1))
                        kstart = l+1
                        break
        if Flag==False:
            break
    

    return pair_k



def remove_detector_annotations(circuit):
    '''
    If we have the ranges for removal ((a_1,a_2),(a_3,a_4),...,(a_{n-1},a_n)) then we need to do the following:

    str[:a_1] + str[a_2:a_3] + str[a_3:a_4] + ... str[a_{n-1}:a_n] + str[a_n:]

    -So, if we have only one range then we do:
        str[:a_1] + str[a_2:]
    -If we have 2 ranges then we do:
        str[:a_1] + str[a_2:a_3] + str[a_3:]
    
    So common thing for all of the ranges:
    do str[:a_1] + str[a_n:] and the ranges in between handle them with a loop.    

    '''

    pair_k   = get_ranges_for_det_removal(circuit)
    if len(pair_k)>0:
        
        str_circ = get_str(circuit)
        pair_0   = pair_k[0]

        new_str  = str_circ[0:pair_0[0]]  #str[:a_1]
        #+ str_circ[pair_0[1]:pair_1[0]]

        if len(pair_k)>=2:
            for k in range(len(pair_k)-1):
                pair1    = pair_k[k]
                pair2    = pair_k[k+1]
                new_str += str_circ[pair1[1]:pair2[0]] 

        
        pair = pair_k[-1] 
        new_str+= str_circ[pair[1]:] #str[a_n:]
        
        return stim.Circuit(new_str)
    else:
        return circuit


def remove_obs_annotations(circuit):
    
    pair_k   = get_ranges_for_obs_removal(circuit)
    
    if len(pair_k)>0:
        
        str_circ = get_str(circuit)
        pair_0   = pair_k[0]
        
        new_str  = str_circ[0:pair_0[0]] #str[:a_1]
        #+ str_circ[pair_0[1]:pair_1[0]]

        if len(pair_k)>=2:
            for k in range(len(pair_k)-1):
                pair1    = pair_k[k]
                pair2    = pair_k[k+1]
                new_str += str_circ[pair1[1]:pair2[0]] 

        
        pair = pair_k[-1] 
        new_str+= str_circ[pair[1]:] #str[a_n:]

        return stim.Circuit(new_str)
    else: 
        return circuit






#TODO: Write a function that get the qubit names and coordinates, and finds which are the data qubits
#and which are the ancilla qubits


def bit_flip_error_before_Meas(circuit,p_before_meas):
    '''General function to add bit-flip error before a measurement'''

    str_circ = get_str(circuit)
    new_str  = str_circ
    L        = 2*len(new_str)

    for k in range(L):

        
        if str_circ[k]=="M" and str_circ[k+1]=="R": #Measurement followed by reset

            new_str = str_circ[:k+2] + """(""" + str(p_before_meas) + """)"""  + str_circ[k+2:]
            

        elif str_circ[k]=="M" and str_circ[k+1]!="X": #not the last MX gates on data qubits

            new_str = str_circ[:k+1] + """(""" + str(p_before_meas) + """)"""  + str_circ[k+1:]
            
        
        str_circ = new_str
        if k==len(new_str)-1:
            break
        
        

    return stim.Circuit(new_str)