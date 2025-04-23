import stim
import numpy as np
from utilities.circuit_control import get_str 



#----- Steane type syndrome extraction for Z-basis memory ---------

def construct_logical_state_on_ancilla(distance: int, logical_state: str,after_CNOT_depol_type: str, p_depol_after:float):
    '''Construct the |0>_L or |+>_L on the ancilla: it should hold that the ancilla qubits start from |0>.
    This functionality is used for the Steane-type syndrome extraction.'''
    
    anc_prep_circuit = stim.Circuit()
    anc_qubits       = (np.arange(distance)+distance).tolist()    
    flag_qubit       = [anc_qubits[-1]+1] 

    if logical_state=='0':

        Q0 = anc_qubits[0]
        anc_prep_circuit.append_operation("H",Q0)

        for k in range(len(anc_qubits)-1):
            Q1 = anc_qubits[k]
            Q2 = anc_qubits[k+1]
            qq = [Q1] + [Q2]
            anc_prep_circuit.append_operation("CX",qq)
            anc_prep_circuit.append(after_CNOT_depol_type,qq,p_depol_after)
        
        # if distance>3:
        #     Qpair = [anc_qubits[1]]+flag_qubit
        #     anc_prep_circuit.append_operation("CX",Qpair)
            #anc_prep_circuit.append(after_CNOT_depol_type,Qpair,p_depol_after)

    elif logical_state=='+':
        
        anc_prep_circuit.append_operation("H",anc_qubits[1:])
        if distance>3:
            anc_prep_circuit.append_operation("H",flag_qubit)

        for k in range(len(anc_qubits)-1):
            Q1 = anc_qubits[k+1]
            Q2 = anc_qubits[k]
            qq = [Q1]+[Q2]
            anc_prep_circuit.append_operation("CX",qq)
        
        if distance>3:
            Qpair = flag_qubit + [anc_qubits[1]]
            anc_prep_circuit.append_operation("CX",Qpair)
            anc_prep_circuit.append(after_CNOT_depol_type,Qpair,p_depol_after)

    else:
        raise Exception('Can only prepare |0>_L or |+>_L on the ancilla qubits.')
    

    return anc_prep_circuit



def entangle_data_with_ancilla(data_qubits: list,anc_qubits: list, logical_state: str, after_CNOT_depol_type: str, p_depol_after:float):

    circuit = stim.Circuit()

    if logical_state=='0':
        
        for k in range(len(data_qubits)):
            
            qq = [data_qubits[k]]+[anc_qubits[k]]
            circuit.append_operation("CX",qq)
            
            #if k==0:
                #circuit.append(after_CNOT_depol_type,qq,p_depol_after)
            
            #circuit.append(after_CNOT_depol_type,qq,p_depol_after)
            circuit.append(after_CNOT_depol_type,qq,p_depol_after)

    elif logical_state == '+':

        for k in range(len(data_qubits)):
            
            qq = [anc_qubits[k]]+ [data_qubits[k]]
            circuit.append_operation("CX",qq)
            
            circuit.append(after_CNOT_depol_type,qq,p_depol_after)

    else: 
        raise Exception('Can only have |0>_L or |+>_L.')


    return circuit



def measure_logical_ancilla(anc_qubits:list,logical_state: str, Reset: bool):
    
    circuit = stim.Circuit()

    if logical_state=='+':

        circuit.append("H",anc_qubits)

    if logical_state!='0' and logical_state!='+':
        raise Exception('Can only have 0 or + logical state.')


    if Reset==True:
        circuit.append("MR",anc_qubits)

    else:
        circuit.append("M",anc_qubits)
    

    return circuit


def construct_repeating_block_for_Steane_Extraction(distance: int, data_qubits: list,anc_qubits: list, flag_qubit: list ,p_data: float, 
                                                    p_anc: float, logical_state: str, Reset: bool, after_CNOT_depol_type: str,
                                                    p_depol_after: float, num_rounds: int):
    
    num_rounds  -= 1
    block        = stim.Circuit()
    
    if Reset==True:

        if num_rounds==0:
            repeat_block = stim.Circuit()
        else:

            # Initialization error
            block.append("DEPOLARIZE1",data_qubits,p_data)
            block.append("DEPOLARIZE1",anc_qubits,p_anc)

            # if distance>3:
            #     block.append("DEPOLARIZE1",flag_qubit,p_anc)            
            
            block.append("TICK",[])
            
            # Prepare the ancilla state (includes a flag qubit too)
            
            block += construct_logical_state_on_ancilla(distance, logical_state,after_CNOT_depol_type, p_depol_after)
            block.append("TICK",[])
            
            # Perform CNOTs between ancilla and data qubits
            block += entangle_data_with_ancilla(data_qubits,anc_qubits,logical_state,after_CNOT_depol_type,p_depol_after)
            block.append("TICK",[])
            
            # Measure the flag qubit

            # if distance>3:    

            #     if logical_state=='+':
            #         block.append_operation("H",flag_qubit)

            #     block.append_operation("MR",flag_qubit)    
            #     block.append("DETECTOR",[stim.target_rec(-1)],(0,0,1))        

            block.append("TICK",[])
            # Measure the ancilla qubits:

            block += measure_logical_ancilla(anc_qubits,logical_state,Reset)

            # Define the detectors

            block.append_operation("SHIFT_COORDS", [], [0,1])
            block.append_operation("SHIFT_COORDS", [], [0,0,1])
            #Detector annotation
            for k in range(len(anc_qubits)-1,0,-1): #range(len(anc_qubits)-1)
                coords = (k,0)
                # block.append("DETECTOR",[stim.target_rec(-1-k),
                #                          stim.target_rec(-2-k),
                #                          stim.target_rec(-1-k-(distance)), 
                #                          stim.target_rec(-2-k-(distance))],coords) 

                # if distance>3:
                #     rec3 = stim.target_rec(-k-distance-1)
                #     rec4 = stim.target_rec(-k-distance-2)
                # else:
                rec3 = stim.target_rec(-k-(distance))
                rec4 = stim.target_rec(-1-k-(distance))
                                        

                block.append("DETECTOR",[stim.target_rec(-k),
                                         stim.target_rec(-1-k),
                                         rec3, 
                                         rec4],coords)                 

            repeat_block = stim.Circuit()
            repeat_block.append_from_stim_program_text("Repeat "+ str(num_rounds) + "{ " + get_str(block) + "\n }")
    

    else:
        raise Exception("Need to update the code for the no-reset case.")


    return repeat_block

def repetition_code_circuit_Steane_Extraction(distance: int, num_rounds: int, p_data: float, p_anc: float, Reset: bool,
                                              p_depol_after: float, after_CNOT_depol_type: str, logical_state: str):
    '''
    Circuit for Steane-style syndrome extraction on repetition code.
    ***Does not contain extra flag qubit which is needed for d \geq 5 if we want to preserve the distance.***

    '''

    data_qubits = (np.arange(distance)).tolist()
    anc_qubits  = (np.arange(distance)+distance).tolist()    
    flag_qubit  = [anc_qubits[-1]+1] 
    circuit     = stim.Circuit()

    circuit.append_operation("R",data_qubits+anc_qubits)

    # if distance>3:
    #     circuit.append_operation("R",flag_qubit)
    # circuit.append("TICK",[])

    # Initialization error
    circuit.append("DEPOLARIZE1",data_qubits,p_data)
    circuit.append("DEPOLARIZE1",anc_qubits,p_anc)

    # if distance>3:
    #     circuit.append("DEPOLARIZE1",flag_qubit,p_anc)

    # Logical 0 on data qubits: we do nothing
    # Logical + on data qubits: apply H on every qubit (repetition code memory |+>_L)
    
    if logical_state=='+':
        circuit.append_operation("H",data_qubits)

    # Prepare the ancilla state (includes a flag qubit too)
    anc_prep_circuit = construct_logical_state_on_ancilla(distance, logical_state,after_CNOT_depol_type, p_depol_after)

    circuit +=  anc_prep_circuit

    # # Measure the flag qubit
    # if distance>3:    

    #     if logical_state=='+':
    #         circuit.append_operation("H",flag_qubit)

    #     circuit.append_operation("MR",flag_qubit)

    #     circuit.append_operation("DETECTOR",stim.target_rec(-1),(0,0,0)) #flag coords

    circuit.append("TICK",[])

    #Now, apply CNOTs from each data qubit to each ancilla qubits

    if logical_state=='0':

        for k in range(len(data_qubits)):

            qq = [data_qubits[k]]+[anc_qubits[k]]
            circuit.append_operation("CX",qq)

            #if k==1:#k==0 or k==len(anc_qubits)-1:
                #circuit.append(after_CNOT_depol_type,qq,p_depol_after)
            #if k==0:
            #circuit.append(after_CNOT_depol_type,qq,p_depol_after)

            circuit.append(after_CNOT_depol_type,qq,p_depol_after)

    elif logical_state=='+':

        for k in range(len(data_qubits)):

            qq = [anc_qubits[k]] + [data_qubits[k]]
            circuit.append_operation("CX",qq)
            circuit.append(after_CNOT_depol_type,qq,p_depol_after)

    circuit.append("TICK",[])
    # Measure the rest of the ancilla:

    #circuit.append("TICK",[])
    circuit += measure_logical_ancilla(anc_qubits,logical_state, Reset)
    

    #Detector annotation: consecutive ancilla qubits are set to 1 detector (this is the stabilizer measurement)
    
    for k in range(len(anc_qubits)-1,0,-1):  #either range(len(anc_qubits)-1) and (-1-k), (-2-k) or the one below

        coords = (k,0)
        circuit.append("DETECTOR",[stim.target_rec(-1-k),
                                   stim.target_rec(-k)],
                                   coords) #stim.target_rec(-1-k-(distance-1)) Dont associate with the previous one
        
        

    #Now, construct the repeating block:
    circuit += construct_repeating_block_for_Steane_Extraction(distance, data_qubits,anc_qubits, flag_qubit ,p_data, 
                                                               p_anc, logical_state, Reset, after_CNOT_depol_type,
                                                               p_depol_after, num_rounds)

    
    #Finally measure the data qubits, and then append the logical observable

    if logical_state=='+':
        circuit.append("H",data_qubits)
    
    circuit.append("M",data_qubits)

    targets = []

    for k in range(len(data_qubits)):

        targets.append(stim.target_rec(-1-k))

    # Put also last detectors and compare them with the previous measurement round of detectors

    cnt = 0
    circuit.append_operation("SHIFT_COORDS", [], [0,1])
    for k in range(len(data_qubits),1,-1): #range(1,len(data_qubits))
        
        coords = (k-1,0)
        # circuit.append("DETECTOR",[stim.target_rec(-1-cnt),
        #                            stim.target_rec(-2-cnt),
        #                            stim.target_rec(-cnt-distance-1),
        #                            stim.target_rec(-cnt-distance-2)],coords)


        circuit.append("DETECTOR",[stim.target_rec(-k+1),
                                   stim.target_rec(-k),
                                   stim.target_rec(-k-distance+1),
                                   stim.target_rec(-k-distance)],coords)


        cnt=cnt+1

    circuit.append("OBSERVABLE_INCLUDE",targets,0)



    return circuit

    
