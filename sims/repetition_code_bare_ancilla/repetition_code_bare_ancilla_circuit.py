import stim
import numpy as np
from utilities.circuit_control import get_str


def construct_repeating_block(distance: int, num_rounds: int, p_data: float, p_anc: float, Reset: bool,
                              p_depol_after: float, after_CNOT_depol_type: str):
    '''Create the repeating block for the Z-memory repetition code.
    
    Input:
        distance: distance of repetition code
        num_rounds: # of QEC rounds
        p_data: input single-qubit depolarizing error rate on data qubits
        p_anc: input single qubit depolarizing error rate on ancilla qubits
        Reset: True or False for the reset of the ancilla after their measurement
        p_depol_after: depolarizing error rate after each CNOT gate
        after_CNOT_depol_type: "DEPOLARIZE1" or "DEPOLARIZE2" for the error channel after the CNOT gates

    Output:
        repeat_block: repeating block for the Z-memory repetition code
    '''

    num_rounds  -= 1
    block       = stim.Circuit()
    data_qubits = (np.arange(distance)).tolist()
    anc_qubits  = (np.arange(distance-1)+distance).tolist()    
    

    if after_CNOT_depol_type!="DEPOLARIZE1" and after_CNOT_depol_type!="DEPOLARIZE2":

        raise Exception("Only DEPOLARIZE1 or DEPOLARIZE2 can be inputs")

    if Reset==True:
        
        meas = "MR" 
        if num_rounds>0:

            #Initialization error
            block.append("DEPOLARIZE1",data_qubits,p_data)
            block.append("DEPOLARIZE1",anc_qubits,p_anc)
            
            #Prep for measurement
            cnt = 0
            for QA in anc_qubits:

                data = data_qubits[cnt:cnt+2]
                
                for Q in data:
                    qq = [Q] + [QA] 
                    block.append_operation("CX",qq)
                    if p_depol_after!=0:
                        block.append(after_CNOT_depol_type,qq,p_depol_after)

                cnt=cnt+1
        
            block.append("TICK",[])
            #Measurement of ancilla
            for QA in anc_qubits:
                block.append(meas,QA)
            
            block.append_operation("SHIFT_COORDS", [], [0,1])
            #Detector annotation
            
            for k in range(len(anc_qubits)):
                coords = (k,0)
                block.append("DETECTOR",[stim.target_rec(-(distance-1)+k),
                                        stim.target_rec(-(distance-1)+k+1-distance)],coords)               
                


            repeat_block = stim.Circuit()
            repeat_block.append_from_stim_program_text("Repeat "+ str(num_rounds) + "{ " + get_str(block) + "\n }")
        else:
            repeat_block = stim.Circuit()

    elif Reset==False:

        meas = "M" 

        if num_rounds==1:

            #Initialization error
            block.append("DEPOLARIZE1",data_qubits,p_data)
            block.append("DEPOLARIZE1",anc_qubits,p_anc)
            
            #Prep for measurement
            cnt = 0
            for QA in anc_qubits:

                data = data_qubits[cnt:cnt+2]
                
                for Q in data:
                    qq = [Q] + [QA] 
                    block.append_operation("CX",qq)
                    if p_depol_after!=0:
                        block.append(after_CNOT_depol_type,qq,p_depol_after)

                cnt=cnt+1
        
            block.append("TICK",[])
            #Measurement of ancilla
            for QA in anc_qubits:
                block.append(meas,QA)
            
            block.append_operation("SHIFT_COORDS", [], [0,1])
            #Detector annotation
            for k in range(len(anc_qubits)):
                coords = (k,0)
                block.append("DETECTOR",[stim.target_rec(-1-k)],coords) 


            repeat_block = block


        elif num_rounds>=2:

            #Initialization error
            block.append("DEPOLARIZE1",data_qubits,p_data)
            block.append("DEPOLARIZE1",anc_qubits,p_anc)

            #Prep for measurement
            cnt = 0
            for QA in anc_qubits:

                data = data_qubits[cnt:cnt+2]
                
                for Q in data:
                    qq = [Q] + [QA] 
                    block.append_operation("CX",qq)
                    if p_depol_after!=0:
                        block.append(after_CNOT_depol_type,qq,p_depol_after)

                cnt=cnt+1
        
            block.append("TICK",[])
            #Measurement of ancilla
            for QA in anc_qubits:
                block.append(meas,QA)
            
            block.append_operation("SHIFT_COORDS", [], [0,1])
            extra_block = block.copy()
            
            
            #Detector annotation
            for k in range(len(anc_qubits)):
                coords = (k,0)
                block.append("DETECTOR",[stim.target_rec(-1-k),
                                         stim.target_rec(-1-k-2*(distance-1))],coords) 
                extra_block.append("DETECTOR",[stim.target_rec(-1-k)],coords) 


            repeat_block = extra_block.copy()
            repeat_block.append_from_stim_program_text("Repeat "+ str(num_rounds-1) + "{ " + get_str(block) + "\n }")

        else:
            repeat_block = stim.Circuit()
        


    return repeat_block

def repetition_code_circuit(distance: int, num_rounds: int, p_data: float, p_anc: float, Reset: bool,
                          p_depol_after: float, after_CNOT_depol_type: str):
    
    '''Create the Z-memory repetition code circuit which has input depolarizing channel and two-qubit gate errors.
    

    Input:
        distance: distance of repetition code
        num_rounds: # of QEC rounds
        p_data: input single-qubit depolarizing error rate on data qubits
        p_anc: input single qubit depolarizing error rate on ancilla qubits
        Reset: True or False for the reset of the ancilla after their measurement
        p_depol_after: depolarizing error rate after each CNOT gate
        after_CNOT_depol_type: "DEPOLARIZE1" or "DEPOLARIZE2" for the error channel after the CNOT gates

    Output:
        circuit: stim circuit of the repetition code
    '''    

    data_qubits = (np.arange(distance)).tolist()
    anc_qubits  = (np.arange(distance-1)+distance).tolist()    
    circuit     = stim.Circuit()

    #Reset condition just of the ancilla qubits
    if Reset:
        meas = "MR" 
    else:
        meas = "M" #When we do no measurement reset, we need to compare or associate detectors that are 2 rounds apart

    circuit.append_operation("R",data_qubits+anc_qubits)
    circuit.append("TICK",[])

    #Initialization error
    circuit.append("DEPOLARIZE1",data_qubits,p_data)
    circuit.append("DEPOLARIZE1",anc_qubits,p_anc)

    #Prep for measurement
    cnt = 0
    for QA in anc_qubits:

        data = data_qubits[cnt:cnt+2]
        
        for Q in data:
            qq = [Q] + [QA] 
            circuit.append_operation("CX",qq)

            if p_depol_after!=0:
                circuit.append(after_CNOT_depol_type,qq,p_depol_after)

        cnt=cnt+1
    
    circuit.append("TICK",[])
    #Measurement of ancilla
    for QA in anc_qubits:
        circuit.append(meas,QA)
    
    #Detector annotation
    for k in range(len(anc_qubits)):
        coords = (k,0)
        circuit.append("DETECTOR",[stim.target_rec(-(distance-1)+k)],coords) 
        
    circuit +=construct_repeating_block(distance, num_rounds, p_data, p_anc, Reset,p_depol_after,after_CNOT_depol_type)


    #Finally, we want to put the last data qubit measurements, and add the detectors

    circuit.append("M",data_qubits)

    #Add the detectors and compare values with neighboring data qubits and appropriate last ancilla outcome.
    
    cnt = 0
    circuit.append_operation("SHIFT_COORDS", [], [0,1])
    for k in range(1,len(data_qubits)):

        coords=(k-1,0)

        #Need last 2 data qubits, and also the last measurement on the ancilla that checked those 2 qubits

        circuit.append("DETECTOR",[stim.target_rec(-(distance-1)+cnt),
                                   stim.target_rec(-(distance-1)+cnt-1),
                                   stim.target_rec(-(distance-1)+cnt-distance)],coords)


        cnt=cnt+1


    #Add the observable:
    obs_recs=[]
    for k in range(len(data_qubits)):

        obs_recs.append(stim.target_rec(-1-k))

    circuit.append("OBSERVABLE_INCLUDE",obs_recs,0)#Add Z's on all qubits

    return circuit

def construct_repeating_block_X_basis(distance: int, num_rounds: int, p_data: float, p_anc: float, Reset: bool,
                              p_depol_after: float, after_CNOT_depol_type: str):
    '''Create the repeating block for the X-memory repetition code.
    
    Input:
        distance: distance of repetition code
        num_rounds: # of QEC rounds
        p_data: input single-qubit depolarizing error rate on data qubits
        p_anc: input single qubit depolarizing error rate on ancilla qubits
        Reset: True or False for the reset of the ancilla after their measurement
        p_depol_after: depolarizing error rate after each CNOT gate
        after_CNOT_depol_type: "DEPOLARIZE1" or "DEPOLARIZE2" for the error channel after the CNOT gates

    Output:
        repeat_block: repeating block for the X-memory repetition code
    '''    

    num_rounds  -= 1
    block       = stim.Circuit()
    data_qubits = (np.arange(distance)).tolist()
    anc_qubits  = (np.arange(distance-1)+distance).tolist()    
    

    if after_CNOT_depol_type!="DEPOLARIZE1" and after_CNOT_depol_type!="DEPOLARIZE2":

        raise Exception("Only DEPOLARIZE1 or DEPOLARIZE2 can be inputs")

    if Reset==True:
        
        meas = "MR" 
        if num_rounds>0:

            #Initialization error
            block.append_operation("H",anc_qubits)
            block.append("DEPOLARIZE1",data_qubits,p_data)
            block.append("DEPOLARIZE1",anc_qubits,p_anc)
            
            #Prep for measurement
            cnt = 0
            for QA in anc_qubits:

                data = data_qubits[cnt:cnt+2]
                
                for Q in data:
                    qq = [QA] + [Q] 
                    block.append_operation("CX",qq)
                    block.append(after_CNOT_depol_type,qq,p_depol_after)

                cnt=cnt+1
        
            block.append("TICK",[])
            block.append_operation("H",anc_qubits)
            #Measurement of ancilla
            for QA in anc_qubits:
                block.append(meas,QA)
            
            block.append_operation("SHIFT_COORDS", [], [0,1])
            #Detector annotation
            for k in range(len(anc_qubits)):
                coords = (k,0)
                block.append("DETECTOR",[stim.target_rec(-1-k),
                                         stim.target_rec(-1-k-(distance-1))],coords) 


            repeat_block = stim.Circuit()
            repeat_block.append_from_stim_program_text("Repeat "+ str(num_rounds) + "{ " + get_str(block) + "\n }")
        else:
            repeat_block = stim.Circuit()

    elif Reset==False:

        meas = "M" 

        if num_rounds==1:

            #Initialization error
            block.append_operation("H",anc_qubits)
            block.append("DEPOLARIZE1",data_qubits,p_data)
            block.append("DEPOLARIZE1",anc_qubits,p_anc)
            
            #Prep for measurement
            cnt = 0
            for QA in anc_qubits:

                data = data_qubits[cnt:cnt+2]
                
                for Q in data:
                    qq = [QA] +[Q] 
                    block.append_operation("CX",qq)
                    block.append(after_CNOT_depol_type,qq,p_depol_after)

                cnt=cnt+1
        
            block.append("TICK",[])
            block.append_operation("H",anc_qubits)
            #Measurement of ancilla
            for QA in anc_qubits:
                block.append(meas,QA)
            
            block.append_operation("SHIFT_COORDS", [], [0,1])
            #Detector annotation
            for k in range(len(anc_qubits)):
                coords = (k,0)
                block.append("DETECTOR",[stim.target_rec(-1-k)],coords) 


            repeat_block = block


        elif num_rounds>=2:

            #Initialization error
            block.append_operation("H",anc_qubits)
            block.append("DEPOLARIZE1",data_qubits,p_data)
            block.append("DEPOLARIZE1",anc_qubits,p_anc)

            #Prep for measurement
            cnt = 0
            for QA in anc_qubits:

                data = data_qubits[cnt:cnt+2]
                
                for Q in data:
                    qq = [QA]+[Q]  
                    block.append_operation("CX",qq)
                    block.append(after_CNOT_depol_type,qq,p_depol_after)

                cnt=cnt+1
        
            block.append("TICK",[])
            block.append_operation("H",anc_qubits)
            #Measurement of ancilla
            for QA in anc_qubits:
                block.append(meas,QA)
            
            block.append_operation("SHIFT_COORDS", [], [0,1])
            extra_block = block.copy()
            
            
            #Detector annotation
            for k in range(len(anc_qubits)):
                coords = (k,0)
                block.append("DETECTOR",[stim.target_rec(-1-k),
                                         stim.target_rec(-1-k-2*(distance-1))],coords) 
                extra_block.append("DETECTOR",[stim.target_rec(-1-k)],coords) 


            repeat_block = extra_block.copy()
            repeat_block.append_from_stim_program_text("Repeat "+ str(num_rounds-1) + "{ " + get_str(block) + "\n }")

        else:
            repeat_block = stim.Circuit()
        


    return repeat_block

def repetition_code_circuit_X_basis(distance: int, num_rounds: int, p_data: float, p_anc: float, Reset: bool,
                          p_depol_after: float, after_CNOT_depol_type: str):
    '''Create the X-memory repetition code circuit which has input depolarizing channel and two-qubit gate errors.
    

    Input:
        distance: distance of repetition code
        num_rounds: # of QEC rounds
        p_data: input single-qubit depolarizing error rate on data qubits
        p_anc: input single qubit depolarizing error rate on ancilla qubits
        Reset: True or False for the reset of the ancilla after their measurement
        p_depol_after: depolarizing error rate after each CNOT gate
        after_CNOT_depol_type: "DEPOLARIZE1" or "DEPOLARIZE2" for the error channel after the CNOT gates

    Output:
        circuit: stim circuit of the repetition code
    '''    

    data_qubits = (np.arange(distance)).tolist()
    anc_qubits  = (np.arange(distance-1)+distance).tolist()    
    circuit     = stim.Circuit()

    #Reset condition just of the ancilla qubits
    if Reset:
        meas = "MR" 
    else:
        meas = "M" #When we do no measurement reset, we need to compare or associate detectors that are 2 rounds apart

    circuit.append_operation("R",data_qubits+anc_qubits)
    circuit.append("TICK",[])
    circuit.append_operation("H",data_qubits+anc_qubits)
    circuit.append("TICK",[])

    #Initialization error
    circuit.append("DEPOLARIZE1",data_qubits,p_data)
    circuit.append("DEPOLARIZE1",anc_qubits,p_anc)

    #Prep for measurement
    cnt = 0
    for QA in anc_qubits:

        data = data_qubits[cnt:cnt+2]
        
        for Q in data:
            qq = [QA]  + [Q] 
            circuit.append_operation("CX",qq)
            circuit.append(after_CNOT_depol_type,qq,p_depol_after)

        cnt=cnt+1
    
    circuit.append("TICK",[])
    circuit.append_operation("H",anc_qubits)
    #Measurement of ancilla
    for QA in anc_qubits:
        circuit.append(meas,QA)
    
    #Detector annotation
    for k in range(len(anc_qubits)):
        coords = (k,0)
        circuit.append("DETECTOR",[stim.target_rec(-1-k)],coords) #stim.target_rec(-1-k-(distance-1)) Dont associate with the previous one
        
    circuit +=construct_repeating_block_X_basis(distance, num_rounds, p_data, p_anc, Reset,p_depol_after,after_CNOT_depol_type)


    #Finally, we want to put the last data qubit measurements, and add the detectors
    circuit.append_operation("H",data_qubits)
    circuit.append("M",data_qubits)

    #Add the detectors and compare values with neighboring data qubits and appropriate last ancilla outcome.
    
    cnt = 0
    for k in range(1,len(data_qubits)):

        circuit.append_operation("SHIFT_COORDS", [], [0,1])


        #Need last 2 data qubits, and also the last measurement on the ancilla that checked those 2 qubits

        circuit.append("DETECTOR",[stim.target_rec(-1-cnt),
                                   stim.target_rec(-2-cnt),
                                   stim.target_rec(-cnt-distance-1)])

        cnt=cnt+1


    #Add the observable:
    obs_recs=[]
    for k in range(len(data_qubits)):

        obs_recs.append(stim.target_rec(-1-k))

    circuit.append("OBSERVABLE_INCLUDE",obs_recs,0)#Add Z's on all qubits




    return circuit







    
        
 

