import stim


def get_str(circuit: stim.Circuit):
    '''Get the circuit as str.
    Input:
        circuit: stim circuit
    Output:
        the str format of the circuit instructions
    '''
    return circuit.__str__()


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