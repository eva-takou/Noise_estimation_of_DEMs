import stim


def get_str(circuit: stim.Circuit):
    '''Get the circuit as str.
    Input:
        circuit: stim circuit
    Output:
        the str format of the circuit instructions
    '''
    return circuit.__str__()

