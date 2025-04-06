


def edge_dicts(num_ancilla: int,num_rounds: int):
    '''Create a dictionary with all the edge names, besides the 4-point events.
    
    Input:
        num_ancilla: # of detectors per round (int)
        num_rounds: # of QEC rounds w/o final stabilizer reconstruction (int)
    
    Output:
        bulk_edges: dictionary with keys the name of bulk space edges of the form ("Di","Dj") and values set to 0
        time_edges: dictionary with keys the name of time edges of the form ("Di","Dj") and values set to 0
        bd_edges  : dictionary with keys the name of time edges of the form ("Di") and values set to 0
    '''


    bulk_edges = set()
    time_edges = set()
    bd_edges   = set()

    #Time edges
    for rd1 in range(num_rounds):
        rd2=rd1+1
        for anc1 in range(num_ancilla):
            anc2 = anc1

            indx1 = anc1+num_ancilla*rd1
            indx2 = anc2+num_ancilla*rd2

            name = (f"D{indx1}",f"D{indx2}")

            time_edges.add(name)
    
    #Space edges
    for rd1 in range(num_rounds+1):
        rd2 = rd1 
        for anc1 in range(num_ancilla-1):
            anc2 = anc1 +1

            indx1 = anc1+num_ancilla*rd1
            indx2 = anc2+num_ancilla*rd2
            name = (f"D{indx1}",f"D{indx2}")

            bulk_edges.add(name)
    
    for rd1 in range(num_rounds+1):

        anc1 = 0
        
        indx1 = anc1+num_ancilla*rd1
        name  = (f"D{indx1}")
        bd_edges.add(name)

        anc1  = num_ancilla-1
        indx1 = anc1+num_ancilla*rd1
        name  = (f"D{indx1}")
        bd_edges.add(name)

    return bulk_edges,time_edges,bd_edges
