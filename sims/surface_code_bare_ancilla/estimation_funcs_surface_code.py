import numpy as np
from utilities.general_utils import bulk_prob_formula

def get_L0_inds(distance: int):
    '''Get the indices of qubits that form the logical operator.
    
    '''
    L       = distance
    shift   = (L-1)*(L-1)-1  #The -1 is because python counts from 0
    L0_inds = np.arange(shift+1,shift+1+L)

    return L0_inds

def get_observable_flips(data_qubit_samples,L0_inds):
    '''Get the observable flips from the data qubit measurements.
    
    Input:
        data_qubit_sample: data qubit measurements of dimensions # of shots x # of qubits (xArray).
        L0_inds: indices of qubits that form the logical operator.
    Output:
        obs_flips: binarry vector of length # of shots where 1 means the observable was flipped, and 0 it was not flipped.
    '''

    obs_flips = np.bitwise_xor.reduce(data_qubit_samples.data[:, L0_inds], axis=1)

    return obs_flips



def get_anc_indices_for_bd_edges(L: int):
    '''Get the indices that correspond to ancilla that check boundary data qubits.
    
    Input:
        L: distance of surface code (int).

    Output:
        anc_indices_checking_bd: list of integers corresponding to the ancilla indices that check boundary qubits.

    '''
    
    #Total # of detectors is L x (L-1).
    #Out of these the first L and the last L check boundary qubits.
    
    anc_indices_checking_bd = np.arange(0,L).tolist() + np.sort([L*(L-1)-1-k for k in range(L)]).tolist()

    return anc_indices_checking_bd

def get_nearest_anc_indices(L: int ,Option):
    '''Return the nearest indices of any ancilla index.
    
    
    '''
    
    num_ancilla = L*(L-1)

    if Option == "all":
        anc_indices = np.arange(0,num_ancilla)
    elif Option == "bd_only":
        anc_indices = get_anc_indices_for_bd_edges(L)


    upper_bd_ancilla = np.arange(L-1,num_ancilla,L)
    lower_bd_ancilla = np.arange(0,num_ancilla,L)
    nearest_anc = {}

    for k in anc_indices:

        temp = []
        
        if k+L<=(num_ancilla-1):
            temp.append(k+L)

        if k-L>=0:
            temp.append(k-L)

        if k+1<=(num_ancilla-1):

            if not (k+1 in upper_bd_ancilla and k in lower_bd_ancilla) and not (k in upper_bd_ancilla and k+1 in lower_bd_ancilla):
                temp.append(k+1)
        if k-1>=0:
            
            if not (k-1 in lower_bd_ancilla and k in upper_bd_ancilla) and not (k-1 in upper_bd_ancilla and k in lower_bd_ancilla) :
            
                temp.append(k-1)
        
        nearest_anc[k]=temp


    return nearest_anc

def nearest_anc_inds_to_unique_pairs(distance: int, Option):

    '''These are all the bulk connections (ancillas defining a qubit edge).'''

    anc_neighbors=get_nearest_anc_indices(distance,Option)

    # pairs = []
    # for key in anc_neighbors.keys():

    #     v1 = key
    #     other = anc_neighbors[key]

    #     for l in range(len(other)):
        
    #         min_v = min([v1,other[l]])
    #         max_v = max([v1,other[l]])

    #         sorted_pair = (min_v,max_v)
    #         if sorted_pair not in pairs:

    #             pairs.append(sorted_pair)

    pairs = set()
    for v1, neighbors in anc_neighbors.items():
        for v2 in neighbors:
            
            # sorted_pair = tuple(sorted((v1, v2)))
            pairs.add((min(v1,v2),max(v1,v2)))

    pairs = list(pairs)    
                

    return pairs

def get_anc_pairs_that_form_L0(distance: int, num_rounds: int):
    '''
    
      
        Get the edge indices of the logical observable, where the edge is formed by the names of the
    detectors. Also, the inds are shifted by the num_rounds. The number of rounds is +1 from the actual
    QEC rounds, because it also includes the projection.'''
    
    num_rounds       = num_rounds-1
    L                = distance   
    num_ancilla      = L*(L-1)
    # anc_pairs_for_L0 = []

    # for k in range(L):

    #     anc_pairs_for_L0.append((k,k+L))
    
    # if num_rounds>=1:
    
    #     cnt       = 0
    #     old_batch = anc_pairs_for_L0.copy()
        
    #     while True:

    #         new_batch = []
    #         for l in range(len(old_batch)):
                
    #             anc0 = old_batch[l][0]
    #             anc1 = old_batch[l][1]


    #             new_pair = (anc0+num_ancilla,  anc1+num_ancilla)            
    #             new_batch.append(new_pair)
    #             anc_pairs_for_L0.append(new_pair)

    #         old_batch = new_batch.copy()

    #         cnt+=1
    #         if cnt==num_rounds:

    #             break

    # Initial batch
    anc_pairs_for_L0 = [(k, k + L) for k in range(L)]

    if num_rounds >= 1:
        old_batch = anc_pairs_for_L0

        for _ in range(num_rounds):
            new_batch = [(a0 + num_ancilla, a1 + num_ancilla) for a0, a1 in old_batch]
            anc_pairs_for_L0.extend(new_batch)
            old_batch = new_batch


    return anc_pairs_for_L0

def first_diag_errors_DEM(distance: int, num_rounds: int):
    '''
    Get rd,anc indices of one set of diagonal errors of the surface code detector error model.
    
    Input: 
        distance: distance of surface code (int)
        num_rounds: # of QEC rounds (int)
    Output:
        detector_pairs: list of tuples of detector names of the form (indx1,indx2), where indx = anc + num_ancilla * rd
        rd_and_anc_pairs: list of tuples of the form (rd1,anc1,rd2,anc2)

    '''
    
    L                = distance
    num_ancilla      = L*(L-1)
    detector_pairs   = []
    rd_and_anc_pairs = []

    for anc1 in range(L,num_ancilla):  #only L through 2L will lead to L0

        anc2  = anc1-(L)

        for rd1 in range(num_rounds-1):

            indx1 = anc1 + num_ancilla * rd1
            rd2   = rd1+1
            indx2 = anc2 + num_ancilla * rd2

            detector_pairs.append((indx1,indx2))
            rd_and_anc_pairs.append((rd1,anc1,rd2,anc2))

            if indx2<indx1:
                raise Exception("Inds not sorted.")


    return detector_pairs, rd_and_anc_pairs



def second_diag_errors_DEM(distance: int, num_rounds: int):
    '''
    Get rd,anc indices of one set of diagonal errors of the surface code detector error model.
    
    Input: 
        distance: distance of surface code (int)
        num_rounds: # of QEC rounds (int)
    Output:
        detector_pairs: list of tuples of detector names of the form (indx1,indx2), where indx = anc + num_ancilla * rd
        rd_and_anc_pairs: list of tuples of the form (rd1,anc1,rd2,anc2)

    '''
    L           = distance
    num_ancilla = L*(L-1)

    lower_bd_ancilla = np.arange(0,num_ancilla,L)

    detector_pairs   = []
    rd_and_anc_pairs = []

    for anc1 in range(1,num_ancilla):
        anc2  = anc1-1

        for rd1 in range(num_rounds-1):

            indx1 = anc1 + num_ancilla * rd1
            rd2   = rd1+1    
            indx2 = anc2 + num_ancilla * rd2

            if indx1 not in lower_bd_ancilla:
                
                detector_pairs.append((indx1,indx2))
                rd_and_anc_pairs.append((rd1,anc1,rd2,anc2))


    return detector_pairs,rd_and_anc_pairs


def estimate_time_edge_probs(num_rounds:int, num_ancilla:int, defects_matrix,vi_mean) -> dict:
    '''
    Estimate the error probabilities of ancilla qubits (time-edges).
    These are all bulk edges because one error creates two detector events.
    We only evaluate the error probabilities of consecutive time-edges.

    Input:
        num_rounds: # of QEC rounds
        num_ancilla: # of ancilla qubits
        vi_mean: # of times each detector fires / # of shots (dims: # of qec rounds x # of ancilla )
        vi_vj_mean: # of times 2 detectors fire together (correlations) / # of shots (dims: # of qec rounds  x # of qec rounds x # of ancilla x # of ancilla)

    Output:
        pij_time: dictionary with keys the names of detectors of time edges, and values the estimated probabilities.

    '''

    num_shots   = np.shape(defects_matrix)[0]
    num_rounds += 1
    pij_time    = {}

    data = defects_matrix.data

    #Need rd and rd+1 for each ancilla.

    anc_array = np.arange(num_ancilla)
    
    for rd1 in range(num_rounds-1):
        
        rd2  = rd1+1

        data1 = data[:, rd1, :]
        data2 = data[:, rd2, :]

        v1   = vi_mean[rd1,:]
        v2   = vi_mean[rd2,:]
        v1v2 = np.sum(data1 & data2, axis=0) / num_shots        

        vals = np.array([bulk_prob_formula(a, b, c) for a, b, c in zip(v1, v2, v1v2)])

        indx1 = anc_array + num_ancilla * rd1
        indx2 = anc_array + num_ancilla * rd2

        pij_time.update({(f"D{i1}", f"D{i2}"): val for i1, i2, val in zip(indx1, indx2, vals)})
        
        

    return pij_time


def estimate_bulk_and_bd_edge_probs(num_rounds:int, num_ancilla:int, distance: int, 
                                             defects_matrix, pij_time: dict,vi_mean):
    '''
    Get the estimated probability for any bulk or boundary edges.

    Input:
        num_rounds: # of QEC rounds (int)
        num_ancilla: # of ancilla qubits (int)
        distance: distance of surface code (int)
        vi_mean: # of times each detector fires / # of shots (dims: # of qec rounds x # of ancilla )
        vi_vj_mean: # of times 2 detectors fire together (correlations) / # of shots (dims: # of qec rounds  x # of qec rounds x # of ancilla x # of ancilla)
    Output:
        pij_bulk: dictionary with keys the bulk detectors and values the estimated probabilities.
        pij_bd:   dictionary with keys the boundary detectors and values the estimated probabilities.
    '''
    L            = distance
    num_rounds  += 1
    pij_bulk     = {}
    Option       = "all"
    anc_pairs    = nearest_anc_inds_to_unique_pairs(distance,Option)
    logical_inds = get_anc_pairs_that_form_L0(distance,num_rounds)
    num_shots    = np.shape(defects_matrix)[0]

    data = defects_matrix.data

    #Same rd, different anc pairs

    LA = len(anc_pairs)

    for rd1 in range(num_rounds):

        rd2 = rd1
        
        for l in range(LA):

            anc1,anc2 = anc_pairs[l]
            
            indx1 = anc1 + num_ancilla * rd1
            indx2 = anc2 + num_ancilla * rd2

            v1   = vi_mean[rd1,anc1]
            v2   = vi_mean[rd2,anc2]
            v1v2 = np.sum(data[:,rd1,anc1]& data[:,rd2,anc2])/num_shots

            det_indx1 = f"D{indx1}"
            det_indx2 = f"D{indx2}"


            if (indx1,indx2) in logical_inds:
            
                pij_bulk[(det_indx1,det_indx2,"L0")]=bulk_prob_formula(v1,v2,v1v2)            
            else:
                pij_bulk[(det_indx1,det_indx2,"")]=bulk_prob_formula(v1,v2,v1v2)            


    #now add the right and the left diagonals
    det_inds_right,right_pairs = first_diag_errors_DEM(distance,num_rounds)
    det_inds_left,left_pairs   = second_diag_errors_DEM(distance,num_rounds)

    RP = len(right_pairs)

    range_for_L0 = set(range(L, 2*L))

    for l in range(RP):
        
        rd1, anc1, rd2, anc2 = right_pairs[l]

        indx1,indx2=det_inds_right[l]

        det_indx1 = f"D{indx1}"
        det_indx2 = f"D{indx2}"

        v1   = vi_mean[rd1,anc1]
        v2   = vi_mean[rd2,anc2]
        v1v2 = np.sum(data[:,rd1,anc1]& data[:,rd2,anc2])/num_shots

        if anc1 in range_for_L0:
            pij_bulk[(det_indx1,det_indx2,"L0")]=bulk_prob_formula(v1,v2,v1v2)            
        else:
            pij_bulk[(det_indx1,det_indx2,"")]=bulk_prob_formula(v1,v2,v1v2)            

    LP = len(left_pairs)

    for l in range(LP):
        
        rd1, anc1, rd2, anc2 = left_pairs[l]

        indx1,indx2 = det_inds_left[l]

        det_indx1 = f"D{indx1}"
        det_indx2 = f"D{indx2}"

        v1   = vi_mean[rd1,anc1]
        v2   = vi_mean[rd2,anc2]
        v1v2 = np.sum(data[:,rd1,anc1] & data[:,rd2,anc2])/num_shots
        
        pij_bulk[(det_indx1,det_indx2,"")]=bulk_prob_formula(v1,v2,v1v2)            

    #Finally, add the bd errors. For this we need the nearest detectors to the currently inspected one
    
    nearest_to_bd = get_nearest_anc_indices(L,"bd_only")
    pij_bd        = {}
    max_det_indx  = (L*(L-1))*num_rounds-1

    for rd1 in range(num_rounds):
    
        for anc in nearest_to_bd.keys():

            DENOM      = 1
            num_of_det = anc + num_ancilla * rd1

            det_indx1 = f"D{num_of_det}"

            v0        = vi_mean[rd1,anc]
            NUMER     = v0-1/2

            #Get nearest time-edges:
            if (anc + num_ancilla * (rd1+1))<=max_det_indx:#anc+num_ancilla<=max_det_indx and num_of_det!=anc+num_ancilla:
                p1     = pij_time[(det_indx1,f"D{anc + num_ancilla * (rd1+1)}")]
                DENOM *= 1-2*p1

            if (anc + num_ancilla * (rd1-1))>=0 and num_of_det!=anc-num_ancilla:
                p1     = pij_time[(f"D{anc + num_ancilla * (rd1-1)}",det_indx1)]
                DENOM *= 1-2*p1

            #Get nearest bulk space edge (not diagonal)

            # neighbors = nearest_to_bd[anc]
            
            # for anc2 in neighbors:
                
            #     num_of_det2         = anc2 + num_ancilla*rd1
            #     new_indx1,new_indx2 = (num_of_det,num_of_det2) if num_of_det<num_of_det2 else (num_of_det2,num_of_det)

                
            #     d1 = f"D{new_indx1}"
            #     d2 = f"D{new_indx2}"

                
            #     key = (d1, d2, "")
            #     if key not in pij_bulk:
            #         key = (d1, d2, "L0")

            #     p2 = pij_bulk[key]

            #     DENOM *= 1 - 2 * p2

            # #-----Add the diagonals-------

            # #Right diagonals:
            
            # if rd1<num_rounds: 

            #     anc2  = anc-L

            #     if anc2>=0:

            #         rd2   = rd1+1
            #         indx2 = anc2 + num_ancilla * rd2 

            #         try_key1 = (det_indx1,f"D{indx2}","")
            #         try_key2 = (det_indx1,f"D{indx2}","L0")

            #         if try_key1 in pij_bulk:
            #             p1 = pij_bulk[try_key1]
            #             DENOM *= 1-2*p1
            #         elif try_key2 in pij_bulk:
            #             p1 = pij_bulk[try_key2]
            #             DENOM *= 1-2*p1
            
            # if rd1>=1: 

            #     anc2  = anc+L

            #     if anc2<=(num_ancilla-1):

            #         rd2   = rd1-1
            #         indx2 = anc2 + num_ancilla * rd2 

            #         try_key1 = (f"D{indx2}",det_indx1,"")
            #         try_key2 = (f"D{indx2}",det_indx1,"L0")

            #         if try_key1 in pij_bulk:
                        
            #             p1 = pij_bulk[try_key1]
            #             DENOM *= 1-2*p1
            #         elif try_key2 in pij_bulk:
                        
            #             p1 = pij_bulk[try_key2]
            #             DENOM *= 1-2*p1


            filtered = {k:v for k,v in pij_bulk.items() if k[0]==det_indx1 or k[1]==det_indx1}

            

            for key,val in filtered.items():
                
                DENOM *=1-2*val



            #Left diagonals:
                

            # for l in range(LP):

            #     temp = det_inds_left[l]
                
            #     if num_of_det in temp:
            #         p1         = pij_bulk[(f"D{temp[0]}",f"D{temp[1]}","")]
            #         DENOM *= 1-2*p1

            

            # for l in range(RP):

            #     temp = det_inds_right[l]

            #     if num_of_det in temp:
            #         key = (f"D{temp[0]}",f"D{temp[1]}","")
            #         if key in pij_bulk:
            #             p1  = pij_bulk[(f"D{temp[0]}",f"D{temp[1]}","")]
            #         else:
            #             p1 = pij_bulk[(f"D{temp[0]}",f"D{temp[1]}","L0")]

            #         # try:
            #         #     p1         = pij_bulk[(f"D{temp[0]}",f"D{temp[1]}","")]
            #         # except KeyError:
            #         #     p1         = pij_bulk[(f"D{temp[0]}",f"D{temp[1]}","L0")]
            #         DENOM *= 1-2*p1


            

            pij_bd[det_indx1]=1/2+NUMER/DENOM


    return pij_bulk,pij_bd




