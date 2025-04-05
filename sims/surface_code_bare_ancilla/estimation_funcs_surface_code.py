import numpy as np
import xarray as xr
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

def get_nearest_anc_indices(L,Option):
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


def nearest_anc_inds_to_unique_pairs(distance,Option):

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
            sorted_pair = tuple(sorted((v1, v2)))
            pairs.add(sorted_pair)

    pairs = list(pairs)    
                

    return pairs


def get_anc_pairs_that_form_L0(distance,num_rounds):
    ''' Get the edge indices of the logical observable, where the edge is formed by the names of the
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

def first_diag_errors_DEM(distance,num_rounds):
    '''Indices of detectors that form
    one of the diagonal errors in the DEM'''
    
    L                = distance
    num_ancilla      = L*(L-1)
    detector_pairs   = []
    rd_and_anc_pairs = []

    for anc1 in range(L,num_ancilla):#range(L,num_ancilla):  #only L through 2L will lead to L0

        for rd1 in range(num_rounds-1):

            indx1 = anc1 + num_ancilla * rd1

            rd2 = rd1+1

            anc2  = anc1-(L)
            indx2 = anc2 + num_ancilla * rd2

            detector_pairs.append((indx1,indx2))

            rd_and_anc_pairs.append((rd1,anc1,rd2,anc2))


    return detector_pairs, rd_and_anc_pairs

def second_diag_errors_DEM(distance,num_rounds):
    '''Indices of detectors that form
    the other diagonal errors in the DEM'''
    L           = distance
    num_ancilla = L*(L-1)

    lower_bd_ancilla = np.arange(0,num_ancilla,L)

    detector_pairs   = []
    rd_and_anc_pairs = []

    for anc1 in range(1,num_ancilla):

        for rd1 in range(num_rounds-1):

            indx1 = anc1 + num_ancilla * rd1

            rd2 = rd1+1

            anc2  = anc1-1
            indx2 = anc2 + num_ancilla * rd2

            if indx1 not in lower_bd_ancilla:
                
                detector_pairs.append((indx1,indx2))
                rd_and_anc_pairs.append((rd1,anc1,rd2,anc2))


    return detector_pairs,rd_and_anc_pairs





def estimate_time_edge_probs(num_rounds:int, num_ancilla:int, vi_mean, vivj_mean):
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

    num_rounds += 1
    pij_time    = {}

    for anc in range(num_ancilla):

        for rd1 in range(num_rounds-1):

            rd2 = rd1+1
            v1  = vi_mean[rd1,anc]
            v2  = vi_mean[rd2,anc]
            v1v2 = vivj_mean[rd1,rd2,anc,anc]

            val = bulk_prob_formula(v1,v2,v1v2)

            indx1 = anc + num_ancilla * rd1
            indx2 = anc + num_ancilla * rd2 

            pij_time[("D"+str(indx1),"D"+str(indx2))] = val

    return pij_time


def estimate_bulk_and_bd_edge_probs(num_rounds:int, num_ancilla:int, distance: int, vi_mean, vivj_mean, pij_time: dict):
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

    #Same rd, different anc pairs
    for rd1 in range(num_rounds):

        rd2 = rd1
        
        for l in range(len(anc_pairs)):

            anc1,anc2 = anc_pairs[l]
            
            indx1 = anc1 + num_ancilla * rd1
            indx2 = anc2 + num_ancilla * rd2

            v1v2 = vivj_mean[rd1,rd2,anc1,anc2]
            v1   = vi_mean[rd1,anc1]
            v2   = vi_mean[rd2,anc2]


            det_indx1 = "D"+str(indx1)
            det_indx2 = "D"+str(indx2)


            if (indx1,indx2) in logical_inds:
            
                pij_bulk[(det_indx1,det_indx2,"L0")]=bulk_prob_formula(v1,v2,v1v2)            
            else:
                pij_bulk[(det_indx1,det_indx2,"")]=bulk_prob_formula(v1,v2,v1v2)            


    #now add the right and the left diagonals
    det_inds,right_pairs = first_diag_errors_DEM(distance,num_rounds)
    det_inds,left_pairs  = second_diag_errors_DEM(distance,num_rounds)

    for l in range(len(right_pairs)):
        
        rd1, anc1, rd2, anc2 = right_pairs[l]

        INDX1 = anc1+num_ancilla*rd1
        INDX2 = anc2+num_ancilla*rd2

        indx1 = min([INDX1,INDX2])
        indx2 = max([INDX1,INDX2])

        det_indx1 = "D"+str(indx1)
        det_indx2 = "D"+str(indx2)

        v1v2 = vivj_mean[rd1,rd2,anc1,anc2]
        v1   = vi_mean[rd1,anc1]
        v2   = vi_mean[rd2,anc2]

        
        if anc1 in range(L,2*L):
            pij_bulk[(det_indx1,det_indx2,"L0")]=bulk_prob_formula(v1,v2,v1v2)            
        else:
            pij_bulk[(det_indx1,det_indx2,"")]=bulk_prob_formula(v1,v2,v1v2)            


    for l in range(len(left_pairs)):
        
        rd1, anc1, rd2, anc2 = left_pairs[l]

        INDX1 = anc1+num_ancilla*rd1
        INDX2 = anc2+num_ancilla*rd2

        indx1 = min([INDX1,INDX2])
        indx2 = max([INDX1,INDX2])

        det_indx1 = "D"+str(indx1)
        det_indx2 = "D"+str(indx2)

        
        v1v2 = vivj_mean[rd1,rd2,anc1,anc2]
        v1   = vi_mean[rd1,anc1]
        v2   = vi_mean[rd2,anc2]


        pij_bulk[(det_indx1,det_indx2,"")]=bulk_prob_formula(v1,v2,v1v2)            

    #Finally, add the bd errors. For this we need the nearest detectors to the currently inspected one
    
    nearest_to_bd = get_nearest_anc_indices(L,"bd_only")
    pij_bd        = {}
    max_det_indx  = (L*(L-1))*num_rounds-1

    for rd1 in range(num_rounds):
    
        for anc in nearest_to_bd.keys():

            DENOM      = 1
            num_of_det = anc + num_ancilla * rd1

            det_indx1 = "D" + str(num_of_det)

            v0        = vi_mean[rd1,anc]
            NUMER     = v0-1/2

            #Get nearest time-edges:
            if (anc + num_ancilla * (rd1+1))<=max_det_indx:#anc+num_ancilla<=max_det_indx and num_of_det!=anc+num_ancilla:
                p1     = pij_time[(det_indx1,"D"+str(anc + num_ancilla * (rd1+1)))]
                DENOM *= 1-2*p1

            if (anc + num_ancilla * (rd1-1))>=0 and num_of_det!=anc-num_ancilla:
                p1     = pij_time[("D"+str(anc + num_ancilla * (rd1-1)),det_indx1)]
                DENOM *= 1-2*p1

            #Get nearest bulk space edge (not diagonal)

            neighbors = nearest_to_bd[anc]
            
            for anc2 in neighbors:
                
                num_of_det2 = anc2 + num_ancilla*rd1
                det_indx2   = "D"+str(num_of_det2)
                
                for KEY in pij_bulk.keys():

                    
                    if KEY[0:2]==(det_indx1,det_indx2) or KEY[0:2]==(det_indx2,det_indx1):
                        
                        p2 = pij_bulk[KEY]
                        DENOM *= 1 -2*p2
            

            #Add the diagonals:

            for l in range(len(left_pairs)):
                
                RD0,A0,RD1,A1 = left_pairs[l]

                INDX0 = A0 + num_ancilla*RD0
                INDX1 = A1 + num_ancilla*RD1

                if INDX0 == num_of_det:
                    
                    indx0      = min([INDX1,num_of_det])
                    indx_other = max([INDX1,num_of_det])
                    p1         = pij_bulk[("D"+str(indx0),"D"+str(indx_other),"")]
                    DENOM *= 1-2*p1

                elif INDX1 == num_of_det:

                    indx0      = min([INDX0,num_of_det])
                    indx_other = max([INDX0,num_of_det])
                    p1         = pij_bulk[("D"+str(indx0),"D"+str(indx_other),"")]
                    DENOM *= 1-2*p1

            
            for l in range(len(right_pairs)):
                
                RD0,A0,RD1,A1 = right_pairs[l]

                INDX0 = A0 + num_ancilla*RD0
                INDX1 = A1 + num_ancilla*RD1

                if INDX0 == num_of_det:
                    
                    indx0      = min([INDX1,num_of_det])
                    indx_other = max([INDX1,num_of_det])

                    if ("D"+str(indx0),"D"+str(indx_other),"L0") in pij_bulk.keys():

                        p1         = pij_bulk[("D"+str(indx0),"D"+str(indx_other),"L0")]
                    else:
                        p1         = pij_bulk[("D"+str(indx0),"D"+str(indx_other),"")]
                    
                    DENOM *= 1-2*p1

                elif INDX1 == num_of_det:

                    indx0      = min([INDX0,num_of_det])
                    indx_other = max([INDX0,num_of_det])

                    if ("D"+str(indx0),"D"+str(indx_other),"L0") in pij_bulk.keys():
                        p1         = pij_bulk[("D"+str(indx0),"D"+str(indx_other),"L0")]
                    else:
                        p1         = pij_bulk[("D"+str(indx0),"D"+str(indx_other),"")]
                    DENOM *= 1-2*p1
            

            pij_bd[det_indx1]=1/2+NUMER/DENOM


    return pij_bulk,pij_bd




