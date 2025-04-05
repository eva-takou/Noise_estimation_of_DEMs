
import numpy as np
import xarray as xr
from utilities.general_utils import bulk_prob_formula


#------------------ Estimation of error probabilities for time edges  ---------------------------


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
        time_edge_probs: dictionary with keys the names of detectors of time edges, and values the estimated probabilities.

    '''

    num_rounds = num_rounds+1 #We add the measurement of data qubits as another QEC round

    time_edge_probs = {}
    
    for l in range(num_ancilla):
        
        for k1 in range(num_rounds-1):
 
            k2   = k1+1
            v1   = vi_mean[k1,l]
            v2   = vi_mean[k2,l]
            v1v2 = vivj_mean[k1,k2,l,l]
            val  = bulk_prob_formula(v1,v2,v1v2)

            indx1 = l+num_ancilla*k1
            indx2 = l+num_ancilla*k2

            time_edge_probs[("D"+str(indx1),"D"+str(indx2))]=val

    return time_edge_probs  


#------------------ Estimation of error probabilities for the bulk  ------------------------



def estimate_data_edge_bulk_probs_all(num_rounds: int, num_ancilla: int, vi_mean: float, vivj_mean: float):
    '''
    Get the estimated probability for any 2 points in the bulk of the code 
    (we do not include potential boundary edges, but include all other space-time edges).

    Input:
        num_rounds: # of QEC rounds
        num_ancilla: # of ancilla qubits
        vi_mean: # of times each detector fires / # of shots (dims: # of qec rounds x # of ancilla )
        vi_vj_mean: # of times 2 detectors fire together (correlations) / # of shots (dims: # of qec rounds  x # of qec rounds x # of ancilla x # of ancilla)
    Output:
        pij_bulk: dictionary with keys the bulk detectors and values the estimated probabilities.

    '''
    num_rounds += 1 #Because we add to the last QEC round the measurement of data qubits as another "QEC round"

    
    pij_bulk = {}

    for anc1 in range(num_ancilla):

        for anc2 in range(num_ancilla):

            for rd1 in range(num_rounds):

                for rd2 in range(num_rounds):

                    if anc1==anc2 and rd1==rd2:
                        continue
                    else:

                        v1   = vi_mean[rd1,anc1]
                        v2   = vi_mean[rd2,anc2]
                        v1v2 = vivj_mean[rd1,rd2,anc1,anc2]
                        
                        val   = bulk_prob_formula(v1,v2, v1v2)
                        indx1 = anc1 + num_ancilla*rd1
                        indx2 = anc2 + num_ancilla*rd2

                        inds  = [indx1,indx2]

                        indx1 = min(inds)
                        indx2 = max(inds)

                        pij_bulk[("D"+str(indx1),"D"+str(indx2))]=val



    return pij_bulk


#----------------- Estimation of error probabilities for boundary edges --------------------------------


def estimate_data_edge_bd_probs(num_rounds: int, num_ancilla: int, vi_mean: float, vivj_mean: float):

    '''
    Estimate the boundary edge probabilities corresponding to data qubits. For the repetition code
    the boundary edge errors are detected by the first ancilla qubit and the last ancilla qubit.

    Input:
        num_rounds:  # of qec rounds
        num_ancilla: # of ancilla qubits
        vi_mean:     # of times each detector fires/ # of shots (dims: # of rounds x # of ancilla)
        vivj_mean:   # of times 2 detectors fire/ # of shots (dims: # of rounds x # of rounds x # of ancilla x # of ancilla)

    Output:
        pij_bd: dictionary of boundary node names with values the estimated probabilities.

    '''
    num_rounds    = num_rounds+1 #Because we add to the last QEC round the measurement of data qubits as another QEC round
    ancilla       = [0,num_ancilla-1]


    pij_bd  = {}
    for anc in ancilla:

        for rd in range(num_rounds):
            
            DENOM = 1
            vi     = vi_mean[rd,anc]

            #Get nearest space edge
            if anc==0:
                anc2 = anc+1
            else:
                anc2 = anc-1
            rd2 = rd
            
            vj   = vi_mean[rd,anc2]
            vivj = vivj_mean[rd,rd2,anc,anc2]
            pij  = bulk_prob_formula(vi,vj,vivj)

            DENOM *=1-2*pij 

            #Get nearest time edges:
            if rd-1>=0:
                vj     = vi_mean[rd-1,anc]
                vivj   = vivj_mean[rd,rd-1,anc,anc]
                pij    = bulk_prob_formula(vi,vj,vivj)
                DENOM *=1-2*pij 
            
            if rd+1<=(num_rounds-1):
                vj   = vi_mean[rd+1,anc]
                vivj = vivj_mean[rd,rd+1,anc,anc]
                pij    = bulk_prob_formula(vi,vj,vivj)
                DENOM *=1-2*pij 
            
            #Get also the diagonal:
            if anc==0:
                anc2 = anc+1
                rd2  = rd-1
                if rd2>=0:
                    vj     = vi_mean[rd2,anc2]
                    vivj   = vivj_mean[rd,rd2,anc,anc2]
                    pij    = bulk_prob_formula(vi,vj,vivj)
                    DENOM *=1-2*pij 

            else:
                anc2 = anc-1
                rd2  = rd+1
                if rd2<=(num_rounds-1):
                    vj     = vi_mean[rd2,anc2]
                    vivj   = vivj_mean[rd,rd2,anc,anc2]
                    pij    = bulk_prob_formula(vi,vj,vivj)
                    DENOM *=1-2*pij 


            
            indx   = anc+num_ancilla*rd 
            
            pij_bd[("D"+str(indx))]=1/2+(vi_mean[rd,anc]-1/2)/DENOM


    return pij_bd

