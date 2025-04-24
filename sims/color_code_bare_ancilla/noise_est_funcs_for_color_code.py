import numpy as np
import stim
from math import prod
from utilities.general_utils import DEM_to_dictionary
from sims.color_code_bare_ancilla.utilities_for_color_code import *
from utilities.general_utils import bulk_prob_formula

def estimate_all_edges_for_defect_type(sols_for_defect_type, obj, num_rounds: int, vi_mean, vivj_mean, stims_DEM: stim.DetectorErrorModel, defects_type):
    '''Estimate the edges and hyperedges for the X-, or Z-DEM.

    Input:
        sols_for_defect_type: dictionary with solutions containing 3-point events
        obj: color code object
        num_rounds: # of QEC rounds (int)
        vi_mean:  average counts of individual detectors  (dims: (num_rounds+1) x # of detectors per round)
        vivj_mean: average 2-point coincidences of detector pairs  (dims: (num_rounds+1) x (num_rounds+1) x # of detectors per round x # of detectors per round)
        stims_DEM: the X-, or Z-DEM of stim (noise-aware)
        defects_type: "X" or "Z"
    Output:
        pij_bulk: dictionary of bulk probabilities
        pij_time: dictionary of time probabilities
        pij_bd:   dictionary of boundary probabilities
        p3:       dictionary of 3-point probabilities
        '''
    
    Z_dets,X_dets = get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds)

    if defects_type=="X":
        dets = X_dets
    elif defects_type=="Z":
        dets = Z_dets     
    else:
        raise Exception("defects_type can be only X or Z")

    errors_in_DEM = DEM_to_dictionary(stims_DEM)
    
    pij_bulk = {}
    pij_time = {}

    all_keys = list(dets.keys())

    #Get all the 3 pnt events
    p3 = {}
    for key in sols_for_defect_type.keys():
        if key in errors_in_DEM.keys():
            p3[key] = sols_for_defect_type[key][key]

    for m in range(len(all_keys)):
        
        det1     = all_keys[m]
        rd1,anc1 = dets[det1]

        for k in range(m+1,len(all_keys)):

            det2     = all_keys[k]
            rd2,anc2 = dets[det2]

            if tuple([det1,det2]) not in errors_in_DEM.keys() and tuple([det1,det2,"L0"]) not in errors_in_DEM.keys():
                continue

            if rd1!=rd2 and anc1==anc2: #time edges

                v1 = vi_mean[rd1,anc1]
                v2 = vi_mean[rd2,anc2]
                v1v2 = vivj_mean[rd1,rd2,anc1,anc2]

                if bulk_prob_formula(v1,v2,v1v2)<=0:
                    continue

                pij_time[tuple([det1,det2])] = bulk_prob_formula(v1,v2,v1v2)

                for key in p3.keys():

                    if det1 in key and det2 in key:
                        p = pij_time[tuple([det1,det2])]
                        pij_time[tuple([det1,det2])] = (p-p3[key])/(1-2*p3[key])            

            else: #space-bulk or space-time bulk

                v1   = vi_mean[rd1,anc1]
                v2   = vi_mean[rd2,anc2]
                v1v2 = vivj_mean[rd1,rd2,anc1,anc2]

                if bulk_prob_formula(v1,v2,v1v2)<=0:
                    continue
                
                pij_bulk[tuple([det1,det2])] = bulk_prob_formula(v1,v2,v1v2)

                for key in p3.keys():

                    if det1 in key and det2 in key:
                        p                            = pij_bulk[tuple([det1,det2])]
                        pij_bulk[tuple([det1,det2])] = (p-p3[key])/(1-2*p3[key])


    #Bulk
    for key in pij_bulk.keys():
        if pij_bulk[key]<0:
            pij_bulk[key]=0
    

    #Time
    for key in pij_time.keys():
        if pij_time[key]<0:
            pij_time[key]=0


    pij_bd = {}
    #Finally, get the boundary edges:

    
    for q in range(len(all_keys)):
    
        det1     = all_keys[q]

        if (det1,) not in errors_in_DEM.keys() and (det1,"L0") not in errors_in_DEM.keys():
            continue

        rd1,anc1 = dets[det1]
        v1       = vi_mean[rd1,anc1]

        DENOM  = 1
        DENOM *= prod(1 - 2 * p for k, p in pij_bulk.items() if det1 in k)
        DENOM *= prod(1 - 2 * p for k, p in pij_time.items() if det1 in k)
        DENOM *= prod(1 - 2 * p for k, p in p3.items() if det1 in k)
        
        pij_bd[det1] = 1/2 + (v1-1/2)/DENOM

    # Boundary:
    
    for key in pij_bd.keys():
        if pij_bd[key]<=0:
            pij_bd[key]=0

    return pij_bulk,pij_time,pij_bd,p3


def get_dets_from_stim_circuit(circuit):
    
    DEM = circuit.detector_error_model()
    cnt = 0
    
    for instruction in DEM:

        if instruction.type=="detector":
            
            det_annotations = DEM[cnt+1:]
            break
        cnt+=1

    return det_annotations


def get_events_that_cause_L0_flips(circuit):
    '''Use the circuit instructions to find which errors cause logical flips.
       These error events are output as a list of tuple ("Dj","Dk"), or ("Dj","").
    '''
    dem    = circuit.detector_error_model()
    events = []

    for instruction in dem:
        if instruction.type=="error":
            targets=instruction.targets_copy()
                
            if targets[-1].is_logical_observable_id():
                temp = []
                for k in range(len(targets)-1):
                    
                    temp.append(str(targets[k]))
                events.append(tuple(temp))

    return events


def get_observable_flips(data_qubit_samples,distance):
    '''The observable include in the circuit is on the first d-final data qubit measurements.
    Thus, we have to XOR the first d data_qubit_samples.'''
    
    restricted_data  = data_qubit_samples.data[:,:distance]
    # cols             = np.shape(restricted_data)[1]

    # obs_flips_alt = restricted_data[:,0]
    # for l in range(1,cols):
    #     obs_flips_alt = np.logical_xor(obs_flips_alt,restricted_data[:,l])

    obs_flips = np.logical_xor.reduce(restricted_data, axis=1)

    # if not (obs_flips_alt==obs_flips_vect).all():
    #     raise Exception("Vectorized code is wrong.")
    # else:
    #     print("All good.")


    return obs_flips

def extract_single_counts_w_and_wo_L0_excluding_nodes(num_shots,obs_flips,defects_matrix,target_node,exclude_nodes):

    rd1                = target_node[0]
    anc1               = target_node[1]
    single_count_w_L0  = 0
    single_count_wo_L0 = 0

    for k in range(num_shots):

        if defects_matrix.data[k,rd1,anc1]==True and obs_flips[k]==True:

            if exclude_nodes is not None:
                for node in exclude_nodes:
                    rd  = node[0]
                    anc = node[1]
                    
                    if defects_matrix.data[k,rd,anc]==False:
                        Flag=True
                    else:
                        Flag=False
                        break
            else:
                Flag=True

            if Flag==True:
                single_count_w_L0 += 1/num_shots

        elif defects_matrix.data[k,rd1,anc1]==True and obs_flips[k]==False:
        
            if exclude_nodes is not None:
                for node in exclude_nodes:
                    rd  = node[0]
                    anc = node[1]
                    
                    if defects_matrix.data[k,rd,anc]==False:
                        Flag=True
                    else:
                        Flag=False
                        break

            else:
                Flag=True

            if Flag==True:
                single_count_wo_L0 += 1/num_shots        
   

    return single_count_wo_L0, single_count_w_L0


def get_3Point_prob(defects_matrix,num_shots,triplets):
    #triplets: list of tuples [(rd1,anc1),(rd2,anc2),(rd3,anc3)]

    p3 = 0

    rd1 = triplets[0][0]
    anc1  = triplets[0][1]

    rd2 = triplets[1][0]
    anc2  = triplets[1][1]

    rd3 = triplets[2][0]
    anc3  = triplets[2][1]

    # for k in range(num_shots):

    #     this_shot = defects_matrix.data[k,:,:]

    #     if this_shot[rd1,anc1] and this_shot[rd2,anc2] and this_shot[rd3,anc3]:

    #         p3+=1

    #Faster alternative method:
    p3 = np.logical_and(defects_matrix.data[:,rd1,anc1],defects_matrix.data[:,rd2,anc2])
    p3 = np.logical_and(p3,defects_matrix.data[:,rd3,anc3])
    p3 = sum(p3)
    # print("alt:",p3_alt)
    # if p3==sum(p3_alt):
    #     print("True!")
    # else:
    #     raise Exception("Alternative method is wrong")
              
    
    return p3/num_shots


def get_3Point_prob_exclude(defects_matrix,num_shots,include_pairs,exclude_pairs):
    #Include_pairs: list of tuples [(rd1,anc1),(rd2,anc2),...]
    #Exclude_pairs: list of tuples [(rd1,anc1),(rd2,anc2),...]

    p3 = 0
    # rds  = np.arange(np.shape(defects_matrix)[1])
    # ancs = np.arange(np.shape(defects_matrix)[2])

    for k in range(num_shots):

        this_shot = defects_matrix.data[k,:,:]
        Flag      = True

        if this_shot[include_pairs[0][0],include_pairs[0][1]]==True and  \
           this_shot[include_pairs[1][0],include_pairs[1][1]]==True and \
           this_shot[include_pairs[2][0],include_pairs[2][1]]==True:
            
            Flag=True
        else:
            Flag=False

        if Flag==False:
            continue

        if exclude_pairs is not None:
            # flags_all = []
            for other_pair in exclude_pairs:

                if this_shot[other_pair[0],other_pair[1]]==True: #This excludes any case where at least 1 node fires
                    Flag = False
                    break
                
                #We might want to exclude cases where all nodes in exclude pairs fire instead
                #The one below seems to be worse
            #     if this_shot[other_pair[0],other_pair[1]]==True:
            #         flags_all.append(True)
            #     else:
            #         flags_all.append(False)
            
            # if False in flags_all:
            #     Flag=True
            # else: #all True
            #     Flag=False

        if Flag==True: #True so we can increase

            p3 +=1

    return p3/num_shots



def get_all_3_point_probs_new(defects_matrix,nodes):
    '''Consider only 3 point events where 
    node1 \in red
    node2 \in blue and 
    node3 \in green'''
    p_3cnts     = {}
    num_shots   = np.shape(defects_matrix)[0]
    num_rounds  = np.shape(defects_matrix)[1]
    num_ancilla = np.shape(defects_matrix)[2]

    #Get all 3 point events:
    #If all detectors are in same rd, then calculate w/o exclusion
    #If detectors are across rds, then calculate by excluding all other nodes

    all_nodes = [] #list of tuples

    for rd in range(num_rounds):
        for anc in range(num_ancilla):
            all_nodes.append((rd,anc))

    red_nodes   = []
    blue_nodes  = []
    green_nodes = []

    for node in nodes['r']:
        red_nodes.append(int(node[1:]))
    for node in nodes['g']:
        green_nodes.append(int(node[1:]))
    for node in nodes['b']:
        blue_nodes.append(int(node[1:]))

    print("red:",red_nodes)
    print("blue:",blue_nodes)
    print("green:",green_nodes)

    for m in range(len(all_nodes)):
        
        pair1   = all_nodes[m]
        rd1     = pair1[0]
        anc1    = pair1[1]
        indx1   = anc1 + num_ancilla * rd1

        if indx1 in red_nodes:
            
            for l in range(len(all_nodes)):

                pair2   = all_nodes[l]
                rd2     = pair2[0]
                anc2    = pair2[1]
                indx2   = anc2 + num_ancilla * rd2

                if indx2 in blue_nodes:
                    
                    for k in range(len(all_nodes)):

                        pair3   = all_nodes[k]
                        rd3     = pair3[0]
                        anc3    = pair3[1]
                        indx3   = anc3 + num_ancilla * rd3

                        if indx3 in green_nodes:
                            
                            triplet = [pair1,pair2,pair3]

                            inds              = np.sort([indx1,indx2,indx3])
                            INDX1,INDX2,INDX3 = inds
                            # inds              = np.sort([indx1,indx2,indx3])
                            # indx1,indx2,indx3 = inds

                            if rd1==rd2 and rd2==rd3: #All in same rd, calculate w/o exclusion
                                p_3cnts[("D"+str(INDX1),"D"+str(INDX2),"D"+str(INDX3))] = get_3Point_prob(defects_matrix,num_shots,triplet)
                            else:
                                exclude_pairs=[]

                                for pair in all_nodes:

                                    if pair!=pair1 and pair!=pair2 and pair!=pair3:
                                        exclude_pairs.append(pair)

                                p_3cnts[("D"+str(INDX1),"D"+str(INDX2),"D"+str(INDX3))] = get_3Point_prob_exclude(defects_matrix,num_shots,triplet,exclude_pairs)
                else:
                    continue


        else:
            continue

    return p_3cnts

def get_all_3_point_probs(defects_matrix):
    
    p_3cnts     = {}
    num_shots   = np.shape(defects_matrix)[0]
    num_rounds  = np.shape(defects_matrix)[1]
    num_ancilla = np.shape(defects_matrix)[2]

    #Get all 3 point events:
    #If all detectors are in same rd, then calculate w/o exclusion
    #If detectors are across rds, then calculate by excluding all other nodes
    
    all_nodes = [] #list of tuples

    for rd in range(num_rounds):
        for anc in range(num_ancilla):
            all_nodes.append((rd,anc))

    for m  in range(len(all_nodes)):
        
        pair1 = all_nodes[m]
        rd1   = pair1[0]
        anc1  = pair1[1]
        

        for l in range(m+1,len(all_nodes)):

            pair2 = all_nodes[l]
            rd2   = pair2[0]
            anc2  = pair2[1]

            for k in range(l+1,len(all_nodes)):

                pair3 = all_nodes[k]

                if pair1!=pair2 and pair2!=pair3 and pair1!=pair3:
                    
                    triplet = [pair1,pair2,pair3]
                    rd3     = pair3[0]
                    anc3    = pair3[1]

                    indx1   = anc1 + num_ancilla * rd1
                    indx2   = anc2 + num_ancilla * rd2
                    indx3   = anc3 + num_ancilla * rd3

                    inds              = np.sort([indx1,indx2,indx3])
                    indx1,indx2,indx3 = inds

                    
                    if rd1==rd2 and rd2==rd3: #All in same rd, calculate w/o exclusion
                        p_3cnts[("D"+str(indx1),"D"+str(indx2),"D"+str(indx3))] = get_3Point_prob(defects_matrix,num_shots,triplet)
                    else: #calculate by exlcusion

                        exclude_pairs=[]

                        for pair in all_nodes:

                            if pair!=pair1 and pair!=pair2 and pair!=pair3:
                                exclude_pairs.append(pair)

                        p_3cnts[("D"+str(indx1),"D"+str(indx2),"D"+str(indx3))] = get_3Point_prob_exclude(defects_matrix,num_shots,triplet,exclude_pairs)




    return p_3cnts




def get_all_single_cnts_w_and_wo_L0(defects_matrix,obs_flips):

    num_shots   = np.shape(defects_matrix)[0]
    num_rounds  = np.shape(defects_matrix)[1]
    num_ancilla = np.shape(defects_matrix)[2]
    
    p1_cnts   = {}
    all_nodes = []
    for rd in range(num_rounds):
        for anc in range(num_ancilla):
            all_nodes.append((rd,anc))
            

    for rd in range(num_rounds):
        for anc in range(num_ancilla):
            
            target_node   = (rd,anc)
            indx          = anc + num_ancilla * rd
            exclude_nodes = [x for x in all_nodes if x != target_node]

            cnts_wo_L0,cnts_w_L0 = extract_single_counts_w_and_wo_L0_excluding_nodes(num_shots,obs_flips,defects_matrix,target_node,exclude_nodes)
            
            p1_cnts[("D"+str(indx),"L0")] = cnts_w_L0
            p1_cnts[("D"+str(indx),"")]  = cnts_wo_L0


    return p1_cnts

#---------------- For bd probs --------------------------------------



def map_pairs_to_detectors(anc,rd,num_ancilla):
    
    name_of_det = "D" + str(anc + num_ancilla * rd)

    return name_of_det


def detector_name_to_rd_anc_name(name_of_det,num_ancilla,num_rounds):
    '''output the (round_indx,ancilla_indx) conversion'''
    Flag=False
    for rd in range(num_rounds+1):

        for anc in range(num_ancilla):

            indx = anc+num_ancilla*rd

            if int(name_of_det[1:])==indx:
                Flag=True
                pair=(rd,anc) 
                break

    if Flag==False:
        raise Exception("Conversion from detector name to (rd,anc) has failed.")

    return pair




#TODO: Consider for more rds and distances
def get_color_restricted_DEMs(pij_bulk,pij_bd,nodes):
    #nodes: a dictionary with keys 'r','b','g'
    #       and values the node names "Dj"
    #DEMs_1: a dictionary that includes all color-restricted lattices.

    dems_1 = {}
    colors = ['r','g','b']

    for color in colors:

        temp_dem    = stim.DetectorErrorModel()

        if color=='r':
            other_nodes = nodes['g']+nodes['b']
        elif color=='b':
            other_nodes = nodes['r']+nodes['g']
        else:
            other_nodes = nodes['r']+nodes['b']
        
        other_nodes = list(np.sort(other_nodes))

        for l in range(len(other_nodes)):

            for m in range(l+1,len(other_nodes)):

                v1 = other_nodes[l]
                v2 = other_nodes[m]

                INDX1 = int(v1[1:])
                INDX2 = int(v2[1:])

                indx1 = min([INDX1,INDX2])
                indx2 = max([INDX1,INDX2])

                p = pij_bulk[("D"+str(indx1),"D"+str(indx2))]
                if p>0:

                    temp_dem.append("error",p,
                                        [stim.target_relative_detector_id(indx1),
                                        stim.target_relative_detector_id(indx2)])


        #Add also the bd edges:
        for v in other_nodes:
            indx1 = int(v[1:])
            p = pij_bd[color][(v)]
            if p>0:
                temp_dem.append("error",p,
                                [stim.target_relative_detector_id(indx1)] )

     
        dems_1[color]=temp_dem



    return dems_1




def get_updated_probs_of_two_point_events(pij,p_3cnts,nodes):
    '''  pij: a dictionary with bulk-type probabilities
     p_3cnts: a dictionary that has the probabilities of each 3 point event
       nodes: a dictionary which with keys r,g,b and the respective nodes.'''
    
    all_nodes = nodes['r']+nodes['g']+nodes['b']

    #Create the pair D_i - D_j and check where this pair is included
    #in the 3-point cnts
    p2_updated = {}

    for m in range(len(all_nodes)):
                   
        v = all_nodes[m]

        for k in range(m+1,len(all_nodes)):
            w = all_nodes[k]

            if v!=w:
                
                numer = 0
                denom = 1
                #Loop through all the keys:
                
                for key in p_3cnts.keys():

                    if v in key and w in key:
                        
                        numer -= p_3cnts[key]
                        denom *= 1-2*p_3cnts[key]

                if (v,w) in pij.keys():
                # INDX1 = int(v[1:])
                # INDX2 = int(w[1:])
                # indx1 = min([INDX1,INDX2])
                # indx2 = max([INDX1,INDX2])
                    val = (pij[(v,w)]+numer)/denom
                    if val<0:
                        val = 0
                    
                    p2_updated[(v,w)]=val
                elif (w,v) in pij.keys():
                    val=(pij[(w,v)]+numer)/denom
                    if val<0:
                        val=0
                    p2_updated[(w,v)]=val

    return p2_updated





