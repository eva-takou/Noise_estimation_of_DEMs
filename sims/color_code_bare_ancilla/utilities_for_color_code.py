import stim
import numpy as np
from noise_est_funcs_for_color_code import *
from math import prod
from utilities.general_utils import *

#These functions work only for d>=5.

def decompose_org_DEM_into_ZX_DEMs(circuit,Z_dets,X_dets):
    
    '''Decompose stim's DEM into the Z and X part. For error processes that
    flip both X and Z detectors, we put the probability p as the X flips and Z flips.
    We drop correlations. The goal then is to try to estimate DEM_Z and DEM_X independently,
    given the 2 types of defect matrices that we have access to.

    Input: 
    circuit: the stim circuit of the color code
    Z_dets: a list of the Z-type detector names e.g. ["D0","D1",...]
    X_dets: a list of the X-type detector names e.g. ["D6",...]

    Output:
    DEM_Z: restricted DEM on Z-type detectors
    DEM_X: restricted DEM on X-type detectors.
    
    '''

    DEM   = circuit.detector_error_model(flatten_loops=True)
    DEM_Z = stim.DetectorErrorModel()
    DEM_X = stim.DetectorErrorModel()

    dict_for_Z = {}
    dict_for_X = {}

    detector_coords = []

    for instruction in DEM:

        if instruction.type=="error":

            prob           = instruction.args_copy()[0]
            targets        = instruction.targets_copy()
            dets_for_Z_dem = []
            dets_for_X_dem = []

            for target in targets:

                if target.is_relative_detector_id():

                    det_name = "D"+str(target.val)
                    
                    if det_name in Z_dets:

                        dets_for_Z_dem.append(det_name)

                    elif det_name in X_dets:
                        

                        dets_for_X_dem.append(det_name)
                        
                else: #logical observable
                    dets_for_Z_dem.append("L0")
            
            if dets_for_Z_dem!=[]:
                key = tuple(dets_for_Z_dem)

                if key in dict_for_Z.keys():
                    p = dict_for_Z[key]
                    q = prob 

                    dict_for_Z[key] = p+q-2*p*q
                else:
                    dict_for_Z[key] = prob
                
            
            if dets_for_X_dem!=[]:

                

                key = tuple(dets_for_X_dem)
                if key in dict_for_X.keys():
                    p = dict_for_X[key]
                    q = prob 
                    dict_for_X[key] = p+q-2*p*q 
                else:
                    dict_for_X[key] = prob

        else: #it's detector coordinates
            detector_coords.append(instruction)


    #Now make the DEMs:

    for key in dict_for_Z.keys():
        prob    = dict_for_Z[key]
        targets = []

        for det in key:
            
            if det[0]=="D":
                indx = int(det[1:])
                targets.append(stim.target_relative_detector_id(indx))
            elif det[0]=="L":
                targets.append(stim.target_logical_observable_id(0))
        
        DEM_Z.append("error",prob,targets=targets)

    for det_coord in detector_coords:
        DEM_Z.append(det_coord)
    
    for key in dict_for_X.keys():
        prob   = dict_for_X[key]
        targets=[]

        for det in key:
            
            if det[0]=="D":
                indx = int(det[1:])
                targets.append(stim.target_relative_detector_id(indx))
            
        DEM_X.append("error",prob,targets=targets)



    return DEM_Z,DEM_X



def get_single_pnt_events_w_L0(defects_matrix,obj,obs_flips,defects_type,num_rounds):
    '''This is simply counting unconditionally counts to see how many times Dj + L0 fire.'''
    
    num_shots   = np.shape(defects_matrix)[0]
    num_ancilla = len(obj.qubit_groups['anc'])
    
    dets_Z,dets_X=get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds)

    if defects_type=="Z":
        dets = dets_Z
        num_ancilla = np.shape(defects_matrix)[2]
    elif defects_type=="X":
        dets = dets_X 
        num_ancilla = np.shape(defects_matrix)[2]
        
    else:
        raise Exception("defects_type can be only Z or X.")


     
    single_cnts  = {}
    for k in range(num_shots):

        locs = np.nonzero(defects_matrix.data[k,:,:])

        if len(locs[0])==1 and len(locs[1])==1 and obs_flips[k]==True:

            rd   = locs[0][0]
            anc  = locs[1][0]
            
            if defects_type=="Z":

                # indx  = anc + num_ancilla * rd
                # names = list(dets_Z.keys())
                # name  = names[indx]
                name = [x for x in dets.keys() if dets[x]==(rd,anc)]
                name = name[0]
            else: #X type
                
                indx  = anc + num_ancilla * rd
                names = list(dets_X.keys())
                name  = names[indx]

            if name in single_cnts.keys():
                single_cnts[name]=single_cnts[name]+1/num_shots
            else:
                single_cnts[name]=1/num_shots


    return single_cnts


def create_Z_DEM(pij_bd,pij_bulk,pij_time,p3,Z_DEM):
    '''Z_DEM: stim's noise aware Z-DEM
       Order the instructions in our estimated Z_DEM according to the order of the Z_DEM. 
        '''
    
    
    #Collect in a dictionary the error instructions
    
    error_events_in_Z_DEM = DEM_to_dictionary(Z_DEM)
    my_DEM = stim.DetectorErrorModel()

    #Now create our own dem:
    for key in error_events_in_Z_DEM.keys():
        
        targets   = []
        dets_only = []
        for det in key:

            if det[0]=="D":
                ind = int(det[1:])
                targets.append(stim.target_relative_detector_id(ind))
                dets_only.append("D"+str(ind))
            else:
                targets.append(stim.target_logical_observable_id(0))
                
        dets_only = tuple(dets_only)
        
        if dets_only in pij_bulk.keys():
            
            p = pij_bulk[dets_only]

        elif dets_only in pij_time.keys():
            
            p = pij_time[dets_only]

        elif dets_only in p3.keys():
            
            p = p3[dets_only]
        else: #has to be a bd edge 
            
            p    = pij_bd[dets_only[0]]    
        
        my_DEM.append("error",p,targets)


    return error_events_in_Z_DEM,my_DEM

def create_X_DEM(pij_bd,pij_bulk,pij_time,p3,X_DEM):
    '''X_DEM: stim's noise aware X-DEM
       Order the instructions in our estimated X_DEM according to the order of the X_DEM. 
        '''
    

    #Collect in a dictionary the error instructions
    error_events_in_X_DEM = DEM_to_dictionary(X_DEM)


    my_DEM = stim.DetectorErrorModel()

    #Now create our own dem:
    for key in error_events_in_X_DEM.keys():
        
        targets   = []
        dets_only = []

        for det in key:

            if det[0]=="D":
                ind = int(det[1:])
                targets.append(stim.target_relative_detector_id(ind))
                dets_only.append("D"+str(ind))
            else:
                
                targets.append(stim.target_logical_observable_id(0))
                
        dets_only = tuple(dets_only)
        
        if dets_only in pij_bulk.keys():
            
            p = pij_bulk[dets_only]

        elif dets_only in pij_time.keys():
            
            p = pij_time[dets_only]

        elif dets_only in p3.keys():
            
            p = p3[dets_only]
        else: #has to be a bd edge 
            
            
            p    = pij_bd[dets_only[0]]    


        
        my_DEM.append("error",p,targets)


    return error_events_in_X_DEM,my_DEM


def estimate_all_edges_for_defect_type(sols_for_defect_type,obj,num_rounds,vi_mean,vivj_mean,stims_DEM,defects_type):

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

            if rd1!=rd2 and anc1==anc2: #time edges:

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


            else: # rd1==rd2: #space-bulk or space-time bulk

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

    # for key in keys_to_delete:
    #     pij_bd.pop(key,None)
    


    return pij_bulk,pij_time,pij_bd,p3


def create_lattices(color,Z_DEM,X_DEM,obj):
    
    colors           = collect_color_of_nodes(obj)
    dets_to_avoid    = colors[color]
    c_restricted_DEM = stim.DetectorErrorModel()
    c_only_DEM       = stim.DetectorErrorModel()
    num_detectors    = len(colors['r'])+len(colors['b'])+len(colors['g'])

    errors_for_c_restricted = {}
    errors_for_c_only       = {}
    dem1_virtual_obs_dict   = {}
    
    #---------------------------------------------------------------------------------
    for instruction in Z_DEM:

        if instruction.type=="error":

            prob            = instruction.args_copy()[0]
            targets         = instruction.targets_copy()
            dets_restricted = []
            dets_only       = []

            flag            = False #flag about whether we encounter a logical flip in the DEM instructions

            for target in targets:

                if target.is_relative_detector_id():

                    id = target.val

                    if "D"+str(id) not in dets_to_avoid:
                        dets_restricted.append("D"+str(id))
                        
                    else: #in dets to avoid

                        dets_only.append("D"+str(id))
                else:
                    
                    flag=True
            
            if dets_restricted:

                if tuple(dets_restricted) not in errors_for_c_restricted.keys():

                    errors_for_c_restricted[tuple(dets_restricted)]=prob
                    
                    dem1_virtual_obs_dict[tuple(dets_restricted)]=len(dem1_virtual_obs_dict)
                    

                else:

                    q                                               = errors_for_c_restricted[tuple(dets_restricted)]
                    p                                               = prob
                    errors_for_c_restricted[tuple(dets_restricted)] = q+p-2*p*q

                    
                virtual_obs=dem1_virtual_obs_dict[tuple(dets_restricted)]
                dets_only.append("D"+str(num_detectors+virtual_obs))

            if flag:
                dets_only.append("L0")
                

            if dets_only:

                errors_for_c_only[tuple(dets_only)]=prob

    
    errors_for_c_only_new = {}
    values       = list(errors_for_c_only.values())
    
    sort_inds    = np.argsort(values)[::-1]
    all_keys     = list(errors_for_c_only.keys())
    
    for indx in sort_inds:
        errors_for_c_only_new[all_keys[indx]]=values[indx]
        
    
    errors_for_c_only = errors_for_c_only_new

    #---------------------------Now continue with the X-DEM-----------------------------------
    if X_DEM: #non-empty
        for instruction in X_DEM:

            if instruction.type=="error":

                prob            = instruction.args_copy()[0]
                targets         = instruction.targets_copy()
                dets_restricted = []
                dets_only       = []

                flag            = False #flag about whether we encounter a logical flip in the DEM instructions

                for target in targets:

                    if target.is_relative_detector_id():

                        id = target.val

                        if "D"+str(id) not in dets_to_avoid:
                            dets_restricted.append("D"+str(id))
                            
                        else: #in dets to avoid

                            dets_only.append("D"+str(id))
                    else:
                        
                        flag=True
                
                if dets_restricted:

                    if tuple(dets_restricted) not in errors_for_c_restricted.keys():

                        errors_for_c_restricted[tuple(dets_restricted)]=prob
                        
                        dem1_virtual_obs_dict[tuple(dets_restricted)]=len(dem1_virtual_obs_dict)
                        
                    else:

                        q                                               = errors_for_c_restricted[tuple(dets_restricted)]
                        p                                               = prob
                        errors_for_c_restricted[tuple(dets_restricted)] = q+p-2*p*q

                        
                    virtual_obs=dem1_virtual_obs_dict[tuple(dets_restricted)]
                    dets_only.append("D"+str(num_detectors+virtual_obs))

                if flag:
                    dets_only.append("L0")
                    

                if dets_only:

                    errors_for_c_only[tuple(dets_only)]=prob
    



    #-------------------------construct the c-only lattice-------------------------------------

        
    for key in errors_for_c_only.keys():
        p = errors_for_c_only[key]
        targets = []

        for det in key:
            
            if det[0]=="D":
            
                ind = int(det[1:])
                targets.append(stim.target_relative_detector_id(ind))
            else:
                targets.append(stim.target_logical_observable_id(0))
        
        c_only_DEM.append("error",p,targets=targets)

    #-------------------------construct the c-restricted lattice-------------------------------------
    
    for key in errors_for_c_restricted.keys():
        p        = errors_for_c_restricted[key]  
        targets = []

        for det in key:
            ind = int(det[1:])
            targets.append(stim.target_relative_detector_id(ind))

        log_id = dem1_virtual_obs_dict[key]
        # log_id = cnt
        
        targets.append(stim.target_logical_observable_id(log_id))

        c_restricted_DEM.append("error",p,targets=targets)
        # cnt+=1
        

    return c_only_DEM,c_restricted_DEM
