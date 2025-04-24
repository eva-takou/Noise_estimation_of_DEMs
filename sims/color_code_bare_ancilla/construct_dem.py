import stim
import numpy as np
from utilities.general_utils import DEM_to_dictionary
from utilities_for_color_code import collect_color_of_nodes

def decompose_org_DEM_into_ZX_DEMs(circuit: stim.Circuit, Z_dets: list, X_dets: list):
    '''Decompose stim's DEM into the Z and X part. Correlations present in Y errors are dropped.

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

        targets = [stim.target_relative_detector_id(int(det[1:])) if det[0]=="D" \
                   else stim.target_logical_observable_id(0)\
                   for det in key]
        
        DEM_Z.append("error",prob,targets=targets)

    for det_coord in detector_coords:
        DEM_Z.append(det_coord)
    
    for key in dict_for_X.keys():
        
        prob    = dict_for_X[key]
        targets = [stim.target_relative_detector_id(int(det[1:]))  \
                   for det in key if det[0]=="D"]

        DEM_X.append("error",prob,targets=targets)


    return DEM_Z,DEM_X

def create_Z_DEM(pij_bd: dict, pij_bulk: dict, pij_time: dict, p3: dict, Z_DEM: stim.DetectorErrorModel):
    '''Create the Z-DEM of the color code using the estimated error probabilities.
       Stim's noise Z-DEM is only used to append detector coordinates.

    Input:
        pij_bd: dictionary with boundary probabilities
        pij_bulk: dictionary with bulk probabilities
        pij_time: dictionary with time probabilities
        p3: dictionary of 3-point probabilities
        Z_DEM: stim's noise-aware DEM

    Output:
        Z_DEM: estimated Z-DEM
        '''
    
    #Collect in a dictionary the error instructions
    
    error_events_in_Z_DEM = DEM_to_dictionary(Z_DEM)
    my_DEM                = stim.DetectorErrorModel()

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

def create_X_DEM(pij_bd: dict, pij_bulk: dict, pij_time: dict, p3: dict, X_DEM: stim.DetectorErrorModel):
    '''Create the X-DEM of the color code using the estimated error probabilities.
       Stim's noise X-DEM is used only to append detector coordinates.

    Input:
        pij_bd: dictionary with boundary probabilities
        pij_bulk: dictionary with bulk probabilities
        pij_time: dictionary with time probabilities
        p3: dictionary of 3-point probabilities
        X_DEM: stim's noise-aware DEM

    Output:
        X_DEM: estimated Z-DEM
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

def create_lattices(color, Z_DEM: stim.DetectorErrorModel, X_DEM: stim.DetectorErrorModel, obj):
    '''Create the color-only and color-restricted lattices.
       This script is based on "Color code decoder with improved scaling for correcting circuit-level noise"
       by Seok-Hyung Lee, Andrew Li, and Stephen D. Bartlett.

       Input: 
            color: 'r' or 'g' or 'b' for the red, green, or blue color
            Z_DEM: our estimated Z-DEM
            X_DEM: our estimated X-DEM
            obj: the color code object
        Output:
            c_only_DEM:       the DEM of the color-only lattice
            c_restricted_DEM: the DEM of the color-restricted lattice
    '''
    
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

