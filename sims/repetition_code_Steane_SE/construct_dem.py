import numpy as np
import xarray as xr
import stim
import pymatching
from pymatching import Matching


def get_initial_state(anc_qubits: list):
    '''Get initial state for ancilla qubits of all 0s.
    
    Input:
        anc_qubits: a list of the ancilla qubit names
    
    Output:
        initial_state: an xarray of length # of anc qubits

    '''
    #Initial ancilla state
    initial_state = np.zeros(len(anc_qubits), dtype=int)
    initial_state = initial_state==True
    initial_state = xr.DataArray ( data=initial_state,dims=[ "anc_qubit" ] ,coords=dict ( anc_qubit=anc_qubits) , ) 
    
    return initial_state


def project_data_meas(data_meas,num_rounds: int,anc_qubits: list):
    '''Project the last data qubit measurement onto stabilizer values.
    
    Input: 
        data_meas: measurement data obtained from stim (# of shots x num of data qubits)
        num_rounds: # of QEC rounds
        anc_qubits: list of names of anc_qubits
    Output:
        syndrome_proj: syndrome projection (xarray of dims # of shots x # of anc_qubits)
    '''

    num_shots = np.shape(data_meas)[0]
    shots     = np.arange(num_shots) 
    
    syndrome_proj = data_meas.data[:,:-1] ^ data_meas.data[:,1:]  #dims: # of shots x # of ancilla
    #the 0th and 1st entries are xored, to give the 1st ancilla measurement equivalent
    #the 1st and 2nd entries are xored, to give the 2nd ancilla measurement equivalent

    syndrome_proj = xr.DataArray(data=syndrome_proj,
                                 dims=["shot","anc_qubit"],
                                 coords = dict(shot=shots,anc_qubit=anc_qubits))

    syndrome_proj["qec_round"]  = num_rounds    

    return syndrome_proj


def get_defects_matrix(distance: int,num_rounds: int,num_shots: int,circuit: stim.Circuit):
    '''Construct the defects matrix given the detection outcomes of the repetition code
    under Steane style syndrome extraction.
    Input:
        distance:   distance of the code
        num_rounds: # of QEC rounds
        num_shots:  # of shots to sample the circuit
        circuit: the stim circuit
    Output:
        defects_matrix: the detector matrix of dimensions (# of shots x # of QEC rounds +1 x # of detectors per round)

    '''
    
    d             = distance    
    data_qubits   = np.arange(0,d).tolist()
    anc_qubits    = (np.arange(d)+d).tolist()        
    SHOTS         = np.arange(num_shots)
    QEC_ROUNDS    = np.arange(num_rounds)

    sampler   = circuit.compile_sampler()
    samples   = sampler.sample(shots=num_shots)

    NUM_ANC = len(anc_qubits)
    NN      = NUM_ANC*num_rounds

    anc_qubit_samples  = samples[:,:NN]
    data_qubit_samples = samples[:,NN:]
    anc_qubit_samples  = anc_qubit_samples.reshape(num_shots,num_rounds,NUM_ANC) 

    #2 consecutive ancilla measurements create the syndrome
    anc_qubit_samples = anc_qubit_samples[:,:,:-1] ^ anc_qubit_samples[:,:,1:]

    #Now redefine the anc qubits as detectors:
    anc_qubits    = (np.arange(d-1)+d).tolist()      

    anc_meas = xr.DataArray(data = anc_qubit_samples, 
                        dims=["shot","qec_round","anc_qubit"],
                        coords = dict(shot=SHOTS,qec_round=QEC_ROUNDS,anc_qubit=anc_qubits),)


    data_meas = xr.DataArray(data=data_qubit_samples, 
                        dims=["shot","data_qubits"],
                        coords = dict(shot=SHOTS,data_qubits=data_qubits))    

    initial_state = get_initial_state(anc_qubits)

    syndrome_proj                     = project_data_meas(data_meas,num_rounds,anc_qubits)
    syndromes                         = xr.concat([anc_meas,syndrome_proj],"qec_round")


    syndrome_matrix_copy              = syndromes.copy()
    syndrome_matrix_copy.data[:,-1,:] = initial_state
    defects_matrix                    = syndromes ^ syndrome_matrix_copy.roll(qec_round=1)

    return defects_matrix



def construct_dem(pij_bulk: dict,pij_bd: dict,pij_time: dict,p4_cnts: dict):
    '''Contruct the detector error model given the estimated values of error probabilities.
    
    Input:
        pij_bulk: dictionary with keys the detector names of space-bulk errors and values the error probabilities
        pij_bd: dictionary with keys the detector names of boundary errors and values the error probabilities
        pij_time: dictionary with keys the detector names of time errors and values the error probabilities
        p4_cnts: dictionary with keys the detector names of 4-point errors and values the error probabilities

    Output:
        reconstructed_DEM: our estimated detector error model
    '''
    reconstructed_DEM = stim.DetectorErrorModel()

    for key in pij_bulk.keys():
        det_list = []
        for det in key:
            ind = int(det[1:])
            det_list.append(stim.target_relative_detector_id(ind))
        
        det_list.append(stim.target_logical_observable_id(0))

        reconstructed_DEM.append("error",pij_bulk[key],det_list)

    for key in pij_bd.keys():
        
        ind      = int(key[1:])
        det_list = [stim.target_relative_detector_id(ind),stim.target_logical_observable_id(0)]

        reconstructed_DEM.append("error",pij_bd[key],det_list)

    for key in pij_time.keys():
        
        det_list = []
        for det in key:
            ind = int(det[1:])
            det_list.append(stim.target_relative_detector_id(ind))
        
        reconstructed_DEM.append("error",pij_time[key],det_list)
    
    for key in p4_cnts.keys():
        det_list=[]
        for det in key:
            ind=int(det[1:])
            det_list.append(stim.target_relative_detector_id(ind))
        
        reconstructed_DEM.append("error",p4_cnts[key],det_list)

    return reconstructed_DEM




#----------- Need to fix those below----------

#Consider also the flag qubit here: 1st measurement in every round is the flag.
def get_defects_matrix_w_Flag(distance,num_rounds,num_shots,circuit):

    #Need to have a tracking system: if the ancilla meas is '1'
    d             = distance    
    data_qubits   = np.arange(0,d).tolist()
    anc_qubits    = (np.arange(d)+d).tolist()  
    SHOTS         = np.arange(num_shots)
    QEC_ROUNDS    = np.arange(num_rounds)

    sampler       = circuit.compile_sampler()
    samples       = sampler.sample(shots=num_shots)
    NUM_ANC       = len(anc_qubits)


    anc_qubit_samples  = samples[:,:((NUM_ANC+1)*num_rounds)]
    data_qubit_samples = samples[:,((NUM_ANC+1)*num_rounds):]

    

    locs_no_error = list(np.where(anc_qubit_samples[:,0]==0)[0])

    for k in range(1,num_rounds):

        new_locs = list(np.where(anc_qubit_samples[:,0+k*(distance+1)]==0)[0])

        for loc in new_locs:
            if loc not in locs_no_error:
                locs_no_error += [loc]

    
        
    anc_qubit_samples  = anc_qubit_samples[locs_no_error,:]
    data_qubit_samples = data_qubit_samples[locs_no_error,:]

    


    print("Old # of shots: ",num_shots)
    # num_shots = len(locs_no_error)
    num_shots = len(locs_no_error)
    SHOTS     = np.arange(num_shots)
    print("New # of shots: ",num_shots)


    anc_qubit_samples  = anc_qubit_samples.reshape(num_shots,num_rounds,NUM_ANC+1)
    #drop the flag qubit
    anc_qubit_samples = anc_qubit_samples[:,:,1:]

    #This is because 2 consecutive ancilla measurements create the syndrome
    anc_qubit_samples = anc_qubit_samples[:,:,:-1] ^ anc_qubit_samples[:,:,1:]


    #Now redefine the anc qubits as detectors
    anc_qubits    = (np.arange(d-1)+d).tolist()      

    anc_meas = xr.DataArray(data = anc_qubit_samples, 
                        dims=["shot","qec_round","anc_qubit"],
                        coords = dict(shot=SHOTS,qec_round=QEC_ROUNDS,anc_qubit=anc_qubits),)


    data_meas = xr.DataArray(data=data_qubit_samples, 
                        dims=["shot","data_qubits"],
                        coords = dict(shot=SHOTS,data_qubits=data_qubits))    

    initial_state = get_initial_state(anc_qubits)

    syndrome_proj                     = project_data_meas(data_meas,num_shots,num_rounds,d,anc_qubits)
    syndromes                         = xr.concat([anc_meas,syndrome_proj],"qec_round")

    syndrome_matrix_copy              = syndromes.copy()
    syndrome_matrix_copy.data[:,-1,:] = initial_state
    defects_matrix                    = syndromes ^ syndrome_matrix_copy.roll(qec_round=1)



    return defects_matrix,num_shots


#Consider also the flag qubit here: 1st measurement in 1st round only is the flag
def get_defects_matrix_w_Flag(distance,num_rounds,num_shots,circuit):

    #Need to have a tracking system: if the ancilla meas is '1'
    d             = distance    
    data_qubits   = np.arange(0,d).tolist()
    anc_qubits    = (np.arange(d)+d).tolist()  
    # flag_qubit    = [anc_qubits[-1]+1]  #Consider a flag qubit too.
    SHOTS         = np.arange(num_shots)
    QEC_ROUNDS    = np.arange(num_rounds)

    sampler       = circuit.compile_sampler()
    samples       = sampler.sample(shots=num_shots)
    NUM_ANC       = len(anc_qubits)


    flag_samples       = samples[:,0]
    samples            = samples[:,1:]
    anc_qubit_samples  = samples[:,:NUM_ANC*num_rounds]
    data_qubit_samples = samples[:,NUM_ANC*num_rounds:]


    if np.shape(data_qubit_samples)[1]!=len(data_qubits):
        print("# of data qubits:",np.shape(data_qubit_samples)[1])
        raise Exception("Error in the assignment of data samples")
    
    if (np.shape(anc_qubit_samples)[1])   !=len(anc_qubits)*num_rounds:
        print("# of anc qubits:",np.shape(anc_qubit_samples)[1])
        raise Exception("Error in the assignment of anc samples")

    #Keep only samples where flag=0
    locs_no_error = np.where(flag_samples==0)[0]
    
    anc_qubit_samples  = anc_qubit_samples[locs_no_error,:]
    data_qubit_samples = data_qubit_samples[locs_no_error,:]

    print("Old # of shots: ",num_shots)
    # num_shots = len(locs_no_error)
    num_shots = np.shape(anc_qubit_samples)[0]
    SHOTS     = np.arange(num_shots)
    print("New # of shots: ",num_shots)


    anc_qubit_samples  = anc_qubit_samples.reshape(num_shots,num_rounds,NUM_ANC) 

    #This is because 2 consecutive ancilla measurements create the syndrome
    anc_qubit_samples = anc_qubit_samples[:,:,:-1] ^ anc_qubit_samples[:,:,1:]


    #Now redefine the anc qubits as detectors
    anc_qubits    = (np.arange(d-1)+d).tolist()      

    anc_meas = xr.DataArray(data = anc_qubit_samples, 
                        dims=["shot","qec_round","anc_qubit"],
                        coords = dict(shot=SHOTS,qec_round=QEC_ROUNDS,anc_qubit=anc_qubits),)


    data_meas = xr.DataArray(data=data_qubit_samples, 
                        dims=["shot","data_qubits"],
                        coords = dict(shot=SHOTS,data_qubits=data_qubits))    

    initial_state = get_initial_state(anc_qubits)

    syndrome_proj                     = project_data_meas(data_meas,num_shots,num_rounds,d,anc_qubits)
    syndromes                         = xr.concat([anc_meas,syndrome_proj],"qec_round")

    syndrome_matrix_copy              = syndromes.copy()
    syndrome_matrix_copy.data[:,-1,:] = initial_state
    defects_matrix                    = syndromes ^ syndrome_matrix_copy.roll(qec_round=1)



    return defects_matrix,num_shots


def decode_both_dems_V2_w_post_selection(my_DEM:stim.DetectorErrorModel,circuit:stim.Circuit,num_shots:int):
    '''
    Decode via MWPM the exact model of stim and our reconstructed DEM. The detection events are the same
    for both DEMs.

    Inputs: 
        my_DEM:    Reconstructed detector error model
        circuit:   The stim circuit from which we extract the exact detector error model
        num_shots: # of shots to use to decode
    Output:
        num_errors_my_DEM: total # of logical errors obtained by decoding the reconstructed DEM
        num_errors_stim:   total # of logical errors obtained by decoding stim's DEM
    '''
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    #Keep only detection events where D0 takes the value of 0
    locs_no_error = np.where(detection_events[:,0]==0)[0]

    detection_events = detection_events[locs_no_error,:]
    # detection_events = detection_events[:,1:] #Drop D0?

    print("shape of detection events:",np.shape(detection_events))

    observable_flips = observable_flips[locs_no_error]
    num_shots        = np.shape(detection_events)[0]

    

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(flatten_loops=True) 


    #I think that now we need to remove the detector "D0" from stim's DEM?
    # new_stims_DEM = stim.DetectorErrorModel()
    # for instruction in detector_error_model:

    #     if instruction.type=="error":

    #         targets = instruction.targets_copy()
    #         prob    = instruction.args_copy()[0]
    #         new_targets =[]
    #         flag = False
    #         for target in targets:

    #             if target.is_relative_detector_id():

    #                 ind = target.val 

    #                 if ind==0:
    #                     flag=True
    #                     break
    #                 else:
    #                     new_targets.append(target) #Shift all other detector names by -1 (?)

    #             else:
    #                 new_targets.append(target)


    #         if new_targets==[] or flag==True:  #Do not append completely this error that include D0
    #             continue
    #         else:
    #             new_stims_DEM.append("error",prob,new_targets)
        
    #     else:

    #         if instruction.type=="detector":

    #             detectors = instruction.targets_copy()
    #             flag = False
    #             for det in detectors:

    #                 print("det:",det)

    #                 if det.is_relative_detector_id():

    #                     ind = det.val 
    #                     if ind==0:
    #                         flag=True
    #                         break
                        
                        
    #             if flag==False:
    #                 new_stims_DEM.append(instruction)

    #         else:
    #             new_stims_DEM.append(instruction)
            


    # print("new_DEM:",new_stims_DEM)

    # detector_error_model = new_stims_DEM

    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors_stim = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors_stim += 1

    #Now do the same for my model.
    detection_events = detection_events[:,1:] #Drop the detector on the flag qubit
    matcher     = Matching.from_detector_error_model(my_DEM)
    predictions = matcher.decode_batch(detection_events)

    num_errors_my_DEM = 0
    for shot in range(num_shots):
        actual_for_shot    = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors_my_DEM += 1    


    return num_errors_my_DEM,num_errors_stim,num_shots


#This seems to not give any gain? This is for dropping "D0" when there is only 1 rd where we put this detector
def decode_both_dems_V2_w_post_selection(my_DEM:stim.DetectorErrorModel,circuit:stim.Circuit,num_shots:int):
    '''
    Decode via MWPM the exact model of stim and our reconstructed DEM. The detection events are the same
    for both DEMs.

    Inputs: 
        my_DEM:    Reconstructed detector error model
        circuit:   The stim circuit from which we extract the exact detector error model
        num_shots: # of shots to use to decode
    Output:
        num_errors_my_DEM: total # of logical errors obtained by decoding the reconstructed DEM
        num_errors_stim:   total # of logical errors obtained by decoding stim's DEM
    '''
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    #Keep only detection events where D0 takes the value of 0
    locs_no_error = np.where(detection_events[:,0]==0)[0]

    detection_events = detection_events[locs_no_error,:]
    

    print("shape of detection events:",np.shape(detection_events))

    observable_flips = observable_flips[locs_no_error]
    num_shots        = np.shape(detection_events)[0]


    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(flatten_loops=True) 


    #Remove "D0" the flag qubit detector from the DEM instructions where it exists
    new_stims_DEM = stim.DetectorErrorModel()
    for instruction in detector_error_model:

        if instruction.type=="error":

            targets = instruction.targets_copy()
            prob    = instruction.args_copy()[0]
            new_targets =[]
            flag = False
            for target in targets:

                if target.is_relative_detector_id():

                    ind = target.val 

                    if ind==0:
                        flag=True
                        # break
                    else:
                        new_targets.append(target) 

                else:
                    new_targets.append(target)


            if new_targets==[]:  
                continue
            else:
                new_stims_DEM.append("error",prob,new_targets)
        
        else:

            if instruction.type=="detector":

                detectors = instruction.targets_copy()
                flag = False
                for det in detectors:

                   
                    if det.is_relative_detector_id():

                        ind = det.val 
                        if ind==0:
                            flag=True
                            break
                        
                        
                if flag==False:
                    new_stims_DEM.append(instruction)

            else:
                new_stims_DEM.append(instruction)
            


    detector_error_model = new_stims_DEM
    
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors_stim = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors_stim += 1

    #Now do the same for my model.
    detection_events = detection_events[:,1:] #Drop the flag qubit
    matcher     = Matching.from_detector_error_model(my_DEM)
    predictions = matcher.decode_batch(detection_events)

    num_errors_my_DEM = 0
    for shot in range(num_shots):
        actual_for_shot    = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors_my_DEM += 1    


    return num_errors_my_DEM,num_errors_stim,num_shots



def decode_both_dems_V2_w_post_selection(my_DEM:stim.DetectorErrorModel,circuit:stim.Circuit,num_shots:int,num_rounds: int, distance: int):
    '''
    Decode via MWPM the exact model of stim and our reconstructed DEM. The detection events are the same
    for both DEMs.

    Inputs: 
        my_DEM:    Reconstructed detector error model
        circuit:   The stim circuit from which we extract the exact detector error model
        num_shots: # of shots to use to decode
    Output:
        num_errors_my_DEM: total # of logical errors obtained by decoding the reconstructed DEM
        num_errors_stim:   total # of logical errors obtained by decoding stim's DEM
    '''
    
    # Sample the circuit.
    sampler                            = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    #Keep only detection events where D0, D_{0+d...} takes the value of 0
    locs_no_error  = list(np.where(detection_events[:,0]==0)[0])
    for k in range(1,num_rounds):
        locs    = list(np.where(detection_events[:,0+k*distance]==0)[0])

        for loc in locs:
            if loc not in locs_no_error:
                locs_no_error += [loc]

    detection_events = detection_events[locs_no_error,:]
    

    print("shape of detection events:",np.shape(detection_events))

    observable_flips = observable_flips[locs_no_error]
    num_shots        = np.shape(detection_events)[0]


    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(flatten_loops=True) 

    inds_of_dets_to_remove = []
    for k in range(num_rounds):
        inds_of_dets_to_remove.append(k*distance)


    #Remove the dets of the flag qubit detector from the DEM instructions where it exists
    new_stims_DEM = stim.DetectorErrorModel()
    for instruction in detector_error_model:

        if instruction.type=="error":

            targets     = instruction.targets_copy()
            prob        = instruction.args_copy()[0]
            new_targets = []
            flag        = False

            for target in targets:

                if target.is_relative_detector_id():

                    ind = target.val 

                    if ind in inds_of_dets_to_remove:
                        flag=True
                        # break
                    else:
                        new_targets.append(target) 

                else:
                    new_targets.append(target)


            if new_targets==[]:  
                continue
            else:
                new_stims_DEM.append("error",prob,new_targets)
        
        else:

            if instruction.type=="detector":

                detectors = instruction.targets_copy()
                flag = False
                for det in detectors:

                   
                    if det.is_relative_detector_id():

                        ind = det.val 
                        if ind in inds_of_dets_to_remove:
                            flag=True
                            break
                        
                        
                if flag==False:
                    new_stims_DEM.append(instruction)

            else:
                new_stims_DEM.append(instruction)
            


    detector_error_model = new_stims_DEM
    
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors_stim = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors_stim += 1
    
    #Now do the same for my model.
    
    #Drop the flag qubit columns. Not sure about this
    cnt=0
    for k in range(num_rounds):
        detection_events = np.delete(detection_events,0+k*distance-cnt)
        cnt+=1 
    


    matcher     = Matching.from_detector_error_model(my_DEM)
    predictions = matcher.decode_batch(detection_events)

    num_errors_my_DEM = 0
    for shot in range(num_shots):
        actual_for_shot    = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors_my_DEM += 1    


    return num_errors_my_DEM,num_errors_stim,num_shots
