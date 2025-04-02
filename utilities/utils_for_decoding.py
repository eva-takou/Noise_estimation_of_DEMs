import stim
import numpy as np
import pymatching


def decode_both_dems_same_data(reconstructed_DEM:stim.DetectorErrorModel,circuit:stim.Circuit,num_shots:int):
    '''
    Decode stim's DEM obtained from a circuit and a reconstructed DEM using MWPM, and on the same data.

    Inputs: 
        my_DEM:    Reconstructed detector error model
        circuit:   The stim circuit from which we extract the exact detector error model
        num_shots: # of shots to use to decode
    Output:
        num_errors_est:    Total # of logical errors obtained by decoding the reconstructed DEM
        num_errors_stim:   Total # of logical errors obtained by decoding stim's DEM
    '''
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(flatten_loops=True) 
    matcher              = pymatching.Matching.from_detector_error_model(detector_error_model)

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

    matcher     = pymatching.Matching.from_detector_error_model(reconstructed_DEM)
    predictions = matcher.decode_batch(detection_events) #use same detection events

    num_errors_est = 0
    for shot in range(num_shots):
        actual_for_shot    = observable_flips[shot]  #use same obs_flips
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors_est += 1    

    return num_errors_est,num_errors_stim


def decode_both_dems_diff_data(reconstructed_DEM,circuit,num_shots):
    '''
    Decode stim's DEM obtained from a circuit and a reconstructed DEM using MWPM, and on the different data.

    Inputs: 
        my_DEM:    Reconstructed detector error model
        circuit:   The stim circuit from which we extract the exact detector error model
        num_shots: # of shots to use to decode
    Output:
        num_errors_est: Total # of logical errors obtained by decoding the reconstructed DEM
        num_errors_stim:   Total # of logical errors obtained by decoding stim's DEM
    '''
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model() #Let's not decompose for rep code
    matcher              = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors_stim = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors_stim += 1

    #------- Now do the same for my model ----------------------


    sampler             = reconstructed_DEM.compile_sampler() 
    events_from_my_DEM  = sampler.sample(shots=num_shots)
    flips_in_my_DEM     = events_from_my_DEM[1]
    detection_events    = events_from_my_DEM[0]


    matcher     = pymatching.Matching.from_detector_error_model(reconstructed_DEM)
    predictions = matcher.decode_batch(detection_events)

    num_errors_est = 0
    for shot in range(num_shots):
        actual_for_shot    = flips_in_my_DEM[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors_my_DEM += 1    


    return num_errors_est,num_errors_stim
