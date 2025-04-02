
import numpy as np
import xarray as xr
import pymatching
from pymatching import Matching
import stim
from scipy.optimize import minimize,least_squares

import sympy as sym
from sympy import symbols
import itertools
from itertools import combinations


def form_equation_terms(target_inds, order):
    '''target_inds: some list that contains 0,1,2,3 elements.
    These correspond to the x0,x1,x2,x3 for P(x0,x1,x2,x3).
    order: the order of p events we want to calculate.
    '''
    if target_inds==[]: #P(0,0,0,0) event: treat separately
        
        
        rest_inds      = [0,1,2,3]
        p0             = [0] ; p1  = [1] ; p2  = [2] ; p3  = [3] 
        p01            = [0,1]; p02 = [0,2] ;p13 = [1,3] ;p23 = [2,3] 
        p0123          = [0,1,2,3]
        all_elements   = [p0,p1,p2,p3,p01,p02,p13,p23,p0123]
        all_combs      = list(combinations(all_elements,order))
        accepted_combs = []

        for comb in all_combs:
            #All should appear an even # of times.

            cnt_0,cnt_1,cnt_2,cnt_3 = 0,0,0,0

            for elem in comb:

                if 3 in elem:
                    cnt_3 +=1 
                if 0 in elem:
                    cnt_0+=1
                if 1 in elem:
                    cnt_1+=1
                if 2 in elem:
                    cnt_2+=1
            
            list_of_cnts = [cnt_0,cnt_1,cnt_2,cnt_3]

            not_acceptable = False
            
            
            for p in rest_inds:
                if list_of_cnts[p]%2==1:
                    not_acceptable=True
            
            if not_acceptable==False:
                accepted_combs.append(comb)
        
        
        return accepted_combs



    for k in target_inds:
        if k>3 or k<0:
            raise Exception("Invalid indices. They should be in [0,1,2,3].")

    all_inds  = [0,1,2,3]
    rest_inds = list(set(all_inds) ^ set(target_inds) )


    p0             = [0] ; p1  = [1] ; p2  = [2] ; p3  = [3] 
    p01            = [0,1]; p02 = [0,2] ;p13 = [1,3] ;p23 = [2,3] 
    p0123          = [0,1,2,3]
    all_elements   = [p0,p1,p2,p3,p01,p02,p13,p23,p0123]
    all_combs      = list(combinations(all_elements,order))
    accepted_combs = []

    for comb in all_combs:
        #check that "3" appears an odd # of times

        cnt_0,cnt_1,cnt_2,cnt_3 = 0,0,0,0

        for elem in comb:

            if 3 in elem:
                cnt_3 +=1 
            if 0 in elem:
                cnt_0+=1
            if 1 in elem:
                cnt_1+=1
            if 2 in elem:
                cnt_2+=1
        
        list_of_cnts = [cnt_0,cnt_1,cnt_2,cnt_3]

        not_acceptable = False
        for p in target_inds:

            if list_of_cnts[p]%2==0:
                not_acceptable=True
        
        for p in rest_inds:
            if list_of_cnts[p]%2==1:
                not_acceptable=True
        
        if not_acceptable==False:
            accepted_combs.append(comb)

    return accepted_combs


def form_particular_eq_for_one_truncation_order(target_inds,order):

    all_terms   = form_equation_terms(target_inds, order)

    p0        = [0] ; p1  = [1] ; p2  = [2] ; p3  = [3] 
    p01       = [0,1]; p02 = [0,2] ;p13 = [1,3] ;p23 = [2,3] 
    p0123     = [0,1,2,3]
    all_eqns   = 0

    for terms in all_terms:

        all_elements   = [p0,p1,p2,p3,p01,p02,p13,p23,p0123]
        this_eq = 1
        
        for term in terms:

            total_elem = "p"
            for elem in term:
                total_elem += str(elem)
            
            
            this_eq*=symbols(total_elem)

        all_elements = [x for x in all_elements if x not in terms]
        
        
        for term in all_elements:
            total_elem = "p"
            for elem in term:
                total_elem += str(elem)
            
            this_eq*=1-symbols(total_elem)
        
        all_eqns+=this_eq



    return all_eqns


def form_all_4_pnt_equations(max_truncation_order):
    '''Form the equations of the form P(x1,x2,x3,x4) which includes 4 pnt events.'''


    #Eqns in a dictionary:
    P = {}

    all_target_inds = [[0],[1],[2],[3], \
                       [0,1],[0,2],[0,3],\
                       [1,2],[1,3],\
                       [2,3],\
                       [0,1,2],[0,1,3],[0,2,3], \
                       [1,2,3],\
                       [0,1,2,3],[]]
    
    all_names = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                 [1,1,0,0],[1,0,1,0],[1,0,0,1],
                 [0,1,1,0],[0,1,0,1],
                 [0,0,1,1],
                 [1,1,1,0],[1,1,0,1],[1,0,1,1],
                 [0,1,1,1],
                 [1,1,1,1],
                 [0,0,0,0]]

    for name in all_names:
        P[tuple(name)]=0
    
    cnt=0
    for target_inds in all_target_inds:


        for order in range(1,max_truncation_order+1):

            eqns = form_particular_eq_for_one_truncation_order(target_inds,order)
            P[tuple(all_names[cnt])]+= eqns
            
        cnt+=1

    return P


#TODO: Verify that we have all terms up to O(p^3), and extend to O(p^4).

#OK
def get_initial_state(anc_qubits):

    #Initial ancilla state
    initial_state = np.zeros(len(anc_qubits), dtype=int)
    initial_state = initial_state==True
    initial_state = xr.DataArray ( data=initial_state,dims=[ "anc_qubit" ] ,coords=dict ( anc_qubit=anc_qubits) , ) 
    
    return initial_state

#OK
def project_data_meas(data_meas, num_shots,num_rounds,d,anc_qubits):

    shots = np.arange(num_shots) 
    
    syndrome_proj = data_meas.data[:,:-1] ^ data_meas.data[:,1:]  #dims: # of shots x # of ancilla
    #the 0th and 1st entries are xored, to give the 1st ancilla measurement equivalent
    #the 1st and 2nd entries are xored, to give the 2nd ancilla measurement equivalent

    syndrome_proj = xr.DataArray(data=syndrome_proj,
                                 dims=["shot","anc_qubit"],
                                 coords = dict(shot=shots,anc_qubit=anc_qubits))

    syndrome_proj["qec_round"]  = num_rounds    

    return syndrome_proj


#I have ommitted the flag qubit for any distance. Need to add it back and repeat the procedure.
def get_defects_matrix(distance,num_rounds,num_shots,circuit):

    #Need to have a tracking system: if the ancilla meas is '1'
    d             = distance    
    data_qubits   = np.arange(0,d).tolist()
    anc_qubits    = (np.arange(d)+d).tolist()        
    SHOTS      = np.arange(num_shots)
    QEC_ROUNDS = np.arange(num_rounds)

    sampler   = circuit.compile_sampler()
    samples   = sampler.sample(shots=num_shots)

    NUM_ANC = len(anc_qubits)

    anc_qubit_samples  = samples[:,:NUM_ANC*num_rounds]
    data_qubit_samples = samples[:,NUM_ANC*num_rounds:]
    anc_qubit_samples  = anc_qubit_samples.reshape(num_shots,num_rounds,NUM_ANC) 

    #This is because 2 consecutive ancilla measurements create the syndrome
    anc_qubit_samples = anc_qubit_samples[:,:,:-1] ^ anc_qubit_samples[:,:,1:]

    # for k in range(num_shots):

    #     print(samples[k,:])
    #     print(anc_qubit_samples[k,0,:])
    #     print("----")


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

    # print("-------------Syndromes for 0th rd:")
    # for k in range(num_shots):
    #     print(syndromes.data[k,0,:])
    #     print('-----')
    # print("-------------Syndromes for 1st rd:")
    # for k in range(num_shots):
    #     print(syndromes.data[k,1,:])
    #     print('-----')

    # syndromes                         = xr.concat([anc_meas,syndrome_proj],"qec_round")
    syndrome_matrix_copy              = syndromes.copy()
    syndrome_matrix_copy.data[:,-1,:] = initial_state
    defects_matrix                    = syndromes ^ syndrome_matrix_copy.roll(qec_round=1)

    # print("-------------Defects for 0th rd:")

    # for k in range(num_shots):
    #     print(defects_matrix.data[k,0,:])

    # print("-------------Defects for 1st rd:")

    # for k in range(num_shots):
    #     print(defects_matrix.data[k,1,:])


    return defects_matrix

def get_defects_matrix_ancilla_only(distance,num_rounds,num_shots,circuit):


    #Need to have a tracking system: if the ancilla meas is '1'
    d             = distance    
    anc_qubits    = np.arange(d).tolist()        
    SHOTS      = np.arange(num_shots)
    QEC_ROUNDS = np.arange(num_rounds+1)

    sampler   = circuit.compile_sampler()
    samples   = sampler.sample(shots=num_shots)

    NUM_ANC = len(anc_qubits)

    anc_qubit_samples  = samples
    anc_qubit_samples  = anc_qubit_samples.reshape(num_shots,num_rounds+1,NUM_ANC) 

    #This is because 2 consecutive ancilla measurements create the syndrome
    anc_qubit_samples = anc_qubit_samples[:,:,:-1] ^ anc_qubit_samples[:,:,1:]

    # for k in range(num_shots):

    #     print(samples[k,:])
    #     print(anc_qubit_samples[k,0,:])
    #     print("----")


    #Now redefine the anc qubits as detectors
    anc_qubits    = np.arange(d-1).tolist()  
    anc_meas = xr.DataArray(data = anc_qubit_samples, 
                        dims=["shot","qec_round","anc_qubit"],
                        coords = dict(shot=SHOTS,qec_round=QEC_ROUNDS,anc_qubit=anc_qubits),)
   

    initial_state = get_initial_state(anc_qubits)

    syndromes = anc_meas

    # print("-------------Syndromes for 0th rd:")
    # for k in range(num_shots):
    #     print(syndromes.data[k,0,:])
    #     print('-----')
    # print("-------------Syndromes for 1st rd:")
    # for k in range(num_shots):
    #     print(syndromes.data[k,1,:])
    #     print('-----')

    # syndromes                         = xr.concat([anc_meas,syndrome_proj],"qec_round")
    syndrome_matrix_copy              = syndromes.copy()
    syndrome_matrix_copy.data[:,-1,:] = initial_state
    defects_matrix                    = syndromes ^ syndrome_matrix_copy.roll(qec_round=1)

    # print("-------------Defects for 0th rd:")

    # for k in range(num_shots):
    #     print(defects_matrix.data[k,0,:])

    # print("-------------Defects for 1st rd:")

    # for k in range(num_shots):
    #     print(defects_matrix.data[k,1,:])


    return defects_matrix


#OK
def get_4_pnt_events(defect_matrix,p_depol_after):
    '''The 4-pnt events have 2 nodes in same rd, and the other 2 nodes in the next rd.
    '''
    num_shots    = np.size(defect_matrix.data,axis=0)     
    num_rounds   = np.size(defect_matrix.data,axis=1)     
    num_anc      = np.size(defect_matrix.data,axis=2)     


    p4_cnts = {}

    for anc1 in range(num_anc-1):
       
        anc2 = anc1+1
        anc3 = anc1
        anc4 = anc2

        for round1 in range(num_rounds-1):

            round2 = round1
            round3 = round1+1
            round4 = round3

            locs1 = np.nonzero(defect_matrix.data[:,round1,anc1])[0]
            locs2 = np.nonzero(defect_matrix.data[:,round2,anc2])[0]
            locs3 = np.nonzero(defect_matrix.data[:,round3,anc3])[0]
            locs4 = np.nonzero(defect_matrix.data[:,round4,anc4])[0]

            locs1 = set(locs1)
            locs2 = set(locs2)
            locs3 = set(locs3)
            locs4 = set(locs4)

            locs  = locs1 & locs2 & locs3 & locs4 #all should be nnz for the same shot
            
            #Indices are sorted
            indx1 = anc1+num_anc*round1
            indx2 = anc2+num_anc*round2
            indx3 = anc3+num_anc*round3
            indx4 = anc4+num_anc*round4

            # inds = np.sort([indx1,indx2,indx3,indx4])

            name_of_4_pnt_event = ("D"+str(indx1),"D"+str(indx2),"D"+str(indx3),"D"+str(indx4))
            
            p4_cnts[name_of_4_pnt_event]=len(locs)/num_shots     

    if p_depol_after!=[]:
        for key in p4_cnts.keys():
            p4_cnts[key]=2/3*4/5*p_depol_after
   

    return p4_cnts


def get_4_pnt_events_alt(defect_matrix):
    '''The 4-pnt events have 2 nodes in same rd, and the other 2 nodes in the next rd.
    '''
    num_shots    = np.size(defect_matrix.data,axis=0)     
    num_rounds   = np.size(defect_matrix.data,axis=1)     
    num_anc      = np.size(defect_matrix.data,axis=2)     


    p4_cnts = {}

    for anc1 in range(num_anc-1):
       
        anc2 = anc1+1
        anc3 = anc1
        anc4 = anc2

        for round1 in range(num_rounds-1):

            round2 = round1
            round3 = round1+1
            round4 = round3

            VAL = 0
            for k in range(num_shots):

                temp1 = defect_matrix.data[k,round1,anc1]
                temp2 = defect_matrix.data[k,round2,anc2]
                temp3 = defect_matrix.data[k,round3,anc3]
                temp4 = defect_matrix.data[k,round4,anc4]

                if temp1 & temp2 & temp3 & temp4:
                    VAL +=1


            #Indices are sorted
            indx1 = anc1+num_anc*round1
            indx2 = anc2+num_anc*round2
            indx3 = anc3+num_anc*round3
            indx4 = anc4+num_anc*round4

            # inds = np.sort([indx1,indx2,indx3,indx4])

            name_of_4_pnt_event = ("D"+str(indx1),"D"+str(indx2),"D"+str(indx3),"D"+str(indx4))
            
            p4_cnts[name_of_4_pnt_event]=VAL/num_shots     


    return p4_cnts


def get_logical_error_rate(my_DEM,circuit,num_shots,decompose_errors):

    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=decompose_errors) 
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

    #Now do the same for my DEM


    sampler       = my_DEM.compile_sampler() 

    #detection_events, observable_flips = sampler.sample(shots = num_shots, separate_observables=True)
    events_from_my_DEM  = sampler.sample(shots=num_shots)
    flips_in_my_DEM     = events_from_my_DEM[1]
    detection_events    = events_from_my_DEM[0]

    matcher     = Matching.from_detector_error_model(my_DEM)
    predictions = matcher.decode_batch(detection_events)

    num_errors_my_DEM = 0
    for shot in range(num_shots):
        actual_for_shot    = flips_in_my_DEM[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors_my_DEM += 1    

    return num_errors_stim/num_shots, num_errors_my_DEM/num_shots


def get_vijkl(p4_cnts,num_rounds,num_ancilla,vi_mean,vivj_mean):

    num_rounds +=1

    events ={}

    for key in p4_cnts.keys():

        temp_dict = {}
        det_inds = []
        for det in key:
            ind = int(det[1:])

            for rd in range(num_rounds):

                for anc in range(num_ancilla):

                    new_ind = anc + rd*num_ancilla
                    if new_ind==ind:
                        det_inds.append((rd,anc))
                        break
        
        inds1 = det_inds[0]
        inds2 = det_inds[1]
        inds3 = det_inds[2]
        inds4 = det_inds[3]
        
        vi = vi_mean[inds1[0],inds1[1]] #v1000
        vj = vi_mean[inds2[0],inds2[1]] #v0100
        vk = vi_mean[inds3[0],inds3[1]] #v0010
        vl = vi_mean[inds4[0],inds4[1]] #v0001
        
        vivj = vivj_mean[inds1[0],inds2[0],inds1[1],inds2[1]] #v1100
        vjvl = vivj_mean[inds2[0],inds4[0],inds2[1],inds4[1]] #v0101
        vivk = vivj_mean[inds1[0],inds3[0],inds1[1],inds3[1]] #v1010
        vkvl = vivj_mean[inds3[0],inds4[0],inds3[1],inds4[1]] #v0011

        vivjvkvl = p4_cnts[key] #v1111

        temp_dict["v1000"]=vi
        temp_dict["v0100"]=vj
        temp_dict["v0010"]=vk
        temp_dict["v0001"]=vl

        temp_dict["v1100"]=vivj
        temp_dict["v0101"]=vjvl
        temp_dict["v1010"]=vivk
        temp_dict["v0011"]=vkvl
        
        temp_dict["v1111"]=vivjvkvl

        events[key] = temp_dict


    return events



#Equations up to O(p^3): It works pretty well!!
def equations(vars, v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011):

    p0, p1, p2, p3, x, p01, p02, p13, p23 = vars
    
    P0001 =   p3*(1-p0)*(1-p1)*(1-p2)*(1-p01)*(1-p02)*(1-p13)*(1-p23)*(1-x) \
            + p2*p23*(1-p0)*(1-p1)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-x) \
            + p1*p13*(1-p0)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p23)*(1-x) \
            + p0*p01*p13*(1-p1)*(1-p2)*(1-p3)*(1-p02)*(1-p23)*(1-x) \
            + p0*p02*p23*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p13)*(1-x) \
            + x*p01*p2*(1-p0)*(1-p1)*(1-p3)*(1-p02)*(1-p13)*(1-p23) \
            + x*p02*p1*(1-p0)*(1-p2)*(1-p3)*(1-p01)*(1-p13)*(1-p23)

    P0010 =   p2*(1-p0)*(1-p1)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-p23)*(1-x) \
            + p0*p02*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p13)*(1-p23)*(1-x) \
            + p3*p23*(1-p0)*(1-p1)*(1-p2)*(1-p01)*(1-p02)*(1-p13)*(1-x) \
            + p1*p13*p23*(1-p0)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-x) \
            + p1*p01*p02*(1-p0)*(1-p2)*(1-p3)*(1-p13)*(1-p23)*(1-x) \
            + x*p01*p3*(1-p0)*(1-p1)*(1-p2)*(1-p02)*(1-p13)*(1-p23) \
            + x*p0*p13*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p23)
    
    P0100 = p1*(1-p0)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-p23)*(1-x)\
          + p3*p13*(1-p0)*(1-p1)*(1-p2)*(1-p01)*(1-p02)*(1-p23)*(1-x) \
          + p0*p01*(1-p1)*(1-p2)*(1-p3)*(1-p02)*(1-p13)*(1-p23)*(1-x) \
          + p2*p02*p01*(1-p0)*(1-p1)*(1-p3)*(1-p13)*(1-p23)*(1-x) \
          + p2*p23*p13*(1-p0)*(1-p1)*(1-p3)*(1-p02)*(1-p01)*(1-x) \
          + x*p23*p0*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p13)\
          + x*p02*p3*(1-p0)*(1-p1)*(1-p2)*(1-p01)*(1-p13)*(1-p23)
    
    P1000 = p0*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-p23)*(1-x) \
          + p01*p1*(1-p0)*(1-p2)*(1-p3)*(1-p02)*(1-p13)*(1-p23)*(1-x) \
          + p02*p2*(1-p0)*(1-p1)*(1-p3)*(1-p01)*(1-p13)*(1-p23)*(1-x) \
          + p02*p23*p3*(1-p0)*(1-p1)*(1-p2)*(1-p01)*(1-p13)*(1-x)\
          + p01*p13*p3*(1-p0)*(1-p1)*(1-p2)*(1-p02)*(1-p23)*(1-x) \
          + x*p1*p23*(1-p0)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p13) \
          + x*p13*p2*(1-p0)*(1-p1)*(1-p3)*(1-p01)*(1-p02)*(1-p23)
    
    

    P1100 = p01*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p02)*(1-p13)*(1-p23)*(1-x) \
          + p0*p1*(1-p2)*(1-p3)*(1-p02)*(1-p01)*(1-p13)*(1-p23)*(1-x) \
          + x*p23*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p13)\
          + x*p2*p3*(1-p0)*(1-p1)*(1-p01)*(1-p02)*(1-p13)*(1-p23) \
          + p02*p13*p23*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-x) \
          + p02*p2*p1*(1-p0)*(1-p3)*(1-p01)*(1-p23)*(1-p13)*(1-x)\
          + p0*p13*p3*(1-p1)*(1-p2)*(1-p01)*(1-p02)*(1-p23)*(1-x)
    
    
    P0011 = p23*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p02)*(1-p13)*(1-p01)*(1-x) \
          + p2*p3*(1-p0)*(1-p1)*(1-p02)*(1-p01)*(1-p13)*(1-p23)*(1-x)\
          + x*p01*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p23)*(1-p02)*(1-p13)\
          + p01*p02*p13*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p23)*(1-x)\
          + x*p0*p1*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-p23) \
          + p2*p13*p1*(1-p0)*(1-p3)*(1-p01)*(1-p02)*(1-p23)*(1-x) \
          + p0*p02*p3*(1-p1)*(1-p2)*(1-p01)*(1-p13)*(1-p23)*(1-x)
    
    P1010 = p02*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p23)*(1-p13)*(1-p01)*(1-x) \
          + p0*p2*(1-p1)*(1-p3)*(1-p02)*(1-p01)*(1-p13)*(1-p23)*(1-x)\
          + x*p13*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p23)*(1-p02)*(1-p01)\
          + p01*p13*p23*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p02)*(1-x) \
          + x*p1*p3*(1-p0)*(1-p2)*(1-p01)*(1-p02)*(1-p13)*(1-p23) \
          + p01*p1*p2*(1-p0)*(1-p3)*(1-p02)*(1-p13)*(1-p23)*(1-x) \
          + p0*p23*p3*(1-p1)*(1-p2)*(1-p01)*(1-p02)*(1-p13)*(1-x)
    
    
    P0101 = p13*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p23)*(1-p02)*(1-p01)*(1-x) \
          + p1*p3*(1-p0)*(1-p2)*(1-p02)*(1-p01)*(1-p13)*(1-p23)*(1-x) \
          + x*p02*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p23)*(1-p13)*(1-p01) \
          + p01*p02*p23*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p13)*(1-x) \
          + x*p0*p2*(1-p1)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-p23) \
          + p1*p23*p2*(1-p0)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-x) \
          + p0*p01*p3*(1-p1)*(1-p2)*(1-p02)*(1-p13)*(1-p23)*(1-x)
    
    P1001 = p0*p3*(1-p1)*(1-p2)*(1-p01)*(1-p02)*(1-p13)*(1-p23)*(1-x) \
          + p01*p13*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p02)*(1-p23)*(1-x) \
          + p02*p23*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p13)*(1-x) \
          + x*p1*p2*(1-p0)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-p23) \
          + p0*p2*p23*(1-p1)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-x) \
          + p01*p1*p3*(1-p0)*(1-p2)*(1-p02)*(1-p13)*(1-p23)*(1-x)
    
    P0110 = p1*p2*(1-p0)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-p23)*(1-x) \
          + p01*p02*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p13)*(1-p23)*(1-x) \
          + p13*p23*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-x) \
          + x*p0*p3*(1-p1)*(1-p2)*(1-p01)*(1-p02)*(1-p13)*(1-p23)\
          + p1*p23*p3*(1-p0)*(1-p2)*(1-p01)*(1-p02)*(1-p13)*(1-x)\
          + p0*p01*p2*(1-p1)*(1-p3)*(1-p02)*(1-p13)*(1-p23)*(1-x)
    
    P1110 = p01*p2*(1-p0)*(1-p1)*(1-p3)*(1-p02)*(1-p23)*(1-p13)*(1-x) \
          + p02*p1*(1-p0)*(1-p2)*(1-p3)*(1-p01)*(1-p23)*(1-p13)*(1-x) \
          + x*p3*(1-p0)*(1-p1)*(1-p2)*(1-p01)*(1-p02)*(1-p13)*(1-p23) \
          + p0*p01*p02*(1-p1)*(1-p2)*(1-p3)*(1-p13)*(1-p23)*(1-x) \
          + x*p23*p2*(1-p0)*(1-p1)*(1-p3)*(1-p01)*(1-p02)*(1-p13) \
          + x*p13*p1*(1-p0)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p23)  \
          + p02*p13*p3*(1-p0)*(1-p1)*(1-p2)*(1-p01)*(1-p23)*(1-x) \
          + p0*p02*p1*(1-p2)*(1-p3)*(1-p01)*(1-p13)*(1-p23)*(1-x) \
          + p0*p1*p2*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-p23)*(1-x)

    P0111 = p13*p2*(1-p0)*(1-p1)*(1-p3)*(1-p02)*(1-p23)*(1-p01)*(1-x)\
          + p23*p1*(1-p0)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-x) \
          + x*p0*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-p23) \
          + p13*p23*p3*(1-p0)*(1-p1)*(1-p2)*(1-p01)*(1-p02)*(1-x) \
          + x*p01*p1*(1-p0)*(1-p2)*(1-p3)*(1-p02)*(1-p13)*(1-p23) \
          + x*p02*p2*(1-p0)*(1-p1)*(1-p3)*(1-p01)*(1-p13)*(1-p23) \
          + p0*p01*p23*(1-p1)*(1-p2)*(1-p3)*(1-p02)*(1-p13)*(1-x) \
          + p0*p02*p13*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p23)*(1-x) \
          + p1*p2*p3*(1-p0)*(1-p01)*(1-p02)*(1-p13)*(1-p23)*(1-x)
    
    P1011 = p0*p23*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-x) \
          + p02*p3*(1-p0)*(1-p1)*(1-p2)*(1-p01)*(1-p13)*(1-p23)*(1-x)\
          + x*p1*(1-p0)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-p23) \
          + p0*p2*p3*(1-p1)*(1-p01)*(1-p02)*(1-p13)*(1-p23)*(1-x) \
          + p01*p1*p23*(1-p0)*(1-p2)*(1-p3)*(1-p02)*(1-p13)*(1-x)\
          + p02*p1*p13*(1-p0)*(1-p2)*(1-p3)*(1-p01)*(1-p23)*(1-x)\
          + p02*p2*p23*(1-p0)*(1-p1)*(1-p3)*(1-p01)*(1-p13)*(1-x)\
          + x*p13*p3*(1-p0)*(1-p1)*(1-p2)*(1-p01)*(1-p02)*(1-p23)\
          + x*p0*p01*(1-p1)*(1-p2)*(1-p3)*(1-p02)*(1-p13)*(1-p23)

    P1101 = p01*p3*(1-p0)*(1-p1)*(1-p2)*(1-p02)*(1-p13)*(1-p23)*(1-x) \
          + p0*p13*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p23)*(1-x) \
          + x*p2*(1-p0)*(1-p1)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-p23) \
          + p01*p1*p13*(1-p0)*(1-p2)*(1-p3)*(1-p02)*(1-p23)*(1-x)\
          + p02*p2*p13*(1-p0)*(1-p1)*(1-p3)*(1-p01)*(1-p23)*(1-x)\
          + p01*p2*p23*(1-p0)*(1-p1)*(1-p3)*(1-p02)*(1-p13)*(1-x)\
          + p0*p1*p3*(1-p2)*(1-p01)*(1-p02)*(1-p13)*(1-p23)*(1-x)\
          + x*p23*p3*(1-p0)*(1-p1)*(1-p2)*(1-p01)*(1-p02)*(1-p13)\
          + x*p01*p1*(1-p0)*(1-p2)*(1-p3)*(1-p02)*(1-p13)*(1-p23)
    
    P1111 = x*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-p23) \
          + p01*p23*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p02)*(1-p13)*(1-x) \
          + p02*p13*(1-p0)*(1-p1)*(1-p2)*(1-p3)*(1-p01)*(1-p23)*(1-x) \
          + p01*p2*p3*(1-p0)*(1-p1)*(1-p02)*(1-p13)*(1-p23)*(1-x) \
          + p0*p2*p13*(1-p1)*(1-p3)*(1-p01)*(1-p02)*(1-p23)*(1-x) \
          + p02*p1*p3*(1-p0)*(1-p2)*(1-p01)*(1-p13)*(1-p23)*(1-x) \
          + p0*p1*p23*(1-p2)*(1-p3)*(1-p01)*(1-p02)*(1-p13)*(1-x)

    
    

    eq1 = v0001 - (  P0001
                   + P1001 + P0101 + P0011
                   + P1101 + P1011 + P0111 
                   + P1111)
    
    eq2 = v0010 - (  P0010
                   + P1010 + P0110 + P0011+
                   + P1011 + P1110 + P0111
                   + P1111)
    
    eq3 = v0100 - (P0100 + 
                   P1100 + P0110 + P0101+
                   P1110 + P0111 + P1101+
                   P1111)
    
    eq4 = v1000 - (P1000 + 
                   P1100 + P1010 + P1001+
                   P1110 + P1011 + P1101
                   + P1111)
    
    eq5 = v1100 - (P1100 +
                    P1110 +P1101
                    +P1111)
    
    eq6 = v1010 - (P1010+
                   P1110 + P1011
                   +P1111)
    
    eq7 = v0011 - (P0011+
                   +P0111+P1011+
                   P1111)
    
    eq8 = v0101-(P0101 +
                 P0111+P1101
                 +P1111)
    
    eq9 = v1111-P1111
    return [eq1,eq2,eq3,eq4,eq5,eq6,eq7,eq8,eq9]

#Equations up to O(p^5)
def equations_NEW(vars, v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011):
    '''These equations were formed given the form_all_4_pnt_equations function, with max trunctation order 4.'''

    p0, p1, p2, p3, x, p01, p02, p13, p23 = vars

    P0001 = p0*p01*p02*x*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p1*p2*p23*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p1*p3*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p13*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p13*p2*(1 - p01)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p2*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p02*p23*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p1*p2*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p1*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2) + p0*p13*p2*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23) + p0*p13*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p01*p02*p1*p2*p3*(1 - p0)*(1 - p13)*(1 - p23)*(1 - x) + p01*p02*p1*p23*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p13*p2*(1 - p0)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p13*p23*p3*(1 - p0)*(1 - p1)*(1 - p2)*(1 - x) + p01*p1*p13*p23*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p3) + p01*p2*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p02*p1*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p3) + p02*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23) + p1*p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - x) + p1*p13*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x)
    P0010 = p0*p01*p02*p23*x*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p01*p1*p2*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p1*p23*p3*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p0*p01*p13*p2*p3*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p0*p01*p13*p23*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p1*p13*p3*(1 - p01)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p2*p23*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - x) + p0*p02*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3) + p0*p1*p3*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p13*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p1*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p13*p2*p23*(1 - p0)*(1 - p1)*(1 - p3)*(1 - x) + p01*p02*p13*p3*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p01*p1*p13*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p01*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p02*p1*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23) + p02*p1*p23*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3) + p02*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3) + p02*p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2) + p1*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - x) + p1*p13*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x)
    P0100 = p0*p01*p02*p13*x*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p1*p13*p3*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p2*p23*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p0*p01*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p2*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p23*p3*(1 - p01)*(1 - p13)*(1 - p2)*(1 - x) + p0*p02*p13*p2*p3*(1 - p01)*(1 - p1)*(1 - p23)*(1 - x) + p0*p02*p13*p23*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p1*p13*p2*x*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3) + p0*p2*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p02*p1*p13*p23*(1 - p0)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p2*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p23*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p01*p1*p2*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23) + p01*p1*p23*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p13*p2*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p01*p13*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2) + p02*p1*p13*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3) + p02*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p1*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - x) + p1*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p13*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x)
    P1000 = p0*p01*p02*p1*p2*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p23*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p2*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p01*p23*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p02*p1*p3*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p02*p13*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p1*p13*p2*p23*(1 - p01)*(1 - p02)*(1 - p3)*(1 - x) + p0*p1*p13*p3*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p0*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p0*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p1*p13*x*(1 - p0)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p2*p23*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p3) + p01*p02*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p1*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - x) + p01*p1*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p13*p2*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p01*p13*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p02*p1*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p23)*(1 - x) + p02*p1*p13*p23*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p3)*(1 - x) + p02*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p02*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p1*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23) + p1*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p13*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)

    P1100 = p0*p01*p02*p2*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p23*p3*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p0*p01*p1*p23*x*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p01*p13*p2*x*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p0*p02*p1*p13*x*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p2*p23*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p02*p3*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p1*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - x) + p0*p1*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p13*p2*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p0*p13*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p1*p3*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p02*p13*x*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p1*p13*p2*p23*(1 - p0)*(1 - p02)*(1 - p3)*(1 - x) + p01*p1*p13*p3*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p01*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p01*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p2*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p23*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - x) + p02*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - x) + p02*p13*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p1*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3) + p1*p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2) + p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)
    P1010 = p0*p01*p02*p1*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p3*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p1*p13*x*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p2*p23*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p01*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p02*p1*p23*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p02*p13*p2*x*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3) + p0*p1*p13*p2*p3*(1 - p01)*(1 - p02)*(1 - p23)*(1 - x) + p0*p1*p13*p23*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p0*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p01*p02*p2*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23) + p01*p02*p23*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p1*p2*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p23*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p01*p13*p2*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p01*p13*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p02*p1*p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p3)*(1 - x) + p02*p1*p13*p3*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - x) + p02*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - x) + p02*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p1*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3) + p1*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p13*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1) + p13*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)
    P1001 = p0*p01*p02*p1*p23*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p2*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p2*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p23*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p0*p02*p1*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p13*p3*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23) + p0*p1*p13*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p2*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p1*p2*p23*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p01*p1*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p13*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - x) + p01*p13*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p13*p2*(1 - p0)*(1 - p01)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p2)*(1 - x) + p02*p2*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p02*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p1*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p1*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2) + p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23) + p13*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)

    P0110 = p0*p01*p1*p13*p23*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p2*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p23*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p0*p02*p1*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p13*p2*p23*(1 - p01)*(1 - p1)*(1 - p3)*(1 - x) + p0*p02*p13*p3*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p0*p1*p13*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p02*p1*p13*p3*(1 - p0)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p2*p23*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - x) + p01*p02*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p3) + p01*p1*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p13*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p1*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p23)*(1 - p3) + p02*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23) + p02*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p1*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p1*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p13*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x)
    P0101 = p0*p01*p02*p1*x*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p1*p13*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p2*p23*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p1*p2*p3*(1 - p01)*(1 - p13)*(1 - p23)*(1 - x) + p0*p02*p1*p23*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p13*p2*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p13*p23*p3*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p0*p1*p13*p23*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3) + p0*p2*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p01*p02*p1*p13*p2*(1 - p0)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p2*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p01*p02*p23*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p1*p2*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p1*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2) + p01*p13*p2*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23) + p01*p13*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p02*p1*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23) + p02*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13) + p02*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p1*p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p1*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p13*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x)
    P0011 = p0*p01*p02*p2*x*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p1*p2*p3*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p1*p23*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p13*p2*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p13*p23*p3*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p0*p02*p1*p13*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p2*p23*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p02*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p1*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p13*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3) + p0*p13*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p01*p02*p1*p2*p23*(1 - p0)*(1 - p13)*(1 - p3)*(1 - x) + p01*p02*p1*p3*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p13*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p13*p3*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23) + p01*p2*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13) + p01*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p1*p2*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3) + p02*p1*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2) + p02*p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23) + p02*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3) + p1*p13*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p1*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x)


    P1110 = p0*p01*p02*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p1*p3*x*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p01*p13*x*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p2*p3*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p02*p23*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p1*p2*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p23*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p0*p13*p2*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p0*p13*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p1*p23*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p02*p13*p2*x*(1 - p0)*(1 - p1)*(1 - p23)*(1 - p3) + p01*p1*p13*p2*p3*(1 - p0)*(1 - p02)*(1 - p23)*(1 - x) + p01*p1*p13*p23*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p01*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p02*p1*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - x) + p02*p1*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p3)*(1 - x) + p02*p13*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p1*p13*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)
    P1101 = p0*p01*p02*p2*p3*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p02*p23*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p1*p2*x*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p13*p23*x*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p0*p02*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p1*p2*p23*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p0*p1*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p13*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p0*p13*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p1*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p13*p3*x*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23) + p01*p1*p13*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p2*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p01*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p02*p1*p2*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - x) + p02*p1*p23*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p02*p13*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p02*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p1*p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23) + p1*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3) + p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)
    P1011 = p0*p01*p02*p1*p3*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p02*p13*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p1*p2*x*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p02*p13*p23*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3) + p0*p1*p13*p2*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p13*p23*p3*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p0*p2*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p2*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p02*p23*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2) + p01*p1*p2*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p01*p1*p23*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p13*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p01*p13*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p02*p1*p13*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p2*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p02*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p1*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13) + p1*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3) + p13*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)
    P0111 = p0*p01*p1*p13*p2*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p2*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p23*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p1*p2*p23*(1 - p01)*(1 - p13)*(1 - p3)*(1 - x) + p0*p02*p1*p3*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p13*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p13*p3*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23) + p0*p2*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13) + p0*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p1*p13*(1 - p0)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p2*p23*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p01*p02*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p1*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p13*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p3) + p01*p13*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p02*p1*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p3) + p02*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p02*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2) + p1*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p1*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p13*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x)

    P1111 = p0*p01*p02*p1*p13*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p2*p23*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p02*p3*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p1*x*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p13*p3*x*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p0*p02*p2*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p02*p23*p3*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2) + p0*p1*p2*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p0*p1*p23*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p13*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p13*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p01*p02*p1*p2*x*(1 - p0)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p02*p13*p23*x*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p3) + p01*p1*p13*p2*(1 - p0)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p13*p23*p3*(1 - p0)*(1 - p02)*(1 - p2)*(1 - x) + p01*p2*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p01*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p02*p1*p2*p23*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p3)*(1 - x) + p02*p1*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p02*p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - x) + p02*p13*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p1*p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p3) + p1*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23) + p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13) + x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)

    eq1 = v0001 - (  P0001
                   + P1001 + P0101 + P0011
                   + P1101 + P1011 + P0111 
                   + P1111)
    
    eq2 = v0010 - (  P0010
                   + P1010 + P0110 + P0011+
                   + P1011 + P1110 + P0111
                   + P1111)
    
    eq3 = v0100 - (P0100 + 
                   P1100 + P0110 + P0101+
                   P1110 + P0111 + P1101+
                   P1111)
    
    eq4 = v1000 - (P1000 + 
                   P1100 + P1010 + P1001+
                   P1110 + P1011 + P1101
                   + P1111)
    
    eq5 = v1100 - (P1100 +
                    P1110 +P1101
                    +P1111)
    
    eq6 = v1010 - (P1010+
                   P1110 + P1011
                   +P1111)
    
    eq7 = v0011 - (P0011+
                   +P0111+P1011+
                   P1111)
    
    eq8 = v0101-(P0101 +
                 P0111+P1101
                 +P1111)
    
    eq9 = v1111-P1111
    return [eq1,eq2,eq3,eq4,eq5,eq6,eq7,eq8,eq9]



def objective(vars, *args):
    return np.sum(np.square(equations(vars, *args)))

def apply_4_pnt_method(v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011,min_bound,max_bound,method):
      
      v_values = [v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011]   #Observed average values
      
      if method=="minimize":

            initial_guess =  [0.04]*9
            
            bounds = [(min_bound, max_bound)] * 9
            result = minimize(objective, initial_guess, args=tuple(v_values), 
                              method='Nelder-Mead', bounds=bounds)
            variable_names = ["p0", "p1", "p2", "p3", "x", "p01", "p02", "p13", "p23"]
            
            solution_dict = dict(zip(variable_names, result.x))
            
            print("cost:",result.fun)
            
      elif method=="least_squares":
           

            # Bounds for variables
            bounds = ([min_bound] * 9, [max_bound] * 9)

            # Initial guess
            initial_guess = np.ones(9)*0.01

            # Example observed values
            v_values = [v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011]  

            # Solve the system
            result = least_squares(equations, initial_guess, args=tuple(v_values), bounds=bounds)

            
            # Output results
            
            solution_dict = dict(zip(["p0", "p1", "p2", "p3", "x", "p01", "p02", "p13", "p23"], result.x))
            
            print("optimality:",result.optimality)
      else:
           raise Exception("Method not available")

             

      return solution_dict


def solve_for_probs_and_update_intermediate_edges(min_bound,max_bound,method,vijkl):
    ''' 
    Solve the system of equations per round. For interemediate rounds, we will over-estimate 
    bulk/time-edges. For example, if we have 2 rounds, d=3, then we have the dets:
    D0 D1 D2 D3
    D2 D3 D4 D5

    We do not estimate correctly D2-D3, D2-, D3-, because these are included in both 4-point regions
    where we solve our equations. For these edges, we just need to update the solution, using the 
    2nd 4-point event these nodes participate in


    min_bound/max_bound: bounds of probabilities for the solver
        method: "least_squares" or "minimize" (both seem to perform pretty well)
        '''
    
    all_dicts={}
    for key in vijkl.keys():

        # solution_dict=apply_4_pnt_method(v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011,min_bound,max_bound,method)
        v0001 = vijkl[key]["v0001"]
        v0010 = vijkl[key]["v0010"]
        v0100 = vijkl[key]["v0100"]
        v1000 = vijkl[key]["v1000"]
        v1111 = vijkl[key]["v1111"]
        v1100 = vijkl[key]["v1100"]
        v1010 = vijkl[key]["v1010"]
        v0101 = vijkl[key]["v0101"]
        v0011 = vijkl[key]["v0011"]

        solution_dict=apply_4_pnt_method(v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011,min_bound,max_bound,method)


        #rename the dicts of solution_dict:
        
        solution_dict[key[0]] = solution_dict.pop("p0")
        solution_dict[key[1]] = solution_dict.pop("p1")
        solution_dict[key[2]] = solution_dict.pop("p2")
        solution_dict[key[3]] = solution_dict.pop("p3")

        solution_dict[(key[0],key[1])] = solution_dict.pop("p01")
        solution_dict[(key[0],key[2])] = solution_dict.pop("p02")
        solution_dict[(key[1],key[3])] = solution_dict.pop("p13")
        solution_dict[(key[2],key[3])] = solution_dict.pop("p23")
        solution_dict[key]             = solution_dict.pop("x")
                            

        all_dicts[key]=solution_dict

    #Find the nodes which participate in both 4-point events
    # all_keys = list(all_dicts.keys())
    # print(all_keys)
    # for l in range(len(all_keys)):

    #     key1 = all_keys[l]
    #     set1 = set(key1)
        
    #     for m in range(l+1,len(all_keys)):

    #         key2 = all_keys[m]
    #         set2 = set(key2)

    #         common_dets = set.intersection(set1,set2)

    #         if common_dets is not None:
    #             #get d_i, d_j, di-dj edges and update using the p4 value from the second dictionary
                
    #             det_name      = list(common_dets)[0]
    #             pval          = all_dicts[key1][det_name]
    #             four_pnt_key  = list(all_dicts[key2].keys())[-1] #get the 4 pnt event
    #             p4            = all_dicts[key2][four_pnt_key]
                
    #             # pval = (pval-p4)/(1-2*p4)

    #             # four_pnt_key  = list(all_dicts[key1].keys())[-1] #get the 4 pnt event
    #             # p4            = all_dicts[key1][four_pnt_key]

    #             # pval = (pval-p4)/(1-2*p4)

    #             temp = all_dicts[key1][("D0","D2")]
    #             pval = (pval-temp)/(1-2*temp)

    #             print("new pval:",pval,"for D:",det_name)
    #             break



    #Update the edge using the last 4-point event
    
    return all_dicts


def edge_dicts(num_ancilla,num_rounds):

    bulk_edges = {}
    time_edges = {}
    bd_edges   = {}
    #Time edges
    for rd1 in range(num_rounds):
        rd2=rd1+1
        for anc1 in range(num_ancilla):
            anc2 = anc1

            indx1 = anc1+num_ancilla*rd1
            indx2 = anc2+num_ancilla*rd2

            name = ("D"+str(indx1),"D"+str(indx2))

            time_edges[name]=0
    
    #Space edges
    for rd1 in range(num_rounds+1):
        rd2 = rd1 
        for anc1 in range(num_ancilla-1):
            anc2 = anc1 +1

            indx1 = anc1+num_ancilla*rd1
            indx2 = anc2+num_ancilla*rd2
            name = ("D"+str(indx1),"D"+str(indx2))
            bulk_edges[name]=0
    
    for rd1 in range(num_rounds+1):

        anc1 = 0
        
        indx1 = anc1+num_ancilla*rd1
        name  = ("D"+str(indx1))
        bd_edges[name]=0

        anc1  = num_ancilla-1
        indx1 = anc1+num_ancilla*rd1
        name  = ("D"+str(indx1))
        
        bd_edges[name]=0

    return bulk_edges,time_edges,bd_edges

def all_dicts_into_pij(all_dicts,bulk_edges,time_edges,bd_edges):
    '''Transform into pij_bulk, pij_time, pij_bd, p4_cnts'''
    
    pij_bulk = {}
    pij_time = {}
    pij_bd   = {}
    p4_cnts  = {}

    for key in all_dicts.keys():

        for key2 in all_dicts[key].keys():

            if key2[0]=="D": #this is a bd edge
                pij_bd[key2]=all_dicts[key][key2]
            elif key2 in bulk_edges:
                pij_bulk[key2]=all_dicts[key][key2]
            elif key2 in time_edges:
                pij_time[key2]=all_dicts[key][key2]
            else: #4 pnt event
                p4_cnts[key2]=all_dicts[key][key2]

    return pij_bulk,pij_time,pij_bd,p4_cnts


def update_edges_after_4_pnt_estimation(pij_bulk,pij_time,pij_bd,p4_cnts,num_ancilla,num_rounds,vi_mean,distance):

    num_rounds +=1

    #Bulk edges (same rd):
    for rd1 in range(1,num_rounds-1):
        rd2=rd1
        for anc1 in range(num_ancilla-1):
            
            anc2  = anc1+1
            indx1 = anc1 + num_ancilla*rd1
            indx2 = anc2 + num_ancilla*rd2
            name  = ("D"+str(indx1),"D"+str(indx2))

            name_of_4_pnt_event =  ("D"+str(indx1-num_ancilla),"D"+str(indx2-num_ancilla),
                                    "D"+str(indx1),"D"+str(indx2))
            
            
            pnew1               = p4_cnts[name_of_4_pnt_event]
            name_of_4_pnt_event =  ("D"+str(indx1),"D"+str(indx2),"D"+str(indx1+num_ancilla),"D"+str(indx2+num_ancilla))            

            # pnew2 = p4_cnts[name_of_4_pnt_event] 
            pij_bulk[name]= (pij_bulk[name]-pnew1)/(1-2*(pnew1))

    #I think the time edges need to be updated for d>3?
    #and we need to update those time edges that do not correspond to anc=0, num_ancilla-1

    if distance>3:

        for rd1 in range(num_rounds-1):
            rd2 = rd1+1

            for anc1 in range(1,num_ancilla-1):
                anc2  = anc1 
                indx1 = anc1 + num_ancilla*rd1
                indx2 = anc2 + num_ancilla*rd2

                name  = ("D"+str(indx1),"D"+str(indx2))

                for key in p4_cnts.keys():

                    if name[0] in key and name[1] in key:
                        pnew1 = p4_cnts[key]
                        pij_time[name]=(pij_time[name]-pnew1)/(1-2*pnew1)
                        break

            
    #Bd-edges: subtract 4-pnt events, subtract time events and subtract bulk events
    pij_bd = {}
    for rd in range(num_rounds):
        
        anc   = 0
        indx1 = anc + num_ancilla*rd
        v0    = vi_mean[rd,anc]
        name  = "D"+str(indx1)

        DENOM = 1

        #subtract previous time event and next time event
        if rd>0: #get previous rd
            rd2   = rd-1
            indx2 = anc+num_ancilla*rd2
            DENOM *= 1-2*pij_time[("D"+str(indx2),"D"+str(indx1))]

        if rd<(num_rounds-1):
            rd2   = rd+1
            indx2 = anc+num_ancilla*rd2
            
            DENOM *= 1-2*pij_time[("D"+str(indx1),"D"+str(indx2))]
        
        #Get bulk edge:
        anc2  = anc+1
        indx2 = anc2+num_ancilla*rd
        DENOM *= 1-2*pij_bulk[("D"+str(indx1),"D"+str(indx2))]

        #Get all relevant 4-pnt events
        for key in p4_cnts.keys():

            if name in key:

                DENOM *=1-2*p4_cnts[key]
                

        pij_bd[name] = 1/2+(v0-1/2)/DENOM


    #Do the same for the other bd edge
    for rd in range(num_rounds):
        
        anc   = num_ancilla-1
        indx1 = anc + num_ancilla*rd
        v0    = vi_mean[rd,anc]
        name  = "D"+str(indx1)

        DENOM = 1

        #subtract previous time event and next time event
        if rd>0: #get previous rd
            rd2   = rd-1
            indx2 = anc+num_ancilla*rd2
            DENOM *= 1-2*pij_time[("D"+str(indx2),"D"+str(indx1))]

        if rd<(num_rounds-1):
            rd2   = rd+1
            indx2 = anc+num_ancilla*rd2
            
            DENOM *= 1-2*pij_time[("D"+str(indx1),"D"+str(indx2))]
        
        #Get bulk edge:
        anc2  = anc-1
        indx2 = anc2+num_ancilla*rd
        DENOM *= 1-2*pij_bulk[("D"+str(indx2),"D"+str(indx1))]

        #Get all relevant 4-pnt events
        for key in p4_cnts.keys():

            if name in key:

                DENOM *=1-2*p4_cnts[key]
                
        pij_bd[name] = 1/2+(v0-1/2)/DENOM


    return pij_bulk, pij_time, pij_bd, p4_cnts


def construct_dem(pij_bulk,pij_bd,pij_time,p4_cnts):

    my_DEM = stim.DetectorErrorModel()

    for key in pij_bulk.keys():
        det_list = []
        for det in key:
            ind = int(det[1:])
            det_list.append(stim.target_relative_detector_id(ind))
        
        det_list.append(stim.target_logical_observable_id(0))

        my_DEM.append("error",pij_bulk[key],det_list)

    for key in pij_bd.keys():
        
        det_list=[]
        ind = int(key[1:])
        det_list.append(stim.target_relative_detector_id(ind))
        det_list.append(stim.target_logical_observable_id(0))

        my_DEM.append("error",pij_bd[key],det_list)

    for key in pij_time.keys():
        det_list = []
        for det in key:
            ind = int(det[1:])
            det_list.append(stim.target_relative_detector_id(ind))
        
        my_DEM.append("error",pij_time[key],det_list)
    
    for key in p4_cnts.keys():
        det_list=[]
        for det in key:
            ind=int(det[1:])
            det_list.append(stim.target_relative_detector_id(ind))
        
        my_DEM.append("error",p4_cnts[key],det_list)

    return my_DEM
