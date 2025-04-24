import numpy as np
from scipy.optimize import minimize,least_squares
from sympy import symbols
from itertools import combinations


def get_4_pnt_events(defect_data,p_depol_after):
    '''
    Collect the <v_{ijkl}> for all possible 4 point events in the detector error model.
    The 4-pnt events consist of 2 consecutive detectors in the same round, and the same 2 detectors shifted by one round.

    Input: 
        defect_matrix: xarray of dims num_shots x num_rounds+1 x num_ancilla
        p_depol_after: either [] or a numerical value of the 2-qubit depolarizing error after the gates.
    
    Output: 
        p4_cnts: dictionary with keys the events of the form ("Di","Dj","Dk","Dl") and values the average # of times where all 4 detectors fire together.
    '''


    num_shots    = np.size(defect_data,axis=0)     
    num_rounds   = np.size(defect_data,axis=1)     
    num_anc      = np.size(defect_data,axis=2)     

    p4_cnts = {}

    for anc1 in range(num_anc-1):
       
        anc2 = anc1+1
        anc3 = anc1
        anc4 = anc2

        for round1 in range(num_rounds-1):

            round2 = round1
            round3 = round1+1
            round4 = round3

            locs = defect_data[:,round1,anc1] & defect_data[:,round2,anc2]\
                 & defect_data[:,round3,anc3] & defect_data[:,round4,anc4]


            #Indices are sorted
            indx1 = anc1+num_anc*round1
            indx2 = anc2+num_anc*round2
            indx3 = anc3+num_anc*round3
            indx4 = anc4+num_anc*round4

            name_of_4_pnt_event = (f"D{indx1}",f"D{indx2}",f"D{indx3}",f"D{indx4}")
            
            p4_cnts[name_of_4_pnt_event]=sum(locs)/num_shots     

    if p_depol_after!=[]:
        for key in p4_cnts.keys():
            p4_cnts[key]=2/3*4/5*p_depol_after
   

    return p4_cnts


def get_vijkl(p4_cnts: dict, num_rounds: int,num_ancilla: int, vi_mean, vivj_mean, det_inds_rd_anc: dict):
    '''
    Prepare the dictionaries for the observed number of counts of single, two-point and 4-point events,
    that will be fed into the numerical equations to solve.

    Input:
        p4_cnts: dictionary with keys the events of the form ("Di","Dj","Dk","Dl") and values the average # of times where all 4 detectors fire together.
        num_rounds: # of QEC rounds (int), excluding the final stabilizer reconstruction
        num_ancilla: # of detectors per rounds (int)
        vi_mean: # of times a detector fires/num_shots (np array of dims # of rounds x # of detectors per round)
        vivj_mean: average # of two-point coincidences (np array of dimds # of rounds x # of roudns x # of detectors per rounds x # of detectors per round)

    Output:
        events: dictionary whose keys are all the 4 point events we want to estimate. Within this outer-most dictionary we have another dictionary which
        holds all <vi>, <vivj> and <vivjvkvl> values for the respective 4-point event we want to solve the equations for.

    '''
    num_rounds +=1

    events ={}

    for key in p4_cnts.keys():

        temp_dict = {}
        det_inds  = []
        
        det_inds = [det_inds_rd_anc[int(det[1:])] for det in key]
        
        inds1,inds2,inds3,inds4 = det_inds
        
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


def form_equation_terms(target_inds: list, order: int):
    '''Form terms for the configuration probabilities P(x0,x1,x2,x3), such that the indices for the probabilities 
       p_{target_ind[0]}..., p_{target_ind[-1]} appear and odd # of times in the equation term, and the remaining indices 
       appear an even # of times in the expression. We have the correspondence [0,1,2,3] <-> [x0,x1,x2,x3], so the term
       we obtain corresponds to odd # of counts for some of the x_j giving rise to a particular term of some input order for P(x0,x1,x2,x3).
       The probability set we have consists of p_0, p_1, p_2, p_3, p_{01}, p_{02}, p_{13}, p_{23}, p_{0123}.

    Input:
        target_inds: list that contains only indices from the elements [0,1,2,3] which correspond to x0,x1,x2,x3 for the configuration probability P(x0,x1,x2,x3)
        order: how many error events combine to give rise to the particular equation term (positive integer)
    
    Output:
        accepted_combs: list of lists corresponding to the indices that need to be combined to give rise to a particular detection pattern
    '''

    if order<=0:
        raise ValueError("Order of equation has to be bigger than 0.")
    

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

                if 0 in elem:
                    cnt_0+=1
                if 1 in elem:
                    cnt_1+=1
                if 2 in elem:
                    cnt_2+=1
                if 3 in elem:
                    cnt_3 +=1 
            
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
            raise Exception("Invalid indices. The input target indices should be in [0,1,2,3].")

    all_inds  = [0,1,2,3]
    rest_inds = [k for k in all_inds if k not in target_inds]

    p0             = [0] ; p1  = [1] ; p2  = [2] ; p3  = [3] 
    p01            = [0,1]; p02 = [0,2] ;p13 = [1,3] ;p23 = [2,3] 
    p0123          = [0,1,2,3]
    all_elements   = [p0,p1,p2,p3,p01,p02,p13,p23,p0123]
    all_combs      = list(combinations(all_elements,order))
    accepted_combs = []

    for comb in all_combs:  

        cnt_0,cnt_1,cnt_2,cnt_3 = 0,0,0,0

        for elem in comb:

            if 0 in elem:
                cnt_0+=1
            if 1 in elem:
                cnt_1+=1
            if 2 in elem:
                cnt_2+=1
            if 3 in elem:
                cnt_3 +=1 

        list_of_cnts = [cnt_0,cnt_1,cnt_2,cnt_3]

        not_acceptable = any(list_of_cnts[p] % 2 == 0 for p in target_inds) #Every element in target_inds needs to appear and odd # of times in the combination
        
        if not_acceptable==True:
            continue
        else:
            not_acceptable = any(list_of_cnts[p] % 2 == 1 for p in rest_inds)  #Every element in rest_inds needs to appear and even # of times in the combination
        
            if not_acceptable==False:
                accepted_combs.append(comb)

    return accepted_combs


def form_particular_eq_for_one_truncation_order(target_inds: list, order: int):
    '''Form the full equation for the configuration probability P(x0,x1,x2,x3) using the individual equation terms, and up to some maximum order.

       Input:
            target_inds: list that contains only indices from the elements [0,1,2,3] which correspond to x0,x1,x2,x3 for the configuration probability P(x0,x1,x2,x3)
            order: how many physical errors combine to give rise to the detection pattern  (int)

        Output:
            all_eqns: symbolic equation for the configuration probability P(x0,x1,x2,x3) where target_inds indicate which detectors fire, for the given order
    '''

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


def form_all_4_pnt_equations(max_truncation_order: int):
    '''Form all the equations for the P(x0,x1,x2,x3) configuration probabilities.
    
    Input:
        max_truncation_order: maximum order to keep for the equations.
    Output:
        P: dictionary of symbolic equations where the names are all bit-combinations on 4 digits. '''


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


def equations_O4(vars, v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011):
    '''Symbolic equations up to O(p^4).

    Input:
        vars: the unknowns that we solve fore
        v0001: the average counts where the last detector fires
        v0010: the average counts where the second-to-last detector fires
        (simularly for other inputs)

    Output:
        System of 9 equations of the form P(x0,x1,x2,x3) = const that we will solve with least-squares method.
        
    '''
    p0, p1, p2, p3, x, p01, p02, p13, p23 = vars
    
    P0001 =   p0*p01*p02*x*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p1*p3*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p13*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p2*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p02*p23*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p1*p2*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p13*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p01*p02*p1*p23*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p13*p2*(1 - p0)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p01*p2*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p02*p1*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23) + p1*p13*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x)

    P0010 =   p0*p01*p1*p2*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p13*p23*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p3*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p13*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p1*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p13*p3*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p01*p1*p13*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p01*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p02*p1*p23*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3) + p02*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3) + p1*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - x) + p1*p13*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x)
    
    P0100 = p0*p01*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p2*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p13*p23*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p2*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p02*p2*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p23*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p01*p1*p23*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p13*p2*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p02*p1*p13*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3) + p02*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p1*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - x) + p1*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p13*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x)
    
    P1000 = p0*p01*p23*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p02*p13*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p1*p13*p3*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p0*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p0*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p1*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p13*p2*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p01*p13*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p02*p1*p13*p23*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p3)*(1 - x) + p02*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p02*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p1*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23) + p1*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p13*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)
    
    

    P1100 = p0*p01*p02*p2*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p3*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p1*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p13*p2*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p0*p13*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p13*x*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p1*p13*p3*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p01*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p01*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p2*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p23*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - x) + p02*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - x) + p02*p13*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p1*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3) + p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)
    
    
    P0011 = p0*p01*p1*p23*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p13*p2*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p13*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p2*p23*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p02*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p1*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p13*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p01*p02*p1*p3*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p13*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p1*p2*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3) + p02*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3) + p1*p13*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p1*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x)
    
    P1010 = p0*p01*p02*p1*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p1*p13*p23*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p0*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p01*p02*p23*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p1*p2*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p23*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p01*p13*p2*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p01*p13*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p02*p1*p13*p3*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - x) + p02*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - x) + p02*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p1*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3) + p1*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p13*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)
    
    
    P0101 = p0*p01*p1*p13*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p2*p23*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p1*p23*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p13*p2*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p2*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p01*p02*p2*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p01*p02*p23*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p1*p2*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p13*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p02*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p1*p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p1*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p13*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x)
    
    P1001 = p0*p01*p2*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p02*p1*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p1*p13*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p2*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p1*p2*p23*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p01*p1*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p13*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p13*p2*(1 - p0)*(1 - p01)*(1 - p23)*(1 - p3)*(1 - x) + p02*p2*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p02*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p1*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p1*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2) + p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23) + p13*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)
    
    P0110 = p0*p01*p2*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p23*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p0*p02*p1*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p13*p3*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p0*p1*p13*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p02*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p13*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23) + p02*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p1*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p1*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p13*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x)
    
    P1110 = p0*p01*p02*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p13*x*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p23*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p1*p2*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p23*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p0*p13*p2*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p0*p13*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p01*p1*p13*p23*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p01*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p02*p1*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p3)*(1 - x) + p02*p13*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p1*p13*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)

    P0111 = p0*p01*p2*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p23*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p1*p3*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p13*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p1*p13*(1 - p0)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p2*p23*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p01*p02*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p1*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p13*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p02*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p02*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2) + p1*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p1*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p13*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x)
    
    P1011 = p0*p01*p02*p13*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p1*p13*p2*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p0*p2*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p2*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p1*p2*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p01*p1*p23*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p13*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p01*p13*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p02*p1*p13*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p2*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p02*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p1*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3) + p13*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)

    P1101 = p0*p01*p02*p23*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p1*p2*p23*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p0*p1*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p13*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p1*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p1*p13*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p2*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p01*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p02*p1*p2*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - x) + p02*p1*p23*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p02*p13*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p02*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p1*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3) + p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)
    
    P1111 = p0*p01*p02*p3*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p1*x*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p2*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p1*p2*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p0*p1*p23*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p13*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p13*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p01*p1*p13*p2*(1 - p0)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p01*p2*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p01*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p02*p1*p2*p23*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p3)*(1 - x) + p02*p1*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p02*p13*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p1*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23) + p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13) + x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)

    
    

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

def equations_O5(vars, v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011):
    '''Symbolic equations up to O(p^5).

    Input:
        vars: the unknowns that we solve fore
        v0001: the average counts where the last detector fires
        v0010: the average counts where the second-to-last detector fires
        (simularly for other inputs)

    Output:
        System of 9 equations of the form P(x0,x1,x2,x3) = const that we will solve with least-squares method.
        
    '''
    p0, p1, p2, p3, x, p01, p02, p13, p23 = vars
    
    P0001 =   p0*p01*p02*x*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p1*p2*p23*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p1*p3*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p13*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p13*p2*(1 - p01)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p2*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p02*p23*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p1*p2*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p1*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2) + p0*p13*p2*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23) + p0*p13*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p01*p02*p1*p2*p3*(1 - p0)*(1 - p13)*(1 - p23)*(1 - x) + p01*p02*p1*p23*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p13*p2*(1 - p0)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p13*p23*p3*(1 - p0)*(1 - p1)*(1 - p2)*(1 - x) + p01*p1*p13*p23*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p3) + p01*p2*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p02*p1*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p3) + p02*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23) + p1*p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - x) + p1*p13*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x)

    P0010 = p0*p01*p02*p23*x*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p01*p1*p2*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p1*p23*p3*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p0*p01*p13*p2*p3*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p0*p01*p13*p23*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p1*p13*p3*(1 - p01)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p2*p23*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - x) + p0*p02*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3) + p0*p1*p3*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p13*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p1*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p13*p2*p23*(1 - p0)*(1 - p1)*(1 - p3)*(1 - x) + p01*p02*p13*p3*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p01*p1*p13*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p01*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p02*p1*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23) + p02*p1*p23*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3) + p02*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3) + p02*p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2) + p1*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - x) + p1*p13*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x)
    
    P0100 = p0*p01*p02*p13*x*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p1*p13*p3*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p2*p23*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p0*p01*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p2*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p23*p3*(1 - p01)*(1 - p13)*(1 - p2)*(1 - x) + p0*p02*p13*p2*p3*(1 - p01)*(1 - p1)*(1 - p23)*(1 - x) + p0*p02*p13*p23*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p1*p13*p2*x*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3) + p0*p2*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p02*p1*p13*p23*(1 - p0)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p2*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p23*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p01*p1*p2*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23) + p01*p1*p23*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p13*p2*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p01*p13*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2) + p02*p1*p13*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3) + p02*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p1*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - x) + p1*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p13*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x)
    
    P1000 = p0*p01*p02*p1*p2*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p23*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p2*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p01*p23*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p02*p1*p3*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p02*p13*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p1*p13*p2*p23*(1 - p01)*(1 - p02)*(1 - p3)*(1 - x) + p0*p1*p13*p3*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p0*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p0*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p1*p13*x*(1 - p0)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p2*p23*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p3) + p01*p02*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p1*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - x) + p01*p1*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p13*p2*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p01*p13*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p02*p1*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p23)*(1 - x) + p02*p1*p13*p23*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p3)*(1 - x) + p02*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p02*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p1*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23) + p1*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p13*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)
    
    

    P1100 = p0*p01*p02*p2*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p23*p3*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p0*p01*p1*p23*x*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p01*p13*p2*x*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p0*p02*p1*p13*x*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p2*p23*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p02*p3*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p1*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - x) + p0*p1*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p13*p2*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p0*p13*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p1*p3*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p02*p13*x*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p1*p13*p2*p23*(1 - p0)*(1 - p02)*(1 - p3)*(1 - x) + p01*p1*p13*p3*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p01*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p01*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p2*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p23*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - x) + p02*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - x) + p02*p13*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p1*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3) + p1*p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2) + p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)
    
    
    P0011 = p0*p01*p02*p2*x*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p1*p2*p3*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p1*p23*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p13*p2*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p13*p23*p3*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p0*p02*p1*p13*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p2*p23*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p02*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p1*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p13*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3) + p0*p13*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p01*p02*p1*p2*p23*(1 - p0)*(1 - p13)*(1 - p3)*(1 - x) + p01*p02*p1*p3*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p13*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p13*p3*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23) + p01*p2*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13) + p01*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p1*p2*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3) + p02*p1*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2) + p02*p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23) + p02*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3) + p1*p13*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p1*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x)
    
    P1010 = p0*p01*p02*p1*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p3*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p1*p13*x*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p2*p23*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p01*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p02*p1*p23*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p02*p13*p2*x*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3) + p0*p1*p13*p2*p3*(1 - p01)*(1 - p02)*(1 - p23)*(1 - x) + p0*p1*p13*p23*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p0*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p01*p02*p2*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23) + p01*p02*p23*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p1*p2*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p23*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p01*p13*p2*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p01*p13*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p02*p1*p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p3)*(1 - x) + p02*p1*p13*p3*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - x) + p02*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - x) + p02*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p1*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3) + p1*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p13*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1) + p13*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)
    
    
    P0101 = p0*p01*p02*p1*x*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p1*p13*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p2*p23*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p1*p2*p3*(1 - p01)*(1 - p13)*(1 - p23)*(1 - x) + p0*p02*p1*p23*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p13*p2*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p13*p23*p3*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p0*p1*p13*p23*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3) + p0*p2*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p01*p02*p1*p13*p2*(1 - p0)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p2*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p01*p02*p23*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p1*p2*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p1*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2) + p01*p13*p2*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23) + p01*p13*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p02*p1*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23) + p02*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13) + p02*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p1*p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p1*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p13*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x)
    
    P1001 = p0*p01*p02*p1*p23*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p2*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p2*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p23*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p0*p02*p1*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p13*p3*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23) + p0*p1*p13*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p2*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p1*p2*p23*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p01*p1*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p13*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - x) + p01*p13*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p13*p2*(1 - p0)*(1 - p01)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p2)*(1 - x) + p02*p2*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p02*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p1*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p1*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2) + p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23) + p13*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)
    
    P0110 = p0*p01*p1*p13*p23*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p2*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p23*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p0*p02*p1*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p13*p2*p23*(1 - p01)*(1 - p1)*(1 - p3)*(1 - x) + p0*p02*p13*p3*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p0*p1*p13*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p02*p1*p13*p3*(1 - p0)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p2*p23*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - x) + p01*p02*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p3) + p01*p1*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p13*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p1*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p23)*(1 - p3) + p02*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23) + p02*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p1*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p1*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p13*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x)
    
    P1110 = p0*p01*p02*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p1*p3*x*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p01*p13*x*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p2*p3*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p02*p23*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p1*p2*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p23*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p0*p13*p2*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p0*p13*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p1*p23*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p02*p13*p2*x*(1 - p0)*(1 - p1)*(1 - p23)*(1 - p3) + p01*p1*p13*p2*p3*(1 - p0)*(1 - p02)*(1 - p23)*(1 - x) + p01*p1*p13*p23*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p01*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p02*p1*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - x) + p02*p1*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p3)*(1 - x) + p02*p13*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p1*p13*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)

    P0111 = p0*p01*p1*p13*p2*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p2*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p23*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p1*p2*p23*(1 - p01)*(1 - p13)*(1 - p3)*(1 - x) + p0*p02*p1*p3*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p13*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p13*p3*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23) + p0*p2*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13) + p0*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p1*p13*(1 - p0)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p2*p23*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p01*p02*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p1*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p13*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p3) + p01*p13*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p02*p1*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p3) + p02*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p02*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2) + p1*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p1*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p13*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x)
    
    P1011 = p0*p01*p02*p1*p3*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p02*p13*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p1*p2*x*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p02*p13*p23*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3) + p0*p1*p13*p2*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p13*p23*p3*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p0*p2*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p2*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p02*p23*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2) + p01*p1*p2*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p01*p1*p23*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p13*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p01*p13*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p02*p1*p13*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p2*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p02*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p1*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13) + p1*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3) + p13*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)

    P1101 = p0*p01*p02*p2*p3*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p02*p23*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p1*p2*x*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p13*p23*x*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p0*p02*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p1*p2*p23*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p0*p1*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p13*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p0*p13*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p1*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p13*p3*x*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23) + p01*p1*p13*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p2*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p01*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p02*p1*p2*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - x) + p02*p1*p23*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p02*p13*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p02*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p1*p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23) + p1*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3) + p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)
    
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


def equations_O6(vars, v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011):
    '''Symbolic equations up to O(p^6).

    Input:
        vars: the unknowns that we solve fore
        v0001: the average counts where the last detector fires
        v0010: the average counts where the second-to-last detector fires
        (simularly for other inputs)

    Output:
        System of 9 equations of the form P(x0,x1,x2,x3) = const that we will solve with least-squares method.
        
    '''
    p0, p1, p2, p3, x, p01, p02, p13, p23 = vars
    
    P0001 =   p0*p01*p02*x*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p1*p2*p23*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p1*p3*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p13*p2*p23*p3*(1 - p02)*(1 - p1)*(1 - x) + p0*p01*p13*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p13*p2*(1 - p01)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p13*p23*p3*(1 - p01)*(1 - p2)*(1 - x) + p0*p02*p2*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p02*p23*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p1*p2*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p1*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2) + p0*p13*p2*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23) + p0*p13*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p01*p02*p1*p2*p3*(1 - p0)*(1 - p13)*(1 - p23)*(1 - x) + p01*p02*p1*p23*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p13*p2*(1 - p0)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p13*p23*p3*(1 - p0)*(1 - p1)*(1 - p2)*(1 - x) + p01*p1*p13*p2*p3*x*(1 - p0)*(1 - p02)*(1 - p23) + p01*p1*p13*p23*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p3) + p01*p2*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p02*p1*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p13) + p02*p1*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p3) + p02*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23) + p1*p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - x) + p1*p13*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x)

    P0010 = p0*p01*p02*p2*p3*x*(1 - p1)*(1 - p13)*(1 - p23) + p0*p01*p02*p23*x*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p01*p1*p2*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p1*p23*p3*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p0*p01*p13*p2*p3*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p0*p01*p13*p23*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p1*p13*p2*p23*(1 - p01)*(1 - p3)*(1 - x) + p0*p02*p1*p13*p3*(1 - p01)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p2*p23*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - x) + p0*p02*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3) + p0*p1*p3*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p13*p2*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p1) + p0*p13*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p1*p2*p23*p3*(1 - p0)*(1 - p13)*(1 - x) + p01*p02*p1*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p13*p2*p23*(1 - p0)*(1 - p1)*(1 - p3)*(1 - x) + p01*p02*p13*p3*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p01*p1*p13*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p01*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p02*p1*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23) + p02*p1*p23*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3) + p02*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3) + p02*p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2) + p1*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - x) + p1*p13*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x)
    
    P0100 = p0*p01*p02*p1*p3*x*(1 - p13)*(1 - p2)*(1 - p23) + p0*p01*p02*p13*x*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p1*p13*p2*p23*(1 - p02)*(1 - p3)*(1 - x) + p0*p01*p1*p13*p3*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p2*p23*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p0*p01*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p2*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p23*p3*(1 - p01)*(1 - p13)*(1 - p2)*(1 - x) + p0*p02*p13*p2*p3*(1 - p01)*(1 - p1)*(1 - p23)*(1 - x) + p0*p02*p13*p23*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p1*p13*p2*x*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3) + p0*p1*p13*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p2) + p0*p2*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p02*p1*p13*p2*p3*(1 - p0)*(1 - p23)*(1 - x) + p01*p02*p1*p13*p23*(1 - p0)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p2*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p23*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p01*p1*p2*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23) + p01*p1*p23*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p13*p2*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p01*p13*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2) + p02*p1*p13*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3) + p02*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p1*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - x) + p1*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p13*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x)
    
    P1000 = p0*p01*p02*p1*p2*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p1*p23*p3*(1 - p13)*(1 - p2)*(1 - x) + p0*p01*p02*p13*p2*p3*(1 - p1)*(1 - p23)*(1 - x) + p0*p01*p02*p13*p23*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p1*p13*p2*x*(1 - p02)*(1 - p23)*(1 - p3) + p0*p01*p2*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p01*p23*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p02*p1*p2*p23*x*(1 - p01)*(1 - p13)*(1 - p3) + p0*p02*p1*p3*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p02*p13*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p1*p13*p2*p23*(1 - p01)*(1 - p02)*(1 - p3)*(1 - x) + p0*p1*p13*p3*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p0*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p0*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p1*p13*x*(1 - p0)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p2*p23*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p3) + p01*p02*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p1*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - x) + p01*p1*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p13*p2*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p01*p13*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p02*p1*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p23)*(1 - x) + p02*p1*p13*p23*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p3)*(1 - x) + p02*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p02*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p1*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23) + p1*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p13*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)
    
    

    P1100 = p0*p01*p02*p1*p13*p23*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p02*p2*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p23*p3*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p0*p01*p1*p2*p3*x*(1 - p02)*(1 - p13)*(1 - p23) + p0*p01*p1*p23*x*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p01*p13*p2*x*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p0*p01*p13*p23*p3*x*(1 - p02)*(1 - p1)*(1 - p2) + p0*p02*p1*p13*x*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p2*p23*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p02*p3*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p1*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - x) + p0*p1*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p13*p2*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p0*p13*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p1*p2*p23*x*(1 - p0)*(1 - p13)*(1 - p3) + p01*p02*p1*p3*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p02*p13*x*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p1*p13*p2*p23*(1 - p0)*(1 - p02)*(1 - p3)*(1 - x) + p01*p1*p13*p3*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p01*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p01*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p2*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p23*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - x) + p02*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - x) + p02*p13*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p1*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3) + p1*p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2) + p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)
    
    
    P0011 = p0*p01*p02*p2*x*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p02*p23*p3*x*(1 - p1)*(1 - p13)*(1 - p2) + p0*p01*p1*p2*p3*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p1*p23*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p13*p2*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p13*p23*p3*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p0*p02*p1*p13*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p2*p23*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p02*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p1*p2*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p13) + p0*p1*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p13*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3) + p0*p13*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p01*p02*p1*p2*p23*(1 - p0)*(1 - p13)*(1 - p3)*(1 - x) + p01*p02*p1*p3*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p13*p2*p23*p3*(1 - p0)*(1 - p1)*(1 - x) + p01*p02*p13*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p13*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p3) + p01*p1*p13*p3*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23) + p01*p2*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13) + p01*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p1*p2*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3) + p02*p1*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2) + p02*p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23) + p02*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3) + p1*p13*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p1*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x)
    
    P1010 = p0*p01*p02*p1*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p2*p23*(1 - p1)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p3*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p1*p13*x*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p2*p23*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p01*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p02*p1*p2*p3*x*(1 - p01)*(1 - p13)*(1 - p23) + p0*p02*p1*p23*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p02*p13*p2*x*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3) + p0*p02*p13*p23*p3*x*(1 - p01)*(1 - p1)*(1 - p2) + p0*p1*p13*p2*p3*(1 - p01)*(1 - p02)*(1 - p23)*(1 - x) + p0*p1*p13*p23*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p0*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p01*p02*p1*p13*p2*x*(1 - p0)*(1 - p23)*(1 - p3) + p01*p02*p2*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23) + p01*p02*p23*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p1*p2*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p23*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p01*p13*p2*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p01*p13*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p02*p1*p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p3)*(1 - x) + p02*p1*p13*p3*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - x) + p02*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - x) + p02*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p1*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3) + p1*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p13*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1) + p13*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)
    
    
    P0101 = p0*p01*p02*p1*x*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p02*p13*p3*x*(1 - p1)*(1 - p2)*(1 - p23) + p0*p01*p1*p13*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p2*p23*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p1*p2*p3*(1 - p01)*(1 - p13)*(1 - p23)*(1 - x) + p0*p02*p1*p23*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p13*p2*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p13*p23*p3*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p0*p1*p13*p2*p3*x*(1 - p01)*(1 - p02)*(1 - p23) + p0*p1*p13*p23*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3) + p0*p2*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p01*p02*p1*p13*p2*(1 - p0)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p1*p13*p23*p3*(1 - p0)*(1 - p2)*(1 - x) + p01*p02*p2*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p01*p02*p23*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p1*p2*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p1*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2) + p01*p13*p2*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23) + p01*p13*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p02*p1*p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p3) + p02*p1*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23) + p02*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13) + p02*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p1*p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p1*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p13*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x)
    
    P1001 = p0*p01*p02*p1*p2*p3*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p02*p1*p23*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p2*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p23*p3*(1 - p1)*(1 - p2)*(1 - x) + p0*p01*p1*p13*p23*x*(1 - p02)*(1 - p2)*(1 - p3) + p0*p01*p2*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p23*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p0*p02*p1*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p13*p2*p23*x*(1 - p01)*(1 - p1)*(1 - p3) + p0*p02*p13*p3*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23) + p0*p1*p13*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - x) + p0*p1*p13*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p2*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p1*p13*p3*x*(1 - p0)*(1 - p2)*(1 - p23) + p01*p02*p2*p23*p3*x*(1 - p0)*(1 - p1)*(1 - p13) + p01*p02*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p1*p2*p23*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p01*p1*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p13*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - x) + p01*p13*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p13*p2*(1 - p0)*(1 - p01)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p2)*(1 - x) + p02*p2*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p02*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p1*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p1*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2) + p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23) + p13*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)
    
    P0110 = p0*p01*p02*p1*p23*x*(1 - p13)*(1 - p2)*(1 - p3) + p0*p01*p02*p13*p2*x*(1 - p1)*(1 - p23)*(1 - p3) + p0*p01*p1*p13*p2*p3*(1 - p02)*(1 - p23)*(1 - x) + p0*p01*p1*p13*p23*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p2*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p23*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p0*p02*p1*p2*p23*p3*(1 - p01)*(1 - p13)*(1 - x) + p0*p02*p1*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p13*p2*p23*(1 - p01)*(1 - p1)*(1 - p3)*(1 - x) + p0*p02*p13*p3*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p0*p1*p13*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p02*p1*p13*p2*p23*(1 - p0)*(1 - p3)*(1 - x) + p01*p02*p1*p13*p3*(1 - p0)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p2*p23*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - x) + p01*p02*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p3) + p01*p1*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p13*p2*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1) + p01*p13*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p1*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p23)*(1 - p3) + p02*p1*p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p2) + p02*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23) + p02*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p1*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p1*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p13*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x)
    
    P1110 = p0*p01*p02*p1*p13*p3*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p02*p2*p23*p3*(1 - p1)*(1 - p13)*(1 - x) + p0*p01*p02*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p1*p2*p23*x*(1 - p02)*(1 - p13)*(1 - p3) + p0*p01*p1*p3*x*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p01*p13*x*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p1*p13*p2*x*(1 - p01)*(1 - p23)*(1 - p3) + p0*p02*p2*p3*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p02*p23*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p1*p2*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p23*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p0*p13*p2*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p0*p13*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p1*p2*p3*x*(1 - p0)*(1 - p13)*(1 - p23) + p01*p02*p1*p23*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p02*p13*p2*x*(1 - p0)*(1 - p1)*(1 - p23)*(1 - p3) + p01*p02*p13*p23*p3*x*(1 - p0)*(1 - p1)*(1 - p2) + p01*p1*p13*p2*p3*(1 - p0)*(1 - p02)*(1 - p23)*(1 - x) + p01*p1*p13*p23*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p01*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p02*p1*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - x) + p02*p1*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p3)*(1 - x) + p02*p13*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p1*p13*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02) + p1*p13*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)

    P0111 = p0*p01*p02*p1*p2*x*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p02*p13*p23*x*(1 - p1)*(1 - p2)*(1 - p3) + p0*p01*p1*p13*p2*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p1*p13*p23*p3*(1 - p02)*(1 - p2)*(1 - x) + p0*p01*p2*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p23*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p1*p2*p23*(1 - p01)*(1 - p13)*(1 - p3)*(1 - x) + p0*p02*p1*p3*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p13*p2*p23*p3*(1 - p01)*(1 - p1)*(1 - x) + p0*p02*p13*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p13*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p3) + p0*p1*p13*p3*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23) + p0*p2*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13) + p0*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p1*p13*(1 - p0)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p2*p23*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p01*p02*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p1*p2*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p13) + p01*p1*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p13*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p3) + p01*p13*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p02*p1*p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p23) + p02*p1*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p3) + p02*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p02*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2) + p1*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p1*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p13*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x)
    
    P1011 = p0*p01*p02*p1*p2*p23*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p02*p1*p3*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p02*p13*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p1*p13*p3*x*(1 - p02)*(1 - p2)*(1 - p23) + p0*p01*p2*p23*p3*x*(1 - p02)*(1 - p1)*(1 - p13) + p0*p01*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p1*p2*x*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p02*p1*p23*p3*x*(1 - p01)*(1 - p13)*(1 - p2) + p0*p02*p13*p2*p3*x*(1 - p01)*(1 - p1)*(1 - p23) + p0*p02*p13*p23*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3) + p0*p1*p13*p2*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p13*p23*p3*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p0*p2*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p1*p13*p23*x*(1 - p0)*(1 - p2)*(1 - p3) + p01*p02*p2*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p02*p23*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2) + p01*p1*p2*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p01*p1*p23*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p13*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p01*p13*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p02*p1*p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - x) + p02*p1*p13*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p2*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p02*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p1*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13) + p1*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3) + p13*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)

    P1101 = p0*p01*p02*p1*p13*p2*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p2*p3*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p02*p23*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p1*p2*x*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p1*p23*p3*x*(1 - p02)*(1 - p13)*(1 - p2) + p0*p01*p13*p2*p3*x*(1 - p02)*(1 - p1)*(1 - p23) + p0*p01*p13*p23*x*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p0*p02*p1*p13*p3*x*(1 - p01)*(1 - p2)*(1 - p23) + p0*p02*p2*p23*p3*x*(1 - p01)*(1 - p1)*(1 - p13) + p0*p02*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p1*p2*p23*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p0*p1*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p13*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p0*p13*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p1*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p13*p2*p23*x*(1 - p0)*(1 - p1)*(1 - p3) + p01*p02*p13*p3*x*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23) + p01*p1*p13*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - x) + p01*p1*p13*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p2*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p01*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p02*p1*p2*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - x) + p02*p1*p23*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p02*p13*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p02*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p1*p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23) + p1*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3) + p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)
    
    P1111 = p0*p01*p02*p1*p13*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p2*p23*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p02*p3*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p1*x*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p13*p2*p23*x*(1 - p02)*(1 - p1)*(1 - p3) + p0*p01*p13*p3*x*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p0*p02*p1*p13*p23*x*(1 - p01)*(1 - p2)*(1 - p3) + p0*p02*p2*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p02*p23*p3*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2) + p0*p1*p2*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p0*p1*p23*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p13*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p13*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p01*p02*p1*p2*x*(1 - p0)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p02*p1*p23*p3*x*(1 - p0)*(1 - p13)*(1 - p2) + p01*p02*p13*p2*p3*x*(1 - p0)*(1 - p1)*(1 - p23) + p01*p02*p13*p23*x*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p3) + p01*p1*p13*p2*(1 - p0)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p13*p23*p3*(1 - p0)*(1 - p02)*(1 - p2)*(1 - x) + p01*p2*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p01*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p02*p1*p2*p23*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p3)*(1 - x) + p02*p1*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p02*p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - x) + p02*p13*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p1*p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p3) + p1*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23) + p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13) + x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)


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

def equations_O7(vars, v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011):
    '''Symbolic equations up to O(p^7).

    Input:
        vars: the unknowns that we solve fore
        v0001: the average counts where the last detector fires
        v0010: the average counts where the second-to-last detector fires
        (simularly for other inputs)

    Output:
        System of 9 equations of the form P(x0,x1,x2,x3) = const that we will solve with least-squares method.
        
    '''
    p0, p1, p2, p3, x, p01, p02, p13, p23 = vars
    
    P0001 =   p0*p01*p02*p1*p13*p3*x*(1 - p2)*(1 - p23) + p0*p01*p02*p2*p23*p3*x*(1 - p1)*(1 - p13) + p0*p01*p02*x*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p1*p2*p23*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p1*p3*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p13*p2*p23*p3*(1 - p02)*(1 - p1)*(1 - x) + p0*p01*p13*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p13*p2*(1 - p01)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p13*p23*p3*(1 - p01)*(1 - p2)*(1 - x) + p0*p02*p2*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p02*p23*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p1*p2*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p1*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2) + p0*p13*p2*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23) + p0*p13*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p01*p02*p1*p2*p3*(1 - p0)*(1 - p13)*(1 - p23)*(1 - x) + p01*p02*p1*p23*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p13*p2*(1 - p0)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p13*p23*p3*(1 - p0)*(1 - p1)*(1 - p2)*(1 - x) + p01*p1*p13*p2*p3*x*(1 - p0)*(1 - p02)*(1 - p23) + p01*p1*p13*p23*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p3) + p01*p2*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p02*p1*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p13) + p02*p1*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p3) + p02*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23) + p1*p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - x) + p1*p13*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x)

    P0010 = p0*p01*p02*p1*p13*p2*x*(1 - p23)*(1 - p3) + p0*p01*p02*p2*p3*x*(1 - p1)*(1 - p13)*(1 - p23) + p0*p01*p02*p23*x*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p01*p1*p2*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p1*p23*p3*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p0*p01*p13*p2*p3*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p0*p01*p13*p23*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p1*p13*p2*p23*(1 - p01)*(1 - p3)*(1 - x) + p0*p02*p1*p13*p3*(1 - p01)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p2*p23*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - x) + p0*p02*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3) + p0*p1*p3*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p13*p2*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p1) + p0*p13*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p1*p2*p23*p3*(1 - p0)*(1 - p13)*(1 - x) + p01*p02*p1*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p13*p2*p23*(1 - p0)*(1 - p1)*(1 - p3)*(1 - x) + p01*p02*p13*p3*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p01*p1*p13*p2*p23*p3*x*(1 - p0)*(1 - p02) + p01*p1*p13*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p01*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p02*p1*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23) + p02*p1*p23*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3) + p02*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3) + p02*p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2) + p1*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - x) + p1*p13*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x)
    
    P0100 = p0*p01*p02*p1*p2*p23*x*(1 - p13)*(1 - p3) + p0*p01*p02*p1*p3*x*(1 - p13)*(1 - p2)*(1 - p23) + p0*p01*p02*p13*x*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p1*p13*p2*p23*(1 - p02)*(1 - p3)*(1 - x) + p0*p01*p1*p13*p3*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p2*p23*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p0*p01*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p2*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p1*p23*p3*(1 - p01)*(1 - p13)*(1 - p2)*(1 - x) + p0*p02*p13*p2*p3*(1 - p01)*(1 - p1)*(1 - p23)*(1 - x) + p0*p02*p13*p23*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p1*p13*p2*x*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3) + p0*p1*p13*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p2) + p0*p2*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p02*p1*p13*p2*p3*(1 - p0)*(1 - p23)*(1 - x) + p01*p02*p1*p13*p23*(1 - p0)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p2*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p23*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p01*p1*p2*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23) + p01*p1*p23*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p13*p2*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p01*p13*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2) + p02*p1*p13*p2*p23*p3*x*(1 - p0)*(1 - p01) + p02*p1*p13*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3) + p02*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p1*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - x) + p1*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p13*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x)
    
    P1000 = p0*p01*p02*p1*p2*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p1*p23*p3*(1 - p13)*(1 - p2)*(1 - x) + p0*p01*p02*p13*p2*p3*(1 - p1)*(1 - p23)*(1 - x) + p0*p01*p02*p13*p23*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p1*p13*p2*x*(1 - p02)*(1 - p23)*(1 - p3) + p0*p01*p1*p13*p23*p3*x*(1 - p02)*(1 - p2) + p0*p01*p2*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p01*p23*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p02*p1*p2*p23*x*(1 - p01)*(1 - p13)*(1 - p3) + p0*p02*p1*p3*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p02*p13*p2*p23*p3*x*(1 - p01)*(1 - p1) + p0*p02*p13*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p1*p13*p2*p23*(1 - p01)*(1 - p02)*(1 - p3)*(1 - x) + p0*p1*p13*p3*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p0*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p0*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p1*p13*x*(1 - p0)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p2*p23*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p3) + p01*p02*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p1*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - x) + p01*p1*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p13*p2*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p01*p13*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p02*p1*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p23)*(1 - x) + p02*p1*p13*p23*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p3)*(1 - x) + p02*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p02*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p1*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23) + p1*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p13*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)
    
    

    P1100 = p0*p01*p02*p1*p13*p2*p3*(1 - p23)*(1 - x) + p0*p01*p02*p1*p13*p23*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p02*p2*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p23*p3*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p0*p01*p1*p2*p3*x*(1 - p02)*(1 - p13)*(1 - p23) + p0*p01*p1*p23*x*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p01*p13*p2*x*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3) + p0*p01*p13*p23*p3*x*(1 - p02)*(1 - p1)*(1 - p2) + p0*p02*p1*p13*x*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p2*p23*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p02*p3*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p1*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - x) + p0*p1*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p13*p2*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3)*(1 - x) + p0*p13*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p1*p2*p23*x*(1 - p0)*(1 - p13)*(1 - p3) + p01*p02*p1*p3*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p02*p13*p2*p23*p3*x*(1 - p0)*(1 - p1) + p01*p02*p13*x*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p1*p13*p2*p23*(1 - p0)*(1 - p02)*(1 - p3)*(1 - x) + p01*p1*p13*p3*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - x) + p01*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - x) + p01*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p2*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p23*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - x) + p02*p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - x) + p02*p13*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p1*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3) + p1*p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2) + p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23) + p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)
    
    
    P0011 = p0*p01*p02*p1*p13*p23*x*(1 - p2)*(1 - p3) + p0*p01*p02*p2*x*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p02*p23*p3*x*(1 - p1)*(1 - p13)*(1 - p2) + p0*p01*p1*p2*p3*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p1*p23*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p13*p2*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p13*p23*p3*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p0*p02*p1*p13*p2*p23*p3*(1 - p01)*(1 - x) + p0*p02*p1*p13*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p2*p23*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p02*p3*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p1*p2*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p13) + p0*p1*x*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p13*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3) + p0*p13*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p01*p02*p1*p2*p23*(1 - p0)*(1 - p13)*(1 - p3)*(1 - x) + p01*p02*p1*p3*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p13*p2*p23*p3*(1 - p0)*(1 - p1)*(1 - x) + p01*p02*p13*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p13*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p3) + p01*p1*p13*p3*x*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23) + p01*p2*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13) + p01*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p1*p2*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3) + p02*p1*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2) + p02*p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23) + p02*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3) + p1*p13*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p1*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x)
    
    P1010 = p0*p01*p02*p1*p2*p23*p3*(1 - p13)*(1 - x) + p0*p01*p02*p1*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p2*p23*(1 - p1)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p3*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p1*p13*x*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p2*p23*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p01*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p02*p1*p2*p3*x*(1 - p01)*(1 - p13)*(1 - p23) + p0*p02*p1*p23*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p02*p13*p2*x*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3) + p0*p02*p13*p23*p3*x*(1 - p01)*(1 - p1)*(1 - p2) + p0*p1*p13*p2*p3*(1 - p01)*(1 - p02)*(1 - p23)*(1 - x) + p0*p1*p13*p23*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p0*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p01*p02*p1*p13*p2*x*(1 - p0)*(1 - p23)*(1 - p3) + p01*p02*p1*p13*p23*p3*x*(1 - p0)*(1 - p2) + p01*p02*p2*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23) + p01*p02*p23*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p1*p2*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p23*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p01*p13*p2*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p01*p13*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p02*p1*p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p3)*(1 - x) + p02*p1*p13*p3*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - x) + p02*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - x) + p02*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p1*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3) + p1*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p13*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1) + p13*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)
    
    
    P0101 = p0*p01*p02*p1*x*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p02*p13*p2*p23*x*(1 - p1)*(1 - p3) + p0*p01*p02*p13*p3*x*(1 - p1)*(1 - p2)*(1 - p23) + p0*p01*p1*p13*p2*p23*p3*(1 - p02)*(1 - x) + p0*p01*p1*p13*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p2*p23*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p1*p2*p3*(1 - p01)*(1 - p13)*(1 - p23)*(1 - x) + p0*p02*p1*p23*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p13*p2*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p13*p23*p3*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p0*p1*p13*p2*p3*x*(1 - p01)*(1 - p02)*(1 - p23) + p0*p1*p13*p23*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3) + p0*p2*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p01*p02*p1*p13*p2*(1 - p0)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p1*p13*p23*p3*(1 - p0)*(1 - p2)*(1 - x) + p01*p02*p2*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p01*p02*p23*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p1*p2*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p1*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2) + p01*p13*p2*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23) + p01*p13*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p02*p1*p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p3) + p02*p1*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23) + p02*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13) + p02*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p1*p2*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p1*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p13*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x)
    
    P1001 = p0*p01*p02*p1*p2*p3*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p02*p1*p23*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p2*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p13*p23*p3*(1 - p1)*(1 - p2)*(1 - x) + p0*p01*p1*p13*p2*p3*x*(1 - p02)*(1 - p23) + p0*p01*p1*p13*p23*x*(1 - p02)*(1 - p2)*(1 - p3) + p0*p01*p2*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p23*p3*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2) + p0*p02*p1*p2*p23*p3*x*(1 - p01)*(1 - p13) + p0*p02*p1*x*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p13*p2*p23*x*(1 - p01)*(1 - p1)*(1 - p3) + p0*p02*p13*p3*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23) + p0*p1*p13*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - x) + p0*p1*p13*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p2*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p1*p13*p2*p23*x*(1 - p0)*(1 - p3) + p01*p02*p1*p13*p3*x*(1 - p0)*(1 - p2)*(1 - p23) + p01*p02*p2*p23*p3*x*(1 - p0)*(1 - p1)*(1 - p13) + p01*p02*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p1*p2*p23*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p01*p1*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p13*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - x) + p01*p13*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p13*p2*(1 - p0)*(1 - p01)*(1 - p23)*(1 - p3)*(1 - x) + p02*p1*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p2)*(1 - x) + p02*p2*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p02*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p1*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p1*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2) + p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23) + p13*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)
    
    P0110 = p0*p01*p02*p1*p2*p3*x*(1 - p13)*(1 - p23) + p0*p01*p02*p1*p23*x*(1 - p13)*(1 - p2)*(1 - p3) + p0*p01*p02*p13*p2*x*(1 - p1)*(1 - p23)*(1 - p3) + p0*p01*p02*p13*p23*p3*x*(1 - p1)*(1 - p2) + p0*p01*p1*p13*p2*p3*(1 - p02)*(1 - p23)*(1 - x) + p0*p01*p1*p13*p23*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p2*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p23*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p0*p02*p1*p2*p23*p3*(1 - p01)*(1 - p13)*(1 - x) + p0*p02*p1*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p02*p13*p2*p23*(1 - p01)*(1 - p1)*(1 - p3)*(1 - x) + p0*p02*p13*p3*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p0*p1*p13*p2*p23*p3*x*(1 - p01)*(1 - p02) + p0*p1*p13*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p0*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p02*p1*p13*p2*p23*(1 - p0)*(1 - p3)*(1 - x) + p01*p02*p1*p13*p3*(1 - p0)*(1 - p2)*(1 - p23)*(1 - x) + p01*p02*p2*p23*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - x) + p01*p02*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p3) + p01*p1*p3*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p01*p13*p2*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p1) + p01*p13*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p02*p1*p13*p2*x*(1 - p0)*(1 - p01)*(1 - p23)*(1 - p3) + p02*p1*p13*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p2) + p02*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23) + p02*p23*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p1*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p1*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p13*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p13*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x)
    
    P1110 = p0*p01*p02*p1*p13*p2*p23*(1 - p3)*(1 - x) + p0*p01*p02*p1*p13*p3*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p02*p2*p23*p3*(1 - p1)*(1 - p13)*(1 - x) + p0*p01*p02*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p1*p2*p23*x*(1 - p02)*(1 - p13)*(1 - p3) + p0*p01*p1*p3*x*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23) + p0*p01*p13*p2*p23*p3*x*(1 - p02)*(1 - p1) + p0*p01*p13*x*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p1*p13*p2*x*(1 - p01)*(1 - p23)*(1 - p3) + p0*p02*p1*p13*p23*p3*x*(1 - p01)*(1 - p2) + p0*p02*p2*p3*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23) + p0*p02*p23*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3) + p0*p1*p2*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p23*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - x) + p0*p13*p2*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - x) + p0*p13*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p1*p2*p3*x*(1 - p0)*(1 - p13)*(1 - p23) + p01*p02*p1*p23*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p3) + p01*p02*p13*p2*x*(1 - p0)*(1 - p1)*(1 - p23)*(1 - p3) + p01*p02*p13*p23*p3*x*(1 - p0)*(1 - p1)*(1 - p2) + p01*p1*p13*p2*p3*(1 - p0)*(1 - p02)*(1 - p23)*(1 - x) + p01*p1*p13*p23*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p3)*(1 - x) + p01*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3)*(1 - x) + p01*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - x) + p02*p1*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - x) + p02*p1*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p13*p2*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p3)*(1 - x) + p02*p13*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - x) + p1*p13*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02) + p1*p13*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3) + p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3) + p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)

    P0111 = p0*p01*p02*p1*p2*x*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p02*p1*p23*p3*x*(1 - p13)*(1 - p2) + p0*p01*p02*p13*p2*p3*x*(1 - p1)*(1 - p23) + p0*p01*p02*p13*p23*x*(1 - p1)*(1 - p2)*(1 - p3) + p0*p01*p1*p13*p2*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p1*p13*p23*p3*(1 - p02)*(1 - p2)*(1 - x) + p0*p01*p2*p3*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p23*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p02*p1*p2*p23*(1 - p01)*(1 - p13)*(1 - p3)*(1 - x) + p0*p02*p1*p3*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p02*p13*p2*p23*p3*(1 - p01)*(1 - p1)*(1 - x) + p0*p02*p13*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p13*p2*p23*x*(1 - p01)*(1 - p02)*(1 - p3) + p0*p1*p13*p3*x*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23) + p0*p2*p23*p3*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13) + p0*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p1*p13*p2*p23*p3*(1 - p0)*(1 - x) + p01*p02*p1*p13*(1 - p0)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p2*p23*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p01*p02*p3*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p01*p1*p2*p23*p3*x*(1 - p0)*(1 - p02)*(1 - p13) + p01*p1*x*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p13*p2*p23*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p3) + p01*p13*p3*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p02*p1*p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p23) + p02*p1*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p3) + p02*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p02*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2) + p1*p2*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p1*p23*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p13*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x)
    
    P1011 = p0*p01*p02*p1*p2*p23*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p02*p1*p3*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p02*p13*p2*p23*p3*(1 - p1)*(1 - x) + p0*p01*p02*p13*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p1*p13*p2*p23*x*(1 - p02)*(1 - p3) + p0*p01*p1*p13*p3*x*(1 - p02)*(1 - p2)*(1 - p23) + p0*p01*p2*p23*p3*x*(1 - p02)*(1 - p1)*(1 - p13) + p0*p01*x*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p02*p1*p2*x*(1 - p01)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p02*p1*p23*p3*x*(1 - p01)*(1 - p13)*(1 - p2) + p0*p02*p13*p2*p3*x*(1 - p01)*(1 - p1)*(1 - p23) + p0*p02*p13*p23*x*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p3) + p0*p1*p13*p2*(1 - p01)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p0*p1*p13*p23*p3*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p0*p2*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p23*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p02*p1*p13*p2*p3*x*(1 - p0)*(1 - p23) + p01*p02*p1*p13*p23*x*(1 - p0)*(1 - p2)*(1 - p3) + p01*p02*p2*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p02*p23*p3*x*(1 - p0)*(1 - p1)*(1 - p13)*(1 - p2) + p01*p1*p2*p3*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p01*p1*p23*(1 - p0)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p01*p13*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p01*p13*p23*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p02*p1*p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - x) + p02*p1*p13*(1 - p0)*(1 - p01)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p02*p2*p23*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p02*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p1*p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13) + p1*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p3) + p13*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)

    P1101 = p0*p01*p02*p1*p13*p2*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p1*p13*p23*p3*(1 - p2)*(1 - x) + p0*p01*p02*p2*p3*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p0*p01*p02*p23*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p01*p1*p2*x*(1 - p02)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p01*p1*p23*p3*x*(1 - p02)*(1 - p13)*(1 - p2) + p0*p01*p13*p2*p3*x*(1 - p02)*(1 - p1)*(1 - p23) + p0*p01*p13*p23*x*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p3) + p0*p02*p1*p13*p2*p23*x*(1 - p01)*(1 - p3) + p0*p02*p1*p13*p3*x*(1 - p01)*(1 - p2)*(1 - p23) + p0*p02*p2*p23*p3*x*(1 - p01)*(1 - p1)*(1 - p13) + p0*p02*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p1*p2*p23*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p3)*(1 - x) + p0*p1*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p13*p2*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p0*p13*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p02*p1*p2*p23*p3*x*(1 - p0)*(1 - p13) + p01*p02*p1*x*(1 - p0)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p01*p02*p13*p2*p23*x*(1 - p0)*(1 - p1)*(1 - p3) + p01*p02*p13*p3*x*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p23) + p01*p1*p13*p2*p23*p3*(1 - p0)*(1 - p02)*(1 - x) + p01*p1*p13*(1 - p0)*(1 - p02)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p01*p2*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p01*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p02*p1*p2*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p23)*(1 - x) + p02*p1*p23*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p02*p13*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p02*p13*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p1*p13*p2*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p23) + p1*p13*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p3) + p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)
    
    P1111 = p0*p01*p02*p1*p13*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p0*p01*p02*p2*p23*(1 - p1)*(1 - p13)*(1 - p3)*(1 - x) + p0*p01*p02*p3*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p0*p01*p1*p2*p23*p3*x*(1 - p02)*(1 - p13) + p0*p01*p1*x*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3) + p0*p01*p13*p2*p23*x*(1 - p02)*(1 - p1)*(1 - p3) + p0*p01*p13*p3*x*(1 - p02)*(1 - p1)*(1 - p2)*(1 - p23) + p0*p02*p1*p13*p2*p3*x*(1 - p01)*(1 - p23) + p0*p02*p1*p13*p23*x*(1 - p01)*(1 - p2)*(1 - p3) + p0*p02*p2*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - p3) + p0*p02*p23*p3*x*(1 - p01)*(1 - p1)*(1 - p13)*(1 - p2) + p0*p1*p2*p3*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p23)*(1 - x) + p0*p1*p23*(1 - p01)*(1 - p02)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p0*p13*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p23)*(1 - p3)*(1 - x) + p0*p13*p23*p3*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p01*p02*p1*p2*x*(1 - p0)*(1 - p13)*(1 - p23)*(1 - p3) + p01*p02*p1*p23*p3*x*(1 - p0)*(1 - p13)*(1 - p2) + p01*p02*p13*p2*p3*x*(1 - p0)*(1 - p1)*(1 - p23) + p01*p02*p13*p23*x*(1 - p0)*(1 - p1)*(1 - p2)*(1 - p3) + p01*p1*p13*p2*(1 - p0)*(1 - p02)*(1 - p23)*(1 - p3)*(1 - x) + p01*p1*p13*p23*p3*(1 - p0)*(1 - p02)*(1 - p2)*(1 - x) + p01*p2*p3*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p23)*(1 - x) + p01*p23*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p3)*(1 - x) + p02*p1*p2*p23*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p3)*(1 - x) + p02*p1*p3*(1 - p0)*(1 - p01)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - x) + p02*p13*p2*p23*p3*(1 - p0)*(1 - p01)*(1 - p1)*(1 - x) + p02*p13*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - p23)*(1 - p3)*(1 - x) + p1*p13*p2*p23*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p3) + p1*p13*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - p23) + p2*p23*p3*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13) + x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p13)*(1 - p2)*(1 - p23)*(1 - p3)


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
    return np.sum(np.square(equations_O7(vars, *args)))

def apply_4_pnt_method(v0001: float, v0010: float, v0100: float, v1000: float, v1111: float, v1100, 
                       v1010: float, v0101: float, v0011: float, min_bound: float, max_bound: float, method):
      '''Solve the system of equations with minimize or least_squares method. The least_squares method is more accurate, the minimize seems unstable in some cases.
      
      Input:
            v0001: average # of times that last detector fires
            v0010: average # of times that second-to-last detector fires
            (similarly for the other counts)
            min_nound: minimum bound where solutions of probabilities can lie in
            max_nound: maximum bound where solutions of probabilities can lie in
      Output:
            solution_dict: dictionary of solutions with names "p0", "p1", "p2", "p3", "x", "p01", "p02", "p13", "p23"

      '''

      v_values = [v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011]   #Observed average values
      
      if method=="minimize":

            initial_guess =  [0.04]*9
            
            bounds = [(min_bound, max_bound)] * 9
            result = minimize(objective, initial_guess, args=tuple(v_values), 
                              method='Nelder-Mead', bounds=bounds)
            variable_names = ["p0", "p1", "p2", "p3", "x", "p01", "p02", "p13", "p23"]
            
            solution_dict = dict(zip(variable_names, result.x))
            
            # print("cost:",result.fun)
            
      elif method=="least_squares":
           

            # Bounds for variables
            bounds = ([min_bound] * 9, [max_bound] * 9)

            # Initial guess
            initial_guess = np.ones(9)*0.01

            # Example observed values
            v_values = [v0001, v0010, v0100, v1000, v1111,v1100, v1010, v0101, v0011]  

            # Solve the system
            result = least_squares(equations_O7, initial_guess, args=tuple(v_values), bounds=bounds,
                                   jac='3-point',loss='soft_l1',verbose=0,gtol=1e-15,ftol=1e-15,xtol=1e-15)

            
            # Output results
            
            solution_dict = dict(zip(["p0", "p1", "p2", "p3", "x", "p01", "p02", "p13", "p23"], result.x))
            
            # print("optimality:",result.optimality)
      else:
           raise Exception("Method not available")

             

      return solution_dict


def solve_system_of_equations(min_bound,max_bound,method,vijkl):
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

        temp = vijkl[key]


        v0001 = temp["v0001"]
        v0010 = temp["v0010"]
        v0100 = temp["v0100"]
        v1000 = temp["v1000"]
        v1111 = temp["v1111"]
        v1100 = temp["v1100"]
        v1010 = temp["v1010"]
        v0101 = temp["v0101"]
        v0011 = temp["v0011"]

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

    
    return all_dicts




def fill_initial_prob_values(all_dicts,bulk_edges,time_edges):
    '''
    Calculate the initial probability values for edges and hyperedges. The edges are updated later using the 4-point event probabilities.
      
    Input: 
        all_dicts: all the dictionaries we obtain as solutions. Each dictionary has another dictionary containing the solutions of each 4-point system we solve
        bulk_edges: set of names of the bulk space edges
        time_edges: set of names of the time edges

    Output:
        pij_bulk: dictionary with keys of the form ("Di","Dj") that contains the probability values of the space-bulk error events
        pij_time: dictionary with keys of the form ("Di","Dj") that contains the probability values of the time-like error events
        pij_bd:   dictionary with keys of the form ("Di") that contains the probability values of the boundary error events
        p4_cnts:  dictionary with keys of the form ("Di","Dj","Dj","Dl") that contains the probability values of the four-point error events

    '''
    
    pij_bulk = {}
    pij_time = {}
    pij_bd   = {}
    p4_cnts  = {}
    

    for key in all_dicts.keys():

        temp = all_dicts[key]

        for key2 in temp.keys():

            if key2[0]=="D": #this is a bd edge
                pij_bd[key2]=temp[key2]
            elif key2 in bulk_edges:
                pij_bulk[key2]=temp[key2]
            elif key2 in time_edges:
                pij_time[key2]=temp[key2]
            else: #4 pnt event
                p4_cnts[key2]=temp[key2]

    return pij_bulk,pij_time,pij_bd,p4_cnts


def update_edges_after_4_pnt_estimation(pij_bulk: dict, pij_time: dict, pij_bd: dict, p4_cnts: dict, 
                                        num_rounds: int, vi_mean, distance: int):
    '''Update the edges by subtracting the contribution from the four-point events.
    
       Input:
            pij_bulk: dictionary for the space bulk edge probabilities
            pij_time: dictionary for the time edge probabilities
            pij_bd: dictionary for the boundary edge probabilities
            p4_cnts: dictionary for the four-point hyperedge probabilities
            num_rounds: # of QEC rounds
            vi_mean: average counts of detectors [array # of QEC rds x # of detectors]
            distance: distance of the repetition code
        Output:
            pij_bulk: updated dictionary of space bulk edge probabiliites
            pij_time: updated dictionary of time edge probabiliites
            pij_bd: updated dictionary of boundary edge probabiliites
            p4_cnts: updated dictionary of four-point hyperedge probabiliites
    '''

    num_rounds +=1
    num_ancilla = distance-1

    #Bulk edges (same rd):
    for rd1 in range(1,num_rounds-1):
        rd2=rd1
        for anc1 in range(num_ancilla-1):
            
            anc2  = anc1+1
            indx1 = anc1 + num_ancilla*rd1
            indx2 = anc2 + num_ancilla*rd2
            
            name                = (f"D{indx1}",f"D{indx2}")
            name_of_4_pnt_event = (f"D{indx1-num_ancilla}",f"D{indx2-num_ancilla}",f"D{indx1}",f"D{indx2}")
            
            pnew1          = p4_cnts[name_of_4_pnt_event]
            pij_bulk[name] = (pij_bulk[name]-pnew1)/(1-2*(pnew1))


    for rd1 in range(num_rounds-1):
        rd2 = rd1+1

        for anc1 in range(1,num_ancilla-1):
            anc2  = anc1 
            indx1 = anc1 + num_ancilla*rd1
            indx2 = anc2 + num_ancilla*rd2

            
            name  = (f"D{indx1}",f"D{indx2}")

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
            DENOM *= 1-2*pij_time[(f"D{indx2}",f"D{indx1}")]

        if rd<(num_rounds-1):
            rd2   = rd+1
            indx2 = anc+num_ancilla*rd2
            DENOM *= 1-2*pij_time[(f"D{indx1}",f"D{indx2}")]
        
        #Get bulk edge:
        anc2  = anc+1
        indx2 = anc2+num_ancilla*rd
        DENOM *= 1-2*pij_bulk[(f"D{indx1}",f"D{indx2}")]

        #Get all relevant 4-pnt events
        for key,val in p4_cnts.items():

            if name in key:

                DENOM *=1-2*val
                

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
            DENOM *= 1-2*pij_time[(f"D{indx2}",f"D{indx1}")]

        if rd<(num_rounds-1):
            rd2   = rd+1
            indx2 = anc+num_ancilla*rd2
            DENOM *= 1-2*pij_time[(f"D{indx1}",f"D{indx2}")]
        
        #Get bulk edge:
        anc2  = anc-1
        indx2 = anc2+num_ancilla*rd
        DENOM *= 1-2*pij_bulk[(f"D{indx2}",f"D{indx1}")]

        #Get all relevant 4-pnt events
        for key in p4_cnts.keys():

            if name in key:

                DENOM *=1-2*p4_cnts[key]
                
        pij_bd[name] = 1/2+(v0-1/2)/DENOM


    return pij_bulk, pij_time, pij_bd, p4_cnts
