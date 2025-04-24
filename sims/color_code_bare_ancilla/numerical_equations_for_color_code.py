import numpy as np
from itertools import combinations
from sympy import symbols
from scipy.optimize import minimize,least_squares
from noise_est_funcs_for_color_code import *
from utilities_for_color_code import *


def get_3_pnt_events_Z_DEM(defects_matrix_Z, obj, Z_DEM: stim.DetectorErrorModel):
    '''Get the average number of 3-point coincidences of the 3-point
       events that exist in the Z-DEM of the color code.

       Input:
            defects_matrix_Z: xArray of the defects matrix for the Z-DEM (dims: # of shots x # of QEC rds+1 x # of ancilla)
            obj: the color code object
            Z_DEM: stim's detector error model for the Z-type detectors
       Output:
            p3_cnts: dictionary with keys of the form ("Di","Dj","Dk") and values the average # of 3-point coincidences
       '''
    num_rounds    = np.shape(defects_matrix_Z)[1]
    num_shots     = np.shape(defects_matrix_Z)[0]
    dets_Z,dets_X = get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds-1) #Need total # of ancilla qubits
    
    p3_cnts = {}
    for instruction in Z_DEM:
        if instruction.type=="error":

            targets = instruction.targets_copy()
            temp    = [f"D{target.val}" for target in targets if target.is_relative_detector_id()]

            if len(temp)==3:

                p3_cnts[tuple(temp)]=0

    for key, detectors in p3_cnts.items():

        pairs = [dets_Z[det] for det in key]

        (rd1, anc1), (rd2, anc2), (rd3, anc3) = pairs
   

        locs1 = np.nonzero(defects_matrix_Z.data[:,rd1,anc1])[0]
        locs2 = np.nonzero(defects_matrix_Z.data[:,rd2,anc2])[0]
        locs3 = np.nonzero(defects_matrix_Z.data[:,rd3,anc3])[0]

        common_locs = np.intersect1d(locs1, np.intersect1d(locs2, locs3), assume_unique=True)

        p3_cnts[key] = len(common_locs)/num_shots 

    return p3_cnts


def get_3_pnt_events_X_DEM(defects_matrix_X, obj, X_DEM: stim.DetectorErrorModel, num_rounds: int):
    '''Get the average number of 3-point coincidences of the 3-point
       events that exist in the X-DEM of the color code.

       Input:
            defects_matrix_X: xArray of the defects matrix for the X-DEM (dims: # of shots x # of QEC rds-2 x # of ancilla)
            obj: the color code object
            X_DEM: stim's detector error model for the X-type detectors
            num_rounds: total # of QEC rounds (int)
       Output:
            p3_cnts: dictionary with keys of the form ("Di","Dj","Dk") and values the average # of 3-point coincidences
       '''
    
    num_shots     = np.shape(defects_matrix_X)[0]
    dets_Z,dets_X = get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds) #Need total # of ancilla qubits
    
    p3_cnts = {}
    for instruction in X_DEM:
        if instruction.type=="error":

            targets = instruction.targets_copy()

            temp    = [f"D{target.val}" for target in targets if target.is_relative_detector_id()]
            if len(temp)==3:
                p3_cnts[tuple(temp)]=0
    
    for key, detectors in p3_cnts.items():

        pairs = [dets_X[det] for det in key]

        (rd1, anc1), (rd2, anc2), (rd3, anc3) = pairs
        

        locs1 = np.nonzero(defects_matrix_X.data[:,rd1,anc1])[0]
        locs2 = np.nonzero(defects_matrix_X.data[:,rd2,anc2])[0]
        locs3 = np.nonzero(defects_matrix_X.data[:,rd3,anc3])[0]

        common_locs = np.intersect1d(locs1, np.intersect1d(locs2, locs3), assume_unique=True)

        p3_cnts[key] = len(common_locs)/num_shots 

    return p3_cnts


def form_equation_terms(target_inds: list, order: int):
    '''Form terms for the configuration probability P(x0,x1,x2), where xj\in[0,1].
       We make the correspondence x0->0, x1->1, x2->2 for a given configuration.
       For example, if target_inds=[0,1], then we find terms for the equation P(1,1,0).
       The order controls which terms we will collect i.e, O(p), O(p^2) etc.

    Input:
        target_inds: the target indices for the configuration probability 
        order: the order in p for the events that are compatible with the detection counts
    Output:
        accepted_combs: list of tuples where the element of the tuples correspond to integers denoting
                        which events combine to give rise to a detection pattern
                        
    '''
    if target_inds==[]: #P(0,0,0) event: treat separately
        
        rest_inds      = [0,1,2]
        p0             = [0] ; p1  = [1] ; p2  = [2] 
        p01            = [0,1]; p02 = [0,2] ; p12 = [1,2]
        p012           = [0,1,2]
        all_elements   = [p0,p1,p2,p01,p02,p12,p012]
        all_combs      = list(combinations(all_elements,order))
        accepted_combs = []

        for comb in all_combs:
            #All should appear an even # of times.

            cnt_0,cnt_1,cnt_2 = 0,0,0

            for elem in comb:

                if 0 in elem:
                    cnt_0+=1
                if 1 in elem:
                    cnt_1+=1
                if 2 in elem:
                    cnt_2+=1
            
            list_of_cnts = [cnt_0,cnt_1,cnt_2]

            not_acceptable = False
            
            
            for p in rest_inds:
                if list_of_cnts[p]%2==1:
                    not_acceptable=True
            
            if not_acceptable==False:
                accepted_combs.append(comb)
        
        
        return accepted_combs



    for k in target_inds:
        if k>2 or k<0:
            raise Exception("Invalid indices. They should be in [0,1,2].")

    all_inds  = [0,1,2]
    rest_inds = list(set(all_inds) ^ set(target_inds) )


    p0             = [0] ; p1  = [1] ; p2  = [2] 
    p01            = [0,1]; p02 = [0,2] ; p12 = [1,2]
    p012           = [0,1,2]
    all_elements   = [p0,p1,p2,p01,p02,p12,p012]
    all_combs      = list(combinations(all_elements,order))
    accepted_combs = []

    for comb in all_combs:
        
        cnt_0,cnt_1,cnt_2 = 0,0,0

        for elem in comb:

            if 0 in elem:
                cnt_0+=1
            if 1 in elem:
                cnt_1+=1
            if 2 in elem:
                cnt_2+=1
        
        list_of_cnts = [cnt_0,cnt_1,cnt_2]

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



def form_particular_eq_for_one_truncation_order(target_inds: list, order: int):
    '''Form the full equation for the configuration probability P(x0,x1,x2) using the individual equation terms, and up to some maximum order.

       Input:
            target_inds: list that contains only indices from the elements [0,1,2] which correspond to x0,x1,x2 for the configuration probability P(x0,x1,x2)
            order: maximum truncation order that we keep in the equations (int)

       Output:
            all_eqns: symbolic equation for the configuration probability P(x0,x1,x2) where target_inds indicate which detectors fire, and for the given order
    '''
    all_terms   = form_equation_terms(target_inds, order)

    p0        = [0] ; p1  = [1] ; p2  = [2] 
    p01       = [0,1]; p02 = [0,2] ; p12=[1,2]
    p012      = [0,1,2]
    all_eqns  = 0

    if target_inds==[] and order==0:

        
        all_elements = [p0,p1,p2,p01,p02,p12,p012]
        this_eq      = 1

        for term in all_elements:

            total_elem = "p"

            for elem in term:

                total_elem+=str(elem)

            this_eq*=1-symbols(total_elem)

        all_eqns = this_eq



    else:

        for terms in all_terms:

            all_elements   = [p0,p1,p2,p01,p02,p12,p012]
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


def form_all_3_pnt_equations(max_truncation_order: int):
    '''Form all the equations for the P(x0,x1,x2) configuration probabilities.
    
    Input:
        max_truncation_order: maximum order to keep for the equations.
    Output:
        P: dictionary of symbolic equations where the names are all bit-combinations on 3 digits. '''

    #Eqns in a dictionary:
    P = {}

    #Target inds denote which location has the entry of "1"
    #e.g. for P(1,1,0) we have target_inds=[0,1]
    #empty is for P(0,0,0).

    all_target_inds = [[0],[1],[2], \
                       [0,1],[0,2],\
                       [1,2],\
                       [0,1,2],[]]
    
    all_names = [[1,0,0],[0,1,0],[0,0,1],
                 [1,1,0],[1,0,1],[0,1,1],
                 [1,1,1],
                 [0,0,0]]

    for name in all_names:
        P[tuple(name)]=0
    
    cnt=0
    for target_inds in all_target_inds:


        for order in range(0,max_truncation_order+1):

            eqns = form_particular_eq_for_one_truncation_order(target_inds,order)
            P[tuple(all_names[cnt])]+= eqns
            
        cnt+=1

    return P



def equations_O3(vars, v100,v010,v001,v110,v101,v011,v111):
    '''Symbolic equations up to O(p^3).

    Input:
        vars: the unknowns that we solve fore
        v100: the average counts where the first detector fires
        v010: the average counts where the second detector fires
        (simularly for other inputs)

    Output:
        System of 9 equations of the form P(x0,x1,x2) = const that we will solve with least-squares method.
        
    '''

    p0, p1, p2, x, p01, p02, p12 = vars
    
    P100 = p0*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p01*p02*x*(1 - p0)*(1 - p1)*(1 - p12)*(1 - p2) + p01*p1*(1 - p0)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p01*p12*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - x) + p02*p1*p12*(1 - p0)*(1 - p01)*(1 - p2)*(1 - x) + p02*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - x) + p1*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12) + p12*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)
    P010 = p0*p01*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p02*p12*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p0*p2*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12) + p01*p02*p2*(1 - p0)*(1 - p1)*(1 - p12)*(1 - x) + p01*p12*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2) + p02*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2) + p1*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p12*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x)
    P001 = p0*p01*p12*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p0*p02*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p1*x*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2) + p01*p02*p1*(1 - p0)*(1 - p12)*(1 - p2)*(1 - x) + p01*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p02*p12*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2) + p1*p12*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x)
    P110 = p0*p02*x*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2) + p0*p1*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p0*p12*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p01*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p02*p1*p2*(1 - p0)*(1 - p01)*(1 - p12)*(1 - x) + p02*p12*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p1*p12*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2) + p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)
    P101 = p0*p01*x*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p0*p1*p12*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p0*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p01*p1*p2*(1 - p0)*(1 - p02)*(1 - p12)*(1 - x) + p01*p12*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p02*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p1*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2) + p12*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)
    P011 = p0*p01*p2*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p0*p02*p1*(1 - p01)*(1 - p12)*(1 - p2)*(1 - x) + p0*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p01*p02*(1 - p0)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p01*p1*x*(1 - p0)*(1 - p02)*(1 - p12)*(1 - p2) + p02*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12) + p1*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - x) + p12*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x)
    P111 = p0*p01*p02*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p1*p2*(1 - p01)*(1 - p02)*(1 - p12)*(1 - x) + p0*p12*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p01*p1*p12*(1 - p0)*(1 - p02)*(1 - p2)*(1 - x) + p01*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p02*p1*(1 - p0)*(1 - p01)*(1 - p12)*(1 - p2)*(1 - x) + p02*p12*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - x) + x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)


    eq1 = v100 - (  P100 + P110 + P101 + P111)
    
    eq2 = v010 - (  P010 + P110 + P011 + P111)

    eq3 = v001 - (P001 + P011 + P101 + P111)

    eq4 = v110 - (P110 + P111)

    eq5 = v101 - (P101 + P111)

    eq6 = v011 - (P011 +P111)

    eq7 = v111 - P111
    
    return [eq1,eq2,eq3,eq4,eq5,eq6,eq7]


def equations_O4(vars, v100,v010,v001,v110,v101,v011,v111):
    '''Symbolic equations up to O(p^4).

    Input:
        vars: the unknowns that we solve fore
        v100: the average counts where the first detector fires
        v010: the average counts where the second detector fires
        (simularly for other inputs)

    Output:
        System of 9 equations of the form P(x0,x1,x2) = const that we will solve with least-squares method.
        
    '''

    p0, p1, p2, x, p01, p02, p12 = vars
    
    P100 = p0*p01*p02*p12*(1 - p1)*(1 - p2)*(1 - x) + p0*p01*p2*x*(1 - p02)*(1 - p1)*(1 - p12) + p0*p02*p1*x*(1 - p01)*(1 - p12)*(1 - p2) + p0*p1*p12*p2*(1 - p01)*(1 - p02)*(1 - x) + p0*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p01*p02*x*(1 - p0)*(1 - p1)*(1 - p12)*(1 - p2) + p01*p1*(1 - p0)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p01*p12*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - x) + p02*p1*p12*(1 - p0)*(1 - p01)*(1 - p2)*(1 - x) + p02*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - x) + p1*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12) + p12*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)
    P010 = p0*p01*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p02*p1*p2*(1 - p01)*(1 - p12)*(1 - x) + p0*p02*p12*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p0*p1*p12*x*(1 - p01)*(1 - p02)*(1 - p2) + p0*p2*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12) + p01*p02*p1*p12*(1 - p0)*(1 - p2)*(1 - x) + p01*p02*p2*(1 - p0)*(1 - p1)*(1 - p12)*(1 - x) + p01*p1*p2*x*(1 - p0)*(1 - p02)*(1 - p12) + p01*p12*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2) + p02*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2) + p1*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p12*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x)
    P001 = p0*p01*p1*p2*(1 - p02)*(1 - p12)*(1 - x) + p0*p01*p12*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p0*p02*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p1*x*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2) + p0*p12*p2*x*(1 - p01)*(1 - p02)*(1 - p1) + p01*p02*p1*(1 - p0)*(1 - p12)*(1 - p2)*(1 - x) + p01*p02*p12*p2*(1 - p0)*(1 - p1)*(1 - x) + p01*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p02*p1*p2*x*(1 - p0)*(1 - p01)*(1 - p12) + p02*p12*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2) + p1*p12*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x)
    P110 = p0*p01*p02*p2*(1 - p1)*(1 - p12)*(1 - x) + p0*p01*p12*x*(1 - p02)*(1 - p1)*(1 - p2) + p0*p02*x*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2) + p0*p1*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p0*p12*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p01*p02*p1*x*(1 - p0)*(1 - p12)*(1 - p2) + p01*p1*p12*p2*(1 - p0)*(1 - p02)*(1 - x) + p01*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p02*p1*p2*(1 - p0)*(1 - p01)*(1 - p12)*(1 - x) + p02*p12*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p1*p12*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2) + p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)
    P101 = p0*p01*p02*p1*(1 - p12)*(1 - p2)*(1 - x) + p0*p01*x*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p0*p02*p12*x*(1 - p01)*(1 - p1)*(1 - p2) + p0*p1*p12*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p0*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p01*p02*p2*x*(1 - p0)*(1 - p1)*(1 - p12) + p01*p1*p2*(1 - p0)*(1 - p02)*(1 - p12)*(1 - x) + p01*p12*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p02*p1*p12*p2*(1 - p0)*(1 - p01)*(1 - x) + p02*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p1*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2) + p12*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)
    P011 = p0*p01*p1*p12*(1 - p02)*(1 - p2)*(1 - x) + p0*p01*p2*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p0*p02*p1*(1 - p01)*(1 - p12)*(1 - p2)*(1 - x) + p0*p02*p12*p2*(1 - p01)*(1 - p1)*(1 - x) + p0*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p01*p02*(1 - p0)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p01*p1*x*(1 - p0)*(1 - p02)*(1 - p12)*(1 - p2) + p01*p12*p2*x*(1 - p0)*(1 - p02)*(1 - p1) + p02*p1*p12*x*(1 - p0)*(1 - p01)*(1 - p2) + p02*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12) + p1*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - x) + p12*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x)
    P111 = p0*p01*p02*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p01*p1*x*(1 - p02)*(1 - p12)*(1 - p2) + p0*p02*p2*x*(1 - p01)*(1 - p1)*(1 - p12) + p0*p1*p2*(1 - p01)*(1 - p02)*(1 - p12)*(1 - x) + p0*p12*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p01*p02*p12*x*(1 - p0)*(1 - p1)*(1 - p2) + p01*p1*p12*(1 - p0)*(1 - p02)*(1 - p2)*(1 - x) + p01*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p02*p1*(1 - p0)*(1 - p01)*(1 - p12)*(1 - p2)*(1 - x) + p02*p12*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - x) + p1*p12*p2*x*(1 - p0)*(1 - p01)*(1 - p02) + x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)


    eq1 = v100 - (  P100 + P110 + P101 + P111)
    
    eq2 = v010 - (  P010 + P110 + P011 + P111)

    eq3 = v001 - (P001 + P011 + P101 + P111)

    eq4 = v110 - (P110 + P111)

    eq5 = v101 - (P101 + P111)

    eq6 = v011 - (P011 +P111)

    eq7 = v111 - P111
    
    return [eq1,eq2,eq3,eq4,eq5,eq6,eq7]


def equations_O5(vars, v100,v010,v001,v110,v101,v011,v111):
    '''Symbolic equations up to O(p^5).

    Input:
        vars: the unknowns that we solve fore
        v100: the average counts where the first detector fires
        v010: the average counts where the second detector fires
        (simularly for other inputs)

    Output:
        System of 9 equations of the form P(x0,x1,x2) = const that we will solve with least-squares method.
        
    '''

    p0, p1, p2, x, p01, p02, p12 = vars
    
    P100 = p0*p01*p02*p1*p2*(1 - p12)*(1 - x) + p0*p01*p02*p12*(1 - p1)*(1 - p2)*(1 - x) + p0*p01*p1*p12*x*(1 - p02)*(1 - p2) + p0*p01*p2*x*(1 - p02)*(1 - p1)*(1 - p12) + p0*p02*p1*x*(1 - p01)*(1 - p12)*(1 - p2) + p0*p02*p12*p2*x*(1 - p01)*(1 - p1) + p0*p1*p12*p2*(1 - p01)*(1 - p02)*(1 - x) + p0*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p01*p02*x*(1 - p0)*(1 - p1)*(1 - p12)*(1 - p2) + p01*p1*(1 - p0)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p01*p12*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - x) + p02*p1*p12*(1 - p0)*(1 - p01)*(1 - p2)*(1 - x) + p02*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - x) + p1*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12) + p12*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)
    P010 = p0*p01*p02*p1*x*(1 - p12)*(1 - p2) + p0*p01*p1*p12*p2*(1 - p02)*(1 - x) + p0*p01*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p02*p1*p2*(1 - p01)*(1 - p12)*(1 - x) + p0*p02*p12*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p0*p1*p12*x*(1 - p01)*(1 - p02)*(1 - p2) + p0*p2*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12) + p01*p02*p1*p12*(1 - p0)*(1 - p2)*(1 - x) + p01*p02*p2*(1 - p0)*(1 - p1)*(1 - p12)*(1 - x) + p01*p1*p2*x*(1 - p0)*(1 - p02)*(1 - p12) + p01*p12*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2) + p02*p1*p12*p2*x*(1 - p0)*(1 - p01) + p02*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2) + p1*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p12*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x)
    P001 = p0*p01*p02*p2*x*(1 - p1)*(1 - p12) + p0*p01*p1*p2*(1 - p02)*(1 - p12)*(1 - x) + p0*p01*p12*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p0*p02*p1*p12*p2*(1 - p01)*(1 - x) + p0*p02*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p1*x*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2) + p0*p12*p2*x*(1 - p01)*(1 - p02)*(1 - p1) + p01*p02*p1*(1 - p0)*(1 - p12)*(1 - p2)*(1 - x) + p01*p02*p12*p2*(1 - p0)*(1 - p1)*(1 - x) + p01*p1*p12*p2*x*(1 - p0)*(1 - p02) + p01*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p02*p1*p2*x*(1 - p0)*(1 - p01)*(1 - p12) + p02*p12*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2) + p1*p12*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x)
    P110 = p0*p01*p02*p1*p12*(1 - p2)*(1 - x) + p0*p01*p02*p2*(1 - p1)*(1 - p12)*(1 - x) + p0*p01*p1*p2*x*(1 - p02)*(1 - p12) + p0*p01*p12*x*(1 - p02)*(1 - p1)*(1 - p2) + p0*p02*x*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2) + p0*p1*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p0*p12*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p01*p02*p1*x*(1 - p0)*(1 - p12)*(1 - p2) + p01*p02*p12*p2*x*(1 - p0)*(1 - p1) + p01*p1*p12*p2*(1 - p0)*(1 - p02)*(1 - x) + p01*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p02*p1*p2*(1 - p0)*(1 - p01)*(1 - p12)*(1 - x) + p02*p12*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p1*p12*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2) + p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)
    P101 = p0*p01*p02*p1*(1 - p12)*(1 - p2)*(1 - x) + p0*p01*p02*p12*p2*(1 - p1)*(1 - x) + p0*p01*x*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p0*p02*p1*p2*x*(1 - p01)*(1 - p12) + p0*p02*p12*x*(1 - p01)*(1 - p1)*(1 - p2) + p0*p1*p12*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p0*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p01*p02*p1*p12*x*(1 - p0)*(1 - p2) + p01*p02*p2*x*(1 - p0)*(1 - p1)*(1 - p12) + p01*p1*p2*(1 - p0)*(1 - p02)*(1 - p12)*(1 - x) + p01*p12*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p02*p1*p12*p2*(1 - p0)*(1 - p01)*(1 - x) + p02*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p1*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2) + p12*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)
    P011 = p0*p01*p02*p12*x*(1 - p1)*(1 - p2) + p0*p01*p1*p12*(1 - p02)*(1 - p2)*(1 - x) + p0*p01*p2*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p0*p02*p1*(1 - p01)*(1 - p12)*(1 - p2)*(1 - x) + p0*p02*p12*p2*(1 - p01)*(1 - p1)*(1 - x) + p0*p1*p12*p2*x*(1 - p01)*(1 - p02) + p0*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p01*p02*p1*p12*p2*(1 - p0)*(1 - x) + p01*p02*(1 - p0)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p01*p1*x*(1 - p0)*(1 - p02)*(1 - p12)*(1 - p2) + p01*p12*p2*x*(1 - p0)*(1 - p02)*(1 - p1) + p02*p1*p12*x*(1 - p0)*(1 - p01)*(1 - p2) + p02*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12) + p1*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - x) + p12*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x)
    P111 = p0*p01*p02*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p01*p1*x*(1 - p02)*(1 - p12)*(1 - p2) + p0*p01*p12*p2*x*(1 - p02)*(1 - p1) + p0*p02*p1*p12*x*(1 - p01)*(1 - p2) + p0*p02*p2*x*(1 - p01)*(1 - p1)*(1 - p12) + p0*p1*p2*(1 - p01)*(1 - p02)*(1 - p12)*(1 - x) + p0*p12*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p01*p02*p1*p2*x*(1 - p0)*(1 - p12) + p01*p02*p12*x*(1 - p0)*(1 - p1)*(1 - p2) + p01*p1*p12*(1 - p0)*(1 - p02)*(1 - p2)*(1 - x) + p01*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p02*p1*(1 - p0)*(1 - p01)*(1 - p12)*(1 - p2)*(1 - x) + p02*p12*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - x) + p1*p12*p2*x*(1 - p0)*(1 - p01)*(1 - p02) + x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)


    eq1 = v100 - (  P100 + P110 + P101 + P111)
    
    eq2 = v010 - (  P010 + P110 + P011 + P111)

    eq3 = v001 - (P001 + P011 + P101 + P111)

    eq4 = v110 - (P110 + P111)

    eq5 = v101 - (P101 + P111)

    eq6 = v011 - (P011 +P111)

    eq7 = v111 - P111
    
    return [eq1,eq2,eq3,eq4,eq5,eq6,eq7]


def equations_O6(vars, v100,v010,v001,v110,v101,v011,v111):
    '''Symbolic equations up to O(p^6).

    Input:
        vars: the unknowns that we solve fore
        v100: the average counts where the first detector fires
        v010: the average counts where the second detector fires
        (simularly for other inputs)

    Output:
        System of 9 equations of the form P(x0,x1,x2) = const that we will solve with least-squares method.
        
    '''

    p0, p1, p2, x, p01, p02, p12 = vars
    
    P100 = p0*p01*p02*p1*p2*(1 - p12)*(1 - x) + p0*p01*p02*p12*(1 - p1)*(1 - p2)*(1 - x) + p0*p01*p1*p12*x*(1 - p02)*(1 - p2) + p0*p01*p2*x*(1 - p02)*(1 - p1)*(1 - p12) + p0*p02*p1*x*(1 - p01)*(1 - p12)*(1 - p2) + p0*p02*p12*p2*x*(1 - p01)*(1 - p1) + p0*p1*p12*p2*(1 - p01)*(1 - p02)*(1 - x) + p0*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p01*p02*p1*p12*p2*x*(1 - p0) + p01*p02*x*(1 - p0)*(1 - p1)*(1 - p12)*(1 - p2) + p01*p1*(1 - p0)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p01*p12*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - x) + p02*p1*p12*(1 - p0)*(1 - p01)*(1 - p2)*(1 - x) + p02*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - x) + p1*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12) + p12*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)
    P010 = p0*p01*p02*p1*x*(1 - p12)*(1 - p2) + p0*p01*p02*p12*p2*x*(1 - p1) + p0*p01*p1*p12*p2*(1 - p02)*(1 - x) + p0*p01*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p02*p1*p2*(1 - p01)*(1 - p12)*(1 - x) + p0*p02*p12*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p0*p1*p12*x*(1 - p01)*(1 - p02)*(1 - p2) + p0*p2*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12) + p01*p02*p1*p12*(1 - p0)*(1 - p2)*(1 - x) + p01*p02*p2*(1 - p0)*(1 - p1)*(1 - p12)*(1 - x) + p01*p1*p2*x*(1 - p0)*(1 - p02)*(1 - p12) + p01*p12*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2) + p02*p1*p12*p2*x*(1 - p0)*(1 - p01) + p02*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2) + p1*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p12*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x)
    P001 = p0*p01*p02*p1*p12*x*(1 - p2) + p0*p01*p02*p2*x*(1 - p1)*(1 - p12) + p0*p01*p1*p2*(1 - p02)*(1 - p12)*(1 - x) + p0*p01*p12*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p0*p02*p1*p12*p2*(1 - p01)*(1 - x) + p0*p02*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p1*x*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2) + p0*p12*p2*x*(1 - p01)*(1 - p02)*(1 - p1) + p01*p02*p1*(1 - p0)*(1 - p12)*(1 - p2)*(1 - x) + p01*p02*p12*p2*(1 - p0)*(1 - p1)*(1 - x) + p01*p1*p12*p2*x*(1 - p0)*(1 - p02) + p01*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p02*p1*p2*x*(1 - p0)*(1 - p01)*(1 - p12) + p02*p12*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2) + p1*p12*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x)
    P110 = p0*p01*p02*p1*p12*(1 - p2)*(1 - x) + p0*p01*p02*p2*(1 - p1)*(1 - p12)*(1 - x) + p0*p01*p1*p2*x*(1 - p02)*(1 - p12) + p0*p01*p12*x*(1 - p02)*(1 - p1)*(1 - p2) + p0*p02*p1*p12*p2*x*(1 - p01) + p0*p02*x*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2) + p0*p1*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p0*p12*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p01*p02*p1*x*(1 - p0)*(1 - p12)*(1 - p2) + p01*p02*p12*p2*x*(1 - p0)*(1 - p1) + p01*p1*p12*p2*(1 - p0)*(1 - p02)*(1 - x) + p01*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p02*p1*p2*(1 - p0)*(1 - p01)*(1 - p12)*(1 - x) + p02*p12*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p1*p12*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2) + p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)
    P101 = p0*p01*p02*p1*(1 - p12)*(1 - p2)*(1 - x) + p0*p01*p02*p12*p2*(1 - p1)*(1 - x) + p0*p01*p1*p12*p2*x*(1 - p02) + p0*p01*x*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p0*p02*p1*p2*x*(1 - p01)*(1 - p12) + p0*p02*p12*x*(1 - p01)*(1 - p1)*(1 - p2) + p0*p1*p12*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p0*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p01*p02*p1*p12*x*(1 - p0)*(1 - p2) + p01*p02*p2*x*(1 - p0)*(1 - p1)*(1 - p12) + p01*p1*p2*(1 - p0)*(1 - p02)*(1 - p12)*(1 - x) + p01*p12*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p02*p1*p12*p2*(1 - p0)*(1 - p01)*(1 - x) + p02*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p1*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2) + p12*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)
    P011 = p0*p01*p02*p1*p2*x*(1 - p12) + p0*p01*p02*p12*x*(1 - p1)*(1 - p2) + p0*p01*p1*p12*(1 - p02)*(1 - p2)*(1 - x) + p0*p01*p2*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p0*p02*p1*(1 - p01)*(1 - p12)*(1 - p2)*(1 - x) + p0*p02*p12*p2*(1 - p01)*(1 - p1)*(1 - x) + p0*p1*p12*p2*x*(1 - p01)*(1 - p02) + p0*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p01*p02*p1*p12*p2*(1 - p0)*(1 - x) + p01*p02*(1 - p0)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p01*p1*x*(1 - p0)*(1 - p02)*(1 - p12)*(1 - p2) + p01*p12*p2*x*(1 - p0)*(1 - p02)*(1 - p1) + p02*p1*p12*x*(1 - p0)*(1 - p01)*(1 - p2) + p02*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12) + p1*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - x) + p12*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x)
    P111 = p0*p01*p02*p1*p12*p2*(1 - x) + p0*p01*p02*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p01*p1*x*(1 - p02)*(1 - p12)*(1 - p2) + p0*p01*p12*p2*x*(1 - p02)*(1 - p1) + p0*p02*p1*p12*x*(1 - p01)*(1 - p2) + p0*p02*p2*x*(1 - p01)*(1 - p1)*(1 - p12) + p0*p1*p2*(1 - p01)*(1 - p02)*(1 - p12)*(1 - x) + p0*p12*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p01*p02*p1*p2*x*(1 - p0)*(1 - p12) + p01*p02*p12*x*(1 - p0)*(1 - p1)*(1 - p2) + p01*p1*p12*(1 - p0)*(1 - p02)*(1 - p2)*(1 - x) + p01*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p02*p1*(1 - p0)*(1 - p01)*(1 - p12)*(1 - p2)*(1 - x) + p02*p12*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - x) + p1*p12*p2*x*(1 - p0)*(1 - p01)*(1 - p02) + x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)


    eq1 = v100 - (  P100 + P110 + P101 + P111)
    
    eq2 = v010 - (  P010 + P110 + P011 + P111)

    eq3 = v001 - (P001 + P011 + P101 + P111)

    eq4 = v110 - (P110 + P111)

    eq5 = v101 - (P101 + P111)

    eq6 = v011 - (P011 +P111)

    eq7 = v111 - P111
    
    return [eq1,eq2,eq3,eq4,eq5,eq6,eq7]



def equations_O7(vars, v100,v010,v001,v110,v101,v011,v111):
    '''Symbolic equations up to O(p^7).

    Input:
        vars: the unknowns that we solve fore
        v100: the average counts where the first detector fires
        v010: the average counts where the second detector fires
        (simularly for other inputs)

    Output:
        System of 9 equations of the form P(x0,x1,x2) = const that we will solve with least-squares method.
        
    '''

    p0, p1, p2, x, p01, p02, p12 = vars
    
    P100 = p0*p01*p02*p1*p2*(1 - p12)*(1 - x) + p0*p01*p02*p12*(1 - p1)*(1 - p2)*(1 - x) + p0*p01*p1*p12*x*(1 - p02)*(1 - p2) + p0*p01*p2*x*(1 - p02)*(1 - p1)*(1 - p12) + p0*p02*p1*x*(1 - p01)*(1 - p12)*(1 - p2) + p0*p02*p12*p2*x*(1 - p01)*(1 - p1) + p0*p1*p12*p2*(1 - p01)*(1 - p02)*(1 - x) + p0*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p01*p02*p1*p12*p2*x*(1 - p0) + p01*p02*x*(1 - p0)*(1 - p1)*(1 - p12)*(1 - p2) + p01*p1*(1 - p0)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p01*p12*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - x) + p02*p1*p12*(1 - p0)*(1 - p01)*(1 - p2)*(1 - x) + p02*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - x) + p1*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12) + p12*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)
    P010 = p0*p01*p02*p1*x*(1 - p12)*(1 - p2) + p0*p01*p02*p12*p2*x*(1 - p1) + p0*p01*p1*p12*p2*(1 - p02)*(1 - x) + p0*p01*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p02*p1*p2*(1 - p01)*(1 - p12)*(1 - x) + p0*p02*p12*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p0*p1*p12*x*(1 - p01)*(1 - p02)*(1 - p2) + p0*p2*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12) + p01*p02*p1*p12*(1 - p0)*(1 - p2)*(1 - x) + p01*p02*p2*(1 - p0)*(1 - p1)*(1 - p12)*(1 - x) + p01*p1*p2*x*(1 - p0)*(1 - p02)*(1 - p12) + p01*p12*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2) + p02*p1*p12*p2*x*(1 - p0)*(1 - p01) + p02*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2) + p1*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p12*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x)
    P001 = p0*p01*p02*p1*p12*x*(1 - p2) + p0*p01*p02*p2*x*(1 - p1)*(1 - p12) + p0*p01*p1*p2*(1 - p02)*(1 - p12)*(1 - x) + p0*p01*p12*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p0*p02*p1*p12*p2*(1 - p01)*(1 - x) + p0*p02*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p1*x*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2) + p0*p12*p2*x*(1 - p01)*(1 - p02)*(1 - p1) + p01*p02*p1*(1 - p0)*(1 - p12)*(1 - p2)*(1 - x) + p01*p02*p12*p2*(1 - p0)*(1 - p1)*(1 - x) + p01*p1*p12*p2*x*(1 - p0)*(1 - p02) + p01*x*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p02*p1*p2*x*(1 - p0)*(1 - p01)*(1 - p12) + p02*p12*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2) + p1*p12*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x)
    P110 = p0*p01*p02*p1*p12*(1 - p2)*(1 - x) + p0*p01*p02*p2*(1 - p1)*(1 - p12)*(1 - x) + p0*p01*p1*p2*x*(1 - p02)*(1 - p12) + p0*p01*p12*x*(1 - p02)*(1 - p1)*(1 - p2) + p0*p02*p1*p12*p2*x*(1 - p01) + p0*p02*x*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2) + p0*p1*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2)*(1 - x) + p0*p12*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - x) + p01*p02*p1*x*(1 - p0)*(1 - p12)*(1 - p2) + p01*p02*p12*p2*x*(1 - p0)*(1 - p1) + p01*p1*p12*p2*(1 - p0)*(1 - p02)*(1 - x) + p01*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p02*p1*p2*(1 - p0)*(1 - p01)*(1 - p12)*(1 - x) + p02*p12*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p2)*(1 - x) + p1*p12*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p2) + p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)
    P101 = p0*p01*p02*p1*(1 - p12)*(1 - p2)*(1 - x) + p0*p01*p02*p12*p2*(1 - p1)*(1 - x) + p0*p01*p1*p12*p2*x*(1 - p02) + p0*p01*x*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p0*p02*p1*p2*x*(1 - p01)*(1 - p12) + p0*p02*p12*x*(1 - p01)*(1 - p1)*(1 - p2) + p0*p1*p12*(1 - p01)*(1 - p02)*(1 - p2)*(1 - x) + p0*p2*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p01*p02*p1*p12*x*(1 - p0)*(1 - p2) + p01*p02*p2*x*(1 - p0)*(1 - p1)*(1 - p12) + p01*p1*p2*(1 - p0)*(1 - p02)*(1 - p12)*(1 - x) + p01*p12*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p02*p1*p12*p2*(1 - p0)*(1 - p01)*(1 - x) + p02*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p1*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - p2) + p12*p2*x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)
    P011 = p0*p01*p02*p1*p2*x*(1 - p12) + p0*p01*p02*p12*x*(1 - p1)*(1 - p2) + p0*p01*p1*p12*(1 - p02)*(1 - p2)*(1 - x) + p0*p01*p2*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p0*p02*p1*(1 - p01)*(1 - p12)*(1 - p2)*(1 - x) + p0*p02*p12*p2*(1 - p01)*(1 - p1)*(1 - x) + p0*p1*p12*p2*x*(1 - p01)*(1 - p02) + p0*x*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2) + p01*p02*p1*p12*p2*(1 - p0)*(1 - x) + p01*p02*(1 - p0)*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p01*p1*x*(1 - p0)*(1 - p02)*(1 - p12)*(1 - p2) + p01*p12*p2*x*(1 - p0)*(1 - p02)*(1 - p1) + p02*p1*p12*x*(1 - p0)*(1 - p01)*(1 - p2) + p02*p2*x*(1 - p0)*(1 - p01)*(1 - p1)*(1 - p12) + p1*p2*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p12)*(1 - x) + p12*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x)
    P111 = p0*p01*p02*p1*p12*p2*(1 - x) + p0*p01*p02*(1 - p1)*(1 - p12)*(1 - p2)*(1 - x) + p0*p01*p1*x*(1 - p02)*(1 - p12)*(1 - p2) + p0*p01*p12*p2*x*(1 - p02)*(1 - p1) + p0*p02*p1*p12*x*(1 - p01)*(1 - p2) + p0*p02*p2*x*(1 - p01)*(1 - p1)*(1 - p12) + p0*p1*p2*(1 - p01)*(1 - p02)*(1 - p12)*(1 - x) + p0*p12*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p2)*(1 - x) + p01*p02*p1*p2*x*(1 - p0)*(1 - p12) + p01*p02*p12*x*(1 - p0)*(1 - p1)*(1 - p2) + p01*p1*p12*(1 - p0)*(1 - p02)*(1 - p2)*(1 - x) + p01*p2*(1 - p0)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - x) + p02*p1*(1 - p0)*(1 - p01)*(1 - p12)*(1 - p2)*(1 - x) + p02*p12*p2*(1 - p0)*(1 - p01)*(1 - p1)*(1 - x) + p1*p12*p2*x*(1 - p0)*(1 - p01)*(1 - p02) + x*(1 - p0)*(1 - p01)*(1 - p02)*(1 - p1)*(1 - p12)*(1 - p2)


    eq1 = v100 - (  P100 + P110 + P101 + P111)
    
    eq2 = v010 - (  P010 + P110 + P011 + P111)

    eq3 = v001 - (P001 + P011 + P101 + P111)

    eq4 = v110 - (P110 + P111)

    eq5 = v101 - (P101 + P111)

    eq6 = v011 - (P011 +P111)

    eq7 = v111 - P111
    
    return [eq1,eq2,eq3,eq4,eq5,eq6,eq7]


def objective(vars, *args):
    return np.sum(np.square(equations_O5(vars, *args)))

def apply_3_pnt_method(v100,v010,v001,v110,v101,v011,v111,min_bound,max_bound,method):
      
      v_values = [v100,v010,v001,v110,v101,v011,v111]   #Observed average values
      
      if method=="minimize":

            initial_guess =  [0.01]*len(v_values)
            
            bounds = [(min_bound, max_bound)] * len(v_values)
            result = minimize(objective, initial_guess, args=tuple(v_values), 
                              method='Nelder-Mead', bounds=bounds)
            variable_names = ["p0", "p1", "p2", "x", "p01", "p02", "p12"]
            
            solution_dict = dict(zip(variable_names, result.x))
            
            
      elif method=="least_squares":
           
            # Example observed values
            v_values =  [v100,v010,v001,v110,v101,v011,v111]

            # Bounds for variables
            bounds = ([min_bound] * len(v_values), [max_bound] * len(v_values))

            # Initial guess
            initial_guess = np.ones(len(v_values))*0.05
            

            result = least_squares(equations_O7, initial_guess, args=tuple(v_values), bounds=bounds,
                                   jac='3-point',loss='soft_l1',verbose=0,gtol=1e-15,ftol=1e-15,xtol=1e-15)
            
            # Output results
            
            solution_dict = dict(zip(["p0", "p1", "p2", "x", "p01", "p02", "p12"], result.x))
            
      else:
           raise Exception("Method not available")

             

      return solution_dict


def get_vijk(p3_cnts: dict, num_rounds: int, vi_mean, vivj_mean, obj, detector_type):
    '''
    Prepare the dictionaries for the observed number of counts of single, two-point and 3-point events,
    that will be fed into the numerical equations to solve.

    Input:
        p3_cnts: dictionary with keys the events of the form ("Di","Dj","Dk") and values the average # of times where all 3 detectors fire together.
        num_rounds: # of QEC rounds (int)
        vi_mean: # of times a detector fires/num_shots for the Z- or X-DEM (np array of dims # of rounds x # of detectors per round)
        vivj_mean: average # of two-point coincidences for the Z- or X-DEM (np array of dimds # of rounds x # of roudns x # of detectors per rounds x # of detectors per round)
        obj: the color code object
        detector_type: "X" or "Z" for the X-DEM or Z-DEM respectively

    Output:
        events: dictionary whose keys are all the 3 point events we want to estimate. Within this outer-most dictionary we have another dictionary which
        holds all <vi>, <vivj> and <vivjvk> values for the respective 3-point event we want to solve the equations for.

    '''
    
    Z_dets,X_dets = get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds)

    if detector_type   == "Z":
        dets = Z_dets 
    elif detector_type == "X":
        dets = X_dets 
    else:
        raise Exception("Detector type can only be X or Z.")

    events = {}

    for key, p3_value in p3_cnts.items():
        try:
            # Collect (rd, anc) pairs
            det_inds = [dets[det] for det in key]

            
            (rd1, anc1), (rd2, anc2), (rd3, anc3) = det_inds

            temp_dict = {
                "v100": vi_mean[rd1, anc1],
                "v010": vi_mean[rd2, anc2],
                "v001": vi_mean[rd3, anc3],
                "v110": vivj_mean[rd1, rd2, anc1, anc2],
                "v101": vivj_mean[rd1, rd3, anc1, anc3],
                "v011": vivj_mean[rd2, rd3, anc2, anc3],
                "v111": p3_value  
            }

            
            events[key] = temp_dict

        except KeyError:
            # If a detector is missing, skip
            continue    
    


    return events



def solve_for_probs(min_bound: float, max_bound: float, method, vijk: dict):
    ''' 
    Solve the system of equations for a given three-point region that encloses a three-point event.
    The equations can be solved with minimize or least-squares method.
    Least-squares is stable and should be prefered.

    Input:
        min_bound: minimum acceptable value for a probability solution (float)
        max_bound: maxmimum acceptable value for a probability solution (float)
        method: "least_squares" or "minimize"
        vijk: dictionary of dictionaries that holds the average counts for P(x0,x1,x2)
    Output:
        all_dicts: dictionary of numerical solutions
        '''
        
    all_dicts = {}

    for key, values in vijk.items():  # Avoids redundant lookups
        solution_dict = apply_3_pnt_method(
            values["v100"], values["v010"], values["v001"],
            values["v110"], values["v101"], values["v011"],
            values["v111"], min_bound, max_bound, method
        )

        
        solution_dict = {
            key[0]: solution_dict.pop("p0"),  # "D0"
            key[1]: solution_dict.pop("p1"),  # "D1"
            key[2]: solution_dict.pop("p2"),  # "D2"
            (key[0], key[1]): solution_dict.pop("p01"),  # D0-D1
            (key[0], key[2]): solution_dict.pop("p02"),  # D0-D2
            (key[1], key[2]): solution_dict.pop("p12"),  # D1-D2
            key: solution_dict.pop("x")  # D0-D1-D2
        }

        all_dicts[key] = solution_dict  # Store the result

        
    return all_dicts




