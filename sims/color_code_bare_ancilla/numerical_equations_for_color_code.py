import numpy as np
import math
from itertools import combinations
import itertools
from sympy import symbols
from scipy.optimize import minimize,least_squares
from noise_est_funcs_for_color_code import *



# def get_3_pnt_events_only_Z_new(defects_matrix_Z,obj,Z_DEM):

#     num_rounds    = np.shape(defects_matrix_Z)[1]
#     num_ancilla   = len(obj.qubit_groups['anc'])
#     num_shots     = np.shape(defects_matrix_Z)[0]
#     dets_Z,dets_X = get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds-1) #Need total # of ancilla qubits
    
#     error_3_events={}
#     for instruction in Z_DEM:
#         if instruction.type=="error":

#             targets = instruction.targets_copy()
#             temp=[]
#             for target in targets:
#                 if target.is_relative_detector_id():
#                     ind = target.val
#                     temp.append("D"+str(ind))

#             if len(temp)==3:
#                 error_3_events[tuple(temp)]=0

#     p3_cnts = {}
#     for key in error_3_events.keys():

#         pairs = []
#         for det in key:

#             pairs.append(dets_Z[det])
        
#         rd1,anc1 = pairs[0]
#         rd2,anc2 = pairs[1]
#         rd3,anc3 = pairs[2]

#         locs1 = np.nonzero(defects_matrix_Z.data[:,rd1,anc1])[0]
#         locs2 = np.nonzero(defects_matrix_Z.data[:,rd2,anc2])[0]
#         locs3 = np.nonzero(defects_matrix_Z.data[:,rd3,anc3])[0]


#         locs1 = set(locs1)
#         locs2 = set(locs2)
#         locs3 = set(locs3)
        

#         locs  = locs1 & locs2 & locs3  #all should be nnz for the same shot

#         p3_cnts[key] = len(locs)/num_shots 

#     return p3_cnts


def get_3_pnt_events_only_Z_new(defects_matrix_Z,obj,Z_DEM):

    num_rounds    = np.shape(defects_matrix_Z)[1]
    
    num_shots     = np.shape(defects_matrix_Z)[0]
    dets_Z,dets_X = get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds-1) #Need total # of ancilla qubits
    
    error_3_events={}
    for instruction in Z_DEM:
        if instruction.type=="error":

            targets = instruction.targets_copy()
            temp=[]
            for target in targets:
                if target.is_relative_detector_id():
                    ind = target.val
                    temp.append("D"+str(ind))

            if len(temp)==3:
                error_3_events[tuple(temp)]=0

    p3_cnts = {}

    for key, detectors in error_3_events.items():

        
        pairs = [dets_Z[det] for det in key]

        (rd1, anc1), (rd2, anc2), (rd3, anc3) = pairs
   

        locs1 = np.nonzero(defects_matrix_Z.data[:,rd1,anc1])[0]
        locs2 = np.nonzero(defects_matrix_Z.data[:,rd2,anc2])[0]
        locs3 = np.nonzero(defects_matrix_Z.data[:,rd3,anc3])[0]

        common_locs = np.intersect1d(locs1, np.intersect1d(locs2, locs3), assume_unique=True)

        p3_cnts[key] = len(common_locs)/num_shots 

    return p3_cnts



def get_3_pnt_events_only_X_new(defects_matrix_X,obj,X_DEM,num_rounds):

    
    num_shots     = np.shape(defects_matrix_X)[0]
    dets_Z,dets_X = get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds) #Need total # of ancilla qubits
    
    error_3_events={}
    for instruction in X_DEM:
        if instruction.type=="error":

            targets = instruction.targets_copy()
            temp=[]
            for target in targets:
                if target.is_relative_detector_id():
                    ind = target.val
                    temp.append("D"+str(ind))

            if len(temp)==3:
                error_3_events[tuple(temp)]=0

    p3_cnts = {}
    for key, detectors in error_3_events.items():

        pairs = [dets_X[det] for det in key]

        (rd1, anc1), (rd2, anc2), (rd3, anc3) = pairs
        

        locs1 = np.nonzero(defects_matrix_X.data[:,rd1,anc1])[0]
        locs2 = np.nonzero(defects_matrix_X.data[:,rd2,anc2])[0]
        locs3 = np.nonzero(defects_matrix_X.data[:,rd3,anc3])[0]

        common_locs = np.intersect1d(locs1, np.intersect1d(locs2, locs3), assume_unique=True)

        p3_cnts[key] = len(common_locs)/num_shots 

    return p3_cnts





#This if for up to 3-pnt events
def form_equation_terms(target_inds, order):
    '''target_inds: some list that contains 0,1,2 elements.
    These correspond to the x0,x1,x2 for P(x0,x1,x2).
    order: the order of p events we want to calculate.
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




def form_particular_eq_for_one_truncation_order(target_inds,order):

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


def form_all_3_pnt_equations(max_truncation_order):
    '''Form the equations of the form P(x1,x2,x3) which includes 3-pnt events.'''


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



#Equations up to O(p^3)
def equations_O3(vars, v100,v010,v001,v110,v101,v011,v111):
    '''These equations were formed given the form_all_3_pnt_equations function, 
    with max trunctation order 3.
    x is the 3-pnt event.'''

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

#Equations up to O(p^4)
def equations_O4(vars, v100,v010,v001,v110,v101,v011,v111):
    '''These equations were formed given the form_all_3_pnt_equations function, 
    with max trunctation order 3.
    x is the 3-pnt event.'''

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

#Equations up to O(p^5)
def equations_O5(vars, v100,v010,v001,v110,v101,v011,v111):
    '''These equations were formed given the form_all_3_pnt_equations function, 
    with max trunctation order 3.
    x is the 3-pnt event.'''

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


#Equations up to O(p^6)
def equations_O6(vars, v100,v010,v001,v110,v101,v011,v111):
    '''These equations were formed given the form_all_3_pnt_equations function, 
    with max trunctation order 3.
    x is the 3-pnt event.'''

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


#Equations up to O(p^7)
def equations_O7(vars, v100,v010,v001,v110,v101,v011,v111):
    '''These equations were formed given the form_all_3_pnt_equations function, 
    with max trunctation order 3.
    x is the 3-pnt event.'''

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
            
            # print("cost:",result.fun)
            
      elif method=="least_squares":
           
            # Example observed values
            v_values =  [v100,v010,v001,v110,v101,v011,v111]

            # Bounds for variables
            bounds = ([min_bound] * len(v_values), [max_bound] * len(v_values))

            # Initial guess
            initial_guess = np.ones(len(v_values))*0.05
            

            # Solve the system
            #objective or equations?
            # result = least_squares(equations_O4, initial_guess, args=tuple(v_values), bounds=bounds,
            #                        jac='3-point',loss='soft_l1',verbose=0,max_nfev=25,gtol=1e-12,ftol=1e-12)

            result = least_squares(equations_O7, initial_guess, args=tuple(v_values), bounds=bounds,
                                   jac='3-point',loss='soft_l1',verbose=0,gtol=1e-15,ftol=1e-15,xtol=1e-15)
            
            # Output results
            
            solution_dict = dict(zip(["p0", "p1", "p2", "x", "p01", "p02", "p12"], result.x))
            
            # print("optimality:",result.optimality)
      else:
           raise Exception("Method not available")

             

      return solution_dict




#I think this is also correct for the X-type defects.
# def get_vijk_new(p3_cnts,num_rounds,vi_mean,vivj_mean,obj,detector_type):
#     '''
#     p3_cnts: The 3pnt coincidences obtained from the Z or X defect matrices. Dictionary with keys of the form ("Di","Dj","Dk")
#     num_rounds: # of QEC rounds (excluding stabilizer reconstruction i.e., last measurement of data qubits)
#     vi_mean: The average # of single-point events obtained from Z or X defect matrix. array # of QEC rds+1 x # of X or Z detectors
#     vivj_mean: The average # of 2-point events obtained from Z or X defect matrix. array # of QEC rds+1 x # QEC rds+1 x # X/Z detectors x # X/Z detectors
#     obj: the color code object
#     '''
    
#     Z_dets,X_dets = get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds)

#     if detector_type   == "Z":
#         dets = Z_dets 
#     elif detector_type == "X":
#         dets = X_dets 
#     else:
#         raise Exception("Detector type can only be X or Z.")

#     events = {}
    
#     for key in p3_cnts.keys():

#         temp_dict              = {}
#         det_inds               = []
#         flag_not_detector_type = False
        
#         for det in key:

#             #find rd,anc pair
#             if det in dets.keys():
#                 rd,anc = dets[det]
#             else:
#                 #Need to go to next iter
#                 flag_not_detector_type = True
#                 break
                
#             det_inds.append((rd,anc))

#         if flag_not_detector_type==True:
#             continue

#         inds1 = det_inds[0]
#         inds2 = det_inds[1]
#         inds3 = det_inds[2]
        
#         vi = vi_mean[inds1[0],inds1[1]] #v100
#         vj = vi_mean[inds2[0],inds2[1]] #v010
#         vk = vi_mean[inds3[0],inds3[1]] #v001
        
#         vivj = vivj_mean[inds1[0],inds2[0],inds1[1],inds2[1]] #v110
#         vivk = vivj_mean[inds1[0],inds3[0],inds1[1],inds3[1]] #v101
#         vjvk = vivj_mean[inds2[0],inds3[0],inds2[1],inds3[1]] #v011

#         vivjvk = p3_cnts[key] #v111

#         temp_dict["v100"]=vi
#         temp_dict["v010"]=vj
#         temp_dict["v001"]=vk

#         temp_dict["v110"]=vivj
#         temp_dict["v101"]=vivk
#         temp_dict["v011"]=vjvk
        
#         temp_dict["v111"]=vivjvk

#         events[key] = temp_dict


#     return events


def get_vijk_new(p3_cnts,num_rounds,vi_mean,vivj_mean,obj,detector_type):
    '''
    p3_cnts: The 3pnt coincidences obtained from the Z or X defect matrices. Dictionary with keys of the form ("Di","Dj","Dk")
    num_rounds: # of QEC rounds (excluding stabilizer reconstruction i.e., last measurement of data qubits)
    vi_mean: The average # of single-point events obtained from Z or X defect matrix. array # of QEC rds+1 x # of X or Z detectors
    vivj_mean: The average # of 2-point events obtained from Z or X defect matrix. array # of QEC rds+1 x # QEC rds+1 x # X/Z detectors x # X/Z detectors
    obj: the color code object
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

            # Unpack detector indices
            (rd1, anc1), (rd2, anc2), (rd3, anc3) = det_inds

            # Fetch values efficiently
            temp_dict = {
                "v100": vi_mean[rd1, anc1],
                "v010": vi_mean[rd2, anc2],
                "v001": vi_mean[rd3, anc3],
                "v110": vivj_mean[rd1, rd2, anc1, anc2],
                "v101": vivj_mean[rd1, rd3, anc1, anc3],
                "v011": vivj_mean[rd2, rd3, anc2, anc3],
                "v111": p3_value  
            }

            # Store result
            events[key] = temp_dict

        except KeyError:
            # If a detector is missing, just skip this iteration
            continue    
    


    return events




# def solve_for_probs(min_bound,max_bound,method,vijk):
#     ''' 
#     Solve the system of equations per round. For interemediate rounds, we will over-estimate 
#     bulk/time-edges. For example, if we have 2 rounds, d=3, then we have the dets:
#     D0 D1 D2
    

#     min_bound/max_bound: bounds of probabilities for the solver
#         method         : "least_squares" or "minimize" (both seem to perform pretty well)
#         vijk           : a dictionary containing the # of times we see each pattern "100", "010", ..., "111", divided by the # of shots.
#         '''
    
#     all_dicts={}
#     for key in vijk.keys(): #e.g. if key = ("D0","D1","D2")

        
#         v100 = vijk[key]["v100"]
#         v010 = vijk[key]["v010"]
#         v001 = vijk[key]["v001"]
        
#         v110 = vijk[key]["v110"]
#         v101 = vijk[key]["v101"]
#         v011 = vijk[key]["v011"]

#         v111 = vijk[key]["v111"]


#         solution_dict=apply_3_pnt_method(v100,v010,v001,v110,v101,v011,v111,min_bound,max_bound,method)

#         #rename the dicts of solution_dict:
        
#         solution_dict[key[0]] = solution_dict.pop("p0") #"D0"
#         solution_dict[key[1]] = solution_dict.pop("p1") #"D1"
#         solution_dict[key[2]] = solution_dict.pop("p2") #"D2"
        
#         solution_dict[(key[0],key[1])] = solution_dict.pop("p01") #D0-D1
#         solution_dict[(key[0],key[2])] = solution_dict.pop("p02") #D0-D2
#         solution_dict[(key[1],key[2])] = solution_dict.pop("p12") #D1-D2
  
#         solution_dict[key]             = solution_dict.pop("x") #D0-D1-D2
                            

#         all_dicts[key]=solution_dict

        
#     return all_dicts


def solve_for_probs(min_bound,max_bound,method,vijk):
    ''' 
    Solve the system of equations per round. For interemediate rounds, we will over-estimate 
    bulk/time-edges. For example, if we have 2 rounds, d=3, then we have the dets:
    D0 D1 D2
    

    min_bound/max_bound: bounds of probabilities for the solver
        method         : "least_squares" or "minimize" (both seem to perform pretty well)
        vijk           : a dictionary containing the # of times we see each pattern "100", "010", ..., "111", divided by the # of shots.
        '''
        
    all_dicts = {}

    for key, values in vijk.items():  # Avoids redundant lookups
        solution_dict = apply_3_pnt_method(
            values["v100"], values["v010"], values["v001"],
            values["v110"], values["v101"], values["v011"],
            values["v111"], min_bound, max_bound, method
        )

        # Rename dictionary keys efficiently
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




#TODO: Update the "bd" edges. I think all of them can be boundaries
def edge_dicts(num_ancilla,num_rounds):

    bulk_edges = {}
    time_edges = {}
    bd_edges   = {}

    #Time edges
    for rd1 in range(num_rounds):
        rd2=rd1+1
        for anc1 in range(num_ancilla):

            anc2  = anc1
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
    
    #"Boundary" edges: each node can be a boundary?

    for rd1 in range(num_rounds+1):

        for anc1 in range(num_ancilla):

            indx1 = anc1+num_ancilla*rd1 
            name = ("D"+str(indx1))
            bd_edges[name] = 0

    return bulk_edges,time_edges,bd_edges


def all_dicts_into_pij(all_dicts,bulk_edges,time_edges,bd_edges):
    '''Transform into pij_bulk, pij_time, pij_bd, p4_cnts'''
    
    pij_bulk = {}
    pij_time = {}
    pij_bd   = {}
    p3_cnts  = {}

    for key in all_dicts.keys():

        for key2 in all_dicts[key].keys():

            if key2[0]=="D": #this is a bd edge
                pij_bd[key2]=all_dicts[key][key2]
            elif key2 in bulk_edges:
                pij_bulk[key2]=all_dicts[key][key2]
            elif key2 in time_edges:
                pij_time[key2]=all_dicts[key][key2]
            else: #4 pnt event
                p3_cnts[key2]=all_dicts[key][key2]

    return pij_bulk,pij_time,pij_bd,p3_cnts

#TODO: Complete this
#TODO: I think this is unused?
def update_edges_after_3_pnt_estimation(pij_bulk,pij_time,pij_bd,p3_cnts,num_ancilla,num_rounds,vi_mean,distance):

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


