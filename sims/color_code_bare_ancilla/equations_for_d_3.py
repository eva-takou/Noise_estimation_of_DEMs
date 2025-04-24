import numpy as np
from utilities_for_color_code import *
from itertools import combinations
from sympy import *
from scipy import *
from scipy.optimize import *
from utilities.general_utils import DEM_to_dictionary
import stim

def get_single_cnts_of_det_w_L0(defects_matrix, obj, obs_flips, defects_type, num_rounds: int, target_node: int):
    '''Get the number of times a detector fires together w/ L0, irrespective of whether or not other detectors fire.
    Input:
        defects_matrix: defects matrix for X or Z detectors (dims: # of shots x # of rounds for X or Z dets x # of dets per round)
        obj: color code object
        obs_flips: True or False array of length # of shots that indicates whether the logical observable was flipped
        defects_type: "Z" or "X" 
        num_rounds: # of QEC rounds (int)
        target_node: the detector node integer value (int)

    Output:
        The # of times the detector and the L0 are flipped divided by the number of shots
    
    '''
    num_shots     = np.shape(defects_matrix)[0]
    dets_Z,dets_X = get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds)

    if defects_type=="Z":
        dets = dets_Z
        
    elif defects_type=="X":
        dets = dets_X 
        
    else:
        raise Exception("defects_type can be only Z or X.")
    
    rd,anc = dets["D"+str(target_node)]

    locs = np.bitwise_and(defects_matrix.data[:,rd,anc] ,obs_flips)

    return sum(locs)/num_shots


def form_all_elements(DEM: stim.DetectorErrorModel):
    '''Get all the error mechanisms of the Z- or X-DEM and write them as p_{Dj..Dk..L0} if they inlcude a logical flip
       or p_{Dj...Dk..} if they do not include a logical flip.

    Input: 
        DEM: The detector error model
    Output:
        all_elements: list of all possible error probabilities written as p_{Dj...Dk..L0} or p_{Dj...Dk}
    '''
    stims_errors = DEM_to_dictionary(DEM)

    all_elements = []
    for key in stims_errors.keys():

        elem = "p"

        if len(key)==1: #Dj w/o L0
            elem += key[0]
        else:
            for det in key:
                elem+=det 
        
        all_elements.append(elem)

    return all_elements


def form_eq_for_bd_and_L0_flipped_and_not_other_nodes(DEM: stim.DetectorErrorModel, max_order: int, target_node: int, obj, num_rounds: int, defects_type):
    '''Form the equation for the configuration probability P(D_j + L_0) which is a function of physical
       errors that flip D_j and L_0, but w/o other detectors firing. In other words, D_j and L_0 are flipped
       and odd # of times, and other detectors and even # of times.

    Input:
        DEM: Z- or X-DEM of the color code 
        max_order: maximum truncation order (int)
        target_node: index of detector "Dj" (int)
        obj: the color code object
        num_rounds: # of QEC rounds (int)
    Output:
        all_eqs: symbolic equation for the P(D_j+L0) configuration probability truncated to max_order

    '''

    dets_Z,dets_X = get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds)

    if defects_type=="Z":
        dets=dets_Z
    else:
        dets=dets_X

    all_elements = form_all_elements(DEM)
    orders       = np.arange(1,max_order+1)
    all_eqs      = 0

    inspected_combs = []

    for order in orders:
        
        all_combs = list(combinations(all_elements,order))
        
        for comb in all_combs:

            #Consider D0 happening odd # of times on its own, and L0 appearing odd # of times on its own
            
            flag1 = [False]*len(comb)
            flag2 = [False]*len(comb)

            this_eq = 1 
            cnt     = 0 
            for elem in comb:
                
                if "L0" in elem:
                    flag1[cnt]=True 
                if "D"+str(target_node) in elem:
                    flag2[cnt]=True

                this_eq *= symbols(elem)
                cnt     +=1 
            
            
            if (sum(flag1)%2) == 1 and (sum(flag2)%2) ==1 and this_eq not in inspected_combs:
                
                #Check which nodes are already in elem besides the D_j
                #then check if they are flipped an even # of times.


                # cnts_of_Djs = [0]*num_nodes_in_Z_DEM
                cnts_of_Djs = {}
                for key in dets.keys():
                    cnts_of_Djs[key]=0

                for elem in comb:

                    name = elem[1:]

                    if name[-2:]=="L0":
                        name=name[:-2]

                    for l in range(len(name)-1):

                        if name[l]=="D":

                            for k in range(l+1,len(name)):

                                if name[k]=="D":

                                    this_ind = name[l+1:k]

                                    cnts_of_Djs["D"+this_ind]+=1
                                    
                                    break
                    
                    #Now read last entry
                    for l in range(len(name)-1,-1,-1):
                        
                        
                        if name[l]=="D":

                            this_ind = name[l+1:]

                            cnts_of_Djs["D"+this_ind]+=1
                            
                            break


                #Drop the counts for D_target_node (passed, is odd)

                cnts_of_Djs.pop("D"+str(target_node), None)


                cnts_of_Djs = list(cnts_of_Djs.values())
                cnts_of_Djs = [x % 2 for x in cnts_of_Djs]



                if sum(cnts_of_Djs)==0:
                    
                    inspected_combs.append(this_eq)
                    all_elements_new = all_elements.copy()
                    all_elements_new = [x for x in all_elements if x not in comb]
                    
                    for elem in all_elements_new:
                        this_eq *= (1-symbols(elem))


                    all_eqs += this_eq



    return all_eqs


def solve_boundary_equations_with_least_squares(pij_time: dict, pij_bulk: dict, p3: dict, pij_bd:dict,
                                                defects_matrix, obj, num_rounds: int, obs_flips, 
                                                max_order: int, DEM:stim.DetectorErrorModel,
                                                defects_type):
    
    '''Solve the equations for the boundary nodes numerically. We want to separate the contributions 
       of p(D_jL0) and p(Dj) events.

    Input:
        pij_time: dictionary of the time-edge probabilities
        pij_bulk: dictionary of the bulk-edge probabilities (excluding time-edges)
        p3: dictionary of the three-point probabilities
        pij_bd: dictionary of the boundary probabilities which do not split into Dj and Dj+L0 events
        defects_matrix: the defects matrix of the Z- or X-DEM (# of shots x # of QEC rounds for X/Z x # of Z or X detectors per round)
        obj: color code object
        num_rounds: # of QEC rounds
        obs_flips: True/False array storing whether or not the logical observable was flipped in each shots (length # of shots)
        max_order: maximum truncation order for the configuration probabilities
        DEM: the detector error model of the circuit
    Output:
        sols: solution dictionary for the p_{DjL0} events
        pij_bd_final: updated dictionary for boundary probabilities

    '''

    #Determine which boundary nodes we learn exactly from the Z_DEM
    
    errors_in_DEM   = DEM_to_dictionary(DEM)
    dets_Z,dets_X   = get_Z_X_det_nodes_as_rd_anc_pairs_dict(obj,num_rounds)

    if defects_type=="Z":
        dets=dets_Z
    else:
        dets=dets_X

    num_shots       = np.shape(defects_matrix)[0]

    vj_w_L0      = {}
    all_unknowns = []
    target_nodes = [] #the ones that we want to solve for
    known        = {}

    for key in dets.keys():

        if (key,) in errors_in_DEM.keys() and (key,"L0") in errors_in_DEM.keys(): 
            
            elem2 = symbols("p"+key+"L0")
            all_unknowns.append(elem2)
            target_nodes.append(int(key[1:]))

        elif (key,) in errors_in_DEM.keys():

            known[(key,)]      = pij_bd[(key,)]
            known[(key,"L0")] = 0
            

        elif (key,"L0") in errors_in_DEM.keys():

            known[(key,"L0")]   = pij_bd[(key,"L0")]
            known[tuple([key])] = 0

        rd,anc        = dets_Z[key]

        temp          = 0 
        for k in range(num_shots):

            locs = np.nonzero(defects_matrix.data[k,:,:])

            if len(locs[0])==1 and obs_flips[k]==True:

                if locs[0]==rd and locs[1]==anc:
                    temp+=1/num_shots


        vj_w_L0[key]  = temp
        
    all_eqs = []
    
    
    for target_node in target_nodes:

        temp_eq  = form_eq_for_bd_and_L0_flipped_and_not_other_nodes(DEM,max_order,target_node,obj,num_rounds,defects_type)
        final_eq = temp_eq - vj_w_L0["D"+str(target_node)]
        
        #Substitute numeric values 

        for key in pij_time.keys():

            elem = "p"
            for det in key:
                elem += det 
            
            final_eq = final_eq.subs(elem,pij_time[key])
            
        for key in pij_bulk.keys():
            elem = "p"
            for det in key:
                elem += det 
            
            final_eq = final_eq.subs(elem,pij_bulk[key])

        for key in p3.keys():
            elem = "p"
            for det in key:
                elem += det 
            
            final_eq = final_eq.subs(elem,p3[key])
        
        #Substitute the values of the boundary nodes that we are certain about

        for key in known.keys():

            
            if len(key)==1:
                to_subs = "p"+key[0]
            else:
                to_subs = "p"+key[0]+key[1]     

            final_eq = final_eq.subs(to_subs,known[key])
            

        #Now substitute pj("Dj") - > (pij_bd["Dj"]-p("Dj + L0"))/(1-2*p("Dj + L0"))
        
        for key in pij_bd.keys():

            if key not in known.keys():
                
                #a=p(Dj)+p(Dj+L0)-2*p(Dj)*p(Dj+L0) -> p[Dj](1-2*p[Dj+L0])=a-p[Dj+L0] -> p[Dj] = ( a-p[Dj+L0] )/(1-2*p[Dj+L0])

                elem1    = symbols("p"+key[0] )
                elem2    = symbols("p"+key[0] +"L0")
                a        = pij_bd[key]

                val      = (a-elem2)/(1-2*elem2)
                final_eq = final_eq.subs(elem1,val )

        print("eq:",final_eq)
        all_eqs.append(final_eq)

    
    func = []
    
    #Finally, we want to minimize with least_squares the sum of the square of these functions
    for eq in all_eqs:

        x                    = IndexedBase("x")
        d                    = {v: x[i] for i, v in enumerate(all_unknowns)}
        eq = eq.subs(d)
        eq = lambdify(x, eq)
        func.append(eq)

    # Define the residual function
    def residuals(vars):
        x = vars
        return np.array([eq(x) for eq in func])  # Call each function separately    
        
    #Lambdify:
    #All the uknowns that we are solving for are the "pDjL0" (not for all j, because some of them are known)

    L_un          = len(all_unknowns)
    initial_guess = [0.01]*L_un
    min_bound     = 1e-8
    max_bound     = 0.6
    
    bounds = ([min_bound] * L_un, [max_bound]*L_un)
        
    result = least_squares(residuals, initial_guess, bounds=bounds,
                           jac='3-point',verbose=1,loss='soft_l1',
                           gtol=1e-15,ftol=1e-15,xtol=1e-15) #tr_solver='exact' 
    

    pij_bd_final = {}
    sols         = {}
    cnt          = 0
    for var in all_unknowns:

        sols[var]=result.x[cnt]

        det_name                       = str(var)[1:-2]
        pij_bd_final[(det_name,"L0") ] = sols[var]
        pij_bd_final[(det_name,)]         = (pij_bd[(det_name,)]-sols[var])/(1-2*sols[var])
        cnt+=1

    for key in pij_bd.keys():

        if key not in pij_bd_final.keys():

            if (key,) in errors_in_DEM.keys():
                pij_bd_final[(key,)] = pij_bd[(key,)]
            elif (key,"L0") in errors_in_DEM.keys():
                pij_bd_final[(key,"L0")] = pij_bd[(key,"L0")]

    return sols,pij_bd_final 
