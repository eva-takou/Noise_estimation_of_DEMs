import numpy as np
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag
import networkx as nx

#The following constructs the surface code lattice.
#If we consider the X_L crossing from bd to bd, then the X stabs have to be plaquettes and the Z stabs have to be star operators.
#In this case, the logical Z_L is the collection of vertical edges.

#


def surface_code_planar(L):
    
    # Here is the diagram for L=3:
    #
    #    q10       q7         q13
    #D9 -----oD3---------oD6-------
    #        |           |
    #      q2|         q4|    q12
    #D8 -----oD2--------oD5-------
    #   q9   |    q6     |
    #      q1|           |q3
    #D7 -----oD1---------oD4-------
    #     q8       q5         q11

    meas_nodes =[]
    all_meas_nodes = []
    
    for p in range(1,2*L-2,2):
        
        for k in range(0,2*L-1,2):
            
            meas_nodes.append([p,k])
            all_meas_nodes.append([p,k])
            
    
    #now create the artificial bd_nodes
    zend = meas_nodes[-1]
    bds  = [-1,zend[0]+2] #x_coordinates of top and bottom boundary
    bd_nodes = []
    
    for p in range(0,len(bds)): #top and bottom boundary

        for l in range(0,2*L-1,2):
        
            bd_nodes.append([bds[p],l])
            all_meas_nodes.append([bds[p],l])
    

    #create the vertical edges (data qubits)
    data_edge=[]
    for k in range(0,len(meas_nodes)-1):
        z1 = meas_nodes[k]
        z2 = meas_nodes[k+1]

        if z1[0]==z2[0]:
            data_edge.append([z1,z2])

    #create the horizontal edges:
    pmin = 0
    pmax = L

    while True:

        for p in range(pmin,pmax):
            
            z1 = meas_nodes[p]
            z2 = meas_nodes[p+L]
            data_edge.append([z1,z2])
       
        pmin=pmin+L
        pmax=pmax+L
        if pmin+L==len(meas_nodes):
            break
    
    #create the edges with bd nodes:
    for p in range(0,2):
        
        if p==0:
            shift=0
            z_edge = meas_nodes[0:L]
        elif p==1:
            shift =L
            z_edge = meas_nodes[-L:-1]
            z_edge.append(meas_nodes[-1])

        for k in range(0,L):

            z1 = bd_nodes[k+shift]
            z2 = z_edge[k]
            data_edge.append([z1,z2])
    
    Dj_checks=[]
    n = L**2 + (L-1)**2

    for detector in meas_nodes: #real detectors, not virtual ones
        
        cnt=1

        qubit_list = []
        
        for edge in data_edge:
            
            P1 = edge[0]
            P2 = edge[1]

            if P1==detector:
                qubit_list.append(cnt)
            elif P2==detector:
                qubit_list.append(cnt)

            if cnt>n:
                raise Exception("Qubit index out of range.")
            
            cnt=cnt+1
        
        
        Dj_checks.append(qubit_list) #Qubits that each detector checks

    #Based on the qubit list now, form the check matrix (X's only)

    shape1 = int((L**2 + (L-1)**2 -1)/2) 
    shape1 = len(meas_nodes) #rows should be equal to the number of real (not virtual) checks
    


    if len(data_edge)!=(L)**2+(L-1)**2:
        raise Exception("Incorrect number of qubits.")

    
    #Since we only create the x-part i think we should make this with n columns..
    
    H = np.zeros((shape1,n)) 
    #Note: changed here 2*n
    for l in range(len(Dj_checks)):

        d = Dj_checks[l]

        
        for k in d:

            if k-1<0:
                raise Exception("Wrong entry assignment in parity check matrix")
            
            H[l,k-1]=1 #X-check matrix  
    
    return data_edge,all_meas_nodes, meas_nodes, bd_nodes #csc_matrix(H)

def to_network_x_graph(L):

    data_edge,all_meas_nodes,real_meas_nodes,bd_nodes=surface_code_planar(L)

    G=nx.Graph()
    for edge in data_edge:

        v=edge[0]
        w=edge[1]
        
        G.add_edge(tuple(v),tuple(w),weight=1)

    return G


                   

def surface_code_x_stabs(L):
#'''Based on the above script, the X stabilizers of the surface code are plaquettes.
#These stabs check for Z-type errors, so we combine them with the Z_logical operator.'''
    n = L**2 + (L-1)**2
    data_edge,all_meas_nodes,real_meas_nodes,bd_nodes=surface_code_planar(L)
    
    l = round(len(bd_nodes)/2)
    left_bd  = bd_nodes[:l]
    right_bd = bd_nodes[l:]
    
    G=to_network_x_graph(L)
    
    #cc=nx.cycle_basis(G)
    cc=nx.simple_cycles(G,length_bound=4)
    
    HX=np.zeros((len(real_meas_nodes),n))
    cnt=0
    
    
    for cycle in cc:
        
        for p in range(len(cycle)):

            if p==len(cycle)-1:

                v = cycle[p]
                w = cycle[0]
             
            else:
                
                v=cycle[p]
                w=cycle[p+1]

            
            
            edge_cnt=0
            for edge in data_edge:
                #print("edge:",edge,"v:",v,"w:",w)
                
                v1 = tuple(edge[0])
                w1 = tuple(edge[1])
                if (v1==v and w==w1) or (v==w1 and w==v1):
                    HX[cnt,edge_cnt]=1
             
                edge_cnt=edge_cnt+1

        #print(HX[cnt])
        #print("cycle:",cycle)
        cnt=cnt+1

    
    #Now we need the boundary stabilizers. For left bd stabs: get shortest path from left_bd[i] left_bd[i+1]
    
    for i in range(len(left_bd)-1):
        v = tuple(left_bd[i])
        w = tuple(left_bd[i+1])
        path=nx.shortest_path(G,v,w)

        
        for m in range(len(path)-1):
            v = path[m]
            w = path[m+1]

            edge_cnt=0
            for edge in data_edge:

                v_o = tuple(edge[0])
                w_o = tuple(edge[1])

                if (v_o==v and w_o==w) or (v_o==w and w_o==v):

                    HX[cnt,edge_cnt]=1
            
                edge_cnt=edge_cnt+1
    
        cnt=cnt+1

    #Do the same for the right bd:
    
    for i in range(len(right_bd)-1):
        v = tuple(right_bd[i])
        w = tuple(right_bd[i+1])
        path=nx.shortest_path(G,v,w)

        for m in range(len(path)-1):
            v = path[m]
            w = path[m+1]

            edge_cnt=0
            for edge in data_edge:

                v_o = tuple(edge[0])
                w_o = tuple(edge[1])

                if (v_o==v and w_o==w) or (v_o==w and w_o==v):

                    HX[cnt,edge_cnt]=1
            
                edge_cnt=edge_cnt+1
    
        cnt=cnt+1

    
    return csc_matrix(HX)


def surface_code_star_stabs(L):


    return surface_code_z_stabs(L)

def surface_code_loop_stabs(L):
    
    return surface_code_x_stabs(L)


def surface_code_z_stabs(L):
#Based on the above script, the stabilizers of the surface code are star operators.
#These stabs check for X-type errors, so we combine them with X_logical error.

    n = L**2 + (L-1)**2

    data_edge,all_meas_nodes,real_meas_nodes,bd_nodes=surface_code_planar(L)

    Dj_checks=[]
    for detector in real_meas_nodes: #real detectors, not virtual ones
        
        cnt=1

        qubit_list = []
        
        for edge in data_edge:
            
            P1 = edge[0]
            P2 = edge[1]

            if P1==detector:
                qubit_list.append(cnt)
            elif P2==detector:
                qubit_list.append(cnt)

            if cnt>n:
                raise Exception("Qubit index out of range.")
            
            cnt=cnt+1
        
        Dj_checks.append(qubit_list) #Qubits that each detector checks
    
   
    H = np.zeros((len(real_meas_nodes),n)) 
    
    for l in range(len(Dj_checks)):

        d = Dj_checks[l]

        for k in d:

            if k-1<0:
                raise Exception("Wrong entry assignment in parity check matrix")
            
            H[l,k-1]=1 #Zstabs

    return csc_matrix(H)
    


def surface_code_x_logical(L):
#'''Based on the above script, the X stabilizers of the surface code are plaquettes.
#These stabs check for Z-type errors, so we combine them with the Z_logical operator.'''
    n = L**2 + (L-1)**2
    data_edge,all_meas_nodes,real_meas_nodes,bd_nodes=surface_code_planar(L)
    
    l        = round(len(bd_nodes)/2)
    left_bd  = bd_nodes[:l]
    right_bd = bd_nodes[l:]
    #print(left_bd)
    #print(right_bd)

    G    = to_network_x_graph(L)
    path = nx.shortest_path(G,tuple(left_bd[0]),tuple(right_bd[0]))

    XL = np.zeros((1,n))
    
    for l in range(np.shape(data_edge)[0]):

        edge = data_edge[l]
        v1 = tuple(edge[0])
        v2 = tuple(edge[1])

        for p in range(len(path)-1):

            w1 = path[p]
            w2 = path[p+1]

            if (v1==w1 and v2==w2) or (v1==w2 and v2==w1):
                
                
                XL[0,l]=1
        
    
    
    return csc_matrix(XL)

def surface_code_z_logical(L):
    #Logical Z of surface code: horizontal edges in vertical direction (straight line)
    
    n = L**2 + (L-1)**2
    
    logical_Z = np.zeros((1,n))

    meas_nodes =[]
   
    for p in range(1,2*L-2,2):
        for k in range(0,2*L-1,2):
            meas_nodes.append([p,k])
            
    #create the vertical edges (data qubits)
    data_edge=[]
    for k in range(0,len(meas_nodes)-1):
        z1 = meas_nodes[k]
        z2 = meas_nodes[k+1]

        if z1[0]==z2[0]:
            data_edge.append([z1,z2])

    pmin = 0
    pmax = L  
    cnt  = len(data_edge)-1
    qubit_cnt = []
    
    for p in range(pmin,pmax):
        cnt=cnt+1
        qubit_cnt.append(cnt)
    
    for p in qubit_cnt:
        logical_Z[0,p]=1

    return csc_matrix(logical_Z)


def repetition_code(n):
    """
    Parity check matrix of a repetition code with length n.
    """
    row_ind, col_ind = zip(*((i, j) for i in range(n) for j in (i, (i+1)%n)))
    data = np.ones(2*n, dtype=np.uint8)
    return csc_matrix((data, (row_ind, col_ind)))


def toric_code_x_stabilisers(L):
    """
    Sparse check matrix for the X stabilisers of a toric code with
    lattice size L, constructed as the hypergraph product of
    two repetition codes.
    """
    Hr = repetition_code(L)
    H = hstack(
            [kron(Hr, eye(Hr.shape[1])), kron(eye(Hr.shape[0]), Hr.T)],
            dtype=np.uint8
        )
    H.data = H.data % 2
    H.eliminate_zeros()
    return csc_matrix(H)


def toric_code_x_logicals(L):
    """
    Sparse binary matrix with each row corresponding to an X logical operator
    of a toric code with lattice size L. Constructed from the
    homology groups of the repetition codes using the Kunneth
    theorem.
    """
    H1 = csc_matrix(([1], ([0],[0])), shape=(1,L), dtype=np.uint8)
    H0 = csc_matrix(np.ones((1, L), dtype=np.uint8))
    x_logicals = block_diag([kron(H1, H0), kron(H0, H1)])
    x_logicals.data = x_logicals.data % 2
    x_logicals.eliminate_zeros()
    return csc_matrix(x_logicals)



def surface_code_planar_Qubit_Coord(L):
    
    # Here is the diagram for L=3:
    #
    #    q10       q7         q13
    #D9 -----oD3---------oD6-------
    #        |           |
    #      q2|         q4|    q12
    #D8 -----oD2--------oD5-------
    #   q9   |    q6     |
    #      q1|           |q3
    #D7 -----oD1---------oD4-------
    #     q8       q5         q11

    meas_nodes =[]
    all_meas_nodes = []
    
    for p in range(1,2*L-2,2):
        
        for k in range(0,2*L-1,2):
            
            meas_nodes.append([p,k])
            all_meas_nodes.append([p,k])
            
    
    #now create the artificial bd_nodes
    zend = meas_nodes[-1]
    bds  = [-1,zend[0]+2] #x_coordinates of top and bottom boundary
    bd_nodes = []
    
    for p in range(0,len(bds)): #top and bottom boundary

        for l in range(0,2*L-1,2):
        
            bd_nodes.append([bds[p],l])
            all_meas_nodes.append([bds[p],l])
    

    #create the vertical edges (data qubits)
    data_edge=[]
    for k in range(0,len(meas_nodes)-1):
        z1 = meas_nodes[k]
        z2 = meas_nodes[k+1]

        if z1[0]==z2[0]:
            data_edge.append([z1,z2])

    #create the horizontal edges:
    pmin = 0
    pmax = L

    while True:

        for p in range(pmin,pmax):
            
            z1 = meas_nodes[p]
            z2 = meas_nodes[p+L]
            data_edge.append([z1,z2])
       
        pmin=pmin+L
        pmax=pmax+L
        if pmin+L==len(meas_nodes):
            break
    
    #create the edges with bd nodes:
    for p in range(0,2):
        
        if p==0:
            shift=0
            z_edge = meas_nodes[0:L]
        elif p==1:
            shift =L
            z_edge = meas_nodes[-L:-1]
            z_edge.append(meas_nodes[-1])

        for k in range(0,L):

            z1 = bd_nodes[k+shift]
            z2 = z_edge[k]
            data_edge.append([z1,z2])
    
    Dj_checks=[]
    n = L**2 + (L-1)**2

    for detector in meas_nodes: #real detectors, not virtual ones
        
        cnt=1

        qubit_list = []
        
        for edge in data_edge:
            
            P1 = edge[0]
            P2 = edge[1]

            if P1==detector:
                qubit_list.append(cnt)
            elif P2==detector:
                qubit_list.append(cnt)

            if cnt>n:
                raise Exception("Qubit index out of range.")
            
            cnt=cnt+1
        
        
        Dj_checks.append(qubit_list) #Qubits that each detector checks

    #Based on the qubit list now, form the check matrix (X's only)

    shape1 = int((L**2 + (L-1)**2 -1)/2) 
    shape1 = len(meas_nodes) #rows should be equal to the number of real (not virtual) checks
    


    if len(data_edge)!=(L)**2+(L-1)**2:
        raise Exception("Incorrect number of qubits.")

    
    
    return data_edge, meas_nodes, Dj_checks,all_meas_nodes
