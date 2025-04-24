import numpy as np
from scipy.sparse import csc_matrix
import networkx as nx


def surface_code_planar(L: int):
    '''Create the unrotate surface code lattice.
    
    Input:
        L: distance (int)
    Output:
        data_edge:      list of data qubit edges [index of anc1 , index of anc2]
        all_meas_nodes: list of [x,y] positions of all measurement nodes 
        meas_nodes:     list of [x,y] positions of measurement nodes that are in the bulk and not the boundary
        bd_nodes        list of [x,y] locations of boundary measurement nodes

    '''
    
    # Diagram for L=3 and names of data & ancilla qubits:
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


    if len(data_edge)!=(L)**2+(L-1)**2:
        raise Exception("Incorrect number of data qubits.") 
    
    return data_edge,all_meas_nodes, meas_nodes, bd_nodes 

def to_network_x_graph(L: int):
    '''Return the networkX graph of the unrotated surfaec code
    
    Input:
        L: distance (int)
    Output:
        G: the graph of the unrotate surface code lattice

    '''
    data_edge,all_meas_nodes,real_meas_nodes,bd_nodes=surface_code_planar(L)

    G=nx.Graph()
    for edge in data_edge:

        v=edge[0]
        w=edge[1]
        
        G.add_edge(tuple(v),tuple(w),weight=1)

    return G

def surface_code_z_stabs(L: int):
    '''Z-type stabilizers for the unrotate surface code. Those stabilizers form loops.

    Input:
        L: distance of the code
    Output:
        csc_matrix(HZ): The HZ parity check matrix (csc matrix)
    '''
    n = L**2 + (L-1)**2
    data_edge,all_meas_nodes,real_meas_nodes,bd_nodes=surface_code_planar(L)
    
    l = round(len(bd_nodes)/2)
    left_bd  = bd_nodes[:l]
    right_bd = bd_nodes[l:]
    
    G=to_network_x_graph(L)
    
    #cc=nx.cycle_basis(G)
    cc=nx.simple_cycles(G,length_bound=4)
    
    HZ=np.zeros((len(real_meas_nodes),n))
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
                    HZ[cnt,edge_cnt]=1
             
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

                    HZ[cnt,edge_cnt]=1
            
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

                    HZ[cnt,edge_cnt]=1
            
                edge_cnt=edge_cnt+1
    
        cnt=cnt+1

    
    return csc_matrix(HZ)

def surface_code_star_stabs(L: int):
    '''Return star-like stabilizers (X-type)
    Input:
        L: distance
    Output:
        HX parity check matrix'''
    
    return surface_code_x_stabs(L)

def surface_code_loop_stabs(L: int):
    '''Return loop-like stabilizers (Z-type)
    Input:
        L: distance
    Output:
        HZ parity check matrix'''
    
    return surface_code_z_stabs(L)

def surface_code_x_stabs(L: int):
    '''X-type stabilizers for the unrotate surface code. Those stabilizers form loops.

    Input:
        L: distance of the code
    Output:
        csc_matrix(H): The HX parity check matrix (csc matrix)
    '''
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
    
def surface_code_z_logical(L: int):
    '''Logical Z operator for the unrotate surface code. This operator is a straight horizontal line from one boundary to the other.
    
    Input: 
        L: distance of the code
    Output:
        csc_matrix(ZL): Z_L operator in binary format of length n, where n is the number of data qubits (csc matrix)
    '''

    n = L**2 + (L-1)**2
    data_edge,all_meas_nodes,real_meas_nodes,bd_nodes=surface_code_planar(L)
    
    l        = round(len(bd_nodes)/2)
    left_bd  = bd_nodes[:l]
    right_bd = bd_nodes[l:]


    G    = to_network_x_graph(L)
    path = nx.shortest_path(G,tuple(left_bd[0]),tuple(right_bd[0]))

    ZL = np.zeros((1,n))
    
    for l in range(np.shape(data_edge)[0]):

        edge = data_edge[l]
        v1 = tuple(edge[0])
        v2 = tuple(edge[1])

        for p in range(len(path)-1):

            w1 = path[p]
            w2 = path[p+1]

            if (v1==w1 and v2==w2) or (v1==w2 and v2==w1):
                
                
                ZL[0,l]=1
        
    return csc_matrix(ZL)

def surface_code_x_logical(L: int):
    '''Logical X operator for the unrotate surface code. This operator is a collection of straight lines in the vertical direction

    Input: 
        L: distance of the code
    Output:
        csc_matrix(logical_X): X_L operator in binary format of length n, where n is the number of data qubits (csc matrix)

    '''
    
    n = L**2 + (L-1)**2
    
    logical_X = np.zeros((1,n))

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
        logical_X[0,p]=1

    return csc_matrix(logical_X)
