#usefull functions
import numpy as np
from math import cos, sin


def geomet(f):

    # Nnode, Nelement, Nmaterial and Nsection
    f.readline()
    line = f.readline()
    line_split = line.split()
    nNode = int(line_split[0])
    nElement = int(line_split[1])
    nMaterial = int(line_split[2])
    nSection = int(line_split[3])

    #coordinate
    f.readline()
    coord = np.empty((nNode,2))
    for i in range(nNode):
        line = f.readline()
        line_split = line.split()
        coord[i,0] = float(line_split[0])
        coord[i,1] = float(line_split[1])

    #connectivity DATA
    f.readline()
    IN = np.empty((nElement,2))
    IMat = np.empty(nElement)
    ISect = np.empty(nElement)
    for i in range(nElement):
        line = f.readline()
        line_split = line.split()
        IN[i,0] = int(line_split[0])
        IN[i,1] = int(line_split[1])
        IMat[i] = int(line_split[2])
        ISect[i] = int(line_split[3])

    #materials data
    f.readline()
    CMat = np.empty((nMaterial,4))
    for i in range(nMaterial):
        line = f.readline()
        line_split = line.split()
        CMat[i] = [float(j) for j in line_split]

    #section data
    f.readline()
    CSect = np.empty((nSection,4))
    for i in range(nSection):
        line = f.readline()
        line_split = line.split()
        CSect[i] = [float(j) for j in line_split]

    #rigid link data
    f.readline()
    nRigid = int(f.readline())
    CRigid = np.empty((nRigid,4))
    for i in range(nRigid):
        line = f.readline()
        line_split = line.split()
        CRigid[i] = [float(j) for j in line_split]

    data_structure = {
        "nNode": nNode,
        "nElement": nElement,
        "nMaterial": nMaterial,
        "nSection": nSection,
        "coord": coord,
        "IN": IN,
        "IMat": IMat,
        "ISect": ISect,
        "CMat": CMat,
        "CSect": CSect,
        "nRigid": nRigid,
        "CRigid": CRigid,
    }

    return data_structure

def scode(data_structure, f):
    nNode = data_structure["nNode"]

    Idof = np.zeros((nNode,3))

    #restraints
    f.readline()
    nRest = int(f.readline())
    for i in range(nRest):
        line = f.readline()
        line_split = line.split()
        line_split = [int(j) for j in line_split]
        if line_split[1] == 4:
            master = f.readline()
            master_split = master.split()
            master_split = [int(j) for j in master_split]
            Idof[line_split[0],master_split[1]] = master_split[0]
        else:
            Idof[line_split[0],line_split[1]] = -1

    nDof = 0

    for i in range(nNode):
        for j in range(3):
            if Idof[i,j] == 0:
                Idof[i,j] = int(nDof)
                nDof = nDof + 1
            elif Idof[i,j] > 0:
                master_node = Idof[i,j]
                Idof[i,j] = Idof[master_node,j]

    #displacement
    f.readline()
    nDisp = int(f.readline())
    CDisp = np.empty((nDisp,3))
    for i in range(nDisp):
        line = f.readline()
        line_split = line.split()
        CDisp[i] = [float(j) for j in line_split]

    #spring
    f.readline()
    nSpring = int(f.readline())
    CSpring = np.empty((nSpring,3))
    for i in range(nSpring):
        line = f.readline()
        line_split = line.split()
        CSpring[i] = [float(j) for j in line_split]



    data_structure["Idof"] = Idof
    data_structure["nRest"] = nRest
    data_structure["nDof"] = nDof
    data_structure["nDisp"] = nDisp
    data_structure["CDisp"] = CDisp
    data_structure["nSpring"] = nSpring
    data_structure["CSpring"] = CSpring

    return 0

def loads(data_structure, f):
    Idof = data_structure["Idof"]
    nDof = data_structure["nDof"]
    nElement = data_structure["nElement"]

    #concentrated LOADS

    VLoads = np.zeros(nDof)

    f.readline()
    nCar = int(f.readline())
    for i in range(nCar):
        line = f.readline()
        line_split = line.split()
        L = int(Idof[int(line_split[0]), int(line_split[1])])
        VLoads[L] = VLoads[L] + float(line_split[2])

    #distributed loads
    Card = np.zeros((nElement, 10))

    f.readline()
    nCard = int(f.readline())

    for i in range(nCard):
        line = f.readline()
        line_split = line.split()
        Card[int(line_split[0]),1] =  float(line_split[1])
        Card[int(line_split[0]),2] =  float(line_split[2])
        Card[int(line_split[0]),3] =  float(line_split[3])

    #thermal VARIATIONS

    f.readline()
    nTemp = int(f.readline())
    for i in range(nTemp):
        line = f.readline()
        line_split = line.split()
        Card[int(line_split[0]), 4] = float(line_split[1])
        Card[int(line_split[0]), 5] = float(line_split[2])

    # PRETENSION
    f.readline()
    nPres = int(f.readline())
    for i in range(nPres):
        line = f.readline()
        line_split = line.split()
        Card[int(line_split[0]), 6] = float(line_split[1])
        Card[int(line_split[0]), 7] = float(line_split[2])
        Card[int(line_split[0]), 8] = float(line_split[3])
        Card[int(line_split[0]), 9] = float(line_split[4])


def rotate_vect(u, alpha):
    A = np.array([[cos(alpha), -sin(alpha), 0, 0, 0, 0],
                [sin(alpha), cos(alpha), 0, 0, 0, 0],
                [0, 0, 1, 0 ,0, 0],
                [0, 0, 0, cos(alpha), -sin(alpha), 0],
                [0, 0, 0, sin(alpha), cos(alpha), 0],
                [0, 0, 0, 0, 0, 1]])
    return A @ u

def rotate_mat(k, alpha):
    A = np.array([[cos(alpha), -sin(alpha), 0, 0, 0, 0],
                [sin(alpha), cos(alpha), 0, 0, 0, 0],
                [0, 0, 1, 0 ,0, 0],
                [0, 0, 0, cos(alpha), -sin(alpha), 0],
                [0, 0, 0, sin(alpha), cos(alpha), 0],
                [0, 0, 0, 0, 0, 1]])
    return A @ k @ A.transpose()

def compute_angle(coord, node1, node2):
    dy = coord[node2,1] - coord[node1,1]
    dx = coord[node2,0] - coord[node1,0]
    return np.arctan2(dy,dx)

def k_local_EB(E, A, I, L):
    k = np.array([[E*A/L, 0, 0, -E*A/L, 0, 0],
                [0, 12*E*I/(L**3), 6*E*I/(L**2), 0, -12*E*I/(L**3), 6*E*I/(L**2)],
                [0, 6*E*I/(L**2), 4*E*I/L, 0, -6*E*I/(L**2), 2*E*I/L],
                [-E*A/L, 0, 0, E*A/L, 0, 0],
                [0, -12*E*I/(L**3), -6*E*I/(L**2), 0, 12*E*I/(L**3), -6*E*I/(L**2)],
                [0, 6*E*I/(L**2), 2*E*I/L, 0, -6*E*I/(L**2), 4*E*I/L]])
    return k

def assemble(data_structure):
    n_max_dof = data_structure["nNode"]*3
    nElement = data_structure["nElement"]
    IMat = data_structure["IMat"]
    CMat = data_structure["CMat"]
    IN = data_structure["IN"]
    ISect = data_structure["ISect"]
    CSect = data_structure["CSect"]
    coord = data_structure["coord"]

    k_global = np.zeros((n_max_dof, n_max_dof))

    for i in range(nElement):
        node1 = int(IN[i,0])
        node2 = int(IN[i,1])
        E = CMat[int(IMat[i]), 0]
        A = CSect[int(ISect[i]), 0]
        I = CSect[int(ISect[i]), i]
        L = ((coord[node1,0] - coord[node2,0])**2 + (coord[node1,1] - coord[node2,1])**2)**0.5
        alpha = compute_angle(coord, node1, node2)

        k_local = k_local_EB(E, A, I, L)
        k_local = rotate_mat(k_local, alpha)

        k_global[node1*3:node1*3+3,node1*3:node1*3+3] = np.add(k_global[node1*3:node1*3+3,node1*3:node1*3+3], k_local[0:3, 0:3])
        k_global[node1*3:node1*3+3,node2*3:node2*3+3] = np.add(k_global[node1*3:node1*3+3,node2*3:node2*3+3], k_local[0:3, 3:6])
        k_global[node2*3:node2*3+3,node1*3:node1*3+3] = np.add(k_global[node2*3:node2*3+3,node1*3:node1*3+3], k_local[3:6, 0:3])
        k_global[node2*3:node2*3+3,node2*3:node2*3+3] = np.add(k_global[node2*3:node2*3+3,node2*3:node2*3+3], k_local[3:6, 3:6])

    print(k_global)

    return 0
