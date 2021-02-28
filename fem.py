#usefull functions
import numpy as np
from math import cos, sin
#from scipy.sparse.linalg import spsolve

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

    data_structure["VLoads"] = VLoads
    data_structure["Card"] = Card
    data_structure["nCard"] = nCard
    data_structure["nTemp"] = nTemp
    data_structure["nPres"] = nPres

    return 0

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

def k_local_mat(E, A, I, L, Phi):
    k = np.array([[E*A/L, 0, 0, -E*A/L, 0, 0],
                [0, 12*E*I/(L**3*(1+Phi)), 6*E*I/(L**2*(1+Phi)), 0, -12*E*I/(L**3*(1+Phi)), 6*E*I/(L**2*(1+Phi))],
                [0, 6*E*I/(L**2*(1+Phi)), (4+Phi)*E*I/(L*(1+Phi)), 0, -6*E*I/(L**2*(1+Phi)), (2-Phi)*E*I/(L*(1+Phi))],
                [-E*A/L, 0, 0, E*A/L, 0, 0],
                [0, -12*E*I/(L**3*(1+Phi)), -6*E*I/(L**2*(1+Phi)), 0, 12*E*I/(L**3*(1+Phi)), -6*E*I/(L**2*(1+Phi))],
                [0, 6*E*I/(L**2*(1+Phi)), (2-Phi)*E*I/(L*(1+Phi)), 0, -6*E*I/(L**2*(1+Phi)), (4+Phi)*E*I/(L*(1+Phi))]])
    return k

def equivalent_nodal_forces(data_structure, NE):
    ENF = np.zeros(6)
    Card = data_structure["Card"]
    coord = data_structure["coord"]
    ISect = data_structure["ISect"]
    IMat = data_structure["IMat"]
    CMat = data_structure["CMat"]
    CSect = data_structure["CSect"]
    IN = data_structure["IN"]

    N1 = int(IN[NE,0])
    N2 = int(IN[NE,1])

    IS = int(ISect[NE])
    IM = int(IMat[NE])
    AA = CSect[IS, 0]
    AJ = CSect[IS, 1]
    CHI = CSect[IS, 2]
    AH = CSect[IS, 3]
    EE = CMat[IM, 0]
    CNU = CMat[IM, 1]
    ALPHA = CMat[IM, 2]
    WEIGHT = CMat[IM, 3]

    if(data_structure["nRigid"]>0):
        CRigid = data_structure["CRigid"]
        A = CRigid[NE,1]
        B = CRigid[NE,2]
        C = CRigid[NE,3]
        D = CRigid[NE,4]
    else:
        A=0; B=0; C=0; D=0;


    DX = coord[N2,0] - coord[N1,0] - A - C
    DY = coord[N2,1] - coord[N1,1] - B - D
    AL = (DX**2 + DY**2)**0.5
    CA = DX/AL
    SA = DY/AL

    PX = Card[NE, 0]
    PY1 = Card[NE, 1]
    PY2 = Card[NE, 2]
    DTI = Card[NE, 3]
    DTS = Card[NE, 4]
    PREC = Card[NE, 5]
    E1 = Card[NE, 6]
    E2 = Card[NE, 7]
    EM = Card[NE, 8]

    ALW = WEIGHT*AA

    PX = PX - ALW*SA
    PY1 = PY1 - ALW*CA
    PY2 = PY2 - ALW*CA

    DTM = (DTS+DTI)/2
    DTD = (DTS-DTI)/2

    TETA1 = (-3*E1 + 4*EM - E2)/AL
    TETA2 = (E1-4*EM + 3*E2)/AL

    CURV = 4*(E1+E2-2*EM)/AL**2

    WW = PREC*CURV
    PY1 = PY1 + WW
    PY2 = PY2 + WW

    DPY = PY2 - PY1

    ENF[0] = PX*AL/2 - ALPHA*DTM*EE*AA
    ENF[1] = PY1*AL/2 + 0.15*DPY*AL
    ENF[2] = (PY1*AL**2)/12 + (DPY*AL**2)/30 + 2*ALPHA*DTD*EE*AJ/AH
    ENF[3] = PX*AL/2 + 0.35*DPY*AL
    ENF[4] = PY1*AL/2 + 0.35*DPY*AL
    ENF[5] = -(PY1*AL**2)/12 - (DPY*AL**2)/20 - 2*ALPHA*DTD*EE*AJ/AH

    if data_structure["nRigid"]>0:
        rigid_vect(ENF, A, B, C, D)

    ENF[0] = ENF[0] + PREC
    ENF[1] = ENF[1] + PREC*TETA1
    ENF[2] = ENF[2] - PREC*E1
    ENF[3] = ENF[3] - PREC
    ENF[4] = ENF[4] - PREC*TETA2
    ENF[5] = ENF[5] + PREC*E2


    return ENF

def rigid_vect(k, A_r, B_r, C_r, D_r):
    mat = np.array([[1, 0, -B_r, 0, 0, 0],
                    [0, 1, A_r, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, D_r],
                    [0, 0, 0, 0, 1, -C_r],
                    [0, 0, 0, 0, 0, 1]])
    return mat.transpose() @ k

def rigid_mat(k, A_r, B_r, C_r, D_r):
    mat = np.array([[1, 0, -B_r, 0, 0, 0],
                    [0, 1, A_r, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, D_r],
                    [0, 0, 0, 0, 1, -C_r],
                    [0, 0, 0, 0, 0, 1]])
    return mat @ k @ mat.transpose()

def assemble(data_structure):
    n_max_dof = data_structure["nNode"]*3
    nElement = data_structure["nElement"]
    IMat = data_structure["IMat"]
    CMat = data_structure["CMat"]
    IN = data_structure["IN"]
    ISect = data_structure["ISect"]
    CSect = data_structure["CSect"]
    coord = data_structure["coord"]
    Idof = data_structure["Idof"]
    nDof = data_structure["nDof"]
    CRigid = data_structure["CRigid"]
    nRigid = data_structure["nRigid"]
    VLoads = data_structure["VLoads"]


    k_global = np.zeros((nDof, nDof))
    N_dof = np.zeros(6)

    for i in range(nElement):
        node1 = int(IN[i,0])
        node2 = int(IN[i,1])
        E = CMat[int(IMat[i]), 0]
        A = CSect[int(ISect[i]), 0]
        I = CSect[int(ISect[i]), 1]
        L = ((coord[node1,0] - coord[node2,0])**2 + (coord[node1,1] - coord[node2,1])**2)**0.5
        CHI = CSect[int(ISect[i]), 2]
        CNU = CMat[int(IMat[i]), 1]

        GG = E/(2*(1+CNU))
        Phi = 12*CHI*(E/GG)*(I/A)/L**2

        alpha = compute_angle(coord, node1, node2)

        k_local = k_local_mat(E, A, I, L, Phi)
        k_local = rotate_mat(k_local, alpha)
        if nRigid > 0:
            A_r, B_r, C_r, D_r = CRigid[1:5]
            k_local = rigid_mat(k_local, A_r, B_r, C_r, D_r)

        enf = equivalent_nodal_forces(data_structure, i)
        enfg = rotate_vect(enf, alpha)
        if nRigid > 0:
            A_r, B_r, C_r, D_r = CRigid[1:5]
            enfg = rigid_mat(enfg, A_r, B_r, C_r, D_r)

        N_dof[0:3] = Idof[node1,:]
        N_dof[3:6] = Idof[node2,:]
        for j in range(6):
            i_dof = int(N_dof[j])
            if i_dof >=0 :
                VLoads[i_dof] += enfg[j]
                for p in range(6):
                     j_dof = int(N_dof[p])
                     if j_dof >=0 :
                         k_global[i_dof, j_dof] += k_local[j,p]



    data_structure["k_global"] = k_global
    data_structure["VLoads"] = VLoads
    return 0

def linear_solver(data_structure):
    k_global = data_structure["k_global"]
    VLoads = data_structure["VLoads"]
    positions = np.linalg.solve(k_global, VLoads)

    data_structure["positions"] = positions
    return 0

def write_output(data_structure):
    nDof = data_structure["nDof"]
    Idof = data_structure["Idof"]
    nNode = data_structure["nNode"]
    positions = data_structure["positions"]

    file = open("output.txt", "w")

    file.write("NODAL DISPLACEMENTS: \n")
    file.write("NODES             U             V           THETA \n")
    data = np.zeros(3)
    for i in range(nNode):
        file.write(str(i) + "           ")

        for j in range(3):
            if(Idof[i,j]<0):
                data[j] = 0.0
            else:
                data[j] = positions[int(Idof[i,j])]

        file.write('{:e}  {:e}  {:e} \n'.format(data[0], data[1], data[2]))
