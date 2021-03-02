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
    coord = np.empty((nNode,3))
    for i in range(nNode):
        line = f.readline()
        line_split = line.split()
        coord[i,0] = float(line_split[0])
        coord[i,1] = float(line_split[1])
        coord[i,2] = float(line_split[2])

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
    CSect = np.empty((nSection,9))
    for i in range(nSection):
        line = f.readline()
        line_split = line.split()
        CSect[i] = [float(j) for j in line_split]

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
        "CSect": CSect
    }

    return data_structure

def scode(data_structure, f):
    nNode = data_structure["nNode"]

    Idof = -2*np.ones((nNode,6))

    #restraints
    f.readline()
    nRest = int(f.readline())
    for i in range(nRest):
        line = f.readline()
        line_split = line.split()
        line_split = [int(j) for j in line_split]
        if line_split[1] == 6:
            master = f.readline()
            master_split = master.split()
            master_split = [int(j) for j in master_split]
            Idof[line_split[0],master_split[1]] = master_split[0]
        else:
            Idof[line_split[0],line_split[1]] = -1

    nDof = 0


    for i in range(nNode):
        for j in range(6):
            if Idof[i,j] == -2:
                Idof[i,j] = int(nDof)
                nDof = nDof + 1
            elif Idof[i,j] >= 0:
                master_node = Idof[i,j]
                Idof[i,j] = Idof[master_node,j]



    data_structure["Idof"] = Idof
    data_structure["nRest"] = nRest
    data_structure["nDof"] = nDof

    return 0

def rotate_vect(u, alpha):
    alpha_xx = alpha[0]; alpha_xy = alpha[1]; alpha_xz = alpha[2]
    alpha_yx = alpha[3]; alpha_yy = alpha[4]; alpha_yz = alpha[5]
    alpha_zx = alpha[6]; alpha_zy = alpha[7]; alpha_zz = alpha[8]
    T = np.array([[cos(alpha_xx), cos(alpha_yx), cos(alpha_zx)],
                  [cos(alpha_xy), cos(alpha_yy), cos(alpha_zy)],
                  [cos(alpha_xz), cos(alpha_yz), cos(alpha_zz)]])
    zero_mat = np.zeros((3,3))
    A = np.block([T, zero_mat, zero_mat, zero_mat],
                 [zero_mat, T, zero_mat, zero_mat],
                 [zero_mat, zero_mat, T, zero_mat],
                 [zero_mat, zero_mat, zero_mat, T])
    return A @ u

def rotate_mat(k, alpha):
    alpha_xx = alpha[0]; alpha_xy = alpha[1]; alpha_xz = alpha[2]
    alpha_yx = alpha[3]; alpha_yy = alpha[4]; alpha_yz = alpha[5]
    alpha_zx = alpha[6]; alpha_zy = alpha[7]; alpha_zz = alpha[8]
    T = np.array([[cos(alpha_xx), cos(alpha_yx), cos(alpha_zx)],
                  [cos(alpha_xy), cos(alpha_yy), cos(alpha_zy)],
                  [cos(alpha_xz), cos(alpha_yz), cos(alpha_zz)]])
    zero_mat = np.zeros((3,3))
    A = np.block([T, zero_mat, zero_mat, zero_mat],
                 [zero_mat, T, zero_mat, zero_mat],
                 [zero_mat, zero_mat, T, zero_mat],
                 [zero_mat, zero_mat, zero_mat, T])
    return A @ k @ A.transpose()

def compute_angle(coord, node1, node2): #à modif
    dy = coord[node2,1] - coord[node1,1]
    dx = coord[node2,0] - coord[node1,0]
    return np.arctan2(dy,dx)

def k_local_mat(E, A, I, L, Phi): #à modif
    k = np.array([[E*A/L, 0, 0, -E*A/L, 0, 0],
                [0, 12*E*I/(L**3*(1+Phi)), 6*E*I/(L**2*(1+Phi)), 0, -12*E*I/(L**3*(1+Phi)), 6*E*I/(L**2*(1+Phi))],
                [0, 6*E*I/(L**2*(1+Phi)), (4+Phi)*E*I/(L*(1+Phi)), 0, -6*E*I/(L**2*(1+Phi)), (2-Phi)*E*I/(L*(1+Phi))],
                [-E*A/L, 0, 0, E*A/L, 0, 0],
                [0, -12*E*I/(L**3*(1+Phi)), -6*E*I/(L**2*(1+Phi)), 0, 12*E*I/(L**3*(1+Phi)), -6*E*I/(L**2*(1+Phi))],
                [0, 6*E*I/(L**2*(1+Phi)), (2-Phi)*E*I/(L*(1+Phi)), 0, -6*E*I/(L**2*(1+Phi)), (4+Phi)*E*I/(L*(1+Phi))]])
    return k

def assemble(data_structure): #à modif
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

        N_dof[0:3] = Idof[node1,:]
        N_dof[3:6] = Idof[node2,:]
        for j in range(6):
            i_dof = int(N_dof[j])
            if i_dof >=0 :
                for p in range(6):
                     j_dof = int(N_dof[p])
                     if j_dof >=0 :
                         k_global[i_dof, j_dof] += k_local[j,p]



    data_structure["k_global"] = k_global
    return 0
