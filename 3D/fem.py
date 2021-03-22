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
                Idof[i,j] = Idof[int(master_node),j]



    data_structure["Idof"] = Idof
    data_structure["nRest"] = nRest
    data_structure["nDof"] = nDof

    return 0

def loads(data_structure, f):
    Idof = data_structure["Idof"]
    nDof = data_structure["nDof"]

    #concentrated LOADS

    VLoads = np.zeros(nDof)

    f.readline()
    nCar = int(f.readline())
    for i in range(nCar):
        line = f.readline()
        line_split = line.split()
        L = int(Idof[int(line_split[0]), int(line_split[1])])
        VLoads[L] = VLoads[L] + float(line_split[2])

    data_structure["VLoads"] = VLoads

    return 0

def rotate_vect(u, coord_1, coord_2):
    dx = coord_1[0] - coord_2[0]
    dy = coord_1[1] - coord_2[1]
    dz = coord_1[2] - coord_2[2]
    L = (dx**2 + dy**2 + dz**2)**0.5
    L_star = (dx**2 + dy**2)**0.5

    T = np.array([[dx/L, dy/L, dz/L],
                  [-dx*dy/(L*L_star), L_star/L, dy*dz/(L*L_star)],
                  [-dz/L_star, 0, dx/L_star]])
    zero_mat = np.zeros((3,3))
    A = np.block([T, zero_mat, zero_mat, zero_mat],
                 [zero_mat, T, zero_mat, zero_mat],
                 [zero_mat, zero_mat, T, zero_mat],
                 [zero_mat, zero_mat, zero_mat, T])
    return A @ u

def rotate_mat(k, coord_1, coord_2):
    dx = coord_1[0] - coord_2[0]
    dy = coord_1[1] - coord_2[1]
    dz = coord_1[2] - coord_2[2]
    L = (dx**2 + dy**2 + dz**2)**0.5
    L_star = (dx**2 + dy**2)**0.5

    T = np.array([[dx/L, dy/L, dz/L],
                  [-dx*dy/(L*L_star), L_star/L, dy*dz/(L*L_star)],
                  [-dz/L_star, 0, dx/L_star]])
    zero_mat = np.zeros((3,3))
    A = np.block([[T, zero_mat, zero_mat, zero_mat],
                 [zero_mat, T, zero_mat, zero_mat],
                 [zero_mat, zero_mat, T, zero_mat],
                 [zero_mat, zero_mat, zero_mat, T]])
    return A @ k @ A.transpose()


def k_local_mat(E, A, I_y, I_z, L, Phi_y, Phi_z, G, J):

    k_z = (E*I_z/((1+Phi_y)*L**2)) * np.array([[12, 6*L, -12, 6*L],
                    [0, (4+Phi_y)*L**2, -6*L, (2-Phi_y)*L**2],
                    [0, 0, 12, -6*L],
                    [0, 0, 0, (4+Phi_y)*L**2]])

    k_y = (E*I_y/((1+Phi_z)*L**2)) * np.array([[12, -6*L, -12, -6*L],
                    [0, (4+Phi_z)*L**2, 6*L, (2-Phi_z)*L**2],
                    [0, 0, 12, 6*L],
                    [0, 0, 0, (4+Phi_z)*L**2]])
    stiffness = np.zeros((12,12))
    stiffness[0,0] = E*A/L; stiffness[0,6] = -E*A/L
    stiffness[1,1] = k_z[0,0]; stiffness[1,5] = k_z[0,1]; stiffness[1,7] = k_z[0,2]; stiffness[1,11]= k_z[0,3]
    stiffness[2,2] = k_y[0,0]; stiffness[2,4] = k_y[0,1]; stiffness[2,8] = k_y[0,2]; stiffness[2,10]= k_y[0,3]
    stiffness[3,3] = G*J/L; stiffness[3,9] = -G*J/L
    stiffness[4,2] = k_y[0,1]; stiffness[4,4] = k_y[1,1]; stiffness[4,8] = k_y[1,2]; stiffness[2,10]= k_y[1,3]
    stiffness[5,1] = k_z[0,1]; stiffness[5,5] = k_z[1,1]; stiffness[5,7] = k_z[1,2]; stiffness[5,11]= k_z[1,3]
    stiffness[6,0] = -E*A/L; stiffness[6,6] = E*A/L
    stiffness[7,1] = k_z[0,2]; stiffness[7,5] = k_z[1,2]; stiffness[7,7] = k_z[2,2]; stiffness[7,11]= k_z[2,3]
    stiffness[8,2] = k_y[0,2]; stiffness[8,4] = k_y[1,2]; stiffness[8,8] = k_y[2,2]; stiffness[8,10]= k_y[2,3]
    stiffness[9,3] = -G*J/L; stiffness[9,9] = G*J/L
    stiffness[10,2] = k_y[0,3]; stiffness[10,4] = k_y[1,3]; stiffness[10,8] = k_y[2,3]; stiffness[10,10]= k_y[3,3]
    stiffness[11,1] = k_z[0,3]; stiffness[11,5] = k_z[1,3]; stiffness[11,7] = k_z[2,3]; stiffness[11,11]= k_z[3,3]

    return stiffness

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
    print(nDof)
    print(Idof)


    k_global = np.zeros((nDof, nDof))

    N_dof = np.zeros(12)

    for i in range(nElement):
        node1 = int(IN[i,0])
        node2 = int(IN[i,1])

        E = CMat[int(IMat[i]), 0]
        A = CSect[int(ISect[i]), 0]
        A_y = CSect[int(ISect[i]), 1]
        A_z = CSect[int(ISect[i]), 2]
        I_y = CSect[int(ISect[i]), 4]
        I_z = CSect[int(ISect[i]), 5]
        J = CSect[int(ISect[i]), 8]
        L = ((coord[node1,0] - coord[node2,0])**2 + (coord[node1,1] - coord[node2,1])**2)**0.5
        CHI = CSect[int(ISect[i]), 6]
        CNU = CMat[int(IMat[i]), 1]

        GG = E/(2*(1+CNU))
        Phi_y = 12*CHI*(E/GG)*(I_z/A_y)/L**2
        Phi_z = 12*CHI*(E/GG)*(I_y/A_z)/L**2

        k_local = k_local_mat(E, A, I_y, I_z, L, Phi_y, Phi_z, GG, J)
        k_local = rotate_mat(k_local, coord[node1][:], coord[node2][:])

        N_dof[0:6] = Idof[node1,:]
        N_dof[6:12] = Idof[node2,:]

        for j in range(12):
            i_dof = int(N_dof[j])
            if i_dof >=0 :
                for p in range(12):
                     j_dof = int(N_dof[p])
                     if j_dof >=0 :
                         k_global[i_dof, j_dof] += k_local[j,p]




    data_structure["k_global"] = k_global
    return 0

def linear_solver(data_structure):
    k_global = data_structure["k_global"]
    # with open('outfile.txt','wb') as f:
    #     for line in np.matrix(k_global):
    #         np.savetxt(f, line, fmt='%.2f')


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
    data = np.zeros(6)
    for i in range(nNode):
        file.write(str(i) + "           ")

        for j in range(6):
            if(Idof[i,j]<0):
                data[j] = 0.0
            else:
                data[j] = positions[int(Idof[i,j])]

        file.write('{:e}  {:e}  {:e} \n'.format(data[0], data[1], data[5]))
