#usefull functions
import numpy as np


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
    coord = np.empty((nNode+1,2))
    for i in range(1,nNode+1):
        line = f.readline()
        line_split = line.split()
        coord[i,0] = float(line_split[0])
        coord[i,1] = float(line_split[1])

    #connectivity DATA
    f.readline()
    IN = np.empty((nElement+1,2))
    IMat = np.empty(nElement+1)
    ISect = np.empty(nElement+1)
    for i in range(1,nElement+1):
        line = f.readline()
        line_split = line.split()
        IN[i,0] = int(line_split[0])
        IN[i,1] = int(line_split[1])
        IMat[i] = int(line_split[2])
        ISect[i] = int(line_split[3])

    #materials data
    f.readline()
    CMat = np.empty((nMaterial+1,4))
    for i in range(1,nMaterial+1):
        line = f.readline()
        line_split = line.split()
        CMat[i] = [float(j) for j in line_split]

    #section data
    f.readline()
    CSect = np.empty((nSection+1,4))
    for i in range(1,nSection+1):
        line = f.readline()
        line_split = line.split()
        CSect[i] = [float(j) for j in line_split]

    #rigid link data
    f.readline()
    nRigid = int(f.readline())
    CRigid = np.empty((nRigid+1,4))
    for i in range(1,nRigid+1):
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

    Idof = np.zeros((nNode+1,4))

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

    for i in range(1,nNode+1):
        for j in range(1,4):
            if Idof[i,j] == 0:
                nDof = nDof + 1
                Idof[i,j] = int(nDof)
            elif Idof[i,j] > 0:
                master_node = Idof[i,j]
                Idof[i,j] = Idof[master_node,j]

    #displacement
    f.readline()
    nDisp = int(f.readline())
    CDisp = np.empty((nDisp+1,3))
    for i in range(1,nDisp+1):
        line = f.readline()
        line_split = line.split()
        CDisp[i] = [float(j) for j in line_split]

    #spring
    f.readline()
    nSpring = int(f.readline())
    CSpring = np.empty((nSpring+1,3))
    for i in range(1,nSpring+1):
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

    VLoads = np.zeros(nDof+1)

    f.readline()
    nCar = int(f.readline())
    for i in range(1,nCar+1):
        line = f.readline()
        line_split = line.split()
        L = int(Idof[int(line_split[0]), int(line_split[1])])
        VLoads[L] = VLoads[L] + float(line_split[2])

    #distributed loads
    Card = np.zeros((nElement+1, 10))

    f.readline()
    nCard = int(f.readline())

    for i in range(1, nCard+1):
        line = f.readline()
        line_split = line.split()
        Card[int(line_split[0]),1] =  float(line_split[1])
        Card[int(line_split[0]),2] =  float(line_split[2])
        Card[int(line_split[0]),3] =  float(line_split[3])

    #thermal VARIATIONS

    f.readline()
    nTemp = int(f.readline())
    for i in range(1,nTemp+1):
        line = f.readline()
        line_split = line.split()
        Card[int(line_split[0]), 4] = float(line_split[1])
        Card[int(line_split[0]), 5] = float(line_split[2])

    # PRETENSION
     f.readline()
     nPres = int(f.readline())
     for i in range(1,nPres+1):
         line = f.readline()
         line_split = line.split()
         Card[int(line_split[0]), 6] = float(line_split[1])
         Card[int(line_split[0]), 7] = float(line_split[2])
         Card[int(line_split[0]), 8] = float(line_split[3])
         Card[int(line_split[0]), 9] = float(line_split[4])
