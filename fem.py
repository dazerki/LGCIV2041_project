#usefull functions
import numpy as np


def geomet(namefile):
    f = open(namefile, "r")
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

    return 0
