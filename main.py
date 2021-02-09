#python file
import numpy as np
import fem

if __name__ == "__main__":
    #main code
    #geomet
    data_structure = fem.geomet("FRAME01.txt")
    print(data_structure["coord"][(1,0)])
