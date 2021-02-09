#python file
import numpy as np
import fem

if __name__ == "__main__":
    #main code
    f = open("FRAME01.txt", "r")
    #geomet
    data_structure = fem.geomet(f)

    data_structure = fem.scode(data_structure, f)

    

    f.close()
