#python file
import numpy as np
import fem

if __name__ == "__main__":
    #main code
    f = open("FRAME03.txt", "r")
    #geomet
    data_structure = fem.geomet(f)

    fem.scode(data_structure, f)

    fem.loads(data_structure, f)




    f.close()
