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

    fem.assemble(data_structure)

    fem.linear_solver(data_structure)

    print(data_structure["positions"])




    f.close()
