#python file
import numpy as np
import fem

if __name__ == "__main__":
    #main code
    f = open("Juventus_3D_2D.txt", "r")
    #geomet
    data_structure = fem.geomet(f)

    fem.scode(data_structure, f)

    fem.loads(data_structure, f)

    fem.assemble(data_structure)

    fem.linear_solver(data_structure)

    fem.write_output(data_structure)

    f.close()
