# Código de planificación de radioterapia con implementación del método de Adam.

## Carpeta Source
Contiene los códigos fuente (.h y .cpp), incluyendo las clases:
- SparseMatrix
- Region
- Plan 
- Optimizer (Tiene descenso con método de Adam)
- Optimizer_Gradient (Tiene la implementación anterior del gardiente por descenso)

## Carpeta Multicore
Contiene el archivo principal (adam_mlk.cpp), junto con el Makefile y el script de ejecución (run.sh)

-Makefile: Dentro del archivo se divide en dos. Pues una parte para crear el archivo compilado adam_mkl y otra parte para crear el archivo compilado gradient_mkl.

-run.sh: Tambien dentro está dividido en dos partes una para leer el adam_mkl (crea una carpeta donde guarda los resultados llamada results_adam) y la otra parte para leer el gradient_mkl (crea una carpeta donde guarda los resultados llamada results_gradient).

## Observaciones
**Obs:** Siempre se debe comentar tanto en el Makefile como en el run.sh la parte que no se irá a usar.
