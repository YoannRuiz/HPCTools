#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "mkl_lapacke.h"
#include <omp.h>

double *generate_matrix(int size)
{
    int i;
    double *matrix = (double *)malloc(sizeof(double) * size * size);
    srand(1);

    for (i = 0; i < size * size; i++)
    {
     	matrix[i] = rand() % 100;
    }

    return matrix;
}

void print_matrix(const char *name, double *matrix, int size)
{
    int i, j;
    printf("matrix: %s \n", name);

    for (i = 0; i < size; i++)
    {
            for (j = 0; j < size; j++)
            {
             	printf("%f ", matrix[i * size + j]);
            }
            printf("\n");
    }
}

int check_result(double *bref, double *b, int size) {
    int i;
    for(i=0;i<size*size;i++) {
        if (abs(bref[i]-b[i]) != 0) return 0;
    }
    return 1;
}


double *mat_zero(int size)
{
    int i;
    double *matrix = (double *)malloc(sizeof(double) * size * size);

    for (i = 0; i < size * size; i++)
    {
     	matrix[i] = 0;
    }

    return matrix;
}

double *transpos(double *mat, int size) {
        int i; // Entier d'itération
        int j; // Entier d'itération
        double *transpos = (double *)malloc(sizeof(double) * size * size); // Matrice transposée

        // Pour chaque ligne
        for(i = 0; i < size; i++) {
                // Pour chaque colonne
                for(j = 0; j < size; j++) {
                        // Attribution des valeurs pour les indices inversés
                        transpos[j*size + i] = mat[i*size + j];
                }
        }
	return transpos;
}

double *prod_mat(double *a, double *b, int size)
{
        double *c = (double *)calloc(size*size, sizeof(double));
       	int i,j,k;
        double *bt = transpos(b,size);
        #pragma omp parallel for private(j,k)
        for(i = 0; i < size; i++)
        {
                
                for(j = 0; j < size; j++)
                {
                        double s = 0;
                        #pragma omp parallel for simd schedule(simd: static) reduction(+:s)
                       	for (k = 0; k < size; k++)
                        {
                                s += a[i*size + k]*bt[j*size + k];
                        }
                c[i*size + j] = s;
                }
        }
        free(bt);
        return c;
}

double norm(double *a, int size, int n){
        int i;
        double s = 0;
        #pragma omp parallel for reduction(+:s)
        for(i = 0; i < size; i+=2){
                s += a[i*size + n]*a[i*size + n];
                s += a[(i+1)*size + n]*a[(i+1)*size + n];
        }
        s = sqrt(s);
        return s;
}

void decompoQR(double *a, int size, double *q, double *r)
{
        int i,j,k;
        
        for(i = 0; i < size; i++)
        {
                r[i*(size + 1)] = norm(a,size,i);
                #pragma omp parallel for private(j)
                #pragma ivdep
                for(j = 0; j < size; j++)
                {
                        q[j*size + i] = a[j*size + i] / r[i*(size + 1)];
                }
                
                #pragma omp parallel for private(j,k) schedule(dynamic)
                for(j = i+1; j < size; j++)
                {
                        double s = 0;
                        #pragma omp parallel for reduction(+:s)
                        for(k = 0; k < size; k++)
                        {
                                s += a[k*size + j] * q[k*size + i];
                        }
                        r[i*size + j] = s;
                        for(k = 0; k < size; k++)
                        {
                                a[k*size + j] -= r[i*size + j] * q[k*size + i];
                        }
                }
        }
}

double *solveQR(double *a, double *b, int size)
{
    double *q = mat_zero(size);
    double *r = mat_zero(size);
    double *x = mat_zero(size);
    double *qt, *y;
    int i, j, k;

    decompoQR(a, size, q, r);
    qt = transpos(q,size);
    y = prod_mat(qt, b, size);
    
    for(i = size - 1; i > -1; i--) {
            #pragma omp parallel for private(j,k) schedule(dynamic)
            for(j = size - 1; j > -1; j--) {
                    double s = 0;
                    #pragma omp parallel for simd schedule(simd: static) reduction(+:s)
                    for(k = i + 1; k < size; k+=4) {
                            s += r[i*size + k]*x[k*size + j];
                            s += r[i*size + k + 1]*x[(k+1)*size + j];
                            s += r[i*size + k + 2]*x[(k+2)*size + j];
                            s += r[i*size + k + 3]*x[(k+3)*size + j];
                    }
            x[i*size + j] = (y[i*size + j] - s)/r[i*(size+1)];
            }
    }
	free(qt);
    free(y);
    free(r);
    free(q);
    return x;
}

void main(int argc, char *argv[])
{

    int size = atoi(argv[1]);

    double *a, *aref;
    double *b, *bref;
    double *x;

    a = generate_matrix(size);
    aref = generate_matrix(size);
    b = generate_matrix(size);
    bref = generate_matrix(size);


    //print_matrix("A", a, size);
    //print_matrix("B", b, size);

    // Using MKL to solve the system
    //MKL_INT n = size, nrhs = size, lda = size, ldb = size, info;
    //MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*size);

    clock_t tStart = clock();
    //info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
    //printf("Time taken by MKL: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    tStart = clock();
    x = solveQR(a, b, size);
    printf("Time taken by my implementation: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

    /*if (check_result(bref,x,size)==1)
        printf("Result is ok!\n");
    else
        printf("Result is wrong!\n");
    free(ipiv);*/
    free(a);
    free(aref);
    
    free(b);
    
    //print_matrix("X", x, size);
    free(x);
    //print_matrix("Xref", bref, size);
    free(bref);
}
