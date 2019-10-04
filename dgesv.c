#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "mkl_lapacke.h"

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
    printf("matrix: %s \n", matrix);

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

double *prod_mat(double *a, double *b, int size)
{
	double *c = (double *)malloc(sizeof(double) * size * size);

	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			c[i*size + j] = 0;

			for (int k = 0; k < size; k++)
			{
				c[i*size + j] += a[i*size + k]*b[k*size + j];
			}
		}
	}
	return c;
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
          		transpos[i*size + j] = mat[j*size + i];
		}
	}
	return transpos;
}


void decompoQR(double *a, int size, double *q, double *r)
{
	int i,j,k;	
	
	for(i = 0; i < size; i++)
	{
		double s = 0;
		
		for(j = 0; j < size; j++)
		{
			s += a[j*size + i]*a[j*size + i];
		}
		r[i*(size + 1)] = sqrt(s);
		for(j = 0; j < size; j++)
		{
			q[j*size + i] = a[j*size + i] / r[i*(size + 1)];
		}
		for(j = i+1; j < size; j++)
		{
			s = 0;
			for(k = 0; k < size; k++)
			{
				s = s + a[k*size + j] * q[k*size + i];
			}
			r[i*size + j] = s;
			for(k = 0; k < size; k++)
			{
				a[k*size + j] = a[k*size + j] - r[i*size + j] * q[k*size + i];
			}
		}
	}
}

double *solveQR(double *a, double *b, int size)
{
	double *q = mat_zero(size);
	double *r = mat_zero(size);
	double *x = mat_zero(size);
	int i, j, k;

    decompoQR(a, size, q, r);
	double *y = prod_mat(transpos(q, size), b, size);
	
	for(i = size - 1; i > -1; i--) {
		for(j = size - 1; j > -1; j--) {
			double s = 0;
			for(k = i + 1; k < size; k++) {
				s += r[i*size + k]*x[k*size + j];
			}
		x[i*size + j] = (y[i*size + j] - s)/r[i*(size+1)];
		}
	}
	return x;
}

    void main(int argc, char *argv[])
    {

     	int size = atoi(argv[1]);

        double *a, *aref;
        double *b, *bref;

        a = generate_matrix(size);
        aref = generate_matrix(size);
        b = generate_matrix(size);
        bref = generate_matrix(size);


        //print_matrix("A", a, size);
        //print_matrix("B", b, size);

        // Using MKL to solve the system
        //MKL_INT n = size, nrhs = size, lda = size, ldb = size, info;
        //MKL_INT *ipiv = (MKL_INT *)malloc(sizeof(MKL_INT)*size);

        //clock_t tStart = clock();
        //info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, aref, lda, ipiv, bref, ldb);
        //printf("Time taken by MKL: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

        clock_t tStart = clock();
            b = solveQR(a, b, size);
        printf("Time taken by my implementation: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

        /*if (check_result(bref,b,size)==1)
            printf("Result is ok!\n");
        else
            printf("Result is wrong!\n");
        */
	//print_matrix("X", b, size);
        //print_matrix("Xref", bref, size);
    }

