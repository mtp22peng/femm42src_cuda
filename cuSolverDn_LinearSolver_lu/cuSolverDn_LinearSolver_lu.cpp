/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 /*
  *  Test three linear solvers, including Cholesky, LU and QR.
  *  The user has to prepare a sparse matrix of "matrix market format" (with
  * extension .mtx). For example, the user can download matrices in Florida
  * Sparse Matrix Collection.
  *  (http://www.cise.ufl.edu/research/sparse/matrices/)
  *
  *  The user needs to choose a solver by switch -R<solver> and
  *  to provide the path of the matrix by switch -F<file>, then
  *  the program solves
  *          A*x = b  where b = ones(m,1)
  *  and reports relative error
  *          |b-A*x|/(|A|*|x|)
  *
  *  The elapsed time is also reported so the user can compare efficiency of
  * different solvers.
  *
  *  How to use
  *      ./cuSolverDn_LinearSolver                     // Default: cholesky
  *     ./cuSolverDn_LinearSolver -R=chol -filefile>   // cholesky factorization
  *     ./cuSolverDn_LinearSolver -R=lu -file<file>     // LU with partial
  * pivoting
  *     ./cuSolverDn_LinearSolver -R=qr -file<file>     // QR factorization
  *
  *  Remark: the absolute error on solution x is meaningless without knowing
  * condition number of A. The relative error on residual should be close to
  * machine zero, i.e. 1.e-15.
  */

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"

#include "helper_cusolver.h"
#include <Header1.h>



void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			double Areg = A[row + col * lda];
			printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Gauss solve lu. </summary>
///
/// <remarks>	, 2020/4/9. </remarks>
///
/// <param name="m">	An int to process. </param>
/// <param name="M">	[in,out] If non-null, a double to process. </param>
/// <param name="B">	[in,out] If non-null, a double to process. </param>
/// <param name="X">	[in,out] If non-null, a double to process. </param>
///
/// <returns>	True if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

BOOL GaussSolve_lu(int m, double **M, double *B, double* X)
{
	int lda = m;
	int ldb = m;
	int n = m;
	double*A;

		for (int row = 0; row < m; row++) {
			for (int col = 0; col < n; col++) {
				A[row + col * lda] = M[row][col];
				printf("(%d,%d) = %f\n", row + 1, col + 1, A[row + col * lda]);
			}
		}





	cusolverDnHandle_t cusolverH = NULL;
	cudaStream_t stream = NULL;

	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;

	/*       | 1 2 3  |
	 *   A = | 4 5 6  |
	 *       | 7 8 10 |
	 *
	 * without pivoting: A = L*U
	 *       | 1 0 0 |      | 1  2  3 |
	 *   L = | 4 1 0 |, U = | 0 -3 -6 |
	 *       | 7 2 1 |      | 0  0  1 |
	 *
	 * with pivoting: P*A = L*U
	 *       | 0 0 1 |
	 *   P = | 1 0 0 |
	 *       | 0 1 0 |
	 *
	 *       | 1       0     0 |      | 7  8       10     |
	 *   L = | 0.1429  1     0 |, U = | 0  0.8571  1.5714 |
	 *       | 0.5714  0.5   1 |      | 0  0       -0.5   |
	 */

	 //double A[lda*m] = { 1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0 };
	 //double B[m] = { 1.0, 2.0, 3.0 };
	//double X[lda]; /* X = A\B */
	//double* X = new double[lda];
	double* LU = new double[lda*lda];

	//double LU[lda*lda]; /* L and U */

	int* Ipiv = new int[lda];
	//int Ipiv[lda];      /* host copy of pivoting sequence */
	int info = 0;     /* host copy of error info */

	double *d_A = NULL; /* device copy of A */
	double *d_B = NULL; /* device copy of B */
	int *d_Ipiv = NULL; /* pivoting sequence */
	int *d_info = NULL; /* error info */
	int  lwork = 0;     /* size of workspace */
	double *d_work = NULL; /* device workspace for getrf */

	const int pivot_on = 0;

	printf("example of getrf \n");

	if (pivot_on) {
		printf("pivot is on : compute P*A = L*U \n");
	}
	else {
		printf("pivot is off: compute A = L*U (not numerically stable)\n");
	}

	printf("A = (matlab base-1)\n");
	printMatrix(m, m, A, lda, "A");
	printf("=====\n");

	printf("B = (matlab base-1)\n");
	printMatrix(m, 1, B, ldb, "B");
	printf("=====\n");

	/* step 1: create cusolver handle, bind a stream */
	status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	assert(cudaSuccess == cudaStat1);

	status = cusolverDnSetStream(cusolverH, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 2: copy A to device */
	cudaStat1 = cudaMalloc((void**)&d_A, sizeof(double) * lda * m);
	cudaStat2 = cudaMalloc((void**)&d_B, sizeof(double) * m);
	cudaStat2 = cudaMalloc((void**)&d_Ipiv, sizeof(int) * m);
	cudaStat4 = cudaMalloc((void**)&d_info, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

	cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)*lda*m, cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_B, B, sizeof(double)*m, cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	/* step 3: query working space of getrf */
	status = cusolverDnDgetrf_bufferSize(
		cusolverH,
		m,
		m,
		d_A,
		lda,
		&lwork);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
	assert(cudaSuccess == cudaStat1);

	/* step 4: LU factorization */
	if (pivot_on) {
		status = cusolverDnDgetrf(
			cusolverH,
			m,
			m,
			d_A,
			lda,
			d_work,
			d_Ipiv,
			d_info);
	}
	else {
		status = cusolverDnDgetrf(
			cusolverH,
			m,
			m,
			d_A,
			lda,
			d_work,
			NULL,
			d_info);
	}
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);
	assert(cudaSuccess == cudaStat1);

	if (pivot_on) {
		cudaStat1 = cudaMemcpy(Ipiv, d_Ipiv, sizeof(int)*m, cudaMemcpyDeviceToHost);
	}
	cudaStat2 = cudaMemcpy(LU, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	if (0 > info) {
		printf("%d-th parameter is wrong \n", -info);
		exit(1);
	}
	if (pivot_on) {
		printf("pivoting sequence, matlab base-1\n");
		for (int j = 0; j < m; j++) {
			printf("Ipiv(%d) = %d\n", j + 1, Ipiv[j]);
		}
	}
	printf("L and U = (matlab base-1)\n");
	printMatrix(m, m, LU, lda, "LU");
	printf("=====\n");

	/*
 * step 5: solve A*X = B
 *       | 1 |       | -0.3333 |
 *   B = | 2 |,  X = |  0.6667 |
 *       | 3 |       |  0      |
 *
 */
	if (pivot_on) {
		status = cusolverDnDgetrs(
			cusolverH,
			CUBLAS_OP_N,
			m,
			1, /* nrhs */
			d_A,
			lda,
			d_Ipiv,
			d_B,
			ldb,
			d_info);
	}
	else {
		status = cusolverDnDgetrs(
			cusolverH,
			CUBLAS_OP_N,
			m,
			1, /* nrhs */
			d_A,
			lda,
			NULL,
			d_B,
			ldb,
			d_info);
	}
	cudaStat1 = cudaDeviceSynchronize();
	assert(CUSOLVER_STATUS_SUCCESS == status);
	assert(cudaSuccess == cudaStat1);

	cudaStat1 = cudaMemcpy(X, d_B, sizeof(double)*m, cudaMemcpyDeviceToHost);
	assert(cudaSuccess == cudaStat1);

	printf("X = (matlab base-1)\n");
	printMatrix(m, 1, X, ldb, "X");
	printf("=====\n");

	/* free resources */
	if (d_A) cudaFree(d_A);
	if (d_B) cudaFree(d_B);
	if (d_Ipiv) cudaFree(d_Ipiv);
	if (d_info) cudaFree(d_info);
	if (d_work) cudaFree(d_work);

	if (cusolverH) cusolverDnDestroy(cusolverH);
	if (stream) cudaStreamDestroy(stream);

	cudaDeviceReset();

	return 0;
}