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


#include "cusparse.h"
#include "cusolverSp.h"







BOOL SparseSolve_lu(int m, double **M, double *B, double* X)
{
	struct testOpts opts;
	cusolverSpHandle_t handle = NULL;
	cusparseHandle_t cusparseHandle = NULL; /* used in residual evaluation */
	cudaStream_t stream = NULL;
	cusparseMatDescr_t descrA = NULL;

	int rowsA = 0; /* number of rows of A */
	int colsA = 0; /* number of columns of A */
	int nnzA = 0;  /* number of nonzeros of A */
	int baseA = 0; /* base index in CSR format */

	/* CSR(A) from I/O */
	int *h_csrRowPtrA = NULL;
	int *h_csrColIndA = NULL;
	double *h_csrValA = NULL;

	double *h_z = NULL;  /* z = B \ (Q*b) */
	double *h_x = NULL;  /* x = A \ b */
	double *h_b = NULL;  /* b = ones(n,1) */
	double *h_Qb = NULL; /* Q*b */
	double *h_r = NULL;  /* r = b - A*x */

	int *h_Q = NULL; /* <int> n */
					 /* reorder to reduce zero fill-in */
					 /* Q = symrcm(A) or Q = symamd(A) */
	/* B = Q*A*Q' or B = A(Q,Q) by MATLAB notation */
	int *h_csrRowPtrB = NULL; /* <int> n+1 */
	int *h_csrColIndB = NULL; /* <int> nnzA */
	double *h_csrValB = NULL; /* <double> nnzA */
	int *h_mapBfromA = NULL;  /* <int> nnzA */

	size_t size_perm = 0;
	void *buffer_cpu = NULL; /* working space for permutation: B = Q*A*Q^T */

	/* device copy of A: used in residual evaluation */
	int *d_csrRowPtrA = NULL;
	int *d_csrColIndA = NULL;
	double *d_csrValA = NULL;

	/* device copy of B: used in B*z = Q*b */
	int *d_csrRowPtrB = NULL;
	int *d_csrColIndB = NULL;
	double *d_csrValB = NULL;

	int *d_Q = NULL;     /* device copy of h_Q */
	double *d_z = NULL;  /* z = B \ Q*b */
	double *d_x = NULL;  /* x = A \ b */
	double *d_b = NULL;  /* a copy of h_b */
	double *d_Qb = NULL; /* a copy of h_Qb */
	double *d_r = NULL;  /* r = b - A*x */

	double tol = 1.e-12;
	const int reorder = 0; /* no reordering */
	int singularity = 0;   /* -1 if A is invertible under tol. */

	/* the constants are used in residual evaluation, r = b - A*x */
	const double minus_one = -1.0;
	const double one = 1.0;

	double b_inf = 0.0;
	double x_inf = 0.0;
	double r_inf = 0.0;
	double A_inf = 0.0;
	int errors = 0;
	int issym = 0;

	double start, stop;
	double time_solve_cpu;
	double time_solve_gpu;



	if (NULL == opts.testFunc) {
		opts.testFunc =
			"chol"; /* By default running Cholesky as NO solver selected with -R
					   option. */
	}


	int argc = 0;

	char ** argv;

	findCudaDevice(argc, (const char **)argv);

	if (opts.sparse_mat_filename == NULL) {
		opts.sparse_mat_filename = sdkFindFilePath("lap2D_5pt_n100.mtx", argv[0]);
		if (opts.sparse_mat_filename != NULL)
			printf("Using default input file [%s]\n", opts.sparse_mat_filename);
		else
			printf("Could not find lap2D_5pt_n100.mtx\n");
	}
	else {
		printf("Using input file [%s]\n", opts.sparse_mat_filename);
	}

	printf("step 1: read matrix market format\n");

	if (opts.sparse_mat_filename == NULL) {
		fprintf(stderr, "Error: input matrix is not provided\n");
		return EXIT_FAILURE;
	}

	
	baseA = h_csrRowPtrA[0];  // baseA = {0,1}
	printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA,
		nnzA, baseA);

	if (rowsA != colsA) {
		fprintf(stderr, "Error: only support square matrix\n");
		return 1;
	}

	checkCudaErrors(cusolverSpCreate(&handle));
	

	


	h_z = (double *)malloc(sizeof(double) * colsA);
	h_x = (double *)malloc(sizeof(double) * colsA);
	h_b = (double *)malloc(sizeof(double) * rowsA);
	h_Qb = (double *)malloc(sizeof(double) * rowsA);
	h_r = (double *)malloc(sizeof(double) * rowsA);

	h_Q = (int *)malloc(sizeof(int) * colsA);
	h_csrRowPtrB = (int *)malloc(sizeof(int) * (rowsA + 1));
	h_csrColIndB = (int *)malloc(sizeof(int) * nnzA);
	h_csrValB = (double *)malloc(sizeof(double) * nnzA);
	h_mapBfromA = (int *)malloc(sizeof(int) * nnzA);

	assert(NULL != h_z);
	assert(NULL != h_x);
	assert(NULL != h_b);
	assert(NULL != h_Qb);
	assert(NULL != h_r);
	assert(NULL != h_Q);
	assert(NULL != h_csrRowPtrB);
	assert(NULL != h_csrColIndB);
	assert(NULL != h_csrValB);
	assert(NULL != h_mapBfromA);

	checkCudaErrors(
		cudaMalloc((void **)&d_csrRowPtrA, sizeof(int) * (rowsA + 1)));
	checkCudaErrors(cudaMalloc((void **)&d_csrColIndA, sizeof(int) * nnzA));
	checkCudaErrors(cudaMalloc((void **)&d_csrValA, sizeof(double) * nnzA));
	checkCudaErrors(
		cudaMalloc((void **)&d_csrRowPtrB, sizeof(int) * (rowsA + 1)));
	checkCudaErrors(cudaMalloc((void **)&d_csrColIndB, sizeof(int) * nnzA));
	checkCudaErrors(cudaMalloc((void **)&d_csrValB, sizeof(double) * nnzA));
	checkCudaErrors(cudaMalloc((void **)&d_Q, sizeof(int) * colsA));
	checkCudaErrors(cudaMalloc((void **)&d_z, sizeof(double) * colsA));
	checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double) * colsA));
	checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double) * rowsA));
	checkCudaErrors(cudaMalloc((void **)&d_Qb, sizeof(double) * rowsA));
	checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(double) * rowsA));

	/* verify if A has symmetric pattern or not */
	checkCudaErrors(cusolverSpXcsrissymHost(handle, rowsA, nnzA, descrA,
		h_csrRowPtrA, h_csrRowPtrA + 1,
		h_csrColIndA, &issym));

	if (0 == strcmp(opts.testFunc, "chol")) {
		if (!issym) {
			printf("Error: A has no symmetric pattern, please use LU or QR \n");
			exit(EXIT_FAILURE);
		}
	}

	printf("step 2: reorder the matrix A to minimize zero fill-in\n");
	printf(
		"        if the user choose a reordering by -P=symrcm, -P=symamd or "
		"-P=metis\n");

	if (NULL != opts.reorder) {
		if (0 == strcmp(opts.reorder, "symrcm")) {
			printf("step 2.1: Q = symrcm(A) \n");
			checkCudaErrors(cusolverSpXcsrsymrcmHost(
				handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrColIndA, h_Q));
		}
		else if (0 == strcmp(opts.reorder, "symamd")) {
			printf("step 2.1: Q = symamd(A) \n");
			checkCudaErrors(cusolverSpXcsrsymamdHost(
				handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrColIndA, h_Q));
		}
		else if (0 == strcmp(opts.reorder, "metis")) {
			printf("step 2.1: Q = metis(A) \n");
			checkCudaErrors(cusolverSpXcsrmetisndHost(handle, rowsA, nnzA, descrA,
				h_csrRowPtrA, h_csrColIndA,
				NULL, /* default setting. */
				h_Q));
		}
		else {
			fprintf(stderr, "Error: %s is unknown reordering\n", opts.reorder);
			return 1;
		}
	}
	else {
		printf("step 2.1: no reordering is chosen, Q = 0:n-1 \n");
		for (int j = 0; j < rowsA; j++) {
			h_Q[j] = j;
		}
	}

	printf("step 2.2: B = A(Q,Q) \n");

	memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int) * (rowsA + 1));
	memcpy(h_csrColIndB, h_csrColIndA, sizeof(int) * nnzA);

	checkCudaErrors(cusolverSpXcsrperm_bufferSizeHost(
		handle, rowsA, colsA, nnzA, descrA, h_csrRowPtrB, h_csrColIndB, h_Q, h_Q,
		&size_perm));

	if (buffer_cpu) {
		free(buffer_cpu);
	}
	buffer_cpu = (void *)malloc(sizeof(char) * size_perm);
	assert(NULL != buffer_cpu);

	/* h_mapBfromA = Identity */
	for (int j = 0; j < nnzA; j++) {
		h_mapBfromA[j] = j;
	}
	checkCudaErrors(cusolverSpXcsrpermHost(handle, rowsA, colsA, nnzA, descrA,
		h_csrRowPtrB, h_csrColIndB, h_Q, h_Q,
		h_mapBfromA, buffer_cpu));

	/* B = A( mapBfromA ) */
	for (int j = 0; j < nnzA; j++) {
		h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
	}

	printf("step 3: b(j) = 1 + j/n \n");
	for (int row = 0; row < rowsA; row++) {
		h_b[row] = 1.0 + ((double)row) / ((double)rowsA);
	}

	/* h_Qb = b(Q) */
	for (int row = 0; row < rowsA; row++) {
		h_Qb[row] = h_b[h_Q[row]];
	}

	printf("step 4: prepare data on device\n");
	checkCudaErrors(cudaMemcpyAsync(d_csrRowPtrA, h_csrRowPtrA,
		sizeof(int) * (rowsA + 1),
		cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_csrColIndA, h_csrColIndA,
		sizeof(int) * nnzA, cudaMemcpyHostToDevice,
		stream));
	checkCudaErrors(cudaMemcpyAsync(d_csrValA, h_csrValA, sizeof(double) * nnzA,
		cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_csrRowPtrB, h_csrRowPtrB,
		sizeof(int) * (rowsA + 1),
		cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_csrColIndB, h_csrColIndB,
		sizeof(int) * nnzA, cudaMemcpyHostToDevice,
		stream));
	checkCudaErrors(cudaMemcpyAsync(d_csrValB, h_csrValB, sizeof(double) * nnzA,
		cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_b, h_b, sizeof(double) * rowsA,
		cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_Qb, h_Qb, sizeof(double) * rowsA,
		cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_Q, h_Q, sizeof(int) * rowsA,
		cudaMemcpyHostToDevice, stream));

	printf("step 5: solve A*x = b on CPU \n");
	start = second();

	/* solve B*z = Q*b */
	if (0 == strcmp(opts.testFunc, "chol")) {
		checkCudaErrors(cusolverSpDcsrlsvcholHost(
			handle, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
			h_Qb, tol, reorder, h_z, &singularity));
	}
	else if (0 == strcmp(opts.testFunc, "lu")) {
		checkCudaErrors(cusolverSpDcsrlsvluHost(
			handle, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
			h_Qb, tol, reorder, h_z, &singularity));

	}
	else if (0 == strcmp(opts.testFunc, "qr")) {
		checkCudaErrors(cusolverSpDcsrlsvqrHost(
			handle, rowsA, nnzA, descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
			h_Qb, tol, reorder, h_z, &singularity));
	}
	else {
		fprintf(stderr, "Error: %s is unknown function\n", opts.testFunc);
		return 1;
	}

	/* Q*x = z */
	for (int row = 0; row < rowsA; row++) {
		h_x[h_Q[row]] = h_z[row];
	}

	if (0 <= singularity) {
		printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
			singularity, tol);
	}

	stop = second();
	time_solve_cpu = stop - start;

	printf("step 6: evaluate residual r = b - A*x (result on CPU)\n");
	checkCudaErrors(cudaMemcpyAsync(d_r, d_b, sizeof(double) * rowsA,
		cudaMemcpyDeviceToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_x, h_x, sizeof(double) * colsA,
		cudaMemcpyHostToDevice, stream));
	cusparseDcsrmv(cusparseHandle,
		CUSPARSE_OPERATION_NON_TRANSPOSE, rowsA, colsA,
		nnzA, &minus_one, descrA, d_csrValA,
		d_csrRowPtrA, d_csrColIndA, d_x, &one, d_r);
	checkCudaErrors(cudaMemcpyAsync(h_r, d_r, sizeof(double) * rowsA,
		cudaMemcpyDeviceToHost, stream));
	/* wait until h_r is ready */
	checkCudaErrors(cudaDeviceSynchronize());

	b_inf = vec_norminf(rowsA, h_b);
	x_inf = vec_norminf(colsA, h_x);
	r_inf = vec_norminf(rowsA, h_r);
	A_inf = csr_mat_norminf(rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA,
		h_csrColIndA);

	printf("(CPU) |b - A*x| = %E \n", r_inf);
	printf("(CPU) |A| = %E \n", A_inf);
	printf("(CPU) |x| = %E \n", x_inf);
	printf("(CPU) |b| = %E \n", b_inf);
	printf("(CPU) |b - A*x|/(|A|*|x| + |b|) = %E \n",
		r_inf / (A_inf * x_inf + b_inf));

	printf("step 7: solve A*x = b on GPU\n");
	start = second();

	/* solve B*z = Q*b */
	if (0 == strcmp(opts.testFunc, "chol")) {
		checkCudaErrors(cusolverSpDcsrlsvchol(
			handle, rowsA, nnzA, descrA, d_csrValB, d_csrRowPtrB, d_csrColIndB,
			d_Qb, tol, reorder, d_z, &singularity));

	}
	else if (0 == strcmp(opts.testFunc, "lu")) {
		printf("WARNING: no LU available on GPU \n");
	}
	else if (0 == strcmp(opts.testFunc, "qr")) {
		checkCudaErrors(cusolverSpDcsrlsvqr(handle, rowsA, nnzA, descrA, d_csrValB,
			d_csrRowPtrB, d_csrColIndB, d_Qb, tol,
			reorder, d_z, &singularity));
	}
	else {
		fprintf(stderr, "Error: %s is unknow function\n", opts.testFunc);
		return 1;
	}
	checkCudaErrors(cudaDeviceSynchronize());
	if (0 <= singularity) {
		printf("WARNING: the matrix is singular at row %d under tol (%E)\n",
			singularity, tol);
	}
	/* Q*x = z */
	cusparseDsctr(cusparseHandle, rowsA, d_z, d_Q, d_x,
		CUSPARSE_INDEX_BASE_ZERO);
	checkCudaErrors(cudaDeviceSynchronize());

	stop = second();
	time_solve_gpu = stop - start;

	printf("step 8: evaluate residual r = b - A*x (result on GPU)\n");
	checkCudaErrors(cudaMemcpyAsync(d_r, d_b, sizeof(double) * rowsA,
		cudaMemcpyDeviceToDevice, stream));
	cusparseDcsrmv(cusparseHandle,
		CUSPARSE_OPERATION_NON_TRANSPOSE, rowsA, colsA,
		nnzA, &minus_one, descrA, d_csrValA,
		d_csrRowPtrA, d_csrColIndA, d_x, &one, d_r);
	checkCudaErrors(cudaMemcpyAsync(h_x, d_x, sizeof(double) * colsA,
		cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaMemcpyAsync(h_r, d_r, sizeof(double) * rowsA,
		cudaMemcpyDeviceToHost, stream));
	/* wait until h_x and h_r are ready */
	checkCudaErrors(cudaDeviceSynchronize());

	b_inf = vec_norminf(rowsA, h_b);
	x_inf = vec_norminf(colsA, h_x);
	r_inf = vec_norminf(rowsA, h_r);

	if (0 != strcmp(opts.testFunc, "lu")) {
		// only cholesky and qr have GPU version
		printf("(GPU) |b - A*x| = %E \n", r_inf);
		printf("(GPU) |A| = %E \n", A_inf);
		printf("(GPU) |x| = %E \n", x_inf);
		printf("(GPU) |b| = %E \n", b_inf);
		printf("(GPU) |b - A*x|/(|A|*|x| + |b|) = %E \n",
			r_inf / (A_inf * x_inf + b_inf));
	}

	fprintf(stdout, "timing %s: CPU = %10.6f sec , GPU = %10.6f sec\n",
		opts.testFunc, time_solve_cpu, time_solve_gpu);

	if (0 != strcmp(opts.testFunc, "lu")) {
		printf("show last 10 elements of solution vector (GPU) \n");
		printf("consistent result for different reordering and solver \n");
		for (int j = rowsA - 10; j < rowsA; j++) {
			printf("x[%d] = %E\n", j, h_x[j]);
		}
	}

	if (handle) {
		checkCudaErrors(cusolverSpDestroy(handle));
	}
	if (cusparseHandle) {
		cusparseDestroy(cusparseHandle);
	}
	if (stream) {
		checkCudaErrors(cudaStreamDestroy(stream));
	}
	if (descrA) {
		cusparseDestroyMatDescr(descrA);
	}

	if (h_csrValA) {
		free(h_csrValA);
	}
	if (h_csrRowPtrA) {
		free(h_csrRowPtrA);
	}
	if (h_csrColIndA) {
		free(h_csrColIndA);
	}
	if (h_z) {
		free(h_z);
	}
	if (h_x) {
		free(h_x);
	}
	if (h_b) {
		free(h_b);
	}
	if (h_Qb) {
		free(h_Qb);
	}
	if (h_r) {
		free(h_r);
	}

	if (h_Q) {
		free(h_Q);
	}

	if (h_csrRowPtrB) {
		free(h_csrRowPtrB);
	}
	if (h_csrColIndB) {
		free(h_csrColIndB);
	}
	if (h_csrValB) {
		free(h_csrValB);
	}
	if (h_mapBfromA) {
		free(h_mapBfromA);
	}

	if (buffer_cpu) {
		free(buffer_cpu);
	}

	if (d_csrValA) {
		checkCudaErrors(cudaFree(d_csrValA));
	}
	if (d_csrRowPtrA) {
		checkCudaErrors(cudaFree(d_csrRowPtrA));
	}
	if (d_csrColIndA) {
		checkCudaErrors(cudaFree(d_csrColIndA));
	}
	if (d_csrValB) {
		checkCudaErrors(cudaFree(d_csrValB));
	}
	if (d_csrRowPtrB) {
		checkCudaErrors(cudaFree(d_csrRowPtrB));
	}
	if (d_csrColIndB) {
		checkCudaErrors(cudaFree(d_csrColIndB));
	}
	if (d_Q) {
		checkCudaErrors(cudaFree(d_Q));
	}
	if (d_z) {
		checkCudaErrors(cudaFree(d_z));
	}
	if (d_x) {
		checkCudaErrors(cudaFree(d_x));
	}
	if (d_b) {
		checkCudaErrors(cudaFree(d_b));
	}
	if (d_Qb) {
		checkCudaErrors(cudaFree(d_Qb));
	}
	if (d_r) {
		checkCudaErrors(cudaFree(d_r));
	}

	return 0;
}



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