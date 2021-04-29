#include "stdafx.h"
#include <stdio.h>
#include <omp.h>
#include "matmul.h"
#include <stdlib.h>

void mat_mul(item_t** a, item_t** b, item_t** c, size_t length) {
	for (size_t i = 0; i < length; i++)
	{
		for (size_t j = 0; j < length; j++)
		{
			c[i][j] = 0;
			for (size_t k = 0; k < length; k++)
			{
				c[i][j] += a[i][k] * b[k][j];
				// *(*(c + i) + j)
			}
		}
	}
}

void mat_mul_b(item_t** a, item_t** b, item_t** c, size_t length) {
	item_t s;
	for (size_t i = 0; i < length; i++)
	{
		for (size_t j = 0; j < length; j++)
		{
			s = 0;
			for (size_t k = 0; k < length; k++)
			{
				s += a[i][k] * b[k][j];
				// *(*(a + i) + k)
			}
			c[i][j] = s;
		}
	}
}

void mat_mul_c(item_t** a, item_t** b, item_t** c, size_t length) {
	item_t* a_ptr;
	item_t s;
	for (size_t i = 0; i < length; i++)
	{
		for (size_t j = 0; j < length; j++)
		{
			s = 0;
			a_ptr = &a[i][0];
			for (size_t k = 0; k < length; k++)
			{
				s += a_ptr[k] * b[k][j];
			}
			c[i][j] = s;
		}
	}
}

void mat_mul_d(item_t** a, item_t** b, item_t** c, size_t length) {
	register item_t* b_ptr;
	register item_t* c_ptr;
	register item_t a_ik;
	for (size_t i = 0; i < length; i++)
	{
		c_ptr = &c[i][0];
		for (size_t j = 0; j < length; j++)
		{
			c_ptr[j] = 0;
		}
		for (size_t k = 0; k < length; k++)
		{
			b_ptr = &b[k][0];
			a_ik = a[i][k];
			for (size_t j = 0; j < length; j++)
			{
				c_ptr[j] += a_ik * b_ptr[j];
			}
		}
	}
}

void mat_mul_e(item_t** a, item_t** b, item_t** c, size_t length) {
	register item_t* b_ptr;
	register item_t* c_ptr;
	register item_t a_ik;
	for (size_t i = 0; i < length; i++)
	{
		c_ptr = &c[i][0];
		for (size_t j = 0; j < length; j++)
		{
			c_ptr[j] = 0;
		}
		for (size_t k = 0; k < length; k++)
		{
			b_ptr = &b[k][0];
			a_ik = a[i][k];
			c_ptr = &c[i][0];
			for (size_t j = 0; j < length; j += 4)
			{
				*c_ptr += a_ik * (*b_ptr);
				*(c_ptr + 1) += a_ik * *(b_ptr + 1);
				*(c_ptr + 2) += a_ik * *(b_ptr + 2);
				*(c_ptr + 3) += a_ik * *(b_ptr + 3);
				b_ptr += 4;
				c_ptr += 4;
			}
		}
	}
}

void mat_mul_d_parallel(item_t** a, item_t** b, item_t** c, size_t length) {
	register item_t* b_ptr;
	register item_t* c_ptr;
	register item_t a_ik;
#pragma omp parallel for private(a_ik, b_ptr, c_ptr)
	for (int i = 0; i < length; i++)
	{
		c_ptr = &c[i][0];
		for (size_t j = 0; j < length; j++)
		{
			c_ptr[j] = 0;
		}
		for (size_t k = 0; k < length; k++)
		{
			b_ptr = &b[k][0];
			a_ik = a[i][k];
			for (size_t j = 0; j < length; j++)
			{
				c_ptr[j] += a_ik * b_ptr[j];
			}
		}
	}
}

void mat_mul_e_parallel(item_t** a, item_t** b, item_t** c, size_t length) {
	register item_t* b_ptr;
	register item_t* c_ptr;
	register item_t a_ik;
#pragma omp parallel for private(b_ptr, c_ptr, a_ik)
	for (int i = 0; i < length; i++)
	{
		c_ptr = &c[i][0];
		for (size_t j = 0; j < length; j++)
		{
			c_ptr[j] = 0;
		}
		for (size_t k = 0; k < length; k++)
		{
			b_ptr = &b[k][0];
			a_ik = a[i][k];
			c_ptr = &c[i][0];
			for (size_t j = 0; j < length; j += 4)
			{
				*c_ptr += a_ik * (*b_ptr);
				*(c_ptr + 1) += a_ik * *(b_ptr + 1);
				*(c_ptr + 2) += a_ik * *(b_ptr + 2);
				*(c_ptr + 3) += a_ik * *(b_ptr + 3);
				b_ptr += 4;
				c_ptr += 4;
			}
		}
	}
}

void test_all_mat_mult(int test_dim) {
	auto a = new item_t * [test_dim];
	//auto b = new item_t* [test_dim];
	auto c = new item_t * [test_dim];
	for (int i = 0; i < test_dim; i++)
	{
		a[i] = new item_t[test_dim];
		//b[i] = new item_t[test_dim];
		c[i] = new item_t[test_dim];
		for (int j = 0; j < test_dim; j++)
		{
			a[i][j] = (item_t)(rand() % 32767);
		}
	}
	
	printf("Test started...\n");
	auto b = a;
	auto i = rand() % test_dim;
	auto j = rand() % test_dim;

	double start = omp_get_wtime();
	mat_mul(a, b, c, test_dim);
	double duration = omp_get_wtime() - start;
	printf("mat_mul (c[%d][%d] = %d) duration: %.4f\n", i, j, c[i][j], duration);

	start = omp_get_wtime();
	mat_mul_b(a, b, c, test_dim);
	auto duration2 = omp_get_wtime() - start;
	printf("mat_mul_b (c[%d][%d] = %d) duration: %.4f, speedup: %e\n", i, j, c[i][j], duration2, duration / duration2);

	start = omp_get_wtime();
	mat_mul_c(a, b, c, test_dim);
	duration2 = omp_get_wtime() - start;
	printf("mat_mul_c (c[%d][%d] = %d) duration: %.4f, speedup: %e\n", i, j, c[i][j], duration2, duration / duration2);

	start = omp_get_wtime();
	mat_mul_d(a, b, c, test_dim);
	duration2 = omp_get_wtime() - start;
	printf("mat_mul_d (c[%d][%d] = %d) duration: %.4f, speedup: %e\n", i, j, c[i][j], duration2, duration / duration2);

	start = omp_get_wtime();
	mat_mul_e(a, b, c, test_dim);
	duration2 = omp_get_wtime() - start;
	printf("mat_mul_e (c[%d][%d] = %d) duration: %.4f, speedup: %e\n", i, j, c[i][j], duration2, duration / duration2);

	start = omp_get_wtime();
	mat_mul_d_parallel(a, b, c, test_dim);
	duration2 = omp_get_wtime() - start;
	printf("mat_mul_d_parallel (c[%d][%d] = %d) duration: %.4f, speedup: %e\n", i, j, c[i][j], duration2, duration / duration2);

	start = omp_get_wtime();
	mat_mul_e_parallel(a, b, c, test_dim);
	duration2 = omp_get_wtime() - start;
	printf("mat_mul_e_parallel (c[%d][%d] = %d) duration: %.4f, speedup: %e\n", i, j, c[i][j], duration2, duration / duration2);

	//free memory!
	for (int i = 0; i < test_dim; i++)
	{
		delete[] a[i];
		delete[] c[i];
	}
	delete[] a;
	delete[] c;
}