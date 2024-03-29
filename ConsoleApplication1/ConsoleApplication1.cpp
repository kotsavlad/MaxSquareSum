// Example of the application of OpenMP
// Search of (square_size x square_size)-window in the given matrix
// with the maximal sum of the absolute values of its elements.
// All functions returns the sought maximal sum.
// Functions with "parallel" suffix uses omp directives.

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <deque>

using namespace std;

typedef double item_t;
//typedef int item_t;


item_t max_square_sum(item_t** matrix, int m, int n, int square_size) {
	item_t max_sum = -1;
	for (int i = 0; i <= m - square_size; i++)
	{
		for (int j = 0; j <= n - square_size; j++)
		{
			item_t square_sum = 0;
			for (int k = i; k < i + square_size; k++)
			{
				for (int l = j; l < j + square_size; l++)
				{
					if (matrix[k][l] >= 0)
						square_sum += matrix[k][l];
					else
						square_sum -= matrix[k][l];
				}
			}
			if (square_sum > max_sum)
				max_sum = square_sum;
		}
	}
	return max_sum;
}

item_t max_square_sum_b(item_t** matrix, int m, int n, int square_size) {
	item_t max_sum = -1;
	item_t* data_ptr;
	for (int i = 0; i <= m - square_size; i++)
	{
		for (int j = 0; j <= n - square_size; j++)
		{
			item_t square_sum = 0;
			for (int k = 0; k < square_size; k++)
			{
				data_ptr = &matrix[i + k][j];
				for (int l = 0; l < square_size; l++)
				{
					if (*data_ptr >= 0)
						square_sum += *data_ptr;
					else
						square_sum -= *data_ptr;
					data_ptr++;
				}
			}
			if (square_sum > max_sum)
				max_sum = square_sum;
		}
	}
	return max_sum;
}

item_t max_square_sum2(item_t** matrix, int m, int n, int square_size) {
	item_t max_sum = -1;
	deque<item_t> sums;
	for (int i = 0; i <= m - square_size; i++)
	{
		sums.clear();
		auto square_sum = 0;
		auto bottom_bound = i + square_size;
		for (int j = 0; j < square_size; j++)
		{
			size_t col_sum = 0;
			for (int k = i; k < bottom_bound; k++)
			{
				auto element = matrix[k][j];
				if (element > 0)
					col_sum += element;
				else
					col_sum -= element;;
			}
			square_sum += col_sum;
			sums.push_back(col_sum);
		}
		if (square_sum > max_sum)
			max_sum = square_size;

		for (int j = square_size; j < n; j++)
		{
			item_t col_sum = 0;
			for (int k = i; k < bottom_bound; k++)
			{
				auto element = matrix[k][j];
				if (element > 0)
					col_sum += element;
				else
					col_sum -= element;;
			}
			square_sum += col_sum - sums.front();
			if (square_sum > max_sum)
				max_sum = square_sum;
			sums.pop_front();
			sums.push_back(col_sum);
		}
	}
	return max_sum;
}

item_t max_square_sum2_b(item_t** matrix, int m, int n, int square_size) {
	item_t max_sum = -1;
	item_t* sums = new item_t[square_size];
	for (int i = 0; i <= m - square_size; i++)
	{
		//sums.clear();
		for (int k = 0; k < square_size; k++)
		{
			sums[k] = 0;
		}
		auto square_sum = 0;
		auto bottom_bound = i + square_size;
		for (int j = 0; j < square_size; j++)
		{
			size_t col_sum = 0;
			for (int k = i; k < bottom_bound; k++)
			{
				auto element = matrix[k][j];
				if (element > 0)
					col_sum += element;
				else
					col_sum -= element;;
			}
			square_sum += col_sum;
			sums[j] = col_sum;
		}
		if (square_sum > max_sum)
			max_sum = square_size;

		for (int j = square_size; j < n; j++)
		{
			item_t col_sum = 0;
			for (int k = i; k < bottom_bound; k++)
			{
				auto element = matrix[k][j];
				if (element > 0)
					col_sum += element;
				else
					col_sum -= element;;
			}
			square_sum += col_sum - sums[j % square_size];
			if (square_sum > max_sum)
				max_sum = square_sum;
			sums[j % square_size] = col_sum;
		}
	}
	delete[] sums;
	return max_sum;
}

item_t max_square_sum_parallel(item_t** matrix, int m, int n, int square_size, int thread_count = 4) {
	item_t max_sum = -1;
	item_t square_sum;
#pragma omp parallel for num_threads(thread_count) private(square_sum)
	for (int i = 0; i <= m - square_size; i++)
	{
		for (int j = 0; j <= n - square_size; j++)
		{
			square_sum = 0;
			for (int k = i; k < i + square_size; k++)
			{
				for (int l = j; l < j + square_size; l++)
				{
					if (matrix[k][l] >= 0)
						square_sum += matrix[k][l];
					else
						square_sum -= matrix[k][l];
				}
			}
			if (square_sum > max_sum)
#pragma omp critical
			{
				if (square_sum > max_sum)
					max_sum = square_sum;
			}
		}
	}
	return max_sum;
}

item_t max_square_sum_parallel_b(item_t** matrix, int m, int n, int square_size, int thread_count = 4) {
	item_t square_sum;
	auto max_sums = new item_t[thread_count];
	item_t max_sum;
#pragma omp parallel num_threads(thread_count) private(square_sum, max_sum)
	{
		int id = omp_get_thread_num();
		max_sum = -1;
#pragma omp for
		for (int i = 0; i <= m - square_size; i++)
		{
			for (int j = 0; j <= n - square_size; j++)
			{
				square_sum = 0;
				for (int k = i; k < i + square_size; k++)
				{
					for (int l = j; l < j + square_size; l++)
					{
						if (matrix[k][l] >= 0)
							square_sum += matrix[k][l];
						else
							square_sum -= matrix[k][l];
					}
				}
				if (square_sum > max_sum)
					max_sum = square_sum;
			}
		}
		max_sums[id] = max_sum;
	}
	max_sum = *std::max_element(max_sums, max_sums + thread_count);
	delete[] max_sums;
	return max_sum;
}

item_t max_square_sum2_b_parallel(item_t** matrix, int m, int n, int square_size, int thread_count = 4) {
	item_t* max_sums = new item_t[thread_count];
	item_t* sums = new item_t[square_size];
	item_t max_sum;
#pragma omp parallel num_threads(thread_count) private(max_sum)
	{
		int id = omp_get_thread_num();
		item_t* sums = new item_t[square_size];
		max_sum = -1;
#pragma omp for
		for (int i = 0; i <= m - square_size; i++)
		{
			for (int k = 0; k < square_size; k++)
			{
				sums[k] = 0;
			}
			auto square_sum = 0;
			auto bottom_bound = i + square_size;
			for (int j = 0; j < square_size; j++)
			{
				size_t col_sum = 0;
				for (int k = i; k < bottom_bound; k++)
				{
					auto element = matrix[k][j];
					if (element > 0)
						col_sum += element;
					else
						col_sum -= element;;
				}
				square_sum += col_sum;
				sums[j] = col_sum;
			}
			if (square_sum > max_sum)
				max_sum = square_size;

			for (int j = square_size; j < n; j++)
			{
				item_t col_sum = 0;
				for (int k = i; k < bottom_bound; k++)
				{
					auto element = matrix[k][j];
					if (element > 0)
						col_sum += element;
					else
						col_sum -= element;;
				}
				square_sum += col_sum - sums[j % square_size];
				if (square_sum > max_sum)
					max_sum = square_sum;
				sums[j % square_size] = col_sum;
			}
		}
		max_sums[id] = max_sum;
		delete[] sums;
	}
	max_sum = *std::max_element(max_sums, max_sums + thread_count);
	delete[] max_sums;
	return max_sum;
}


int main()
{
	const int MAX_DIM = 1000;
	int square_size = 60;
	auto matrix = new item_t * [MAX_DIM];
	for (int i = 0; i < MAX_DIM; i++)
	{
		matrix[i] = new item_t[MAX_DIM];
		for (int j = 0; j < MAX_DIM; j++)
		{
			matrix[i][j] = (item_t)(rand() % 32767);
			//matrix[i][j] = (item_t)(i + j + 1);
		}
	}

	//auto test_dim = MAX_DIM;
	//auto start = omp_get_wtime();
	//auto res = max_square_sum_b(matrix, test_dim, test_dim, square_size);
	//auto duration = omp_get_wtime() - start;
	//printf("max_square_sum_b result: %e, duration: %e\n", res, duration);

	//start = omp_get_wtime();
	//res = max_square_sum2(matrix, test_dim, test_dim, square_size);
	//auto duration2 = omp_get_wtime() - start;
	//printf("max_square_sum2 result: %e, duration: %e, speedup: %.4f\n", res, duration2, duration / duration2);

	//start = omp_get_wtime();
	//res = max_square_sum2_b(matrix, test_dim, test_dim, square_size);
	//duration2 = omp_get_wtime() - start;
	//printf("max_square_sum2_b result: %e, duration: %e, speedup: %.4f\n", res, duration2, duration / duration2);

	//start = omp_get_wtime();
	//res = max_square_sum_parallel(matrix, test_dim, test_dim, square_size);
	//duration2 = omp_get_wtime() - start;
	//printf("max_square_sum_parallel result: %e, duration: %e, speedup: %.4f\n", res, duration2, duration / duration2);

	//start = omp_get_wtime();
	//res = max_square_sum_parallel_b(matrix, test_dim, test_dim, square_size);
	//duration2 = omp_get_wtime() - start;
	//printf("max_square_sum_parallel2 result: %e, duration: %e, speedup: %.4f\n", res, duration2, duration / duration2);

	//start = omp_get_wtime();
	//res = max_square_sum2_b_parallel(matrix, test_dim, test_dim, square_size);
	//duration2 = omp_get_wtime() - start;
	//printf("max_square_sum2_b_parallel result: %e, duration: %e, speedup: %.4f\n", res, duration2, duration / duration2);

	auto dims = { 100, 200, 300, 400, 500, MAX_DIM };
	auto nums_thread = { 2, 4, 6, 8, 10, 20 };
	fstream fs = fstream("d:/result.txt", ios::out);
	for (int m : dims) {
		auto start = omp_get_wtime();
		auto res = max_square_sum(matrix, m, m, square_size);
		auto duration = omp_get_wtime() - start;
		cout << "Dimension: " << m << endl;
		printf("Result in serial mode: %e, duration: %e\n", res, duration);

		for (int t : nums_thread) {
			start = omp_get_wtime();
			res = max_square_sum_parallel_b(matrix, m, m, square_size);
			auto duration2 = omp_get_wtime() - start;
			auto speedup = duration / duration2;
			printf("Result in parallel mode for %d threads: %e, duration: %e, speedup: %.4f\n", t, res, duration2, speedup);
			fs << speedup << '\t';
		}
		fs << '\n';
	}
	fs.close();

	// Memory cleanup!
	for (size_t i = 0; i < MAX_DIM; i++)
		delete[] matrix[i];
	delete[] matrix;
}
