#pragma once
//typedef double item_t;
typedef int item_t;

void mat_mul(item_t** a, item_t** b, item_t** c, size_t length);

void mat_mul_b(item_t** a, item_t** b, item_t** c, size_t length);

void mat_mul_c(item_t** a, item_t** b, item_t** c, size_t length);

void mat_mul_d(item_t** a, item_t** b, item_t** c, size_t length);

void mat_mul_e(item_t** a, item_t** b, item_t** c, size_t length);

void mat_mul_d_parallel(item_t** a, item_t** b, item_t** c, size_t length);

void mat_mul_e_parallel(item_t** a, item_t** b, item_t** c, size_t length);

void test_all_mat_mult(int test_dim = 1000);

