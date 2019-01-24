__kernel void mat_mul_naive(__global const float *mat_a, __global const float *mat_b, __global float *mat_res, const int row_count_a, const int col_count_a, const int col_count_b) {
	const int col = get_global_id(0);
        const int row = get_global_id(1);
        float res = 0;
        for (int i = 0; i < col_count_a; i++) {
  		res += mat_a[row * col_count_a + i] * mat_b[i * col_count_b + col];
	}
	mat_res[row * col_count_b + col] = res;
}
