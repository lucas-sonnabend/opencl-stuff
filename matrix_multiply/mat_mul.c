#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>

#include <sys/time.h>
#include <math.h>

#include <CL/cl.h>

// define matrix dimensions here
#define SIZE 32
#define LOCAL_SIZE 16

#define MAX_SOURCE_SIZE (0x1000000)

void multiply_matrices(float *mat_a, float *mat_b, float *mat_result, int row_count_a, int col_count_a, int col_count_b) {
	for (int r_idx = 0; r_idx < row_count_a; r_idx++) {
		for (int c_idx = 0; c_idx < col_count_b; c_idx++) {
			int sum = 0;
			for (int i = 0; i < col_count_a; i++) {
				sum = sum + mat_a[r_idx * col_count_a + i] * mat_b[i * col_count_b + c_idx];
			}
			mat_result[r_idx * col_count_b + c_idx] = sum;
		}
	}
}

void print_matrix(float *matrix, int col_count, int row_count) {
	for (int r_idx = 0; r_idx < row_count; r_idx++) {
		for (int c_idx = 0; c_idx < col_count; c_idx++) {
			printf("%6.2f ", matrix[r_idx * col_count + c_idx]);
		}
		printf("\n");
	}
}

void check_matrices(float *mat_a, float *mat_b, int col_count, int row_count) {
	int index;
	for (int r_idx = 0; r_idx < row_count; r_idx++) {
		for (int c_idx = 0; c_idx < col_count; c_idx++) {
			index = r_idx * col_count + c_idx;
			if (mat_a[index] != mat_b[index]) {
				printf("Matrices are different at [%d][%d]\n", r_idx, c_idx);
				return;
			}
		}
	}
	printf("Matrices are the same!\n");
}

void checkErr(cl_int err, const char *func_name) {
    if(err != CL_SUCCESS)
        {
            printf(">>> [ERROR] %s %d", func_name, err);
            switch(err) {
                case CL_DEVICE_NOT_FOUND :printf("(CL_DEVICE_NOT_FOUND)");break;
                case CL_DEVICE_NOT_AVAILABLE :printf("(CL_DEVICE_NOT_AVAILABLE)");break;
                case CL_COMPILER_NOT_AVAILABLE :printf("(CL_COMPILER_NOT_AVAILABLE)");break;
                case CL_MEM_OBJECT_ALLOCATION_FAILURE :printf("(CL_MEM_OBJECT_ALIOCATION_FAILURE)");break;
                case CL_OUT_OF_RESOURCES :printf("(CL_OUT_OF_RESOURCES)");break;
                case CL_OUT_OF_HOST_MEMORY :printf("(CL_OUT_OF_HOST_MEMORY)");break;
                case CL_MEM_COPY_OVERLAP :printf("(CL_MEM_COPY_OVERLAP)");break;
                case CL_BUILD_PROGRAM_FAILURE:printf("(CL_BUILD_PROGRAM_FAILURE)");break;
                case CL_INVALID_VALUE:printf("(CL_INVALID_VALUE)");break;
                case CL_INVALID_DEVICE_TYPE:printf("(CL_INVALID_DEVICE_TYPE)");break;
                case CL_INVALID_DEVICE:printf("(CL_INVALID_DEVICE)");break;
                case CL_INVALID_CONTEXT:printf("(CL_INVALID_CONTEXT)");break;
                case CL_INVALID_BINARY:printf("(CL_INVALID_BINARY)");break;
                case CL_INVALID_BUILD_OPTIONS:printf("(CL_INVALID_BUILD_OPTIONS)");break;
                case CL_INVALID_PROGRAM:printf("(CL_INVALID_PROGRAM)");break;
                case CL_INVALID_PROGRAM_EXECUTABLE:printf("(CL_INVALID_PROGRAM_EXECUTABLE)");break;
                case CL_INVALID_KERNEL_DEFINITION:printf("(CL_INVALID_KERNEL_DEFINITION)");break;
                case CL_INVALID_KERNEL:printf("(CL_INVALID_KERNEL)");break;
                case CL_INVALID_KERNEL_ARGS:printf("(CL_INVALID_KERNEL_ARGS)");break;
                case CL_INVALID_OPERATION:printf("(CL_INVALID_OPERATION)");break;
                case CL_INVALID_COMMAND_QUEUE:printf("(CL_INVALID_COMMAND_QUEUE)");break;
                case CL_INVALID_WORK_DIMENSION:printf("(CL_INVALID_WORK_DIMENSION)");break;
                case CL_INVALID_WORK_GROUP_SIZE:printf("(CL_INVALID_WORK_GROUP_SIZE)");break;
                case CL_INVALID_WORK_ITEM_SIZE:printf("(CL_INVALID_WORK_ITEM_SIZE)");break;
                case CL_INVALID_GLOBAL_WORK_SIZE:printf("(CL_INVALID_GLOBAL_WORK_SIZE)");break;
                case CL_INVALID_GLOBAL_OFFSET:printf("(CL_INVALID_GLOBAL_OFFSET)");break;
                case CL_INVALID_IMAGE_SIZE:printf("(CL_INVALID_IMAGE_SIZE)");break;
                case CL_INVALID_EVENT_WAIT_LIST:printf("(CL_INVALID_EVENT_WAIT_LIST)");break;
                case CL_MISALIGNED_SUB_BUFFER_OFFSET:printf("(CL_MISALIGNED_SUB_BUFFER_OFFSET)");break;

            default: break;
        }
        printf("\n");
    }
}

size_t read_source_program(char *source_str, const char file_name[]) {
        FILE *fp;
        size_t source_size;

        fp = fopen(file_name, "r");
        if (!fp) {
                fprintf(stderr, "Failed to laod kernel. \n");
                exit(1);
        }
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);
	return source_size;
}

#define OCL_ERROR_LEN (1024)

int init_opencl(cl_context *context, cl_command_queue *queue, cl_program *program, const char *inputfile) {
	char BufferError[OCL_ERROR_LEN];
	cl_int ret_code;
	cl_platform_id platform;
	cl_uint num_platforms;
	cl_device_id *device_ids;
	cl_uint num_devices;

	printf("initialise opencl stuff\n");

	ret_code = clGetPlatformIDs(1, &platform, &num_platforms);
	checkErr(ret_code, "clGetPlatformIds()");

	// step one get the number of devices
	ret_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	checkErr(ret_code, "clGetDeviceIDs()");
	printf("got %d devices:\n", (int)num_devices);
	// get all the device ids
	device_ids = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
	ret_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, device_ids, NULL);
	checkErr(ret_code, "clGetDeviceIDs()");

	*context = clCreateContext(NULL, 1, device_ids, NULL, NULL, &ret_code);
	checkErr(ret_code, "clCreateContext()");

	*queue = clCreateCommandQueue(*context, *device_ids, CL_QUEUE_PROFILING_ENABLE, &ret_code);
	checkErr(ret_code, "clCreateCommandQueue()");

	char *source_str = (char *) malloc(MAX_SOURCE_SIZE);
	size_t source_size;
	source_size = read_source_program(source_str, inputfile);
	printf("source program: %s\n", source_str);
	
	*program = clCreateProgramWithSource(*context, 1, (const char**) &source_str, (const size_t *)&source_size, &ret_code);
	checkErr(ret_code, "clCreateProgramWithSource");

	ret_code = clBuildProgram(*program, 1, device_ids, NULL, NULL, NULL);
	if(ret_code != CL_SUCCESS) {
		printf("[ERROR] failed to build program: %d\n", (int)ret_code);
		clGetProgramBuildInfo(*program, *device_ids, CL_PROGRAM_BUILD_LOG, sizeof(BufferError), BufferError, NULL);
		BufferError[OCL_ERROR_LEN] = '\0';
		printf("[ERROR] kernel build log: %s\n", BufferError);
		return -1;
	}
	return 0;
}

int main(int argc, char *argv[]) {
	int matrix_mem_size = SIZE * SIZE * sizeof(float);
	int ret_code = 0;
	float *matrix_a = (float *) malloc(matrix_mem_size);
	float *matrix_b = (float *) malloc(matrix_mem_size);
	float *matrix_res = (float *) malloc(matrix_mem_size);
	float *matrix_res2 = (float *) malloc(matrix_mem_size);

	struct timeval start, finish;
	double duration;

	// initialise the matrices
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			
			matrix_a[i * SIZE + j] = rand() % 4;
			matrix_b[i * SIZE + j] = rand() % 4;  // alternatively i == j ? 1 : 0
			matrix_res[i * SIZE + j] = 0;
		}
	}
	printf("matrix a: \n");
	//print_matrix(matrix_a, SIZE, SIZE);
	printf("matrix b: \n");
	//print_matrix(matrix_b, SIZE, SIZE);


	// multiplying the matrices on the cpu:
	gettimeofday(&start, NULL);
	multiply_matrices(matrix_a, matrix_b, matrix_res, SIZE, SIZE, SIZE);
	gettimeofday(&finish, NULL);

	printf("The time for cpu execution in milliseconds: %ld ms\n",  (finish.tv_sec - start.tv_sec)*1000000L + finish.tv_usec - start.tv_usec);

	
	//printf("resulting matrix: \n");
	//print_matrix(matrix_res, SIZE, SIZE);

	// do the same on the GPU
	//
	cl_kernel mat_mul_kernel = NULL;
	cl_int status = 0;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_program program = NULL;
	cl_event event = NULL;

	const char *inputfile = "mat_mul.cl";
	init_opencl(&context, &command_queue, &program, inputfile);

	// is it always true that a C array has the same size as a cl array (of floats, for example)??
	cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_mem_size, NULL, &status);
	cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_mem_size, NULL, &status);
	cl_mem res_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_mem_size, NULL, &status);
	ret_code = clEnqueueWriteBuffer(command_queue, a_buffer, CL_TRUE, 0, matrix_mem_size, (void *) matrix_a, 0, NULL, NULL);
	ret_code |= clEnqueueWriteBuffer(command_queue, b_buffer, CL_TRUE, 0, matrix_mem_size, (void *) matrix_b, 0, NULL, NULL);
        if (ret_code != CL_SUCCESS) {
		checkErr(ret_code, "clEnqueueWriteBuffer()");
		return -1;
	}

	mat_mul_kernel = clCreateKernel(program, "mat_mul_naive", &ret_code);
	checkErr(ret_code, "clCreateKernel() for malt_mul_kernel");
	int size = SIZE;
	ret_code = clSetKernelArg(mat_mul_kernel, 0, sizeof(cl_mem), (void *) &a_buffer);
	ret_code |= clSetKernelArg(mat_mul_kernel, 1, sizeof(cl_mem), (void *) &b_buffer);
	ret_code |= clSetKernelArg(mat_mul_kernel, 2, sizeof(cl_mem), (void *) &res_buffer);
	ret_code |= clSetKernelArg(mat_mul_kernel, 3, sizeof(cl_int), (void *) &size);
	ret_code |= clSetKernelArg(mat_mul_kernel, 4, sizeof(cl_int), (void *) &size);
	ret_code |= clSetKernelArg(mat_mul_kernel, 5, sizeof(cl_int), (void *) &size);
	checkErr(status, "clSetKernelArg() for mat_mul_kernel");

	// TODO: play around with the global work size and work size groups
	// for a matrix you should only set the first two dimension
	size_t global_work_size[3] = {SIZE, SIZE, 1};
	size_t local_work_size[3] = {LOCAL_SIZE, LOCAL_SIZE, 1};
	size_t global_size = global_work_size[0] * global_work_size[1] * global_work_size[2];
	int task_size = SIZE * SIZE;
	if (global_size < task_size) {
		printf("[WARN] global work size (%ld) is smaller than task size (%d)\n", global_size, task_size);
	}
	if (global_work_size[0] % local_work_size[0] != 0) {
		printf("[WARN] global work size not a multiple of local work size!\n");
	}

	gettimeofday(&start, NULL);
	clEnqueueNDRangeKernel(command_queue, mat_mul_kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &event);
	clFinish(command_queue);
	gettimeofday(&finish, NULL);
	printf("The time for gpu execution in milliseconds: %ld ms\n",  (finish.tv_sec - start.tv_sec)*1000000L + finish.tv_usec - start.tv_usec);

	ret_code = clEnqueueReadBuffer(command_queue, res_buffer, CL_TRUE, 0, matrix_mem_size, (void *) matrix_res2, 0, NULL, NULL);
	checkErr(ret_code, "clEnqueueReadBuffer()");
	clFinish(command_queue);

	printf("result from GPU:\n");
	//print_matrix(matrix_res2, SIZE, SIZE);
	printf("compare results:\n");
	check_matrices(matrix_res, matrix_res2, SIZE, SIZE);

	ret_code = clReleaseCommandQueue(command_queue);
	ret_code = clReleaseProgram(program);
	ret_code = clReleaseContext(context);
	ret_code = clReleaseKernel(mat_mul_kernel);
	ret_code = clReleaseMemObject(a_buffer);
	ret_code = clReleaseMemObject(b_buffer);
	ret_code = clReleaseMemObject(res_buffer);

	return 0;
}

