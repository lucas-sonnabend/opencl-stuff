#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

#include <time.h>
#include <sys/time.h>

/* gpu */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define 	MEM_SIZE                (128)
#define 	MAX_SOURCE_SIZE 	(0x100000)
#define         PRINT_LINE(title)       printf("\n========== %s ==========\n", title);


void init_vec(int *vec, int len, int set_one_flag) {
    for (int i = 0; i < len; i++) {
		if (set_one_flag)
			vec[i] = 1;
		else
			vec[i] = 0;
	}
}

void rand_vec(int *vec, int len) {
	srand( (unsigned) time(0) );
	for (int i = 0; i < len; i++) {
		vec[i] = rand() % 4;
	}
}

void add_vec_cpu(const int *a, const int *b, int *res, const int len) {
	for (int i = 0; i < len; i++) {
		res[i] = a[i] + b[i];
	}
}

void print_vec(int *vec, int len) {
	for (int i = 0; i < len; i++) {
		printf("%d ", vec[i]);
	}
	printf("\n");
}

void check_result(int *v1, int *v2, int len) {
    int correct_num = 0;
	for (int i = 0; i < len; i++) {
		if (v1[i] == v2[i]) {
			correct_num += 1;
		}
	}
	printf("correct rate: %d / %d , %1.2f\n", correct_num, len, (float)correct_num/len);
}

size_t read_source_program(char *source_str) {
        FILE *fp;
        char fileName[] = "./vector-add.cl";
        size_t source_size;

        fp = fopen(fileName, "r");
        if (!fp) {
                fprintf(stderr, "Failed to laod kernel. \n");
                exit(1);
        }
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);
	return source_size;
}


int main(void) {
	struct timeval start, finish;
	double duration;
	srand((unsigned) time(NULL));

	// generate two vectors a, b, 
	// and c (result of cpu add) and d (result of gpu add)
	int len = 16;
	size_t data_size = len * sizeof(int);
	int *a = (int *) malloc(data_size);
	int *b = (int *) malloc(data_size);
	int *c = (int *) malloc(data_size);
	int *d = (int *) malloc(data_size);

	PRINT_LINE("INIT_VALUE");
	printf("a: ");
	init_vec(a, len, 1);
	print_vec(a, len);

	printf("b: ");
	rand_vec(b, len);
	print_vec(b, len);

	printf("c: ");
	init_vec(c, len, 0);
	gettimeofday(&start, NULL);
	add_vec_cpu(a, b, c, len);
	gettimeofday(&finish, NULL);
	print_vec(c, len);

	printf("Time for cpu execution in microseconds: %ld ms \n", (finish.tv_sec - start.tv_sec)*1000000L + finish.tv_usec - start.tv_usec);
	
	// vector addition on cpu
	cl_mem a_buff, b_buff, d_buff;

	cl_platform_id platform_id = NULL;
	cl_uint ret_num_platforms;
	cl_device_id device_id = NULL;
	cl_uint re_num_devices;
	cl_context context = NULL;
	cl_kernel kernel = NULL;
	cl_program program = NULL;

	cl_command_queue command_queue = NULL;
	cl_int ret;

	// load source code containing the kernel
	char result_string[MEM_SIZE];
	char *source_str;
        source_str = (char *) malloc(MAX_SOURCE_SIZE);
	size_t source_size;
	source_size = read_source_program(source_str);
	printf("Source program: %s\n", source_str);

	// get platform
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	if (ret != CL_SUCCESS) {
	    printf("Failed to get platform ID.\n");
		goto error;
	}
	// get Device
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if (ret != CL_SUCCESS) {
	    printf("Failed to get device ID.\n");
		goto error;
	}
	// create Context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);//&ret);
	if (ret != CL_SUCCESS) {
		printf("Failed to create OpenCL context.\n");
	    goto error;
	}
	// create command Q
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	if (ret != CL_SUCCESS) {
	   printf("Failed to create command queue %d\n", (int) ret);
	   goto error;
	}

	// memory buffer to hold the vectors
	a_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &ret);
	b_buff = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size, NULL, &ret);
	d_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size, NULL, &ret);

	ret = clEnqueueWriteBuffer(command_queue, a_buff, CL_TRUE, 0, data_size, (void *)a, 0, NULL, NULL);

	ret |= clEnqueueWriteBuffer(command_queue, b_buff, CL_TRUE, 0, data_size, (void *)b, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Failed to cpy data from host to device %d\n", (int) ret);
		goto error;
	}
	// create Kernel program from source
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	if (ret != CL_SUCCESS) {
		printf("Failed to create OpenCL program from source %d\n", (int) ret);
		if (ret == CL_INVALID_CONTEXT) {
			printf("invalid context!\n");
		} else if (ret == CL_INVALID_VALUE) {
			printf("invalid value\n");
		} else if (ret == CL_OUT_OF_HOST_MEMORY) {
			printf("out of host memory\n");
		} else {
			printf("unknown return value\n");
		}
		
		goto error;
	}
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Failed to build program %d\n", (int) ret);

		char build_log[16348];
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
		printf("Error in kernel: %s\n", build_log);
		goto error;
	}

	// create kernel
	kernel = clCreateKernel(program, "add_vec_gpu", &ret);
	if (ret != CL_SUCCESS) {
		printf("Failed to create kernel %d\n", (int) ret);
		goto error;
	}

	ret  = clSetKernelArg(kernel, 0, sizeof (cl_mem), (void *) &a_buff);
	ret |= clSetKernelArg(kernel, 1, sizeof (cl_mem), (void *) &b_buff);
	ret |= clSetKernelArg(kernel, 2, sizeof (cl_mem), (void *) &d_buff);
	ret |= clSetKernelArg(kernel, 3, sizeof (cl_int), (void *) &len);
	if (ret != CL_SUCCESS) {
		printf("Failed to set kernel arguments %d\n", (int) ret);
		goto error;
	}
	// execute the kernel
	size_t global_work_size, local_work_size;
	// number of work items in each local work group
	// TODO: experiment with different local work_sizes
	local_work_size = 256;
	// number of total work items
	// global work size has to be a multiple of local_work_size
	global_work_size = (size_t) ceil( len / (float) local_work_size) * local_work_size;

	init_vec(d, len, 0);
	gettimeofday(&start, NULL);	
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("Failed to execute kernel for execution %d\n", (int) ret);
		goto error;
	}

	ret = clEnqueueReadBuffer(command_queue, d_buff, CL_TRUE, 0, data_size, (void *)d, 0, NULL, NULL);
	gettimeofday(&finish, NULL);
	if (ret != CL_SUCCESS) {
		printf("Failed to copy data from device to host %d\n", (int) ret);
		goto error;
	}

        PRINT_LINE("CHECK RESULT cpu-verison && gpu-version");
	printf("d: ");
	print_vec(d, len);
	check_result(c, d, len);
	printf("len-1=%d, d[%d]==c[%d]: %d, d[%d]=%d, c[%d]=%d \n", len-1, len-1, len-1, d[len-1]==c[len-1], len-1, d[len-1], len-1, c[len-1]);

        
	printf("Time for gpu execution in microseconds: %ld ms \n", (finish.tv_sec - start.tv_sec)*1000000L + finish.tv_usec - start.tv_usec);
	
         PRINT_LINE("CHECK RESULT ELEMENT BY ELEMENT");
	 printf("idx  c  d\n");
	 for(int i = 0; i < len; i++) {
	 	printf("%2d %2d %2d \n", i, c[i], d[i]);
	 }
	

// Finalisation (TODO: don't use a todo for this!)
error:
	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	clReleaseMemObject(a_buff);
	clReleaseMemObject(b_buff);
	clReleaseMemObject(d_buff);

	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	free(source_str);
	free(a);
	free(b);
	free(c);
	free(d);

	return 0;
}

