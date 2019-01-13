#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

int main() {
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem memobj = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;

	char result_string[MEM_SIZE];

	FILE *fp;
	char filename[] = "./hello.cl";
	char *source_str;
	size_t source_size;

        // load the source code containing the kernel
	fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel. \n");
		exit(1);
	}
	source_str = (char *) malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	printf("source_size: %d\n", (int)source_size);
	pclose(fp);

	// Get Platform and Device info
        ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
	printf("platform_id: %d\n", (int)platform_id);
	printf("ret: %d\n", (int)ret);
	printf("ret_num_platforms: %d\n", (int)ret_num_platforms);
	printf("device_id: %d\n", (int)device_id);
	printf("ret_number_devices: %d\n", (int)ret_num_devices);

	// create openCl context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	printf("context %s\n", context);

	// created command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(char), NULL, &ret);
	// create kernel program from source
	program = clCreateProgramWithSource(
		context, 1, (const char **)&source_str,
		(const size_t *)&source_size, &ret
	);
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        // create kernel
	kernel = clCreateKernel(program, "hello", &ret);
        // set args
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);
	// execute kernel
	ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);

	// copy result
	ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, 
			MEM_SIZE * sizeof(char), result_string, 0, NULL, NULL);
        puts(result_string);

        ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(memobj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	return 0;
}
