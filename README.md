# OpenCV examples

These are some examples to get yourself familiar with OpenCV. I am by no means an expert. This is heavily based on [this project](https://github.com/ysh329/OpenCL-101).

## Install OpenCV

For installing OpenCV these commands worked on ubuntu 18.10:
```
$ sudo apt-get update
$ sudo apt-get install build-essentials g++ cmake
$ sudo apt-get install ocl-icd-libopencl1
$ sudo apt-get install opencl-headers
$ sudo apt-get install clinfo
$ sudo apt-get install ocl-icd-opencl-dev
$ sudo apt-get install beignet
```

To check whether it was successfull try
```
$ clinfo
```
It should spew out a lot of information about the detected platforms and devices.


## Exercises

### 1. Hello world

compile the code using
```
$ gcc hello.c -lOpenCL -o hello 
```
and running it should produce something like this
```
$ ./hello
platform_id: 1512359136
ret: 0
ret_num_platforms: 1
device_id: 1512376576
ret_number_devices: 1
context @S!Z�
Hello, World!
```

Have a look at the `hello.c` and `hello.cl` source code. What steps are needed to set up and execute a kernel in OpenCl? If you want to know details about a command just google it. For example `opencl clCreateProgramWithSource`. Should get yout [this link](https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clCreateProgramWithSource.html), the official documentation as the first result.


### 2. Vector addition

compile the code using:
```
$ gcc vector-add.c -lOpenCL -lm -o vector-add
```
and running it should produce something like this (for a vector length of 16):
```
$ ./vector-add                              
========== INIT_VALUE ==========
a: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
b: 0 0 1 2 3 2 0 3 0 3 3 1 2 1 1 3 
c: 1 1 2 3 4 3 1 4 1 4 4 2 3 2 2 4 
Time for cpu execution in microseconds: 1 ms 
Source program: __kernel void add_vec_gpu(__global const int *a, __global const int *b, __global int *res, const int len) {
	const int idx = get_global_id(0);
        if (idx < len)
            res[idx] = a[idx] + b[idx];
}


========== CHECK RESULT cpu-verison && gpu-version ==========
d: 1 1 2 3 4 3 1 4 1 4 4 2 3 2 2 4 
correct rate: 16 / 16 , 1.00
len-1=15, d[15]==c[15]: 1, d[15]=4, c[15]=4 
Time for gpu execution in microseconds: 728 ms 

========== CHECK RESULT ELEMENT BY ELEMENT ==========
idx  c  d
 0  1  1 
 1  1  1 
 2  2  2 
 3  3  3 
 4  4  4 
 5  3  3 
 6  1  1 
 7  4  4 
 8  1  1 
 9  4  4 
10  4  4 
11  2  2 
12  3  3 
13  2  2 
14  2  2 
15  4  4 
```

What are the purpose of local and global work size? What are work groups vs work items?

Play around with the numbers for `len` and `local_work_size`, how does it affect the runtime? Try vectors of length `32`, `1024*32` and `1024*1024*32`.

[This paper](https://cims.nyu.edu/~schlacht/OpenCLModel.pdf) gives a decent introduction to the openCL concepts!
