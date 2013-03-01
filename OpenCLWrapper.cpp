#include "OpenCLWrapper.h"
#include <array>
#include <cstdio>
#include <fstream>
#ifdef MATLAB_MEX_FILE
#define __STDC_UTF_16__ 1
#include "mex.h"
#endif
using namespace std;

template<typename... Args>
void printErr(const char *fmt, Args... args){
#ifndef MATLAB_MEX_FILE
	printf(fmt, args...);
#else
	char x[10000];
	sprintf(x, fmt, args...);
	mexErrMsgIdAndTxt("Matlab:OpenCLWrapper", x);
#endif
}

Device::Device(cl_device_id devId):deviceId(devId){
	cl_device_type type;
	cl_uint tmpClUInt;
	cl_ulong tmpClUlong;
	std::array<char,2000> tmpCharArray;
	clGetDeviceInfo(devId, CL_DEVICE_TYPE, sizeof(type),&type,nullptr);
	deviceType = type;

	clGetDeviceInfo(devId, CL_DEVICE_VENDOR, tmpCharArray.size(),tmpCharArray.data(),nullptr);
	vendor = tmpCharArray.data();

	clGetDeviceInfo(devId, CL_DEVICE_NAME, tmpCharArray.size(),tmpCharArray.data(),nullptr);
	deviceName = tmpCharArray.data();

	clGetDeviceInfo(devId, CL_DRIVER_VERSION, tmpCharArray.size(),tmpCharArray.data(),nullptr);
	driverVersion = tmpCharArray.data();

	clGetDeviceInfo(devId, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(tmpClUInt),&tmpClUInt,nullptr);
	maxComputeUnits = tmpClUInt;

	clGetDeviceInfo(devId, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(tmpClUInt),&tmpClUInt,nullptr);
	maxWorkItemsDimensions = tmpClUInt;

	maxWorkItemsSizes.reset(new size_t[maxWorkItemsDimensions],default_delete<size_t[]>());
	// new (&maxWorkItemsDimensions) shared_ptr<size_t>(new size_t[maxWorkItemsSizes],default_delete<size_t[]>()); //a7a

	clGetDeviceInfo(devId, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*maxWorkItemsDimensions,maxWorkItemsSizes.get(),nullptr);
	// for(auto i = maxWorkItemsDimensions.get(); i < maWxorkItemsDimensions.get() + maxWorkItemsDimensions; i ++)
	// 	cout<<*i<<endl;

	clGetDeviceInfo(devId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(tmpClUlong),&tmpClUlong,nullptr);
	globalMemorySize = tmpClUlong;
}

void Device::releaseAll(){
	// clReleaseDevice(deviceId);
}

bool debug=true;

vector<Device *> OpenCLQuery(cl_device_type deviceType){
	cl_uint numDevices;
	cl_int err;
	cl_uint numPlatt;
	cl_platform_id platform;
	cl_device_id *dev;

	err = clGetPlatformIDs(1,&platform,&numPlatt);
	if(debug and CL_SUCCESS != err) printErr("%d platform detection failed", err);

	err = clGetDeviceIDs(platform,deviceType,0,nullptr,&numDevices);
	if(debug and CL_SUCCESS != err) printErr("%d Device detection failed", err);

	dev = new cl_device_id[numDevices];

	err = clGetDeviceIDs(platform,deviceType,numDevices,dev,nullptr);
	if(debug and CL_SUCCESS != err) printErr("%d Device detection failed", err);

	vector<Device *> v;
	
	if(CL_SUCCESS==err)
		for(cl_uint i=0;i<numDevices;i++)
			v.push_back(new Device(dev[i]));

	return v;
}

Environment::Environment(Device dev):device(dev){
	cl_int err;
	context = clCreateContext(nullptr,1,&dev.deviceId,nullptr,nullptr,&err);
	if(debug && err){ printErr("%d context creation failed",err); }

	queue = clCreateCommandQueue(context, device.deviceId, 0, &err);
	if(debug && err){ printErr("%d queue creation failed",err); }
}

bool Environment::setKernel(std::string kernelSrc, std::string kernelName){
	cl_int err;
	const char *src = kernelSrc.c_str();
	
	program = clCreateProgramWithSource(context, 1, const_cast<const char**> (&src), nullptr, &err);
	if(debug && err)
	{ 
		printErr("%d program creation failed",err);
		return false;
	}

	err = clBuildProgram(program, 1, &(device.deviceId), nullptr, nullptr, nullptr);
	if(debug && err)
	{
		size_t s = 2000;
		char *tmp = new char[s];
		clGetProgramBuildInfo(program, device.deviceId, CL_PROGRAM_BUILD_LOG, s, tmp, &s);
		printErr("program build failed\n Build Log:\n %s", tmp);
		delete[] tmp;
		return false;
	}

	kernel = clCreateKernel(program, kernelName.c_str(), &err);
	if(debug && err) { printErr("%d kernel creation failed",err); return false; }

	return true;
}

void Environment::releaseAll(){
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context)
;}

kernelArgument::kernelArgument(Environment e, void *d, size_t s, cl_mem_flags memflags):size(s),memoryFlags(memflags),env(e),scalar(false){
	data = d;
	cl_int err;
	memory = clCreateBuffer(e.context, memflags, s, d, &err);
	if(debug && err){	printErr("%d kernelArgument creation failed", err);	}
}

void kernelArgument::readBufferFromDevice(void *&d, bool allocate = true){
	if(allocate)
		d = malloc(size);

	cl_int err = clEnqueueReadBuffer(env.queue, memory, CL_TRUE, 0, size, d, 0, nullptr, nullptr);
	if(debug && err){ printErr("%d kernelArgument read failed", err);	}
}

void kernelArgument::releaseAll(){
	clReleaseMemObject(memory);
}

void Run(cl_uint threadDimensionsCount, size_t *threadDimensions, std::vector<kernelArgument> inputs){
	cl_uint i=0;
	cl_int err=0;
	for(kernelArgument b : inputs){
		if(b.scalar)
			err = clSetKernelArg(b.env.kernel, i++, b.size, b.data);
		else
			err = clSetKernelArg(b.env.kernel, i++, sizeof(b.memory), &b.memory);
		
		if(debug && err) { 
			printErr((b.scalar?"%d Couldn't set scalar as kernel argument":"%d Couldn't set buffer as kernel argument"), err);
			 return;
		}
	}
	
	err = clEnqueueNDRangeKernel(inputs[0].env.queue, inputs[0].env.kernel, threadDimensionsCount,
		 nullptr, threadDimensions, nullptr, 0, nullptr,	nullptr);

	clFinish(inputs[0].env.queue);
	if(debug && err){ printErr("%d Failed to Enqueue", err); return; }
}

int main()
{
	vector<Device *> devs = OpenCLQuery(CL_DEVICE_TYPE_CPU);

	Environment e(*devs[0]);
	std::ifstream ifs("Test.cl");
  	std::string kernel( (std::istreambuf_iterator<char>(ifs) ),
                       (std::istreambuf_iterator<char>()    ) );

	cout<<e.setKernel(kernel, "memset_float")<<endl;
	float *x=(float*)malloc(10*sizeof(float));
	for(int q = 0; q<10;q++)
		x[q] = 5;

	kernelArgument b(e,x,10*sizeof(float),CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR);
	float r=10;
	kernelArgument scalar(e,&r,sizeof(float));
	cout<<b.size<<endl<<scalar.size<<endl;
	size_t threads = 10;
	Run(1, &threads, {scalar,b});

	void *tmp = nullptr;
	b.readBufferFromDevice(tmp);
	float *tmpFloat = (float*)tmp;
	// cout<<tmpInt;
	for(int w = 0; w<10;w++)
		cout<<tmpFloat[w]<<endl;

	b.releaseAll();
	free(x);
	e.releaseAll();

	for(int i=0;i<devs.size();i++)
		devs[i]->releaseAll();

	return 0;
}