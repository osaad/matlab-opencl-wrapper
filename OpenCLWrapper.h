#include <iostream>
#include <string>
#include <CL/cl.h>
#include <memory>
#include <vector>
//#include "mex.h"
struct Device{
	// explicit Device(int x):maxWorkItemsDimensions(new int[x], [](int *p){delete[] p;}){}
	// explicit Device(int dimensionsNo) 
	// : maxWorkItemsDimensions(new int[dimensionsNo], std::default_delete<int[]>()),
	// maxComputeUnits(0) // for example
	// {}
	explicit Device(cl_device_id devId);
	explicit Device(){}
	cl_device_id deviceId;
	cl_device_type deviceType; 
	std::string vendor;
	std::string deviceName; 
	std::string driverVersion;
	cl_uint maxComputeUnits;
	cl_uint maxWorkItemsDimensions;
	std::shared_ptr<size_t> maxWorkItemsSizes;
	cl_ulong globalMemorySize;

	Device(const Device& ) = default; //copy constructor
	Device(      Device&&) = default; //move constructor
	Device& operator = (const Device& ) = default; //copy assignment operator
	Device& operator = (      Device&&) = default; //move assignment operator
	void releaseAll();
};

struct Environment{
	cl_kernel kernel;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	Device device;
	explicit Environment(Device dev);
	explicit Environment(){}
	
	bool setKernel(std::string kernel, std::string kernelName);

	Environment(const Environment& ) = default; //copy constructor
	Environment(      Environment&&) = default; //move constructor
	Environment& operator = (const Environment& ) = default; //copy assignment operator
	Environment& operator = (      Environment&&) = default; //move assignment operator

	void releaseAll();
};

struct kernelArgument{
	cl_mem memory;
	size_t size;
	cl_mem_flags memoryFlags;
	Environment env;
	void *data;
	bool scalar;
	explicit kernelArgument():size(0),memoryFlags(0),data(nullptr),scalar(false){}
	explicit kernelArgument(Environment e, void *d, size_t s, cl_mem_flags memFlags); //size is in BYTES
	explicit kernelArgument(Environment e, void *d, size_t s):size(s),env(e),data(d),scalar(true){}; //size is in BYTES

	void readBufferFromDevice(void *&d, bool allocate);

	kernelArgument(const kernelArgument& ) = default; //copy constructor
	kernelArgument(      kernelArgument&&) = default; //move constructor
	kernelArgument& operator = (const kernelArgument& ) = default; //copy assignment operator
	kernelArgument& operator = (      kernelArgument&&) = default; //move assignment operator
	void releaseAll();
};

// Output copying will be called from the Mex Wrapper
void Run(cl_uint threadDimensionsCount, size_t *threadDimensions, std::vector<kernelArgument> inputs);

std::vector<Device *> OpenCLQuery(cl_device_type deviceType);