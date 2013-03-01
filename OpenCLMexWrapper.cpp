#define __STDC_UTF_16__ 1
#include "mex.h"
#include "OpenCLWrapper.h"
#include "class_handle.hpp"
#include <iostream>
#include <functional>
#include <sstream>
using namespace std;

/**********************************************HELPER FUNCTIONS*****************************************************/
string getString(const mxArray *prhs){
	char *buf;
	size_t buflen;
	buflen = mxGetN(prhs)*sizeof(mxChar)+1;
	buf = (char*)mxMalloc(buflen);
	/* Copy the string data into buf. */
	mxGetString(prhs, buf, (mwSize)buflen);
	string s = ""; s += buf;
	mxFree(buf);
	return s;
}

// FNV-1a constants
static constexpr unsigned long long basis = 14695981039346656037ULL;
static constexpr unsigned long long prime = 1099511628211ULL;
 
// compile-time hash helper function
constexpr unsigned long long hash_one(char c, const char* remain, unsigned long long value)
{
    return c == 0 ? value : hash_one(remain[0], remain + 1, (value ^ c) * prime);
}
 
// compile-time hash
constexpr unsigned long long hash_(const char* str)
{
    return hash_one(str[0], str + 1, basis);
}

constexpr unsigned long long operator"" _hash(const char* str, size_t n ){
	return hash_(str);
}
/**********************************************HELPER FUNCTIONS*****************************************************/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if(!mxIsChar(prhs[0]))
		mexErrMsgIdAndTxt("Matlab:mexFunction","String Command Expected");
	
	
	string command = getString(prhs[0]);
	
	switch(hash_(command.c_str())){
		case "deviceQuery"_hash:
		{	
			vector<Device *> devs = OpenCLQuery(CL_DEVICE_TYPE_ALL);
			// plhs[0] = convertPtr2Mat<Device>(devs[0]);

			mwSize ndim = 1;
			mwSize dims[] = {devs.size()};

			mxArray *devsCellArrayPtr = mxCreateCellArray(ndim, dims);


			stringstream ss;
			for(int i=0;i<devs.size();i++){

				mwSize dim[] = {8 + devs[i]->maxWorkItemsDimensions};
				mxArray *deviceCellArrayPtr = mxCreateCellArray(ndim, dim);

				mxSetCell(deviceCellArrayPtr, 0, convertPtr2Mat<Device>(devs[i]));
				ss.str(std::string()); // Clear stringstream
				ss << devs[i]->deviceId;
				mxSetCell(deviceCellArrayPtr, 1,  mxCreateString(ss.str().c_str()));
				ss.str(std::string());
				ss << devs[i]->deviceName;
				mxSetCell(deviceCellArrayPtr, 2,  mxCreateString(ss.str().c_str()));
				// Device Type
				mxSetCell(deviceCellArrayPtr, 3,  mxCreateLogicalScalar(devs[i]->deviceType & CL_DEVICE_TYPE_GPU));
				ss.str(std::string());
				ss << devs[i]->driverVersion;
				mxSetCell(deviceCellArrayPtr, 4,  mxCreateString(ss.str().c_str()));
				ss.str(std::string());
				ss << devs[i]->globalMemorySize/1024/1024;
				mxSetCell(deviceCellArrayPtr, 5,  mxCreateString(ss.str().c_str()));
				ss.str(std::string());
				ss << devs[i]->maxComputeUnits;
				mxSetCell(deviceCellArrayPtr, 6,  mxCreateString(ss.str().c_str()));
				ss.str(std::string());
				ss << devs[i]->maxWorkItemsDimensions;
				mxSetCell(deviceCellArrayPtr, 7,  mxCreateString(ss.str().c_str()));

				int j=8;
				for(auto a = devs[i]->maxWorkItemsSizes.get(); a < devs[i]->maxWorkItemsSizes.get()+devs[i]->maxWorkItemsDimensions;a++){
					ss.str(std::string());
					ss<<*a;
					mxSetCell(deviceCellArrayPtr, j++, mxCreateString(ss.str().c_str()));	
				}
				mxSetCell(devsCellArrayPtr, i, deviceCellArrayPtr);
			}

			plhs[0] = devsCellArrayPtr;

			break;
		}

		case "buildKernel"_hash:
		{
			Device * d = convertMat2Ptr<Device>(prhs[1]);
			Environment *e = new Environment(*d);
			if(!mxIsChar(prhs[2]))
				mexErrMsgIdAndTxt("Matlab:mexFunction","String kernel expected");

			if(!mxIsChar(prhs[3]))
				mexErrMsgIdAndTxt("Matlab:mexFunction","String kernelName expected");
			
			if(!e->setKernel(getString(prhs[2]), getString(prhs[3])))
				mexErrMsgIdAndTxt("Matlab:OpenCLWrapper","Can't set kernel");

			plhs[0] = convertPtr2Mat(e);
			break;
		}

		case "Run"_hash:
		{
			Environment *e = convertMat2Ptr<Environment>(prhs[1]);
			if(!mxIsNumeric(prhs[2]) || mxGetN(prhs[2]) != 1 || mxGetM(prhs[2]) != 1)
				mexErrMsgIdAndTxt("Matlab:mexFunction","Scalar count expected");
			if(!mxIsNumeric(prhs[3]) || mxGetN(prhs[2]) != 1 || mxGetM(prhs[2]) != 1)
				mexErrMsgIdAndTxt("Matlab:mexFunction","Buffer modifiers expected");
			if(!mxIsNumeric(prhs[4]))
				mexErrMsgIdAndTxt("Matlab:mexFunction","Thread dimensions expected");
			if(!mxIsCell(prhs[5]))
				mexErrMsgIdAndTxt("Matlab:mexFunction","Data cell array expected");

			size_t dimensionsCount = mxGetN(prhs[4]);
			double *tmpDoublePtr = mxGetPr(prhs[4]);
			size_t *threadDimensions = new size_t[dimensionsCount];
			for(int i=0;i<dimensionsCount;i++){
				threadDimensions[i] = *(tmpDoublePtr+i);
				mexPrintf("Thread dimensions: %d\n",threadDimensions[i]);
			}

			vector<kernelArgument> kernArgs;
			double scalarCount = mxGetScalar(prhs[2]);
			tmpDoublePtr = mxGetPr(prhs[3]); // Buffer flags
			vector<tuple<int,int,mxClassID>> outputBufferIdx;
			for(int i=0;i<mxGetN(prhs[5]);i++){
				mxArray *c = mxGetCell(prhs[5], i);
				if(i < scalarCount){
					if(mxIsNumeric(c) && mxGetN(c) == 1 && mxGetM(c) == 1){
						kernArgs.push_back(kernelArgument(*e,mxGetData(c),mxGetElementSize(c)));
						mexPrintf("kernel arg data : %f\n",*((float*)kernArgs[i].data));
					}else{
						mexPrintf("Scalar input expected");
					}
				}else{
					// WARNING! Matlab calling interface must do sanity check on memory flags!


					double flags = *tmpDoublePtr;
					int flag = static_cast<int>(flags);
					cl_mem_flags memFlags = 0;
					if(flag & 4) // Read
						if(flag & 8) // Write
							memFlags |= CL_MEM_READ_WRITE;
						else // Read only
							memFlags |= CL_MEM_READ_ONLY;
					else // Write only
						memFlags |= CL_MEM_WRITE_ONLY;

					if(flag & 16) // Copy_host_ptr
						memFlags |= CL_MEM_COPY_HOST_PTR;
					else if(flag & 32) // Use_host_ptr
						memFlags |= CL_MEM_USE_HOST_PTR;
					else // Allocate_host_ptr
						memFlags |= CL_MEM_ALLOC_HOST_PTR;

					if(flag & 2)
						outputBufferIdx.push_back(make_tuple(i,mxGetN(c)*mxGetM(c),mxGetClassID(c)));

					mexPrintf("pointer: %p size: %d\n",mxGetData(c),mxGetN(c)*mxGetM(c)*mxGetElementSize(c));
					mexPrintf("%f %d %d %p\n",flags, flag,mxGetElementSize(c),mxGetData(c));
					kernArgs.push_back(kernelArgument(*e,mxGetData(c),mxGetN(c)*mxGetM(c)*mxGetElementSize(c),memFlags));
					tmpDoublePtr++;
					++i; // skip the text arguments
				}
			}
			Run(static_cast<cl_uint>(dimensionsCount), threadDimensions, kernArgs);
			for(int i=0; i<outputBufferIdx.size();i++){
				plhs[i] = mxCreateNumericMatrix(get<1>(outputBufferIdx[i]),1,  get<2>(outputBufferIdx[i]), mxREAL);
				void *ptr = mxGetData(plhs[i]);
				kernArgs[get<0>(outputBufferIdx[i])].readBufferFromDevice(ptr, false);
			}
			break;
		}

		case "deleteDevice"_hash:
		{
			mexPrintf("Device:before clas_handle %p\n", mxGetScalar(prhs[1]));

			Device *p = convertMat2Ptr<Device>(prhs[1]);
			mexPrintf("Device:before class_handle %s\n",p->deviceName.c_str());
			destroyObject<Device>(prhs[1]);
			mexPrintf("Device:class_handle done\n");
			
			break;
		}

		case "deleteEnvironment"_hash:
		{
			mexPrintf("Environment:before clas_handle %p\n", mxGetScalar(prhs[1]));
			Environment *p = convertMat2Ptr<Environment>(prhs[1]);
			mexPrintf("Environment:before class_handle %u\n",p->program);
			destroyObject<Environment>(prhs[1]);
			mexPrintf("Environment:class_handle done\n");

			break;
		}

		default:
			mexErrMsgIdAndTxt("Matlab:mexFunction", "Command not found");
			break;
	}

}