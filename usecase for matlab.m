a = OpenCLInterface;

Internal funcs:
printDevices() % print available devices
% [Devices] = printDevices('GPU') % return list of devices info

OpenCLCreateFunction(Devices(1), 'kernel.cl', 'kernelName') % path to file or string (Function Should Check For IT!!!!!!)

buffer1 = 1:10;
buffer2 = zeros(10,1); % Output
%r = read, w = write, default = rw, u = use_host_ptr, a = alloc_host_ptr, c = copy_host_ptr, default = c
Run(threadDimensionsCount, threadCount, scalar1, scalar2, buffer1, 'Irwc', buffer2, 'O', ...)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% buffer reusability

[Devices] = OpenCLQuery() % return list of devices info
[Devices] = OpenCLQuery('GPU') % return list of devices info

obj1 = OpenCLCreateFunction(Devices(1), 'kernel.cl', 'kernelName', ['I','O']) % path to file or string (Function Should Check For IT!!!!!!)
obj2 = OpenCLCreateFunction(Devices(1), 'kernel.cl', 'kernelName', ['I','O']) % path to file or string (Function Should Check For IT!!!!!!)

buffer1 = 1:10;
%buffer2 = zeros(10,1); % Output

bufHandle = OpenCLAllocateMem(scalarSize) % Return Object!!!!

obj1.Run(threadCount, scalar1, scalar2, buffer1, bufHandle, ...)
obj2.Run(threadCount, scalar1, scalar2, buffer1, bufHandle, ...)

data = OpenCLReadMem(bufHandle)

OpenCLFree(bufHandle)

