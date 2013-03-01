%CLASS_INTERFACE Example MATLAB class wrapper to an underlying C++ class
classdef OpenCLInterface < handle
    % properties (SetAccess = private, Hidden = true)
    properties
        Devices; % Handle to the underlying C++ class instance
        Environment_ptr;
    end
    methods
        %% Constructor - Create a new C++ class instance 
        function this = OpenCLInterface(this)
            this.Devices = OpenCLMex('deviceQuery');
            Environment_ptr = 0;
        end
        
        %% Destructor - Destroy the C++ class instance
        function delete(this)
            for i = 1:size(this.Devices,1)
                OpenCLMex('deleteDevice', this.Devices{i}{1});
                % OpenCLMex('delete',this.Devices);
            end
            if this.Environment_ptr ~= 0
                OpenCLMex('deleteEnvironment',this.Environment_ptr);
            end
        end

        function varargout = printDevices(this)
            for i = 1:size(this.Devices,1)
                message = sprintf('Device No: %d\n ',i);
                message = [message sprintf('Device ID: %s\n ',this.Devices{i}{2})];
                message = [message sprintf('Device Name: %s\n ',this.Devices{i}{3})];
                if this.Devices{i}{4}
                    message = [message sprintf('Device Type: %s\n ', 'GPU')];
                else
                    message = [message sprintf('Device Type: %s\n ', 'CPU')];
                end
                message = [message sprintf('Driver Version: %s\n ',this.Devices{i}{5})];
                message = [message sprintf('Global Memory Size: %s\n ',this.Devices{i}{6})];
                message = [message sprintf('Max Compute Units: %s\n ',this.Devices{i}{7})];
                message = [message sprintf('Max Workitem Dimensions: %s\n ',this.Devices{i}{8})];
                
                message = [message sprintf('Max Workitem Sizes:')];
                for j = 9:size(this.Devices{i},1)
                    message = [message sprintf(' %s ',this.Devices{i}{j})];
                end
                sprintf('Device Info:\n %s', message)
            end
        end

        %% Train - an example class method call
        function varargout = createFunction(this, DeviceID, kernelFile, kernelName)
            % [varargout{1:nargout}] = OpenCLMex('train', this.Devices, varargin{:});
            if exist(kernelFile,'file')
                this.Environment_ptr = OpenCLMex('buildKernel', this.Devices{DeviceID}{1}, fileread(kernelFile), kernelName);
            else
                error('File not found');
            end
        end

        %% Test - another example class method call
        function varargout = Run(this, threadDimensions, varargin)
            scalarCount = 0;
            bufferFlags = [];
            nextStringFlag = false;
            if length(threadDimensions) == 0
                error('Thread dimensions CANNOT be empty');
            end
            if length(varargin) == 0
                error('No data supplied');
            end
                
            for i=1:size(varargin,2)
                if  ~iscellstr(varargin(i))
                    if nextStringFlag
                        error('Buffer No. %d type expected',floor((i-scalarCount)/2));
                    end

                    if length(varargin{i}) == 1
                        if i-scalarCount ~= 1
                            error('All scalars must be sent after each other');
                        end
                        scalarCount = i;
                    else  %Buffer
                         nextStringFlag = true;
                         continue;
                    end
                elseif nextStringFlag % Expected string
                    str = varargin{i};
                    flags = 0;
                    ioset = false;
                    readset = false;
                    hostptrset = false;
                    %r = read, w = write
                    %c = copy_host_ptr, u = use_host_ptr, a = alloc_host_ptr
                    for j=1:length(str)
                        switch str(j)
                            case 'I'
                                if ~ioset
                                    flags = bitor(flags , 1);
                                    ioset = true;
                                else
                                    error('CANNOT set I/O flag twice in parameter %d', i);   
                                end 
                            case 'O'
                                if ~ioset
                                    flags = bitor(flags , 2);
                                    ioset = true;
                                else
                                    error('CANNOT set I/O flag twice in parameter %d', i);
                                end 
                            case 'r'
                                flags = bitor(flags , 4);
                                readset = true;
                            case 'w'
                                flags = bitor(flags , 8);
                                readset = true;
                            case 'c'
                                if ~hostptrset
                                    flags = bitor(flags , 16);
                                    hostptrset = true;
                                else
                                    error('CANNOT set host_ptr flag twice in parameter %d', i);
                                end
                            case 'u'
                                if ~hostptrset
                                    flags = bitor(flags , 32);
                                    hostptrset = true;
                                else
                                    error('CANNOT set host_ptr flag twice in parameter %d', i);
                                end
                            case 'a'
                                if ~hostptrset
                                    flags = bitor(flags , 64);
                                    hostptrset = true;
                                else
                                    error('CANNOT set host_ptr flag twice in parameter %d', i);
                                end
                            otherwise
                                error('Unkown buffer memory flag');
                        end
                    end
                    if ~ioset || ~readset || ~hostptrset
                        error('Missing buffer memory flag(s) in parameter %d',i);
                    end
                    bufferFlags = [bufferFlags flags];
                    nextStringFlag = false;
                else
                    error('Unexpected String Input in parameter %d',i);
                end
            end
            if nextStringFlag
                error('Last buffer type expected');
            end
            bufferFlags
            [varargout{1:nargout}] = OpenCLMex('Run', this.Environment_ptr, scalarCount, bufferFlags, threadDimensions, varargin);
        end
    end
end