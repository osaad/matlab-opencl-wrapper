all:
	g++ -std=c++11 "OpenCLWrapper.cpp" -g -lOpenCL -Wall -Wextra -pedantic -Wall -Wextra -Wfloat-equal \
	-pedantic -Wno-unused-parameter -Wunreachable-code -Winline -Wshadow -Wsign-conversion -Wlogical-op \
	-Wno-multichar -Wredundant-decls -Woverloaded-virtual -fno-stack-protector -Wctor-dtor-privacy -Weffc++ \
	-Wold-style-cast -Wzero-as-null-pointer-constant -Wcast-qual -Wcast-align -Wconversion -Wctor-dtor-privacy \
	-Wdelete-non-virtual-dtor -Wdisabled-optimization -Wdouble-promotion -Wnoexcept -Wnon-virtual-dtor -Wold-style-cast \
	-Woverloaded-virtual -Wpointer-arith  -Wsuggest-attribute=const -Wsuggest-attribute=noreturn -Wsuggest-attribute=pure \
	-Wunused-macros -Wunused-parameter -Wvector-operation-performance -Wzero-as-null-pointer-constant -Wstrict-overflow -I"C:\Program Files\ATI Stream\include" -L"C:\Program Files\ATI Stream\lib\x86"