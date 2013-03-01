__kernel void memset_float( float val, __global float* mem) {
 	mem[get_global_id(0)] = val;
 }