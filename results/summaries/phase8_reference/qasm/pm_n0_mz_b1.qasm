OPENQASM 3.0;
include "stdgates.inc";
bit[1] m;
sx $75;
sx $75;
barrier $59, $75;
barrier $59, $75;
rz(-pi) $75;
m[0] = measure $75;
