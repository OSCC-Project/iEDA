# TK10Nov2021 qubo2miqp.zpl
#
# Converting a QUBO in BiqMac format to MIQP in LP format
# example see here biqmac.uni-kla.ac.at
# 
# usage: zimpl -Dfilename="bqp50-1.sparse" -o bqp50-1 qubo2miqp.zpl
# Needs zimpl 3.5.0 up

# We could do without the nodes by using
# set I := proj(E, <1>) + proj(E,<2>);
param nodes := read filename as "1n" use 1 comment "#";

# should also work with using edges. But this way there can be other stuff
# behind the edges and we won't care.
param edges := read filename as "2n" use 1 comment "#"; 

set I := { 1 .. nodes };

set E := { read filename as "<1n,2n>" skip 1 use edges comment "#" };

param weight[I*I] := read filename as "<1n,2n> 3n" skip 1 use edges comment "#";

var x[I] binary;

minimize cost:
   sum <i,j> in E : x[i] * x[j] * weight[i,j] * if i == j then 1 else 2 end;



