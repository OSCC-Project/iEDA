# $Id: e142_2.zpl,v 1.1 2013/02/24 16:01:57 bzfkocht Exp $
set T := { 0 .. 1 };
set TR:= { "0900", "0915" } cross { "R1", "R2" };
var x[TR];
param bonus[T] := <0> 5 default 0;
subto c1: sum <t,r> in TR : bonus[t] * x[t,r] <= 5;
