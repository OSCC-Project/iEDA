set I := { 1 .. 4 };

var x[I] binary;

minimize cost: 1000 * x[1];

subto c1: sum <i> in I do x[i] == 1, qubo, penalty4;
subto c2: x[1] == 1, qubo, penalty4;



