# $Id: queens1.zpl,v 1.5 2009/09/13 16:15:53 bzfkocht Exp $
#
# This is a formulation of the n queens problem using general integer
# variables. Please note that this particular formulation only works,
# if a queen can be placed in each row, so queens should be greater
# equal to 4
#
param queens := 8;

set I := { 1 .. queens };
set P := { <i,j> in I * I with i < j };
 
var x[I] integer >= 1 <= queens;

# All x have to be different 
#
subto c1: forall <i,j> in P do vabs(x[i] - x[j]) >= 1;

# Block diagonals => 
# never the same distance between two queens in x and y direction =>
# abs(x[i] - x[j]) != abs(i - j) =>
# a != b modeled as abs(a - b) >= 1
#
subto c2: forall <i,j> in P do vabs(vabs(x[i] - x[j]) - abs(i - j)) >= 1;
