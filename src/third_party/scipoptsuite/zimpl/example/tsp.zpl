# $Id: tsp.zpl,v 1.9 2009/09/13 16:15:54 bzfkocht Exp $
#
# Generic formulation of the Travelling Salesmen Problem
#
set V   := { read "tsp.dat" as "<1s>" comment "#" };
set E   := { <i,j> in V * V with i < j };
set P[] := powerset(V \ { ord(V,1,1) });
set K   := indexset(P) \ { 0 };

param px[V] := read "tsp.dat" as "<1s> 2n" comment "#";
param py[V] := read "tsp.dat" as "<1s> 3n" comment "#";

defnumb dist(a,b) := sqrt((px[a] - px[b])^2 + (py[a] - py[b])^2);

var x[E] binary;

minimize cost: sum <i,j> in E : dist(i,j) * x[i, j];

subto two_connected:
   forall <v> in V do
      (sum <v,j> in E : x[v,j]) + (sum <i,v> in E : x[i,v]) == 2;

subto no_subtour:
   forall <k> in K with card(P[k]) > 2 and card(P[k]) < card(V) - 2 do
      sum <i,j> in E with <i> in P[k] and <j> in P[k] : x[i,j] 
      <= card(P[k]) - 1;





