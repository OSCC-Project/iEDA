# $Id: steinerbaum.zpl,v 1.4 2009/09/13 16:15:54 bzfkocht Exp $
#
# This is a classical Steiner-Tree Problem in Networks formulation.
# BEWARE: This grows exponentially
#
set V := { 1 .. 5 };
set E := { <1,2>, <1,4>, <2,3>, <2,4>, <3,4>, <3,5>, <4,5> };
set T := { 1, 3, 5 };
 
param c[E] := <1,2> 1, <1,4> 2, <2,3> 3, <2,4> 4, <3,4> 5, <3,5> 6, <4,5> 7;

var x[E] binary;

minimize cost: sum <a,b> in E : c[a,b] * x[a,b];

set P[] := powerset(V);
set I   := indexset(P);

# If we partition V and there is a Terminal in both parts, there has
# to be at least one edge to connect them.
subto partition: 
   forall <i> in I with      P[i]  inter T != {}
                    and (V \ P[i]) inter T != {} do
      sum <a,b> in E with (<a> in P[i] and not <b> in P[i]) 
                       or (<b> in P[i] and not <a> in P[i]) : x[a,b] >= 1;

# That's it.
