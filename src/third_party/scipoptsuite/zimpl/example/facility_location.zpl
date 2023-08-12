# $Id: facility_location.zpl,v 1.4 2009/09/13 16:15:53 bzfkocht Exp $
#
# Generic formulation for the Capacitated Facility Location Problem
#
# Of course this is also kind of a bin packing problem with packing costs
# and variable sized bins, or a cutting stock problem with cutting costs.
#
set PLANTS := { "A", "B", "C", "D" };
set STORES := { 1 .. 9 };
set PS     := PLANTS * STORES;

# How much does it cost to build a plant and what capacity will 
# it then have?
#
param building[PLANTS] := <"A"> 500, <"B"> 600, <"C"> 700, <"D"> 800;
param capacity[PLANTS] := <"A">  40, <"B">  55, <"C">  73, <"D">  90;

# Here is the demand of each store
#
param demand  [STORES] := <1> 10, <2> 14, <3> 17, <4> 8, <5> 9,
                          <6> 12, <7> 11, <8> 15, <9> 16;

# Transportation cost from each plant to each store
#
param transport[PS] := 
      |  1,  2,  3,  4,  5,  6,  7,  8,  9 |
  |"A"| 55,  4, 17, 33, 47, 98, 19, 10,  6 |
  |"B"| 42, 12,  4, 23, 16, 78, 47,  9, 82 | 
  |"C"| 17, 34, 65, 25,  7, 67, 45, 13, 54 |
  |"D"| 60,  8, 79, 24, 28, 19, 62, 18, 45 |;

var x[PS]     binary;  # Is plant p suppling store s ?
var z[PLANTS] binary;  # Is plant p build ?

# We want it cheap
#
minimize cost: sum <p> in PLANTS : building[p] * z[p]
             + sum <p,s> in PS : transport[p,s] * x[p,s];

# Each store is supplied by exactly one plant
#
subto assign: forall <s> in STORES : sum <p> in PLANTS : x[p,s] == 1;
   
# To be able to supply a store, a plant must be build
#
subto build: forall <p,s> in PS : x[p,s] <= z[p];

# The plant must be able to meet the demands from all stores
# that are connected to it
#
subto limit: forall <p> in PLANTS :
   sum <s> in STORES : demand[s] * x[p,s] <= capacity[p];





