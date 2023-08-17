# $Id: chvatal_diet.zpl,v 1.4 2009/09/13 16:15:53 bzfkocht Exp $
#
# From V. Chvatal: Linear Programming
# Chapter 1, Page 3ff.
#
# A diet problem
#
set Food      := { "Oatmeal", "Chicken", "Eggs", "Milk", "Pie", "Pork" };
set Nutrients := { "Energy", "Protein", "Calcium" };
set Attr      := Nutrients + {"Servings", "Price"};

param needed[Nutrients] := <"Energy"> 2000, <"Protein"> 55, <"Calcium"> 800;

param data[Food * Attr] := 
           | "Servings", "Energy", "Protein", "Calcium", "Price" |
|"Oatmeal" |         4 ,     110 ,        4 ,        2 ,      3  |
|"Chicken" |         3 ,     205 ,       32 ,       12 ,     24  |
|"Eggs"    |         2 ,     160 ,       13 ,       54 ,     13  |
|"Milk"    |         8 ,     160 ,        8 ,      284 ,      9  |
|"Pie"     |         2 ,     420 ,        4 ,       22 ,     20  |
|"Pork"    |         2 ,     260 ,       14 ,       80 ,     19  |;
#                          (kcal)        (g)        (mg)  (cents)       

var x[<f> in Food] integer >= 0 <= data[f, "Servings"];

minimize cost: sum <f> in Food : data[f, "Price"] * x[f];

subto need :
  forall <n> in Nutrients do
    sum <f> in Food : data[f, n] * x[f] >= needed[n];



