# $Id: set.zpl,v 1.17 2010/06/10 19:42:40 bzfkocht Exp $
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*                                                                           */
#*   File....: set.zpl                                                       */
#*   Name....: Set test                                                      */
#*   Author..: Thorsten Koch                                                 */
#*   Copyright by Author, All rights reserved                                */
#*                                                                           */
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*
#* Copyright (C) 2006-2022 by Thorsten Koch <koch@zib.de>
#* 
#* This program is free software; you can redistribute it and/or
#* modify it under the terms of the GNU General Public License
#* as published by the Free Software Foundation; either version 2
#* of the License, or (at your option) any later version.
#* 
#* This program is distributed in the hope that it will be useful,
#* but WITHOUT ANY WARRANTY; without even the implied warranty of
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#* GNU General Public License for more details.
#* 
#* You should have received a copy of the GNU General Public License
#* along with this program; if not, write to the Free Software
#* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
#*
set A := { "a", "b", "c" };
set B := { 1, 2, 3 };
set B2:= argmax <b> in B : abs(3-b);
set C := { 1 to 4 };
set C2:= argmin <c> in C : c;
set D := { 7 .. 8 by 2 };
set E2:= (B + C) * A;
set E := argmax(card(E2)) <i,j> in E2 : i; # does nothing ;-)
set F := B inter C;
set G := { 1 .. 5 } union D without B;
set H := C symdiff { <1>, <2>, <7>, <8> };
set J := { <g> in G with g > 3 }; 
set K := { <"a", 1>, <"b", 7>, <"c", 9> };
set L[B] := <1> { "a", "b" },
            <2> { "c", "d", "e" },
            <3> { "f" };
set LU   := union <b> in B : L[b];
set L2[] := <1> { "a", "b" },
            <2> { "c", "d", "e" },
            <3> { "f" };
set L2I  := inter <b> in B : (L2[b] + A);
set M[<i> in D] := { <3 * i, 7, "d">, <i + 2, 9, "c"> };
set N[<i,j> in C * B with i > j] := { i + j, i - j, i * j };
set P[] := powerset(C);
set Q[] := subsets(C, 3);
set R   := indexset(P);
set S   := R;
set T   := { 1 to 9 } * { 10 to 19 } * { "A", "B" };
set T2  := argmin(15) <i,j,k> in T : j - 2 * i;
set U   := proj(T2, <3,1>);
set V   := { <a,2> in A*B with a == "a" or a == "b" };
set W[<i> in B] := { <c> in C with c <= i };
set X   := { <i> in { <k> in C with k > 2 } with i mod 2 == 0 }; # { 4 }
set Y   := { -2 .. -2 } + { -4 .. -8 by 2 } + { -16 .. -10 by -2 } + { 7 .. 1 } ;
set Y2  := { -2 to -2 } + { -4 to -8 by 2 } + { -16 to -10 by -2 } + { 7 to 1 } ;
set Z   := C * (A * B);
set AA  := { <b> in B : 2 * b };
set BB  := { <a,b> in A*B : <a,b+2> };
set CC  := { 1 ..  1000 } * { 1001 .. 2000 } * { "A", "B", "C", "D", "E"} * { 2001 .. 9999 } * { 1 .. 100 };
set DD  := { <2,2000,"D",3000,e> in CC };
var a[L2I];
var b[B];
var c[C];
var d[D];
var e[E];
var f[F];
var g[G];
var h[H];
var j[J];
var k[K];
var l[LU];
var m[M[7]];
var n[{ 1 .. 100 }];
var p[C];
var q[C];
var u[U];
var v[V];
var xabcdefghijklmno[X];
var y[Y];
var y2[Y2];
var z[Z];
var aa[AA];
var bb[BB];
var dd[DD];
subto a1: sum <i> in L2I : a[i] >= 0;
subto b1: sum <i> in B : b[i] >= ord(B2,1,1);
subto c1: sum <i> in C : c[i] >= ord(C2,1,1);
subto d1: sum <i> in D : d[i] >= 0;
subto e1: sum <i1,i2> in E : e[i1,i2] >= 0;
subto f1: sum <i> in F : f[i] >= 0;
subto g1: sum <i> in G : g[i] >= 0;
subto h1: sum <i> in H : h[i] >= 0;
subto j1: sum <i> in J : j[i] >= 0;
subto k1: sum <i1,i2> in K : k[i1,i2] >= 0;
subto l1: forall <i1> in indexset(L) with L[i1] == L2[i1] do
   sum <i2> in L[i1] : l[i2] >= 0;
subto m1: sum <i1,i2,i3> in M[7] : m[i1,i2,i3] >= 0;
subto n1: forall <i1,i2> in indexset(N) do
   sum <i3> in N[i1,i2] : n[i3] >= 0;
subto p1: forall <i1> in R do
   sum <i2> in P[i1] : p[i2] >= 0;
subto q1: forall <i1> in indexset(Q) do
   sum <i2> in Q[i1] : q[i2] >= 0;
subto u1: forall <i1,i2> in U do u[i1,i2] >= 0;
subto v1: sum <i1,i2> in V : v[i1,i2] >= 0;
subto w1: forall <i> in B do
   sum <w> in W[i]: n[w] >= 0; 
subto x1: sum <i> in X : xabcdefghijklmno[i] >= 0;
subto y1: sum <i> in Y : y[i] >= 0;
subto y2: sum <i> in Y2 : y2[i] >= 2;
subto z1: sum <i> in {1..card(E)} do e[ord(E,i,1),ord(E,i,2)] >= 5;
subto z2: sum <c1,a1,b1> in Z with a1 > "b" do z[c1,a1,b1] <= 8;
subto aa1: sum <i> in AA : aa[i] >= 5;
subto bb1: sum <i1,i2> in BB : bb[i1,i2] <= 3;
subto dd1: b[1] == card(DD);






