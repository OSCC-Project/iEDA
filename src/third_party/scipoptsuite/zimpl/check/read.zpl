# $Id: read.zpl,v 1.7 2011/09/16 09:11:49 bzfkocht Exp $
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*   File....: read.zpl                                                      *
#*   Name....: read test                                                     *
#*   Author..: Thorsten Koch                                                 *
#*   Copyright by Author, All rights reserved                                *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*
#* Copyright (C) 2007-2022 by Thorsten Koch <koch@zib.de>
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
set A := { read "read1.dat" as "<1s>" };
set B := { read "read1.dat" as "<1n>" skip 5 use 100 comment "#" };
set C := { read "read2.dat" as "<1s,2n,3n,4n,5n,6n,7n,8n,9n,10n,11n,12n,13s>" comment "#[" };
set D := { read "read3.dat" as "<3n>" match " [0-9][0-9][0-9] " comment "ABCDEF" };
set E := {1..8}*{1..10000};

param a1 := card(A);
param b1 := sum <b> in B : b;
param c1 := sum <a,b,c,d,e,f,g,h,i,j,k,l,m> in C :
   (length(a) + b + c + d + e + f + g + h + i + j / (k + l + length(m)));
param d1 := sum <d> in D : d;

param mat[E] := read "read1.dat" as "n+" comment "#";
param e1     := (sum <1,i> in E : mat[1,i]) / (sum <5,i> in E : mat[5,i]) 
              + (sum <i,j> in { 2, 3, 4, 6, 7, 8 }*{1..10000} : mat[i,j]);

param fix    := read "read1.dat" as "2n" comment "#" use 1;

var x real >= -infinity <= infinity;

subto c1: x <= a1; # <= 20001
subto c2: x >= b1 - sum <i> in { 21 .. 417 by 4 } : i; # >= 0
subto c3: e1 * x <= c1;
subto c4: fix * x == d1;

set F := { read "read3.dat" as "<1n>" match "Z" }; # empty
set I := {};
set J := I * I;
set K := I + { 1..6 };
set L := I * { 7..9 };
set M := I + { <"a","b">, <"c","d"> };
set N := { <i> in {} : i };

do check card(F) == 0;
do check card(I) == 0;
do check card(K) == 6;
do check card(L) == 0;
do check card(M) == 2;
do check card(N) == 0;

var z[I];

subto c5: forall <i> in I do
   z[i] == z[j];

subto c6: forall <i> in F do
   z[i] == z[j];

subto c7: forall <i,j> in J with i > j do
   z[i] == z[j];

subto c8: forall <i> in L do
   z[i] == z[j];

