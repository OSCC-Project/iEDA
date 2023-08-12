# $Id: var.zpl,v 1.10 2010/06/10 19:42:40 bzfkocht Exp $
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*                                                                           */
#*   File....: var.zpl                                                       */
#*   Name....: Var test                                                      */
#*   Author..: Thorsten Koch                                                 */
#*   Copyright by Author, All rights reserved                                */
#*                                                                           */
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*
#* Copyright (C) 2001-2022 by Thorsten Koch <koch@zib.de>
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
set I := { 1 .. 5 };
set K := { -5 .. -1 };

var a1;
var b1 real;
var b2 real <= 10;
var b3 real >= 5/2;
var b4 real >= 2 <= 7/3;
var c1 <= 10;
var c2 >= -5/4;
var c3 >= 2 <= 7;
var c4 <= infinity;
var c5 >= -infinity <= -30;
var d1 binary;
var d2 binary priority 50;
var d3 binary startval 1;
var d4 binary priority 100 startval 1;
var e1 integer;
var e2 integer >= -5;
var e3 integer <= 10;
var e4 integer >= -2 <= 6;
var e5 integer >= 5 priority 10;
var e6 integer <= 10 startval 5;
var e7 integer >= 2 <= 6 priority 15 startval 4;

var x[I];
var y[<i> in I] integer >= -5 * i <= 6 * i priority 3 startval i;
var z[K];
var w[<i> in I] <= if (i <= 3) then i else infinity end;

subto ca1: a1 + sum <i> in I : x[i] >= 17;
subto cb1: b1 + b2 + b3 + b4 <= 99;
subto cc1: c1 + c2 + c3 + c4 + c5 <= 77;
subto cd1: d1 + d2 + d3 + d4>= 2;
subto ce1: e1 + e2 + e3 + e4 + e5 + e6 + e7 <= 18;
subto cy: sum <i> in I : y[i] == -6;
subto cd: sum <k> in K : k * z[k] <= -8; 
subto cf: sum <i> in I : w[i] <= 17;
