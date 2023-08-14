# $Id: presol.zpl,v 1.7 2010/06/10 19:42:40 bzfkocht Exp $
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*   File....: presol.zpl                                                    *
#*   Name....: Presolve test                                                 *
#*   Author..: Thorsten Koch                                                 *
#*   Copyright by Author, All rights reserved                                *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*
#* Copyright (C) 2004-2022 by Thorsten Koch <koch@zib.de>
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
set I := { 1..5 };

var x[<i> in I] binary startval i mod 2;
var y[<i> in I] integer <= 2 * i priority i * 10 startval istart;
var z[<i> in I] <= 4 * i;
var w[<i> in I] integer >= 1 <= if i mod 2 == 1 then 1 else 2 end;

minimize cost: sum <i> in I : -(x[i] + y[i]);
 
subto c1: forall <i> in I: x[i] <= 1;
subto c2: y[1] + y[2] == 6;
subto c3: sum <i> in I: x[i] <= 3;
subto c4: sum <i> in I: y[i] == 15;
subto c5: forall <i> in I: y[i] >= z[i];
subto c6: forall <i> in I with i < 3: x[i] <= z[i];
subto c7: forall <i> in I with i >= 3: 1 <= x[i] + y[i] <= 100;
subto c8: forall <i> in I with i mod 2 == 1: -3 * w[i] == -3;
subto c9: w[2] >= 2;
subto c10: w[4] <= 2;
