# $Id: define.zpl,v 1.8 2010/06/10 19:42:40 bzfkocht Exp $
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*                                                                           */
#*   File....: define.zpl                                                    */
#*   Name....: Define test                                                   */
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
set I := { 1..5 } * { 4..7};
var x[I];

defnumb dist(a,b) := a*a + b*b;

subto c1: sum <i,j> in I do dist(i,j) * x[i,j] >= 0;

defnumb ack(i,j) := 
   if i == 0 then j + 1 
             else if j == 0 then ack(i - 1, 1)
                            else ack(i - 1, ack(i, j - 1))
                  end
   end;

subto c2: ack(3,3) * x[ack(1,3),ack(2,2)] >= 0;            

set K := { 1..10 };
var y[K];

defbool goodone(a,b) := a > b; 
defset  bigger(i) := { <j> in K with goodone(j,i) };

subto c3: sum <i> in bigger(5) : y[i] >= 0;

set G := { 1..3 };
set H := { "hallo", "tach", "moin" };
param greet[G] := <1> "hallo", <2> "tach", <3> "moin";
var z[H]; 
defstrg greeting(m) := greet[(m + 1) mod card(G)];

subto c4: z[greeting(1)] + z[greeting(3)] >= 0; 


