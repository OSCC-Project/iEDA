# $Id: vinst.zpl,v 1.7 2014/01/12 12:21:25 bzfkocht Exp $
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*                                                                           */
#*   File....: vinst.zpl                                                     */
#*   Name....: Variable Instruction Test                                     */
#*   Author..: Thorsten Koch                                                 */
#*   Copyright by Author, All rights reserved                                */
#*                                                                           */
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*
#* Copyright (C) 2003-2022 by Thorsten Koch <koch@zib.de>
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
set I := { 1..10 };

var x[I] integer >= -10 <= 10;

subto c1: vif x[1] <  0 then x[2] ==  9 end;
subto c2: vif x[1] <= 0 then x[2] ==  8 end;
subto c3: vif x[1] == 0 then x[2] ==  7 end;
subto c4: vif x[1] != 0 then x[2] ==  6 end;
subto c5: vif x[1] >= 0 then x[2] ==  5 end;
subto c6: vif x[1] >  0 then x[2] ==  4 end;

subto c7: forall <i> in I with i < 4 :
   vif vabs(x[1] + x[2]) <= 5 then x[3] + x[4] >= 5 else x[5] + x[6] >= 8 end;

subto c8: vif 3 * x[1] + x[2] != 7 
   then sum <i> in I : x[i] <= 17
   else sum <i> in I : x[i] >= 5 end;

subto c9: vif x[1] == 1 and x[2] > 5 
           or x[1] == 2 and x[2] < 8 then x[3] == 7 end;

subto c10: vif x[1] != 1 xor not x[2] == 5 then x[3] <= 2 end;

subto c11: vif x[1] == 1  then x[2] == 0 else x[3] == 0 end, indicator;
