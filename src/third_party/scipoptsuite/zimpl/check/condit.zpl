# $Id: condit.zpl,v 1.2 2011/09/16 09:11:49 bzfkocht Exp $
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*                                                                           */
#*   File....: condit.zpl                                                    */
#*   Name....: Condition test                                                */
#*   Author..: Thorsten Koch                                                 */
#*   Copyright by Author, All rights reserved                                */
#*                                                                           */
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*
#* Copyright (C) 2011-2022 by Thorsten Koch <koch@zib.de>
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
set I := { 1 to 5 };

var x;
var y;
var z[<i> in I] >= if (i >= 3) then -infinity else 10 end <= 100 ;

subto c1: forall <i> in I do
   if i > 2 then 
      if i > 4 then 
         i * x <= 5 and i * 2 * y <= 10 
      else
         i * z[i] <= 7 
      end
   end;

subto c2: forall <i> in I do
   if i > 2 then 
      if i > 4 then 
         i * x <= 5 and i * 2 * y <= 10 
      end
   else
       i * z[i] <= 7 
   end;


subto c3: forall <i> in I:
  if (i > 3)
  then 
     if (i == 4) 
        then z[1] + z[2] == i
        else z[2] + z[3] == i
     end and
     z[3] + z[4] == i and
     z[4] + z[5] == i
  else
     if (i == 2) 
        then z[1] + z[2] == i + 10
        else z[2] + z[3] == i + 10
     end and
     z[3] + z[4] == i + 10 and
     z[4] + z[5] == i + 10
  end;

