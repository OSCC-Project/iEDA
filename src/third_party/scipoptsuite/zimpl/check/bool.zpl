# $Id: bool.zpl,v 1.9 2010/06/10 19:42:40 bzfkocht Exp $
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*                                                                           */
#*   File....: bool.zpl                                                      */
#*   Name....: bool test                                                     */
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
set A := { 1 .. 10 };
set B := { 5 .. 15 };

param a1 := if A == B or A <= B then 1 else 2 end;
param a2 := if A != B and A < B then 1 else 2 end;
param a3 := if not A > B and not A < B then 1 else 2 end;
param a4 := if a1 < a2 or a2 != a3 then 1 else 2 end;
param a5 := if if a1 != a2 then a3 < a4 else a3 > a4 end then 1 else 2 end;
param a6 := if A == B xor A <= B then 3 else 4 end;
param a7 := if "aaaa" >= "aaaab" then 5 else 1 end;
param a8 := if exists(<i> in { 1 .. 20 } with i * i == 16) then 1 else 2 end;
param a9 := if <4> in { 1 .. 4 } then 7 else 8 end
          + if <"hallo"> in { "moin", "tach", "hi"} then 2 else 5 end
          + if <5> in {} then 100 else 0 end;
var x[A];

maximize c1: a1 * x[1] + a2 * x[2] + a3 * x[3] + a4 * x[4] + a5 * x[5] 
           + a6 * x[6] + a7 * x[7] + a8 * x[8] + a9 * x[9];
 








