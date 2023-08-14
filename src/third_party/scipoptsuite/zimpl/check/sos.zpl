# $Id: sos.zpl,v 1.7 2010/06/10 19:42:40 bzfkocht Exp $
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*                                                                           */
#*   File....: sos.zpl                                                     */
#*   Name....: SOS test                                                    */
#*   Author..: Thorsten Koch                                                 */
#*   Copyright by Author, All rights reserved                                */
#*                                                                           */
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*
#* Copyright (C) 2005-2022 by Thorsten Koch <koch@zib.de>
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
set A := { 1 to 5 };
set B := { 1 to 5 };

var x[A];
var y[B];

param a[A] := <1> 1, <2> 2, <3> 3, <4> 4, <5> 5 ;

sos c01:  type1: x[1];
sos c02:  type1 priority 10 : x[1] + 2 * x[2];
sos c03:  type2: x[1] * 3 + 2 * x[2] + x[1];
sos c04:  type1 priority 60 : 3 * (x[1] + 2 * x[2]) + x[4];
sos c05:  type2 : sum <i> in A do i * x[i];
sos c06:  type1 : - x[5] - 2 * sum <i> in A with i > 3 do i * x[i] + x[3] * 2 
         + 90 * y[1] + 62 * y[2];
sos c07:  type2 : sum <i> in A: a[i] * x[i];
sos c08:  forall <i> in A do type2 priority 5 : sum <j> in B: (a[j] + i) * y[j];


