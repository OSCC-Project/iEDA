# $Id: subto.zpl,v 1.12 2010/06/10 19:42:40 bzfkocht Exp $
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*                                                                           */
#*   File....: subto.zpl                                                     */
#*   Name....: Subto test                                                    */
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
set A := { 1 to 5 };
set B := { 1 to 5 };

var x[A];
var y[B];

param a[A] := <1> 1, <2> 2, <3> 3, <4> 4, <5> 5 ;

minimize cost: x[1] + x[2] + 17;

subto c01:  x[1] <= 5;
subto c02:  x[1] + x[2] <= 5;
subto c03:  3 * x[1] <= 5;
subto c04:  x[1] * 3 <= 5;
subto c05:  3 * x[1] + x[2] <= 5;
subto c06:  x[1] + 3 * x[2] <= 5;
subto c07:  3 * (x[1] + x[2]) <= 5;
subto c08:  (x[1] + x[2]) * 3 <= 5;

subto c10:  sum <i> in A do x[i] <= 5;
subto c11:  sum <i> in A do 3 * x[i] <= 5;

subto c12:  sum <i> in A do (x[i] * 4) <= 5;
subto c13:  (sum <i> in A do x[i]) * 4 <= 5;
subto c14:  sum <i> in A do x[i] * 4 <= 5;

subto c15:  5 * sum <i> in A do x[i] <= 5;
subto c16: - x[5] - 2 * sum <i> in A with i < 3 do x[i] + x[4] * 3 <= 5;
subto c17: + x[5] - 2 * sum <i> in A with i < 3 do 7 * x[i] + 3 * x[4] <= 5;
subto c18: + x[5] - 1.5 * (2 * sum <i> in A with i < 3 do 7 * x[i] + 3 * x[4]) <= 5;

subto c20: sum <i> in A : x[i] + sum <j> in B : y[j] >= 2;
subto c21: 6 * (sum <i> in A : x[i] + sum <j> in B : y[j]) >= 2;
subto c22: sum <i> in A : sum <j> in B with j > i : (x[i] + y[j]) >= 2;

subto c23: 3 * sum <i> in A : 2 * sum <j> in B with j > i : (x[i] + y[j]) + 77 * x[5] <= 88;

subto c24: forall <i> in A : sum <j> in B : 3 * x[j] * 4 <= 17;

subto c25: y[1] + sum <i> in A : x[i] + y[2] <= 15;
subto c26: y[1] + sum <i> in A : (x[i] + y[2]) <= 15;
subto c27: y[1] + (sum <i> in A : x[i] + y[2]) <= 15;

subto c30: x[1] <= y[2];
subto c31: sum <i> in A : x[i] -5 <= sum <j> in B : y[j] + 12;
subto c32: x[1] + 5 >= x[2];
subto c33: 5 <= x[1];
subto c34: x[1] >= 5 + x[2];
subto c35: 5 + x[1] >= x[2];
subto c36: 5 <= 6;
subto c37: -5 + x[1] >= x[2];
subto c38: x[1] == a[1] + sum <i> in A do a[i]*x[i];
subto c39: x[1] == a[1] + sum <i> in A do x[i]*a[i];
subto c40: x[1] == a[1] + 2 * sum <i> in A do 2*a[i]*x[i]*3 + 4;
subto c41: x[1] == a[1] + 2 * sum <i> in A do 2*x[i]*a[i]*3 + 4;
subto c42: 17 <= x[1] + x[2] <= 23;
subto c43: 13 * a[1] / 2 >= sum <i> in A do a[i]*x[i] + 6 >= a[2];
subto c44: sum <i> in A with i < 5: 10^-i * x[i] >= 5, scale;
subto c45: sum <i> in A : 0.0001 * x[i] >= 6, scale;



