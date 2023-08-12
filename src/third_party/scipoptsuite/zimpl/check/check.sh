#!/bin/sh
# $Id: check.sh,v 1.21 2011/10/25 08:18:01 bzfkocht Exp $
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#*                                                                           */
#*   File....: check.sh                                                      */
#*   Name....: check script                                                  */
#*   Author..: Thorsten Koch                                                 */
#*   Copyright by Author, All rights reserved                                */
#*                                                                           */
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
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
# $1 = Binary
PASS=0
COUNT=0
for i in expr param set subto condit var bool define vinst sos read
do
   COUNT=`expr $COUNT + 1` 
   $1 -v0 $i.zpl
   diff $i.lp $i.lp.ref >/dev/null
   case $? in
    0) echo Test $i "(lp)" OK; PASS=`expr $PASS + 1` ;;
    1) echo Test $i "(lp)" FAIL ;;
    *) echo Test $i "(lp)" ERROR ;;
   esac
   COUNT=`expr $COUNT + 1` 
   diff -b $i.tbl $i.tbl.ref >/dev/null
   case $? in
    0) echo Test $i "(tbl)" OK; PASS=`expr $PASS + 1`  ;;
    1) echo Test $i "(tbl)" FAIL ;;
    *) echo Test $i "(tbl)" ERROR ;;
   esac
   rm $i.tbl $i.lp
done 
for i in presol
do
   COUNT=`expr $COUNT + 1` 
   $1 -v0 -Distart=5 -t mps -r -m -n cm $i.zpl
   diff $i.mps $i.mps.ref >/dev/null
   case $? in
    0) echo Test $i "(mps)" OK; PASS=`expr $PASS + 1` ;;
    1) echo Test $i "(mps)" FAIL ;;
    *) echo Test $i "(mps)" ERROR ;;
   esac
   COUNT=`expr $COUNT + 1` 
   diff $i.tbl $i.tbl.ref >/dev/null
   case $? in
    0) echo Test $i "(tbl)" OK; PASS=`expr $PASS + 1`  ;;
    1) echo Test $i "(tbl)" FAIL ;;
    *) echo Test $i "(tbl)" ERROR ;;
   esac
   COUNT=`expr $COUNT + 1` 
   diff $i.mst $i.mst.ref >/dev/null
   case $? in
    0) echo Test $i "(mst)" OK; PASS=`expr $PASS + 1`  ;;
    1) echo Test $i "(mst)" FAIL ;;
    *) echo Test $i "(mst)" ERROR ;;
   esac
   COUNT=`expr $COUNT + 1` 
   diff $i.ord $i.ord.ref >/dev/null
   case $? in
    0) echo Test $i "(ord)" OK; PASS=`expr $PASS + 1`  ;;
    1) echo Test $i "(ord)" FAIL ;;
    *) echo Test $i "(ord)" ERROR ;;
   esac
   rm $i.tbl $i.mps $i.mst $i.ord
done 
#
   COUNT=`expr $COUNT + 1` 
   $1 -v0 -Distart=4 -t hum -n cf presol.zpl
   diff presol.hum presol.hum.ref >/dev/null
   case $? in
    0) echo Test presol.zpl "(hum)" OK; PASS=`expr $PASS + 1` ;;
    1) echo Test presol.zpl "(hum)" FAIL ;;
    *) echo Test presol.zpl "(hum)" ERROR ;;
   esac
   rm presol.hum
#
   COUNT=`expr $COUNT + 1` 
   $1 -v0 print.zpl >print.out
   diff print.out print.out.ref >/dev/null
   case $? in
    0) echo Test print.zpl "(out)" OK; PASS=`expr $PASS + 1` ;;
    1) echo Test print.zpl "(out)" FAIL ;;
    *) echo Test print.zpl "(out)" ERROR ;;
   esac
   rm print.out print.tbl print.lp
# 
#
   COUNT=`expr $COUNT + 1` 
   $1 -v0 -t pip minlp.zpl 
   diff minlp.pip minlp.pip.ref >/dev/null
   case $? in
    0) echo Test minlp.zpl "(pip)" OK; PASS=`expr $PASS + 1` ;;
    1) echo Test minlp.zpl "(pip)" FAIL ;;
    *) echo Test minlp.zpl "(pip)" ERROR ;;
   esac
   rm minlp.pip minlp.tbl
# 
#
   $1 -v0 -Dcities=5 -o metaio @selftest_tspste.zpl >metaio.out
   COUNT=`expr $COUNT + 1` 
   diff metaio.lp metaio.lp.ref >/dev/null
   case $? in
    0) echo Test metaio "(lp)" OK; PASS=`expr $PASS + 1` ;;
    1) echo Test metaio "(lp)" FAIL ;;
    *) echo Test metaio "(lp)" ERROR ;;
   esac
   COUNT=`expr $COUNT + 1` 
   diff metaio.out metaio.out.ref >/dev/null
   case $? in
    0) echo Test metaio "(out)" OK; PASS=`expr $PASS + 1` ;;
    1) echo Test metaio "(out)" FAIL ;;
    *) echo Test metaio "(out)" ERROR ;;
   esac
   rm metaio.lp metaio.tbl metaio.out
# 
#
for i in qubo
do
   COUNT=`expr $COUNT + 1` 
   $1 -v0 -t q -o $i.q $i.zpl 
   diff $i.q.qs $i.q.ref >/dev/null
   case $? in
    0) echo Test qubo.zpl "(qbo: q)" OK; PASS=`expr $PASS + 1` ;;
    1) echo Test qubo.zpl "(qbo: q)" FAIL ;;
    *) echo Test qubo.zpl "(qbo: q)" ERROR ;;
   esac
   COUNT=`expr $COUNT + 1` 
   $1 -v0 -t q0cp -o $i.q0cp $i.zpl 
   diff $i.q0cp.qs $i.q0cp.ref >/dev/null
   case $? in
    0) echo Test qubo.zpl "(qbo: q0cp)" OK; PASS=`expr $PASS + 1` ;;
    1) echo Test qubo.zpl "(qbo: q0cp)" FAIL ;;
    *) echo Test qubo.zpl "(qbo: q0cp)" ERROR ;;
   esac
   COUNT=`expr $COUNT + 1` 
   diff -b $i.q0cp.tbl $i.q0cp.tbl.ref >/dev/null
   case $? in
    0) echo Test $i "(tbl)" OK; PASS=`expr $PASS + 1`  ;;
    1) echo Test $i "(tbl)" FAIL ;;
    *) echo Test $i "(tbl)" ERROR ;;
   esac
   rm $i.q.tbl $i.q0cp.tbl $i.q.qs $i.q0cp.qs
done
#
for i in bqp50-1 ps5-1
do
    COUNT=`expr $COUNT + 1` 
    $1 -v0 -Dfilename=$i".sparse" -t q1 -o $i qubo2miqp.zpl
    grep -v "#" $i.qs | diff -b - $i.sparse
    case $? in
     0) echo Test $i "(qbo2)" OK; PASS=`expr $PASS + 1`  ;;
     1) echo Test $i "(qbo2)" FAIL ;;
     *) echo Test $i "(qbo2)" ERROR ;;
    esac
    rm $i.tbl $i.qs
done
for i in bqp50-1 ps5-1
do
    COUNT=`expr $COUNT + 1` 
    $1 -v0 -Dfilename=$i".sparse" -t q1 -o $i qubo2miqp2.zpl
    grep -v "#" $i.qs | diff -b - $i.sparse
    case $? in
     0) echo Test $i "(qbo3)" OK; PASS=`expr $PASS + 1`  ;;
     1) echo Test $i "(qbo3)" FAIL ;;
     *) echo Test $i "(qbo3)" ERROR ;;
    esac
    rm $i.tbl $i.qs
done
#
cd warnings
for i in w*.zpl
do
   COUNT=`expr $COUNT + 1` 
   NAME=`basename $i .zpl`
   case $NAME in
       w601|w602|w603) ../$1 -t q $i 2>$NAME.warn >/dev/null ;;
       *)  ../$1 $i 2>$NAME.warn >/dev/null ;;
   esac
   diff $NAME.warn $NAME.warn.ref >/dev/null
   case $? in
    0) echo Test $i "(warn)" OK; PASS=`expr $PASS + 1`  ;;
    1) echo Test $i "(warn)" FAIL ;;
    *) echo Test $i "(warn)" ERROR ;;
   esac
   rm -f $NAME.warn $NAME.tbl $NAME.lp $NAME.qs $NAME.sos
done 2>/dev/null
# 
# Special w215 test
COUNT=`expr $COUNT + 2` 
NAME=w215
../$1 -m $NAME 2>$NAME.warn >/dev/null
diff $NAME.warn $NAME-m.warn.ref >/dev/null
case $? in
 0) echo Test $NAME "-1 (warn)" OK; PASS=`expr $PASS + 1`  ;;
 1) echo Test $NAME "-1 (warn)" FAIL ;;
 *) echo Test $NAME "-1 (warn)" ERROR ;;
esac
diff $NAME.mst $NAME.mst.ref >/dev/null
case $? in
 0) echo Test $NAME "-2 (warn)" OK; PASS=`expr $PASS + 1`  ;;
 1) echo Test $NAME "-2 (warn)" FAIL ;;
 *) echo Test $NAME "-2 (warn)" ERROR ;;
esac
rm -f $NAME.warn $NAME.tbl $NAME.lp $NAME.mst
#
cd ..
cd errors
#
for i in e[1-6]*.zpl
do
   COUNT=`expr $COUNT + 1` 
   NAME=`basename $i .zpl`
   ../$1 -v0 $i 2>$NAME.err
   fgrep -v "Aborted (core dumped)" $NAME.err | diff - $NAME.err.ref >/dev/null
   case $? in
    0) echo Test $i "(err)" OK; PASS=`expr $PASS + 1`  ;;
    1) echo Test $i "(err)" FAIL ;;
    *) echo Test $i "(err)" ERROR ;;
   esac
   rm $NAME.err
done 2>/dev/null
#
# Error 700 to 900 can vary
#
for i in e[789]*.zpl
do
#   COUNT=`expr $COUNT + 1` 
   NAME=`basename $i .zpl`
   # DIFFOPT=`awk -f ../exdiffopt.awk $NAME.err.ref`
   ../$1 -v0 $i 2>$NAME.err
   fgrep -v "Aborted (core dumped)" $NAME.err | diff - $NAME.err.ref >/dev/null
   case $? in
    0) echo Test $i "(err)" OK;; 
    1) echo Test $i "(err)" FAIL "(ignored)";;
    *) echo Test $i "(err)" ERROR ;;
   esac
   rm $NAME.err
done 2>/dev/null

if [ $PASS -eq $COUNT ] ; then echo All $PASS tests passed; 
else echo FAILURE! Only $PASS of $COUNT tests passed; 
fi







