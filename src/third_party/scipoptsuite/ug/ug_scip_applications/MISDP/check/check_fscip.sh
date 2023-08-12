#!/usr/bin/env bash
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
#*                                                                           *#
#*          This file is part of the program and software framework          *#
#*                    UG --- Ubquity Generator Framework                     *#
#*                                                                           *#
#*  Copyright Written by Yuji Shinano <shinano@zib.de>,                      *#
#*            Copyright (C) 2021 by Zuse Institute Berlin,                   *#
#*            licensed under LGPL version 3 or later.                        *#
#*            Commercial licenses are available through <licenses@zib.de>    *#
#*                                                                           *#
#* This code is free software; you can redistribute it and/or                *#
#* modify it under the terms of the GNU Lesser General Public License        *#
#* as published by the Free Software Foundation; either version 3            *#
#* of the License, or (at your option) any later version.                    *#
#*                                                                           *#
#* This program is distributed in the hope that it will be useful,           *#
#* but WITHOUT ANY WARRANTY; without even the implied warranty of            *#
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *#
#* GNU Lesser General Public License for more details.                       *#
#*                                                                           *#
#* You should have received a copy of the GNU Lesser General Public License  *#
#* along with this program.  If not, see <http://www.gnu.org/licenses/>.     *#
#*                                                                           *#
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
TSTNAME=$1
BINNAME=$2
SETNAME=$3
BINID=$4
TIMELIMIT=$5
MEMLIMIT=$6
CONTINUE=$7
LOCK=$8
LPS=$9
THREADS=${10}
SDPSETNAME=${11}

VERSION=0.1.0

echo "testname = $TSTNAME"
echo "binname = $BINNAME"
echo "setname = $SETNAME"
echo "binid = $BINID"
echo "timelimit = $TIMELIMIT"
echo "memlimit = $MEMLIMIT"
echo "continue = $CONTINUE"
echo "lock = $LOCK"
echo "version = $VERSION"
echo "lps = $LPS"
echo "threads = $THREADS"
echo "sdpsetnmae = $SDPSETNAME"

if test ! -e results
then
    mkdir results
fi
if test ! -e locks
then
    mkdir locks
fi

SETDIR=../settings
LOCKFILE=locks/$TSTNAME.$SETNAME.$VERSION.$LPS.lock
RUNFILE=locks/$TSTNAME.$SETNAME.$VERSION.$LPS.run.$BINID
DONEFILE=locks/$TSTNAME.$SETNAME.$VERSION.$LPS.done

OUTFILE=results/check.$TSTNAME.$BINID.$SETNAME.$SDPSETNAME.$THREADS.out
ERRFILE=results/check.$TSTNAME.$BINID.$SETNAME.$SDPSETNAME.$THREADS.err
RESFILE=results/check.$TSTNAME.$BINID.$SETNAME.$SDPSETNAME.$THREADS.res
TEXFILE=results/check.$TSTNAME.$BINID.$SETNAME.$SDPSETNAME.$THREADS.tex
TMPFILE=results/check.$TSTNAME.$BINID.$SETNAME.$SDPSETNAME.$THREADS.tmp
SETFILE=results/check.$TSTNAME.$BINID.$SETNAME.$SDPSETNAME.$THREADS.set
SDPSETFILE=results/check.$TSTNAME.$BINID.$SETNAME.$SDPSETNAME.$THREADS.sdpset

echo $TSTNAME
# echo $BINNAME
# echo $SETNAME
# echo $BINID
# echo $TIMELIMIT
# echo $MEMLIMIT
# echo $CONTINUE
# echo $LOCK
# echo $VERSION
# echo $LPS
# echo $THREADS 

SETTINGS=$SETDIR/$SETNAME.set
if [ "$SDPSETTINGS" != "" ]
then
        SDPSETTINGS=$SETDIR/$SDPSETNAME.set
fi

if test "$LOCK" = "true"
then
    if test -e $DONEFILE
    then
        echo skipping test due to existing done file $DONEFILE
        exit
    fi
    if test -e $LOCKFILE
    then
        if test -e $RUNFILE
        then
            echo continuing aborted run with run file $RUNFILE
        else
            echo skipping test due to existing lock file $LOCKFILE
            exit
        fi
    fi
    date > $LOCKFILE
    date > $RUNFILE
fi

if test ! -e $OUTFILE
then
    CONTINUE=false
fi

if test "$CONTINUE" = "true"
then
    MVORCP=cp
else
    MVORCP=mv
fi

DATEINT=`date +"%s"`
if test -e $OUTFILE
then
    $MVORCP $OUTFILE $OUTFILE.old-$DATEINT
fi
if test -e $ERRFILE
then
    $MVORCP $ERRFILE $ERRFILE.old-$DATEINT
fi

if test "$CONTINUE" = "true"
then
    LASTPROB=`./getlastprob.awk $OUTFILE`
    echo Continuing benchmark. Last solved instance: $LASTPROB
    echo "" >> $OUTFILE
    echo "----- Continuing from here. Last solved: $LASTPROB -----" >> $OUTFILE
    echo "" >> $OUTFILE
else
    LASTPROB=""
fi

uname -a >>$OUTFILE
uname -a >>$ERRFILE
date >>$OUTFILE
date >>$ERRFILE

# we add 10% to the hard time limit and additional 10 seconds in case of small time limits
HARDTIMELIMIT=`expr \`expr $TIMELIMIT + 600\` + $TIMELIMIT`

# we add 10% to the hard memory limit and additional 100mb to the hard memory limit
HARDMEMLIMIT=`expr \`expr $MEMLIMIT + 100\` + \`expr $MEMLIMIT / 10\``
HARDMEMLIMIT=`expr $HARDMEMLIMIT \* 1024`

echo "hard time limit: $HARDTIMELIMIT s" >>$OUTFILE
echo "hard mem limit: $HARDMEMLIMIT k" >>$OUTFILE

for i in `cat testset/$TSTNAME.test` DONE
do
    if test "$i" = "DONE"
    then
        date > $DONEFILE
        break
    fi

    if test "$LASTPROB" = ""
    then
        LASTPROB=""
        if test -f $i
        then
            echo @01 $i ===========
            echo @01 $i ===========                >> $ERRFILE
            cat $SETTINGS > $TMPFILE
            cat ../settings/default.set > results/default.set
            echo set limits time $TIMELIMIT          >> $OUTFILE
            echo TimeLimit = $TIMELIMIT              >> $TMPFILE
            echo -----------------------------
            date
            date >>$ERRFILE
            echo -----------------------------
            date +"@03 %s"
            if [ "$SDPSETTINGS" != "" ]
            then
                echo -e "ulimit -t $HARDTIMELIMIT s;../$BINNAME $TMPFILE $i -sth $THREADS -sl $SDPSETTINGS -sr $SDPSETTINGS -s $SDPSETTINGS"
                bash -c "ulimit -t $HARDTIMELIMIT s;../$BINNAME $TMPFILE $i -sth $THREADS -sl $SDPSETTINGS -sr $SDPSETTINGS -s $SDPSETTINGS" 2>>$ERRFILE
            else
                echo -e "ulimit -t $HARDTIMELIMIT s;../$BINNAME $TMPFILE $i -sth $THREADS"
                bash -c "ulimit -t $HARDTIMELIMIT s;../$BINNAME $TMPFILE $i -sth $THREADS" 2>>$ERRFILE
            fi
            echo "Number of threads = " $THREADS >> $OUTFILE
            date +"@04 %s"
            echo -----------------------------
            date
            date >>$ERRFILE
            echo -----------------------------
            echo
            echo =ready=
        else
            echo @02 FILE NOT FOUND: $i ===========
            echo @02 FILE NOT FOUND: $i =========== >>$ERRFILE
        fi
    else
        echo skipping $i
        if test "$LASTPROB" = "$i"
        then
            LASTPROB=""
        fi
    fi
done | tee -a $OUTFILE

rm -f $TMPFILE

date >>$OUTFILE
date >>$ERRFILE

if test -e $DONEFILE
then
    ./evalcheck.sh $OUTFILE
    
    if test "$LOCK" = "true"
    then
        rm -f $RUNFILE
    fi
fi
