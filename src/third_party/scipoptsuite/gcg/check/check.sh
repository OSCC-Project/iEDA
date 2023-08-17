#!/usr/bin/env bash
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program                         *
#*          GCG --- Generic Column Generation                                *
#*                  a Dantzig-Wolfe decomposition based extension            *
#*                  of the branch-cut-and-price framework                    *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#* Copyright (C) 2010-2022 Operations Research, RWTH Aachen University       *
#*                         Zuse Institute Berlin (ZIB)                       *
#*                                                                           *
#* This program is free software; you can redistribute it and/or             *
#* modify it under the terms of the GNU Lesser General Public License        *
#* as published by the Free Software Foundation; either version 3            *
#* of the License, or (at your option) any later version.                    *
#*                                                                           *
#* This program is distributed in the hope that it will be useful,           *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of            *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
#* GNU Lesser General Public License for more details.                       *
#*                                                                           *
#* You should have received a copy of the GNU Lesser General Public License  *
#* along with this program; if not, write to the Free Software               *
#* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.*
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#
# @author Martin Bergner
# @author Gerald Gamrath
# @author Christian Puchert

TSTNAME=$1
BINNAME=$2
SETNAME=$3
MSETNAME=$4
BINID=$5
TIMELIMIT=$6
NODELIMIT=$7
MEMLIMIT=$8
THREADS=$9
FEASTOL=${10}
DISPFREQ=${11}
CONTINUE=${12}
LOCK=${13}
VERSION=${14}
LPS=${15}
VALGRIND=${16}
MODE=${17}
SETCUTOFF=${18}
STATISTICS=${19}
SHARED=${20}
VISU=${21}
LAST_STATISTICS=${22}
SCRIPTSETTINGS=${23}
DETECTIONSTATISTICS=${24}

SETDIR=../settings

if test "${SHARED}" = "true"
then
  LD="LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:../lib/shared/"
fi

SETDIR=../settings

if test "${VISU}" = "true"
then
    DETECTIONSTATISTICS=true
    STATISTICS=true
fi
if test ! -e results
then
    mkdir results
fi
if test ! -e results/vbc
then
    if test "${STATISTICS}" = "true"
    then
        mkdir results/vbc
    fi
fi
if test ! -e locks
then
    mkdir locks
fi
if test "${DETECTIONSTATISTICS}" = "true"
then
    mkdir -p results/decomps
fi

LOCKFILE=locks/$TSTNAME.$SETNAME.$MSETNAME.$VERSION.$LPS.lock
RUNFILE=locks/$TSTNAME.$SETNAME.$MSETNAME.$VERSION.$LPS.run.$BINID
DONEFILE=locks/$TSTNAME.$SETNAME.$MSETNAME.$VERSION.$LPS.done

OUTFILE=results/check.$TSTNAME.$BINID.$SETNAME.$MSETNAME.out
ERRFILE=results/check.$TSTNAME.$BINID.$SETNAME.$MSETNAME.err
RESFILE=results/check.$TSTNAME.$BINID.$SETNAME.$MSETNAME.res
TEXFILE=results/check.$TSTNAME.$BINID.$SETNAME.$MSETNAME.tex
TMPFILE=results/check.$TSTNAME.$BINID.$SETNAME.$MSETNAME.tmp
SETFILE=results/check.$TSTNAME.$BINID.$SETNAME.$MSETNAME.set

SETTINGS=$SETDIR/$SETNAME.set
MSETTINGS=$SETDIR/$MSETNAME.set

if test "$LOCK" = "true"
then
    if test -e "$DONEFILE"
    then
        echo skipping test due to existing done file $DONEFILE
        exit
    fi
    if test -e "$LOCKFILE"
    then
        if test -e "$RUNFILE"
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
if test -e "$OUTFILE"
then
    $MVORCP $OUTFILE $OUTFILE.old-$DATEINT
fi
if test -e "$ERRFILE"
then
    $MVORCP $ERRFILE $ERRFILE.old-$DATEINT
fi

if test "$CONTINUE" = "true"
then
    LASTPROB=`awk -f getlastprob.awk $OUTFILE`
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

# we add 50% to the hard time limit and additional 10 seconds in case of small time limits
HARDTIMELIMIT=`expr \`expr $TIMELIMIT + 10\` + \`expr $TIMELIMIT / 2\``

# we add 10% to the hard memory limit and additional 1000mb to the hard memory limit
HARDMEMLIMIT=`expr \`expr $MEMLIMIT + 1000\` + \`expr $MEMLIMIT / 10\``
HARDMEMLIMIT=`expr $HARDMEMLIMIT \* 1024`

echo "hard time limit: $HARDTIMELIMIT s" >>$OUTFILE
echo "hard mem limit: $HARDMEMLIMIT k" >>$OUTFILE

VALGRINDCMD=
if test "$VALGRIND" = "true"
then
   VALGRINDCMD="valgrind --log-fd=1 --leak-check=full"
fi

SOLUFILE=""
for SOLU in testset/$TSTNAME.solu testset/_all.solu
do
    if test -e "$SOLU"
    then
        SOLUFILE=$SOLU
        break
    fi
done

# if cutoff should be passed, solu file must exist
if test $SETCUTOFF = "true"
then
    if test $SOLUFILE = ""
    then
        echo "Skipping test: SETCUTOFF=true set, but no .solu file (testset/$TSTNAME.solu or testset/_all.solu) available"
        exit
    fi
fi

for i in `cat testset/$TSTNAME.test` DONE
do
    if test "$i" = "DONE"
    then
        date > $DONEFILE
        break
    fi

    PROB=`echo $i|cut -d";" -f1`
    DECFILE=`echo $i|cut -d";" -f2`
    DIR=`dirname $PROB`
    NAME=`basename $PROB .gz`
    NAME=`basename $NAME .mps`
    NAME=`basename $NAME .lp`

    if test "$PROB" == "$DECFILE"
    then
        EXT=${PROB##*.}
        if test "$EXT" = "gz"
        then
            BLKFILE=$DIR/$NAME.blk.gz
            DECFILE=$DIR/$NAME.dec.gz
        else
            BLKFILE=$DIR/$NAME.blk
            DECFILE=$DIR/$NAME.dec
        fi
    fi
    if test "$LASTPROB" = ""
    then
        if test -f "$PROB"
        then
            echo @01 $PROB ===========
            echo @01 $PROB ===========             >> $ERRFILE
            echo > $TMPFILE
            if test "$SETNAME" != "default"
            then
                echo set load $SETTINGS            >> $TMPFILE
            fi
            if test "$MSETNAME" != "default"
            then
                echo set loadmaster $MSETTINGS     >>  $TMPFILE
            fi
            if test "$FEASTOL" != "default"
            then
                echo set numerics feastol $FEASTOL >> $TMPFILE
            fi

            if test -e "$SOLUFILE"
            then
                OBJECTIVEVAL=`grep "$NAME" $SOLUFILE | grep -v =feas= | grep -v =inf= | tail -n 1 | awk '{print $3}'`
            else
                OBJECTIVEVAL=""
            fi

            echo set limits time $TIMELIMIT        >> $TMPFILE
            echo set limits nodes $NODELIMIT       >> $TMPFILE
            echo set limits memory $MEMLIMIT       >> $TMPFILE
            echo set lp advanced threads $THREADS  >> $TMPFILE
            echo set timing clocktype 1            >> $TMPFILE
            echo set display verblevel 4           >> $TMPFILE
            echo set display freq $DISPFREQ        >> $TMPFILE
            if test $STATISTICS = "true"
            then
              if test $VISU = "true"
              then
                mkdir -p results/vbc/$TSTNAME/
                echo set visual vbcfilename results/vbc/$TSTNAME/$NAME.$SETNAME.vbc >> $TMPFILE
              else
                echo set visual vbcfilename results/vbc/$NAME.$SETNAME.vbc >> $TMPFILE
              fi
            fi
            echo set memory savefac 1.0            >> $TMPFILE # avoid switching to dfs - better abort with memory error
            if test "$LPS" = "none"
            then
                echo set lp solvefreq -1           >> $TMPFILE # avoid solving LPs in case of LPS=none
            fi
            echo set save $SETFILE                 >> $TMPFILE
            echo read $PROB                        >> $TMPFILE

	    # set objective limit: optimal solution value from solu file, if existent
	    if test $SETCUTOFF = "true"
	    then
	        if test $SOLUFILE == ""
	        then
	            echo Exiting test because no solu file can be found for this test
	            exit
	        fi
	        if test ""$OBJECTIVEVAL != ""
	        then
	            echo set limits objective $OBJECTIVEVAL >> $TMPFILE
	            echo set heur emph off                  >> $TMPFILE
	            echo master                             >> $TMPFILE
	            echo set heur emph off                  >> $TMPFILE
	            echo quit                               >> $TMPFILE
	        fi
	    fi


            if test $MODE = "detect"
            then
                echo presolve                      >> $TMPFILE
                echo detect                        >> $TMPFILE
                echo display statistics            >> $TMPFILE
                if test $STATISTICS = "true"
                then
                    echo display additionalstatistics  >> $TMPFILE
                fi
            elif test $MODE = "miplibfeaturesoriginal"
            then
		if test ! -e results/features_original
		then
		    mkdir results/features_original
		fi
		echo set detection consclassifier consnamelevenshtein enabled FALSE    >> $TMPFILE
		echo set detection consclassifier consnamenonumbers enabled FALSE    >> $TMPFILE
		echo set detection varclassifier objectivevalues enabled FALSE    >> $TMPFILE
		echo set detection varclassifier objectivevaluesigns enabled FALSE    >> $TMPFILE
		echo set detection varclassifier scipvartype enabled FALSE    >> $TMPFILE
		echo set write miplib2017features TRUE  >> $TMPFILE
		echo set write miplib2017featurefilepath results/features_original/featurefile >> $TMPFILE
                echo detect                        >> $TMPFILE
		echo quit                          >> $TMPFILE
            elif test $MODE = "miplibfeaturespresolved"
            then
		if test ! -e results/features_presolved
		then
		    mkdir results/features_presolved
		fi
		echo set detection consclassifier consnamelevenshtein enabled FALSE    >> $TMPFILE
		echo set detection consclassifier consnamenonumbers enabled FALSE    >> $TMPFILE
		echo set detection varclassifier objectivevalues enabled FALSE    >> $TMPFILE
		echo set detection varclassifier objectivevaluesigns enabled FALSE    >> $TMPFILE
		echo set detection varclassifier scipvartype enabled FALSE    >> $TMPFILE

		echo set write miplib2017features TRUE  >> $TMPFILE
		echo set write miplib2017featurefilepath results/features_presolved/featurefile >> $TMPFILE
                echo presolve                      >> $TMPFILE
                echo detect                        >> $TMPFILE
		echo quit                          >> $TMPFILE
            elif test $MODE = "miplibfeaturesplotsoriginal"
            then
		if test ! -e $DIR/features_original
		then
		    mkdir $DIR/features_original
		fi
		if test ! -e $DIR/features_original/decs
		then
		    mkdir $DIR/features_original/decs
		fi
		if test ! -e $DIR/features_original/matrix
		then
		    mkdir $DIR/features_original/matrix
		fi
		echo set detection consclassifier consnamelevenshtein enabled FALSE    >> $TMPFILE
		echo set detection consclassifier consnamenonumbers enabled FALSE    >> $TMPFILE
		echo set detection varclassifier objectivevalues enabled FALSE    >> $TMPFILE
		echo set detection varclassifier objectivevaluesigns enabled FALSE    >> $TMPFILE
		echo set detection varclassifier scipvartype enabled FALSE    >> $TMPFILE
		echo set presolving maxrounds 0    >> $TMPFILE
		echo set write miplib2017features TRUE  >> $TMPFILE
		echo set write miplib2017plotsanddecs TRUE  >> $TMPFILE
		echo set write miplib2017shortbasefeatures TRUE  >> $TMPFILE
#		echo set display verblevel 5 >> $TMPFILE

		echo set write miplib2017featurefilepath $DIR/features_original/featurefile >> $TMPFILE
		echo set write miplib2017matrixfilepath $DIR/features_original/matrix >> $TMPFILE
		echo set write miplib2017decompfilepath $DIR/features_original/decs >> $TMPFILE
		echo set visual colorscheme 1 >> $TMPFILE
                echo detect                        >> $TMPFILE
		echo write problem  $DIR/features_original/decs/$NAME.dec     >> $TMPFILE
		echo write problem  $DIR/features_original/decs/$NAME.gp      >> $TMPFILE
		echo quit                          >> $TMPFILE
            elif test $MODE = "miplibfeaturesplotspresolved"
            then
		if test ! -e $DIR/features_presolved
		then
		    mkdir $DIR/features_presolved
		fi
		if test ! -e $DIR/features_presolved/decs
		then
		    mkdir $DIR/features_presolved/decs
		fi
		if test ! -e $DIR/features_presolved/matrix
		then
		    mkdir $DIR/features_presolved/matrix
		fi

		echo set detection consclassifier consnamelevenshtein enabled FALSE    >> $TMPFILE
		echo set detection consclassifier consnamenonumbers enabled FALSE    >> $TMPFILE
		echo set detection varclassifier objectivevalues enabled FALSE    >> $TMPFILE
		echo set detection varclassifier objectivevaluesigns enabled FALSE    >> $TMPFILE
		echo set detection varclassifier scipvartype enabled FALSE    >> $TMPFILE

		echo set write miplib2017features TRUE  >> $TMPFILE
		echo set write miplib2017plotsanddecs TRUE  >> $TMPFILE
		echo set write miplib2017shortbasefeatures TRUE  >> $TMPFILE

		echo set write miplib2017featurefilepath $DIR/features_presolved/featurefile >> $TMPFILE
		echo set write miplib2017matrixfilepath $DIR/features_presolved/matrix >> $TMPFILE
		echo set write miplib2017decompfilepath $DIR/features_presolved/decs >> $TMPFILE
		echo set visual colorscheme 1 >> $TMPFILE
		echo presolve                      >> $TMPFILE
                echo detect                        >> $TMPFILE
		echo write trans  $DIR/features_presolved/decs/$NAME.dec     >> $TMPFILE
		echo write trans  $DIR/features_presolved/decs/$NAME.gp      >> $TMPFILE
		echo quit                          >> $TMPFILE

	    elif test $MODE = "detectionstatistics"
	    then
		echo set detection allowclassifier enabled TRUE >> $TMPFILE
		echo presolve                      >> $TMPFILE
		echo detect                        >> $TMPFILE
                echo display detectionst           >> $TMPFILE
	    elif test $MODE = "checkexistence"
	    then
		echo set presolving maxrounds 0    >> $TMPFILE
		echo presolve                      >> $TMPFILE
		echo detect                        >> $TMPFILE
		if test -f "$DECFILE"
                    then
                        BLKFILE=$DECFILE
                    fi
                    if test -f "$BLKFILE"
                    then
                        EXT=${BLKFILE##*.}
                        if test "$EXT" = "gz"
                        then
                            presol=`zgrep -A1 PRESOLVE $BLKFILE`
                        else
                            presol=`grep -A1 PRESOLVE $BLKFILE`
                        fi
                        echo $presol
                        # If the decomposition contains presolving information ...
                        echo read $BLKFILE         >> $TMPFILE
                    fi
            elif test $MODE = "bip"
            then
                echo presolve                      >> $TMPFILE
                echo write prob bip\/$NAME-dec.bip >> $TMPFILE
                echo display statistics            >> $TMPFILE
                if test $STATISTICS = "true"
                then
                    echo display additionalstatistics  >> $TMPFILE
                fi

            elif test $MODE = "detectall"
            then
                echo presolve                      >> $TMPFILE
                echo detect                        >> $TMPFILE
                mkdir -p decs/$TSTNAME.$SETNAME
                mkdir -p images/$TSTNAME.$SETNAME
                echo write alld decs\/$TSTNAME.$SETNAME dec >> $TMPFILE
                echo write alld images\/$TSTNAME.$SETNAME gp >> $TMPFILE
            else
                if test $MODE = "readdec"
                then
                    if test -f "$DECFILE"
                    then
                        BLKFILE=$DECFILE
                    fi
                    if test -f "$BLKFILE"
                    then
                        EXT=${BLKFILE##*.}
                        if test "$EXT" = "gz"
                        then
                            presol=`zgrep -A1 PRESOLVE $BLKFILE`
                        else
                            presol=`grep -A1 PRESOLVE $BLKFILE`
                        fi
                        echo $presol
                        # If the decomposition contains presolving information ...
                        if test $? = 0
                        then
                            # ... check if it belongs to a presolved problem
                            if grep -xq 1 - <<EOF
$presol
EOF
                            then
                                echo presolve      >> $TMPFILE
                            fi
                        fi
                        echo read $BLKFILE         >> $TMPFILE
                    fi
                fi
                GP_BASE=`basename $DECFILE .dec`

#                echo detect                        >> $TMPFILE
#                echo write problem $HOME\/results\/gpsBench\/$GP_BASE.gp >> $TMPFILE
#                echo write problem $HOME\/results\/decsBench\/$GP_BASE.dec >> $TMPFILE
                echo optimize                      >> $TMPFILE
                echo display statistics            >> $TMPFILE
                if test $STATISTICS = "true"
                then
                    echo display additionalstatistics  >> $TMPFILE
                fi

                if [[ $DETECTIONSTATISTICS == "true" ]]
                then
                    echo display detectionstatistics     >> $TMPFILE
                    echo explore export 0 quit           >> $TMPFILE
                fi

#               echo display solution              >> $TMPFILE
                echo checksol                      >> $TMPFILE
            fi
            echo quit                              >> $TMPFILE
            echo -----------------------------
            date
            date >>$ERRFILE
            echo -----------------------------
            date +"@03 %s"
            bash -c " ulimit -t $HARDTIMELIMIT s; ulimit -v $HARDMEMLIMIT k; ulimit -f 200000; $VALGRINDCMD $LD ../$BINNAME < $TMPFILE" 2>>$ERRFILE
            date +"@04 %s"
            echo -----------------------------
            date
            date >>$ERRFILE
            echo -----------------------------
            echo
            echo =ready=
            if test $MODE = "detectall"
            then
                mv *_*.dec decs\/
            fi
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
rm -f cipreadparsetest.cip

date >>$OUTFILE
date >>$ERRFILE

if test -e "$DONEFILE"
then
    ./evalcheck.sh $OUTFILE

    if test "$LOCK" = "true"
    then
        rm -f $RUNFILE
    fi
fi
if test "$VISU" = "true"
then
  ./writeTestsetReport.sh $SCRIPTSETTINGS $BINID $VERSION $MODE $LPS $THREADS $FEASTOL $LAST_STATISTICS $OUTFILE $RESFILE results/vbc/$TSTNAME/ $TSTNAME $SETNAME $TIMELIMIT $MEMLIMIT
fi

if test "$DETECTIONSTATISTICS" = "true"
then
    for gpfile in $(ls *--gp)
    do
        sed -i.bak "s/--pdf/\.pdf/g" $gpfile && rm *.bak
        mv $gpfile results/decomps/$(echo $gpfile | sed "s/--gp/\.gp/g")
    done
fi
