#!/usr/bin/awk -f
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
#@file    check.awk
#@brief   SCIP Check Report Generator
#@author  Thorsten Koch
#@author  Tobias Achterberg
#@author  Alexander Martin
#@author  Timo Berthold
#@author  Robert Waniek
#@author  Gregor Hendel
#@author  Christian Puchert
#@author  Gerald Gamrath
#@author  Martin Bergner
#
function abs(x)
{
   return x < 0 ? -x : x;
}
function min(x,y)
{
   return (x) < (y) ? (x) : (y);
}
function max(x,y)
{
   return (x) > (y) ? (x) : (y);
}
BEGIN {
   timegeomshift = 10.0;
   nodegeomshift = 100.0;
   pricegeomshift = 5.0;
   lpgeomshift = 2.0;
   pricecallsgeomshift = 100;
   priceprobsgeomshift = 100;
   pricevarsgeomshift = 100;
   sblpgeomshift = 0.0;
   pavshift = 0.0;
   onlyinsolufile = 0;          # should only instances be reported that are included in the .solu file?
   onlyintestfile = 0;          # should only instances be reported that are included in the .test file?  TEMPORARY HACK!
   onlypresolvereductions = 0;  # should only instances with presolve reductions be shown?
   useshortnames = 1;           # should problem name be truncated to fit into column?
   writesolufile = 0;           # should a solution file be created from the results
   printsoltimes = 0;           # should the times until first and best solution be shown
   printdecname = 0;            # should the name of the read decomposition be read?
   checksol = 1;                # should the solution check of SCIP be parsed and counted as a fail if best solution is not feasible?
   NEWSOLUFILE = "new_solufile.solu";
   infty = +1e+20;
   headerprinted = 0;
   namelength = 18;             # maximal length of instance names (can be increased)


   nprobs = 0;
   sbab = 0;
   slp = 0;
   ssim = 0;
   ssblp = 0;
   stottime = 0.0;
   stimetofirst = 0.0;
   stimetobest = 0.0;
   spricetime = 0.0;
   slptime = 0.0;
   spricecalls = 0.0;
   spriceprobs = 0.0;
   spricevars = 0.0;
   nodegeom = 0.0;
   timegeom = 0.0;
   pricegeom = 0.0;
   pricecallsgeom = 0.0;
   priceprobsgeom = 0.0;
   pricevarsgeom = 0.0;
   lpgeom = 0.0;
   timetofirstgeom = 0.0;
   timetobestgeom = 0.0;
   sblpgeom = 0.0;
   conftimegeom = 0.0;
   basictimegeom = 0.0;
   overheadtimegeom = 0.0;
   shiftednodegeom = nodegeomshift;
   shiftedtimegeom = timegeomshift;
   shiftedpricegeom = pricegeomshift;
   shiftedpricecallsgeom = pricecallsgeomshift;
   shiftedpriceprobsgeom = priceprobsgeomshift;
   shiftedpricevarsgeom = pricevarsgeomshift;
   shiftedlpgeom = pricegeomshift;
   shiftedsblpgeom = sblpgeomshift;
   shiftedconftimegeom = timegeomshift;
   shiftedbasictimegeom = timegeomshift;
   shiftedoverheadtimegeom = timegeomshift;
   shiftedtimetofirstgeom = timegeomshift;
   shiftedtimetobestgeom = timegeomshift;
   timeouttime = 0.0;
   timeouts = 0;
   failtime = 0.0;
   fail = 0;
   pass = 0;
   settings = "default";
   lpsname = "?";
   lpsversion = "?";
   scipversion = "?";
   gcgversion = "?";
   githash = "?";
   gcggithash = "?";
   conftottime = 0.0;
   overheadtottime = 0.0;
   timelimit = 0.0;
   decname = "";
   #initialize paver input file
   if( PAVFILE != "" ) {
      printf("* Trace Record Definition\n") > PAVFILE;
      printf("* InputFileName,ModelType,SolverName,Direction,ModelStatus,SolverStatus,ObjectiveValue,ObjectiveValueEstimate,SolverTime\n") > PAVFILE;
      printf("* NumberOfNodes,NumberOfIterations,NumberOfEquations,NumberOfVariables\n") > PAVFILE;
   }
}
/^IP\// {  # TEMPORARY HACK to parse .test files
   intestfile[$1] = 1;
}
/=opt=/  { solstatus[$2] = "opt"; sol[$2] = $3; }   # get optimum
/=inf=/  { solstatus[$2] = "inf"; }                 # problem infeasible (no feasible solution exists)
/=best=/ { solstatus[$2] = "best"; sol[$2] = $3; }  # get best known solution value
/=unkn=/ { solstatus[$2] = "unkn"; }                # no feasible solution known
#
# problem name
#
/^@01/ {
   filename = $2;

   n  = split ($2, a, "/");
   m = split(a[n], b, ".");
   prob = b[1];
   if( b[m] == "gz" || b[m] == "z" || b[m] == "GZ" || b[m] == "Z" )
      m--;
   if( b[m] == "lp" || b[m] == "mps" )
      m--;
   for( i = 2; i <= m; ++i )
      prob = prob "." b[i];

   if( useshortnames && length(prob) > namelength )
      shortprob = substr(prob, length(prob)-namelength-1, namelength);
   else
      shortprob = prob;

   # Escape _ for TeX
   n = split(prob, a, "_");
   pprob = a[1];
   for( i = 2; i <= n; i++ )
      pprob = pprob "\\_" a[i];
   vars = 0;
   binvars = 0;
   intvars = 0;
   implvars = 0;
   contvars = 0;
   cons = 0;
   lincons = 0;
   origvars = 0;
   origcons = 0;
   objsense = 0;
   timeout = 0;
   feasible = 0;
   pb = +infty;
   objectivelimit = +infty;
   firstpb = +infty;
   db = -infty;
   rootdb = -infty;
   simpiters = 0;
   bbnodes = 0;
   primlps = 0;
   primiter = 0;
   duallps = 0;
   dualiter = 0;
   sblps = 0;
   sbiter = 0;
   tottime = 0.0;
   timetofirst = -1.0;
   timetobest = -1.0;
   inconflict = 0;
   inconstime = 0;
   confclauses = 0;
   confliterals = 0.0;
   conftime = 0.0;
   pricetime = 0.0;
   lptime = 0.0;
   npriceprobs = 0;
   overheadtime = 0.0;
   aborted = 1;
   readerror = 0;
   gapreached = 0;
   sollimitreached = 0;
   memlimitreached = 0;
   nodelimitreached = 0;
   objlimitreached = 0;
   starttime = 0.0;
   endtime = 0.0;
   timelimit = 0.0;
   inoriginalprob = 1;
   inmasterprob = 1;
   incons = 0;
   valgrinderror = 0;
   valgrindleaks = 0;
   bestsolfeas = 1;
   blocks = 0
   rel = 0
   maxw = 0.0
   detector = "--"
   linkvars = 0
   staticvars = 0
   linkconss = 0
   pricetime = 0.0
   pricevars = 0
   priceall = 0
   pricecall = 0
}

/@03/ { starttime = $2; }
/@04/ { endtime = $2; }

/^GCG version/ {
   # get GCG version
   gcgversion = $3;

      # get git hash
   if( $(NF-1) == "[GitHash:" ) {
      split($NF, v, "]");
      gcggithash = v[1];
   }
}

/^SCIP version/ {
   # get SCIP version
   scipversion = $3;

   # get name of LP solver
   if( $13 == "SoPlex" )
      lpsname = "spx";
   else if( $13 == "CPLEX" )
      lpsname = "cpx";
   else if( $13 == "NONE" )
      lpsname = "none";
   else if( $13 == "Clp" )
      lpsname = "clp";
   else if( $13 == "MOSEK" )
      lpsname = "msk";
   else if( $13 == "Gurobi" )
      lpsname = "grb";
   else if( $13 == "NONE" )
      lpsname = "none";
   else if( $13 == "QSopt" )
      lpsname = "qso";
#   else if( $13 == "???" )
#      lpsname = "xprs";

    # get LP solver version
   if( NF >= 14 ) {
      split($14, v, "]");
      lpsversion = v[1];
   }

   # get git hash
   if( $(NF-1) == "[GitHash:" ) {
      split($NF, v, "]");
      githash = v[1];
   }
}
/^SCIP> SCIP> / { $0 = substr($0, 13, length($0)-12); }
/^SCIP> / { $0 = substr($0, 7, length($0)-6); }
/^GCG> GCG> / { $0 = substr($0, 11, length($0)-10); }
/^GCG> / { $0 = substr($0, 6, length($0)-5); }
/^loaded parameter file/ { settings = $4; sub(/<.*settings\//, "", settings); sub(/\.set>/, "", settings); }
/^parameter <limits\/time> set to/ { timelimit = $5; }
/^limits\/time =/ { timelimit = $3; }
/set limits objective/ {
   objectivelimit = $4;
}

#
# problem: master or original?
#
/^Original Program statistics:/ { inmasterprob = 0; inoriginalprob = 1; }
/^Master Program statistics:/ { inmasterprob = 1; inoriginalprob = 0; }

#
# get objective sense
#
/^  Objective sense  :/ {
   if( inoriginalprob )
   {
      if ( $4 == "minimize" )
         objsense = 1;
      if ( $4 == "maximize" )
         objsense = -1;
      # objsense is 0 otherwise
   }
}
# SCIP API version >= 9
/^  Objective        :/ {
   if( inoriginalprob )
   {
      if( objsense == 0 )
      {
         if ( $3 == "minimize," || $3 == "minimize,\r")
            objsense = 1;
         if ( $3 == "maximize," || $3 == "maximize,\r" )
            objsense = -1;

         # objsense is 0 otherwise
      }
   }
}

#
# conflict analysis
#
/^Conflict Analysis  :/ { inconflict = 1; }
/^  propagation      :/ {
   if( inconflict == 1 ) {
      conftime += $3; #confclauses += $5 + $7; confliterals += $5 * $6 + $7 * $8;
   }
}
/^  infeasible LP    :/ {
   if( inconflict == 1 ) {
      conftime += $4; #confclauses += $6 + $8; confliterals += $6 * $7 + $8 * $9;
   }
}
/^  strong branching :/ {
   if( inconflict == 1 ) {
      conftime += $4; #confclauses += $6 + $8; confliterals += $6 * $7 + $8 * $9;
   }
}
/^  pseudo solution  :/ {
   if( inconflict == 1 ) {
      conftime += $4; #confclauses += $6 + $8; confliterals += $6 * $7 + $8 * $9;
   }
}
/^  applied globally :/ {
   if( inconflict == 1 ) {
      confclauses += $7; confliterals += $7 * $8;
   }
}
/^  applied locally  :/ {
   if( inconflict == 1 ) {
      confclauses += $7; confliterals += $7 * $8;
   }
}
#
# pricing
#
/^  gcg              :/ {
   if( inmasterprob )
   {
      pricetime = $3;
      pricecall = $5;
      pricevars = $6;
   }
}

/^Separators         :/ { inconflict = 0; }
/^Constraint Timings :/ { inconstime = 1; }
#/^  logicor          :/ { if( inconstime == 1 ) { overheadtime += $3; } }
/^Propagators        :/ { inconstime = 0; }
/^  switching time   :/ { overheadtime += $4; }
#
# problem size
#
/^Presolved Problem  :/ { inoriginalprob = 0; }
/^  Variables        :/ {
   if( !inmasterprob )
   {
      if( inoriginalprob )
         origvars = $3;
      else
      {
         vars = $3;
         intvars = $6;
         implvars = $8;
         contvars = $11;
         binvars = vars - intvars - implvars - contvars;
      }
   }
}
/^  Constraints      :/ {
   if( !inmasterprob )
   {
      if( inoriginalprob )
         origcons = $3;
      else
         cons = $3;
   }
   else
   {
#      if( inoriginalprob )
         mcons = $3;
   }
}


/^Chosen decomposition/ {
   blocks = $4;
   rel = $4;
}

/^This decomposition has/ {
    maxw = $8
}


/^Matrix has / {
   blocks = $3;
   rel = $6;
}
/^GCG                : Performing Dantzig-Wolfe with [0-9]+ blocks./ {
   blocks = $6;
}

/^  blocks           :/ {
   blocks = $3;
}

/^  aggr\. blocks     :/ {
   rel = $4;
}

/^  master           :/ {
   linkconss = $9;
   linkvars = $3;
   staticvars = $4
}

/^  detector         :/ {
   detector = $3
}
#
# count number of linear constraints
#
/^Constraints        :/ {
   incons = 1;
   lincons = 0;
}
/^  knapsack         :/ {
   if( incons == 1 ) {
      n  = split ($3, a, "+");
      lincons += a[1];
   }
}
/^  setppc           :/ {
   if( incons == 1 ) {
      n  = split ($3, a, "+");
      lincons += a[1];
   }
}
/^  linear           :/ {
   if( incons  == 1 ) {
      n  = split ($3, a, "+");
      lincons += a[1];
   }
}
/^  logicor          :/ {
   if( incons == 1 ) {
      n  = split ($3, a, "+");
      lincons += a[1];
   }
}
/^  varbound         :/ {
   if( incons == 1 ) {
      n  = split ($3, a, "+");
      lincons += a[1];
   }
}
/^Constraint Timings :/ {
   incons = 0;
}

/^read problem </ {
   split($3,tmp1,"<")
   split(tmp1[2],tmp2,">")
   n = split(tmp2[1],tmp1,"/")
   sub(/\.dec$/, "", tmp1[n])
   decname =  tmp1[n];
}

#
# solution
#
/^Original Problem   : no problem exists./ { if( inoriginalprob ) readerror = 1; }
/^SCIP Status        :/ { aborted = 0; }
# grep for an objective limit induced infeasibility
/^SCIP Status        : problem is solved \[infeasible\] \(objective limit reached\)/ {
    objlimitreached = 1;
}
/solving was interrupted/ { if( inoriginalprob ) timeout = 1; }
/gap limit reached/ { if( inoriginalprob ) gapreached = 1; }
/solution limit reached/ { if( inoriginalprob ) sollimitreached = 1; }
/memory limit reached/ { if( inoriginalprob ) memlimitreached = 1; }
/node limit reached/ { if( inoriginalprob ) nodelimitreached = 1; }
/problem is solved/ { if( inoriginalprob ) timeout = 0; }
/best solution is not feasible in original problem/  { if( inoriginalprob ) bestsolfeas = 0; }

/^  First Solution   :/ {
   timetofirst = $11;
   firstpb = $4;
}
/^  Primal Bound     :/ {
   if( $4 == "infeasible" ) {
      pb = +infty;
      db = +infty;
      feasible = 0;
   }
   else if( $4 == "-" ) {
      pb = +infty;
      feasible = 0;
   }
   else {
      pb = $4;
      feasible = 1;
      timetobest = $11;
   }
}
/^Dual Bound         :/ {
   if( $4 != "-" )
      db = $4;
}
/^  Final Dual Bound :/ {
   if( !inmasterprob )
   {
      if( $5 != "-" )
         rootdb = $5;
      else
         rootdb = db;  # SCIP most likely finished during root node, perhaps due to a solution limit. the rootdb is NOT printed then, but needed later
   }

}
#
# iterations
#
/^  primal LP        :/ { simpiters += $6; lptime += $4 }
/^  dual LP          :/ { simpiters += $6; lptime += $4 }
/^  barrier LP       :/ { simpiters += $6; }
/^  nodes \(total\)    :/ { bbnodes = $4 }
/^  primal LP        :/ { primlps = $5; primiter = $6; }
/^  dual LP          :/ { duallps = $5; dualiter = $6; }
/^  strong branching :/ { sblps = $5; sbiter = $6; }
#
# time
#
/^Solving Time       :/ { tottime = $4 } # for older scip version ( < 2.0.1.3 )
/^  solving          :/ { tottime = $3 }
#
# valgrind check
#
/^==[0-9]*== ERROR SUMMARY:/       { valgrinderror = $4 }
/^==[0-9]*==    definitely lost:/  { valgrindleaks += $4 }
/^==[0-9]*==    indirectly lost:/  { valgrindleaks += $4 }
/^==[0-9]*==    possibly lost:/    { valgrindleaks += $4 }
#
# solver status overview (in order of priority):
# 1) solver broke before returning solution => abort
# 2) solver cut off the optimal solution (solu-file-value is not between primal and dual bound) => fail
#    (especially if problem is claimed to be solved but solution is not the optimal solution)
# 3) solver solved problem with the value in solu-file (if existing) => ok
# 4) solver solved problem which has no (optimal) value in solu-file => solved
#    (since we here don't detect the direction of optimization, it is possible
#     that a solver claims an optimal solution which contradicts a known feasible solution)
# 5) solver found solution better than known best solution (or no solution was known so far) => better
# 7) solver reached gaplimit or limit of number of solutions => gaplimit, sollimit
# 8) solver reached any other limit (like time or nodes) => timeout
# 9) otherwise => unknown
#
/^=ready=/ {

   #since the header depends on the parameter printsoltimes and settings it is no longer possible to print it in the BEGIN section
   if( !headerprinted ) {
      ntexcolumns = 8 + (2 * printsoltimes);


      if (TEXFILE != "") {
         #print header of tex file table
         printf("\\documentclass[leqno]{article}\n")                      >TEXFILE;
         printf("\\usepackage{a4wide}\n")                                 >TEXFILE;
         printf("\\usepackage{amsmath,amsfonts,amssymb,booktabs}\n")      >TEXFILE;
         printf("\\usepackage{supertabular}\n")                           >TEXFILE;
         printf("\\pagestyle{empty}\n\n")                                 >TEXFILE;
         printf("\\begin{document}\n\n")                                  >TEXFILE;
         printf("\\begin{center}\n")                                      >TEXFILE;
         printf("\\setlength{\\tabcolsep}{2pt}\n")                        >TEXFILE;
         printf("\\newcommand{\\g}{\\raisebox{0.25ex}{\\tiny $>$}}\n")    >TEXFILE;
         printf("\\tablehead{\n\\toprule\n")                              >TEXFILE;
         printf("Name             &  Conss &   Vars &       Dual Bound &     Primal Bound &  Gap\\%% &   Calls &   Probs &    Vars &  P-Time & LP-Time &      Nodes &     Time ") > TEXFILE;
         if( printsoltimes )
            printf(" &     To First      &    To Last   ") > TEXFILE;
         printf("\\\\\n") > TEXFILE;
         printf("\\midrule\n}\n")                                         >TEXFILE;
         printf("\\tabletail{\n\\midrule\n")                              >TEXFILE;
         printf("\\multicolumn{%d}{r} \\; continue next page \\\\\n", ntexcolumns) >TEXFILE;
         printf("\\bottomrule\n}\n")                                      >TEXFILE;
         printf("\\tablelasttail{\\bottomrule}\n")                        >TEXFILE;
         printf("\\tablecaption{SCIP with %s settings}\n",settings)       >TEXFILE;
         printf("\\begin{supertabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrrrrrrrrrrr") >TEXFILE;
         if( printsoltimes )
            printf("rr") > TEXFILE;
         printf("@{}}\n") > TEXFILE;
      }

      #print header of table when this regular expression is matched for the first time

      # prepare header
      hyphenstr = "";
      for (i = 0; i < namelength; ++i)
         hyphenstr = sprintf("%s-", hyphenstr);

      # first part: name of given length
      tablehead1 = hyphenstr;
      tablehead2 = sprintf("Name%*s", namelength-4, " ");
      tablehead3 = hyphenstr;

      tablehead1 = tablehead1"+----+--- Original --+-- Presolved --+------------------- Decomposition -----------+---------------------------+------+------- Pricing -------+---- Master ----+-------+-------+";
      tablehead2 = tablehead2"|Type| Conss |  Vars | Conss |  Vars | Det| Blks| Agg | MWhi | MConss| LVars| SVars|  Dual Bound | Primal Bound| Gap %%| Calls |  Vars |  Time |LP-Time|  Iters | Nodes |  Time |";
      tablehead3 = tablehead3"+----+-------+-------+-------+-------+----+-----+-----+------+-------+------+------+-------------+-------------+------+-------+-------+-------+-------+--------+-------+-------+";

      if( printsoltimes == 1 ) {
         tablehead1 = tablehead1"----------+---------+";
         tablehead2 = tablehead2" To First | To Best |";
         tablehead3 = tablehead3"----------+---------+";
      }

      if( printdecname == 1 ) {
         tablehead1 = tablehead1"-----Decomposition-----+";
         tablehead2 = tablehead2"          Name         |";
         tablehead3 = tablehead3"-----------------------+";
      }
      tablehead1 = tablehead1"--------\n";
      tablehead2 = tablehead2"       \n";
      tablehead3 = tablehead3"--------\n";

      printf(tablehead1);
      printf(tablehead2);
      printf(tablehead3);

      headerprinted = 1;
   }

   if( (!onlyinsolufile || solstatus[prob] != "") &&
       (!onlyintestfile || intestfile[filename]) ) {

      #avoid problems when comparing floats and integer (make everything float)
      temp = pb;
      pb = 1.0*temp;
      temp = db;
      db = 1.0*temp;
      temp = rootdb;
      rootdb = 1.0*temp;

      # if objsense could not be determined so far (output is maybe too old)
      if ( objsense == 0 )
      {
         reltol = 1e-5 * max(abs(pb),1.0);
         abstol = 1e-4;

	 # firstpb and rootdb are used to detect the direction of optimization (min or max)
	 if( timetofirst < 0.0 )
	    temp = pb;
	 else
	    temp = firstpb;
	 firstpb = 1.0*temp;

	 if ( firstpb - rootdb > max(abstol,reltol) )
	    objsense = 1;   # minimize
	 else
	    objsense = -1;  # maximize
      }

      # treat primal and dual bound differently if objective limit was reached
      if( objlimitreached && objectivelimit < +infty )
      {
          pb = objectivelimit;
          db = objectivelimit;
      }

      # modify primal bound for maximization problems without primal solution
      if ( objsense == -1 && pb >= +infty )
	 pb = -1.0 * pb;

      # modify dual bound for infeasible maximization problems
      if ( objsense == -1 && db >= +infty )
	 db = -1.0 * db;

      nprobs++;

      optimal = 0;
      markersym = "\\g";
      if( abs(pb - db) < 1e-06 && pb < infty ) {
         gap = 0.0;
         optimal = 1;
         markersym = "  ";
      }
      else if( abs(db) < 1e-06 )
         gap = -1.0;
      else if( abs(pb) < 1e-06 )
         gap = -1.0;
      else if( pb*db < 0.0 )
         gap = -1.0;
      else if( abs(db) >= +infty )
         gap = -1.0;
      else if( abs(pb) >= +infty )
         gap = -1.0;
      else
         gap = 100.0*abs((pb-db)/min(abs(db),abs(pb)));
      if( gap < 0.0 )
         gapstr = "  --  ";
      else if( gap < 1e+04 )
         gapstr = sprintf("%6.1f", gap);
      else
         gapstr = " Large";

      if( vars == 0 )
         probtype = "--";
      else if( lincons < cons )
         probtype = "CIP";
      else if( binvars == 0 && intvars == 0 )
         probtype = "LP";
      else if( contvars == 0 ) {
         if( intvars == 0 && implvars == 0 )
            probtype = "BP";
         else
            probtype = "IP";
      }
      else {
         if( intvars == 0 )
            probtype = "MBP";
         else
            probtype = "MIP";
      }

      if( aborted && endtime - starttime > timelimit && timelimit > 0.0 ) {
         timeout = 1;
         aborted = 0;
         tottime = endtime - starttime;
      }
      else if( gapreached || sollimitreached || memlimitreached || nodelimitreached )
         timeout = 0;

      if( aborted && tottime == 0.0 )
         tottime = timelimit;
      if( timelimit > 0.0 )
         tottime = min(tottime, timelimit);

      if( aborted || timetobest < 0.0 ) {
         timetofirst = tottime;
         timetobest = tottime;
      }

      lps = primlps + duallps;
      simplex = primiter + dualiter;
      stottime += tottime;
      stimetofirst += timetofirst;
      stimetobest += timetobest;
      spricetime += pricetime;
      spricecalls += pricecall;
      spriceprobs += npriceprobs;
      spricevars += pricevars;
      slptime += lptime;
      sbab += bbnodes;
      slp += lps;
      ssim += simplex;
      ssblp += sblps;
      conftottime += conftime;
      overheadtottime += overheadtime;
      basictime = tottime - conftime - overheadtime;

      nodegeom = nodegeom^((nprobs-1)/nprobs) * max(bbnodes, 1.0)^(1.0/nprobs);
      sblpgeom = sblpgeom^((nprobs-1)/nprobs) * max(sblps, 1.0)^(1.0/nprobs);
      timegeom = timegeom^((nprobs-1)/nprobs) * max(tottime, 1.0)^(1.0/nprobs);
      pricegeom = pricegeom^((nprobs-1)/nprobs) * max(pricetime, 1.0)^(1.0/nprobs);
      pricecallsgeom = pricecallsgeom^((nprobs-1)/nprobs) * max(pricecall, 1.0)^(1.0/nprobs);
      priceprobsgeom = priceprobsgeom^((nprobs-1)/nprobs) * max(npriceprobs, 1.0)^(1.0/nprobs);
      pricevarsgeom = pricevarsgeom^((nprobs-1)/nprobs) * max(pricevars, 1.0)^(1.0/nprobs);
      lpgeom = lpgeom^((nprobs-1)/nprobs) * max(lptime, 1.0)^(1.0/nprobs);

      conftimegeom = conftimegeom^((nprobs-1)/nprobs) * max(conftime, 1.0)^(1.0/nprobs);
      overheadtimegeom = overheadtimegeom^((nprobs-1)/nprobs) * max(overheadtime, 1.0)^(1.0/nprobs);
      basictimegeom = basictimegeom^((nprobs-1)/nprobs) * max(basictime, 1.0)^(1.0/nprobs);

      shiftednodegeom = shiftednodegeom^((nprobs-1)/nprobs) * max(bbnodes+nodegeomshift, 1.0)^(1.0/nprobs);
      shiftedsblpgeom = shiftedsblpgeom^((nprobs-1)/nprobs) * max(sblps+sblpgeomshift, 1.0)^(1.0/nprobs);
      shiftedtimegeom = shiftedtimegeom^((nprobs-1)/nprobs) * max(tottime+timegeomshift, 1.0)^(1.0/nprobs);
      shiftedpricegeom = shiftedpricegeom^((nprobs-1)/nprobs) * max(pricetime+pricegeomshift, 1.0)^(1.0/nprobs);
      shiftedpricecallsgeom = shiftedpricecallsgeom^((nprobs-1)/nprobs) * max(pricecall+pricecallsgeomshift, 1.0)^(1.0/nprobs);
      shiftedpriceprobsgeom = shiftedpriceprobsgeom^((nprobs-1)/nprobs) * max(npriceprobs+priceprobsgeomshift, 1.0)^(1.0/nprobs);
      shiftedpricevarsgeom = shiftedpricevarsgeom^((nprobs-1)/nprobs) * max(pricevars+pricevarsgeomshift, 1.0)^(1.0/nprobs);
      shiftedlpgeom = shiftedlpgeom^((nprobs-1)/nprobs) * max(lptime+lpgeomshift, 1.0)^(1.0/nprobs);
      shiftedconftimegeom = shiftedconftimegeom^((nprobs-1)/nprobs) * max(conftime+timegeomshift, 1.0)^(1.0/nprobs);
      shiftedoverheadtimegeom = shiftedoverheadtimegeom^((nprobs-1)/nprobs) * max(overheadtime+timegeomshift, 1.0)^(1.0/nprobs);
      shiftedbasictimegeom = shiftedbasictimegeom^((nprobs-1)/nprobs) * max(basictime+timegeomshift, 1.0)^(1.0/nprobs);

      timetobestgeom = timetobestgeom^((nprobs-1)/nprobs) * max(timetobest,1.0)^(1.0/nprobs);
      timetofirstgeom = timetofirstgeom^((nprobs-1)/nprobs) * max(timetofirst,1.0)^(1.0/nprobs);
      shiftedtimetofirstgeom = shiftedtimetofirstgeom^((nprobs-1)/nprobs) * max(timetofirst + timegeomshift, 1.0)^(1.0/nprobs);
      shiftedtimetobestgeom = shiftedtimetobestgeom^((nprobs-1)/nprobs) * max(timetobest + timegeomshift, 1.0)^(1.0/nprobs);

      status = "";
      if( readerror ) {
         status = "readerror";
         failtime += tottime;
         fail++;
      }
      else if( aborted ) {
         status = "abort";
         failtime += tottime;
         fail++;
      }
      else if( checksol && !bestsolfeas ) {
         status = "fail";
         failtime += tottime;
         fail++;
      }
      else if( solstatus[prob] == "opt" ) {
         reltol = 1e-5 * max(abs(pb),1.0);
         abstol = 1e-4;

	 # objsense = 1 -> minimize; objsense = -1 -> maximize
         if( ( objsense == 1 && (db-sol[prob] > reltol || sol[prob]-pb > reltol) ) || ( objsense == -1 && (sol[prob]-db > reltol || pb-sol[prob] > reltol) ) ) {
            status = "fail";
            failtime += tottime;
            fail++;
         }
         else {
            if( timeout || gapreached || sollimitreached || memlimitreached || nodelimitreached )
	    {
               if( timeout )
                  status = "timeout";
               else if( gapreached )
                  status = "gaplimit";
               else if( sollimitreached )
                  status = "sollimit";
               else if( memlimitreached )
                  status = "memlimit";
               else if( nodelimitreached )
                  status = "nodelimit";

               timeouttime += tottime;
               timeouts++;
            }
            else {
               if( (abs(pb - db) <= max(abstol, reltol)) && abs(pb - sol[prob]) <= reltol ) {
                  status = "ok";
                  pass++;
               }
               else {
                  status = "fail";
                  failtime += tottime;
                  fail++;
               }
            }
         }
      }
      else if( solstatus[prob] == "best" ) {
         reltol = 1e-5 * max(abs(pb),1.0);
         abstol = 1e-4;

	 # objsense = 1 -> minimize; objsense = -1 -> maximize
         if( ( objsense == 1 && db-sol[prob] > reltol) || ( objsense == -1 && sol[prob]-db > reltol) ) {
            status = "fail";
            failtime += tottime;
            fail++;
         }
         else {
            if( timeout || gapreached || sollimitreached || memlimitreached || nodelimitreached ) {
               if( (objsense == 1 && sol[prob]-pb > reltol) || (objsense == -1 && pb-sol[prob] > reltol) ) {
                  status = "better";
                  timeouttime += tottime;
                  timeouts++;
               }
               else {
                  if( timeout )
                     status = "timeout";
                  else if( gapreached )
                     status = "gaplimit";
                  else if( sollimitreached )
                     status = "sollimit";
                  else if( memlimitreached )
                     status = "memlimit";
                  else if( nodelimitreached )
                     status = "nodelimit";
                  timeouttime += tottime;
                  timeouts++;
               }
            }
            else {
               if( abs(pb - db) <= max(abstol, reltol) ) {
                  if( abs(firstpb - rootdb) <= max(abstol,reltol) )
                     status = "solved not verified";
                  else
                     status = "solved";
                  pass++;
               }
               else {
                  status = "fail";
                  failtime += tottime;
                  fail++;
               }
            }
         }
      }
      else if( solstatus[prob] == "unkn" ) {
         reltol = 1e-5 * max(abs(pb),1.0);
         abstol = 1e-4;

	 if( timeout || gapreached || sollimitreached || memlimitreached || nodelimitreached ) {
            if( abs(pb) < infty ) {
               status = "better";
               timeouttime += tottime;
               timeouts++;
            }
	    else {
	       if( timeout )
		  status = "timeout";
	       else if( gapreached )
		  status = "gaplimit";
	       else if( sollimitreached )
		  status = "sollimit";
	       else if( memlimitreached )
		  status = "memlimit";
	       else if( nodelimitreached )
		  status = "nodelimit";
	       timeouttime += tottime;
	       timeouts++;
	    }
	 }
         else if( abs(pb - db) <= max(abstol, reltol) ) {
            status = "solved not verified";
            pass++;
         }
         else {
	    status = "unknown";
         }
      }
      else if( solstatus[prob] == "inf" ) {
         if( !feasible ) {
	    if( timeout || memlimitreached || nodelimitreached ) {
	       if( timeout )
		  status = "timeout";
	       else if( memlimitreached )
		  status = "memlimit";
	       else if( nodelimitreached )
		  status = "nodelimit";
	       timeouttime += tottime;
	       timeouts++;
	    }
            else {
               status = "ok";
               pass++;
            }
         }
         else {
            status = "fail";
            failtime += tottime;
            fail++;
         }
      }
      else {
         reltol = 1e-5 * max(abs(pb),1.0);
         abstol = 1e-4;

         if( timeout || gapreached || sollimitreached || memlimitreached || nodelimitreached ) {
	    if( timeout )
	       status = "timeout";
	    else if( gapreached )
	       status = "gaplimit";
	    else if( sollimitreached )
	       status = "sollimit";
	    else if( memlimitreached )
	       status = "memlimit";
	    else if( nodelimitreached )
	       status = "nodelimit";
	    timeouttime += tottime;
	    timeouts++;
	 }
         else if( abs(pb - db) < max(abstol,reltol) ) {
            status = "solved not verified";
            pass++;
         }
         else {
               status = "unknown";
         }
      }

      if( valgrinderror > 0 ) {
         status = "fail"
         failtime += tottime;
         fail++;
      } else if( valgrindleaks > 0 ) {
         status = "fail"
         failtime += tottime;
         fail++;
      }

      if( writesolufile ) {
         if( pb == +infty && db == +infty )
            printf("=inf= %-18s\n",prob)>NEWSOLUFILE;
         else if( pb == db )
            printf("=opt= %-18s %16.9g\n",prob,pb)>NEWSOLUFILE;
         else if( pb < +infty )
            printf("=best= %-18s %16.9g\n",prob,pb)>NEWSOLUFILE;
         else
            printf("=unkn= %-18s\n",prob)>NEWSOLUFILE;
      }

      #write output to both the tex file and the console depending on whether printsoltimes is activated or not
      if( !onlypresolvereductions || origcons > cons || origvars > vars ) {
         if (TEXFILE != "") {
            printf("%-*s & %6d & %6d & %16.9g & %16.9g & %6s &  %6d &  %6d &  %6d & %7.1f & %7.1f & %s%8d &%s%7.1f",
                   namelength, pprob, cons, vars, db, pb, gapstr, pricecall, npriceprobs, pricevars, pricetime, lptime,
                   markersym, bbnodes, markersym, tottime) >TEXFILE;
            if( printsoltimes )
               printf(" & %7.1f & %7.1f", timetofirst, timetobest) > TEXFILE;
            printf("\\\\\n") > TEXFILE;
         }
         printf("%-*s %-4s %7d %7d %7d %7d %-4s %5d %5d %6.3g %7d %6d %6d %13.9g %13.9g %6s %7d %7d %7.1f %7.1f %8d %7d %7.1f ",
                namelength, shortprob, probtype, origcons, origvars, cons, vars, detector, blocks, rel, maxw, linkconss, linkvars, staticvars, db, pb, gapstr,
                pricecall, pricevars, pricetime, lptime, simpiters, bbnodes, tottime);
         if( printsoltimes )
            printf("%9.1f %9.1f ", timetofirst, timetobest);
         if( printdecname )
            printf("%-23s ", decname);
         printf("%s\n", status);
      }

      if( PAVFILE != "" ) {
         #PAVER output: see http://www.gamsworld.org/performance/paver/pprocess_submit.htm
         if( status == "abort" ) {
            modelstat = 13;
            solverstat = 13;
         } else if( status == "fail" || status == "unknown" ) {
            modelstat = 7;
            solverstat = 1;
         } else if( status == "timeout" ) {
            modelstat = abs(pb) < infty ? 8 : 14;
            solverstat = 3;
         } else if( status == "gaplimit" || status == "better" ) {
            modelstat = 8;
            solverstat = 1;
         } else if( status == "ok" || status == "solved not verified" ) {
            modelstat = 1;
            solverstat = 1;
         } else {
            modelstat = 13;
            solverstat = 13;
         }
         pavprob = prob;
         if( length(pavprob) > 25 )
              pavprob = substr(pavprob, length(pavprob)-24,25);
         if( vars == 0 )
            gamsprobtype = "LP";
         else if( lincons < cons && binvars == 0 && intvars == 0 )
            gamsprobtype = "NLP";
         else if( lincons < cons )
            gamsprobtype = "MINLP";
         else if( binvars == 0 && intvars == 0 )
            gamsprobtype = "LP";
         else
            gamsprobtype = "MIP";
         #InputFileName,ModelType,SolverName,Direction,ModelStatus,SolverStatus,ObjectiveValue,ObjectiveValueEstimate,SolverTime
         #NumberOfNodes,NumberOfIterations,NumberOfEquations,NumberOfVariables
         printf("%s,%s,SCIP_%s,%d,%d,%d,%g,%g,%g,", pavprob, gamsprobtype, settings, objsense == 1 ? 1 : 0, modelstat, solverstat, pb, db, tottime+pavshift) > PAVFILE;
         printf("%d,%d,%d,%d\n", bbnodes, simpiters, cons, vars) > PAVFILE;
      }
   }
}
END {
   shiftednodegeom -= nodegeomshift;
   shiftedsblpgeom -= sblpgeomshift;
   shiftedtimegeom -= timegeomshift;
   shiftedconftimegeom -= timegeomshift;
   shiftedoverheadtimegeom -= timegeomshift;
   shiftedbasictimegeom -= timegeomshift;
   shiftedtimetofirstgeom -= timegeomshift;
   shiftedtimetobestgeom -= timegeomshift;
   shiftedpricecallsgeom -= pricecallsgeomshift;
   shiftedpriceprobsgeom -= priceprobsgeomshift;
   shiftedpricevarsgeom -= pricevarsgeomshift;
   if (TEXFILE != "" ) {
      printf("\\midrule\n")                                                 >TEXFILE;
      printf("%-11s (%2d) &        &        &                  &                  &        & %7d & %7d & %7d &%8.1f &%8.1f &  %9d & %8.1f ",
             "Total", nprobs, spricecalls, spriceprobs, spricevars, spricetime, slptime, sbab, stottime) >TEXFILE;
      if( printsoltimes )
         printf(" & %8.1f & %8.1f", stimetofirst, stimetobest) > TEXFILE;
      printf("\\\\\n") > TEXFILE;
      printf("%-11s      &        &        &                  &                  &        & %7d & %7d & %7d &%8.1f &%8.1f &  %9d & %8.1f ",
             "Geom. Mean", pricecallsgeom, priceprobsgeom, pricevarsgeom, pricegeom, lpgeom, nodegeom, timegeom) >TEXFILE;
      if( printsoltimes )
         printf(" & %8.1f & %8.1f", timetofirstgeom, timetobestgeom) > TEXFILE;
      printf("\\\\\n") > TEXFILE;
      printf("%-13s    &        &        &                  &                  &        & %7d & %7d & %7d &%8.1f &%8.1f &  %9d & %8.1f ",
             "Shifted Geom.", shiftedpricecallsgeom, shiftedpriceprobsgeom, shiftedpricevarsgeom, shiftedpricegeom,
	     shiftedlpgeom, shiftednodegeom, shiftedtimegeom) >TEXFILE;
      if( printsoltimes )
         printf(" & %8.1f & %8.1f", shiftedtimetofirstgeom, shiftedtimetobestgeom) > TEXFILE;
      printf("\\\\\n") > TEXFILE;
   }
   printf(tablehead3);
   printf("\n");

   tablebottom1 = "------------------------------[Nodes]---------------[Time]----------[Pricing-Time]--------[LP-Time]-------[Pricing-Calls]--";
   tablebottom2 = "  Cnt  Pass  Time  Fail  total(k)     geom.     total     geom.     total     geom.     total     geom.    total     geom. ";
   tablebottom3 = "---------------------------------------------------------------------------------------------------------------------------";

   if( printsoltimes ) {
      tablebottom1 = tablebottom1"--------[ToFirst]-----------[ToLast]-----";
      tablebottom2 = tablebottom2"     total     geom.     total     geom.";
      tablebottom3 = tablebottom3"-----------------------------------------";
   }

   tablebottom1 = tablebottom1"\n";
   tablebottom2 = tablebottom2"\n";
   tablebottom3 = tablebottom3"\n";

   printf(tablebottom1);
   printf(tablebottom2);
   printf(tablebottom3);

   printf("%5d %5d %5d %5d %9d %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9d %9.1f ",
	  nprobs, pass, timeouts, fail, sbab / 1000, nodegeom, stottime, timegeom, spricetime, pricegeom,
	  slptime, lpgeom, spricecalls, pricecallsgeom);
   if( printsoltimes )
      printf("%9.1f %9.1f %9.1f %9.1f", stimetofirst, timetofirstgeom, stimetobest, timetobestgeom);

   printf("\n");
   printf(" shifted geom. [%4d/%4.1f/%3.1f/%3.1f]%9.1f           %9.1f           %9.1f           %9.1f           %9.1f ",
	  nodegeomshift, timegeomshift, pricegeomshift, lpgeomshift, shiftednodegeom, shiftedtimegeom,
	  shiftedpricegeom, shiftedlpgeom, shiftedpriceprobsgeom);

   if( printsoltimes )
      printf("          %9.1f           %9.1f ", shiftedtimetofirstgeom, shiftedtimetobestgeom);
   printf("\n");
   printf(tablebottom3);

   if (TEXFILE != "" ) {
      printf("\\noalign{\\vspace{6pt}}\n")                                  >TEXFILE;
      printf("\\end{supertabular*}\n")                                      >TEXFILE;
      printf("\\end{center}\n")                                             >TEXFILE;
      printf("\\end{document}\n")                                           >TEXFILE;
   }

   printf("@02 timelimit: %g\n", timelimit);
   printf("@01 GCG(%s)SCIP(%s)%s(%s):%s\n", gcgversion, scipversion, lpsname, lpsversion, settings);
}
