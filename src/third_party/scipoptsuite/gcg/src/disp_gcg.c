/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program                         */
/*          GCG --- Generic Column Generation                                */
/*                  a Dantzig-Wolfe decomposition based extension            */
/*                  of the branch-cut-and-price framework                    */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/* Copyright (C) 2010-2022 Operations Research, RWTH Aachen University       */
/*                         Zuse Institute Berlin (ZIB)                       */
/*                                                                           */
/* This program is free software; you can redistribute it and/or             */
/* modify it under the terms of the GNU Lesser General Public License        */
/* as published by the Free Software Foundation; either version 3            */
/* of the License, or (at your option) any later version.                    */
/*                                                                           */
/* This program is distributed in the hope that it will be useful,           */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/* GNU Lesser General Public License for more details.                       */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with this program; if not, write to the Free Software               */
/* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.*/
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   disp_gcg.c
 * @ingroup DISPLAYS
 * @brief  GCG display columns
 * @author Gerald Gamrath
 * @author Christian Puchert
 * @author Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "disp_gcg.h"
#include "scip/disp_default.h"
#include "gcg.h"

#include "relax_gcg.h"
#include "pricer_gcg.h"

#define DISP_NAME_SOLFOUND      "solfound"
#define DISP_DESC_SOLFOUND      "letter that indicates the heuristic, that found the solution"
#define DISP_HEAD_SOLFOUND      "  "
#define DISP_WIDT_SOLFOUND      2
#define DISP_PRIO_SOLFOUND      80000
#define DISP_POSI_SOLFOUND      0
#define DISP_STRI_SOLFOUND      FALSE

#define DISP_NAME_TIME          "time"
#define DISP_DESC_TIME          "total solution time"
#define DISP_HEAD_TIME          "time"
#define DISP_WIDT_TIME          5
#define DISP_PRIO_TIME          4000
#define DISP_POSI_TIME          50
#define DISP_STRI_TIME          TRUE

#define DISP_NAME_NNODES        "nnodes"
#define DISP_DESC_NNODES        "number of processed nodes"
#define DISP_HEAD_NNODES        "node"
#define DISP_WIDT_NNODES        7
#define DISP_PRIO_NNODES        100000
#define DISP_POSI_NNODES        100
#define DISP_STRI_NNODES        TRUE

#define DISP_NAME_NODESLEFT     "nodesleft"
#define DISP_DESC_NODESLEFT     "number of unprocessed nodes"
#define DISP_HEAD_NODESLEFT     "left"
#define DISP_WIDT_NODESLEFT     7
#define DISP_PRIO_NODESLEFT     90000
#define DISP_POSI_NODESLEFT     200
#define DISP_STRI_NODESLEFT     TRUE

#define DISP_NAME_LPITERATIONS  "lpiterations"
#define DISP_DESC_LPITERATIONS  "number of simplex iterations"
#define DISP_HEAD_LPITERATIONS  "LP iter"
#define DISP_WIDT_LPITERATIONS  7
#define DISP_PRIO_LPITERATIONS  1000
#define DISP_POSI_LPITERATIONS  1000
#define DISP_STRI_LPITERATIONS  TRUE

#define DISP_NAME_SLPITERATIONS "sumlpiterations"
#define DISP_DESC_SLPITERATIONS "number of simplex iterations in master and pricing problems"
#define DISP_HEAD_SLPITERATIONS "SLP iter"
#define DISP_WIDT_SLPITERATIONS 8
#define DISP_PRIO_SLPITERATIONS 30000
#define DISP_POSI_SLPITERATIONS 1050
#define DISP_STRI_SLPITERATIONS TRUE

#define DISP_NAME_LPAVGITERS    "lpavgiterations"
#define DISP_DESC_LPAVGITERS    "average number of LP iterations since the last output line"
#define DISP_HEAD_LPAVGITERS    "LP it/n"
#define DISP_WIDT_LPAVGITERS    7
#define DISP_PRIO_LPAVGITERS    25000
#define DISP_POSI_LPAVGITERS    1400
#define DISP_STRI_LPAVGITERS    TRUE

#define DISP_NAME_LPCOND        "lpcond"
#define DISP_DESC_LPCOND        "estimate on condition number of LP solution"
#define DISP_HEAD_LPCOND        "LP cond"
#define DISP_WIDT_LPCOND        7
#define DISP_PRIO_LPCOND        0
#define DISP_POSI_LPCOND        1450
#define DISP_STRI_LPCOND        TRUE

#define DISP_NAME_MEMUSED       "memused"
#define DISP_DESC_MEMUSED       "total number of bytes used in block memory"
#define DISP_HEAD_MEMUSED       "mem"
#define DISP_WIDT_MEMUSED       5
#define DISP_PRIO_MEMUSED       20000
#define DISP_POSI_MEMUSED       1500
#define DISP_STRI_MEMUSED       TRUE

#define DISP_NAME_DEPTH         "depth"
#define DISP_DESC_DEPTH         "depth of current node"
#define DISP_HEAD_DEPTH         "depth"
#define DISP_WIDT_DEPTH         5
#define DISP_PRIO_DEPTH         500
#define DISP_POSI_DEPTH         2000
#define DISP_STRI_DEPTH         TRUE

#define DISP_NAME_MAXDEPTH      "maxdepth"
#define DISP_DESC_MAXDEPTH      "maximal depth of all processed nodes"
#define DISP_HEAD_MAXDEPTH      "mdpt"
#define DISP_WIDT_MAXDEPTH      5
#define DISP_PRIO_MAXDEPTH      5000
#define DISP_POSI_MAXDEPTH      2100
#define DISP_STRI_MAXDEPTH      TRUE

#define DISP_NAME_PLUNGEDEPTH   "plungedepth"
#define DISP_DESC_PLUNGEDEPTH   "current plunging depth"
#define DISP_HEAD_PLUNGEDEPTH   "pdpt"
#define DISP_WIDT_PLUNGEDEPTH   5
#define DISP_PRIO_PLUNGEDEPTH   10
#define DISP_POSI_PLUNGEDEPTH   2200
#define DISP_STRI_PLUNGEDEPTH   TRUE

#define DISP_NAME_NFRAC         "nfrac"
#define DISP_DESC_NFRAC         "number of fractional variables in the current solution"
#define DISP_HEAD_NFRAC         "frac"
#define DISP_WIDT_NFRAC         5
#define DISP_PRIO_NFRAC         700
#define DISP_POSI_NFRAC         2500
#define DISP_STRI_NFRAC         TRUE

#define DISP_NAME_NEXTERNCANDS  "nexternbranchcands"
#define DISP_DESC_NEXTERNCANDS  "number of extern branching variables in the current node"
#define DISP_HEAD_NEXTERNCANDS  "extbr"
#define DISP_WIDT_NEXTERNCANDS  5
#define DISP_PRIO_NEXTERNCANDS  650
#define DISP_POSI_NEXTERNCANDS  2600
#define DISP_STRI_NEXTERNCANDS  TRUE

#define DISP_NAME_VARS          "vars"
#define DISP_DESC_VARS          "number of variables in the original problem"
#define DISP_HEAD_VARS          "ovars"
#define DISP_WIDT_VARS          5
#define DISP_PRIO_VARS          3000
#define DISP_POSI_VARS          3000
#define DISP_STRI_VARS          TRUE

#define DISP_NAME_CONSS         "conss"
#define DISP_DESC_CONSS         "number of globally valid constraints in the problem"
#define DISP_HEAD_CONSS         "ocons"
#define DISP_WIDT_CONSS         5
#define DISP_PRIO_CONSS         3100
#define DISP_POSI_CONSS         3100
#define DISP_STRI_CONSS         TRUE

#define DISP_NAME_CURCONSS      "curconss"
#define DISP_DESC_CURCONSS      "number of enabled constraints in current node"
#define DISP_HEAD_CURCONSS      "ccons"
#define DISP_WIDT_CURCONSS      5
#define DISP_PRIO_CURCONSS      600
#define DISP_POSI_CURCONSS      3200
#define DISP_STRI_CURCONSS      TRUE

#define DISP_NAME_CURCOLS       "curcols"
#define DISP_DESC_CURCOLS       "number of LP columns in current node"
#define DISP_HEAD_CURCOLS       "cols"
#define DISP_WIDT_CURCOLS       5
#define DISP_PRIO_CURCOLS       800
#define DISP_POSI_CURCOLS       3300
#define DISP_STRI_CURCOLS       TRUE

#define DISP_NAME_CURROWS       "currows"
#define DISP_DESC_CURROWS       "number of LP rows in current node"
#define DISP_HEAD_CURROWS       "rows"
#define DISP_WIDT_CURROWS       5
#define DISP_PRIO_CURROWS       900
#define DISP_POSI_CURROWS       3400
#define DISP_STRI_CURROWS       TRUE

#define DISP_NAME_CUTS          "cuts"
#define DISP_DESC_CUTS          "total number of cuts applied to the original LPs"
#define DISP_HEAD_CUTS          "ocuts"
#define DISP_WIDT_CUTS          5
#define DISP_PRIO_CUTS          100
#define DISP_POSI_CUTS          3500
#define DISP_STRI_CUTS          TRUE

#define DISP_NAME_SEPAROUNDS    "separounds"
#define DISP_DESC_SEPAROUNDS    "number of separation rounds performed at the current node"
#define DISP_HEAD_SEPAROUNDS    "sepa"
#define DISP_WIDT_SEPAROUNDS    4
#define DISP_PRIO_SEPAROUNDS    100
#define DISP_POSI_SEPAROUNDS    3600
#define DISP_STRI_SEPAROUNDS    TRUE

#define DISP_NAME_POOLSIZE      "poolsize"
#define DISP_DESC_POOLSIZE      "number of LP rows in the cut pool"
#define DISP_HEAD_POOLSIZE      "pool"
#define DISP_WIDT_POOLSIZE      5
#define DISP_PRIO_POOLSIZE      50
#define DISP_POSI_POOLSIZE      3700
#define DISP_STRI_POOLSIZE      TRUE

#define DISP_NAME_CONFLICTS     "conflicts"
#define DISP_DESC_CONFLICTS     "total number of conflicts found in conflict analysis"
#define DISP_HEAD_CONFLICTS     "confs"
#define DISP_WIDT_CONFLICTS     5
#define DISP_PRIO_CONFLICTS     2000
#define DISP_POSI_CONFLICTS     4000
#define DISP_STRI_CONFLICTS     TRUE

#define DISP_NAME_STRONGBRANCHS "strongbranchs"
#define DISP_DESC_STRONGBRANCHS "total number of strong branching calls"
#define DISP_HEAD_STRONGBRANCHS "strbr"
#define DISP_WIDT_STRONGBRANCHS 5
#define DISP_PRIO_STRONGBRANCHS 1000
#define DISP_POSI_STRONGBRANCHS 5000
#define DISP_STRI_STRONGBRANCHS TRUE

#define DISP_NAME_PSEUDOOBJ     "pseudoobj"
#define DISP_DESC_PSEUDOOBJ     "current pseudo objective value"
#define DISP_HEAD_PSEUDOOBJ     "pseudoobj"
#define DISP_WIDT_PSEUDOOBJ     14
#define DISP_PRIO_PSEUDOOBJ     300
#define DISP_POSI_PSEUDOOBJ     6000
#define DISP_STRI_PSEUDOOBJ     TRUE

#define DISP_NAME_LPOBJ         "lpobj"
#define DISP_DESC_LPOBJ         "current LP objective value"
#define DISP_HEAD_LPOBJ         "lpobj"
#define DISP_WIDT_LPOBJ         14
#define DISP_PRIO_LPOBJ         300
#define DISP_POSI_LPOBJ         6500
#define DISP_STRI_LPOBJ         TRUE

#define DISP_NAME_CURDUALBOUND  "curdualbound"
#define DISP_DESC_CURDUALBOUND  "dual bound of current node"
#define DISP_HEAD_CURDUALBOUND  "curdualbound"
#define DISP_WIDT_CURDUALBOUND  14
#define DISP_PRIO_CURDUALBOUND  400
#define DISP_POSI_CURDUALBOUND  7000
#define DISP_STRI_CURDUALBOUND  TRUE

#define DISP_NAME_ESTIMATE      "estimate"
#define DISP_DESC_ESTIMATE      "estimated value of feasible solution in current node"
#define DISP_HEAD_ESTIMATE      "estimate"
#define DISP_WIDT_ESTIMATE      14
#define DISP_PRIO_ESTIMATE      200
#define DISP_POSI_ESTIMATE      7500
#define DISP_STRI_ESTIMATE      TRUE

#define DISP_NAME_AVGDUALBOUND  "avgdualbound"
#define DISP_DESC_AVGDUALBOUND  "average dual bound of all unprocessed nodes"
#define DISP_HEAD_AVGDUALBOUND  "avgdualbound"
#define DISP_WIDT_AVGDUALBOUND  14
#define DISP_PRIO_AVGDUALBOUND  40
#define DISP_POSI_AVGDUALBOUND  8000
#define DISP_STRI_AVGDUALBOUND  TRUE

#define DISP_NAME_DUALBOUND     "dualbound"
#define DISP_DESC_DUALBOUND     "current global dual bound"
#define DISP_HEAD_DUALBOUND     "dualbound"
#define DISP_WIDT_DUALBOUND     14
#define DISP_PRIO_DUALBOUND     70000
#define DISP_POSI_DUALBOUND     9000
#define DISP_STRI_DUALBOUND     TRUE

#define DISP_NAME_PRIMALBOUND   "primalbound"
#define DISP_DESC_PRIMALBOUND   "current primal bound"
#define DISP_HEAD_PRIMALBOUND   "primalbound"
#define DISP_WIDT_PRIMALBOUND   14
#define DISP_PRIO_PRIMALBOUND   80000
#define DISP_POSI_PRIMALBOUND   10000
#define DISP_STRI_PRIMALBOUND   TRUE

#define DISP_NAME_CUTOFFBOUND   "cutoffbound"
#define DISP_DESC_CUTOFFBOUND   "current cutoff bound"
#define DISP_HEAD_CUTOFFBOUND   "cutoffbound"
#define DISP_WIDT_CUTOFFBOUND   14
#define DISP_PRIO_CUTOFFBOUND   10
#define DISP_POSI_CUTOFFBOUND   10100
#define DISP_STRI_CUTOFFBOUND   TRUE

#define DISP_NAME_DEGENERACY    "degeneracy"
#define DISP_DESC_DEGENERACY    "current average degeneracy"
#define DISP_HEAD_DEGENERACY    "deg"
#define DISP_WIDT_DEGENERACY    8
#define DISP_PRIO_DEGENERACY    40000
#define DISP_POSI_DEGENERACY    18000
#define DISP_STRI_DEGENERACY    TRUE

#define DISP_NAME_GAP           "gap"
#define DISP_DESC_GAP           "current (relative) gap using |primal-dual|/MIN(|dual|,|primal|)"
#define DISP_HEAD_GAP           "gap"
#define DISP_WIDT_GAP           8
#define DISP_PRIO_GAP           60000
#define DISP_POSI_GAP           20000
#define DISP_STRI_GAP           TRUE

#define DISP_NAME_PRIMALGAP          "primalgap"
#define DISP_DESC_PRIMALGAP          "current (relative) gap using |primal-dual|/|primal|"
#define DISP_HEAD_PRIMALGAP          "primgap"
#define DISP_WIDT_PRIMALGAP          8
#define DISP_PRIO_PRIMALGAP          20000
#define DISP_POSI_PRIMALGAP          21000
#define DISP_STRI_PRIMALGAP          TRUE

#define DISP_NAME_NSOLS         "nsols"
#define DISP_DESC_NSOLS         "current number of solutions found"
#define DISP_HEAD_NSOLS         "nsols"
#define DISP_WIDT_NSOLS         5
#define DISP_PRIO_NSOLS         0
#define DISP_POSI_NSOLS         30000
#define DISP_STRI_NSOLS         TRUE

#define DISP_NAME_MLPITERATIONS  "mlpiterations"
#define DISP_DESC_MLPITERATIONS  "number of simplex iterations in the master"
#define DISP_HEAD_MLPITERATIONS  "MLP iter"
#define DISP_WIDT_MLPITERATIONS  8
#define DISP_PRIO_MLPITERATIONS  80000
#define DISP_POSI_MLPITERATIONS  1100
#define DISP_STRI_MLPITERATIONS  TRUE

#define DISP_NAME_MVARS         "mvars"
#define DISP_DESC_MVARS         "number of variables in the master problem"
#define DISP_HEAD_MVARS         "mvars"
#define DISP_WIDT_MVARS         5
#define DISP_PRIO_MVARS         70000
#define DISP_POSI_MVARS         3050
#define DISP_STRI_MVARS         TRUE

#define DISP_NAME_MCONSS        "mconss"
#define DISP_DESC_MCONSS        "number of globally valid constraints in the master problem"
#define DISP_HEAD_MCONSS        "mcons"
#define DISP_WIDT_MCONSS        5
#define DISP_PRIO_MCONSS        70000
#define DISP_POSI_MCONSS        3150
#define DISP_STRI_MCONSS        TRUE

#define DISP_NAME_MCUTS         "mcuts"
#define DISP_DESC_MCUTS         "total number of cuts applied to the master LPs"
#define DISP_HEAD_MCUTS         "mcuts"
#define DISP_WIDT_MCUTS         5
#define DISP_PRIO_MCUTS         80000
#define DISP_POSI_MCUTS         3550
#define DISP_STRI_MCUTS         TRUE

/*
 * Callback methods
 */

/** copy method for display plugins (called when SCIP copies plugins) */
static
SCIP_DECL_DISPCOPY(dispCopyGcg)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(disp != NULL);

   /* call inclusion method of default SCIP display plugin */
   SCIP_CALL( SCIPincludeDispDefault(scip) );

   return SCIP_OKAY;
}

/** solving process initialization method of display column (called when branch and bound process is about to begin) */
static
SCIP_DECL_DISPINITSOL(SCIPdispInitsolSolFound)
{  /*lint --e{715}*/

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_SOLFOUND) == 0);
   assert(scip != NULL);

   SCIPdispSetData(disp, (SCIP_DISPDATA*)SCIPgetBestSol(scip));

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for character of best solution */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputSolFound)
{  /*lint --e{715}*/
   SCIP* masterprob;
   SCIP_SOL* origsol;
   SCIP_SOL* mastersol;
   SCIP_DISPDATA* dispdata;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_SOLFOUND) == 0);
   assert(scip != NULL);

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   origsol = SCIPgetBestSol(scip);
   if( origsol == NULL )
      SCIPdispSetData(disp, NULL);

   if( SCIPgetStage(masterprob) >= SCIP_STAGE_SOLVING )
      mastersol = SCIPgetBestSol(masterprob);
   else
      mastersol = NULL;

   dispdata = SCIPdispGetData(disp);
   if( origsol != (SCIP_SOL*)dispdata )
   {
      SCIPinfoMessage(scip, file, "%c", (SCIPgetSolHeur(scip, origsol) == NULL ? '*'
            : SCIPheurGetDispchar(SCIPgetSolHeur(scip, origsol))));
      /* If the solution was obtained in the master problem, display whether it came from its
       * LP relaxation or from the master heuristics */
      if( SCIPgetSolHeur(scip, origsol) == NULL && (mastersol != NULL) )
      {
         SCIPinfoMessage(scip, file, "%c", (SCIPgetSolHeur(masterprob, mastersol) == NULL ? '*'
               : SCIPheurGetDispchar(SCIPgetSolHeur(masterprob, mastersol))));
      }
      else
      {
         SCIPinfoMessage(scip, file, " ");
      }
      SCIPdispSetData(disp, (SCIP_DISPDATA*)origsol);
   }
   else
      SCIPinfoMessage(scip, file, "  ");

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for solving time */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputSolvingTime)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_TIME) == 0);
   assert(scip != NULL);

   SCIPdispTime(SCIPgetMessagehdlr(scip), file, SCIPgetSolvingTime(scip), DISP_WIDT_TIME);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of nodes */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNNodes)
{  /*lint --e{715}*/
   SCIP* masterprob;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_NNODES) == 0);
   assert(scip != NULL);

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   if( SCIPgetStage(masterprob) >= SCIP_STAGE_SOLVING && GCGgetDecompositionMode(scip) != DEC_DECMODE_DANTZIGWOLFE )
      SCIPdispLongint(SCIPgetMessagehdlr(scip), file, SCIPgetNNodes(masterprob), DISP_WIDT_NNODES);
   else
      SCIPdispLongint(SCIPgetMessagehdlr(scip), file, SCIPgetNNodes(scip), DISP_WIDT_NNODES);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of open nodes */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNNodesLeft)
{  /*lint --e{715}*/
   SCIP* masterprob;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_NODESLEFT) == 0);
   assert(scip != NULL);

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   if( SCIPgetStage(masterprob) >= SCIP_STAGE_SOLVING && GCGgetDecompositionMode(scip) != DEC_DECMODE_DANTZIGWOLFE )
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNNodesLeft(masterprob), DISP_WIDT_NODESLEFT);
   else
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNNodesLeft(scip), DISP_WIDT_NODESLEFT);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of LP iterations */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNLPIterations)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_LPITERATIONS) == 0);
   assert(scip != NULL);

   SCIPdispLongint(SCIPgetMessagehdlr(scip), file, SCIPgetNLPIterations(scip), DISP_WIDT_LPITERATIONS);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of average LP iterations */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNLPAvgIters)
{  /*lint --e{715}*/
   SCIP* masterprob;
   SCIP_Longint nnodes;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_LPAVGITERS) == 0);
   assert(scip != NULL);

   /**@todo Currently we are using the total number of nodes to compute the average LP iterations number. The reason for
    *       that is, that for the LP iterations only the total number (over all runs) are stored in the statistics. It
    *       would be nicer if the statistic also stores the number of LP iterations for the current run similar to the
    *       nodes.
    */

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   if( SCIPgetStage(masterprob) >= SCIP_STAGE_SOLVING && GCGgetDecompositionMode(scip) != DEC_DECMODE_DANTZIGWOLFE )
      nnodes = SCIPgetNNodes(GCGgetMasterprob(scip));
   else
      nnodes = SCIPgetNNodes(scip);

   if( nnodes < 2 )
      SCIPinfoMessage(scip, file, "     - ");
   else
      SCIPinfoMessage(scip, file, "%6.1f ",
         (SCIPgetNLPIterations(GCGgetMasterprob(scip)) - SCIPgetNRootLPIterations(GCGgetMasterprob(scip)))
         / (SCIP_Real)(SCIPgetNNodes(GCGgetMasterprob(scip)) - 1) );

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for estimate on LP condition */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputLPCondition)
{  /*lint --e{715}*/
   SCIP_LPI* lpi;
   SCIP_Real cond;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_LPCOND) == 0);
   assert(scip != NULL);

   SCIP_CALL( SCIPgetLPI(scip, &lpi) );
   if( lpi == NULL )
   {
      SCIPinfoMessage(scip, file, "     - ");
      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPlpiGetRealSolQuality(lpi, SCIP_LPSOLQUALITY_ESTIMCONDITION, &cond) );

   if( cond == SCIP_INVALID )  /*lint !e777*/
      SCIPinfoMessage(scip, file, "   n/a ");
   else
      SCIPinfoMessage(scip, file, "%.1e", cond);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for depth */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputDepth)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_DEPTH) == 0);
   assert(scip != NULL);

   SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetDepth(scip), DISP_WIDT_DEPTH);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputMemUsed)
{  /*lint --e{715}*/
   SCIP_Longint memused;
   int i;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_MEMUSED) == 0);
   assert(scip != NULL);

   memused = SCIPgetMemUsed(scip);
   memused += SCIPgetMemUsed(GCGgetMasterprob(scip));
   for( i = 0; i < GCGgetNPricingprobs(scip); i++ )
   {
      memused += SCIPgetMemUsed(GCGgetPricingprob(scip, i));
   }

   SCIPdispLongint(SCIPgetMessagehdlr(scip), file, memused, DISP_WIDT_MEMUSED);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for maximal depth */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputMaxDepth)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_MAXDEPTH) == 0);
   assert(scip != NULL);

   SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetMaxDepth(scip), DISP_WIDT_MAXDEPTH);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for plunging depth */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputPlungeDepth)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_PLUNGEDEPTH) == 0);
   assert(scip != NULL);

   SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetPlungeDepth(scip), DISP_WIDT_PLUNGEDEPTH);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of LP branch candidates */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNFrac)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_NFRAC) == 0);
   assert(scip != NULL);

   if( SCIPhasCurrentNodeLP(scip) && SCIPgetLPSolstat(scip) == SCIP_LPSOLSTAT_OPTIMAL )
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNLPBranchCands(scip), DISP_WIDT_NFRAC);
   else
      SCIPinfoMessage(scip, file, "   - ");

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of external branch candidates */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNExternCands)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_NEXTERNCANDS) == 0);
   assert(scip != NULL);

   SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNExternBranchCands(scip), DISP_WIDT_NEXTERNCANDS);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of variables */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNVars)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_VARS) == 0);
   assert(scip != NULL);

   SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNVars(scip), DISP_WIDT_VARS);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of constraints */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNConss)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_CONSS) == 0);
   assert(scip != NULL);

   SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNConss(scip), DISP_WIDT_CONSS);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of enabled constraints */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNCurConss)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_CURCONSS) == 0);
   assert(scip != NULL);

   SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNEnabledConss(scip), DISP_WIDT_CURCONSS);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of columns in the LP */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNCurCols)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_CURCOLS) == 0);
   assert(scip != NULL);

   SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNLPCols(scip), DISP_WIDT_CURCOLS);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of rows in the LP */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNCurRows)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_CURROWS) == 0);
   assert(scip != NULL);

   SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNLPRows(scip), DISP_WIDT_CURROWS);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of applied cuts */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNAppliedCuts)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_CUTS) == 0);
   assert(scip != NULL);

   SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNCutsApplied(scip), DISP_WIDT_CUTS);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of separation rounds */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNSepaRounds)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_SEPAROUNDS) == 0);
   assert(scip != NULL);

   if( SCIPgetStage(GCGgetMasterprob(scip)) == SCIP_STAGE_SOLVING )
   {
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNSepaRounds(GCGgetMasterprob(scip)), DISP_WIDT_SEPAROUNDS);
   }
   else
   {
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, 0, DISP_WIDT_SEPAROUNDS);
   }

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of current rows in the cut pool */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputCutPoolSize)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_POOLSIZE) == 0);
   assert(scip != NULL);

   if( SCIPgetStage(GCGgetMasterprob(scip)) >= SCIP_STAGE_SOLVING )
   {
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNPoolCuts(GCGgetMasterprob(scip)), DISP_WIDT_POOLSIZE);
   }
   else
   {
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, 0, DISP_WIDT_POOLSIZE);
   }

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of conflicts */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNConflicts)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_CONFLICTS) == 0);
   assert(scip != NULL);

   SCIPdispLongint(SCIPgetMessagehdlr(scip), file, SCIPgetNConflictConssApplied(scip), DISP_WIDT_CONFLICTS);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of strong branchings */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNStrongbranchs)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_STRONGBRANCHS) == 0);
   assert(scip != NULL);

   SCIPdispLongint(SCIPgetMessagehdlr(scip), file, SCIPgetNStrongbranchs(scip), DISP_WIDT_STRONGBRANCHS);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for pseudo objective value */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputPseudoObjval)
{  /*lint --e{715}*/
   SCIP_Real pseudoobj;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_PSEUDOOBJ) == 0);
   assert(scip != NULL);

   pseudoobj = SCIPgetPseudoObjval(scip);

   if( SCIPisInfinity(scip, -pseudoobj) )
      SCIPinfoMessage(scip, file, "      --      ");
   else if( SCIPisInfinity(scip, pseudoobj) )
      SCIPinfoMessage(scip, file, "    cutoff    ");
   else
      SCIPinfoMessage(scip, file, "%13.6e ", pseudoobj);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for LP objective value */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputLPObjval)
{  /*lint --e{715}*/

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_LPOBJ) == 0);
   assert(scip != NULL);

   if( SCIPgetStage(GCGgetMasterprob(scip)) != SCIP_STAGE_SOLVING || SCIPgetLPSolstat(GCGgetMasterprob(scip)) == SCIP_LPSOLSTAT_NOTSOLVED )
   {
      SCIPinfoMessage(scip, file, "      --      ");
   }
   else
   {
      SCIP_Real lpobj = SCIPgetLPObjval(GCGgetMasterprob(scip));
      if( SCIPisInfinity(scip, -lpobj) )
         SCIPinfoMessage(scip, file, "      --      ");
      else if( SCIPisInfinity(scip, lpobj) )
         SCIPinfoMessage(scip, file, "    cutoff    ");
      else
         SCIPinfoMessage(scip, file, "%13.6e ", lpobj);
   }

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for the current dualbound */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputCurDualbound)
{  /*lint --e{715}*/
   SCIP_Real curdualbound;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_CURDUALBOUND) == 0);
   assert(scip != NULL);

   curdualbound = SCIPgetLocalDualbound(scip);

   if( SCIPisInfinity(scip, (SCIP_Real) SCIPgetObjsense(scip) * curdualbound ) )
      SCIPinfoMessage(scip, file, "    cutoff    ");
   else if( SCIPisInfinity(scip, -1.0 * (SCIP_Real) SCIPgetObjsense(scip) * curdualbound ) )
      SCIPinfoMessage(scip, file, "      --      ");
   else
      SCIPinfoMessage(scip, file, "%13.6e ", curdualbound);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for estimate of best primal solution w.r.t. original
 *  problem contained in current subtree */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputLocalOrigEstimate)
{  /*lint --e{715}*/
   SCIP_Real estimate;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_ESTIMATE) == 0);
   assert(scip != NULL);

   estimate = SCIPgetLocalOrigEstimate(scip);
   if( SCIPisInfinity(scip, REALABS(estimate)) )
      SCIPinfoMessage(scip, file, "      --      ");
   else
      SCIPinfoMessage(scip, file, "%13.6e ", estimate);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for average dualbound */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputAvgDualbound)
{  /*lint --e{715}*/
   SCIP_Real avgdualbound;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_AVGDUALBOUND) == 0);
   assert(scip != NULL);

   avgdualbound = SCIPgetAvgDualbound(scip);
   if( SCIPisInfinity(scip, REALABS(avgdualbound)) )
      SCIPinfoMessage(scip, file, "      --      ");
   else
      SCIPinfoMessage(scip, file, "%13.6e ", avgdualbound);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for dualbound */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputDualbound)
{  /*lint --e{715}*/
   SCIP_Real dualbound;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_DUALBOUND) == 0);
   assert(scip != NULL);

   dualbound = GCGgetDualbound(scip);

   if( SCIPisInfinity(scip, (SCIP_Real) SCIPgetObjsense(scip) * dualbound ) )
      SCIPinfoMessage(scip, file, "    cutoff    ");
   else if( SCIPisInfinity(scip, -1.0 * (SCIP_Real) SCIPgetObjsense(scip) * dualbound ) )
      SCIPinfoMessage(scip, file, "      --      ");
   else
      SCIPinfoMessage(scip, file, "%13.6e ", dualbound);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for primalbound */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputPrimalbound)
{  /*lint --e{715}*/
   SCIP_Real primalbound;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_PRIMALBOUND) == 0);
   assert(scip != NULL);

   primalbound = GCGgetPrimalbound(scip);

   if( SCIPisInfinity(scip, REALABS(primalbound)) )
      SCIPinfoMessage(scip, file, "      --      ");
   else
      SCIPinfoMessage(scip, file, "%13.6e%c", primalbound, SCIPisPrimalboundSol(scip) ? ' ' : '*');

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for cutoffbound */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputCutoffbound)
{  /*lint --e{715}*/
   SCIP_Real cutoffbound;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_CUTOFFBOUND) == 0);
   assert(scip != NULL);

   cutoffbound = SCIPgetCutoffbound(scip);
   if( SCIPisInfinity(scip, REALABS(cutoffbound)) )
      SCIPinfoMessage(scip, file, "      --      ");
   else
      SCIPinfoMessage(scip, file, "%13.6e ", SCIPretransformObj(scip, cutoffbound));

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for gap */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputGap)
{  /*lint --e{715}*/
   SCIP_Real gap;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_GAP) == 0);
   assert(scip != NULL);

   gap = GCGgetGap(scip);

   if( SCIPisInfinity(scip, gap) )
      SCIPinfoMessage(scip, file, "    Inf ");
   else if( gap >= 100.00 )
      SCIPinfoMessage(scip, file, "  Large ");
   else
      SCIPinfoMessage(scip, file, "%7.2f%%", 100.0*gap);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for the sum of simplex iterations */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputSlpiterations)
{  /*lint --e{715}*/

   SCIP* masterprob;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_SLPITERATIONS) == 0);
   assert(scip != NULL);

   masterprob = GCGgetMasterprob(scip);

   if( masterprob != NULL && SCIPgetStage(masterprob) >= SCIP_STAGE_SOLVING )
   {
      SCIPdispLongint(SCIPgetMessagehdlr(scip), file, SCIPgetNLPIterations(masterprob) + GCGmasterGetPricingSimplexIters(masterprob), DISP_WIDT_SLPITERATIONS);
   }
   else
   {
      SCIPdispLongint(SCIPgetMessagehdlr(scip), file, 0LL, DISP_WIDT_SLPITERATIONS);
   }

   return SCIP_OKAY;
}


/** output method of display column to output file stream 'file' for degeneracy */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputDegeneracy)
{  /*lint --e{715}*/
   SCIP_Real degeneracy;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_DEGENERACY) == 0);
   assert(scip != NULL);

   degeneracy = GCGgetDegeneracy(scip);

   if( SCIPisInfinity(scip, degeneracy) )
      SCIPinfoMessage(scip, file, "   --   ");
   else
      SCIPinfoMessage(scip, file, "%7.2f%%", 100.0*degeneracy);

   return SCIP_OKAY;
}


/** output method of display column to output file stream 'file' for primalgap */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputPrimalgap)
{  /*lint --e{715}*/
   SCIP_Real primalbound;
   SCIP_Real dualbound;
   SCIP_Real gap;

   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_PRIMALGAP) == 0);
   assert(scip != NULL);

   primalbound = SCIPgetPrimalbound(scip);
   dualbound = SCIPgetDualbound(scip);

   if( SCIPisEQ(scip, primalbound, dualbound) )
      gap = 0.0;
   else if( SCIPisZero(scip, primalbound) || SCIPisInfinity(scip, REALABS(primalbound)) || primalbound * dualbound < 0.0 )
      gap = SCIPinfinity(scip);
   else
      gap = REALABS((primalbound - dualbound))/REALABS(primalbound + SCIPepsilon(scip));

   if( SCIPisInfinity(scip, gap) )
      SCIPinfoMessage(scip, file, "    Inf ");
   else if( gap >= 100.00 )
      SCIPinfoMessage(scip, file, "  Large ");
   else
      SCIPinfoMessage(scip, file, "%7.2f%%", 100.0*gap);

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' for number of found solutions */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputNSols)
{  /*lint --e{715}*/
   SCIPinfoMessage(scip, file, "%5"SCIP_LONGINT_FORMAT, SCIPgetNSolsFound(scip));

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputMlpiterations)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_MLPITERATIONS) == 0);
   assert(scip != NULL);

   if( SCIPgetStage(GCGgetMasterprob(scip)) >= SCIP_STAGE_SOLVING )
   {
      SCIPdispLongint(SCIPgetMessagehdlr(scip), file, SCIPgetNLPIterations(GCGgetMasterprob(scip)), DISP_WIDT_MLPITERATIONS);
   }
   else
   {
      SCIPdispLongint(SCIPgetMessagehdlr(scip), file, 0LL, DISP_WIDT_MLPITERATIONS);
   }

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputMvars)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_MVARS) == 0);
   assert(scip != NULL);

   if( SCIPgetStage(GCGgetMasterprob(scip)) >= SCIP_STAGE_SOLVING )
   {
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNVars(GCGgetMasterprob(scip)), DISP_WIDT_MVARS);
   }
   else
   {
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, 0, DISP_WIDT_MVARS);
   }

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputMconss)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_MCONSS) == 0);
   assert(scip != NULL);

   if( SCIPgetStage(GCGgetMasterprob(scip)) >= SCIP_STAGE_SOLVING )
   {
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNConss(GCGgetMasterprob(scip)), DISP_WIDT_MCONSS);
   }
   else
   {
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, 0, DISP_WIDT_MCONSS);
   }

   return SCIP_OKAY;
}

/** output method of display column to output file stream 'file' */
static
SCIP_DECL_DISPOUTPUT(SCIPdispOutputMcuts)
{  /*lint --e{715}*/
   assert(disp != NULL);
   assert(strcmp(SCIPdispGetName(disp), DISP_NAME_MCUTS) == 0);
   assert(scip != NULL);

   if( SCIPgetStage(GCGgetMasterprob(scip)) >= SCIP_STAGE_SOLVING )
   {
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, SCIPgetNCutsApplied(GCGgetMasterprob(scip)), DISP_WIDT_MCUTS);
   }
   else
   {
      SCIPdispInt(SCIPgetMessagehdlr(scip), file, 0, DISP_WIDT_MCUTS);
   }



   return SCIP_OKAY;
}

/*
 * default display columns specific interface methods
 */

/** includes the default display columns in SCIP */
SCIP_RETCODE SCIPincludeDispGcg(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_DISP* tmpdisp;

   tmpdisp = SCIPfindDisp(scip, DISP_NAME_SOLFOUND);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_SOLFOUND, DISP_DESC_SOLFOUND, DISP_HEAD_SOLFOUND,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, SCIPdispInitsolSolFound, NULL, SCIPdispOutputSolFound, NULL,
            DISP_WIDT_SOLFOUND, DISP_PRIO_SOLFOUND, DISP_POSI_SOLFOUND, DISP_STRI_SOLFOUND) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_TIME);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_TIME, DISP_DESC_TIME, DISP_HEAD_TIME,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputSolvingTime, NULL,
            DISP_WIDT_TIME, DISP_PRIO_TIME, DISP_POSI_TIME, DISP_STRI_TIME) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_NNODES);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_NNODES, DISP_DESC_NNODES, DISP_HEAD_NNODES,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNNodes, NULL,
            DISP_WIDT_NNODES, DISP_PRIO_NNODES, DISP_POSI_NNODES, DISP_STRI_NNODES) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_NODESLEFT);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_NODESLEFT, DISP_DESC_NODESLEFT, DISP_HEAD_NODESLEFT,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNNodesLeft, NULL,
            DISP_WIDT_NODESLEFT, DISP_PRIO_NODESLEFT, DISP_POSI_NODESLEFT, DISP_STRI_NODESLEFT) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_LPITERATIONS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_LPITERATIONS, DISP_DESC_LPITERATIONS, DISP_HEAD_LPITERATIONS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNLPIterations, NULL,
            DISP_WIDT_LPITERATIONS, DISP_PRIO_LPITERATIONS, DISP_POSI_LPITERATIONS, DISP_STRI_LPITERATIONS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_LPAVGITERS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_LPAVGITERS, DISP_DESC_LPAVGITERS, DISP_HEAD_LPAVGITERS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNLPAvgIters, NULL,
            DISP_WIDT_LPAVGITERS, DISP_PRIO_LPAVGITERS, DISP_POSI_LPAVGITERS, DISP_STRI_LPAVGITERS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_LPCOND);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_LPCOND, DISP_DESC_LPCOND, DISP_HEAD_LPCOND,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputLPCondition, NULL,
            DISP_WIDT_LPCOND, DISP_PRIO_LPCOND, DISP_POSI_LPCOND, DISP_STRI_LPCOND) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_MEMUSED);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_MEMUSED, DISP_DESC_MEMUSED, DISP_HEAD_MEMUSED,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputMemUsed, NULL,
            DISP_WIDT_MEMUSED, DISP_PRIO_MEMUSED, DISP_POSI_MEMUSED, DISP_STRI_MEMUSED) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_DEPTH);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_DEPTH, DISP_DESC_DEPTH, DISP_HEAD_DEPTH,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputDepth, NULL,
            DISP_WIDT_DEPTH, DISP_PRIO_DEPTH, DISP_POSI_DEPTH, DISP_STRI_DEPTH) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_MAXDEPTH);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_MAXDEPTH, DISP_DESC_MAXDEPTH, DISP_HEAD_MAXDEPTH,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputMaxDepth, NULL,
            DISP_WIDT_MAXDEPTH, DISP_PRIO_MAXDEPTH, DISP_POSI_MAXDEPTH, DISP_STRI_MAXDEPTH) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_PLUNGEDEPTH);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_PLUNGEDEPTH, DISP_DESC_PLUNGEDEPTH, DISP_HEAD_PLUNGEDEPTH,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputPlungeDepth, NULL,
            DISP_WIDT_PLUNGEDEPTH, DISP_PRIO_PLUNGEDEPTH, DISP_POSI_PLUNGEDEPTH, DISP_STRI_PLUNGEDEPTH) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_NFRAC);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_NFRAC, DISP_DESC_NFRAC, DISP_HEAD_NFRAC,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNFrac, NULL,
            DISP_WIDT_NFRAC, DISP_PRIO_NFRAC, DISP_POSI_NFRAC, DISP_STRI_NFRAC) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_NEXTERNCANDS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_NEXTERNCANDS, DISP_DESC_NEXTERNCANDS, DISP_HEAD_NEXTERNCANDS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNExternCands, NULL,
            DISP_WIDT_NEXTERNCANDS, DISP_PRIO_NEXTERNCANDS, DISP_POSI_NEXTERNCANDS, DISP_STRI_NEXTERNCANDS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_VARS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_VARS, DISP_DESC_VARS, DISP_HEAD_VARS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNVars, NULL,
            DISP_WIDT_VARS, DISP_PRIO_VARS, DISP_POSI_VARS, DISP_STRI_VARS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_CONSS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_CONSS, DISP_DESC_CONSS, DISP_HEAD_CONSS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNConss, NULL,
            DISP_WIDT_CONSS, DISP_PRIO_CONSS, DISP_POSI_CONSS, DISP_STRI_CONSS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_CURCONSS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_CURCONSS, DISP_DESC_CURCONSS, DISP_HEAD_CURCONSS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNCurConss, NULL,
            DISP_WIDT_CURCONSS, DISP_PRIO_CURCONSS, DISP_POSI_CURCONSS, DISP_STRI_CURCONSS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_CURCOLS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_CURCOLS, DISP_DESC_CURCOLS, DISP_HEAD_CURCOLS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNCurCols, NULL,
            DISP_WIDT_CURCOLS, DISP_PRIO_CURCOLS, DISP_POSI_CURCOLS, DISP_STRI_CURCOLS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_CURROWS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_CURROWS, DISP_DESC_CURROWS, DISP_HEAD_CURROWS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNCurRows, NULL,
            DISP_WIDT_CURROWS, DISP_PRIO_CURROWS, DISP_POSI_CURROWS, DISP_STRI_CURROWS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_CUTS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_CUTS, DISP_DESC_CUTS, DISP_HEAD_CUTS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNAppliedCuts, NULL,
            DISP_WIDT_CUTS, DISP_PRIO_CUTS, DISP_POSI_CUTS, DISP_STRI_CUTS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_SEPAROUNDS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_SEPAROUNDS, DISP_DESC_SEPAROUNDS, DISP_HEAD_SEPAROUNDS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNSepaRounds, NULL,
            DISP_WIDT_SEPAROUNDS, DISP_PRIO_SEPAROUNDS, DISP_POSI_SEPAROUNDS, DISP_STRI_SEPAROUNDS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_POOLSIZE);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_POOLSIZE, DISP_DESC_POOLSIZE, DISP_HEAD_POOLSIZE,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputCutPoolSize, NULL,
            DISP_WIDT_POOLSIZE, DISP_PRIO_POOLSIZE, DISP_POSI_POOLSIZE, DISP_STRI_POOLSIZE) );
   }
   tmpdisp = SCIPfindDisp(scip,DISP_NAME_CONFLICTS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_CONFLICTS, DISP_DESC_CONFLICTS, DISP_HEAD_CONFLICTS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNConflicts, NULL,
            DISP_WIDT_CONFLICTS, DISP_PRIO_CONFLICTS, DISP_POSI_CONFLICTS, DISP_STRI_CONFLICTS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_STRONGBRANCHS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_STRONGBRANCHS, DISP_DESC_STRONGBRANCHS, DISP_HEAD_STRONGBRANCHS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNStrongbranchs, NULL,
            DISP_WIDT_STRONGBRANCHS, DISP_PRIO_STRONGBRANCHS, DISP_POSI_STRONGBRANCHS, DISP_STRI_STRONGBRANCHS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_PSEUDOOBJ);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_PSEUDOOBJ, DISP_DESC_PSEUDOOBJ, DISP_HEAD_PSEUDOOBJ,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputPseudoObjval, NULL,
            DISP_WIDT_PSEUDOOBJ, DISP_PRIO_PSEUDOOBJ, DISP_POSI_PSEUDOOBJ, DISP_STRI_PSEUDOOBJ) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_LPOBJ);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_LPOBJ, DISP_DESC_LPOBJ, DISP_HEAD_LPOBJ,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputLPObjval, NULL,
            DISP_WIDT_LPOBJ, DISP_PRIO_LPOBJ, DISP_POSI_LPOBJ, DISP_STRI_LPOBJ) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_CURDUALBOUND);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_CURDUALBOUND, DISP_DESC_CURDUALBOUND, DISP_HEAD_CURDUALBOUND,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputCurDualbound, NULL,
            DISP_WIDT_CURDUALBOUND, DISP_PRIO_CURDUALBOUND, DISP_POSI_CURDUALBOUND, DISP_STRI_CURDUALBOUND) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_ESTIMATE);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_ESTIMATE, DISP_DESC_ESTIMATE, DISP_HEAD_ESTIMATE,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputLocalOrigEstimate, NULL,
            DISP_WIDT_ESTIMATE, DISP_PRIO_ESTIMATE, DISP_POSI_ESTIMATE, DISP_STRI_ESTIMATE) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_AVGDUALBOUND);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_AVGDUALBOUND, DISP_DESC_AVGDUALBOUND, DISP_HEAD_AVGDUALBOUND,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputAvgDualbound, NULL,
            DISP_WIDT_AVGDUALBOUND, DISP_PRIO_AVGDUALBOUND, DISP_POSI_AVGDUALBOUND, DISP_STRI_AVGDUALBOUND) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_DUALBOUND);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_DUALBOUND, DISP_DESC_DUALBOUND, DISP_HEAD_DUALBOUND,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputDualbound, NULL,
            DISP_WIDT_DUALBOUND, DISP_PRIO_DUALBOUND, DISP_POSI_DUALBOUND, DISP_STRI_DUALBOUND) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_PRIMALBOUND);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_PRIMALBOUND, DISP_DESC_PRIMALBOUND, DISP_HEAD_PRIMALBOUND,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputPrimalbound, NULL,
            DISP_WIDT_PRIMALBOUND, DISP_PRIO_PRIMALBOUND, DISP_POSI_PRIMALBOUND, DISP_STRI_PRIMALBOUND) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_CUTOFFBOUND);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_CUTOFFBOUND, DISP_DESC_CUTOFFBOUND, DISP_HEAD_CUTOFFBOUND,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputCutoffbound, NULL,
            DISP_WIDT_CUTOFFBOUND, DISP_PRIO_CUTOFFBOUND, DISP_POSI_CUTOFFBOUND, DISP_STRI_CUTOFFBOUND) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_GAP);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_GAP, DISP_DESC_GAP, DISP_HEAD_GAP,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputGap, NULL,
            DISP_WIDT_GAP, DISP_PRIO_GAP, DISP_POSI_GAP, DISP_STRI_GAP) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_PRIMALGAP);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_PRIMALGAP, DISP_DESC_PRIMALGAP, DISP_HEAD_PRIMALGAP,
            SCIP_DISPSTATUS_OFF,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputPrimalgap, NULL,
            DISP_WIDT_PRIMALGAP, DISP_PRIO_PRIMALGAP, DISP_POSI_PRIMALGAP, DISP_STRI_PRIMALGAP) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_NSOLS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_NSOLS, DISP_DESC_NSOLS, DISP_HEAD_NSOLS,
            SCIP_DISPSTATUS_AUTO,
            dispCopyGcg,
            NULL, NULL, NULL, NULL, NULL, SCIPdispOutputNSols, NULL,
            DISP_WIDT_NSOLS, DISP_PRIO_NSOLS, DISP_POSI_NSOLS, DISP_STRI_NSOLS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_MLPITERATIONS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_MLPITERATIONS, DISP_DESC_MLPITERATIONS, DISP_HEAD_MLPITERATIONS,
            SCIP_DISPSTATUS_AUTO, dispCopyGcg, NULL, NULL, NULL, NULL, NULL, SCIPdispOutputMlpiterations, NULL,
            DISP_WIDT_MLPITERATIONS, DISP_PRIO_MLPITERATIONS, DISP_POSI_MLPITERATIONS, DISP_STRI_MLPITERATIONS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_MVARS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_MVARS, DISP_DESC_MVARS, DISP_HEAD_MVARS,
            SCIP_DISPSTATUS_AUTO, dispCopyGcg, NULL, NULL, NULL, NULL, NULL, SCIPdispOutputMvars, NULL,
            DISP_WIDT_MVARS, DISP_PRIO_MVARS, DISP_POSI_MVARS, DISP_STRI_MVARS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_MCONSS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_MCONSS, DISP_DESC_MCONSS, DISP_HEAD_MCONSS,
            SCIP_DISPSTATUS_AUTO, dispCopyGcg, NULL, NULL, NULL, NULL, NULL, SCIPdispOutputMconss, NULL,
            DISP_WIDT_MCONSS, DISP_PRIO_MCONSS, DISP_POSI_MCONSS, DISP_STRI_MCONSS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_MCUTS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_MCUTS, DISP_DESC_MCUTS, DISP_HEAD_MCUTS,
            SCIP_DISPSTATUS_AUTO, dispCopyGcg, NULL, NULL, NULL, NULL, NULL, SCIPdispOutputMcuts, NULL,
            DISP_WIDT_MCUTS, DISP_PRIO_MCUTS, DISP_POSI_MCUTS, DISP_STRI_MCUTS) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_DEGENERACY);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_DEGENERACY, DISP_DESC_DEGENERACY, DISP_HEAD_DEGENERACY,
            SCIP_DISPSTATUS_AUTO, NULL, NULL, NULL, NULL, NULL, NULL, SCIPdispOutputDegeneracy, NULL,
            DISP_WIDT_DEGENERACY, DISP_PRIO_DEGENERACY, DISP_POSI_DEGENERACY, DISP_STRI_DEGENERACY) );
   }
   tmpdisp = SCIPfindDisp(scip, DISP_NAME_SLPITERATIONS);
   if( tmpdisp == NULL )
   {
      SCIP_CALL( SCIPincludeDisp(scip, DISP_NAME_SLPITERATIONS, DISP_DESC_SLPITERATIONS, DISP_HEAD_SLPITERATIONS,
            SCIP_DISPSTATUS_AUTO, NULL, NULL, NULL, NULL, NULL, NULL, SCIPdispOutputSlpiterations, NULL,
            DISP_WIDT_SLPITERATIONS, DISP_PRIO_SLPITERATIONS, DISP_POSI_SLPITERATIONS, DISP_STRI_SLPITERATIONS) );
   }

   return SCIP_OKAY;
}
