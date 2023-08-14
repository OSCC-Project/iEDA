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

/**@file   heur_restmaster.c
 * @brief  Restricted Master Heuristic
 * @author Christian Puchert
 * @author Jonas Witt
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "heur_restmaster.h"
#include "gcg.h"
#include "pricer_gcg.h"

#include "scip/scipdefplugins.h"


#define HEUR_NAME             "restmaster"
#define HEUR_DESC             "LNS heuristic for the master problem that fixes some master variables to zero"
#define HEUR_DISPCHAR         'P'
#define HEUR_PRIORITY         100
#define HEUR_FREQ             30
#define HEUR_FREQOFS          0
#define HEUR_MAXDEPTH         -1
#define HEUR_TIMING           SCIP_HEURTIMING_DURINGLPLOOP | SCIP_HEURTIMING_DURINGPRICINGLOOP
#define HEUR_USESSUBSCIP      TRUE

#define DEFAULT_MAXNODES      5000LL    /**< maximum number of nodes to regard in the subproblem                 */
#define DEFAULT_MINFIXINGRATE 0.5       /**< minimum percentage of integer variables that have to be fixed       */
#define DEFAULT_MINIMPROVE    0.01      /**< factor by which restricted master should at least improve the incumbent */
#define DEFAULT_MINNODES      500LL     /**< minimum number of nodes to regard in the subproblem                 */
#define DEFAULT_NODESOFS      500LL     /**< number of nodes added to the contingent of the total nodes          */
#define DEFAULT_NODESQUOT     0.1       /**< subproblem nodes in relation to nodes of the original problem       */
#define DEFAULT_USELPROWS     FALSE     /**< should subproblem be created out of the rows in the LP rows,
                                          *  otherwise, the copy constructor of the constraints handlers are used*/
#define DEFAULT_COPYCUTS      TRUE      /**< if DEFAULT_USELPROWS is FALSE, then should all active cuts from the cutpool
                                          *  of the original scip be copied to constraints of the subscip        */

#define DEFAULT_PBHEUR        FALSE     /**< default value for using the restricted master heuristic as
                                          *  price-and-branch heuristic?
                                          *  (this changes the HEUR_TIMING to SCIP_HEURTIMING_AFTERNODE,
                                          *  and it changes the HEUR_FREQ to 0.                                  */

/*
 * Data structures
 */

/** primal heuristic data */
struct SCIP_HeurData
{
   SCIP_Longint          maxnodes;           /**< maximum number of nodes to regard in the subproblem                 */
   SCIP_Longint          minnodes;           /**< minimum number of nodes to regard in the subproblem                 */
   SCIP_Longint          nodesofs;           /**< number of nodes added to the contingent of the total nodes          */
   SCIP_Longint          usednodes;          /**< nodes already used by restricted master in earlier calls            */
   SCIP_Real             minfixingrate;      /**< minimum percentage of integer variables that have to be fixed       */
   SCIP_Real             minimprove;         /**< factor by which restricted master should at least improve the incumbent */
   SCIP_Real             nodesquot;          /**< subproblem nodes in relation to nodes of the original problem       */
   SCIP_Bool             uselprows;          /**< should subproblem be created out of the rows in the LP rows?        */
   SCIP_Bool             copycuts;           /**< if uselprows == FALSE, should all active cuts from cutpool be copied
                                              *   to constraints in subproblem?
                                              */
   SCIP_Bool             pbheur;             /**< value for using the restricted master heuristic as
                                              *  price-and-branch heuristic?
                                              *  (this changes the HEUR_TIMING to SCIP_HEURTIMING_AFTERNODE,
                                              *  and it changes the HEUR_FREQ to 0.                                   */
};




/*
 * Local methods
 */

/** set up the restricted master problem by fixing master variables to zero */
static
SCIP_RETCODE setupSubproblem(
   SCIP*                 scip,               /**< SCIP data structure for master problem                         */
   SCIP*                 restmaster,         /**< SCIP data structure for restricted master problem              */
   SCIP_VAR**            restmastervars,     /**< the variables of the restricted                                */
   SCIP_Real             minfixingrate,      /**< percentage of integer variables that have to be fixed          */
   SCIP_Bool             uselprows,          /**< should subproblem be created out of the rows in the LP rows?   */
   SCIP_Bool             pbheur,             /**< use heuristic as price-and-branch heuristic?                   */
   SCIP_Bool*            success             /**< pointer to store whether the problem was created successfully  */
   )
{
   SCIP_VAR** mastervars;
   int nmastervars;
   SCIP_Real fixingrate;

   int i;
   int fixingcounter;

   /* get variable data of the master problem */
   SCIP_CALL( SCIPgetVarsData(scip, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(mastervars != NULL);
   assert(nmastervars >= 0);

   fixingcounter = 0;

   /* fix zero variables in the restricted master problem */
   for( i = 0; i < nmastervars && !pbheur; i++ )
   {
      SCIP_Real mastersolval;

      mastersolval = SCIPgetSolVal(scip, NULL, mastervars[i]);

      /* if LP solution value of master variable is zero, fix it to zero in restricted master */
      if( SCIPisFeasZero(scip, mastersolval) )
      {
         SCIP_CALL( SCIPchgVarLbGlobal(restmaster, restmastervars[i], 0.0) );
         SCIP_CALL( SCIPchgVarUbGlobal(restmaster, restmastervars[i], 0.0) );
         fixingcounter++;
      }
   }

   /* abort, if all variables were fixed (which should not happen) */
   if( fixingcounter == nmastervars )
   {
      SCIPdebugMessage(" -> all master variables fixed, not solving problem.\n");
      *success = FALSE;
      return SCIP_OKAY;
   }
   else
      fixingrate = fixingcounter / (SCIP_Real)(MAX(nmastervars, 1));

   SCIPdebugMessage(" -> %d out of %d (%.2f percent) master variables fixed.\n", fixingcounter, nmastervars, fixingrate * 100.0);

   /* abort, if the amount of fixed variables is insufficient */
   if( fixingrate < minfixingrate && !pbheur)
   {
      SCIPdebugMessage(" -> not enough variables fixed.\n");
      *success = FALSE;
      return SCIP_OKAY;
   }

   if( uselprows )
   {
      SCIP_ROW** rows;                          /* original scip rows                         */
      int nrows;

      /* get the rows and their number */
      SCIP_CALL( SCIPgetLPRowsData(scip, &rows, &nrows) );

      /* copy all rows to linear constraints */
      for( i = 0; i < nrows; i++ )
      {
         SCIP_CONS* cons;
         SCIP_VAR** consvars;
         SCIP_COL** cols;
         SCIP_Real constant;
         SCIP_Real lhs;
         SCIP_Real rhs;
         SCIP_Real* vals;
         int nnonz;
         int j;

         /* ignore rows that are only locally valid */
         if( SCIProwIsLocal(rows[i]) )
            continue;

         /* get the row's data */
         constant = SCIProwGetConstant(rows[i]);
         lhs = SCIProwGetLhs(rows[i]) - constant;
         rhs = SCIProwGetRhs(rows[i]) - constant;
         vals = SCIProwGetVals(rows[i]);
         nnonz = SCIProwGetNNonz(rows[i]);
         cols = SCIProwGetCols(rows[i]);

         assert(lhs <= rhs);

         /* allocate memory array to be filled with the corresponding subproblem variables */
         SCIP_CALL( SCIPallocBufferArray(restmaster, &consvars, nnonz) );
         for( j = 0; j < nnonz; j++ )
            consvars[j] = restmastervars[SCIPvarGetProbindex(SCIPcolGetVar(cols[j]))];

         /* create a new linear constraint and add it to the subproblem */
         SCIP_CALL( SCIPcreateConsLinear(restmaster, &cons, SCIProwGetName(rows[i]), nnonz, consvars, vals, lhs, rhs,
               TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE) );
         SCIP_CALL( SCIPaddCons(restmaster, cons) );
         SCIP_CALL( SCIPreleaseCons(restmaster, &cons) );

         /* free temporary memory */
         SCIPfreeBufferArray(restmaster, &consvars);
      }
   }

   *success = TRUE;
   return SCIP_OKAY;
}

/** creates a new solution for the original problem by translating the solution of the restricted master problem */
static
SCIP_RETCODE createNewSol(
   SCIP*                 origprob,           /**< original SCIP data structure                        */
   SCIP*                 scip,               /**< SCIP data structure of master problem               */
   SCIP*                 restmaster,         /**< SCIP structure of restricted master problem         */
   SCIP_VAR**            restmastervars,     /**< the variables of the restricted master problem      */
   SCIP_HEUR*            heur,               /**< Restricted Master heuristic structure               */
   SCIP_SOL*             restmastersol,      /**< solution of the restricted master problem           */
   SCIP_Bool*            success             /**< used to store whether new solution was found or not */
   )
{
   SCIP_VAR** vars;                          /* the original problem's variables                */
   SCIP_VAR** mastervars;                    /* the master problem's variables                  */
   int        nvars;
   int        nmastervars;
   SCIP_Real* restmastervals;                /* solution values of the subproblem               */
   SCIP_SOL*  newmastersol;                  /* solution for the master problem                 */

   assert(origprob != NULL);
   assert(scip != NULL);
   assert(restmaster != NULL);
   assert(restmastervars != NULL);
   assert(restmastersol != NULL);

   /* get variables' data */
   SCIP_CALL( SCIPgetVarsData(origprob, &vars, &nvars, NULL, NULL, NULL, NULL) );
   SCIP_CALL( SCIPgetVarsData(scip, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(nmastervars == SCIPgetNOrigVars(restmaster));

   SCIP_CALL( SCIPallocBufferArray(scip, &restmastervals, nmastervars) );

   /* copy the solution */
   SCIP_CALL( SCIPgetSolVals(restmaster, restmastersol, nmastervars, restmastervars, restmastervals) );

   /* create new solution for the master problem */
   SCIP_CALL( SCIPcreateSol(scip, &newmastersol, heur) );
   SCIP_CALL( SCIPsetSolVals(scip, newmastersol, nmastervars, mastervars, restmastervals) );

#ifdef SCIP_DEBUG
   SCIP_CALL( SCIPtrySolFree(scip, &newmastersol, TRUE, TRUE, TRUE, TRUE, TRUE, success) );
#else
   SCIP_CALL( SCIPtrySolFree(scip, &newmastersol, FALSE, FALSE, TRUE, TRUE, TRUE, success) );
#endif

   SCIPfreeBufferArray(scip, &restmastervals);

   return SCIP_OKAY;
}


/*
 * Callback methods of primal heuristic
 */

/** copy method for primal heuristic plugins (called when SCIP copies plugins) */
#define heurCopyRestmaster NULL

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeRestmaster)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* free heuristic data */
   SCIPfreeMemory(scip, &heurdata);
   SCIPheurSetData(heur, NULL);

   return SCIP_OKAY;
}


/** initialization method of primal heuristic (called after problem was transformed) */
static
SCIP_DECL_HEURINIT(heurInitRestmaster)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic's data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* initialize data */
   heurdata->usednodes = 0;

   /* change timing to after node, call heuristic only at root and increase maxnodes for price-and-branch heuristic */
   if( heurdata->pbheur )
   {
      heurdata->maxnodes = INT_MAX;

      SCIPheurSetTimingmask(heur, SCIP_HEURTIMING_AFTERNODE);
      SCIPheurSetFreq(heur, 0);
   }

   return SCIP_OKAY;
}


/** deinitialization method of primal heuristic (called before transformed problem is freed) */
#define heurExitRestmaster NULL


/** solving process initialization method of primal heuristic (called when branch and bound process is about to begin) */
#define heurInitsolRestmaster NULL


/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed) */
#define heurExitsolRestmaster NULL


/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecRestmaster)
{  /*lint --e{715}*/
   SCIP* origprob;                           /* SCIP structure of original problem    */
   SCIP_HEURDATA* heurdata;                  /* heuristic's data                    */
   SCIP_Real timelimit;                      /* timelimit for the subproblem        */
   SCIP_Real memorylimit;
   SCIP_Bool discretization;
   SCIP_Bool success;
   SCIP_Longint nstallnodes;                 /* number of stalling nodes for the restricted master problem */

   SCIP* restmaster;                         /* SCIP structure of the restricted master problem            */
   SCIP_HASHMAP* varmapfw;                   /* mapping of master variables to restricted master variables */
   SCIP_VAR** mastervars;
   SCIP_VAR** restmastervars;
   int nmastervars;

   int i;

#ifdef NDEBUG
   SCIP_RETCODE retstat;
#endif

   assert(heur != NULL);
   assert(scip != NULL);
   assert(result != NULL);
   assert(SCIPhasCurrentNodeLP(scip));

   /* get original problem */
   origprob = GCGmasterGetOrigprob(scip);
   assert(origprob != NULL);

   /* get heuristic's data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   *result = SCIP_DIDNOTRUN;

   /* this heuristic works only for the discretization approach */
   SCIP_CALL( SCIPgetBoolParam(origprob, "relaxing/gcg/discretization", &discretization) );
   if( !discretization )
      return SCIP_OKAY;

   *result = SCIP_DELAYED;

   /* only call heuristic, if an optimal LP solution is at hand */
   if( SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL )
      return SCIP_OKAY;

   *result = SCIP_DIDNOTRUN;

   /* calculate the maximal number of branching nodes until heuristic is aborted */
   nstallnodes = (SCIP_Longint)(heurdata->nodesquot * SCIPgetNNodes(origprob));

   if( !heurdata->pbheur )
   {
      /* reward restricted master if it succeeded often */
      nstallnodes = (SCIP_Longint)(nstallnodes * 3.0 * (SCIPheurGetNBestSolsFound(heur)+1.0)/(SCIPheurGetNCalls(heur) + 1.0));
      nstallnodes -= 100 * SCIPheurGetNCalls(heur);  /* count the setup costs for the sub-MIP as 100 nodes */
      nstallnodes += heurdata->nodesofs;

      /* determine the node limit for the current process */
      nstallnodes -= heurdata->usednodes;
      nstallnodes = MIN(nstallnodes, heurdata->maxnodes);


      /* check whether we have enough nodes left to call subproblem solving */
      if( nstallnodes < heurdata->minnodes )
      {
   /*       SCIPdebugMessage("skipping Restricted Master Heuristic: nstallnodes=%"SCIP_LONGINT_FORMAT", minnodes=%"SCIP_LONGINT_FORMAT"\n", nstallnodes, heurdata->minnodes); */
         return SCIP_OKAY;
      }
   }

   /* check whether there is enough time and memory left */
   SCIP_CALL( SCIPgetRealParam(origprob, "limits/time", &timelimit) );
   if( !SCIPisInfinity(origprob, timelimit) )
      timelimit -= SCIPgetSolvingTime(origprob);
   SCIP_CALL( SCIPgetRealParam(origprob, "limits/memory", &memorylimit) );
   if( !SCIPisInfinity(origprob, memorylimit) )
      memorylimit -= SCIPgetMemUsed(origprob)/1048576.0;
   if( timelimit < 10.0 || memorylimit <= 0.0 )
      return SCIP_OKAY;

   if( SCIPisStopped(scip) )
      return SCIP_OKAY;

   SCIPdebugMessage("Executing Restricted Master Heuristic ...\n");

   *result = SCIP_DIDNOTFIND;

   /* get variable data of the master problem */
   SCIP_CALL( SCIPgetVarsData(scip, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(mastervars != NULL);
   assert(nmastervars >= 0);

   /* initializing the subproblem */
   SCIP_CALL( SCIPcreate(&restmaster) );

   /* create the variable mapping hash map */
   SCIP_CALL( SCIPhashmapCreate(&varmapfw, SCIPblkmem(restmaster), nmastervars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &restmastervars, nmastervars) );

   if( heurdata->uselprows )
   {
      char probname[SCIP_MAXSTRLEN];

      /* copy all plugins */
      SCIP_CALL( SCIPincludeDefaultPlugins(restmaster) );

      /* get name of the original problem and add the string "_restricted" */
      (void) SCIPsnprintf(probname, SCIP_MAXSTRLEN, "%s_restricted", SCIPgetProbName(scip));

      /* create the subproblem */
      SCIP_CALL( SCIPcreateProb(restmaster, probname, NULL, NULL, NULL, NULL, NULL, NULL, NULL) );

      /* copy all variables */
      SCIP_CALL( SCIPcopyVars(scip, restmaster, varmapfw, NULL, NULL, NULL, 0, TRUE) );
   }
   else
   {
      SCIP_Bool valid;

      valid = FALSE;

      SCIP_CALL( SCIPcopy(scip, restmaster, varmapfw, NULL, "restmaster", TRUE, FALSE, FALSE, TRUE, &valid) ); /** @todo check for thread safeness */

      if( heurdata->copycuts )
      {
         /* copies all active cuts from cutpool of sourcescip to linear constraints in targetscip */
         SCIP_CALL( SCIPcopyCuts(scip, restmaster, varmapfw, NULL, TRUE, NULL) );
      }

      SCIPdebugMessage("Copying the SCIP instance was %s complete.\n", valid ? "" : "not ");
   }

   for( i = 0; i < nmastervars; i++ )
      restmastervars[i] = (SCIP_VAR*) SCIPhashmapGetImage(varmapfw, mastervars[i]);

   /* free hash map */
   SCIPhashmapFree(&varmapfw);

   success = FALSE;

   /* set up restricted master problem by fixing variables to zero */
   SCIP_CALL( setupSubproblem(scip, restmaster, restmastervars, heurdata->minfixingrate, heurdata->uselprows, heurdata->pbheur, &success) );
   SCIPdebugMessage("restricted master problem: %d vars, %d cons, success=%u\n", SCIPgetNVars(restmaster), SCIPgetNConss(restmaster), success);

   /* do not abort subproblem on CTRL-C */
   SCIP_CALL( SCIPsetBoolParam(restmaster, "misc/catchctrlc", FALSE) );

   /* disable output to console */
   SCIP_CALL( SCIPsetIntParam(restmaster, "display/verblevel", 0) );

   /* if price-and-branch heuristic, set node limits according to original problem */
   if( heurdata->pbheur )
      nstallnodes = heurdata->maxnodes;

   /* set limits for the subproblem */
   SCIP_CALL( SCIPsetLongintParam(restmaster, "limits/stallnodes", nstallnodes) );
   SCIP_CALL( SCIPsetLongintParam(restmaster, "limits/nodes", heurdata->maxnodes) );
   SCIP_CALL( SCIPsetRealParam(restmaster, "limits/time", timelimit) );
   SCIP_CALL( SCIPsetRealParam(restmaster, "limits/memory", memorylimit) );

   /* set specific parameters only of price-and-branch is not used */
   if( !heurdata->pbheur )
   {
      /* forbid recursive call of heuristics solving subMIPs */
      SCIP_CALL( SCIPsetSubscipsOff(restmaster, TRUE) );
      /* disable cutting plane separation */
      SCIP_CALL( SCIPsetSeparating(restmaster, SCIP_PARAMSETTING_OFF, TRUE) );

      /* disable expensive presolving */
      SCIP_CALL( SCIPsetPresolving(restmaster, SCIP_PARAMSETTING_FAST, TRUE) );

      /* use best estimate node selection */
      if( SCIPfindNodesel(scip, "estimate") != NULL )
      {
         SCIP_CALL( SCIPsetIntParam(restmaster, "nodeselection/estimate/stdpriority", INT_MAX/4) );
      }

      /* use inference branching */
      if( SCIPfindBranchrule(scip, "inference") != NULL )
      {
         SCIP_CALL( SCIPsetIntParam(restmaster, "branching/inference/priority", INT_MAX/4) );
      }

      /* disable conflict analysis */
      if( !SCIPisParamFixed(restmaster, "conflict/enable") )
      {
         SCIP_CALL( SCIPsetBoolParam(restmaster, "conflict/enable", FALSE) );
      }
   }

   /* if the subproblem could not be created, free memory and return */
   if( !success )
   {
      SCIPdebugMessage("restricted master problem not created.\n");
      *result = SCIP_DIDNOTRUN;
      SCIPfreeBufferArray(scip, &restmastervars);
      SCIP_CALL( SCIPfree(&restmaster) );
      return SCIP_OKAY;
   }

   /* if there is already a solution, add an objective cutoff */
   /* @todo origprob or scip? */
   if( SCIPgetNSols(origprob) > 0 )
   {
      SCIP_Real cutoff;                         /* objective cutoff for the restricted master problem */
      SCIP_Real upperbound;
      assert(!SCIPisInfinity(origprob,SCIPgetUpperbound(origprob)));

      upperbound = SCIPgetUpperbound(origprob) - SCIPsumepsilon(origprob);

      if( !SCIPisInfinity(origprob,-1.0*SCIPgetLowerbound(origprob)) )
      {
         cutoff = (1-heurdata->minimprove)*SCIPgetUpperbound(origprob) + heurdata->minimprove*SCIPgetLowerbound(origprob);
      }
      else
      {
         if( SCIPgetUpperbound ( origprob ) >= 0 )
            cutoff = ( 1 - heurdata->minimprove ) * SCIPgetUpperbound ( origprob );
         else
            cutoff = ( 1 + heurdata->minimprove ) * SCIPgetUpperbound ( origprob );
      }
      cutoff = MIN(upperbound, cutoff);
      SCIP_CALL( SCIPsetObjlimit(restmaster, cutoff) );
   }

   /* solve the restricted master problem */
   /* Errors in the LP solver should not kill the overall solving process, if the LP is just needed for a heuristic.
    * Hence in optimized mode, the return code is catched and a warning is printed, only in debug mode, SCIP will stop.
    */
#ifdef NDEBUG
   retstat = SCIPpresolve(restmaster);
   if( retstat != SCIP_OKAY )
   {
      SCIPwarningMessage(scip, "Error while presolving subMIP in Restricted Master Heuristic; Restricted Master terminated with code <%d>\n",retstat);
   }
#else
   SCIP_CALL( SCIPpresolve(restmaster) );
#endif

   SCIPdebugMessage("presolved restricted master problem: %d vars, %d cons, success=%u\n", SCIPgetNVars(restmaster), SCIPgetNConss(restmaster), success);

   /* after presolving, we should have at least reached a certain fixing rate over ALL variables (including continuous)
    * to ensure that not only the MIP but also the LP relaxation is easy enough
    */
   if( heurdata->pbheur || ( nmastervars - SCIPgetNVars(restmaster) ) / (SCIP_Real)nmastervars >= heurdata->minfixingrate / 2.0 )
   {
      SCIP_SOL** restmastersols;
      int nrestmastersols;

      SCIPdebugMessage("solving restricted master problem: nstallnodes=%"SCIP_LONGINT_FORMAT", maxnodes=%"SCIP_LONGINT_FORMAT"\n", nstallnodes, heurdata->maxnodes);

#ifdef NDEBUG
      retstat = SCIPsolve(restmaster);
      if( retstat != SCIP_OKAY )
      {
         SCIPwarningMessage(scip, "Error while solving subMIP in Restricted Master Heuristic; Restricted Master terminated with code <%d>\n",retstat);
      }
#else
      SCIP_CALL( SCIPsolve(restmaster) );
#endif

      SCIPdebugMessage(" -> %d feasible solution(s) found.\n", SCIPgetNSols(restmaster));

      /* check, whether a solution was found;
       * due to numerics, it might happen that not all solutions are feasible -> try all solutions until one was accepted
       */
      nrestmastersols = SCIPgetNSols(restmaster);
      restmastersols = SCIPgetSols(restmaster);
      success = FALSE;
      for( i = 0; i < nrestmastersols && !success; ++i )
      {
         SCIP_CALL( createNewSol(origprob, scip, restmaster, restmastervars, heur, restmastersols[i], &success) );
      }
      if( success )
         *result = SCIP_FOUNDSOL;
   }

   /* free subproblem */
   SCIPfreeBufferArray(scip, &restmastervars);
   SCIP_CALL( SCIPfree(&restmaster) );


   return SCIP_OKAY;
}




/*
 * primal heuristic specific interface methods
 */

/** creates the Restricted Master primal heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurRestmaster(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEURDATA* heurdata;

   /* create Restricted Master primal heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &heurdata) );

   /* include primal heuristic */
   SCIP_CALL( SCIPincludeHeur(scip, HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP,
         heurCopyRestmaster, heurFreeRestmaster, heurInitRestmaster, heurExitRestmaster,
         heurInitsolRestmaster, heurExitsolRestmaster, heurExecRestmaster,
         heurdata) );

   /* add Restricted Master primal heuristic parameters */
   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/minfixingrate",
         "minimum percentage of integer variables that have to be fixable ",
         &heurdata->minfixingrate, FALSE, DEFAULT_MINFIXINGRATE, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddLongintParam(scip, "heuristics/"HEUR_NAME"/maxnodes",
         "maximum number of nodes to regard in the subproblem",
         &heurdata->maxnodes,  TRUE,DEFAULT_MAXNODES, 0LL, SCIP_LONGINT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddLongintParam(scip, "heuristics/"HEUR_NAME"/nodesofs",
         "number of nodes added to the contingent of the total nodes",
         &heurdata->nodesofs, FALSE, DEFAULT_NODESOFS, 0LL, SCIP_LONGINT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddLongintParam(scip, "heuristics/"HEUR_NAME"/minnodes",
         "minimum number of nodes required to start the subproblem",
         &heurdata->minnodes, TRUE, DEFAULT_MINNODES, 0LL, SCIP_LONGINT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/nodesquot",
         "contingent of sub problem nodes in relation to the number of nodes of the original problem",
         &heurdata->nodesquot, FALSE, DEFAULT_NODESQUOT, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/minimprove",
         "factor by which restricted master should at least improve the incumbent  ",
         &heurdata->minimprove, TRUE, DEFAULT_MINIMPROVE, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/uselprows",
         "should subproblem be created out of the rows in the LP rows?",
         &heurdata->uselprows, TRUE, DEFAULT_USELPROWS, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/copycuts",
         "if uselprows == FALSE, should all active cuts from cutpool be copied to constraints in subproblem?",
         &heurdata->copycuts, TRUE, DEFAULT_COPYCUTS, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/pbheur",
         "should the restricted master heuristic be used as price-and-branch heuristic?",
         &heurdata->pbheur, FALSE, DEFAULT_PBHEUR, NULL, NULL) );

   return SCIP_OKAY;
}
