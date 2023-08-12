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

/**@file   heur_gcgdins.c
 * @brief  DINS primal heuristic (according to Ghosh)
 * @author Robert Waniek
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "heur_gcgdins.h"
#include "gcg.h"

#include "scip/scipdefplugins.h"
#include "scip/cons_linear.h"

#define HEUR_NAME             "gcgdins"
#define HEUR_DESC             "distance induced neighborhood search by Ghosh"
#define HEUR_DISPCHAR         'D'
#define HEUR_PRIORITY         -1105000
#define HEUR_FREQ             -1
#define HEUR_FREQOFS          0
#define HEUR_MAXDEPTH         -1
#define HEUR_TIMING           SCIP_HEURTIMING_AFTERNODE
#define HEUR_USESSUBSCIP      TRUE      /**< does the heuristic use a secondary SCIP instance? */

#define DEFAULT_NODESOFS      5000LL    /**< number of nodes added to the contingent of the total nodes          */
#define DEFAULT_MAXNODES      5000LL    /**< maximum number of nodes to regard in the subproblem                 */
#define DEFAULT_MINNODES      500LL     /**< minimum number of nodes to regard in the subproblem                 */
#define DEFAULT_MINIMPROVE    0.01      /**< factor by which DINS should at least improve the incumbent          */
#define DEFAULT_NODESQUOT     0.05      /**< subproblem nodes in relation to nodes of the original problem       */
#define DEFAULT_NWAITINGNODES 0LL       /**< number of nodes without incumbent change that heuristic should wait */
#define DEFAULT_NEIGHBORHOODSIZE  18    /**< radius of the incumbents neighborhood to be searched                */
#define DEFAULT_SOLNUM        5         /**< number of pool-solutions to be checked for flag array update        */
#define DEFAULT_USELPROWS     FALSE     /**< should subproblem be created out of the rows in the LP rows,
                                         * otherwise, the copy constructors of the constraints handlers are used */
#define DEFAULT_COPYCUTS      TRUE      /**< if DEFAULT_USELPROWS is FALSE, then should all active cuts from the cutpool
                                         * of the original scip be copied to constraints of the subscip        */


/*
 * Data structures
 */

/** DINS primal heuristic data */
struct SCIP_HeurData
{
   SCIP_Longint          nodesofs;           /**< number of nodes added to the contingent of the total nodes          */
   SCIP_Longint          maxnodes;           /**< maximum number of nodes to regard in the subproblem                 */
   SCIP_Longint          minnodes;           /**< minimum number of nodes to regard in the subproblem                 */
   SCIP_Longint          nwaitingnodes;      /**< number of nodes without incumbent change that heuristic should wait */
   SCIP_Real             minimprove;         /**< factor by which DINS should at least improve the incumbent          */
   SCIP_Longint          usednodes;          /**< nodes already used by DINS in earlier calls                         */
   SCIP_Real             nodesquot;          /**< subproblem nodes in relation to nodes of the original problem       */
   int                   neighborhoodsize;   /**< radius of the incumbent's neighborhood to be searched               */
   SCIP_Bool*            delta;              /**< stores whether a variable kept its value from root LP all the time  */
   int                   deltalength;        /**< if there are no binary variables, we need no flag array             */
   SCIP_Longint          lastnsolsfound;     /**< solutions found until the last call of DINS                         */
   int                   solnum;             /**< number of pool-solutions to be checked for flag array update        */
   SCIP_Bool             uselprows;          /**< should subproblem be created out of the rows in the LP rows?        */
   SCIP_Bool             copycuts;           /**< if uselprows == FALSE, should all active cuts from cutpool be copied
                                              *   to constraints in subproblem?
                                              */
   SCIP_SOL*             rootsol;            /**< relaxation solution at the root node */
   SCIP_Bool             firstrun;           /**< is the heuristic running for the first time? */

#ifdef SCIP_STATISTIC
   SCIP_Real             avgfixrate;         /**< average rate of variables that are fixed                            */
   SCIP_Real             avgzerorate;        /**< average rate of fixed variables that are zero                       */
   SCIP_Longint          totalsols;          /**< total number of subSCIP solutions (including those which have not
                                              *   been added)
                                              */
   SCIP_Real             subsciptime;        /**< total subSCIP solving time in seconds                               */
   SCIP_Real             bestprimalbd;       /**< objective value of best solution found by this heuristic            */
#endif
};


/*
 * Local methods
 */

/** creates a subproblem for subscip by fixing a number of variables */
static
SCIP_RETCODE createSubproblem(
   SCIP*                 scip,               /**< SCIP data structure of the original problem                    */
   SCIP*                 subscip,            /**< SCIP data structure of the subproblem                          */
   SCIP_VAR**            vars,               /**< variables of the original problem                              */
   SCIP_VAR**            subvars,            /**< variables of the subproblem                                    */
   int                   nbinvars,           /**< number of binary variables of problem and subproblem           */
   int                   nintvars,           /**< number of general integer variables of problem and subproblem  */
   SCIP_Bool             uselprows,          /**< should subproblem be created out of the rows in the LP rows?   */
   int*                  fixingcounter,      /**< number of integers that get actually fixed                     */
   int*                  zerocounter         /**< number of variables fixed to zero                              */
   )
{
   SCIP_SOL* bestsol;

   int i;

   assert(scip != NULL);
   assert(subscip != NULL);
   assert(vars != NULL);
   assert(subvars != NULL);

   /* get the best MIP-solution known so far */
   bestsol = SCIPgetBestSol(scip);
   assert(bestsol != NULL);

   /* create the rebounded general integer variables of the subproblem */
   for( i = nbinvars; i < nbinvars + nintvars; i++ )
   {
      SCIP_Real mipsol;
      SCIP_Real lpsol;

      SCIP_Real lbglobal;
      SCIP_Real ubglobal;

      /* get the bounds for each variable */
      lbglobal = SCIPvarGetLbGlobal(vars[i]);
      ubglobal = SCIPvarGetUbGlobal(vars[i]);

      assert(SCIPvarGetType(vars[i]) == SCIP_VARTYPE_INTEGER);
      /* get the current relaxation solution for each variable */
      lpsol = SCIPgetRelaxSolVal(scip, vars[i]);
      /* get the current MIP solution for each variable */
      mipsol = SCIPgetSolVal(scip, bestsol, vars[i]);

      /* if the solution values differ by 0.5 or more, the variable is rebounded, otherwise it is just copied */
      if( REALABS(lpsol-mipsol) >= 0.5 )
      {
         SCIP_Real lb;
         SCIP_Real ub;
         SCIP_Real range;

         lb = lbglobal;
         ub = ubglobal;

         /* create a equally sized range around lpsol for general integers: bounds are lpsol +- (mipsol-lpsol) */
         range = 2*lpsol-mipsol;

         if( mipsol >= lpsol )
         {
            range = SCIPfeasCeil(scip, range);
            lb = MAX(lb, range);

            /* when the bound new upper bound is equal to the current MIP solution, we set both bounds to the integral bound (without eps) */
            if( SCIPisFeasEQ(scip, mipsol, lb) )
               ub = lb;
            else
               ub = mipsol;
         }
         else
         {
            range = SCIPfeasFloor(scip, range);
            ub = MIN(ub, range);

            /* when the bound new upper bound is equal to the current MIP solution, we set both bounds to the integral bound (without eps) */
            if( SCIPisFeasEQ(scip, mipsol, ub) )
               lb = ub;
            else
               lb = mipsol;
         }

         /* the global domain of variables might have been reduced since incumbent was found: adjust lb and ub accordingly */
         lb = MAX(lb, lbglobal);
         ub = MIN(ub, ubglobal);

         /* perform the bound change */
         SCIP_CALL( SCIPchgVarLbGlobal(subscip, subvars[i], lb) );
         SCIP_CALL( SCIPchgVarUbGlobal(subscip, subvars[i], ub) );
         if( SCIPisEQ(scip, lb, ub) )
         {
            (*fixingcounter)++;
            if( SCIPisZero(scip, ub) )
               (*zerocounter)++;
         }
      }
      else
      {
         /* the global domain of variables might have been reduced since incumbent was found: adjust it accordingly */
         mipsol = MAX(mipsol, lbglobal);
         mipsol = MIN(mipsol, ubglobal);

         /* hard fixing for general integer variables with abs(mipsol-lpsol) < 0.5 */
         SCIP_CALL( SCIPchgVarLbGlobal(subscip, subvars[i], mipsol) );
         SCIP_CALL( SCIPchgVarUbGlobal(subscip, subvars[i], mipsol) );
         (*fixingcounter)++;
         if( SCIPisZero(scip, mipsol) )
            (*zerocounter)++;
      }
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
         SCIP_CALL( SCIPallocBufferArray(subscip, &consvars, nnonz) );
         for( j = 0; j < nnonz; j++ )
            consvars[j] = subvars [ SCIPvarGetProbindex(SCIPcolGetVar(cols[j])) ];

         /* create a new linear constraint and add it to the subproblem */
         SCIP_CALL( SCIPcreateConsLinear(subscip, &cons, SCIProwGetName(rows[i]), nnonz, consvars, vals, lhs, rhs,
               TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE) );
         SCIP_CALL( SCIPaddCons(subscip, cons) );
         SCIP_CALL( SCIPreleaseCons(subscip, &cons) );

         /* free temporary memory */
         SCIPfreeBufferArray(subscip, &consvars);
      }
   }

   return SCIP_OKAY;
}

/** create the extra constraint of local branching and add it to subscip */
static
SCIP_RETCODE addLocalBranchingConstraint(
   SCIP*                 scip,               /**< SCIP data structure of the original problem */
   SCIP*                 subscip,            /**< SCIP data structure of the subproblem       */
   SCIP_VAR**            subvars,            /**< variables of the subproblem                 */
   SCIP_HEURDATA*        heurdata,           /**< heuristic's data structure                  */
   SCIP_Bool*            fixed               /**< TRUE --> include variable in LB constraint  */
   )
{
   SCIP_CONS* cons;                     /* local branching constraint to create          */
   SCIP_VAR** consvars;
   SCIP_VAR** vars;
   SCIP_SOL* bestsol;

   SCIP_Real* consvals;
   SCIP_Real solval;
   SCIP_Real lhs;
   SCIP_Real rhs;

   char consname[SCIP_MAXSTRLEN];

   int nbinvars;
   int i;

   (void) SCIPsnprintf(consname, SCIP_MAXSTRLEN, "%s_dinsLBcons", SCIPgetProbName(scip));

   /* get the data of the variables and the best solution */
   SCIP_CALL( SCIPgetVarsData(scip, &vars, NULL, &nbinvars, NULL, NULL, NULL) );
   bestsol = SCIPgetBestSol(scip);
   assert(bestsol != NULL);

   /* memory allocation */
   SCIP_CALL( SCIPallocBufferArray(scip, &consvars, nbinvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &consvals, nbinvars) );

   /* set initial left and right hand sides of local branching constraint */
   lhs = 0.0;
   rhs = (SCIP_Real) heurdata->neighborhoodsize;

   /* create the distance function of the binary variables (to incumbent solution) */
   for( i = 0; i < nbinvars; i++ )
   {
      consvars[i] = subvars[i];
      assert(SCIPvarGetType(consvars[i]) == SCIP_VARTYPE_BINARY);
      if( fixed[i] )
      {
         consvals[i]=0.0;
         continue;
      }

      solval = SCIPgetSolVal(scip, bestsol, vars[i]);
      assert(SCIPisFeasIntegral(scip, solval));

      /* is variable i part of the binary support of the current solution? */
      if( SCIPisFeasEQ(scip, solval, 1.0) )
      {
         consvals[i] = -1.0;
         rhs -= 1.0;
         lhs -= 1.0;
      }
      else
         consvals[i] = 1.0;
   }

   /* creates local branching constraint and adds it to subscip */
   SCIP_CALL( SCIPcreateConsLinear(subscip, &cons, consname, nbinvars, consvars, consvals,
         lhs, rhs, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE) );
   SCIP_CALL( SCIPaddCons(subscip, cons) );
   SCIP_CALL( SCIPreleaseCons(subscip, &cons) );

   /* free local memory */
   SCIPfreeBufferArray(scip, &consvals);
   SCIPfreeBufferArray(scip, &consvars);

   return SCIP_OKAY;
}

/** get relaxation solution of root node (in original variables) */
static
SCIP_RETCODE getRootRelaxSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            rootsol             /**< pointer to store root relaxation solution */
   )
{
   SCIP* masterprob;
   SCIP_SOL* masterrootsol;
   SCIP_VAR** mastervars;
   int nmastervars;
   int i;

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   /* allocate memory for master root LP solution */
   SCIP_CALL( SCIPcreateSol(masterprob, &masterrootsol, NULL) );

   /* get master variable information */
   SCIP_CALL( SCIPgetVarsData(masterprob, &mastervars, &nmastervars, NULL, NULL, NULL, NULL) );
   assert(mastervars != NULL);
   assert(nmastervars >= 0);

   /* store root LP values in working master solution */
   for( i = 0; i < nmastervars; i++ )
      SCIP_CALL( SCIPsetSolVal(masterprob, masterrootsol, mastervars[i], SCIPvarGetRootSol(mastervars[i])) );

   /* calculate original root LP solution */
   SCIP_CALL( GCGtransformMastersolToOrigsol(scip, masterrootsol, rootsol) );

   /* free memory */
   SCIP_CALL( SCIPfreeSol(masterprob, &masterrootsol) );

   return SCIP_OKAY;
}

/** creates a new solution for the original problem by copying the solution of the subproblem */
static
SCIP_RETCODE createNewSol(
   SCIP*                 scip,               /**< original SCIP data structure                        */
   SCIP*                 subscip,            /**< SCIP structure of the subproblem                    */
   SCIP_VAR**            subvars,            /**< the variables of the subproblem                     */
   SCIP_HEUR*            heur,               /**< DINS heuristic structure                            */
   SCIP_SOL*             subsol,             /**< solution of the subproblem                          */
   SCIP_Bool*            success             /**< used to store whether new solution was found or not */
   )
{
#ifdef SCIP_STATISTIC
   SCIP_HEURDATA* heurdata;
#endif
   SCIP_VAR** vars;                          /* the original problem's variables                */
   int        nvars;
   SCIP_Real* subsolvals;                    /* solution values of the subproblem               */
   SCIP_SOL*  newsol;                        /* solution to be created for the original problem */

   assert(scip != NULL);
   assert(heur != NULL);
   assert(subscip != NULL);
   assert(subvars != NULL);
   assert(subsol != NULL);

#ifdef SCIP_STATISTIC
   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert( heurdata != NULL );
#endif

   /* get variables' data */
   SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, NULL, NULL, NULL, NULL) );
   /* sub-SCIP may have more variables than the number of active (transformed) variables in the main SCIP
    * since constraint copying may have required the copy of variables that are fixed in the main SCIP
    */
   assert(nvars <= SCIPgetNOrigVars(subscip));

   SCIP_CALL( SCIPallocBufferArray(scip, &subsolvals, nvars) );

   /* copy the solution */
   SCIP_CALL( SCIPgetSolVals(subscip, subsol, nvars, subvars, subsolvals) );

   /* create new solution for the original problem */
   SCIP_CALL( SCIPcreateSol(scip, &newsol, heur) );
   SCIP_CALL( SCIPsetSolVals(scip, newsol, nvars, vars, subsolvals) );

   /* try to add new solution to scip and free it immediately */
   SCIP_CALL( SCIPtrySol(scip, newsol, FALSE, FALSE, TRUE, TRUE, TRUE, success) );

#ifdef SCIP_STATISTIC
   if( *success )
   {
      if( SCIPgetSolTransObj(scip, newsol) < heurdata->bestprimalbd )
         heurdata->bestprimalbd = SCIPgetSolTransObj(scip, newsol);
   }
#endif

   SCIP_CALL( SCIPfreeSol(scip, &newsol) );

   SCIPfreeBufferArray(scip, &subsolvals);
   return SCIP_OKAY;
}


/*
 * Callback methods of primal heuristic
 */

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeGcgdins)
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


/** solving process initialization method of primal heuristic (called when branch and bound process is about to begin) */
static
SCIP_DECL_HEURINITSOL(heurInitsolGcgdins)
{
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic's data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* initialize data */
   heurdata->usednodes = 0;
   heurdata->lastnsolsfound = 0;
   heurdata->rootsol = NULL;
   heurdata->firstrun = TRUE;

   /* create flag array */
   heurdata->deltalength = SCIPgetNBinVars(scip);

   /* no binvars => no flag array needed */
   if( heurdata->deltalength > 0 )
   {
      int i;

      SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(heurdata->delta), heurdata->deltalength) );
      for( i = 0; i < heurdata->deltalength; i++ )
         heurdata->delta[i] = TRUE;
   }

#ifdef SCIP_STATISTIC
   /* initialize statistical data */
   heurdata->avgfixrate = 0.0;
   heurdata->avgzerorate = 0.0;
   heurdata->totalsols = 0;
   heurdata->subsciptime = 0.0;
   heurdata->bestprimalbd = SCIPinfinity(scip);
#endif

   return SCIP_OKAY;
}

/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed) */
static
SCIP_DECL_HEUREXITSOL(heurExitsolGcgdins)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;
#ifdef SCIP_STATISTIC
   SCIP_Longint ncalls;
#endif

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* free flag array if exist */
   if( heurdata->deltalength > 0 )
   {
      SCIPfreeBlockMemoryArray(scip, &(heurdata->delta), heurdata->deltalength);
   }

   /* free root relaxation solution */
   if( heurdata->rootsol != NULL )
      SCIP_CALL( SCIPfreeSol(scip, &heurdata->rootsol) );

#ifdef SCIP_STATISTIC
   ncalls = SCIPheurGetNCalls(heur);
   heurdata->avgfixrate /= MAX((SCIP_Real)ncalls, 1.0);
   heurdata->avgzerorate /= MAX((SCIP_Real)ncalls, 1.0);

   /* print detailed statistics */
   SCIPstatisticPrintf("LNS Statistics -- %s:\n", SCIPheurGetName(heur));
   SCIPstatisticPrintf("Calls            : %13"SCIP_LONGINT_FORMAT"\n", ncalls);
   SCIPstatisticPrintf("Sols             : %13"SCIP_LONGINT_FORMAT"\n", SCIPheurGetNSolsFound(heur));
   SCIPstatisticPrintf("Improving Sols   : %13"SCIP_LONGINT_FORMAT"\n", SCIPheurGetNBestSolsFound(heur));
   SCIPstatisticPrintf("Total Sols       : %13"SCIP_LONGINT_FORMAT"\n", heurdata->totalsols);
   SCIPstatisticPrintf("subSCIP time     : %13.2f\n", heurdata->subsciptime);
   SCIPstatisticPrintf("subSCIP nodes    : %13"SCIP_LONGINT_FORMAT"\n", heurdata->usednodes);
   SCIPstatisticPrintf("Avg. fixing rate : %13.2f\n", 100.0 * heurdata->avgfixrate);
   SCIPstatisticPrintf("Avg. zero rate   : %13.2f\n", 100.0 * heurdata->avgzerorate);
   SCIPstatisticPrintf("Best primal bd.  :");
   if( SCIPisInfinity(scip, heurdata->bestprimalbd) )
      SCIPstatisticPrintf("      infinity\n");
   else
      SCIPstatisticPrintf(" %13.6e\n", heurdata->bestprimalbd);
   SCIPstatisticPrintf("\n");
#endif

   return SCIP_OKAY;
}

/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecGcgdins)
{  /*lint --e{715}*/
   SCIP* masterprob;
   SCIP_HEURDATA* heurdata;
   SCIP* subscip;                            /* the subproblem created by DINS                               */
   SCIP_VAR** subvars;                       /* subproblem's variables                                       */
   SCIP_VAR** vars;                          /* variables of the original problem                            */
   SCIP_HASHMAP* varmapfw;                   /* mapping of SCIP variables to sub-SCIP variables */
   SCIP_SOL* bestsol;                        /* best solution known so far                                   */
   SCIP_SOL** sols;                          /* list of known solutions                                      */

   SCIP_Bool* fixed;                         /* fixing flag array                                            */
   SCIP_Bool* delta;                         /* flag array if variable value changed during solution process */


   SCIP_Longint maxnnodes;                   /* maximum number of subnodes                                   */
   SCIP_Longint nsubnodes;                   /* nodelimit for subscip                                        */
   SCIP_Longint nsolsfound;

   SCIP_Real timelimit;                      /* timelimit for subscip (equals remaining time of scip)        */
   SCIP_Real cutoff;                         /* objective cutoff for the subproblem                          */
   SCIP_Real upperbound;
   SCIP_Real memorylimit;                    /* memory limit for solution process of subscip                 */
   SCIP_Real lpsolval;
   SCIP_Real rootlpsolval;
   SCIP_Real mipsolval;
   SCIP_Real solval;
#ifdef SCIP_STATISTIC
   SCIP_Real allfixingrate;                  /* percentage of all variables fixed               */
   SCIP_Real intfixingrate;                  /* percentage of integer variables fixed           */
   SCIP_Real zerofixingrate;                 /* percentage of variables fixed to zero           */
#endif

   int ufcount;                              /* counts the number of true fixing flag entries                */
   int nvars;                                /* number of variables in original SCIP                         */
   int nbinvars;                             /* number of binary variables in original SCIP                  */
   int nintvars;                             /* number of general integer variables in original SCIP         */
   int nsols;                                /* number of known solutions                                    */
   int nsubsols;
   int checklength;
   int fixingcounter;
   int zerocounter;
   int i;
   int j;

   SCIP_Bool success;                        /* used to store whether new solution was found or not          */
   SCIP_Bool infeasible;                     /* stores whether the hard fixing of a variables was feasible or not */

   SCIP_RETCODE retcode;

   assert(heur != NULL);
   assert(scip != NULL);
   assert(result != NULL);

   /* get master problem */
   masterprob = GCGgetMasterprob(scip);
   assert(masterprob != NULL);

   *result = SCIP_DELAYED;

   /* do not execute the heuristic on invalid relaxation solutions
    * (which is the case if the node has been cut off)
    */
   if( !SCIPisRelaxSolValid(scip) )
      return SCIP_OKAY;

   /* only call heuristic, if feasible solution is available */
   if( SCIPgetNSols(scip) <= 0 )
      return SCIP_OKAY;

   /* only call heuristic, if an optimal LP solution is at hand */
   if( SCIPgetStage(masterprob) > SCIP_STAGE_SOLVING || SCIPgetLPSolstat(masterprob) != SCIP_LPSOLSTAT_OPTIMAL )
      return SCIP_OKAY;

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);
   delta = heurdata->delta;

   /* only call heuristic, if enough nodes were processed since last incumbent */
   if( SCIPgetNNodes(scip) - SCIPgetSolNodenum(scip, SCIPgetBestSol(scip)) < heurdata->nwaitingnodes )
      return SCIP_OKAY;

   *result = SCIP_DIDNOTRUN;

   /* determine the node limit for the current process */
   maxnnodes = (SCIP_Longint) (heurdata->nodesquot * SCIPgetNNodes(scip));

   /* reward DINS if it succeeded often */
   maxnnodes = (SCIP_Longint) (maxnnodes * (1.0 + 2.0 * (SCIPheurGetNBestSolsFound(heur)+1.0) / (SCIPheurGetNCalls(heur) + 1.0)));

   /* count the setup costs for the sub-MIP as 100 nodes */
   maxnnodes -= 100 * SCIPheurGetNCalls(heur);
   maxnnodes += heurdata->nodesofs;

   /* determine the node limit for the current process */
   nsubnodes = maxnnodes - heurdata->usednodes;
   nsubnodes = MIN(nsubnodes , heurdata->maxnodes);

   /* check whether we have enough nodes left to call sub problem solving */
   if( nsubnodes < heurdata->minnodes )
      return SCIP_OKAY;

   if( SCIPisStopped(scip) )
     return SCIP_OKAY;

   /* get required data of the original problem */
   SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, &nbinvars, &nintvars, NULL, NULL) );
   assert(nbinvars <= nvars);

   /* do not run heuristic if only continuous variables are present */
   if( nbinvars == 0 && nintvars == 0 )
      return SCIP_OKAY;

   assert(vars != NULL);

   /* if the heuristic is running for the first time, the root relaxation solution needs to be stored */
   if( heurdata->firstrun )
   {
      assert(heurdata->rootsol == NULL);
      SCIP_CALL( getRootRelaxSol(scip, &heurdata->rootsol) );
      assert(heurdata->rootsol != NULL);
      heurdata->firstrun = FALSE;
   }

   /* initialize the subproblem */
   SCIP_CALL( SCIPcreate(&subscip) );

   /* create the variable mapping hash map */
   SCIP_CALL( SCIPallocBufferArray(scip, &subvars, nvars) );
   SCIP_CALL( SCIPhashmapCreate(&varmapfw, SCIPblkmem(subscip), nvars) );

   success = FALSE;
   if( heurdata->uselprows )
   {
      char probname[SCIP_MAXSTRLEN];

      /* copy all plugins */
      SCIP_CALL( SCIPincludeDefaultPlugins(subscip) );

      /* get name of the original problem and add the string "_gcgdinssub" */
      (void) SCIPsnprintf(probname, SCIP_MAXSTRLEN, "%s_gcgdinssub", SCIPgetProbName(scip));

      /* create the subproblem */
      SCIP_CALL( SCIPcreateProb(subscip, probname, NULL, NULL, NULL, NULL, NULL, NULL, NULL) );

      /* copy all variables */
      SCIP_CALL( SCIPcopyVars(scip, subscip, varmapfw, NULL, NULL, NULL, 0, TRUE) );
   }
   else
   {
      SCIP_CALL( SCIPcopy(scip, subscip, varmapfw, NULL, "gcgdins", TRUE, FALSE, FALSE, TRUE, &success) );

      if( heurdata->copycuts )
      {
         /* copies all active cuts from cutpool of sourcescip to linear constraints in targetscip */
         SCIP_CALL( SCIPcopyCuts(scip, subscip, varmapfw, NULL, TRUE, NULL) );
      }

      SCIPdebugMessage("Copying the SCIP instance was %ssuccessful.\n", success ? "" : "not ");
   }

   for( i = 0; i < nvars; i++ )
     subvars[i] = (SCIP_VAR*) SCIPhashmapGetImage(varmapfw, vars[i]);

   /* free hash map */
   SCIPhashmapFree(&varmapfw);

   fixingcounter = 0;
   zerocounter = 0;

   SCIPstatisticPrintf("gcgdins statistic: called at node %"SCIP_LONGINT_FORMAT"\n", SCIPgetNNodes(scip));

   /* create variables and rebound them if their bounds differ by more than 0.5 */
   SCIP_CALL( createSubproblem(scip, subscip, vars, subvars, nbinvars, nintvars, heurdata->uselprows, &fixingcounter, &zerocounter) );
   SCIPdebugMessage("DINS subproblem: %d vars (%d binvars & %d intvars), %d cons\n",
      SCIPgetNVars(subscip), SCIPgetNBinVars(subscip) , SCIPgetNIntVars(subscip) , SCIPgetNConss(subscip));

   *result = SCIP_DIDNOTFIND;

   /* do not abort subproblem on CTRL-C */
   SCIP_CALL( SCIPsetBoolParam(subscip, "misc/catchctrlc", FALSE) );

   /* disable output to console */
   SCIP_CALL( SCIPsetIntParam(subscip, "display/verblevel", 0) );

   /* check whether there is enough time and memory left */
   SCIP_CALL( SCIPgetRealParam(scip, "limits/time", &timelimit) );
   if( !SCIPisInfinity(scip, timelimit) )
      timelimit -= SCIPgetSolvingTime(scip);
   SCIP_CALL( SCIPgetRealParam(scip, "limits/memory", &memorylimit) );

   /* substract the memory already used by the main SCIP and the estimated memory usage of external software */
   if( !SCIPisInfinity(scip, memorylimit) )
   {
      memorylimit -= SCIPgetMemUsed(scip)/1048576.0;
      memorylimit -= SCIPgetMemExternEstim(scip)/1048576.0;
   }

   /* abort if no time is left or not enough memory to create a copy of SCIP, including external memory usage */
   if( timelimit <= 0.0 || memorylimit <= 2.0*SCIPgetMemExternEstim(scip)/1048576.0 )
      goto TERMINATE;

   /* set limits for the subproblem */
   SCIP_CALL( SCIPsetLongintParam(subscip, "limits/nodes", nsubnodes) );
   SCIP_CALL( SCIPsetIntParam(subscip, "limits/bestsol", 3) );
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/time", timelimit) );
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/memory", memorylimit) );

   /* forbid recursive call of heuristics and separators solving subMIPs */
   SCIP_CALL( SCIPsetSubscipsOff(subscip, TRUE) );

   /* disable cutting plane separation */
   SCIP_CALL( SCIPsetSeparating(subscip, SCIP_PARAMSETTING_OFF, TRUE) );

   /* disable expensive presolving */
   SCIP_CALL( SCIPsetPresolving(subscip, SCIP_PARAMSETTING_FAST, TRUE) );

   /* use best estimate node selection */
   if( SCIPfindNodesel(subscip, "estimate") != NULL && !SCIPisParamFixed(subscip, "nodeselection/estimate/stdpriority") )
   {
      SCIP_CALL( SCIPsetIntParam(subscip, "nodeselection/estimate/stdpriority", INT_MAX/4) );
   }

   /* use inference branching */
   if( SCIPfindBranchrule(subscip, "inference") != NULL && !SCIPisParamFixed(subscip, "branching/inference/priority") )
   {
      SCIP_CALL( SCIPsetIntParam(subscip, "branching/inference/priority", INT_MAX/4) );
   }

   /* disable conflict analysis */
   if( !SCIPisParamFixed(subscip, "conflict/enable") )
   {
      SCIP_CALL( SCIPsetBoolParam(subscip, "conflict/enable", FALSE) );
   }

   /* get the best MIP-solution known so far */
   bestsol = SCIPgetBestSol(scip);
   assert(bestsol != NULL);

   /* get solution pool and number of solutions in pool */
   sols = SCIPgetSols(scip);
   nsols = SCIPgetNSols(scip);
   nsolsfound = SCIPgetNSolsFound(scip);
   checklength = MIN(nsols, heurdata->solnum);
   assert(sols != NULL);
   assert(nsols > 0);

   /* create fixing flag array */
   SCIP_CALL( SCIPallocBufferArray(scip, &fixed, nbinvars) );

   /* if new binary variables have been created, e.g., due to column generation, reallocate the delta array */
   if( heurdata->deltalength < nbinvars )
   {
      int newsize;

      newsize = SCIPcalcMemGrowSize(scip, nbinvars);
      assert(newsize >= nbinvars);

      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &heurdata->delta, heurdata->deltalength, newsize) );
      delta = heurdata->delta;

      /* initialize new part of delta array */
      for( i = heurdata->deltalength; i < newsize; i++ )
         delta[i] = TRUE;

      heurdata->deltalength = newsize;
   }

   /* fixing for binary variables */
   /* hard fixing for some with mipsol(s)=lpsolval=rootlpsolval and preparation for soft fixing for the remaining */
   ufcount = 0;
   for( i = 0; i < nbinvars; i++ )
   {
      /* soft fixing if the variable somewhen changed its value or the relaxations differ by adding a local branching constraint */
      fixed[i] = FALSE;

      /* get the current relaxation solution for each variable */
      lpsolval = SCIPgetRelaxSolVal(scip, vars[i]);
      /* get the current MIP solution for each variable */
      mipsolval = SCIPgetSolVal(scip, bestsol, vars[i]);
      /* get the root relaxation solution for each variable */
      rootlpsolval = SCIPgetSolVal(scip, heurdata->rootsol, vars[i]);

      if( SCIPisFeasEQ(scip, lpsolval, mipsolval) && SCIPisFeasEQ(scip, mipsolval, rootlpsolval) )
      {
         /* update delta */
         if( nsols > 1 && heurdata->lastnsolsfound != nsolsfound && delta[i] ) /* no need to update delta[i] if already FALSE */
         {
            /* no need to update delta[i] if already FALSE or sols[i] already checked on previous run or worse than DINS-solution of last run */
            for( j = 0; delta[i] && j < checklength && SCIPgetSolHeur(scip, sols[j]) != heur ; j++ )
            {
               solval = SCIPgetSolVal(scip, sols[j], vars[i]);
               delta[i] = delta[i] && SCIPisFeasEQ(scip, mipsolval, solval);
            }
         }

         /* hard fixing if rootlpsolval=nodelpsolval=mipsolval(s) and delta (is TRUE) */
         if( delta[i] && SCIPisFeasEQ(scip, mipsolval, lpsolval) && SCIPisFeasEQ(scip, mipsolval, rootlpsolval )
            && SCIPisFeasEQ(scip, rootlpsolval, lpsolval) )
         {
            SCIP_CALL( SCIPfixVar(subscip, subvars[i], mipsolval, &infeasible, &success) );
            fixed[i] = !infeasible;
            if( !success )
            {
               SCIPdebugMessage("variable %d was already fixed\n", i);
            }
            else
            {
               ++fixingcounter;
               if( SCIPisZero(scip, mipsolval) )
                  ++zerocounter;
            }
            if( infeasible )
            {
               SCIPdebugMessage("fixing of variable %d to value %f was infeasible\n", i, mipsolval);
            }
         }
      }
      if( !fixed[i] )
         ufcount++;
   }

#ifdef SCIP_STATISTIC
   intfixingrate = (SCIP_Real)fixingcounter / (SCIP_Real)(MAX(nbinvars + nintvars, 1));
   zerofixingrate = (SCIP_Real)zerocounter / MAX((SCIP_Real)fixingcounter, 1.0);
#endif

   /* store the number of found solutions for next run */
   heurdata->lastnsolsfound = nsolsfound;

   /* perform prepared softfixing for all unfixed vars if the number of unfixed vars is larger than the neighborhoodsize (otherwise it will be useless) */
   if( ufcount > heurdata->neighborhoodsize )
   {
      SCIP_CALL( addLocalBranchingConstraint(scip, subscip, subvars, heurdata, fixed) );
   }

   /* free fixing flag array */
   SCIPfreeBufferArray(scip, &fixed);

   /* add an objective cutoff */
   assert(!SCIPisInfinity(scip, SCIPgetUpperbound(scip)));

   if( !SCIPisInfinity(scip, -1.0*SCIPgetLowerbound(scip)) )
   {
      cutoff = (1 - heurdata->minimprove) * SCIPgetUpperbound(scip) + heurdata->minimprove * SCIPgetLowerbound(scip);
      upperbound = SCIPgetUpperbound(scip) - SCIPsumepsilon(scip);
      cutoff = MIN(upperbound, cutoff);
   }
   else
   {
      if( SCIPgetUpperbound(scip) >= 0 )
         cutoff = (1 - heurdata->minimprove) * SCIPgetUpperbound(scip);
      else
         cutoff = (1 + heurdata->minimprove) * SCIPgetUpperbound(scip);
      upperbound = SCIPgetUpperbound(scip) - SCIPsumepsilon(scip);
      cutoff = MIN(upperbound, cutoff);
   }
   SCIP_CALL( SCIPsetObjlimit(subscip, cutoff) );

#ifdef SCIP_STATISTIC
   heurdata->avgfixrate += intfixingrate;
   heurdata->avgzerorate += zerofixingrate;
#endif

   /* presolve the subproblem */
   retcode = SCIPpresolve(subscip);

   /* errors in solving the subproblem should not kill the overall solving process;
    * hence, the return code is caught and a warning is printed, only in debug mode, SCIP will stop.
    */
   if( retcode != SCIP_OKAY )
   {
#ifndef NDEBUG
      SCIP_CALL( retcode );
#endif
      SCIPwarningMessage(scip, "Error while presolving subproblem in GCG RINS heuristic; sub-SCIP terminated with code <%d>\n",retcode);
      goto TERMINATE;
   }

   SCIPdebugMessage("GCG DINS presolved subproblem: %d vars, %d cons, success=%u\n", SCIPgetNVars(subscip), SCIPgetNConss(subscip), success);

#ifdef SCIP_STATISTIC
   allfixingrate = (SCIPgetNOrigVars(subscip) - SCIPgetNVars(subscip)) / (SCIP_Real)SCIPgetNOrigVars(subscip);

   /* additional variables added in presolving may lead to the subSCIP having more variables than the original */
   allfixingrate = MAX(allfixingrate, 0.0);
#endif

   /* solve the subproblem */
   SCIPdebugMessage("solving DINS sub-MIP with neighborhoodsize %d and maxnodes %"SCIP_LONGINT_FORMAT"\n", heurdata->neighborhoodsize, nsubnodes);
   retcode = SCIPsolve(subscip);

   /* Errors in solving the subproblem should not kill the overall solving process
    * Hence, the return code is caught and a warning is printed, only in debug mode, SCIP will stop.
    */
   if( retcode != SCIP_OKAY )
   {
#ifndef NDEBUG
      SCIP_CALL( retcode );
#endif
      SCIPwarningMessage(scip, "Error while solving subproblem in DINS heuristic; sub-SCIP terminated with code <%d>\n", retcode);
   }

   heurdata->usednodes += SCIPgetNNodes(subscip);
   nsubsols = SCIPgetNSols(subscip);
#ifdef SCIP_STATISTIC
   heurdata->subsciptime += SCIPgetTotalTime(subscip);
   heurdata->totalsols += nsubsols;
#endif
   SCIPdebugMessage("DINS used %"SCIP_LONGINT_FORMAT"/%"SCIP_LONGINT_FORMAT" nodes and found %d solutions\n", SCIPgetNNodes(subscip), nsubnodes, nsubsols);

   /* check, whether a  (new) solution was found */
   if( nsubsols > 0 )
   {
      SCIP_SOL** subsols;

      /* check, whether a solution was found; due to numerics, it might happen that not all solutions are feasible -> try all solutions until one was accepted */
      subsols = SCIPgetSols(subscip);
      success = FALSE;
      for( i = 0; i < nsubsols && !success; ++i )
      {
         SCIP_CALL( createNewSol(scip, subscip, subvars, heur, subsols[i], &success) );
         if( success )
            *result = SCIP_FOUNDSOL;
      }

#ifdef SCIP_STATISTIC
      SCIPstatisticPrintf("gcgdins statistic: fixed %6.3f integer variables ( %6.3f zero), %6.3f all variables, needed %6.1f sec (SCIP time: %6.1f sec), %"SCIP_LONGINT_FORMAT" nodes, found %d solutions, solution %10.4f found at node %"SCIP_LONGINT_FORMAT"\n",
         intfixingrate, zerofixingrate, allfixingrate, SCIPgetSolvingTime(subscip), SCIPgetSolvingTime(scip), SCIPgetNNodes(subscip), nsubsols,
         success ? SCIPgetPrimalbound(scip) : SCIPinfinity(scip), nsubsols > 0 ? SCIPsolGetNodenum(SCIPgetBestSol(subscip)) : -1 );
#endif

   }

 TERMINATE:
   /* free subproblem */
   SCIPfreeBufferArray(scip, &subvars);
   SCIP_CALL( SCIPfree(&subscip) );

   return SCIP_OKAY;
}


/*
 * primal heuristic specific interface methods
 */

/** creates the GCG DINS primal heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurGcgdins(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEURDATA* heurdata;
   SCIP_HEUR* heur;

   /* create Gcgdins primal heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &heurdata) );

   /* include primal heuristic */
   SCIP_CALL( SCIPincludeHeurBasic(scip, &heur,
         HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP, heurExecGcgdins, heurdata) );

   assert(heur != NULL);

   /* set non-NULL pointers to callback methods */
   SCIP_CALL( SCIPsetHeurFree(scip, heur, heurFreeGcgdins) );
   SCIP_CALL( SCIPsetHeurInitsol(scip, heur, heurInitsolGcgdins) );
   SCIP_CALL( SCIPsetHeurExitsol(scip, heur, heurExitsolGcgdins) );

   /* add DINS primal heuristic parameters */
   SCIP_CALL( SCIPaddLongintParam(scip, "heuristics/"HEUR_NAME"/nodesofs",
         "number of nodes added to the contingent of the total nodes",
         &heurdata->nodesofs, FALSE, DEFAULT_NODESOFS, 0LL, SCIP_LONGINT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/nodesquot",
         "contingent of sub problem nodes in relation to the number of nodes of the original problem",
         &heurdata->nodesquot, FALSE, DEFAULT_NODESQUOT, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddLongintParam(scip, "heuristics/"HEUR_NAME"/minnodes",
         "minimum number of nodes required to start the subproblem",
         &heurdata->minnodes, FALSE, DEFAULT_MINNODES, 0LL, SCIP_LONGINT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/"HEUR_NAME"/solnum",
         "number of pool-solutions to be checked for flag array update (for hard fixing of binary variables)",
         &heurdata->solnum, FALSE, DEFAULT_SOLNUM, 1, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/"HEUR_NAME"/neighborhoodsize",
         "radius (using Manhattan metric) of the incumbent's neighborhood to be searched",
         &heurdata->neighborhoodsize, FALSE, DEFAULT_NEIGHBORHOODSIZE, 1, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddLongintParam(scip, "heuristics/"HEUR_NAME"/maxnodes",
         "maximum number of nodes to regard in the subproblem",
         &heurdata->maxnodes,TRUE,DEFAULT_MAXNODES, 0LL, SCIP_LONGINT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/minimprove",
         "factor by which "HEUR_NAME" should at least improve the incumbent",
         &heurdata->minimprove, TRUE, DEFAULT_MINIMPROVE, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddLongintParam(scip, "heuristics/"HEUR_NAME"/nwaitingnodes",
         "number of nodes without incumbent change that heuristic should wait",
         &heurdata->nwaitingnodes, TRUE, DEFAULT_NWAITINGNODES, 0LL, SCIP_LONGINT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/uselprows",
         "should subproblem be created out of the rows in the LP rows?",
         &heurdata->uselprows, TRUE, DEFAULT_USELPROWS, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/copycuts",
         "if uselprows == FALSE, should all active cuts from cutpool be copied to constraints in subproblem?",
         &heurdata->copycuts, TRUE, DEFAULT_COPYCUTS, NULL, NULL) );

   return SCIP_OKAY;
}
