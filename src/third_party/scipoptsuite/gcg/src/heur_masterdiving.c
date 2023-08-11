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

/**@file   heur_masterdiving.c
 * @brief  primal heuristic interface for LP diving heuristics on the master variables
 * @author Tobias Achterberg
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "heur_masterdiving.h"
#include "pricer_gcg.h"
#include "relax_gcg.h"


#define HEUR_TIMING           SCIP_HEURTIMING_AFTERLPPLUNGE
#define HEUR_USESSUBSCIP      FALSE  /**< does the heuristic use a secondary SCIP instance? */


/*
 * Default parameter settings for all diving heuristics
 */

#define DEFAULT_MINRELDEPTH         0.0 /**< minimal relative depth to start diving */
#define DEFAULT_MAXRELDEPTH         1.0 /**< maximal relative depth to start diving */
#define DEFAULT_MAXLPITERQUOT      0.05 /**< maximal fraction of diving LP iterations compared to node LP iterations */
#define DEFAULT_MAXLPITEROFS       1000 /**< additional number of allowed LP iterations */
#define DEFAULT_MAXPRICEROUNDS        0 /**< maximal number of allowed pricing rounds (-1: no limit) */
#define DEFAULT_USEFARKASONLY     FALSE /**< perform pricing only if infeasibility is encountered */
#define DEFAULT_MAXDIVEUBQUOT       0.8 /**< maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound)
                                         *   where diving is performed (0.0: no limit) */
#define DEFAULT_MAXDIVEAVGQUOT      0.0 /**< maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound)
                                         *   where diving is performed (0.0: no limit) */
#define DEFAULT_MAXDIVEUBQUOTNOSOL  0.1 /**< maximal UBQUOT when no solution was found yet (0.0: no limit) */
#define DEFAULT_MAXDIVEAVGQUOTNOSOL 0.0 /**< maximal AVGQUOT when no solution was found yet (0.0: no limit) */
#define DEFAULT_BACKTRACK         FALSE /**< single backtracking by choosing another variable in case of infeasibility */
#define DEFAULT_MAXDISCREPANCY        2 /**< maximal discrepancy allowed in backtracking and limited discrepancy search */
#define DEFAULT_MAXDISCDEPTH          0 /**< maximal depth until which a limited discrepancy search is performed */

#define MINLPITER                 10000 /**< minimal number of LP iterations allowed in each LP solving call */

#ifdef SCIP_STATISTIC
#define EVENTHDLR_NAME         "masterdiving"
#define EVENTHDLR_DESC         "event handler for masterdiving solution statistics"
#endif


/* locally defined heuristic data for all diving heuristics */
struct SCIP_HeurData
{
   GCG_DECL_DIVINGFREE   ((*divingfree));    /**< destructor of diving heuristic */
   GCG_DECL_DIVINGINIT   ((*divinginit));    /**< initialize diving heuristic */
   GCG_DECL_DIVINGEXIT   ((*divingexit));    /**< deinitialize diving heuristic */
   GCG_DECL_DIVINGINITSOL ((*divinginitsol)); /**< solving process initialization method of diving heuristic */
   GCG_DECL_DIVINGEXITSOL ((*divingexitsol)); /**< solving process deinitialization method of diving heuristic */
   GCG_DECL_DIVINGINITEXEC ((*divinginitexec)); /**< execution initialization method of diving heuristic */
   GCG_DECL_DIVINGEXITEXEC ((*divingexitexec)); /**< execution deinitialization method of diving heuristic */
   GCG_DECL_DIVINGSELECTVAR ((*divingselectvar)); /**< variable selection method of diving heuristic */
   GCG_DIVINGDATA*       divingdata;         /**< diving rule specific data */

   SCIP_SOL*             sol;                /**< working solution */
   SCIP_Real             minreldepth;        /**< minimal relative depth to start diving */
   SCIP_Real             maxreldepth;        /**< maximal relative depth to start diving */
   SCIP_Real             maxlpiterquot;      /**< maximal fraction of diving LP iterations compared to node LP iterations */
   int                   maxlpiterofs;       /**< additional number of allowed LP iterations */
   int                   maxpricerounds;     /**< maximal number of allowed pricing rounds (-1: no limit) */
   SCIP_Bool             usefarkasonly;      /**< perform pricing only if infeasibility is encountered */
   SCIP_Real             maxdiveubquot;      /**< maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound)
                                              *   where diving is performed (0.0: no limit) */
   SCIP_Real             maxdiveavgquot;     /**< maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound)
                                              *   where diving is performed (0.0: no limit) */
   SCIP_Real             maxdiveubquotnosol; /**< maximal UBQUOT when no solution was found yet (0.0: no limit) */
   SCIP_Real             maxdiveavgquotnosol;/**< maximal AVGQUOT when no solution was found yet (0.0: no limit) */
   SCIP_Bool             backtrack;          /**< single backtracking by choosing another variable in case of infeasibility */
   int                   maxdiscrepancy;     /**< maximal discrepancy allowed in backtracking and limited discrepancy search */
   int                   maxdiscdepth;       /**< maximal depth until which a limited discrepancy search is performed */
   SCIP_Longint          nlpiterations;      /**< LP iterations used in this heuristic */
   SCIP_Longint          npricerounds;       /**< pricing rounds used in this heuristic */
   int                   nsuccess;           /**< number of runs that produced at least one feasible solution */

#ifdef SCIP_STATISTIC
   SCIP_Longint          ncalls;             /**< number of calls                                                           */
   SCIP_Longint          nsols;              /**< number of solutions                                                       */
   SCIP_Longint          nimpsols;           /**< number of improving solutions                                             */
   SCIP_Longint          ndivesols;          /**< number of integral diving LP solutions                                    */
   SCIP_Longint          nimpdivesols;       /**< number of improving integral diving LP solutions                          */
   SCIP_Longint          nroundsols;         /**< number of integral solutions that have been obtained by rounding          */
   SCIP_Longint          nimproundsols;      /**< number of improving integral solutions obtained by rounding               */
   SCIP_Longint          ndivenodes;         /**< number of diving nodes                                                    */
   SCIP_Longint          nfarkas;            /**< number of times an infeasibility was resolved by Farkas pricing           */
   SCIP_Longint          notherdirections;   /**< number of times a cutoff was resolved by branching in the other direction */
   SCIP_Longint          nbacktracks;        /**< number of times a single backtracking at a deeper node was performed      */
   SCIP_Longint          ndiscsearches;      /**< number of times a limited discrepancy search was performed                */
   SCIP_Real             bestprimalbd;       /**< objective value of best solution found by this heuristic                  */
   SCIP_Bool             bestsolrounded;     /**< was the best solution obtained by rounding?                               */
#endif
};

#ifdef SCIP_STATISTIC
/** event handler data */
struct SCIP_EventhdlrData
{
   SCIP_HEUR**           heurs;              /**< diving heuristics known to the event handler */
   int                   nheurs;             /**< number of diving heuristics known to the event handler */
   SCIP_HEUR*            runningheur;        /**< the diving heuristic that is currently running, or NULL */
};
#endif


/*
 * local methods
 */



/*
 * Callback methods
 */

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeMasterdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   if( heurdata->divingfree != NULL )
   {
      SCIP_CALL( heurdata->divingfree(scip, heur) );
   }

   /* free heuristic data */
   SCIPfreeMemory(scip, &heurdata);
   SCIPheurSetData(heur, NULL);

   return SCIP_OKAY;
}


/** initialization method of primal heuristic (called after problem was transformed) */
static
SCIP_DECL_HEURINIT(heurInitMasterdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* create working solution */
   SCIP_CALL( SCIPcreateSol(scip, &heurdata->sol, heur) );

   heurdata->nlpiterations = 0;
   heurdata->npricerounds = 0;
   heurdata->nsuccess = 0;

   /* diving rule specific initialization */
   if( heurdata->divinginit != NULL )
   {
      SCIP_CALL( heurdata->divinginit(scip, heur) );
   }

   return SCIP_OKAY;
}


/** deinitialization method of primal heuristic (called before transformed problem is freed) */
static
SCIP_DECL_HEUREXIT(heurExitMasterdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* diving rule specific deinitialization */
   if( heurdata->divingexit != NULL )
   {
      SCIP_CALL( heurdata->divingexit(scip, heur) );
   }

   /* free working solution */
   SCIP_CALL( SCIPfreeSol(scip, &heurdata->sol) );

   return SCIP_OKAY;
}


/** solving process initialization method of primal heuristic (called when branch and bound process is about to begin) */
static
SCIP_DECL_HEURINITSOL(heurInitsolMasterdiving)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

#ifdef SCIP_STATISTIC
   /* initialize statistics */
   heurdata->ncalls = 0;
   heurdata->nsols = 0;
   heurdata->nimpsols = 0;
   heurdata->ndivesols = 0;
   heurdata->nimpdivesols = 0;
   heurdata->nroundsols = 0;
   heurdata->nimproundsols = 0;
   heurdata->ndivenodes = 0;
   heurdata->nfarkas = 0;
   heurdata->notherdirections = 0;
   heurdata->nbacktracks = 0;
   heurdata->ndiscsearches = 0;
   heurdata->bestprimalbd = SCIPinfinity(scip);
   heurdata->bestsolrounded = FALSE;
#endif

   /* diving rule specific initialization */
   if( heurdata->divinginitsol != NULL )
   {
      SCIP_CALL( heurdata->divinginitsol(scip, heur) );
   }

   return SCIP_OKAY;
}


/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed) */
static
SCIP_DECL_HEUREXITSOL(heurExitsolMasterdiving)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* diving rule specific deinitialization */
   if( heurdata->divingexitsol != NULL )
   {
      SCIP_CALL( heurdata->divingexitsol(scip, heur) );
   }

   return SCIP_OKAY;
}


/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecMasterdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP* origprob;
#ifdef SCIP_STATISTIC
   SCIP_EVENTHDLR* eventhdlr;
   SCIP_EVENTHDLRDATA* eventhdlrdata;
#endif
   SCIP_HEURDATA* heurdata;
   SCIP_LPSOLSTAT lpsolstat;
   SCIP_VAR** selectedvars;
   SCIP_VAR** tabulist;
   int* discrepancies;
   SCIP_Real searchubbound;
   SCIP_Real searchavgbound;
   SCIP_Real searchbound;
   SCIP_Real objval;
   SCIP_Real oldobjval;
   SCIP_Bool lperror;
   SCIP_Bool cutoff;
   SCIP_Longint ncalls;
   SCIP_Longint nsolsfound;
   SCIP_Longint nlpiterations;         /* lp iterations performed in one single diving loop */
   SCIP_Longint maxnlpiterations;
   int totalpricerounds;               /* pricing rounds performed in one call of the heuristic */
   int nlpcands;
   int startnlpcands;
   int depth;
   int maxdepth;
   int maxdivedepth;
   int divedepth;
   int discrepancy;

#ifdef NDEBUG
   SCIP_RETCODE retstat;
#endif

#ifdef SCIP_STATISTIC
   /* variable declarations for additional statistics */
   int ndivenodes;                     /* number of diving nodes */
   int maxreacheddepth;                /* maximal diving depth reached in this call */
   int nfarkas;                        /* number of times an infeasibility was resolved by Farkas pricing */
   int nbacktracks;                    /* number of times a single backtracking at a deeper node was performed */
   int ndiscsearches;                  /* number of times a limited discrepancy search was performed */
   SCIP_Longint totallpiters;          /* lp iterations performed in one call of the heuristic */
   SCIP_CLOCK* lptime;                 /* time spent for solving diving LPs */
#endif

   int i;

   assert(heur != NULL);
   assert(scip != NULL);
   assert(result != NULL);

   /* get original problem */
   origprob = GCGmasterGetOrigprob(scip);
   assert(origprob != NULL);

#ifdef SCIP_STATISTIC
   /* get the masterdiving event handler and its data */
   eventhdlr = SCIPfindEventhdlr(scip, EVENTHDLR_NAME);
   assert(eventhdlr != NULL);
   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);
#endif

   *result = SCIP_DELAYED;

   /* only call heuristic, if an optimal LP solution is at hand */
   if( !SCIPhasCurrentNodeLP(scip) || SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL )
      return SCIP_OKAY;

   /* only call heuristic, if the LP solution is basic (which allows fast resolve in diving) */
   if( !SCIPisLPSolBasic(scip) )
      return SCIP_OKAY;

   /* don't dive two times at the same node */
   if( SCIPgetLastDivenode(scip) == SCIPgetNNodes(scip) && SCIPgetDepth(scip) > 0 )
      return SCIP_OKAY;

   *result = SCIP_DIDNOTRUN;

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* check if fundamental diving callbacks are present */
   assert(heurdata->divingselectvar != NULL);

   /* only try to dive, if we are in the correct part of the tree, given by minreldepth and maxreldepth */
   depth = SCIPgetDepth(scip);
   maxdepth = SCIPgetMaxDepth(scip);
   maxdepth = MAX(maxdepth, 30);
   if( depth < heurdata->minreldepth*maxdepth || depth > heurdata->maxreldepth*maxdepth )
      return SCIP_OKAY;

   /* calculate the maximal number of LP iterations until heuristic is aborted */
   nlpiterations = SCIPgetNNodeLPIterations(scip);
   ncalls = SCIPheurGetNCalls(heur);
   nsolsfound = 10*SCIPheurGetNBestSolsFound(heur) + heurdata->nsuccess;
   maxnlpiterations = (SCIP_Longint)((1.0 + 10.0*(nsolsfound+1.0)/(ncalls+1.0)) * heurdata->maxlpiterquot * nlpiterations);
   maxnlpiterations += heurdata->maxlpiterofs;

   /* don't try to dive, if we took too many LP iterations during diving */
   if( heurdata->nlpiterations >= maxnlpiterations )
      return SCIP_OKAY;

   /* allow at least a certain number of LP iterations in this dive */
   maxnlpiterations = MAX(maxnlpiterations, heurdata->nlpiterations + MINLPITER);

   /* get number of fractional variables that should be integral */
   nlpcands = SCIPgetNLPBranchCands(scip);

   /* don't try to dive, if there are no fractional variables */
   if( nlpcands == 0 )
      return SCIP_OKAY;

   /* calculate the objective search bound */
   if( SCIPgetNSolsFound(scip) == 0 )
   {
      if( heurdata->maxdiveubquotnosol > 0.0 )
         searchubbound = SCIPgetLowerbound(scip)
            + heurdata->maxdiveubquotnosol * (SCIPgetCutoffbound(scip) - SCIPgetLowerbound(scip));
      else
         searchubbound = SCIPinfinity(scip);
      if( heurdata->maxdiveavgquotnosol > 0.0 )
         searchavgbound = SCIPgetLowerbound(scip)
            + heurdata->maxdiveavgquotnosol * (SCIPgetAvgLowerbound(scip) - SCIPgetLowerbound(scip));
      else
         searchavgbound = SCIPinfinity(scip);
   }
   else
   {
      if( heurdata->maxdiveubquot > 0.0 )
         searchubbound = SCIPgetLowerbound(scip)
            + heurdata->maxdiveubquot * (SCIPgetCutoffbound(scip) - SCIPgetLowerbound(scip));
      else
         searchubbound = SCIPinfinity(scip);
      if( heurdata->maxdiveavgquot > 0.0 )
         searchavgbound = SCIPgetLowerbound(scip)
            + heurdata->maxdiveavgquot * (SCIPgetAvgLowerbound(scip) - SCIPgetLowerbound(scip));
      else
         searchavgbound = SCIPinfinity(scip);
   }
   searchbound = MIN(searchubbound, searchavgbound);
   if( SCIPisObjIntegral(scip) )
      searchbound = SCIPceil(scip, searchbound);

   /* calculate the maximal diving depth: 10 * min{number of integer variables, max depth} */
   maxdivedepth = SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip);
   maxdivedepth = MIN(maxdivedepth, maxdepth);
   maxdivedepth *= 10;

   /* allocate memory */
   SCIP_CALL( SCIPallocBufferArray(scip, &discrepancies, heurdata->maxdiscdepth) );
   SCIP_CALL( SCIPallocBufferArray(scip, &tabulist, heurdata->maxdiscrepancy) );
   SCIP_CALL( SCIPallocBufferArray(scip, &selectedvars, heurdata->maxdiscdepth) );

   SCIPstatistic( SCIP_CALL( SCIPcreateClock(scip, &lptime) ) );

   /* initialize arrays */
   for( i = 0; i < heurdata->maxdiscdepth; ++i )
   {
      discrepancies[i] = 0;
      selectedvars[i] = NULL;
   }
   for( i = 0; i < heurdata->maxdiscrepancy; ++i )
      tabulist[i] = NULL;

   /* diving rule specific initialization */
   if( heurdata->divinginitexec != NULL )
   {
      SCIP_CALL( heurdata->divinginitexec(scip, heur) );
   }


   *result = SCIP_DIDNOTFIND;

#ifdef SCIP_STATISTIC
   /* notify the event handler of the diving heuristic that is now running */
   eventhdlrdata->runningheur = heur;
   ++heurdata->ncalls;
#endif

   /* start diving */
   SCIP_CALL( SCIPstartProbing(scip) );

   /* enables collection of variable statistics during probing */
   SCIPenableVarHistory(scip);

   /* get LP objective value*/
   lpsolstat = SCIP_LPSOLSTAT_OPTIMAL;
   objval = SCIPgetLPObjval(scip);

   SCIPdebugMessage("(node %"SCIP_LONGINT_FORMAT") executing %s heuristic: depth=%d, %d fractionals, dualbound=%g, searchbound=%g\n",
      SCIPgetNNodes(scip), SCIPheurGetName(heur), SCIPgetDepth(scip), nlpcands, SCIPgetDualbound(scip), SCIPretransformObj(scip, searchbound));

   /* dive as long we are in the given objective, depth and iteration limits and fractional variables exist, but
    * - if possible, we dive at least with the depth 10
    * - if the number of fractional variables decreased at least with 1 variable per 2 dive depths, we continue diving
    */
   lperror = FALSE;
   cutoff = FALSE;
   divedepth = 0;
   discrepancy = 0;
   totalpricerounds = 0;
   startnlpcands = nlpcands;

#ifdef SCIP_STATISTIC
   ndivenodes = 0;
   maxreacheddepth = 0;
   nfarkas = 0;
   nbacktracks = 0;
   ndiscsearches = 0;
   totallpiters = 0;
#endif

   while( !lperror && !cutoff && lpsolstat == SCIP_LPSOLSTAT_OPTIMAL && nlpcands > 0
      && (divedepth < 10
         || nlpcands <= startnlpcands - divedepth/2
         || (divedepth < maxdivedepth && heurdata->nlpiterations < maxnlpiterations && objval < searchbound))
      && !SCIPisStopped(scip) )
   {
      SCIP_VAR* bestcand;
      SCIP_Real bestcandsol;
      SCIP_Real bestfrac;
      SCIP_Bool bestcandmayround;

      SCIP_Bool backtracked;
      SCIP_Bool farkaspricing;

      SCIP_CALL( SCIPnewProbingNode(scip) );
      divedepth++;

#ifdef SCIP_STATISTIC
      maxreacheddepth = MAX(maxreacheddepth, divedepth);
      ++ndivenodes;
#endif

      /* get the current LP solution */
      SCIP_CALL( SCIPlinkLPSol(scip, heurdata->sol) );

      bestcand = NULL;
      bestcandmayround = TRUE;

      /* choose a variable to dive on */
      SCIP_CALL( heurdata->divingselectvar(scip, heur, tabulist, heurdata->maxdiscrepancy, &bestcand, &bestcandmayround) );

      /* if no variable could be chosen, abort diving */
      if( bestcand == NULL )
      {
         SCIPdebugMessage("No variable for diving could be selected, diving aborted\n");
         break;
      }
      assert(bestcand != NULL);

      bestcandsol = SCIPgetSolVal(scip, heurdata->sol, bestcand);
      bestfrac = SCIPfeasFrac(scip, bestcandsol);

      assert(SCIPisFeasGT(scip, bestcandsol, SCIPvarGetLbLocal(bestcand))
         && SCIPisFeasLT(scip, bestcandsol, SCIPvarGetUbLocal(bestcand)));

      /* memorize selected variables up to the maximal depth for discrepancy search */
      if( divedepth-1 < heurdata->maxdiscdepth )
         selectedvars[divedepth-1] = bestcand;

      /* if all candidates are roundable, try to round the solution */
      if( bestcandmayround )
      {
         SCIP_Bool success;

         /* try to round solution from diving LP */
         SCIP_CALL( SCIProundSol(scip, heurdata->sol, &success) );

         if( success )
         {
            SCIPdebugMessage("%s found roundable primal solution: obj=%g\n", SCIPheurGetName(heur), SCIPgetSolOrigObj(scip, heurdata->sol));

            /* try to add solution to SCIP */
            SCIP_CALL( SCIPtrySol(scip, heurdata->sol, FALSE, FALSE, FALSE, FALSE, FALSE, &success) );

            /* check, if solution was feasible and good enough */
            if( success )
            {
               SCIPdebugMessage(" -> solution was feasible and good enough\n");
               *result = SCIP_FOUNDSOL;
            }
         }
      }

      /* If the variable is already fixed, numerical troubles may have occurred => abort diving! */
      if( SCIPvarGetLbLocal(bestcand) >= SCIPvarGetUbLocal(bestcand) - 0.5 )
      {
         SCIPdebugMessage("Selected variable <%s> already fixed to [%g,%g] (solval: %.9f), diving aborted \n",
            SCIPvarGetName(bestcand), SCIPvarGetLbLocal(bestcand), SCIPvarGetUbLocal(bestcand), bestcandsol);
         cutoff = TRUE;
         break;
      }

      /* round variable up */
      SCIPdebugMessage("  dive %d/%d, LP iter %"SCIP_LONGINT_FORMAT"/%"SCIP_LONGINT_FORMAT", pricerounds %d/%d: var <%s>, round=%u, sol=%g, oldbounds=[%g,%g], newbounds=[%g,%g]\n",
         divedepth, maxdivedepth, heurdata->nlpiterations, maxnlpiterations, totalpricerounds, heurdata->maxpricerounds,
         SCIPvarGetName(bestcand), bestcandmayround,
         bestcandsol, SCIPvarGetLbLocal(bestcand), SCIPvarGetUbLocal(bestcand),
         SCIPfeasCeil(scip, bestcandsol), SCIPvarGetUbLocal(bestcand));
      SCIP_CALL( SCIPchgVarLbProbing(scip, bestcand, SCIPfeasCeil(scip, bestcandsol)) );

      backtracked = FALSE;
      farkaspricing = FALSE;
      do
      {
         /* apply domain propagation */
         SCIP_CALL( SCIPpropagateProbing(scip, 0, &cutoff, NULL) );

         if( !cutoff || backtracked || farkaspricing )
         {
            int npricerounds;                   /* pricing rounds performed in one single diving loop */

            /* resolve the diving LP */
            /* Errors in the LP solver should not kill the overall solving process, if the LP is just needed for a heuristic.
             * Hence in optimized mode, the return code is caught and a warning is printed, only in debug mode, SCIP will stop.
             */
            nlpiterations = SCIPgetNLPIterations(scip);
            npricerounds = SCIPgetNPriceRounds(scip);
            SCIPstatistic( SCIP_CALL( SCIPstartClock(scip, lptime) ) );
#ifdef NDEBUG
            if( (!heurdata->usefarkasonly || farkaspricing )
               && (heurdata->maxpricerounds == -1 || totalpricerounds < heurdata->maxpricerounds) )
               retstat = SCIPsolveProbingLPWithPricing(scip, FALSE, TRUE, heurdata->maxpricerounds == -1 ? -1 : heurdata->maxpricerounds - totalpricerounds, &lperror, &cutoff);

            else
               retstat = SCIPsolveProbingLP(scip, MAX((int)(maxnlpiterations - heurdata->nlpiterations), MINLPITER), &lperror, &cutoff);
            if( retstat != SCIP_OKAY )
            {
               SCIPwarningMessage(scip, "Error while solving LP in %s heuristic; LP solve terminated with code <%d>\n", SCIPheurGetName(heur), retstat);
            }
#else
            if( (!heurdata->usefarkasonly || farkaspricing )
               && (heurdata->maxpricerounds == -1 || totalpricerounds < heurdata->maxpricerounds) )
               SCIP_CALL( SCIPsolveProbingLPWithPricing(scip, FALSE, TRUE, heurdata->maxpricerounds == -1 ? -1 : heurdata->maxpricerounds - totalpricerounds, &lperror, &cutoff) );
            else
               SCIP_CALL( SCIPsolveProbingLP(scip, MAX((int)(maxnlpiterations - heurdata->nlpiterations), MINLPITER), &lperror, &cutoff) );
#endif
            SCIPstatistic( SCIP_CALL( SCIPstopClock(scip, lptime) ) );

            if( lperror )
               break;

            /* update iteration counts */
            heurdata->nlpiterations += SCIPgetNLPIterations(scip) - nlpiterations;
            heurdata->npricerounds += (SCIP_Longint) SCIPgetNPriceRounds(scip) - (SCIP_Longint) npricerounds;
            totalpricerounds += SCIPgetNPriceRounds(scip) - npricerounds;
            SCIPstatistic( totallpiters += SCIPgetNLPIterations(scip) - nlpiterations );

            /* get LP solution status */
            lpsolstat = SCIPgetLPSolstat(scip);
         }

         /* If infeasibility is encountered, perform Farkas pricing in order to reach feasibility again */
         if( lpsolstat == SCIP_LPSOLSTAT_INFEASIBLE && heurdata->usefarkasonly
            && !farkaspricing && (heurdata->maxpricerounds == -1 || totalpricerounds < heurdata->maxpricerounds)
            && !backtracked )
         {
            SCIPdebugMessage("  *** infeasibility detected at level %d - perform Farkas pricing\n", SCIPgetProbingDepth(scip));
#ifdef SCIP_STATISTIC
            ++nfarkas;
#endif
            farkaspricing = TRUE;
         }
         else
            farkaspricing = FALSE;

         /* perform backtracking if a cutoff or an infeasibility was detected and if Farkas pricing did not help */
         if( (lpsolstat == SCIP_LPSOLSTAT_INFEASIBLE || cutoff) && !backtracked && !farkaspricing )
         {
            /* Single backtracking (go back only one node) */
            if( heurdata->backtrack && divedepth > heurdata->maxdiscdepth && discrepancy < heurdata->maxdiscrepancy )
            {
               SCIPdebugMessage("  *** cutoff or infeasibility detected at level %d - backtracking one node\n", SCIPgetProbingDepth(scip));

               /* go back one depth in the search tree */
               SCIP_CALL( SCIPbacktrackProbing(scip, SCIPgetProbingDepth(scip)-1) );
               --divedepth;

               tabulist[discrepancy] = bestcand;
               ++discrepancy;

#ifdef SCIP_STATISTIC
                  ++nbacktracks;
#endif
               backtracked = TRUE;
            }
            /* Limited discrepancy search: If single backtracking unsuccessful, backtrack further */
            else if( heurdata->maxdiscdepth > 0 )
            {
               SCIPdebugMessage("  *** cutoff or infeasibility detected at level %d - performing discrepancy search\n", SCIPgetProbingDepth(scip));

               /* go back until the search can differ from the previous search tree */
               do
               {
                  SCIP_CALL( SCIPbacktrackProbing(scip, SCIPgetProbingDepth(scip)-1) );
                  --divedepth;
               }
               while( divedepth > 0 &&
                  (divedepth >= heurdata->maxdiscdepth || discrepancies[divedepth] >= heurdata->maxdiscrepancy) );

               assert(divedepth < heurdata->maxdiscdepth);

               if( discrepancies[divedepth] < heurdata->maxdiscrepancy )
               {
                  /* add variable selected previously at this depth to the tabu list */
                  tabulist[discrepancies[divedepth]] = selectedvars[divedepth];
                  ++discrepancies[divedepth];
                  discrepancy = discrepancies[divedepth];
                  for( i = discrepancy; i < heurdata->maxdiscrepancy; ++i )
                     tabulist[i] = NULL;
                  for( i = divedepth + 1; i < heurdata->maxdiscdepth; ++i )
                     discrepancies[i] = discrepancies[divedepth];

#ifdef SCIP_STATISTIC
                  ++ndiscsearches;
#endif
                  backtracked = TRUE;
               }
               else
               {
                  assert(divedepth == 0);
               }
            }
         }
         else
            backtracked = FALSE;
      }
      while( backtracked || farkaspricing );

      if( !lperror && !cutoff && lpsolstat == SCIP_LPSOLSTAT_OPTIMAL )
      {
         /* get new objective value */
         oldobjval = objval;
         objval = SCIPgetLPObjval(scip);

         /* update pseudo cost values */
         if( SCIPisGT(scip, objval, oldobjval) )
         {
            SCIP_CALL( SCIPupdateVarPseudocost(scip, bestcand, 0.0-bestfrac,
                  objval - oldobjval, 1.0) );
         }

         /* get new number of fractional variables */
         nlpcands = SCIPgetNLPBranchCands(scip);

         if( GCGrelaxIsOrigSolFeasible(origprob) )
         {
            SCIPdebugMessage("   -> LP solution is feasible in the original problem\n");
         }
      }
      SCIPdebugMessage("   -> lpsolstat=%d, objval=%g/%g, nfrac=%d\n", lpsolstat, objval, searchbound, nlpcands);
   }

   /* check if a solution has been found */
   if( nlpcands == 0 && !lperror && !cutoff && lpsolstat == SCIP_LPSOLSTAT_OPTIMAL )
   {
      SCIP_Bool success;

      /* create solution from diving LP */
      SCIP_CALL( SCIPlinkLPSol(scip, heurdata->sol) );
      SCIPdebugMessage("%s found primal solution: obj=%g\n", SCIPheurGetName(heur), SCIPgetSolOrigObj(scip, heurdata->sol));

      /* try to add solution to SCIP */
      SCIP_CALL( SCIPtrySol(scip, heurdata->sol, FALSE, FALSE, FALSE, FALSE, FALSE, &success) );

      /* check, if solution was feasible and good enough */
      if( success )
      {
         SCIPdebugMessage(" -> solution was feasible and good enough\n");
         *result = SCIP_FOUNDSOL;
      }
   }

   /* end diving */
   SCIP_CALL( SCIPendProbing(scip) );

   if( *result == SCIP_FOUNDSOL )
      heurdata->nsuccess++;

#ifdef SCIP_STATISTIC
   eventhdlrdata->runningheur = NULL;
   heurdata->ndivenodes += ndivenodes;
   heurdata->nfarkas += nfarkas;
   heurdata->nbacktracks += nbacktracks;
   heurdata->ndiscsearches += ndiscsearches;

   if( ndivenodes > 0 )
   {
      SCIPstatisticPrintf("Masterdiving statistic: %s at node %"SCIP_LONGINT_FORMAT" , %d dive nodes, max depth = %d, lptime = %6.1f sec, %"SCIP_LONGINT_FORMAT" lp iters, %d pricing rds, %d Farkas repairs, %d single backtracks, %d disc searches\n",
         SCIPheurGetName(heur), SCIPgetNNodes(scip), ndivenodes, maxreacheddepth, SCIPgetClockTime(scip, lptime), totallpiters, totalpricerounds, nfarkas, nbacktracks, ndiscsearches);
   }
#endif

   /* free memory */
   if( heurdata->divingexitexec != NULL )
   {
      SCIP_CALL( heurdata->divingexitexec(scip, heur) );
   }
   SCIPstatistic( SCIP_CALL( SCIPfreeClock(scip, &lptime) ) );
   SCIPfreeBufferArray(scip, &selectedvars);
   SCIPfreeBufferArray(scip, &tabulist);
   SCIPfreeBufferArray(scip, &discrepancies);

   SCIPdebugMessage("%s heuristic finished\n", SCIPheurGetName(heur));

   return SCIP_OKAY;
}


#ifdef SCIP_STATISTIC
/** destructor of event handler to free user data (called when SCIP is exiting) */
static
SCIP_DECL_EVENTFREE(eventFreeMasterdiving)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;

   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   assert((eventhdlrdata->heurs == NULL) == (eventhdlrdata->nheurs == 0));
   SCIPfreeMemoryArrayNull(scip, &eventhdlrdata->heurs);

   SCIPfreeMemory(scip, &eventhdlrdata);

   SCIPeventhdlrSetData(eventhdlr, NULL);

   return SCIP_OKAY;
}

/** initialization method of event handler (called after problem was transformed) */
static
SCIP_DECL_EVENTINIT(eventInitMasterdiving)
{  /*lint --e{715}*/
   assert(eventhdlr != NULL);

   /* notify GCG that this event should catch the SOLFOUND event */
   SCIP_CALL( SCIPcatchEvent(scip, SCIP_EVENTTYPE_SOLFOUND, eventhdlr, NULL, NULL) );

   return SCIP_OKAY;
}

/** deinitialization method of event handler (called before transformed problem is freed) */
static
SCIP_DECL_EVENTEXIT(eventExitMasterdiving)
{  /*lint --e{715}*/
   assert(eventhdlr != NULL);

   /* notify GCG that this event should drop the SOLFOUND event */
   SCIP_CALL( SCIPdropEvent(scip, SCIP_EVENTTYPE_SOLFOUND, eventhdlr, NULL, -1) );

   return SCIP_OKAY;
}

/** solving process deinitialization method of event handler (called before branch and bound process data is freed) */
static
SCIP_DECL_EVENTEXITSOL(eventExitsolMasterdiving)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;
   int i;

   assert(eventhdlr != NULL);

   /* get event handler data */
   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   /* print detailed statistics */
   SCIPstatisticPrintf("Master Diving Heuristics   :      Calls       Sols  Improving   DiveSols  Improving  RoundSols  Improving      Nodes   LP iters  Price rds        max    nFarkas  Single bt   Discsrch    BestPrimal Rounded?\n");
   for( i = 0; i < eventhdlrdata->nheurs; ++i )
   {
      SCIP_HEUR* heur;
      SCIP_HEURDATA* heurdata;

      heur = eventhdlrdata->heurs[i];

      /* get heuristic data */
      heurdata = SCIPheurGetData(heur);
      assert(heurdata != NULL);

      SCIPstatisticPrintf("%-17.17s          : %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10d %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT" %10"SCIP_LONGINT_FORMAT,
         SCIPheurGetName(heur), heurdata->ncalls, heurdata->nsols, heurdata->nimpsols, heurdata->ndivesols, heurdata->nimpdivesols, heurdata->nroundsols, heurdata->nimproundsols, heurdata->ndivenodes, heurdata->nlpiterations, heurdata->npricerounds, heurdata->maxpricerounds, heurdata->nfarkas, heurdata->nbacktracks, heurdata->ndiscsearches);
      if( SCIPisInfinity(scip, heurdata->bestprimalbd) )
         SCIPstatisticPrintf("      infinity");
      else
         SCIPstatisticPrintf(" %13.6e", heurdata->bestprimalbd);
      SCIPstatisticPrintf(heurdata->bestsolrounded ? "      yes\n" : "       no\n");
   }
   SCIPstatisticPrintf("END\n");
   SCIPstatisticPrintf("\n");

   return SCIP_OKAY;
}


/** execution method of event handler */
static
SCIP_DECL_EVENTEXEC(eventExecMasterdiving)
{  /*lint --e{715}*/
   SCIP_EVENTHDLRDATA* eventhdlrdata;
   SCIP_HEUR* heur;
   SCIP_HEURDATA* heurdata;
   SCIP_SOL* sol;
   SCIP_HEUR* solheur;
   SCIP_Bool rounded;
   SCIP_Bool improving;

   assert(eventhdlr != NULL);

   /* get event handler data */
   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);

   /* get the diving heuristic which is currently running;
    * if no diving heuristic is currently running, abort
    */
   heur = eventhdlrdata->runningheur;
   if( heur == NULL )
      return SCIP_OKAY;
   assert(heur != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* get new primal solution */
   sol = SCIPeventGetSol(event);
   assert(sol != NULL);

   /* get the heuristic that found the solution (might differ from the diving heuristic) */
   solheur = SCIPgetSolHeur(scip, sol);

   /* update solution statistics */
   ++heurdata->nsols;

   if( SCIPeventGetType(event) == SCIP_EVENTTYPE_BESTSOLFOUND )
   {
      ++heurdata->nimpsols;
      improving = TRUE;
   }
   else
      improving = FALSE;

   rounded = FALSE;
   if( solheur != NULL && strcmp(SCIPheurGetName(solheur), "simplerounding") == 0 )
   {
      rounded = TRUE;
      ++heurdata->nroundsols;
      if( improving )
         ++heurdata->nimproundsols;
   }
   else if( solheur == heur )
   {
      ++heurdata->ndivesols;
      if( improving )
         ++heurdata->nimpdivesols;
   }

   if( SCIPgetSolTransObj(scip, sol) < heurdata->bestprimalbd )
   {
      heurdata->bestprimalbd = SCIPgetSolTransObj(scip, sol);
      heurdata->bestsolrounded = rounded;
   }

   SCIPstatisticPrintf("Masterdiving statistic: %s found solution %13.6e , improving = %u , rounded = %u\n",
      SCIPheurGetName(heur), SCIPgetSolTransObj(scip, sol), improving, rounded);

   return SCIP_OKAY;
}
#endif


/*
 * heuristic specific interface methods
 */

/** gets diving rule specific data of a diving heuristic */
GCG_DIVINGDATA* GCGheurGetDivingDataMaster(
   SCIP_HEUR*               heur                    /**< primal heuristic */
   )
{
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   return heurdata->divingdata;
}

/** sets diving rule specific data of a diving heuristic */
void GCGheurSetDivingDataMaster(
   SCIP_HEUR*               heur,                   /**< primal heuristic */
   GCG_DIVINGDATA*          divingdata              /**< diving rule specific data */
   )
{
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   heurdata->divingdata = divingdata;
}

/** creates a master diving heuristic and includes it in GCG */
SCIP_RETCODE GCGincludeDivingHeurMaster(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_HEUR**           heur,               /**< pointer to diving heuristic */
   const char*           name,               /**< name of primal heuristic */
   const char*           desc,               /**< description of primal heuristic */
   char                  dispchar,           /**< display character of primal heuristic */
   int                   priority,           /**< priority of the primal heuristic */
   int                   freq,               /**< frequency for calling primal heuristic */
   int                   freqofs,            /**< frequency offset for calling primal heuristic */
   int                   maxdepth,           /**< maximal depth level to call heuristic at (-1: no limit) */
   GCG_DECL_DIVINGFREE   ((*divingfree)),    /**< destructor of diving heuristic */
   GCG_DECL_DIVINGINIT   ((*divinginit)),    /**< initialize diving heuristic */
   GCG_DECL_DIVINGEXIT   ((*divingexit)),    /**< deinitialize diving heuristic */
   GCG_DECL_DIVINGINITSOL ((*divinginitsol)), /**< solving process initialization method of diving heuristic */
   GCG_DECL_DIVINGEXITSOL ((*divingexitsol)), /**< solving process deinitialization method of diving heuristic */
   GCG_DECL_DIVINGINITEXEC ((*divinginitexec)), /**< execution initialization method of diving heuristic */
   GCG_DECL_DIVINGEXITEXEC ((*divingexitexec)), /**< execution deinitialization method of diving heuristic */
   GCG_DECL_DIVINGSELECTVAR ((*divingselectvar)), /**< variable selection method of diving heuristic */
   GCG_DIVINGDATA*       divingdata          /**< diving rule specific data (or NULL) */
   )
{
#ifdef SCIP_STATISTIC
   SCIP_EVENTHDLRDATA* eventhdlrdata;
   SCIP_EVENTHDLR* eventhdlr;
#endif
   SCIP_HEURDATA* heurdata;
   char paramname[SCIP_MAXSTRLEN];

#ifdef SCIP_STATISTIC
   /* get masterdiving event handler and its data */
   eventhdlr = SCIPfindEventhdlr(scip, EVENTHDLR_NAME);
   assert(eventhdlr != NULL);
   eventhdlrdata = SCIPeventhdlrGetData(eventhdlr);
   assert(eventhdlrdata != NULL);
#endif

   /* create Masterdiving primal heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &heurdata) );

   /* set diving rule callbacks and data */
   heurdata->divingfree = divingfree;
   heurdata->divinginit = divinginit;
   heurdata->divingexit = divingexit;
   heurdata->divinginitsol = divinginitsol;
   heurdata->divingexitsol = divingexitsol;
   heurdata->divinginitexec = divinginitexec;
   heurdata->divingexitexec = divingexitexec;
   heurdata->divingselectvar = divingselectvar;
   heurdata->divingdata = divingdata;

   /* include primal heuristic */
   SCIP_CALL( SCIPincludeHeurBasic(scip, heur,
         name, desc, dispchar, priority, freq, freqofs,
         maxdepth, HEUR_TIMING, HEUR_USESSUBSCIP, heurExecMasterdiving, heurdata) );

   assert(*heur != NULL);

   /* set non-NULL pointers to callback methods */
   SCIP_CALL( SCIPsetHeurFree(scip, *heur, heurFreeMasterdiving) );
   SCIP_CALL( SCIPsetHeurInit(scip, *heur, heurInitMasterdiving) );
   SCIP_CALL( SCIPsetHeurExit(scip, *heur, heurExitMasterdiving) );
   SCIP_CALL( SCIPsetHeurInitsol(scip, *heur, heurInitsolMasterdiving) );
   SCIP_CALL( SCIPsetHeurExitsol(scip, *heur, heurExitsolMasterdiving) );

   /* masterdiving heuristic parameters */
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/minreldepth", name);
   SCIP_CALL( SCIPaddRealParam(scip,
        paramname,
        "minimal relative depth to start diving",
        &heurdata->minreldepth, TRUE, DEFAULT_MINRELDEPTH, 0.0, 1.0, NULL, NULL) );
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/maxreldepth", name);
   SCIP_CALL( SCIPaddRealParam(scip,
        paramname,
        "maximal relative depth to start diving",
        &heurdata->maxreldepth, TRUE, DEFAULT_MAXRELDEPTH, 0.0, 1.0, NULL, NULL) );
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/maxlpiterquot", name);
   SCIP_CALL( SCIPaddRealParam(scip,
        paramname,
        "maximal fraction of diving LP iterations compared to node LP iterations",
        &heurdata->maxlpiterquot, FALSE, DEFAULT_MAXLPITERQUOT, 0.0, SCIP_REAL_MAX, NULL, NULL) );
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/maxlpiterofs", name);
   SCIP_CALL( SCIPaddIntParam(scip,
        paramname,
        "additional number of allowed LP iterations",
        &heurdata->maxlpiterofs, FALSE, DEFAULT_MAXLPITEROFS, 0, INT_MAX, NULL, NULL) );
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/maxpricerounds", name);
   SCIP_CALL( SCIPaddIntParam(scip,
        paramname,
        "maximal number of allowed pricing rounds (-1: no limit)",
        &heurdata->maxpricerounds, FALSE, DEFAULT_MAXPRICEROUNDS, -1, INT_MAX, NULL, NULL) );
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/usefarkasonly", name);
   SCIP_CALL( SCIPaddBoolParam(scip,
        paramname,
        "perform pricing only if infeasibility is encountered",
        &heurdata->usefarkasonly, TRUE, DEFAULT_USEFARKASONLY, NULL, NULL) );
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/maxdiveubquot", name);
   SCIP_CALL( SCIPaddRealParam(scip,
        paramname,
        "maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)",
        &heurdata->maxdiveubquot, TRUE, DEFAULT_MAXDIVEUBQUOT, 0.0, 1.0, NULL, NULL) );
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/maxdiveavgquot", name);
   SCIP_CALL( SCIPaddRealParam(scip,
        paramname,
        "maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)",
        &heurdata->maxdiveavgquot, TRUE, DEFAULT_MAXDIVEAVGQUOT, 0.0, SCIP_REAL_MAX, NULL, NULL) );
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/maxdiveubquotnosol", name);
   SCIP_CALL( SCIPaddRealParam(scip,
        paramname,
        "maximal UBQUOT when no solution was found yet (0.0: no limit)",
        &heurdata->maxdiveubquotnosol, TRUE, DEFAULT_MAXDIVEUBQUOTNOSOL, 0.0, 1.0, NULL, NULL) );
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/maxdiveavgquotnosol", name);
   SCIP_CALL( SCIPaddRealParam(scip,
        paramname,
        "maximal AVGQUOT when no solution was found yet (0.0: no limit)",
        &heurdata->maxdiveavgquotnosol, TRUE, DEFAULT_MAXDIVEAVGQUOTNOSOL, 0.0, SCIP_REAL_MAX, NULL, NULL) );
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/backtrack", name);
   SCIP_CALL( SCIPaddBoolParam(scip,
        paramname,
        "single backtracking by choosing another variable in case of infeasibility",
        &heurdata->backtrack, TRUE, DEFAULT_BACKTRACK, NULL, NULL) );
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/maxdiscrepancy", name);
   SCIP_CALL( SCIPaddIntParam(scip,
        paramname,
        "maximal discrepancy allowed in backtracking and limited discrepancy search",
        &heurdata->maxdiscrepancy, TRUE, DEFAULT_MAXDISCREPANCY, 0, INT_MAX, NULL, NULL) );
   (void) SCIPsnprintf(paramname, SCIP_MAXSTRLEN, "heuristics/%s/maxdiscdepth", name);
   SCIP_CALL( SCIPaddIntParam(scip,
        paramname,
        "maximal depth until which a limited discrepancy search is performed",
        &heurdata->maxdiscdepth, TRUE, DEFAULT_MAXDISCDEPTH, 0, INT_MAX, NULL, NULL) );

#ifdef SCIP_STATISTIC
   /* register the diving heuristic to the masterdiving event handler */
   assert((eventhdlrdata->heurs == NULL) == (eventhdlrdata->nheurs == 0));
   if( eventhdlrdata->nheurs == 0 )
   {
      SCIP_CALL( SCIPallocMemoryArray(scip, &eventhdlrdata->heurs, 1) ); /*lint !e506*/
   }
   else
   {
      SCIP_CALL( SCIPreallocMemoryArray(scip, &eventhdlrdata->heurs, eventhdlrdata->nheurs+1) );
   }
   eventhdlrdata->heurs[eventhdlrdata->nheurs] = *heur;
   ++eventhdlrdata->nheurs;
#endif

   return SCIP_OKAY;
}

/** creates event handler for masterdiving event */
SCIP_RETCODE SCIPincludeEventHdlrMasterdiving(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
#ifdef SCIP_STATISTIC
   SCIP_EVENTHDLRDATA* eventhdlrdata;
   SCIP_EVENTHDLR* eventhdlr;

   /* create master event handler data */
   SCIP_CALL( SCIPallocMemory(scip, &eventhdlrdata) );
   assert(eventhdlrdata != NULL);

   eventhdlr = NULL;

   /* include event handler into GCG */
   SCIP_CALL( SCIPincludeEventhdlrBasic(scip, &eventhdlr, EVENTHDLR_NAME, EVENTHDLR_DESC,
         eventExecMasterdiving, eventhdlrdata) );
   assert(eventhdlr != NULL);

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetEventhdlrFree(scip, eventhdlr, eventFreeMasterdiving) );
   SCIP_CALL( SCIPsetEventhdlrInit(scip, eventhdlr, eventInitMasterdiving) );
   SCIP_CALL( SCIPsetEventhdlrExit(scip, eventhdlr, eventExitMasterdiving) );
   SCIP_CALL( SCIPsetEventhdlrExitsol(scip, eventhdlr, eventExitsolMasterdiving) );

   /* initialize masterdiving event handler data */
   eventhdlrdata->heurs = NULL;
   eventhdlrdata->nheurs = 0;
   eventhdlrdata->runningheur = NULL;
#endif

   return SCIP_OKAY;
}
