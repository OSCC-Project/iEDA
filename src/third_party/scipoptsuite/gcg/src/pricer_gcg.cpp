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

/**@file   pricer_gcg.cpp
 * @brief  pricer for generic column generation
 * @author Gerald Gamrath
 * @author Martin Bergner
 * @author Alexander Gross
 * @author Christian Puchert
 * @author Michael Bastubbe
 * @author Jonas Witt
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/* #define SCIP_DEBUG */
/* #define PRINTDUALSOLS */

#include <cassert>
#include <cstring>

/*lint -e64 disable useless and wrong lint warning */

#ifdef __INTEL_COMPILER
#ifndef _OPENMP
#pragma warning disable 3180  /* disable wrong and useless omp warnings */
#endif
#endif
#include "scip/scip.h"
#include "gcg.h"

#include "scip/cons_linear.h"
#include "scip/cons_knapsack.h"

#include "pricer_gcg.h"
#include "objpricer_gcg.h"
#include "sepa_master.h"
#include "sepa_basis.h"

#include "relax_gcg.h"
#include "scip_misc.h"
#include "pub_gcgvar.h"
#include "pub_gcgcol.h"
#include "pub_pricingjob.h"
#include "pub_pricingprob.h"
#include "pub_solver.h"
#include "solver.h"
#include "cons_masterbranch.h"
#include "objscip/objscip.h"
#include "objpricer_gcg.h"
#include "class_pricingtype.h"
#include "class_pricingcontroller.h"
#include "class_stabilization.h"
#include "branch_generic.h"
#include "event_display.h"
#include "pub_colpool.h"

#ifdef SCIP_STATISTIC
#include "scip/struct_scip.h"
#include "scip/struct_stat.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace scip;

#define PRICER_NAME            "gcg"
#define PRICER_DESC            "pricer for gcg"
#define PRICER_PRIORITY        5000000
#define PRICER_DELAY           TRUE     /* only call pricer if all problem variables have non-negative reduced costs */

#define DEFAULT_ABORTPRICINGINT          TRUE       /**< should the pricing be aborted when integral */
#define DEFAULT_ABORTPRICINGGAP          0.00       /**< gap between dual bound and RMP objective at which pricing is aborted */
#define DEFAULT_DISPINFOS                FALSE      /**< should additional information be displayed */
#define DEFAULT_DISABLECUTOFF            2          /**< should the cutoffbound be applied in master LP solving? (0: on, 1:off, 2:auto) */
#define DEFAULT_THREADS                  0          /**< number of threads (0 is OpenMP default) */
#define DEFAULT_STABILIZATION            TRUE       /**< should stabilization be used */
#define DEFAULT_STABILIZATIONTREE        FALSE      /**< should stabilization be used in nodes other than the root node*/
#define DEFAULT_HYBRIDASCENT             FALSE      /**< should hybridization of smoothing with an ascent method be enabled */
#define DEFAULT_HYBRIDASCENT_NOAGG       FALSE      /**< should hybridization of smoothing with an ascent method be enabled
                                                     *   if pricing problems cannot be aggregation */

#define DEFAULT_USECOLPOOL               TRUE       /**< should the colpool be checked for negative redcost cols before solving the pricing problems? */
#define DEFAULT_COLPOOL_AGELIMIT         100        /**< default age limit for columns in column pool */

#define DEFAULT_PRICE_ORTHOFAC 0.0
#define DEFAULT_PRICE_OBJPARALFAC 0.0
#define DEFAULT_PRICE_REDCOSTFAC 1.0
#define DEFAULT_PRICE_MINCOLORTH 0.0
#define DEFAULT_PRICE_EFFICIACYCHOICE 0

#define DEFAULT_USEARTIFICIALVARS        FALSE      /**< add artificial vars to master (instead of using Farkas pricing) */
#define DEFAULT_USEMAXOBJ                TRUE       /**< default value for using maxobj for big M objective of artificial variables */
#define DEFAULT_ONLYRELIABLEBIGM         TRUE       /**< default value for only using maxobj for big M objective of artificial variables if it is reliable */
#define DEFAULT_FACTORUNRELIABLE         1000       /**< default factor to use for objective of unbounded variables */
#define DEFAULT_BIGMARTIFICIAL           1000       /**< default value for big M objective of artificial variables (if maxobj is not used)*/

#define EVENTHDLR_NAME         "probdatavardeleted"
#define EVENTHDLR_DESC         "event handler for variable deleted event"

/** small macro to simplify printing pricer information */
#define GCGpricerPrintInfo(scip, pricerdata, ...) do { \
   if( pricerdata->dispinfos ) { \
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, __VA_ARGS__);\
   } else {\
      SCIPdebugMessage(__VA_ARGS__); \
   }\
   }while( FALSE )

#define PRICER_STAT_ARRAYLEN_TIME 1024                /**< length of the array for Time histogram representation */
#define PRICER_STAT_BUCKETSIZE_TIME 10                /**< size of the buckets for Time histogram representation */
#define PRICER_STAT_ARRAYLEN_VARS 1024                /**< length of the array for foundVars histogram representation */
#define PRICER_STAT_BUCKETSIZE_VARS 1                 /**< size of the buckets for foundVars histogram representation */

/*
 * Data structures
 */

/** variable pricer data */
struct SCIP_PricerData
{
   int                   npricingprobs;      /**< number of pricing problems */
   SCIP**                pricingprobs;       /**< pointers to the pricing problems */
   SCIP_Real*            dualsolconv;        /**< array of dual solutions for the convexity constraints */
   SCIP_Real*            solvals;            /**< solution values of variables in the pricing problems */
   int                   maxsolvals;         /**< capacity of solvals */
   int*                  npointsprob;        /**< number of variables representing points created by the pricing probs */
   int*                  nraysprob;          /**< number of variables representing rays created by the pricing probs */
   SCIP_Longint          currnodenr;         /**< current node number in the masterproblem */
   SCIP_Bool             newnode;            /**< indicate whether we are at a new branch-and-bound node */
   SCIP_HASHMAP*         mapcons2idx;        /**< hashmap mapping constraints to their index in the conss array */
   int                   npricingprobsnotnull; /**< number of non-Null pricing problems*/

   SCIP_VAR**            pricedvars;         /**< array of all priced variables */
   int                   npricedvars;        /**< number of priced variables */
   int                   maxpricedvars;      /**< maximal number of priced variables */

   SCIP_VAR**            artificialvars;     /**< array of artificial variables */
   int                   nartificialvars;    /**< number of artificial variables */
   int                   maxartificialvars;  /**< capacity of artificialvars */
   SCIP_Bool             artificialused;     /**< returns if artificial variables are used in current node's LP solution */

   SCIP_Real**           realdualvalues;     /**< real dual values for pricing variables */
   int*                  maxrealdualvalues;  /**< capacities of realdualvalues */
   int                   maxrealdualvaluescapacity;  /**< capacity of maxrealdualvalues */

   /* variables used for statistics */
   SCIP_CLOCK*           freeclock;          /**< time for freeing pricing problems */
   SCIP_CLOCK*           transformclock;     /**< time for transforming pricing problems */
   int                   solvedsubmipsoptimal; /**< number of optimal pricing runs */
   int                   solvedsubmipsheur;  /**< number of heuristical pricing runs*/
   int                   calls;              /**< number of total pricing calls */
   SCIP_Longint          pricingiters;       /**< sum of all pricing simplex iterations */

   /* solver data */
   GCG_SOLVER**          solvers;            /**< pricing solvers array */
   int                   nsolvers;           /**< number of pricing solvers */

   /* event handler */
   SCIP_EVENTHDLR*       eventhdlr;          /**< event handler */

   /* parameter values */
   SCIP_VARTYPE          vartype;            /**< vartype of created master variables */
   int                   nroundsredcost;     /**< number of reduced cost rounds */
   SCIP_Bool             abortpricingint;    /**< should the pricing be aborted on integral solutions? */
   SCIP_Bool             dispinfos;          /**< should pricing information be displayed? */
   int                   disablecutoff;      /**< should the cutoffbound be applied in master LP solving (0: on, 1:off, 2:auto)? */
   SCIP_Real             abortpricinggap;    /**< gap between dual bound and RMP objective at which pricing is aborted */
   SCIP_Bool             stabilization;      /**< should stabilization be used */
   SCIP_Bool             stabilizationtree;  /**< should stabilization be used in nodes other than the root node */
   SCIP_Bool             usecolpool;         /**< should the colpool be checked for negative redcost cols before solving the pricing problems? */
   SCIP_Bool             useartificialvars;  /**< use artificial variables to make RMP feasible (instead of applying Farkas pricing) */
   SCIP_Real             maxobj;             /**< maxobj bound that can be used for big M objective of artificial variables */
   SCIP_Bool             usemaxobj;          /**< use maxobj for big M objective of artificial variables */
   SCIP_Bool             onlyreliablebigm;   /**< only use maxobj for big M objective of artificial variables if it is reliable */
   SCIP_Real             factorunreliable;   /**< factor to use for objective of unbounded variables */
   SCIP_Real             bigmartificial;     /**< value for for big M objective of artificial variables (if maxobj is not used) */
   SCIP_Bool             hybridascent;       /**< should hybridization of smoothing with an ascent method be enabled */
   SCIP_Bool             hybridascentnoagg;  /**< should hybridization of smoothing with an ascent method be enabled
                                              *   if pricing problems cannot be aggregation */
   int                   colpoolagelimit;    /**< agelimit of columns in colpool */

   /* price storage */
   SCIP_Real             redcostfac;         /**< factor of -redcost/norm in score function */
   SCIP_Real             objparalfac;        /**< factor of objective parallelism in score function */
   SCIP_Real             orthofac;           /**< factor of orthogonalities in score function */
   SCIP_Real             mincolorth;         /**< minimal orthogonality of columns to add
                                                  (with respect to columns added in the current round) */
   SCIP_Real             maxpricecols;       /**< maximum number of columns per round */
   SCIP_Real             maxpricecolsfarkas; /**< maximum number of columns per Farkas round */
   GCG_EFFICIACYCHOICE   efficiacychoice;    /**< choice to base efficiacy on */

   /* statistics */
   int                   oldvars;            /**< Vars of last pricing iteration */
   int*                  farkascallsdist;    /**< Calls of each farkas pricing problem */
   int*                  farkasfoundvars;    /**< Found vars of each farkas pricing problem */
   double*               farkasnodetimedist; /**< Time spend in each farkas pricing problem */

   int*                  redcostcallsdist;   /**< Calls of each redcost pricing problem */
   int*                  redcostfoundvars;   /**< Found vars of each redcost pricing problem */
   double*               redcostnodetimedist; /**< Time spend in each redcost pricing problem */
   double                rootnodedegeneracy; /**< degeneracy of the root node */
   double                avgrootnodedegeneracy; /**< average degeneray of all nodes */
   int                   ndegeneracycalcs;   /**< number of observations */

#ifdef SCIP_STATISTIC
   int                   nrootbounds;        /**< number of stored bounds */
   SCIP_Real*            rootpbs;            /**< array of primal bounds for the root LP, one bound for each pricing call */
   SCIP_Real*            rootdbs;            /**< array of dual bounds for the root LP, one bound for each pricing call */
   SCIP_Real*            roottimes;          /**< array of times spent for root LP */
   SCIP_Real*            rootdualdiffs;      /**< array of differences to last dual solution */
   int                   maxrootbounds;      /**< maximal number of bounds */
   SCIP_Real             rootfarkastime;     /**< time of last Farkas call */
   SCIP_Real             dualdiff;           /**< difference to last dual solution */
   int                   dualdiffround;      /**< value of nrootbounds when difference to last dual solution was computed */
   SCIP_SOL*             rootlpsol;          /**< optimal root LP solution */
   SCIP_Real***          dualvalues;         /**< array of dual values for pricing variables for each root redcost call*/
   SCIP_Real**           dualsolconvs;       /**< array of dual solutions for the convexity constraints for each root redcost call*/
   int*                  nodetimehist;       /**< Histogram of nodetime distribution */
   int*                  foundvarshist;      /**< Histogram of foundvars distribution */
#endif
};


int ObjPricerGcg::threads;

/** information method for a parameter change of disablecutoff */
static
SCIP_DECL_PARAMCHGD(paramChgdDisablecutoff)
{  /*lint --e{715}*/
   SCIP* masterprob;
   int newval;

   masterprob = GCGgetMasterprob(scip);
   newval = SCIPparamGetInt(param);

   SCIP_CALL( SCIPsetIntParam(masterprob, "lp/disablecutoff", newval) );

   return SCIP_OKAY;
}


/*
 * Callback methods of event handler
 */

/** destructor of event handler to free user data (called when SCIP is exiting) */
#define eventFreeVardeleted NULL

/** initialization method of event handler (called after problem was transformed) */
#define eventInitVardeleted NULL

/** deinitialization method of event handler (called before transformed problem is freed) */
#define eventExitVardeleted NULL

/** solving process initialization method of event handler (called when branch and bound process is about to begin) */
#define eventInitsolVardeleted NULL

/** solving process deinitialization method of event handler (called before branch and bound process data is freed) */
#define eventExitsolVardeleted NULL

/** frees specific event data */
#define eventDeleteVardeleted NULL

/** execution method of event handler */
static
SCIP_DECL_EVENTEXEC(eventExecVardeleted)
{  /*lint --e{715}*/
   SCIP_VAR* var;
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;
   SCIP_VAR** origvars;
   int i;

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   assert(SCIPeventGetType(event) == SCIP_EVENTTYPE_VARDELETED);
   var = SCIPeventGetVar(event);
   assert(var != NULL);

   SCIPdebugMessage("remove master variable %s from pricerdata and corresponding original variables\n", SCIPvarGetName(var));

   assert(GCGvarIsMaster(var));
   origvars = GCGmasterVarGetOrigvars(var);
   assert(origvars != NULL);

   /* remove master variable from corresponding pricing original variables */
   for( i = 0; i < GCGmasterVarGetNOrigvars(var); ++i )
   {
      SCIP_CALL( GCGoriginalVarRemoveMasterVar(scip, origvars[i], var) );
   }

   /* remove variable from array of stored priced variables */
   i = GCGmasterVarGetIndex(var);
   assert(i < pricerdata->npricedvars);

   if( i >= 0 )
   {
      assert(pricerdata->pricedvars[i] == var);

      /* drop vardeleted event on variable */
      SCIP_CALL(
         SCIPdropVarEvent(scip, pricerdata->pricedvars[i], SCIP_EVENTTYPE_VARDELETED, pricerdata->eventhdlr, NULL, -1));

      SCIP_CALL(SCIPreleaseVar(scip, &(pricerdata->pricedvars[i])));
      (pricerdata->npricedvars)--;
      assert(pricerdata->pricedvars[pricerdata->npricedvars] != NULL || i == pricerdata->npricedvars);
      pricerdata->pricedvars[i] = pricerdata->pricedvars[pricerdata->npricedvars];
      if( i < pricerdata->npricedvars )
         GCGmasterVarSetIndex(pricerdata->pricedvars[i], i);
      (pricerdata->oldvars)--;

#ifndef NDEBUG
      for ( ; i < pricerdata->npricedvars; ++i )
      {
         assert(pricerdata->pricedvars[i] != var);
      }
#endif
   }

   return SCIP_OKAY;
}


/*
 * Local methods
 */

/** return TRUE or FALSE whether the master LP is solved to optimality */
SCIP_Bool ObjPricerGcg::isMasterLPOptimal() const
{
   assert(GCGisMaster(scip_));

   return SCIPgetLPSolstat(scip_) == SCIP_LPSOLSTAT_OPTIMAL;
}

/** get the number of columns to be added to the master LP in the current pricing round */
int ObjPricerGcg::getMaxColsRound() const
{
   assert(pricingtype != NULL);

   return pricingtype->getMaxcolsround();
}

/** get the number of columns per pricing problem to be added to the master LP in the current pricing round */
int ObjPricerGcg::getMaxColsProb() const
{
   assert(pricingtype != NULL);

   return pricingtype->getMaxcolsprob();
}

/** ensures size of pricedvars array */
SCIP_RETCODE ObjPricerGcg::ensureSizePricedvars(
   int                   size                /**< needed size */
   )
{
   assert(pricerdata != NULL);
   assert(pricerdata->pricedvars != NULL);

   if( pricerdata->maxpricedvars < size )
   {
      int oldsize = pricerdata->maxpricedvars;
      pricerdata->maxpricedvars = SCIPcalcMemGrowSize(scip_, size);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &(pricerdata->pricedvars), oldsize, pricerdata->maxpricedvars) );
   }
   assert(pricerdata->maxpricedvars >= size);

   return SCIP_OKAY;
}


#ifdef SCIP_STATISTIC
/** ensures size of root bounds arrays */
SCIP_RETCODE ObjPricerGcg::ensureSizeRootBounds(
   int                   size                /**< needed size */
   )
{
   assert(pricerdata != NULL);
   assert(pricerdata->rootdbs != NULL);
   assert(pricerdata->rootpbs != NULL);
   assert(pricerdata->roottimes != NULL);
   assert(pricerdata->rootdualdiffs != NULL);
   assert(pricerdata->dualvalues != NULL);
   assert(pricerdata->dualsolconvs != NULL);

   if( pricerdata->maxrootbounds < size )
   {
      int oldsize = pricerdata->maxrootbounds;
      pricerdata->maxrootbounds = SCIPcalcMemGrowSize(scip_, size);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &(pricerdata->rootpbs), oldsize, pricerdata->maxrootbounds) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &(pricerdata->rootdbs), oldsize, pricerdata->maxrootbounds) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &(pricerdata->roottimes), oldsize, pricerdata->maxrootbounds) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &(pricerdata->rootdualdiffs), oldsize, pricerdata->maxrootbounds) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &(pricerdata->dualvalues), oldsize, pricerdata->maxrootbounds) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &(pricerdata->dualsolconvs), oldsize, pricerdata->maxrootbounds) );
   }
   assert(pricerdata->maxrootbounds >= size);

   return SCIP_OKAY;
}
#endif

#ifdef SCIP_STATISTIC
/** gets the NodeTimeDistribution in the form of a histogram */
static
void GCGpricerGetNodeTimeHistogram(
   SCIP_PRICERDATA*      pricerdata,         /**< pricerdata data structure */
   SCIP_Real             time                /**< time the pricingproblem needed */
   )
{
   int i;
   assert(pricerdata != NULL);
   /* 1000* because mapping milliseconds on the index i */
   i = 1000*time/PRICER_STAT_BUCKETSIZE_TIME; /*lint !e524 */

   if( i >= PRICER_STAT_ARRAYLEN_TIME )
   {
      i = PRICER_STAT_ARRAYLEN_TIME-1;
   }

   assert(i < PRICER_STAT_ARRAYLEN_TIME);
   assert(i >= 0);
   pricerdata->nodetimehist[i]++;

}


/** gets the FoundVarsDistribution in form of a histogram */
static
void GCGpricerGetFoundVarsHistogram(
   SCIP_PRICERDATA*      pricerdata,         /**< pricerdata data structure */
   int                   foundvars           /**< foundVars in pricingproblem */
   )
{
   int i;
   assert(pricerdata != NULL);
   i = foundvars/PRICER_STAT_BUCKETSIZE_VARS;
   if( i >= PRICER_STAT_ARRAYLEN_VARS )
   {
      i = PRICER_STAT_ARRAYLEN_VARS-1;
   }

   assert(i < PRICER_STAT_ARRAYLEN_VARS);
   assert(i >= 0);
   pricerdata->foundvarshist[i]++;

}


/** gets the statistics of the pricingprobs like calls, foundvars and time */
static
void GCGpricerCollectStatistic(
   SCIP_PRICERDATA*      pricerdata,         /**< pricerdata data structure */
   GCG_PRICETYPE         type,               /**< type of pricing: optimal or heuristic */
   int                   probindex,          /**< index of the pricingproblem */
   SCIP_Real             time                /**< time the pricingproblem needed */
   )
{
   int foundvars;
   assert(pricerdata != NULL);
   foundvars = pricerdata->npricedvars - pricerdata->oldvars;

   if( type == GCG_PRICETYPE_FARKAS )
   {

      pricerdata->farkascallsdist[probindex]++; /*Calls*/
      pricerdata->farkasfoundvars[probindex] += foundvars;
      pricerdata->farkasnodetimedist[probindex] += time;   /*Time*/

   }
   else if( type == GCG_PRICETYPE_REDCOST )
   {

      pricerdata->redcostcallsdist[probindex]++;
      pricerdata->redcostfoundvars[probindex] += foundvars;
      pricerdata->redcostnodetimedist[probindex] += time;

   }

   GCGpricerGetNodeTimeHistogram(pricerdata, time);
   GCGpricerGetFoundVarsHistogram(pricerdata, foundvars);

   pricerdata->oldvars = pricerdata->npricedvars;
}
#endif

/** frees all solvers */
SCIP_RETCODE ObjPricerGcg::solversFree()
{
   int i;
   assert(pricerdata != NULL);
   assert((pricerdata->solvers == NULL) == (pricerdata->nsolvers == 0));
   assert(pricerdata->nsolvers > 0);

   for( i = 0; i < pricerdata->nsolvers; i++ )
   {
      SCIP_CALL( GCGsolverFree(scip_, &pricerdata->solvers[i]) );
   }

   return SCIP_OKAY;
}

/** calls the init method on all solvers */
SCIP_RETCODE ObjPricerGcg::solversInit()
{
   int i;
   assert(pricerdata != NULL);
   assert((pricerdata->solvers == NULL) == (pricerdata->nsolvers == 0));
   assert(pricerdata->nsolvers > 0);

   for( i = 0; i < pricerdata->nsolvers; i++ )
   {
      SCIP_CALL( GCGsolverInit(scip_, pricerdata->solvers[i]) );
   }

   return SCIP_OKAY;
}

/** calls the exit method on all solvers */
SCIP_RETCODE ObjPricerGcg::solversExit()
{
   int i;
   assert(pricerdata != NULL);
   assert((pricerdata->solvers == NULL) == (pricerdata->nsolvers == 0));
   assert(pricerdata->nsolvers > 0);

   for( i = 0; i < pricerdata->nsolvers; i++ )
   {
      SCIP_CALL( GCGsolverExit(scip_, pricerdata->solvers[i]) );
   }

   return SCIP_OKAY;
}

/** calls the initsol method on all solvers */
SCIP_RETCODE ObjPricerGcg::solversInitsol()
{
   int i;
   assert(pricerdata != NULL);

   if( pricerdata->npricingprobs == 0 )
      return SCIP_OKAY;

   assert((pricerdata->solvers == NULL) == (pricerdata->nsolvers == 0));
   assert(pricerdata->nsolvers > 0);

   for( i = 0; i < pricerdata->nsolvers; i++ )
   {
      SCIP_CALL( GCGsolverInitsol(scip_, pricerdata->solvers[i]) );
   }

   return SCIP_OKAY;
}

/** calls the exitsol method of all solvers */
SCIP_RETCODE ObjPricerGcg::solversExitsol()
{
   int i;
   assert(pricerdata != NULL);
   assert((pricerdata->solvers == NULL) == (pricerdata->nsolvers == 0));
   assert(pricerdata->nsolvers > 0);

   if( pricerdata->npricingprobs == 0 )
      return SCIP_OKAY;

   for( i = 0; i < pricerdata->nsolvers; i++ )
   {
      SCIP_CALL( GCGsolverExitsol(scip_, pricerdata->solvers[i]) );
   }

   return SCIP_OKAY;
}

/** returns the gegeneracy of the masterproblem */
SCIP_RETCODE ObjPricerGcg::computeCurrentDegeneracy(
   double*               degeneracy          /**< pointer to store degeneracy */
   )
{
   int nrows;
   SCIP_COL** cols;
   SCIP_ROW** rows;
   int i;
   int countz = 0;
   int* indizes = NULL;

   assert(degeneracy != NULL);

   nrows = SCIPgetNLPRows(scip_);

   if( nrows > 0 ) {
      cols = SCIPgetLPCols(scip_);
      rows = SCIPgetLPRows(scip_);

      SCIP_CALL(SCIPallocBufferArray(scip_, &indizes, (size_t) nrows));

      /* gives indices of Columns in Basis and indices of vars in Basis */
      SCIP_CALL(SCIPgetLPBasisInd(scip_, indizes));

      for (i = 0; i < nrows; i++) {
         SCIP_Real currentVal;
         int colindex = indizes[i];

         /* is column if >= 0 it is column in basis, < 0 is for row */
         if (colindex >= 0) {
            SCIP_COL *col;
            col = cols[colindex];
            currentVal = SCIPcolGetPrimsol(col);

            if (SCIPisEQ(scip_, currentVal, SCIPcolGetLb(col)) || SCIPisEQ(scip_, currentVal, SCIPcolGetUb(col)))
               countz++;
         } else {
            SCIP_ROW *row = rows[-indizes[i] - 1];

            currentVal = SCIPgetRowActivity(scip_, row);

            if (SCIPisEQ(scip_, currentVal, SCIProwGetRhs(row)) || SCIPisEQ(scip_, currentVal, SCIProwGetLhs(row)))
               countz++;
         }
      }
      SCIPfreeBufferArray(scip_, &indizes);

      /* Degeneracy in % */
      *degeneracy = ((double) countz / nrows);
   } else {
      *degeneracy = 0.0;
   }

   assert(*degeneracy <= 1.0 && *degeneracy >= 0);

   return SCIP_OKAY;
}

/** set subproblem memory limit */
SCIP_RETCODE ObjPricerGcg::setPricingProblemMemorylimit(
   SCIP*                 pricingscip         /**< SCIP of the pricingproblem */
   )
{
   SCIP_Real memlimit;

   assert(pricingscip != NULL);

   assert(GCGisOriginal(origprob));

   SCIP_CALL( SCIPgetRealParam(origprob, "limits/memory", &memlimit) );

   if( !SCIPisInfinity(origprob, memlimit) )
   {
      memlimit -= SCIPgetMemUsed(origprob)/1048576.0 + GCGgetPricingprobsMemUsed(origprob) - SCIPgetMemUsed(pricingscip)/1048576.0;
      if( memlimit < 0 )
         memlimit = 0.0;
      SCIP_CALL( SCIPsetRealParam(pricingscip, "limits/memory", memlimit) );
   }

   return SCIP_OKAY;
}


/** for a pricing problem, get the dual solution value or Farkas value of the convexity constraint */
SCIP_Real ObjPricerGcg::getConvconsDualsol(
   PricingType*          pricetype,           /**< Farkas or Reduced cost pricing */
   int                   probnr               /**< index of corresponding pricing problem */
   )
{
   if( !GCGisPricingprobRelevant(origprob, probnr) )
      return -1.0 * SCIPinfinity(scip_);
   else
      return pricetype->consGetDual(scip_, GCGgetConvCons(origprob, probnr));
}

/** computes the pricing problem objectives
 *  @todo this method could use more parameters as it is private
 */
SCIP_RETCODE ObjPricerGcg::setPricingObjs(
   PricingType*          pricetype,          /**< Farkas or Reduced cost pricing */
   SCIP_Bool             stabilize           /**< do we use stabilization ? */
   )
{
   SCIP_CONS** origconss = NULL;
   SCIP_CONS** masterconss = NULL;
   int nmasterconss;
   SCIP_VAR** probvars = NULL;
   int nprobvars;

   SCIP_ROW** mastercuts = NULL;
   int nmastercuts;
   SCIP_ROW** origcuts = NULL;
   SCIP_COL** cols = NULL;
   SCIP_Real* consvals = NULL;
   SCIP_Real dualsol;

   SCIP_VAR** consvars = NULL;
   int nconsvars;
   int i;
   int j;

   assert(pricerdata != NULL);
   assert(stabilization != NULL);

   /* get the constraints of the master problem and the corresponding constraints in the original problem */
   nmasterconss = GCGgetNMasterConss(origprob);
   masterconss = GCGgetMasterConss(origprob);
   origconss = GCGgetOrigMasterConss(origprob);

   /* set objective value of all variables in the pricing problems to 0 (for farkas pricing) /
    * to the original objective of the variable (for redcost pricing)
    */
   for( i = 0; i < pricerdata->npricingprobs; i++ )
   {
      if( pricerdata->pricingprobs[i] == NULL )
         continue;
      probvars = SCIPgetVars(pricerdata->pricingprobs[i]);
      nprobvars = SCIPgetNVars(pricerdata->pricingprobs[i]);

      for( j = 0; j < nprobvars; j++ )
      {
         assert(GCGvarGetBlock(probvars[j]) == i);
         assert( GCGoriginalVarIsLinking(GCGpricingVarGetOrigvars(probvars[j])[0]) || (GCGvarGetBlock(GCGpricingVarGetOrigvars(probvars[j])[0]) == i));

         SCIP_CALL( SCIPchgVarObj(pricerdata->pricingprobs[i], probvars[j], pricetype->varGetObj(probvars[j])));

         pricerdata->realdualvalues[i][j] = pricetype->varGetObj(probvars[j]);
#ifdef PRINTDUALSOLS
         SCIPdebugMessage("pricingobj var <%s> %f, realdualvalues %f\n", SCIPvarGetName(probvars[j]), pricetype->varGetObj(probvars[j]), pricerdata->realdualvalues[i][j]);
#endif
      }
   }

   /* compute reduced cost for linking variable constraints and update objectives in the pricing problems
    * go through constraints, and select correct variable
    */

   int nlinkconss;
   SCIP_CONS** linkconss = NULL;
   int* linkconssblock = NULL;
   nlinkconss = GCGgetNVarLinkingconss(origprob);
   linkconss = GCGgetVarLinkingconss(origprob);
   linkconssblock = GCGgetVarLinkingconssBlock(origprob);

   for( i = 0; i < nlinkconss; ++i)
   {
      SCIP_VAR** linkconsvars = NULL;
      SCIP_CONS* linkcons = linkconss[i];
      int block = linkconssblock[i];

      linkconsvars = SCIPgetVarsLinear(scip_, linkcons);

      SCIP_VAR* linkvar = linkconsvars[0];

      SCIP_VAR* pricingvar = GCGlinkingVarGetPricingVars(GCGmasterVarGetOrigvars(linkvar)[0])[block];
      assert(GCGvarIsPricing(pricingvar));
      if( stabilize )
      {
         dualsol = stabilization->linkingconsGetDual(i);
      }
      else
      {
         dualsol = pricetype->consGetDual(scip_, linkcons);
      }

      /* add dual solution value to the pricing variable:
       * lambda variables get coef -1 in linking constraints --> add dualsol
       */
      SCIP_CALL( SCIPaddVarObj(pricerdata->pricingprobs[block], pricingvar, dualsol) );
      assert(SCIPvarGetProbindex(pricingvar) >= 0 && SCIPvarGetProbindex(pricingvar) < SCIPgetNVars(pricerdata->pricingprobs[block]));
      pricerdata->realdualvalues[block][SCIPvarGetProbindex(pricingvar)] +=  pricetype->consGetDual(scip_, linkcons);

#ifdef PRINTDUALSOLS
      SCIPdebugMessage("pricingobj var <%s> %f, realdualvalues %f\n", SCIPvarGetName(pricingvar), dualsol, pricetype->consGetDual(scip_, linkcons));
#endif
   }

   /* compute reduced cost and update objectives in the pricing problems */
   for( i = 0; i < nmasterconss; i++ )
   {
      if( stabilize )
      {
         SCIP_CALL( stabilization->consGetDual(i, &dualsol) );
      }
      else
      {
         dualsol = pricetype->consGetDual(scip_, masterconss[i]);
      }

      if( !SCIPisZero(scip_, dualsol) || !SCIPisZero(scip_, pricetype->consGetDual(scip_, masterconss[i])) )
      {
#ifdef PRINTDUALSOLS
         SCIPdebugMessage("mastercons <%s> dualsol: %g\n", SCIPconsGetName(masterconss[i]), dualsol);
#endif

         /* for all variables in the constraint, modify the objective of the corresponding variable in a pricing problem */
         nconsvars = GCGconsGetNVars(origprob, origconss[i]);
         SCIP_CALL( SCIPallocBufferArray(scip_, &consvars, nconsvars) );
         SCIP_CALL( SCIPallocBufferArray(scip_, &consvals, nconsvars) );
         GCGconsGetVars(origprob, origconss[i], consvars, nconsvars);
         GCGconsGetVals(origprob, origconss[i], consvals, nconsvars);
         for( j = 0; j < nconsvars; j++ )
         {
            int blocknr;
            blocknr = GCGvarGetBlock(consvars[j]);
            assert(GCGvarIsOriginal(consvars[j]));
            /* nothing to be done if variable belongs to redundant block or variable was directly transferred to the master
             * or variable is linking variable (which means, the directly transferred copy is part of the master cons)
             */
            if( blocknr >= 0 && pricerdata->pricingprobs[blocknr] != NULL )
            {
               assert(GCGoriginalVarGetPricingVar(consvars[j]) != NULL);
               /* modify the objective of the corresponding variable in the pricing problem */
               SCIP_CALL( SCIPaddVarObj(pricerdata->pricingprobs[blocknr],
                     GCGoriginalVarGetPricingVar(consvars[j]), -1.0 * dualsol * consvals[j]) );

               pricerdata->realdualvalues[blocknr][SCIPvarGetProbindex(GCGoriginalVarGetPricingVar(consvars[j]))] += -1.0 * consvals[j] * pricetype->consGetDual(scip_, masterconss[i]);

#ifdef PRINTDUALSOLS
               SCIPdebugMessage("pricingobj var <%s> %f, realdualvalues %f\n",
                     SCIPvarGetName(GCGoriginalVarGetPricingVar(consvars[j])), dualsol, -1.0 * consvals[j] * pricetype->consGetDual(scip_, masterconss[i]));
#endif
            }
         }
         SCIPfreeBufferArray(scip_, &consvals);
         SCIPfreeBufferArray(scip_, &consvars);
      }
   }

   /* get the cuts of the master problem and the corresponding cuts in the original problem */
   mastercuts = GCGsepaGetMastercuts(scip_);
   nmastercuts = GCGsepaGetNCuts(scip_);
   origcuts = GCGsepaGetOrigcuts(scip_);

   assert(mastercuts != NULL);
   assert(origcuts != NULL);

   /* compute reduced cost and update objectives in the pricing problems */
   for( i = 0; i < nmastercuts; i++ )
   {
      if( stabilize )
      {
         SCIP_CALL( stabilization->rowGetDual(i, &dualsol) );
      }
      else
      {
         dualsol = pricetype->rowGetDual(mastercuts[i]);
      }

      if( !SCIPisZero(scip_, dualsol) || !SCIPisZero(scip_, pricetype->rowGetDual(mastercuts[i])) )
      {
         /* get columns and vals of the cut */
         nconsvars = SCIProwGetNNonz(origcuts[i]);
         cols = SCIProwGetCols(origcuts[i]);
         consvals = SCIProwGetVals(origcuts[i]);

         /* get the variables corresponding to the columns in the cut */
         SCIP_CALL( SCIPallocBufferArray(scip_, &consvars, nconsvars) );
         for( j = 0; j < nconsvars; j++ )
            consvars[j] = SCIPcolGetVar(cols[j]);

         /* for all variables in the cut, modify the objective of the corresponding variable in a pricing problem */
         for( j = 0; j < nconsvars; j++ )
         {
            int blocknr;
            blocknr = GCGvarGetBlock(consvars[j]);
            assert(GCGvarIsOriginal(consvars[j]));
            /* nothing to be done if variable belongs to redundant block or
             * variable was directly transferred to the master
             * or variable is linking variable (which means, the directly transferred copy is part of the master cut) */
            if( blocknr >= 0 && pricerdata->pricingprobs[blocknr] != NULL )
            {
               assert(GCGoriginalVarGetPricingVar(consvars[j]) != NULL);
               /* modify the objective of the corresponding variable in the pricing problem */
               SCIP_CALL( SCIPaddVarObj(pricerdata->pricingprobs[blocknr],
                     GCGoriginalVarGetPricingVar(consvars[j]), -1.0 * dualsol * consvals[j]) );

               pricerdata->realdualvalues[blocknr][SCIPvarGetProbindex(GCGoriginalVarGetPricingVar(consvars[j]))] += -1.0 *consvals[j]* pricetype->rowGetDual(mastercuts[i]);

#ifdef PRINTDUALSOLS
               SCIPdebugMessage("pricingobj var <%s> %f, realdualvalues %f\n",
                     SCIPvarGetName(GCGoriginalVarGetPricingVar(consvars[j])), dualsol, -1.0 * consvals[j] * pricetype->rowGetDual(mastercuts[i]));
#endif
            }
         }
         SCIPfreeBufferArray(scip_, &consvars);
      }
   }

   /* get dual solutions / farkas values of the convexity constraints */
   for( i = 0; i < pricerdata->npricingprobs; i++ )
   {
      assert( GCGisPricingprobRelevant(origprob, i) == (GCGgetConvCons(origprob, i) != NULL) );

      if( !GCGisPricingprobRelevant(origprob, i) )
      {
         pricerdata->dualsolconv[i] = -1.0 * SCIPinfinity(scip_);
         continue;
      }

      pricerdata->dualsolconv[i] = pricetype->consGetDual(scip_, GCGgetConvCons(origprob, i));

#ifdef PRINTDUALSOLS
      if( GCGisPricingprobRelevant(origprob, i) )
      {
         SCIPdebugMessage("convcons <%s> dualsol: %g\n", SCIPconsGetName(GCGgetConvCons(origprob, i)), pricerdata->dualsolconv[i]);
      }
#endif
   }

   return SCIP_OKAY;
}

/** add master variable to all constraints */
SCIP_RETCODE ObjPricerGcg::addVariableToMasterconstraints(
   SCIP_VAR*             newvar,             /**< The new variable to add */
   int                   prob,               /**< number of the pricing problem the solution belongs to */
   SCIP_VAR**            solvars,            /**< array of variables with non-zero value in the solution of the pricing problem */
   SCIP_Real*            solvals,            /**< array of values in the solution of the pricing problem for variables in array solvars*/
   int                   nsolvars            /**< number of variables in array solvars */
   )
{
   int i;
   int c;
   int idx;

   SCIP_CONS** masterconss = NULL;
   int nmasterconss;
   SCIP_Real* mastercoefs = NULL;
   SCIP_CONS* linkcons = NULL;

   assert(pricerdata != NULL);

   nmasterconss = GCGgetNMasterConss(origprob);
   masterconss = GCGgetMasterConss(origprob);

   SCIP_CALL( SCIPallocBufferArray(scip_, &mastercoefs, nmasterconss) ); /*lint !e530*/
   BMSclearMemoryArray(mastercoefs, nmasterconss);

   /* compute coef of the variable in the master constraints */
   for( i = 0; i < nsolvars; i++ )
   {
      if( !SCIPisZero(scip_, solvals[i]) )
      {
         SCIP_CONS** linkconss;
         SCIP_VAR** origvars;
         SCIP_Real* coefs;
         int ncoefs;

         assert(GCGvarIsPricing(solvars[i]));
         origvars = GCGpricingVarGetOrigvars(solvars[i]);
         assert(GCGvarIsOriginal(origvars[0]));

         coefs = GCGoriginalVarGetCoefs(origvars[0]);
         ncoefs = GCGoriginalVarGetNCoefs(origvars[0]);
         assert(!SCIPisInfinity(scip_, solvals[i]));

         /* original variable is a linking variable, just add it to the linkcons */
         if( GCGoriginalVarIsLinking(origvars[0]) )
         {
#ifndef NDEBUG
            SCIP_VAR** pricingvars;
            pricingvars = GCGlinkingVarGetPricingVars(origvars[0]);
#endif
            linkconss = GCGlinkingVarGetLinkingConss(origvars[0]);

            /* the linking constraints could be NULL if the Benders' decomposition is used. */
            if( linkconss != NULL )
            {
               assert(pricingvars[prob] == solvars[i]);
               assert(linkconss[prob] != NULL);
               SCIP_CALL( SCIPaddCoefLinear(scip_, linkconss[prob], newvar, -solvals[i]) );
            }
            continue;
         }

         /* for each coef, add coef * solval to the coef of the new variable for the corresponding constraint */
         for( c = 0; c < ncoefs; c++ )
         {
            linkconss = GCGoriginalVarGetMasterconss(origvars[0]);
            assert(!SCIPisZero(scip_, coefs[c]));
            SCIP_CALL( SCIPgetTransformedCons(scip_, linkconss[c], &linkcons) );

            idx = (int)(size_t)SCIPhashmapGetImage(pricerdata->mapcons2idx, linkcons); /*lint !e507*/
            assert(0 <= idx && idx < nmasterconss);
            assert(masterconss[idx] == linkcons);
            mastercoefs[idx] += coefs[c] * solvals[i];
         }

      }
   }

   /* add the variable to the master constraints */
   for( i = 0; i < nmasterconss; i++ )
   {
      if( !SCIPisZero(scip_, mastercoefs[i]) )
      {
         assert(!SCIPisInfinity(scip_, mastercoefs[i]) && !SCIPisInfinity(scip_, -mastercoefs[i]));
         SCIP_CALL( SCIPaddCoefLinear(scip_, masterconss[i], newvar, mastercoefs[i]) );
      }
   }

   SCIPfreeBufferArray(scip_, &mastercoefs);
   return SCIP_OKAY;
}

/** add master variable to all constraints */
SCIP_RETCODE ObjPricerGcg::addVariableToMasterconstraintsFromGCGCol(
   SCIP_VAR*             newvar,             /**< The new variable to add */
   GCG_COL*              gcgcol              /**< GCG column data structure */
   )
{
   SCIP_CONS** masterconss;
   int nmasterconss;
   SCIP_Real* mastercoefs;
   int nlinkvars;
   int* linkvars;

   SCIP_VAR**            solvars;            /* array of variables with non-zero value in the solution of the pricing problem */
   SCIP_Real*            solvals;            /* array of values in the solution of the pricing problem for variables in array solvars*/
#ifndef NDEBUG
   int                   nsolvars;            /* number of variables in array solvars */
#endif

   int i;
   int prob;

   assert(pricerdata != NULL);

   nmasterconss = GCGgetNMasterConss(origprob);
   masterconss = GCGgetMasterConss(origprob);

   SCIP_CALL( computeColMastercoefs(gcgcol) );

   mastercoefs = GCGcolGetMastercoefs(gcgcol);

   nlinkvars = GCGcolGetNLinkvars(gcgcol);
   linkvars = GCGcolGetLinkvars(gcgcol);
   solvars = GCGcolGetVars(gcgcol);
   solvals = GCGcolGetVals(gcgcol);
#ifndef NDEBUG
   nsolvars = GCGcolGetNVars(gcgcol);
#endif

   prob = GCGcolGetProbNr(gcgcol);

   /* compute coef of the variable in the master constraints */
   for( i = 0; i < nlinkvars; i++ )
   {
      SCIP_CONS** linkconss;
      SCIP_VAR** origvars;

      assert(linkvars[i] < nsolvars );
      assert(GCGvarIsPricing(solvars[linkvars[i]]));
      origvars = GCGpricingVarGetOrigvars(solvars[linkvars[i]]);
      assert(GCGvarIsOriginal(origvars[0]));

      assert(!SCIPisInfinity(scip_, solvals[linkvars[i]]));

      assert(GCGoriginalVarIsLinking(origvars[0]));
      /* original variable is a linking variable, just add it to the linkcons */
#ifndef NDEBUG
      SCIP_VAR** pricingvars;
      pricingvars = GCGlinkingVarGetPricingVars(origvars[0]);
#endif
      linkconss = GCGlinkingVarGetLinkingConss(origvars[0]);

      assert(pricingvars[prob] == solvars[linkvars[i]]);
      assert(linkconss[prob] != NULL);
      SCIP_CALL( SCIPaddCoefLinear(scip_, linkconss[prob], newvar, -solvals[linkvars[i]]) );
   }

   /* add the variable to the master constraints */
   for( i = 0; i < nmasterconss; i++ )
   {
      if( !SCIPisZero(scip_, mastercoefs[i]) )
      {
         assert(!SCIPisInfinity(scip_, mastercoefs[i]) && !SCIPisInfinity(scip_, -mastercoefs[i]));
         SCIP_CALL( SCIPaddCoefLinear(scip_, masterconss[i], newvar, mastercoefs[i]) );
      }
   }

   return SCIP_OKAY;
}


/** compute master coefficients of column */
SCIP_RETCODE ObjPricerGcg::computeColMastercoefs(
   GCG_COL*              gcgcol              /**< GCG column data structure */
   )
{
   int i;

   SCIP_VAR** solvars = NULL;
   SCIP_Real* solvals = NULL;
   int nsolvars;

   int c;
   int idx;

   assert(scip_ != NULL);
   assert(gcgcol != NULL);

   nsolvars = GCGcolGetNVars(gcgcol);
   solvars = GCGcolGetVars(gcgcol);
   solvals = GCGcolGetVals(gcgcol);

   int nmasterconss;
   SCIP_Real* mastercoefs = NULL;
   SCIP_CONS* linkcons = NULL;
   SCIP* pricingprob = NULL;

   int* linkvars = NULL;
   int nlinkvars;

   pricingprob = GCGcolGetPricingProb(gcgcol);

   nmasterconss = GCGgetNMasterConss(origprob);

   assert(GCGcolGetNMastercoefs(gcgcol) == 0 || GCGcolGetNMastercoefs(gcgcol) == nmasterconss);

   if( GCGcolGetInitializedCoefs(gcgcol) )
   {
//      SCIPdebugMessage("Coefficients already computed, nmastercoefs = %d\n", GCGcolGetNMastercoefs(gcgcol));
      return SCIP_OKAY;
   }

   if( nmasterconss > 0)
   {
      SCIP_CALL( SCIPallocBufferArray(pricingprob, &mastercoefs, nmasterconss) ); /*lint !e530*/
      BMSclearMemoryArray(mastercoefs, nmasterconss);
   }

   SCIP_CALL( SCIPallocBufferArray(pricingprob, &linkvars, nsolvars) ); /*lint !e530*/

   nlinkvars = 0;

   /* compute coef of the variable in the master constraints */
   for( i = 0; i < nsolvars; i++ )
   {
      if( !SCIPisZero(origprob, solvals[i]) )
      {
         SCIP_CONS** linkconss;
         SCIP_VAR** origvars;
         SCIP_Real* coefs;
         int ncoefs;

         assert(GCGvarIsPricing(solvars[i]));
         origvars = GCGpricingVarGetOrigvars(solvars[i]);
         assert(GCGvarIsOriginal(origvars[0]));

         coefs = GCGoriginalVarGetCoefs(origvars[0]);
         ncoefs = GCGoriginalVarGetNCoefs(origvars[0]);
         assert(!SCIPisInfinity(origprob, solvals[i]));

         /* original variable is a linking variable, just add it to the linkcons */
         if( GCGoriginalVarIsLinking(origvars[0]) )
         {
            linkvars[nlinkvars] = i;
            ++nlinkvars;

            continue;
         }

         /* for each coef, add coef * solval to the coef of the new variable for the corresponding constraint */
         for( c = 0; c < ncoefs; c++ )
         {
            linkconss = GCGoriginalVarGetMasterconss(origvars[0]);
            assert(!SCIPisZero(origprob, coefs[c]));
            SCIP_CALL( SCIPgetTransformedCons(scip_, linkconss[c], &linkcons) );

            idx = (int)(size_t)SCIPhashmapGetImage(pricerdata->mapcons2idx, linkcons); /*lint !e507*/
            assert(0 <= idx && idx < nmasterconss);
            assert(!SCIPisInfinity(scip_, ABS(coefs[c] * solvals[i])));
            mastercoefs[idx] += coefs[c] * solvals[i];
            assert(!SCIPisInfinity(scip_, ABS(mastercoefs[idx])));
         }
      }
   }

   SCIP_CALL( GCGcolSetMastercoefs(gcgcol, mastercoefs, nmasterconss) );

   SCIP_CALL( GCGcolSetLinkvars(gcgcol, linkvars, nlinkvars) );

   SCIP_CALL( GCGcolSetInitializedCoefs(gcgcol) );

   SCIPfreeBufferArray(pricingprob, &linkvars); /*lint !e530*/

   if( nmasterconss > 0)
      SCIPfreeBufferArray(pricingprob, &mastercoefs); /*lint !e530*/

   return SCIP_OKAY;
}

/** add variable with computed coefficients to the master cuts */
SCIP_RETCODE ObjPricerGcg::addVariableToMastercuts(
   SCIP_VAR*             newvar,             /**< The new variable to add */
   int                   prob,               /**< number of the pricing problem the solution belongs to */
   SCIP_VAR**            solvars,            /**< array of variables with non-zero value in the solution of the pricing problem */
   SCIP_Real*            solvals,            /**< array of values in the solution of the pricing problem for variables in array solvars*/
   int                   nsolvars            /**< number of variables in array solvars */
   )
{
   SCIP_ROW** mastercuts;
   int nmastercuts;
   SCIP_ROW** origcuts;

   SCIP_COL** cols;
   SCIP_Real conscoef;
   SCIP_VAR* var;
   SCIP_Real* consvals;

   int i;
   int j;
   int k;

   assert(scip_ != NULL);
   assert(newvar != NULL);
   assert(solvars != NULL || nsolvars == 0);
   assert(solvals != NULL || nsolvars == 0);

   /* get the cuts of the master problem and the corresponding cuts in the original problem */
   mastercuts = GCGsepaGetMastercuts(scip_);
   nmastercuts = GCGsepaGetNCuts(scip_);
   origcuts = GCGsepaGetOrigcuts(scip_);

   assert(mastercuts != NULL);
   assert(origcuts != NULL);

   /* compute coef of the variable in the cuts and add it to the cuts */
   for( i = 0; i < nmastercuts; i++ )
   {
      if( !SCIProwIsInLP(mastercuts[i]) )
         continue;

      /* get columns of the cut and their coefficients */
      cols = SCIProwGetCols(origcuts[i]);
      consvals = SCIProwGetVals(origcuts[i]);

      conscoef = 0;

      for( j = 0; j < SCIProwGetNNonz(origcuts[i]); j++ )
      {
         int blocknr;
         var = SCIPcolGetVar(cols[j]);
         blocknr = GCGvarGetBlock(var);
         assert(GCGvarIsOriginal(var));

         /* if the belongs to the same block and is no linking variable, update the coef */
         if( blocknr == prob )
            for( k = 0; k < nsolvars; k++ )
               if( solvars[k] == GCGoriginalVarGetPricingVar(var) )
               {
                  conscoef += (consvals[j] * solvals[k]);
                  break;
               }
      }

      if( !SCIPisZero(scip_, conscoef) )
         SCIP_CALL( SCIPaddVarToRow(scip_ , mastercuts[i], newvar, conscoef) );
   }

   return SCIP_OKAY;
}

/** add variable with computed coefficients to the master cuts */
SCIP_RETCODE ObjPricerGcg::addVariableToMastercutsFromGCGCol(
   SCIP_VAR*             newvar,             /**< The new variable to add */
   GCG_COL*              gcgcol              /**< GCG column data structure */
   )
{
   SCIP_ROW** mastercuts;
   int nmastercuts;
   SCIP_Real* mastercutcoefs;
   int i;

   assert(scip_ != NULL);
   assert(newvar != NULL);

   /* get the cuts of the master problem and the corresponding cuts in the original problem */
   mastercuts = GCGsepaGetMastercuts(scip_);
   nmastercuts = GCGsepaGetNCuts(scip_);

   assert(mastercuts != NULL);

   SCIP_CALL( computeColMastercuts(gcgcol) );

   mastercutcoefs = GCGcolGetMastercuts(gcgcol);

   /* compute coef of the variable in the cuts and add it to the cuts */
   for( i = 0; i < nmastercuts; i++ )
   {
      if( !SCIProwIsInLP(mastercuts[i]) )
         continue;

      if( !SCIPisZero(scip_, mastercutcoefs[i]) )
         SCIP_CALL( SCIPaddVarToRow(scip_ , mastercuts[i], newvar, mastercutcoefs[i]) );
   }

   return SCIP_OKAY;
}


/** compute master cut coefficients of column */
SCIP_RETCODE ObjPricerGcg::computeColMastercuts(
   GCG_COL*              gcgcol              /**< GCG column data structure */
   )
{
   int prob;
   int i;

   SCIP_VAR** solvars = NULL;
   SCIP_Real* solvals = NULL;
   int nsolvars;
   int noldmastercuts;
   int nnewmastercuts;
   SCIP_Real* newmastercuts = NULL;

   assert(scip_ != NULL);
   assert(gcgcol != NULL);

   prob = GCGcolGetProbNr(gcgcol);
   nsolvars = GCGcolGetNVars(gcgcol);
   solvars = GCGcolGetVars(gcgcol);
   solvals = GCGcolGetVals(gcgcol);

   noldmastercuts = GCGcolGetNMastercuts(gcgcol);

   SCIP_ROW** mastercuts = NULL;
   int nmastercuts;
   SCIP_ROW** origcuts = NULL;

   SCIP_COL** cols = NULL;
   SCIP_Real conscoef;
   SCIP_VAR* var = NULL;
   SCIP_Real* consvals = NULL;

   int j;
   int k;

   assert(scip_ != NULL);
   assert(solvars != NULL);
   assert(solvals != NULL);

   /* get the cuts of the master problem and the corresponding cuts in the original problem */
   mastercuts = GCGsepaGetMastercuts(scip_);
   nmastercuts = GCGsepaGetNCuts(scip_);
   origcuts = GCGsepaGetOrigcuts(scip_);

   assert(mastercuts != NULL);
   assert(origcuts != NULL);

   assert(nmastercuts - noldmastercuts >= 0);

   if( nmastercuts - noldmastercuts == 0 )
      return SCIP_OKAY;

   SCIP_CALL( SCIPallocBufferArray(origprob, &newmastercuts, nmastercuts - noldmastercuts) );

   nnewmastercuts = 0;

   /* compute coef of the variable in the cuts and add it to the cuts */
   for( i = noldmastercuts; i < nmastercuts; i++ )
   {
      if( !SCIProwIsInLP(mastercuts[i]) )
      {
         newmastercuts[nnewmastercuts] = 0.0;
         ++nnewmastercuts;
         continue;
      }

      /* get columns of the cut and their coefficients */
      cols = SCIProwGetCols(origcuts[i]);
      consvals = SCIProwGetVals(origcuts[i]);

      conscoef = 0;

      for( j = 0; j < SCIProwGetNNonz(origcuts[i]); j++ )
      {
         int blocknr;
         var = SCIPcolGetVar(cols[j]);
         blocknr = GCGvarGetBlock(var);
         assert(GCGvarIsOriginal(var));

         /* if the belongs to the same block and is no linking variable, update the coef */
         if( blocknr == prob )
            for( k = 0; k < nsolvars; k++ )
               if( solvars[k] == GCGoriginalVarGetPricingVar(var) )
               {
                  conscoef += ( consvals[j] * solvals[k] );
                  break;
               }
      }

      newmastercuts[nnewmastercuts] = conscoef;
      ++nnewmastercuts;
   }

   GCGcolUpdateMastercuts(gcgcol, newmastercuts, nnewmastercuts);

   SCIPfreeBufferArray(origprob, &newmastercuts);

   return SCIP_OKAY;
}

/** adds new variable to the end of the priced variables array */
SCIP_RETCODE ObjPricerGcg::addVariableToPricedvars(
   SCIP_VAR*             newvar              /**< variable to add */
   )
{
   SCIP_CALL( ensureSizePricedvars(pricerdata->npricedvars + 1) );
   pricerdata->pricedvars[pricerdata->npricedvars] = newvar;
   GCGmasterVarSetIndex(newvar, pricerdata->npricedvars);
   pricerdata->npricedvars++;

   return SCIP_OKAY;
}

#ifdef SCIP_STATISTIC
/** adds new bounds to the bound arrays as well as some additional information on dual variables and root lp solution */
SCIP_RETCODE ObjPricerGcg::addRootBounds(
   SCIP_Real             primalbound,        /**< new primal bound for the root master LP */
   SCIP_Real             dualbound           /**< new dual bound for the root master LP */
   )
{
   int nprobvars;
   int i;
   int j;

   SCIP_SOL* sol;
   SCIP_Real* solvals;
   SCIP_VAR** vars;
   int nvars;

   nvars = SCIPgetNVars(scip_);
   vars = SCIPgetVars(scip_);

   SCIP_CALL( ensureSizeRootBounds(pricerdata->nrootbounds + 1) );
   pricerdata->rootpbs[pricerdata->nrootbounds] = primalbound;
   pricerdata->rootdbs[pricerdata->nrootbounds] = dualbound;
   pricerdata->roottimes[pricerdata->nrootbounds] = SCIPgetSolvingTime(scip_) - pricerdata->rootfarkastime;
   pricerdata->rootdualdiffs[pricerdata->nrootbounds] = pricerdata->dualdiff;

   SCIPdebugMessage("Add new bounds: \n pb = %f\n db = %f\n", primalbound, dualbound);

   SCIP_CALL( SCIPallocBlockMemoryArray(scip_, &pricerdata->dualvalues[pricerdata->nrootbounds], pricerdata->npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip_, &pricerdata->dualsolconvs[pricerdata->nrootbounds], pricerdata->npricingprobs) );

   for( i = 0; i < pricerdata->npricingprobs; i++ )
   {
      if( pricerdata->pricingprobs[i] == NULL )
         continue;

      nprobvars = SCIPgetNVars(pricerdata->pricingprobs[i]);

      pricerdata->dualsolconvs[pricerdata->nrootbounds][i] = pricerdata->dualsolconv[i];
      SCIP_CALL( SCIPallocBlockMemoryArray(scip_, &(pricerdata->dualvalues[pricerdata->nrootbounds][i]), nprobvars) );

      for( j = 0; j < nprobvars; j++ )
         pricerdata->dualvalues[pricerdata->nrootbounds][i][j] = pricerdata->realdualvalues[i][j];
   }

   pricerdata->nrootbounds++;

   SCIP_CALL( SCIPallocBufferArray(scip_, &solvals, nvars) );

   SCIP_CALL( SCIPgetSolVals(scip_, NULL, nvars, vars, solvals) );

   SCIP_CALL( SCIPcreateSol(scip_, &sol, NULL) );

   SCIP_CALL( SCIPsetSolVals(scip_, sol, nvars, vars, solvals) );

   if( pricerdata->rootlpsol != NULL)
      SCIPfreeSol(scip_, &pricerdata->rootlpsol);

   pricerdata->rootlpsol = sol;

   SCIPfreeBufferArray(scip_, &solvals);

   return SCIP_OKAY;
}
#endif

SCIP_Real ObjPricerGcg::computeRedCost(
   PricingType*          pricetype,          /**< type of pricing */
   SCIP_SOL*             sol,                /**< solution to compute reduced cost for */
   SCIP_Bool             solisray,           /**< is the solution a ray? */
   int                   probnr,             /**< number of the pricing problem the solution belongs to */
   SCIP_Real*            objvalptr           /**< pointer to store the computed objective value */
   ) const
{
   GCG_PRICINGPROB* pricingprob;
   SCIP* pricingscip;
   SCIP_CONS** branchconss;                  /* stack of genericbranching constraints */
   int nbranchconss;                         /* number of generic branching constraints */
   SCIP_Real* branchduals;                   /* dual values of branching constraints in the master (sigma) */
   int i;

   SCIP_VAR** solvars;
   SCIP_Real* solvals = NULL;
   int nsolvars;
   SCIP_Real objvalue;

   assert(pricerdata != NULL);

   pricingprob = pricingcontroller->getPricingprob(probnr);
   assert(pricingprob != NULL);

   objvalue = 0.0;
   pricingscip = pricerdata->pricingprobs[probnr];
   solvars = SCIPgetOrigVars(pricingscip);
   nsolvars = SCIPgetNOrigVars(pricingscip);
   SCIP_CALL_ABORT( SCIPallocBlockMemoryArray(scip_, &solvals, nsolvars) );
   SCIP_CALL_ABORT( SCIPgetSolVals(pricingscip, sol, nsolvars, solvars, solvals) );

   /* compute the objective function value of the solution */
   for( i = 0; i < nsolvars; i++ )
      objvalue += solvals[i] * pricerdata->realdualvalues[probnr][SCIPvarGetProbindex(solvars[i])];

   if( objvalptr != NULL )
      *objvalptr = objvalue;

   /* get path to the last generic branching node */
   GCGpricingprobGetGenericBranchData(pricingprob, &branchconss, &branchduals, &nbranchconss);

   for( i = nbranchconss - 1; i >= 0; --i )
   {
      SCIP_Bool feasible;
      SCIP_CALL_ABORT( checkBranchingBoundChanges(probnr, sol, branchconss[i], &feasible) );
      if( feasible )
      {
         objvalue -= branchduals[i];
      }
   }
   SCIPfreeBlockMemoryArray(scip_, &solvals, nsolvars);

   /* compute reduced cost of variable (i.e. subtract dual solution of convexity constraint, if solution corresponds to a point) */
   return (solisray ? objvalue : objvalue - pricerdata->dualsolconv[probnr]);
}

SCIP_Real ObjPricerGcg::computeRedCostGcgCol(
   PricingType*          pricetype,          /**< type of pricing */
   GCG_Col*              gcgcol,             /**< gcg column to compute reduced cost for */
   SCIP_Real*            objvalptr           /**< pointer to store the computed objective value */
   ) const
{
   GCG_PRICINGPROB* pricingprob;
   SCIP_CONS** branchconss;                  /* stack of genericbranching constraints */
   int nbranchconss;                         /* number of generic branching constraints */
   SCIP_Real* branchduals;                   /* dual values of branching constraints in the master (sigma) */
   int i;
   int probnr;

   SCIP_Bool isray;

   SCIP_Real redcost;

   SCIP_VAR** solvars;
   SCIP_Real* solvals;
   int nsolvars;
   SCIP_Real objvalue;

   assert(pricerdata != NULL);

   objvalue = 0.0;
   probnr = GCGcolGetProbNr(gcgcol);

   solvars = GCGcolGetVars(gcgcol);
   nsolvars = GCGcolGetNVars(gcgcol);
   solvals = GCGcolGetVals(gcgcol);
   isray = GCGcolIsRay(gcgcol);

   pricingprob = pricingcontroller->getPricingprob(probnr);
   assert(pricingprob != NULL);

   /* compute the objective function value of the column */
   for( i = 0; i < nsolvars; i++ )
      objvalue += solvals[i] * pricerdata->realdualvalues[probnr][SCIPvarGetProbindex(solvars[i])];

   if( objvalptr != NULL )
      *objvalptr = objvalue;

   /* get path to the last generic branching node */
   GCGpricingprobGetGenericBranchData(pricingprob, &branchconss, &branchduals, &nbranchconss);

   for( i = nbranchconss -1; i >= 0; --i )
   {
      SCIP_Bool feasible;
      SCIP_CALL_ABORT( checkBranchingBoundChangesGcgCol(gcgcol, branchconss[i], &feasible) );
      if( feasible )
      {
         objvalue -= branchduals[i];
      }
   }

   redcost = (isray ? objvalue : objvalue - pricerdata->dualsolconv[probnr]);

   GCGcolUpdateRedcost(gcgcol, redcost, FALSE);

   return redcost;
}

/** for given columns, (re-)compute and update their reduced costs */
void ObjPricerGcg::updateRedcosts(
   PricingType*          pricetype,          /**< type of pricing */
   GCG_COL**             cols,               /**< columns to compute reduced costs for */
   int                   ncols               /**< number of columns */
   ) const
{
   SCIPdebugMessage("Update reduced costs\n");

   for( int i = 0; i < ncols; ++i )
   {
      SCIP_Real redcost = computeRedCostGcgCol(pricetype, cols[i], NULL);
      GCGcolUpdateRedcost(cols[i], redcost, FALSE);

      SCIPdebugMessage("  -> column %d/%d <%p>, reduced cost = %g\n", i+1, ncols, (void*) cols[i], redcost);
   }
}

/** add a new column to the pricing storage */
SCIP_RETCODE ObjPricerGcg::addColToPricestore(
   GCG_COL*              col                 /**< priced col */
   )
{
   SCIP_RETCODE retcode;

   assert(col != NULL);
   assert(pricingtype != NULL);

   SCIP_Real redcost = computeRedCostGcgCol(pricingtype, col, NULL);
   GCGcolUpdateRedcost(col, redcost, FALSE);

   SCIPdebugMessage("  -> new column <%p>, reduced cost = %g\n", (void*) col, redcost);

   #pragma omp critical (update)
   {
      retcode = GCGpricestoreAddCol(scip_, pricestore, col, FALSE);
   }
   SCIP_CALL( retcode );

   return SCIP_OKAY;
}

/** for each pricing problem, get the best found column from the pricing storage */
void ObjPricerGcg::getBestCols(
   GCG_COL**             pricingprobcols     /**< array to be filled with best column per pricing problem */
   )
{
   GCG_COL** cols;
   int ncols;

   int i;

   assert(pricingprobcols != NULL);

   cols = GCGpricestoreGetCols(pricestore);
   ncols = GCGpricestoreGetNCols(pricestore);

   for( i = 0; i < pricerdata->npricingprobs; ++i )
      pricingprobcols[i] = NULL;

   for( i = 0; i < ncols; ++i )
   {
      int probnr = GCGcolGetProbNr(cols[i]);

      if( pricingprobcols[probnr] == NULL
         || SCIPisDualfeasLT(scip_, GCGcolGetRedcost(cols[i]), GCGcolGetRedcost(pricingprobcols[probnr])) )
         pricingprobcols[probnr] = cols[i];
   }
}

/** get the sum over the dual values of convexity constraints */
SCIP_Real ObjPricerGcg::getDualconvsum(
   GCG_COL**             bestcols            /**< best columns found per pricing problem */
   )
{
   SCIP_Real dualconvsum = 0.0;

   assert(pricingtype != NULL);

   for( int i = 0; i < pricerdata->npricingprobs; ++i )
   {
      if( GCGisPricingprobRelevant(origprob, i) && (bestcols[i] == NULL || !GCGcolIsRay(bestcols[i])) )
         dualconvsum += GCGgetNIdenticalBlocks(origprob, i) * pricingtype->consGetDual(scip_, GCGgetConvCons(origprob, i));
   }

   return dualconvsum;
}

/* computes the objective value of the current (stabilized) dual variables) in the dual program */
 SCIP_RETCODE ObjPricerGcg::getStabilizedDualObjectiveValue(
    PricingType*         pricetype,          /**< type of pricing */
    SCIP_Real*           stabdualval,        /**< pointer to store stabilized dual objective value */
    SCIP_Bool            stabilize           /**< stabilize? */
)
{
   SCIP_VAR** mastervars = NULL;
   int nmastervars;

   SCIP_CONS** origconss = NULL;

   SCIP_ROW** origcuts = NULL;
   SCIP_COL** cols = NULL;
   SCIP_Real* consvals = NULL;

   SCIP_VAR** consvars = NULL;
   int nconsvars;
   int j;

   SCIP_Real dualobjval;
   SCIP_Real dualsol;
   SCIP_Real boundval;

   SCIP_CONS** masterconss = NULL;
   int nmasterconss;

   int nlinkconss;
   SCIP_CONS** linkconss = NULL;

   SCIP_ROW** mastercuts = NULL;
   int nmastercuts;
   int i;

   SCIP_Real* stabredcosts = NULL;

   assert(stabilization != NULL);
   assert(stabdualval != NULL);

   *stabdualval = 0.0;

   /* get the constraints of the master problem and the corresponding constraints in the original problem */
   nmasterconss = GCGgetNMasterConss(origprob);
   masterconss = GCGgetMasterConss(origprob);
   origconss = GCGgetOrigMasterConss(origprob);

   dualobjval = 0.0;

   nlinkconss = GCGgetNVarLinkingconss(origprob);
   linkconss = GCGgetVarLinkingconss(origprob);

   /* get the cuts of the master problem */
   mastercuts = GCGsepaGetMastercuts(scip_);
   nmastercuts = GCGsepaGetNCuts(scip_);

   assert(mastercuts != NULL);

   /* compute lhs/rhs * dual for linking constraints and add it to dualobjval */
   for( i = 0; i < nlinkconss; ++i )
   {
      SCIP_CONS* linkcons = linkconss[i];
#ifndef NDEBUG
      SCIP_VAR** linkconsvars;
      int block = GCGgetVarLinkingconssBlock(origprob)[i];

      linkconsvars = SCIPgetVarsLinear(scip_, linkcons);

      SCIP_VAR* linkvar = linkconsvars[0];

      assert(GCGvarIsPricing(GCGlinkingVarGetPricingVars(GCGmasterVarGetOrigvars(linkvar)[0])[block]));
#endif

      if( stabilize )
         dualsol = stabilization->linkingconsGetDual(i);
      else
         dualsol = pricetype->consGetDual(scip_, linkcons);

      if( SCIPisFeasPositive(scip_, dualsol) )
         boundval = SCIPgetLhsLinear(scip_, linkcons);
      else if( SCIPisFeasNegative(scip_, dualsol) )
         boundval = SCIPgetRhsLinear(scip_, linkcons);
      else
         continue;

      assert(SCIPisZero(scip_, boundval));

      if( !SCIPisZero(scip_, boundval) )
         dualobjval += boundval * dualsol;
   }


   /* compute lhs/rhs * dual for master constraints and add it to dualobjval */
   for( i = 0; i < nmasterconss; i++ )
   {
      if( stabilize )
         SCIP_CALL( stabilization->consGetDual(i, &dualsol) );
      else
         dualsol = pricetype->consGetDual(scip_, masterconss[i]);

      if( SCIPisFeasPositive(scip_, dualsol) )
         boundval = GCGconsGetLhs(scip_, origconss[i]);
      else if( SCIPisFeasNegative(scip_, dualsol) )
         boundval = GCGconsGetRhs(scip_, origconss[i]);
      else
         continue;

      if( !SCIPisZero(scip_, boundval) )
         dualobjval += boundval * dualsol;
   }

   /* compute lhs/rhs * dual for master cuts and add it to dualobjval */
   for( i = 0; i < nmastercuts; i++ )
   {
      if( stabilize )
         SCIP_CALL( stabilization->rowGetDual(i, &dualsol) );
      else
         dualsol = pricetype->rowGetDual(mastercuts[i]);

      if( SCIPisFeasPositive(scip_, dualsol) )
         boundval = SCIProwGetLhs(mastercuts[i]);
      else if( SCIPisFeasNegative(scip_, dualsol) )
         boundval = SCIProwGetRhs(mastercuts[i]);
      else
         continue;

      if( !SCIPisZero(scip_, boundval) )
         dualobjval += boundval * dualsol;
   }

   /* get master variables that were directly transferred or that are linking */
   mastervars = SCIPgetOrigVars(scip_);
   nmastervars = GCGgetNTransvars(origprob) + GCGgetNLinkingvars(origprob);

   assert(nmastervars <= SCIPgetNOrigVars(scip_));

   /* no linking or directly transferred variables exist, set stabdualval pointer and exit */
   if( nmastervars == 0 )
   {
      *stabdualval = dualobjval;

      return SCIP_OKAY;
   }

   /* allocate memory for array with (stabilizied) reduced cost coefficients */
   SCIP_CALL( SCIPallocBufferArray(scip_, &stabredcosts, nmastervars) );

   /* initialize (stabilized) reduced cost with objective coefficients */
   for( i = 0; i < nmastervars; i++ )
   {
      assert(GCGvarGetBlock(mastervars[i]) == -1);
      assert( GCGoriginalVarIsLinking(GCGmasterVarGetOrigvars(mastervars[i])[0]) || GCGoriginalVarIsTransVar(GCGmasterVarGetOrigvars(mastervars[i])[0]) );

      stabredcosts[i] = SCIPvarGetObj(mastervars[i]);
   }

   /* compute reduced cost for linking variable constraints and update (stabilized) reduced cost coefficients
    * go through constraints, and select correct variable
    */
   nlinkconss = GCGgetNVarLinkingconss(origprob);
   linkconss = GCGgetVarLinkingconss(origprob);

   for( i = 0; i < nlinkconss; ++i )
   {
      SCIP_VAR** linkconsvars;
      SCIP_CONS* linkcons = linkconss[i];
      int varindex;

      linkconsvars = SCIPgetVarsLinear(scip_, linkcons);

      SCIP_VAR* linkvar = linkconsvars[0];

      varindex = SCIPvarGetProbindex(linkvar);
      assert(varindex < nmastervars);

      if( stabilize )
      {
         dualsol = stabilization->linkingconsGetDual(i);
      }
      else
      {
         dualsol = pricetype->consGetDual(scip_, linkcons);
      }

      /* substract dual solution value to the linking variable:
       * linking variables get coef 11 in linking constraints --> substract dualsol
       */
      stabredcosts[varindex] -= dualsol;
   }

   /* compute reduced cost for master constraints and update (stabilized) reduced cost coefficients */
   for( i = 0; i < nmasterconss; i++ )
   {
      if( stabilize )
      {
         SCIP_CALL( stabilization->consGetDual(i, &dualsol) );
      }
      else
      {
         dualsol = pricetype->consGetDual(scip_, masterconss[i]);
      }

      if( !SCIPisZero(scip_, dualsol) )
      {
         /* for all variables in the constraint, modify the objective of the corresponding variable in a pricing problem */
         nconsvars = GCGconsGetNVars(origprob, origconss[i]);
         SCIP_CALL( SCIPallocBufferArray(scip_, &consvars, nconsvars) );
         SCIP_CALL( SCIPallocBufferArray(scip_, &consvals, nconsvars) );
         GCGconsGetVars(origprob, origconss[i], consvars, nconsvars);
         GCGconsGetVals(origprob, origconss[i], consvals, nconsvars);
         for( j = 0; j < nconsvars; j++ )
         {
            SCIP_VAR* mastervar;
            int blocknr;

            assert(GCGvarIsOriginal(consvars[j]));

            if( GCGoriginalVarGetNMastervars(consvars[j]) == 0 )
               continue;
            assert( GCGoriginalVarGetNMastervars(consvars[j]) > 0 );

            mastervar = GCGoriginalVarGetMastervars(consvars[j])[0];
            blocknr = GCGvarGetBlock(mastervar);

            /* nothing to be done if variable belongs to redundant block or variable was directly transferred to the master
             * or variable is linking variable (which means, the directly transferred copy is part of the master cons)
             */
            if( blocknr < 0 )
            {
               int varindex;
               varindex = SCIPvarGetProbindex(mastervar);
               assert(varindex < nmastervars);

               stabredcosts[varindex] -= dualsol * consvals[j];
            }
         }
         SCIPfreeBufferArray(scip_, &consvals);
         SCIPfreeBufferArray(scip_, &consvars);
      }
   }

   /* get the cuts of the master problem and the corresponding cuts in the original problem */
   mastercuts = GCGsepaGetMastercuts(scip_);
   nmastercuts = GCGsepaGetNCuts(scip_);
   origcuts = GCGsepaGetOrigcuts(scip_);

   assert(mastercuts != NULL);
   assert(origcuts != NULL);

   /* compute reduced cost for master cuts and update (stabilized) reduced cost coefficients */
   for( i = 0; i < nmastercuts; i++ )
   {
      if( stabilize )
      {
         SCIP_CALL( stabilization->rowGetDual(i, &dualsol) );
      }
      else
      {
         dualsol = pricetype->rowGetDual(mastercuts[i]);
      }

      if( !SCIPisZero(scip_, dualsol) )
      {
         /* get columns and vals of the cut */
         nconsvars = SCIProwGetNNonz(origcuts[i]);
         cols = SCIProwGetCols(origcuts[i]);
         consvals = SCIProwGetVals(origcuts[i]);

         /* get the variables corresponding to the columns in the cut */
         SCIP_CALL( SCIPallocBufferArray(scip_, &consvars, nconsvars) );
         for( j = 0; j < nconsvars; j++ )
            consvars[j] = SCIPcolGetVar(cols[j]);

         /* for all variables in the cut, modify the objective of the corresponding variable in a pricing problem */
         for( j = 0; j < nconsvars; j++ )
         {
            SCIP_VAR* mastervar;
            int blocknr;

            assert(GCGvarIsOriginal(consvars[j]));

            if( GCGoriginalVarGetNMastervars(consvars[j]) == 0 )
               continue;
            assert( GCGoriginalVarGetNMastervars(consvars[j]) > 0 );

            mastervar = GCGoriginalVarGetMastervars(consvars[j])[0];
            blocknr = GCGvarGetBlock(mastervar);

            /* nothing to be done if variable belongs to redundant block or variable was directly transferred to the master
             * or variable is linking variable (which means, the directly transferred copy is part of the master cons)
             */
            if( blocknr < 0 )
            {
               int varindex;
               varindex = SCIPvarGetProbindex(mastervar);
               assert(varindex < nmastervars);

               stabredcosts[varindex] -= dualsol * consvals[j];
            }
         }
         SCIPfreeBufferArray(scip_, &consvars);
      }
   }

   /* add redcost coefficients * lb/ub of linking or directly transferred variables */
   for( i = 0; i < nmastervars; ++i )
   {
      SCIP_Real stabredcost;
      SCIP_VAR* mastervar;

      mastervar = mastervars[i];
      stabredcost = stabredcosts[i];
      if( SCIPisPositive(scip_, stabredcost) )
      {
         boundval = SCIPvarGetLbLocal(mastervar);
      }
      else if( SCIPisNegative(scip_, stabredcost) )
      {
         boundval = SCIPvarGetUbLocal(mastervar);
      }
      else
         continue;

      if( SCIPisPositive(scip_, boundval) )
         dualobjval += boundval * stabredcost;

   }

   SCIPfreeBufferArray(scip_, &stabredcosts);

   *stabdualval = dualobjval;

   return SCIP_OKAY;
}

/** creates a new master variable corresponding to the given solution and problem */
SCIP_RETCODE ObjPricerGcg::createNewMasterVar(
   SCIP*                 scip,               /**< SCIP data structure */
   PricingType*          pricetype,          /**< type of pricing */
   SCIP_SOL*             sol,                /**< solution to compute reduced cost for */
   SCIP_VAR**            solvars,            /**< array of variables with non-zero value in the solution of the pricing problem */
   SCIP_Real*            solvals,            /**< array of values in the solution of the pricing problem for variables in array solvars*/
   int                   nsolvars,           /**< number of variables in array solvars */
   SCIP_Bool             solisray,           /**< is the solution a ray? */
   int                   prob,               /**< number of the pricing problem the solution belongs to */
   SCIP_Bool             force,              /**< should the given variable be added also if it has non-negative reduced cost? */
   SCIP_Bool*            added,              /**< pointer to store whether the variable was successfully added */
   SCIP_VAR**            addedvar            /**< pointer to store the created variable */
   )
{
   char varname[SCIP_MAXSTRLEN];

   SCIP_Real objcoeff;
   SCIP_VAR* newvar;

   SCIP_Real objvalue;
   SCIP_Real redcost;
   int i;

   assert(scip != NULL);
   assert(solvars != NULL || nsolvars == 0);
   assert(solvals != NULL || nsolvars == 0);
   assert(nsolvars >= 0);
   assert(pricerdata != NULL);
   assert((pricetype == NULL) == (force));
   assert((pricetype == NULL) == (sol == NULL));
   if( addedvar != NULL )
      *addedvar = NULL;

   objvalue = 0.0;
   redcost = 0.0;

   if( !force )
   {
      /* compute the objective function value of the solution */
      redcost = computeRedCost(pricetype, sol, solisray, prob, &objvalue);

      if( !SCIPisDualfeasNegative(scip, redcost) )
      {
         SCIPdebugMessage("var with redcost %g (objvalue=%g, dualsol=%g, ray=%u) was not added\n", redcost, objvalue, pricerdata->dualsolconv[prob], solisray);
         *added = FALSE;

         return SCIP_OKAY;
      }
      SCIPdebugMessage("found var with redcost %g (objvalue=%g, dualsol=%g, ray=%u)\n", redcost, objvalue, pricerdata->dualsolconv[prob], solisray);
   }
   else
   {
      SCIPdebugMessage("force var (objvalue=%g, dualsol=%g, ray=%u)\n",  objvalue, pricerdata->dualsolconv[prob], solisray);
   }

   *added = TRUE;

   /* compute objective coefficient of the variable */
   objcoeff = 0;
   for( i = 0; i < nsolvars; i++ )
   {
      SCIP_Real solval;
      solval = solvals[i];

      if( !SCIPisZero(scip, solval) )
      {
         SCIP_VAR* origvar;

         assert(GCGvarIsPricing(solvars[i]));
         origvar = GCGpricingVarGetOrigvars(solvars[i])[0];

         if( SCIPisZero(scip, SCIPvarGetObj(origvar)) )
            continue;

         /* original variable is linking variable --> directly transferred master variable got the full obj,
          * priced-in variables get no objective value for this origvar */
         if( GCGoriginalVarIsLinking(origvar) )
            continue;

         /* round solval if possible to avoid numerical troubles */
         if( SCIPvarIsIntegral(solvars[i]) && SCIPisFeasIntegral(scip, solval) )
            solval = SCIPround(scip, solval);

         /* add quota of original variable's objcoef to the master variable's coef */
         objcoeff += solval * SCIPvarGetObj(origvar);
      }
   }

   if( SCIPisInfinity(scip, objcoeff) )
   {
      SCIPwarningMessage(scip, "variable with infinite objective value found in pricing, change objective to SCIPinfinity()/2\n");
      objcoeff = SCIPinfinity(scip) / 2;
   }

   if( solisray )
   {
      (void) SCIPsnprintf(varname, SCIP_MAXSTRLEN, "r_%d_%d", prob, pricerdata->nraysprob[prob]);
      pricerdata->nraysprob[prob]++;
   }
   else
   {
      (void) SCIPsnprintf(varname, SCIP_MAXSTRLEN, "p_%d_%d", prob, pricerdata->npointsprob[prob]);
      pricerdata->npointsprob[prob]++;
   }

   SCIP_CALL( GCGcreateMasterVar(scip, origprob, pricerdata->pricingprobs[prob], &newvar, varname, objcoeff,
         pricerdata->vartype, solisray, prob, nsolvars, solvals, solvars, FALSE));

   SCIPvarMarkDeletable(newvar);

   SCIP_CALL( SCIPcatchVarEvent(scip, newvar, SCIP_EVENTTYPE_VARDELETED,
         pricerdata->eventhdlr, NULL, NULL) );


   /* add variable */
   if( !force )
   {
      SCIP_CALL( SCIPaddPricedVar(scip, newvar, pricerdata->dualsolconv[prob] - objvalue) );
   }
   else
   {
      SCIP_CALL( SCIPaddVar(scip, newvar) );
   }

   SCIP_CALL( addVariableToPricedvars(newvar) );
   SCIP_CALL( addVariableToMasterconstraints(newvar, prob, solvars, solvals, nsolvars) );
   SCIP_CALL( addVariableToMastercuts(newvar, prob, solvars, solvals, nsolvars) );

   /* add variable to convexity constraint */
   if( !solisray )
   {
      SCIP_CALL( SCIPaddCoefLinear(scip, GCGgetConvCons(origprob, prob), newvar, 1.0) );
   }

   if( addedvar != NULL )
   {
      *addedvar = newvar;
   }

   GCGupdateVarStatistics(scip, origprob, newvar, redcost);

#ifdef SCIP_STATISTIC
   if( SCIPgetCurrentNode(scip) == SCIPgetRootNode(scip) && pricetype != NULL && pricetype->getType() == GCG_PRICETYPE_REDCOST )
      GCGsetRootRedcostCall(newvar, pricerdata->nrootbounds );
#else
   GCGsetRootRedcostCall(newvar, -1LL);
#endif

   SCIPdebugMessage("Added variable <%s>\n", varname);

   return SCIP_OKAY;
}

/** creates a new master variable corresponding to the given gcg column */
SCIP_RETCODE ObjPricerGcg::createNewMasterVarFromGcgCol(
   SCIP*                 scip,               /**< SCIP data structure */
   PricingType*          pricetype,          /**< type of pricing */
   GCG_COL*              gcgcol,             /**< GCG column data structure */
   SCIP_Bool             force,              /**< should the given variable be added also if it has non-negative reduced cost? */
   SCIP_Bool*            added,              /**< pointer to store whether the variable was successfully added */
   SCIP_VAR**            addedvar,           /**< pointer to store the created variable */
   SCIP_Real             score               /**< score of column (or -1.0 if not specified) */
   )
{
   char varname[SCIP_MAXSTRLEN];

   SCIP_Real objcoeff;
   SCIP_VAR* newvar;

   SCIP_Real redcost;
   SCIP_Bool isray;
   int prob;
   int i;

   SCIP_VAR** solvars;
   SCIP_Real* solvals;
   int nsolvars;

   assert(scip != NULL);
   assert(pricerdata != NULL);
   assert(gcgcol != NULL);
   assert((pricetype == NULL) == (force));

   if( addedvar != NULL )
      *addedvar = NULL;

   redcost = 0.0;

   prob = GCGcolGetProbNr(gcgcol);
   isray = GCGcolIsRay(gcgcol);
   nsolvars = GCGcolGetNVars(gcgcol);
   solvars = GCGcolGetVars(gcgcol);
   solvals = GCGcolGetVals(gcgcol);

   if( !force )
   {
      /* compute the objective function value of the solution */
      redcost = GCGcolGetRedcost(gcgcol);

      if( !SCIPisDualfeasNegative(scip, redcost) )
      {
         SCIPdebugMessage("    var with redcost %g (dualsol=%g, ray=%u) was not added\n", redcost, pricerdata->dualsolconv[prob], isray);
         *added = FALSE;

         return SCIP_OKAY;
      }
      SCIPdebugMessage("    found var with redcost %g (dualsol=%g, ray=%u)\n", redcost, pricerdata->dualsolconv[prob], isray);
   }
   else
   {
      SCIPdebugMessage("    force var (dualsol=%g, ray=%u)\n", pricerdata->dualsolconv[prob], isray);
   }

   *added = TRUE;

   /* compute objective coefficient of the variable */
   objcoeff = 0.0;
   for( i = 0; i < nsolvars; i++ )
   {
      SCIP_Real solval;
      solval = solvals[i];

      if( !SCIPisZero(scip, solvals[i]) )
      {
         SCIP_VAR* origvar;

         assert(GCGvarIsPricing(solvars[i]));
         origvar = GCGpricingVarGetOrigvars(solvars[i])[0];
         solval = solvals[i];

         if( SCIPisZero(scip, SCIPvarGetObj(origvar)) )
            continue;

         /* original variable is linking variable --> directly transferred master variable got the full obj,
          * priced-in variables get no objective value for this origvar */
         if( GCGoriginalVarIsLinking(origvar) )
            continue;

         /* add quota of original variable's objcoef to the master variable's coef */
         objcoeff += solval * SCIPvarGetObj(origvar);
      }
   }

   if( SCIPisInfinity(scip, objcoeff) )
   {
      SCIPwarningMessage(scip, "variable with infinite objective value found in pricing, change objective to SCIPinfinity()/2\n");
      objcoeff = SCIPinfinity(scip) / 2;
   }

   if( isray )
   {
      (void) SCIPsnprintf(varname, SCIP_MAXSTRLEN, "r_%d_%d", prob, pricerdata->nraysprob[prob]);
      pricerdata->nraysprob[prob]++;
   }
   else
   {
      (void) SCIPsnprintf(varname, SCIP_MAXSTRLEN, "p_%d_%d", prob, pricerdata->npointsprob[prob]);
      pricerdata->npointsprob[prob]++;
   }

   SCIP_CALL( GCGcreateMasterVar(scip, GCGmasterGetOrigprob(scip), pricerdata->pricingprobs[prob], &newvar, varname, objcoeff,
         pricerdata->vartype, isray, prob, nsolvars, solvals, solvars, FALSE));

   SCIPvarMarkDeletable(newvar);

   SCIP_CALL( SCIPcatchVarEvent(scip, newvar, SCIP_EVENTTYPE_VARDELETED,
         pricerdata->eventhdlr, NULL, NULL) );

   if( SCIPisNegative(scip, score) )
      score = pricerdata->dualsolconv[prob] - objcoeff;

   /* add variable */
   if( !force )
   {
      SCIP_CALL( SCIPaddPricedVar(scip, newvar, score /* pricerdata->dualsolconv[prob] - objvalue */ ) );
   }
   else
   {
      SCIP_CALL( SCIPaddVar(scip, newvar) );
   }

   SCIP_CALL( addVariableToPricedvars(newvar) );
   SCIP_CALL( addVariableToMasterconstraintsFromGCGCol(newvar, gcgcol) );
   SCIP_CALL( addVariableToMastercutsFromGCGCol(newvar, gcgcol) );

   /* add variable to convexity constraint */
   if( !isray )
   {
      SCIP_CALL( SCIPaddCoefLinear(scip, GCGgetConvCons(origprob, prob), newvar, 1.0) );
   }

   if( addedvar != NULL )
   {
      *addedvar = newvar;
   }

   GCGupdateVarStatistics(scip, origprob, newvar, redcost);

#ifdef SCIP_STATISTIC
   if( SCIPgetCurrentNode(scip) == SCIPgetRootNode(scip) && pricetype->getType() == GCG_PRICETYPE_REDCOST )
      GCGsetRootRedcostCall(newvar, pricerdata->nrootbounds );
#else
   GCGsetRootRedcostCall(newvar, -1LL);
#endif

   SCIPdebugMessage("    added variable <%s>\n", varname);

   return SCIP_OKAY;
}

/**
 * check whether pricing can be aborted:
 * if objective value is always integral and the current node's current
 * lowerbound rounded up equals the current lp objective value rounded
 * up we don't need to continue pricing since the best possible feasible
 * solution must have at least this value
 */
SCIP_Bool  ObjPricerGcg::canPricingBeAborted() const
{
   SCIP_Bool canabort = FALSE;

   assert(pricerdata != NULL);

   if( pricerdata->abortpricingint && SCIPisObjIntegral(scip_)
      && SCIPisEQ(scip_, SCIPceil(scip_, SCIPgetNodeLowerbound(scip_, SCIPgetCurrentNode(scip_))), SCIPceil(scip_, SCIPgetLPObjval(scip_))) /* && SCIPgetNNodes(scip) > 1 ??????*/)
   {
      GCGpricerPrintInfo(scip_, pricerdata, "pricing aborted due to integral objective: node LB = %g, LP obj = %g\n",
            SCIPgetNodeLowerbound(scip_, SCIPgetCurrentNode(scip_)), SCIPgetLPObjval(scip_));

      canabort = TRUE;
   }

   if( !canabort && pricerdata->abortpricinggap > 0.0 )
   {
      SCIP_Real gap;
      gap = (SCIPgetLPObjval(scip_) - SCIPgetNodeLowerbound(scip_, SCIPgetCurrentNode(scip_)))/SCIPgetNodeLowerbound(scip_, SCIPgetCurrentNode(scip_));
      gap = ABS(gap);

      if( gap < pricerdata->abortpricinggap )
      {
         GCGpricerPrintInfo(scip_, pricerdata, "pricing aborted due to small gap: node LB = %g, LP obj = %g, gap = %g\n",
               SCIPgetNodeLowerbound(scip_, SCIPgetCurrentNode(scip_)), SCIPgetLPObjval(scip_), gap);

         canabort = TRUE;
      }
   }

   return canabort;
}


/** free pricing problems */
SCIP_RETCODE ObjPricerGcg::freePricingProblems()
{
   int j;
   assert(pricerdata != NULL);
   assert(pricerdata->pricingprobs != NULL);

   for( j = 0; j < pricerdata->npricingprobs; j++ )
      if( pricerdata->pricingprobs[j] != NULL
         && SCIPgetStage(pricerdata->pricingprobs[j]) > SCIP_STAGE_PROBLEM)
         {
            SCIP_CALL( SCIPstartClock(scip_, pricerdata->freeclock) );
            SCIP_CALL( SCIPfreeTransform(pricerdata->pricingprobs[j]) );
            SCIP_CALL( SCIPstopClock(scip_, pricerdata->freeclock) );
         }

   return SCIP_OKAY;
}

/** add bounds change from constraint from the pricing problem at this node
 * @note This method has to be threadsafe!
 */
SCIP_RETCODE ObjPricerGcg::addBranchingBoundChangesToPricing(
   int                   prob,               /**< index of pricing problem */
   SCIP_CONS*            branchcons          /**< branching constraints from which bound should applied */
) const
{
   GCG_BRANCHDATA* branchdata = GCGconsMasterbranchGetBranchdata(branchcons);
   GCG_COMPSEQUENCE* components = GCGbranchGenericBranchdataGetConsS(branchdata);
   int ncomponents = GCGbranchGenericBranchdataGetConsSsize(branchdata);
   int i;

   assert(pricerdata != NULL);

   for( i = 0; i < ncomponents; ++i)
   {
      SCIP_Real bound = components[i].bound;
      SCIP_VAR* var = GCGoriginalVarGetPricingVar(components[i].component);
      SCIP_Bool infeasible = FALSE;
      SCIP_Bool tightened = TRUE;

      if( components[i].sense == GCG_COMPSENSE_GE )
      {
         SCIP_CALL( SCIPtightenVarLb(pricerdata->pricingprobs[prob], var, bound, TRUE, &infeasible, &tightened));
         SCIPdebugMessage("Added <%s> >= %.2f\n", SCIPvarGetName(var), bound);
         assert(infeasible || tightened ||  SCIPisGE(pricerdata->pricingprobs[prob], SCIPvarGetLbGlobal(var), bound));
      }
      else
      {
         SCIP_CALL( SCIPtightenVarUb(pricerdata->pricingprobs[prob], var, bound-1, TRUE, &infeasible, &tightened));
         SCIPdebugMessage("Added <%s> <= %.2f\n", SCIPvarGetName(var), bound-1);
         assert(infeasible || tightened || SCIPisLE(pricerdata->pricingprobs[prob], SCIPvarGetUbGlobal(var), bound-1));
      }
   }

   return SCIP_OKAY;
}

/** check bounds change from constraint from the pricing problem at this node
 * @note This method has to be threadsafe!
 */
SCIP_RETCODE ObjPricerGcg::checkBranchingBoundChanges(
   int                   prob,               /**< index of pricing problem */
   SCIP_SOL*             sol,                /**< solution to check */
   SCIP_CONS*            branchcons,         /**< branching constraints from which bound should applied */
   SCIP_Bool*            feasible            /**< check whether the solution is feasible */
) const
{
   GCG_BRANCHDATA* branchdata = GCGconsMasterbranchGetBranchdata(branchcons);
   GCG_COMPSEQUENCE* components = GCGbranchGenericBranchdataGetConsS(branchdata);
   int ncomponents = GCGbranchGenericBranchdataGetConsSsize(branchdata);
   int i;

   assert(pricerdata != NULL);

   for( i = 0; i < ncomponents; ++i)
   {
      SCIP_VAR* pricingvar = GCGoriginalVarGetPricingVar(components[i].component);
      SCIP_Real val = SCIPgetSolVal(pricerdata->pricingprobs[prob], sol, pricingvar);

      if( components[i].sense == GCG_COMPSENSE_GE )
      {
         *feasible = SCIPisFeasGE(pricerdata->pricingprobs[prob], val, components[i].bound);
         SCIPdebugMessage("<%s> %.4f >= %.4f\n", SCIPvarGetName(pricingvar), val, components[i].bound);
      }
      else
      {
         *feasible = SCIPisFeasLT(pricerdata->pricingprobs[prob], val, components[i].bound);
         SCIPdebugMessage("<%s> %.4f < %.4f\n", SCIPvarGetName(pricingvar), val, components[i].bound);
      }
      if( !*feasible )
         break;
   }

   return SCIP_OKAY;
}


/** check bounds change from constraint from the pricing problem at this node
 * @note This method has to be threadsafe!
 */
SCIP_RETCODE ObjPricerGcg::checkBranchingBoundChangesGcgCol(
   GCG_COL*              gcgcol,             /**< gcg column to check */
   SCIP_CONS*            branchcons,         /**< branching constraints from which bound should applied */
   SCIP_Bool*            feasible            /**< check whether the solution is feasible */
) const
{
   int prob = GCGcolGetProbNr(gcgcol);
   GCG_BRANCHDATA* branchdata = GCGconsMasterbranchGetBranchdata(branchcons);
   GCG_COMPSEQUENCE* components = GCGbranchGenericBranchdataGetConsS(branchdata);
   int ncomponents = GCGbranchGenericBranchdataGetConsSsize(branchdata);
   int i;

   assert(pricerdata != NULL);

   for( i = 0; i < ncomponents; ++i)
   {
      SCIP_VAR* pricingvar = GCGoriginalVarGetPricingVar(components[i].component);

      SCIP_Real val = GCGcolGetSolVal(pricerdata->pricingprobs[prob], gcgcol, pricingvar);

      if( components[i].sense == GCG_COMPSENSE_GE )
      {
         *feasible = SCIPisFeasGE(pricerdata->pricingprobs[prob], val, components[i].bound);
/*         SCIPdebugMessage("<%s> %.4f >= %.4f\n", SCIPvarGetName(pricingvar), val, components[i].bound); */
      }
      else
      {
         *feasible = SCIPisFeasLT(pricerdata->pricingprobs[prob], val, components[i].bound);
/*         SCIPdebugMessage("<%s> %.4f < %.4f\n", SCIPvarGetName(pricingvar), val, components[i].bound); */
      }
      if( !*feasible )
         break;
   }

   return SCIP_OKAY;
}


/** perform a pricing job, i.e. apply the corresponding solver to the pricing problem
 * @note This method has to be threadsafe!
 */
SCIP_RETCODE ObjPricerGcg::performPricingjob(
   GCG_PRICINGJOB*       pricingjob,         /**< pricing job */
   PricingType*          pricetype,          /**< type of pricing: reduced cost or Farkas */
   GCG_PRICINGSTATUS*    status,             /**< pointer to store pricing status */
   SCIP_Real*            lowerbound          /**< pointer to store the obtained lower bound */
   )
{
   GCG_PRICINGPROB* pricingprob;
   SCIP* pricingscip;
   int probnr;
   GCG_SOLVER* solver;
   SCIP_Bool heuristic;
   SCIP_Bool solved;
   SCIP_RETCODE retcode;

   pricingprob = GCGpricingjobGetPricingprob(pricingjob);
   assert(pricingprob != NULL);

   probnr = GCGpricingprobGetProbnr(pricingprob);

   pricingscip = GCGpricingprobGetPricingscip(pricingprob);
   assert(pricingscip != NULL);

   solver = GCGpricingjobGetSolver(pricingjob);
   assert(solver != NULL);

   heuristic = GCGpricingjobIsHeuristic(pricingjob);

   // @todo: this should be done by the pricing solvers
   #pragma omp critical (limits)
   {
      retcode = setPricingProblemMemorylimit(pricingscip);
   }
   SCIP_CALL( retcode );

   /* add the next generic branching constraint if necessary */
   if( !GCGpricingprobBranchconsIsAdded(pricingprob) )
   {
      SCIP_CONS** branchconss;
      int nbranchconss;
      int branchconsidx;

      int i;

      GCGpricingprobGetGenericBranchData(pricingprob, &branchconss, NULL, &nbranchconss);
      branchconsidx = GCGpricingprobGetBranchconsIdx(pricingprob);
      assert(branchconsidx >= 0);
      assert(branchconsidx < nbranchconss);

      SCIP_CALL( SCIPfreeTransform(pricingscip) );

      SCIPdebugMessage("*** Apply generic branching bound change of depth %d\n", -branchconsidx);
      SCIP_CALL( SCIPtransformProb(pricingscip) );
      SCIP_CALL( addBranchingBoundChangesToPricing(probnr, branchconss[branchconsidx]) );

      for( i = 0; i < pricerdata->nsolvers; ++i )
      {
         SCIP_CALL( GCGsolverUpdate(pricingscip, pricerdata->solvers[i], probnr, FALSE, TRUE, FALSE) );
      }

      GCGpricingprobMarkBranchconsAdded(pricingprob);
   }

   SCIP_CALL( GCGsolverSolve(scip_, pricingscip, solver, pricetype->getType() == GCG_PRICETYPE_REDCOST,
         heuristic, probnr, pricerdata->dualsolconv[probnr], lowerbound, status, &solved) );

   if( !solved )
      return SCIP_OKAY;

   if( !heuristic )
   {
      #pragma omp atomic
      pricerdata->solvedsubmipsoptimal++;
   }
   else
   {
      #pragma omp atomic
      pricerdata->solvedsubmipsheur++;
   }

#ifdef SCIP_STATISTIC
   #pragma omp critical (collectstats)
   GCGpricerCollectStatistic(pricerdata, pricetype->getType(), probnr,
      SCIPgetSolvingTime(pricingscip));
#endif
   /* @todo: This should actually be a MIP solver specific statistic */
   if( SCIPgetStage(pricingscip) > SCIP_STAGE_SOLVING )
   {
      #pragma omp atomic
      pricerdata->pricingiters += SCIPgetNLPIterations(pricingscip);
   }

   return SCIP_OKAY;
}


/* Compute difference of two dual solutions */
SCIP_RETCODE ObjPricerGcg::computeDualDiff(
   SCIP_Real**          dualvals1,           /**< array of dual values for each pricing problem */
   SCIP_Real*           dualconv1,           /**< array of dual solutions for the convexity constraints  */
   SCIP_Real**          dualvals2,           /**< array of dual values for each pricing problem */
   SCIP_Real*           dualconv2,           /**< array of dual solutions for the convexity constraints  */
   SCIP_Real*           dualdiff             /**< pointer to store difference of duals solutions */
   )
{
   int i;
   int j;
   int nprobvars;

   *dualdiff = 0.0;
   for( i = 0; i < pricerdata->npricingprobs; i++ )
   {
      if( pricerdata->pricingprobs[i] == NULL )
         continue;

      nprobvars = SCIPgetNVars(pricerdata->pricingprobs[i]);

      for( j = 0; j < nprobvars; j++ )
      {
         *dualdiff += SQR(dualvals1[i][j] - dualvals2[i][j]);
      }

      *dualdiff += SQR(dualconv1[i] - dualconv2[i]);

   }
   *dualdiff = SQRT(ABS(*dualdiff));

   return SCIP_OKAY;
}

/** the pricing loop: solve the pricing problems */
SCIP_RETCODE ObjPricerGcg::pricingLoop(
   PricingType*          pricetype,          /**< type of pricing */
   SCIP_RESULT*          result,             /**< result pointer */
   int*                  pnfoundvars,        /**< pointer to store number of found variables */
   SCIP_Real*            lowerbound,         /**< pointer to store lowerbound obtained due to lagrange bound */
   SCIP_Bool*            bestredcostvalid    /**< pointer to store if bestredcost are valid (pp solvedoptimal) */
   )
{
   GCG_PRICINGJOB* pricingjob = NULL;
   GCG_COL** bestcols = NULL;
   SCIP_LPI* lpi = NULL;
   SCIP_Real* bestobjvals = NULL;
   SCIP_Real bestredcost;
   SCIP_Real beststabobj;
   SCIP_RETCODE retcode;
   SCIP_Bool infeasible;
   SCIP_Bool nextchunk = TRUE;
   SCIP_Bool stabilized;
   SCIP_Bool colpoolupdated;
   SCIP_Bool enableppcuts;
   SCIP_Bool enablestab;
   int nsuccessfulprobs;
   int maxniters;
   int niters;
   int i;
   int j;
   int nfoundvars;
   SCIP_Bool optimal;
   bool probingnode;

#ifdef SCIP_STATISTIC
   SCIP_Real** olddualvalues;
   SCIP_Real* olddualconv;

   int nprobvars;
   int nstabrounds;
   SCIP_Real pricingtime;
#endif

   assert(pricerdata != NULL);
   assert(stabilization != NULL);
   assert(farkaspricing != NULL);
   assert(reducedcostpricing != NULL);

   assert(result != NULL);
   assert(pnfoundvars != NULL);

   /* initializations */
   retcode = SCIP_OKAY;
   *pnfoundvars = 0;
   nfoundvars = 0;
   infeasible = FALSE;
   stabilized = FALSE;
   optimal = FALSE;
   if( lowerbound != NULL )
      *lowerbound = -SCIPinfinity(scip_);

   maxniters = pricingcontroller->getMaxNIters();

   // disable colpool while probing mode is active (can be removed after a feasibility check (#586) is implemented)
   probingnode = (SCIPnodeGetType(SCIPgetCurrentNode(scip_)) == SCIP_NODETYPE_PROBINGNODE);

   SCIP_CALL( SCIPgetLPI(scip_, &lpi) );

   /* check preliminary conditions for stabilization */
   enablestab = pricerdata->stabilization
      && (SCIPgetCurrentNode(scip_) == SCIPgetRootNode(scip_) || pricerdata->stabilizationtree)
      && (pricerdata->stabilization && pricetype->getType() == GCG_PRICETYPE_REDCOST)
      && !GCGisBranchruleGeneric(GCGconsMasterbranchGetBranchrule(GCGconsMasterbranchGetActiveCons(scip_)));

   /* allocate memory */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip_, &bestcols, pricerdata->npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip_, &bestobjvals, pricerdata->npricingprobs) );

   enableppcuts = FALSE;
   SCIP_CALL( SCIPgetBoolParam(GCGmasterGetOrigprob(scip_), "sepa/basis/enableppcuts", &enableppcuts) );
   /* set parameters for adding pool cuts to separation basis */
   if( enableppcuts && SCIPgetCurrentNode(scip_) != SCIPgetRootNode(scip_) )
   {
      for( i = 0; i < pricerdata->npricingprobs; i++ )
      {
         if( GCGisPricingprobRelevant(origprob, i) )
         {
            SCIP_CALL( SCIPsetIntParam(pricerdata->pricingprobs[i], "branching/pscost/priority", 2000) );
            SCIP_CALL( SCIPsetIntParam(pricerdata->pricingprobs[i], "propagating/maxroundsroot", 1000) );
            SCIP_CALL( SCIPsetPresolving(pricerdata->pricingprobs[i], SCIP_PARAMSETTING_DEFAULT, TRUE) );
         }
      }
   }

#ifdef _OPENMP
   if( threads > 0 )
      omp_set_num_threads(MIN(threads, GCGgetNRelPricingprobs(origprob)));
   else
      omp_set_num_threads(GCGgetNRelPricingprobs(origprob));
#endif

   /* todo: We avoid checking for feasibility of the columns using this hack */
   if( pricerdata->usecolpool && !probingnode )
      SCIP_CALL( GCGcolpoolUpdateNode(colpool) );

   colpoolupdated = FALSE;

#ifdef SCIP_STATISTIC
   if( pricerdata->nroundsredcost > 0 && pricetype->getType() == GCG_PRICETYPE_REDCOST )
   {
      SCIP_CALL( SCIPallocBufferArray(scip_, &olddualvalues, pricerdata->npricingprobs) );
      SCIP_CALL( SCIPallocBufferArray(scip_, &olddualconv, pricerdata->npricingprobs) );

      for( i = 0; i < pricerdata->npricingprobs; i++ )
      {
         if( pricerdata->pricingprobs[i] == NULL )
            continue;

         nprobvars = SCIPgetNVars(pricerdata->pricingprobs[i]);

         olddualconv[i] = pricerdata->dualsolconv[i];
         SCIP_CALL( SCIPallocBufferArray(scip_, &(olddualvalues[i]), nprobvars) );

         for( j = 0; j < nprobvars; j++ )
            olddualvalues[i][j] = pricerdata->realdualvalues[i][j];
      }
   }
#endif

#ifdef SCIP_STATISTIC
   SCIPstatisticMessage("New pr, node %" SCIP_LONGINT_FORMAT "\n", SCIPgetNNodes(scip_));
   SCIPstatisticMessage("MLP t: %g\n", SCIPgetClockTime(scip_, scip_->stat->primallptime) + SCIPgetClockTime(scip_, scip_->stat->duallptime));
   nstabrounds = 0;
#endif

   SCIPdebugMessage("***** New pricing round at node %" SCIP_LONGINT_FORMAT " (depth = %d), maxniters = %d\n",
      SCIPgetNNodes(scip_), SCIPnodeGetDepth(SCIPgetCurrentNode(scip_)), maxniters);

   /* stabilization loop */
   do
   {
      optimal = FALSE;
#ifndef NDEBUG
      if( nextchunk )
      {
         SCIPdebugMessage("*** get next chunk of pricing problems\n");
      }
#endif

      nsuccessfulprobs = 0;
      *bestredcostvalid = isMasterLPOptimal() && !GCGisBranchruleGeneric(GCGconsMasterbranchGetBranchrule(GCGconsMasterbranchGetActiveCons(scip_)));
      nextchunk = FALSE;

      if( stabilized )
      {
         SCIPdebugMessage("****************************** Mispricing iteration ******************************\n");
#ifdef SCIP_STATISTIC
         ++nstabrounds;
         SCIPstatisticMessage("Sr %d\n", nstabrounds);
#endif
      }

      /* initialize stabilization parameters if we are at a new node */
      if( enablestab )
      {
         stabilization->updateNode();
         SCIP_CALL( stabilization->updateHybrid() );
      }

      stabilized = enablestab && stabilization->isStabilized();

      /* set the objective function */
      SCIP_CALL( freePricingProblems() );
      SCIP_CALL( setPricingObjs(pricetype, stabilized) );

      /* call update method of pricing solvers to update objectives;
       * also, let them update their bounds since the transformed pricing problems
       * might have contained generic branching bounds before freeing
       */
      for( i = 0; i < pricerdata->nsolvers; ++i  )
      {
         for( j = 0; j < pricerdata->npricingprobs; ++j )
         {
            if( pricerdata->pricingprobs[j] != NULL )
            {
               SCIP_CALL( GCGsolverUpdate(pricerdata->pricingprobs[j], pricerdata->solvers[i], j, TRUE, TRUE, FALSE) );
            }
         }
      }

      /* todo: do this inside the updateRedcostColumnPool */
      if( !colpoolupdated && pricerdata->usecolpool && !probingnode )
      {
         /* update reduced cost of cols in colpool */
         SCIP_CALL( GCGcolpoolUpdateRedcost(colpool) );

         colpoolupdated = TRUE;
      }

      /* check if colpool already contains columns with negative reduced cost */
      if( pricerdata->usecolpool && !probingnode )
      {
         SCIP_Bool foundvarscolpool;
         int oldnfoundcols;

         foundvarscolpool = FALSE;
         oldnfoundcols = GCGpricestoreGetNCols(pricestore);

         SCIP_CALL( GCGcolpoolPrice(scip_, colpool, pricestore, NULL, &foundvarscolpool) );
         SCIPstatisticMessage("cp: %d impr c\n", GCGpricestoreGetNCols(pricestore) - oldnfoundcols);

         if( foundvarscolpool )
         {
            SCIPdebugMessage("*** Found column(s) with negative reduced cost in column pool\n");
            assert(GCGpricestoreGetNCols(pricestore) > 0);
            break;
         }
      }

      pricingcontroller->setupPriorityQueue(pricerdata->dualsolconv);

      /* actual pricing loop: perform the pricing jobs until none are left or an abortion criterion is met */
      #pragma omp parallel for ordered firstprivate(pricingjob) shared(retcode, pricetype, nfoundvars, nsuccessfulprobs) schedule(static,1)
      for( niters = 0; niters < maxniters; ++niters )
      {
         GCG_PRICINGPROB* pricingprob;
         GCG_PRICINGSTATUS status;
         SCIP_Real problowerbound;
         SCIP_RETCODE private_retcode;

         int oldnimpcols = GCGpricestoreGetNEfficaciousCols(pricestore);

         #pragma omp flush(retcode)
         if( retcode != SCIP_OKAY )
            continue;

         /* retrieve the next pricing job from the queue */
         #pragma omp critical (update)
         {
            pricingjob = pricingcontroller->getNextPricingjob();
         }
         if( pricingjob == NULL )
            continue;

         #pragma omp flush(nfoundvars, nsuccessfulprobs)
         if( (pricingcontroller->canPricingloopBeAborted(pricetype, nfoundvars, nsuccessfulprobs) || infeasible) && !stabilized )
         {
            SCIPdebugMessage("*** Abort pricing loop, infeasible = %u, stabilized = %u\n", infeasible, stabilized);
            continue;
         }

         /* initializations */
         pricingprob = GCGpricingjobGetPricingprob(pricingjob);
         status = GCG_PRICINGSTATUS_UNKNOWN;
         problowerbound = -SCIPinfinity(scip_);

         SCIPdebugMessage("*** Solve pricing problem %d, solver <%s>, stabilized = %u, %s\n",
            GCGpricingprobGetProbnr(pricingprob), GCGsolverGetName(GCGpricingjobGetSolver(pricingjob)), stabilized,
            GCGpricingjobIsHeuristic(pricingjob) ? "heuristic" : "exact");

         /* @todo: this should be done by the pricing solvers */
         #pragma omp critical (limits)
         SCIP_CALL_ABORT( pricingcontroller->setPricingjobTimelimit(pricingjob) );

#ifdef SCIP_STATISTIC
         /* @todo: this can interfere with parallelization */
         pricingtime = pricetype->getClockTime();
#endif

         /* solve the pricing problem */
         private_retcode = performPricingjob(pricingjob, pricetype, &status, &problowerbound);

#ifdef SCIP_STATISTIC
         pricingtime = pricetype->getClockTime() - pricingtime;
#endif

         SCIPdebugMessage("  -> status: %d\n", status);
         SCIPdebugMessage("  -> problowerbound: %.4g\n", problowerbound);

         /* update pricing problem results, store columns */
         #pragma omp critical (update)
         {
            pricingcontroller->updatePricingprob(pricingprob, status, problowerbound, GCGpricestoreGetNEfficaciousCols(pricestore) - oldnimpcols);
         }

         /* update solving statistics, needed for checking the abortion criterion */
         #pragma omp ordered
         {
            #pragma omp critical (retcode)
            retcode = private_retcode; // @todo: handle return code correctly

            #pragma omp atomic
            nfoundvars += GCGpricestoreGetNEfficaciousCols(pricestore) - oldnimpcols;

            if( oldnimpcols == 0 && GCGpricingprobGetNImpCols(pricingprob) > 0 )
            {
               #pragma omp atomic
               ++nsuccessfulprobs;
            }

#ifdef SCIP_STATISTIC
            if( status != GCG_PRICINGSTATUS_NOTAPPLICABLE )
            {
               SCIPstatisticMessage("P p %d : %d in %g\n",
                  GCGpricingprobGetProbnr(pricingprob), GCGpricestoreGetNEfficaciousCols(pricestore) - oldnimpcols, pricingtime);
            }
#endif
         }

         #pragma omp critical (update)
         {
            pricingcontroller->evaluatePricingjob(pricingjob, status);
         }
      }

      SCIP_CALL( retcode );

      /* collect results from all performed pricing jobs */
      getBestCols(bestcols);
      pricingcontroller->collectResults(bestcols, &infeasible, &optimal, bestobjvals, &beststabobj, &bestredcost, bestredcostvalid);

      if( infeasible )
         break;

      SCIPdebugMessage("optimal = %u, bestredcostvalid = %u, stabilized = %u\n", optimal, *bestredcostvalid, stabilized);

      /* update stabilization information and lower bound */
      if( pricetype->getType() == GCG_PRICETYPE_REDCOST )
      {
         SCIP_Real lowerboundcandidate;
         SCIP_Real stabdualval = 0.0;

         assert(lowerbound != NULL);

         SCIP_CALL( getStabilizedDualObjectiveValue(pricetype, &stabdualval, stabilized) );

         lowerboundcandidate = stabdualval + beststabobj;

         SCIPdebugMessage("lpobjval = %.8g, bestredcost = %.8g, stabdualval = %.8g, beststabobj = %.8g\n",
            SCIPgetLPObjval(scip_), bestredcost, stabdualval, beststabobj);
         SCIPdebugMessage("lowerboundcandidate = %.8g\n", lowerboundcandidate);

         assert(!optimal || !*bestredcostvalid || stabilized || SCIPisDualfeasEQ(scip_, SCIPgetLPObjval(scip_) + bestredcost, lowerboundcandidate));

         if( enablestab )
         {
            SCIP_Real beststabredcost = beststabobj - getDualconvsum(bestcols);

            SCIPdebugMessage("beststabredcost = %.8g\n", beststabredcost);

            /* If all pricing problems have been solved to optimality, update subgradient product and stability center */
            if( optimal )
            {
               SCIPdebugMessage("update subgradient product and stability center\n");

               /* update subgradient product before a potential change of the stability center */
               SCIP_CALL( stabilization->updateSubgradientProduct(bestcols) );
               SCIP_CALL( stabilization->updateStabilityCenter(lowerboundcandidate, bestobjvals, bestcols) );
            }

            /* activate or deactivate mispricing schedule, depending on whether improving columns have been found */
            if( nfoundvars == 0 )
            {
               if( stabilized )
               {
                  SCIPdebugMessage("enabling mispricing schedule\n");
                  stabilization->activateMispricingSchedule();
                  stabilization->updateAlphaMisprice();
               }
               else
                  stabilization->disablingMispricingSchedule();
            }
            else if( *bestredcostvalid && SCIPisDualfeasNegative(scip_, beststabredcost) )
            {
               if( stabilization->isInMispricingSchedule() )
                  stabilization->disablingMispricingSchedule();
               stabilization->updateAlpha();
            }
         }

         if( *bestredcostvalid )
         {
            SCIP_Bool enableppobjcg;

            *lowerbound = MAX(*lowerbound, lowerboundcandidate);

            /* add cuts based on the latest pricing problem objective to the original problem */
            SCIP_CALL( SCIPgetBoolParam(GCGmasterGetOrigprob(scip_), "sepa/basis/enableppobjcg", &enableppobjcg) );
            if( enableppobjcg && SCIPgetCurrentNode(scip_) == SCIPgetRootNode(scip_) )
            {
               for( i = 0; i < pricerdata->npricingprobs; ++i )
               {
                  if( !GCGisPricingprobRelevant(GCGmasterGetOrigprob(scip_), i) )
                     continue;

                  SCIP_CALL( SCIPsepaBasisAddPPObjConss(scip_, i, bestobjvals[i], TRUE) );
               }
            }
         }
      }

      /* if no column has negative reduced cost, consider the next chunk of pricing problems */
      if( nfoundvars == 0 && !stabilized )
         nextchunk = pricingcontroller->checkNextChunk();

      /** @todo perhaps solve remaining pricing problems, if only few left? */
      /** @todo solve all pricing problems all k iterations? */
   }
   while( nextchunk || (stabilized && nfoundvars == 0) );

   SCIPdebugMessage("*** Pricing loop finished, found %d improving columns.\n", nfoundvars);

   /* Add new columns as variables to the master problem or move them to the column pool */
   SCIP_CALL( GCGpricestoreApplyCols(pricestore, colpool, (pricerdata->usecolpool && !probingnode), &nfoundvars) );

   SCIPdebugMessage("Added %d new variables.\n", nfoundvars);

   SCIPfreeBlockMemoryArray(scip_, &bestobjvals, pricerdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip_, &bestcols, pricerdata->npricingprobs);

   enableppcuts = FALSE;
   SCIP_CALL( SCIPgetBoolParam(GCGmasterGetOrigprob(scip_), "sepa/basis/enableppcuts", &enableppcuts) );

   /* add pool cuts to sepa basis */
   if( enableppcuts && SCIPgetCurrentNode(scip_) == SCIPgetRootNode(scip_) )
   {
      for( j = 0; j < pricerdata->npricingprobs; j++ )
      {
         if( pricerdata->pricingprobs[j] != NULL
            && SCIPgetStage(pricerdata->pricingprobs[j]) >= SCIP_STAGE_SOLVING )
         {
            SCIP_CUT** cuts;
            int ncuts;

            ncuts = SCIPgetNPoolCuts(pricerdata->pricingprobs[j]);
            cuts = SCIPgetPoolCuts(pricerdata->pricingprobs[j]);

            for( i = 0; i < ncuts; ++i )
            {
               SCIP_ROW* row;
               row = SCIPcutGetRow(cuts[i]);

               if( !SCIProwIsLocal(row) && SCIProwGetRank(row) >= 1 && nfoundvars == 0 )
                  SCIP_CALL( GCGsepaBasisAddPricingCut(scip_, j, row) );
            }
         }
      }
   }

#ifdef SCIP_STATISTIC
   SCIPstatisticMessage("MLP t: %g\n", SCIPgetClockTime(scip_, scip_->stat->primallptime) + SCIPgetClockTime(scip_, scip_->stat->duallptime));
#endif

   /* free the pricingproblems if they exist and need to be freed */
   // @todo: actually, only the transformed problems are freed
   SCIP_CALL( freePricingProblems() );
   *pnfoundvars = nfoundvars;

   if( infeasible )
      *result = SCIP_SUCCESS;
   else if( *pnfoundvars > 0 || optimal )
      *result = SCIP_SUCCESS;
   else
      *result = SCIP_DIDNOTRUN;

#ifdef SCIP_STATISTIC
   if( pricerdata->nroundsredcost > 0 && pricetype->getType() == GCG_PRICETYPE_REDCOST )
   {
      if( pricerdata->nrootbounds != pricerdata->dualdiffround )
      {
         SCIP_Real dualdiff;
         SCIP_CALL( computeDualDiff(olddualvalues, olddualconv, pricerdata->realdualvalues, pricerdata->dualsolconv, &dualdiff) );
         pricerdata->dualdiffround = pricerdata->nrootbounds;
         pricerdata->dualdiff = dualdiff;
      }

      for( i = pricerdata->npricingprobs - 1; i >= 0; i-- )
      {
         if( pricerdata->pricingprobs[i] == NULL )
            continue;
         SCIPfreeBufferArray(scip_, &(olddualvalues[i]));
      }
      SCIPfreeBufferArray(scip_, &olddualconv);
      SCIPfreeBufferArray(scip_, &olddualvalues);

   }
   else if( pricerdata->nrootbounds != pricerdata->dualdiffround )
   {
      pricerdata->dualdiff = 0.0;
   }
#endif

   return SCIP_OKAY;
}

/** set pricing objectives */
extern "C"
SCIP_RETCODE GCGsetPricingObjs(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            dualsolconv         /**< array of dual solutions corresponding to convexity constraints */
)
{
  ObjPricerGcg* pricer;
  SCIP_Bool stabilizationtmp;
  int i;
  int j;

  assert(scip != NULL);

  pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
  assert(pricer != NULL);

  stabilizationtmp = pricer->pricerdata->stabilization;

  pricer->pricerdata->stabilization = FALSE;

  SCIP_CALL( pricer->setPricingObjs(pricer->getReducedCostPricingNonConst(), FALSE) );

   /* notify the pricing solvers that the objective values have changed */
   for( i = 0; i < pricer->pricerdata->nsolvers; ++i  )
   {
      for( j = 0; j < pricer->pricerdata->npricingprobs; ++j )
      {
         if( pricer->pricerdata->pricingprobs[j] != NULL )
         {
            SCIP_CALL( GCGsolverUpdate(pricer->pricerdata->pricingprobs[j], pricer->pricerdata->solvers[i], j, TRUE, FALSE, FALSE) );
         }
      }
   }

  if(dualsolconv != NULL)
  {
     for(i = 0; i < pricer->pricerdata->npricingprobs; ++i)
     {
        dualsolconv[i] = pricer->pricerdata->dualsolconv[i];
     }
  }
  pricer->pricerdata->stabilization = stabilizationtmp;

  return SCIP_OKAY;
}

/** creates a new master variable corresponding to the given gcg column */
extern "C"
SCIP_RETCODE GCGcreateNewMasterVarFromGcgCol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             infarkas,           /**< in Farkas pricing? */
   GCG_COL*              gcgcol,             /**< GCG column data structure */
   SCIP_Bool             force,              /**< should the given variable be added also if it has non-negative reduced cost? */
   SCIP_Bool*            added,              /**< pointer to store whether the variable was successfully added */
   SCIP_VAR**            addedvar,           /**< pointer to store the created variable */
   SCIP_Real             score               /**< score of column (or -1.0 if not specified) */
)
{
  ObjPricerGcg* pricer;
  PricingType* pricetype;

  assert(scip != NULL);

  pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
  assert(pricer != NULL);

  if( infarkas )
     pricetype = pricer->getFarkasPricingNonConst();
  else
     pricetype = pricer->getReducedCostPricingNonConst();


  SCIP_CALL( pricer->createNewMasterVarFromGcgCol(scip, pricetype, gcgcol, force, added, addedvar, score) );

  return SCIP_OKAY;
}

/** compute master and cut coefficients of column */
extern "C"
SCIP_RETCODE GCGcomputeColMastercoefs(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_COL*              gcgcol              /**< GCG column data structure */
   )
{
   ObjPricerGcg* pricer;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricer->computeColMastercoefs(gcgcol);
   pricer->computeColMastercuts(gcgcol);

   return SCIP_OKAY;

}
/** computes the reduced cost of a column */
extern "C"
SCIP_Real GCGcomputeRedCostGcgCol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Bool             infarkas,           /**< in Farkas pricing? */
   GCG_COL*              gcgcol,             /**< gcg column to compute reduced cost for */
   SCIP_Real*            objvalptr           /**< pointer to store the computed objective value */
   )
{
   ObjPricerGcg* pricer;
   PricingType* pricetype;
   SCIP_Real redcost;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   if( infarkas )
      pricetype = pricer->getFarkasPricingNonConst();
   else
      pricetype = pricer->getReducedCostPricingNonConst();

   redcost = pricer->computeRedCostGcgCol(pricetype, gcgcol, objvalptr);

   return redcost;
}

/** performs the pricing routine, gets the type of pricing that should be done: farkas or redcost pricing */
SCIP_RETCODE ObjPricerGcg::priceNewVariables(
   PricingType*          pricetype,          /**< type of the pricing */
   SCIP_RESULT*          result,             /**< result pointer */
   SCIP_Real*            lowerbound          /**< lowerbound pointer */
   )
{
   int nfoundvars;
   SCIP_Bool bestredcostvalid;

   assert(result != NULL);
   assert(lowerbound != NULL || pricetype->getType() == GCG_PRICETYPE_FARKAS);
   assert(pricerdata != NULL);

   if( lowerbound != NULL )
      *lowerbound = -SCIPinfinity(scip_);

   GCGpricerPrintInfo(scip_, pricerdata, "nvars = %d, current LP objval = %g, time = %f, node = %lld\n",
         SCIPgetNVars(scip_), SCIPgetLPObjval(scip_), SCIPgetSolvingTime(scip_), SCIPgetNNodes(scip_));

   if( pricetype->getType() == GCG_PRICETYPE_REDCOST )
   {
      assert(result != NULL);

      /* terminate early, if applicable */
      if( canPricingBeAborted() )
      {
         *result = SCIP_DIDNOTRUN;
         return SCIP_OKAY;
      }
   }

   *result = SCIP_SUCCESS;

   pricingtype = pricetype;
   pricetype->incCalls();

   pricerdata->calls++;
   nfoundvars = 0;

   bestredcostvalid = TRUE;

   /* If pricing is performed for the first time at this node, update variable bounds and pricing constraints */
   if( pricerdata->newnode )
   {
      int i;
      int j;

      for( i = 0; i < pricerdata->nsolvers; ++i  )
      {
         for( j = 0; j < pricerdata->npricingprobs; ++j )
         {
            if( pricerdata->pricingprobs[j] != NULL )
            {
               SCIP_CALL( GCGsolverUpdate(pricerdata->pricingprobs[j], pricerdata->solvers[i], j, FALSE, TRUE, TRUE) );
            }
         }
      }
   }

   pricingcontroller->initPricing(pricetype);

   SCIP_CALL( pricingLoop(pricetype, result, &nfoundvars, lowerbound, &bestredcostvalid) );

   if( pricetype->getType() == GCG_PRICETYPE_REDCOST && bestredcostvalid )
   {
      assert(lowerbound != NULL);
      GCGpricerPrintInfo(scip_, pricerdata, "lower bound = %g\n", *lowerbound);

      pricingcontroller->resetEagerage();
   }


   SCIPdebugMessage("%s pricing: found %d new vars\n", (pricetype->getType() == GCG_PRICETYPE_REDCOST ? "Redcost" : "Farkas"), nfoundvars);

   if( GCGisRootNode(scip_) && pricetype->getType() == GCG_PRICETYPE_REDCOST && pricetype->getCalls() > 0 )
   {
      double degeneracy = 0.0;

      /* only compute degeneracy if current solution is basic */
      if( SCIPisLPSolBasic(scip_) )
         SCIP_CALL( computeCurrentDegeneracy(&degeneracy) );

      pricerdata->rootnodedegeneracy = degeneracy;

      /* Complicated calculation for numerical stability:
       *     E[\sum_{i=1}^n x_i] = (E[\sum_{i=1}^{n-1} x_i]*(n-1) + x_n)/n
       *     E[\sum_{i=1}^n x_i] = E[\sum_{i=1}^{n-1} x_i]*(n-1)/n + x_n/n
       * <=> E[\sum_{i=1}^n x_i] = E[\sum_{i=1}^{n-1} x_i]-E[\sum_{i=1}^{n-1} x_i]/n + x_n/n
       * <=> E_n = E_{n-1} - E_{n-1}/n + x_n/n
       * <=> E -= E/n - x_n/n
       */
      ++pricerdata->ndegeneracycalcs;
      pricerdata->avgrootnodedegeneracy -= (pricerdata->avgrootnodedegeneracy/(pricerdata->ndegeneracycalcs) - degeneracy/(pricerdata->ndegeneracycalcs));
   }

   pricingcontroller->exitPricing();

   pricingtype = NULL;

   return SCIP_OKAY;
}

/*
 * Callback methods of variable pricer
 */

 ObjPricerGcg::ObjPricerGcg(
    SCIP*              scip,               /**< SCIP data structure */
    SCIP*              origscip,           /**< SCIP data structure of original problem */
    const char*        name,               /**< name of variable pricer */
    const char*        desc,               /**< description of variable pricer */
    int                priority,           /**< priority of the variable pricer */
    SCIP_Bool          delay,
    SCIP_PRICERDATA*   p_pricerdata
    ) : ObjPricer(scip, name, desc, priority, delay), colpool(NULL), pricestore(NULL), reducedcostpricing(NULL), farkaspricing(NULL), pricingtype(NULL), pricingcontroller(NULL), stabilization(NULL)
 {

    assert(origscip!= NULL);
    pricerdata = p_pricerdata;
    origprob = origscip;
 }

/** destructor of variable pricer to free user data (called when SCIP is exiting) */
SCIP_DECL_PRICERFREE(ObjPricerGcg::scip_free)
{
   assert(scip == scip_);
   SCIP_CALL( solversFree() );

   SCIPfreeMemoryArray(scip, &pricerdata->solvers);

   /* free memory for pricerdata*/
   if( pricerdata != NULL )
   {
      SCIPfreeMemory(scip, &pricerdata);
   }

   delete pricingcontroller;

   if( reducedcostpricing != NULL )
      delete reducedcostpricing;

   if( farkaspricing != NULL )
      delete farkaspricing;

   SCIPpricerSetData(pricer, NULL);
   return SCIP_OKAY;
}


/** initialization method of variable pricer (called after problem was transformed) */
SCIP_DECL_PRICERINIT(ObjPricerGcg::scip_init)
{ /*lint --e{715}*/
   assert(scip == scip_);
   assert(reducedcostpricing != NULL);
   assert(farkaspricing != NULL);

   SCIP_CALL( solversInit() );

   SCIP_CALL( reducedcostpricing->resetCalls() );
   SCIP_CALL( farkaspricing->resetCalls() );

   return SCIP_OKAY;
}


/** deinitialization method of variable pricer (called before transformed problem is freed) */
SCIP_DECL_PRICEREXIT(ObjPricerGcg::scip_exit)
{ /*lint --e{715}*/
   assert(scip == scip_);
   SCIP_CALL( solversExit() );

   return SCIP_OKAY;
}


/** solving process initialization method of variable pricer (called when branch and bound process is about to begin) */
SCIP_DECL_PRICERINITSOL(ObjPricerGcg::scip_initsol)
{
   int i;
   int norigvars;
   SCIP_Bool discretization;
   SCIP_CONS** masterconss;
   int nmasterconss;
   int origverblevel;
   int size;

   assert(scip == scip_);
   assert(pricer != NULL);
   assert(pricerdata != NULL);

   /* at the beginning, the output of the master problem gets the same verbosity level
    * as the output of the original problem */
   SCIP_CALL( SCIPgetIntParam(origprob, "display/verblevel", &origverblevel) );
   SCIP_CALL( SCIPsetIntParam(scip, "display/verblevel", origverblevel) );

   pricerdata->currnodenr = -1;
   pricerdata->newnode = TRUE;
   pricerdata->artificialused = FALSE;

   nmasterconss = GCGgetNMasterConss(origprob);
   masterconss = GCGgetMasterConss(origprob);

   pricerdata->artificialvars = NULL;
   pricerdata->nartificialvars = 0;
   pricerdata->maxartificialvars = 0;

   /* init array containing all pricing problems */
   pricerdata->npricingprobs = GCGgetNPricingprobs(origprob);
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->pricingprobs), pricerdata->npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->npointsprob), pricerdata->npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->nraysprob), pricerdata->npricingprobs) );

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->farkascallsdist), pricerdata->npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->farkasfoundvars), pricerdata->npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->farkasnodetimedist), pricerdata->npricingprobs) );

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->redcostcallsdist), pricerdata->npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->redcostfoundvars), pricerdata->npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->redcostnodetimedist), pricerdata->npricingprobs) );

   size = SCIPcalcMemGrowSize(scip, pricerdata->npricingprobs);
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->realdualvalues), size) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->maxrealdualvalues), size) );
   pricerdata->maxrealdualvaluescapacity = size;

   pricerdata->oldvars = 0;

   pricerdata->npricingprobsnotnull = 0;

   for( i = 0; i < pricerdata->npricingprobs; i++ )
   {

      pricerdata->farkascallsdist[i] = 0;
      pricerdata->farkasfoundvars[i] = 0;
      pricerdata->farkasnodetimedist[i] = 0;
      pricerdata->redcostcallsdist[i] = 0;
      pricerdata->redcostfoundvars[i] = 0;
      pricerdata->redcostnodetimedist[i]= 0;


      if( GCGisPricingprobRelevant(origprob, i) )
      {
         pricerdata->pricingprobs[i] = GCGgetPricingprob(origprob, i);
         pricerdata->npricingprobsnotnull++;
         pricerdata->maxrealdualvalues[i] = SCIPcalcMemGrowSize(scip, SCIPgetNVars(pricerdata->pricingprobs[i]));
         SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->realdualvalues[i]), pricerdata->maxrealdualvalues[i]) ); /*lint !e666 !e866*/
      }
      else
      {
         pricerdata->realdualvalues[i] = NULL;
         pricerdata->pricingprobs[i] = NULL;
      }
      pricerdata->npointsprob[i] = 0;
      pricerdata->nraysprob[i] = 0;
   }

   /* alloc memory for arrays of reduced cost */
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->dualsolconv), pricerdata->npricingprobs) );

   /* alloc memory for solution values of variables in pricing problems */
   norigvars = SCIPgetNOrigVars(origprob);
   pricerdata->maxsolvals = SCIPcalcMemGrowSize(scip, norigvars);
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(pricerdata->solvals), pricerdata->maxsolvals) );

   SCIP_CALL( SCIPcreateCPUClock(scip, &(pricerdata->freeclock)) );
   SCIP_CALL( SCIPcreateCPUClock(scip, &(pricerdata->transformclock)) );

   pricerdata->solvedsubmipsoptimal = 0;
   pricerdata->solvedsubmipsheur = 0;
   pricerdata->calls = 0;
   pricerdata->pricingiters = 0;

   /* set variable type for master variables */
   SCIP_CALL( SCIPgetBoolParam(origprob, "relaxing/gcg/discretization", &discretization) );
   if( discretization && SCIPgetNContVars(origprob) == 0 )
   {
      pricerdata->vartype = SCIP_VARTYPE_INTEGER;
   }
   else
   {
      pricerdata->vartype = SCIP_VARTYPE_CONTINUOUS;
   }

   SCIP_CALL( SCIPhashmapCreate(&(pricerdata->mapcons2idx), SCIPblkmem(scip), 10 * nmasterconss +1) );
   for( i = 0; i < nmasterconss; i++ )
   {
      SCIP_CALL( SCIPhashmapInsert(pricerdata->mapcons2idx, masterconss[i], (void*)(size_t)i) );
      assert((int)(size_t)SCIPhashmapGetImage(pricerdata->mapcons2idx, masterconss[i]) == i); /*lint !e507*/
   }

   pricerdata->npricedvars = 0;
   pricerdata->maxpricedvars = SCIPcalcMemGrowSize(scip, 50);
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &pricerdata->pricedvars, pricerdata->maxpricedvars) );

   pricerdata->nroundsredcost = 0;
#ifdef SCIP_STATISTIC
   pricerdata->rootlpsol = NULL;
   pricerdata->rootfarkastime = 0.0;
   pricerdata->dualdiff = 0.0;
   pricerdata->dualdiffround = -1;
   pricerdata->nrootbounds = 0;
   pricerdata->maxrootbounds = 50;
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &pricerdata->rootpbs, pricerdata->maxrootbounds) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &pricerdata->rootdbs, pricerdata->maxrootbounds) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &pricerdata->roottimes, pricerdata->maxrootbounds) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &pricerdata->rootdualdiffs, pricerdata->maxrootbounds) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &pricerdata->dualvalues, pricerdata->maxrootbounds) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &pricerdata->dualsolconvs, pricerdata->maxrootbounds) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &(pricerdata->nodetimehist), PRICER_STAT_ARRAYLEN_TIME) ); /*lint !e506*/
   SCIP_CALL( SCIPallocMemoryArray(scip, &(pricerdata->foundvarshist), PRICER_STAT_ARRAYLEN_VARS) ); /*lint !e506*/

   BMSclearMemoryArray(pricerdata->nodetimehist, PRICER_STAT_ARRAYLEN_TIME);
   BMSclearMemoryArray(pricerdata->foundvarshist, PRICER_STAT_ARRAYLEN_VARS);
#endif

   pricerdata->rootnodedegeneracy = 0.0;
   pricerdata->avgrootnodedegeneracy = 0.0;
   pricerdata->ndegeneracycalcs = 0;

   /* sort solvers by priority */
   SCIPsortPtr((void**)pricerdata->solvers, GCGsolverComp, pricerdata->nsolvers);

   SCIP_CALL( pricingcontroller->initSol() );

   SCIP_CALL( solversInitsol() );

   /* if maxobj should be used, compute it */
   if( pricerdata->usemaxobj )
   {
      SCIP_Bool reliable;
      pricerdata->maxobj = 0.0;
      reliable = TRUE;
      for( i = 0; i < SCIPgetNVars(origprob); ++i )
      {
         SCIP_VAR* var;
         SCIP_Real obj;
         SCIP_Real ub;
         SCIP_Real lb;

         var = SCIPgetVars(origprob)[i];
         obj = SCIPvarGetObj(var);
         ub = SCIPvarGetUbGlobal(var);
         lb = SCIPvarGetLbGlobal(var);

         /* check if influence of variable on objective is bounded */
         if( (SCIPisInfinity(origprob, ub) && SCIPisPositive(origprob, obj))
          || (SCIPisInfinity(origprob, -lb) && SCIPisNegative(origprob, obj)) )
         {
            /* if it is not bounded, maxobj is not reliable; use large, heuristic value */
            pricerdata->maxobj += pricerdata->factorunreliable* ABS(obj);
            reliable = FALSE;
         }
         else
            /* if it is bounded, add maximum difference to maxobj */
            pricerdata->maxobj += MAX(ub * obj, lb * obj) - MIN(ub * obj, lb * obj);
      }
      if( !reliable && pricerdata->onlyreliablebigm )
      {
         pricerdata->useartificialvars = FALSE;
         pricerdata->maxobj = SCIPinfinity(origprob);
         SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Big M to be used for artificial variables not reliable; use regular Farkas pricing instead.\n");
      }
      else if( !reliable && !pricerdata->onlyreliablebigm )
         SCIPwarningMessage(scip, "Big M used for artificial variables not reliable. This might lead to wrong solutions.\n");
   }
   else
      pricerdata->maxobj = SCIPinfinity(origprob);

   createStabilization();
   SCIP_CALL( stabilization->setNLinkingconsvals(GCGgetNVarLinkingconss(origprob)) );
   SCIP_CALL( stabilization->setNConvconsvals(GCGgetNPricingprobs(origprob)) );

   if( pricerdata->usecolpool )
      SCIP_CALL( createColpool() );

   SCIP_CALL( createPricestore() );

   SCIP_CALL( SCIPactivateEventHdlrDisplay(scip_) );

   return SCIP_OKAY;
}


/** solving process deinitialization method of variable pricer (called before branch and bound process data is freed) */
SCIP_DECL_PRICEREXITSOL(ObjPricerGcg::scip_exitsol)
{
   int i;

   assert(scip == scip_);
   assert(pricer != NULL);
   assert(pricerdata != NULL);

   SCIP_CALL( solversExitsol() );

   SCIP_CALL( pricingcontroller->exitSol() );

   if( stabilization != NULL )
      delete stabilization;

   stabilization = NULL;

   if( pricerdata->usecolpool )
      SCIP_CALL( GCGcolpoolFree(scip_, &colpool) );

   SCIP_CALL( GCGpricestoreFree(scip_, &pricestore) );

   SCIPhashmapFree(&(pricerdata->mapcons2idx));

   SCIPfreeBlockMemoryArray(scip, &(pricerdata->pricingprobs), pricerdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &(pricerdata->npointsprob), pricerdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &(pricerdata->nraysprob), pricerdata->npricingprobs);

   SCIPfreeBlockMemoryArray(scip, &(pricerdata->farkascallsdist), pricerdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &(pricerdata->farkasfoundvars), pricerdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &(pricerdata->farkasnodetimedist), pricerdata->npricingprobs);

   SCIPfreeBlockMemoryArray(scip, &(pricerdata->redcostcallsdist), pricerdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &(pricerdata->redcostfoundvars), pricerdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &(pricerdata->redcostnodetimedist), pricerdata->npricingprobs);


   SCIPfreeBlockMemoryArray(scip, &(pricerdata->dualsolconv), pricerdata->npricingprobs);

   SCIPfreeBlockMemoryArray(scip, &(pricerdata->solvals), pricerdata->maxsolvals);

   for( i = 0; i < pricerdata->nartificialvars; i++ )
   {
//      SCIP_CALL( SCIPdropVarEvent(scip, pricerdata->artificialvars[i], SCIP_EVENTTYPE_VARDELETED,
//            pricerdata->eventhdlr, NULL, -1) );

      SCIP_CALL( SCIPreleaseVar(scip, &pricerdata->artificialvars[i]) );
   }
   SCIPfreeBlockMemoryArrayNull(scip, &(pricerdata->artificialvars), pricerdata->maxartificialvars);
   pricerdata->nartificialvars = 0;

   for( i = 0; i < pricerdata->npricedvars; i++ )
   {
      SCIP_CALL( SCIPdropVarEvent(scip, pricerdata->pricedvars[i], SCIP_EVENTTYPE_VARDELETED,
            pricerdata->eventhdlr, NULL, -1) );

      SCIP_CALL( SCIPreleaseVar(scip, &pricerdata->pricedvars[i]) );
   }
   SCIPfreeBlockMemoryArray(scip, &pricerdata->pricedvars, pricerdata->maxpricedvars);
   pricerdata->maxpricedvars = 0;
   pricerdata->npricedvars = 0;

#ifdef SCIP_STATISTIC
   SCIPfreeBlockMemoryArray(scip, &pricerdata->rootpbs, pricerdata->maxrootbounds);
   SCIPfreeBlockMemoryArray(scip, &pricerdata->rootdbs, pricerdata->maxrootbounds);
   SCIPfreeBlockMemoryArray(scip, &pricerdata->roottimes, pricerdata->maxrootbounds);
   SCIPfreeBlockMemoryArray(scip, &pricerdata->rootdualdiffs, pricerdata->maxrootbounds);
   SCIPfreeBlockMemoryArray(scip, &pricerdata->dualvalues, pricerdata->maxrootbounds);
   SCIPfreeBlockMemoryArray(scip, &pricerdata->dualsolconvs, pricerdata->maxrootbounds);
   SCIPfreeSol(scip, &pricerdata->rootlpsol);
   pricerdata->rootlpsol = NULL;
   pricerdata->maxrootbounds = 0;
   pricerdata->nrootbounds = 0;
   pricerdata->rootfarkastime = 0.0;
   pricerdata->dualdiff = 0.0;
   SCIPfreeMemoryArrayNull(scip, &(pricerdata->nodetimehist));
   SCIPfreeMemoryArrayNull(scip, &(pricerdata->foundvarshist));

   pricerdata->nodetimehist = NULL;
   pricerdata->foundvarshist = NULL;
#endif

   SCIP_CALL( SCIPfreeClock(scip, &(pricerdata->freeclock)) );
   SCIP_CALL( SCIPfreeClock(scip, &(pricerdata->transformclock)) );

   for( i = 0; i < pricerdata->npricingprobs; ++i )
   {
      SCIPfreeBlockMemoryArrayNull(scip, &(pricerdata->realdualvalues[i]), pricerdata->maxrealdualvalues[i]);
   }
   SCIPfreeBlockMemoryArray(scip, &(pricerdata->realdualvalues), pricerdata->maxrealdualvaluescapacity);
   SCIPfreeBlockMemoryArray(scip, &(pricerdata->maxrealdualvalues), pricerdata->maxrealdualvaluescapacity);

   return SCIP_OKAY;
}


/** reduced cost pricing method of variable pricer for feasible LPs */
SCIP_DECL_PRICERREDCOST(ObjPricerGcg::scip_redcost)
{ /*lint -esym(715, stopearly)*/
   SCIP_RETCODE retcode;

   assert(scip == scip_);
   assert(pricer != NULL);
   assert(pricerdata != NULL);
   assert(reducedcostpricing != NULL);
   assert(farkaspricing != NULL);

   *result = SCIP_DIDNOTRUN;

   if( reducedcostpricing->getCalls() == 0 )
   {
      /** @todo This is just a workaround around SCIP stages! */
      if( farkaspricing->getCalls() == 0 )
      {
         SCIP_CALL( GCGconsMasterbranchAddRootCons(scip) );
      }
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Starting reduced cost pricing...\n");
   }

   if( SCIPisStopped(scip_) )
   {
      return SCIP_OKAY;
   }

   if( SCIPgetCurrentNode(scip) == SCIPgetRootNode(scip) && GCGsepaGetNCuts(scip) == 0 && reducedcostpricing->getCalls() > 0
      && GCGmasterIsCurrentSolValid(scip) && pricerdata->artificialused )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Starting reduced cost pricing without artificial variables...\n");
   }

   if( !GCGmasterIsCurrentSolValid(scip) )
      pricerdata->artificialused = TRUE;
   else
      pricerdata->artificialused = FALSE;


   /* update number of reduced cost pricing rounds at the current node */
   if( SCIPgetNNodes(scip) == pricerdata->currnodenr )
   {
      pricerdata->nroundsredcost++;
   }
   else
   {
      pricerdata->currnodenr = SCIPgetNNodes(scip);
      pricerdata->newnode = TRUE;
      pricerdata->nroundsredcost = 0;
   }

   /* if the number of reduced cost pricing rounds at the current node exceeds the limit (and we are not at the root), stop pricing;
    * we always stop pricing, if the maximum number of reduced cost rounds is set to 0
    */
   if( reducedcostpricing->getMaxrounds() == 0 || (pricerdata->nroundsredcost >= reducedcostpricing->getMaxrounds() && pricerdata->currnodenr != 1) )
   {
      SCIPdebugMessage("pricing aborted at node %lld\n", pricerdata->currnodenr);
      return SCIP_OKAY;
   }

   *result = SCIP_SUCCESS;

   /* perform pricing */

   pricingcontroller->increaseEagerage();

   GCGpricestoreEndFarkas(pricestore);
   if( pricerdata->usecolpool )
      GCGcolpoolEndFarkas(colpool);

   SCIP_CALL( reducedcostpricing->startClock() );
   retcode = priceNewVariables(reducedcostpricing, result, lowerbound);
   SCIP_CALL( reducedcostpricing->stopClock() );

#ifdef SCIP_STATISTIC
   if( SCIPgetCurrentNode(scip_) == SCIPgetRootNode(scip_) && GCGsepaGetNCuts(scip_) == 0 )
   {
      SCIP_CALL( addRootBounds(SCIPgetLPObjval(scip_), *lowerbound) );
      SCIPdebugMessage("Add bounds, %f\n", *lowerbound);
   }
#endif
   return retcode;
}

SCIP_RETCODE ObjPricerGcg::ensureSizeArtificialvars(
   int                   size                /**< needed size */
)
{
   assert(pricerdata != NULL);
   assert(pricerdata->artificialvars != NULL);

   if( pricerdata->maxartificialvars < size )
   {
      int oldsize = pricerdata->maxartificialvars;
      pricerdata->maxartificialvars = SCIPcalcMemGrowSize(scip_, size);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip_, &(pricerdata->artificialvars), oldsize, pricerdata->maxartificialvars) );
   }
   assert(pricerdata->maxartificialvars >= size);

   return SCIP_OKAY;
}

/** add artificial vars */
SCIP_RETCODE ObjPricerGcg::addArtificialVars(
   )
{
   SCIP_CONS** masterconss;
   int nmasterconss;
   int nconvconss;
   char varname[SCIP_MAXSTRLEN];
   int i;
   SCIP_Real bigm;

   assert(pricerdata != NULL);
   assert(pricerdata->pricedvars != NULL);

   masterconss = GCGgetMasterConss(origprob);
   nmasterconss = GCGgetNMasterConss(origprob);

   nconvconss = GCGgetNPricingprobs(origprob);

   /* if bigmartificial is negative, use maxobj */
   if( pricerdata->usemaxobj )
      bigm = pricerdata->maxobj;
   else
      bigm = pricerdata->bigmartificial;

   for( i = 0; i < nmasterconss; ++i )
   {
      if( !SCIPisInfinity(scip_, -1.0*GCGconsGetLhs(scip_, masterconss[i]) ))
      {
         (void) SCIPsnprintf(varname, SCIP_MAXSTRLEN, "art_lhs_%s", SCIPconsGetName(masterconss[i]));
         ensureSizeArtificialvars(pricerdata->nartificialvars + 1);
         SCIP_CALL( GCGcreateArtificialVar(scip_, &(pricerdata->artificialvars[pricerdata->nartificialvars]), varname, bigm) );
         SCIP_CALL( SCIPaddCoefLinear(scip_, masterconss[i], pricerdata->artificialvars[pricerdata->nartificialvars], 1.0) );
         SCIP_CALL( SCIPaddVar(scip_, pricerdata->artificialvars[pricerdata->nartificialvars]) );
         ++(pricerdata->nartificialvars);
      }

      if( !SCIPisInfinity(scip_, GCGconsGetRhs(scip_, masterconss[i]) ))
      {
         (void) SCIPsnprintf(varname, SCIP_MAXSTRLEN, "art_rhs_%s", SCIPconsGetName(masterconss[i]));
         ensureSizeArtificialvars(pricerdata->nartificialvars + 1);
         SCIP_CALL( GCGcreateArtificialVar(scip_, &(pricerdata->artificialvars[pricerdata->nartificialvars]), varname, bigm) );
         SCIP_CALL( SCIPaddCoefLinear(scip_, masterconss[i], pricerdata->artificialvars[pricerdata->nartificialvars], -1.0) );
         SCIP_CALL( SCIPaddVar(scip_, pricerdata->artificialvars[pricerdata->nartificialvars]) );
         ++(pricerdata->nartificialvars);
      }
   }

   for( i = 0; i < nconvconss; ++i )
   {
      SCIP_CONS* convcons;

      if( !GCGisPricingprobRelevant(origprob, i) )
         continue;

      convcons = GCGgetConvCons(origprob, i);

      if( !SCIPisInfinity(scip_, -1.0*GCGconsGetLhs(scip_, convcons) ))
      {
         (void) SCIPsnprintf(varname, SCIP_MAXSTRLEN, "art_lhs_%s", SCIPconsGetName(convcons));
         ensureSizeArtificialvars(pricerdata->nartificialvars + 1);
         SCIP_CALL( GCGcreateArtificialVar(scip_, &(pricerdata->artificialvars[pricerdata->nartificialvars]), varname, bigm) );
         SCIP_CALL( SCIPaddCoefLinear(scip_, convcons, pricerdata->artificialvars[pricerdata->nartificialvars], 1.0) );
         SCIP_CALL( SCIPaddVar(scip_, pricerdata->artificialvars[pricerdata->nartificialvars]) );
         ++(pricerdata->nartificialvars);
      }

      if( !SCIPisInfinity(scip_, GCGconsGetRhs(scip_, convcons) ))
      {
         (void) SCIPsnprintf(varname, SCIP_MAXSTRLEN, "art_rhs_%s", SCIPconsGetName(convcons));
         ensureSizeArtificialvars(pricerdata->nartificialvars + 1);
         SCIP_CALL( GCGcreateArtificialVar(scip_, &(pricerdata->artificialvars[pricerdata->nartificialvars]), varname, bigm) );
         SCIP_CALL( SCIPaddCoefLinear(scip_, convcons, pricerdata->artificialvars[pricerdata->nartificialvars], -1.0) );
         SCIP_CALL( SCIPaddVar(scip_, pricerdata->artificialvars[pricerdata->nartificialvars]) );
         ++(pricerdata->nartificialvars);
      }

   }

   pricerdata->artificialused = TRUE;

   return SCIP_OKAY;
}

/** farcas pricing method of variable pricer for infeasible LPs */
SCIP_DECL_PRICERFARKAS(ObjPricerGcg::scip_farkas)
{
   SCIP_RETCODE retcode;
   SCIP_SOL** origsols;
   int norigsols;
   int nstoredsols;

   assert(scip == scip_);
   assert(pricer != NULL);
   assert(pricerdata != NULL);
   assert(reducedcostpricing != NULL);
   assert(farkaspricing != NULL);

   *result = SCIP_DIDNOTRUN;

   /** @todo This is just a workaround around SCIP stages! */
   if( reducedcostpricing->getCalls() == 0 && farkaspricing->getCalls() == 0 )
   {
      SCIP_CALL( GCGconsMasterbranchAddRootCons(scip) );
   }

   /* get solutions from the original problem */
   origsols = SCIPgetSols(origprob);
   norigsols = SCIPgetNSols(origprob);
   assert(norigsols >= 0);

   /* update current node */
   if( SCIPgetNNodes(scip) != pricerdata->currnodenr )
   {
      pricerdata->currnodenr = SCIPgetNNodes(scip);
      pricerdata->newnode = TRUE;
   }

   *result = SCIP_SUCCESS;

   /* Add already known solutions for the original problem to the master variable space */
   /** @todo This is just a workaround around SCIP stages! */
   nstoredsols = 0;
   if( farkaspricing->getCalls() == 0 )
   {
      int i;

      for( i = 0; i < norigsols; ++i )
      {
         SCIP_Bool stored;
         assert(origsols[i] != NULL);
         SCIPdebugMessage("Transferring original feasible solution found by <%s> to master problem\n",
            SCIPsolGetHeur(origsols[i]) == NULL ? "relaxation" : SCIPheurGetName(SCIPsolGetHeur(origsols[i])));
         SCIP_CALL( GCGmasterTransOrigSolToMasterVars(scip, origsols[i], &stored) );
         if( stored )
         {
            ++nstoredsols;
         }
      }
      SCIPdebugMessage("GCGmasterTransOrigSolToMasterVars() transferred %d original feasible solutions\n",
         nstoredsols);
      /* return if we transferred solutions as the master should be feasible */
      if( nstoredsols > 0 )
      {
         farkaspricing->incCalls();
#ifdef SCIP_STATISTIC
         pricerdata->rootfarkastime =  SCIPgetSolvingTime(scip_);
#endif
         return SCIP_OKAY;
      }
   }

   if( pricerdata->useartificialvars && farkaspricing->getCalls() == 0 )
   {
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Add artificial variables ensuring the restricted master problem is feasible.\n");
      SCIP_CALL( addArtificialVars() );
      farkaspricing->incCalls();
      return SCIP_OKAY;
   }

   GCGpricestoreStartFarkas(pricestore);
   if( pricerdata->usecolpool )
      GCGcolpoolStartFarkas(colpool);

   SCIP_CALL( farkaspricing->startClock() );
   retcode = priceNewVariables(farkaspricing, result, NULL);
   SCIP_CALL( farkaspricing->stopClock() );

#ifdef SCIP_STATISTIC
   pricerdata->rootfarkastime =  SCIPgetSolvingTime(scip_);
#endif

   return retcode;
}

/** create the pointers for the pricing types */
SCIP_RETCODE ObjPricerGcg::createPricingTypes()
{
   farkaspricing = new FarkasPricing(scip_);
   SCIP_CALL( farkaspricing->addParameters() );

   reducedcostpricing = new ReducedCostPricing(scip_);
   SCIP_CALL( reducedcostpricing->addParameters() );

   pricingtype = NULL;

   return SCIP_OKAY;
}

/** create the pricing controller */
SCIP_RETCODE ObjPricerGcg::createPricingcontroller()
{
   pricingcontroller = new Pricingcontroller(scip_);
   SCIP_CALL( pricingcontroller->addParameters() );

   return SCIP_OKAY;
}

/** create the pointers for the stabilization */
void ObjPricerGcg::createStabilization()
{
   SCIP_Bool usehybridascent;

   usehybridascent = pricerdata->hybridascent ||
                     (GCGgetNPricingprobs(origprob) == GCGgetNRelPricingprobs(origprob) && pricerdata->hybridascentnoagg);


   stabilization = new Stabilization(scip_, reducedcostpricing, usehybridascent);
}

SCIP_RETCODE ObjPricerGcg::createColpool()
{
   assert(farkaspricing != NULL);
   assert(reducedcostpricing != NULL);
   assert(pricerdata != NULL);

   SCIP_CALL( GCGcolpoolCreate(scip_, &colpool, pricerdata->colpoolagelimit) );

   return SCIP_OKAY;
}

SCIP_RETCODE ObjPricerGcg::createPricestore()
{
   SCIP_CALL( GCGpricestoreCreate(scip_, &pricestore,
      pricerdata->redcostfac, pricerdata->objparalfac, pricerdata->orthofac,
      pricerdata->mincolorth,
      pricerdata->efficiacychoice) );

   return SCIP_OKAY;
}
/*
 * variable pricer specific interface methods
 */

/** creates the GCG variable pricer and includes it in SCIP */
extern "C"
SCIP_RETCODE SCIPincludePricerGcg(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP*                 origprob            /**< SCIP data structure of the original problem */
   )
{  /*lint -esym(429,pricer) */
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata = NULL;

   SCIP_CALL( SCIPallocMemory(scip, &pricerdata) );

   /* initialize solvers array */
   pricerdata->solvers = NULL;
   pricerdata->nsolvers = 0;
   pricerdata->realdualvalues = NULL;

#ifdef SCIP_STATISTIC
   pricerdata->nodetimehist = NULL;
   pricerdata->foundvarshist = NULL;
#endif

   pricer = new ObjPricerGcg(scip, origprob, PRICER_NAME, PRICER_DESC, PRICER_PRIORITY, PRICER_DELAY, pricerdata);
   /* include variable pricer */
   SCIP_CALL( SCIPincludeObjPricer(scip, pricer, TRUE) );

   SCIP_CALL( pricer->createPricingTypes() );
   SCIP_CALL( pricer->createPricingcontroller() );

   /* include event handler into master SCIP */
   SCIP_CALL( SCIPincludeEventhdlr(scip, EVENTHDLR_NAME, EVENTHDLR_DESC,
         NULL, eventFreeVardeleted, eventInitVardeleted, eventExitVardeleted,
         eventInitsolVardeleted, eventExitsolVardeleted, eventDeleteVardeleted, eventExecVardeleted,
         NULL) );

   pricerdata->eventhdlr = SCIPfindEventhdlr(scip, EVENTHDLR_NAME);

   SCIP_CALL( SCIPaddBoolParam(origprob, "pricing/masterpricer/abortpricingint",
         "should pricing be aborted due to integral objective function?",
         &pricerdata->abortpricingint, TRUE, DEFAULT_ABORTPRICINGINT, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricing/masterpricer/abortpricinggap",
         "gap between dual bound and RMP objective at which pricing is aborted",
         &pricerdata->abortpricinggap, TRUE, DEFAULT_ABORTPRICINGGAP, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origprob, "pricing/masterpricer/dispinfos",
         "should additional informations concerning the pricing process be displayed?",
         &pricerdata->dispinfos, FALSE, DEFAULT_DISPINFOS, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origprob, "pricing/masterpricer/threads",
         "how many threads should be used to concurrently solve the pricing problem (0 to guess threads by OpenMP)",
         &ObjPricerGcg::threads, FALSE, DEFAULT_THREADS, 0, 4096, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origprob, "pricing/masterpricer/stabilization",
         "should stabilization be performed?",
         &pricerdata->stabilization, FALSE, DEFAULT_STABILIZATION, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origprob, "pricing/masterpricer/stabilizationtree",
         "should stabilization be performed in the tree (in nodes other than the root node)?",
         &pricerdata->stabilizationtree, FALSE, DEFAULT_STABILIZATIONTREE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origprob, "pricing/masterpricer/usecolpool",
         "should the colpool be checked for negative redcost cols before solving the pricing problems?",
         &pricerdata->usecolpool, FALSE, DEFAULT_USECOLPOOL, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origprob, "pricing/masterpricer/stabilization/hybridascent",
         "should hybridization of smoothing with an ascent method be enabled?",
         &pricerdata->hybridascent, FALSE, DEFAULT_HYBRIDASCENT, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origprob, "pricing/masterpricer/stabilization/hybridascentnoagg",
         "should hybridization of smoothing with an ascent method be enabled if pricing problems cannot be aggregation?",
         &pricerdata->hybridascentnoagg, FALSE, DEFAULT_HYBRIDASCENT_NOAGG, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origprob, "pricing/masterpricer/useartificialvars",
         "should artificial variables be used to make the RMP feasible (instead of applying Farkas pricing)?",
         &pricerdata->useartificialvars, FALSE, DEFAULT_USEARTIFICIALVARS, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origprob, "pricing/masterpricer/usemaxobj",
         "use maxobj for big M objective of artificial variables",
         &pricerdata->usemaxobj, FALSE, DEFAULT_USEMAXOBJ, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(origprob, "pricing/masterpricer/onlyreliablebigm",
         "only use maxobj for big M objective of artificial variables if it is reliable",
         &pricerdata->onlyreliablebigm, FALSE, DEFAULT_ONLYRELIABLEBIGM, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricing/masterpricer/factorunreliable",
         "factor to use for objective of unbounded variables",
         &pricerdata->factorunreliable, FALSE, DEFAULT_FACTORUNRELIABLE, 0.0, SCIPinfinity(origprob), NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricing/masterpricer/bigmartificial",
         "value for for big M objective of artificial variables (negative if max obj should be used)",
         &pricerdata->bigmartificial, FALSE, DEFAULT_BIGMARTIFICIAL, 0.0, SCIPinfinity(origprob), NULL, NULL) );

   SCIP_CALL( SCIPsetIntParam(scip, "lp/disablecutoff", DEFAULT_DISABLECUTOFF) );

   SCIP_CALL( SCIPaddIntParam(origprob, "pricing/masterpricer/disablecutoff",
         "should the cutoffbound be applied in master LP solving (0: on, 1:off, 2:auto)?",
         &pricerdata->disablecutoff, FALSE, DEFAULT_DISABLECUTOFF, 0, 2, paramChgdDisablecutoff, NULL) );

   SCIP_CALL( SCIPaddIntParam(origprob, "pricing/masterpricer/colpool/agelimit",
         "age limit for columns in column pool? (-1 for no limit)",
         &pricerdata->colpoolagelimit, FALSE, DEFAULT_COLPOOL_AGELIMIT, -1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricing/masterpricer/pricestore/redcostfac",
         "factor of -redcost/norm in score function",
         &pricerdata->redcostfac, FALSE, DEFAULT_PRICE_REDCOSTFAC, 0.0, 10.0, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricing/masterpricer/pricestore/objparalfac",
            "factor of objective parallelism in score function",
            &pricerdata->objparalfac, FALSE, DEFAULT_PRICE_OBJPARALFAC, 0.0, 10.0, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricing/masterpricer/pricestore/orthofac",
            "factor of orthogonalities in score function",
            &pricerdata->orthofac, FALSE, DEFAULT_PRICE_ORTHOFAC, 0.0, 10.0, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricing/masterpricer/pricestore/mincolorth",
            "minimal orthogonality of columns to add",
            &pricerdata->mincolorth, FALSE, DEFAULT_PRICE_MINCOLORTH, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(origprob, "pricing/masterpricer/pricestore/efficiacychoice",
            "choice to base efficiacy on",
            (int*)&pricerdata->efficiacychoice, FALSE, DEFAULT_PRICE_EFFICIACYCHOICE, 0, 2, NULL, NULL) );


   return SCIP_OKAY;
}

/** returns the pointer to the scip instance representing the original problem */
extern "C"
SCIP* GCGmasterGetOrigprob(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   ObjPricerGcg* pricer;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   return pricer->getOrigprob();
}

/** returns the array of variables that were priced in during the solving process */
extern "C"
SCIP_VAR** GCGmasterGetPricedvars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   return pricerdata->pricedvars;
}

/** returns the number of variables that were priced in during the solving process */
extern "C"
int GCGmasterGetNPricedvars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   return pricerdata->npricedvars;
}


/** adds the given constraint and the given position to the hashmap of the pricer */
extern "C"
SCIP_RETCODE GCGmasterAddMasterconsToHashmap(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_CONS*            cons,               /**< the constraint that should be added */
   int                   pos                 /**< the position of the constraint in the relaxator's masterconss array */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);
   assert(cons != NULL);
   assert(pos >= 0);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   SCIP_CALL( SCIPhashmapInsert(pricerdata->mapcons2idx, cons, (void*)(size_t)pos) );
   assert((int)(size_t)SCIPhashmapGetImage(pricerdata->mapcons2idx, cons) == pos); /*lint !e507*/

   SCIPdebugMessage("Added cons %s (%p) to hashmap with index %d\n", SCIPconsGetName(cons), (void*) cons, pos);

   return SCIP_OKAY;
}

#ifdef SCIP_STATISTIC
/** sets the optimal LP solution in the pricerdata */
extern "C"
SCIP_RETCODE GCGmasterSetRootLPSol(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL**            sol                 /**< pointer to optimal solution to root LP */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   pricerdata->rootlpsol = *sol;

   return SCIP_OKAY;
}

/** gets the optimal LP solution in the pricerdata */
extern "C"
SCIP_SOL* GCGmasterGetRootLPSol(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   return pricerdata->rootlpsol;
}
#endif

/** includes a solver into the pricer data */
extern "C"
SCIP_RETCODE GCGpricerIncludeSolver(
   SCIP*                 scip,               /**< SCIP data structure */
   const char*           name,               /**< name of solver */
   const char*           desc,               /**< description of solver */
   int                   priority,           /**< priority of solver */
   SCIP_Bool             heurenabled,        /**< flag to indicate whether heuristic solving method of the solver is enabled */
   SCIP_Bool             exactenabled,        /**< flag to indicate whether exact solving method of the solver is enabled */
   GCG_DECL_SOLVERUPDATE((*solverupdate)),   /**< update method for solver */
   GCG_DECL_SOLVERSOLVE  ((*solversolve)),   /**< solving method for solver */
   GCG_DECL_SOLVERSOLVEHEUR((*solversolveheur)), /**< heuristic solving method for solver */
   GCG_DECL_SOLVERFREE   ((*solverfree)),    /**< free method of solver */
   GCG_DECL_SOLVERINIT   ((*solverinit)),    /**< init method of solver */
   GCG_DECL_SOLVEREXIT   ((*solverexit)),    /**< exit method of solver */
   GCG_DECL_SOLVERINITSOL((*solverinitsol)), /**< initsol method of solver */
   GCG_DECL_SOLVEREXITSOL((*solverexitsol)), /**< exitsol method of solver */
   GCG_SOLVERDATA*       solverdata          /**< pricing solver data */
   )
{
   GCG_SOLVER* solver;
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   /* create pricing solver */
   solver = NULL;
   SCIP_CALL( GCGsolverCreate(scip, &solver, name, desc, priority, heurenabled, exactenabled,
      solverupdate, solversolve, solversolveheur, solverfree, solverinit, solverexit, solverinitsol, solverexitsol,
      solverdata) );
   assert(solver != NULL);

   /* add solver to pricer data */
   if( pricerdata->nsolvers == 0 )
   {
      SCIP_CALL( SCIPallocMemoryArray(scip_, &(pricerdata->solvers), 1) ); /*lint !e506*/
   }
   else
   {
      SCIP_CALL( SCIPreallocMemoryArray(scip_, &(pricerdata->solvers), (size_t)pricerdata->nsolvers+1) );
   }
   pricerdata->solvers[pricerdata->nsolvers] = solver;
   ++pricerdata->nsolvers;

   return SCIP_OKAY;
}

/** returns the available pricing solvers */
extern "C"
GCG_SOLVER** GCGpricerGetSolvers(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   return pricerdata->solvers;
}

/** returns the number of available pricing solvers */
extern "C"
int GCGpricerGetNSolvers(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   return pricerdata->nsolvers;
}

/** writes out a list of all pricing problem solvers */
extern "C"
void GCGpricerPrintListOfSolvers(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;
   int nsolvers;
   int i;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   assert((pricerdata->solvers == NULL) == (pricerdata->nsolvers == 0));

   nsolvers = pricerdata->nsolvers;

   SCIPdialogMessage(scip, NULL, " solver               priority heur  exact description\n");
   SCIPdialogMessage(scip, NULL, " --------------       -------- ----- ----- -----------\n");

   for( i = 0; i < nsolvers; ++i )
   {
      SCIPdialogMessage(scip, NULL,  " %-20s", GCGsolverGetName(pricerdata->solvers[i]));
      SCIPdialogMessage(scip, NULL,  " %8d", GCGsolverGetPriority(pricerdata->solvers[i]));
      SCIPdialogMessage(scip, NULL,  " %5s", GCGsolverIsHeurEnabled(pricerdata->solvers[i]) ? "TRUE" : "FALSE");
      SCIPdialogMessage(scip, NULL,  " %5s", GCGsolverIsExactEnabled(pricerdata->solvers[i]) ? "TRUE" : "FALSE");
      SCIPdialogMessage(scip, NULL,  " %s\n", GCGsolverGetDesc(pricerdata->solvers[i]));
   }
}

/** prints pricing solver statistics */
extern "C"
void GCGpricerPrintPricingStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
)
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   /**@todo add constraint statistics: how many constraints (instead of cuts) have been added? */
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file,
         "Pricing Solver     : #HeurFarkas  #OptFarkas  #HeurRedcost #OptRedcost Time: HeurFarkas  OptFarkas  HeurRedcost OptRedcost\n");
   for( int i = 0; i < pricerdata->nsolvers; ++i )
   {
      GCG_SOLVER* solver;
      solver = pricerdata->solvers[i];
      assert(solver != NULL);

      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  %-17.17s:",
            GCGsolverGetName(solver));
      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file,
            " %11d %11d   %11d %11d       %10.2f %10.2f   %10.2f %10.2f \n",
            GCGsolverGetHeurFarkasCalls(solver), GCGsolverGetOptFarkasCalls(solver),
            GCGsolverGetHeurRedcostCalls(solver), GCGsolverGetOptRedcostCalls(solver),
            GCGsolverGetHeurFarkasTime(scip, solver),
            GCGsolverGetOptFarkasTime(scip, solver),
            GCGsolverGetHeurRedcostTime(scip, solver),
            GCGsolverGetOptRedcostTime(scip, solver));
   }
}

/** prints pricer statistics */
extern "C"
void GCGpricerPrintStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;
   const PricingType* farkas;
   const PricingType* redcost;
   int i;
#ifdef SCIP_STATISTIC
   double start;
   double end;
#endif

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   /**@todo add constraint statistics: how many constraints (instead of cuts) have been added? */

   /* print of Pricing Statistics */

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Farkas pricing Statistic:\nno.\t#Calls\t\t#Vars\t\ttime(s)\n");

   for( i = 0; i < pricerdata->npricingprobs; i++ )
   {
      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%d  \t %d \t\t %d \t\t %.2f \n", i, pricerdata->farkascallsdist[i],
         pricerdata->farkasfoundvars[i], pricerdata->farkasnodetimedist[i]);

   }

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Reduced Cost pricing Statistic:\nno.\t#Calls\t\t#Vars\t\ttime(s)\n");

   for( i = 0; i < pricerdata->npricingprobs; i++ )
   {
      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%d  \t %d \t\t %d \t\t %.2f \n", i, pricerdata->redcostcallsdist[i],
         pricerdata->redcostfoundvars[i], pricerdata->redcostnodetimedist[i]);

   }

#ifdef SCIP_STATISTIC
   /* print of Histogram Buckets !=0      */

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Histogram Time\n");
   for( i = 0; i < PRICER_STAT_ARRAYLEN_TIME; i++ )
   {
      start = (1.0 * i * PRICER_STAT_BUCKETSIZE_TIME)/1000.0;
      end = start + PRICER_STAT_BUCKETSIZE_TIME/1000.0;

      if( pricerdata->nodetimehist[i] != 0 )
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "From\t%.4f\t-\t%.4f\ts:\t\t%d \n", start, end, pricerdata->nodetimehist[i]);
   }

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Histogram Found Vars\n");

   for( i = 0; i < PRICER_STAT_ARRAYLEN_VARS; i++ )
   {
      start = i * PRICER_STAT_BUCKETSIZE_VARS;
      end = start + PRICER_STAT_BUCKETSIZE_VARS;

      if( pricerdata->foundvarshist[i] != 0 )
         SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "From\t%.0f\t-\t%.0f\tvars:\t\t%d \n", start, end, pricerdata->foundvarshist[i]);
   }

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Root bounds \n");
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, " iter            pb            db            time        dualdiff     dualoptdiff\n");

   for( i = 0; i < pricerdata->nrootbounds; i++ )
   {
      SCIP_Real pb;
      SCIP_Real db;
      SCIP_Real time;
      SCIP_Real dualdiff;
      SCIP_Real dualoptdiff;

      pb = pricerdata->rootpbs[i];
      db = pricerdata->rootdbs[i];
      time = pricerdata->roottimes[i];
      dualdiff = pricerdata->rootdualdiffs[i];
      pricer->computeDualDiff(pricerdata->dualvalues[i], pricerdata->dualsolconvs[i], pricerdata->dualvalues[pricerdata->nrootbounds - 1], pricerdata->dualsolconvs[pricerdata->nrootbounds - 1], &dualoptdiff);

      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "%5d %13.6e %13.6e %15.5f %15.5f %15.5f\n", i, pb, db, time, dualdiff, dualoptdiff);
   }
#endif

   redcost = pricer->getReducedCostPricing();
   assert( redcost != NULL );
   farkas = pricer->getFarkasPricing();
   assert( farkas != NULL );

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Pricing Summary:\n");
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Calls                            : %d\n", pricerdata->calls);
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Farkas Pricing Calls             : %d\n", farkas->getCalls());
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Farkas Pricing Time              : %f\n", farkas->getClockTime());
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Reduced Cost Pricing Calls       : %d\n", redcost->getCalls());
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Reduced Cost Pricing Time        : %f\n", redcost->getClockTime());
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Solved subMIPs Heuristic Pricing : %d\n", pricerdata->solvedsubmipsheur);
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Solved subMIPs Optimal Pricing   : %d\n", pricerdata->solvedsubmipsoptimal);
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Time for transformation          : %f\n", SCIPgetClockTime(scip, pricerdata->transformclock));
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Time for freeing subMIPs         : %f\n", SCIPgetClockTime(scip, pricerdata->freeclock));

}

/** method to get existence of rays */
extern "C"
SCIP_RETCODE GCGpricerExistRays(
   SCIP*                 scip,               /**< master SCIP data structure */
   SCIP_Bool*            exist               /**< pointer to store if there exists any ray */
   )
{
   int prob;

   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   *exist = FALSE;

   for( prob = 0; prob < pricerdata->npricingprobs; ++prob )
   {
      if( pricerdata->nraysprob[prob] > 0 )
      {
         *exist = TRUE;
         break;
      }
   }

   return SCIP_OKAY;
}

/** get the number of extreme points that a pricing problem has generated so far */
extern "C"
int GCGpricerGetNPointsProb(
   SCIP*                 scip,               /**< master SCIP data structure */
   int                   probnr              /**< index of pricing problem */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   if( !GCGisPricingprobRelevant(GCGmasterGetOrigprob(scip), probnr) )
      return 0;
   else
      return pricerdata->npointsprob[probnr];
}

/** get the number of extreme rays that a pricing problem has generated so far */
extern "C"
int GCGpricerGetNRaysProb(
   SCIP*                 scip,               /**< master SCIP data structure */
   int                   probnr              /**< index of pricing problem */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   if( !GCGisPricingprobRelevant(GCGmasterGetOrigprob(scip), probnr) )
      return 0;
   else
      return pricerdata->nraysprob[probnr];
}

/** get the number of columns to be added to the master LP in the current pricing round */
extern "C"
int GCGpricerGetMaxColsRound(
   SCIP*                 scip                /**< master SCIP data structure */
   )
{
   ObjPricerGcg* pricer;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   return pricer->getMaxColsRound();
}

/** get the number of columns per pricing problem to be added to the master LP in the current pricing round */
extern "C"
int GCGpricerGetMaxColsProb(
   SCIP*                 scip                /**< master SCIP data structure */
   )
{
   ObjPricerGcg* pricer;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   return pricer->getMaxColsProb();
}

/** add a new column to the pricing storage */
extern "C"
SCIP_RETCODE GCGpricerAddCol(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_COL*              col                 /**< priced col */
   )
{
   ObjPricerGcg* pricer;

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   SCIP_CALL( pricer->addColToPricestore(col) );

   return SCIP_OKAY;
}

/** transfers a primal solution of the original problem into the master variable space,
 *  i.e. creates one master variable for each block and adds the solution to the master problem  */
extern "C"
SCIP_RETCODE GCGmasterTransOrigSolToMasterVars(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             origsol,            /**< the solution that should be transferred */
   SCIP_Bool*            stored              /**< pointer to store if transferred solution is feasible (or NULL) */
   )
{
   ObjPricerGcg* pricer = NULL;
   SCIP_PRICERDATA* pricerdata = NULL;
   SCIP_SOL* mastersol = NULL;
   SCIP_VAR* newvar = NULL;
   SCIP* origprob = NULL;
   SCIP_Bool added;
   int prob;
   int i;
   int j;

   SCIP_VAR** origvars = NULL;
   SCIP_Real* origsolvals = NULL;
   int norigvars;

   SCIP_VAR*** pricingvars = NULL;
   SCIP_Real** pricingvals = NULL;
   int* npricingvars = NULL;

   assert(scip != NULL);

   origprob = GCGgetOriginalprob(scip);
   assert(origprob != NULL);

   /*
    * Due to global domain changes, it might happen that the solution is not feasible for the
    * transformed original problem anymore; in that case, do not tranfer it to the master
    */
   SCIP_CALL( SCIPcheckSol(origprob, origsol, FALSE, FALSE, TRUE, TRUE, TRUE, &added) );
   if( !added )
   {
      if( stored != NULL)
         *stored = FALSE;
      return SCIP_OKAY;
   }

   pricerdata = NULL;   /* the pricerdata is set to NULL when the Benders' decomposition mode is used. */
   if( GCGgetDecompositionMode(origprob) == DEC_DECMODE_DANTZIGWOLFE )
   {
      pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
      assert(pricer != NULL);

      pricerdata = pricer->getPricerdata();
      assert(pricerdata != NULL);
   }

   /* now compute coefficients of the master variables in the master constraint */
   origvars = SCIPgetVars(origprob);
   norigvars = SCIPgetNVars(origprob);

   /* allocate memory for storing variables and solution values from the solution */
   SCIP_CALL( SCIPallocBufferArray(scip, &origsolvals, norigvars) ); /*lint !e530*/

   if( pricerdata != NULL )
   {
      SCIP_CALL( SCIPallocBufferArray(scip, &pricingvars, pricerdata->npricingprobs) ); /*lint !e530*/
      SCIP_CALL( SCIPallocBufferArray(scip, &pricingvals, pricerdata->npricingprobs) ); /*lint !e530*/
      SCIP_CALL( SCIPallocBufferArray(scip, &npricingvars, pricerdata->npricingprobs) ); /*lint !e530*/

      for( i = 0; i < pricerdata->npricingprobs; i++ )
      {
         int representative;
         representative = GCGgetBlockRepresentative(origprob, i);
         npricingvars[i] = 0;

         SCIP_CALL( SCIPallocBufferArray(scip, &(pricingvars[i]), SCIPgetNVars(pricerdata->pricingprobs[representative])) ); /*lint !e866*/
         SCIP_CALL( SCIPallocBufferArray(scip, &(pricingvals[i]), SCIPgetNVars(pricerdata->pricingprobs[representative])) ); /*lint !e866*/
      }
   }

   /* get solution values */
   SCIP_CALL( SCIPgetSolVals(scip, origsol, norigvars, origvars, origsolvals) );
   SCIP_CALL( SCIPcreateSol(scip, &mastersol, NULL) );

   /* store variables and solutions into arrays */
   for( i = 0; i < norigvars; i++ )
   {
      int blocknr;
      assert(GCGvarIsOriginal(origvars[i]));
      blocknr = GCGvarGetBlock(origvars[i]);
      assert(blocknr < 0 || GCGoriginalVarGetPricingVar(origvars[i]) != NULL);

      if( blocknr >= 0 )
      {
         if( pricerdata != NULL && !SCIPisZero(scip, origsolvals[i]) )
         {
            pricingvars[blocknr][npricingvars[blocknr]] = GCGoriginalVarGetPricingVar(origvars[i]);
            pricingvals[blocknr][npricingvars[blocknr]] = origsolvals[i];
            npricingvars[blocknr]++;
         }
      }
      else
      {
         SCIP_VAR* mastervar;

         assert((GCGoriginalVarGetNMastervars(origvars[i]) == 1) || (GCGoriginalVarIsLinking(origvars[i])));
         assert(GCGoriginalVarGetMastervars(origvars[i])[0] != NULL);

         mastervar = GCGoriginalVarGetMastervars(origvars[i])[0];

         if( SCIPisEQ(scip, SCIPvarGetUbGlobal(mastervar), SCIPvarGetLbGlobal(mastervar)) )
         {
            SCIP_CALL( SCIPsetSolVal(scip, mastersol, mastervar, SCIPvarGetUbGlobal(mastervar)) );
         }
         else
         {
            SCIP_CALL( SCIPsetSolVal(scip, mastersol, mastervar, origsolvals[i]) );
         }

         if( GCGoriginalVarIsLinking(origvars[i]) )
         {
            if( pricerdata != NULL )
            {
               if( !SCIPisZero(scip, origsolvals[i]) )
               {
                  int* blocks = NULL;
                  int nblocks = GCGlinkingVarGetNBlocks(origvars[i]);
                  SCIP_CALL( SCIPallocBufferArray(scip, &blocks, nblocks) ); /*lint !e530*/
                  SCIP_CALL( GCGlinkingVarGetBlocks(origvars[i], nblocks, blocks) );
                  for ( j = 0; j < nblocks; ++j)
                  {
                     prob = blocks[j];

                     pricingvars[prob][npricingvars[prob]] = GCGlinkingVarGetPricingVars(origvars[i])[prob];
                     pricingvals[prob][npricingvars[prob]] = origsolvals[i];
                     npricingvars[prob]++;
                  }
                  SCIPfreeBufferArray(scip, &blocks);

               }
            }
            else
            {
               if( SCIPisEQ(scip, SCIPvarGetUbGlobal(mastervar), SCIPvarGetLbGlobal(mastervar)) )
               {
                  SCIP_CALL( SCIPsetSolVal(scip, mastersol, mastervar, SCIPvarGetUbGlobal(mastervar)) );
               }
               else
               {
                  SCIP_CALL( SCIPsetSolVal(scip, mastersol, mastervar, origsolvals[i]) );
               }
            }
         }
      }
   }

   /* create variables in the master problem */
   if( pricerdata != NULL )
   {
      for( prob = 0; prob < pricerdata->npricingprobs; prob++ )
      {
         int representative;

         representative = GCGgetBlockRepresentative(origprob, prob);

         SCIP_CALL( pricer->createNewMasterVar(scip, NULL, NULL, pricingvars[prob], pricingvals[prob], npricingvars[prob], FALSE, representative, TRUE, &added, &newvar) );
         assert(added);

         SCIP_CALL( SCIPsetSolVal(scip, mastersol, newvar, 1.0) );
      }
   }

#ifdef SCIP_DEBUG
   SCIP_CALL( SCIPtrySolFree(scip, &mastersol, TRUE, TRUE, TRUE, TRUE, TRUE, &added) );
#else
   SCIP_CALL( SCIPtrySolFree(scip, &mastersol, FALSE, FALSE, TRUE, TRUE, TRUE, &added) );
#endif

   /* set external pointer if it is not NULL */
   if( stored != NULL )
      *stored = added;

   /* free memory for storing variables and solution values from the solution */
   if( pricerdata != NULL )
   {
      for( i = pricerdata->npricingprobs - 1; i>= 0; i-- )
      {
         SCIPfreeBufferArray(scip, &(pricingvals[i]));
         SCIPfreeBufferArray(scip, &(pricingvars[i]));
      }

      SCIPfreeBufferArray(scip, &npricingvars);
      SCIPfreeBufferArray(scip, &pricingvals);
      SCIPfreeBufferArray(scip, &pricingvars);
   }
   SCIPfreeBufferArray(scip, &origsolvals);

   return SCIP_OKAY;
}


/** create initial master variables */
extern "C"
SCIP_RETCODE GCGmasterCreateInitialMastervars(
   SCIP*                 scip                /**< master SCIP data structure */
   )
{
   SCIP* origprob;
   int i;
   SCIP_VAR** vars;
   int nvars;
   int npricingprobs;
   int v;

   assert(scip != NULL);

   origprob = GCGgetOriginalprob(scip);
   assert(origprob != NULL);

   npricingprobs = GCGgetNPricingprobs(origprob);
   assert(npricingprobs >= 0);

   /* for variables in the original problem that do not belong to any block,
    * create the corresponding variable in the master problem
    */
   vars = SCIPgetVars(origprob);
   nvars = SCIPgetNVars(origprob);
   for( v = 0; v < nvars; v++ )
   {
      SCIP_Real* coefs;
      int blocknr;
      int ncoefs;
      SCIP_VAR* var;

      /* var = SCIPvarGetProbvar(vars[v]); */
      var = vars[v];
      blocknr = GCGvarGetBlock(var);
      coefs = GCGoriginalVarGetCoefs(var);
      ncoefs = GCGoriginalVarGetNCoefs(var);

      assert(GCGvarIsOriginal(var));
      if( blocknr < 0 )
      {
         SCIP_CONS** linkconss;
         SCIP_VAR* newvar;

         SCIP_CALL( GCGcreateInitialMasterVar(scip, var, &newvar) );
         SCIP_CALL( SCIPaddVar(scip, newvar) );

         SCIP_CALL( GCGoriginalVarAddMasterVar(origprob, var, newvar, 1.0) );

         linkconss = GCGoriginalVarGetMasterconss(var);

         /* add variable in the master to the master constraints it belongs to */
         for( i = 0; i < ncoefs; i++ )
         {
            assert(!SCIPisZero(scip, coefs[i]));

            SCIP_CALL( SCIPaddCoefLinear(scip, linkconss[i], newvar, coefs[i]) );
         }

         /* we copied a linking variable into the master, add it to the linkcons */
         if( GCGoriginalVarIsLinking(var) )
         {
            SCIP_CONS** linkingconss;
            linkingconss = GCGlinkingVarGetLinkingConss(var);

            /* the linking constraints could be NULL if the Benders' decomposition is used. */
            if( linkingconss != NULL )
            {
               for( i = 0; i < npricingprobs; i++ )
               {
                  if( linkingconss[i] != NULL )
                  {
                     SCIP_CALL( SCIPaddCoefLinear(scip, linkingconss[i], newvar, 1.0) );
                  }
               }
            }
         }

         SCIP_CALL( SCIPreleaseVar(scip, &newvar) );
      }
   }
   return SCIP_OKAY;
}

/** get root node degeneracy */
extern "C"
SCIP_Real GCGmasterGetDegeneracy(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   if( SCIPgetStage(scip) >= SCIP_STAGE_INITPRESOLVE && SCIPgetStage(scip) <= SCIP_STAGE_SOLVING && GCGisRootNode(scip) )
   {
      return pricerdata->avgrootnodedegeneracy;
   }
   else
      return SCIPinfinity(scip);
}

/** check if current sol is valid */
extern "C"
SCIP_Bool GCGmasterIsCurrentSolValid(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;
   SCIP_SOL* sol;
   int i;

   assert(scip != NULL);

   /* checking the decomposition mode. If Benders' is used, then the solution is assumed to be valid. */
   if( GCGgetMasterDecompMode(scip) != DEC_DECMODE_DANTZIGWOLFE )
      return TRUE;

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   if( pricerdata->nartificialvars == 0 )
      return TRUE;

   if( SCIPgetStage(scip) == SCIP_STAGE_SOLVING )
      sol = NULL;
   else if( SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL )
      sol = SCIPgetBestSol(scip);
   else
      return TRUE;

   for( i = 0; i < pricerdata->nartificialvars; ++i )
   {
      SCIP_Real solval;
      solval = SCIPgetSolVal(scip, sol, pricerdata->artificialvars[i]);

      if( SCIPisPositive(scip, solval) )
         return FALSE;
   }

   return TRUE;
}

extern "C"
SCIP_Bool GCGmasterIsBestsolValid(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;
   SCIP_SOL* sol;
   int i;

   assert(scip != NULL);

   /* checking the decomposition mode. If Benders' is used, then the solution is assumed to be valid. */
   if( GCGgetMasterDecompMode(scip) != DEC_DECMODE_DANTZIGWOLFE )
      return TRUE;

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   sol = SCIPgetBestSol(scip);

   if( sol == NULL )
      return TRUE;

   for( i = 0; i < pricerdata->nartificialvars; ++i )
   {
      SCIP_Real solval;
      solval = SCIPgetSolVal(scip, sol, pricerdata->artificialvars[i]);

      if( SCIPisPositive(scip, solval) )
         return FALSE;
   }

   return TRUE;
}

extern "C"
SCIP_Bool GCGmasterIsSolValid(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SOL*             mastersol           /**< solution of the master problem, or NULL for current LP solution */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;
   int i;

   assert(scip != NULL);

   /* checking the decomposition mode. If Benders' is used, then the solution is assumed to be valid. */
   if( GCGgetMasterDecompMode(scip) != DEC_DECMODE_DANTZIGWOLFE )
      return TRUE;

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   for( i = 0; i < pricerdata->nartificialvars; ++i )
   {
      SCIP_Real solval;
      solval = SCIPgetSolVal(scip, mastersol, pricerdata->artificialvars[i]);

      if( SCIPisPositive(scip, solval) )
         return FALSE;
   }

   return TRUE;
}


/* get number of iterations in pricing problems */
extern "C"
SCIP_Longint GCGmasterGetPricingSimplexIters(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   return pricerdata->pricingiters;
}

/** print simplex iteration statistics */
extern "C"
SCIP_RETCODE GCGmasterPrintSimplexIters(
   SCIP*                 scip,               /**< SCIP data structure */
   FILE*                 file                /**< output file */
   )
{
   ObjPricerGcg* pricer;
   SCIP_PRICERDATA* pricerdata;

   assert(scip != NULL);

   pricer = static_cast<ObjPricerGcg*>(SCIPfindObjPricer(scip, PRICER_NAME));
   assert(pricer != NULL);

   pricerdata = pricer->getPricerdata();
   assert(pricerdata != NULL);

   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "Simplex iterations :       iter\n");
   if( SCIPgetStage(scip) >= SCIP_STAGE_SOLVING )
   {
      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  Master LP        : %10lld\n", SCIPgetNLPIterations(scip));
   }
   else
   {
      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  Master LP        : %10lld\n", 0LL);
   }
   SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  Pricing LP       : %10lld\n", pricerdata->pricingiters);

   if( SCIPgetStage(pricer->getOrigprob()) >= SCIP_STAGE_SOLVING )
   {
      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  Original LP      : %10lld\n", SCIPgetNLPIterations(pricer->getOrigprob()));
   }
   else
   {
      SCIPmessageFPrintInfo(SCIPgetMessagehdlr(scip), file, "  Original LP      : %10lld\n", 0LL);
   }

   return SCIP_OKAY;
}
