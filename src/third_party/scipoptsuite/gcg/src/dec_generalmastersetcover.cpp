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

/**@file   dec_generalmastersetcover.cpp
 * @ingroup DETECTORS
 * @brief  detector for set covering constraints
 * @author Michael Bastubbe
 *
 * This detector sets the following constraints to master:
 * - set covering constraints
 * - logical OR constraints
 * - constraints with infinity rhs and nonnegative lhs
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "dec_generalmastersetcover.h"
#include "cons_decomp.h"
#include "class_partialdecomp.h"
#include "class_detprobdata.h"
#include "gcg.h"
#include "scip/cons_setppc.h"
#include "scip/scip.h"
#include "scip_misc.h"
#include "scip/clock.h"

#include <iostream>

/* constraint handler properties */
#define DEC_DETECTORNAME          "generalmastersetcover"       /**< name of detector */
#define DEC_DESC                  "detector generalmastersetcover" /**< description of detector*/
#define DEC_FREQCALLROUND         1           /** frequency the detector gets called in detection loop ,ie it is called in round r if and only if minCallRound <= r <= maxCallRound AND  (r - minCallRound) mod freqCallRound == 0 */
#define DEC_MAXCALLROUND          0     /** last round the detector gets called                              */
#define DEC_MINCALLROUND          0           /** first round the detector gets called                              */
#define DEC_FREQCALLROUNDORIGINAL 1           /** frequency the detector gets called in detection loop while detecting the original problem   */
#define DEC_MAXCALLROUNDORIGINAL  0     /** last round the detector gets called while detecting the original problem                            */
#define DEC_MINCALLROUNDORIGINAL  0           /** first round the detector gets called while detecting the original problem    */
#define DEC_PRIORITY              0           /**< priority of the constraint handler for separation */
#define DEC_DECCHAR               '?'         /**< display character of detector */
#define DEC_ENABLED               TRUE        /**< should the detection be enabled */
#define DEC_ENABLEDFINISHING      FALSE       /**< should the finishing be enabled */
#define DEC_ENABLEDPOSTPROCESSING FALSE          /**< should the postprocessing be enabled */
#define DEC_SKIP                  FALSE       /**< should detector be skipped if other detectors found decompositions */
#define DEC_USEFULRECALL          FALSE       /**< is it useful to call this detector on a descendant of the propagated partialdec */

/*
 * Data structures
 */

/** @todo fill in the necessary detector data */

/** detector handler data */
struct DEC_DetectorData
{
};

/*
 * Local methods
 */

/* put your local methods here, and declare them static */

/*
 * detector callback methods
 */

/** destructor of detector to free user data (called when GCG is exiting) */
#define freeGeneralmastersetcover NULL

/** destructor of detector to free detector data (called before the solving process begins) */
#define exitGeneralmastersetcover NULL

/** detection initialization function of detector (called before solving is about to begin) */
#define initGeneralmastersetcover NULL

static DEC_DECL_PROPAGATEPARTIALDEC(propagatePartialdecGeneralmastersetcover)
{
   *result = SCIP_DIDNOTFIND;

   char decinfo[SCIP_MAXSTRLEN];
   SCIP_CLOCK* temporaryClock;
   SCIP_CALL_ABORT(SCIPcreateClock(scip, &temporaryClock) );
   SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );

   SCIP_CONS* cons;
   SCIP_VAR** vars;
   SCIP_Real* vals;
   int nvars;
   bool relevant = true;


   gcg::PARTIALDECOMP* partialdec = partialdecdetectiondata->workonpartialdec;
   auto& openconss = partialdec->getOpenconssVec();
   for( auto itr = openconss.cbegin(); itr != openconss.cend(); )
   {
      bool found = false;
      cons = partialdecdetectiondata->detprobdata->getCons(*itr);

      /* set open setcovering and logicor constraints to master */
      if( GCGconsGetType(scip, cons) == setcovering || GCGconsGetType(scip, cons) == logicor )
      {
         itr = partialdec->fixConsToMaster(itr);
         found = true;
      }
      /* set constraints with infinity rhs and nonnegative lhs to master */
      else if(GCGconsGetType(scip, cons) != logicor && GCGconsGetType(scip, cons) != setpacking && GCGconsGetType(scip, cons) != setpartitioning )
      {
         nvars = GCGconsGetNVars(scip, cons);
         vars = NULL;
         vals = NULL;
         if( !SCIPisInfinity(scip, GCGconsGetRhs(scip, cons)) )
            relevant = false;
         if( SCIPisNegative(scip, GCGconsGetLhs(scip, cons)) )
            relevant = false;
         if( nvars > 0 )
         {
            SCIP_CALL( SCIPallocMemoryArray(scip, &vars, nvars) );
            SCIP_CALL( SCIPallocMemoryArray(scip, &vals, nvars) );
            SCIP_CALL( GCGconsGetVars(scip, cons, vars, nvars) );
            SCIP_CALL( GCGconsGetVals(scip, cons, vals, nvars) );
         }
         for( int j = 0; j < nvars && relevant; ++j )
         {
            assert(vars != NULL);
            assert(vals != NULL);

            if( !SCIPvarIsIntegral(vars[j]) && !SCIPvarIsBinary(vars[j]) )
            {
               SCIPdebugPrintf("(%s is not integral) ", SCIPvarGetName(vars[j]) );
               relevant = FALSE;
            }
            if( !SCIPisEQ(scip, vals[j], 1.0) )
            {
               SCIPdebugPrintf("(coeff for var %s is %.2f != 1.0) ", SCIPvarGetName(vars[j]), vals[j] );
               relevant = FALSE;
            }
         }
         SCIPfreeMemoryArrayNull(scip, &vals);
         SCIPfreeMemoryArrayNull(scip, &vars);

         if(relevant)
         {
            itr = partialdec->fixConsToMaster(itr);
            found = true;
         }
      }
      if( !found )
      {
         ++itr;
      }
   }

   partialdec->sort();
   SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock) );

   partialdecdetectiondata->detectiontime = SCIPgetClockTime(scip, temporaryClock);
   SCIP_CALL( SCIPallocMemoryArray(scip, &(partialdecdetectiondata->newpartialdecs), 1) );
   partialdecdetectiondata->newpartialdecs[0] = partialdec;
   partialdecdetectiondata->nnewpartialdecs = 1;
   (void) SCIPsnprintf(decinfo, SCIP_MAXSTRLEN, "genmastersetcover");
   partialdecdetectiondata->newpartialdecs[0]->addDetectorChainInfo(decinfo);
   partialdecdetectiondata->newpartialdecs[0]->addClockTime(SCIPgetClockTime(scip, temporaryClock));
   // we used the provided partialdec -> prevent deletion
   partialdecdetectiondata->workonpartialdec = NULL;
   SCIP_CALL_ABORT(SCIPfreeClock(scip, &temporaryClock) );

   *result = SCIP_SUCCESS;

   return SCIP_OKAY;
}

#define finishPartialdecGeneralmastersetcover NULL
#define detectorPostprocessPartialdecGeneralmastersetcover NULL

static
DEC_DECL_SETPARAMAGGRESSIVE(setParamAggressiveGeneralmastersetcover)
{
   char setstr[SCIP_MAXSTRLEN];
   const char* name = DECdetectorGetName(detector);
   int newval;

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, TRUE) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE ) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxcallround", name);
   SCIP_CALL( SCIPgetIntParam(scip, setstr, &newval) );
   ++newval;
   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPinfoMessage(scip, NULL, "After Setting %s = %d\n", setstr, newval);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/origmaxcallround", name);
   SCIP_CALL( SCIPgetIntParam(scip, setstr, &newval) );
   ++newval;
   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPinfoMessage(scip, NULL, "%s = %d\n", setstr, newval);

   return SCIP_OKAY;
}


static
DEC_DECL_SETPARAMDEFAULT(setParamDefaultGeneralmastersetcover)
{
   char setstr[SCIP_MAXSTRLEN];

   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, DEC_ENABLED) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, DEC_ENABLEDFINISHING ) );

   return SCIP_OKAY;
}

static
DEC_DECL_SETPARAMFAST(setParamFastGeneralmastersetcover)
{
   char setstr[SCIP_MAXSTRLEN];

   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE ) );

   return SCIP_OKAY;
}



/*
 * detector specific interface methods
 */

/** creates the handler for generalmastersetcover detector and includes it in SCIP */
SCIP_RETCODE SCIPincludeDetectorGeneralmastersetcover(SCIP* scip /**< SCIP data structure */
)
{
   DEC_DETECTORDATA* detectordata;

   /**@todo create generalmastersetcover detector data here*/
   detectordata = NULL;

   SCIP_CALL(
      DECincludeDetector(scip, DEC_DETECTORNAME, DEC_DECCHAR, DEC_DESC, DEC_FREQCALLROUND, DEC_MAXCALLROUND, DEC_MINCALLROUND,
         DEC_FREQCALLROUNDORIGINAL, DEC_MAXCALLROUNDORIGINAL, DEC_MINCALLROUNDORIGINAL, DEC_PRIORITY,
         DEC_ENABLED, DEC_ENABLEDFINISHING,DEC_ENABLEDPOSTPROCESSING, DEC_SKIP, DEC_USEFULRECALL,
         detectordata, freeGeneralmastersetcover, initGeneralmastersetcover, exitGeneralmastersetcover, propagatePartialdecGeneralmastersetcover, finishPartialdecGeneralmastersetcover, detectorPostprocessPartialdecGeneralmastersetcover, setParamAggressiveGeneralmastersetcover, setParamDefaultGeneralmastersetcover, setParamFastGeneralmastersetcover));

   /**@todo add generalmastersetcover detector parameters */

   return SCIP_OKAY;
}









