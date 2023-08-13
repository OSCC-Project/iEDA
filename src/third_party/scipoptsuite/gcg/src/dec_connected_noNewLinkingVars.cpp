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

/**@file   dec_connected_noNewLinkingVars.cpp
 * @ingroup DETECTORS
 * @brief  detector connected_noNewLinkingVars (assigns all dependent open conss and vars and completes the partialdec by bfs)
 * @author Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "dec_connected_noNewLinkingVars.h"
#include "cons_decomp.h"
#include "gcg.h"
#include "class_partialdecomp.h"
#include "class_detprobdata.h"
#include "scip/scip.h"
#include "scip_misc.h"
#include "scip/clock.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>

/* constraint handler properties */
#define DEC_DETECTORNAME          "connected_nonewlinkingvars"       /**< name of detector */
#define DEC_DESC                  "detector connected_noNewLinkingVars" /**< description of detector*/
#define DEC_FREQCALLROUND         1           /** frequency the detector gets called in detection loop ,ie it is called in round r if and only if minCallRound <= r <= maxCallRound AND  (r - minCallRound) mod freqCallRound == 0 */
#define DEC_MAXCALLROUND          INT_MAX     /** last round the detector gets called                              */
#define DEC_MINCALLROUND          0           /** first round the detector gets called                              */
#define DEC_FREQCALLROUNDORIGINAL 1           /** frequency the detector gets called in detection loop while detecting the original problem   */
#define DEC_MAXCALLROUNDORIGINAL  INT_MAX     /** last round the detector gets called while detecting the original problem                            */
#define DEC_MINCALLROUNDORIGINAL  0           /** first round the detector gets called while detecting the original problem    */
#define DEC_PRIORITY              0           /**< priority of the constraint handler for separation */
#define DEC_DECCHAR               '?'         /**< display character of detector */
#define DEC_ENABLED               FALSE        /**< should the detection be enabled */
#define DEC_ENABLEDFINISHING      FALSE        /**< should the finishing be enabled */
#define DEC_ENABLEDPOSTPROCESSING FALSE          /**< should the finishing be enabled */
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
#define freeConnected_noNewLinkingVars NULL

/** destructor of detector to free detector data (called before the solving process begins) */
#define exitConnected_noNewLinkingVars NULL

/** detection initialization function of detector (called before solving is about to begin) */
#define initConnected_noNewLinkingVars NULL



/** detection function for partialdecs */
static
SCIP_RETCODE detection(
   SCIP*                   scip,                         /**< SCIP data structure */
   Partialdec_Detection_Data* partialdecdetectiondata    /**< partialdecdetectiondata (including the detprobdata and workonpartialdec) where to store the new Partialdecs */
)
{
   SCIP_CLOCK* temporaryClock;
   SCIP_CALL_ABORT( SCIPcreateClock(scip, &temporaryClock) );
   SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );

   gcg::PARTIALDECOMP* partialdec = partialdecdetectiondata->workonpartialdec;

   partialdec->considerImplicits();

   //assign all dependent open vars and conss
   partialdec->refineToBlocks();

   //complete the partialdec by bfs
   partialdec->completeByConnected();

   SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock) );

   partialdecdetectiondata->detectiontime = SCIPgetClockTime(scip, temporaryClock);
   partialdecdetectiondata->nnewpartialdecs = 1;
   SCIP_CALL( SCIPallocMemoryArray(scip, &(partialdecdetectiondata->newpartialdecs), 1) );
   partialdecdetectiondata->newpartialdecs[0] = partialdec;
   partialdecdetectiondata->newpartialdecs[0]->addDetectorChainInfo(DEC_DETECTORNAME);
   partialdecdetectiondata->newpartialdecs[0]->addClockTime(SCIPgetClockTime(scip, temporaryClock));
   // we used the provided partialdec -> prevent deletion
   partialdecdetectiondata->workonpartialdec = NULL;
   SCIP_CALL_ABORT(SCIPfreeClock(scip, &temporaryClock) );

   return SCIP_OKAY;
}


static
DEC_DECL_PROPAGATEPARTIALDEC(propagatePartialdecConnected_noNewLinkingVars)
{
   *result = SCIP_DIDNOTFIND;
   detection(scip, partialdecdetectiondata);
   *result = SCIP_SUCCESS;

   return SCIP_OKAY;
}

static
DEC_DECL_FINISHPARTIALDEC(finishPartialdecConnected_noNewLinkingVars)
{
   *result = SCIP_DIDNOTFIND;
   detection(scip, partialdecdetectiondata);
   *result = SCIP_SUCCESS;

   return SCIP_OKAY;
}
#define detectorPostprocessPartialdecConnected_noNewLinkingVars NULL


#define setParamAggressiveConnected_noNewLinkingVars NULL
#define setParamDefaultConnected_noNewLinkingVars NULL
#define setParamFastConnected_noNewLinkingVars NULL



/*
 * detector specific interface methods
 */

/** creates the handler for connected_noNewLinkingVars detector and includes it in SCIP */
SCIP_RETCODE SCIPincludeDetectorConnected_noNewLinkingVars(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   DEC_DETECTORDATA* detectordata;

   /**@todo create connected_noNewLinkingVars detector data here*/
   detectordata = NULL;

   SCIP_CALL( DECincludeDetector(scip, DEC_DETECTORNAME, DEC_DECCHAR, DEC_DESC, DEC_FREQCALLROUND, DEC_MAXCALLROUND, DEC_MINCALLROUND, DEC_FREQCALLROUNDORIGINAL, DEC_MAXCALLROUNDORIGINAL, DEC_MINCALLROUNDORIGINAL, DEC_PRIORITY, DEC_ENABLED, DEC_ENABLEDFINISHING, DEC_ENABLEDPOSTPROCESSING, DEC_SKIP, DEC_USEFULRECALL, detectordata, freeConnected_noNewLinkingVars, initConnected_noNewLinkingVars, exitConnected_noNewLinkingVars, propagatePartialdecConnected_noNewLinkingVars, finishPartialdecConnected_noNewLinkingVars, detectorPostprocessPartialdecConnected_noNewLinkingVars, setParamAggressiveConnected_noNewLinkingVars, setParamDefaultConnected_noNewLinkingVars, setParamFastConnected_noNewLinkingVars) );

   /**@todo add connected_noNewLinkingVars detector parameters */

   return SCIP_OKAY;
}
