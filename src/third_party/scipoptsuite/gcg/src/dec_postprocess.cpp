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

/**@file   dec_postprocess.cpp
 * @ingroup DETECTORS
 * @brief  checks if there are master constraints that can be assigned to one block (without any other changes)
 * @author Michael Bastubbe
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "dec_postprocess.h"
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
#define DEC_DETECTORNAME          "postprocess"       /**< name of detector */
#define DEC_DESC                  "detector postprocess" /**< description of detector*/
#define DEC_FREQCALLROUND         1           /**< frequency the detector gets called in detection loop ,ie it is called in round r if and only if minCallRound <= r <= maxCallRound AND  (r - minCallRound) mod freqCallRound == 0 */
#define DEC_MAXCALLROUND          INT_MAX     /**< last round the detector gets called                              */
#define DEC_MINCALLROUND          0           /**< first round the detector gets called                              */
#define DEC_PRIORITY              1000000     /**< priority of the constraint handler for separation */
#define DEC_FREQCALLROUNDORIGINAL 1           /**< frequency the detector gets called in detection loop while detecting the original problem   */
#define DEC_MAXCALLROUNDORIGINAL  INT_MAX     /**< last round the detector gets called while detecting the original problem                            */
#define DEC_MINCALLROUNDORIGINAL  0           /**< first round the detector gets called while detecting the original problem    */
#define DEC_DECCHAR               'p'         /**< display character of detector */
#define DEC_ENABLED               FALSE        /**< should the detection be enabled */
#define DEC_ENABLEDFINISHING      FALSE        /**< should the finishing be enabled */
#define DEC_ENABLEDPOSTPROCESSING TRUE          /**< should the postprocessing be enabled */
#define DEC_SKIP                  FALSE       /**< should detector be skipped if other detectors found decompositions */
#define DEC_USEFULRECALL          FALSE       /**< is it useful to call this detector on a descendant of the propagated partialdec */
#define DEFAULT_USECONSSADJ       TRUE
/*
 * Data structures
 */

/** @todo fill in the necessary detector data */

/** detector handler data */
struct DEC_DetectorData
{
   SCIP_Bool useconssadj;
};


/*
 * Local methods
 */

/* put your local methods here, and declare them static */


/*
 * detector callback methods
 */

/** destructor of detector to free user data (called when GCG is exiting) */
/** destructor of detector to free detector data (called when SCIP is exiting) */
static
DEC_DECL_FREEDETECTOR(freePostprocess)
{  /*lint --e{715}*/
   DEC_DETECTORDATA *detectordata;

   assert(scip != NULL);
   assert(detector != NULL);

   assert(strcmp(DECdetectorGetName(detector), DEC_DETECTORNAME) == 0);

   detectordata = DECdetectorGetData(detector);
   assert(detectordata != NULL);

   SCIPfreeMemory(scip, &detectordata);

   return SCIP_OKAY;
}




/** destructor of detector to free detector data (called before the solving process begins) */
#define exitPostprocess NULL

/** detection initialization function of detector (called before solving is about to begin) */
#define initPostprocess NULL

#define propagatePartialdecPostprocess NULL
#define finishPartialdecPostprocess NULL

static
DEC_DECL_POSTPROCESSPARTIALDEC(postprocessPartialdecPostprocess)
{
   *result = SCIP_DIDNOTFIND;

   SCIP_CLOCK* temporaryClock;
   SCIP_CALL_ABORT( SCIPcreateClock(scip, &temporaryClock) );
   SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );
   char decinfo[SCIP_MAXSTRLEN];
   SCIP_Bool success;
   SCIP_Bool byconssadj;

   gcg::PARTIALDECOMP* partialdec = partialdecdetectiondata->workonpartialdec;
   gcg::DETPROBDATA* detprobdata = partialdecdetectiondata->detprobdata;
   assert(partialdecdetectiondata->workonpartialdec->getDetprobdata() == detprobdata);

   SCIPgetBoolParam(scip, "detection/detectors/postprocess/useconssadj", &byconssadj);

   if ( byconssadj && !detprobdata->isConssAdjInitialized() )
      detprobdata->createConssAdjacency();

   //complete the partialdec by bfs
   if ( byconssadj )
   {
      success = FALSE;
      std::vector<int> constoreassign;
      std::vector<int> blockforconstoreassign;

      partialdec->sort();

      std::vector<int> blockforvar(partialdec->getNVars(), -1 );

      for( int b = 0; b < partialdec->getNBlocks(); ++b )
      {
         for( size_t j  = 0; j < (size_t) partialdec->getNVarsForBlock(b); ++j )
         {
            blockforvar[partialdec->getVarsForBlock(b)[j]] = b;
         }
      }

      for( int mc = 0; mc < partialdec->getNMasterconss(); ++mc )
      {
         int masterconsid = partialdec->getMasterconss()[mc];
         int hittenblock  = -1;

         SCIP_Bool hitsmastervar = FALSE;
         SCIP_Bool varhitsotherblock = FALSE;

         for( int var = 0; var < detprobdata->getNVarsForCons(masterconsid); ++var )
         {
            int varid = detprobdata->getVarsForCons(masterconsid)[var];
            if( partialdec->isVarMastervar(varid) )
            {
               hitsmastervar = TRUE;
               break;
            }

            if ( blockforvar[varid] != -1 )
            {
               if( hittenblock == -1 )
                  hittenblock = blockforvar[varid];
               else if( hittenblock != blockforvar[varid] )
               {
                  varhitsotherblock = TRUE;
                  break;
               }
            }
         }

         if( hitsmastervar || varhitsotherblock )
            continue;

         if ( hittenblock != -1 )
         {
            constoreassign.push_back(masterconsid);
            blockforconstoreassign.push_back(hittenblock);
         }
      }

      for( size_t i = 0; i < constoreassign.size() ; ++i )
      {
         partialdec->setConsToBlock(constoreassign[i], blockforconstoreassign[i]);
         partialdec->removeMastercons(constoreassign[i]);
      }

      if( !constoreassign.empty() )
         success = TRUE;

      partialdec->prepare();
   }
   else
      success = FALSE;

   if ( !success )
   {
     partialdecdetectiondata->nnewpartialdecs = 0;
     *result = SCIP_DIDNOTFIND;
     SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock ) );
     SCIP_CALL_ABORT(SCIPfreeClock(scip, &temporaryClock) );
     return SCIP_OKAY;
   }
   SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock ) );

   partialdecdetectiondata->detectiontime = SCIPgetClockTime(scip, temporaryClock);
   SCIP_CALL( SCIPallocMemoryArray(scip, &(partialdecdetectiondata->newpartialdecs), 1) );
   partialdecdetectiondata->newpartialdecs[0] = partialdec;
   partialdecdetectiondata->nnewpartialdecs = 1;
   (void) SCIPsnprintf(decinfo, SCIP_MAXSTRLEN, "postprocess");
   partialdecdetectiondata->newpartialdecs[0]->addDetectorChainInfo(decinfo);
   partialdecdetectiondata->newpartialdecs[0]->addClockTime(SCIPgetClockTime(scip, temporaryClock));
   SCIP_CALL_ABORT(SCIPfreeClock(scip, &temporaryClock) );
   // we used the provided partialdec -> prevent deletion
   partialdecdetectiondata->workonpartialdec = NULL;

   *result = SCIP_SUCCESS;

   return SCIP_OKAY;
}


static
DEC_DECL_SETPARAMAGGRESSIVE(setParamAggressivePostprocess)
{
   char setstr[SCIP_MAXSTRLEN];

   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, TRUE ) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/postprocessingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, TRUE) );


   return SCIP_OKAY;

}


static
DEC_DECL_SETPARAMDEFAULT(setParamDefaultPostprocess)
{
   char setstr[SCIP_MAXSTRLEN];

   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, DEC_ENABLED) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, DEC_ENABLEDFINISHING) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/postprocessingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, DEC_ENABLEDPOSTPROCESSING ) );


   return SCIP_OKAY;

}

static
DEC_DECL_SETPARAMFAST(setParamFastPostprocess)
{
   char setstr[SCIP_MAXSTRLEN];

   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE ) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/postprocessingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE ) );


   return SCIP_OKAY;

}



/*
 * detector specific interface methods
 */

/** creates the handler for postprocess detector and includes it in SCIP */
SCIP_RETCODE SCIPincludeDetectorPostprocess(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   DEC_DETECTORDATA* detectordata;

   /**@todo create postprocess detector data here*/
   detectordata = NULL;
   SCIP_CALL( SCIPallocMemory(scip, &detectordata) );
   assert(detectordata != NULL);

   detectordata->useconssadj = TRUE;

   SCIP_CALL( DECincludeDetector(scip, DEC_DETECTORNAME, DEC_DECCHAR, DEC_DESC, DEC_FREQCALLROUND, DEC_MAXCALLROUND, DEC_MINCALLROUND, DEC_FREQCALLROUNDORIGINAL, DEC_MAXCALLROUNDORIGINAL, DEC_MINCALLROUNDORIGINAL, DEC_PRIORITY, DEC_ENABLED, DEC_ENABLEDFINISHING, DEC_ENABLEDPOSTPROCESSING, DEC_SKIP, DEC_USEFULRECALL, detectordata, freePostprocess,
      initPostprocess, exitPostprocess, propagatePartialdecPostprocess, finishPartialdecPostprocess,
      postprocessPartialdecPostprocess, setParamAggressivePostprocess, setParamDefaultPostprocess, setParamFastPostprocess) );

   /* add consname detector parameters */
      /**@todo add postprocess detector parameters */
   SCIP_CALL( SCIPaddBoolParam(scip, "detection/detectors/postprocess/useconssadj", "should the constraint adjacency be used", &detectordata->useconssadj, FALSE, DEFAULT_USECONSSADJ, NULL, NULL) );


   return SCIP_OKAY;
}
