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

/**@file   dec_consclass.cpp
 * @ingroup DETECTORS
 * @brief  detector consclass (put your description here)
 * @author Michael Bastubbe
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "dec_consclass.h"
#include "cons_decomp.h"
#include "class_partialdecomp.h"
#include "class_detprobdata.h"
#include "class_conspartition.h"
#include "gcg.h"
#include "scip/cons_setppc.h"
#include "scip/scip.h"
#include "scip_misc.h"
#include "scip/clock.h"

#include <sstream>

#include <iostream>
#include <algorithm>

/* constraint handler properties */
#define DEC_DETECTORNAME          "consclass"       /**< name of detector */
#define DEC_DESC                  "detector consclass" /**< description of detector*/
#define DEC_FREQCALLROUND         1           /** frequency the detector gets called in detection loop ,ie it is called in round r if and only if minCallRound <= r <= maxCallRound AND  (r - minCallRound) mod freqCallRound == 0 */
#define DEC_MAXCALLROUND          0           /** last round the detector gets called                              */
#define DEC_MINCALLROUND          0           /** first round the detector gets called                              */
#define DEC_FREQCALLROUNDORIGINAL 1           /** frequency the detector gets called in detection loop while detecting the original problem   */
#define DEC_MAXCALLROUNDORIGINAL  INT_MAX     /** last round the detector gets called while detecting the original problem                            */
#define DEC_MINCALLROUNDORIGINAL  0           /** first round the detector gets called while detecting the original problem    */
#define DEC_PRIORITY              0           /**< priority of the constraint handler for separation */
#define DEC_DECCHAR               'c'         /**< display character of detector */
#define DEC_ENABLED               TRUE        /**< should the detection be enabled */
#define DEC_ENABLEDFINISHING      FALSE       /**< should the detection be enabled */
#define DEC_ENABLEDPOSTPROCESSING FALSE       /**< should the finishing be enabled */
#define DEC_SKIP                  FALSE       /**< should detector be skipped if other detectors found decompositions */
#define DEC_USEFULRECALL          FALSE       /**< is it useful to call this detector on a descendant of the propagated partialdec */

#define DEFAULT_MAXIMUMNCLASSES     5
#define AGGRESSIVE_MAXIMUMNCLASSES  9
#define FAST_MAXIMUMNCLASSES        3

#define SET_MULTIPLEFORSIZETRANSF   12500

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
#define freeConsclass NULL

/** destructor of detector to free detector data (called before the solving process begins) */
#define exitConsclass NULL

/** detection initialization function of detector (called before solving is about to begin) */
#define initConsclass NULL

#define detectConsclass NULL

#define finishPartialdecConsclass NULL

static DEC_DECL_PROPAGATEPARTIALDEC(propagatePartialdecConsclass)
{
   *result = SCIP_DIDNOTFIND;
   char decinfo[SCIP_MAXSTRLEN];

   SCIP_CLOCK* temporaryClock;

   SCIP_CALL_ABORT( SCIPcreateClock(scip, &temporaryClock) );
   SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );

   std::vector<gcg::PARTIALDECOMP*> foundpartialdecs(0);

   gcg::PARTIALDECOMP* partialdecOrig;
   gcg::PARTIALDECOMP* partialdec;

   int maximumnclasses;

   if( partialdecdetectiondata->detprobdata->getNConss() + partialdecdetectiondata->detprobdata->getNVars() >= 50000 )
      SCIPgetIntParam(scip, "detection/classification/maxnclassesperpartitionforlargeprobs", &maximumnclasses);
   else
      SCIPgetIntParam(scip, "detection/classification/maxnclassesperpartition", &maximumnclasses);

   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " in dec_consclass: there are %d different constraint classes   \n ",
                  partialdecdetectiondata->detprobdata->getNConsPartitions() );


   for( int classifierIndex = 0; classifierIndex < partialdecdetectiondata->detprobdata->getNConsPartitions(); ++classifierIndex )
   {
      gcg::ConsPartition* classifier = partialdecdetectiondata->detprobdata->getConsPartition(classifierIndex);
      std::vector<int> consclassindices_master;

      if( classifier->getNClasses() > maximumnclasses )
      {
         SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL,
            " the current consclass distribution includes %d classes but only %d are allowed for propagatePartialdec() of cons class detector\n",
            classifier->getNClasses(), maximumnclasses);
         continue;
      }

      SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, " the current constraint classifier \"%s\" consists of %d different classes   \n ", classifier->getName(), classifier->getNClasses() );

      partialdecOrig = partialdecdetectiondata->workonpartialdec;

      for( int i = 0; i < classifier->getNClasses(); ++ i )
      {
         if ( classifier->getClassDecompInfo(i) == gcg::ONLY_MASTER )
            consclassindices_master.push_back(i);
      }

      std::vector< std::vector<int> > subsetsOfConsclasses = classifier->getAllSubsets( true, false, false );

      for( auto& subset : subsetsOfConsclasses )
      {
         if( subset.empty() && consclassindices_master.empty() )
            continue;

         partialdec = new gcg::PARTIALDECOMP(partialdecOrig);

         /* fix open conss that have a) type of the current subset or b) decomp info ONLY_MASTER as master conss */
         auto& openconss = partialdec->getOpenconssVec();
         for( auto itr = openconss.cbegin(); itr != openconss.cend(); )
         {
            bool foundCons = false;
            for( int consclassId : subset )
            {
               if( classifier->getClassOfCons(*itr) == consclassId )
               {
                  itr = partialdec->fixConsToMaster(itr);
                  foundCons = true;
                  break;
               }
            }
            /* only check consclassindices_master if current cons has not already been found in a subset */
            if ( !foundCons )
            {
               for(int consclassId : consclassindices_master )
               {
                  if( classifier->getClassOfCons(*itr) == consclassId )
                  {
                     itr = partialdec->fixConsToMaster(itr);
                     foundCons = true;
                     break;
                  }
               }
            }
            if( !foundCons )
            {
               ++itr;
            }
         }

         if( partialdec->getNOpenconss() < partialdecOrig->getNOpenconss() )
         {
            /* set decinfo to: consclass_<classfier_name>:<master_class_name#1>-...-<master_class_name#n> */
            std::stringstream decdesc;
            decdesc << "consclass" << "\\_" << classifier->getName() << ": \\\\ ";
            std::vector<int> curmasterclasses(consclassindices_master);
            for( size_t consclassId = 0; consclassId < subset.size(); ++consclassId )
            {
               if( consclassId > 0 )
               {
                  decdesc << "-";
               }
               decdesc << classifier->getClassName(subset[consclassId]);
               if( std::find(consclassindices_master.begin(), consclassindices_master.end(),
                             subset[consclassId]) == consclassindices_master.end())
               {
                  curmasterclasses.push_back(subset[consclassId]);
               }
            }
            for( size_t consclassId = 0; consclassId < consclassindices_master.size(); ++consclassId )
            {
               if( consclassId > 0 || !subset.empty())
               {
                  decdesc << "-";
               }
               decdesc << classifier->getClassName(consclassindices_master[consclassId]);
            }

            partialdec->sort();
            (void) SCIPsnprintf(decinfo, SCIP_MAXSTRLEN, decdesc.str().c_str());
            partialdec->addDetectorChainInfo(decinfo);
            partialdec->setConsPartitionStatistics(partialdec->getNDetectors(), classifier, curmasterclasses);

            foundpartialdecs.push_back(partialdec);
         }
         else
            delete partialdec;
      }
   }

   SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock) );

   partialdecdetectiondata->detectiontime = SCIPgetClockTime(scip, temporaryClock);
   SCIP_CALL( SCIPallocMemoryArray(scip, &(partialdecdetectiondata->newpartialdecs), foundpartialdecs.size()) );
   partialdecdetectiondata->nnewpartialdecs  = foundpartialdecs.size();

   SCIPverbMessage(scip, SCIP_VERBLEVEL_HIGH, NULL, "dec_consclass found %d new partialdecs \n", partialdecdetectiondata->nnewpartialdecs  );

   for( int s = 0; s < partialdecdetectiondata->nnewpartialdecs; ++s )
   {
      partialdecdetectiondata->newpartialdecs[s] = foundpartialdecs[s];
      partialdecdetectiondata->newpartialdecs[s]->addClockTime(partialdecdetectiondata->detectiontime / partialdecdetectiondata->nnewpartialdecs);
   }

   SCIP_CALL_ABORT(SCIPfreeClock(scip, &temporaryClock) );

   *result = SCIP_SUCCESS;

   return SCIP_OKAY;
}


#define detectorPostprocessPartialdecConsclass NULL

static
DEC_DECL_SETPARAMAGGRESSIVE(setParamAggressiveConsclass)
{
   char setstr[SCIP_MAXSTRLEN];
   SCIP_Real modifier;

   int newval;
   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, TRUE) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE ) );

   if( SCIPgetStage(scip) < SCIP_STAGE_PROBLEM )
   {
      return SCIP_OKAY;
   }

   modifier = ((SCIP_Real)SCIPgetNConss(scip) + (SCIP_Real)SCIPgetNVars(scip) ) / SET_MULTIPLEFORSIZETRANSF;
   modifier = log(modifier) / log(2.);

   if (!SCIPisFeasPositive(scip, modifier) )
      modifier = -1.;

   modifier = SCIPfloor(scip, modifier);

   newval = MAX( 6, AGGRESSIVE_MAXIMUMNCLASSES - modifier );
   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxnclasses", name);

   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "\n%s = %d\n", setstr, newval);


   return SCIP_OKAY;

}


static
DEC_DECL_SETPARAMDEFAULT(setParamDefaultConsclass)
{
   char setstr[SCIP_MAXSTRLEN];
   SCIP_Real modifier;

   int newval;
   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, DEC_ENABLED) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, DEC_ENABLEDFINISHING ) );

   if( SCIPgetStage(scip) < SCIP_STAGE_PROBLEM )
   {
      return SCIP_OKAY;
   }


   modifier = ( (SCIP_Real)SCIPgetNConss(scip) + (SCIP_Real)SCIPgetNVars(scip) ) / SET_MULTIPLEFORSIZETRANSF;
   modifier = log(modifier) / log(2);

   if (!SCIPisFeasPositive(scip, modifier) )
      modifier = -1.;

   modifier = SCIPfloor(scip, modifier);

   newval = MAX( 6, DEFAULT_MAXIMUMNCLASSES - modifier );
   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxnclasses", name);

   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "\n%s = %d\n", setstr, newval);

   return SCIP_OKAY;

}

static
DEC_DECL_SETPARAMFAST(setParamFastConsclass)
{
   char setstr[SCIP_MAXSTRLEN];
   SCIP_Real modifier;
   int newval;

   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, TRUE) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE ) );

   if( SCIPgetStage(scip) < SCIP_STAGE_PROBLEM )
   {
      return SCIP_OKAY;
   }


   modifier = ( (SCIP_Real)SCIPgetNConss(scip) + (SCIP_Real)SCIPgetNVars(scip) ) / SET_MULTIPLEFORSIZETRANSF;

   modifier = log(modifier) / log(2);

   if (!SCIPisFeasPositive(scip, modifier) )
      modifier = -1.;

   modifier = SCIPfloor(scip, modifier);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxnclasses", name);

   newval = MAX( 6, FAST_MAXIMUMNCLASSES - modifier );

   SCIP_CALL( SCIPsetIntParam(scip, setstr, newval ) );
   SCIPverbMessage(scip, SCIP_VERBLEVEL_DIALOG, NULL, "\n%s = %d\n", setstr, newval);

   return SCIP_OKAY;

}


/*
 * detector specific interface methods
 */

/** creates the handler for consclass detector and includes it in SCIP */
SCIP_RETCODE SCIPincludeDetectorConsclass(SCIP* scip /**< SCIP data structure */
)
{
   DEC_DETECTORDATA* detectordata;
   char setstr[SCIP_MAXSTRLEN];

   /**@todo create consclass detector data here*/
   detectordata = NULL;

   SCIP_CALL(
      DECincludeDetector(scip, DEC_DETECTORNAME, DEC_DECCHAR, DEC_DESC, DEC_FREQCALLROUND, DEC_MAXCALLROUND,
         DEC_MINCALLROUND, DEC_FREQCALLROUNDORIGINAL, DEC_MAXCALLROUNDORIGINAL, DEC_MINCALLROUNDORIGINAL, DEC_PRIORITY, DEC_ENABLED, DEC_ENABLEDFINISHING, DEC_ENABLEDPOSTPROCESSING, DEC_SKIP, DEC_USEFULRECALL, detectordata,
         freeConsclass, initConsclass, exitConsclass, propagatePartialdecConsclass, finishPartialdecConsclass, detectorPostprocessPartialdecConsclass, setParamAggressiveConsclass, setParamDefaultConsclass, setParamFastConsclass));

   /**@todo add consclass detector parameters */

   const char* name = DEC_DETECTORNAME;
   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/maxnclasses", name);
   SCIP_CALL( SCIPaddIntParam(scip, setstr, "maximum number of classes ",  NULL, FALSE, DEFAULT_MAXIMUMNCLASSES, 1, INT_MAX, NULL, NULL ) );

   return SCIP_OKAY;
}
