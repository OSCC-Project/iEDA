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

/**@file   dec_dbscan.cpp
 * @ingroup DETECTORS
 * @brief  detector DBSCAN
 * @author Igor Pesic
 *
 * @note requires package to be installed: GSL library, requires flag to be set: `GSL=true`
 *
 * This detector performs DBSCAN clustering.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#include <time.h>     // for measuring time performance
#include <string>
#include "dec_dbscan.h"
#include "cons_decomp.h"
#include "graph/matrixgraph.h"
#include "graph/rowgraph_weighted.h"
#include "graph/graph_gcg.h"
#include "scip/clock.h"
#include "iostream"

using gcg::RowGraphWeighted;
using gcg::Weights;
using gcg::GraphGCG;


/* constraint handler properties */
#define DEC_DETECTORNAME          "dbscan"    /**< name of detector */
#define DEC_DESC                  "detector based on DBSCAN clustering"  /**< description of detector */
#define DEC_PRIORITY              901         /**< priority of the constraint handler for separation */
#define DEC_FREQCALLROUND         1           /**< frequency the detector gets called in detection loop, i.e. it is called in round r if and only if minCallRound <= r <= maxCallRound AND  (r - minCallRound) mod freqCallRound == 0 */
#define DEC_MAXCALLROUND          INT_MAX     /**< last round the detector gets called */
#define DEC_MINCALLROUND          0           /**< first round the detector gets called */
#define DEC_FREQCALLROUNDORIGINAL 1           /**< frequency the detector gets called in detection loop while detecting the original problem */
#define DEC_MAXCALLROUNDORIGINAL  INT_MAX     /**< last round the detector gets called while detecting the original problem */
#define DEC_MINCALLROUNDORIGINAL  0           /**< first round the detector gets called while detecting the original problem  */
#define DEC_DECCHAR               'D'         /**< display character of detector */
#define DEC_ENABLED               FALSE       /**< should the detection be enabled */
#define DEC_ENABLEDFINISHING      FALSE       /**< should the finishing be enabled */
#define DEC_ENABLEDPOSTPROCESSING FALSE       /**< should the postprocessing be enabled */
#define DEC_SKIP                  FALSE       /**< should detector be skipped if other detectors found decompositions */
#define DEC_USEFULRECALL          FALSE       /**< is it useful to call this detector on a descendant of the propagated partialdec */


/* Default parameter settings*/
#define DEFAULT_N_ITERATIONS              51
#define DEFAULT_JOHNSON_ENABLE            true
#define DEFAULT_INTERSECTION_ENABLE       false
#define DEFAULT_JACCARD_ENABLE            false
#define DEFAULT_COSINE_ENABLE             false
#define DEFAULT_SIMPSON_ENABLE            false
#define DEFAULT_POSTPROC_ENABLE           true
#define MAX_N_BLOCKS                      100

/*
 * Data structures
 */

/** detector handler data */
struct DEC_DetectorData
{
   SCIP_RESULT result;                                 /**< result pointer to indicate success or failure */
   SCIP_Bool found;
   int n_iterations;
   int n_similarities;                                  /**< number of active similarities */
   SCIP_Bool johnsonenable;
   SCIP_Bool intersectionenable;
   SCIP_Bool jaccardenable;
   SCIP_Bool cosineenable;
   SCIP_Bool simpsonenable;
   SCIP_Bool postprocenable;
};


/*
 * Local methods
 */

static std::vector<double> getEpsList(int length, double mid, bool isintersection)
{
   int n1, n2;
   if(isintersection)
   {
      n1 = (int) round((length+1) / 2.0);
      n2 = n1;
   }
   else
   {
      n2 = (int) round((length+1) / 4.0);
      n1 = abs(length - n2) + 1;
   }

   double s = mid;
   double end1 = mid + 0.9;      // lower boundary
   double end2 = mid + 0.4;      // upper boundary

   double q1 = pow(end1/s, 1.0/(double)(n1-1));
   double q2 = pow(end2/s, 1.0/(double)(n2-1));

   std::vector<double> geom_seq1(n1-1);
   std::vector<double> geom_seq2(n2);

   int j = 0;
   for(int i = n1 - 1; i > 0; i--)
   {
      geom_seq1[j] = 2*s-s*pow(q1, (double)i);
      j++;
   }
   for(int i = 0; i < n2; i++)
   {
      geom_seq2[i] = s*pow(q2, (double)i);
   }

   geom_seq1.insert( geom_seq1.end(), geom_seq2.begin(), geom_seq2.end() );

   assert((int)geom_seq1.size() == length);

   return geom_seq1;
}

/*
 * detector callback methods
 */

/** destructor of detector to free user data (called when GCG is exiting) */
static
DEC_DECL_FREEDETECTOR(freeDBSCAN)
{
   DEC_DETECTORDATA* detectordata;

   assert(scip != NULL);

   detectordata = DECdetectorGetData(detector);
   assert(detectordata != NULL);
   assert(strcmp(DECdetectorGetName(detector), DEC_DETECTORNAME) == 0);

   SCIPfreeMemory(scip, &detectordata);

   return SCIP_OKAY;
}

/** destructor of detector to free detector data (called before the solving process begins) */
static
DEC_DECL_EXITDETECTOR(exitDBSCAN)
{
   return SCIP_OKAY;
}


/** detection initialization function of detector (called before solving is about to begin) */
static
DEC_DECL_INITDETECTOR(initDBSCAN)
{  /*lint --e{715}*/

   DEC_DETECTORDATA* detectordata;
   assert(scip != NULL);


   detectordata = DECdetectorGetData(detector);
   assert(detectordata != NULL);
   assert(strcmp(DECdetectorGetName(detector), DEC_DETECTORNAME) == 0);

   detectordata->n_similarities = -1;
   detectordata->found = FALSE;
   return SCIP_OKAY;
}

/** are there conss and vars to be included by the graph and have the conss common vars included by the graph */
static
bool graphCompletible(
   gcg::DETPROBDATA*  detprobdata,
   gcg::PARTIALDECOMP*      partialdec
   )
{
   bool completible;

   //have the open conss open vars?
   for(int c = 0; c < partialdec->getNOpenconss() && !completible; ++c)
   {
      int cons = partialdec->getOpenconss()[c];
      for(int v = 0; v < partialdec->getNOpenvars() && !completible; ++v)
      {
         int var = partialdec->getOpenvars()[v];
         for(int i = 0; i < detprobdata->getNVarsForCons(cons) && !completible; ++i)
         {
            if(var == detprobdata->getVarsForCons(cons)[i])
            {
               completible = true;
            }
         }
      }
   }
   if(!completible)
      return false;

   //have the open conss common open vars?
   for(int c = 0; c < partialdec->getNOpenconss(); ++c)
   {
      int cons1 = partialdec->getOpenconss()[c];
      for(int d = c + 1; d < partialdec->getNOpenconss(); ++d)
      {
         int cons2 = partialdec->getOpenconss()[d];
         for(int v = 0; v < detprobdata->getNVarsForCons(cons1); ++v)
         {
            int var1 = detprobdata->getVarsForCons(cons1)[v];
            if(!partialdec->isVarOpenvar(var1))
               continue;
            for(int w = 0; w < detprobdata->getNVarsForCons(cons2); ++w)
            {
               int var2 = detprobdata->getVarsForCons(cons2)[w];
               if(var1 == var2)
                  return true;
            }
         }
      }
   }
   return false;
}


static
DEC_DECL_PROPAGATEPARTIALDEC(propagatePartialdecDBSCAN)
{ /*lint --e{715}*/

   int nnewpartialdecs;
   gcg::PARTIALDECOMP* partialdec;
   DEC_DETECTORDATA* detectordata = DECdetectorGetData(detector);
   std::vector<SCIP_Real> clockTimes1;        /* vector containing times in seconds  */
   std::vector<SCIP_Real> clockTimes2;        /* vector containing times in seconds  */
   std::vector< RowGraphWeighted<GraphGCG>*> graphs;
   SCIP_CLOCK* overallClock;

   assert(scip != NULL);
   assert(detectordata != NULL);
   *result = SCIP_DIDNOTFIND;

   SCIP_CALL_ABORT( SCIPcreateClock(scip, &overallClock) );
   SCIP_CALL_ABORT( SCIPstartClock(scip, overallClock) );

   partialdec = partialdecdetectiondata->workonpartialdec;
   partialdec->refineToBlocks();

   if( !graphCompletible(partialdecdetectiondata->detprobdata, partialdec) )
   {
      delete partialdec;
      partialdecdetectiondata->nnewpartialdecs = 0;
      SCIP_CALL_ABORT( SCIPstopClock(scip, overallClock) );
      partialdecdetectiondata->detectiontime = SCIPgetClockTime(scip, overallClock);
      SCIP_CALL_ABORT(SCIPfreeClock(scip, &overallClock) );
      *result = SCIP_SUCCESS;
      return SCIP_OKAY;
   }

   Weights w(1, 1, 1, 1, 1, 1);

   SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Detecting DBSCAN structure:");

   time_t start, cp0, d_s, d_e;
   time(&start);

   std::vector<std::string> sim;
   SCIP_CLOCK* temporaryClock;

   SCIP_CALL_ABORT( SCIPcreateClock(scip, &temporaryClock) );

   if( detectordata->johnsonenable )
   {
      SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );
      RowGraphWeighted<GraphGCG>* g = new RowGraphWeighted<GraphGCG>(scip, w);
      SCIP_CALL( g->createFromPartialMatrix(partialdecdetectiondata->detprobdata, partialdec, gcg::DISTANCE_MEASURE::JOHNSON, gcg::WEIGHT_TYPE::DIST));
      graphs.push_back(g);
      sim.emplace_back("Johnson");
      SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock ) );
      clockTimes1.push_back(SCIPgetClockTime( scip, temporaryClock));
      SCIP_CALL_ABORT( SCIPresetClock(scip, temporaryClock ) );
   }
   if( detectordata->intersectionenable )
   {
      SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );
      RowGraphWeighted<GraphGCG>* g = new RowGraphWeighted<GraphGCG>(scip, w);
      SCIP_CALL( g->createFromPartialMatrix(partialdecdetectiondata->detprobdata, partialdec, gcg::DISTANCE_MEASURE::INTERSECTION, gcg::WEIGHT_TYPE::DIST));
      graphs.push_back(g);
      sim.emplace_back("Intersection");
      SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock ) );
      clockTimes1.push_back(SCIPgetClockTime( scip, temporaryClock));
      SCIP_CALL_ABORT( SCIPresetClock(scip, temporaryClock ) );
   }
   if( detectordata->jaccardenable )
   {
      SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );
      RowGraphWeighted<GraphGCG>* g = new RowGraphWeighted<GraphGCG>(scip, w);
      SCIP_CALL( g->createFromPartialMatrix(partialdecdetectiondata->detprobdata, partialdec, gcg::DISTANCE_MEASURE::JACCARD, gcg::WEIGHT_TYPE::DIST));
      graphs.push_back(g);
      sim.emplace_back("Jaccard");
      SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock ) );
      clockTimes1.push_back(SCIPgetClockTime( scip, temporaryClock));
      SCIP_CALL_ABORT( SCIPresetClock(scip, temporaryClock ) );
   }
   if( detectordata->cosineenable )
   {
      SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );
      RowGraphWeighted<GraphGCG>* g = new RowGraphWeighted<GraphGCG>(scip, w);
      SCIP_CALL( g->createFromPartialMatrix(partialdecdetectiondata->detprobdata, partialdec, gcg::DISTANCE_MEASURE::COSINE, gcg::WEIGHT_TYPE::DIST));
      graphs.push_back(g);
      sim.emplace_back("Cosine");
      SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock ) );
      clockTimes1.push_back(SCIPgetClockTime( scip, temporaryClock));
      SCIP_CALL_ABORT( SCIPresetClock(scip, temporaryClock ) );
   }
   if( detectordata->simpsonenable )
   {
      SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );
      RowGraphWeighted<GraphGCG>* g = new RowGraphWeighted<GraphGCG>(scip, w);
      SCIP_CALL( g->createFromPartialMatrix(partialdecdetectiondata->detprobdata, partialdec, gcg::DISTANCE_MEASURE::SIMPSON, gcg::WEIGHT_TYPE::DIST));
      graphs.push_back(g);
      sim.emplace_back("Simspon");
      SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock ) );
      clockTimes1.push_back(SCIPgetClockTime( scip, temporaryClock));
      SCIP_CALL_ABORT( SCIPresetClock(scip, temporaryClock ) );
   }
   time(&cp0);
   detectordata->n_similarities = (int) graphs.size();

   double q = 10; // quantile to search for the percentile needed for the mid of the eps list
   std::vector<double> mids(graphs.size());      // middle values for each eps list
   std::vector<std::vector<double> > epsLists(graphs.size());
   for( int i = 0; i < (int)graphs.size(); i++ )
   {
      SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );
      mids[i] = graphs.at(i)->getEdgeWeightPercentile(q);
      if( i == 1 && detectordata->intersectionenable )
      {
         epsLists[i] = getEpsList(detectordata->n_iterations, mids[i], true); // case for intersection
      }
      else
      {
         epsLists[i] = getEpsList(detectordata->n_iterations, mids[i], false); // case for all except intersection
      }
      SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock) );
      clockTimes2.push_back(SCIPgetClockTime( scip, temporaryClock));
      SCIP_CALL_ABORT( SCIPresetClock(scip, temporaryClock) );
   }

   int nMaxPartialdecs = detectordata->n_iterations * graphs.size();
   const int max_blocks = std::min((int)round(0.3 * SCIPgetNConss(scip)), MAX_N_BLOCKS);
   char decinfo[SCIP_MAXSTRLEN];

   SCIP_CALL( SCIPallocMemoryArray(scip, &(partialdecdetectiondata->newpartialdecs), 2 * nMaxPartialdecs) );
   nnewpartialdecs = 0;
   time(&d_s);
   for( int i = 0; i < (int)graphs.size(); i++ )
   {
      RowGraphWeighted<GraphGCG>* graph = graphs.at(i);
      int old_n_blocks = -1;
      int old_non_cl = -1;
      std::vector<gcg::PARTIALDECOMP*> createddecomps;
      std::vector<std::pair<double, SCIP_Real>> clockTimes3;
      gcg::PARTIALDECOMP *decomp1 = NULL, *decomp2 = NULL;

      SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );
      SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "\n  %s similarity:", sim[i].c_str());
      createddecomps.reserve(2 * epsLists[i].size());
      clockTimes3.reserve(epsLists[i].size());

      for( double eps : epsLists[i] )
      {
         if( eps <= 0.0 )
         {
            continue;
         }
         if( eps >= 1.0 )
         {
            break;
         }

         // run DBSCAN with different eps
         SCIP_CALL( graph->computePartitionDBSCANForPartialGraph(partialdecdetectiondata->detprobdata, partialdec, eps, detectordata->postprocenable) );

         int n_blocks;
         int non_cl;
         SCIP_CALL( graph->getNBlocks(n_blocks) );
         SCIP_CALL( graph->nonClustered(non_cl) );

         // skip the case if we have too many blocks (it means we must increase eps) or if the clustering is the same as the last one
         if( n_blocks > max_blocks || n_blocks == 0 || (n_blocks == old_n_blocks && non_cl == old_non_cl) )
         {
            continue;
         }
         // stop. eps is already too big
         if( n_blocks == 1 && non_cl == 0)
         {
            break;
         }
         SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "\n    Blocks: %d, Master Conss: %d/%d, ", n_blocks, non_cl, SCIPgetNConss(scip));
         old_n_blocks = n_blocks;
         old_non_cl = non_cl;

         SCIP_CALL( graph->createPartialdecFromPartition(partialdec, &decomp1, &decomp2, partialdecdetectiondata->detprobdata));
         SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock ) );

         if( decomp1 != NULL )
         {
            assert(decomp2 != NULL);
            detectordata->found = TRUE;
            createddecomps.push_back(decomp1);
            createddecomps.push_back(decomp2);
            clockTimes3.emplace_back(eps, SCIPgetClockTime(scip, temporaryClock));
         }

         SCIP_CALL_ABORT( SCIPresetClock(scip, temporaryClock ) );
      }

      size_t ncreateddecomps = createddecomps.size();
      for( unsigned int j = 0; j < ncreateddecomps; ++j )
      {
         double eps = clockTimes3[j / 2].first;
         SCIP_Real epstime = clockTimes3[j / 2].second;
         (void) SCIPsnprintf(decinfo, SCIP_MAXSTRLEN, "dbscan_%s_%f", sim[i].c_str(), eps);
         createddecomps[j]->addDetectorChainInfo(decinfo);
         createddecomps[j]->addClockTime(clockTimes1[i] / ncreateddecomps + clockTimes2[i] / ncreateddecomps + epstime / 2.0);
         partialdecdetectiondata->newpartialdecs[nnewpartialdecs + j] = createddecomps[j];
      }
      nnewpartialdecs += ncreateddecomps;

      delete graphs.at(i);
      graphs[i] = NULL;
   }

   SCIP_CALL( SCIPreallocMemoryArray(scip, &(partialdecdetectiondata->newpartialdecs), nnewpartialdecs) );
   partialdecdetectiondata->nnewpartialdecs = nnewpartialdecs;
   SCIP_CALL_ABORT( SCIPstopClock(scip, overallClock) );
   partialdecdetectiondata->detectiontime = SCIPgetClockTime(scip, overallClock);
   SCIP_CALL_ABORT(SCIPfreeClock(scip, &overallClock) );

   time(&d_e);
   double elapsed_graphs = difftime(cp0, start);
   double elapsed_dbscan = difftime(d_e, d_s);

   SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, " done, %d similarities used, %d partialdecs found.\n", detectordata->n_similarities, nnewpartialdecs);
   SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "DBSCAN Runtime: graphs: %.2lf, dbscan: %.2lf. \n", elapsed_graphs, elapsed_dbscan);

   *result = nnewpartialdecs > 0 ? SCIP_SUCCESS: SCIP_DIDNOTFIND;
   if( nnewpartialdecs == 0 )
   {
      SCIPfreeMemoryArrayNull(scip, &(partialdecdetectiondata->newpartialdecs));
   }
   SCIP_CALL_ABORT(SCIPfreeClock(scip, &temporaryClock) );
   return SCIP_OKAY;
}

#define finishPartialdecDBSCAN NULL
#define detectorPostprocessPartialdecDBSCAN NULL

static
DEC_DECL_SETPARAMAGGRESSIVE(setParamAggressiveDBSCAN)
{
   char setstr[SCIP_MAXSTRLEN];
   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE ) );

   return SCIP_OKAY;
}


static
DEC_DECL_SETPARAMDEFAULT(setParamDefaultDBSCAN)
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
DEC_DECL_SETPARAMFAST(setParamFastDBSCAN)
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

/** creates the handler for xyz detector and includes it in SCIP */
SCIP_RETCODE SCIPincludeDetectorDBSCAN(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
#if !defined(_WIN32) && !defined(_WIN64)
   DEC_DETECTORDATA *detectordata = NULL;
   assert(scip != NULL);

   SCIP_CALL( SCIPallocMemory(scip, &detectordata) );

   assert(detectordata != NULL);
   detectordata->found = FALSE;

   SCIP_CALL( DECincludeDetector(scip, DEC_DETECTORNAME, DEC_DECCHAR, DEC_DESC, DEC_FREQCALLROUND, DEC_MAXCALLROUND, DEC_MINCALLROUND, DEC_FREQCALLROUNDORIGINAL, DEC_MAXCALLROUNDORIGINAL, DEC_MINCALLROUNDORIGINAL, DEC_PRIORITY, DEC_ENABLED, DEC_ENABLEDFINISHING, DEC_ENABLEDPOSTPROCESSING, DEC_SKIP, DEC_USEFULRECALL,
      detectordata, freeDBSCAN, initDBSCAN, exitDBSCAN, propagatePartialdecDBSCAN, finishPartialdecDBSCAN, detectorPostprocessPartialdecDBSCAN, setParamAggressiveDBSCAN, setParamDefaultDBSCAN, setParamFastDBSCAN) );

   /* add arrowheur presolver parameters */
   SCIP_CALL( SCIPaddIntParam(scip, "detection/detectors/dbscan/niterations", "Number of iterations to run dbscan with different eps.", &detectordata->n_iterations, FALSE, DEFAULT_N_ITERATIONS, 11, 1001, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip, "detection/detectors/dbscan/johson", "Enable johson distance measure.", &detectordata->johnsonenable, FALSE, DEFAULT_JOHNSON_ENABLE, NULL, NULL ) );
   SCIP_CALL( SCIPaddBoolParam(scip, "detection/detectors/dbscan/intersection", "Enable intersection distance measure.", &detectordata->intersectionenable, FALSE, DEFAULT_INTERSECTION_ENABLE, NULL, NULL ) );
   SCIP_CALL( SCIPaddBoolParam(scip, "detection/detectors/dbscan/jaccard", "Enable jaccard distance measure.", &detectordata->jaccardenable, FALSE, DEFAULT_JACCARD_ENABLE, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip, "detection/detectors/dbscan/cosine", "Enable cosine distance measure.", &detectordata->cosineenable, FALSE, DEFAULT_COSINE_ENABLE, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip, "detection/detectors/dbscan/simpson", "Enable simpson distance measure.", &detectordata->simpsonenable, FALSE, DEFAULT_SIMPSON_ENABLE, NULL, NULL ) );
   SCIP_CALL( SCIPaddBoolParam(scip, "detection/detectors/dbscan/postprocenable", "Enable post-processing step.", &detectordata->postprocenable, FALSE, DEFAULT_POSTPROC_ENABLE, NULL, NULL ) );

#endif
   return SCIP_OKAY;
}
