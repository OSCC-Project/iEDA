/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*          This file is part of the program and software framework          */
/*                    UG --- Ubquity Generator Framework                     */
/*                                                                           */
/*  Copyright Written by Yuji Shinano <shinano@zib.de>,                      */
/*            Copyright (C) 2021 by Zuse Institute Berlin,                   */
/*            licensed under LGPL version 3 or later.                        */
/*            Commercial licenses are available through <licenses@zib.de>    */
/*                                                                           */
/* This code is free software; you can redistribute it and/or                */
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
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.     */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file    scipParaInitiator.h
 * @brief   ParaInitiator extension for SCIP solver.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_INITIATOR_H__
#define __SCIP_PARA_INITIATOR_H__

#include <string>
#include "ug/paraDef.h"
#include "ug_bb/bbParaInitiator.h"
#include "scipParaComm.h"
#include "scipUserPlugins.h"
#include "scipDiffParamSet.h"
#include "scipParaInstance.h"
#include "scipParaSolution.h"
#include "scipParaDiffSubproblem.h"
#include "objscip/objscip.h"
#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#ifdef UG_WITH_UGS
#include "ugs/ugsParaCommMpi.h"
#endif

namespace ParaSCIP
{

/** Initiator class */
class ScipParaInitiator : public UG::BbParaInitiator
{
   UG::ParaParamSet     *paraParams;
   ScipParaInstance     *instance;
   ScipParaSolution     *solution;
   ScipDiffParamSet     *scipDiffParamSetRoot;
   ScipDiffParamSet     *scipDiffParamSet;
   SCIP_MESSAGEHDLR     *messagehdlr;
   FILE                 *logfile;
   FILE                 *solutionFile;
   FILE                 *transSolutionFile;
   SCIP                 *scip;
   char                 *probname;
   char                 *settingsNameLC;
   char                 *settingsNameRoot;
   char                 *settingsName;
   char                 *racingSettingsName;
   char                 *logname;
   char                 *isolname;
   char                 *generatedIsolname;
   char                 *solutionFileName;
   ScipUserPlugins      *userPlugins;
   SCIP_Real            finalDualBound;
   UG::FinalSolverState finalState;
   long long            nSolved;
   double               absgap;
   double               gap;
   double               objlimit;
#ifdef UG_WITH_UGS
   int                  seqNumber;
#endif

   bool addRootNodeCuts();
   void outputProblemInfo(int *nNonLinearConsHdlrs);
   bool onlyLinearConsHandler();

public:
   /** constructor */
   ScipParaInitiator(
         UG::ParaComm *inComm,
         UG::ParaTimer *inTimer
         )
         :  UG::BbParaInitiator(inComm, inTimer), paraParams(0), instance(0), solution(0), scipDiffParamSetRoot(0), scipDiffParamSet(0), messagehdlr(0), logfile(0),
            solutionFile(0), transSolutionFile(0), scip(0), probname(0), settingsNameLC(0), settingsNameRoot(0), settingsName(0), racingSettingsName(0),
            logname(0), isolname(0), generatedIsolname(0), solutionFileName(0), userPlugins(0), finalDualBound(-DBL_MAX), finalState(UG::Aborted), nSolved(0),
            absgap(-1.0), gap(-1.0), objlimit(DBL_MAX)
#ifdef UG_WITH_UGS
            , seqNumber(0)
#endif
   {
   }

   /** destructor */
   ~ScipParaInitiator(
         )
   {
      if( instance ) delete instance;
      if( solution ) delete solution;
      if( scipDiffParamSetRoot ) delete scipDiffParamSetRoot;
      if( scipDiffParamSet ) delete scipDiffParamSet;
      if( userPlugins ) delete userPlugins;
      if( generatedIsolname ) delete [] generatedIsolname;

      // message handler is mangaed within scip. It is freed at SCIPfree
#ifndef SCIP_THREADSAFE_MESSAGEHDLRS
      if( messagehdlr )
      {
         SCIP_CALL_ABORT( SCIPsetDefaultMessagehdlr() );
         SCIP_CALL_ABORT( SCIPfreeObjMessagehdlr(&messagehdlr) );
      }
#endif
      /******************
       * Close files *
       ******************/
      if( solutionFile )
      {
         fclose(solutionFile);
      }
      if( transSolutionFile )
      {
         fclose(transSolutionFile);
      }

      /********************
       * Deinitialization *
       ********************/
      if( !paraParams->getBoolParamValue(UG::Quiet) )
      {
         SCIP_CALL_ABORT( SCIPprintStatistics(scip, NULL) );    // output statistics (only for problem info)
      }
      if( scip )
      {
    	  SCIP_CALL_ABORT( SCIPfree(&scip) );
      }

      if( logfile != NULL )
         fclose(logfile);

      BMScheckEmptyMemory();
   }

   /** init function */
   int init(
         UG::ParaParamSet *paraParams,
         int          argc,
         char**       argv
         );


   int reInit(
         int nRestartedRacing
         );

   /** get instance */
   UG::ParaInstance *getParaInstance(
         )
   {
      return instance;
   }

   /** make DiffSubproblem object for root node */
   UG::BbParaDiffSubproblem *makeRootNodeDiffSubproblem(
         )
   {
      return 0;
   }

   /** try to set incumbent solution */
   bool tryToSetIncumbentSolution(UG::BbParaSolution *sol, bool checksol);

   /** send solver initialization message */
   void sendSolverInitializationMessage();

   /** generate racing ramp-up parameter sets */
   void generateRacingRampUpParameterSets(int nParamSets, UG::ParaRacingRampUpParamSet **racingRampUpParamSets);

   UG::BbParaSolution *getGlobalBestIncumbentSolution()
   {
      return solution;
   }

   int getNSolutions()
   {
      return SCIPgetNSols(scip);
   }

   /** convert an internal value to external value */
   double convertToExternalValue(
         double internalValue
         )
   {
      return instance->convertToExternalValue(internalValue);
   }

   /** convert an external value to internal value */
   double convertToInternalValue(
         double externalValue
         )
   {
      return instance->convertToInternalValue(externalValue);
   }

   /** get solution file name */
   char *getSolutionFileName(
         )
   {
      return solutionFileName;
   }

   /** get absgap */
   double getAbsgap(double dualBoundValue);

   /** get gap */
   double getGap(double dualBoundValue);

   /** get absgap value specified */
   double getAbsgapValue()
   {
      if( absgap < 0.0 )
      {
         SCIP_CALL_ABORT( SCIPgetRealParam(scip, "limits/absgap",&absgap ) );
      }
      return absgap;
   }

   /** get gap value specified */
   double getGapValue()
   {
      if( gap < 0.0 )
      {
         SCIP_CALL_ABORT( SCIPgetRealParam(scip, "limits/gap",&gap ) );
      }
      return gap;
   }

   /** get epsilon */
   double getEpsilon();

   /** write solution */
   void writeSolution(const std::string& message);

   /** write ParaInstance */
   void writeParaInstance(const std::string& filename);

   /** write solver runtime parameters */
   void writeSolverParameters(std::ostream *os);

#ifdef UG_WITH_ZLIB
   /** write checkpoint solution */
   void writeCheckpointSolution(const std::string& filename);

   /** read solution from checkpoint file */
   double readSolutionFromCheckpointFile(char *afterCheckpointingSolutionFileName);
#endif

   /** get solving status string */
   std::string getStatus();

   /** print solver version **/
   void printSolverVersion(std::ostream *os);   /**< output file (or NULL for standard output) */

   /** check if feasilbe soltuion exists or not */
   bool isFeasibleSolution()
   {
      return ( SCIPgetBestSol(scip) != NULL );
   }

   /** set initial stat on initiator */
   void accumulateInitialStat(UG::ParaInitialStat *initialStat);

   /** set initial stat on DiffSubproblem */
   void setInitialStatOnDiffSubproblem(int minDepth, int maxDepth, UG::BbParaDiffSubproblem *diffSubproblem);

   /** set final solver status */
   void setFinalSolverStatus(UG::FinalSolverState status);

   /** set number of nodes solved */
   void setNumberOfNodesSolved(long long n);

   /** set final dual bound  */
   void setDualBound(double bound);

   /** output solution status */
   void outputFinalSolverStatistics(std::ostream *os, double time);

   /** set user plugins */
   void setUserPlugins(ScipUserPlugins *inUi);
   /*
   { 
      userPlugins = inUi; 
      assert(userPlugins != 0);
   }
   */

   /** include user plugins */
   void includeUserPlugins(SCIP *inScip)
   {
      if( userPlugins )
      {
         (*userPlugins)(inScip);
      }
   }

   /** returns whether the objective value is known to be integral in every feasible solution */
   bool isObjIntegral(){ return ( SCIPisObjIntegral(scip) == TRUE );; }

   void interrupt()
   {
       SCIP_STAGE stage = SCIPgetStage(scip);

       if(stage == SCIP_STAGE_PRESOLVING || stage == SCIP_STAGE_SOLVING)
           SCIP_CALL_ABORT( SCIPinterruptSolve(scip) );
   }

#ifdef UG_WITH_UGS
   /** read ugs incumbent solution **/
   bool readUgsIncumbentSolution(UGS::UgsParaCommMpi *ugsComm, int source);

   /** write ugs incumbent solution **/
   void writeUgsIncumbentSolution(UGS::UgsParaCommMpi *ugsComm);
#endif

};

typedef ScipParaInitiator *ScipParaInitiatorPtr;

}

#endif // __SCIP_PARA_INITIATOR_H__

