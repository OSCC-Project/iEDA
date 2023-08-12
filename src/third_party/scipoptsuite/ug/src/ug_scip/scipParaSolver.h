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

/**@file    scipParaSolver.h
 * @brief   ParaSolver extension for SCIP: Parallelized solver implementation for SCIP.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_SOLVER_H__
#define __SCIP_PARA_SOLVER_H__

#include <list>
#include <thread>
#include "ug_bb/bbParaSolver.h"
#include "scipUserPlugins.h"
#include "scipParaDiffSubproblem.h"
#include "scipParaInterruptMsgMonitor.h"
#include "scipDiffParamSet.h"

#define ENFORCED_THRESHOLD 5

namespace ParaSCIP
{

class ScipParaObjCommPointHdlr;
class ScipParaObjNodesel;
class ScipParaObjSelfSplitNodesel;
class ScipParaObjProp;

typedef struct LocalNodeInfo_t
{
   /********************************
    * for local cuts and conflicts *
    * *****************************/
   SCIP_Real      linearLhs;           /**< array of lhs */
   SCIP_Real      linearRhs;           /**< array of rhs */
   int            nLinearCoefs;         /**< array of number of coefficient values for linear constrains */
   SCIP_Real      *linearCoefs;         /**< array of non-zero coefficient values of linear constrains */
   int            *idxLinearCoefsVars;  /**< array of indices of no-zero coefficient values of linear constrains */
} LocalNodeInfo;
typedef LocalNodeInfo * LocalNodeInfoPtr;

class ScipParaSolver : public UG::BbParaSolver
{
   typedef int(ScipParaSolver::*ScipMessageHandlerFunctionPointer)(int, int);

   SCIP                      *scip;
   SCIP                      *scipToCheckEffectOfRootNodeProcesses;
   ScipDiffParamSet          *scipDiffParamSetRoot;
   ScipDiffParamSet          *scipDiffParamSet;
   SCIP_MESSAGEHDLR          *messagehdlr;
   FILE                      *logfile;
   ScipDiffParamSet          *originalParamSet;
   std::list<LocalNodeInfoPtr> *conflictConsList;
   ScipUserPlugins           *userPlugins;

   ScipParaObjCommPointHdlr  *commPointHdlr;
   ScipParaObjNodesel          *nodesel;
#if SCIP_APIVERSION >= 101
   ScipParaObjSelfSplitNodesel *selfSplitNodesel;
#endif

   ScipParaObjProp           *scipPropagator;

   ScipParaInterruptMsgMonitor *interruptMsgMonitor;     ///< interrupt message monitor
   std::thread               interruptMsgMonitorThread;  ///< interrupt message monitor thread

   long long                 nPreviousNodesLeft; /**< number of nodes left in the previous notification */

   int                       originalPriority;  /**< original priority of changing node selector */
   int                       orgMaxRestart;     /**< original value of presolving/maxrestart parameter */

   int           nOrgVars;                   /**< number of original variables in LC */
   int           nOrgVarsInSolvers;          /**< number of original variables in Solvers */
   SCIP_Real     *orgVarLbs;                 /**< array of original lower bound of variable */
   SCIP_Real     *orgVarUbs;                 /**< array of original upper bound of variable */
   SCIP_Real     *tightenedVarLbs;           /**< array of tightened lower bound of variable */
   SCIP_Real     *tightenedVarUbs;           /**< array of tightened upper bound of variable */
   int           *mapToOriginalIndecies;     /**< array of indices to map to original problem's probindices */
   int           *mapToSolverLocalIndecies;  /**< array of reverse indices mapToOriginalIndecies */
   int           *mapToProbIndecies;         /**< map from index to probindex */

   // Dropping settings when variable bounds exchange is performed during racing
   // int           stuffingMaxrounds;
   // int           domcolMaxrounds;
   // int           dualcompMaxrounds;
   // int           dualinferMaxrounds;
   // int           dualaggMaxrounds;
   // unsigned int  abspowerDualpresolve;
   // unsigned int  andDualpresolving;
   // unsigned int  cumulativeDualpresolve;
   // unsigned int  knapsackDualpresolving;
   // unsigned int  linearDualpresolving;
   // unsigned int  setppcDualpresolving;
   // unsigned int  logicorDualpresolving;
   unsigned int  miscAllowdualreds;

   //
   int          nAddedConss;
   SCIP_CONS** addedConss;
   SCIP_CONS* addedDualCons;

   char    *settingsNameLC;                  /**< parameter settings file name for LC */
   bool    fiberSCIP;
   bool    quiet;
   bool collectingModeIsProhibited;          /**< indicate that collecting mode is prohibited */

   const char  *problemFileName;             /**< keep the name for restart, in case of file read */

   SCIP_Real orgFeastol;                     /**< original feasibility tolerance */
   SCIP_Real orgLpfeastol;                   /**< original lp feasibility tolerance */

   bool copyIncreasedVariables;              /**< indicate that SCIP copy increaded variables */

   void setRacingParams(UG::ParaRacingRampUpParamSet *inRacingParams, bool winnerParam);
   void setWinnerRacingParams(UG::ParaRacingRampUpParamSet *inRacingParams);
   void createSubproblem();
   void freeSubproblem();
   void solve();
   long long getNNodesSolved();
   int getNNodesLeft();
   double getDualBoundValue();
   void reinitialize();
   void setOriginalNodeSelectionStrategy();

   void solveToCheckEffectOfRootNodePreprocesses();
   void saveOrgProblemBounds();

   void setBakSettings();

   int lbBoundTightened(int source, int tag);
   int ubBoundTightened(int source, int tag);

   void saveOriginalSettings();
   void dropSettingsForVariableBoundsExchnage();
   void recoverOriginalSettings();

   void saveImprovedSolution();

   static void runInterruptMsgMonitorThread(void *threadData)
   {
      ScipParaInterruptMsgMonitor *monitor = static_cast<ScipParaInterruptMsgMonitor *>(threadData);
      monitor->run();
   }


public:
   ScipParaSolver(int argc, char **argv,
         UG::ParaComm *comm, UG::ParaParamSet *paraParamSet, UG::ParaInstance *paraInstance, UG::ParaDeterministicTimer *detTimer);
   ScipParaSolver(int argc, char **argv,
         UG::ParaComm *comm, UG::ParaParamSet *paraParamSet, UG::ParaInstance *paraInstance, UG::ParaDeterministicTimer *detTimer, double timeOffset, bool thread);
   ~ScipParaSolver();
   const char *getChangeNodeSelName(){
      if( paraParams->getIntParamValue(UG::NodeTransferMode) == 0 ) return "estimate";
      // else return "bfs";
      else return "ScipParaObjNodesel";
   }
   int getOriginalPriority(){
      return originalPriority;
   }

   void saveOriginalPriority(){
      int numnodesels = SCIPgetNNodesels( scip );
      SCIP_NODESEL** nodesels = SCIPgetNodesels( scip );
      const char     *changeNodeSelName = getChangeNodeSelName();
      int i;
      for( i = 0; i < numnodesels; ++i )
      {
         std::string nodeselname(SCIPnodeselGetName(nodesels[i]));
         if( std::string(nodeselname) == std::string(changeNodeSelName) )
         {
            originalPriority = SCIPnodeselGetStdPriority(nodesels[i]);
            break;
         }
      }
      assert( i != numnodesels );
   }

   void setOriginalPriority(){
      int numnodesels = SCIPgetNNodesels( scip );
      SCIP_NODESEL** nodesels = SCIPgetNodesels( scip );
      const char     *changeNodeSelName = getChangeNodeSelName();
      int i;
      for( i = 0; i < numnodesels; ++i )
      {
         std::string nodeselname(SCIPnodeselGetName(nodesels[i]));
         if( std::string(nodeselname) == std::string(changeNodeSelName) )
         {
            SCIP_CALL_ABORT( SCIPsetNodeselStdPriority(scip, nodesels[i], originalPriority ) );
            break;
         }
      }
      assert( i != numnodesels );
   }

   void setOriginalMaxRestart(){
      SCIP_CALL_ABORT( SCIPgetIntParam(scip, "presolving/maxrestarts", &orgMaxRestart) );
   }

   int getOriginalMaxRestart(){ return orgMaxRestart; }

   ScipParaDiffSubproblem *getParentDiffSubproblem(){ return dynamic_cast< ScipParaDiffSubproblem * >(currentTask->getDiffSubproblem() ); }
   void writeCurrentTaskProblem(const std::string& filename);
   void tryNewSolution(UG::ParaSolution *sol);
   void setLightWeightRootNodeProcess();
   void setOriginalRootNodeProcess();

   int getOffsetDepth()
   {
      if( currentTask->getDiffSubproblem() )
      {
         ScipParaDiffSubproblem *scipDiffSubproblem = dynamic_cast<ScipParaDiffSubproblem *>(currentTask->getDiffSubproblem());
         return scipDiffSubproblem->getOffset();
      }
      else
      {
         return 0;
      }
   }

   SCIP *getScip(){
      return scip;
   }

   std::list<LocalNodeInfoPtr> *getConflictConsList(
         )
   {
      return conflictConsList;
   }

   void writeSubproblem();

   long long getSimplexIter(
         )
   {
      SCIP_STAGE stage = SCIPgetStage(scip);
      if( stage == SCIP_STAGE_PRESOLVED || stage == SCIP_STAGE_SOLVING || stage == SCIP_STAGE_SOLVED )
      {
         return SCIPgetNLPIterations(scip);
      }
      else
      {
         return 0;
      }
   }

   int getNRestarts()
   {
      SCIP_STAGE stage = SCIPgetStage(scip);
      if( stage != SCIP_STAGE_INIT )
      {
         return SCIPgetNRuns(scip);
      }
      else
      {
         return 0;
      }
   }

   /** set user plugins */
   void setUserPlugins(ScipUserPlugins *inUi) { userPlugins = inUi; }

   /** include user plugins */
   void includeUserPlugins(SCIP *inScip)
   {
      if( userPlugins )
      {
         (*userPlugins)(inScip);
      }
   }

   bool wasTerminatedNormally(
         )
   {
      return true;
   }

   void setProblemFileName(
         const char *fileName
         )
   {
      problemFileName = fileName;
   }

   bool isCollectingModeProhibited(
         )
   {
      return collectingModeIsProhibited;
   }

   void allowCollectingMode(
         )
   {
      collectingModeIsProhibited = false;
   }

   void prohibitCollectingMode(
         )
   {
      collectingModeIsProhibited = true;
   }

   bool isOriginalIndeciesMap() { return (mapToOriginalIndecies != 0); }

   int getOriginalIndex(int index)
   {
      assert(mapToOriginalIndecies);
      return mapToOriginalIndecies[index];
   }

   bool isProbIndeciesMap() { return (mapToProbIndecies != 0); }

   int getProbIndex(int index)
   {
      assert(mapToProbIndecies);
      return mapToProbIndecies[index];
   }

   long long getNPreviousNodesLeft(){ return nPreviousNodesLeft; }
   void setNPreviousNodesLeft(long long n){ nPreviousNodesLeft = n; }
   int getNOrgVars() { return nOrgVars; }
   double getOrgVarLb(int i){ return orgVarLbs[i]; }
   double getOrgVarUb(int i){ return orgVarUbs[i]; }
   void setOrgVarLb(int i, double v){ orgVarLbs[i] = v; }
   void setOrgVarUb(int i, double v){ orgVarUbs[i] = v; }

   double getTightenedVarLb(int i){ return tightenedVarLbs[i]; }
   double getTightenedVarUb(int i){ return tightenedVarUbs[i]; }
   void setTightenedVarLb(int i, double v){ tightenedVarLbs[i] = v; }
   void setTightenedVarUb(int i, double v){ tightenedVarUbs[i] = v; }

   void checkVarsAndIndex(const char *string, SCIP* inScip)
   {
      std::cout << "R" << paraComm->getRank() << ":" << string << std::endl;
      int nVars;
      SCIP_VAR **vars;
      SCIP_CALL_ABORT( SCIPgetOrigVarsData(inScip, &vars, &nVars, NULL, NULL, NULL, NULL) );
      for( int i = 0; i < nVars; i++ )
      {
         std::cout <<  "R" << paraComm->getRank() << ": idx = " << i << ": " << SCIPvarGetName(vars[i]) << std::endl;
      }

   }

   /** get number of tightened variables during racing */
   int getNTightened();

   /** get number of tightened integral variables during racing */
   int getNTightenedInt();

   bool isCopyIncreasedVariables()
   {
      return copyIncreasedVariables;
   }

   void copyIncrasedVariables()
   {
      copyIncreasedVariables = true;
   }

   void issueInterruptSolve();

   bool isInterrupting();

   int processTagInterruptRequest(int source, int tag)
   {
      if( scip )
      {
         return UG::BbParaSolver::processTagInterruptRequest(source, tag);
      }
      else
      {
         return 0;
      }
   }

};

}

#endif // __SCIP_PARA_SOLVER_H__
