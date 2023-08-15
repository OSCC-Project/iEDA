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

/**@file    scipParaCommTh.h
 * @brief   SCIP ParaComm extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_COMM_TH_H__
#define __SCIP_PARA_COMM_TH_H__
#include "ug_bb/bbParaComm.h"
#include "scipParaTagDef.h"
#include "scipParaInstanceTh.h"
#include "scipParaSolutionTh.h"
#include "scipParaDiffSubproblemTh.h"
#include "scipParaInitialStatTh.h"
#include "scipParaRacingRampUpParamSetTh.h"
#include "scip/scip.h"

namespace ParaSCIP
{

///
/// scip defined transfer data types
///
static const int SCIP_USER_TYPE_FIRST           = UG::BB_USER_TYPE_LAST + 1;
static const int ParaInitialStatType            = SCIP_USER_TYPE_FIRST + 0;

class ScipParaSolver;
class ScipDiffParamSet;

#ifdef _COMM_PTH
class ScipParaCommTh : public UG::BbParaCommPth
#endif
#ifdef _COMM_CPP11
class ScipParaCommTh : public UG::BbParaCommCPP11
#endif
{

   std::mutex                 interruptMsgMonitorLockMutex;    ///< mutex for interrupt message monitor

protected:

   static const char          *tagStringTable[];               ///< tag name string table

   ///
   /// check if tag string table (for debugging) set up correctly
   /// @return true if tag string table is set up correctly, false otherwise
   ///
   bool tagStringTableIsSetUpCoorectly(
         );

   ///
   /// get Tag string for debugging
   /// @return string which shows Tag
   ///
   const char *getTagString(
         int tag                 /// tag to be converted to string
         );

public:
   ScipParaCommTh(){}
   ~ScipParaCommTh();

   ///
   /// lock interrupt message monitor to synchronize with the monitor thread
   ///
   void lockInterruptMsg(
         )
   {
      interruptMsgMonitorLockMutex.lock();
   }

   ///
   /// unlock interrupt message monitor to synchronize with the monitor thread
   ///
   void unlockInterruptMsg(
         )
   {
      interruptMsgMonitorLockMutex.unlock();
   }

   /*******************************************************************************
   * transfer object factory
   *******************************************************************************/
   UG::ParaDiffSubproblem *createParaDiffSubproblem();
   UG::ParaInitialStat* createParaInitialStat();
   UG::ParaRacingRampUpParamSet* createParaRacingRampUpParamSet();
   UG::ParaInstance *createParaInstance();
   UG::ParaSolution *createParaSolution();
   UG::ParaParamSet *createParaParamSet();

   ScipParaInstance *createScipParaInstance(SCIP *scip, int method);
   ScipParaSolution *createScipParaSolution(ScipParaSolver *solver, SCIP_Real objval, int inNvars, SCIP_VAR ** vars, SCIP_Real *vals);
   ScipParaSolution *createScipParaSolution(SCIP_Real objval, int inNvars, int *inIndicesAmongSolvers, SCIP_Real *vals);

   ScipParaDiffSubproblem *createScipParaDiffSubproblem(         
            SCIP *scip,
            ScipParaSolver *scipParaSolver,
            int nNewBranchVars,
            SCIP_VAR **newBranchVars,
            SCIP_Real *newBranchBounds,
            SCIP_BOUNDTYPE *newBoundTypes,
            int nAddedConss,
            SCIP_CONS **addedConss
         );
   ScipParaInitialStat *createScipParaInitialStat(SCIP *scip);
   ScipParaInitialStat *createScipParaInitialStat(
            int inMaxDepth,
            int inMaxTotalDepth,
            int inNVarBranchStatsDown,
            int inNVarBranchStatsUp,
            int *inIdxLBranchStatsVarsDown,
            int *inNVarBranchingDown,
            int *inIdxLBranchStatsVarsUp,
            int *inNVarBranchingUp,
            SCIP_Real *inDownpscost,
            SCIP_Real *inDownvsids,
            SCIP_Real *inDownconflen,
            SCIP_Real *inDowninfer,
            SCIP_Real *inDowncutoff,
            SCIP_Real *inUppscost,
            SCIP_Real *inUpvsids,
            SCIP_Real *inUpconflen,
            SCIP_Real *inUpinfer,
            SCIP_Real *inUpcutoff
         );
   ScipParaRacingRampUpParamSet *createScipParaRacingRampUpParamSet(
         int inTerminationCriteria,
         int inNNodesLeft,
         double inTimeLimit,
         int inScipRacingParamSeed,
         int inPermuteProbSeed,
         int inGenerateBranchOrderSeed,
         ScipDiffParamSet *inScipDiffParamSet
         );
   ScipDiffParamSet *createScipDiffParamSet();
   ScipDiffParamSet *createScipDiffParamSet( SCIP *scip );
};

#define DEF_SCIP_PARA_COMM( scip_para_comm, comm ) ScipParaCommTh *scip_para_comm = dynamic_cast< ScipParaCommTh* >(comm)

}
#endif // __SCIP_PARA_COMM_TH_H__
