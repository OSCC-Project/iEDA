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

/**@file    paraParamSet.h
 * @brief   Parameter set for UG framework.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_PARAM_SET_H__
#define __BB_PARA_PARAM_SET_H__
#include <algorithm>
#include <string>
#include <iostream>
#include <map>
#include <cmath>
#include "ug/paraComm.h"
#if defined(_COMM_PTH) || defined(_COMM_CPP11)
#include "ug/paraParamSetTh.h"
#endif
#if defined(_COMM_MPI_WORLD)
#include "ug/paraParamSetMpi.h"
#endif

#define OUTPUT_PARAM_VALUE_ERROR( msg1, msg2, msg3, msg4 ) \
   std::cout << "[PARAM VALUE ERROR] Param type = " << msg1 << ", Param name = " << msg2 \
     << ", Param value = " <<  msg3 <<  ": Param comment is as follows: " << std::endl \
     << msg4 << std::endl;  \
   return (-1)

namespace UG
{

///
///  Bool parameters
///
static const int BbParaParamsFirst                    = ParaParamsLast + 1;
static const int BbParaParamsBoolFirst                = BbParaParamsFirst;
//-------------------------------------------------------------------------
static const int LogSubtreeInfo                      = BbParaParamsBoolFirst +  0;
static const int OutputTabularSolvingStatus          = BbParaParamsBoolFirst +  1;
static const int DeterministicTabularSolvingStatus   = BbParaParamsBoolFirst +  2;
static const int UseRootNodeCuts                     = BbParaParamsBoolFirst +  3;
static const int TransferLocalCuts                   = BbParaParamsBoolFirst +  4;
static const int TransferConflictCuts                = BbParaParamsBoolFirst +  5;
static const int TransferConflicts                   = BbParaParamsBoolFirst +  6;
static const int TransferBranchStats                 = BbParaParamsBoolFirst +  7;
static const int TransferVarValueStats               = BbParaParamsBoolFirst +  8;
static const int TransferBendersCuts                 = BbParaParamsBoolFirst +  9;
static const int CheckEffectOfRootNodePreprocesses   = BbParaParamsBoolFirst + 10;
static const int CollectOnce                         = BbParaParamsBoolFirst + 11;
static const int ProvingRun                          = BbParaParamsBoolFirst + 12;
static const int SetAllDefaultsAfterRacing           = BbParaParamsBoolFirst + 13;
static const int DistributeBestPrimalSolution        = BbParaParamsBoolFirst + 14;
static const int LightWeightRootNodeProcess          = BbParaParamsBoolFirst + 15;
static const int RacingStatBranching                 = BbParaParamsBoolFirst + 16;
static const int IterativeBreakDown                  = BbParaParamsBoolFirst + 17;
static const int NoPreprocessingInLC                 = BbParaParamsBoolFirst + 18;
static const int NoUpperBoundTransferInRacing        = BbParaParamsBoolFirst + 19;
static const int MergeNodesAtRestart                 = BbParaParamsBoolFirst + 20;
static const int NChangeIntoCollectingModeNSolvers   = BbParaParamsBoolFirst + 21;
static const int EventWeightedDeterministic          = BbParaParamsBoolFirst + 22;
static const int NoSolverPresolvingAtRoot            = BbParaParamsBoolFirst + 23;
static const int NoSolverPresolvingAtRootDefaultSet  = BbParaParamsBoolFirst + 24;
static const int NoAggressiveSeparatorInRacing       = BbParaParamsBoolFirst + 25;
static const int AllBoundChangesTransfer             = BbParaParamsBoolFirst + 26;
static const int NoAllBoundChangesTransferInRacing   = BbParaParamsBoolFirst + 27;
static const int BreakFirstSubtree                   = BbParaParamsBoolFirst + 28;
static const int InitialNodesGeneration              = BbParaParamsBoolFirst + 29;
static const int RestartRacing                       = BbParaParamsBoolFirst + 30;
static const int CheckFeasibilityInLC                = BbParaParamsBoolFirst + 31;
static const int ControlCollectingModeOnSolverSide   = BbParaParamsBoolFirst + 32;
static const int CleanUp                             = BbParaParamsBoolFirst + 33;
static const int DualBoundGainTest                   = BbParaParamsBoolFirst + 34;
static const int GenerateReducedCheckpointFiles      = BbParaParamsBoolFirst + 35;
static const int OutputPresolvedInstance             = BbParaParamsBoolFirst + 36;
static const int CommunicateTighterBoundsInRacing    = BbParaParamsBoolFirst + 37;
static const int KeepRacingUntilToFindFirstSolution  = BbParaParamsBoolFirst + 38;
static const int AllowTreeSearchRestart              = BbParaParamsBoolFirst + 39;
static const int OmitInfeasibleTerminationInRacing   = BbParaParamsBoolFirst + 40;
static const int WaitTerminationOfThreads            = BbParaParamsBoolFirst + 41;
static const int EnhancedFinalCheckpoint             = BbParaParamsBoolFirst + 42;
//-------------------------------------------------------------------------
static const int BbParaParamsBoolLast                = BbParaParamsBoolFirst + 42;
static const int BbParaParamsBoolN                   = BbParaParamsBoolLast - BbParaParamsBoolFirst + 1;
///
/// Int parameters
///
static const int BbParaParamsIntFirst                = BbParaParamsBoolLast  + 1;
//-------------------------------------------------------------------------
static const int RampUpPhaseProcess                  = BbParaParamsIntFirst +  0;
static const int NChangeIntoCollectingMode           = BbParaParamsIntFirst +  1;
static const int NodeTransferMode                    = BbParaParamsIntFirst +  2;
static const int MinNumberOfCollectingModeSolvers    = BbParaParamsIntFirst +  3;
static const int MaxNumberOfCollectingModeSolvers    = BbParaParamsIntFirst +  4;
static const int SolverOrderInCollectingMode         = BbParaParamsIntFirst +  5;
static const int RacingRampUpTerminationCriteria     = BbParaParamsIntFirst +  6;
static const int StopRacingNumberOfNodesLeft         = BbParaParamsIntFirst +  7;
static const int NumberOfNodesKeepingInRootSolver    = BbParaParamsIntFirst +  8;
static const int NumberOfInitialNodes                = BbParaParamsIntFirst +  9;
static const int MaxNRacingParamSetSeed              = BbParaParamsIntFirst + 10;
static const int TryNVariablegOrderInRacing          = BbParaParamsIntFirst + 11;
static const int TryNBranchingOrderInRacing          = BbParaParamsIntFirst + 12;
static const int NEvaluationSolversToStopRacing      = BbParaParamsIntFirst + 13;
static const int NMaxCanditatesForCollecting         = BbParaParamsIntFirst + 14;
static const int NSolverNodesStartBreaking           = BbParaParamsIntFirst + 15;
static const int NStopBreaking                       = BbParaParamsIntFirst + 16;
static const int NTransferLimitForBreaking           = BbParaParamsIntFirst + 17;
static const int NStopSolvingMode                    = BbParaParamsIntFirst + 18;
static const int NCollectOnce                        = BbParaParamsIntFirst + 19;
static const int AggressivePresolveDepth             = BbParaParamsIntFirst + 20;
static const int AggressivePresolveStopDepth         = BbParaParamsIntFirst + 21;
static const int BigDualGapSubtreeHandling           = BbParaParamsIntFirst + 22;
static const int InstanceTransferMethod              = BbParaParamsIntFirst + 23;
static const int KeepNodesDepth                      = BbParaParamsIntFirst + 24;
static const int NoAlternateSolving                  = BbParaParamsIntFirst + 25;
static const int NNodesTransferLogging               = BbParaParamsIntFirst + 26;
static const int NIdleSolversToTerminate             = BbParaParamsIntFirst + 27;
static const int FinalCheckpointNSolvers             = BbParaParamsIntFirst + 28;
static const int NMergingNodesAtRestart              = BbParaParamsIntFirst + 29;
static const int NBoundChangesOfMergeNode            = BbParaParamsIntFirst + 30;
static const int NNodesToKeepInCheckpointFile        = BbParaParamsIntFirst + 31;
static const int NMaxRacingBaseParameters            = BbParaParamsIntFirst + 32;
static const int NBoundChangesForTransferNode        = BbParaParamsIntFirst + 33;
static const int OmitTerminationNSolutionsInRacing   = BbParaParamsIntFirst + 34;
static const int NEagerToSolveAtRestart              = BbParaParamsIntFirst + 35;
static const int SelfSplitTreeDepth                  = BbParaParamsIntFirst + 36;
static const int LightWeightNodePenartyInCollecting  = BbParaParamsIntFirst + 37;
static const int EnhancedCheckpointInterval          = BbParaParamsIntFirst + 38;
//-------------------------------------------------------------------------
static const int BbParaParamsIntLast                 = BbParaParamsIntFirst + 38;
static const int BbParaParamsIntN                    = BbParaParamsIntLast - BbParaParamsIntFirst + 1;
///
/// Longint parameters
///
static const int BbParaParamsLongintFirst            = BbParaParamsIntLast + 1;
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
static const int BbParaParamsLongintLast             = BbParaParamsLongintFirst - 1;  // No params -1
static const int BbParaParamsLongintN                = BbParaParamsLongintLast - BbParaParamsLongintFirst + 1;
///
/// Real parameters
///
static const int BbParaParamsRealFirst                 = BbParaParamsLongintLast + 1;
//-------------------------------------------------------------------------
static const int MultiplierForCollectingMode           = BbParaParamsRealFirst  +  0;
static const int MultiplierToDetermineThresholdValue   = BbParaParamsRealFirst  +  1;
static const int BgapCollectingMode                    = BbParaParamsRealFirst  +  2;
static const int MultiplierForBgapCollectingMode       = BbParaParamsRealFirst  +  3;
static const int ABgapForSwitchingToBestSolver         = BbParaParamsRealFirst  +  4;
static const int BgapStopSolvingMode                   = BbParaParamsRealFirst  +  5;
static const int StopRacingTimeLimit                   = BbParaParamsRealFirst  +  6;
static const int StopRacingTimeLimitMultiplier         = BbParaParamsRealFirst  +  7;
static const int StopRacingNumberOfNodesLeftMultiplier = BbParaParamsRealFirst  +  8;
static const int TimeToIncreaseCMS                     = BbParaParamsRealFirst  +  9;
static const int TabularSolvingStatusInterval          = BbParaParamsRealFirst  + 10;
static const int RatioToApplyLightWeightRootProcess    = BbParaParamsRealFirst  + 11;
static const int MultiplierForBreakingTargetBound      = BbParaParamsRealFirst  + 12;
static const int FixedVariablesRatioInMerging          = BbParaParamsRealFirst  + 13;
static const int AllowableRegressionRatioInMerging     = BbParaParamsRealFirst  + 14;
static const int CountingSolverRatioInRacing           = BbParaParamsRealFirst  + 15;
static const int ProhibitCollectOnceMultiplier         = BbParaParamsRealFirst  + 16;
static const int TNodesTransferLogging                 = BbParaParamsRealFirst  + 17;
static const int RandomNodeSelectionRatio              = BbParaParamsRealFirst  + 18;
static const int DualBoundGainBranchRatio              = BbParaParamsRealFirst  + 19;
static const int CollectingModeInterval                = BbParaParamsRealFirst  + 20;
static const int RestartInRampDownThresholdTime        = BbParaParamsRealFirst  + 21;
static const int RestartInRampDownActiveSolverRatio    = BbParaParamsRealFirst  + 22;
static const int HugeImbalanceThresholdTime            = BbParaParamsRealFirst  + 23;
static const int HugeImbalanceActiveSolverRatio        = BbParaParamsRealFirst  + 24;
static const int TimeStopSolvingMode                   = BbParaParamsRealFirst  + 25;
static const int NoTransferThresholdReductionRatio     = BbParaParamsRealFirst  + 26;
static const int EnhancedCheckpointStartTime           = BbParaParamsRealFirst  + 27;
//-------------------------------------------------------------------------
static const int BbParaParamsRealLast                  = BbParaParamsRealFirst  + 27;
static const int BbParaParamsRealN                     = BbParaParamsRealLast - BbParaParamsRealFirst + 1;
///
/// Char parameters
///
static const int BbParaParamsCharFirst                 = BbParaParamsRealLast + 1;
//-------------------------------------------------------------------------
//-------------------------------------------------------------------------
static const int BbParaParamsCharLast                  = BbParaParamsCharFirst - 1;   // No params -1
static const int BbParaParamsCharN                     = BbParaParamsCharLast - BbParaParamsCharFirst + 1;
///
/// String parameters
///
static const int BbParaParamsStringFirst             = BbParaParamsCharLast      +1;
//-------------------------------------------------------------------------
static const int SolverSettingsForInitialPresolving  = BbParaParamsStringFirst + 0;
static const int SolverSettingsAtRootNode            = BbParaParamsStringFirst + 1;
static const int SolverSettingsExceptRootNode        = BbParaParamsStringFirst + 2;
static const int SolverSettingsAtRacing              = BbParaParamsStringFirst + 3;
//-------------------------------------------------------------------------
static const int BbParaParamsStringLast              = BbParaParamsStringFirst + 3;
static const int BbParaParamsStringN                 = BbParaParamsStringLast - BbParaParamsStringFirst + 1;
static const int BbParaParamsLast                    = BbParaParamsStringLast;
static const int BbParaParamsSize                    = BbParaParamsLast + 1;


class ParaComm;
///
/// class BbParaParamSet
///
#if defined(_COMM_PTH) || defined(_COMM_CPP11)
class BbParaParamSet : public ParaParamSetTh
{

public:

   ///
   /// constructor
   ///
   BbParaParamSet(
         )
         : ParaParamSetTh(BbParaParamsSize)
   {
   }

   ///
   /// constructor
   ///
   BbParaParamSet(
         size_t inNParaParams
         );
//
//         : ParaParamSetTh(inNParaParams)
//   {
//   }

#endif
#if defined(_COMM_MPI_WORLD)
class BbParaParamSet : public ParaParamSetMpi
{

public:

   ///
   /// constructor
   ///
   BbParaParamSet(
         )
         : ParaParamSetMpi(BbParaParamsSize)
   {
   }

   ///
   /// constructor
   ///
   BbParaParamSet(
         size_t inNParaParams
         );
//         : ParaParamSetMpi(inNParaParams)
//   {
//   }

#endif

   ///
   /// destructor
   ///
   virtual ~BbParaParamSet(
         )
   {
   }

   ///
   /// read ParaParams from file
   ///
   void read(
         ParaComm *comm,       ///< communicator used
         const char* filename  ///< reading file name
         );

   ///
   /// get number of bool parameters
   /// @return size of parameter table
   ///
   virtual size_t getNumBoolParams(
         )
   {
      return (ParaParamsBoolN + BbParaParamsBoolN);
   }

   ///
   /// get number of int parameters
   /// @return size of parameter table
   ///
   virtual size_t getNumIntParams(
         )
   {
      return (ParaParamsIntN + BbParaParamsIntN);
   }

   ///
   /// get number of longint parameters
   /// @return size of parameter table
   ///
   virtual size_t getNumLongintParams(
         )
   {
      return (ParaParamsLongintN + BbParaParamsLongintN);
   }

   ///
   /// get number of real parameters
   /// @return size of parameter table
   ///
   virtual size_t getNumRealParams(
         )
   {
      return (ParaParamsCharN + BbParaParamsCharN);
   }

   ///
   /// get number of char parameters
   /// @return size of parameter table
   ///
   virtual size_t getNumCharParams(
         )
   {
      return (ParaParamsCharN + BbParaParamsCharN);
   }

   ///
   /// get number of string parameters
   /// @return size of parameter table
   ///
   virtual size_t getNumStringParams(
         )
   {
      return (ParaParamsStringN + BbParaParamsStringN);
   }

};

}

#endif  // __BB_PARA_PARAM_SET_H__
