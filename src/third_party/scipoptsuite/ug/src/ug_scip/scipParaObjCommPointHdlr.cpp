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

/**@file    scipParaObjCommPointHdlr.cpp
 * @brief   Event handlr for communication point.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cassert>
#include <cmath>
#include <string>
#include <sstream>
#include <iostream>

#include "scipParaObjCommPointHdlr.h"
#include "scip/struct_nodesel.h"
#include "scip/tree.h"
#ifdef UG_DEBUG_SOLUTION
#ifndef WITH_DEBUG_SOLUTION
#define WITH_DEBUG_SOLUTION
#endif
#include "scip/debug.h"
#endif

#include "scipParaSolution.h"
#include "scipParaDiffSubproblem.h"

using namespace ParaSCIP;

void
ScipParaObjCommPointHdlr::processNewSolution(
      SCIP*              scip,               /**< SCIP data structure */
      SCIP_EVENT*        event               /**< event to process */
      )
{
   SCIP_SOL *sol = SCIPeventGetSol(event);
   int nVars = SCIPgetNOrigVars(scip);
   SCIP_VAR **vars = SCIPgetOrigVars(scip);
   SCIP_Real *vals = new SCIP_Real[nVars];
   SCIP_CALL_ABORT( SCIPgetSolVals(scip, sol, nVars, vars, vals) );

   // SCIP_CALL_ABORT( SCIPprintSol(scip,sol,NULL, FALSE) );
   SCIP_Bool feasible;
   SCIP_CALL_ABORT( SCIPcheckSolOrig(scip,sol,&feasible,1,1 ) );

   DEF_SCIP_PARA_COMM( scipParaComm, paraComm);

   if( scipParaSolver->isCopyIncreasedVariables() )
   {
      SCIP_VAR **varsInOrig = new SCIP_VAR*[nVars];
      SCIP_Real *valsInOrig = new SCIP_Real[nVars]();
      int nVarsInOrig = 0;
      for( int i = 0; i < nVars; i++ )
      {
         if( scipParaSolver->getOriginalIndex(SCIPvarGetIndex(vars[i])) >= 0 )
         {
            varsInOrig[nVarsInOrig] = vars[i];
            valsInOrig[nVarsInOrig] = vals[i];
            nVarsInOrig++;
         }
      }

      scipParaSolver->saveIfImprovedSolutionWasFound(
            scipParaComm->createScipParaSolution(
                  scipParaSolver,
                  SCIPgetSolOrigObj(scip, sol),
                  nVarsInOrig,
                  varsInOrig,
                  valsInOrig
                  )
            );
      delete [] varsInOrig;
      delete [] valsInOrig;
   }
   else
   {
      scipParaSolver->saveIfImprovedSolutionWasFound(
            scipParaComm->createScipParaSolution(
                  scipParaSolver,
                  SCIPgetSolOrigObj(scip, sol),
                  nVars,
                  vars,
                  vals
                  )
            );
      // std::cout << "*** event handler ***" << std::endl;
      // SCIP_CALL_ABORT(  SCIPprintSol(scip, sol, NULL, FALSE) );
      // scipParaSolver->checkVarsAndIndex("***found solution***",scip);
   }
   delete [] vals;
}

SCIP_RETCODE
ScipParaObjCommPointHdlr::scip_exec(
      SCIP*              scip,               /**< SCIP data structure */
      SCIP_EVENTHDLR*    eventhdlr,          /**< the event handler itself */
      SCIP_EVENT*        event,              /**< event to process */
      SCIP_EVENTDATA*    eventdata           /**< user data for the event */
      )
{
   DEF_SCIP_PARA_COMM( scipParaComm, paraComm);
   if( !scipParaSolver->getParaParamSet()->getBoolParamValue(UG::Deterministic) )
   {
      scipParaComm->lockInterruptMsg();
   }
   assert(eventhdlr != NULL);
   assert(strcmp(SCIPeventhdlrGetName(eventhdlr), "ScipParaObjCommPointHdlr") == 0);
   assert(event != NULL);
   assert(SCIPeventGetType(event) &
          ( SCIP_EVENTTYPE_GBDCHANGED |
            SCIP_EVENTTYPE_BOUNDTIGHTENED |
            SCIP_EVENTTYPE_LPEVENT |
            SCIP_EVENTTYPE_ROWEVENT |
            // SCIP_EVENTTYPE_NODEFOCUSED |
            SCIP_EVENTTYPE_NODEEVENT |
            SCIP_EVENTTYPE_BESTSOLFOUND // |
            // SCIP_EVENTTYPE_COMM
         )
         );

#if SCIP_VERSION == 211 && SCIP_SUBVERSION == 0
   if( SCIPgetStage(scip) >= SCIP_STAGE_FREESOLVE )
#else
   if( SCIPgetStage(scip) >= SCIP_STAGE_EXITSOLVE )
#endif
   {
      if( !scipParaSolver->getParaParamSet()->getBoolParamValue(UG::Deterministic) )
      {
         scipParaComm->unlockInterruptMsg();
      }
      return  SCIP_OKAY;
   }

   double detTime = -1.0;
   SCIP *subScip = 0;
   if( cloned )
   {
      subScip = scip;
      scip = originalScip;
   }

   if( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::Deterministic) )
   {
      assert(scipParaSolver->getDeterministicTimer());
      if( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::EventWeightedDeterministic) )
      {
         if( SCIPeventGetType(event) == SCIP_EVENTTYPE_FIRSTLPSOLVED )
         {
            SCIP_Longint totalLpIter = 0;
            if( cloned )
            {
               totalLpIter = SCIPgetNLPIterations(subScip);
            }
            else
            {
               totalLpIter = SCIPgetNLPIterations(scip);
            }
            SCIP_Longint lpIter = totalLpIter - previousLpIter;
            if( lpIter < 0 )
            {
               lpIter = totalLpIter;
            }
            scipParaSolver->getDeterministicTimer()->update(lpIter);
            previousLpIter = totalLpIter;

         }
         else if ( SCIPeventGetType(event) == SCIP_EVENTTYPE_LPSOLVED )
         {
            SCIP_Longint totalLpIter = 0;
            if( cloned )
            {
               totalLpIter = SCIPgetNLPIterations(subScip);
            }
            else
            {
               totalLpIter = SCIPgetNLPIterations(scip);
            }
            SCIP_Longint lpIter = totalLpIter - previousLpIter;
            if( lpIter < 0 )
            {
               lpIter = totalLpIter;
            }
            scipParaSolver->getDeterministicTimer()->update(lpIter);
            previousLpIter = totalLpIter;
         }
         else if ( SCIPeventGetType(event) == SCIP_EVENTTYPE_BESTSOLFOUND
               || SCIPeventGetType(event) == SCIP_EVENTTYPE_NODEFOCUSED )
         {
         }
         else
         {
            scipParaSolver->getDeterministicTimer()->update(1.0);
         }
      }
      else
      {
         if ( SCIPeventGetType(event) != SCIP_EVENTTYPE_BESTSOLFOUND
               && SCIPeventGetType(event) != SCIP_EVENTTYPE_NODEFOCUSED )
         {
            scipParaSolver->getDeterministicTimer()->update(1.0);
         }
      }
      detTime = scipParaSolver->getDeterministicTimer()->getElapsedTime();
      if( ( detTime - scipParaSolver->getPreviousCommTime() )
            < ( scipParaSolver->getParaParamSet()->getRealParamValue(UG::NotificationInterval) ) )
      {
         if( SCIPeventGetType(event) == SCIP_EVENTTYPE_BESTSOLFOUND )
         {
            if( cloned )
            {
               if( !scipParaSolver->getParaParamSet()->getBoolParamValue(UG::Deterministic) )
               {
                  scipParaComm->unlockInterruptMsg();
               }
               return SCIP_OKAY;
            }
            else
            {
               processNewSolution(scip, event);
            }
         }
      }
      do
      {
         scipParaSolver->iReceiveMessages();
      } while( !scipParaSolver->waitToken(scipParaSolver->getRank()) );
   }
   else
   {
      if( SCIPeventGetType(event) == SCIP_EVENTTYPE_BESTSOLFOUND )
      {
         if( cloned )
         {
            if( !scipParaSolver->getParaParamSet()->getBoolParamValue(UG::Deterministic) )
            {
               scipParaComm->unlockInterruptMsg();
            }
            return  SCIP_OKAY;
         }
         else
         {
            processNewSolution(scip, event);
         }
      }
   }

   //
   //  the following interrupt routine moved from the inside of the following if clause
   //  and  scipParaSolver->iReceiveMessages() before solution sending is also removed (comment out)
   //
   /*********************************************************************************************
     * check if there are messages: this is the first thing to do for communication *
     ********************************************************************************************/
   scipParaSolver->iReceiveMessages();
   if( !scipParaSolver->getParaParamSet()->getBoolParamValue(UG::Deterministic) )
   {
      scipParaComm->unlockInterruptMsg();
   }

   if( scipParaSolver->getParaParamSet()->getIntParamValue(UG::RampUpPhaseProcess) == 3 &&
         !scipParaSolver->isRampUp() )
   {
      return SCIP_OKAY;
   }

   if( scipParaSolver->isGlobalIncumbentUpdated() &&
         SCIPgetStage(scip) != SCIP_STAGE_PRESOLVING
                  && SCIPgetStage(scip) != SCIP_STAGE_INITSOLVE )
   {
      if( !cloned
            // && SCIPeventGetType(event) != SCIP_EVENTTYPE_BOUNDTIGHTENED
            // && SCIPeventGetType(event) != SCIP_EVENTTYPE_ROWDELETEDLP
            &&
            ( SCIPeventGetType(event) & (SCIP_EVENTTYPE_NODEEVENT | SCIP_EVENTTYPE_LPEVENT)
            )
         )
      {
         /** set cutoff value */
         if( scipParaSolver->getGlobalBestIncumbentValue() < SCIPgetObjlimit(scip) )
         {
            scipParaObjLimitUpdator->update();
            // SCIP_CALL_ABORT( SCIPsetObjlimit(scip, scipParaSolver->getGlobalBestIncumbentValue()) );
         }
         // scipParaSolver->globalIncumbnetValueIsReflected();
         // std::cout << "***** R." << scipParaSolver->getRank() << ", in event = " << SCIPeventGetType(event) << std::endl;
         if( !interrupting  && SCIPisObjIntegral(scip) )
         {
            if( SCIPfeasCeil(scip, dynamic_cast<UG::BbParaNode *>(scipParaSolver->getCurrentNode())->getDualBoundValue())
                  >= scipParaSolver->getGlobalBestIncumbentValue() )
            {
#ifdef UG_DEBUG_SOLUTION
               if( SCIPdebugSolIsEnabled(scip) )
               {
                  throw "Optimal solution going to be lost!";
               }
#endif
               SCIP_CALL_ABORT( SCIPinterruptSolve(scip) );
               interrupting = true;
               // std::cout << "***** R." << scipParaSolver->getRank() << "Interrupted!" << std::endl;
            }
         }
      }
   }

   if( scipParaSolver->isRacingStage()
         && scipParaSolver->getParaParamSet()->getBoolParamValue(UG::CommunicateTighterBoundsInRacing)
         // && SCIPgetStage(scip) != SCIP_STAGE_PRESOLVING
         // && SCIPgetStage(scip) != SCIP_STAGE_INITSOLVE
         && (SCIPeventGetType(event) & SCIP_EVENTTYPE_GBDCHANGED )
         && (SCIPvarGetStatus(SCIPeventGetVar(event)) != SCIP_VARSTATUS_LOOSE )
         && (SCIPvarGetStatus(SCIPeventGetVar(event)) != SCIP_VARSTATUS_AGGREGATED )
         )
   {
      // assert( scipParaSolver->getCurrentNode()->isRootTask() );  /// This is fail for racing at restart
      SCIP_BOUNDTYPE      boundtype;
      SCIP_Var *var = SCIPeventGetVar(event);
      switch( SCIPeventGetType(event) )
      {
         case SCIP_EVENTTYPE_GLBCHANGED:
            boundtype = SCIP_BOUNDTYPE_LOWER;
            break;
         case SCIP_EVENTTYPE_GUBCHANGED:
            boundtype = SCIP_BOUNDTYPE_UPPER;
            break;
         default:
            SCIPABORT();
            return SCIP_ERROR;
      }
      SCIP_Real newbound = SCIPeventGetNewbound(event);
      SCIP_Real constant = 0.0;
      SCIP_Real scalar = 1.0;
      SCIP_CALL_ABORT( SCIPvarGetOrigvarSum(&var, &scalar, &constant) );
      if( var != NULL )
      {
         assert(SCIPvarIsOriginal(var));
         int index = SCIPvarGetIndex(var);
         boundtype = scalar < 0.0 ? (SCIP_BOUNDTYPE)(1 - boundtype) : boundtype;
         newbound = (newbound - constant) / scalar;
         if( scipParaSolver->isProbIndeciesMap() )
         {
            index = scipParaSolver->getProbIndex(index);
         }
         if( index < scipParaSolver->getNOrgVars() )
         {
            if( boundtype == SCIP_BOUNDTYPE_LOWER )
            {
               if( SCIPvarGetType(var) != SCIP_VARTYPE_CONTINUOUS )
               {
                  assert( SCIPisFeasIntegral(scip, newbound) );
                  newbound = SCIPfeasCeil(scip, newbound);
               }
   #ifdef UG_DEBUG_SOLUTION
               SCIP_Real solvalue = 0.0;
               SCIP_CALL(SCIPdebugGetSolVal(scip,var, &solvalue));
               std::cout << "Sender side SolValue: " << SCIPvarGetName(var) << " = " << solvalue << std::endl;
               std::cout << "Sender side (SCIP_BOUNDTYPE_LOWER): " << SCIPvarGetName(var) << " = " << newbound << std::endl;
               SCIP_CALL_ABORT( SCIPdebugCheckLbGlobal(scip,var,newbound) );
   #endif
               // assert( SCIPisLE(scip,scipParaSolver->getOrgVarLb(index), newbound) &&
               assert( SCIPisGE(scip,scipParaSolver->getOrgVarUb(index), newbound) );
               if( SCIPisGT(scip, newbound, scipParaSolver->getTightenedVarLb(index) ) ) // &&
                   //  SCIPisLE(scip, newbound, scipParaSolver->getTightenedVarUb(index) ) )  // just for safety
               {
                  // this assertion may not be hold
                  // assert( SCIPisLE(scip,scipParaSolver->getTightenedVarLb(index), newbound) &&
                  //      SCIPisGE(scip,scipParaSolver->getTightenedVarUb(index), newbound)  );
                  scipParaSolver->setTightenedVarLb(index, newbound);
                  int orgIndex = SCIPvarGetIndex(var);
                  if( scipParaSolver->isOriginalIndeciesMap() )
                  {
                     orgIndex = scipParaSolver->getOriginalIndex(orgIndex);
                  }
                  PARA_COMM_CALL(
                        paraComm->send((void *)&orgIndex, 1, UG::ParaINT, 0, UG::TagLbBoundTightenedIndex )
                        );
                  PARA_COMM_CALL(
                        paraComm->send((void *)&newbound, 1, UG::ParaDOUBLE, 0, UG::TagLbBoundTightenedBound )
                        );
                  /*
                  std::cout << "Rank " << paraComm->getRank()
                        << ": send tightened lower bond. idx = " << index
                        << ", bound = " << newbound
                        << ", var status = " << SCIPvarGetStatus(SCIPeventGetVar(event))
                        << std::endl;  */
               }
            }
            else
            {
               if( SCIPvarGetType(var) != SCIP_VARTYPE_CONTINUOUS )
               {
                  assert( SCIPisFeasIntegral(scip, newbound) );
                  newbound = SCIPfeasFloor(scip, newbound);
               }
   #ifdef UG_DEBUG_SOLUTION
               SCIP_Real solvalue = 0.0;
               SCIP_CALL(SCIPdebugGetSolVal(scip,var, &solvalue));
               std::cout << "Sender side SolValue: " << SCIPvarGetName(var) << " = " << solvalue << std::endl;
               std::cout << "Sender side (SCIP_BOUNDTYPE_UPPER): " << SCIPvarGetName(var) << " = " << newbound << std::endl;
               SCIP_CALL_ABORT( SCIPdebugCheckUbGlobal(scip,var,newbound) );
   #endif
               assert( SCIPisLE(scip,scipParaSolver->getOrgVarLb(index), newbound) );
               //     && SCIPisGE(scip,scipParaSolver->getOrgVarUb(index), newbound)  );
               if( SCIPisLT(scip, newbound, scipParaSolver->getTightenedVarUb(index) ) ) // &&
                     // SCIPisGE(scip, newbound, scipParaSolver->getTightenedVarLb(index) )  )   // just for safety
               {
                  // This asertion may not be hold
                  // assert( SCIPisLE(scip,scipParaSolver->getTightenedVarLb(index), newbound) &&
                  //      SCIPisGE(scip,scipParaSolver->getTightenedVarUb(index), newbound)  );
                  scipParaSolver->setTightenedVarUb(index, newbound);
                  int orgIndex = SCIPvarGetIndex(var);
                  if( scipParaSolver->isOriginalIndeciesMap() )
                  {
                     orgIndex = scipParaSolver->getOriginalIndex(orgIndex);
                  }
                  PARA_COMM_CALL(
                        paraComm->send((void *)&orgIndex, 1, UG::ParaINT, 0, UG::TagUbBoundTightenedIndex )
                        );
                  PARA_COMM_CALL(
                        paraComm->send((void *)&newbound, 1, UG::ParaDOUBLE, 0, UG::TagUbBoundTightenedBound )
                        );
                  /*
                  std::cout << "Rank " << paraComm->getRank()
                        << ": send tightened upper bond. idx = " << index
                        << ", bound = " << newbound
                        << ", var status = " << SCIPvarGetStatus(SCIPeventGetVar(event))
                        << std::endl;  */
               }
            }
         }
      }
   }


   /*********************************************************************************************
     * update solution. This have to do right after receiving message *
     ********************************************************************************************/
   scipParaSolver->sendLocalSolution();   // if local solution exists, it should be sent right after iReceiveMessages
   if( !cloned )
   {
      scipParaSolver->updatePendingSolution();
   } 
   if( scipParaSolver->newParaNodeExists() ||
         scipParaSolver->isInterrupting() ||
         scipParaSolver->isRacingInterruptRequested() ||
         scipParaSolver->isTerminationRequested() ||
         scipParaSolver->isGivenGapReached()
         )
   {
      if( !interrupting  )
      {
         if( cloned )
         {
            if( SCIPgetStage(subScip) != SCIP_STAGE_INITSOLVE )
            {
               SCIP_CALL_ABORT( SCIPinterruptSolve(subScip) );
               interrupting = true;
            }
         }
         else
         {
            if( SCIPgetStage(scip) != SCIP_STAGE_INITSOLVE )
            {
               if( scipParaSolver->getParaParamSet()->getRealParamValue(UG::FinalCheckpointGeneratingTime) < 0.0 ||
                     ( scipParaSolver->isInterrupting() && (!scipParaSolver->isCollecingInterrupt()) ) ||
                     ( scipParaSolver->isRacingStage() && (!scipParaSolver->isRacingWinner()) ) )
               {
   #ifdef UG_DEBUG_SOLUTION
                  if( scipParaSolver->isRacingStage() && scipParaSolver->isRacingWinner() )
                  {
                     std::cout <<  "racing stage = " << scipParaSolver->isRacingStage() << ", winner = " << scipParaSolver->isRacingWinner() << std::endl;
                     if( SCIPdebugSolIsEnabled(scip) )
                     {
                        throw "Optimal solution going to be lost!";
                     }
                  }
                  SCIPdebugSolDisable(scip);
                  std::cout << "R." << paraComm->getRank() << ": disable debug, this solver is interrupted." << std::endl;
   #endif
                  SCIP_CALL_ABORT( SCIPinterruptSolve(scip) );
                  interrupting = true;
               }
            }
         }
         return SCIP_OKAY;
      }
      if( scipParaSolver->getParaParamSet()->getRealParamValue(UG::FinalCheckpointGeneratingTime) < 0.0 ||
            ( !scipParaSolver->isCollecingInterrupt() ) ||
            ( !scipParaSolver->isRampUp() ) )
      {
         return SCIP_OKAY;
      }
   }

   if( (!cloned ) &&
         scipParaSolver->getParaParamSet()->getBoolParamValue(UG::ControlCollectingModeOnSolverSide) &&
         scipParaSolver->isCollectingModeProhibited() )
   {
      int maxrestarts = 0;
      SCIP_CALL( SCIPgetIntParam(scip, "presolving/maxrestarts", &maxrestarts) );
      if( SCIPgetNRuns( scip ) >= maxrestarts  )
      {
         if( !scipParaSolver->isRacingStage() )
         {
            PARA_COMM_CALL(
                   paraComm->send( NULL, 0, UG::ParaBYTE, 0, UG::TagAllowToBeInCollectingMode);
                   );
         }
         scipParaSolver->allowCollectingMode();
      }
      /*
      else
      {
         if( scipParaSolver->isRacingStage() )
         {
            return SCIP_OKAY;      // during racing stage, just keep running for all restart
         }
      }
      */
   }

   SCIP_Longint nNodes = 0;
   if( SCIPgetStage(scip) != SCIP_STAGE_PRESOLVING
         && SCIPgetStage(scip) != SCIP_STAGE_INITSOLVE
         && (!scipParaSolver->isCollectingModeProhibited() ) ) // restarting is considered as root node process in ug[SCIP,*])
   {
      nNodes = SCIPgetNTotalNodes(scip);
   }

   if( previousNNodesSolved == nNodes
         // && (!scipParaSolver->isCollectingModeProhibited() ) ) // setting cutoff value works well
         || scipParaSolver->isCollectingModeProhibited() ) // setting cutoff value works well
   {
      if( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::Deterministic)
            && ( !scipParaSolver->isInCollectingMode() )
            && ( !scipParaSolver->newParaNodeExists() )
            && ( !scipParaSolver->isInterrupting() )
            && ( !scipParaSolver->isRacingInterruptRequested() )
            && ( !scipParaSolver->isTerminationRequested() )
            && ( !scipParaSolver->isGivenGapReached() )
            )
      {
         scipParaSolver->passToken(scipParaSolver->getRank());
         scipParaSolver->setPreviousCommTime(detTime);
      }

      return SCIP_OKAY;
   }
   previousNNodesSolved = nNodes;

   /** if root node is solved, set root node time */
   if( nNodes == 2 && SCIPgetNNodes(scip) == 2 )
   {
      /** when a problem is solved at root,
       * its root node process time is set on paraSolver main loop */
      scipParaSolver->setRootNodeTime();
      scipParaSolver->setRootNodeSimplexIter(scipParaSolver->getSimplexIter());
      if( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::DualBoundGainTest) &&
            scipParaSolver->isDualBoundGainTestNeeded()
            )
      {
         if( ( SCIPgetDualbound(scip) - dynamic_cast<UG::BbParaNode *>(scipParaSolver->getCurrentNode())->getDualBoundValue() )
               < scipParaSolver->getAverageDualBoundGain()*scipParaSolver->getParaParamSet()->getRealParamValue(UG::DualBoundGainBranchRatio)  )
         {
            scipParaSolver->setNotEnoughGain();
         }
      }
      if( scipParaSolver->getCurrentSolivingNodeMergingStatus() == 0 &&
            SCIPgetDualbound(scip) <
            scipParaSolver->getCurrentSolvingNodeInitialDualBound() * ( 1.0 - scipParaSolver->getParaParamSet()->getRealParamValue(UG::AllowableRegressionRatioInMerging)) )
      {
         if( !interrupting  )
         {
            if( cloned )
            {
               SCIP_CALL_ABORT( SCIPinterruptSolve(subScip) );
            }
            else
            {
               dynamic_cast<UG::BbParaNode *>(scipParaSolver->getCurrentNode())->setMergingStatus(3);  // cannot be merged
               SCIP_CALL_ABORT( SCIPinterruptSolve(scip) );
            }
            interrupting = true;
         }
         return SCIP_OKAY;
      }
   }

   /*****************************************************************************
    * sends solver state message if it is necessary, that is,                   *
    * notification interval time has been passed from the previous notification *
    *****************************************************************************/
   double bestDualBoundValue = -SCIPinfinity(scip);
   if( SCIPgetStage(scip) == SCIP_STAGE_TRANSFORMED ||
         SCIPgetStage(scip) == SCIP_STAGE_PRESOLVING ||
         SCIPgetStage(scip) == SCIP_STAGE_INITSOLVE
         )
   {
      bestDualBoundValue = dynamic_cast<UG::BbParaNode *>(scipParaSolver->getCurrentNode())->getDualBoundValue();
   }
   else
   {
      bestDualBoundValue = std::max(SCIPgetDualbound(scip), dynamic_cast<UG::BbParaNode *>(scipParaSolver->getCurrentNode())->getDualBoundValue());
   }

   /************************************************************
    *  check if global best incumbent value is updated or not. *
    *  if it is updated, set cutoff value                      *
    ************************************************************/
//   if( scipParaSolver->isGlobalIncumbentUpdated() &&
//         SCIPgetStage(scip) != SCIP_STAGE_PRESOLVING
//                  && SCIPgetStage(scip) != SCIP_STAGE_INITSOLVE )
//   {
//      if( !cloned
//            && SCIPeventGetType(event) != SCIP_EVENTTYPE_BOUNDTIGHTENED
//            )
//      {
//         /** set cutoff value */
//         // if( scipParaSolver->getGlobalBestIncumbentValue() > bestDualBoundValue )
//         // {
//         SCIP_CALL_ABORT( SCIPsetObjlimit(scip, scipParaSolver->getGlobalBestIncumbentValue()) );
//         // }
//         scipParaSolver->globalIncumbnetValueIsReflected();
//std::cout <<  "R." << paraComm->getRank() << " ** UPDATED **" << std::endl;
//      }
//   }

   if( scipParaSolver->getGlobalBestIncumbentValue() < bestDualBoundValue )
   {
      if( !interrupting  )
      {
         if( cloned )
         {
            if( SCIPgetStage(subScip) != SCIP_STAGE_INITSOLVE )
            {
               SCIP_CALL_ABORT( SCIPinterruptSolve(subScip) );
               interrupting = true;
            }
         }
         else
         {
            if( SCIPgetStage(scip) != SCIP_STAGE_INITSOLVE )
            {
#ifdef UG_DEBUG_SOLUTION
               if( scipParaSolver->isRacingStage() && scipParaSolver->isRacingWinner() )
               { 
                  std::cout <<  "racing stage = " << scipParaSolver->isRacingStage() << ", winner = " << scipParaSolver->isRacingWinner() << std::endl;
                  if( SCIPdebugSolIsEnabled(scip) )
                  {
                     throw "Optimal solution going to be lost!";
                  }
               }
               SCIPdebugSolDisable(scip);
               std::cout << "R." << paraComm->getRank() << ": disable debug, this solver is interrupted." << std::endl;
#endif
               SCIP_CALL_ABORT( SCIPinterruptSolve(scip) );
               interrupting = true;
            }
         }
      }
      return SCIP_OKAY;
   }

   if( scipParaSolver->isCollectingModeProhibited() || nNodes < 2 )  // do not have to send status to LC yet.
   {
      return SCIP_OKAY;
   }

   if( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::GenerateReducedCheckpointFiles) )
   {
      needToSendNode = true;
   }

   if ( needToSendNode
         || scipParaSolver->notificationIsNecessary()
         || nNodes == 2
         || scipParaSolver->getParaParamSet()->getBoolParamValue(UG::Deterministic) )  // always notify
   {
      if( bestDualBoundValue >= -1e+10 )  // only after some dual bound value is computed, notify its status
      {
         int nNodesLeft = 0;
         if( SCIPgetStage(scip) != SCIP_STAGE_TRANSFORMED &&
               SCIPgetStage(scip) != SCIP_STAGE_PRESOLVING &&
               SCIPgetStage(scip) != SCIP_STAGE_INITSOLVE )
         {
            nNodesLeft = SCIPgetNNodesLeft(scip);
         }
         scipParaSolver->sendSolverState(nNodes, nNodesLeft, bestDualBoundValue, detTime);
         scipParaSolver->waitMessageIfNecessary();
         if( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::GenerateReducedCheckpointFiles)  &&
               nNodes > 2 )
         {
            dynamic_cast<UG::BbParaNode *>(scipParaSolver->getCurrentNode())->setMergingStatus(3);  // cannot be merged: this is a dummy status
            SCIP_CALL_ABORT( SCIPinterruptSolve(scip) );
            return SCIP_OKAY;
         }
         if( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::Deterministic)
               && ( !needToSendNode )
               && ( !scipParaSolver->isManyNodesCollectionRequested() )
               && !( !scipParaSolver->isRacingStage()
                     &&  scipParaSolver->isAggressivePresolvingSpecified()
                     &&  scipParaSolver->getSubMipDepth() <
                     ( scipParaSolver->getOffsetDepth() + scipParaSolver->getAggresivePresolvingStopDepth() ) &&
                       SCIPgetDepth(scip) > scipParaSolver->getAggresivePresolvingDepth() )
           )
         {
            scipParaSolver->passToken(scipParaSolver->getRank());
            scipParaSolver->setPreviousCommTime(detTime);
         }
      }
   }

   if( scipParaSolver->getNStopSolvingMode() > 0 &&
         !scipParaSolver->isRacingStage() &&
         ( nNodes < scipParaSolver->getNStopSolvingMode() ) &&
         ( bestDualBoundValue - scipParaSolver->getLcBestDualBoundValue() ) > 0.0 &&
          REALABS( ( bestDualBoundValue - scipParaSolver->getLcBestDualBoundValue() )
         / std::max(std::fabs(scipParaSolver->getLcBestDualBoundValue()), 1.0) ) > scipParaSolver->getBoundGapForStopSolving() )
   {
      if( scipParaSolver->getBigDualGapSubtreeHandlingStrategy() == 0 )
      {
         if( scipParaSolver->getTimeStopSolvingMode() < 0 ||
             ( scipParaSolver->getTimeStopSolvingMode() > 0 &&
               scipParaSolver->getElapsedTimeOfNodeSolving() < scipParaSolver->getTimeStopSolvingMode() ) )
         {
            scipParaSolver->sendAnotherNodeRequest(bestDualBoundValue);   // throw away nodes
         }
      }
      else if( scipParaSolver->getBigDualGapSubtreeHandlingStrategy() == 1 )
      {
         if(  scipParaSolver->getCurrentSolivingNodeMergingStatus() != 0 )
         {
            scipParaSolver->setSendBackAllNodes();
         }
      }
      else
      {
         THROW_LOGICAL_ERROR2("Invalid big dual gap handling strategy. startegy = ",
               scipParaSolver->getBigDualGapSubtreeHandlingStrategy() );
      }
   }

   if( scipParaSolver->getTimeStopSolvingMode() > 0 &&
         !scipParaSolver->isRacingStage() &&
         ( bestDualBoundValue - scipParaSolver->getLcBestDualBoundValue() ) > 0.0 &&
          REALABS( ( bestDualBoundValue - scipParaSolver->getLcBestDualBoundValue() )
         / scipParaSolver->getLcBestDualBoundValue() ) > scipParaSolver->getBoundGapForStopSolving() &&
         ( scipParaSolver->getElapsedTimeOfNodeSolving() < scipParaSolver->getTimeStopSolvingMode() ) )
   {
      if( scipParaSolver->getBigDualGapSubtreeHandlingStrategy() == 0 )
      {
         if( scipParaSolver->getNStopSolvingMode() < 0 ||
             ( scipParaSolver->getNStopSolvingMode() > 0 &&
               nNodes < scipParaSolver->getNStopSolvingMode() ) ) 
         {
            scipParaSolver->sendAnotherNodeRequest(bestDualBoundValue);   // throw away nodes
         }
      }
      else if( scipParaSolver->getBigDualGapSubtreeHandlingStrategy() == 1 )
      {
         if(  scipParaSolver->getCurrentSolivingNodeMergingStatus() != 0 )
         {
            scipParaSolver->setSendBackAllNodes();
         }
      }
      else
      {
         THROW_LOGICAL_ERROR2("Invalid big dual gap handling strategy. startegy = ",
               scipParaSolver->getBigDualGapSubtreeHandlingStrategy() );
      }
   }
   /*******************************************************************
    * if new ParaNode was received or Interrupt message was received, *
    * stop solving the current ParaNode                               *
    ******************************************************************/
   if( cloned ||
         ( SCIPeventGetType(event) != SCIP_EVENTTYPE_NODEFOCUSED
         && !scipParaSolver->isInCollectingMode() ) )
   {
      return SCIP_OKAY; // process until SCIP_EVENTTYPE_NODEFOCUSED event
   }

   if( scipParaSolver->getCurrentNode()->isRootTask() &&
         (!startedCollectingNodesForInitialRampUp ) &&
         scipParaSolver->getParaParamSet()->getIntParamValue(UG::NumberOfNodesKeepingInRootSolver) > 0
         )
   {
      if( SCIPgetNNodesLeft(scip) < scipParaSolver->getParaParamSet()->getIntParamValue(UG::NumberOfNodesKeepingInRootSolver) )
      {
         return SCIP_OKAY;
      }
      else
      {
         startedCollectingNodesForInitialRampUp = true;
      }
   }

   /******************************************************************
    * if node depth is smaller than that specified by parameter,
    * not to send nodes and keep them.
    *****************************************************************/
   if( SCIPnodeGetDepth( SCIPgetCurrentNode( scip ) )
         < scipParaSolver->getParaParamSet()->getIntParamValue(UG::KeepNodesDepth) &&
         !scipParaSolver->isCollectingAllNodes() )
   {
      return SCIP_OKAY;
   }

   /*******************************************************************
    * if ParaNode transfer has been requested, send a ParaNode        *
    ******************************************************************/
   if( ( needToSendNode &&
           (
                SCIPgetNNodesLeft( scip ) > scipParaSolver->getThresholdValue(SCIPgetNNodes(scip)) ||
                scipParaSolver->isInterrupting() ||
                ( SCIPgetNNodesLeft( scip ) > 1 &&
                      ( scipParaSolver->isAggressiveCollecting() ||
                            ( !scipParaSolver->isRampUp() &&
                                  ( bestDualBoundValue - scipParaSolver->getLcBestDualBoundValue() < 0.0 ||
                                         ( ( bestDualBoundValue - scipParaSolver->getLcBestDualBoundValue() ) > 0.0 &&
                                                          ( REALABS( ( bestDualBoundValue - scipParaSolver->getLcBestDualBoundValue() )
                                                                / std::max(REALABS(scipParaSolver->getLcBestDualBoundValue()),1.0) ) < scipParaSolver->getBoundGapForCollectingMode() )
                                          )
                                   )
                            )
                      )
                )
           )
       )
       || scipParaSolver->isManyNodesCollectionRequested()
       || ( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::DualBoundGainTest) &&
             scipParaSolver->isDualBoundGainTestNeeded() &&
            (!scipParaSolver->isEnoughGainObtained()) )
       || ( !scipParaSolver->isRacingStage() &&
             scipParaSolver->isAggressivePresolvingSpecified() &&
             scipParaSolver->getSubMipDepth() <
             ( scipParaSolver->getOffsetDepth() + scipParaSolver->getAggresivePresolvingStopDepth() ) &&
             SCIPgetDepth(scip) > scipParaSolver->getAggresivePresolvingDepth() )
   )
   {
      if( !scipParaSolver->getParaParamSet()->getBoolParamValue(UG::GenerateReducedCheckpointFiles) &&
            !scipParaSolver->isAnotherNodeIsRequested() )
      {
         if( checkRootNodeSolvabilityAndSendParaNode(scip) )
         {
            if( ( scipParaSolver->isManyNodesCollectionRequested() && (!scipParaSolver->isCollectingAllNodes() ) ) &&
                  scipParaSolver->isBreaking() &&
                  ( scipParaSolver->isTransferLimitReached() ||
                        ( bestDualBoundValue > scipParaSolver->getTargetBound() ) ) )
            {
               scipParaSolver->resetBreakingInfo();
            }
         }
         else
         {
            return SCIP_OKAY;
         }

      }
      else
      {
         assert( nNodes <=2 );
         return SCIP_OKAY;
      }
   }

   if( !scipParaSolver->isManyNodesCollectionRequested() )
   {
      if( !scipParaSolver->isRampUp() )
      {
         // if( ( !scipParaSolver->isWarmStarted() ) && scipParaSolver->isRacingRampUp() )
         if( scipParaSolver->isRacingRampUp() )
         {
            if( scipParaSolver->isRacingWinner() || !scipParaSolver->isRacingStage() )
            {
               if( originalSelectionStrategy )
               // if( originalSelectionStrategy && SCIPgetNNodesLeft(scip) > scipParaSolver->getNPreviousNodesLeft() )
               {
                  changeSearchStrategy(scip);
               }
               needToSendNode = true;
            }
         }
         else
         {
            if( ( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::BreakFirstSubtree)
                  && paraComm->getRank() == 1 ) ||
                  bestDualBoundValue - scipParaSolver->getLcBestDualBoundValue() < 0.0 ||
                  ( ( bestDualBoundValue - scipParaSolver->getLcBestDualBoundValue() ) > 0.0 &&
                        ( REALABS( ( bestDualBoundValue - scipParaSolver->getLcBestDualBoundValue() )
                              / scipParaSolver->getLcBestDualBoundValue() ) < scipParaSolver->getBoundGapForCollectingMode() ) ) )
            {
               if( originalSelectionStrategy )
               // if( originalSelectionStrategy && SCIPgetNNodesLeft(scip) > scipParaSolver->getNPreviousNodesLeft() )
               {
                  changeSearchStrategy(scip);
               }
               if( scipParaSolver->getParaParamSet()->getIntParamValue(UG::NoAlternateSolving) > 0 &&
                     SCIPgetNNodesLeft(scip) > scipParaSolver->getParaParamSet()->getIntParamValue(UG::NoAlternateSolving) )
               {
                  needToSendNode = true;
               }
               else
               {
                  if( needToSendNode ) needToSendNode = false;
                  else needToSendNode = true;
               }
            }
            else
            {
               needToSendNode = false;
            }
         }
      }
      else
      {
         /*********************************************
          * check if solver is in collecting mode     *
          ********************************************/
         if( scipParaSolver->isInCollectingMode() )
         {
            if( originalSelectionStrategy )
            // if( originalSelectionStrategy && SCIPgetNNodesLeft(scip) > scipParaSolver->getNPreviousNodesLeft() )
            {
               changeSearchStrategy(scip);
            }
            if( scipParaSolver->getNSendInCollectingMode() > 0 || scipParaSolver->isAggressiveCollecting() )
            {
               needToSendNode = true;
            }
            else
            {
               if( scipParaSolver->getParaParamSet()->getIntParamValue(UG::NoAlternateSolving) > 0 &&
                     SCIPgetNNodesLeft(scip) > scipParaSolver->getParaParamSet()->getIntParamValue(UG::NoAlternateSolving) )
               {
                  needToSendNode = true;
               }
               else
               {
                  if( needToSendNode ) needToSendNode = false;
                  else needToSendNode = true;
               }
            }
         }
         else
         {
            if( !originalSelectionStrategy )
            {
               SCIP_NODESEL *nodesel = SCIPgetNodesel(scip);
               SCIP_CALL_ABORT( SCIPsetNodeselStdPriority(scip, nodesel,
                     scipParaSolver->getOriginalPriority() ) );
               originalSelectionStrategy = true;
               scipParaSolver->setNPreviousNodesLeft(SCIPgetNNodesLeft(scip));
            }
            needToSendNode = false;   // In iReceive, more than on collecting mode message may be received
         }
      }
   }

   return SCIP_OKAY;
}

// void
bool
ScipParaObjCommPointHdlr::checkRootNodeSolvabilityAndSendParaNode(
      SCIP*  scip
      )
{
   SCIP_NODE* node = SCIPgetCurrentNode( scip );
   int depth = SCIPnodeGetDepth( node );
   SCIP_VAR **branchVars = new SCIP_VAR*[depth];
   SCIP_Real *branchBounds = new SCIP_Real[depth];
   SCIP_BOUNDTYPE *boundTypes = new  SCIP_BOUNDTYPE[depth];
   int nBranchVars;
   SCIPnodeGetAncestorBranchings( node, branchVars, branchBounds, boundTypes, &nBranchVars, depth );
   if( nBranchVars > depth )  // did not have enough memory, then reallocate
   {
      delete [] branchVars;
      delete [] branchBounds;
      delete [] boundTypes;
      branchVars = new SCIP_VAR*[nBranchVars];
      branchBounds = new SCIP_Real[nBranchVars];
      boundTypes = new  SCIP_BOUNDTYPE[nBranchVars];
      SCIPnodeGetAncestorBranchings( node, branchVars, branchBounds, boundTypes, &nBranchVars, nBranchVars );
   }

   if( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::AllBoundChangesTransfer) &&
         !( scipParaSolver->getParaParamSet()->getBoolParamValue(UG::NoAllBoundChangesTransferInRacing) &&
               scipParaSolver->isRacingStage() ) )
   {
      int nVars = SCIPgetNVars(scip);
      SCIP_VAR **vars = SCIPgetVars(scip);
      int *iBranchVars = new int[nBranchVars];
      /* create the variable mapping hash map */
      SCIP_HASHMAP* varmapLb;
      SCIP_HASHMAP* varmapUb;
      SCIP_CALL_ABORT( SCIPhashmapCreate(&varmapLb, SCIPblkmem(scip), nVars) );
      SCIP_CALL_ABORT( SCIPhashmapCreate(&varmapUb, SCIPblkmem(scip), nVars) );
      for( int i = 0; i < nBranchVars; i++ )
      {
         iBranchVars[i] = i;
         if( boundTypes[i] == SCIP_BOUNDTYPE_LOWER )
         {
            if( !SCIPhashmapGetImage(varmapLb, branchVars[i]) )
            {
               SCIP_CALL_ABORT( SCIPhashmapInsert(varmapLb, branchVars[i], &iBranchVars[i] ) );
            }
         }
         else
         {
            if( !SCIPhashmapGetImage(varmapUb, branchVars[i]) )
            {
               SCIP_CALL_ABORT( SCIPhashmapInsert(varmapUb, branchVars[i], &iBranchVars[i] ) );
            }
         }
      }
      SCIP_VAR **preBranchVars = branchVars;
      SCIP_Real *preBranchBounds = branchBounds;
      SCIP_BOUNDTYPE *preBboundTypes = boundTypes;
      branchVars = new SCIP_VAR*[nBranchVars+nVars*2];
      branchBounds = new SCIP_Real[nBranchVars+nVars*2];
      boundTypes = new  SCIP_BOUNDTYPE[nBranchVars+nVars*2];
      for( int i = 0; i < nBranchVars; i++ )
      {
         branchVars[i] = preBranchVars[i];
         branchBounds[i] = preBranchBounds[i];
         boundTypes[i] = preBboundTypes[i];
      }
      int *iBranchVar = NULL;
      for( int i = 0; i < nVars; i++ )
      {
         if( scipParaSolver->isCopyIncreasedVariables() &&
            scipParaSolver->getOriginalIndex(i) >= scipParaSolver->getNOrgVars() )
         {
            continue;
         }
         iBranchVar =  (int *)SCIPhashmapGetImage(varmapLb, vars[i]);
         if( iBranchVar )
         {
            // assert( EPSGE(preBranchBounds[*iBranchVar], SCIPvarGetLbLocal(vars[i]) ,DEFAULT_NUM_EPSILON ) );
            if( EPSLT(preBranchBounds[*iBranchVar], SCIPvarGetLbLocal(vars[i]) ,DEFAULT_NUM_EPSILON ) )
            {
               branchBounds[*iBranchVar] = SCIPvarGetLbLocal(vars[i]);  // node is current node
               if ( EPSGT(branchBounds[*iBranchVar], SCIPvarGetUbGlobal(vars[i]), DEFAULT_NUM_EPSILON) ) abort();
            }
         }
         else
         {
            if( EPSGT( SCIPvarGetLbLocal(vars[i]), SCIPvarGetLbGlobal(vars[i]), MINEPSILON ) )
            {
               branchVars[nBranchVars] = vars[i];
               branchBounds[nBranchVars] = SCIPvarGetLbLocal(vars[i]);
               boundTypes[nBranchVars] = SCIP_BOUNDTYPE_LOWER;
               if ( EPSGT(branchBounds[nBranchVars], SCIPvarGetUbGlobal(vars[i]), DEFAULT_NUM_EPSILON) ) abort();
               nBranchVars++;
            }
         }
         iBranchVar = (int *)SCIPhashmapGetImage(varmapUb, vars[i]);
         if( iBranchVar )
         {
            // assert( EPSLE(preBranchBounds[*iBranchVar], SCIPvarGetUbLocal(vars[i]) ,DEFAULT_NUM_EPSILON ) );
            if( EPSGT(preBranchBounds[*iBranchVar], SCIPvarGetUbLocal(vars[i]) ,DEFAULT_NUM_EPSILON ) )
            {
               branchBounds[*iBranchVar] = SCIPvarGetUbLocal(vars[i]); // node is current node
               if ( EPSLT(branchBounds[*iBranchVar], SCIPvarGetLbGlobal(vars[i]),DEFAULT_NUM_EPSILON) ) abort();
            }
         }
         else
         {
            if( EPSLT( SCIPvarGetUbLocal(vars[i]), SCIPvarGetUbGlobal(vars[i]), MINEPSILON ) )
            {
               branchVars[nBranchVars] = vars[i];
               branchBounds[nBranchVars] = SCIPvarGetUbLocal(vars[i]);
               boundTypes[nBranchVars] = SCIP_BOUNDTYPE_UPPER;
               if ( EPSLT(branchBounds[nBranchVars], SCIPvarGetLbGlobal(vars[i]),DEFAULT_NUM_EPSILON) ) abort();
               nBranchVars++;
            }
         }
      }
      SCIPhashmapFree(&varmapLb);
      SCIPhashmapFree(&varmapUb);
      delete [] preBranchVars;
      delete [] preBranchBounds;
      delete [] preBboundTypes;
      delete [] iBranchVars;
   }

   /** check root node solvability */
   if( scipParaSolver->getParaParamSet()->getBoolParamValue(RootNodeSolvabilityCheck) )
   {
      SCIP_CALL_ABORT( SCIPtransformProb(scipToCheckRootSolvability) );
      SCIP_VAR **copyVars = SCIPgetVars(scipToCheckRootSolvability);
      for(int v = 0; v < nBranchVars; v++)
      {
         int index = SCIPvarGetProbindex(branchVars[v]);
         assert(index != -1);
         assert(std::string(SCIPvarGetName(branchVars[v]))==std::string(SCIPvarGetName(copyVars[index])));
         if( boundTypes[v] == SCIP_BOUNDTYPE_LOWER )
         {
            SCIP_CALL_ABORT(SCIPchgVarLbGlobal(scipToCheckRootSolvability,copyVars[index], branchBounds[v]));
         }
         else if (boundTypes[v] == SCIP_BOUNDTYPE_UPPER)
         {
            SCIP_CALL_ABORT(SCIPchgVarUbGlobal(scipToCheckRootSolvability,copyVars[index], branchBounds[v]));
         }
         else
         {
            THROW_LOGICAL_ERROR2("Invalid bound type: type = ", static_cast<int>(boundTypes[v]));
         }
      }
      SCIP_CALL_ABORT(SCIPsetLongintParam(scipToCheckRootSolvability,"limits/nodes", 1));
      SCIP_CALL_ABORT(SCIPsetObjlimit(scipToCheckRootSolvability, scipParaSolver->getGlobalBestIncumbentValue()));
      SCIP_CALL_ABORT(SCIPsolve(scipToCheckRootSolvability));

      SCIP_STATUS status = SCIPgetStatus(scipToCheckRootSolvability);

      switch(status)
      {
      case SCIP_STATUS_OPTIMAL :
      {
         SCIP_SOL *bestSol = SCIPgetBestSol( scip );
         int nVars = SCIPgetNOrigVars(scip);
         SCIP_VAR **vars = SCIPgetOrigVars(scip);
         SCIP_Real *vals = new SCIP_Real[nVars];
         SCIP_CALL_ABORT( SCIPgetSolVals(scip, bestSol, nVars, vars, vals) );
         DEF_SCIP_PARA_COMM( scipParaComm, paraComm);
         scipParaSolver->saveIfImprovedSolutionWasFound(
               scipParaComm->createScipParaSolution(
                     scipParaSolver,
                     SCIPgetSolOrigObj(scip, bestSol),
                     nVars,
                     vars,
                     vals
                  )
               );
         delete [] vals;
         /** remove the node sent from SCIP environment */
         SCIP_CALL_ABORT( SCIPcutoffNode( scip, node) );
         needToSendNode = false;   // This means that a node is sent in the next time again,
                                   // because this flag is flipped when collecting mode is checked
         scipParaSolver->countInPrecheckSolvedParaNodes();
         break;
      }
      case SCIP_STATUS_INFEASIBLE :
      case SCIP_STATUS_INFORUNBD :
      {
         /** remove the node sent from SCIP environment */
         SCIP_CALL_ABORT( SCIPcutoffNode( scip, node ) );
         needToSendNode = false;   // This means that a node is sent in the next time again,
                                    // because this flag is flipped when collecting mode is checked
         scipParaSolver->countInPrecheckSolvedParaNodes();
         break;
      }
      case SCIP_STATUS_NODELIMIT :
      {
         sendNode(scip, node, depth, nBranchVars, branchVars, branchBounds, boundTypes);
         break;
      }
      default:
         THROW_LOGICAL_ERROR2("Invalid status after root solvability check: status = ", static_cast<int>(status));
      }

      SCIP_CALL_ABORT( SCIPfreeTransform(scipToCheckRootSolvability) );
   }
   else
   {
      if( scipParaSolver->isCopyIncreasedVariables() ) // this may not need, but only for this error occurred so far.
      {
         if( !ifFeasibleInOriginalProblem(scip, nBranchVars, branchVars, branchBounds) )
         {
            delete [] branchVars;
            delete [] branchBounds;
            delete [] boundTypes;
            return false;
         }
      }
      sendNode(scip, node, depth, nBranchVars, branchVars, branchBounds, boundTypes);
   }

   delete [] branchVars;
   delete [] branchBounds;
   delete [] boundTypes;

   return true;
}

void
ScipParaObjCommPointHdlr::sendNode(
      SCIP *scip,
      SCIP_NODE* node,
      int depth,
      int nNewBranchVars,
      SCIP_VAR **newBranchVars,
      SCIP_Real *newBranchBounds,
      SCIP_BOUNDTYPE *newBoundTypes
      )
{

   SCIP_CONS** addedcons = 0;
   int addedconsssize = SCIPnodeGetNAddedConss(node);
   int naddedconss = 0;
   if( addedconsssize > 0 )
   {
      SCIP_CALL_ABORT( SCIPallocBufferArray(scip, &addedcons, addedconsssize) );
      SCIPnodeGetAddedConss(node, addedcons, &naddedconss, addedconsssize);
   }


   DEF_SCIP_PARA_COMM( scipParaComm, paraComm);
   ScipParaDiffSubproblem *diffSubproblem = scipParaComm->createScipParaDiffSubproblem(
         scip,
         scipParaSolver,
         nNewBranchVars,
         newBranchVars,
         newBranchBounds,
         newBoundTypes,
         naddedconss,
         addedcons
         );

   if( naddedconss  )
   {
      SCIPfreeBufferArray(scip, &addedcons);
   }


   long long n = SCIPnodeGetNumber( node );
   double dualBound = SCIPgetDualbound(scip);
   // if( SCIPisObjIntegral(scip) )
   // {
   //    dualBound = ceil(dualBound);
   // }
   assert(SCIPisFeasGE(scip, SCIPnodeGetLowerbound(node) , SCIPgetLowerbound(scip)));
   double estimateValue = SCIPnodeGetEstimate( node );
   assert(SCIPisFeasGE(scip, estimateValue, SCIPnodeGetLowerbound(node) ));
#ifdef UG_DEBUG_SOLUTION
   SCIP_Bool valid = 0;
   SCIP_CALL_ABORT( SCIPdebugSolIsValidInSubtree(scip, &valid) );
   diffSubproblem->setOptimalSolIndicator(valid);
   std::cout << "* R." << scipParaSolver->getRank() << ", debug = " << SCIPdebugSolIsEnabled(scip) << ", valid = " << valid << std::endl;
#endif
   scipParaSolver->sendParaNode(n, depth, dualBound, estimateValue, diffSubproblem);

   /** remove the node sent from SCIP environment */
#ifdef UG_DEBUG_SOLUTION
   if( valid )
   {
      SCIPdebugSolDisable(scip);
      std::cout << "R." << paraComm->getRank() << ": disable debug, node which contains optmal solution is sent." << std::endl;
   }
#endif
   SCIP_CALL_ABORT( SCIPcutoffNode( scip, node) );
}

void
ScipParaObjCommPointHdlr::changeSearchStrategy(
      SCIP*  scip
      )
{
   scipParaSolver->setNPreviousNodesLeft(SCIPgetNNodesLeft(scip));
   int numnodesels = SCIPgetNNodesels( scip );
   SCIP_NODESEL** nodesels = SCIPgetNodesels( scip );
   int i;
   for( i = 0; i < numnodesels; ++i )
   {
      std::string nodeselname(SCIPnodeselGetName(nodesels[i]));
      if( std::string(nodeselname) == std::string(changeNodeSelName) )
      {
         SCIP_CALL_ABORT( SCIPsetNodeselStdPriority(scip, nodesels[i], 536870911 ) );
         originalSelectionStrategy = false;
         int maxrestarts;
         SCIP_CALL_ABORT( SCIPgetIntParam(scip, "presolving/maxrestarts", &maxrestarts) );
         if( maxrestarts != 0 )
         {
            SCIP_CALL_ABORT( SCIPsetIntParam(scip, "presolving/maxrestarts", 0) );
         }
         break;
      }
   }
   assert( i != numnodesels );
}

bool
ScipParaObjCommPointHdlr::ifFeasibleInOriginalProblem(
      SCIP *scip,
      int nNewBranchVars,
      SCIP_VAR **newBranchVars,
      SCIP_Real *newBranchBounds)
{

   bool feasible = true;
   SCIP_Real *branchBounds = new SCIP_Real[nNewBranchVars];
   for( int v = nNewBranchVars -1 ; v >= 0; --v )
   {
      SCIP_VAR *transformVar = newBranchVars[v];
      SCIP_Real scalar = 1.0;
      SCIP_Real constant = 0.0;
      SCIP_CALL_ABORT( SCIPvarGetOrigvarSum(&transformVar, &scalar, &constant ) );
      if( transformVar == NULL ) continue;
      // assert( scalar == 1.0 && constant == 0.0 );
      branchBounds[v] = ( newBranchBounds[v] - constant ) / scalar;
      if( SCIPvarGetType(transformVar) != SCIP_VARTYPE_CONTINUOUS
          && SCIPvarGetProbindex(transformVar) < scipParaSolver->getNOrgVars() )
      {
         if( !(SCIPisLE(scip,scipParaSolver->getOrgVarLb(SCIPvarGetProbindex(transformVar)), branchBounds[v]) &&
               SCIPisGE(scip,scipParaSolver->getOrgVarUb(SCIPvarGetProbindex(transformVar)), branchBounds[v])) )
         {
            feasible = false;
            break;
         }
      }
   }
   delete [] branchBounds;

   return feasible;
}
