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

/**@file    paraInitiator.h
 * @brief   Base class of initiator that maintains original problem and incumbent solution.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_INITIATOR_H__
#define __BB_PARA_INITIATOR_H__

#include <string>

#include "ug/uggithash.h"
#include "ug/paraInitiator.h"
#include "bbParaDiffSubproblem.h"
#include "bbParaSolution.h"
#include "bbParaComm.h"
#include "bbParaNode.h"

#ifdef UG_WITH_UGS
#include "ugs/ugsDef.h"
#include "ugs/ugsParaCommMpi.h"
#endif

namespace UG
{

///
/// Final status of computation
///
enum FinalSolverState {
   InitialNodesGenerated,            ///< initial nodes were generated
   Aborted,                          ///< aborted
   HardTimeLimitIsReached,           ///< hard time limit is reached
   MemoryLimitIsReached,             ///< memory limit is reached in a solver
   GivenGapIsReached,                ///< given gap is reached for the computation
   ComputingWasInterrupted,          ///< computing was interrupted
   ProblemWasSolved,                 ///< problem was solved
   RequestedSubProblemsWereSolved    ///< requested subproblem was solved
};

///
/// Class for initiator
///
class BbParaInitiator : public ParaInitiator
{

protected:

   bool       solvedAtInit;                     ///< solved at init
   bool       solvedAtReInit;                   ///< solved at reInit
   double     *tightenedVarLbs;                 ///< array of tightened lower bound of variable
   double     *tightenedVarUbs;                 ///< array of tightened upper bound of variable

#ifdef UG_WITH_ZLIB
   gzstream::igzstream  checkpointTasksStream;  ///< gzstream for checkpoint tasks file
#endif

public:

   ///
   /// constructor
   ///
   BbParaInitiator(
         ParaComm *inComm,                     ///< communicator used
         ParaTimer *inTimer                    ///< timer used
         )
         : ParaInitiator(inComm, inTimer),
           solvedAtInit(false),
           solvedAtReInit(false),
           tightenedVarLbs(0),
           tightenedVarUbs(0)
   {
   }

   ///
   /// destructor
   ///
   virtual ~BbParaInitiator(
         )
   {
      if( tightenedVarLbs ) delete [] tightenedVarLbs;
      if( tightenedVarUbs ) delete [] tightenedVarUbs;
   }

#ifdef UG_WITH_ZLIB

   ///
   /// read a ParaNode from checkpoint file
   /// @return ParaNode object, 0 in the end of file
   ///
   BbParaNode *readParaNodeFromCheckpointFile(
         bool onlyBoundChanges        ///< indicate if it read only bound changes of not.
                                      ///< true: only bound changes, fase: else
         // bool hasMergingStatus
         )
   {
      BbParaNode *paraNode = dynamic_cast<BbParaNode *>(paraComm->createParaTask());
      if( paraNode->read(paraComm, checkpointTasksStream, onlyBoundChanges) )
      {
         return paraNode;
      }
      else
      {
         delete paraNode;
         checkpointTasksStream.close();
         return 0;
      }
   }

#endif

   ///
   /// check if problem is solved at init or not
   /// @return true if problem is solved, false otherwise
   ///
   bool isSolvedAtInit(
         )
   {
      return solvedAtInit;
   }

   ///
   /// check if problem is solved at reInit or not
   /// @return true if problem is solved, false otherwise
   ///
   bool isSolvedAtReInit(
         )
   {
      return solvedAtReInit;
   }
   
   ///
   /// set tightened variable lower bound
   /// TODO: this function should be in inherited class
   ///
   void setTightenedVarLbs(
         int i,           ///< index of variable
         double v         ///< tightened bound
         )
   {
      assert(tightenedVarLbs);
      tightenedVarLbs[i] = v;
      // could detect infeasibility
      // assert( EPSLE(tightenedVarLbs[i],tightenedVarUbs[i], MINEPSILON) );
   }

   ///
   /// set tightened variable upper bound
   /// TODO: this function should be in inherited class
   ///
   void setTightenedVarUbs(
         int i,          ///< index of variable
         double v        ///< tightened bound
         )
   {
      assert(tightenedVarUbs);
      tightenedVarUbs[i] = v;
      // could detect infeasibility
      // assert( EPSLE(tightenedVarLbs[i],tightenedVarUbs[i], MINEPSILON) );
   }

   ///
   /// get tightened variable lower bound
   /// TODO: this function should be in inherited class
   /// @return lower bound
   ///
   double getTightenedVarLbs(
         int i            ///< index of variable
         )
   {
      if( tightenedVarLbs )
      {
         return tightenedVarLbs[i];
      }
      else
      {
         return DBL_MAX;
      }
   }

   ///
   /// get tightened variable upper bound
   /// TODO: this function should be in inherited class
   /// @return uppper bound
   ///
   double getTightenedVarUbs(
         int i            ///< index of variable
         )
   {
      if( tightenedVarUbs )
      {
         return tightenedVarUbs[i];
      }
      else
      {
         return -DBL_MAX;
      }
   }

   ///
   /// check if there are tightened lower or upper bound
   /// TODO: this function should be in inherited class
   /// @return true if there is a tightened lower or upper bound, false otherwise
   ///
   bool areTightenedVarBounds(
         )
   {
      assert( (tightenedVarLbs && tightenedVarUbs) || ( (!tightenedVarLbs) && (!tightenedVarUbs) ) );
      return ( tightenedVarLbs != 0 );
   }

   ///
   /// make DiffSubproblem object for root node
   /// @return pointer to the root ParaDiffSubproblem object
   ///
   virtual BbParaDiffSubproblem *makeRootNodeDiffSubproblem(
         ) = 0;

   ///
   /// convert objective function value to external value
   /// TODO: this function may be in inherited class
   /// @return objective function value as in external value
   ///
   virtual double convertToExternalValue(
         double internalValue                               ///< internal value of the objective function
         ) = 0;


   ///
   /// get global best incumbent solution
   /// @return pinter to ParaSolution object
   ///
   virtual BbParaSolution *getGlobalBestIncumbentSolution(
         ) = 0;

   ///
   /// get the number of incumbent solutions
   /// @return the number of incumbent solutions
   ///
   virtual int getNSolutions(
         ) = 0;

   ///
   /// try to set incumbent solution
   /// @return true if solution is set successfully, false otherwise
   ///
   virtual bool tryToSetIncumbentSolution(
         BbParaSolution *sol,                                 ///< pointer to ParaSolution object to be set
         bool checksol                                      ///< true if the solution feasibility need to be checked
         ) = 0;

   ///
   /// get absolute gap of dual bound value
   /// @return absolute gap
   ///
   virtual double getAbsgap(
         double dualBoundValue                              ///< dual bound value
         ) = 0;

   ///
   /// get relative gap of dual bound value
   /// @return relative gap
   ///
   virtual double getGap(
         double dualBoundValue                               ///< dual bound value
         ) = 0;

   ///
   /// get absgap value specified
   /// @return absgap value
   ///
   virtual double getAbsgapValue(
         ) = 0;

   ///
   /// get gap value specified
   /// @return gap value
   ///
   virtual double getGapValue(
         ) = 0;

   ///
   /// set final solver status
   ///
   virtual void setFinalSolverStatus(
         FinalSolverState status                        ///< solver status
         ) = 0;

   ///
   /// set number of nodes solved
   ///
   virtual void setNumberOfNodesSolved(
         long long n                                    ///< the number of nodes solved
         ) = 0;

   ///
   /// set final dual bound
   ///
   virtual void setDualBound(
         double bound                                   ///< dual bound value
         ) = 0;


   ///
   /// check if feasible solution exists or not
   /// @return true if a feasible solution exists, false otherwise
   ///
   virtual bool isFeasibleSolution(
         ) = 0;

   ///
   /// accumulate initial status
   ///
   virtual void accumulateInitialStat(
         ParaInitialStat *initialStat                    ///< initial status collected
         )
   {
   }

   ///
   /// set initial status on DiffSubproblem
   ///
   virtual void setInitialStatOnDiffSubproblem(
         int minDepth,                                  ///< minimum depth
         int maxDepth,                                  ///< maximum depth
         BbParaDiffSubproblem *diffSubproblem           ///< pointer to ParaDiffSubproblem object
         )
   {
   }

   ///
   /// check if objective function value is always integral or not
   /// @return true if it is always integral, false others
   ///
   virtual bool isObjIntegral(
         )
   {
      return false;
   }

   ///
   /// check if solver can generate special cut off value or not
   /// @return true if it can be generated, false others
   ///
   virtual bool canGenerateSpecialCutOffValue(
         )
   {
      return false;
   }

};

typedef ParaInitiator *ParaInitiatorPtr;

}

#endif // __BB_PARA_INITIATOR_HPP__
