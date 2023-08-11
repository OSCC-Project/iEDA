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

/**@file    paraCalculationState.h
 * @brief   Base class for calculation state.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_CALCULATION_STATE_H__
#define __BB_PARA_CALCULATION_STATE_H__

#include <climits>
#include <cfloat>
#include "ug/paraComm.h"
#include "ug/paraCalculationState.h"

namespace UG
{

///
/// \class BbParaCalculationState
/// Base class of Calculation state in a ParaSolver
///
class BbParaCalculationState : public ParaCalculationState
{
protected:
   double rootTime;                   ///< computation time of the root node
   int    nSent;                      ///< the number of ParaNodes sent
   int    nImprovedIncumbent;         ///< the number of improved solution generated in this ParaSolver
   int    nSolvedWithNoPreprocesses;  ///< number of nodes solved when it is solved with no preprocesses
   int    nSimplexIterRoot;           ///< number of simplex iteration at root node
   double averageSimplexIter;         ///< average number of simplex iteration except root node
   int    nTransferredLocalCuts;      ///< number of local cuts transferred from a ParaNode
   int    minTransferredLocalCuts;    ///< minimum number of local cuts transferred from a ParaNode
   int    maxTransferredLocalCuts;    ///< maximum number of local cuts transferred from a ParaNode
   int    nTransferredBendersCuts;    ///< number of benders cuts transferred from a ParaNode
   int    minTransferredBendersCuts;  ///< minimum number of benders cuts transferred from a ParaNode
   int    maxTransferredBendersCuts;  ///< maximum number of benders cuts transferred from a ParaNode
   int    nRestarts;                  ///< number of restarts
   double minIisum;                   ///< minimum sum of integer infeasibility
   double maxIisum;                   ///< maximum sum of integer infeasibility
   int    minNii;                     ///< minimum number of integer infeasibility
   int    maxNii;                     ///< maximum number of integer infeasibility
   double dualBound;                  ///< final dual bound value
   int    nSelfSplitNodesLeft;        ///< number of self-split nodes left
public:

   ///
   /// Default Constructor
   ///
   BbParaCalculationState(
         )
         : ParaCalculationState(),
           rootTime(0.0),
           nSent(-1),
           nImprovedIncumbent(-1),
           nSolvedWithNoPreprocesses(-1),
           nSimplexIterRoot(0),
           averageSimplexIter(0.0),
           nTransferredLocalCuts(0),
           minTransferredLocalCuts(INT_MAX),
           maxTransferredLocalCuts(INT_MIN),
           nTransferredBendersCuts(0),
           minTransferredBendersCuts(INT_MAX),
           maxTransferredBendersCuts(INT_MIN),
           nRestarts(0),
           minIisum(0.0),
           maxIisum(0.0),
           minNii(0),
           maxNii(0),
           dualBound(-DBL_MAX),
           nSelfSplitNodesLeft(0)
   {
   }

   ///
   /// Constructor
   ///
   BbParaCalculationState(
         double inCompTime,                   ///< computation time of this ParaNode
         double inRootTime,                   ///< computation time of the root node
         int    inNSolved,                    ///< the number of nodes solved
         int    inNSent,                      ///< the number of ParaNodes sent
         int    inNImprovedIncumbent,         ///< the number of improved solution generated in this ParaSolver
         int    inTerminationState,           ///< indicate whether if this computation is terminationState or not. 0: no, 1: terminationState
         int    inNSolvedWithNoPreprocesses,  ///< number of nodes solved when it is solved with no preprocesses
         int    inNSimplexIterRoot,           ///< number of simplex iteration at root node
         double inAverageSimplexIter,         ///< average number of simplex iteration except root node
         int    inNTransferredLocalCuts,      ///< number of local cuts transferred from a ParaNode
         int    inMinTransferredLocalCuts,    ///< minimum number of local cuts transferred from a ParaNode
         int    inMaxTransferredLocalCuts,    ///< maximum number of local cuts transferred from a ParaNode
         int    inNTransferredBendersCuts,    ///< number of benders cuts transferred from a ParaNode
         int    inMinTransferredBendersCuts,  ///< minimum number of benders cuts transferred from a ParaNode
         int    inMaxTransferredBendersCuts,  ///< maximum number of benders cuts transferred from a ParaNode
         int    inNRestarts,                  ///< number of restarts
         double inMinIisum,                   ///< minimum sum of integer infeasibility
         double inMaxIisum,                   ///< maximum sum of integer infeasibility
         int    inMinNii,                     ///< minimum number of integer infeasibility
         int    inMaxNii,                     ///< maximum number of integer infeasibility
         double inDualBound,                  ///< final dual Bound value
         int    inNSelfSplitNodesLeft         ///< number of self-split nodes left
         )
         : ParaCalculationState(inCompTime, inNSolved, inTerminationState),
           rootTime(inRootTime),
           nSent(inNSent),
           nImprovedIncumbent(inNImprovedIncumbent),
           nSolvedWithNoPreprocesses(inNSolvedWithNoPreprocesses),
           nSimplexIterRoot(inNSimplexIterRoot),
           averageSimplexIter(inAverageSimplexIter),
           nTransferredLocalCuts(inNTransferredLocalCuts),
           minTransferredLocalCuts(inMaxTransferredLocalCuts),
           maxTransferredLocalCuts(inMaxTransferredLocalCuts),
           nTransferredBendersCuts(inNTransferredBendersCuts),
           minTransferredBendersCuts(inMaxTransferredBendersCuts),
           maxTransferredBendersCuts(inMaxTransferredBendersCuts),
           nRestarts(inNRestarts),
           minIisum(inMinIisum),
           maxIisum(inMaxIisum),
           minNii(inMinNii),
           maxNii(inMaxNii),
           dualBound(inDualBound),
           nSelfSplitNodesLeft(inNSelfSplitNodesLeft)
   {
   }

   ///
   /// Destructor
   ///
   virtual
   ~BbParaCalculationState(
         )
   {
   }

   ///
   /// getter of root node computing time
   /// @return root node computing time
   ///
   double getRootTime(
         )
   {
      return rootTime;
   }

   ///
   /// getter of the number of restart occurred in solving a subproblem
   /// @return the number of restarts
   ///
   int getNRestarts(
         )
   {
      return nRestarts;
   }

   ///
   /// getter of average computing time of a node except root node
   /// @return the average computing time
   ///
   double getAverageNodeCompTimeExcpetRoot(
         )
   {
      if( nSolved > 1 )
      {
         return ((compTime - rootTime)/(nSolved - 1));
      }
      else
      {
         return 0.0;
      }
   }

   ///
   /// getter of the number of nodes transferred from the subproblem solving
   /// @return the number of nodes sent
   ///
   int getNSent(
         )
   {
      return nSent;
   }

   ///
   /// getter of the number of improved incumbents during solving the subproblem
   /// @return the number of the improved incumbents
   ///
   int getNImprovedIncumbent(
         )
   {
      return nImprovedIncumbent;
   }

   ///
   /// getter of the termination state for solving the subproblem
   /// @return the termination state
   ///
   int getTerminationState(
         )
   {
      return terminationState;
   }

   ///
   /// getter of the number of solved nodes in the case that a node is solved without
   /// presolving. This is an experimental routine only used for SCIP parallelization
   /// @return the number of solved node without presolving
   ///
   int getNSolvedWithNoPreprocesses(
         )
   {
      return nSolvedWithNoPreprocesses;
   }

   ///
   /// getter of the final dual bound value
   /// @return the final dual bound value
   ///
   double getDualBoundValue(
         )
   {
      return dualBound;
   }

   ///
   /// getter of the number of self-split nodes left
   /// @return the number of self-split nodes left
   ///
   double getNSelfSplitNodesLeft(
         )
   {
      return nSelfSplitNodesLeft;
   }

   ///
   /// stringfy BbParaCalculationState
   /// @return string to show this object
   ///
   std::string toString(
         )
   {
      std::ostringstream s;
      if( terminationState )
      {
         s << "Termination state of this computation was " << terminationState << " : [ "
         << compTime << " sec. computed ]"
         << nSolved << " nodes were solved, "
         << nSent << " nodes were sent, "
         << nImprovedIncumbent << " improved solutions were found";
      }
      else
      {
         s << "Computation was normally terminated: [ "
         << compTime << " sec. computed ]"
         << nSolved << " nodes were solved, "
         << nSent << " nodes were sent, "
         << nImprovedIncumbent << " improved solutions were found";
      }
      return s.str();
   }

   ///
   /// stringfy BbParaCalculationState (simple string version)
   /// @return simple string to show this object
   ///
   std::string toSimpleString(
         )
   {
      std::ostringstream s;

      s << compTime
            << ", "
            << rootTime
            << ", "
            << nSolved
            << ", "
            << nSent
            << ", "
            << nImprovedIncumbent
            << ", "
            << nSimplexIterRoot
            << ", "
            << averageSimplexIter
            << ", "
            << nRestarts
            << ", ";

      if( maxNii > 0 )
      {
         s << minIisum
               << ", "
               << maxIisum
               << ", "
               << minNii
               << ", "
               << maxNii;
      }
      else
      {
         s << ", -, -, -, -";
      }
      s << ", " << dualBound;
      s << ", " << nSelfSplitNodesLeft;

      return s.str();
   }

};

}

#endif // __BB_PARA_CALCULATION_STATE_H__
