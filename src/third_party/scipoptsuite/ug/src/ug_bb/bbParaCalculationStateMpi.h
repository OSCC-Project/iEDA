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

/**@file    paraCalculationStateMpi.h
 * @brief   CalcutationStte object extension for MPI communication
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __BB_PARA_CALCULATION_STATE_MPI_H__
#define __BB_PARA_CALCULATION_STATE_MPI_H__

#include <mpi.h>
#include "bbParaCommMpi.h"
#include "bbParaCalculationState.h"

namespace UG
{

///
/// Calculation state object for MPI communications
///
class BbParaCalculationStateMpi : public BbParaCalculationState
{

   ///
   /// create MPI datatype of this object
   /// @return MPI dataypte of this object
   ///
   MPI_Datatype createDatatype(
         );

public:

   ///
   /// default constructor of this object
   ///
   BbParaCalculationStateMpi(
         )
   {
   }

   ///
   /// constructor of this object
   ///
   BbParaCalculationStateMpi(
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
         : BbParaCalculationState(inCompTime,inRootTime, inNSolved, inNSent,inNImprovedIncumbent,inTerminationState,inNSolvedWithNoPreprocesses,
               inNSimplexIterRoot, inAverageSimplexIter,
               inNTransferredLocalCuts, inMinTransferredLocalCuts, inMaxTransferredLocalCuts,
               inNTransferredBendersCuts, inMinTransferredBendersCuts, inMaxTransferredBendersCuts,
               inNRestarts, inMinIisum, inMaxIisum, inMinNii, inMaxNii, inDualBound, inNSelfSplitNodesLeft )
   {
   }

   ///
   /// destructor of this object
   ///
   ~BbParaCalculationStateMpi(
         )
   {
   }

   ///
   /// send this object to destination
   ///
   void send(
         ParaComm *comm,     ///< communicator used to send this object
         int destination,    ///< destination rank to send
         int tag             ///< tag to show this object
         );

   ///
   /// receive this object from source
   ///
   void receive(
         ParaComm *comm,     ///< communicator used to receive this object
         int source,         ///< source rank to receive this object
         int tag             ///< tag to show this object
         );

};

}

#endif // __BB_PARA_CALCULATION_STATE_MPI_H__

