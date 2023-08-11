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

/**@file    paraSolverTerminationState.cpp
 * @brief   This class contains solver termination state which is transferred form Solver to LC.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#include "ug/paraDef.h"
#include "ug/paraComm.h"
#include "bbParaSolverTerminationState.h"

using namespace UG;

/** stringfy BbParaSolverTerminationState */
std::string
BbParaSolverTerminationState::toString(
      ParaInitiator *initiator
      )
{
   std::ostringstream os;
   switch( interrupted )
   {
   case 0:
   {
      os << "######### Solver Rank = " << rank << " is terminated. #########" << std::endl;
      break;
   }
   case 1:
   {
      os << "######### Solver Rank = " << rank << " is interrupted. #########" << std::endl;
      break;
   }
   case 2:
   {
      os << "######### Solver Rank = " << rank << " is at checkpoint. #########" << std::endl;
      break;
   }
   case 3:
   {
      os << "######### Solver Rank = " << rank << " is at the end of racing process. #########" << std::endl;
      break;
   }
   default:
   {
      THROW_LOGICAL_ERROR1("invalid interrupted flag in BbParaSolverTerminationState!");
   }
   }

    os << "#=== Elapsed time to terminate this Solver = " << runningTime << std::endl;
    os << "#=== Total computing time = " << runningTime - (idleTimeToFirstParaTask+idleTimeBetweenParaTasks+idleTimeAfterLastParaTask+idleTimeToWaitNotificationId+idleTimeToWaitToken ) << std::endl;
    os << "#=== Total idle time = " << (idleTimeToFirstParaTask+idleTimeBetweenParaTasks+idleTimeAfterLastParaTask+idleTimeToWaitNotificationId+idleTimeToWaitToken) << std::endl;
    os << "#=== ( Idle time to start first ParaNode = " << idleTimeToFirstParaTask
    << ", Idle time between ParaNods = " << idleTimeBetweenParaTasks
    << ", Idle Time after last ParaNode = " << idleTimeAfterLastParaTask
    << " )" << std::endl;
    os << "#=== ( Idle time to wait notification Id messages = " << idleTimeToWaitNotificationId << " )" << std::endl;
    os << "#=== ( Idle time to wait acknowledgment of completion = " << idleTimeToWaitAckCompletion << " )" << std::endl;
    os << "#=== ( Idle time to wait token = " << idleTimeToWaitToken << " )" << std::endl;
    if( nParaTasksSolved > 0 )
    {
       os << "#=== Total root node process time = " << totalRootNodeTime << " ( Mean = " << (totalRootNodeTime/nParaTasksSolved)
       << ", Min = " << minRootNodeTime << ", Max = " << maxRootNodeTime << " )" << std::endl;
    }
    else
    {
       os << "#=== Total root node process time = 0.0 ( Mean = 0.0, Min = 0.0, Max = 0.0 )" << std::endl;
    }
   os << "#=== The number of ParaNodes received in this solver = " << nParaTasksReceived << std::endl;
   os << "#=== The number of ParaNodes sent from this solver = " << totalNSent << std::endl;
   if( nParaTasksSolved > 0 )
   {
      os << "#=== The number of nodes solved in this solver = " << totalNSolved
      << " ( / Subtree : Mean = " << totalNSolved/nParaTasksSolved <<  ", Min = " << minNSolved << ", Max = " << maxNSolved << " )"<< std::endl;
      os << "#=== Total number of restarts in this solver = " << nTotalRestarts
            << "( / Subtree : Mean = " << nTotalRestarts/nParaTasksSolved
            << ", Min = " << minRestarts << ", Max = " << maxRestarts << " )"<< std::endl;
      os << "#=== Total number of cuts sent from this solver = " << nTransferredLocalCutsFromSolver
            << "( / Subtree : Mean = " << nTransferredLocalCutsFromSolver/nParaTasksSolved;
      if( nTransferredLocalCutsFromSolver > 0 )
      {
         os << ", Min = " << minTransferredLocalCutsFromSolver << ", Max = " << maxTransferredLocalCutsFromSolver << " )"<< std::endl;
      }
      else
      {
         os << ", Min = 0, Max = 0 )" << std::endl;
      }
      os << "#=== Total number of benders cuts sent from this solver = " << nTransferredBendersCutsFromSolver
            << "( / Subtree : Mean = " << nTransferredBendersCutsFromSolver/nParaTasksSolved;
      if( nTransferredBendersCutsFromSolver > 0 )
      {
         os << ", Min = " << minTransferredBendersCutsFromSolver << ", Max = " << maxTransferredBendersCutsFromSolver << " )"<< std::endl;
      }
      else
      {
         os << ", Min = 0, Max = 0 )" << std::endl;
      }
   }
   else
   {
      os << "#=== The number of nodes solved in this solver = 0 ( / Subtree : Mean = 0, Min = 0, Max = 0 )" << std::endl;
   }
   os << "#=== The number of ParaNodes solved in this solver = " << nParaTasksSolved << std::endl;
   os << "#=== ( Solved at root node  =  " << nParaNodesSolvedAtRoot << ", Solved at pre-checking of root node solvability = "
   << nParaNodesSolvedAtPreCheck << " )" << std::endl;
   os << "#=== The number of improved solutions found in this solver = " << totalNImprovedIncumbent << std::endl;
   os << "#=== The number of tightened variable bounds in this solver = " << nTightened << " ( Int: " << nTightenedInt << " )" << std::endl;
   return os.str();
}

#ifdef UG_WITH_ZLIB
void
BbParaSolverTerminationState::write(
      gzstream::ogzstream &out
      )
{
   out.write((char *)&interrupted, sizeof(int));
   out.write((char *)&rank, sizeof(int));
   out.write((char *)&totalNSolved, sizeof(int));
   out.write((char *)&minNSolved, sizeof(int));
   out.write((char *)&maxNSolved, sizeof(int));
   out.write((char *)&totalNSent, sizeof(int));
   out.write((char *)&totalNImprovedIncumbent, sizeof(int));
   out.write((char *)&nParaTasksReceived, sizeof(int));
   out.write((char *)&nParaTasksSolved, sizeof(int));
   out.write((char *)&nParaNodesSolvedAtRoot, sizeof(int));
   out.write((char *)&nParaNodesSolvedAtPreCheck, sizeof(int));
   out.write((char *)&runningTime, sizeof(double));
   out.write((char *)&idleTimeToFirstParaTask, sizeof(double));
   out.write((char *)&idleTimeBetweenParaTasks, sizeof(double));
   out.write((char *)&idleTimeAfterLastParaTask, sizeof(double));
   out.write((char *)&idleTimeToWaitNotificationId, sizeof(double));
   out.write((char *)&idleTimeToWaitToken, sizeof(double));
   out.write((char *)&totalRootNodeTime, sizeof(double));
   out.write((char *)&minRootNodeTime, sizeof(double));
   out.write((char *)&maxRootNodeTime, sizeof(double));
   // detTime and dualBound are not saved
}

bool
BbParaSolverTerminationState::read(
      ParaComm *comm,
      gzstream::igzstream &in
      )
{
   in.read((char *)&interrupted, sizeof(int));
   if( in.eof() ) return false;
   in.read((char *)&rank, sizeof(int));
   in.read((char *)&totalNSolved, sizeof(int));
   in.read((char *)&minNSolved, sizeof(int));
   in.read((char *)&maxNSolved, sizeof(int));
   in.read((char *)&totalNSent, sizeof(int));
   in.read((char *)&totalNImprovedIncumbent, sizeof(int));
   in.read((char *)&nParaTasksReceived, sizeof(int));
   in.read((char *)&nParaTasksSolved, sizeof(int));
   in.read((char *)&nParaNodesSolvedAtRoot, sizeof(int));
   in.read((char *)&nParaNodesSolvedAtPreCheck, sizeof(int));
   in.read((char *)&runningTime, sizeof(double));
   in.read((char *)&idleTimeToFirstParaTask, sizeof(double));
   in.read((char *)&idleTimeBetweenParaTasks, sizeof(double));
   in.read((char *)&idleTimeAfterLastParaTask, sizeof(double));
   in.read((char *)&idleTimeToWaitNotificationId, sizeof(double));
   in.read((char *)&idleTimeToWaitToken, sizeof(double));
   in.read((char *)&totalRootNodeTime, sizeof(double));
   in.read((char *)&minRootNodeTime, sizeof(double));
   in.read((char *)&maxRootNodeTime, sizeof(double));
   // detTime and dualBound are not saved
   return true;
}

#endif
