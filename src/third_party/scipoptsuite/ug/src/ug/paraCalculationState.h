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


#ifndef __PARA_CALCULATION_STATE_H__
#define __PARA_CALCULATION_STATE_H__

#include <climits>
#include <cfloat>
#include "paraComm.h"

namespace UG
{

///
/// \class ParaCalculationState
/// Base class of Calculation state in a ParaSolver
///
class ParaCalculationState
{
protected:
   double compTime;                   ///< computation time of this ParaTask
   int    nSolved;                    ///< the number of tasks solved
   int    terminationState;           ///< indicate whether if this computation is terminationState or not. 0: no, 1: terminationState
                                      ///< meaning can be defined in derived class
public:

   ///
   /// Default Constructor
   ///
   ParaCalculationState(
         )
         : compTime(0.0),
           nSolved(-1),
           terminationState(-1)
   {
   }

   ///
   /// Constructor
   ///
   ParaCalculationState(
         double inCompTime,                   ///< computation time of this ParaTask
         int    inNSolved,                     ///< the number of tasks solved
         int    inTerminationState            ///< indicate whether if this computation is terminationState or not. 0: no, 1: terminationState
         )
         : compTime(inCompTime),
           nSolved(inNSolved),
           terminationState(inTerminationState)
   {
   }

   ///
   /// Destructor
   ///
   virtual
   ~ParaCalculationState(
         )
   {
   }

   ///
   /// getter of computing time of a subproblem
   /// @return subroblem computing time
   ///
   double
   getCompTime(
         )
   {
      return compTime;
   }

   ///
   /// geeter of the number of tasks solved in a subproblem
   /// @return the number of tasks
   ///
   int getNSolved(
         )
   {
      return nSolved;
   }

   ///
   /// getter of the termination state for solving the subproblem
   /// @return the termination state
   int getTerminationState(
         )
   {
      return terminationState;
   }

   ///
   /// stringfy ParaCalculationState
   /// @return string to show this object
   ///
   virtual std::string toString(
         ) = 0;

   ///
   /// stringfy ParaCalculationState (simple string version)
   /// @return simple string to show this object
   ///
   virtual std::string toSimpleString(
         ) = 0;

   ///
   /// send this object to destination
   ///
   virtual void send(
         ParaComm *comm,   ///< communicator used to send this object
         int destination,  ///< destination rank to send
         int tag           ///< tag to show this object
         ) = 0;

   ///
   /// send this object to destination
   ///
   virtual void receive(
         ParaComm *comm,   ///< communicator used to receive this object
         int source,       ///< source rank to receive this object
         int tag           ///< tag to show this object
         ) = 0;

};

}

#endif // __PARA_CALCULATION_STATE_H__
