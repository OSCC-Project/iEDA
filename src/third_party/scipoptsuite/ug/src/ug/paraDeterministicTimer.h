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

/**@file    paraDeterministicTimer.h
 * @brief   Base class for deterministic timer
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_DETERMINISTIC_TIMER_H__
#define __PARA_DETERMINISTIC_TIMER_H__

#include "paraComm.h"

namespace UG
{

///
/// class for deterministic timer
///
/// void init() as a public function:\n
/// if you want to set original initial time, you can do the init() function.
/// arguments of the init() should be different depending on MIP solver parallelized
/// So, add init() function in derived classes and use it in constructor of
/// xxxParaSolver class
///
class ParaDeterministicTimer
{

public:

   ///
   /// default constructor of ParaDeterministicTimer
   ///
   ParaDeterministicTimer(
         )
   {
   }

   ///
   /// destructor of ParaDeterministicTimer
   ///
   virtual ~ParaDeterministicTimer(
         )
   {
   }

   ///
   /// some normalization for the deterministic time might be needed
   /// user can do it in this function
   ///
   virtual void normalize(
         ParaComm *comm       ///< communicator used
         )
   {
   }

   ///
   /// update function of the deterministic time.
   /// the deterministic time is a kind of counter
   ///
   virtual void update(
         double value         ///< added value to the deterministic time
         ) = 0;

   ///
   /// getter of the deterministic time
   ///
   virtual double getElapsedTime(
         ) = 0;

};

}

#endif  // __PARA_TIMER_H__
