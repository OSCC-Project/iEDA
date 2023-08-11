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

/**@file    paraInitialStat.h
 * @brief   Base class for initial statistics collecting class
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_INITIAL_STAT_H__
#define __PARA_INITIAL_STAT_H__

#include <iostream>
#include "paraComm.h"

namespace UG
{

///
/// class for initial statistics collecting after racing
///
/// This class should NOT have any data member.
///
class ParaInitialStat {

   ///
   /// DO NOT HAVE DATA MEMBER!!
   ///

public:

   ///
   /// default constructor
   ///
   ParaInitialStat(
         )
   {
   }

   ///
   /// destructor
   ///
   virtual ~ParaInitialStat(
         )
   {
   }

   ///
   /// create clone of this object
   /// @return pointer to ParaInitialStat object
   ///
   virtual ParaInitialStat *clone(
         ParaComm *comm          ///< communicator used
         ) = 0;

   ///
   /// send function for ParaInitialStat object
   /// @return always 0 (for future extensions)
   ///
   virtual void send(
         ParaComm *comm,         ///< communicator used
         int dest                ///< destination rank
         ) = 0;

   ///
   /// receive function for ParaInitialStat object
   /// @return always 0 (for future extensions)
   ///
   virtual void receive(
         ParaComm *comm,         ///< communicator used
         int source              ///< source rank
         ) = 0;

   ///
   /// stringfy ParaInitialStat object (for debugging)
   /// @return string to show inside of this object
   ///
   virtual const std::string toString(
         ) = 0;
};

}

#endif    // __PARA_INITIAL_STAT_H__

