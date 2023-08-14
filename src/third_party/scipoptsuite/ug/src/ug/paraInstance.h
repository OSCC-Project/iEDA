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

/**@file    paraInstance.h
 * @brief   Base class for instance data.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_INSTANCE_H__
#define __PARA_INSTANCE_H__

#include "paraComm.h"

namespace UG
{

///
/// class for instance data
///
/// This class should NOT have any data member.
///
class ParaInstance
{

   ///
   /// DO NOT HAVE DATA MEMBER!!
   ///

public:

   ///
   /// default constructor
   ///
   ParaInstance(
         )
   {
   }

   ///
   /// destructor
   ///
   virtual ~ParaInstance(
        )
   {
   }

   ////
   /// get problem name
   /// @return problem name
   ///
   virtual const char* getProbName(
         ) = 0;

   ///
   /// broadcast function to all solvers
   /// @return always expected be 1
   ///
   virtual int bcast(
         ParaComm *comm,      ///< communicator used
         int rank,            ///< root rank
         int method           ///< method for the transferring
         ) = 0;

   ///
   /// Stringfy this object
   /// @return object as string
   ///
   virtual const std::string toString(
         ) = 0;

};

}

#endif  // __PARA_INSTANCE_H__
