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

/**@file    paraDiffSubproblem.h
 * @brief   Base class for a container which has difference between instance and subproblem.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __PARA_DIFF_SUBPROBLEM_H__
#define __PARA_DIFF_SUBPROBLEM_H__

#include <iostream>
#include <fstream>

#ifdef UG_WITH_ZLIB
#include "gzstream.h"
#endif

#include "paraComm.h"
#include "paraInstance.h"
// #include "paraMergeNodesStructs.h"

/** uncomment this define to activate debugging on given solution */
/** PARASCIP_DEBUG only valid for PARASCIP */
// #define UG_DEBUG_SOLUTION "timtab2-trans.sol"

namespace UG
{

class ParaInitiator;

///
/// Class for the difference between instance and subproblem
///
/// This class should NOT have any data member.
///
class ParaDiffSubproblem {

   ///
   /// DO NOT HAVE DATA MEMBER!!
   ///

public:

   ///
   ///  default constructor
   ///
   ParaDiffSubproblem(
         )
   {
   }

   ///
   ///  destractor¥
   ///
   virtual ~ParaDiffSubproblem(
         )
   {
   }

   ///
   /// create clone of this object
   /// @return pointer to ParaDiffSubproblem object
   ///
   virtual ParaDiffSubproblem *clone(
         ParaComm *comm           ///< communicator used
         ) = 0;

   ///
   /// broadcast function for ParaDiffSubproblem object
   /// @return always 0 (for future extensions)
   ///
   virtual int bcast(
         ParaComm *comm,          ///< communicator used
         int root                 ///< root rank of broadcast
         ) = 0;

   ///
   /// send function for ParaDiffSubproblem object
   /// @return always 0 (for future extensions)
   ///
   virtual int send(
         ParaComm *comm,          ///< communicator used
         int dest                 ///< destination rank
         ) = 0;

   ///
   /// receive function for ParaDiffSubproblem object
   /// @return always 0 (for future extensions)
   ///
   virtual int receive(
         ParaComm *comm,          ///< communicator used
         int source               ///< source rank
         ) = 0;

#ifdef UG_WITH_ZLIB

   ///
   /// function to write ParaDiffSubproblem object to checkpoint file
   ///
   virtual void write(
         gzstream::ogzstream &out  ///< gzstream for output
         ) = 0;

#endif

   ///
   /// stringfy ParaDiffSubproblem object ( for debugging )
   /// @return string to show inside of this object
   ///
   virtual const std::string toString(
         ) = 0;

};

}

#endif    // __PARA_DIFF_SUBPROBLEM_H__

