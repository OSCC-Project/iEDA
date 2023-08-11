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

/**@file    scipParaInterruptMsgMonitor.cpp
 * @brief   Interrupt message monitor thread class.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "cassert"
#if defined(_MSC_VER)
#include "windows.h"
#else
#include "unistd.h"
#endif
#include "scipParaComm.h"
#include "scipParaSolver.h"
#include "scipParaInterruptMsgMonitor.h"

using namespace ParaSCIP;

ScipParaInterruptMsgMonitor::ScipParaInterruptMsgMonitor(
    UG::ParaComm *comm,
    ScipParaSolver *inScipParaSolver
    ) : terminateRequested(false), paraComm(comm), scipParaSolver(inScipParaSolver)
{
   rank = paraComm->getRank();
   // std::cout << typeid(*paraComm).name() << std::endl;
}

void 
ScipParaInterruptMsgMonitor::run()
{
   paraComm->setLocalRank(rank);
   // DEF_SCIP_PARA_COMM( scipParaComm, paraComm);
   for(;;)
   {
#ifdef _MSC_VER
     _sleep(static_cast<unsigned int>(1));
#else
      sleep(static_cast<unsigned int>(1));
#endif
      if( terminateRequested ) return;  // the flag update can be delayed
      DEF_SCIP_PARA_COMM( scipParaComm, paraComm);    // should be writen in here. I do not know the reason
      // std::cout << typeid(*paraComm).name() << std::endl;
      // std::cout << typeid(*scipParaComm).name() << std::endl;
      if( !scipParaSolver->getScip() ) return;   // solver is in destructor
      if( !scipParaComm ) return;
      scipParaComm->lockInterruptMsg();
      int source;
      int tag = UG::TagInterruptRequest;
      if( !scipParaSolver->getScip() )         // check gain. solver is in destructor
      {
         scipParaComm->unlockInterruptMsg();
         return;
      }
      /************************************
       * check if there are some messages *
       ************************************/
      if ( scipParaComm->iProbe(&source, &tag) )
      {
         assert( tag == UG::TagInterruptRequest );
         assert( !scipParaSolver->isInterrupting() );
         if( !scipParaSolver->getScip() )         // check gain. solver is in destructor
         {
            scipParaComm->unlockInterruptMsg();
            return;
         }
         scipParaSolver->processTagInterruptRequest(source, tag);
         // scipParaSolver->issueInterruptSolve();    // this is performed above function
      }
      scipParaComm->unlockInterruptMsg();
      if( terminateRequested ) break;
   }
}
