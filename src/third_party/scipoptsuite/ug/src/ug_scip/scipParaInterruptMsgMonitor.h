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

/**@file    scipParaInterruptMsgMonitor.h
 * @brief   Interrupt message monitor thread class.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_INTERRUPT_MSG_MONITOR_H__
#define __SCIP_PARA_INTERRUPT_MSG_MONITOR_H__

#include "ug/paraDef.h"
#include "ug/paraComm.h"
// #include "scipParaComm.h"
// #include "scipParaSolver.h"

namespace ParaSCIP
{

class ScipParaSolver;

class ScipParaInterruptMsgMonitor
{
   bool terminateRequested;
   int  rank;                              ///< rank of this Monitor
protected:
   UG::ParaComm    *paraComm;              ///< ParaCommunicator object 
   ScipParaSolver  *scipParaSolver;        ///< pointer to ScipParaSolver object
public:
   ScipParaInterruptMsgMonitor() : terminateRequested(false), paraComm(0), scipParaSolver(0)
   {
      THROW_LOGICAL_ERROR1("Default constructor of ParaTimeLimitMonitor is called");
   }
   ScipParaInterruptMsgMonitor(
       UG::ParaComm *comm,
       ScipParaSolver *inScipParaSolver
       );

   virtual ~ScipParaInterruptMsgMonitor(){}

   void run();

   void terminate()
   {
      terminateRequested = true;
      scipParaSolver = 0;
   }
};

}

#endif // __SCIP_PARA_INTERRUPT_MSG_MONITOR_H__

