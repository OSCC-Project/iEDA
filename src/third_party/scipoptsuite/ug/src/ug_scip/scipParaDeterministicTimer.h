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

/**@file    scipParaDeterministicTimer.h
 * @brief   ParaDeterministicTimer extension for SCIP.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_DETERMINISITC_TIMER_H__
#define __SCIP_PARA_DETERMINISITC_TIMER_H__

#include "ug/paraDeterministicTimer.h"

namespace ParaSCIP
{

class ScipParaDeterministicTimer : public UG::ParaDeterministicTimer
{
   double current; 
   int    normalizeFactor;
public:
   ScipParaDeterministicTimer() : current(0.0), normalizeFactor(1) {}
   virtual ~ScipParaDeterministicTimer() {}
   /**********************************************
    * if you want to set original initial time,  *
    * you can do it init()                       *
    **********************************************/
   void normalize(UG::ParaComm *comm){ normalizeFactor = comm->getSize() - 1; }
   void update(double value) { current += value; }
   double getElapsedTime() { return current/normalizeFactor; }
};

}

#endif  // __SCIP_PARA_DETERMINISTIC_TIMER_H__
