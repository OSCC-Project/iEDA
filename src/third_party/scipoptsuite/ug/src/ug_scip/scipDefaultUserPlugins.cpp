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

/**@file    scipDefaultUserPlugins.cpp
 * @brief   Set SCIP default user plugins.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "scipUserPlugins.h"
#include "scipParaInstance.h"
#include "scipParaSolver.h"
#include "scipParaInitiator.h"

using namespace UG;
using namespace ParaSCIP;

void
setUserPlugins(
   ParaInitiator *inInitiator
   )
{
   ScipParaInitiator *initiator = dynamic_cast<ScipParaInitiator *>(inInitiator);
   initiator->setUserPlugins(0);
}

void
setUserPlugins(
   ParaInstance *inInstance
   )
{
   ScipParaInstance *instance = dynamic_cast<ScipParaInstance *>(inInstance);
   instance->setUserPlugins(0);
}

void
setUserPlugins(
   ParaSolver *inSolver
   )
{
   ScipParaSolver *solver = dynamic_cast<ScipParaSolver *>(inSolver);
   solver->setUserPlugins(0);
}
