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

/**@file   ScipParaObjLimitUpdator.cpp
 * @brief  heuristic to update objlimit 
 * @author Yuji Shinano
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cassert>
#include <iostream>

#include "objscip/objscip.h"
#include "scipParaObjLimitUpdator.h"

using namespace ParaSCIP;
using namespace std;


/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
SCIP_DECL_HEURFREE(ScipParaObjLimitUpdator::scip_free)
{  /*lint --e{715}*/
   return SCIP_OKAY;
}


/** initialization method of primal heuristic (called after problem was transformed) */
SCIP_DECL_HEURINIT(ScipParaObjLimitUpdator::scip_init)
{  /*lint --e{715}*/
   return SCIP_OKAY;
}


/** deinitialization method of primal heuristic (called before transformed problem is freed) */
SCIP_DECL_HEUREXIT(ScipParaObjLimitUpdator::scip_exit)
{  /*lint --e{715}*/
   return SCIP_OKAY;
}


/** solving process initialization method of primal heuristic (called when branch and bound process is about to begin)
 *
 *  This method is called when the presolving was finished and the branch and bound process is about to begin.
 *  The primal heuristic may use this call to initialize its branch and bound specific data.
 *
 */
SCIP_DECL_HEURINITSOL(ScipParaObjLimitUpdator::scip_initsol)
{  /*lint --e{715}*/
   return SCIP_OKAY;
}


/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed)
 *
 *  This method is called before the branch and bound process is freed.
 *  The primal heuristic should use this call to clean up its branch and bound data.
 */
SCIP_DECL_HEUREXITSOL(ScipParaObjLimitUpdator::scip_exitsol)
{  /*lint --e{715}*/
   return SCIP_OKAY;
}


/** execution method of primal heuristic 2-Opt */
SCIP_DECL_HEUREXEC(ScipParaObjLimitUpdator::scip_exec)
{  /*lint --e{715}*/
   if( updated )
   {
      if( scipParaSolver->getGlobalBestIncumbentValue() < SCIPgetObjlimit(scip) )
      {
         SCIP_CALL_ABORT( SCIPsetObjlimit(scip, scipParaSolver->getGlobalBestIncumbentValue()) );
      }
      scipParaSolver->globalIncumbnetValueIsReflected();
      updated = false;
   }
   return SCIP_OKAY;
}


/** clone method which will be used to copy a objective plugin */
SCIP_DECL_HEURCLONE(scip::ObjCloneable* ScipParaObjLimitUpdator::clone) /*lint !e665*/
{
   return new ScipParaObjLimitUpdator(scip, scipParaSolver);
}
