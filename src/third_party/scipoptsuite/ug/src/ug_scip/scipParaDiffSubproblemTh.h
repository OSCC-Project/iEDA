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

/**@file    scipParaDiffSubproblemTh.h
 * @brief   ScipParaDiffSubproblem extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_DIFF_SUBPROBLEM_TH_H__
#define __SCIP_PARA_DIFF_SUBPROBLEM_TH_H__

#include "ug/paraDef.h"
#include "ug_bb/bbParaComm.h"
#include "scipParaDiffSubproblem.h"
#include "scip/scip.h"

namespace ParaSCIP
{

class ScipParaSolver;

/** The difference between instance and subproblem: this is base class */
class ScipParaDiffSubproblemTh : public ScipParaDiffSubproblem
{
public:
   /** default constructor */
   ScipParaDiffSubproblemTh()
   {
   }

   /** Constructor */
   ScipParaDiffSubproblemTh(
         SCIP *inScip,
         ScipParaSolver *inScipParaSolver,
         int inNNewBranchVars,
         SCIP_VAR **inNewBranchVars,
         SCIP_Real *inNewBranchBounds,
         SCIP_BOUNDTYPE *inNewBoundTypes,
         int nAddedConss,
         SCIP_CONS **addedConss
         ) : ScipParaDiffSubproblem(inScip, inScipParaSolver,
               inNNewBranchVars, inNewBranchVars, inNewBranchBounds, inNewBoundTypes, nAddedConss, addedConss)
   {
   }

   /** Constructor */
   ScipParaDiffSubproblemTh(
         ScipParaDiffSubproblem *paraDiffSubproblem
         ) : ScipParaDiffSubproblem(paraDiffSubproblem)
   {
   }


   /** destractor */
   ~ScipParaDiffSubproblemTh()
   {
   }

   /** create clone of this object */
   ScipParaDiffSubproblemTh *clone(
         UG::ParaComm *comm
         )
   {
      return(
            new ScipParaDiffSubproblemTh(this)
      );
   }

   int bcast(
         UG::ParaComm *comm,
         int root
         )
   {
      THROW_LOGICAL_ERROR1("bcast is issued in ScipParaDiffSubproblemTh");
   }

   int send(
         UG::ParaComm *comm,
         int dest
         )
   {
      THROW_LOGICAL_ERROR1("send is issued in ScipParaDiffSubproblemTh");
   }

   int receive(
         UG::ParaComm *comm,
         int source
         )
   {
      THROW_LOGICAL_ERROR1("receive is issued in ScipParaDiffSubproblemTh");
   }
};

}

#endif    // __SCIP_PARA_DIFF_SUBPROBLEM_TH_H__

