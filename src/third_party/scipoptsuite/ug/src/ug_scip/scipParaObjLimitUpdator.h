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

/**@file   ScipParaObjLimitUpdator.h
 * @brief  heuristic to update objlimit
 * @author Yuji Shinano
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_PARA_OBJ_LIMIT_UPDATOR_H__
#define __SCIP_PARA_OBJ_LIMIT_UPDATOR_H__

#include "objscip/objscip.h"
#include "scipParaSolver.h"

namespace ParaSCIP
{

/** C++ wrapper */
class ScipParaObjLimitUpdator : public scip::ObjHeur
{
   ScipParaSolver *scipParaSolver;
   bool            updated;
public:

   /** default constructor */
   ScipParaObjLimitUpdator(
      SCIP* scip,
      ScipParaSolver *solver
      )
      : ObjHeur(scip, "ObjLim", "Objlimit updator", 'u',1000000, 1, 0, -1, 
		      SCIP_HEURTIMING_DURINGLPLOOP | SCIP_HEURTIMING_AFTERNODE | 
		      SCIP_HEURTIMING_BEFORENODE | SCIP_HEURTIMING_AFTERLPLOOP | 
		      SCIP_HEURTIMING_AFTERPROPLOOP, FALSE),
      scipParaSolver(solver), updated(false)
   {
   }

   /** destructor */
   virtual ~ScipParaObjLimitUpdator()
   {
   } /*lint !e1540*/

   /** destructor of primal heuristic to free user data (called when SCIP is exiting) */
   virtual SCIP_DECL_HEURFREE(scip_free);

   /** initialization method of primal heuristic (called after problem was transformed) */
   virtual SCIP_DECL_HEURINIT(scip_init);

   /** deinitialization method of primal heuristic (called before transformed problem is freed) */
   virtual SCIP_DECL_HEUREXIT(scip_exit);

   /** solving process initialization method of primal heuristic (called when branch and bound process is about to begin)
    *
    *  This method is called when the presolving was finished and the branch and bound process is about to begin.
    *  The primal heuristic may use this call to initialize its branch and bound specific data.
    *
    */
   virtual SCIP_DECL_HEURINITSOL(scip_initsol);

   /** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed)
    *
    *  This method is called before the branch and bound process is freed.
    *  The primal heuristic should use this call to clean up its branch and bound data.
    */
   virtual SCIP_DECL_HEUREXITSOL(scip_exitsol);

   /** execution method of primal heuristic
    *
    *  Searches for feasible primal solutions. The method is called in the node processing loop.
    *
    *  possible return values for *result:
    *  - SCIP_FOUNDSOL   : at least one feasible primal solution was found
    *  - SCIP_DIDNOTFIND : the heuristic searched, but did not find a feasible solution
    *  - SCIP_DIDNOTRUN  : the heuristic was skipped
    *  - SCIP_DELAYED    : the heuristic was skipped, but should be called again as soon as possible, disregarding
    *                      its frequency
    */
   virtual SCIP_DECL_HEUREXEC(scip_exec);

   /** clone method which will be used to copy a objective plugin */
   virtual SCIP_DECL_HEURCLONE(ObjCloneable* clone); /*lint !e665*/

   /** returns whether the objective plugin is copyable */
   virtual SCIP_DECL_HEURISCLONEABLE(iscloneable)
   {
      return true;
   }

   void update()
   {
      updated = true;
   }

}; /*lint !e1712*/

}

#endif
