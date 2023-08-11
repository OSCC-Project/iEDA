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

/**@file   objnodesel.h
 * @brief  C++ wrapper for node selectors
 * @author Yuji Shinano
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_PARA_OBJNODESEL_H__
#define __SCIP_PARA_OBJNODESEL_H__

#include <cstring>
// #include "scipParaComm.h"
#include "scipParaSolver.h"
#include "objscip/objnodesel.h"
#include "scip/scipdefplugins.h"

namespace ParaSCIP
{

class ScipParaSolver;

/** @brief C++ wrapper for primal heuristics
 *
 *  This class defines the interface for node selectors implemented in C++. Note that there is a pure virtual
 *  function (this function has to be implemented). This function is: scip_comp().
 *
 *  - \ref NODESEL "Instructions for implementing a  node selector"
 *  - \ref NODESELECTORS "List of available node selectors"
 *  - \ref type_nodesel.h "Corresponding C interface"
 */
class ScipParaObjNodesel : public scip::ObjNodesel
{
   ScipParaSolver *scipParaSolver;

   int getNBoundChanges(SCIP_NODE* node);

public:
   /** default constructor */
   ScipParaObjNodesel(
      ScipParaSolver *solver
      )
      : scip::ObjNodesel::ObjNodesel(solver->getScip(), "ScipParaObjNodesel", "Node selector when a node is sent to LC",
        0, 0), scipParaSolver(solver)
   {
   }

   /** destructor */
   virtual ~ScipParaObjNodesel()
   {
   }

   /** node selection method of node selector
    *
    *  @see SCIP_DECL_NODESELSELECT(x) in @ref type_nodesel.h
    */
   virtual SCIP_DECL_NODESELSELECT(scip_select);

   /** node comparison method of node selector
    *
    *  @see SCIP_DECL_NODESELCOMP(x) in @ref type_nodesel.h
    */
   virtual SCIP_DECL_NODESELCOMP(scip_comp);
  
   void reset()
   {
   }

};

} /* namespace scip */

#endif
