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

/**@file   scipParaObjSelfSplitNodesel.h
 * @brief  node selector for self-split ramp-up
 * @author Yuji Shinano
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_PARA_OBJ_SELF_SPLIT_NODESEL_H__
#define __SCIP_PARA_OBJ_SELF_SPLIT_NODESEL_H__

#include <cstring>

#include "scipParaComm.h"
#include "scipParaSolver.h"
#include "scip/scipdefplugins.h"
#include "objscip/objnodesel.h"

#if SCIP_APIVERSION >= 101

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
// class ScipParaObjSelfSplitNodesel : public ObjCloneable
class ScipParaObjSelfSplitNodesel : public scip::ObjNodesel
{
   // SCIP_NODESEL* nodesel_estimate;

   int selfsplitrank;
   int selfsplitsize;
   int depthlimit;

   bool sampling;

   UG::ParaComm *paraComm;
   ScipParaSolver *scipParaSolver;


   void keepParaNode(SCIP *scip, int depth, SCIP_NODE* node);

   bool ifFeasibleInOriginalProblem(
         SCIP *scip,
         int nBranchVars,
         SCIP_VAR **branchVars,
         SCIP_Real *inBranchBounds);

public:
   /*lint --e{1540}*/

   /** default constructor */
   ScipParaObjSelfSplitNodesel(
         int rank,
         int size,
         int depth,
         UG::ParaComm *comm,
         ScipParaSolver *solver,
         SCIP *scip
      )
      : scip::ObjNodesel::ObjNodesel(scip, "ScipParaObjSelfSplitNodeSel", "Node selector for self-split ramp-up",
                                     INT_MAX/4, 0),
                                     // nodesel_estimate(0),
                                     selfsplitrank(rank),
                                     selfsplitsize(size),
                                     depthlimit(depth),
                                     sampling(true),
                                     paraComm(comm),
                                     scipParaSolver(solver)
   {
      // SCIP *scip = solver->getScip();
      // nodesel_estimate = SCIPfindNodesel(scip, "estimate");
      // assert(nodesel_estimate != NULL);
   }

   /** destructor */
   virtual ~ScipParaObjSelfSplitNodesel()
   {
      /* the macro SCIPfreeMemoryArray does not need the first argument: */
      /*lint --e{64}*/
//      SCIPfreeMemoryArray(scip_, &scip_name_);
//      SCIPfreeMemoryArray(scip_, &scip_desc_);
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

   bool inSampling(
         )
   {
      return sampling;
   }

};

} /* namespace scip */

#endif
#endif
