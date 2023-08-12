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

/**@file   scipParaObjProp.h
 * @brief  C++ wrapper for propagators
 * @author Yuji Shinano
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef __SCIP_PARA_OBJPROP_H__
#define __SCIP_PARA_OBJPROP_H__

#include <cstring>
#include <list>

#include "scipParaSolver.h"
#include "objscip/objprop.h"
#include "ug/paraComm.h"
#ifdef UG_DEBUG_SOLUTION
#ifndef WITH_DEBUG_SOLUTION
#define WITH_DEBUG_SOLUTION
#endif
#include "scip/debug.h"
#endif

namespace ParaSCIP
{

struct BoundChange{
   SCIP_BOUNDTYPE boundType;
   int            index;
   SCIP_Real      bound;
};

/** @brief C++ wrapper for propagators
 *
 *  This class defines the interface for propagators implemented in C++. Note that there is a pure virtual
 *  function (this function has to be implemented). This function is: scip_exec().
 *
 *  - \ref PROP "Instructions for implementing a propagator"
 *  - \ref PROPAGATORS "List of available propagators"
 *  - \ref type_prop.h "Corresponding C interface"
 */
class ScipParaObjProp : public scip::ObjProp
{
   // UG::ParaComm *paraComm;
   std::list<BoundChange *> boundChanges;
   ScipParaSolver *solver;
   int             ntotaltightened;
   int             ntotaltightenedint;
public:
   /** default constructor */
   ScipParaObjProp(
         UG::ParaComm   *comm,
         ScipParaSolver *inSolver
      ) : scip::ObjProp::ObjProp(
            inSolver->getScip(),
            "ScipParaObjProp",
            "Propagator for updating variable bounds",
            (INT_MAX/4),
            -1,
            0,
            SCIP_PROPTIMING_ALWAYS,
            (INT_MAX/4) ,
            -1,
            SCIP_PRESOLTIMING_FAST
            )// , paraComm(comm)
             , solver(inSolver), ntotaltightened(0), ntotaltightenedint(0)
   {
   }

   /** destructor */
   virtual ~ScipParaObjProp()
   {
      std::list<BoundChange *>::iterator it = boundChanges.begin();
      while( it != boundChanges.end() )
      {
         BoundChange *bc = boundChanges.front();
         it = boundChanges.erase(it);
         delete bc;
      }
   }

   /** execution method of propagator
    *
    *  @see SCIP_DECL_PROPEXEC(x) in @ref type_prop.h
    */
   SCIP_RETCODE applyBoundChanges(SCIP *scip, int& ntightened, int& ntightenedint, SCIP_RESULT *result )
   {
      // std::cout << "#### exec propagator ##### Rank = " << paraComm->getRank() << std::endl;
      ntightened = 0;
      ntightenedint = 0;

      *result = SCIP_DIDNOTFIND;

      std::list<BoundChange *>::iterator it = boundChanges.begin();
      while( it != boundChanges.end() )
      {
         BoundChange *bc = boundChanges.front();
         SCIP_Var **orgVars = SCIPgetOrigVars(scip);
         SCIP_Var *var = SCIPvarGetTransVar(orgVars[bc->index]);
         if ( *result != SCIP_CUTOFF && var && SCIPvarGetStatus(var) != SCIP_VARSTATUS_FIXED && SCIPvarGetStatus(var) != SCIP_VARSTATUS_MULTAGGR && SCIPvarGetStatus(var) != SCIP_VARSTATUS_AGGREGATED )  // Can recive bounds during presolving
         {
            if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_NEGATED )
            {
               SCIP_Var *varNeg = 0;
               SCIP_CALL_ABORT ( SCIPgetNegatedVar(scip, var, &varNeg) );
               if( SCIPvarIsActive(varNeg) )
               {
#ifdef UG_DEBUG_SOLUTION
                  SCIP_Real solvalue = 0.0;
                  SCIP_CALL(SCIPdebugGetSolVal(scip,orgVars[bc->index], &solvalue));  // this can happen, when there are several optimal solutions with DDP
                  std::cout << "Receiver side SolValue: " << SCIPvarGetName(orgVars[bc->index]) << " = " << solvalue << std::endl;
                  if( bc->boundType == SCIP_BOUNDTYPE_LOWER )
                  {
                      std::cout << "Receiver side (SCIP_BOUNDTYPE_LOWER): " << SCIPvarGetName(orgVars[bc->index]) << " = " << bc->bound << std::endl;
                      SCIP_CALL_ABORT( SCIPdebugCheckLbGlobal(scip,orgVars[bc->index],bc->bound) );
                  }
                  else
                  {
                      std::cout << "Receiver side (SCIP_BOUNDTYPE_UPPER): " << SCIPvarGetName(orgVars[bc->index]) << " = " << bc->bound << std::endl;
                      SCIP_CALL_ABORT( SCIPdebugCheckUbGlobal(scip,orgVars[bc->index],bc->bound) );
                  }
#endif
                  SCIP_CALL( tryToTightenBound(scip, bc->boundType, orgVars[bc->index], bc->bound, result, ntightened, ntightenedint ) );
               }
            }
            else
            {
#ifdef UG_DEBUG_SOLUTION
               SCIP_Real solvalue = 0.0;
               SCIP_CALL(SCIPdebugGetSolVal(scip,orgVars[bc->index], &solvalue));   // this can happen, when there are several optimal solutions with DDP
               std::cout << "Receiver side SolValue: " << SCIPvarGetName(orgVars[bc->index]) << " = " << solvalue << std::endl;
               if( bc->boundType == SCIP_BOUNDTYPE_LOWER )
               {
                   std::cout << "Receiver side (SCIP_BOUNDTYPE_LOWER): " << SCIPvarGetName(orgVars[bc->index]) << " = " << bc->bound << std::endl;
                   SCIP_CALL_ABORT( SCIPdebugCheckLbGlobal(scip,orgVars[bc->index],bc->bound) );
               }
               else
               {
                   std::cout << "Receiver side (SCIP_BOUNDTYPE_UPPER): " << SCIPvarGetName(orgVars[bc->index]) << " = " << bc->bound << std::endl;
                   SCIP_CALL_ABORT( SCIPdebugCheckUbGlobal(scip,orgVars[bc->index],bc->bound) );
               }
#endif
               SCIP_CALL( tryToTightenBound(scip, bc->boundType, orgVars[bc->index], bc->bound, result, ntightened, ntightenedint ) );
            }
         }
         it = boundChanges.erase(it);
         delete bc;
      }

      return SCIP_OKAY;
   }

   /** presolving method of propagator
       *
       *  @see SCIP_DECL_PROPPRESOL(x) in @ref type_prop.h
       */
   virtual SCIP_DECL_PROPPRESOL(scip_presol)
   {
      int             ntightened;
      int             ntightenedint;
      *result = SCIP_DIDNOTRUN;

      if( boundChanges.empty() || SCIPinProbing(scip) )
         return SCIP_OKAY;

      // if pending incumbent value exists, set it before apply bounds
      if( solver->getPendingIncumbentValue() < SCIPgetObjlimit(scip) )
      {
         SCIPsetObjlimit(scip, solver->getPendingIncumbentValue());
      }

      applyBoundChanges(scip, ntightened, ntightenedint, result);

      if( ntightened > 0 )
      {
         *nchgbds += ntightened;
         ntotaltightened += ntightened;
         ntotaltightenedint += ntightenedint;
         if( *result != SCIP_CUTOFF )
            *result = SCIP_SUCCESS;
         // std::cout << "$$$$$ tightened " << ntightened << " var bounds in Rank " << paraComm->getRank() << " of which " << ntightenedint << " where integral vars." << std::endl;
      }
      SCIPpropSetFreq(prop, -1);
      return SCIP_OKAY;
   }

   /** execution method of propagator
    *
    *  @see SCIP_DECL_PROPEXEC(x) in @ref type_prop.h
    */
   virtual SCIP_DECL_PROPEXEC(scip_exec)
   {
      // std::cout << "#### exec propagator ##### Rank = " << paraComm->getRank() << std::endl;
     int             ntightened;
     int             ntightenedint;
	  *result = SCIP_DIDNOTRUN;
	  
      if( SCIPinProbing(scip) || SCIPinRepropagation(scip) )
         return SCIP_OKAY;

      // if pending incumbent value exists, set it before apply bounds
      if( solver->getPendingIncumbentValue() < SCIPgetObjlimit(scip) )
      {
         SCIPsetObjlimit(scip, solver->getPendingIncumbentValue());
      }

      applyBoundChanges(scip, ntightened, ntightenedint, result);
      
      if( ntightened > 0 )
      {
         ntotaltightened += ntightened;
         ntotaltightenedint += ntightenedint;
         if( *result != SCIP_CUTOFF )
            *result = SCIP_REDUCEDDOM;
         // std::cout << "$$$$$ tightened " << ntightened << " var bounds in Rank " << paraComm->getRank() << " of which " << ntightenedint << " where integral vars." << std::endl;
      }
      SCIPpropSetFreq(prop, -1);
      return SCIP_OKAY;
   }

   SCIP_RETCODE tryToTightenBound(SCIP *scip, SCIP_BOUNDTYPE boundType, SCIP_VAR *var, SCIP_Real bound, SCIP_Result *result, int& ntightened, int& ntightenedint )
   {
      SCIP_Bool infeas, tightened;
      if( boundType == SCIP_BOUNDTYPE_LOWER )
      {
         // std::cout << "### idx = " << bc->index << " Local lb = " << SCIPvarGetLbGlobal(orgVars[bc->index]) << ", bound = " << bc->bound << " #### Rank = " << paraComm->getRank() << std::endl;
         SCIP_CALL( SCIPtightenVarLbGlobal(scip, var, bound, FALSE, &infeas, &tightened) );
      }
      else
      {
         assert(boundType == SCIP_BOUNDTYPE_UPPER);
         // std::cout << "### idx = " << bc->index << " Local ub = " << SCIPvarGetUbGlobal(orgVars[bc->index]) << ", bound = " << bc->bound << " #### Rank = " << paraComm->getRank() << std::endl;
         SCIP_CALL( SCIPtightenVarUbGlobal(scip, var, bound, FALSE, &infeas, &tightened) );
      }
      // std::cout << "#### call SCIPtightenVarLbGlobal or SCIPtightenVarUbGlobal ##### Rank = " << paraComm->getRank()
      //     << ", infeas = " << infeas << ", tightened = " << tightened << std::endl;
      if( infeas )
      {
         ++ntightened;
         ++ntightenedint;
         *result = SCIP_CUTOFF;
         return SCIP_OKAY;
      }
      if( tightened )
      {
         ++ntightened;
         if( SCIPvarGetType(var) == SCIP_VARTYPE_BINARY
               || SCIPvarGetType(var) == SCIP_VARTYPE_INTEGER )
            ++ntightenedint;
      }
      return SCIP_OKAY;
   }

   void addBoundChange(SCIP *scip, SCIP_BOUNDTYPE boundType, int index, SCIP_Real bound)
   {
      BoundChange *bc = new BoundChange;
      bc->boundType = boundType;
      bc->index = index;
      bc->bound = bound;
      boundChanges.push_back(bc);
      SCIPsetIntParam(scip, "propagating/ScipParaObjProp/freq", 1);
   }

   int getNtightened(){ return ntotaltightened; }
   int getNtightenedInt(){ return ntotaltightenedint; }
};

} /* namespace ParaSCIP */

#endif // __SCIP_PARA_OBJPROP_H__
