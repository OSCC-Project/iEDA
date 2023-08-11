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

/**@file    scipParaSolution.h
 * @brief   ParaSolution extension for SCIP solver.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_SOLUTION_H__
#define __SCIP_PARA_SOLUTION_H__

#include <sstream>
#include "ug_bb/bbParaSolution.h"
#include "scip/scip.h"
#include "scipParaSolver.h"

namespace ParaSCIP
{

/** ScipParaSolution class */
class ScipParaSolution : public UG::BbParaSolution
{
protected:
   double objectiveFunctionValue;
   int nVars;                       /**< number of variables */
   int *indicesAmongSolvers;        /**< array of variable indices, mapping SCIPvarGetProbindex to SCIPvarGetIndex  */
   SCIP_Real *values;               /**< array of bounds which the branchings     */
public:
   /** constructor */
   ScipParaSolution(
         )
         : objectiveFunctionValue(0.0), nVars(0), indicesAmongSolvers(0), values(0)
   {
   }

   ScipParaSolution(
         ScipParaSolver *solver,
         SCIP_Real      objval,
         int            inNvars,
         SCIP_VAR **    vars,
         SCIP_Real *    vals
         )
        : indicesAmongSolvers(0), values(0)
   {
      objectiveFunctionValue = objval;
      nVars = inNvars;
      if( nVars > 0 )
      {
         indicesAmongSolvers = new int[nVars];
         values = new SCIP_Real[nVars];
         if( solver && solver->isOriginalIndeciesMap() )
         {
            for(int i = 0; i < nVars; i++ )
            {
               indicesAmongSolvers[i] = solver->getOriginalIndex(SCIPvarGetIndex(vars[i]));
               // indicesAmongSolvers[i] = SCIPvarGetIndex(vars[i]);
               values[i] = vals[i];
               // std::cout << i << ": " << indicesAmongSolvers[i] << std::endl;
               // std::cout << i << ": " << SCIPvarGetName(vars[indicesAmongSolvers[i]]) << ": " << values[i] << std::endl;
               // std::cout << i << ": " << SCIPvarGetName(vars[i]) << ": " << values[i] << std::endl;
            }
         }
         else
         {
            // std::cout << "*** index: name: value ***" << std::endl;
            for(int i = 0; i < nVars; i++ )
            {
               indicesAmongSolvers[i] = SCIPvarGetIndex(vars[i]);
               values[i] = vals[i];
               // std::cout << i << ": " << SCIPvarGetName(vars[indicesAmongSolvers[i]]) << ": " << values[i] << std::endl;
               // std::cout << i << ": " << SCIPvarGetName(vars[i]) << ": " << values[i] << std::endl;
            }
         }

      }
   }

   ScipParaSolution(
         double inObjectiveFunctionValue,
         int inNVars,                       /**< number of variables */
         int *inIndicesAmongSolvers,        /**< array of variable indices ( probindex )  */
         SCIP_Real *inValues                /**< array of bounds which the branchings     */
         ) : objectiveFunctionValue(inObjectiveFunctionValue), nVars(inNVars), indicesAmongSolvers(0), values(0)
   {
      if( nVars > 0 )
      {
         indicesAmongSolvers = new int[inNVars];
         values = new SCIP_Real[inNVars];
         for( int i = 0; i < inNVars; i++ )
         {
            indicesAmongSolvers[i] = inIndicesAmongSolvers[i];
            values[i] = inValues[i];
         }
      }
   }

   /** destructor */
   virtual ~ScipParaSolution(
         )
   {
      if( indicesAmongSolvers ) delete [] indicesAmongSolvers;
      if( values ) delete [] values;
   }

   /** get objective function value */
   double getObjectiveFunctionValue(){ return objectiveFunctionValue; }

   /** set objective function value */
   void setObjectiveFuntionValue(SCIP_Real val){ objectiveFunctionValue = val; }

   /** get number of variables */
   int getNVars(){ return nVars; }

   int indexAmongSolvers(int index){ return indicesAmongSolvers[index]; }

   SCIP_Real *getValues(){ return values; }
   void setValue(int i, SCIP_Real val){ assert(i < nVars && i >= 0 ); values[i] = val; }

#ifdef UG_WITH_ZLIB
   /** user should implement write method */
   void write(gzstream::ogzstream &out);

   /** user should implement read method */
   bool read(UG::ParaComm *comm, gzstream::igzstream &in);
#endif

   /** stringfy solution */
   const std::string toString(
         )
   {
      std::ostringstream s;
      s << "Obj = " << objectiveFunctionValue << std::endl;
      for( int i = 0; i < nVars; i++ )
      {
          s << "i = " << i;
          s << ", idx = " << indicesAmongSolvers[i];
          s << ", val = " << values[i] << std::endl;
      }
      return s.str();
   }

};

}

#endif // __SCIP_PARA_SOLUTION_H__

