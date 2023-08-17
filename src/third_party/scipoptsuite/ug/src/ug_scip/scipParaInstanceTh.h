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

/**@file    scipParaInstanceTh.h
 * @brief   ScipParaInstance extension for threads communication.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_INSTANCE_TH_H__
#define __SCIP_PARA_INSTANCE_TH_H__

#include <cassert>
#include <cstring>
#include "ug_bb/bbParaComm.h"
#include "ug_bb/bbParaInstance.h"
#include "scipUserPlugins.h"
#include "scip/scip.h"
#include "scip/cons_linear.h"
#include "scipDiffParamSet.h"
#include "scipParaSolution.h"

namespace ParaSCIP
{

class ScipParaInstance : public UG::BbParaInstance
{
protected:
   SCIP *scip;     // this pointer should point to the scip environment of LoadCoordinator, in case that
                   // LC does not have two scip environment
   virtual const char *getFileName() = 0;
   int           nVars;
   int           varIndexRange;
   int           *mapToOriginalIndecies;  /**< array of indices to map to original problem's probindices
                                               LC does not have this map in general, it is in the transformed prob. in scip */
   int           *mapToSolverLocalIndecies; /**< array of reverse indices mapToOriginalIndecies */
   // int           *mapToProbIndecies;      /**< in Solver, one more variable map is need in some cases */
   SCIP          *orgScip;                /**< if LC has the above MAP, variables need to be converted to this scip */
   bool          copyIncreasedVariables;   /**< indicate if SCIP copy increase the number of veariables or not */
public:
   /** constructor */
   ScipParaInstance(
         ) : scip(0), nVars(0), varIndexRange(0), mapToOriginalIndecies(0), mapToSolverLocalIndecies(0), // mapToProbIndecies(0),
         orgScip(0), copyIncreasedVariables(false)
   {
   }

   ScipParaInstance(
		 SCIP *inScip
         ) : scip(inScip), nVars(0), varIndexRange(0), mapToOriginalIndecies(0), mapToSolverLocalIndecies(0), // mapToProbIndecies(0),
               orgScip(0),  copyIncreasedVariables(false)
   {
   }
   /** destractor */
   virtual ~ScipParaInstance(
         )
   {
      if( mapToOriginalIndecies )
      {
         delete [] mapToOriginalIndecies;
      }
      if( mapToSolverLocalIndecies )
      {
         delete [] mapToSolverLocalIndecies;
      }
      /*
      if( mapToProbIndecies )
      {
         delete [] mapToProbIndecies;
      }
      */
      if( orgScip )    // If this instance has orgScip pointer, scip has to be freed
      {
         SCIPfree(&scip);
      }
   }

   /** convert an internal value to external value */
   double convertToExternalValue(double internalValue)
   {
      if( orgScip )
      {
         return SCIPretransformObj(orgScip, SCIPretransformObj(scip, internalValue) );
      }
      else
      {
         return SCIPretransformObj(scip, internalValue);
      }
   }

   /** convert an external value to internal value */
   double convertToInternalValue(double externalValue)
   {
      if( orgScip )
      {
         return SCIPtransformObj(scip, SCIPtransformObj(orgScip, externalValue));
      }
      else
      {
         return SCIPtransformObj(scip, externalValue);
      }
   }

   /** get solution values for the original problem */
   void getSolValuesForOriginalProblem(ScipParaSolution *sol, SCIP_Real *vals)
   {
      assert( orgScip );
      assert( mapToOriginalIndecies );
      SCIP_SOL*  newsol;
      SCIP_CALL_ABORT( SCIPcreateSol(scip, &newsol, 0) );
      SCIP_VAR **vars = SCIPgetVars(scip);
      assert( SCIPgetNVars(scip) == sol->getNVars() );
      int j = 0;
      for( int i = 0; i < varIndexRange; i++ )
      {
         if( mapToOriginalIndecies[i] >= 0 )
         {
            // assert( sol->indexAmongSolvers(j) == mapToOriginalIndecies[i]);
            if( sol->indexAmongSolvers(j) == mapToOriginalIndecies[i] )
            {
               SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[sol->indexAmongSolvers(j)], sol->getValues()[j]) );
               j++;
            }
         }
         //  SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[sol->indexAmongSolvers(i)], sol->getValues()[i]) );
         // SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[mapToOriginalIndecies[sol->indexAmongSolvers(i)]], sol->getValues()[i]) );
         /* this is not necessary
         if( mapToOriginalIndecies )
         {
            assert( mapToOriginalIndecies[sol->indexAmongSolvers(i)] < sol->getNVars() );
         }
         else
         {
            assert( sol->indexAmongSolvers(i) < sol->getNVars() );
            SCIP_CALL_ABORT( SCIPsetSolVal(scip, newsol, vars[i], sol->getValues()[i]) );
         }
         */
      }
      SCIP_CALL_ABORT(SCIPgetSolVals(scip, newsol, SCIPgetNVars(scip), SCIPgetVars(scip), vals) );
      SCIP_Bool success;
      // SCIP_CALL_ABORT( SCIPaddSolFree(scip, &newsol, &success) );
#if (SCIP_VERSION < 321 || ( SCIP_VERSION == 321 && SCIP_SUBVERSION < 2) )
      SCIP_CALL_ABORT( SCIPtrySolFree(scip, &newsol, FALSE, TRUE, TRUE, TRUE, &success) );
#else
      SCIP_CALL_ABORT( SCIPtrySolFree(scip, &newsol, FALSE, TRUE, TRUE, TRUE, TRUE, &success) );
#endif
      // std::cout << "** 1 ** success = " << success << std::endl;
   }

   /** create presolved problem instance that is solved by ParaSCIP form scip environment in this object */
   void copyScipEnvironment(
         SCIP **scip
         );

   SCIP *getScip(
         )
   {
      return scip;
   }

   /** create presolved problem instance that is solved by ParaSCIP */
   void createProblem(
        SCIP *scip,
        int  method,               // transferring method
        bool noPreprocessingInLC,  // LC preprocesing settings
        bool usetRootNodeCuts,
        ScipDiffParamSet  *scipDiffParamSetRoot,
        ScipDiffParamSet  *scipDiffParamSet,
        char *settingsNameLC,       // LC preprocesing settings
        char *isolname
        );

   /** stringfy ParaCalculationState */
   const std::string toString(
        )
   {
      return std::string("Should be written from scip environment.");
   }

   // int getOrigProbIndex(int index){ return mapToOriginalIndecies[index]; }
   const char *getProbName()
   {
      if( orgScip )
      {
         return SCIPgetProbName(orgScip);
      }
      else
      {
         return SCIPgetProbName(scip);
      }
   }

   int getNVars(){ return nVars; }
   int getVarIndexRange(){ return varIndexRange; }
   SCIP_Real getVarLb(int i)
   {
      SCIP_VAR **vars = SCIPgetVars(scip);
      return SCIPvarGetLbGlobal(vars[i]);
   }
   SCIP_Real getVarUb(int i)
   {
      SCIP_VAR **vars = SCIPgetVars(scip);
      return SCIPvarGetUbGlobal(vars[i]);
   }
   int getVarType(int i)
   {
      SCIP_VAR **vars = SCIPgetVars(scip);
      return SCIPvarGetType(vars[i]);
   }
   SCIP_Real getObjCoef(int i)
   {
      SCIP_VAR **vars = SCIPgetVars(scip);
      return SCIPvarGetObj(vars[i]);
   }

   const char *getVarName(int i)
   {
      SCIP_VAR **vars = SCIPgetVars(scip);
      return SCIPvarGetName(vars[i]);
   }

   int getNConss()
   {
      return SCIPgetNConss(scip);
   }

   SCIP_Real getLhsLinear(int i)
   {
      SCIP_CONS **conss = SCIPgetConss(scip);
      SCIP_CONSHDLR* conshdlr = SCIPconsGetHdlr(conss[i]);
      assert( conshdlr != NULL );
      if( strcmp(SCIPconshdlrGetName(conshdlr),"linear") != 0 )
      {
         THROW_LOGICAL_ERROR2("invalid constraint type exists; consname = ", SCIPconshdlrGetName(conshdlr));
      }
      return SCIPgetLhsLinear(scip, conss[i]);
   }

   SCIP_Real getRhsLinear(int i)
   {
      SCIP_CONS **conss = SCIPgetConss(scip);
      SCIP_CONSHDLR* conshdlr = SCIPconsGetHdlr(conss[i]);
      assert( conshdlr != NULL );
      if( strcmp(SCIPconshdlrGetName(conshdlr),"linear") != 0 )
      {
         THROW_LOGICAL_ERROR2("invalid constraint type exists; consname = ", SCIPconshdlrGetName(conshdlr));
      }
      return SCIPgetRhsLinear(scip, conss[i]);
   }

   int getNVarsLinear(int i)
   {
      SCIP_CONS **conss = SCIPgetConss(scip);
      SCIP_CONSHDLR* conshdlr = SCIPconsGetHdlr(conss[i]);
      assert( conshdlr != NULL );
      if( strcmp(SCIPconshdlrGetName(conshdlr),"linear") != 0 )
      {
         THROW_LOGICAL_ERROR2("invalid constraint type exists; consname = ", SCIPconshdlrGetName(conshdlr));
      }
      return SCIPgetNVarsLinear(scip, conss[i]);
   }

   int getIdxLinearCoefVar(int i, int j) // i-th constraint, j-th variable
   {
      SCIP_CONS **conss = SCIPgetConss(scip);
      SCIP_VAR **vars = SCIPgetVarsLinear(scip, conss[i]);
      return SCIPvarGetProbindex(vars[j]);
   }

   SCIP_Real *getLinearCoefs(int i) // i-th constraint, j-th variable
   {
      SCIP_CONS **conss = SCIPgetConss(scip);
      return SCIPgetValsLinear(scip, conss[i]);
   }

   const char *getConsName(int i)
   {
      SCIP_CONS **conss = SCIPgetConss(scip);
      return SCIPconsGetName(conss[i]);
   }

   /** set user plugins */
   void setUserPlugins(ScipUserPlugins *inUi) { /** maybe called, no need to do anything */ }

   /** include user plugins */
   void includeUserPlugins(SCIP *inScip){/** should not be called **/}

   virtual void setFileName( const char *file ) = 0;

  bool isOriginalIndeciesMap() { return (mapToOriginalIndecies != 0); }

  bool isSolverLocalIndeciesMap() { return (mapToSolverLocalIndecies != 0); }

  int getOrigProbIndex(int index)
  {
     assert(mapToOriginalIndecies);
     return mapToOriginalIndecies[index];
  }

  int *extractOrigProbIndexMap()
  {
     assert(mapToOriginalIndecies);
     int *extract = mapToOriginalIndecies;
     mapToOriginalIndecies = 0;
     return extract;
  }

  int *extractSolverLocalIndexMap()
  {
     assert(mapToSolverLocalIndecies);
     int *extract = mapToSolverLocalIndecies;
     mapToSolverLocalIndecies = 0;
     return extract;
  }

  /*
  int *extractProbIndexMap()
  {
     assert(mapToProbIndecies);
     int *extract = mapToProbIndecies;
     mapToProbIndecies = 0;
     return extract;
  }
  */

  SCIP *getParaInstanceScip()
  {
     assert( orgScip );
     return scip;
  }

  int getOrigObjSense()
  {
     return SCIPgetObjsense(scip);;
  }

  bool isCopyIncreasedVariables()
  {
     return copyIncreasedVariables;
  }

  void copyIncrasedVariables()
  {
     copyIncreasedVariables = true;
  }

};

}

namespace ParaSCIP
{

/** ScipInstanceTh */
class ScipParaInstanceTh : public ScipParaInstance
{
   const char *getFileName()
   {
      std::cout << "This function should name be used for FiberSCIP!" << std::endl;
      abort();
   }
public:
   /** constructor */
   ScipParaInstanceTh(
         )
   {
   }

   /** constructor : only called from ScipInitiator */
   ScipParaInstanceTh(
         SCIP *inScip,
         int method
         );

   /** destractor */
   ~ScipParaInstanceTh(
         )
   {
   }

   /** broadcasts instance to all solvers */
   int bcast(UG::ParaComm *comm, int rank, int method);

   void setFileName( const char *file )
   {
      std::cout << "This function should name be used for FiberSCIP!" << std::endl;
      abort();
   }

};

typedef ScipParaInstanceTh *ScipParaInstanceThPtr;

}

#endif  // __SCIP_PARA_INSTANCE_TH_H__

