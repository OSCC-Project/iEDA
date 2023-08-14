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

/**@file    scipParaDiffSubproblem.h
 * @brief   ParaInitialStat extension for SCIP solver.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_DIFF_SUBPROBLEM_H__
#define __SCIP_PARA_DIFF_SUBPROBLEM_H__

#include <cstring>
#include <iostream>
#include <fstream>
#include "ug/paraComm.h"
#ifdef UG_WITH_ZLIB
#include "ug/gzstream.h"
#endif
#include "ug_bb/bbParaInstance.h"
#include "ug_bb/bbParaDiffSubproblem.h"
#include "scip/scip.h"
#include "scip/history.h"
#include "scip/cons_bounddisjunction.h"

namespace ParaSCIP
{

class ScipParaSolver;

typedef struct BranchConsLinearInfo_t
{
   /********************************
    * for branching constraint     *
    * *****************************/
   char *         consName;             /**< name of this constraint */
   SCIP_Real      linearLhs;            /**< array of lhs */
   SCIP_Real      linearRhs;            /**< array of rhs */
   int            nLinearCoefs;         /**< array of number of coefficient values for linear constrains */
   SCIP_Real      *linearCoefs;         /**< array of non-zero coefficient values of linear constrains */
   int            *idxLinearCoefsVars;  /**< array of indices of no-zero coefficient values of linear constrains */
} BranchConsLinearInfo;
typedef BranchConsLinearInfo * BranchConsLinearInfoPtr;

typedef struct BranchConsSetppcInfo_t
{
   /********************************
    * for branching constraint     *
    * *****************************/
   char *         consName;            /**< name of this constraint */
   int            nSetppcVars;         /**< array of numbers of indices of variables for setppc constrains */
   int            setppcType;          /**< setppc Type */
   int            *idxSetppcVars;      /**< array of indices of variables for setppc constrains */
} BranchConsSetppcInfo;
typedef BranchConsSetppcInfo * BranchConsSetppcInfoPtr;

class ScipParaDiffSubproblemBranchLinearCons
{
public:
   int            nLinearConss;          /**< number of linear constrains */
   SCIP_Real      *linearLhss;           /**< array of lhs */
   SCIP_Real      *linearRhss;           /**< array of rhs */
   int            *nLinearCoefs;         /**< array of number of coefficient values for linear constrains */
   SCIP_Real      **linearCoefs;         /**< array of non-zero coefficient values of linear constrains */
   int            **idxLinearCoefsVars;  /**< array of indices of no-zero coefficient values of linear constrains */
   int            lConsNames;            /**< length of cons names space: each name is separated by \0 */
   char           *consNames;            /**< constraint names */

   /** default constructor */
   ScipParaDiffSubproblemBranchLinearCons(
         ) :nLinearConss(0),
            linearLhss(0),
            linearRhss(0),
            nLinearCoefs(0),
            linearCoefs(0),
            idxLinearCoefsVars(0),
            lConsNames(0),
            consNames(0)
   {
   }
   /** destractor */
   ~ScipParaDiffSubproblemBranchLinearCons(
         )
   {
      if( linearLhss ) delete[] linearLhss;
      if( linearRhss ) delete[] linearRhss;
      for( int c = 0; c < nLinearConss; c++ )
      {
         if( nLinearCoefs[c] > 0 )
         {
            if( linearCoefs[c] ) delete[] linearCoefs[c];
            if( idxLinearCoefsVars[c] ) delete[] idxLinearCoefsVars[c];
         }
      }
      if( nLinearCoefs ) delete[] nLinearCoefs;
      if( linearCoefs ) delete[] linearCoefs;
      if( idxLinearCoefsVars ) delete[] idxLinearCoefsVars;
      if( consNames ) delete[] consNames;
   }
};

class ScipParaDiffSubproblemBranchSetppcCons
{
public:
   int           nSetppcConss;             /**< number of setppc constrains */
   int           *nSetppcVars;             /**< array of numbers of indices of variables for setppc constrains */
   int           *setppcTypes;             /**< setppc Types */
   int           **idxSetppcVars;          /**< array of indices of variables for setppc constrains */
   int           lConsNames;               /**< length of cons names space: each name is separated by \0 */
   char          *consNames;               /**< constraint names */
   /** default constructor */
   ScipParaDiffSubproblemBranchSetppcCons(
         ) :nSetppcConss(0),
            nSetppcVars(0),
            setppcTypes(0),
            idxSetppcVars(0),
            lConsNames(0),
            consNames(0)
   {
   }
   /** destractor */
   ~ScipParaDiffSubproblemBranchSetppcCons(
         )
   {
      if( nSetppcVars ) delete[] nSetppcVars;
      if( setppcTypes ) delete[] setppcTypes;
      for( int c = 0; c < nSetppcConss; c++ )
      {
         if( idxSetppcVars[c] ) delete[] idxSetppcVars[c];
      }
      if (idxSetppcVars ) delete [] idxSetppcVars;
      if( consNames ) delete[] consNames;
   }
};

class ScipParaDiffSubproblemLinearCons
{
public:
   int            nLinearConss;          /**< number of linear constrains */
   SCIP_Real      *linearLhss;           /**< array of lhs */
   SCIP_Real      *linearRhss;           /**< array of rhs */
   int            *nLinearCoefs;         /**< array of number of coefficient values for linear constrains */
   SCIP_Real      **linearCoefs;         /**< array of non-zero coefficient values of linear constrains */
   int            **idxLinearCoefsVars;  /**< array of indices of no-zero coefficient values of linear constrains */

   /** default constructor */
   ScipParaDiffSubproblemLinearCons(
         ) :nLinearConss(0),
            linearLhss(0),
            linearRhss(0),
            nLinearCoefs(0),
            linearCoefs(0),
            idxLinearCoefsVars(0)
   {
   }
   /** destractor */
   ~ScipParaDiffSubproblemLinearCons(
         )
   {
      if( linearLhss ) delete[] linearLhss;
      if( linearRhss ) delete[] linearRhss;
      for( int c = 0; c < nLinearConss; c++ )
      {
         if( nLinearCoefs[c] > 0 )
         {
            if( linearCoefs[c] ) delete[] linearCoefs[c];
            if( idxLinearCoefsVars[c] ) delete[] idxLinearCoefsVars[c];
         }
      }
      if( nLinearCoefs ) delete[] nLinearCoefs;
      if( linearCoefs ) delete[] linearCoefs;
      if( idxLinearCoefsVars ) delete[] idxLinearCoefsVars;
   }
};

class ScipParaDiffSubproblemBoundDisjunctions
{
public:
   int           nBoundDisjunctions;    /**< number of bound disjunction constraints */
   int           nTotalVarsBoundDisjunctions; /**< total number of vars in bound disjunction constraints */
   int           *nVarsBoundDisjunction;/**< number of variables in bound disjunction constraint */
   SCIP_Bool     *flagBoundDisjunctionInitial;   /**< should the LP relaxation of constraint be in the initial LP? Usually set to TRUE. Set to FALSE for 'lazy constraints'. */
   SCIP_Bool     *flagBoundDisjunctionSeparate;  /**< should the constraint be separated during LP processing? Usually set to TRUE. */
   SCIP_Bool     *flagBoundDisjunctionEnforce;   /**< should the constraint be enforced during node processing? TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool     *flagBoundDisjunctionCheck;     /**< should the constraint be checked for feasibility? TRUE for model constraints, FALSE for additional, redundant constraints. */
   SCIP_Bool     *flagBoundDisjunctionPropagate; /**< should the constraint be propagated during node processing? Usually set to TRUE. */
   SCIP_Bool     *flagBoundDisjunctionLocal;     /**< is constraint only valid locally? Usually set to FALSE. Has to be set to TRUE, e.g., for branching constraints. */
   SCIP_Bool     *flagBoundDisjunctionModifiable;/**< is constraint modifiable (subject to column generation)? Usually set to FALSE. In column generation applications, set to TRUE if pricing adds coefficients to this constraint. */
   SCIP_Bool     *flagBoundDisjunctionDynamic;   /**< is constraint subject to aging? Usually set to FALSE. Set to TRUE for own cuts which are separated as constraints.*/
   SCIP_Bool     *flagBoundDisjunctionRemovable; /**< should the relaxation be removed from the LP due to aging or cleanup? Usually set to FALSE. Set to TRUE for 'lazy constraints' and 'user cuts'.*/
   SCIP_Bool     *flagBoundDisjunctionStickingatnode; /**< should the constraint always be kept at the node where it was added, even if it may be moved to a more global node? Usually set to FALSE. Set to TRUE to for constraints that represent node data. */
   int           **idxBoundDisjunctionVars;    /**< index of bound disjunction vars */
   SCIP_BOUNDTYPE  **boundTypesBoundDisjunction; /**< array of bound types in bound disjunction constraint */
   SCIP_Real     **boundsBoundDisjunction;    /**< array of bounds in bound disjunction constraint */

   /** default constructor */
   ScipParaDiffSubproblemBoundDisjunctions(
         ) : nBoundDisjunctions(0),
               nTotalVarsBoundDisjunctions(0),
               nVarsBoundDisjunction(0),
               flagBoundDisjunctionInitial(0),
               flagBoundDisjunctionSeparate(0),
               flagBoundDisjunctionEnforce(0),
               flagBoundDisjunctionCheck(0),
               flagBoundDisjunctionPropagate(0),
               flagBoundDisjunctionLocal(0),
               flagBoundDisjunctionModifiable(0),
               flagBoundDisjunctionDynamic(0),
               flagBoundDisjunctionRemovable(0),
               flagBoundDisjunctionStickingatnode(0),
               idxBoundDisjunctionVars(0),
               boundTypesBoundDisjunction(0),
               boundsBoundDisjunction(0)
   {
   }
   /** destractor */
   ~ScipParaDiffSubproblemBoundDisjunctions(
         )
   {
      for( int i = 0; i < nBoundDisjunctions; i++ )
      {
         if( nVarsBoundDisjunction[i] > 0 )
         {
            if( idxBoundDisjunctionVars[i] ) delete[] idxBoundDisjunctionVars[i];
            if( boundTypesBoundDisjunction[i] ) delete[]  boundTypesBoundDisjunction[i];
            if( boundsBoundDisjunction[i] ) delete[]  boundsBoundDisjunction[i];
         }
      }
      if( idxBoundDisjunctionVars ) delete[] idxBoundDisjunctionVars;
      if( boundTypesBoundDisjunction ) delete[]  boundTypesBoundDisjunction;
      if( boundsBoundDisjunction ) delete[]  boundsBoundDisjunction;
      if( nVarsBoundDisjunction ) delete[] nVarsBoundDisjunction;
      if( flagBoundDisjunctionInitial ) delete[] flagBoundDisjunctionInitial;
      if( flagBoundDisjunctionSeparate ) delete[] flagBoundDisjunctionSeparate;
      if( flagBoundDisjunctionEnforce )  delete[] flagBoundDisjunctionEnforce;
      if( flagBoundDisjunctionCheck ) delete[] flagBoundDisjunctionCheck;
      if( flagBoundDisjunctionPropagate ) delete[] flagBoundDisjunctionPropagate;
      if( flagBoundDisjunctionLocal ) delete[] flagBoundDisjunctionLocal;
      if( flagBoundDisjunctionModifiable ) delete[] flagBoundDisjunctionModifiable;
      if( flagBoundDisjunctionDynamic ) delete[] flagBoundDisjunctionDynamic;
      if( flagBoundDisjunctionRemovable ) delete[] flagBoundDisjunctionRemovable;
      if( flagBoundDisjunctionStickingatnode ) delete[] flagBoundDisjunctionStickingatnode;
   }
};

class ScipParaDiffSubproblemVarBranchStats
{
public:
   int           offset;                /**< root node depth of the collected nodes */
   int           nVarBranchStats;       /**< number of branch stats */
   int           *idxBranchStatsVars;  /**< indices of branch stats vars */
   SCIP_Real     *downpscost;           /**< values to which pseudocosts for downwards branching */
   SCIP_Real     *uppscost;             /**< values to which pseudocosts for upwards branching */
   SCIP_Real     *downvsids;            /**< values to which VSIDS score for downwards branching */
   SCIP_Real     *upvsids;              /**< values to which VSIDS score for upwards branching */
   SCIP_Real     *downconflen;          /**< values to which conflict length score for downwards branching */
   SCIP_Real     *upconflen;            /**< values to which conflict length score for upwards branching */
   SCIP_Real     *downinfer;            /**< values to which inference counter for downwards branching */
   SCIP_Real     *upinfer;              /**< values to which inference counter for upwards branching */
   SCIP_Real     *downcutoff;           /**< values to which cutoff counter for downwards branching */
   SCIP_Real     *upcutoff;             /**< values to which cutoff counter for upwards branching */

   /** default constructor */
   ScipParaDiffSubproblemVarBranchStats(
         ) : offset(0),
             nVarBranchStats(0),
             idxBranchStatsVars(0),
             downpscost(0),
             uppscost(0),
             downvsids(0),
             upvsids(0),
             downconflen(0),
             upconflen(0),
             downinfer(0),
             upinfer(0),
             downcutoff(0),
             upcutoff(0)
   {
   }
   /** destractor */
   ~ScipParaDiffSubproblemVarBranchStats(
         )
   {
      if( idxBranchStatsVars ) delete[] idxBranchStatsVars;
      if( downpscost ) delete[] downpscost;
      if( uppscost ) delete[] uppscost;
      if( downvsids ) delete[] downvsids;
      if( upvsids ) delete[] upvsids;
      if( downconflen ) delete[] downconflen;
      if( upconflen ) delete[] upconflen;
      if( downinfer ) delete[] downinfer;
      if( upinfer ) delete[] upinfer;
      if( downcutoff ) delete[] downcutoff;
      if( upcutoff ) delete[] upcutoff;
   }
};

class ScipParaDiffSubproblemVarValues
{
public:
   int           nVarValueVars;         /**< number of variable value variables */
   int           nVarValues;            /**< number of variable values, for getting statistics */
   int           *idxVarValueVars;      /**< indicies of variable value vars */
   int           *nVarValueValues;      /**< number of values for a variable */
   SCIP_Real     **varValue;            /**< domain value, or SCIP_UNKNOWN */
   SCIP_Real     **varValueDownvsids;   /**< value to which VSIDS score for downwards branching should be initialized */
   SCIP_Real     **varVlaueUpvsids;     /**< value to which VSIDS score for upwards branching should be initialized */
   SCIP_Real     **varValueDownconflen; /**< value to which conflict length score for downwards branching should be initialized */
   SCIP_Real     **varValueUpconflen;   /**< value to which conflict length score for upwards branching should be initialized */
   SCIP_Real     **varValueDowninfer;   /**< value to which inference counter for downwards branching should be initialized */
   SCIP_Real     **varValueUpinfer;     /**< value to which inference counter for upwards branching should be initialized */
   SCIP_Real     **varValueDowncutoff;  /**< value to which cutoff counter for downwards branching should be initialized */
   SCIP_Real     **varValueUpcutoff;    /**< value to which cutoff counter for upwards branching should be initialized */
   /** default constructor */
   ScipParaDiffSubproblemVarValues(
         ) : nVarValueVars(0),
             nVarValues(0),
             idxVarValueVars(0),
             nVarValueValues(0),
             varValue(0),
             varValueDownvsids(0),
             varVlaueUpvsids(0),
             varValueDownconflen(0),
             varValueUpconflen(0),
             varValueDowninfer(0),
             varValueUpinfer(0),
             varValueDowncutoff(0),
             varValueUpcutoff(0)
   {
   }
   /** destractor */
   ~ScipParaDiffSubproblemVarValues(
         )
   {
      for( int i = 0; i < nVarValueVars; i++ )
      {
         if( nVarValueValues[i] > 0 )
         {
            if( varValue[i] ) delete[] varValue[i];
            if( varValueDownvsids[i] ) delete[] varValueDownvsids[i];
            if( varVlaueUpvsids[i] ) delete[] varVlaueUpvsids[i];
            if( varValueDownconflen[i] ) delete[] varValueDownconflen[i];
            if( varValueUpconflen[i] ) delete[] varValueUpconflen[i];
            if( varValueDowninfer[i] ) delete[] varValueDowninfer[i];
            if( varValueUpinfer[i] ) delete[] varValueUpinfer[i];
            if( varValueDowncutoff[i] ) delete[] varValueDowncutoff[i];
            if( varValueUpcutoff[i] ) delete[] varValueUpcutoff[i];
         }
      }
      if( varValue ) delete [] varValue;
      if( varValueDownvsids ) delete[] varValueDownvsids;
      if( varVlaueUpvsids ) delete[] varVlaueUpvsids;
      if( varValueDownconflen ) delete[] varValueDownconflen;
      if( varValueUpconflen ) delete[] varValueUpconflen;
      if( varValueDowninfer ) delete[] varValueDowninfer;
      if( varValueUpinfer ) delete[] varValueUpinfer;
      if( varValueDowncutoff ) delete[] varValueDowncutoff;
      if( varValueUpcutoff ) delete[] varValueUpcutoff;
      if( idxVarValueVars ) delete[] idxVarValueVars;
      if( nVarValueValues ) delete[] nVarValueValues;
   }
};

/** The difference between instance and subproblem: this is base class */
class ScipParaDiffSubproblem : public UG::BbParaDiffSubproblem
{
protected:
   int localInfoIncluded;                /**< 0 (0000 0000): not included
                                              1 (0000 0001): if local cuts are included
                                              2 (0000 0010): if conflicts are included
                                              3 (0000 0011): if local cuts and conflicts are included */
   /******************************
    * for variable bound changes *
    * ***************************/
   int             nBoundChanges;            /**< number of branching variables */
   int             *indicesAmongSolvers; /**< array of variable indices ( unique index )  */
   SCIP_Real       *branchBounds;        /**< array of bounds which the branchings     */
   SCIP_BOUNDTYPE  *boundTypes;          /**< array of boundtypes which the branchings */
   /*******************************************
    * for the case of branching by constraint *
    * ****************************************/
   ScipParaDiffSubproblemBranchLinearCons *branchLinearConss;  /**< point to branch constraints */
   ScipParaDiffSubproblemBranchSetppcCons *branchSetppcConss;  /**< point to branch constraints */
   /************************************
    * for local cuts and conflict cuts *
    * *********************************/
   ScipParaDiffSubproblemLinearCons *linearConss;   /**< point to linear constraint data */
   /************************************
    * for benders cuts                 *
    * *********************************/
   ScipParaDiffSubproblemLinearCons *bendersLinearConss;   /**< point to benders linear constraint data */
   /********************************
    * for conflicts                *
    * *****************************/
   ScipParaDiffSubproblemBoundDisjunctions *boundDisjunctions; /**< point to bound disjunctions */
   /********************************
    * for local var brnach stats   *
    * *****************************/
   ScipParaDiffSubproblemVarBranchStats *varBranchStats; /**< point to varialbe branch stats */
   /**************************************
    * for local var value brnach stats   *
    * ***********************************/
   ScipParaDiffSubproblemVarValues *varValues; /**< point to variable values */
#ifdef UG_DEBUG_SOLUTION
   int           includeOptimalSol;     /**< indicate if sub-MIP includes optimal solution or not. 1: include, 0: not */
#endif
public:
   /** default constructor */
   ScipParaDiffSubproblem(
         )
         : localInfoIncluded(0),
           nBoundChanges(0), indicesAmongSolvers(0), branchBounds(0), boundTypes(0),
           branchLinearConss(0),
           branchSetppcConss(0),
           linearConss(0),
           bendersLinearConss(0),
           boundDisjunctions(0),
           varBranchStats(0),
           varValues(0)
   {
#ifdef UG_DEBUG_SOLUTION
      includeOptimalSol = 0;
#endif 
   }

   ScipParaDiffSubproblem(
         SCIP *scip,
         ScipParaSolver *scipParaSolver,
         int nNewBranchVars,
         SCIP_VAR **newBranchVars,
         SCIP_Real *newBranchBounds,
         SCIP_BOUNDTYPE *newBoundTypes,
         int nAddedConss,
         SCIP_CONS **addedConss
         );

   ScipParaDiffSubproblem(
         ScipParaDiffSubproblem *diffSubproblem
         ) :   localInfoIncluded(0),
               nBoundChanges(0), indicesAmongSolvers(0), branchBounds(0), boundTypes(0),
               branchLinearConss(0),
               branchSetppcConss(0),
               linearConss(0),
               bendersLinearConss(0),
               boundDisjunctions(0),
               varBranchStats(0),
               varValues(0)

   {
      if( !diffSubproblem ) return;

      localInfoIncluded = diffSubproblem->localInfoIncluded;
      nBoundChanges = diffSubproblem->nBoundChanges;
      if( nBoundChanges )
      {
         indicesAmongSolvers = new int[nBoundChanges];
         branchBounds = new SCIP_Real[nBoundChanges];
         boundTypes = new SCIP_BOUNDTYPE[nBoundChanges];
         for( int i = 0; i < nBoundChanges; i++ )
         {
            indicesAmongSolvers[i] = diffSubproblem->indicesAmongSolvers[i];
            branchBounds[i] = diffSubproblem->branchBounds[i];
            boundTypes[i] = diffSubproblem->boundTypes[i];
         }
      }

      if( diffSubproblem->branchLinearConss )
      {
         branchLinearConss = new ScipParaDiffSubproblemBranchLinearCons();
         branchLinearConss->nLinearConss = diffSubproblem->branchLinearConss->nLinearConss;
         assert( branchLinearConss->nLinearConss > 0 );
         branchLinearConss->linearLhss = new SCIP_Real[branchLinearConss->nLinearConss];
         branchLinearConss->linearRhss = new SCIP_Real[branchLinearConss->nLinearConss];
         branchLinearConss->nLinearCoefs = new int[branchLinearConss->nLinearConss];
         branchLinearConss->linearCoefs = new SCIP_Real*[branchLinearConss->nLinearConss];
         branchLinearConss->idxLinearCoefsVars = new int*[branchLinearConss->nLinearConss];
         for( int c = 0; c < branchLinearConss->nLinearConss; c++ )
         {
            branchLinearConss->linearLhss[c] = diffSubproblem->branchLinearConss->linearLhss[c];
            branchLinearConss->linearRhss[c] = diffSubproblem->branchLinearConss->linearRhss[c];
            branchLinearConss->nLinearCoefs[c] = diffSubproblem->branchLinearConss->nLinearCoefs[c];
            branchLinearConss->linearCoefs[c] = new SCIP_Real[branchLinearConss->nLinearCoefs[c]];
            branchLinearConss->idxLinearCoefsVars[c] = new int[branchLinearConss->nLinearCoefs[c]];
            for( int v = 0; v < branchLinearConss->nLinearCoefs[c]; v++ )
            {
               branchLinearConss->linearCoefs[c][v] = diffSubproblem->branchLinearConss->linearCoefs[c][v];
               branchLinearConss->idxLinearCoefsVars[c][v] = diffSubproblem->branchLinearConss->idxLinearCoefsVars[c][v];
            }
         }
         branchLinearConss->lConsNames = diffSubproblem->branchLinearConss->lConsNames;
         branchLinearConss->consNames = new char[branchLinearConss->lConsNames+1];
         for( int c = 0; c < branchLinearConss->lConsNames; c++ )
         {
            branchLinearConss->consNames[c] = diffSubproblem->branchLinearConss->consNames[c];
         }
         branchLinearConss->consNames[branchLinearConss->lConsNames] = '\0';
      }

      if( diffSubproblem->branchSetppcConss )
      {
         branchSetppcConss = new ScipParaDiffSubproblemBranchSetppcCons();
         branchSetppcConss->nSetppcConss = diffSubproblem->branchSetppcConss->nSetppcConss;
         assert( branchSetppcConss->nSetppcConss > 0 );
         branchSetppcConss->nSetppcVars = new int[branchSetppcConss->nSetppcConss];
         branchSetppcConss->setppcTypes = new int[branchSetppcConss->nSetppcConss];
         branchSetppcConss->idxSetppcVars = new int*[branchSetppcConss->nSetppcConss];
         for( int c = 0; c < branchSetppcConss->nSetppcConss; c++ )
         {
            branchSetppcConss->nSetppcVars[c] = diffSubproblem->branchSetppcConss->nSetppcVars[c];
            branchSetppcConss->setppcTypes[c] = diffSubproblem->branchSetppcConss->setppcTypes[c];
            branchSetppcConss->idxSetppcVars[c] = new int[branchSetppcConss->nSetppcVars[c]];
            for( int v = 0; v < branchSetppcConss->nSetppcVars[c]; v++ )
            {
               branchSetppcConss->idxSetppcVars[c][v] = diffSubproblem->branchSetppcConss->idxSetppcVars[c][v];
            }
         }
         branchSetppcConss->lConsNames = diffSubproblem->branchSetppcConss->lConsNames;
         branchSetppcConss->consNames = new char[branchSetppcConss->lConsNames+1];
         for( int c = 0; c < branchSetppcConss->lConsNames; c++ )
         {
            branchSetppcConss->consNames[c] = diffSubproblem->branchSetppcConss->consNames[c];
         }
         branchSetppcConss->consNames[branchSetppcConss->lConsNames] = '\0';
      }

      if( diffSubproblem->linearConss )
      {
         linearConss = new ScipParaDiffSubproblemLinearCons();
         linearConss->nLinearConss = diffSubproblem->linearConss->nLinearConss;
         assert( linearConss->nLinearConss > 0 );
         // std::cout << "linearConss->nLinearConss = " << linearConss->nLinearConss << std::endl;
         linearConss->linearLhss = new SCIP_Real[linearConss->nLinearConss];
         linearConss->linearRhss = new SCIP_Real[linearConss->nLinearConss];
         linearConss->nLinearCoefs = new int[linearConss->nLinearConss];
         linearConss->linearCoefs = new SCIP_Real*[linearConss->nLinearConss];
         linearConss->idxLinearCoefsVars = new int*[linearConss->nLinearConss];
         for( int c = 0; c < linearConss->nLinearConss; c++ )
         {
            linearConss->linearLhss[c] = diffSubproblem->linearConss->linearLhss[c];
            linearConss->linearRhss[c] = diffSubproblem->linearConss->linearRhss[c];
            linearConss->nLinearCoefs[c] = diffSubproblem->linearConss->nLinearCoefs[c];
            linearConss->linearCoefs[c] = new SCIP_Real[linearConss->nLinearCoefs[c]];
            linearConss->idxLinearCoefsVars[c] = new int[linearConss->nLinearCoefs[c]];
            for( int v = 0; v < linearConss->nLinearCoefs[c]; v++ )
            {
               linearConss->linearCoefs[c][v] = diffSubproblem->linearConss->linearCoefs[c][v];
               linearConss->idxLinearCoefsVars[c][v] = diffSubproblem->linearConss->idxLinearCoefsVars[c][v];
            }
         }
      }

      if( diffSubproblem->bendersLinearConss )
      {
         bendersLinearConss = new ScipParaDiffSubproblemLinearCons();
         bendersLinearConss->nLinearConss = diffSubproblem->bendersLinearConss->nLinearConss;
         assert( bendersLinearConss->nLinearConss > 0 );
         // std::cout << "bendersLinearConss->nLinearConss = " << bendersLinearConss->nLinearConss << std::endl;
         bendersLinearConss->linearLhss = new SCIP_Real[bendersLinearConss->nLinearConss];
         bendersLinearConss->linearRhss = new SCIP_Real[bendersLinearConss->nLinearConss];
         bendersLinearConss->nLinearCoefs = new int[bendersLinearConss->nLinearConss];
         bendersLinearConss->linearCoefs = new SCIP_Real*[bendersLinearConss->nLinearConss];
         bendersLinearConss->idxLinearCoefsVars = new int*[bendersLinearConss->nLinearConss];
         for( int c = 0; c < bendersLinearConss->nLinearConss; c++ )
         {
            bendersLinearConss->linearLhss[c] = diffSubproblem->bendersLinearConss->linearLhss[c];
            bendersLinearConss->linearRhss[c] = diffSubproblem->bendersLinearConss->linearRhss[c];
            bendersLinearConss->nLinearCoefs[c] = diffSubproblem->bendersLinearConss->nLinearCoefs[c];
            bendersLinearConss->linearCoefs[c] = new SCIP_Real[bendersLinearConss->nLinearCoefs[c]];
            bendersLinearConss->idxLinearCoefsVars[c] = new int[bendersLinearConss->nLinearCoefs[c]];
            for( int v = 0; v < bendersLinearConss->nLinearCoefs[c]; v++ )
            {
               bendersLinearConss->linearCoefs[c][v] = diffSubproblem->bendersLinearConss->linearCoefs[c][v];
               bendersLinearConss->idxLinearCoefsVars[c][v] = diffSubproblem->bendersLinearConss->idxLinearCoefsVars[c][v];
            }
         }
      }

      if( diffSubproblem->boundDisjunctions )
      {
         boundDisjunctions = new ScipParaDiffSubproblemBoundDisjunctions();
         boundDisjunctions->nBoundDisjunctions = diffSubproblem->boundDisjunctions->nBoundDisjunctions;
         boundDisjunctions->nTotalVarsBoundDisjunctions = diffSubproblem->boundDisjunctions->nTotalVarsBoundDisjunctions;
         boundDisjunctions->nVarsBoundDisjunction = new int[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->flagBoundDisjunctionInitial = new SCIP_Bool[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->flagBoundDisjunctionSeparate = new SCIP_Bool[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->flagBoundDisjunctionEnforce = new SCIP_Bool[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->flagBoundDisjunctionCheck = new SCIP_Bool[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->flagBoundDisjunctionPropagate = new SCIP_Bool[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->flagBoundDisjunctionLocal = new SCIP_Bool[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->flagBoundDisjunctionModifiable = new SCIP_Bool[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->flagBoundDisjunctionDynamic = new SCIP_Bool[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->flagBoundDisjunctionRemovable = new SCIP_Bool[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->flagBoundDisjunctionStickingatnode = new SCIP_Bool[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->idxBoundDisjunctionVars = new int*[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->boundTypesBoundDisjunction = new SCIP_BOUNDTYPE*[boundDisjunctions->nBoundDisjunctions];
         boundDisjunctions->boundsBoundDisjunction = new SCIP_Real*[boundDisjunctions->nBoundDisjunctions];
         for( int i = 0; i <  boundDisjunctions->nBoundDisjunctions; i++ )
         {
            boundDisjunctions->nVarsBoundDisjunction[i] = diffSubproblem->boundDisjunctions->nVarsBoundDisjunction[i];
            boundDisjunctions->flagBoundDisjunctionInitial[i] = diffSubproblem->boundDisjunctions->flagBoundDisjunctionInitial[i];
            boundDisjunctions->flagBoundDisjunctionSeparate[i] = diffSubproblem->boundDisjunctions->flagBoundDisjunctionSeparate[i];
            boundDisjunctions->flagBoundDisjunctionEnforce[i] = diffSubproblem->boundDisjunctions->flagBoundDisjunctionEnforce[i];
            boundDisjunctions->flagBoundDisjunctionCheck[i] = diffSubproblem->boundDisjunctions->flagBoundDisjunctionCheck[i];
            boundDisjunctions->flagBoundDisjunctionPropagate[i] = diffSubproblem->boundDisjunctions->flagBoundDisjunctionPropagate[i];
            boundDisjunctions->flagBoundDisjunctionLocal[i] = diffSubproblem->boundDisjunctions->flagBoundDisjunctionLocal[i];
            boundDisjunctions->flagBoundDisjunctionModifiable[i] = diffSubproblem->boundDisjunctions->flagBoundDisjunctionModifiable[i];
            boundDisjunctions->flagBoundDisjunctionDynamic[i] = diffSubproblem->boundDisjunctions->flagBoundDisjunctionDynamic[i];
            boundDisjunctions->flagBoundDisjunctionRemovable[i] = diffSubproblem->boundDisjunctions->flagBoundDisjunctionRemovable[i];
            boundDisjunctions->flagBoundDisjunctionStickingatnode[i] = diffSubproblem->boundDisjunctions->flagBoundDisjunctionStickingatnode[i];
            boundDisjunctions->idxBoundDisjunctionVars[i] = diffSubproblem->boundDisjunctions->idxBoundDisjunctionVars[i];
            if( boundDisjunctions->nVarsBoundDisjunction[i] > 0 )
            {
               boundDisjunctions->idxBoundDisjunctionVars[i] = new int[boundDisjunctions->nVarsBoundDisjunction[i]];
               boundDisjunctions->boundTypesBoundDisjunction[i] = new SCIP_BOUNDTYPE[boundDisjunctions->nVarsBoundDisjunction[i]];
               boundDisjunctions->boundsBoundDisjunction[i] = new SCIP_Real[boundDisjunctions->nVarsBoundDisjunction[i]];
               for( int j = 0; j < boundDisjunctions->nVarsBoundDisjunction[i]; j++ )
               {
                  boundDisjunctions->idxBoundDisjunctionVars[i][j] = diffSubproblem->boundDisjunctions->idxBoundDisjunctionVars[i][j];
                  boundDisjunctions->boundTypesBoundDisjunction[i][j] = diffSubproblem->boundDisjunctions->boundTypesBoundDisjunction[i][j];
                  boundDisjunctions->boundsBoundDisjunction[i][j] = diffSubproblem->boundDisjunctions->boundsBoundDisjunction[i][j];
               }
            }
         }
      }


      if( diffSubproblem->varBranchStats )
      {
         varBranchStats = new ScipParaDiffSubproblemVarBranchStats();
         varBranchStats->offset = diffSubproblem->varBranchStats->offset;
         varBranchStats->nVarBranchStats = diffSubproblem->varBranchStats->nVarBranchStats;
         varBranchStats->idxBranchStatsVars = new int[varBranchStats->nVarBranchStats];
         varBranchStats->downpscost = new SCIP_Real[varBranchStats->nVarBranchStats];
         varBranchStats->uppscost = new SCIP_Real[varBranchStats->nVarBranchStats];
         varBranchStats->downvsids = new SCIP_Real[varBranchStats->nVarBranchStats];
         varBranchStats->upvsids = new SCIP_Real[varBranchStats->nVarBranchStats];
         varBranchStats->downconflen = new SCIP_Real[varBranchStats->nVarBranchStats];
         varBranchStats->upconflen = new SCIP_Real[varBranchStats->nVarBranchStats];
         varBranchStats->downinfer = new SCIP_Real[varBranchStats->nVarBranchStats];
         varBranchStats->upinfer = new SCIP_Real[varBranchStats->nVarBranchStats];
         varBranchStats->downcutoff = new SCIP_Real[varBranchStats->nVarBranchStats];
         varBranchStats->upcutoff = new SCIP_Real[varBranchStats->nVarBranchStats];
         for( int i = 0; i < varBranchStats->nVarBranchStats; ++i )
         {
            varBranchStats->idxBranchStatsVars[i] = diffSubproblem->varBranchStats->idxBranchStatsVars[i];
            varBranchStats->downpscost[i] = diffSubproblem->varBranchStats->downpscost[i];
            varBranchStats->uppscost[i] = diffSubproblem->varBranchStats->uppscost[i];
            varBranchStats->downvsids[i] = diffSubproblem->varBranchStats->downvsids[i];
            varBranchStats->upvsids[i] = diffSubproblem->varBranchStats->upvsids[i];
            varBranchStats->downconflen[i] = diffSubproblem->varBranchStats->downconflen[i];
            varBranchStats->upconflen[i] = diffSubproblem->varBranchStats->upconflen[i];
            varBranchStats->downinfer[i] = diffSubproblem->varBranchStats->downinfer[i];
            varBranchStats->upinfer[i] = diffSubproblem->varBranchStats->upinfer[i];
            varBranchStats->downcutoff[i] = diffSubproblem->varBranchStats->downcutoff[i];
            varBranchStats->upcutoff[i] = diffSubproblem->varBranchStats->upcutoff[i];
         }
      }

      if( diffSubproblem->varValues )
      {
         varValues = new ScipParaDiffSubproblemVarValues();
         varValues->nVarValueVars = diffSubproblem->varValues->nVarValueVars;
         varValues->nVarValues = diffSubproblem->varValues->nVarValues;
         varValues->idxVarValueVars = new int[varValues->nVarValueVars];
         varValues->nVarValueValues = new int[varValues->nVarValueVars];
         varValues->varValue            = new SCIP_Real*[varValues->nVarValueVars];
         varValues->varValueDownvsids   = new SCIP_Real*[varValues->nVarValueVars];
         varValues->varVlaueUpvsids     = new SCIP_Real*[varValues->nVarValueVars];
         varValues->varValueDownconflen = new SCIP_Real*[varValues->nVarValueVars];
         varValues->varValueUpconflen   = new SCIP_Real*[varValues->nVarValueVars];
         varValues->varValueDowninfer   = new SCIP_Real*[varValues->nVarValueVars];
         varValues->varValueUpinfer     = new SCIP_Real*[varValues->nVarValueVars];
         varValues->varValueDowncutoff  = new SCIP_Real*[varValues->nVarValueVars];
         varValues->varValueUpcutoff    = new SCIP_Real*[varValues->nVarValueVars];
         for( int i = 0; i < varValues->nVarValueVars; i++ )
         {
            varValues->idxVarValueVars[i] = diffSubproblem->varValues->idxVarValueVars[i];
            varValues->nVarValueValues[i] = diffSubproblem->varValues->nVarValueValues[i];
            if( varValues->nVarValueValues[i] > 0 )
            {
               varValues->varValue[i]            = new SCIP_Real[varValues->nVarValueValues[i]];
               varValues->varValueDownvsids[i]   = new SCIP_Real[varValues->nVarValueValues[i]];
               varValues->varVlaueUpvsids[i]     = new SCIP_Real[varValues->nVarValueValues[i]];
               varValues->varValueDownconflen[i] = new SCIP_Real[varValues->nVarValueValues[i]];
               varValues->varValueUpconflen[i]   = new SCIP_Real[varValues->nVarValueValues[i]];
               varValues->varValueDowninfer[i]   = new SCIP_Real[varValues->nVarValueValues[i]];
               varValues->varValueUpinfer[i]     = new SCIP_Real[varValues->nVarValueValues[i]];
               varValues->varValueDowncutoff[i]  = new SCIP_Real[varValues->nVarValueValues[i]];
               varValues->varValueUpcutoff[i]    = new SCIP_Real[varValues->nVarValueValues[i]];
               for( int j = 0; j < varValues->nVarValueValues[i]; j++ )
               {
                  varValues->varValue[i][j]            = diffSubproblem->varValues->varValue[i][j];
                  varValues->varValueDownvsids[i][j]   = diffSubproblem->varValues->varValueDownvsids[i][j];
                  varValues->varVlaueUpvsids[i][j]     = diffSubproblem->varValues->varVlaueUpvsids[i][j];
                  varValues->varValueDownconflen[i][j] = diffSubproblem->varValues->varValueDownconflen[i][j];
                  varValues->varValueUpconflen[i][j]   = diffSubproblem->varValues->varValueUpconflen[i][j];
                  varValues->varValueDowninfer[i][j]   = diffSubproblem->varValues->varValueDowninfer[i][j];
                  varValues->varValueUpinfer[i][j]     = diffSubproblem->varValues->varValueUpinfer[i][j];
                  varValues->varValueDowncutoff[i][j]  = diffSubproblem->varValues->varValueDowncutoff[i][j];
                  varValues->varValueUpcutoff[i][j]    = diffSubproblem->varValues->varValueUpcutoff[i][j];
               }
            }
         }
      }
#ifdef UG_DEBUG_SOLUTION
      includeOptimalSol = diffSubproblem->includeOptimalSol;
#endif 
   }

   /** destractor */
   virtual ~ScipParaDiffSubproblem()
   {
      if( indicesAmongSolvers ) delete[] indicesAmongSolvers;
      if( branchBounds ) delete[] branchBounds;
      if( boundTypes ) delete[] boundTypes;
      if( branchLinearConss )
      {
         delete branchLinearConss;
      }
      if( branchSetppcConss )
      {
         delete branchSetppcConss;
      }
      if( linearConss )
      {
         delete linearConss;
      }
      if( bendersLinearConss )
      {
         delete bendersLinearConss;
      }
      if( boundDisjunctions )
      {
         delete boundDisjunctions;
      }

      if( varBranchStats  )
      {
         delete varBranchStats;
      }
      if( varValues )
      {
         delete varValues;
      }
   }

   int getNBoundChanges(){ return nBoundChanges; }
   int getIndex(int i){ return indicesAmongSolvers[i]; }
   SCIP_Real getBranchBound(int i){ return branchBounds[i]; }
   SCIP_BOUNDTYPE getBoundType(int i){ return boundTypes[i]; }

   ScipParaDiffSubproblemBranchLinearCons *getBranchLinearConss()
   {
      return branchLinearConss;
   }

   ScipParaDiffSubproblemBranchSetppcCons *getBranchSetppcConss()
   {
      return branchSetppcConss;
   }

   int getNBranchConsLinearConss()
   {
      if( branchLinearConss )
      {
         return branchLinearConss->nLinearConss;
      }
      else
      {
         return 0;
      }
   }

   SCIP_Real getBranchConsLinearLhs(int i)
   {
      assert(branchLinearConss);
      return branchLinearConss->linearLhss[i];
   }

   SCIP_Real getBranchConsLinearRhs(int i)
   {
      assert(branchLinearConss);
      return branchLinearConss->linearRhss[i];
   }

   int getBranchConsNLinearCoefs(int i)
   {
      assert(branchLinearConss);
      return branchLinearConss->nLinearCoefs[i];
   }

   SCIP_Real getBranchConsLinearCoefs(int i, int j)
   {
      assert(branchLinearConss);
      return branchLinearConss->linearCoefs[i][j];
   }

   int getBranchConsLinearIdxCoefsVars(int i, int j)
   {
      assert(branchLinearConss);
      return branchLinearConss->idxLinearCoefsVars[i][j];
   }

   int getBranchConsLinearConsNames()
   {
      assert(branchLinearConss);
      return branchLinearConss->lConsNames;
   }

   char *getBranchConsLinearConsNames(int i)
   {
      char *name = branchLinearConss->consNames;
      for( int j = 0; j < i; j++)
      {
         assert(*name);
         name += (std::strlen(name) + 1);
      }
      assert(*name);
      return name;
   }


   int getNBranchConsSetppcConss()
   {
      if( branchSetppcConss )
      {
         return branchSetppcConss->nSetppcConss;
      }
      else
      {
         return 0;
      }
   }

   int getBranchConsSetppcNVars(int i)
   {
      assert(branchSetppcConss);
      return branchSetppcConss->nSetppcVars[i];
   }

   int getBranchConsSetppcType(int i)
   {
      assert(branchSetppcConss);
      return branchSetppcConss->setppcTypes[i];
   }

   int getBranchConsSetppcVars(int i, int j)
   {
      assert(branchSetppcConss);
      return branchSetppcConss->idxSetppcVars[i][j];
   }

   int getBranchConsSetppcConsNames()
   {
      assert(branchSetppcConss);
      return branchSetppcConss->lConsNames;
   }

   char *getBranchConsSetppcConsNames(int i)
   {
      char *name = branchSetppcConss->consNames;
      for( int j = 0; j < i; j++)
      {
         assert(*name);
         name += (std::strlen(name) + 1);
      }
      assert(*name);
      return name;
   }


   int getNLinearConss()
   {
      if( linearConss )
      {
         return linearConss->nLinearConss;
      }
      else
      {
         return 0;
      }
   }


   SCIP_Real getLinearLhs(int i)
   {
      assert(linearConss);
      return linearConss->linearLhss[i];
   }

   SCIP_Real getLinearRhs(int i)
   {
      assert(linearConss);
      return linearConss->linearRhss[i];
   }

   int getNLinearCoefs(int i)
   {
      assert(linearConss);
      return linearConss->nLinearCoefs[i];
   }

   SCIP_Real getLinearCoefs(int i, int j)
   {
      assert(linearConss);
      return linearConss->linearCoefs[i][j];
   }

   int getIdxLinearCoefsVars(int i, int j)
   {
      assert(linearConss);
      return linearConss->idxLinearCoefsVars[i][j];
   }

   int getNBendersLinearConss()
   {
      if( bendersLinearConss )
      {
         return bendersLinearConss->nLinearConss;
      }
      else
      {
         return 0;
      }
   }

   SCIP_Real getBendersLinearLhs(int i)
   {
      assert(bendersLinearConss);
      return bendersLinearConss->linearLhss[i];
   }

   SCIP_Real getBendersLinearRhs(int i)
   {
      assert(bendersLinearConss);
      return bendersLinearConss->linearRhss[i];
   }

   int getNBendersLinearCoefs(int i)
   {
      assert(bendersLinearConss);
      return bendersLinearConss->nLinearCoefs[i];
   }

   SCIP_Real getBendersLinearCoefs(int i, int j)
   {
      assert(bendersLinearConss);
      return bendersLinearConss->linearCoefs[i][j];
   }

   int getIdxBendersLinearCoefsVars(int i, int j)
   {
      assert(bendersLinearConss);
      return bendersLinearConss->idxLinearCoefsVars[i][j];
   }

   int getNBoundDisjunctions()
   {
      if( boundDisjunctions )
      {
         return boundDisjunctions->nBoundDisjunctions;
      }
      else
      {
         return 0;
      }
   }

   int getNTotalVarsBoundDisjunctions()
   {
      assert(boundDisjunctions);
      return boundDisjunctions->nTotalVarsBoundDisjunctions;
   }

   int getNVarsBoundDisjunction(int i){
      assert(boundDisjunctions);
      assert( boundDisjunctions->nVarsBoundDisjunction && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      return boundDisjunctions->nVarsBoundDisjunction[i];
   }
   SCIP_Bool getFlagBoundDisjunctionInitial(int i)
   {
      assert(boundDisjunctions);
      assert( boundDisjunctions->flagBoundDisjunctionInitial && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      return boundDisjunctions->flagBoundDisjunctionInitial[i];
   }
   SCIP_Bool getFlagBoundDisjunctionSeparate(int i)
   {
      assert(boundDisjunctions);
      assert( boundDisjunctions->flagBoundDisjunctionSeparate && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      return boundDisjunctions->flagBoundDisjunctionSeparate[i];
   }
   SCIP_Bool getFlagBoundDisjunctionEnforce(int i)
   {
      assert(boundDisjunctions);
      assert( boundDisjunctions->flagBoundDisjunctionEnforce && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      return boundDisjunctions->flagBoundDisjunctionEnforce[i];
   }
   SCIP_Bool getFlagBoundDisjunctionCheck(int i)
   {
      assert(boundDisjunctions);
      assert( boundDisjunctions->flagBoundDisjunctionCheck && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      return boundDisjunctions->flagBoundDisjunctionCheck[i];
   }
   SCIP_Bool getFlagBoundDisjunctionPropagate(int i)
   {
      assert(boundDisjunctions);
      assert( boundDisjunctions->flagBoundDisjunctionPropagate && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      return boundDisjunctions->flagBoundDisjunctionPropagate[i];
   }
   SCIP_Bool getFlagBoundDisjunctionLocal(int i)
   {
      assert( boundDisjunctions->flagBoundDisjunctionLocal && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      return boundDisjunctions->flagBoundDisjunctionLocal[i];
   }
   SCIP_Bool getFlagBoundDisjunctionModifiable(int i)
   {
      assert(boundDisjunctions);
      assert( boundDisjunctions->flagBoundDisjunctionModifiable && i >= 0 && i <boundDisjunctions->nBoundDisjunctions );
      return boundDisjunctions->flagBoundDisjunctionModifiable[i];
   }
   SCIP_Bool getFlagBoundDisjunctionDynamic(int i)
   {
      assert(boundDisjunctions);
      assert( boundDisjunctions->flagBoundDisjunctionDynamic && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      return boundDisjunctions->flagBoundDisjunctionDynamic[i];
   }
   SCIP_Bool getFlagBoundDisjunctionRemovable(int i)
   {
      assert(boundDisjunctions);
      assert( boundDisjunctions->flagBoundDisjunctionRemovable && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      return boundDisjunctions->flagBoundDisjunctionRemovable[i];
   }
   SCIP_Bool getFlagBoundDisjunctionStickingatnode(int i)
   {
      assert(boundDisjunctions);
      assert( boundDisjunctions->flagBoundDisjunctionStickingatnode && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      return boundDisjunctions->flagBoundDisjunctionStickingatnode[i];
   }

   int getIdxBoundDisjunctionVars(int i, int j)
   {
      assert(boundDisjunctions);
      assert( boundDisjunctions->idxBoundDisjunctionVars && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      assert( boundDisjunctions->idxBoundDisjunctionVars[i] && j >= 0 && j < boundDisjunctions->nVarsBoundDisjunction[i] );
      return boundDisjunctions->idxBoundDisjunctionVars[i][j];
   }
   SCIP_BOUNDTYPE  getBoundTypesBoundDisjunction(int i, int j)
   {
      assert(boundDisjunctions);
      assert( boundDisjunctions->boundTypesBoundDisjunction && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      assert( boundDisjunctions->boundTypesBoundDisjunction[i] && j >= 0 && j < boundDisjunctions->nVarsBoundDisjunction[i] );
      return boundDisjunctions->boundTypesBoundDisjunction[i][j];
   }
   SCIP_Real getBoundsBoundDisjunction(int i, int j)
   {
      assert(boundDisjunctions);
      assert( boundDisjunctions->boundsBoundDisjunction && i >= 0 && i < boundDisjunctions->nBoundDisjunctions );
      assert( boundDisjunctions->boundsBoundDisjunction[i] && j >= 0 && j < boundDisjunctions->nVarsBoundDisjunction[i] );
      return boundDisjunctions->boundsBoundDisjunction[i][j];
   }

   int getNVarBranchStats()
   {

      if( varBranchStats )
      {
         return varBranchStats->nVarBranchStats;
      }
      else
      {
         return 0;
      }
   }

   int getIdxLBranchStatsVars(int i)
   {
      assert(varBranchStats);
      assert( i >= 0 && i <varBranchStats->nVarBranchStats );
      return varBranchStats->idxBranchStatsVars[i];
   }

   SCIP_Real getDownpscost(int i)
   {
      assert(varBranchStats);
      assert( i >= 0 && i <varBranchStats->nVarBranchStats );
      return varBranchStats->downpscost[i];
   }

   SCIP_Real getUppscost(int i)
   {
      assert(varBranchStats);
      assert( i >= 0 && i <varBranchStats->nVarBranchStats );
      return varBranchStats->uppscost[i];
   }

   SCIP_Real getDownvsids(int i)
   {
      assert(varBranchStats);
      assert( i >= 0 && i <varBranchStats->nVarBranchStats );
      return varBranchStats->downvsids[i];
   }

   SCIP_Real getUpvsids(int i)
   {
      assert(varBranchStats);
      assert( i >= 0 && i <varBranchStats->nVarBranchStats );
      return varBranchStats->upvsids[i];
   }

   SCIP_Real getDownconflen(int i)
   {
      assert(varBranchStats);
      assert( i >= 0 && i <varBranchStats->nVarBranchStats );
      return varBranchStats->downconflen[i];
   }

   SCIP_Real getUpconflen(int i)
   {
      assert(varBranchStats);
      assert( i >= 0 && i <varBranchStats->nVarBranchStats );
      return varBranchStats->upconflen[i];
   }

   SCIP_Real getDowninfer(int i)
   {
      assert(varBranchStats);
      assert( i >= 0 && i <varBranchStats->nVarBranchStats );
      return varBranchStats->downinfer[i];
   }

   SCIP_Real getUpinfer(int i)
   {
      assert(varBranchStats);
      assert( i >= 0 && i <varBranchStats->nVarBranchStats );
      return varBranchStats->upinfer[i];
   }

   SCIP_Real getDowncutoff(int i)
   {
      assert(varBranchStats);
      assert( i >= 0 && i <varBranchStats->nVarBranchStats );
      return varBranchStats->downcutoff[i];
   }

   SCIP_Real getUpcutoff(int i)
   {
      assert(varBranchStats);
      assert( i >= 0 && i <varBranchStats->nVarBranchStats );
      return varBranchStats->upcutoff[i];
   }


   int getNVarValueVars()
   {
      if( varValues )
      {
         return varValues->nVarValueVars;
      }
      else
      {
         return 0;
      }
   }

   int getNVarValues()
   {
      assert(varValues);
      return varValues->nVarValues;
   }

   int getIdxVarValueVars(int i)
   {
      assert(varValues);
      assert(varValues->idxVarValueVars && i >= 0 && i < varValues->nVarValueVars );
      return varValues->idxVarValueVars[i];
   }

   int getNVarValueValues(int i)
   {
      assert(varValues);
      assert(varValues->idxVarValueVars && i >= 0 && i < varValues->nVarValueVars );
      return varValues->nVarValueValues[i];
   }

   SCIP_Real getVarValue(int i, int j)
   {
      assert(varValues);
      assert(varValues->varValue && i >= 0 && i < varValues->nVarValueVars );
      assert(varValues->varValue[i] && j >= 0 && j < varValues->nVarValueValues[i] );
      return varValues->varValue[i][j];
   }

   SCIP_Real getVarValueDownvsids(int i, int j)
   {
      assert(varValues);
      assert(varValues->varValueDownvsids && i >= 0 && i < varValues->nVarValueVars );
      assert(varValues->varValueDownvsids[i] && j >= 0 && j < varValues->nVarValueValues[i] );
      return varValues->varValueDownvsids[i][j];
   }

   SCIP_Real getVarVlaueUpvsids(int i, int j)
   {
      assert(varValues);
      assert(varValues->varVlaueUpvsids && i >= 0 && i < varValues->nVarValueVars );
      assert(varValues->varVlaueUpvsids[i] && j >= 0 && j < varValues->nVarValueValues[i] );
      return varValues->varVlaueUpvsids[i][j];
   }

   SCIP_Real getVarValueDownconflen(int i, int j)
   {
      assert(varValues);
      assert(varValues->varValueDownconflen && i >= 0 && i < varValues->nVarValueVars );
      assert(varValues->varValueDownconflen[i] && j >= 0 && j < varValues->nVarValueValues[i] );
      return varValues->varValueDownconflen[i][j];
   }

   SCIP_Real getVarValueUpconflen(int i, int j)
   {
      assert(varValues);
      assert(varValues->varValueUpconflen && i >= 0 && i < varValues->nVarValueVars );
      assert(varValues->varValueUpconflen[i] && j >= 0 && j < varValues->nVarValueValues[i] );
      return varValues->varValueUpconflen[i][j];
   }

   SCIP_Real getVarValueDowninfer(int i, int j)
   {
      assert(varValues);
      assert(varValues->varValueDowninfer && i >= 0 && i < varValues->nVarValueVars );
      assert(varValues->varValueDowninfer[i] && j >= 0 && j < varValues->nVarValueValues[i] );
      return varValues->varValueDowninfer[i][j];
   }

   SCIP_Real getVarValueUpinfer(int i, int j)
   {
      assert(varValues);
      assert(varValues->varValueUpinfer && i >= 0 && i < varValues->nVarValueVars );
      assert(varValues->varValueUpinfer[i] && j >= 0 && j < varValues->nVarValueValues[i] );
      return varValues->varValueUpinfer[i][j];
   }

   SCIP_Real getVarValueDowncutoff(int i, int j)
   {
      assert(varValues);
      assert(varValues->varValueDowncutoff && i >= 0 && i < varValues->nVarValueVars );
      assert(varValues->varValueDowncutoff[i] && j >= 0 && j < varValues->nVarValueValues[i] );
      return varValues->varValueDowncutoff[i][j];
   }

   SCIP_Real getVarValueUpcutoff(int i, int j)
   {
      assert(varValues);
      assert(varValues->varValueUpcutoff && i >= 0 && i < varValues->nVarValueVars );
      assert(varValues->varValueUpcutoff[i] && j >= 0 && j < varValues->nVarValueValues[i] );
      return varValues->varValueUpcutoff[i][j];
   }

   void addBranchLinearConss(
         SCIP *scip,
         ScipParaSolver *scipParaSolver,
         int nLenarConss,
         int nAddedConss,
         SCIP_CONS **addedConss
         );

   void addBranchSetppcConss(
         SCIP *scip,
         ScipParaSolver *scipParaSolver,
         int nSetpartConss,
         int nAddedConss,
         SCIP_CONS **addedConss
         );

   void addLocalNodeInfo(
         SCIP *scip,
         ScipParaSolver *scipParaSolver
         );

   void addBoundDisjunctions(
         SCIP *scip,
         ScipParaSolver *scipParaSolver
         );

   void addBranchVarStats(
         SCIP *scip,
         ScipParaSolver *scipParaSolver
         );

   void addVarValueStats(
         SCIP *scip,
         ScipParaSolver *scipParaSolver
         );

   void addInitialBranchVarStats(
         int minDepth,
         int maxDepth,
         SCIP *scip
         );

   int getOffset()
   {
      assert(varBranchStats);
      return varBranchStats->offset;
   }

#ifdef UG_WITH_ZLIB
   void write(gzstream::ogzstream &out);
   void read(UG::ParaComm *comm, gzstream::igzstream &in, bool onlyBoundChanges);
#endif

   /** get fixed variables **/
   int getFixedVariables(
         UG::ParaInstance *instance,
         UG::BbParaFixedVariable **fixedVars );

   /** create new ParaDiffSubproblem using fixed variables information */
   BbParaDiffSubproblem* createDiffSubproblem(
		   UG::ParaComm *comm,
		   UG::ParaInitiator *initiator,
		   int n,
		   UG::BbParaFixedVariable *fixedVars );

   /** stringfy ParaCalculationState */
   const std::string toString(
         );

   /** stringfy statistics to log file */
   const std::string toStringStat(){
      std::ostringstream s;
      if (branchLinearConss)
      {
         s << ", blc: " << branchLinearConss->nLinearConss;
         s << "," << branchLinearConss->lConsNames;
      }
      else
      {
         s << ", blc: 0";
      }
      if (branchSetppcConss)
      {
         s << ", bsc: " << branchSetppcConss->nSetppcConss;
         s << "," << branchSetppcConss->lConsNames;
      }
      else
      {
         s << ", bsc: 0";
      }
      if( linearConss )
      {
         s << ", nl: " << linearConss->nLinearConss;
      }
      else
      {
         s << ", nl: 0";
      }
      if( bendersLinearConss )
      {
         s << ", bnl: " << bendersLinearConss->nLinearConss;
      }
      else
      {
         s << ", nl: 0";
      }


      if ( boundDisjunctions )
      {
         s << ", nbd: " << boundDisjunctions->nBoundDisjunctions << ", nbdt: " << boundDisjunctions->nTotalVarsBoundDisjunctions;
      }
      else
      {
         s << ", nbd: 0, nbdt: 0";
      }

      if( varBranchStats )
      {
         s << ", nbs: " << varBranchStats->nVarBranchStats;
      }
      else
      {
         s << ", nbs: 0";
      }

      if( varValues )
      {
         s << ", nvv: " << varValues->nVarValueVars << ", nvvt: " << varValues->nVarValues;
      }
      else
      {
         s << ", nvv: 0, nvvt: 0";
      }
      return s.str();
   }

#ifdef UG_DEBUG_SOLUTION
   bool isOptimalSolIncluded(){ return (includeOptimalSol != 0); }
   void setOptimalSolIndicator(int i){ includeOptimalSol = i; }
#endif

};

}

#endif    // __SCIP_PARA_DIFF_SUBPROBLEM_H__

