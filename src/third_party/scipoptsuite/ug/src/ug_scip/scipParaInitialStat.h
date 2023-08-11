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

/**@file    scipParaInitialStat.h
 * @brief   ParaInitialStat extension for SCIP solver.
 * @author  Yuji Shinano
 *
 *
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef __SCIP_PARA_INITIAL_STAT_H__
#define __SCIP_PARA_INITIAL_STAT_H__

#include <iostream>
#include "ug_bb/bbParaComm.h"
#include "scip/scip.h"
#include "ug/paraInitialStat.h"

namespace ParaSCIP
{

/** The initial statistic collecting data class: this is base class */
class ScipParaInitialStat : public UG::ParaInitialStat
{
protected:
   int           maxDepth;                /**< maximal depth of all processed nodes in current branch and bound run (excluding probing nodes) */
   int           maxTotalDepth;           /**< maximal depth of all processed nodes over all branch and bound runs */
   /********************************
    * for var brnach stats        *
    * *****************************/
   int           nVarBranchStatsDown;      /**< number of branch stats downward */
   int           nVarBranchStatsUp;        /**< number of branch stats upward */
   int           *idxLBranchStatsVarsDown; /**< indices of branch stats vars downward */
   int           *nVarBranchingDown;       /**< number of branchings of this var to downward */
   int           *idxLBranchStatsVarsUp;   /**< indices of branch stats vars upward  */
   int           *nVarBranchingUp;         /**< number of branchings of this var to upward */
   SCIP_Real     *downpscost;              /**< values to which pseudocosts for downwards branching */
   SCIP_Real     *downvsids;               /**< values to which VSIDS score for downwards branching */
   SCIP_Real     *downconflen;             /**< values to which conflict length score for downwards branching */
   SCIP_Real     *downinfer;               /**< values to which inference counter for downwards branching */
   SCIP_Real     *downcutoff;              /**< values to which cutoff counter for downwards branching */
   SCIP_Real     *uppscost;                /**< values to which pseudocosts for upwards branching */
   SCIP_Real     *upvsids;                 /**< values to which VSIDS score for upwards branching */
   SCIP_Real     *upconflen;               /**< values to which conflict length score for upwards branching */
   SCIP_Real     *upinfer;                 /**< values to which inference counter for upwards branching */
   SCIP_Real     *upcutoff;                /**< values to which cutoff counter for upwards branching */
public:
   /** default constructor */
   ScipParaInitialStat(
         ) :
            maxDepth(0),
            maxTotalDepth(0),
            nVarBranchStatsDown(0),
            nVarBranchStatsUp(0),
            idxLBranchStatsVarsDown(0),
            nVarBranchingDown(0),
            idxLBranchStatsVarsUp(0),
            nVarBranchingUp(0),
            downpscost(0),
            downvsids(0),
            downconflen(0),
            downinfer(0),
            downcutoff(0),
            uppscost(0),
            upvsids(0),
            upconflen(0),
            upinfer(0),
            upcutoff(0)
   {
   }

   /** constructor to create this object */
   ScipParaInitialStat(SCIP *scip);

   /** destractor */
   virtual ~ScipParaInitialStat()
   {
      if( idxLBranchStatsVarsDown ) delete[] idxLBranchStatsVarsDown;
      if( nVarBranchingDown ) delete[] nVarBranchingDown;
      if( idxLBranchStatsVarsUp ) delete[] idxLBranchStatsVarsUp;
      if( nVarBranchingUp ) delete[] nVarBranchingUp;
      if( downpscost ) delete[] downpscost;
      if( downvsids ) delete[] downvsids;
      if( downconflen ) delete[] downconflen;
      if( downinfer ) delete[] downinfer;
      if( downcutoff ) delete[] downcutoff;
      if( uppscost ) delete[] uppscost;
      if( upvsids ) delete[] upvsids;
      if( upconflen ) delete[] upconflen;
      if( upinfer ) delete[] upinfer;
      if( upcutoff ) delete[] upcutoff;
   }

   /** create clone of this object */
   ParaInitialStat *clone(UG::ParaComm *comm);

   /** accumulate initial statistics into target scip environment */
   void accumulateOn(SCIP *scip);

   /** get maximum depth */
   int getMaxDepth(){ return maxDepth; }

   /** stringfy subproblem ( for debugging ) */
   const std::string toString(){ return std::string(); }
};

}

#endif    // __SCIP_PARA_INITIAL_STAT_H__

