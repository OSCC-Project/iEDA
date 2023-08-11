/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program                         */
/*          GCG --- Generic Column Generation                                */
/*                  a Dantzig-Wolfe decomposition based extension            */
/*                  of the branch-cut-and-price framework                    */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/* Copyright (C) 2010-2022 Operations Research, RWTH Aachen University       */
/*                         Zuse Institute Berlin (ZIB)                       */
/*                                                                           */
/* This program is free software; you can redistribute it and/or             */
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
/* along with this program; if not, write to the Free Software               */
/* Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.*/
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   rowgraph.h
 * @brief  A row graph where each row is a node and rows are adjacent if they share a variable
 * @author Martin Bergner
 * @author Annika Thome
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_ROWGRAPH_H_
#define GCG_ROWGRAPH_H_

#include "graph.h"
#include "bipartitegraph.h"
#include "matrixgraph.h"

namespace gcg {
template <class T>
class RowGraph : public gcg::MatrixGraph<T>
{
protected:
   gcg::Graph<T> graph;

public:
   RowGraph(
         SCIP*                 scip,              /**< SCIP data structure */
         Weights               w                  /**< weights for the given graph */
      );
   virtual ~RowGraph();

   virtual SCIP_RETCODE createDecompFromPartition(
      DEC_DECOMP**       decomp              /**< decomposition structure to generate */
      );

   /** amplifies a partialdec by dint of a graph created with open constraints and open variables of the partialdec */
   virtual SCIP_RETCODE createPartialdecFromPartition(
      PARTIALDECOMP*      oldpartialdec,            /**< partialdec which should be amplifies */
      PARTIALDECOMP**     firstpartialdec,          /**< pointer to buffer the new partialdec amplified by dint of the graph */
      PARTIALDECOMP**     secondpartialdec,         /**< pointer to buffer the new partialdec whose border is amplified by dint of the graph */
      DETPROBDATA*        detprobdata          /**< datprobdata the partialdecs correspond to */
      );

   virtual SCIP_RETCODE createFromMatrix(
      SCIP_CONS**           conss,              /**< constraints for which graph should be created */
      SCIP_VAR**            vars,               /**< variables for which graph should be created */
      int                   nconss_,            /**< number of constraints */
      int                   nvars_              /**< number of variables */
      );
};

} /* namespace gcg */
#endif /* GCG_ROWGRAPH_H_ */
