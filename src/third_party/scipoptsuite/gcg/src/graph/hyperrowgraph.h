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

/**@file   hyperrowgraph.h
 * @brief  Column hypergraph
 * @author Martin Bergner
 * @author Annika Thome
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef GCG_HYPERROWGRAPH_H_
#define GCG_HYPERROWGRAPH_H_

#include "matrixgraph.h"
#include "hypergraph.h"
#include "class_partialdecomp.h"
#include "class_detprobdata.h"

namespace gcg
{
template <class T>
class HyperrowGraph: public gcg::MatrixGraph<T>
{
private:
   Hypergraph<T> graph;
public:
   HyperrowGraph(
      SCIP*                 scip,              /**< SCIP data structure */
      Weights               w                  /**< weights for the given graph */
   );

   virtual ~HyperrowGraph();

   /** writes the graph to the given file.
    *  The format is graph dependent
    */
   SCIP_RETCODE writeToFile(
      int                fd,                 /**< filename where the graph should be written to */
      SCIP_Bool          edgeweights         /**< whether to write edgeweights */
    );

   /** return the number of nodes */
   virtual int getNNodes();

   /** return the number of edges (or hyperedges) */
   virtual int getNEdges();

   /** return node degree */
   virtual int getNNeighbors(
      int i
      );

   virtual std::vector<int> getNeighbors(
         int i
      )
      {
      return this->graph.getNeighbors(i);
      }

   virtual std::vector<int> getHyperedgeNodes(
         int i
      );

   /**
    * reads the partition from the given file.
    * The format is graph dependent. The default is a file with one line for each node a
    */
   virtual SCIP_RETCODE readPartition(
      const char*        filename            /**< filename where the partition is stored */
   )
   {
      SCIP_CALL( this->graph.readPartition(filename) );
      return SCIP_OKAY;
   }

   /** return a partition of the nodes */
   virtual std::vector<int> getPartition()
   {
      return this->graph.getPartition();
   }

   virtual SCIP_RETCODE createDecompFromPartition(
      DEC_DECOMP**       decomp              /**< decomposition structure to generate */
      );

   /** amplifies a partialdec by dint of a graph created with open constraints and open variables of the partialdec */
   virtual SCIP_RETCODE createPartialdecFromPartition(
      PARTIALDECOMP*      oldpartialdec,            /**< partialdec which should be amplifies */
      PARTIALDECOMP**     firstpartialdec,          /**< pointer to buffer the new partialdec amplified by dint of the graph */
      PARTIALDECOMP**     secondpartialdec,         /**< pinter to buffer the new partialdec whose border is amplified by dint of the graph */
      DETPROBDATA*        detprobdata               /**< detection process information and data */
      );

   /** creates a new partialdec by dint of a graph created with all constraints and variables */
   virtual SCIP_RETCODE createPartialdecFromPartition(
      PARTIALDECOMP**      firstpartialdec,         /**< pointer to buffer the new partialdec created by dint of the graph */
      PARTIALDECOMP**      secondpartialdec,        /**< pointer to buffer the new partialdec whose border is amplified by dint of the graph */
      DETPROBDATA*         detprobdata              /**< detection process information and data */
      );

   virtual SCIP_RETCODE createFromMatrix(
      SCIP_CONS**           conss,              /**< constraints for which graph should be created */
      SCIP_VAR**            vars,               /**< variables for which graph should be created */
      int                   nconss_,            /**< number of constraints */
      int                   nvars_              /**< number of variables */
      );

   /** creates a graph with open constraints and open variables of the partialdec */
   virtual SCIP_RETCODE createFromPartialMatrix(
      DETPROBDATA*          detprobdata,     /**< detection process information and data */
      PARTIALDECOMP*        partialdec       /**< partial decomposition to use for matrix */
      );

};

} /* namespace gcg */
#endif /* GCG_HYPERROWGRAPH_H_ */
