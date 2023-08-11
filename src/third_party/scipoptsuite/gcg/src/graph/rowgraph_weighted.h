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

/**@file   rowgraph_weighted.h
 * @brief  A row graph where each row is a node and rows are adjacent if they share a variable.
 *         The edges are weighted according to specified similarity measure.
 * @author Igor Pesic
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_ROWGRAPH_WEIGHTED_H_
#define GCG_ROWGRAPH_WEIGHTED_H_

#include "graph.h"
#include "rowgraph.h"

namespace gcg {

enum DistanceMeasure
{
   JOHNSON           = 1,
   INTERSECTION      = 2,
   JACCARD           = 5,
   COSINE            = 6,
   SIMPSON           = 9,
};
typedef enum DistanceMeasure DISTANCE_MEASURE;

enum WeightType
{
   SIM       = 0,
   DIST      = 1
};
typedef enum WeightType WEIGHT_TYPE;


template <class T>
class RowGraphWeighted : public gcg::RowGraph<T>
{
private:
   int n_blocks;
   int non_cl;

   /** removes the colliding rows from the clusters (should be used just after clustering)
    *  Method: Assign labels to columns and remove conss where column label doesn't match the conss label.
    */
   virtual SCIP_RETCODE postProcess(std::vector<int>& labels, bool enabled);

   virtual SCIP_RETCODE postProcessForPartialGraph(gcg::DETPROBDATA* detprobdata, gcg::PARTIALDECOMP* partialdec, std::vector<int>& labels, bool enabled);

   /** removes the colliding rows from the clusters (should be used just after clustering)
    *  Method: solve the stable set problem with greedy heuristic
    *  NOTE: this function is obsolete because the new version of postProcess has the same results, and is faster
    */
   virtual SCIP_RETCODE postProcessStableSet(std::vector<int>& labels, bool enabled);

   virtual SCIP_RETCODE postProcessStableSetForPartialGraph(gcg::DETPROBDATA* detprobdata, gcg::PARTIALDECOMP* partialdec, std::vector<int>& labels, bool enabled);

public:
   RowGraphWeighted(
         SCIP*                 scip,              /**< SCIP data structure */
         Weights               w                  /**< weights for the given graph */
      );
   virtual ~RowGraphWeighted();

   // we need this to avoid the warning of hidding the parent function
   using gcg::RowGraph<T>::createFromMatrix;

   virtual SCIP_RETCODE createFromMatrix(
      SCIP_CONS**           conss,              /**< constraints for which graph should be created */
      SCIP_VAR**            vars,               /**< variables for which graph should be created */
      int                   nconss_,            /**< number of constraints */
      int                   nvars_,             /**< number of variables */
      DISTANCE_MEASURE      dist,               /**< Here we define the distance measure between two rows */
      WEIGHT_TYPE           w_type              /**< Depending on the algorithm we can build distance or similarity graph */
      );

   // we need this to avoid the warning of hidding the parent function
   using gcg::RowGraph<T>::createFromPartialMatrix;

   /** creates a graph with open constraints and open variables of the partialdec */
   virtual SCIP_RETCODE createFromPartialMatrix(
      DETPROBDATA*          detprobdata,        /**< detection process information and data */
      PARTIALDECOMP*        partialdec,         /**< partial decomposition to use for matrix */
      DISTANCE_MEASURE      dist,               /**< Here we define the distance measure between two rows */
      WEIGHT_TYPE           w_type              /**< Depending on the algorithm we can build distance or similarity graph */
      );

   static double calculateSimilarity(
      int a,               /**< number of common variables in two rows */
      int b,               /**< number of variables that appear only in the 2nd row */
      int c,               /**< number of variables that appear only in the 1st row */
      DISTANCE_MEASURE dist,   /**< Here we define the distance measure between two rows */
      WEIGHT_TYPE w_type,      /**< Depending on the algorithm we can build distance or similarity graph */
      bool itself          /**< true if we calculate similarity between the row itself */
      );

   /** return a partition of the nodes with the help of DBSCAN */
   virtual SCIP_RETCODE computePartitionDBSCAN(double eps, bool postprocenable);

   /** return a partition of the nodes with the help of DBSCAN */
   virtual SCIP_RETCODE computePartitionDBSCANForPartialGraph(gcg::DETPROBDATA* detprobdata, gcg::PARTIALDECOMP* partialdec, double eps, bool postprocenable);

   /** return a partition of the nodes with the help of MST */
   virtual SCIP_RETCODE computePartitionMST(double eps, bool postprocenable);

   /** return a partition of the nodes with the help of MST */
   virtual SCIP_RETCODE computePartitionMSTForPartialGraph(gcg::DETPROBDATA* detprobdata, gcg::PARTIALDECOMP* partialdec, double eps, bool postprocenable);

   /** return a partition of the nodes with the help of MST */
   virtual SCIP_RETCODE computePartitionMCL(int& stoppedAfter, double inflatefactor, bool postprocenable);

   /** return a partition of the nodes with the help of MST */
   virtual SCIP_RETCODE computePartitionMCLForPartialGraph(gcg::DETPROBDATA* detprobdata, gcg::PARTIALDECOMP* partialdec, int& stoppedAfter, double inflatefactor, bool postprocenable);

   /** return the number of clusters after post-processing */
   virtual SCIP_RETCODE getNBlocks(int& _n_blocks);

   virtual SCIP_RETCODE nonClustered(int& _non_cl);

   virtual double getEdgeWeightPercentile(double q);


};

} /* namespace gcg */
#endif /* GCG_ROWGRAPH_WEIGHTED_H_ */
