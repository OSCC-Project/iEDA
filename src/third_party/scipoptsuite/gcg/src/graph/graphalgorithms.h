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

/**@file    graphalgorithms.h
 * @brief   several metrics for graphs
 * @author  Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GRAPHALGORITHMS_H_
#define GRAPHALGORITHMS_H_

#include "hypergraph.h"
#include "graph.h"
#include "graph_gcg.h"

namespace gcg {

// A structure to represent a subset for union-find
typedef struct subset
{
    int parent;
    int rank;
} subset;


template<class T>
class GraphAlgorithms {

public:
   /** compute weighted sum of external degrees */
   static SCIP_Real computeSoed(
      Hypergraph<T>&     graph               /**< the hypergraph */
   );

   /** compute minimum hyperedge cut */
   static SCIP_Real computeMincut(
      Hypergraph<T>&     graph               /**< the hypergraph */
   );

   /** compute k-1 metric */
   static SCIP_Real computekMetric(
      Hypergraph<T>&     graph               /**< the hypergraph */
   );

   /** run DBSCAN on the distance graph */
   static std::vector<int> dbscan(
      Graph<GraphGCG>& graph,          /**< the graph with weighted edges */
      double eps,               /**< radius in which we search for the neighbors */
      int minPts = 4            /**< minimum number of neighbors needed to define a core point (can be fixed to 4 as stated in the paper) */
   );


   /** run MST on the distance graph */
   static std::vector<int> mst(
      Graph<GraphGCG>& graph,          /**< the graph with weighted edges */
      double cutoff,            /**< threshold below which we cut the edges */
      int minPts = 4            /**< minimum number of points needed in a cluster */
   );


   /** run MCL on the similarity graph */
   static std::vector<int> mcl(
      Graph<GraphGCG>& graph,       /**< the graph with weighted edges */
      int& stoppedAfter,            /**< number of iterations after which the clustering terminated */
      double inflatefac,            /**< inflate factor */
      int maxiters = 25,            /**< max number of iterations, set to 25 per default */
      int expandfac = 2             /**< expand factor, should be always set to 2 */
   );




//private:

   /** help function for DBSCAN */
   static void expandCluster(
      Graph<T>& graph,
      std::vector<bool>& visited,
      std::vector<bool>& is_core,
      std::vector<int>& labels,
      int point,
      std::vector<int>& NeighborPts,
      int curr_cluster,
      double eps,
      int minPts
   );

   static double cutoff;

   // Returns true if the weight of the edge is bigger than the this->cutoff
   static bool cutoffif(
      EdgeGCG &a
   );

   // Compare two edges according to their weights.
   // Used in sort() for sorting an array of edges
   static int weightComp(
      const void* a,
      const void* b
   );

   // A utility function to find set of an element i
   // (uses path compression technique)
   static int mstfind(
      std::vector<subset>& subsets,
      int i
   );

   // A function that does union of two sets of x and y
   // (uses union by rank)
   static void mstunion(
      std::vector<subset>& subsets,
      int x,
      int y
   );

};

} /* namespace gcg */



#endif /* GRAPHALGORITHMS_H_ */
