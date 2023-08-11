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

/**@file   graph_gcg.h
 * @brief  Implementation of the graph which supports both node and edge weights.
 * @author Igor Pesic
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_GRAPH_GCG_H_
#define GCG_GRAPH_GCG_H_

#include <map>
#include "bridge.h"

#ifdef WITH_GSL
   #include <gsl/gsl_spmatrix.h>
   #include <gsl/gsl_spblas.h>
#endif

namespace gcg {

class EdgeGCG
{
public:
    int src, dest;
    double weight;
    EdgeGCG(): src(-1), dest(-1), weight(0.0) {}
    EdgeGCG(int s, int d, double w): src(s), dest(d), weight(w) {}
} ;

class GraphGCG: public gcg::Bridge
{
private:
   bool undirected;
   bool locked;           // true if we are not allowed to change the graph anymore
   bool initialized;      // true if at least 1 node
   std::vector<int> nodes;
#ifdef WITH_GSL
   gsl_spmatrix* adj_matrix_sparse;
   gsl_spmatrix* working_adj_matrix;      // this one is used ONLY during MCL algorithm!
#else
   std::vector<std::vector<double>> adj_matrix;      /** For undirected graphs, this matrix is symmetrical */
#endif
   std::vector<EdgeGCG*> edges;

public:

   GraphGCG();
   GraphGCG(int _n_nodes, bool _undirected);

   virtual ~GraphGCG();
   virtual SCIP_RETCODE addNNodes(int _n_nodes);
   virtual SCIP_RETCODE addNNodes(int _n_nodes, std::vector<int> weights);
   virtual int getNNodes();
   virtual int getNEdges();

#ifdef WITH_GSL
   virtual gsl_spmatrix* getAdjMatrix();
   virtual void expand(int factor);
   virtual void inflate(double factor);
   virtual void colL1Norm();
   virtual void prune();
   virtual bool stopMCL(int iter);
   virtual std::vector<int> getClustersMCL();
   virtual void initMCL();
   virtual void clearMCL();
#else
   virtual std::vector<std::vector<double>> getAdjMatrix();
#endif
   virtual SCIP_RETCODE getEdges(std::vector<void*>& edges);
   virtual SCIP_Bool isEdge(int node_i, int node_j);
   virtual int getNNeighbors(int node);
   virtual std::vector<int> getNeighbors(int node);
   virtual std::vector<std::pair<int, double> > getNeighborWeights(int node);
   virtual SCIP_RETCODE addNode(int node, int weight);
   virtual SCIP_RETCODE addNode();                    /** Sets node weight to 0 and the ID to the next available. */
   virtual SCIP_RETCODE deleteNode(int node);
   virtual SCIP_RETCODE addEdge(int i, int j);        /** Sets edge weight to 1. */
   virtual SCIP_RETCODE addEdge(int node_i, int node_j, double weight);
   virtual SCIP_RETCODE setEdge(int node_i, int node_j, double weight);
   virtual SCIP_RETCODE deleteEdge(int node_i, int node_j);
   virtual int graphGetWeights(int node);
   virtual double getEdgeWeight(int node_i, int node_j);

   virtual int edgeComp(const EdgeGCG* a, const EdgeGCG* b);

   virtual SCIP_RETCODE flush();       // lock the graph and compresses the adj matrix if we use GSL
   virtual SCIP_RETCODE normalize();
   virtual double getEdgeWeightPercentile(double q);
};

} /* namespace gcg */
#endif /* GCG_GRAPH_GCG_H_ */
