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

/**@file   bridge.h
 * @brief  bridge
 * @author Annika Thome
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_BRIDGE_H
#define GCG_BRIDGE_H
#include "objscip/objscip.h"
#include <vector>


namespace gcg
{

class Bridge
{
public:
   Bridge() {}
   virtual ~Bridge() {}

   /** add n nodes in the graph at the same time. it is much faster than to call addNode() many times */
   virtual SCIP_RETCODE addNNodes(int _n_nodes) = 0;

   /** add n nodes in the graph at the same time. it is much faster than to call addNode() many times. weights represent node weights */
   virtual SCIP_RETCODE addNNodes(int _n_nodes, std::vector<int> weights) = 0;

   /** get number of nodes in the graph */
   virtual int getNNodes() = 0;

   /** get number of edges in the graph */
   virtual int getNEdges() = 0;

   /** get list of edges in the graph (not defined how edges are implemented) */
   virtual SCIP_RETCODE getEdges(std::vector<void*>& edges) = 0;

   /** return whether a given pair of vertices is connected by an edge */
   virtual SCIP_Bool isEdge(int i, int j) = 0;

   /** get number of neighbors of a given node */
   virtual int getNNeighbors(int i) = 0;

   /** get a vector of all neighbors of a given node */
   virtual std::vector<int> getNeighbors(int i) = 0;

   /** adds the node with the given weight to the graph */
   virtual SCIP_RETCODE addNode(int i,int weight) = 0;

   /** adds the node with 0 weight to the graph */
   virtual SCIP_RETCODE addNode() = 0;

   /** adds the weighted edge to the graph */
   virtual SCIP_RETCODE addEdge(int i, int j, double weight) = 0;

   /** sets the weight of the edge in the graph */
   virtual SCIP_RETCODE setEdge(int i, int j, double weight) = 0;

   /** returns the weight of the edge in the graph */
   virtual double getEdgeWeight(int i, int j) = 0;

   virtual std::vector<std::pair<int, double> > getNeighborWeights(int i) {return std::vector<std::pair<int, double> >();};

   /** deletes the given node from the graph */
   virtual SCIP_RETCODE deleteNode(int i) = 0;

   /** adds the edge to the graph */
   virtual SCIP_RETCODE addEdge(int i, int j) = 0;

   /** deletes the edge from the graph */
   virtual SCIP_RETCODE deleteEdge(int i, int j) = 0;

   /** return the weight of a node */
   virtual int graphGetWeights(int i) = 0;

   /** flushes the data stuctures, if needed */
   virtual SCIP_RETCODE flush() = 0;

   /** normalizes the edge weights, so that the biggest edge egiht in the graph is 1 */
   virtual SCIP_RETCODE normalize() = 0;

   virtual double getEdgeWeightPercentile(double q) = 0;

#ifdef WITH_GSL
   /** function needed for MST clustering */
   virtual void expand(int factor) = 0;

   /** function needed for MST clustering */
   virtual void inflate(double factor) = 0;

   /** function needed for MST clustering */
   virtual void colL1Norm() = 0;

   /** function needed for MST clustering */
   virtual void prune() = 0;

   /** function needed for MST clustering */
   virtual bool stopMCL(int iter) {return true;}

   virtual std::vector<int> getClustersMCL() {return std::vector<int>();}

   /** function needed for MST clustering */
   virtual void initMCL() = 0;

   /** function needed for MST clustering */
   virtual void clearMCL() = 0;

#endif

};


} /* namespace gcg*/

#endif
