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

/**@file    graphalgorithms_def.h
 * @brief   several metrics for graphs
 * @author  Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_GRAPHALGORITHMS_DEF_H_
#define GCG_GRAPHALGORITHMS_DEF_H_

#include "graph/graphalgorithms.h"
#include <vector>
#include <algorithm>
#include <map>
#include <iostream>
#include "stdlib.h"
#include "graph/graph_tclique.h"
#include "graph/graph_gcg.h"


using std::vector;


namespace gcg {

template <typename T>
double GraphAlgorithms<T>::cutoff = 0.0;


/** compute weighted sum of external degrees */
template<class T>
double GraphAlgorithms<T>::computeSoed(
      Hypergraph<T>&     graph               /**< the hypergraph */
)
{
   SCIP_Real soed = 0.0;
   size_t nedges = graph.getNHyperedges();
   vector<int> partition = vector<int>(graph.getPartition());

   for( size_t i = 0; i < nedges; ++i )
   {
      vector<int> nodes = graph.getHyperedgeNodes(i);
      for( auto &it : nodes)
      {
         it = partition[it];
      }
      auto end = std::unique(nodes.begin(), nodes.end());
      if( end - nodes.begin() > 1)
         soed += ( end - nodes.begin())*graph.getHyperedgeWeight(i);
   }
   return soed;
}

/** compute minimum hyperedge cut */
template<class T>
double GraphAlgorithms<T>::computeMincut(
      Hypergraph<T>&     graph               /**< the hypergraph */
)
{
   SCIP_Real mincut = 0.0;
   size_t nedges = graph.getNHyperedges();
   vector<int> partition = vector<int>(graph.getPartition());

   for( size_t i = 0; i < nedges; ++i )
   {
      vector<int> nodes = graph.getHyperedgeNodes(i);
      for( auto &it : nodes)
      {
         it = partition[it];
      }
      auto end = std::unique(nodes.begin(), nodes.end());

      if( end - nodes.begin() > 1)
         mincut += graph.getHyperedgeWeight(i);
   }

   return mincut;
}

/** compute k-1 metric */
template<class T>
double GraphAlgorithms<T>::computekMetric(
      Hypergraph<T>&     graph               /**< the hypergraph */
)
{
   SCIP_Real kmetric = 0.0;
   size_t nedges = graph.getNHyperedges();
   vector<int> partition = vector<int>(graph.getPartition());

   for( size_t i = 0; i < nedges; ++i )
   {
      vector<int> nodes = graph.getHyperedgeNodes(i);
      for( auto &it : nodes)
      {
         it = partition[it];
      }
      auto end = std::unique(nodes.begin(), nodes.end());

      kmetric += ( end - nodes.begin() -1)*graph.getHyperedgeWeight(i);
   }

   return kmetric;
}


/** run DBSCAN on the distance graph */
template<class T>
std::vector<int> GraphAlgorithms<T>::dbscan(
   Graph<GraphGCG>& graph,          /**< the graph with weighted edges */
   double eps,               /**< radius in which we search for the neighbors */
   int minPts                /**< minimum number of neighbors needed to define a core point (can be fixed to 4 as stated in the paper) */
)
{

   int n_nodes = graph.getNNodes();
   std::vector<bool> visited(n_nodes, false);
   std::vector<bool> is_core(n_nodes, false);
   std::vector<int> labels(n_nodes, -1);
   int curr_cluster = -1;

   // visit all the points and check for the core points
   for(int i = 0; i < n_nodes; i++)
   {
      if(visited[i])
      {
         continue;
      }
      visited[i] = true;

      // check if i is a core point
      std::vector<std::pair<int, double> > NeighborPts_row = graph.getNeighborWeights(i);       // gets ALL connected points
      std::vector<int> NeighborPts;                                                             // holds ONLY the neighbors within the radius
      for(int j = 0; j < (int)NeighborPts_row.size(); j++)
      {
         if(NeighborPts_row[j].second <= eps)
         {
            NeighborPts.push_back(NeighborPts_row[j].first);
         }

      }
      if((int)NeighborPts.size() < minPts)
      {
         labels[i] = -1;   // mark point as noise
      }
      else {
         curr_cluster++;
         expandCluster(graph, visited, is_core, labels, i, NeighborPts, curr_cluster, eps, minPts);
      }
   }

   // Add border points also. We assign the border point the same label as the first core point. This is basically arbitrary, but not random.
   for(int i = 0; i < n_nodes; i++)
   {
      if(is_core[i])
      {
         continue;
      }
      std::vector<std::pair<int, double> > NeighborPts_row = graph.getNeighborWeights(i);      // gets ALL connected points                                                                // holds ONLY the neighbors within the radius
      for(int j = 0; j < (int)NeighborPts_row.size(); j++)
      {
         if(NeighborPts_row[j].second <= eps && is_core[NeighborPts_row[j].first])
         {
            labels[i] = labels[NeighborPts_row[j].first];
            break;
         }
      }
   }

   return labels;
}

/** help function for DBSCAN */
template<class T>
void GraphAlgorithms<T>::expandCluster(
   Graph<T>& graph,
   std::vector<bool>& visited,
   std::vector<bool>& is_core,
   std::vector<int>& labels,
   int point,
   std::vector<int>& NeighborPts,
   int curr_cluster,
   double eps,
   int minPts
)
{
   // add P to cluster C
   labels[point] = curr_cluster;
   is_core[point] = true;

   // for each point P' in NeighborPts
   for(int i = 0; i < (int)NeighborPts.size();  i++)
   {
      int neighbor = NeighborPts[i];
      // if P' is not visited {
      if(!visited[neighbor])
      {
         // mark P' as visited
         visited[neighbor] = true;
         // NeighborPts' = regionQuery(P', eps)
         std::vector<std::pair<int, double> > NeighborPts_tmp_row = graph.getNeighborWeights(neighbor);     // gets ALL connected points
         std::vector<int> NeighborPts_tmp;                                                                         // holds ONLY the neighbors within the radius
         for(int j = 0; j < (int)NeighborPts_tmp_row.size(); j++)
         {
            if(NeighborPts_tmp_row[j].second <= eps)
            {
               NeighborPts_tmp.push_back(NeighborPts_tmp_row[j].first);
            }

         }
         if((int)NeighborPts_tmp.size() >= minPts)
         {
            // NeighborPts = NeighborPts joined with NeighborPts'
            NeighborPts.insert(NeighborPts.end(), NeighborPts_tmp.begin(), NeighborPts_tmp.end());
         }

      }

      // if P' is not yet member of any cluster
      //    add P' to cluster C
      if(labels[neighbor] < 0)
      {
         labels[neighbor] = curr_cluster;
         is_core[neighbor] = true;
      }
   }
}

/*
 * template<class T>
 * std::vector<int> GraphAlgorithms<T>::mst(
 */
template<class T>
std::vector<int> GraphAlgorithms<T>::mst(
   Graph<GraphGCG>& graph,          /**< the graph with weighted edges */
   double _cutoff,            /**< threshold below which we cut the edges */
   int minPts                /**< minimum number of points needed in a cluster */
)
{
   cutoff = _cutoff;
   // Step 1: find a minimum spanning tree using Kruskal's algorithm

   int nnodes = graph.getNNodes();
   std::vector<void*> edges(graph.getNEdges());
   graph.getEdges(edges);

   vector<EdgeGCG> resultMST(nnodes-1);
   int e = 0;  // An index variable, used for resultMST
   unsigned int j = 0;  // An index variable, used for sorted edges

   // Step 1:  Sort all the edges in non-decreasing order of their weight
   // If we are not allowed to change the given graph, we can create a copy of
   // array of edges
   sort(edges.begin(), edges.end(), weightComp);

   // Create V subsets (one for each node)
   vector<subset> subsetsmst(nnodes);

   // Create V subsets with single elements
   for (int v = 0; v < nnodes; ++v)
   {
     subsetsmst[v].parent = v;
     subsetsmst[v].rank = 0;
   }

   // Number of edges to be taken is equal to V-1
   while (e < nnodes - 1)
   {
      // Step 2: Pick the smallest edge. And increment the index
      // for next iteration
      if(j == edges.size()) break;
      EdgeGCG next_edge = *(EdgeGCG *)(edges[j++]);
      assert(next_edge.src < graph.getNNodes());
      assert(next_edge.dest < graph.getNNodes());

      int x = mstfind(subsetsmst, next_edge.src);     // O(tree height)
      int y = mstfind(subsetsmst, next_edge.dest);

      // If including this edge does't cause cycle, include it
      // in result and increment the index of result for next edge
      if (x != y)
      {
         resultMST[e++] = next_edge;
         mstunion(subsetsmst, x, y);
      }
     // Else discard the next_edge
   }

   resultMST.resize(e);

   // Step 2: remove all the edges from the MST that are greater or equal to cutoff

   //std::vector<EdgeGCG> resultDBG = resultMST;
   resultMST.erase(std::remove_if(resultMST.begin(), resultMST.end(), cutoffif), resultMST.end());
   // Step 3: use find-union to find the graph components

   // saves the component labels (i.e. cluster labels)
   std::vector<int> labels(nnodes, -1);

   // Create V subsets (one for each node)
   std::vector<subset> subsetscomp(nnodes);

   // Create V subsets with single elements
   for (int v = 0; v < nnodes; ++v)
   {
     subsetscomp[v].parent = v;
     subsetscomp[v].rank = 0;
   }


   // iterate all the edges and assign its nodes to the root node of the set
   for(unsigned int edge_it = 0; edge_it < resultMST.size(); edge_it++)
   {
      auto edge = resultMST[edge_it];
      /*if(!(edge.src < graph.getNNodes()) || !(edge.dest < graph.getNNodes())){
         cout << "DEBUG ME!!!" << endl;
      }*/
      assert(edge.src < graph.getNNodes());
      assert(edge.dest < graph.getNNodes());
      // if the nodes are directly connected, put them in the same set
      mstunion(subsetscomp, edge.src, edge.dest);
   }

   for(int i = 0; i < nnodes; i++)
   {
      labels[i] = mstfind(subsetscomp, i);
   }


   // remove the clusters that are smaller than minPts
   std::map<int,int> labelcount;
   for(int i = 0; i < nnodes; i++)
   {
      labelcount[labels[i]]++;
   }

   for(int i = 0; i < nnodes; i++)
   {
      if(labelcount[labels[i]] < minPts)
         labels[i] = -1;
   }


   // reassign the labels so that they start from 0 (actually this is done in postProcess, so we can skip this part here)


   return labels;
}


template<class T>
bool GraphAlgorithms<T>::cutoffif(EdgeGCG &a)
{
   return a.weight > cutoff;
}


// Compare two edges according to their weights.
// Used in sort() for sorting an array of edges
template<class T>
int GraphAlgorithms<T>::weightComp(const void* a, const void* b)
{
   const EdgeGCG* a1 = static_cast<const EdgeGCG*>(a);
   const EdgeGCG* b1 = static_cast<const EdgeGCG*>(b);
   return a1->weight < b1->weight;
}


// A utility function to find set of an element i
// (uses path compression technique)
template<class T>
int GraphAlgorithms<T>::mstfind(std::vector<subset>& subsets, int i)
{
    // find root and make root as parent of i (path compression)
    if (subsets[i].parent != i)
        subsets[i].parent = mstfind(subsets, subsets[i].parent);

    return subsets[i].parent;
}


// A function that does union of two sets of x and y
// (uses union by rank)
template<class T>
void GraphAlgorithms<T>::mstunion(std::vector<subset>& subsets, int x, int y)
{
    int xroot = mstfind(subsets, x);
    int yroot = mstfind(subsets, y);

    // Attach smaller rank tree under root of high rank tree
    // (Union by Rank)
    if (subsets[xroot].rank < subsets[yroot].rank)
        subsets[xroot].parent = yroot;
    else if (subsets[xroot].rank > subsets[yroot].rank)
        subsets[yroot].parent = xroot;

    // If ranks are same, then make one as root and increment
    // its rank by one
    else
    {
        subsets[yroot].parent = xroot;
        subsets[xroot].rank++;
    }
}


template<class T>
std::vector<int> GraphAlgorithms<T>::mcl(
   Graph<GraphGCG>& graph,   /**< the graph with weighted edges */
   int& stoppedAfter,        /**< number of iterations after which the clustering terminated */
   double inflatefac,        /**< inflate factor */
   int maxiters,             /**< max number of iterations, set to 25 per default */
   int expandfac             /**< expand factor, should be always set to 2 */
)
{
#ifdef WITH_GSL
   graph.initMCL();
   graph.colL1Norm();
   graph.prune();

   int i = 0;
   for(; i < maxiters; i++)
   {
      graph.inflate(inflatefac);
      graph.expand(expandfac);
      graph.prune();

      if(graph.stopMCL(i))
      {
         break;
      }
   }
   assert(i > 6);
   stoppedAfter = i;

   std::vector<int> res = graph.getClustersMCL();
   graph.clearMCL();
   return res;
#else
   return vector<int>();
#endif
}




} /* namespace gcg */
#endif
