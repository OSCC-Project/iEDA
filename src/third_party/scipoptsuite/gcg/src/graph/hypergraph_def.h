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

/**@file   hypergraph_def.h
 * @brief  miscellaneous hypergraph methods for structure detection
 * @author Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/


#ifndef GCG_HYPERGRAPH_DEF_H_
#define GCG_HYPERGRAPH_DEF_H_

#include "scip/scip.h"
#include "hypergraph.h"

namespace gcg {

template <class T>
Hypergraph<T>::Hypergraph(
   SCIP*                 scip               /**< SCIP data structure */
) : name("hypergraph"),scip_(scip),graph(NULL),lastnode(0),dummynodes(0)
{
   SCIPdebugMessage("Creating graph\n");
   graph = new Graph<T>(scip);
}

template <class T>
Hypergraph<T>::~Hypergraph()
{
   if(graph != NULL)
      delete graph;

}

template <class T>
int Hypergraph<T>::computeNodeId(int i)
{
   int nodeid;
   if( i < (int) nodes.size())
      nodeid = nodes[i];
   else
      nodeid = lastnode;

   SCIPdebugMessage("Nodeid %d is %d\n", i, nodeid);
   return nodeid;
}

template <class T>
SCIP_RETCODE  Hypergraph<T>::addNode(int i,int weight)
{
   int nodeid = lastnode;
   SCIPdebugMessage("Adding node %d (id=%d)\n", i, nodeid);
   SCIP_CALL( graph->addNode(nodeid, weight) );
   nodes.push_back(nodeid);
   mapping.resize(nodeid+1);
   mapping[nodeid] = i;
   ++lastnode;
   return SCIP_OKAY;
}

/** adds the edge to the graph */
template <class T>
SCIP_RETCODE Hypergraph<T>::addHyperedge(std::vector<int> &edge, int weight)
{
   int edgenodeid = lastnode;
   ++lastnode;
   SCIPdebugMessage("Adding hyperedge %lu (id=%d)\n", hedges.size(), edgenodeid);
   SCIP_CALL( graph->addNode(edgenodeid, weight) );

   for( size_t i = 0; i < edge.size(); ++i )
   {
      SCIP_CALL( graph->addEdge(edgenodeid, computeNodeId(edge[i])) );
   }
   hedges.push_back(edgenodeid);
   mapping.resize(edgenodeid+1);
   mapping[edgenodeid] = hedges.size()-1;
   return SCIP_OKAY;
}

/** adds the edge to the graph */
template <class T>
SCIP_RETCODE Hypergraph<T>::addNodeToHyperedge(int node, int hedge)
{
   int edgenodeid = hedges[hedge];
   int nodeid = nodes[node];
   SCIP_CALL( graph->addEdge(edgenodeid, nodeid) );

   return SCIP_OKAY;
}


template <class T>
int Hypergraph<T>::getNNodes() {
   return nodes.size();
}

template <class T>
int Hypergraph<T>::getNHyperedges() {
   return hedges.size();
}

template <class T>
int Hypergraph<T>::getNNeighbors(int i) {
   assert( i >= 0);
   return graph->getNNeighbors(i);
}

template <class T>
std::vector<int> Hypergraph<T>::getNeighbors(int i) {
   assert(i >= 0);
   int nodeid = computeNodeId(i);
   std::vector<int> edges = graph->getNeighbors(nodeid);

   std::set<int> neighbors;
   for( size_t j = 0; j < edges.size(); ++j )
   {
      std::vector<int> tempneighbors = graph->getNeighbors(edges[j]);
      neighbors.insert(tempneighbors.begin(), tempneighbors.end());
   }
   std::vector<int> r(neighbors.begin(), neighbors.end());
   for( size_t j = 0; j < r.size(); ++j)
   {
      r[j] = mapping[r[j]];
   }
   std::vector<int>::iterator it = std::remove(r.begin(), r.end(), nodeid);

   return std::vector<int>(r.begin(), it);
}

template <class T>
std::vector<int> Hypergraph<T>::getHyperedgeNodes(
   int i
   )
{
   std::vector<int> hnodes =  graph->getNeighbors(hedges[i]);
   for( size_t j = 0; j < hnodes.size(); ++j)
   {
      hnodes[j] = computeNodeId(hnodes[j]);
   }
   return hnodes;
}

template <class T>
int Hypergraph<T>::getNHyperedgeNodes(
   int i
   )
{
   return graph->getNNeighbors(hedges[i]);
}

template <class T>
void Hypergraph<T>::setPartition(int i, int ID) {
   partition.resize(getNNodes(), -1);
   partition[i] = ID;
}

/** write the hypergraph to a file */
template <class T>
SCIP_RETCODE Hypergraph<T>::writeToFile(
      int                fd,                    /**< filename where the graph should be written to */
      SCIP_Bool writeweights
    )
{
   FILE* file;
   file = fdopen(fd, "w");
   if( file == NULL )
      return SCIP_FILECREATEERROR;

   SCIPinfoMessage(scip_, file, "%ld %ld\n", nodes.size()+dummynodes, hedges.size());

   for( size_t i = 0; i < hedges.size(); ++i )
   {
      int nneighbors = graph->getNNeighbors(hedges[i]);
      std::vector<int> neighbors = graph->getNeighbors(hedges[i]);

      if( writeweights )
      {
         SCIPinfoMessage(scip_, file, "%d ", graph->getWeight(i));
      }
      for( int j = 0; j < nneighbors; ++j )
      {
         SCIPinfoMessage(scip_, file, "%d ", computeNodeId(neighbors[j])+1);
      }
      SCIPinfoMessage(scip_, file, "\n");
   }

   return SCIP_OKAY;
}

/** read in the partition from a file */
template <class T>
SCIP_RETCODE Hypergraph<T>::readPartition(
   const char* filename
)
{
   ifstream input(filename);
   if( !input.good() )
   {
      SCIPerrorMessage("Could not open file <%s> for reading\n", filename);
      return SCIP_READERROR;
   }
   partition.resize(getNNodes(), -1);
   for( int i = 0; i < getNNodes(); ++i )
   {
      int part = 0;
      if( !(input >> part) )
      {
         SCIPerrorMessage("Could not read from file <%s>. It may be in the wrong format\n", filename);
         return SCIP_READERROR;
      }
      partition[i] = part;
   }

   input.close();
   return SCIP_OKAY;
}

/** return the weight of given node */
template <class T>
int Hypergraph<T>::getWeight(
   int                i                   /**< the given node */
   )
{
   return graph->getWeight(i);
}

/** return the weight of given hyperedge */
template <class T>
int Hypergraph<T>::getHyperedgeWeight(
   int                i                   /**< the given hyperedge */
   )
{
   int edgenodeid = hedges[i];

   return graph->getWeight(edgenodeid);
}

template <class T>
SCIP_RETCODE Hypergraph<T>::flush()
{
   SCIP_CALL( graph->flush() );
   return SCIP_OKAY;
}

} /* namespace gcg */

#endif
