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

/**@file   graph_tclique.cpp
 * @brief  interface to the SCIP tclique graph library
 * @author Annika Thome
 * @author Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/*lint -e39*/
#include <cassert>
#include "graph_tclique.h"

#define TCLIQUE_CALL_EXC(x)   do                                                                              \
                       {                                                                                      \
                          SCIP_Bool _restat_;                                                                 \
                          if( (_restat_ = (x)) != TRUE )                                                      \
                          {                                                                                   \
                             SCIPerrorMessage("Error <%d> in function call\n", _restat_);                     \
                             throw std::exception();                          \
                           }                                                                                  \
                       }                                                                                      \
                       while( FALSE )

#define TCLIQUE_CALL(x)   do                                                                                  \
                       {                                                                                      \
                          SCIP_Bool _restat_;                                                                 \
                          if( (_restat_ = (x)) != TRUE )                                                      \
                          {                                                                                   \
                             SCIPerrorMessage("Error <%d> in function call\n", _restat_);                     \
                             return SCIP_ERROR;                                                               \
                           }                                                                                  \
                       }                                                                                      \
                       while( FALSE )


namespace gcg {

GraphTclique::GraphTclique()
{
   TCLIQUE_CALL_EXC( tcliqueCreate(&graph) );
}

SCIP_RETCODE GraphTclique::addNNodes(int _n_nodes)
{
   return SCIP_INVALIDCALL;
}

SCIP_RETCODE GraphTclique::addNNodes(int _n_nodes, std::vector<int> weights)
{
   return SCIP_INVALIDCALL;
}


GraphTclique::~GraphTclique()
{
   tcliqueFree(&graph);
}

int GraphTclique::getNNodes()
{
   return tcliqueGetNNodes(graph);
}

int GraphTclique::getNEdges()
{
   return tcliqueGetNEdges(graph);
}

SCIP_RETCODE GraphTclique::getEdges(std::vector<void*>& edges)
{
   return SCIP_INVALIDCALL;
}


SCIP_Bool GraphTclique::isEdge(int i, int j)
{
   assert(i >= 0);
   assert(j >= 0);

   return tcliqueIsEdge(graph, i, j);
}

int GraphTclique::getNNeighbors(int i)
{
   assert( i >= 0);
   return int( tcliqueGetLastAdjedge(graph,i)-tcliqueGetFirstAdjedge(graph, i)+1 );
}

std::vector<int> GraphTclique::getNeighbors(int i)
{
   assert(i >= 0);
   std::vector<int> part(tcliqueGetFirstAdjedge(graph, i), tcliqueGetLastAdjedge(graph,i)+1);
   return part;
}

SCIP_RETCODE GraphTclique::addNode(int i, int weight)
{
   assert(i >= getNNodes());
   TCLIQUE_CALL( tcliqueAddNode(graph,i,weight) );
   return SCIP_OKAY;
}

SCIP_RETCODE GraphTclique::addNode()
{
   return SCIP_INVALIDCALL;
}

SCIP_RETCODE GraphTclique::deleteNode(int i)
{ /*lint -e715*/
   return SCIP_ERROR;
}

SCIP_RETCODE GraphTclique::addEdge(int i, int j)
{
   assert(i >=0);
   assert(i < getNNodes());
   assert(j >=0);
   assert(j < getNNodes());

   TCLIQUE_CALL( tcliqueAddEdge(graph,i,j) );
   return SCIP_OKAY;

}

SCIP_RETCODE GraphTclique::addEdge(int i, int j, double weight)
{
   return SCIP_INVALIDCALL;
}

SCIP_RETCODE GraphTclique::setEdge(int i, int j, double weight)
{
   return SCIP_INVALIDCALL;
}

double GraphTclique::getEdgeWeight(int i, int j)
{
   return 0.0;
}

std::vector<std::pair<int, double> > GraphTclique::getNeighborWeights(int i)
{
   return std::vector<std::pair<int, double> >();
}

SCIP_RETCODE GraphTclique::deleteEdge(int i, int j)
{ /*lint -e715*/
   return SCIP_ERROR;
}


SCIP_RETCODE GraphTclique::flush()
{
   TCLIQUE_CALL( tcliqueFlush(graph) );

   return SCIP_OKAY;
}

int GraphTclique::graphGetWeights(int i)
{
   assert( i >= 0);
   assert( i <= getNNodes());
   const TCLIQUE_WEIGHT* weights;
   weights = tcliqueGetWeights(graph);
   return weights[i];
}

SCIP_RETCODE GraphTclique::normalize(){
   // this function is used only in GraphGCG
   return SCIP_INVALIDCALL;
}

double GraphTclique::getEdgeWeightPercentile(double q)
{
   return 0.0;
}


} /* namespace gcg */
