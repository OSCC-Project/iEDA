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

/**@file   dec_staircase_lsp.cpp
 * @ingroup DETECTORS
 * @brief  detector for staircase_lsp matrices
 * @author Martin Bergner
 * @author Michael Bastubbe
 *
 * This detector detects staircase_lsp structures in the constraint matrix by searching for the longest shortest path
 * in the row graph of the matrix.
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>
#include <iostream>


#include "dec_staircase_lsp.h"
#include "cons_decomp.h"
#include "scip_misc.h"
#include "pub_decomp.h"
#include "tclique/tclique.h"
#include "class_partialdecomp.h"
#include "class_detprobdata.h"
#include "scip/clock.h"


/* constraint handler properties */
#define DEC_DETECTORNAME          "staircase_lsp"    /**< name of detector */
#define DEC_DESC                  "Staircase detection via shortest paths" /**< description of detector */
#define DEC_PRIORITY              200            /**< priority of the detector */
#define DEC_FREQCALLROUND         1           /**< frequency the detector gets called in detection loop ,ie it is called in round r if and only if minCallRound <= r <= maxCallRound AND  (r - minCallRound) mod freqCallRound == 0 */
#define DEC_MAXCALLROUND          INT_MAX     /**< last round the detector gets called                              */
#define DEC_MINCALLROUND          0           /**< first round the detector gets called                              */
#define DEC_FREQCALLROUNDORIGINAL 1           /**< frequency the detector gets called in detection loop while detecting the original problem   */
#define DEC_MAXCALLROUNDORIGINAL  INT_MAX     /**< last round the detector gets called while detecting the original problem                            */
#define DEC_MINCALLROUNDORIGINAL  0           /**< first round the detector gets called while detecting the original problem    */
#define DEC_DECCHAR               'S'            /**< display character of detector */
#define DEC_ENABLED               FALSE           /**< should the detection be enabled */
#define DEC_ENABLEDFINISHING      FALSE       /**< should the finishing be enabled */
#define DEC_ENABLEDPOSTPROCESSING FALSE          /**< should the postprocessing be enabled */
#define DEC_SKIP                  FALSE          /**< should detector be skipped if others found detections */
#define DEC_USEFULRECALL          FALSE       /**< is it useful to call this detector on a descendant of the propagated partialdec */

#define TCLIQUE_CALL(x) do                                                                                    \
                       {                                                                                      \
                          if((x) != TRUE )                                                      \
                          {                                                                                   \
                             SCIPerrorMessage("Error in function call\n");                                    \
                             return SCIP_ERROR;                                                               \
                           }                                                                                  \
                       }                                                                                      \
                       while( FALSE )

/*
 * Data structures
 */

/** constraint handler data */
struct DEC_DetectorData
{
   SCIP_HASHMAP*  constoblock;
   SCIP_HASHMAP*  vartoblock;
   TCLIQUE_GRAPH* graph;
   int*           components;
   int            ncomponents;
   int            nblocks;
   std::vector<int>* oldToNew;
   std::vector<int>* newToOld;
};


/*
 * Local methods
 */

/* put your local methods here, and declare them static */


/** creates the graph from the constraint matrix */
static
SCIP_RETCODE createGraphFromPartialMatrix(
   SCIP*                 scip,               /**< SCIP data structure */
   TCLIQUE_GRAPH**       graph,              /**< Graph data structure */
   gcg::PARTIALDECOMP*   partialdec,         /**< partial decomposition to use for matrix */
   gcg::DETPROBDATA*     detprobdata,        /**< detection process information and data */
   DEC_DETECTORDATA*     detectordata        /**< detector data data structure */
   )
{
   int i;
   int v;

   assert(scip != NULL);
   assert(graph != NULL);

   TCLIQUE_CALL( tcliqueCreate(graph) );
   assert(*graph != NULL);

   if(detectordata->newToOld != NULL)
      delete detectordata->newToOld;
   if(detectordata->oldToNew != NULL)
         delete detectordata->oldToNew;

   detectordata->oldToNew = new std::vector<int>(detprobdata->getNConss(), -1) ;
   detectordata->newToOld = new std::vector<int>(partialdec->getNOpenconss(), -1) ;

   for( i = 0; i < partialdec->getNOpenconss(); ++i )
   {
      int cons = partialdec->getOpenconss()[i];
      detectordata->oldToNew->at(cons) = i;
      detectordata->newToOld->at(i) = cons;
      TCLIQUE_CALL( tcliqueAddNode(*graph, i, 0) );
   }

   std::vector<bool> alreadyConsidered(detprobdata->getNConss(), false );


   for( i = 0; i < partialdec->getNOpenconss(); ++i )
   {
      int cons = partialdec->getOpenconss()[i];
      std::vector<int> neighborNodes(0);
      std::vector<bool> isNeighbor(detprobdata->getNConss(), false);

      alreadyConsidered[cons] = true;

      for( v = 0; v < detprobdata->getNVarsForCons(cons ) ; ++v )
      {
         int var = detprobdata->getVarsForCons(cons)[v];

         if(!partialdec->isVarOpenvar(var) )
            continue;

         for( int c = 0; c < detprobdata->getNConssForVar(var); ++c )
         {
            int otherCons = detprobdata->getConssForVar(var)[c];
            if( isNeighbor[otherCons] || alreadyConsidered[otherCons] || !partialdec->isConsOpencons(otherCons) )
               continue;
            isNeighbor[otherCons] = true;

            TCLIQUE_CALL( tcliqueAddEdge(*graph, detectordata->oldToNew->at(cons), detectordata->oldToNew->at(otherCons)) );
         }
      }
   }

   TCLIQUE_CALL( tcliqueFlush(*graph) );
   /*SCIPdebug(tcliquePrintGraph(*graph));*/

   return SCIP_OKAY;
}




/** finds the diameter of a connected component of a graph and computes all distances from some vertex of maximum eccentricity to all other vertices */
static
SCIP_RETCODE findDiameter(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DETECTORDATA*     detectordata,       /**< constraint handler data structure */
   int*                  maxdistance,        /**< diameter of the graph */
   int*                  ncomp,              /**< number of vertices the component contains */
   int*                  vertices,           /**< vertices of the connected component (ordered by BFS) */
   int*                  distances,          /**< distances of vertices to some vertex of maximum eccentricity */
   int                   component           /**< connected component to investigate */
)
{
   TCLIQUE_GRAPH* graph;
   int diameter = -1;
   int* queue = NULL;   /* queue, first entry is queue[squeue], last entry is queue[equeue] */
   int squeue;          /* index of first entry of the queue */
   int equeue;          /* index of last entry of the queue */
   int nnodes;          /* number of vertices the graph contains */
   int ncompnodes = 0;  /* number of vertices the component contains */
   SCIP_Bool* marked = NULL;
   int* eccentricity = NULL;   /* upper bounds on the eccentricities */
   int* dist = NULL;           /* distances from some start vertex to all other vertices (gets updated in each BFS) */
   int* origdegree = NULL;     /* degrees of the vertices */
   int* degree = NULL;         /* degrees of the vertices sorted in decreasing order */
   int* degreepos = NULL;
   int i;
   int j;
   int k = 50;          /* number of low-degree vertices that are visited before high-degree vertices are visited */

   /* This method computes the diameter of a connected component by starting BFS at each vertex and memorizing the depth of the search tree.
    * If a tree has greater depth than any other tree that was computed before, the vertices itself and their distances to the root of the
    * tree are stored in 'vertices' and 'distances', respectively.
    *
    * As these steps require $O(nm)$ time in the worst case, we apply a simple heuristic that stops BFS from vertices that do not have
    * maximum eccentricity, and in some cases it even prevents starting BFS at such vertices.
    * a) For each vertex 'v' there is an upper bound 'eccentricity[v]' on the actual eccentricity of 'v'. Initially, this is set to a very large number.
    * b) Whenever a BFS with root 'v' has finished, 'eccentricity[v]' contains the actual eccentricity of 'v'.
    * c) If during a BFS with root 'v' a vertex 'u' is discovered, then dist(v, u) + eccentricty[u] is an upper bound on the eccentricity of 'v',
    *    and therefore the BFS is stopped if dist(v, u) + eccentricity[u] <= diameter_lb, where diameter_lb is the greatest eccentricity known so far.
    */

   assert(scip != NULL);
   assert(detectordata != NULL);
   assert(detectordata->graph != NULL);
   assert(detectordata->components != NULL);
   assert(maxdistance != NULL);
   assert(vertices != NULL);
   assert(distances != NULL);
   assert(ncomp != NULL);

   graph = detectordata->graph;
   nnodes = tcliqueGetNNodes(graph);

   SCIP_CALL( SCIPallocMemoryArray(scip, &queue, nnodes) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &marked, nnodes) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &eccentricity, nnodes) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &dist, nnodes) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &degree, nnodes) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &degreepos, nnodes) );

   /* get degrees of vertices and initialize all eccentricities of vertices to values representing upper bounds */
   origdegree = tcliqueGetDegrees(graph);
   for( i = 0; i < nnodes; ++i )
   {
      if( detectordata->components[i] != component )
         continue;

      eccentricity[i] = 2 * nnodes; /* do not use INT_MAX because it could lead to an overflow when adding values */
      degree[ncompnodes] = origdegree[i];    /* we copy the degree array because we are going to sort it */
      degreepos[ncompnodes] = i;

      ++ncompnodes;
   }

   /* sort vertices by their degrees in decreasing order */
   SCIPsortDownIntInt(degree, degreepos, ncompnodes);

   if( ncompnodes < k )
      k = ncompnodes;

   /* for each vertex a BFS will be performed */
   for( j = 0; j < ncompnodes; j++ )
   {
      int eccent = 0; /* eccentricity of this vertex, only valid if vertex has not been pruned */
      int startnode;
      SCIP_Bool pruned = FALSE;

      /* change order in which BFSes are performed: first start at 'k' low-degree vertices, then start BFS at high-degree vertices */
      if(j < k)
         startnode = degreepos[ncompnodes - k + j];
      else
         startnode = degreepos[j - k];

      /* eccentricity[startnode] always represents an UPPER BOUND on the actual eccentricity! */
      if( eccentricity[startnode] <= diameter )
         continue;

      /* unmark all vertices */
      BMSclearMemoryArray(marked, nnodes);

      /* add 'startnode' to the queue */
      queue[0] = startnode;
      equeue = 1;
      squeue = 0;
      marked[startnode] = TRUE;
      dist[startnode] = 0;

      /* continue BFS until vertex gets pruned or all vertices have been visited */
      while( !pruned && (equeue > squeue) )
      {
         int currentnode;
         int currentdistance;
         int* lastneighbour;
         int* node;

         /* dequeue new node */
         currentnode = queue[squeue];
         currentdistance = dist[currentnode];
         ++squeue;

         lastneighbour = tcliqueGetLastAdjedge(graph, currentnode);
         /* go through all neighbours */
         for( node = tcliqueGetFirstAdjedge(graph, currentnode); !pruned && (node <= lastneighbour); ++node )
         {
            const int v = *node;
            /* visit 'v' if it has not been visited yet */
            if( !marked[v] )
            {
               /* mark 'v' and add it to the queue */
               marked[v] = TRUE;
               queue[equeue] = v;
               dist[v] = currentdistance + 1;
               ++equeue;

               /* if 'v' is further away from the startnode than any other vertex, update the eccentricity */
               if( dist[v] > eccent )
                  eccent = dist[v];

               /* prune the startnode if its eccentricity will certainly not lead to a new upper bound */
               if( eccentricity[v] + dist[v] <= diameter )
               {
                  pruned = TRUE;
                  eccent = eccentricity[v] + dist[v];
               }

               /* update upper bound on eccentricity of 'v' */
               /*if( eccentricity[currentnode] + dist[v] < eccentricity[v] )
                  eccentricity[v] = eccentricity[currentnode] + dist[v];*/
            }
         }
      }

      eccentricity[startnode] = eccent;

      if( eccent > diameter )
      {
         SCIPdebugMessage("new incumbent in component %i: path of length %i starts at %i\n", component, eccent, startnode);
         diameter = eccent;

         *maxdistance = diameter;
         *ncomp = ncompnodes;
         /*detectordata->nblocks = diameter + 1;*/

         for( i = 0; i < ncompnodes; ++i )
         {
            vertices[i] = queue[i];
            distances[i] = dist[queue[i]];
         }
      }
   }

   SCIPfreeMemoryArray(scip, &degreepos);
   SCIPfreeMemoryArray(scip, &degree);
   SCIPfreeMemoryArray(scip, &dist);
   SCIPfreeMemoryArray(scip, &eccentricity);
   SCIPfreeMemoryArray(scip, &marked);
   SCIPfreeMemoryArray(scip, &queue);

   return SCIP_OKAY;
}

/** finds the connected components of the row graph. a staircase_lsp decomposition will be built for each component separately. */
static
SCIP_RETCODE findConnectedComponents(
   SCIP*                 scip,               /**< SCIP data structure */
   DEC_DETECTORDATA*     detectordata        /**< constraint handler data structure */
   )
{
   int i;
   int nnodes;
   int ncomps = 0;
   int curcomp;
   int* queue = NULL;
   int squeue;
   int equeue;
   TCLIQUE_GRAPH* graph;
   int* component;

   assert(scip != NULL);
   assert(detectordata != NULL);
   assert(detectordata->graph != NULL);
   assert(tcliqueGetNNodes(detectordata->graph) >= 0);

   graph = detectordata->graph;
   nnodes = tcliqueGetNNodes(graph);

   /* for each vertex the 'component' array contains a number from [0, ncomponents) */
   assert(detectordata->components == NULL);
   SCIP_CALL( SCIPallocMemoryArray(scip, &(detectordata->components), nnodes) );
   component = detectordata->components;

   /* component[i] == -1 if and only if vertex i has not been assigned to a component yet */
   for( i = 0; i < nnodes; ++i )
      component[i] = -1;

   SCIP_CALL( SCIPallocMemoryArray(scip, &queue, nnodes) );

   for( i = 0; i < nnodes; ++i )
   {
      /* find first node that has not been visited yet and start BFS */
      if( component[i] >= 0 )
         continue;

      SCIPdebugMessage("found new component; starting at %i\n", i);
      squeue = 0;
      equeue = 1;
      queue[0] = i;
      curcomp = ncomps++;
      component[i] = curcomp;

      /* dequeue a vertex as long as the queue is not empty */
      while( equeue > squeue )
      {
         int curnode;
         int* lastneighbour;
         int* node;

         curnode = queue[squeue++];

         assert(curnode < nnodes);

         /* add unvisited neighbors of this vertex to the queue */
         lastneighbour = tcliqueGetLastAdjedge(graph, curnode);
         for( node = tcliqueGetFirstAdjedge(graph, curnode); node <= lastneighbour; ++node )
         {
            assert(*node < nnodes);

            if( component[*node] < 0 )
            {
               assert(equeue < nnodes);
               component[*node] = curcomp;
               queue[equeue++] = *node;
            }
         }
      }
   }

   detectordata->ncomponents = ncomps;
   SCIPdebugMessage("found %i components\n", ncomps);

   SCIPfreeMemoryArray(scip, &queue);
   return SCIP_OKAY;
}


/** destructor of detector to free user data (called when GCG is exiting) */
static
DEC_DECL_FREEDETECTOR(detectorFreeStaircaseLsp)
{  /*lint --e{715}*/
   DEC_DETECTORDATA *detectordata;

   assert(scip != NULL);
   assert(detector != NULL);

   assert(strcmp(DECdetectorGetName(detector), DEC_DETECTORNAME) == 0);

   detectordata = DECdetectorGetData(detector);
   assert(detectordata != NULL);

   delete detectordata->newToOld;
   delete detectordata->oldToNew;

   SCIPfreeMemory(scip, &detectordata);

   return SCIP_OKAY;
}

/** detector initialization method (called after the problem has been transformed) */
static
DEC_DECL_INITDETECTOR(detectorInitStaircaseLsp)
{  /*lint --e{715}*/

   DEC_DETECTORDATA *detectordata;

   assert(scip != NULL);
   assert(detector != NULL);

   assert(strcmp(DECdetectorGetName(detector), DEC_DETECTORNAME) == 0);

   detectordata = DECdetectorGetData(detector);
   assert(detectordata != NULL);

   detectordata->graph = NULL;
   detectordata->components = NULL;
   detectordata->ncomponents = 0;
   detectordata->constoblock = NULL;
   detectordata->vartoblock = NULL;
   detectordata->nblocks = 0;

   return SCIP_OKAY;
}

/** detector deinitialization method (called before the transformed problem is freed) */
static
DEC_DECL_EXITDETECTOR(detectorExitStaircaseLsp)
{  /*lint --e{715}*/
   DEC_DETECTORDATA *detectordata;

   assert(scip != NULL);
   assert(detector != NULL);

   assert(strcmp(DECdetectorGetName(detector), DEC_DETECTORNAME) == 0);

   detectordata = DECdetectorGetData(detector);
   assert(detectordata != NULL);

   if( detectordata->graph != NULL )
   {
      tcliqueFree(&detectordata->graph);
   }

   if( detectordata->components != NULL )
   {
      SCIPfreeMemoryArray(scip, &detectordata->components);
   }

   return SCIP_OKAY;
}


/** detection function for partialdecs */
static
SCIP_RETCODE detection(
   SCIP*                   scip,                         /**< SCIP data structure */
   DEC_DETECTORDATA*       detectordata,                 /**< detectordata of the detector */
   Partialdec_Detection_Data*   partialdecdetectiondata  /**< partialdecdetectiondata (including the detprobdata) where to store the new Partialdecs */
)
{
   int i;
   int j;
   int* nodes = NULL;
   int nnodes;
   int* distances = NULL;
   int* blocks = NULL;
   int nblocks = 0;

   char decinfo[SCIP_MAXSTRLEN];

   gcg::DETPROBDATA* detprobdata = partialdecdetectiondata->detprobdata;
   gcg::PARTIALDECOMP* currpartialdec = new gcg::PARTIALDECOMP(partialdecdetectiondata->workonpartialdec);

   SCIP_CLOCK* temporaryClock;
   SCIP_CALL_ABORT(SCIPcreateClock(scip, &temporaryClock) );
   SCIP_CALL_ABORT( SCIPstartClock(scip, temporaryClock) );

   currpartialdec->refineToMaster();

   currpartialdec->sort();

   //currpartialdec->showScatterPlot(detprobdata);

   //SCIPverbMessage(scip, SCIP_VERBLEVEL_NORMAL, NULL, "Detecting staircase structure:");

   SCIP_CALL( createGraphFromPartialMatrix(scip, &(detectordata->graph), currpartialdec, detprobdata, detectordata) );
   SCIP_CALL( SCIPhashmapCreate(&detectordata->constoblock, SCIPblkmem(scip), detprobdata->getNConss() ) );

   if( tcliqueGetNNodes(detectordata->graph) > 0 )
   {
      nnodes = tcliqueGetNNodes(detectordata->graph);

      /* find connected components of the graph. the result will be stored in 'detectordata->components' */
      SCIP_CALL( findConnectedComponents(scip, detectordata) );

      SCIP_CALL( SCIPallocMemoryArray(scip, &nodes, nnodes) );
      SCIP_CALL( SCIPallocMemoryArray(scip, &distances, nnodes) );
      SCIP_CALL( SCIPallocMemoryArray(scip, &blocks, nnodes) );

      for( i = 0; i < nnodes; ++i)
         blocks[i] = -1;

      /* find the diameter of each connected component */
      for( i = 0; i < detectordata->ncomponents; ++i )
      {
         int diameter = 0;
         int ncompsize = 0;

         SCIP_CALL( findDiameter(scip, detectordata, &diameter, &ncompsize, nodes, distances, i) );
         SCIPdebugMessage("component %i has %i vertices and diameter %i\n", i, ncompsize, diameter);

         for( j = 0; j < ncompsize; j++ )
         {
            assert(nodes[j] >= 0);
            assert(nodes[j] < nnodes);
            assert(distances[j] >= 0);
            assert(distances[j] <= diameter);
            assert(distances[j] + nblocks < nnodes);

            blocks[nodes[j]] = nblocks + distances[j];
            SCIPdebugMessage("\tnode %i to block %i\n", nodes[j], nblocks + distances[j]);
         }

         nblocks += (diameter + 1);
      }
      if( nblocks > 0 )
      {
         detectordata->nblocks = nblocks;

         for( i = 0; i < nnodes; ++i )
         {
            assert(blocks[i] >= 0);
            SCIP_CALL( SCIPhashmapInsert(detectordata->constoblock, (void*) (size_t) detectordata->newToOld->at(i), (void*) (size_t) (blocks[i] + 1)) );
         }
      }

      SCIPfreeMemoryArray(scip, &blocks);
      SCIPfreeMemoryArray(scip, &nodes);
      SCIPfreeMemoryArray(scip, &distances);
      SCIPfreeMemoryArray(scip, &(detectordata->components));
   }

   SCIP_CALL( currpartialdec->assignPartialdecFromConstoblock(detectordata->constoblock, nblocks) );

   currpartialdec->assignCurrentStairlinking();
   currpartialdec->prepare();

   if( detectordata->constoblock != NULL )
      SCIPhashmapFree(&detectordata->constoblock);
   if( detectordata->vartoblock != NULL )
      SCIPhashmapFree(&detectordata->vartoblock);

   SCIP_CALL_ABORT( SCIPstopClock(scip, temporaryClock ) );

   assert(currpartialdec->checkConsistency());

   partialdecdetectiondata->detectiontime = SCIPgetClockTime(scip, temporaryClock);
   SCIP_CALL( SCIPallocMemoryArray(scip, &(partialdecdetectiondata->newpartialdecs), 1) );
   partialdecdetectiondata->newpartialdecs[0] = currpartialdec;
   partialdecdetectiondata->nnewpartialdecs = 1;
   (void) SCIPsnprintf(decinfo, SCIP_MAXSTRLEN, "staircase\\_lsp");
   partialdecdetectiondata->newpartialdecs[0]->addDetectorChainInfo(decinfo);
   partialdecdetectiondata->newpartialdecs[0]->addClockTime(SCIPgetClockTime(scip, temporaryClock));

   tcliqueFree(&detectordata->graph);
   SCIP_CALL_ABORT(SCIPfreeClock(scip, &temporaryClock) );

   return SCIP_OKAY;
}


/** detector structure detection method, tries to detect a structure in the problem */
static
DEC_DECL_PROPAGATEPARTIALDEC(detectorPropagatePartialdecStaircaseLsp)
{

   DEC_DETECTORDATA* detectordata = NULL;

   SCIP_CALL( SCIPallocMemory(scip, &detectordata) );
   assert(detectordata != NULL);
   detectordata->graph = NULL;
   detectordata->constoblock = NULL;
   detectordata->vartoblock = NULL;
   detectordata->nblocks = 0;
   detectordata->newToOld = NULL;
   detectordata->oldToNew = NULL;
   detectordata->components = NULL;
   detectordata->ncomponents = 0;

   *result = SCIP_DIDNOTFIND;

   detection(scip, detectordata, partialdecdetectiondata);

   delete detectordata->newToOld;
   delete detectordata->oldToNew;

   SCIPfreeMemory(scip, &detectordata);

   *result = SCIP_SUCCESS;

   return SCIP_OKAY;
}

static
DEC_DECL_FINISHPARTIALDEC(detectorFinishPartialdecStaircaseLsp)
{
   DEC_DETECTORDATA* detectordata = NULL;

   SCIP_CALL( SCIPallocMemory(scip, &detectordata) );
   assert(detectordata != NULL);
   detectordata->graph = NULL;
   detectordata->constoblock = NULL;
   detectordata->vartoblock = NULL;
   detectordata->nblocks = 0;
   detectordata->newToOld = NULL;
   detectordata->oldToNew = NULL;
   detectordata->components = NULL;
   detectordata->ncomponents = 0;

   *result = SCIP_DIDNOTFIND;

   detection(scip, detectordata, partialdecdetectiondata);

   delete detectordata->newToOld;
   delete detectordata->oldToNew;

   SCIPfreeMemory(scip, &detectordata);

   *result = SCIP_SUCCESS;

   return SCIP_OKAY;
}
//#define detectorExitStaircase NULL

#define detectorPostprocessPartialdecStaircaseLsp NULL


static
DEC_DECL_SETPARAMFAST(setParamAggressiveStaircaseLsp)
{
   char setstr[SCIP_MAXSTRLEN];
   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, TRUE) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, TRUE ) );

   return SCIP_OKAY;

}

static
DEC_DECL_SETPARAMFAST(setParamDefaultStaircaseLsp)
{
   char setstr[SCIP_MAXSTRLEN];
   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, DEC_ENABLED) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, DEC_ENABLEDFINISHING ) );

   return SCIP_OKAY;

}



static
DEC_DECL_SETPARAMFAST(setParamFastStaircaseLsp)
{
   char setstr[SCIP_MAXSTRLEN];
   const char* name = DECdetectorGetName(detector);

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/enabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE) );

   (void) SCIPsnprintf(setstr, SCIP_MAXSTRLEN, "detection/detectors/%s/finishingenabled", name);
   SCIP_CALL( SCIPsetBoolParam(scip, setstr, FALSE ) );

   return SCIP_OKAY;

}


/*
 * constraint specific interface methods
 */

/** creates the handler for staircase constraints and includes it in SCIP */
SCIP_RETCODE SCIPincludeDetectorStaircaseLsp(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   DEC_DETECTORDATA* detectordata;

   /* for thread safety: create staircase constraint handler data in callback methods*/
   detectordata = NULL;

   SCIP_CALL( SCIPallocMemory(scip, &detectordata) );
   assert(detectordata != NULL);
   detectordata->graph = NULL;
   detectordata->constoblock = NULL;
   detectordata->vartoblock = NULL;
   detectordata->nblocks = 0;
   detectordata->newToOld = NULL;
   detectordata->oldToNew = NULL;


   SCIP_CALL( DECincludeDetector( scip, DEC_DETECTORNAME, DEC_DECCHAR, DEC_DESC, DEC_FREQCALLROUND,
         DEC_MAXCALLROUND, DEC_MINCALLROUND, DEC_FREQCALLROUNDORIGINAL, DEC_MAXCALLROUNDORIGINAL, DEC_MINCALLROUNDORIGINAL, DEC_PRIORITY, DEC_ENABLED, DEC_ENABLEDFINISHING,DEC_ENABLEDPOSTPROCESSING, DEC_SKIP,
         DEC_USEFULRECALL, detectordata, detectorFreeStaircaseLsp,
         detectorInitStaircaseLsp, detectorExitStaircaseLsp, detectorPropagatePartialdecStaircaseLsp, detectorFinishPartialdecStaircaseLsp, detectorPostprocessPartialdecStaircaseLsp, setParamAggressiveStaircaseLsp, setParamDefaultStaircaseLsp, setParamFastStaircaseLsp) );


   return SCIP_OKAY;
}
