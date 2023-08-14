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

/**@file   hypercolgraph_def.h
 * @brief  Column hypergraph
 * @author Martin Bergner
 * @author Annika Thome
 *
 * A hypergraph structure with a node for every constraint and a hyperedge for every variable.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_HYPERCOLGRAPH_DEF_H_
#define GCG_HYPERCOLGRAPH_DEF_H_

#include "hypercolgraph.h"
#include "class_partialdecomp.h"
#include "class_detprobdata.h"
#include <set>
#include <algorithm>
#include <vector>
#include <iostream>

namespace gcg
{
template <class T>
HypercolGraph<T>::HypercolGraph(
   SCIP*                 scip,              /**< SCIP data structure */
   Weights               w                  /**< weights for the given graph */
):  MatrixGraph<T>(scip, w), graph(scip)
{
   this->graphiface = &graph;
   this->name = std::string("hypercol");
}

template <class T>
HypercolGraph<T>::~HypercolGraph()
{
   // TODO Auto-generated destructor stub
}


/** writes the graph to the given file.
 *  The format is graph dependent
 */
template <class T>
SCIP_RETCODE HypercolGraph<T>::writeToFile(
   int                fd,                    /**< filename where the graph should be written to */
   SCIP_Bool          edgeweights            /**< whether to write edgeweights */
 )
{
   function f(this->nvars);
   FILE* file;
   file = fdopen(fd, "w");
   if( file == NULL )
      return SCIP_FILECREATEERROR;

   SCIPinfoMessage(this->scip_, file, "%d %d %d\n", getNEdges(), getNNodes()+this->dummynodes, edgeweights ? 1 :0);

   for( int i = 0; i < getNEdges(); ++i )
   {
      std::vector<int> neighbors = getHyperedgeNodes(i);

      if( edgeweights )
      {
         SCIPinfoMessage(this->scip_, file, "%d ", graph.getHyperedgeWeight(i));
      }
      for( size_t j = 0; j < neighbors.size(); ++j )
      {
         SCIPinfoMessage(this->scip_, file, "%d ",neighbors[j]+1);
      }
      SCIPinfoMessage(this->scip_, file, "\n");
   }
   if( !fclose(file) )
      return SCIP_OKAY;
   else
      return SCIP_WRITEERROR;
}

template <class T>
int HypercolGraph<T>::getNEdges()
{
   return this->nvars;
}

template <class T>
int HypercolGraph<T>::getNNodes()
{
   return this->nconss;
}


template <class T>
std::vector<int> HypercolGraph<T>::getHyperedgeNodes(
   int i
)
{
   assert(i >= 0);
   assert(i < getNEdges());

   std::vector<int> neighbors = graph.getHyperedgeNodes(i);
   return neighbors;
}


template <class T>
SCIP_RETCODE HypercolGraph<T>::createFromMatrix(
   SCIP_CONS**           conss,              /**< constraints for which graph should be created */
   SCIP_VAR**            vars,               /**< variables for which graph should be created */
   int                   nconss_,             /**< number of constraints */
   int                   nvars_               /**< number of variables */
   )
{
   int i;
   int k;
   SCIP_Bool success;
   std::vector< std::vector<int> > hyperedges;

   assert(conss != NULL);
   assert(vars != NULL);
   assert(nvars_ > 0);
   assert(nconss_ > 0);

   this->nvars = nvars_;
   this->nconss = nconss_;

   /* go through all constraints */
   for( i = 0; i < this->nconss; ++i )
   {
      TCLIQUE_WEIGHT weight;

      /* calculate weight of node */
      weight = this->weights.calculate(conss[i]);

      this->graph.addNode(i, weight);
   }

   hyperedges.resize(this->nvars);

   /* go through all constraints */
   for( i = 0; i < this->nconss; ++i )
   {
      SCIP_VAR** curvars1 = NULL;

      int ncurvars1;
      SCIP_CALL( SCIPgetConsNVars(this->scip_, conss[i], &ncurvars1, &success) );
      assert(success);
      if( ncurvars1 == 0 )
         continue;

      /*
       * may work as is, as we are copying the constraint later regardless
       * if there are variables in it or not
       */
      SCIP_CALL( SCIPallocBufferArray(this->scip_, &curvars1, ncurvars1) );
      SCIP_CALL( SCIPgetConsVars(this->scip_, conss[i], curvars1, ncurvars1, &success) );
      assert(success);

      /** @todo skip all variables that have a zero coeffient or where all coefficients add to zero */
      /** @todo Do more then one entry per variable actually work? */

      for( k = 0; k < ncurvars1; ++k )
      {
         SCIP_VAR* var1 = NULL;
         int varIndex1;

         if( SCIPgetStage(this->scip_) >= SCIP_STAGE_TRANSFORMED)
            var1 = SCIPvarGetProbvar(curvars1[k]);
         else
            var1 = curvars1[k];

         if( !GCGisVarRelevant(var1) )
            continue;

         assert(var1 != NULL);
         varIndex1 = SCIPvarGetProbindex(var1);
         assert(varIndex1 >= 0);
         assert(varIndex1 < this->nvars);

         hyperedges[varIndex1].insert(hyperedges[varIndex1].end(), i);
      }
      SCIPfreeBufferArray(this->scip_, &curvars1);
   }

   /* go through all variables */
   for( i = 0; i < this->nvars; ++i )
   {
      TCLIQUE_WEIGHT weight;

      /* calculate weight of node */
      weight = this->weights.calculate(vars[i]);

      this->graph.addHyperedge(hyperedges[i], weight);
   }
   this->graph.flush();

   return SCIP_OKAY;
}

template <class T>
SCIP_RETCODE HypercolGraph<T>::createFromPartialMatrix(
      DETPROBDATA*        detprobdata,
      PARTIALDECOMP*            partialdec
   )
{
   int i;
   int j;
   TCLIQUE_WEIGHT weight;
   std::vector< std::vector<int> > hyperedges;
   unordered_map<int, int> oldToNewConsIndex;
   vector<bool> varsBool(partialdec->getNVars(), false); /**< true, if the var will be part of the graph */
   vector<bool> conssBool(partialdec->getNConss(), false); /**< true, if the cons will be part of the graph */
   vector<int> conssForGraph; /** stores the conss included by the graph */
   vector<int> varsForGraph; /** stores the vars included by the graph */

   //fillout conssForGraph and varsForGraph
   for(int c = 0; c < partialdec->getNOpenconss(); ++c)
   {
      int cons = partialdec->getOpenconss()[c];
      for(int v = 0; v < partialdec->getNOpenvars(); ++v)
      {
         int var = partialdec->getOpenvars()[v];
         for(i = 0; i < detprobdata->getNVarsForCons(cons); ++i)
         {
            if(var == detprobdata->getVarsForCons(cons)[i])
            {
               varsBool[var] = true;
               conssBool[cons] = true;
            }
         }
      }
   }

   for(int v = 0; v < partialdec->getNOpenvars(); ++v)
   {
      int var = partialdec->getOpenvars()[v];
      if(varsBool[var])
         varsForGraph.push_back(var);
   }
   for(int c = 0; c < partialdec->getNOpenconss(); ++c)
   {
      int cons = partialdec->getOpenconss()[c];
      if(conssBool[cons])
         conssForGraph.push_back(cons);
   }

   this->nconss = (int)conssForGraph.size();
   this->nvars = (int)varsForGraph.size();

   /* go through all open constraints */
   for( i = 0; i < this->nconss; ++i )
   {
      int oldConsId = conssForGraph[i];

      /* calculate weight of node */
      weight = this->weights.calculate(detprobdata->getCons(oldConsId));

      oldToNewConsIndex.insert({oldConsId,i});

      this->graph.addNode(i, weight);
   }



   /* go through all open variables */
   for( i = 0; i < this->nvars; ++i )
   {
      std::vector<int> hyperedge;
      int oldVarId = varsForGraph[i];

      for( j = 0; j < detprobdata->getNConssForVar(oldVarId); ++j )
      {
         int oldConsId = detprobdata->getConssForVar(oldVarId)[j];
         if(!conssBool[oldConsId])
            continue;
         hyperedge.insert(hyperedge.end(), oldToNewConsIndex[oldConsId]);
      }
      /* calculate weight of hyperedge */
      weight = this->weights.calculate(detprobdata->getVar(oldVarId));
      this->graph.addHyperedge(hyperedge, weight);
   }


   this->graph.flush();

   return SCIP_OKAY;
}


template <class T>
SCIP_RETCODE HypercolGraph<T>::createDecompFromPartition(
   DEC_DECOMP**          decomp           /**< decomposition structure to generate */
   )
{
   SCIP_HASHMAP* constoblock;
   SCIP_CONS** conss;
   int nblocks;

   assert(decomp != NULL);
   std::vector<int> partition = this->getPartition();
   conss = SCIPgetConss(this->scip_);

   SCIP_CALL( SCIPhashmapCreate(&constoblock, SCIPblkmem(this->scip_), this->nconss) );

   assert((size_t)SCIPgetNConss(this->scip_) == partition.size());
   nblocks = 1+*std::max_element(partition.begin(), partition.end() );

   for( int c = 0; c < this->nconss; ++c )
   {
      int consblock = partition[c]+1;

      SCIP_CALL( SCIPhashmapInsert(constoblock, conss[c], (void*) (size_t) consblock) );
   }

   SCIP_CALL( DECdecompCreate(this->scip_, decomp) );
   SCIP_CALL( DECfilloutDecompFromConstoblock(this->scip_, *decomp, constoblock, nblocks, FALSE) );

   return SCIP_OKAY;
}

template <class T>
SCIP_RETCODE HypercolGraph<T>::createPartialdecFromPartition(
   PARTIALDECOMP**     firstpartialdec,
   PARTIALDECOMP**     secondpartialdec,
   DETPROBDATA*  detprobdata
   )
{
   SCIP_HASHMAP* constoblock;
   SCIP_CONS** conss;
   int nblocks;

   std::vector<int> partition = this->getPartition();
   conss = SCIPgetConss(this->scip_);
   std::vector<bool> isEmptyBlock;
   std::vector<int> nEmptyBlocksBefore;

   SCIP_CALL( SCIPhashmapCreate(&constoblock, SCIPblkmem(this->scip_), this->nconss) );

   assert((size_t)SCIPgetNConss(this->scip_) == partition.size());
   nblocks = 1+*std::max_element(partition.begin(), partition.end() );

   /** add data structures to handle empty blocks */

   isEmptyBlock = std::vector<bool>(nblocks, true);
   nEmptyBlocksBefore = std::vector<int>(nblocks, 0);

   for( int c = 0; c < this->nconss; ++c )
   {
      int consblock = partition[c]+1;
      isEmptyBlock[consblock-1] = false;
   }

   for(int b1 = 0; b1 < nblocks; ++b1)
   {
       if (isEmptyBlock[b1] )
       {
           std::cout << "block  " << b1 << "  is an empty block " << std::endl;
           for(int b2 = b1+1; b2 < nblocks; ++b2)
               nEmptyBlocksBefore[b2]++;
       }
   }

   for( int c = 0; c < this->nconss; ++c )
   {
       int consblock = partition[c]+1;
       consblock -= nEmptyBlocksBefore[partition[c] ];
       SCIP_CALL( SCIPhashmapInsert(constoblock, (void*) (size_t) detprobdata->getIndexForCons(conss[c]), (void*) (size_t) consblock) );
   }

   bool original = detprobdata->isAssignedToOrigProb();
   if( firstpartialdec != NULL )
   {
      (*firstpartialdec) = new PARTIALDECOMP(this->scip_, original);
      SCIP_CALL((*firstpartialdec)->filloutPartialdecFromConstoblock(constoblock, nblocks));
   }
   if( secondpartialdec != NULL )
   {
      (*secondpartialdec) = new PARTIALDECOMP(this->scip_, original);
      SCIP_CALL((*secondpartialdec)->filloutBorderFromConstoblock(constoblock, nblocks));
   }
   SCIPhashmapFree(&constoblock);

   return SCIP_OKAY;
}

template <class T>
SCIP_RETCODE HypercolGraph<T>::createPartialdecFromPartition(
   PARTIALDECOMP*      oldpartialdec,
   PARTIALDECOMP**     firstpartialdec,
   PARTIALDECOMP**     secondpartialdec,
   DETPROBDATA*  detprobdata
   )
{
   SCIP_HASHMAP* constoblock;
   int nblocks;
   std::vector<bool> isEmptyBlock;
   std::vector<int> nEmptyBlocksBefore;
   int nEmptyBlocks = 0;

   if(this->nconss == 0)
   {
      (*firstpartialdec) = NULL;
      (*secondpartialdec) = NULL;
      return SCIP_OKAY;
   }

   std::vector<int> partition = this->getPartition();

   //fillout conssForGraph
   vector<int> conssForGraph; /** stores the conss included by the graph */
   vector<bool> conssBool(oldpartialdec->getNConss(), false); /**< true, if the cons will be part of the graph */
   bool found;

   for(int c = 0; c < oldpartialdec->getNOpenconss(); ++c)
   {
      int cons = oldpartialdec->getOpenconss()[c];
      found = false;
      for(int v = 0; v < oldpartialdec->getNOpenvars() && !found; ++v)
      {
         int var = oldpartialdec->getOpenvars()[v];
         for(int i = 0; i < detprobdata->getNVarsForCons(cons) && !found; ++i)
         {
            if(var == detprobdata->getVarsForCons(cons)[i])
            {
               conssBool[cons] = true;
               found = true;
            }
         }
      }
   }

   for(int c = 0; c < oldpartialdec->getNOpenconss(); ++c)
   {
      int cons = oldpartialdec->getOpenconss()[c];
      if(conssBool[cons])
         conssForGraph.push_back(cons);
   }

   SCIP_CALL( SCIPhashmapCreate(&constoblock, SCIPblkmem(this->scip_), this->nconss) );
   nblocks = 1+*std::max_element(partition.begin(), partition.end() );
   /** add data structures to handle empty blocks */

   isEmptyBlock = std::vector<bool>(nblocks, true);
   nEmptyBlocksBefore = std::vector<int>(nblocks, 0);

   for( int c = 0; c < this->nconss; ++c )
   {
       int consblock = partition[c]+1;
       isEmptyBlock[consblock-1] = false;
   }

   for(int b1 = 0; b1 < nblocks; ++b1)
   {
       if (isEmptyBlock[b1] )
       {
           nEmptyBlocks++;
           for(int b2 = b1+1; b2 < nblocks; ++b2)
               nEmptyBlocksBefore[b2]++;
       }
   }

   for( int c = 0; c < this->nconss; ++c )
   {
       int consblock = partition[c]+1;
       consblock -= nEmptyBlocksBefore[partition[c] ];
       SCIP_CALL( SCIPhashmapInsert(constoblock, (void*) (size_t) conssForGraph[c], (void*) (size_t) consblock) );
   }

   nblocks -= nEmptyBlocks;

   if( firstpartialdec != NULL )
   {
      (*firstpartialdec) = new PARTIALDECOMP(oldpartialdec);
      SCIP_CALL((*firstpartialdec)->assignPartialdecFromConstoblock(constoblock, nblocks));
   }
   if( secondpartialdec != NULL )
   {
      (*secondpartialdec) = new PARTIALDECOMP(oldpartialdec);
      SCIP_CALL((*secondpartialdec)->assignBorderFromConstoblock(constoblock, nblocks));
   }
   SCIPhashmapFree(&constoblock);

   return SCIP_OKAY;
}


} /* namespace gcg */

#endif
