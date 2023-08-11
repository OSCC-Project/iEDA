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

/**@file   hyperrowcolgraph_def.h
 * @brief  A hypergraph with row and column nodes
 * @author Martin Bergner
 * @author Annika Thome
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_HYPERROWCOLGRAPH_DEF_H_
#define GCG_HYPERROWCOLGRAPH_DEF_H_

#include "hyperrowcolgraph.h"
#include "scip_misc.h"
#include "class_partialdecomp.h"
#include "class_detprobdata.h"
#include <algorithm>
#include <set>

namespace gcg {
template <class T>
HyperrowcolGraph<T>::HyperrowcolGraph(
   SCIP*                 scip,              /**< SCIP data structure */
   Weights               w                  /**< weights for the given graph */
): MatrixGraph<T>(scip, w),graph(scip)
{
   this->graphiface = &graph;
   this->name = std::string("hyperrowcol");
}

template <class T>
HyperrowcolGraph<T>::~HyperrowcolGraph()
{

}


/**
 * Builds a bipartite representation of the hyperrowcol graph out of the matrix.
 *
 * The function will create an node for every constraint, every variable and every nonzero entry of the matrix.
 * One side of the bipartite graph are the nonzero entries (nodes), the constraints and variables are on the other side (hyperedges).
 * A nonzero entry a_{ij} is incident to the constraint i and the variable j.
 *
 * @todo The nonzeroness is not checked, all variables in the variable array are considered
 */
template <class T>
SCIP_RETCODE HyperrowcolGraph<T>::createFromMatrix(
   SCIP_CONS**           conss,              /**< constraints for which graph should be created */
   SCIP_VAR**            vars,               /**< variables for which graph should be created */
   int                   nconss_,            /**< number of constraints */
   int                   nvars_              /**< number of variables */
   )
{
   int i;
   int j;
   SCIP_Bool success;

   assert(conss != NULL);
   assert(vars != NULL);
   assert(nvars_ > 0);
   assert(nconss_ > 0);

   this->nvars = nvars_;
   this->nconss = nconss_;

   /* create nodes for constraints and variables (hyperedges) */
   for( i = 0; i < this->nvars + this->nconss; ++i )
   {
      TCLIQUE_WEIGHT weight;

      /* note that the first nvars nodes correspond to variables */
      if( i < this->nvars )
      {
         weight = this->weights.calculate(vars[i]);
         SCIPdebugMessage("Weight for var <%s> is %d\n", SCIPvarGetName(vars[i]), weight);
      }

      else
      {
         weight = this->weights.calculate(conss[i-this->nvars]);
         SCIPdebugMessage("Weight for cons <%s> is %d\n", SCIPconsGetName(conss[i-this->nvars]), weight);
      }

      SCIP_CALL( this->graph.addNode(i, weight) );
   }

   /* go through all constraints */
   for( i = 0; i < this->nconss; ++i )
   {
      SCIP_VAR **curvars = NULL;

      int ncurvars;
      SCIP_CALL( SCIPgetConsNVars(this->scip_, conss[i], &ncurvars, &success) );
      assert(success);
      if( ncurvars == 0 )
         continue;

      /*
       * may work as is, as we are copying the constraint later regardless
       * if there are variables in it or not
       */
      SCIP_CALL( SCIPallocBufferArray(this->scip_, &curvars, ncurvars) );
      SCIP_CALL( SCIPgetConsVars(this->scip_, conss[i], curvars, ncurvars, &success) );
      assert(success);

      /** @todo skip all variables that have a zero coeffient or where all coefficients add to zero */
      /** @todo Do more then one entry per variable actually work? */

      for( j = 0; j < ncurvars; ++j )
      {
         SCIP_VAR* var;
         int varIndex;

         if( SCIPgetStage(this->scip_) >= SCIP_STAGE_TRANSFORMED)
            var = SCIPvarGetProbvar(curvars[j]);
         else
            var = curvars[j];

         if( !GCGisVarRelevant(var) )
            continue;

         assert(var != NULL);
         varIndex = SCIPvarGetProbindex(var);
         assert(varIndex >= 0);
         assert(varIndex < this->nvars);

         SCIPdebugMessage("Cons <%s> (%d), var <%s> (%d), nonzero %d\n", SCIPconsGetName(conss[i]), i, SCIPvarGetName(var), varIndex, this->nnonzeroes);
         /* add nonzero node and edge to variable and constraint) */;
         SCIP_CALL( this->graph.addNode( this->nvars+this->nconss+this->nnonzeroes, 0) );
         SCIP_CALL( this->graph.addEdge(varIndex, this->nvars+this->nconss+this->nnonzeroes) );
         SCIP_CALL( this->graph.addEdge(this->nvars+i, this->nvars+this->nconss+this->nnonzeroes) );

         this->nnonzeroes++;
      }
      SCIPfreeBufferArray(this->scip_, &curvars);
   }

   SCIP_CALL( this->graph.flush() );

   return SCIP_OKAY;
}

template <class T>
SCIP_RETCODE HyperrowcolGraph<T>::createFromPartialMatrix(
   DETPROBDATA*         detprobdata,
   PARTIALDECOMP*             partialdec
   )
{
   int i;
   int j;
   unordered_map<int, int> oldToNewConsIndex;
   unordered_map<int, int> oldToNewVarIndex;
   vector<int> conssForGraph; /** stores the conss included by the graph */
   vector<int> varsForGraph; /** stores the vars included by the graph */
   vector<bool> varsBool(partialdec->getNVars(), false); /**< true, if the var will be part of the graph */
   vector<bool> conssBool(partialdec->getNConss(), false); /**< true, if the cons will be part of the graph */


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

   /** add node for every var */
   for( i = 0 ; i < (int)varsForGraph.size(); ++i )
   {
      int oldVarId = varsForGraph[i];
      assert(varsBool[oldVarId]);
      TCLIQUE_WEIGHT weight;

      /* note that the first nvars nodes correspond to variables */
      weight = this->weights.calculate(detprobdata->getVar(oldVarId));
      oldToNewVarIndex.insert({oldVarId ,i});

      this->graph.addNode(i, weight);
   }

   /** add node for every cons */
   for(  j = 0 ; j < (int)conssForGraph.size(); ++j  )
   {
      int oldConsId = conssForGraph[j];
      assert(conssBool[oldConsId]);
      TCLIQUE_WEIGHT weight;

      /* note that the first nvars nodes correspond to variables (legacy implementation) */
      weight = this->weights.calculate(detprobdata->getCons(oldConsId));
      oldToNewConsIndex.insert({ oldConsId, j});
      this->graph.addNode( this->nvars + j, weight);
   }

   this->nnonzeroes = 0;
   /* go through all open constraints */
   for( i = 0; i < (int)conssForGraph.size(); ++i )
   {
      int oldConsId = conssForGraph[i];

      for( j = 0; j < detprobdata->getNVarsForCons(oldConsId); ++j )
      {
         int oldVarId = detprobdata->getVarsForCons(oldConsId)[j];
         if(!varsBool[oldVarId])
            continue;
         SCIPdebugMessage("Cons <%s> (%d), var <%s> (%d), nonzero %d\n", SCIPconsGetName(detprobdata->getCons(oldConsId)), i,
            SCIPvarGetName(detprobdata->getVar(oldVarId)), oldToNewVarIndex[oldVarId], this->nnonzeroes);
         /* add nonzero node and edge to variable and constraint) */;
         SCIP_CALL( this->graph.addNode( this->nvars+this->nconss+this->nnonzeroes, 0) );
         SCIP_CALL( this->graph.addEdge(oldToNewVarIndex[oldVarId], this->nvars+this->nconss+this->nnonzeroes) );
         SCIP_CALL( this->graph.addEdge(this->nvars+oldToNewConsIndex[oldConsId], this->nvars+this->nconss+this->nnonzeroes) );

         this->nnonzeroes++;
      }
   }

   this->graph.flush();
   return SCIP_OKAY;
}


/** writes the graph to the given file.
 *  The format is graph dependent
 */
template <class T>
SCIP_RETCODE HyperrowcolGraph<T>::writeToFile(
   int                fd,                    /**< filename where the graph should be written to */
   SCIP_Bool          edgeweights            /**< whether to write edgeweights */
 )
{
   FILE* file;
   file = fdopen(fd, "wx");
   if( file == NULL )
      return SCIP_FILECREATEERROR;

   SCIPinfoMessage(this->scip_, file, "%d %d %d\n", this->nvars+this->nconss, this->nnonzeroes+this->dummynodes, edgeweights ? 1 :0);

   for( int i = 0; i < this->nvars+this->nconss; ++i )
   {
      std::vector<int> neighbors = graph.getNeighbors(i);
      int nneighbors = graph.getNNeighbors(i);
      if( edgeweights )
      {
         SCIPinfoMessage(this->scip_, file, "%d ", graph.getWeight(i));
      }
      for( int j = 0; j < nneighbors; ++j )
      {
         SCIPinfoMessage(this->scip_, file, "%d ",neighbors[j]+1-this->nvars-this->nconss);
      }
      SCIPinfoMessage(this->scip_, file, "\n");
   }

   if( !fclose(file) )
      return SCIP_OKAY;
   else
      return SCIP_WRITEERROR;
}


class function {
   int diff;
public:
   function(int i):diff(i) {}
   int operator()(int i) const { return i-diff;}
};

template <class T>
std::vector<int> HyperrowcolGraph<T>::getNeighbors(
   int i
)
{
   assert(i >= 0);
   assert(i < this->nnonzeroes);
   function f(this->nconss+this->nvars);
   std::vector<int>::iterator it;
   std::set<int> neighbors;
   std::vector<int> immediateneighbors = this->graph.getNeighbors(i+this->nconss+this->nvars);
   for( size_t j = 0; j < immediateneighbors.size(); ++j)
   {
      std::vector<int> alternateneighbor = this->graph.getNeighbors(immediateneighbors[j]);
      neighbors.insert(alternateneighbor.begin(), alternateneighbor.end() );
   }
   std::vector<int> r(neighbors.size(), 0);
   std::transform(neighbors.begin(), neighbors.end(), r.begin(), f);
   it = std::remove(r.begin(), r.end(), i);

   return std::vector<int>(r.begin(), it);
}

template <class T>
std::vector<int> HyperrowcolGraph<T>::getHyperedgeNodes(
   int i
)
{
   function f(this->nconss+this->nvars);
   assert(i >= 0);
   assert(i < this->nconss+this->nvars);

   std::vector<int> neighbors = this->graph.getNeighbors(i);
   std::transform(neighbors.begin(), neighbors.end(), neighbors.begin(), f);
   return neighbors;
}

template <class T>
std::vector<int> HyperrowcolGraph<T>::getConsNonzeroNodes(
   int i
)
{
   function f(this->nconss+this->nvars);
   assert(i >= 0);
   assert(i < this->nconss);

   std::vector<int> neighbors = this->graph.getNeighbors(i+this->nvars);
   std::transform(neighbors.begin(), neighbors.end(), neighbors.begin(), f);
   return neighbors;
}

template <class T>
std::vector<int> HyperrowcolGraph<T>::getVarNonzeroNodes(
   int i
)
{
   function f(this->nconss+this->nvars);
   assert(i >= 0);
   assert(i < this->nvars);

   std::vector<int> neighbors = this->graph.getNeighbors(i);
   std::transform(neighbors.begin(), neighbors.end(), neighbors.begin(), f);
   return neighbors;
}

template <class T>
SCIP_RETCODE HyperrowcolGraph<T>::createDecompFromPartition(
   DEC_DECOMP**       decomp              /**< decomposition structure to generate */
   )
{
   int nblocks;
   SCIP_HASHMAP* constoblock = NULL;

   int* nsubscipconss = NULL;
   int i;
   SCIP_CONS** conss = NULL;
   SCIP_Bool emptyblocks = FALSE;
   std::vector<int> partition = graph.getPartition();
   conss = SCIPgetConss(this->scip_);

   nblocks = *(std::max_element(partition.begin(), partition.end()))+1;
   SCIP_CALL( SCIPallocBufferArray(this->scip_, &nsubscipconss, nblocks) );
   BMSclearMemoryArray(nsubscipconss, nblocks);

   SCIP_CALL( SCIPhashmapCreate(&constoblock, SCIPblkmem(this->scip_), this->nconss) );

   /* assign constraints to partition */
   for( i = 0; i < this->nconss; i++ )
   {
      std::set<int> blocks;
      std::vector<int> nonzeros = getConsNonzeroNodes(i);
      for( size_t k = 0; k < nonzeros.size(); ++k )
      {
         blocks.insert(partition[nonzeros[k]]);
      }
      if( blocks.size() > 1 )
      {
         SCIP_CALL( SCIPhashmapInsert(constoblock, conss[i], (void*) (size_t) (nblocks+1)) );
      }
      else
      {
         int block = *(blocks.begin());
         SCIP_CALL( SCIPhashmapInsert(constoblock, conss[i], (void*) (size_t) (block +1)) );
         ++(nsubscipconss[block]);
      }
   }

   /* first, make sure that there are constraints in every block, otherwise the hole thing is useless */
   for( i = 0; i < nblocks; ++i )
   {
      if( nsubscipconss[i] == 0 )
      {
         SCIPdebugMessage("Block %d does not have any constraints!\n", i);
         emptyblocks = TRUE;
      }
   }

   if( !emptyblocks )
   {
      SCIP_CALL( DECdecompCreate(this->scip_, decomp) );
      SCIP_CALL( DECfilloutDecompFromConstoblock(this->scip_, *decomp, constoblock, nblocks, FALSE) );
   }
   else {
      SCIPhashmapFree(&constoblock);
      *decomp = NULL;
   }

   SCIPfreeBufferArray(this->scip_, &nsubscipconss);
   return SCIP_OKAY;
}

template <class T>
SCIP_RETCODE HyperrowcolGraph<T>::createPartialdecFromPartition(
   PARTIALDECOMP**      firstpartialdec,
   PARTIALDECOMP**      secondpartialdec,
   DETPROBDATA*  detprobdata
   )
{
   int nblocks;
   SCIP_HASHMAP* constoblock = NULL;

   int *nsubscipconss = NULL;
   int i;
   SCIP_CONS **conss = NULL;
   SCIP_Bool emptyblocks = FALSE;
   std::vector<int> partition = graph.getPartition();
   conss = SCIPgetConss(this->scip_);

   nblocks = *(std::max_element(partition.begin(), partition.end()))+1;
   SCIP_CALL( SCIPallocBufferArray(this->scip_, &nsubscipconss, nblocks) );
   BMSclearMemoryArray(nsubscipconss, nblocks);

   SCIP_CALL( SCIPhashmapCreate(&constoblock, SCIPblkmem(this->scip_), this->nconss) );

   /* assign constraints to partition */
   for( i = 0; i < this->nconss; i++ )
   {
      std::set<int> blocks;
      std::vector<int> nonzeros = getConsNonzeroNodes(i);
      for( size_t k = 0; k < nonzeros.size(); ++k )
      {
         blocks.insert(partition[nonzeros[k]]);
      }
      if( blocks.size() > 1 )
      {
         SCIP_CALL( SCIPhashmapInsert(constoblock, (void*) (size_t)detprobdata->getIndexForCons(conss[i]), (void*) (size_t) (nblocks+1)) );
      }
      else
      {
         int block = *(blocks.begin());
         SCIP_CALL( SCIPhashmapInsert(constoblock, (void*) (size_t) detprobdata->getIndexForCons(conss[i]), (void*) (size_t) (block +1)) );
         ++(nsubscipconss[block]);
      }
   }

   /* first, make sure that there are constraints in every block, otherwise the whole thing is useless */
   for( i = 0; i < nblocks; ++i )
   {
      if( nsubscipconss[i] == 0 )
      {
         SCIPdebugMessage("Block %d does not have any constraints!\n", i);
         emptyblocks = TRUE;
      }
   }

   if( !emptyblocks )
   {
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
         for( int col = 0; col < (*firstpartialdec)->getNLinkingvars(); ++col )
         {
            (*secondpartialdec)->setVarToLinking((*firstpartialdec)->getLinkingvars()[col]);
            (*secondpartialdec)->deleteOpenvar(col);
         }
      }
      SCIPhashmapFree(&constoblock);
   }
   else {
      SCIPhashmapFree(&constoblock);
      if( firstpartialdec != NULL )
      {
         (*firstpartialdec) = NULL;
      }
      if( secondpartialdec != NULL)
      {
         (*secondpartialdec) = NULL;
      }
   }

   SCIPfreeBufferArray(this->scip_, &nsubscipconss);
   return SCIP_OKAY;
}

template <class T>
SCIP_RETCODE HyperrowcolGraph<T>::createPartialdecFromPartition(
   PARTIALDECOMP*       oldpartialdec,
   PARTIALDECOMP**      firstpartialdec,
   PARTIALDECOMP**      secondpartialdec,
   DETPROBDATA*   detprobdata
   )
{
   int nblocks;
   SCIP_HASHMAP* constoblock = NULL;

   int* nsubscipconss = NULL;
   int i;
   SCIP_Bool emptyblocks = FALSE;

   if(this->nconss == 0)
   {
      (*firstpartialdec) = NULL;
      (*secondpartialdec) = NULL;
      return SCIP_OKAY;
   }

   std::vector<int> partition = graph.getPartition();

   nblocks = *(std::max_element(partition.begin(), partition.end()))+1;
   SCIP_CALL( SCIPallocBufferArray(this->scip_, &nsubscipconss, nblocks) );
   BMSclearMemoryArray(nsubscipconss, nblocks);

   SCIP_CALL( SCIPhashmapCreate(&constoblock, SCIPblkmem(this->scip_), this->nconss) );

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
         for(i = 0; i < detprobdata->getNVarsForCons(cons) && !found; ++i)
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


   /* assign constraints to partition */
   for( i = 0; i < this->nconss; i++ )
   {
      std::set<int> blocks;
      std::vector<int> nonzeros = getConsNonzeroNodes(i);
      for( size_t k = 0; k < nonzeros.size(); ++k )
      {
         blocks.insert(partition[nonzeros[k]]);
      }
      if( blocks.size() > 1 )
      {
         SCIP_CALL( SCIPhashmapInsert(constoblock, (void*) (size_t) conssForGraph[i], (void*) (size_t) (nblocks+1)) );
      }
      else
      {
         int block = *(blocks.begin());
         SCIP_CALL( SCIPhashmapInsert(constoblock, (void*) (size_t) conssForGraph[i], (void*) (size_t) (block +1)) );
         ++(nsubscipconss[block]);
      }
   }

   /* first, make sure that there are constraints in every block, otherwise the hole thing is useless */
   for( i = 0; i < nblocks; ++i )
   {
      if( nsubscipconss[i] == 0 )
      {
         SCIPdebugMessage("Block %d does not have any constraints!\n", i);
         emptyblocks = TRUE;
      }
   }

   if( !emptyblocks )
   {
      (*firstpartialdec) = new PARTIALDECOMP(oldpartialdec);
      SCIP_CALL( (*firstpartialdec)->assignPartialdecFromConstoblock(constoblock, nblocks) );
      (*secondpartialdec) = new PARTIALDECOMP(oldpartialdec);
      SCIP_CALL( (*secondpartialdec)->assignBorderFromConstoblock(constoblock, nblocks) );
      SCIPhashmapFree(&constoblock);
   }
   else {
      SCIPhashmapFree(&constoblock);
      (*firstpartialdec) = NULL;
      (*secondpartialdec) = NULL;
   }

   SCIPfreeBufferArray(this->scip_, &nsubscipconss);
   return SCIP_OKAY;
}

template <class T>
SCIP_RETCODE HyperrowcolGraph<T>::readPartition(
   const char*        filename            /**< filename where the partition is stored */
   )
{

   ifstream input(filename);
   if( !input.good() )
   {
      SCIPerrorMessage("Could not open file <%s> for reading\n", filename);
      return SCIP_READERROR;
   }
   for( int i = 0; i < this->nnonzeroes; ++i )
   {
      int part = 0;
      if( !(input >> part) )
      {
         SCIPerrorMessage("Could not read from file <%s>. It may be in the wrong format\n", filename);
         return SCIP_READERROR;
      }
      graph.setPartition(i,part);
   }

   input.close();
   return SCIP_OKAY;

}

} /* namespace gcg */
#endif
