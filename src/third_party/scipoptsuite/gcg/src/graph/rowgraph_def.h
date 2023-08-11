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

/**@file   rowgraph_def.h
 * @brief  A row graph where each row is a node and rows are adjacent if they share a variable
 * @author Martin Bergner
 * @author Annika Thome
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
// #define SCIP_DEBUG

#ifndef GCG_ROWGRAPH_DEF_H_
#define GCG_ROWGRAPH_DEF_H_

#include "rowgraph.h"
#include <algorithm>

namespace gcg {

template <class T>
RowGraph<T>::RowGraph(
   SCIP*                 scip,              /**< SCIP data structure */
   Weights               w                  /**< weights for the given graph */
   ) : MatrixGraph<T>(scip,w), graph(scip)
{
   this->graphiface = &graph;
   this->name = std::string("rowgraph");
}

template <class T>
RowGraph<T>::~RowGraph()
{
   // TODO Auto-generated destructor stub
}

template <class T>
SCIP_RETCODE RowGraph<T>::createDecompFromPartition(
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
      int block = partition[i];

      if(block == -1)
      {
         SCIP_CALL( SCIPhashmapInsert(constoblock, conss[i], (void*) (size_t) (nblocks +1)) );
      }
      else
      {  assert(block >= 0);
         assert(block < nblocks);
         SCIP_CALL( SCIPhashmapInsert(constoblock, conss[i], (void*) (size_t) (block +1)) );
         ++(nsubscipconss[block]);
      }

   }


   // TODO: remove- FOR DEBUG ONLY!!!
   std::vector<int> nsubscipconss_dbg(nblocks);

   /* first, make sure that there are constraints in every block, otherwise the hole thing is useless */
   for( i = 0; i < nblocks; ++i )
   {
      // TODO: remove- FOR DEBUG ONLY!!!
      nsubscipconss_dbg[i] = nsubscipconss[i];
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
SCIP_RETCODE RowGraph<T>::createPartialdecFromPartition(
   PARTIALDECOMP*      oldpartialdec,
   PARTIALDECOMP**     firstpartialdec,
   PARTIALDECOMP**     secondpartialdec,
   DETPROBDATA*  detprobdata
   )
{
   int nblocks;
   SCIP_HASHMAP* constoblock = NULL;

   int* nsubscipconss = NULL;
   int i;
   SCIP_Bool emptyblocks = FALSE;

   //fillout conssForGraph
   std::vector<int> conssForGraph; /** stores the conss included by the graph */
   std::vector<bool> conssBool(oldpartialdec->getNConss(), false); /**< true, if the cons will be part of the graph */
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

   std::vector<int> partition = graph.getPartition();
   nblocks = *(std::max_element(partition.begin(), partition.end()))+1;

   SCIP_CALL( SCIPallocBufferArray(this->scip_, &nsubscipconss, nblocks) );
   BMSclearMemoryArray(nsubscipconss, nblocks);

   SCIP_CALL( SCIPhashmapCreate(&constoblock, SCIPblkmem(this->scip_), this->nconss) );

   /* assign constraints to partition */
   for( i = 0; i < this->nconss; i++ )
   {
      int block = partition[i];

      if(block == -1)
      {
         SCIP_CALL( SCIPhashmapInsert(constoblock, (void*) (size_t) conssForGraph[i], (void*) (size_t) (nblocks +1)) );
      }
      else
      {  assert(block >= 0);
         assert(block < nblocks);
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
   }
   else
   {
      SCIPhashmapFree(&constoblock);
      if( firstpartialdec != NULL )
      {
         (*firstpartialdec) = NULL;
      }
      if( secondpartialdec != NULL )
      {
         (*secondpartialdec) = NULL;
      }
   }

   SCIPfreeBufferArray(this->scip_, &nsubscipconss);
   return SCIP_OKAY;
}

template <class T>
SCIP_RETCODE RowGraph<T>::createFromMatrix(
   SCIP_CONS**           conss,              /**< constraints for which graph should be created */
   SCIP_VAR**            vars,               /**< variables for which graph should be created */
   int                   nconss_,             /**< number of constraints */
   int                   nvars_               /**< number of variables */
   )
{
   int i;
   int j;
   int k;
   int l;
   SCIP_Bool success;

   std::pair< int, int> edge;
   std::vector< std::pair< int, int> > edges;

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

      /* go through all constraints */
      for( j = 0; j < i; ++j )
      {
         SCIP_VAR** curvars2 = NULL;
         SCIP_Bool continueloop;
         int ncurvars2;
         SCIP_CALL( SCIPgetConsNVars(this->scip_, conss[j], &ncurvars2, &success) );
         assert(success);
         if( ncurvars2 == 0 )
            continue;

         edge = std::make_pair(MIN(i, j), MAX(i, j) );

          /* check if edge was not already added to graph */
          if(edges.end() != std::find(edges.begin(), edges.end(), edge) )
             continue;

          /*if(this->graph.edge(i, j))
            continue;
           */
         continueloop = FALSE;
         /*
          * may work as is, as we are copying the constraint later regardless
          * if there are variables in it or not
          */
         SCIP_CALL( SCIPallocBufferArray(this->scip_, &curvars2, ncurvars2) );
         SCIP_CALL( SCIPgetConsVars(this->scip_, conss[j], curvars2, ncurvars2, &success) );
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

            for( l = 0; l < ncurvars2; ++l )
            {
               SCIP_VAR* var2;
               int varIndex2;

               if( SCIPgetStage(this->scip_) >= SCIP_STAGE_TRANSFORMED)
                  var2 = SCIPvarGetProbvar(curvars2[l]);
               else
                  var2 = curvars2[l];

               if( !GCGisVarRelevant(var2) )
                  continue;

               assert(var2 != NULL);
               varIndex2 = SCIPvarGetProbindex(var2);
               assert(varIndex2 >= 0);
               assert(varIndex2 < this->nvars);

               if(varIndex1 == varIndex2)
               {
                  SCIP_CALL( this->graph.addEdge(i, j) );

                  edges.push_back(edge);
                  std::sort(edges.begin(), edges.end());

                  /*
                   * this->graph.flush();
                   */

                  continueloop = TRUE;
                  break;
               }
            }
            if(continueloop)
               break;
         }
         SCIPfreeBufferArray(this->scip_, &curvars2);
      }
      SCIPfreeBufferArray(this->scip_, &curvars1);
   }
   this->graph.flush();

   return SCIP_OKAY;
}

} /* namespace gcg */
#endif
