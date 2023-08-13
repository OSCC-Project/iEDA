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

/**@file   columngraph_def.h
 * @brief  A row graph where each column is a node and columns are adjacent if they appear in one row
 * @author Martin Bergner
 * @author Annika Thome
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_COLUMNGRAPH_DEF_H_
#define GCG_COLUMNGRAPH_DEF_H_

#include "columngraph.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace gcg {

template <class T>
ColumnGraph<T>::ColumnGraph(
   SCIP*                 scip,              /**< SCIP data structure */
   Weights               w                  /**< weights for the given graph */
   ) : MatrixGraph<T>(scip, w), graph(scip)
{
   this->graphiface = &graph;
   this->name = std::string("columngraph");
}

template <class T>
ColumnGraph<T>::~ColumnGraph()
{
   // TODO Auto-generated destructor stub
}

template <class T>
SCIP_RETCODE ColumnGraph<T>::createDecompFromPartition(
   DEC_DECOMP**         decomp
)
{
   int nblocks;
   SCIP_HASHMAP* constoblock = NULL;

   int* nsubscipconss = NULL;
   int i;
   SCIP_CONS **conss;
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
      SCIP_CALL( SCIPhashmapInsert(constoblock, conss[i], (void*) (size_t) (block +1)) );
      ++(nsubscipconss[block]);
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
SCIP_RETCODE ColumnGraph<T>::createFromMatrix(
   SCIP_CONS**           conss,              /**< constraints for which graph should be created */
   SCIP_VAR**            vars,               /**< variables for which graph should be created */
   int                   nconss_,             /**< number of constraints */
   int                   nvars_               /**< number of variables */
   )
{
   int i;
   int j;
   int k;
   SCIP_Bool success;
   std::pair< int, int> edge;
   std::vector< std::pair< int, int> > edges;

   assert(conss != NULL);
   assert(vars != NULL);
   assert(nvars_ > 0);
   assert(nconss_ > 0);

   this->nvars = nvars_;
   this->nconss = nconss_;

   /* go through all variables */
   for( i = 0; i < this->nvars; ++i )
   {
      TCLIQUE_WEIGHT weight;

      /* calculate weight of node */
      weight = this->weights.calculate(vars[i]);

      this->graph.addNode(i, weight);
   }

   /* go through all constraints */
   for( i = 0; i < this->nconss; ++i )
   {
      SCIP_VAR** curvars = NULL;

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
         SCIP_VAR* var1;
         int varIndex1;

         if( SCIPgetStage(this->scip_) >= SCIP_STAGE_TRANSFORMED)
            var1 = SCIPvarGetProbvar(curvars[j]);
         else
            var1 = curvars[j];

         if( !GCGisVarRelevant(var1) )
            continue;

         assert(var1 != NULL);
         varIndex1 = SCIPvarGetProbindex(var1);
         assert(varIndex1 >= 0);
         assert(varIndex1 < this->nvars);

         for( k = 0; k < j; ++k )
         {
            SCIP_VAR* var2;
            int varIndex2;

            if( SCIPgetStage(this->scip_) >= SCIP_STAGE_TRANSFORMED)
               var2 = SCIPvarGetProbvar(curvars[k]);
            else
               var2 = curvars[k];

            if( !GCGisVarRelevant(var2) )
               continue;

            assert(var2 != NULL);
            varIndex2 = SCIPvarGetProbindex(var2);
            assert(varIndex2 >= 0);
            assert(varIndex2 < this->nvars);

            edge = std::make_pair(MIN(varIndex1, varIndex2), MAX(varIndex1, varIndex2) );

            /* check if edge was not already added to graph */
            if(edges.end() == std::find(edges.begin(), edges.end(), edge) )
            {

               SCIP_CALL( this->graph.addEdge(varIndex1, varIndex2) );
               edges.push_back(edge);
               std::sort(edges.begin(), edges.end());
            }
         }
      }
      SCIPfreeBufferArray(this->scip_, &curvars);
   }

   this->graph.flush();
   return SCIP_OKAY;
}

} /* namespace gcg */

#endif
