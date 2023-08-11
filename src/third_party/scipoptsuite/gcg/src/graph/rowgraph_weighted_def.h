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

/**@file   rowgraph_weighted_def.h
 * @brief  A row graph where each row is a node and rows are adjacent if they share a variable.
 *         The edges are weighted according to specified similarity measure.
 * @author Igor Pesic
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
// #define SCIP_DEBUG

#ifndef GCG_ROWGRAPH_WEIGHTED_DEF_H_
#define GCG_ROWGRAPH_WEIGHTED_DEF_H_

#include "rowgraph_weighted.h"
#include "graph_gcg.h"
#include "graphalgorithms.h"
#include "priority_graph.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <climits>
#include <queue>

using namespace std;

namespace gcg {

template <class T>
RowGraphWeighted<T>::RowGraphWeighted(
   SCIP*                 scip,              /**< SCIP data structure */
   Weights               w                  /**< weights for the given graph */
   ) : RowGraph<T>(scip,w)
{
   this->name = string("rowgraph_weighted");
   n_blocks = -1;
   non_cl = INT_MAX;
}

template <class T>
RowGraphWeighted<T>::~RowGraphWeighted()
{
   // Auto-generated destructor stub
}


template <class T>
SCIP_RETCODE RowGraphWeighted<T>::createFromMatrix(
   SCIP_CONS**           conss,              /**< constraints for which graph should be created */
   SCIP_VAR**            vars,               /**< variables for which graph should be created */
   int                   nconss_,            /**< number of constraints */
   int                   nvars_,             /**< number of variables */
   DISTANCE_MEASURE      dist,               /**< Here we define the distance measure between two rows */
   WEIGHT_TYPE           w_type             /**< Depending on the algorithm we can build distance or similarity graph */
   )
{
   int i;
   int j;
   int k;
   int l;
   SCIP_Bool success;

   assert(conss != NULL);
   assert(vars != NULL);
   assert(nvars_ > 0);
   assert(nconss_ > 0);

   this->nvars = nvars_;
   this->nconss = nconss_;

   SCIP_CALL( this->graph.addNNodes(this->nconss) );

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

      /* go through all constraints again */
      for( j = 0; j < i; ++j )
      {
         SCIP_VAR** curvars2 = NULL;
         int ncurvars2;
         SCIP_CALL( SCIPgetConsNVars(this->scip_, conss[j], &ncurvars2, &success) );
         assert(success);
         if( ncurvars2 == 0 )
            continue;


         /*
          * may work as is, as we are copying the constraint later regardless
          * if there are variables in it or not
          */
         SCIP_CALL( SCIPallocBufferArray(this->scip_, &curvars2, ncurvars2) );
         SCIP_CALL( SCIPgetConsVars(this->scip_, conss[j], curvars2, ncurvars2, &success) );
         assert(success);


         /** @todo skip all variables that have a zero coeffient or where all coefficients add to zero */
         /** @todo Do more then one entry per variable actually work? */

         int a = 0;   // number of common variables
         int b = 0;   // number of variables that appear ONLY in the second row
         int c = 0;   // number of variables that appear ONLY in the first row
         for( k = 0; k < ncurvars1; ++k )
         {
            SCIP_VAR* var1 = NULL;
            int varIndex1;

            if( !GCGisVarRelevant(curvars1[k]) )
               continue;

            if( SCIPgetStage(this->scip_) >= SCIP_STAGE_TRANSFORMED)
               var1 = SCIPvarGetProbvar(curvars1[k]);
            else
               var1 = curvars1[k];

            assert(var1 != NULL);
            varIndex1 = SCIPvarGetProbindex(var1);
            assert(varIndex1 >= 0);
            assert(varIndex1 < this->nvars);

            for( l = 0; l < ncurvars2; ++l )
            {
               SCIP_VAR* var2 = NULL;
               int varIndex2;

               if( !GCGisVarRelevant(curvars2[l]) )
                  continue;

               if( SCIPgetStage(this->scip_) >= SCIP_STAGE_TRANSFORMED)
                  var2 = SCIPvarGetProbvar(curvars2[l]);
               else
                  var2 = curvars2[l];

               assert(var2 != NULL);
               varIndex2 = SCIPvarGetProbindex(var2);
               assert(varIndex2 >= 0);
               assert(varIndex2 < this->nvars);

               if(varIndex1 == varIndex2)
               {
                  a++;
                  break;      // stop comparing the variable from the 1st const. with the rest of vars. in the 2nd const.
               }
            }
         }
         b = ncurvars2 - a;
         c = ncurvars1 - a;
         assert(a >= 0);   // number of common var
         assert(b >= 0);   // number of variables in the second conss
         assert(c >= 0);   // number of variables in the first conss
         if(a != 0){
            double edge_weight = calculateSimilarity(a, b, c, dist, w_type, i==j);
            this->graph.addEdge(i, j, edge_weight);
         }

         SCIPfreeBufferArray(this->scip_, &curvars2);
      }
      SCIPfreeBufferArray(this->scip_, &curvars1);
   }

   if(dist == INTERSECTION)
   {
      this->graph.normalize();
      if(w_type == DIST)
      {
         return SCIP_INVALIDCALL;
      }
   }

   SCIP_CALL( this->graph.flush() );

   assert(this->graph.getNNodes() == nconss_);

   return SCIP_OKAY;
}

template <class T>
SCIP_RETCODE RowGraphWeighted<T>::createFromPartialMatrix(
   DETPROBDATA*            detprobdata,
   PARTIALDECOMP*                partialdec,
   DISTANCE_MEASURE      dist,               /**< Here we define the distance measure between two rows */
   WEIGHT_TYPE           w_type             /**< Depending on the algorithm we can build distance or similarity graph */
   )
{

   int i;
   int j;
   int k;
   int l;
   int m;
   vector<int> conssForGraph; /** stores the conss included by the graph */
   vector<int> varsForGraph; /** stores the vars included by the graph */
   vector<bool> varsBool(partialdec->getNVars(), false); /**< true, if the var will be part of the graph */
   vector<bool> conssBool(partialdec->getNConss(), false); /**< true, if the cons will be part of the graph */
   unordered_map<int, int> oldToNewConsIndex; /** stores new index of the conss */
   unordered_map<int, int> oldToNewVarIndex; /** stores new index of the vars */

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

   this->nvars = (int)varsForGraph.size();
   this->nconss = (int)conssForGraph.size();
   assert(this->nconss > 0);
   assert(this->nvars > 0);

   for(int v = 0; v < (int)varsForGraph.size(); ++v)
   {
      int oldVarId = varsForGraph[v];
      oldToNewVarIndex.insert({oldVarId,v});
   }
   for(int c = 0; c < (int)conssForGraph.size(); ++c)
   {
      int oldConsId = conssForGraph[c];
      oldToNewConsIndex.insert({oldConsId,c});
   }



   SCIP_CALL( this->graph.addNNodes(this->nconss) );

   /* go through all constraints */
   for( i = 0; i < this->nconss; ++i )
   {
      int cons1 = conssForGraph[i];
      assert(conssBool[cons1]);

      /* go through all constraints again */
      for( j = 0; j < i; ++j )
      {
         int cons2 = conssForGraph[j];

         int a = 0;   // number of common variables
         int b = 0;   // number of variables that appear ONLY in the second row
         int c = 0;   // number of variables that appear ONLY in the first row


         for( k = 0; k < detprobdata->getNVarsForCons(cons1); ++k)
         {
            int var1 = detprobdata->getVarsForCons(cons1)[k];
            if(!varsBool[var1])
               continue;
            assert(varsBool[var1]);

            for(l = 0; l < detprobdata->getNVarsForCons(cons2); ++l)
            {
               int var2 = detprobdata->getVarsForCons(cons2)[l];
               if(!varsBool[var2])
                  continue;

               if(var1 == var2)
               {
                  a++;
                  break;   // stop comparing the variable from the 1st const. with the rest of vars. in the 2nd const.
               }
            }
            for(m = 0; m < detprobdata->getNVarsForCons(cons2); ++m)
            {
               int var = detprobdata->getVarsForCons(cons2)[m];
               if(varsBool[var])
                  b++;
            }
            b = b - a;

            for(m = 0; m < detprobdata->getNVarsForCons(cons1); ++m)
            {
               int var = detprobdata->getVarsForCons(cons1)[m];
               if(varsBool[var])
                  c++;
            }
            c = c - a;

            assert(a >= 0);   // number of common var
            assert(b >= 0);   // number of variables in the second conss
            assert(c >= 0);   // number of variables in the first conss
            if(a != 0){
               double edge_weight = calculateSimilarity(a, b, c, dist, w_type, i==j);
               this->graph.addEdge(oldToNewConsIndex[cons1], oldToNewConsIndex[cons2], edge_weight);
            }
         }
      }
   }

   if(dist == INTERSECTION)
   {
      this->graph.normalize();
      if(w_type == DIST)
      {
         return SCIP_INVALIDCALL;
      }
   }

   SCIP_CALL( this->graph.flush() );

   return SCIP_OKAY;
}

template <class T>
double RowGraphWeighted<T>::calculateSimilarity(int _a, int _b, int _c, DISTANCE_MEASURE dist, WEIGHT_TYPE w_type, bool itself)
{
   if(w_type == DIST )
   {
      if(_c == 0 && _b == 0) return 1e-10;
   }
   else
   {
      if(_c == 0 && _b == 0) return (1.0 - 1e-10);
   }
   double result = 0.0;
   double a = (double)_a;
   double b = (double)_b;
   double c = (double)_c;
   switch( dist ) {
   case JOHNSON:
      result = (a/(a+b) + a/(a+c)) / 2.0;
      break;
   case INTERSECTION:
      result = a;
      break;
   case JACCARD:
      result = a / (a+b+c);
      break;
   case COSINE:
      result = a / (sqrt(a+b)*sqrt(a+c));
      break;
   case SIMPSON:
      result = a / MIN(a+b, a+c);
      break;
   }



   if(dist != INTERSECTION)
   {
      assert(result >= 0.0);
      assert(result <= 1.0);
   }

   if(!itself)
   {
      result *= (1 - 1e-10);
   }

   if(w_type == DIST && dist != INTERSECTION)
   {
      result = 1.0 - result;
   }

   return result;
}

template <>
SCIP_RETCODE RowGraphWeighted<GraphGCG>::postProcess(vector<int>& labels, bool enabled)
{
   assert((int)labels.size() == graph.getNNodes());
   set<int> diff_blocks_beginning;
   for(auto curr_int = labels.begin(), end = labels.end(); curr_int != end; ++curr_int)
   {
      diff_blocks_beginning.insert(*curr_int);
   }
   //std::cout << "diff_blocks_beginning: " << diff_blocks_beginning.size() << std::endl;
   bool skip_me = false;
   if(diff_blocks_beginning.size() == labels.size())
   {
      skip_me = true;
   }
   // If the post processing is enabled, remove the coliding conss, otherwise just set the partition
   if(enabled && !skip_me)
   {
      this->non_cl = (int)labels.size();
      // for each column in the conss matrix we save labels of all the constraints where the variable appears
      vector< vector<int> > all_labels_in_col(this->nvars);

      // for each column in the conss matrix we count the number of occurrences of each label
      vector< map<int, int> > all_label_occ_in_col(this->nvars);

      // item i saves number which appears most often in the all_labels_in_col[i]
      vector<int> col_labels(this->nvars, -1);

      SCIP_CONS** conss = SCIPgetConss(this->scip_);
      SCIP_Bool success;

      // For each var save the labels of all the constraints where this var appears.
      for(auto i = 0; i < this->nconss; i++)
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

         for(auto j = 0; j < ncurvars; j++)
         {
            SCIP_VAR* var;

            if( !GCGisVarRelevant(curvars[j]) )
               continue;

            if( SCIPgetStage(this->scip_) >= SCIP_STAGE_TRANSFORMED)
               var = SCIPvarGetProbvar(curvars[j]);
            else
               var = curvars[j];

            assert(var != NULL);
            int varIndex = SCIPvarGetProbindex(var);
            assert(varIndex >= 0);
            assert(varIndex < this->nvars);

            // push the label of the constraint to the columns where the variable appears
            all_labels_in_col[varIndex].push_back(labels[i]);
            all_label_occ_in_col[varIndex][labels[i]]++;
         }
         SCIPfreeBufferArray(this->scip_, &curvars);
      }

      // fill the col_labels
      for(auto i = 0; i < this->nvars; i++)
      {
         auto pr = max_element
         (
             all_label_occ_in_col[i].begin(), all_label_occ_in_col[i].end(),
             [] (const pair<int,int> & p1, const pair<int,int> & p2) {
                 return p1.second < p2.second;
             }
         );
         // new code
         col_labels[i] = pr->first;
      }

      // Iterate all the conss and remove them (i.e. set label to -1) if necessary
      for(auto i = 0; i < this->nconss; i++)
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

         for(auto j = 0; j < ncurvars1; j++)
         {
            SCIP_VAR* var1;

            if( !GCGisVarRelevant(curvars1[j]) )
               continue;

            if( SCIPgetStage(this->scip_) >= SCIP_STAGE_TRANSFORMED)
               var1 = SCIPvarGetProbvar(curvars1[j]);
            else
               var1 = curvars1[j];

            assert(var1 != NULL);
            int varIndex = SCIPvarGetProbindex(var1);
            assert(varIndex >= 0);
            assert(varIndex < this->nvars);

            // Check if in a conss we have found a var with different label than the conss label.
            // This means that this var is mostly belonging to some other block, so remove it
            if(col_labels[varIndex] != labels[i])
            {
               labels[i] = -1;

               // this is new part
               //       it updates the column labels each time after eliminating some conss
               all_label_occ_in_col[varIndex][labels[i]]--;
               auto pr = max_element
               (
                   all_label_occ_in_col[varIndex].begin(), all_label_occ_in_col[varIndex].end(),
                   [] (const pair<int,int> & p1, const pair<int,int> & p2) {
                       return p1.second < p2.second;
                   }
               );
               col_labels[varIndex] = pr->first;
            }
         }
         SCIPfreeBufferArray(this->scip_, &curvars1);

      }
   }

   if(!skip_me)
   {
      // Fix the labeling so that it starts from 0 without skipping the enumeration order
      // And set the graph partition accordingly.
      set<int> diff_blocks;

      for(auto curr_int = labels.begin(), end = labels.end(); curr_int != end; ++curr_int)
      {
         diff_blocks.insert(*curr_int);
      }

      const int non_cl_pts = count(labels.begin(), labels.end(), -1);
      if(non_cl_pts > 0)
      {
         this->n_blocks = diff_blocks.size() - 1;
      }
      else
      {
         this->n_blocks = diff_blocks.size();
      }
      this->non_cl = non_cl_pts;

      map<int,int> labels_fix;
      int new_label;
      if(this->non_cl > 0) new_label = -1;
      else                 new_label =  0;

      for(int old_label: diff_blocks)
      {
         labels_fix[old_label] = new_label;
         new_label++;
      }
      for(int i = 0; i < (int)labels.size(); i++)
      {
         labels[i] = labels_fix[labels[i]];
      }
   }

   // put the labels as the partition...
   for(int i = 0; i < (int)labels.size(); i++)
   {
      this->graph.setPartition(i, labels[i]);
   }
   return SCIP_OKAY;
}

template <>
SCIP_RETCODE RowGraphWeighted<GraphGCG>::postProcessForPartialGraph(gcg::DETPROBDATA* detprobdata, gcg::PARTIALDECOMP* partialdec, vector<int>& labels, bool enabled)
{
   assert((int)labels.size() == graph.getNNodes());
   set<int> diff_blocks_beginning;
   for(auto curr_int = labels.begin(), end = labels.end(); curr_int != end; ++curr_int)
   {
      diff_blocks_beginning.insert(*curr_int);
   }
   bool skip_me = false;
   if(diff_blocks_beginning.size() == labels.size())
   {
      skip_me = true;
   }
   // If the post processing is enabled, remove the coliding conss, otherwise just set the partition
   if(enabled && !skip_me)
    {
      //fillout conssForGraph and varsForGraph
      vector<int> conssForGraph; /* stores the conss included by the graph */
      vector<int> varsForGraph; /* stores the vars included by the graph */
      vector<bool> varsBool(partialdec->getNVars(), false); /* true, if the var will be part of the graph */
      vector<bool> conssBool(partialdec->getNConss(), false); /* true, if the cons will be part of the graph */
      unordered_map<int, int> oldToNewConsIndex; /* stores new index of the conss */
      unordered_map<int, int> oldToNewVarIndex; /* stores new index of the vars */

      for(int c = 0; c < partialdec->getNOpenconss(); ++c)
      {
         int cons = partialdec->getOpenconss()[c];
         for(int v = 0; v < partialdec->getNOpenvars(); ++v)
         {
            int var = partialdec->getOpenvars()[v];
            for(int i = 0; i < detprobdata->getNVarsForCons(cons); ++i)
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
         if(varsBool[var] == true)
            varsForGraph.push_back(var);
      }
      for(int c = 0; c < partialdec->getNOpenconss(); ++c)
      {
         int cons = partialdec->getOpenconss()[c];
         if(conssBool[cons] == true)
            conssForGraph.push_back(cons);
      }

       assert(this->nvars == (int) varsForGraph.size());
       assert(this->nconss == (int) conssForGraph.size());

       sort(conssForGraph.begin(), conssForGraph.end());
       sort(varsForGraph.begin(), varsForGraph.end());

       //fillout oldToNewVarIndex and oldToNewConsIndex
       for(int v = 0; v < this->nvars; ++v)
       {
          int oldVarId = varsForGraph[v];
          oldToNewVarIndex.insert({oldVarId,v});
       }

       for(int c = 0; c < this->nconss; ++c)
       {
          int oldConsId = conssForGraph[c];
          oldToNewConsIndex.insert({oldConsId,c});
       }

       this->non_cl = (int)labels.size();
       // for each column in the conss matrix we save labels of all the constraints where the variable appears
       vector< vector<int> > all_labels_in_col(this->nvars);

       // for each column in the conss matrix we count the number of occurrences of each label
       vector< map<int, int> > all_label_occ_in_col(this->nvars);

       // item i saves number which appears most often in the all_labels_in_col[i]
       vector<int> col_labels(this->nvars, -1);


       // For each var save the labels of all the constraints where this var appears.
       for(size_t c = 0; c < conssForGraph.size(); c++)
       {
          int cons = conssForGraph[c];
          int consIndex = oldToNewConsIndex[cons];
          assert(consIndex >= 0);
          assert(consIndex < this->nconss);
          for(int v = 0; v < detprobdata->getNVarsForCons(cons); ++v)
          {
             int var = detprobdata->getVarsForCons(cons)[v];
             if(find(varsForGraph.begin(), varsForGraph.end(), var) == varsForGraph.end())
                continue;
             int varIndex = oldToNewVarIndex[var];
             assert(varIndex >= 0);
             assert(varIndex < this->nvars);
             all_labels_in_col[varIndex].push_back(labels[consIndex]);
          }
       }

       // fill the col_labels
       for(size_t v = 0; v < varsForGraph.size(); ++v)
       {
          int var = varsForGraph[v];
          int varIndex = oldToNewVarIndex[var];
          assert(varIndex >= 0);
          assert(varIndex < this->nvars);

          auto pr = max_element
          (
              all_label_occ_in_col[varIndex].begin(), all_label_occ_in_col[varIndex].end(),
              [] (const pair<int,int> & p1, const pair<int,int> & p2) {
                  return p1.second < p2.second;
              }
          );
          col_labels[varIndex] = pr->first;
       }


       // Iterate all the conss and remove them (i.e. set label to -1) if necessary
       for(size_t c = 0; c < conssForGraph.size(); ++c)
       {
          int cons = conssForGraph[c];
          assert(cons >= 0);
          int consIndex = oldToNewConsIndex[cons];
          assert(consIndex >= 0);
          assert(consIndex < this->nconss);
          for(int v = 0; v < detprobdata->getNVarsForCons(cons); ++v)
          {
             int var = detprobdata->getVarsForCons(cons)[v];
             if(find(varsForGraph.begin(), varsForGraph.end(), var) == varsForGraph.end())
                continue;
             int varIndex = oldToNewVarIndex[var];
             assert(varIndex >= 0);
             assert(varIndex < this->nvars);
             // Check if in a conss we have found a var with different label than the conss label.
             // This means that this var is mostly belonging to some other block, so remove it
             if(col_labels[varIndex] != labels[consIndex])
             {
                labels.at(consIndex) = -1;

                // this is new part
                //       it updates the column labels each time after eliminating some conss
                all_label_occ_in_col[varIndex][labels[consIndex]]--;
                auto pr = max_element
                (
                    all_label_occ_in_col[varIndex].begin(), all_label_occ_in_col[varIndex].end(),
                    [] (const pair<int,int> & p1, const pair<int,int> & p2) {
                        return p1.second < p2.second;
                    }
                );
                col_labels[varIndex] = pr->first;
             }
          }
       }
    }

   if(!skip_me)
   {
      // Fix the labeling so that it starts from 0 without skipping the enumeration order
      // And set the graph partition accordingly.
      set<int> diff_blocks;

      for(auto curr_int = labels.begin(), end = labels.end(); curr_int != end; ++curr_int)
      {
         diff_blocks.insert(*curr_int);
      }

      const int non_cl_pts = count(labels.begin(), labels.end(), -1);
      if(non_cl_pts > 0)
      {
         this->n_blocks = diff_blocks.size() - 1;
      }
      else
      {
         this->n_blocks = diff_blocks.size();
      }
      this->non_cl = non_cl_pts;

      map<int,int> labels_fix;
      int new_label;
      if(this->non_cl > 0) new_label = -1;
      else                 new_label =  0;

      for(int old_label: diff_blocks)
      {
         labels_fix[old_label] = new_label;
         new_label++;
      }
      for(int i = 0; i < (int)labels.size(); i++)
      {
         labels[i] = labels_fix[labels[i]];
      }
   }

   // put the labels as the partition...
   for(int i = 0; i < (int)labels.size(); i++)
   {
      this->graph.setPartition(i, labels[i]);
   }
   return SCIP_OKAY;
}


// this function is obsolete
template <>
SCIP_RETCODE RowGraphWeighted<GraphGCG>::postProcessStableSet(vector<int>& labels, bool enabled)
{
   assert((int)labels.size() == graph.getNNodes());
   set<int> diff_blocks_beginning;
   for(auto curr_int = labels.begin(), end = labels.end(); curr_int != end; ++curr_int)
   {
      diff_blocks_beginning.insert(*curr_int);
   }
   bool skip_me = false;
   if(diff_blocks_beginning.size() == labels.size())
   {
      skip_me = true;
   }


   // If the post processing is enabled, remove the coliding conss, otherwise just set the partition
   if(enabled && !skip_me)
   {
      priority_graph stable_set_graph;
      //SCIPverbMessage(this->scip_, SCIP_VERBLEVEL_NORMAL, NULL, " running postProcessStableSet\n");
      this->non_cl = (int)labels.size();
      // for each column in the conss matrix we save labels of all the constraints where the variable appears
      vector< vector<int> > all_ind_in_col(this->nvars);


      SCIP_CONS** conss = SCIPgetConss(this->scip_);
      SCIP_Bool success;

      /* go through all constraints */
      for(int i = 0; i < this->nconss; ++i )
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
         for(int j = 0; j < i; ++j )
         {
            if(labels[i] == labels[j]) continue;
            SCIP_VAR** curvars2 = NULL;
            SCIP_Bool continueloop;
            int ncurvars2;
            SCIP_CALL( SCIPgetConsNVars(this->scip_, conss[j], &ncurvars2, &success) );
            assert(success);
            if( ncurvars2 == 0 )
               continue;

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

            for(int k = 0; k < ncurvars1; ++k )
            {
               SCIP_VAR* var1 = NULL;
               int varIndex1;

               if( !GCGisVarRelevant(curvars1[k]) )
                  continue;

               if( SCIPgetStage(this->scip_) >= SCIP_STAGE_TRANSFORMED)
                  var1 = SCIPvarGetProbvar(curvars1[k]);
               else
                  var1 = curvars1[k];

               assert(var1 != NULL);
               varIndex1 = SCIPvarGetProbindex(var1);
               assert(varIndex1 >= 0);
               assert(varIndex1 < this->nvars);

               for(int l = 0; l < ncurvars2; ++l )
               {
                  SCIP_VAR* var2 = NULL;
                  int varIndex2;

                  if( !GCGisVarRelevant(curvars2[l]) )
                     continue;

                  if( SCIPgetStage(this->scip_) >= SCIP_STAGE_TRANSFORMED)
                     var2 = SCIPvarGetProbvar(curvars2[l]);
                  else
                     var2 = curvars2[l];

                  assert(var2 != NULL);
                  varIndex2 = SCIPvarGetProbindex(var2);
                  assert(varIndex2 >= 0);
                  assert(varIndex2 < this->nvars);

                  if(varIndex1 == varIndex2)
                  {
                     stable_set_graph.addNode(i);
                     stable_set_graph.addNode(j);
                     stable_set_graph.addEdge(i, j);

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

      // run greedy heuristic for stable set
      vector<int> stable_set;
      vector<int> no_stable_set;
      while(!stable_set_graph.empty()){
         int current = stable_set_graph.top().first;
         stable_set.push_back(current);
         set<int> neighbors = stable_set_graph.getNeighbors(current);
         stable_set_graph.pop();
         for(auto neighbor : neighbors){
            stable_set_graph.removeNode(neighbor, no_stable_set);
         }
      }


      // Iterate all the conss and remove them (i.e. set label to -1) if necessary
      for(int to_remove : no_stable_set)
      {
         //SCIPverbMessage(this->scip_, SCIP_VERBLEVEL_NORMAL, NULL, " to_remove: %d \n", to_remove);
         labels[to_remove] = -1;
      }
   }


   if(!skip_me)
   {
      // Fix the labeling so that it starts from 0 without skipping the enumeration order
      // And set the graph partition accordingly.
      set<int> diff_blocks;
      for(auto curr_int = labels.begin(), end = labels.end(); curr_int != end; ++curr_int)
      {
         diff_blocks.insert(*curr_int);
      }

      const int non_cl_pts = count(labels.begin(), labels.end(), -1);
      if(non_cl_pts > 0)
      {
         this->n_blocks = diff_blocks.size() - 1;
      }
      else
      {
         this->n_blocks = diff_blocks.size();
      }
      this->non_cl = non_cl_pts;

      map<int,int> labels_fix;
      int new_label;
      if(this->non_cl > 0) new_label = -1;
      else                 new_label =  0;

      for(int old_label: diff_blocks)
      {
         labels_fix[old_label] = new_label;
         new_label++;
      }
      for(int i = 0; i < (int)labels.size(); i++)
      {
         labels[i] = labels_fix[labels[i]];
      }
   }

   // put the labels as the partition...
   for(int i = 0; i < (int)labels.size(); i++)
   {
      this->graph.setPartition(i, labels[i]);
   }
   return SCIP_OKAY;
}

// this function is obsolete
template <>
SCIP_RETCODE RowGraphWeighted<GraphGCG>::postProcessStableSetForPartialGraph(gcg::DETPROBDATA* detprobdata, gcg::PARTIALDECOMP* partialdec, vector<int>& labels, bool enabled)
{
   assert((int)labels.size() == graph.getNNodes());
   set<int> diff_blocks_beginning;
   for(auto curr_int = labels.begin(), end = labels.end(); curr_int != end; ++curr_int)
   {
      diff_blocks_beginning.insert(*curr_int);
   }
   bool skip_me = false;
   if(diff_blocks_beginning.size() == labels.size())
   {
      skip_me = true;
   }



   // If the post processing is enabled, remove the coliding conss, otherwise just set the partition
   if(enabled && !skip_me)
   {
      //fillout conssForGraph and varsForGraph
      vector<int> conssForGraph; /** stores the conss included by the graph */
      vector<int> varsForGraph; /** stores the vars included by the graph */
      vector<bool> varsBool(partialdec->getNVars(), false); /**< true, if the var will be part of the graph */
      vector<bool> conssBool(partialdec->getNConss(), false); /**< true, if the cons will be part of the graph */
      unordered_map<int, int> oldToNewConsIndex; /** stores new index of the conss */
      unordered_map<int, int> oldToNewVarIndex; /** stores new index of the vars */

      for(int c = 0; c < partialdec->getNOpenconss(); ++c)
      {
         int cons = partialdec->getOpenconss()[c];
         for(int v = 0; v < partialdec->getNOpenvars(); ++v)
         {
            int var = partialdec->getOpenvars()[v];
            for(int i = 0; i < detprobdata->getNVarsForCons(cons); ++i)
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
         if(varsBool[var] == true)
            varsForGraph.push_back(var);
      }
      for(int c = 0; c < partialdec->getNOpenconss(); ++c)
      {
         int cons = partialdec->getOpenconss()[c];
         if(conssBool[cons] == true)
            conssForGraph.push_back(cons);
      }

      assert(this->nvars == (int) varsForGraph.size());
      assert(this->nconss == (int) conssForGraph.size());

      sort(conssForGraph.begin(), conssForGraph.end());
      sort(varsForGraph.begin(), varsForGraph.end());

      //fillout oldToNewVarIndex and oltToNewConsIndex
      for(int v = 0; v < this->nvars; ++v)
      {
         int oldVarId = varsForGraph[v];
         oldToNewVarIndex.insert({oldVarId,v});
      }

      for(int c = 0; c < this->nconss; ++c)
      {
         int oldConsId = conssForGraph[c];
         oldToNewConsIndex.insert({oldConsId,c});
      }

      priority_graph stable_set_graph;
      this->non_cl = (int)labels.size();
      // for each column in the conss matrix we save labels of all the constraints where the variable appears
      vector< vector<int> > all_ind_in_col(this->nvars);

      /* go through all constraints */
      for(int c = 0; c < this->nconss; ++c)
      {
         int cons1 = conssForGraph[c];
         int consIndex1 = oldToNewConsIndex[cons1];
         assert(consIndex1 >= 0);
         assert(consIndex1 < this->nconss);

         /*
          * may work as is, as we are copying the constraint later regardless
          * if there are variables in it or not
          */

         /* go through all constraints */
         for(int d = 0; d < c; ++d )
         {
            int cons2 = conssForGraph[d];
            int consIndex2 = oldToNewConsIndex[cons2];
            assert(consIndex2 >= 0);
            assert(consIndex2 < this->nconss);
            if(labels[consIndex1] == labels[consIndex2])
               continue;
            SCIP_Bool continueloop = FALSE;
            /*
             * may work as is, as we are copying the constraint later regardless
             * if there are variables in it or not
             */

            /** @todo skip all variables that have a zero coeffient or where all coefficients add to zero */
            /** @todo Do more then one entry per variable actually work? */
            for(int v = 0; v < detprobdata->getNVarsForCons(cons1); ++v)
            {
               int var1 = detprobdata->getVarsForCons(cons1)[v];
               if(find(varsForGraph.begin(), varsForGraph.end(), var1) == varsForGraph.end())
                  continue;
               int varIndex1 = oldToNewVarIndex[var1];
               assert(varIndex1 >= 0);
               assert(varIndex1 < this->nvars);
               for(int w = 0; w < detprobdata->getNVarsForCons(cons2); ++w)
               {
                  int var2 = detprobdata->getVarsForCons(cons2)[w];
                     if(find(varsForGraph.begin(), varsForGraph.end(), var2) == varsForGraph.end())
                        continue;
                     int varIndex2 = oldToNewVarIndex[var2];
                     assert(varIndex2 >= 0);
                     assert(varIndex2 < this->nvars);
                  if(varIndex1 == varIndex2)
                  {
                     stable_set_graph.addNode(consIndex1);
                     stable_set_graph.addNode(consIndex2);
                     stable_set_graph.addEdge(consIndex1, consIndex2);
                     continueloop = TRUE;
                     break;
                  }
               }
               if(continueloop)
                  break;
            }
         }
      }

      // run greedy heuristic for stable set
      vector<int> stable_set;
      vector<int> no_stable_set;
      while(!stable_set_graph.empty()){
         int current = stable_set_graph.top().first;
         stable_set.push_back(current);
         set<int> neighbors = stable_set_graph.getNeighbors(current);
         stable_set_graph.pop();
         for(auto neighbor : neighbors){
            stable_set_graph.removeNode(neighbor, no_stable_set);
         }
      }


      // Iterate all the conss and remove them (i.e. set label to -1) if necessary
      for(int to_remove : no_stable_set)
      {
         labels[to_remove] = -1;
      }
   }


   if(!skip_me)
   {
      // Fix the labeling so that it starts from 0 without skipping the enumeration order
      // And set the graph partition accordingly.
      set<int> diff_blocks;
      for(auto curr_int = labels.begin(), end = labels.end(); curr_int != end; ++curr_int)
      {
         diff_blocks.insert(*curr_int);
      }

      const int non_cl_pts = count(labels.begin(), labels.end(), -1);
      if(non_cl_pts > 0)
      {
         this->n_blocks = diff_blocks.size() - 1;
      }
      else
      {
         this->n_blocks = diff_blocks.size();
      }
      this->non_cl = non_cl_pts;

      map<int,int> labels_fix;
      int new_label;
      if(this->non_cl > 0) new_label = -1;
      else                 new_label =  0;

      for(int old_label: diff_blocks)
      {
         labels_fix[old_label] = new_label;
         new_label++;
      }
      for(int i = 0; i < (int)labels.size(); i++)
      {
         labels[i] = labels_fix[labels[i]];
      }
   }

   // put the labels as the partition...
   for(int i = 0; i < (int)labels.size(); i++)
   {
      this->graph.setPartition(i, labels[i]);
   }
   return SCIP_OKAY;
}

template <>
SCIP_RETCODE RowGraphWeighted<GraphGCG>::computePartitionDBSCAN(double eps, bool postprocenable)
{
   vector<int> labels;
   labels = GraphAlgorithms<GraphGCG>::dbscan(graph, eps);
   assert((int)labels.size() == graph.getNNodes());

   SCIP_CALL( postProcess(labels, postprocenable) );
   return SCIP_OKAY;
}

template <>
SCIP_RETCODE RowGraphWeighted<GraphGCG>::computePartitionDBSCANForPartialGraph(gcg::DETPROBDATA* detprobdata, gcg::PARTIALDECOMP* partialdec, double eps, bool postprocenable)
{
   vector<int> labels;
   labels = GraphAlgorithms<GraphGCG>::dbscan(graph, eps);
   assert((int)labels.size() == graph.getNNodes());

   SCIP_CALL( postProcessForPartialGraph(detprobdata, partialdec, labels, postprocenable) );
   return SCIP_OKAY;
}


template <>
SCIP_RETCODE RowGraphWeighted<GraphGCG>::computePartitionMST(double eps, bool postprocenable)
{
   vector<int> labels;
   labels = GraphAlgorithms<GraphGCG>::mst(graph, eps);
   assert((int)labels.size() == graph.getNNodes());

   SCIP_CALL( postProcess(labels, postprocenable) );
   return SCIP_OKAY;
}

template <>
SCIP_RETCODE RowGraphWeighted<GraphGCG>::computePartitionMSTForPartialGraph(gcg::DETPROBDATA* detprobdata, gcg::PARTIALDECOMP* partialdec, double eps, bool postprocenable)
{
   vector<int> labels;
   labels = GraphAlgorithms<GraphGCG>::mst(graph, eps);
   assert((int)labels.size() == graph.getNNodes());

   SCIP_CALL( postProcessForPartialGraph(detprobdata, partialdec, labels, postprocenable) );
   return SCIP_OKAY;
}

template <>
SCIP_RETCODE RowGraphWeighted<GraphGCG>::computePartitionMCL(int& stoppedAfter, double inflatefactor, bool postprocenable)
{
   vector<int> labels;
   labels = GraphAlgorithms<GraphGCG>::mcl(graph, stoppedAfter, inflatefactor);
   assert((int)labels.size() == graph.getNNodes());

   SCIP_CALL( postProcess(labels, postprocenable) );
   return SCIP_OKAY;
}

template <>
SCIP_RETCODE RowGraphWeighted<GraphGCG>::computePartitionMCLForPartialGraph(gcg::DETPROBDATA* detprobdata, gcg::PARTIALDECOMP* partialdec, int& stoppedAfter, double inflatefactor, bool postprocenable)
{
   vector<int> labels;
   labels = GraphAlgorithms<GraphGCG>::mcl(graph, stoppedAfter, inflatefactor);
   assert((int)labels.size() == graph.getNNodes());

   SCIP_CALL( postProcessForPartialGraph(detprobdata, partialdec, labels, postprocenable) );
   return SCIP_OKAY;
}

template <class T>
SCIP_RETCODE RowGraphWeighted<T>::getNBlocks(int& _n_blocks)
{
   _n_blocks = n_blocks;
   return SCIP_OKAY;
}

template <class T>
SCIP_RETCODE RowGraphWeighted<T>::nonClustered(int& _non_cl)
{
   _non_cl = non_cl;
   return SCIP_OKAY;
}

template <class T>
double RowGraphWeighted<T>::getEdgeWeightPercentile(double q)
{
   return this->graph.getEdgeWeightPercentile(q);
}

} /* namespace gcg */
#endif
