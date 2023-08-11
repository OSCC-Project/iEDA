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

/**@file   graph_gcg.h
 * @brief  Implementation of the graph which supports both node and edge weights.
 * @author Igor Pesic
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
#include <algorithm>
#include <iostream>
#include <cmath>
#include <set>
#include "graph_gcg.h"

#ifdef WITH_GSL
#include <gsl/gsl_errno.h>
#endif

using namespace std;

namespace gcg {


GraphGCG::GraphGCG()
{
   undirected = true;
   locked = false;
   initialized = false;
   nodes = vector<int>(0);
#ifdef WITH_GSL
   adj_matrix_sparse = gsl_spmatrix_alloc(1,1);
   working_adj_matrix = NULL;
#endif
}

GraphGCG::GraphGCG(int _n_nodes, bool _undirected)
{
   undirected = _undirected;
   locked = false;
   initialized = true;
   assert(_n_nodes >= 0);
   nodes = vector<int>(_n_nodes, 0);
#ifdef WITH_GSL
   adj_matrix_sparse = gsl_spmatrix_alloc(_n_nodes, _n_nodes);
   working_adj_matrix = NULL;
#else
   adj_matrix = vector<vector<double>>(_n_nodes, vector<double>(_n_nodes, 0.0));
#endif
}

GraphGCG::~GraphGCG()
{
   vector<int>().swap(this->nodes);
#ifndef WITH_GSL
   vector<vector<double>>().swap(adj_matrix);
   assert((int)adj_matrix.size() == 0);
#else
   gsl_spmatrix_free(adj_matrix_sparse);
   /*if(working_adj_matrix != NULL)
      gsl_spmatrix_free(working_adj_matrix);*/
#endif
   for(auto edge: edges)
   {
      if (edge != NULL)
         delete edge;
   }
   initialized = false;
}

SCIP_RETCODE GraphGCG::addNNodes(int _n_nodes)
{
   if(locked || initialized)
   {
      return SCIP_ERROR;
   }
#ifdef WITH_GSL
   if(adj_matrix_sparse == NULL)
      adj_matrix_sparse = gsl_spmatrix_alloc(_n_nodes, _n_nodes);
   // add diagonal because of MCL algorithm
   for(auto i = 0; i < _n_nodes; i++)
   {
      gsl_spmatrix_set(adj_matrix_sparse, i, i, 1.0);
   }
#else
   assert(adj_matrix.size() == 0);
   adj_matrix = vector<vector<double>>(_n_nodes, vector<double>(_n_nodes, 0.0));

#endif
   nodes = vector<int>(_n_nodes, 0);
   initialized = true;
   return SCIP_OKAY;
}

SCIP_RETCODE GraphGCG::addNNodes(int _n_nodes, std::vector<int> weights)
{
   auto res = addNNodes(_n_nodes);
   nodes = vector<int>(weights);
   return res;
}


int GraphGCG::getNNodes()
{
   if(!initialized) return 0;
#ifdef WITH_GSL
   assert(nodes.size() == adj_matrix_sparse->size1);
   assert(adj_matrix_sparse->size1 == adj_matrix_sparse->size2);
   return adj_matrix_sparse->size1;
#else
   assert(nodes.size() == adj_matrix.size());
   return (int)adj_matrix.size();
#endif
}


#ifdef WITH_GSL
gsl_spmatrix* GraphGCG::getAdjMatrix()
{
   cout << "Return adj_matrix_sparse..." << endl;
   return adj_matrix_sparse;
}


void GraphGCG::expand(int factor)
{
   assert(working_adj_matrix->sptype == (size_t)1);
   const double alpha = 1.0;
   gsl_spmatrix *tmp;
   for(int i = 1; i < factor; i++)
   {
      tmp = gsl_spmatrix_alloc_nzmax(working_adj_matrix->size1, working_adj_matrix->size2, working_adj_matrix->nz, GSL_SPMATRIX_CCS);
      gsl_spblas_dgemm(alpha, working_adj_matrix, working_adj_matrix, tmp);
      gsl_spmatrix_free(working_adj_matrix);
      working_adj_matrix = gsl_spmatrix_alloc_nzmax(tmp->size1, tmp->size2, tmp->nz, GSL_SPMATRIX_CCS);
      gsl_spmatrix_memcpy(working_adj_matrix, tmp);
      gsl_spmatrix_free(tmp);
   }
}


// inflate columns and normalize them
void GraphGCG::inflate(double factor)
{
   assert(working_adj_matrix->sptype == (size_t)1);
   size_t i = 0;
   while(i < working_adj_matrix->nz)
   {
      working_adj_matrix->data[i] = pow(working_adj_matrix->data[i], factor);
      i++;
   }
   colL1Norm();
}


// normalize columns, but remove values below 1e-6
void GraphGCG::colL1Norm()
{
   assert(working_adj_matrix->sptype == (size_t)1);
   // normalize the columns to sum up to 1
   for(int col = 0; col < (int)working_adj_matrix->size2; col++)
   {
      double col_sum = 0.0;
      size_t begin_ = working_adj_matrix->p[col];
      const size_t end_ = working_adj_matrix->p[col+1];
      while(begin_ < end_)
      {
         col_sum += working_adj_matrix->data[begin_++];
      }
      begin_ = working_adj_matrix->p[col];
      while(begin_ < end_)
      {
         working_adj_matrix->data[begin_] /= col_sum;
         begin_++;
      }
   }

   // We need this part to remove very very small values, otherwise GSL behaves strangely
   double threshold = 1e-8;
   gsl_spmatrix *tmp = gsl_spmatrix_alloc(working_adj_matrix->size1, working_adj_matrix->size2);

   for (int col = 0; col < (int)working_adj_matrix->size1; col++)
   {
      size_t begin_ = working_adj_matrix->p[col];
      const size_t end_ = working_adj_matrix->p[col+1];
      while(begin_ < end_)
      {
         if(working_adj_matrix->data[begin_] > threshold)
         {
            gsl_spmatrix_set(tmp, working_adj_matrix->i[begin_], col, working_adj_matrix->data[begin_]);
         }
         begin_++;
      }
   }

   gsl_spmatrix *tmp_comp = gsl_spmatrix_compcol(tmp);
   gsl_spmatrix_free(working_adj_matrix);
   working_adj_matrix = gsl_spmatrix_alloc_nzmax(tmp_comp->size1, tmp_comp->size2, tmp_comp->nz, GSL_SPMATRIX_CCS);
   gsl_spmatrix_memcpy(working_adj_matrix, tmp_comp);
   gsl_spmatrix_free(tmp);
   gsl_spmatrix_free(tmp_comp);
}


// remove values below 1e-3 or 5*1e-4 or 1e-4 and then normalize
void GraphGCG::prune()
{
   double threshold_start = 1e-3;
   gsl_spmatrix *tmp = gsl_spmatrix_alloc(working_adj_matrix->size1, working_adj_matrix->size2);

   double ave_col_sum = 0.0;
   for (int col = 0; col < (int)working_adj_matrix->size1; col++)
   {
      size_t begin_ = working_adj_matrix->p[col];
      const size_t end_ = working_adj_matrix->p[col+1];
      while(begin_ < end_)
      {
         if(working_adj_matrix->data[begin_] > threshold_start)
         {
            gsl_spmatrix_set(tmp, working_adj_matrix->i[begin_], col, working_adj_matrix->data[begin_]);
            ave_col_sum += working_adj_matrix->data[begin_];
         }
         begin_++;
      }
   }
   ave_col_sum /= (double)(tmp->size1);

   double thresholds[2] = {5*1e-4, 1e-4};
   for(int i = 0; i < 2; i++)
   {
      if(ave_col_sum > 0.85)
      {
         break;
      }

      gsl_spmatrix_set_zero(tmp);
      ave_col_sum = 0.0;
      for (int col = 0; col < (int)working_adj_matrix->size1; col++)
      {
         size_t begin_ = working_adj_matrix->p[col];
         const size_t end_ = working_adj_matrix->p[col+1];
         while(begin_ < end_)
         {
            if(working_adj_matrix->data[begin_] > thresholds[i])
            {
               gsl_spmatrix_set(tmp, working_adj_matrix->i[begin_], col, working_adj_matrix->data[begin_]);
               ave_col_sum += working_adj_matrix->data[begin_];
            }
            begin_++;
         }
      }
      ave_col_sum /= (double)(tmp->size1);
   }

   gsl_spmatrix *tmp_comp = gsl_spmatrix_compcol(tmp);
   gsl_spmatrix_free(working_adj_matrix);
   working_adj_matrix = gsl_spmatrix_alloc_nzmax(tmp_comp->size1, tmp_comp->size2, tmp_comp->nz, GSL_SPMATRIX_CCS);
   gsl_spmatrix_memcpy(working_adj_matrix, tmp_comp);

   gsl_spmatrix_free(tmp);
   gsl_spmatrix_free(tmp_comp);

   colL1Norm();
}


// checks if A*A - A stays unchanged
bool GraphGCG::stopMCL(int iter)
{
   if(iter > 6)
   {
      gsl_spmatrix *tmpSquare = gsl_spmatrix_alloc_nzmax(working_adj_matrix->size1, working_adj_matrix->size2, working_adj_matrix->nz, GSL_SPMATRIX_CCS);
      gsl_spmatrix *tmpMinus = gsl_spmatrix_alloc_nzmax(working_adj_matrix->size1, working_adj_matrix->size2, working_adj_matrix->nz, GSL_SPMATRIX_CCS);
      gsl_spmatrix *res = gsl_spmatrix_alloc_nzmax(working_adj_matrix->size1, working_adj_matrix->size2, working_adj_matrix->nz, GSL_SPMATRIX_CCS);
      gsl_spmatrix_memcpy(tmpMinus, working_adj_matrix);
      gsl_spmatrix_scale(tmpMinus, -1.0);
      double alpha = 1.0;
      gsl_spblas_dgemm(alpha, working_adj_matrix, working_adj_matrix, tmpSquare);
      gsl_spmatrix_add(res, tmpSquare, tmpMinus);
      double min, max;
      gsl_set_error_handler_off();
      int status = gsl_spmatrix_minmax(res, &min, &max);
      if(status == GSL_EINVAL)
      {
         min = 0.0;
         max = 0.0;
      }

      gsl_spmatrix_free(res);
      gsl_spmatrix_free(tmpSquare);
      gsl_spmatrix_free(tmpMinus);
      if(abs(max - min) < 1e-8)
      {
         return true;
      }
   }
   return false;
}

vector<int> GraphGCG::getClustersMCL()
{
   assert(working_adj_matrix->size1 == working_adj_matrix->size2);
   vector<int> res(working_adj_matrix->size1, 0);
   map<int, int> row_to_label;
   set<int> nzrows;
   vector<int> labels(working_adj_matrix->size1);

   for(int i = 0; i < (int)working_adj_matrix->nz; i++)
   {
      nzrows.insert(working_adj_matrix->i[i]);
   }
   int c = 0;
   for(auto nzrow: nzrows)
   {
      row_to_label[nzrow] = c;
      c++;
   }

   map<int, vector<int> > clust_map;
   // all data that belongs to that row belongs to its cluster
   for (int col = 0; col < (int)working_adj_matrix->size2; col++)
   {
      size_t begin_ = working_adj_matrix->p[col];
      const size_t end_ = working_adj_matrix->p[col+1];
      while(begin_ < end_)
      {
         labels[col] = row_to_label[working_adj_matrix->i[begin_]];
         clust_map[row_to_label[working_adj_matrix->i[begin_]]].push_back(col);     // python version
         begin_++;
      }
   }

   for(auto item : clust_map)
   {
      for(auto val : item.second)
      {
         labels[val] = item.first;
      }
   }
   // enumerations of labels may be broken (e.g. we can have labels 0,1,4,5), so we have to fix it to 0,1,2,3
   set<int> existinglabels;
   map<int, int> fixlabels;
   for(int i = 0; i < (int)labels.size(); i++)
   {
      existinglabels.insert(labels[i]);
   }

   c = 0;
   for(auto existinglabel: existinglabels)
   {
      fixlabels[existinglabel] = c;
      c++;
   }
   for(int i = 0; i < (int)labels.size(); i++)
   {
      labels[i] = fixlabels[labels[i]];
   }
   return labels;
}


void GraphGCG::initMCL()
{
   working_adj_matrix = gsl_spmatrix_alloc_nzmax(adj_matrix_sparse->size1, adj_matrix_sparse->size2, adj_matrix_sparse->nz, GSL_SPMATRIX_CCS);
   gsl_spmatrix_memcpy(working_adj_matrix, adj_matrix_sparse);
}


void GraphGCG::clearMCL()
{
   gsl_spmatrix_free(working_adj_matrix);
}

#else
vector<vector<double>> GraphGCG::getAdjMatrix()
{
   return adj_matrix;
}
#endif


// TODO: we can use edges.size() here
int GraphGCG::getNEdges()
{
   int n_edges = 0;
#ifdef WITH_GSL
   if(initialized)
      n_edges = adj_matrix_sparse->nz - adj_matrix_sparse->size1;
   else
      n_edges = 0;
#else
   for(int i = 0; i < (int)adj_matrix.size(); i++)
   {
      n_edges += getNNeighbors(i);
   }
#endif
   if(undirected)
   {
      assert(n_edges % 2 == 0);
      n_edges = (int) ( n_edges / 2.0);
   }
   assert(n_edges == (int)edges.size());
   return n_edges;
}

SCIP_Bool GraphGCG::isEdge(int node_i, int node_j)
{
   assert(node_i >= 0);
   assert(node_j >= 0);
#ifdef WITH_GSL
   if(gsl_spmatrix_get(adj_matrix_sparse, node_i, node_j) != 0.0)
   {
      return 1;
   }
#else
   if(adj_matrix[node_i][node_j] != 0.0)
   {
      return 1;
   }
#endif
   return 0;
}

int GraphGCG::getNNeighbors(int node)
{
   assert(node >= 0);
   int n_neighbors;
#ifdef WITH_GSL
   if(!initialized) return 0;
   assert(adj_matrix_sparse->sptype == (size_t)1);
   assert(node < (int)adj_matrix_sparse->size2);
   const size_t begin_ = adj_matrix_sparse->p[node];
   const size_t end_ = adj_matrix_sparse->p[node+1];
   n_neighbors = (int)(end_ - begin_);
   if(gsl_spmatrix_get(adj_matrix_sparse, node, node) != 0.0)
   {
      n_neighbors -=1 ;
   }
#else
   assert(node < (int)adj_matrix.size());
   n_neighbors = count_if(adj_matrix[node].begin(), adj_matrix[node].end(), [](double i) {return i != 0.0;});
   // remove itself as a neighbor
   if(adj_matrix[node][node] != 0.0)
   {
      n_neighbors -= 1;
   }
#endif
   return n_neighbors;
}

vector<int> GraphGCG::getNeighbors(int node)
{
   vector<int> res;
#ifdef WITH_GSL
   if(!initialized || !locked) return res;
   assert(adj_matrix_sparse->sptype == (size_t)1);
   assert(node < (int)adj_matrix_sparse->size2);
   size_t begin_ = adj_matrix_sparse->p[node];
   size_t end_ = adj_matrix_sparse->p[node+1];
   res.resize(end_ - begin_);
   size_t curr = 0;
   bool self_found = false;
   while(begin_ < end_){
      if((int)adj_matrix_sparse->i[begin_] == node)  self_found = true;
      res[curr++] = adj_matrix_sparse->i[begin_++];
   }
   assert(curr == res.size());
   if(self_found)
      res.erase(remove(res.begin(), res.end(), node), res.end());
#else
   for(int i = 0; i < (int)adj_matrix[node].size(); i++)
   {
      if(adj_matrix[node][i] != 0.0 && node != i)
      {
         res.push_back(i);
      }
   }
#endif
   return res;
}

vector<pair<int, double> > GraphGCG::getNeighborWeights(int node)
{
   vector<pair<int, double> > res;
#ifdef WITH_GSL
   if(!initialized || !locked) return res;
   assert(adj_matrix_sparse->sptype == (size_t)1);
   assert(node < (int)adj_matrix_sparse->size2);
   size_t begin_ = adj_matrix_sparse->p[node];
   size_t end_ = adj_matrix_sparse->p[node+1];
   res.resize(end_ - begin_);
   size_t curr = 0;
   bool self_found = false;
   int found_pos = 0;
   while(begin_ < end_){
      if((int)adj_matrix_sparse->i[begin_] == node)
      {
         self_found = true;
         found_pos = curr;
      }
      double value = adj_matrix_sparse->data[begin_];
      int row = (int)adj_matrix_sparse->i[begin_];
      res[curr++] = pair<int, double>(row, value);
      begin_++;
   }
   assert(curr == res.size());
   if(self_found)
   {
      res.erase(res.begin() + found_pos);
   }
#else
   for(int i = 0; i < (int)adj_matrix[node].size(); i++)
   {
      if(adj_matrix[node][i] != 0.0 && node != i)
      {
         res.push_back(make_pair(i, adj_matrix[node][i]));
      }
   }
#endif


   return res;
}

/** int node is obsolete, it must be the next available id  */
SCIP_RETCODE GraphGCG::addNode(int node, int weight)
{

   if(locked)
   {
      return SCIP_ERROR;
   }
   int next_id = (int)nodes.size();
   assert(node == next_id);

#ifdef WITH_GSL
   if(adj_matrix_sparse == NULL)
      adj_matrix_sparse = gsl_spmatrix_alloc(1, 1);
   // add diagonal because of MCL algorithm
   gsl_spmatrix_set(adj_matrix_sparse, next_id, next_id, 1.0);

   //return SCIP_NOTIMPL;
#else

   // add new column
   for(size_t i = 0; i < adj_matrix.size(); i++)
   {
      adj_matrix[i].push_back(0.0);
   }
   // add new row
   adj_matrix.push_back(vector<double>(next_id+1));

   // add self loop
   adj_matrix[next_id][next_id] = 1.0;
   assert(adj_matrix.size() == adj_matrix[0].size());
#endif
   initialized = true;
   nodes.push_back(weight);
   return SCIP_OKAY;
}

SCIP_RETCODE GraphGCG::addNode()
{
#ifdef WITH_GSL
   return addNode((int)adj_matrix_sparse->size2, 0);
#else
   return addNode((int)adj_matrix.size(), 0);
#endif
}

SCIP_RETCODE GraphGCG::deleteNode(int node)
{
   if(locked)
   {
      return SCIP_ERROR;
   }
   // Would be very inefficient and is not necessary
   return SCIP_INVALIDCALL;
}

SCIP_RETCODE GraphGCG::addEdge(int node_i, int node_j)
{
   return addEdge(node_i, node_j, 1);
}

SCIP_RETCODE GraphGCG::addEdge(int node_i, int node_j, double weight)
{
   if(locked || !initialized || node_i == node_j)
   {
      return SCIP_ERROR;
   }
   bool new_edge = true;
   assert(weight >= 0.0);
   assert(node_i >= 0);
   assert(node_j >= 0);
#ifdef WITH_GSL
   assert(node_i < (int)adj_matrix_sparse->size2);
   assert(node_j < (int)adj_matrix_sparse->size2);
   if (gsl_spmatrix_get(adj_matrix_sparse, node_i, node_j) != 0.0)
   {
      new_edge = false;
   }
   if(new_edge)
   {
      gsl_spmatrix_set(adj_matrix_sparse, node_i, node_j, weight);
      if(undirected)
      {
         gsl_spmatrix_set(adj_matrix_sparse, node_j, node_i, weight);
      }
   }
#else
   assert(node_i < (int)adj_matrix.size());
   assert(node_j < (int)adj_matrix.size());
   (adj_matrix[node_i][node_j] == 0.0) ? new_edge = true : new_edge = false;
   adj_matrix[node_i][node_j] = weight;
   if(undirected)
   {
      adj_matrix[node_j][node_i] = weight;
   }
#endif
   if(new_edge && node_i != node_j)
   {
      int first, second;
      first = node_i < node_j ? node_i : node_j;
      second = node_i == first ? node_j : node_i;
      EdgeGCG* edge1 = new EdgeGCG(first, second, weight);
      edges.push_back(edge1);
   }

   return SCIP_OKAY;
}

SCIP_RETCODE GraphGCG::setEdge(int node_i, int node_j, double weight)
{
   if(locked || !initialized)
   {
      return SCIP_ERROR;
   }
   assert(weight >= 0.0);
   assert(node_i >= 0);
   assert(node_j >= 0);
#ifdef WITH_GSL
   assert(node_i < (int)adj_matrix_sparse->size2);
   assert(node_j < (int)adj_matrix_sparse->size2);
   gsl_spmatrix_set(adj_matrix_sparse, node_i, node_j, weight);
   if(undirected)
   {
      gsl_spmatrix_set(adj_matrix_sparse, node_j, node_i, weight);
   }
#else
   assert(node_i < (int)adj_matrix.size());
   assert(node_j < (int)adj_matrix.size());
   adj_matrix[node_i][node_j] = weight;
   if(undirected)
   {
      adj_matrix[node_j][node_i] = weight;
   }
#endif
   if(node_i != node_j)
      for(auto edge: edges)
      {
         if((edge->src == node_i && edge->dest == node_j) || (edge->src == node_j && edge->dest == node_i))
         {
            edge->weight = weight;
         }
      }
   return SCIP_OKAY;
}

SCIP_RETCODE GraphGCG::deleteEdge(int node_i, int node_j)
{
   return setEdge(node_i, node_j, 0);
}

int GraphGCG::graphGetWeights(int node)
{
   return nodes[node];
}

double GraphGCG::getEdgeWeight(int node_i, int node_j)
{
   assert(node_i >= 0);
   assert(node_j >= 0);
   double weight;
#ifdef WITH_GSL
   assert(node_i < (int)adj_matrix_sparse->size1);
   assert(node_j < (int)adj_matrix_sparse->size1);
   weight = gsl_spmatrix_get(adj_matrix_sparse, node_i, node_j);
#else
   assert(node_i < (int)adj_matrix.size());
   assert(node_j < (int)adj_matrix.size());
   weight = adj_matrix[node_i][node_j];
#endif
   return weight;
}

SCIP_RETCODE GraphGCG::flush()
{
   locked = true;
#ifdef WITH_GSL
   gsl_spmatrix *adj_matrix_sparse_tmp = gsl_spmatrix_compcol(adj_matrix_sparse);
   gsl_spmatrix_free(adj_matrix_sparse);
   adj_matrix_sparse = adj_matrix_sparse_tmp;
#endif
   return SCIP_OKAY;
}

SCIP_RETCODE GraphGCG::normalize()
{
   double scaler = 0.0;
#ifdef WITH_GSL
   double min, max;
   gsl_spmatrix_minmax(adj_matrix_sparse, &min, &max);
   scaler = max;
   gsl_spmatrix_scale(adj_matrix_sparse, double(1.0/scaler));

#else
   for(int i = 0; i < (int)adj_matrix.size(); i++)
   {
      vector<double>::iterator tmp = max_element(adj_matrix[i].begin(), adj_matrix[i].end());
      double curr_max = *tmp;
      if( curr_max > scaler )
      {
         scaler = curr_max;
      }
   }

   for(int i = 0; i < (int)adj_matrix.size(); i++)
   {
      for(int j = 0; j < (int)adj_matrix[i].size(); j++)
      {
         adj_matrix[i][j] /= scaler;
         assert(adj_matrix[i][j] <= 1);
         assert(adj_matrix[i][j] >= 0);
      }
   }
#endif
   return SCIP_OKAY;
}

double GraphGCG::getEdgeWeightPercentile(double q)
{
   double res = -1;
   vector<double> all_weights;
#ifdef WITH_GSL
   all_weights = vector<double>(adj_matrix_sparse->data, adj_matrix_sparse->data + adj_matrix_sparse->nz);
#else
   for(int i = 0; i < (int)adj_matrix.size(); i++)
   {
      for(int j = 0; j < (int)adj_matrix[i].size(); j++)
      {
         if(adj_matrix[i][j] != 0.0)
         {
            all_weights.push_back(adj_matrix[i][j]);
         }
      }
   }
#endif
   sort(all_weights.begin(), all_weights.end());

   int upper_pos = (int) floor((q/100.0)*all_weights.size());
   int lower_pos = (int) ceil((q/100.0)*all_weights.size());
   if(upper_pos != lower_pos)
   {
      res = (all_weights[upper_pos] + all_weights[lower_pos]) / 2.0;
   }
   else
   {
      res = all_weights[lower_pos];
   }
   return res;
}


SCIP_RETCODE GraphGCG::getEdges(vector<void *>& _edges)
{
   _edges.resize(edges.size());
   for(int i = 0; i < (int)edges.size(); i++)
   {
      assert(edges[i]->src < (int)nodes.size());
      assert(edges[i]->dest < (int)nodes.size());
      assert(edges[i]->src != edges[i]->dest);

      (_edges[i]) = (void *)(edges[i]);
   }
   return SCIP_OKAY;
}


// Compare two edges according to their weights.
// Used in sort() for sorting an array of edges
int GraphGCG::edgeComp(const EdgeGCG* a, const EdgeGCG* b)
{
   return (a->src < b->src) || (a->src == b->src && a->weight < b->weight);
}




}
