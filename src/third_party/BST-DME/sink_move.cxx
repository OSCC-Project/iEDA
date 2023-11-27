///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2018, The Regents of the University of California
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////


/*
#**************************************************************************
#**   Please read README to see how to run this program
***   Created by Chung-Wen Albert Tsao  on May 2, 2000*
#**
#**
#**************************************************************************
*/


#include "bst_header.h"
#include "Global_var.h"

extern double pt2ms_distance(PointType *pt, TrrType *ms);

extern void make_intersect( TrrType *trr1, TrrType *trr2, TrrType *t );
extern void core_mid_point(TrrType *trr, PointType *p);
extern void make_core(TrrType *trr,TrrType *core);


/********************************************************************/
/*                                                                  */
/********************************************************************/
int sink_move_compare_inc(const void *a, const void *b) {

   PairType *p = (PairType *) a;
   PairType *q = (PairType *) b;

   return( (p->cost > q->cost) ? YES: NO);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
void _calc_cluster_center(int cid, TrrType *ms) {
TrrType temp;

  double r = INT_MAX;
  unsigned n = 0; 
  unsigned npoints = gBoundedSkewTree->Npoints() ;
  for (unsigned i=0;i<npoints;++i) {
    PointType &qt = Node[ i   ].m_stnPt ;
    if (cid == Cluster_id[i]) {
      if (n==0) {
        ms->MakeDiamond (  qt     , r);
      } else {
        temp.MakeDiamond (  qt     , r);
        make_intersect(ms, &temp, ms);
      }
      n++;
    }
  }
  make_core(ms, ms);
}

/********************************************************************/
/*  calc neighbors of cluster cid.                                  */
/********************************************************************/
int _calc_cluster_neighbors(int cid, PairType pair[], int size) {
int i, n;
TrrType ms;

  n = 0;
  _calc_cluster_center(cid,  &ms);
  int nterms = (int) gBoundedSkewTree->Nterms() ;
  for (i=0;i<nterms;++i) {
    if (Cluster_id[i]!=cid) {
      pair[n].x = i;
      pair[n].y = cid;
      PointType &qt = Node[ i   ].m_stnPt ;
      pair[n].cost = pt2ms_distance(&(  qt     ), &ms);
      n++;
    }
  }
  assert(n>0 && n <= MAX_N_NODES);
  qsort(pair, n, sizeof(PairType), sink_move_compare_inc);
  n = min (n, size);
  return(n);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
int calc_all_cluster_neighbors(PairType pair[], int n_clusters, int size) {
int i,j,m, n;
PairType temp[MAX_N_NODES];

  n = 0;
  for (i=0;i< n_clusters;++i) {
    m = _calc_cluster_neighbors(i,temp, size);
    for (j=0;j<m;++j) {
      pair[n++] = temp[j];
    }
  }
  assert(n>0 && n <= MAX_N_NODES);
  qsort(pair, n, sizeof(PairType), sink_move_compare_inc);
  n = min (n, size*n_clusters);
  return(n);
}

