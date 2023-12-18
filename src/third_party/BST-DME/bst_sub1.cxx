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


#include <fstream.h>
#include <iostream.h>
#include <cmath>
#include <vector>
#include <strings.h>
#include "bst_header.h"
#include "Global_var.h"
#include "bst.h"
#include "bst_sub1.h"
#include "bst_sub3.h"
#include "facility.h"

extern double path_finder(PointType p1, PointType p2, PointType path[], int *n);
extern double calc_pathlength(PointType path[], int n, int mode);
extern void print_obstacles_sub(FILE *f);

void print_node_info(NodeType *node);
void print_node(NodeType *node);
double area_minskew(AreaType *area); 
void print_node_informatio(NodeType *node,NodeType *node_L, NodeType *node_R);
void check_trr(TrrType *t);
int area_Manhattan_arc_JS(AreaType *area);
double calc_edge_delay(int v, int par, int mode);

/********************************************************************/
// split string into tokens   
/********************************************************************/
static void
tSplit ( char *buf, vector< char * > tokv, const char *sep = 0 )
{

    if( sep == 0 ) {
        sep = "\t ";
    }
    tokv.clear();
    char *temp ;
    char *s  = buf ;
    
    while( ( temp = strtok( s, sep )) != 0 ) {
      tokv.push_back ( temp ) ;
      // cerr << temp  << " " ;
      s = 0 ;
    }
    // cerr << endl ;
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
int calc_n_buffer_level() {
int i, n;

  i = 0;
  n = 0;
  while (N_Clusters[i] > 1) {
    i++;
    n++;
  }
  return(n);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int level(int v) {
int n=0;

  assert(Buffered[v] > 0 );
  NodeType *from  = gBoundedSkewTree->TreeNode (v ) ;
  NodeType *to1  = gBoundedSkewTree->SuperRootNode () ;
  NodeType *to2  = gBoundedSkewTree->RootNode () ;
  
  while (from != to1  && from != to2 ) {
    if (Buffered[v] > 0 ) n++;
    v = Node[v].parent;
  }
  return(n);
}
/******************************************************************/
/*                                                                */
/******************************************************************/
int calc_line_type(PointType pt1,PointType pt2) {
double a,b;
  a = ABS(pt1.x-pt2.x);
  b = ABS(pt1.y-pt2.y);

  if (equal(a,b)) {
    return(MANHATTAN_ARC);
  } else if (equal(a,0)) {
    return(VERTICAL);
  } else if (equal(b,0)) {
    return(HORIZONTAL);
  } else if (a > b) {
    return(FLAT);
  } else if (a < b) {
    return(TILT);
  }
  assert ( 0 ) ;
  return ( MANHATTAN_ARC ) ;
}

/*****************************************************************/
/*                                                                */
/******************************************************************/
double calc_buffered_node_delay(int v) {
AreaType *area;
double t;

  area = &(Node[v].area[Node[v].ca]);
  PointType &pt = Node[v].m_stnPt ;
  t = pt       .max + area->unbuf_capac*area->R_buffer + Delay_buffer;
  return(t);
}

/********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/********************************************************************/
int JR_corner_exist(int i) {
double x0,x1,y0,y1;

  x0 = JS[0][i].x;
  y0 = JS[0][i].y;
  x1 = JS[1][i].x;
  y1 = JS[1][i].y;
  if (equal(x0,x1) || equal(y0,y1)) {
    return(NO);
  } else {
    return(YES);
  }
}
/********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/********************************************************************/
int empty_trr(TrrType *t) {
double a,b;

  a=t->xlow;
  b=t->xhi;
  if (equal(a,b) ){
    t->xlow = t->xhi = (a+b)/2;
  }
  a=t->ylow;
  b=t->yhi;
  if (equal(a,b) ){
    t->ylow = t->yhi = (a+b)/2;
  }

  if (t->xlow > t->xhi || t->ylow > t->yhi) {
    return(YES);
  } else {
    return(NO);
  }
}
/********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/********************************************************************/

void check_trr(TrrType *t) {
  empty_trr(t);
  assert(t->xlow <= t->xhi);
  assert(t->ylow <= t->yhi);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/

void check_Point_delay(PointType *pt) {

  assert(pt->min > -FUZZ);
  assert(pt->max > pt->min - FUZZ);

  if (pt->min < 0) {
    pt->min = 0;
  }
  if (pt->max < pt->min) {
    pt->max = pt->min;
  }
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
static 
void check_Point_skew(PointType *pt) {
  assert(pt->max - pt->min < Skew_B + FUZZ);
  pt->max = tMIN(pt->max, pt->min + Skew_B);
}


/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void check_Point(PointType *pt) {

  check_Point_delay(pt);
  check_Point_skew(pt);
}

/************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew Kahng, Chung-Wen Albert Tsao     */
/************************************************************************/
int equal(double x,double y) {
  if (ABS(x-y) <= FUZZ) {
    return(YES);
  }
  return(NO);
}
/************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew Kahng, Chung-Wen Albert Tsao     */
/************************************************************************/
int equivalent(double x,double y, double fuzz) {
  if (ABS(x-y) <= fuzz) {
    return(YES);
  }
  return(NO);
}


/********************************************************************/
/*    merge_cost  = dist + detour wirien = |e_a| + |e_b|           */
/********************************************************************/
double area_merge_cost(AreaType *area) {
double t;

  t = area->L_EdgeLen + area->R_EdgeLen;
  if (t >= 0 ) {
    assert( equal(t, area->dist) || t >= area->dist);
    return(t);
  } else {
    return(area->dist);
  }
}
/********************************************************************/
/*    merge_cost  = dist + detour wirien = |e_a| + |e_b|           */
/********************************************************************/
double merge_cost(NodeType *node) {
  return(area_merge_cost(&(node->area[node->ca])));
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
static 
void print_area_sub(FILE *f, AreaType *area) {

  int n = area->n_mr;
  assert( n <= MAX_mr_PTS);
  if (n==0) {
    fprintf(f, "move 0 0 Empty Region \n");
  } else {
    PointType pt = area->mr[0];
    fprintf(f, "move %.8f  %.8f  (JS_type=%d)(n_mr=%d)(cap=%.1f)\n",
      pt.x, pt.y, areaJS_line_type(area), n, area->capac);
    for (int i=1; i<=n; i++) {
      pt = area->mr[i%n];
      fprintf(f, "     %.8f  %.8f  skew:%.2f, max:%.2f, min:%.2f\n",
          pt.x, pt.y, pt.max-pt.min,pt.max,pt.min);
    }
  }
  fflush(f);
}

/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/

void print_area(AreaType *area) {
  print_area_sub(stdout,area);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
static 
void print_node_sub(FILE *f, NodeType *node) {

  fprintf(f, "\"node[%d]%d:%d\n", node->id, node->ca, node->n_area);
  fprintf(f, "\"mr\n");
  for (int i=0;i<node->n_area;++i) {
    print_area_sub(f, &(node->area[i]));
  }
}
/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void print_node(NodeType *node) {
  print_node_sub(stdout,node);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void set_K() {
  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
    PURES[V_] = PURES[H_] = 0; 
    PUCAP[V_] = PUCAP[H_] = 0; 
    K[H_] = K[V_] = 0;
  } else {
    PUCAP[H_] *= PUCAP_SCALE;
    PURES[V_] = PURES[H_]*PURES_V_SCALE ;
    PUCAP[V_] = PUCAP[H_]*PUCAP_V_SCALE ;
    K[H_] = 0.5*PURES[H_]*PUCAP[H_];
    K[V_] = 0.5*PURES[V_]*PUCAP[V_];
  }
}
/****************************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void alloca_NodeType_with_n_areas(NodeType *node, int n) {
  node->ca = 0;
  node->n_area = 1;
  assert(n <= N_Area_Per_Node);

  /* Nate: purify */
  assert(node->area == 0);
  node->area = new AreaType[n];
  memset(node->area,0,n*sizeof(AreaType));
  assert(node->area !=NULL );
  assert(node->ms == 0);
  node->ms = new TrrType;
  assert(node->ms !=NULL );

}
/****************************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void alloca_NodeType(NodeType *node) {
  alloca_NodeType_with_n_areas(node, N_Area_Per_Node);
}
/****************************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void free_NodeType(NodeType *node) {
  free(node->area);
  free(node->ms);
}
/************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew Kahng, Chung-Wen Albert Tsao     */
/************************************************************************/
void assign_NodeType(NodeType *node1, NodeType *node2) {
int i, n; 

  node1->ca = node2->ca;
  n = node1->n_area = node2->n_area;
  assert(node2->ca <= n);
  assert(n <= N_Area_Per_Node);
  for (i=0;i<n; ++i) {
    node1->area[i] = node2->area[i];
  }
  if (node1->ms!= NULL  && node2->ms != NULL) {
    *(node1->ms) = *(node2->ms);
  }
  node1->parent = node2->parent;
  node1->L = node2->L;
  node1->R = node2->R;
  node1->id = node2->id;
  node1->root_id = node2->root_id;
  node1->pattern = node2->pattern;
  node1->buffered = node2->buffered;
}

/****************************************************************************/
/*      tell if Node[v] uses detour wirelength.                             */
/****************************************************************************/
int detour_Node(int v) {
int ans, i;

  i = Node[v].ca;
  if (Node[v].area[i].L_EdgeLen + Node[v].area[i].R_EdgeLen > 
      Node[v].area[i].dist) {
    ans = YES;
  } else {
    ans = NO;
  }
  return(NO);
}

/*******************************************************************/
/*  Return the skew of a given point */
/*******************************************************************/
double pt_skew(PointType pt) {
  return(pt.max-pt.min);
}

/********************************************************************/
/* find the mid-point of the core of trr */
/********************************************************************/
void core_mid_point(TrrType *trr, PointType *p) {
double tx,ty;

  tx = (trr->xlow + trr->xhi) / 2;
  ty = (trr->ylow + trr->yhi) / 2;

  p->x = (tx+ty)/2;
  p->y = (ty-tx)/2;

} /* core_mid_point */

/********************************************************************/
/*                                                                  */
/********************************************************************/
void make_intersect_sub( TrrType *trr1, TrrType *trr2, TrrType *t ) {

  t->xlow = tMAX(trr1->xlow,trr2->xlow);
  t->xhi  = tMIN(trr1->xhi, trr2->xhi );

  t->ylow = tMAX(trr1->ylow,trr2->ylow);
  t->yhi  = tMIN(trr1->yhi, trr2->yhi );

}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void make_intersect( TrrType *trr1, TrrType *trr2, TrrType *t ) {

   make_intersect_sub(trr1, trr2, t);

  /* check if intersection is non-empty */
  check_trr(t); 

} /* make_intersect */

/********************************************************************/
/* find the radius of a TRR */
/********************************************************************/
double radius(TrrType *trr)  {
double t1,t2;

  t1 = (trr->xhi - trr->xlow) / 2;
  t2 = (trr->yhi - trr->ylow) / 2;

  return( min(t1,t2) );
}

/********************************************************************/
/* return the core of trr in core */
/********************************************************************/
void make_core(TrrType *trr,TrrType *core)  {
double r;

  r = radius(trr);
  core->xlow = trr->xlow + r;
  core->xhi  = trr->xhi  - r;
  core->ylow = trr->ylow + r;
  core->yhi  = trr->yhi  - r;

} /* make_core */
 
/********************************************************************/
/* return the core of trr in core */
/********************************************************************/
void make_1D_TRR(TrrType *trr,TrrType *core)  {

  if (trr->xhi - trr->xlow >= trr->yhi - trr->ylow) {
    core->xlow = trr->xlow ;
    core->xhi  = trr->xhi  ;
    core->ylow = core->yhi = (trr->yhi + trr->ylow)/2.0 ;
  } else {
    core->ylow = trr->ylow ;
    core->yhi  = trr->yhi  ;
    core->xlow = core->xhi = (trr->xhi + trr->xlow)/2.0 ;
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int a_segment_TRR(TrrType *trr) {
  /* check if *trr is a valid merging segment */
  if (equal(trr->xlow,trr->xhi) || equal(trr->ylow,trr->yhi)) {
    return(YES);
  } else {
    return(NO);
  }
}
/********************************************************************/
/* check if *ms is a valid merging segment */
/********************************************************************/
void check_ms(TrrType *ms) {

  /* *ms must be a valid trr */
  check_trr(ms);

  /* check if *ms is a valid merging segment */
  assert(a_segment_TRR(ms));

  /* remove the epsilon error */
  if (equal(ms->xlow,ms->xhi)) {
    ms->xlow = ms->xhi = (ms->xlow+ms->xhi)/2.0;
  }
  if (equal(ms->ylow,ms->yhi)) {
    ms->ylow = ms->yhi = (ms->ylow+ms->yhi)/2.0;
  }
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
int ms_type(TrrType *trr) {
double a,b;

  check_trr(trr);
  a = trr->xhi - trr->xlow;
  b = trr->yhi - trr->ylow;

  assert(equal(a,0) || equal(b,0) );  /* must be a segment */

  if (equal(a,0) && equal(b,0) ) {  /* a point TRR */ 
    return(0); 
  } else if (equal(a,0)) {     /* segment with slope +1 */
    return(1); 
  } else if (equal(b,0)) {    /* segment with slope -1 */
    return(-1); 
  }
    assert(0); 
    return(0); 
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
void build_trr(TrrType *ms,double d,TrrType *trr) {
  trr->xlow = ms->xlow - d;
  trr->xhi  = ms->xhi + d;

  trr->ylow = ms->ylow - d;
  trr->yhi  = ms->yhi + d;

} /* build_trr */

/****************************************************************************/
/*                                                                       */
/****************************************************************************/
int Manhattan_arc(PointType p1,PointType p2) {
double a,b;
char ans;

  a = ABS(p2.y-p1.y);
  b = ABS(p2.x-p1.x);
  ans = equal(a, b)? YES:NO;
  return( ans );
}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
int Manhattan_arc_area(AreaType  *area) {
int n;

  n= area->n_mr;
  if (n==1) return(YES);
  if (n==2 && Manhattan_arc(area->mr[0],area->mr[1])) return(YES);
  return(NO);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
int Manhattan_arc_node(NodeType  *node) {
  return(Manhattan_arc_area(&(node->area[node->ca])));
}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
int merging_segment_area(AreaType  *area) {

  if (Manhattan_arc_area(area) && equal(pt_skew(area->mr[0]), Skew_B))  {
    return(YES);
  } else {
    return(NO);
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int TRR_area(AreaType *area) {
int i, j, n, count = 0;
double t0,t1;

  if (Manhattan_arc_area(area)) return(YES);
  n = area->n_mr;
  if ( n != 4) return(NO);
  for (i=0;i<n;++i) {
    j = (i+1)%n;
    if (Manhattan_arc(area->mr[i], area->mr[j])==YES) count++;
    t0 = ABS(area->mr[i].max - area->mr[j].max);
    t1 = ABS(area->mr[i].min - area->mr[j].min);
    if (t0 > 1E-10 || t1 > 1E-10 ) return(NO);
  }
  if (count!=2) return(NO);
  return(YES);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
int Manhattan_Arc_Node(int v) {
  return(Manhattan_arc_node(&(Node[v])));
}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
int case_Manhattan_arc() {
int ans;

  ans = NO;

  if (Skew_B==0) {
    ans = YES;
  } else if (equal(pt_skew(JS[0][0]), Skew_B)) {
    if (equal(pt_skew(JS[1][0]), Skew_B) ) {
      if (Manhattan_arc(JS[0][0], JS[0][1]) ) {
        assert(Manhattan_arc(JS[1][0], JS[1][1]) );
        ans = YES;
      }
    }
  }
  return(ans);
}

/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void print_Point(FILE *f, const PointType& pt) {
  fprintf(f,"move %.9f %.9f skew = %.9f\n", pt.x,pt.y, pt.max-pt.min);
  fprintf(f,"     %.9f %.9f max=%.9f, min=%.9f\n",pt.x,pt.y,pt.max,pt.min);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int count_trees() {
int count = 0;
 
  unsigned npoints = gBoundedSkewTree->Npoints() ;
  
  for (unsigned i=0;i < npoints ;++i) {
    if (Node[i].parent <0 && !Marked[i]     )
      count++;
  }
  return count ;
}


/****************************************************************************/
/*  the merging area of node1 =  node2->area[n].              */
/****************************************************************************/
void get_an_area(NodeType *node, int i) {
  node->ca = i; 
  assert(i < node->n_area); 
  assert(node->n_area <= N_Area_Per_Node); 
  assert(node->area[i].npts >0); 
  assert(node->area[i].n_mr >0); 
  /*
  JS[0][0] = node->area[i].line[0][0] ;
  JS[0][1] = node->area[i].line[0][1] ;
  JS[1][0] = node->area[i].line[1][0]; 
  JS[1][1] = node->area[i].line[1][1] ;
  */
}
/****************************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void get_area_for_Node(int v, int i) {
  get_an_area(&(Node[v]), i);
}
/*************************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/*************************************************************************/
void get_all_areas(int v, int i) {
int j;

  unsigned nterms = gBoundedSkewTree->Nterms() ;
  if ((unsigned) v< nterms ) return;
  get_area_for_Node(v, i);
  j = Node[v].ca; 
  get_all_areas(Node[v].L, Node[v].area[j].L_area);
  get_all_areas(Node[v].R, Node[v].area[j].R_area);
}

/****************************************************************************/
/*  store merging area of node1 to node2->area[n]                           */
/****************************************************************************/
void store_an_area(NodeType *node1, NodeType *node2, int n) {
int i, k;

  k = node1->ca;
  assert(k < node1->n_area);
  assert(n < N_Area_Per_Node);
  node2->area[n].n_mr         = node1->area[k].n_mr;
  node2->area[n].npts         = node1->area[k].npts;
  node2->area[n].L_area       = node1->area[k].L_area   ;
  node2->area[n].R_area       = node1->area[k].R_area   ;
  node2->area[n].L_EdgeLen    = node1->area[k].L_EdgeLen;
  node2->area[n].R_EdgeLen    = node1->area[k].R_EdgeLen;
  node2->area[n].subtree_cost = node1->area[k].subtree_cost;
  node2->area[n].dist         = node1->area[k].dist;
  node2->area[n].capac        = node1->area[k].capac;

  for (i=0;i<node1->area[k].npts;++i) {
    node2->area[n].vertex[i] = node1->area[k].vertex[i];
  }
  for (i=0;i<node1->area[k].n_mr;++i) {
    node2->area[n].mr[i] = node1->area[k].mr[i];
  }
  node2->area[n].line[0][0] = node1->area[k].line[0][0];
  node2->area[n].line[0][1] = node1->area[k].line[0][1];
  node2->area[n].line[1][0] = node1->area[k].line[1][0];
  node2->area[n].line[1][1] = node1->area[k].line[1][1];
}
/****************************************************************************/
/*  store area_array tmpnode[n] to node                                    */
/****************************************************************************/
void store_n_areas(AreaType tmparea[], int n, NodeType *node) {
int i;

  node->n_area = n;
  for (i=0;i<n;++i) {
    node->area[i] = tmparea[i];
  }
}
/******************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/******************************************************************/
void store_n_areas_IME(NodeType *node,AreaSetType *result) {
int i;

  node->n_area = result->npoly;
  for (i=0;i<result->npoly;++i) { 
    node->area[i] = result->freg[i];
  }
}
/******************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/******************************************************************/
void store_last_n_areas_IME(NodeType *node,AreaSetType *stair) {
int i, n;

  node->n_area = n = min (stair->npoly, N_Area_Per_Node);
  for (i=0;i<n;++i) { 
    node->area[i] = stair->freg[stair->npoly-1-i];
  }
}

/*****************************************************************************/
/*                                                                           */
/*****************************************************************************/
void build_NodeTRR_sub1(NodeType *node) {
int i, j;
double x,y;
 
  node->ms->MakeDiamond   (node->area[node->ca].mr[0], 0       );
  for (i=1;i<node->area[node->ca].npts;++i) {
    j = node->area[node->ca].vertex[i]; 
    x = node->area[node->ca].mr[j].x;
    y = node->area[node->ca].mr[j].y;
    node->ms->xlow = tMIN(node->ms->xlow, x-y);
    node->ms->xhi  = tMAX(node->ms->xhi , x-y);
    node->ms->ylow = tMIN(node->ms->ylow, x+y);
    node->ms->yhi  = tMAX(node->ms->yhi , x+y);
  }
}
/*************************************************************************** */
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/*************************************************************************** */
void build_NodeTRR_sub2(NodeType *node) {
int i,j,k, n; 
double x,y;

  node->ms->xlow = node->ms->ylow =  DBL_MAX;
  node->ms->xhi  = node->ms->yhi  = -DBL_MAX;
  for (i=0;i<node->n_area;++i) {
    n = node->area[i].npts; 
    for (j=0;j<n;++j) {
      k = node->area[i].vertex[j]; 
      x = node->area[i].mr[k].x;
      y = node->area[i].mr[k].y;
      node->ms->xlow = tMIN(node->ms->xlow, x-y);
      node->ms->xhi  = tMAX(node->ms->xhi , x-y);
      node->ms->ylow = tMIN(node->ms->ylow, x+y);
      node->ms->yhi  = tMAX(node->ms->yhi , x+y);
    }
  }
}

/****************************************************************************/ 
/* make a smallest TRR containing the merging Node */
/* for the inplementation of the bucket decomposition */ 
/****************************************************************************/ 
void build_NodeTRR(NodeType *node) {
  if ( BST_Mode != BME_MODE   ) {
    build_NodeTRR_sub2(node);
  } else {
    build_NodeTRR_sub1(node);
  }
}
 
/*******************************************************************/
/*                                                                 */
/*******************************************************************/
double minskew_IME_sub(NodeType *node, int i) {
double min_skew;
int j;

  min_skew=DBL_MAX;
  for (j=0;j<node->area[i].n_mr;++j) {
    min_skew = tMIN(pt_skew(node->area[i].mr[j]), min_skew);
  }
  return(min_skew);
}
/*******************************************************************/
/*                                                                 */
/*******************************************************************/
double minskew_IME(NodeType *node) {
double min_skew, x;
int i;

  min_skew=DBL_MAX;
  for (i=0; i<node->n_area; i++) {
    x = minskew_IME_sub(node, i);
    min_skew = min (x, min_skew);
  }
  return(min_skew);
}
/*******************************************************************/
/*                                                                 */
/*******************************************************************/
double area_minskew(AreaType *area) {
double min_skew;
int i;

  min_skew=DBL_MAX;
  for (i=0; i<area->n_mr; i++) {
    min_skew = min (pt_skew(area->mr[i]), min_skew);
  }
  return(min_skew);
}

/*******************************************************************/
/*                                                                 */
/*******************************************************************/
double minskew_BME(NodeType *node) {
  return(area_minskew(&(node->area[node->ca])));
}
/*******************************************************************/
/*  Return the minimum skew of the given merging region            */
/*******************************************************************/
double minskew(NodeType *node, int mode) {
double min_skew=DBL_MAX;

  if (mode == BME_MODE) {
    min_skew = minskew_BME(node);
  } else {
    min_skew = minskew_IME(node);
  }
  return(min_skew);
}


/*******************************************************************/
/*                                                                 */
/*******************************************************************/
double maxskew_sub1(NodeType *node) {
double max_skew = 0 ;
int i, j;


  for (i=0; i<node->n_area; i++) {
    for (j=0;j<node->area[i].n_mr;++j) {
      max_skew = tMAX(pt_skew(node->area[i].mr[j]), max_skew);
    }
  }
  return(max_skew);
}

/*******************************************************************/
/*                                                                 */
/*******************************************************************/
double maxskew_sub2(NodeType *node) {
double max_skew = 0;
int i;

  for (i=0; i<node->area[node->ca].n_mr; i++) {
    max_skew = tMAX(pt_skew(node->area[node->ca].mr[i]), max_skew);
  }
  return(max_skew);
}
/*******************************************************************/
/*  Return the max skew of the given merging region            */
/*******************************************************************/
double maxskew(NodeType *node, int mode) {
double  max_skew=0;

  if (mode != BME_MODE) {
    max_skew = maxskew_sub1(node);
  } else {
    max_skew = maxskew_sub2(node);
  }
  return(max_skew);
}

/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void trace() {
  printf("******* # trees = %d*********", count_trees());
  iShowTime();
  fflush(stdout);
}

/****************************************************************************/
/*                   check if three points p1, p2, p3 Node colinear */
/****************************************************************************/
int colinear(PointType p1, PointType p2, PointType p3) {

  if ((p1.x==p2.x) && (p2.x==p3.x))
      return(YES);
  if ((p1.y==p2.y) && (p2.y==p3.y))
      return(YES);
  if ((p1.y-p2.y)/(p1.x-p2.x) == (p1.y-p3.y)/(p1.x-p3.x))
      return (YES);

  return(NO);
}
/******************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/******************************************************************/
void print_a_trr(TrrType *trr) {

  printf("xl=%f,xh=%f,yl=%f,yh=%f\n",trr->xlow, trr->xhi, 
                                         trr->ylow,trr->yhi);
}
/******************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/******************************************************************/
void print_IME_areas(NodeType *node,NodeType *node_L, NodeType *node_R,int n, int m) {
int i, L,R,n1,n2, L_area, R_area;
double cap, minskew;

  L = node_L->id;
  R = node_R->id;
  
  n1 = node_L->n_area;
  n2 = node_R->n_area;
  printf("\nnode %d(%d,%d)(n1=%d,n2=%d) has %d areas out of %d(total:%d): \n", 
     node->id, L,R, n1,n2, node->n_area, n,m);
  for (i=0;i<n1;++i) { printf("%d ",node_L->area[i].n_mr); }
  printf("\n");
  for (i=0;i<n2;++i) { printf("%d ",node_R->area[i].n_mr); }
  printf("\n");
  

  for (i=0;i<node->n_area;++i) { 
    cap = node->area[i].capac;
    minskew = minskew_IME_sub(node, i);
    L_area = node->area[i].L_area;
    R_area = node->area[i].R_area;
    printf("	cap=%f, minskew=%f (%d,%d)\n", cap, minskew, L_area,R_area);
  }

}
/******************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/******************************************************************/
void print_double_array(double *a, int  n) {
int i;

  for (i=0;i<n;++i) { printf("%.9f, ", a[i]); }
  printf("\n");
}

/****************************************************************************/
/*  check if two bounding boxes intesect.                                   */
/****************************************************************************/
int _bbox_overlap(double x1, double y1, double x2, double y2,
                 double x3, double y3, double x4, double y4, double epsilon) {

  if (tMIN(x1,x2)>=tMAX(x3,x4) + epsilon ) return(NO);
  if (tMIN(x3,x4)>=tMAX(x1,x2) + epsilon ) return(NO);
  if (tMIN(y1,y2)>=tMAX(y3,y4) + epsilon ) return(NO);
  if (tMIN(y3,y4)>=tMAX(y1,y2) + epsilon ) return(NO);
  return(YES);
}

/****************************************************************************/
/*  check if two bounding boxes intesect.                                   */
/****************************************************************************/
int bbox_overlap(double x1, double y1, double x2, double y2,
                 double x3, double y3, double x4, double y4) {

  return(_bbox_overlap(x1,y1,x2,y2,x3,y3,x4,y4,FUZZ));
}

/***********************************************************************/
/* check if (x,y) is in the box defined by max_x,min_x,max_y,min_y     */
/***********************************************************************/
int in_bbox(double x, double y, double x1,double y1, double x2,double y2) {
  if (x <= tMAX(x1,x2) + FUZZ && x >= tMIN(x1,x2) - FUZZ && y <= tMAX(y1,y2)
      + FUZZ && y >= tMIN(y1,y2)- FUZZ )
   { return(YES); }
  return(NO);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
void TRR2pt(TrrType *trr, PointType *pt) {
  pt->x = (trr->ylow + trr->xhi )/2;
  pt->y = (trr->ylow - trr->xhi )/2;
}

/********************************************************************/
/*  merging segment to line segment                                  */
/********************************************************************/
void ms_to_line(TrrType *ms,double *x1,double *y1,double *x2,double *y2) {

  /* *ms must be a valide merging segement */
  check_ms(ms);
  *x1 = (ms->ylow + ms->xhi )/2;
  *y1 = (ms->ylow - ms->xhi )/2;
  *x2 = (ms->yhi  + ms->xlow)/2;
  *y2 = (ms->yhi  - ms->xlow)/2;
  /* p2 must be higher than p1 */
  assert(*y1 <= *y2);
}
/********************************************************************/
/*  merging segment to line segment                                  */
/********************************************************************/
void ms2line(TrrType *ms, PointType *p1, PointType *p2) {
  ms_to_line(ms,&(p1->x),&(p1->y),&(p2->x),&(p2->y)); 
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int TRR2area(TrrType *trr, PointType pt[]) {
TrrType ms;
int n;
double a,b;

  a = trr->xhi - trr->xlow;
  b = trr->yhi - trr->ylow;

  check_trr(trr);

  if (equal(a,0) && equal(b,0) ) {
    TRR2pt(trr, &(pt[0]));
    n = 1;
  } else if (equal(a,0) || equal(b,0) ) {
    ms2line(trr, &(pt[0]), &(pt[1]));
    n = 2;
  } else {
    ms.yhi = trr->yhi;
    ms.ylow = trr->ylow;
    ms.xhi = ms.xlow = trr->xhi;
    ms2line(&ms, &(pt[0]), &(pt[1]));
    ms.xhi = ms.xlow = trr->xlow;
    ms2line(&ms, &(pt[3]), &(pt[2]));
    n = 4;
  }
  return(n);

}

/********************************************************************/
/*                                                                  */
/********************************************************************/
void draw_a_TRR(TrrType *trr) {
PointType pt[4];
int i, j, n;

  n = TRR2area(trr,pt);
  printf("move %f %f \n", pt[0].x, pt[0].y);
  for (i=0;i<n;++i) {
    j = (i+1)%n;
    printf("draw %f %f \n", pt[j].x, pt[j].y);
  }
}
/****************************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void recalculate_JS() {
  ms2line(&L_MS,  &(JS[0][0]), &(JS[0][1]) );
  ms2line(&R_MS,  &(JS[1][0]), &(JS[1][1]) );
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void line_to_ms(TrrType *ms,double x1,double y1,double x2,double y2) {
double a,b;

  if (y1 >y2) { /* p2 must be higher than p1 */
    a = x1; 
    b = y1; 
    x1 = x2; 
    y1 = y2; 
    x2 = a; 
    y2 = b; 
  }
  ms->ylow = (x1+y1); 
  ms->yhi  = (x2+y2);
  ms->xlow = (x2-y2); 
  ms->xhi  = (x1-y1);

  
  /* *ms must be a Manhattan arc. */
  check_ms(ms); 
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void line2ms(TrrType *ms, PointType p1, PointType p2) {
  line_to_ms(ms,p1.x,p1.y,p2.x,p2.y);  
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void pts2TRR(PointType pts[], int n, TrrType *trr) {
TrrType ms0, ms1;
  
  if (n==1) {
    trr->MakeDiamond   (pts[0],  0);
  } else if (n==2) {
    line2ms(trr, pts[0], pts[1]);
  } else if (n==4) {
    line2ms(&ms0, pts[0], pts[1]);
    line2ms(&ms1, pts[3], pts[2]);

    *trr = ms0;
    trr->Enclose ( ms1 ) ;
    
  } else {
    assert(0);
  }
  
}
/********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/*                                                                  */
/********************************************************************/
int trrContain(TrrType *t1,TrrType *t2) {

 if ( t1->xhi>=t2->xhi-FUZZ && t1->xlow <= t2->xlow + FUZZ   &&
    t1->yhi>=t2->yhi-FUZZ && t1->ylow <= t2->ylow + FUZZ) {
      return(YES);
 } else { return(NO); 
 }
}


/*******************************************************************/
/* return min of 3 real numbers                                    */
/*******************************************************************/
double min3(double x1, double x2, double x3)
{ return(min (x1, min (x2, x3))); }

/*******************************************************************/
/* return max of 3 real numbers                                    */
/*******************************************************************/
double max3(double x1, double x2, double x3)
{ return( tMAX(x1,  tMAX(x2, x3))); }


/*******************************************************************/
/* return min of 4 real numbers                                    */
/*******************************************************************/
double min4(double x1, double x2, double x3,double x4)
{ return(min (x1, min (x2, min (x3,x4)))); }

/*******************************************************************/
/* return max of 4 real numbers                                    */
/*******************************************************************/
double max4(double x1, double x2, double x3,double x4)
{ return( tMAX(x1,  tMAX(x2,  tMAX(x3,x4)))); }


/****************************************************************************/
/*                                                                          */
/****************************************************************************/
double Point_dist(PointType p1, PointType p2) {
    return(ABS(p1.x - p2.x) + ABS(p1.y - p2.y));
}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
int same_Point(PointType p1, PointType p2) {
double dist;

  dist = ABS(p1.x - p2.x) + ABS(p1.y - p2.y);
  return(equal(dist,0));
}
/******************************************************************/
/*                                                               */
/******************************************************************/
int Same_Point_delay(PointType *p, PointType *q) {
 if ( same_Point(*p, *q) && equal(p->max, q->max) && equal(p->min, q->min) ) {
   return(YES);
 } else {
   return(NO);
 }
}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
int same_line(PointType p0,PointType p1,PointType q0,PointType q1) {
  if (same_Point(p0,q0) && same_Point(p1,q1)) {
    return(YES);
  }
  if (same_Point(p1,q0) && same_Point(p0,q1)) {
    return(YES);
  }
  return(NO);
}


/******************************************************************/
/*                                                                */
/******************************************************************/
int parallel_line(PointType p1,PointType p2,PointType p3,PointType p4) {
double dx1,dx2,dy1,dy2;

  if (same_Point(p1,p2) || same_Point(p3,p4)) return(NO);

  dx1 = p1.x-p2.x; dy1 = p1.y-p2.y;
  dx2 = p3.x-p4.x; dy2 = p3.y-p4.y;

  if ( equal(dx1,0) && equal(dx2,0)) { /* parallel vertical lines */
    return(YES);
  } else if ( !equal(dx1,0)&& ! equal(dx2,0) && equal(dy1/dx1,dy2/dx2)) {
     return(YES);  /* other or horizontal parallel lines */
  }
  return(NO);
}

/****************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao     */
/****************************************************************************/
int areaJS_line_type(AreaType *area) {
int line_type;

  line_type = calc_line_type(area->line[0][0], area->line[0][1]);
  return(line_type); 
}
/****************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao     */
/****************************************************************************/
int JS_line_type(NodeType *node) {
  return(areaJS_line_type(&(node->area[node->ca])) ); 
}
/****************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao     */
/****************************************************************************/
int area_Manhattan_arc_JS(AreaType *area) {
  return(areaJS_line_type(area) == MANHATTAN_ARC);
}
/****************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao     */
/****************************************************************************/
int Manhattan_arc_JS(NodeType *node) {
  return(JS_line_type(node) == MANHATTAN_ARC);
}

/****************************************************************************/
/*   check if (x,y) is in line_segment p1,p2 if yes, eliminate the epsilon 
     error.  */
/****************************************************************************/
int ON_line_segment(double *x,double *y,double x1,double y1,double x2,double y2) {
double d,d1,d2,t, new_x, new_y;
int ans = NO;

  d = ABS(x1-x2) + ABS(y1 - y2);      
  d1 = ABS(x1 - *x) + ABS(y1 - *y);                  
  d2 = ABS(x2 - *x) + ABS(y2 - *y);                  
/*
  if ( equal(d1+d2,d) ) {
*/
  if ( ABS(d1+d2-d)<= 2*FUZZ ) {
    if ( equal(d1,0)) {
      *x   = x1;     
      *y   = y1;      
      ans = YES;
    } else if ( equal(d2,0) ) {
      *x   = x2;     
      *y   = y2;      
      ans = YES;
    } else {
      if (ABS(y2  - y1) > ABS(x2-x1)) {
        new_x = (x2-x1)*(*y-y1)/(y2-y1) + x1;
        new_y = *y;
        t = ABS(new_x - *x);      
      } else {
        new_x = *x; 
        new_y = (y2-y1)*(*x-x1)/(x2-x1) + y1;
        t = ABS(new_y - *y);  
      }
   /* 
     if ( t < LARGE_FUZZ) {      
   */
     if ( t < FUZZ) {
        ans = YES;
        *x = new_x;
        *y = new_y;
      }
    }
  }
  return(ans);
}

/****************************************************************************/
/* check if (x,y) is in line_segment p1,p2                                  */
/* if yes, eliminate the epsilon error. */
/****************************************************************************/
int PT_on_line_segment(PointType *pt,PointType pt1,PointType pt2) {
int ans;

  ans = ON_line_segment(&(pt->x),&(pt->y),pt1.x,pt1.y,pt2.x,pt2.y);
  return(ans); 
}

/****************************************************************************/
/* check if (x,y) is in line_segment p1,p2                                  */
/****************************************************************************/
int on_line_segment(double x,double y,double x1,double y1,double x2,double y2)
{
  return(ON_line_segment(&x,&y,x1,y1,x2,y2));
}
/****************************************************************************/
/* check if point pt is in line_segment p1, p2                           */
/****************************************************************************/
int pt_on_line_segment(PointType pt,PointType pt1,PointType pt2)
{
  return(ON_line_segment(&(pt.x),&(pt.y),pt1.x,pt1.y,pt2.x,pt2.y));
}


/********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/********************************************************************/
double ms_distance(TrrType *ms1,TrrType *ms2) {
  double x1a,x1b,y1a,y1b,x2a,x2b,y2a,y2b;
  double d1,d2,d3,d4;
  double t1,t2,t3;

  /* remember that ms1 and ms2 are represented in the Linfinity metric */
  /* first check that these are valid merging segments */
  if (0==1) check_ms(ms1);
  if (0==1) check_ms(ms2);

  x1a = ms1->xhi; y1a = ms1->yhi;
  x1b = ms1->xlow; y1b = ms1->ylow;

  x2a = ms2->xhi; y2a = ms2->yhi;
  x2b = ms2->xlow; y2b = ms2->ylow;

  /*there are four cases to consider:
      (1) there is no intersetion tween x-coords of ms1 and x-coords of ms2
          and no intersetion tween y-coords of ms1 and y-coords of ms2
      (2) no intersection tween x-coords, but some intersection in y-coords
      (3) no intersection tween y-coords, but some intersection in x-coords
      (4) intersection in both coords (distance=0)

      (1) find min distance between endpoints
      (2) find min distance in x-coords
      (3) find min distance in y-coords
      (4) return distance = 0
      */

    /* if no intersection between x & y coordinates at all, take
       the min distance between endpoints */
  if ( ((x1a < x2b) || (x2a < x1b))  && ((y1a < y2b) || (y2a < y1b)) ) {

    /* find the distance between all 4 pairs of endpoints */
    d1 =  tMAX(fabs(x1a-x2a),fabs(y1a-y2a));
    d2 =  tMAX(fabs(x1b-x2a),fabs(y1b-y2a));
    d3 =  tMAX(fabs(x1a-x2b),fabs(y1a-y2b));
    d4 =  tMAX(fabs(x1b-x2b),fabs(y1b-y2b));

    t1 =  tMIN(d1,d2);
    t2 =  tMIN(d3,d4);
    t3 =  tMIN(t1,t2);
    return( t3 );
  }
  else if ((x1a < x2b) || (x2a < x1b)) {
    t1 = x2b - x1a;
    t2 = x1b - x2a;
    t3 =  tMAX(t1,t2);  /* take max here because one will be negative */
    return( t3 );
  }
  else if ((y1a < y2b) || (y2a < y1b)) {
    t1 = y2b - y1a;
    t2 = y1b - y2a;
    t3 =  tMAX(t1,t2); /* take max here because one will be negative */
    return( t3 );
  }
  else return( 0 );
  
} /* ms_distance() */

/********************************************************************/
/*                                                                  */
/********************************************************************/
double pt2ms_distance(PointType *pt, TrrType *ms) {
TrrType trr;

  trr.MakeDiamond ( *pt, 0 ) ;
  return(ms_distance(&trr, ms));
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
double pt2TRR_distance_sub(PointType *pt, TrrType *trr) {
TrrType ms[4];
double dist;
int i;

  TrrType PtTrr ( *pt, 0 ) ;
  if (trrContain(trr,&PtTrr)) {
    dist = 0;
  } else {
    for (i=0;i<4;++i) ms[i] = *trr;
    ms[0].ylow = trr->yhi;
    ms[1].yhi = trr->ylow;
    ms[2].xlow = trr->xhi;
    ms[3].xhi = trr->xlow;
    dist = ms_distance(&PtTrr, &(ms[0]));
    for (i=1;i<4;++i) {
      dist = min (dist, ms_distance(&PtTrr, &(ms[i])));
    }
  }
  return(dist);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
double pt2TRR_distance(PointType *pt, PointType pts[], int n) {
TrrType trr;
double dist;

  pts2TRR(pts, n, &trr);
  dist = pt2TRR_distance_sub(pt, &trr);

  return(dist);
}
/****************************************************************************/
/* return the shortest distance between point p1 and line p2-p3 */
/****************************************************************************/
int pt2linedist_sub(PointType p1, PointType p2, PointType p3,
                    PointType candidate_pt[4]) {
int n;


  n = 0;
  if ( !equal(p2.x,p3.x) && (p1.x-p2.x)*(p1.x-p3.x) <= 0) {
    candidate_pt[n].x = p1.x;
    candidate_pt[n].y = (p3.x-p1.x)*(p2.y-p3.y)/(p3.x-p2.x) + p3.y;
    n++;
  }
  if ( !equal(p2.y,p3.y) && (p1.y-p2.y)*(p1.y-p3.y) <= 0) {
    candidate_pt[n].x = (p3.y-p1.y)*(p2.x-p3.x)/(p3.y-p2.y) + p3.x;
    candidate_pt[n].y = p1.y;
    n++;
  }

  if (n < 2) {
    candidate_pt[n++]=p2; candidate_pt[n++]=p3;
  }

  return(n);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
double pt2linedist_case_ms(PointType p1, PointType p2, PointType p3, 
       PointType *ans) {
TrrType ms0, ms1, trr0, ms2;
double dist;

  ms0.MakeDiamond(p1, 0);
  line2ms(&ms1,p2,p3);
  dist = ms_distance(&ms0, &ms1);
  trr0.MakeDiamond(p1, dist);
  make_intersect(&trr0,&ms1,&ms2);
  core_mid_point(&ms2, ans);

  return(dist);
}
/****************************************************************************/
/* return the shortest distance between point p1 and line p2-p3 */
/****************************************************************************/
double pt2linedist(PointType p1, PointType p2, PointType p3, PointType *ans) {
int i,n;
PointType candidate_pt[4];
double d, min_d, a, b;


  a = ABS(p3.x - p2.x); 
  b = ABS(p3.y - p2.y); 
  if (same_Point(p2,p3)) {
    *ans = p2;
    min_d = Point_dist(p1,p2);
  } else if (pt_on_line_segment(p1,p2,p3)) {
    *ans = p1;
    min_d = 0;
  } else if (equal(a,b)) {  /* p1, p2 is Manhattan arc */
    min_d = pt2linedist_case_ms(p1,p2,p3,ans);
  } else {
    n = pt2linedist_sub(p1, p2, p3, candidate_pt);
    min_d = DBL_MAX;
    for (i=0;i<n;++i) {
      d = Point_dist(candidate_pt[i],p1);
      if (d < min_d) { min_d = d; *ans=candidate_pt[i]; }
    }
  }
  assert(pt_on_line_segment(*ans,p2,p3));
  return(min_d);
}



/****************************************************************************/
/*  check if two line segments intersect.                                   */
/****************************************************************************/
int L_intersect(double *x, double *y, double x1, double y1,double x2, 
           double y2, double x3, double y3, double x4, double y4) {
double mm,nn;
int count;

  /* zero-length_edge */ 
  if (ABS(x1-x2)+ABS(y1-y2) < FUZZ || ABS(x3-x4)+ABS(y3-y4) < FUZZ) 
    assert(0);             
  
  if (!bbox_overlap(x1,y1,x2,y2,x3,y3,x4,y4)) return(NO);

  if (ABS(x1-x3)+ABS(y1-y3) + ABS(x2-x4)+ABS(y2-y4) < FUZZ)  return(SAME_EDGE);

  count = 0;
   *x=DBL_MAX; *y=DBL_MAX;
  if (on_line_segment(x1,y1,x3,y3,x4,y4)) {
    *x = x1;
    *y = y1;
    count++;
  }
  if (on_line_segment(x2,y2,x3,y3,x4,y4)) {
    *x = x2;
    *y = y2;
    count++;
  }
  if (on_line_segment(x3,y3,x1,y1,x2,y2)) {
    *x = x3;
    *y = y3;
    count++;
  }
  if (on_line_segment(x4,y4,x1,y1,x2,y2)) {
    *x = x4;
    *y = y4;
    count++;
  }
  if (ABS(x1-x3)+ABS(y1-y3) < FUZZ) count--;        
  if (ABS(x1-x4)+ABS(y1-y4) < FUZZ) count--;        
  if (ABS(x2-x3)+ABS(y2-y3) < FUZZ) count--;        
  if (ABS(x2-x4)+ABS(y2-y4) < FUZZ) count--;        
  if (count>=2) return(OVERLAPPING);


   if (equal(x1,x2) &&  equal(x3,x4)) {
      /* parallel vertical lines  has been considered */
     
   } else if ( equal(x1,x2) &&  !equal(x3,x4)) {  /* p1p2 is a vertical line */
     *x=x1;
     *y=y3+(y3-y4)*(x1-x3)/(x3-x4);
   } else if ( !equal(x1,x2) &&  equal(x3,x4)) {  /* p3p4 is a vertical line */
     *x=x3;
     *y=y1+(y2-y1)*(x3-x1)/(x2-x1);
   } else if ( !equal(x1,x2) &&  !equal(x3,x4)) {
     mm=(y2-y1)/(x2-x1);
     nn=(y4-y3)/(x4-x3);
   /* overlapping parallel line segments have been considered */ 
     if (equal(mm,nn)) {    
        return(NO);         /* disjoint parallel line segments */
     } else {               /* Non-parallel line segments */
       *x=(y3-y1+mm*x1-nn*x3)/(mm-nn);
       *y=(y1+y3+mm*(*x - x1)+nn*(*x - x3))/2.0;
     }
   }

   if (in_bbox(*x,*y,x1,y1,x2,y2) && in_bbox(*x,*y,x3,y3,x4,y4) ){

     if ( equal(*x,x1) && equal(*y,y1) ) {
       return(P1);
     } else if ( equal(*x,x2) && equal(*y,y2) ) {
       return(P2);
     } else if ( equal(*x,x3) && equal(*y,y3) ) {
       return(P3);
     } else if ( equal(*x,x4) && equal(*y,y4) ) {
       return(P4);
     } else {
       return(XING);
     }
   }
   return(NO);
}
/****************************************************************************/
/*  check if two line segments intersect.                                   */
/****************************************************************************/
int lineIntersect(PointType *p, PointType p1, PointType p2, PointType p3, 
              PointType p4) {
double x, y; 
int ans; 

  ans = L_intersect(&x,&y,p1.x,p1.y,p2.x,p2.y,p3.x,p3.y,p4.x,p4.y);
  p->x = x;
  p->y = y;
  return(ans);
}

/*****************************************************************************/
/* calc Manhattan distance between two lines; ans[2] = two closest points */
/*****************************************************************************/
double linedist(PointType lpt0,PointType lpt1, PointType lpt2, PointType lpt3,
                PointType ans[2]) {
double dist, mindist = DBL_MAX;
PointType intersect, pt[2][2];
int i,j,k,n0,n1; 

  pt[0][0] = lpt0;
  pt[0][1] = lpt1;
  pt[1][0] = lpt2;
  pt[1][1] = lpt3;

  n0=n1 =2;
  if (same_Point(lpt0,lpt1)) n0=1;
  if (same_Point(lpt2,lpt3)) n1=1;

  if (n0==2 && n1==2) {
    k = lineIntersect(&intersect,lpt0,lpt1,lpt2,lpt3);
    if (k>0) {
     ans[0] = ans[1] = intersect; 
     return(0.0);
    }
  }

  for (i=0;i<n0;++i) {
    k = (i+1)%2; 
    for (j=0;j<n1;++j) {
      dist = pt2linedist(pt[i][j], pt[k][0], pt[k][1], &intersect);
      if (dist < mindist ) {
        mindist = dist;
        ans[i] = pt[i][j]; 
        ans[k] = intersect; 
      }
    }
  }
  return(mindist);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void calc_Linear_merge_distance(double delay1,double delay2,double d,
                           double *d1,double *d2) {
double diff;

  diff = fabs(delay1 - delay2);

  if (d > diff) {
    *d1 = (d + delay2 - delay1) / 2;
    *d2 = d - *d1;
  } else if (delay1 > delay2) {
    *d1 = 0;
    *d2 = diff;
  } else {
    *d1 = diff;
    *d2 = 0;
  }
}


/********************************************************************/
/* calculate the merging distance using calcs from Tsay's article */
/********************************************************************/
void calc_Elmore_merge_distance(double r, double c, double cap1,double delay1,
     double cap2, double delay2,double d,double *d1,double *d2)  {
double x,y,z;
 
  y = (delay2 - delay1 + r*d*(cap2 + c*d/2));
  z = (r*(cap1 + cap2 + c*d));
  x = y/z;
 
  if (x<0) {
    cap2 *= r;
    x = (double) (sqrt(cap2*cap2 + 2*r*c* (delay1-delay2)) - cap2) / (r*c);
    d1[0] = 0;
    d2[0] = x;
  } else if (x>d) {
    cap1 *= r;
    x = (double)  (sqrt(cap1*cap1 + 2*r*c* (delay2-delay1)) - cap1) / (r*c);
    d1[0] = x;
    d2[0] = 0;
  } else {
    d1[0] = x;
    d2[0] = d-x;
  }
}  /* calc_merge_distance */
 
/********************************************************************/
/* calculate the merging distance using calcs from Tsay's article */
/********************************************************************/
void calc_merge_distance(double r,double c, double cap1,double delay1,
      double cap2, double delay2, double d,double *d1,double *d2)  {
  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
    calc_Linear_merge_distance(delay1,delay2,d,d1,d2);
  } else {
    calc_Elmore_merge_distance(r,c, cap1,delay1,cap2, 
                               delay2,d,d1,d2);   
  }
}

/********************************************************************/
/* calculate the merging distance using calcs from Tsay's article */
/********************************************************************/
void new_calc_merge_distance(PointType pt1, PointType pt2, int delay_id,
                         double *d1,double *d2) {
double x, y, delay1, delay2, d, r, c;
 
  if (delay_id==0 ) {
    delay1 = pt1.max;
    delay2 = pt2.max;
  } else {
    delay1 = pt1.min;
    delay2 = pt2.min;
  }
 
  x = ABS(pt1.x - pt2.x);
  y = ABS(pt1.y - pt2.y);
 
  d = x+y;
 
  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
    calc_Linear_merge_distance(delay1,delay2,d,d1,d2);
  } else {
    if (equal(y,0)) {  /* Horizontal line segments */
      r = PURES[H_]; c = PUCAP[H_];
    } else {
      r = PURES[V_]; c = PUCAP[V_];
    }
    calc_Elmore_merge_distance(r, c, pt1.t,delay1,pt2.t, delay2,d,d1,d2);
  }
}
 
/******************************************************************/
/*                                                               */
/******************************************************************/
double calc_delay_increase(double pattern, double cap,double x, double y) {
double t, t0, t1;

  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
     t = x+y; 
  } else {/* Elmore delay model */
    t0 = PURES[H_]*x*(PUCAP[H_]*x/2+cap) + 
      PURES[V_]*y*(PUCAP[V_]*y/2+cap+x*PUCAP[H_]);
    t1 = PURES[V_]*y*(PUCAP[V_]*y/2+cap) + 
      PURES[H_]*x*(PUCAP[H_]*x/2+cap+y*PUCAP[V_]);
    t = pattern*t0 + (1-pattern)*t1;
  }
  return(t);
}

/******************************************************************/
/*                                                               */
/******************************************************************/
double pt_delay_increase(double pat, double cap, PointType *q0, PointType *q1) {
double t;

  t = calc_delay_increase(pat, cap, ABS(q1->x - q0->x),ABS(q1->y - q0->y));
  assert(t>=0);
  return(t);
}

/******************************************************************/
/*                                                               */
/******************************************************************/
double _pt_delay_increase(double pat, double leng,double cap,PointType *q0,PointType *q1) {
double h,v,t;

  h = ABS(q0->x - q1->x);
  v = ABS(q0->y - q1->y);
  assert( equal(leng, h+v) || leng >= h+v);
  if (equal(h,0)) {
    t = calc_delay_increase(pat, cap, 0,leng);
  } else if (equal(v,0)) {
    t = calc_delay_increase(pat, cap, leng, 0);
  } else {
    t = calc_delay_increase(pat, cap,h,v);
    if (leng > h + v) {
      t += calc_delay_increase(pat, cap + h*PUCAP[H_]+v*PUCAP[V_], 0, leng-h-v);
    }
  }
  return(t);
}

/********************************************************************/
/*  calculate the points with d0 to q0 and d1 to q1           */
/*  , where all the points are on a line.                           */
/********************************************************************/
void calc_pt_coor_on_a_line(PointType q0,PointType q1, double d0,double d1, 
                            PointType *pts) {
double d, t;
 
  d = d0+d1;
  t= Point_dist(q0,q1);
  assert( equal(d,t) || d > t);
  assert(d0 >=0 );
  assert(d1 >=0 );
  if ( equal(d0,0) ) {
    pts->x = q0.x;
    pts->y = q0.y;
  } else if ( equal(d1,0) ) {
    pts->x = q1.x;
    pts->y = q1.y;
  } else {
    pts->x = (q0.x*d1+q1.x*d0)/d;
    pts->y = (q0.y*d1+q1.y*d0)/d;
  }
}
 
/******************************************************************/
/*                                                               */
/******************************************************************/
void check_calc_Bal_of_2pt(PointType *pt0, PointType *pt1, int delay_id, 
                           PointType *balPt, double d0, double d1) {
double t0, t1;

  t0 =  _pt_delay_increase(Gamma, d0, pt0->t, pt0, balPt);
  t1 =  _pt_delay_increase(Gamma, d1, pt1->t, pt1, balPt);

  if (delay_id == 0) {
    assert(equal(pt0->max + t0, pt1->max + t1));
  } else {
    assert(equal(pt0->min + t0, pt1->min + t1));
  }

  if (pt_skew(*pt0) <= Skew_B && pt_skew(*pt1) <= Skew_B ) {
    check_Point(balPt);
  }
  if (equal(d0,0)) {
    assert(same_Point(*pt0,*balPt));
  }
  if (equal(d1,0)) {
    assert(same_Point(*pt1,*balPt));
  }

}

/******************************************************************/
/*  balace pt on the recilinear line (pt0,pt1)                   */
/******************************************************************/
void Balance_of_line(PointType *pt0, PointType *pt1, int delay_id,
     double *d0, double *d1, PointType *BalPt) {
double h,v, t0, t1;
double r,c, delay0, delay1;
 
  if (delay_id==0 ) {
    delay0 = pt0->max;
    delay1 = pt1->max;
  } else {
    delay0 = pt0->min;
    delay1 = pt1->min;
  }
 
  h = ABS(pt0->x - pt1->x);
  v = ABS(pt0->y - pt1->y);
 
  assert(equal(h,0) || equal(v,0) );
  if (equal(h,0)) {   /* vertical line (default) */
    r = PURES[V_]; 
    c = PUCAP[V_];
  } else {            /* horizontal line */
    r = PURES[H_]; 
    c = PUCAP[H_];
  }
  calc_merge_distance(r,c, pt0->t,delay0,pt1->t,delay1,h+v,d0,d1);
  calc_pt_coor_on_a_line(*pt0,*pt1, *d0, *d1, BalPt);
  if (equal(h,0)) { 
    t0 = calc_delay_increase(Gamma, pt0->t, 0, *d0);
    t1 = calc_delay_increase(Gamma, pt1->t, 0, *d1);
  } else {
    t0 = calc_delay_increase(Gamma, pt0->t, *d0, 0);
    t1 = calc_delay_increase(Gamma, pt1->t, *d1, 0);
  }
  BalPt->max =  tMAX(pt0->max + t0, pt1->max + t1);
  BalPt->min =  tMIN(pt0->min + t0, pt1->min + t1);
  assert(equal(delay0+t0, delay1 + t1) );

  /* check */
  check_calc_Bal_of_2pt(pt0, pt1, delay_id, BalPt, *d0, *d1);
}

/******************************************************************/
/*                                                               */
/******************************************************************/
double calc_y_Bal_position(double delay0, double delay1, double cap0,
     double cap1, double h, double v, int bal_pt_id) {
double y, t0, t1, t;

  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
    if (bal_pt_id==0) {
      y = (delay1 - delay0 + v+h)/2;
    } else {
      y = (delay1 - delay0 + v-h)/2;
    }
  } else {
    t = Gamma*PURES[V_]*PUCAP[H_] + (1-Gamma)*PURES[H_]*PUCAP[V_];
    t1 = PURES[V_]*(cap0+cap1)+ 2*v*K[V_]+ t*h;
    if (bal_pt_id==0) {
      t0 = delay1-delay0+K[H_]*h*h+K[V_]*v*v +
           cap1*(PURES[H_]*h+PURES[V_]*v)+ t*h*v;
      y = t0/t1;
      assert(y <= v);
    } else {
      t0 = delay1-delay0+K[V_]*v*v - K[H_]*h*h + PURES[V_]*v*cap1
           -PURES[H_]*cap0*h;
      y = t0/t1;
      assert(y >= 0);
    }
  }

  return(y);
}

/******************************************************************/
/*                                                               */
/******************************************************************/
double calc_x_Bal_position(double delay0, double delay1, double cap0,
     double cap1, double h, double v, int bal_pt_id) {
double x, t0, t1, t;

  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
    if (bal_pt_id==0) {
      x = (delay1 - delay0 + h-v)/2;
    } else {
      x = (delay1 - delay0 + h+v)/2;
    }
  } else {
    t = Gamma*PURES[V_]*PUCAP[H_] + (1-Gamma)*PURES[H_]*PUCAP[V_];
    if (bal_pt_id==0) {
      t0 = delay1-delay0+K[H_]*h*h - K[V_]*v*v + PURES[H_]*h*cap1
         -PURES[V_]*v*cap0;
    } else {
      t0 = delay1-delay0+K[H_]*h*h+  K[V_]*v*v +
           cap1*(PURES[H_]*h+PURES[V_]*v)+ t*h*v;
    }
    t1 = PURES[H_]*(cap0+cap1)+ t*v + 2*h*K[H_];
    x = t0/t1;
  }

  return(x);
}

/******************************************************************/
/*                                                               */
/******************************************************************/
void Bal_Pt_not_on_line(PointType *pt0, PointType *pt1, int delay_id,
     int bal_pt_id, double *d0, double *d1, PointType *BalPt) {
 
double delay0, delay1 ;
double x, y, t0, t1, t;
PointType tmp_PT;
 
  double h = ABS(pt0->x - pt1->x);
  double v = ABS(pt0->y - pt1->y);
 
  assert( !equal(h,0) && !equal(v,0) );
  assert(  pt0->x <= pt1->x);
 
  if (delay_id==0 ) {
    delay0 = pt0->max;
    delay1 = pt1->max;
  } else {
    delay0 = pt0->min;
    delay1 = pt1->min;
  }
 
  x= calc_x_Bal_position(delay0,delay1,pt0->t,pt1->t,h,v,bal_pt_id);
  if (x<0) {
    if (bal_pt_id==0) {
      y = calc_y_Bal_position(delay0,delay1,pt0->t,pt1->t,h,v,bal_pt_id);
    } else {
      y = -1;
    }
    if (y>=0) x=0;
  } else if (x > h) {
    if (bal_pt_id==0) {
      y = v + 1;
    } else {
      y = calc_y_Bal_position(delay0,delay1,pt0->t,pt1->t,h,v,bal_pt_id);
    }
    if (y<=v) x=h;
  }  else {
    if (bal_pt_id==0) { y = v; } else {y=0;}
  }
  if ( x <0 ) {
    assert(y<0);
    tmp_PT.x = pt0->x;
    tmp_PT.y = pt0->y;
    tmp_PT.t = pt1->t + PUCAP[H_]*h + PUCAP[V_]*v;
    t1 = calc_delay_increase(Gamma, pt1->t, h, v);
    tmp_PT.max = pt1->max + t1;
    tmp_PT.min = pt1->min + t1;
    Balance_of_line(pt0,&tmp_PT,delay_id,d0,d1, BalPt);
    assert(*d0==0);
    t = calc_delay_increase(Gamma, tmp_PT.t, 0, *d1);
    assert(equal(delay0,t+t1+delay1));
    *(d1) += h+v;
  } else if  (x>h) {
    assert(y>v);
    tmp_PT.x = pt1->x;
    tmp_PT.y = pt1->y;
    tmp_PT.t = pt0->t + PUCAP[H_]*h + PUCAP[V_]*v;
    t0 = calc_delay_increase(Gamma, pt0->t, h, v);
    tmp_PT.max = pt0->max + t0;
    tmp_PT.min = pt0->min + t0;
    Balance_of_line(&tmp_PT,pt1, delay_id,d0,d1, BalPt);
    assert(*d1==0);
    t = calc_delay_increase(Gamma, tmp_PT.t, 0, *d0);
    assert(equal(t+t0+delay0,delay1));
    *(d0) += h+v;
  } else {
    assert( y>=0 && y <= v);
    BalPt->x = pt0->x + x;
    if (pt0->y < pt1->y) {
      BalPt->y = pt0->y + y;
    } else {
      BalPt->y = pt0->y - y;
    }
    t0 = calc_delay_increase(Gamma, pt0->t, x, y);
    t1 = calc_delay_increase(Gamma, pt1->t, h-x,v-y);
    BalPt->max =  tMAX( pt0->max + t0, pt1->max + t1);
    BalPt->min =  tMIN( pt0->min + t0, pt1->min + t1);
    *d0 = x + y;
    *d1 = h+v -x -y;
    assert(equal(t0+delay0,t1+delay1));
  }
  assert(*d0+*d1 >= h+v - FUZZ);
  check_calc_Bal_of_2pt(pt0, pt1, delay_id, BalPt, *d0, *d1);
}
/******************************************************************/
/*                                                               */
/******************************************************************/
void calc_Bal_of_2pt(PointType *pt0, PointType *pt1, int delay_id,
     int bal_pt_id, double *d0, double *d1, PointType *BalPt) {
double h, v;
 
  h = ABS(pt0->x - pt1->x);
  v = ABS(pt0->y - pt1->y);
 
  if (equal(h,0) || equal(v,0) ) {
    Balance_of_line(pt0,pt1,delay_id, d0,d1, BalPt);
  } else if (pt0->x <= pt1->x ) {
    Bal_Pt_not_on_line(pt0, pt1, delay_id, bal_pt_id, d0,d1, BalPt);
  } else {
    Bal_Pt_not_on_line(pt1, pt0, delay_id, bal_pt_id, d1,d0, BalPt);
  }
  check_calc_Bal_of_2pt(pt0, pt1, delay_id, BalPt, *d0, *d1);
}


/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void calc_vertices_sub(AreaType *area) {
int i,j;
PointType pt, line0, line1;

  area->vertex[0] = 0; 
  for (i=1, j = 1;i<area->n_mr;++i) { /* remove colinear turn points on mr */
    pt = area->mr[i];
    line0 = area->mr[area->vertex[j-1]];
    line1 = area->mr[(i+1)%area->n_mr];
    if (!pt_on_line_segment(pt,line0,line1)) {
      area->vertex[j++] = i ; 
    }
  }
  area->npts = j; 
}
/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void calc_vertices(AreaType *area) {
  if (area->n_mr<=2) {
    area->vertex[0] = 0;
    area->vertex[1] = 1;
    area->npts = area->n_mr;
  } else {
    calc_vertices_sub(area);
  }
}

/******************************************************************/
/* calculate the delays of pt on a well-behaved line segment p0p2; */
/******************************************************************/
void calc_pt_delays_sub(AreaType *area,PointType *q1,PointType q0,PointType q2){
double A, B, m, t, tL, tR, d1, d2, a, b;
int type;

  a = ABS(q0.x-q2.x);
  b = ABS(q0.y-q2.y);

  type = areaJS_line_type(area);
  if (type == MANHATTAN_ARC) {
    assert( PURES_V_SCALE != 1);
    assert(same_Point(area->line[0][0], area->line[0][1]));
    assert(same_Point(area->line[1][0], area->line[1][1]));
    tL = pt_delay_increase(Gamma, area->area_L->capac, &(area->line[0][0]), q1);
    tR = pt_delay_increase(Gamma, area->area_R->capac, &(area->line[1][0]), q1);
    q1->max =  tMAX(area->line[0][0].max+tL, area->line[1][0].max+tR);
    q1->min =  tMIN(area->line[0][0].min+tL, area->line[1][0].min+tR);
    assert(pt_skew(*q1) < Skew_B + FUZZ);
  } else {
    assert(type==VERTICAL || type==HORIZONTAL);
    d1 = Point_dist(q0,*q1);
    d2 = Point_dist(q0,q2);
    if (a > b) {
      m =b/a;
      t = (1+ABS(m))*(1+ABS(m));
      A = (K[H_]+m*m*K[V_])/t;
    } else {
      m = a/b;
      t = (1+ABS(m))*(1+ABS(m));
      A = (m*m*K[H_]+K[V_])/t;
    }
    B = (q2.max-q0.max)/d2 -  A*d2;
    q1->max = q0.max + A*d1*d1 + B*d1;
    B = (q2.min-q0.min)/d2 -  A*d2;
    q1->min = q0.min + A*d1*d1 + B*d1;
  }
}



/******************************************************************/
/* calculate the delays of pt on a well-behaved line segment p0p2*/
/******************************************************************/
void calc_pt_delays(AreaType *area, PointType *q1,PointType q0,PointType q2) {
double d1,d2,a,b, A,B;
 
  assert(pt_on_line_segment(*q1,q0,q2));
  d1 = Point_dist(q0,*q1);
  a = ABS(q0.x-q2.x);
  b = ABS(q0.y-q2.y);
  d2 = a+b;
 
  if (equal(d1,0)) {
    q1->max = q0.max;
    q1->min = q0.min;
  } else if (same_Point(*q1,q2)) {
    q1->max = q2.max;
    q1->min = q2.min;
  } else if (equal(a, b) ) { /* p0p1 is a Manhattan arc, a point */
    assert(equal(q0.max,q2.max) && equal(q0.min,q2.min));
    q1->max = q0.max = q2.max;
    q1->min = q0.min = q2.min;
  } else if (equal(a,0) || equal(b,0)) {  /* Vertical || Horizontal */
    A = equal(a,0)? K[V_]:K[H_];
    B = (q2.max-q0.max)/d2 -  A*d2;
    q1->max = q0.max + A*d1*d1 + B*d1;
    B = (q2.min-q0.min)/d2 -  A*d2;
    q1->min = q0.min + A*d1*d1 + B*d1;
  } else {
    assert(area!=NULL);
    /* no skew reservation */
      assert(equal(pt_skew(q0),Skew_B) && equal(pt_skew(q2),Skew_B));
    
    calc_pt_delays_sub(area, q1, q0, q2);
  }
  check_Point_delay(q1);
}



/****************************************************************************/
/*  calc the maximum joining segment length for Node[v]             */
/****************************************************************************/
double calc_JR_area_sub(PointType p0,PointType p1,PointType p2,PointType p3) {
double max_x, max_y, min_x, min_y, t;
double a,b,c,d;

  max_x =  max4(p0.x,p1.x,p2.x,p3.x);
  max_y =  max4(p0.y,p1.y,p2.y,p3.y);
  min_x =  min4(p0.x,p1.x,p2.x,p3.x);
  min_y =  min4(p0.y,p1.y,p2.y,p3.y);
  t = (max_x -min_x)*(max_y-min_y);
  a = ABS(p0.x-p1.x); b = ABS(p0.y-p1.y);
  c = ABS(p2.x-p3.x); d = ABS(p2.y-p3.y);
  t -= 0.5*(a*b + c*d);
  return(t);

}


/****************************************************************************/
/*  calc the maximum joining segment length for Node[v]             */
/****************************************************************************/
double calc_boundary_length(AreaType *area) {
int i, v1, v2;
double t;

  t = 0;
  for (i=0;i<area->npts;++i) {
    v1 = area->vertex[i];
    v2 = area->vertex[(i+1)%area->npts];
    t += Point_dist(area->mr[v1],area->mr[v2]);
  }
  return(t);
}

/****************************************************************************/
/*  calc the maximum joining segment length for Node[v]             */
/****************************************************************************/
double calc_area(AreaType *area) {
int i, v1, v2;
double t, h, min_y;

  min_y = DBL_MAX;
  for (i=0;i<area->npts;++i) {
     v1 = area->vertex[i];
     min_y = min (min_y, area->mr[v1].y);
  }
  t = 0;
  for (i=0;i<area->npts;++i) {
    v1 = area->vertex[i];
    v2 = area->vertex[(i+1)%area->npts];
    h = area->mr[v2].x - area->mr[v1].x;
    t += h*((area->mr[v1].y + area->mr[v2].y)/2 - min_y);
  }
  return(t);
}


/*************************************************************************/
/*                                                                       */
/*************************************************************************/
void calc_TreeCost_sub(int par, int v, double *Tcost, double *Tdist, int
    *n_detour) {
double dist, length;

  if (v<0) return;
  PointType &pt = Node[par].m_stnPt ;
  PointType &qt = Node[v].m_stnPt ;
  dist = Point_dist(pt, qt );
  length = EdgeLength[v];
  assert( equal(length,dist) || dist < length);
  if (dist < length) {
    (*n_detour)++;
  }
  *Tcost += length;
  *Tdist += dist;
  calc_TreeCost_sub(v,Node[v].L, Tcost, Tdist, n_detour);
  calc_TreeCost_sub(v,Node[v].R, Tcost, Tdist, n_detour);
}
/*************************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/*************************************************************************/
int calc_TreeCost(int v, double *Tcost, double *Tdist) {
int n_detour;

  n_detour = 0; 
  *Tcost = *Tdist = 0;

  calc_TreeCost_sub(v, Node[v].L, Tcost, Tdist, &n_detour);
  calc_TreeCost_sub(v, Node[v].R, Tcost, Tdist, &n_detour);
  return(n_detour);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
static 
void calc_fanout_sub(int v, int *n) {

  int nterms = (int) gBoundedSkewTree->Nterms() ;
  if (v<nterms || Buffered[v]>0) {
    (*n)++;
  } else {
    calc_fanout_sub(Node[v].L, n);
    calc_fanout_sub(Node[v].R, n);
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int calc_buffer_fanout(int v) {
int n=0;

  int nterms = (int) gBoundedSkewTree->Nterms() ;
  assert(v>=nterms && Buffered[v] > 0 );

  calc_fanout_sub(Node[v].L, &n);
  calc_fanout_sub(Node[v].R, &n);

  return(n);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
int calc_cluster_size(int cid) {

  unsigned npoints = gBoundedSkewTree->Npoints() ;
  for (unsigned i=0;i<npoints;++i) {
    if (Buffered[i] > 0  && Cluster_id[Node[i].L]==cid) {
      assert(Cluster_id[Node[i].R] == cid);
      return(calc_buffer_fanout(i));
    }
  }
  return(0);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void print_buffer_delay_info(int L) {
int i, buffer[MAX_N_NODES], n_buffer;
double max_t, min_t, ave_t, sum, dev, t, Obj_Cost;

  n_buffer = 0;
  int npoints = (int) gBoundedSkewTree->Npoints() ;
  int superRoot = gBoundedSkewTree->SuperRootNodeIndex () ;
  for (i=0;i<npoints;++i) {
    if (Buffered[i]  > 0 && i != superRoot && level(i)==L ) {
      buffer[n_buffer++] = i; 
    }
  }
  max_t = -DBL_MAX;
  min_t =  DBL_MAX;
  sum = 0;
  for (i=0;i<n_buffer;++i) {
    PointType &pt = Node[ buffer[i] ].m_stnPt ;
    max_t =  tMAX(max_t, pt               .max);
    min_t =  tMIN(min_t, pt               .max);
    sum += pt               .max;
  }
  ave_t = sum/n_buffer;
  sum = Obj_Cost = 0;
  for (i=0;i<n_buffer;++i) {
    PointType &pt = Node[ buffer[i] ].m_stnPt ;
    t = pt               .max;
    sum +=  (t - ave_t)*(t - ave_t); 
    Obj_Cost +=  pow(t, 5.0); 
  }
  dev = sqrt(sum/(double) n_buffer);
  printf("****************\n");
  printf("deviation = %E (max_diff =%E) \n", dev, max_t-min_t);
  printf("Obj_Cost = %E \n", Obj_Cost);
}

/****************************************************************************/
/* calculate delay from a node to the root.                                 */
/****************************************************************************/
double calc_node_delay(int v, double delay, int mode) {
   
  int superRoot = gBoundedSkewTree->SuperRootNodeIndex () ;
  int root = gBoundedSkewTree->RootNodeIndex () ;

  if (v== root ) {
    PointType &pt = Node[ root ].m_stnPt ;
    PointType &qt = Node[ superRoot ].m_stnPt ;
    assert(same_Point(pt, qt ) ) ;
  }
  while (v!= root &&  Node[v].parent != NIL) {
    assert(EdgeLength[v] >= StubLength[v] - FUZZ);
    if (EdgeLength[v] < StubLength[v]) {
      printf("\n warning: 2340 \n"); 
    }
    delay +=  calc_edge_delay(v,Node[v].parent, mode);
    v = Node[v].parent;
  }
  return(delay);
}
/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void print_cluster_cost_at_level(int L, double wl[], double cap[]) {
int i, cid ,n, total_fanout=0;
double total_wl = 0, total_cap = 0, max_t, min_t, t;

  max_t = 0;
  min_t = DBL_MAX;
  int npoints = (int) gBoundedSkewTree->Npoints() ;
  
  int superRoot = gBoundedSkewTree->SuperRootNodeIndex () ;

  for (i=0;i<npoints;++i) {
    if (Buffered[i] > 0    && i != superRoot && level(i)==L ) {
      cid = Cluster_id[Node[i].L];
      printf("WL[%2d]=%+9.1f ", cid, wl[cid]); 
      n =  calc_buffer_fanout(i);
      total_fanout += n;
      total_wl +=  wl[cid];
      total_cap += cap[cid];
      PointType &pt = Node[i].m_stnPt ;
      printf("skew=%.0f (%.1f-%.1f)", pt       .max-pt       .min,
                                        pt       .max,pt       .min);
      printf("C%.3f ", Node[i].area[Node[i].ca].unbuf_capac);
      printf("(n%3d)",n);
      printf(" R_b%d ",Buffered[i]);
      printf(" Stub%.0f(%.0f) ",StubLength[i], EdgeLength[i]);
      t = calc_node_delay(i, pt       .max, 1);
      printf(" t%.0f", t);
      printf("\n");
      max_t =  tMAX(max_t, t);
      min_t = min (min_t, t);
    }
  }
  printf("total capacitance of  level[%2d] = %.3f Frad\n",L, total_cap);
  if (L>0) {
    printf("****************\n");
    printf("skew = %f\n", max_t - min_t);
    printf("****************\n");
    printf("wirelength of level[%2d] = %+9.1f (total_fanout = %3d)\n",
           L, total_wl, total_fanout);
    print_buffer_delay_info(L);
  }
}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
double get_area_max_delay(AreaType *area) {
double t= 0;
int i;

  for (i=0;i<area->n_mr;++i) {
    t =  tMAX(t, area->mr[i].max);
  }
  return(t);
}

/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void print_cluster_cost(double cost) {
double wl[MAX_N_SINKS], cap[MAX_N_SINKS], total_wl=0;
int i, j, k, n;

  // for (i=0;i<Total_CL;++i) { wl[i]=0; cap[i] = 0;}

  // nate 
  assert ( Total_CL +1 < MAX_N_SINKS ) ; 
  for (i=0;i<=Total_CL;++i) { wl[i]=0; cap[i] = 0;}

  
  int npoints = (int) gBoundedSkewTree->Npoints() ;
  for (i=0;i<npoints;++i) {
    j= Cluster_id[i];
    
    if ( j==NIL) continue ; 
    assert(j <= Total_CL); /* XXXnate */

    if (StubLength[i]>0) {
      assert(Buffered[i]>0);
      k = Cluster_id[Node[i].L];
      assert(k!=j);
      wl[k] += StubLength[i];
      cap[k] += StubLength[i]*PUCAP[H_];
      wl[j] += EdgeLength[i] - StubLength[i];
      cap[j] += (EdgeLength[i] - StubLength[i])*PUCAP[H_];
    } else {
      wl[j] += EdgeLength[i];
      cap[j] += EdgeLength[i]*PUCAP[H_];
    }
    int nterms = (int) gBoundedSkewTree->Nterms() ;
    if ( Node[i].parent >=0 && (Buffered[i]>0 || i < nterms) ) {
      cap[j] += Node[i].area[Node[i].ca].capac;
    }
    total_wl += EdgeLength[i];
  }

  n = calc_n_buffer_level();
  printf("\n");
  for (i=n-1;i>=0;--i) {
    printf("\n");
    printf("================== level %d ===============================\n",i);
    print_cluster_cost_at_level(i, wl, cap);
  }
  printf("===========================================================\n");
  printf("\n");

  printf("total wirelength = %.0f \n", total_wl);
  if ( !equal(cost, total_wl) ) {
    printf("warning: cost = %.0f != wirelength \n", cost);
   //  assert(equal(cost, total_wl));
  }
}

/****************************************************************************/
/*  mark off descendants of node v.                                         */
/****************************************************************************/
void init_marked(int v) {

  if (v<0) return;
  Marked[v]      = YES;
  init_marked(Node[v].L);
  init_marked(Node[v].R);
}

/****************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao     */
/****************************************************************************/
static 
void print_BST_delay_error(int v, int L, int R, double tL, double tR) {
int ca;
double t;

  PointType &pt = Node[v].m_stnPt ;
  t = pt_skew(pt) ;
  ca = Node[v].ca;
  if (Node[v].area[ca].n_mr==1) {
     assert(same_Point( pt,       Node[v].area[ca].mr[0]));
     assert(equal(pt_skew(pt       ), pt_skew(Node[v].area[ca].mr[0])));
  }
  assert(t >=0);
  if ( (Gamma==0 || Gamma==1) && t> tMAX(Skew_B*1.01, FUZZ*10) ) {
    printf("\n---------------------------\n");
    printf("error 6060: skew[%d] = %.9f \n",v,t);
    printf("Buffered[%d] = %d \n",v,Buffered[v]);
    print_Point(stdout,pt       );
    PointType &ptL = Node[L].m_stnPt ;
    PointType &ptR = Node[R].m_stnPt ;
    print_Point(stdout, ptL     );
    print_Point(stdout, ptR     );
    printf("tL,tR = (%f, %f) \n", tL,tR);
    printf("EdgeLength[%d] = %f, EdgeLength[%d] = %f\n",
        L, EdgeLength[L], R, EdgeLength[R]);
    printf("StubLength[%d] = %f, StubLength[%d] = %f\n",
        L, StubLength[L], R, StubLength[R]);
    printf("Buffered[%d] = %d , Buffered[%d] = %d \n",
        L, Buffered[L], R, Buffered[R]);
    printf("wiredist = (%f, %f) \n", Point_dist(pt, ptL ),            
             Point_dist( pt, ptR ) ) ;           
    printf("---------------------------\n");
  
    print_node_info(&(Node[v]));
    assert(0);
  }
}
/****************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao     */
/****************************************************************************/
double calc_buffered_edge_delay(int v, int par) {
double t0, t1, t2, cap, ucap, dv, dp;
PointType buffer_pt;
int ca;

  ca = Node[v].ca;

  dv = StubLength[v];
  dp = EdgeLength[v] - StubLength[v];
  PointType &pt = Node[ par ].m_stnPt ;
  PointType &qt = Node[  v  ].m_stnPt ;
  if (equal(dv,0)) {
    buffer_pt = Node[v].m_stnPt ;
  } else if (equal(dp,0)) {
    buffer_pt = Node[ par ].m_stnPt ;
  } else {
    buffer_pt.x = (qt       .x * dp + pt         .x *dv) /EdgeLength[v];
    buffer_pt.y = (qt       .y * dp + pt         .y *dv) /EdgeLength[v];
  }

  ucap = Node[v].area[ca].unbuf_capac;
  cap = Node[v].area[ca].capac;
  t0 =  _pt_delay_increase((double) Node[v].pattern, dv,ucap, 
                         &(qt       ), &buffer_pt);

  t1 =  _pt_delay_increase((double) Node[v].pattern, dp,cap, 
                         &buffer_pt, &(pt         ));

  t2 = Delay_buffer + Buffered[v]*(ucap + dv*PUCAP[H_]);

  return(t0+t1+t2);
}

/****************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao     */
/****************************************************************************/
double calc_edge_delay(int v, int par, int mode) {
double t, cap;

  if (Buffered[v] > 0) {
    t = calc_buffered_edge_delay(v, par);
  } else {
    PointType &pt = Node[ par ].m_stnPt ;
    PointType &qt = Node[ v   ].m_stnPt ;
    cap = Node[v].area[Node[v].ca].capac;
    t =  _pt_delay_increase((double) Node[v].pattern, EdgeLength[v],cap, 
                         &(qt       ), &(pt         ));
  }
  return(t);
}
/****************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao     */
/****************************************************************************/
double _calc_edge_delay_(int v, int par) {
double t, cap;

  cap = Node[v].area[Node[v].ca].capac;
    PointType &pt = Node[ par ].m_stnPt ;
    PointType &qt = Node[ v   ].m_stnPt ;
  t =  _pt_delay_increase((double) Node[v].pattern, EdgeLength[v],cap,
                         &(qt       ), &(pt         ));
  return(t);
}

/****************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao     */
/****************************************************************************/
void calc_BST_delay_sub(int v) {
 
  int L=Node[v].L;
  int R=Node[v].R;
 
    PointType &ptL = Node[ L   ].m_stnPt ;
    PointType &ptR = Node[ R   ].m_stnPt ;
  if (ptL.max == NIL) {
    calc_BST_delay_sub(L);
  }
 
  if (ptR.max == NIL) {
    calc_BST_delay_sub(R);
  }
 
  double tL =  calc_edge_delay(L,v, 1);
  double tR =  calc_edge_delay(R,v, 1);

  PointType &pt = Node[ v   ].m_stnPt ;
  pt.max =  tMAX( ptL     .max+tL, ptR      .max+tR);
  pt       .min =  tMIN( ptL     .min+tL, ptR      .min+tR);

  print_BST_delay_error(v, L, R, tL, tR);
   
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void calc_BST_delay(int v) {
int i; 

  int npoints = (int) gBoundedSkewTree->Npoints() ;
  int nterms = (int) gBoundedSkewTree->Nterms() ;
  for (i=nterms; i<npoints;++i) {
    PointType &pt = Node[ i ].m_stnPt ;
    pt       .max = pt       .min = NIL;
  }
  calc_BST_delay_sub(v);
}


/******************************************************************/
/* the relative location of JS(L) and JS(R) */
/******************************************************************/
int calc_side_loc(int side) {
PointType p0,p1,q0,q1;
double t1,t2;
int lineType;

  p0 = JS[0][0];
  p1 = JS[0][1];
  q0 = JS[1][0];
  q1 = JS[1][1];
  lineType = calc_line_type(p0,p1);

  if (lineType==VERTICAL || lineType==TILT) {
   /* parallel vertical/titled lines */
   t1 =  tMAX(p0.x,p1.x);
   t2 =  tMAX(q0.x,q1.x);
   if ( (t1 <= t2 && side==0) || (t1>=t2 && side==1) ) { return(LEFT); }
   else { return(RIGHT); }
  } else if (lineType==HORIZONTAL || lineType==FLAT) {
    /* parallel horizontal/flat lines */
   t1 =  tMAX(p0.y,p1.y);
   t2 =  tMAX(q0.y,q1.y);
   if ( (t1 <= t2 && side==0) || (t1>=t2 && side==1) ) { return(BOTTOM); }
   else { return(TOP); }
  } else if (lineType==MANHATTAN_ARC) {
    return(PARA_MANHATTAN_ARC);
  } else {
    assert(0);             
  }
}

/****************************************************************************/
/*  Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao     */
/****************************************************************************/
double calc_slope(PointType p1, PointType p2, int *type) {
double m;

  *type = calc_line_type(p1,p2);
  if (same_Point(p1,p2) || *type == VERTICAL) {
    m = 0;
  } else {
    m = (p1.y-p2.y)/(p1.x-p2.x);
  };
  return(m);
}


/****************************************************************************/
/* calculate Boundary Segments (BS) (p1,p2) where point *pt is located.     */
/* return error if point pt is not at any boundary segment of the node      */
/****************************************************************************/
void calc_BS_located(PointType *pt,AreaType *area, PointType *p1,
                        PointType *p2) {
int i,n, found;


  n = area->n_mr;
  for (i=0;i<n;++i) {
    *p1 = area->mr[i];
    *p2 = area->mr[(i+1)%n];
    found = PT_on_line_segment(pt,*p1, *p2);
    if (found) return;
  }
  print_Point(stdout, *pt);
  print_area(area);
  assert(0);
}

/****************************************************************************/
/* calculate the merging boundary for node's children                    */
/****************************************************************************/
void calc_JS_delay(AreaType *area, AreaType *area_L,AreaType *area_R) {
int i;
PointType p1,p2;
/*
  print_node(node_L);
  print_node(node_R);
    print_Point(stdout, JS[0][0]);
    print_Point(stdout, JS[0][1]);
    print_Point(stdout, JS[1][0]);
    print_Point(stdout, JS[1][1]);
*/

  for (i=0;i<2;++i) {

    calc_BS_located(&(JS[0][i]),area_L, &p1, &p2);
    calc_pt_delays(area_L, &(JS[0][i]),p1,p2);

    calc_BS_located(&(JS[1][i]),area_R, &p1, &p2);
    calc_pt_delays(area_R, &(JS[1][i]),p1,p2);
  }
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void itoa(int n, char s[], int *i) {

  if (n/10) {
    itoa(n/10,s, i);
  } else {
    *i = 0;
    if (n<0) s[(*i)++] = '-';
  }
  s[(*i)++] = abs(n)%10 + '0';
  s[(*i)] = '\0';
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void dtoa_sub(double x, char s[], int *i) {
int n;
double f;

  x *= 10.0;

  n = (int) x;
  f = x-n;
  
  s[(*i)++] = n + '0';
  if ( equal(f,0) || *i > 7 ) {
    s[(*i)] = '\0';
  } else {
    dtoa_sub(f,s,i);
  }
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void dtoa(double x, char s[], int *i) {

  if (x==DBL_MAX) x = -1;

  int n = (int) x;
  itoa(n, s, i);

  double f = x-n;
  if ( !equal(f,0)) {
    s[(*i)++] = '.';
    dtoa_sub(f, s, i);
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void new_dtoa(double x, char s[]) {
int i=0;

  dtoa(x,s,&i);
}

/***********************************************************************/
/*  Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao     */
/***********************************************************************/
void print_max_npts() {

  unsigned max_npts=0;
  unsigned npoints = gBoundedSkewTree->Npoints() ;
  unsigned max_i = 0 ;
  for (unsigned i=0;i<npoints;++i) {
    unsigned j = Node[i].ca; 
    unsigned npts= Node[i].area[j].npts;
    if (npts>max_npts) {
      max_npts=npts;
      max_i = i;
    }
  }
  printf("Node[%d].npts = %d\n",max_i,max_npts);
  assert( !(BST_Mode==BME_MODE && Skew_B  == DBL_MAX && max_npts >4));
  assert( !(BST_Mode==IME_MODE && max_npts >6));
}


/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_Fms_Pt_sub(NodeType *node, int i) {
int j,n;

  printf("\"Fms_Pt[%d]\n", i);
  n = n_Fms_Pt[i];
  for (j=0;j<n;++j) {
    print_Point(stdout, Fms_Pt[i][j]);
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_Fms_Pt(NodeType *node) {
  print_Fms_Pt_sub(node,0);
  print_Fms_Pt_sub(node,1);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_Bal_Pt_sub(NodeType *node, int i) {
int j,n;

  printf("\"Bal_Pt[%d]\n", i);
  n = N_Bal_Pt[i];
  for (j=0;j<n;++j) {
    print_Point(stdout, Bal_Pt[i][j]);
  }
}
/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void print_Bal_Pt(NodeType *node) {
  print_Bal_Pt_sub(node,0);
  print_Bal_Pt_sub(node,1);
}

/***********************************************************************/
/* print the pointset */
/***********************************************************************/
static 
void print_sinks (FILE *f, const char fn[]) {
int i;

  fprintf(f, "\n");
  
  fprintf(f, "\"sinks\n" );
  int nterms = (int) gBoundedSkewTree->Nterms() ;
  for (i=0; i<nterms; i++) {
    const PointType &pt = Node[ i ].m_stnPt ;
    print_Point(f, pt );
  }
  print_obstacles_sub(f);
}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
static 
void fprint_Point_array(FILE *f, int n, PointType pt[]) {
int i;

  fprintf(f, "move");
  for (i=0;i<n;++i) {
    fprintf(f, "     %6.1f %6.1f skew:%f, (%f, %f)\n",
                pt[i].x,pt[i].y,pt_skew(pt[i]),pt[i].max,pt[i].min);
  }
  fflush(f);

}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
static 
void print_JR(FILE *f, NodeType *node) {

  int id = node->id;
  int nterms = (int) gBoundedSkewTree->Nterms() ;
  if ( id< nterms && id >=0) return;

  fprintf(f, "\n\"JR\n");
  fprintf(f, "mr[%5d] child:%d,%d \n", id,node->L,node->R);

  fprintf(f, "\n\"JR\n");
  fprint_Point_array(f, n_JR[0], JR[0]);
  fprint_Point_array(f, n_JR[1], JR[1]);

  fflush(f);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void print_a_JS_sub(FILE *f, NodeType *node, int i) {
PointType p0,p1;

  p0 = node->area[node->ca].line[i][0];
  p1 = node->area[node->ca].line[i][1];
    fprintf(f, "move %6.1f  %6.1f \n", p0.x,p0.y);
    fprintf(f, "     %6.1f  %6.1f \n", p1.x,p1.y);
  if (!same_Point(p0,p1) )  {
  }
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void print_a_JS(FILE *f, NodeType *node) {

  int nterms = (int) gBoundedSkewTree->Nterms() ;
  if (node->L >nterms) {
    print_a_JS_sub(f, node, 0);
  }
  if (node->R >nterms) {
    print_a_JS_sub(f, node, 1);
  }
/*
  if (!Manhattan_Arc_Node(node->L)) {
    print_a_JS_sub(f, node, 0);
  }
  if (!Manhattan_Arc_Node(node->R)) {
    print_a_JS_sub(f, node, 1);
  }
*/
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void print_MS(FILE *f) {
PointType p0,p1;

  ms2line(&L_MS, &p0,&p1);
  fprintf(f, "move %6.1f  %6.1f \n", p0.x,p0.y);
  fprintf(f, "     %6.1f  %6.1f \n", p1.x,p1.y);

  ms2line(&R_MS, &p0,&p1);
  fprintf(f, "move %6.1f  %6.1f \n", p0.x,p0.y);
  fprintf(f, "     %6.1f  %6.1f \n", p1.x,p1.y);
}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void print_area_line_sub(FILE *f, AreaType *area) {
PointType p0,p1;
int i;

  printf("\ndist=%.0f \n",area->dist);

  for (i=0;i<2;++i) {  /* for each side */
    p0 = area->line[i][0];
    p1 = area->line[i][1];
    fprintf(f, "move %6.1f %6.1f skew:%6.1f, max=%6.1f, min=%6.1f\n",
                p0.x,p0.y, p0.max - p0.min, p0.max, p0.min);

    fprintf(f, "     %6.1f %6.1f skew:%6.1f, max=%6.1f, min=%6.1f\n",
                   p1.x,p1.y, p1.max -p1.min, p1.max, p1.min);
  }
  fflush(f);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_area_line(AreaType *area) {
  print_area_line_sub(stdout, area);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_Js(FILE *f, NodeType *node) {

  fprintf(f, "\n\"JS\n");
  for (int i=0;i<2;++i) {  /* for each side */
    int j = calc_side_loc(i);
    fprintf(f, "\"type=%d\n",j);
    int n = n_JS[i];
    for (j=0;j<n;++j) {
      PointType p0 = JS[i][j];
      fprintf(f, "move %6.1f  %6.1f    JS[%d][%d] \n", p0.x,p0.y, i, j);
      fprintf(f, "     %6.1f %6.1f skew:%f, max=%f, min=%f\n",
                  p0.x,p0.y, p0.max -p0.min, p0.max, p0.min);
    }
  }
  fprintf(f, "\n\"line\n");

  print_area_line_sub(f, &(node->area[node->ca]));
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_vertices_sub(FILE *f, AreaType *area) {
int i, j, n;
PointType  pt;

  pt = area->mr[0];
  n = area->npts;
  fprintf(f, "move %6.1f %6.1f  %d vertex ",   pt.x, pt.y, n);
  fprintf(f, "merge_cost=%6.1f, dist=%6.1f, tree_cost=%6.1f\n",
              area_merge_cost(area), area->dist, area->subtree_cost);
  for (i=1; i<=n; i++) {
    j = area->vertex[i%n];
    pt = area->mr[j];
    fprintf(f, "     %6.1f %6.1f skew:%f, max:%f, min:%f\n",
          pt.x, pt.y, pt.max-pt.min,pt.max,pt.min);
  }
  fflush(f);
}
/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void print_vertices(NodeType *node) {
  print_vertices_sub(stdout, &(node->area[node->ca]));
}

/******************************************************************/
/*                                                                */
/******************************************************************/
void print_child_region(NodeType *node) {
int L,R;

  L= node->L;
  R= node->R;
  printf("\n\"L%d(Area:%d) \n",L, node->area[node->ca].L_area);
  // unsigned npoints = gBoundedSkewTree->Npoints() ;
  int npoints = (int) gBoundedSkewTree->Npoints() ;
  if (L>=0 && L < npoints) {
    print_area_line(node->area[node->ca].area_L);
    print_area(node->area[node->ca].area_R);
  }
  printf("\"R%d(Area:%d) \n",R, node->area[node->ca].R_area);
  if (R>=0 && R < npoints) {
    print_area_line(node->area[node->ca].area_R);
    print_area(node->area[node->ca].area_R);
  }
}


/***********************************************************************/
/*                                                                     */
/***********************************************************************/
static
void print_merging_tree_sub(FILE *f, int i) {
 
  int root = gBoundedSkewTree->RootNodeIndex () ;

  int nterms = (int) gBoundedSkewTree->Nterms() ;
  if (i < nterms) return;
  print_node_sub(f, &(Node[i]));
  if (i==root || Buffered[i]==0 ) {
    print_merging_tree_sub(f, Node[i].L);
    print_merging_tree_sub(f, Node[i].R);
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_all_JS_sub(FILE *f, int v) {
  int nterms = (int) gBoundedSkewTree->Nterms() ;
  if (v<nterms) return;

  int root = gBoundedSkewTree->RootNodeIndex () ;

  if (v==root || Buffered[v]==0) {
    print_a_JS(f, &(Node[v]));
    print_all_JS_sub(f, Node[v].L);
    print_all_JS_sub(f, Node[v].R);
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_all_JS(FILE *f, int v) {
  fprintf(f, "\n\n\"JS\n");
  print_all_JS_sub(f,v);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_all_merging_regions(FILE *f, int v) {

  fprintf(f, "\n\n\"MS\n");
  unsigned npoints = gBoundedSkewTree->Npoints() ;
  unsigned nterms = gBoundedSkewTree->Nterms() ;
  for (unsigned i=nterms+1;i<npoints;++i) {
    print_area_sub(f, &(Node[i].area[Node[i].ca]));
  }
}

/***********************************************************************/
/*  print merging regions for all internal nodes                       */
/***********************************************************************/
void print_merging_tree (const char* fn, int v) {

  string msg = string ( fn) + "_area.xg_B" + U_Printf ("%.0f",Skew_B ) ;
  cout << "open file (for write) " << msg << endl << endl;

  FILE* f = fopen( msg.c_str() ,"w");
  assert(f != NULL);
  fprintf(f, "\n\n");
  fprintf(f, "\"mr\n");
  print_merging_tree_sub(f,v);

  

  print_all_JS(f,v);

  fclose(f);
}


/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void print_overlapped_regions() {

  unsigned npoints = gBoundedSkewTree->Npoints() ;
  unsigned nterms = gBoundedSkewTree->Nterms() ;
  unsigned j = 0 ;
  for (unsigned i=nterms+1 ;i<npoints;++i) {
    if (equal(Node[i].area[Node[i].ca].dist,0)) {
      j++;
    }
  }
  double t = j*100.0/(nterms-1);
  printf("%d pairs of overlapped regions (%.0f%%)\n",j,t);
}

/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void print_max_n_mr() {
int i, max_n_mr, n_mr, max_i;

  max_n_mr=0;
  // unsigned npoints = gBoundedSkewTree->Npoints() ;
  int npoints = (int) gBoundedSkewTree->Npoints() ;
  for (i=0;i<npoints;++i) {
    n_mr= Node[i].area[Node[i].ca].n_mr;
    if (n_mr>max_n_mr) {
      max_n_mr=n_mr;
      max_i = i;
    }
  }
  printf("Node[%d].n_mr = %d\n",max_i,max_n_mr);
}
/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void print_n_region_type() {
unsigned n_type[5];

  for (unsigned i=0;i<5;++i) n_type[i]=0;
  unsigned npoints = gBoundedSkewTree->Npoints() ;
  for (unsigned i=0;i<npoints;++i) {
    unsigned k = JS_line_type(&(Node[i]));
    n_type[k]++;
    assert(k<5);
  }
  printf("n_region_type:(V=%d,H=%d,M=%d,F=%d,T=%d)\n",
          n_type[0],n_type[1],n_type[2],n_type[3],n_type[4]);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
static 
void draw_a_buffer(FILE *out,int v) {
double x, y ;

  int par = Node[v].parent;
  const PointType& pt = Node[ par ].m_stnPt ;
  const PointType& qt = Node[ v   ].m_stnPt ;
  double x0 = qt.x;
  double y0 = qt.y;
  double x1 = pt.x;
  double y1 = pt.y;
  if (equal(EdgeLength[v],0) || equal(StubLength[v],0) ) {
    x = x0;
    y = y0;
  } else if ( StubLength[v] >= 0.99*EdgeLength[v]) {
    x = (x0*0.01 + x1*0.99);
    y = (y0*0.01 + y1*0.99);
  } else {
    double d0 = StubLength[v]/EdgeLength[v];
    double d1 = 1 - d0;                           
    x = (x0*d1 + x1*d0);
    y = (y0*d1 + y1*d0);
  }
  int cid = Cluster_id[Node[v].L];
  int n=calc_buffer_fanout(v);

  fprintf(out,"move %f %f v:%d cid:%d n = %d, leng=%f\n", 
     x,y,v, cid, n, EdgeLength[v]);   
  fprintf(out,"     %f %f StubLength = %f\n", x, y, StubLength[v]);   
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
static
void draw_buffers_sub(FILE *out, int v) {
  
  
  int nterms = (int) gBoundedSkewTree->Nterms() ;
  if (v<nterms) return;
  if (Buffered[v] > 0  ) {
    draw_a_buffer(out, v);
  }
  draw_buffers_sub(out, Node[v].L);
  draw_buffers_sub(out, Node[v].R);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
int calc_n_buffer() {

  unsigned j=0;
  unsigned npoints = gBoundedSkewTree->Npoints() ;
  for (unsigned i=0;i<npoints;i++) {
    if (Buffered[i] > 0 ) j++;
  }
  return(j-1);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
static
void draw_buffers (FILE *out) {
  int root = gBoundedSkewTree->RootNodeIndex () ;

  fprintf(out,"\"buffer %d\n", calc_n_buffer());
  draw_buffers_sub(out, root );
  fprintf(out,"\n" ) ;
  fprintf(out,"\n" ) ;
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_bst_sub3(FILE *f, int v) {
int p, n, sib;
double x,y,len,  dist, px, py, detour, off, xc,yc;
char pat;
PointType path[100];

  p = Node[v].parent;
  len  = EdgeLength[v];
  pat = Node[v].pattern;

    PointType &pt = Node[ p   ].m_stnPt ;
    PointType &qt = Node[ v   ].m_stnPt ;
  dist = path_finder(qt, pt, path, &n);
  assert(n<100);
  x = path[n-2].x;
  y = path[n-2].y;
  px = path[n-1].x;
  py = path[n-1].y;
  assert(equal(px, pt       .x));
  assert(equal(py, pt       .y));
  detour =  len-dist;
  if (detour < -FUZZ) {
    printf("p=%d, v=%d \n", p, v);
    printf("\"error \n");
    print_path(stdout,path,n);
    if (Node[p].L==v) {
      sib = Node[p].R;
    } else {
      sib = Node[p].L;
    }
    PointType &ptSib = Node[ sib ].m_stnPt ;
    printf("move %f %f len = %f \n", ptSib      .x,ptSib      .y,  len);
    printf("     %f %f dist = %f \n", pt       .x,pt       .y, dist);
  }
  if (n>2) {
    print_path(f,path,n-1);
  } else {
    assert(equal(x, qt       .x));
    assert(equal(y, qt       .y));
    assert( pat == 0 || pat == 1);
    fprintf(f,"move %.2f %.2f id:%-3d child:%d %d par:%d len:%.1f pat:%d\n",
	      x, y,v,Node[v].L,Node[v].R,p, len, pat);
  }
  if (detour > 10000*FUZZ) {  /* detour */
    xc = (x+px)/2; 
    yc = (y+py)/2;
    off = detour/8.0;
    fprintf(f, "     %.2f %.2f detour: (dist=%f)\n", xc, yc, dist);
    fprintf(f, "     %.2f %.2f detour: (dist=%f)\n", xc+off, yc + off, dist);
    fprintf(f, "     %.2f %.2f detour: (dist=%f)\n", xc-off, yc - off, dist);
    fprintf(f, "     %.2f %.2f detour: (dist=%f)\n", xc, yc, dist);
    fprintf(f, "     %.2f %.2f detour: (dist=%f)\n", xc-off, yc + off, dist);
    fprintf(f, "     %.2f %.2f detour: (dist=%f)\n", xc+off, yc - off, dist);
    fprintf(f, "     %.2f %.2f detour: (dist=%f)\n", xc, yc, dist);
  }
  fprintf(f, "     %.2f %.2f delay:(%f,%f) skew:%f\n",
          px, py,qt       .max,qt       .min,pt_skew(qt       ));
  if (Buffered[v] > 0 ) draw_a_buffer(f,v);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
static 
void _print_bst_sub2(FILE *f, int v) {

  if (v>=0) {
    if (Node[v].parent != NIL) {
      print_bst_sub3(f,v);
    }
    if (Buffered[v]==NO) {
      _print_bst_sub2(f,Node[v].L);
      _print_bst_sub2(f,Node[v].R);
    }
  }
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
static 
void print_bst_sub2(FILE *f, int v) {
  if (v>=0) {
    _print_bst_sub2(f,Node[v].L);
    _print_bst_sub2(f,Node[v].R);
  }
}
/******************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/******************************************************************/
void set_SuperRoot() {

  int superRoot = gBoundedSkewTree->SuperRootNodeIndex () ;
  int root = gBoundedSkewTree->RootNodeIndex () ;
  
  Node[superRoot]= Node[root];
  Node[superRoot].parent = Node[superRoot].R = NIL;
  Node[superRoot].L = root;
  Node[root].parent = superRoot;
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
static 
void print_bst_at_level(FILE *out, int L) {

  unsigned npoints = gBoundedSkewTree->Npoints() ;
  int superRoot = gBoundedSkewTree->SuperRootNodeIndex () ;

  unsigned j=0;
  for (unsigned i=0 ;i<npoints;++i) {
    if (Buffered[i] > 0  && i != (unsigned) superRoot && level(i)==L ) {
      // unsigned n =  calc_buffer_fanout(i);
      // fprintf(out, "\"%d:%d:%d\n",L,j, n);
      j++ ;
      fprintf(out, "\"tree%d\n", L);
      print_bst_sub2(out, i);
    }
  }

}
/***********************************************************************/
/*          print the b-skew routing tree in XGRAPH format             */
/***********************************************************************/
static
void print_bst_sub1 (FILE *out, int v, const char* fn) {

  int n = calc_n_buffer_level();
  for (int i=n-1;i>=0;--i) {
    print_bst_at_level(out, i);
    fprintf(out, "\n\n");
  }
  print_sinks (out, fn);

  fprintf(out, "\n" );
  draw_buffers (out);

}

/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void print_Skew_B(FILE *f, double skew) {
  if (skew == DBL_MAX) { fprintf(f, "B=infty ");
  } else {
    fprintf(f, "B%.0f",skew);
  }
}
/***********************************************************************/
/*          print the b-skew routing tree in XGRAPH format             */
/***********************************************************************/
void print_bst(const char* fn, int v, double Tcost, double Tdist) {
double t;

  string msg = string ( fn) + ".xg_B" + U_Printf ("%.0f",Skew_B ) ;
  cout << "open file (for write) " << msg << endl << endl;
  

  FILE* f = fopen( msg.c_str(),"w");
  assert(f != NULL);

  fprintf(f, "TitleText:  ");
  if ( N_Clusters[1]>1 ) { /* exists a buffer tree structure */
    fprintf(f, "C%.0f", Start_Tcost/1000.0);
    
      t = (Tcost- Tdist)*100.0/Tdist;
      fprintf(f, "(%.1f%%) ",t);
    
    print_Skew_B(f, Start_Skew_B);
    fprintf(f, "-");
  }
  if (BST_Mode != BME_MODE) {
    fprintf(f, "N%d", N_Area_Per_Node);
    fprintf(f, "S%d", N_Sampling);
  }
  if ( PURES_V_SCALE != 1) fprintf(f, "R%.1f",   PURES_V_SCALE);
  if ( PUCAP_V_SCALE != 1) fprintf(f, "C%.1f",   PUCAP_V_SCALE);
  if (k_Parameter != -1) fprintf(f, "k%d",k_Parameter);
  if ( equivalent(Cost_Function,1, 0)) {
    fprintf(stdout, "C%d",Cost_Function);
    fprintf(f, "C%d",Cost_Function);
  }
  fprintf(f, "\n\n");

  print_bst_sub1 (f,v, fn);
  fclose(f);
}

/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
static 
void print_top_sub(FILE *f, int v) {

  if (v<0) return;
  int p = Node[v].parent;
  int L = Node[v].L;
  int R = Node[v].R;
  double Lx, Ly, Rx, Ry;
  if ( L >= 0 ) {
	  Lx = Node[L].m_stnPt.x; 
	  Ly = Node[L].m_stnPt.y;
  } else {
	  Lx = 0.0;
	  Ly = 0.0;
  }
  if ( R >= 0 ) {
	  Rx = Node[R].m_stnPt.x; 
	  Ry = Node[R].m_stnPt.y;
  } else {
	  Rx = 0.0;
	  Ry = 0.0;
  }
  //fprintf(f, "id:%-3d child:%d %d par:%d \n", v,L,R,p);
  //hyein: to get coordinate
  fprintf(f, "id:%-3d (%f %f) child1:%d (%f %f) child2:%d (%f %f) par:%d (%f %f) \n", \
		  v, Node[v].m_stnPt.x, Node[v].m_stnPt.y, \
		  L, Lx, Ly, \
		  R, Rx, Ry, \
		  p, Node[p].m_stnPt.x, Node[p].m_stnPt.y \
		  ); 
  print_top_sub(f,L);
  print_top_sub(f,R);
}

/***********************************************************************/
/*          print the b-skew routing tree in XGRAPH format             */
/***********************************************************************/
void print_topology(const char fn[], int v, double Tcost, double Tdist) {
FILE *f;

  string msg = string ( fn) + ".top_B" + U_Printf ("%.0f",Skew_B ) ;
  cout << "open file (for write) " << msg << endl << endl;
  
  f = fopen( msg.c_str(),"w");
  assert( f != NULL);

  fprintf(f, "#BST-DME will only read lines begingging with \"id:\" \n" ) ; 
  fprintf(f, "#id corresponds to the one in sink input file %s\n", fn ) ; 
  fprintf(f, "#each node has two children and one parent\n" ) ;
  fprintf(f, "#Children or the parent can have id=-1 if they don't exist\n" ) ;
  fprintf(f, "#This file is generated by running bst -i %s -B ", fn);
  if (Skew_B == DBL_MAX) { fprintf(f, " -1 ");
  } else {
    fprintf(f, "%.1f",Skew_B);
  }
  fprintf(f, "\n" ) ;

  fprintf(f, "#The resulting total length=%.2f \n\n", Tcost);

  fprintf(f, "#Topology\n");
  print_top_sub(f,v);
  fclose(f);

}

/***********************************************************************/
/*   print another inputfile in different format                       */
/***********************************************************************/
void print_inputfile(char fn[]) {
FILE *f;
char a[20], *b = "_B", c[10];
double t, x,y, cap, delay;
double min_t= DBL_MAX, max_t = 0, skew, total_delay;
int i=0;

  strcpy(a,fn);
  strcat(a,b); 
  dtoa( Skew_B, c, &i);
  strcat(a,c);

  string msg = string ( fn) + "_B" + U_Printf ("%.0f",Skew_B ) ;
  cout << "open file (for write) " << msg << endl << endl;


  f = fopen(a,"w");
  assert( f != NULL);

  int root = gBoundedSkewTree->RootNodeIndex () ;

  cap = PUCAP[H_]/PUCAP_SCALE;
    PointType &pt = Node[  root  ].m_stnPt ;
  skew = pt_skew( pt         )/PUCAP_SCALE;
  int nterms = (int) gBoundedSkewTree->Nterms() ;
  fprintf(f, "%d  %f  %e %e \n", nterms, PURES[H_], cap, skew);

  total_delay =  pt         .max;
  for (i=0;i<nterms;++i) {
    PointType &qt = Node[ i   ].m_stnPt ;
    x = qt       .x;
    y = qt       .y;
    cap = (Node[i].area[0].capac)/PUCAP_SCALE;
    delay = calc_node_delay(i, qt       .max, 1);
    fprintf(f, "%d ",i);
    t = (total_delay - delay) /PUCAP_SCALE;
    fprintf(f, "%.0f  %.0f  %e  %e \n",x, y, cap, t);
    min_t =  tMIN(min_t, delay + qt       .min);
    max_t =  tMAX(max_t, delay + qt       .max);
  }
  assert( max_t - min_t <= Skew_B + FUZZ); 
  fclose(f);
}

/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void print_area_info(AreaType *area) {
   print_area_line(area) ;
   print_area(area) ;

   print_area_line(area->area_L) ;
   print_area(area->area_L);

   print_area_line(area->area_R) ;
   print_area(area->area_R);
}
/***********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void print_node_info(NodeType *node) {

   print_area_info( &(node->area[node->ca]));
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_node_informatio(NodeType *node,NodeType *node_L, NodeType *node_R) {
AreaType *area;

  area = &(node->area[node->ca]);
  print_node(node);
  print_JR(stdout, node);
  print_Js(stdout, node);
  print_Fms_Pt(node);
  print_Bal_Pt(node);
  printf("\n\"L(%d)", node->area[node->ca].L_area);
  print_node(node_L);
  printf("\n\"R(%d)", node->area[node->ca].R_area);
  print_node(node_R);
  iShowTime();
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void check_JS_line(NodeType *node, NodeType *node_L,NodeType *node_R) {
PointType p0, p1, tmp_pt0, tmp_pt1;
AreaType *area, *area_L, *area_R;

  area = &(node->area[node->ca]);
  area_L = &(node_L->area[node_L->ca]);
  area_R = &(node_R->area[node_R->ca]);

  p0 = node->area[node->ca].line[0][0];
  p1 = node->area[node->ca].line[0][1];
  calc_BS_located(&p0, area_L, &tmp_pt0, &tmp_pt1);
  calc_BS_located(&p1, area_L, &tmp_pt0, &tmp_pt1);
  p0 = node->area[node->ca].line[1][0];
  p1 = node->area[node->ca].line[1][1];
  calc_BS_located(&p0, area_R, &tmp_pt0, &tmp_pt1);
  calc_BS_located(&p1, area_R, &tmp_pt0, &tmp_pt1);
}

/***************************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***************************************************************************/
void check_mss(AreaType *area,AreaType *area_L, AreaType *area_R) {
int err=NO;
PointType p0,p1;


  if (area->n_mr>2 || pt_skew(area->mr[0]) <= Skew_B ) {
     err = YES;
  } else if ( area->n_mr == 2) {
    p0 = area->mr[0];
    p1 = area->mr[1];
    if (same_Point(p0,p1) || !equal(pt_skew(p0),pt_skew(p1)) ||
        pt_skew(area->mr[1]) <= Skew_B) {
      err = YES;
    }
  }

  if (err) {
/*
    print_area_informatio(node,node_L,node_R);
*/
    assert(0);             
  }
}
/********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/********************************************************************/
void check_x(AreaType *area, AreaType *area_L, AreaType *area_R, double *x) {
  printf("error: x=%.11f \n", *x);
  if (*x < -FUZZ) {
    print_area(area);
    print_area(area_L);
    print_area(area_R);
    assert(0);             
  } else { *x = 0; }
}

/********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/********************************************************************/
void check_fms_sub(AreaType *area_L, AreaType *area_R, int side) {
int i;
  printf("\"bal\n");
  for (i=0;i<N_Bal_Pt[side];++i) {
    print_Point(stdout, Bal_Pt[side][i]);
  }
  printf("\"fms\n");
  for (i=0;i<n_Fms_Pt[side];++i) {
    print_Point(stdout, Fms_Pt[side][i]);
  }
  printf("\"LR\n");
  fflush(stdout);
  assert(0);             
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void check_fms(AreaType *area_L, AreaType *area_R, int side) {
int i, n;
PointType tmp_pt;

  n =  n_Fms_Pt[side];
  for (i=0;i<n;++i) {
    tmp_pt = Fms_Pt[side][i];
    if ( pt_skew(tmp_pt) > Skew_B + FUZZ ) {
      printf("\"err_fms\n");
      print_Point(stdout, tmp_pt);
      check_fms_sub(area_L, area_R, side);
    }
  }
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
void check_ZST_detour(NodeType *node, NodeType *node_L, NodeType *node_R) {
double d0, d1, t0,t1;
PointType pt0, pt1;


  pt0=node_L->area[node_L->ca].mr[0];
  pt1=node_R->area[node_R->ca].mr[0];

  if (!equal(pt0.max, pt0.min) )
    assert(0);             
  if (!equal(pt1.max, pt1.min) )
    assert(0);             

  d0 = node->area[node->ca].L_EdgeLen;
  d1 = node->area[node->ca].R_EdgeLen;

  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
    t0 = pt0.max + d0;
    t1 = pt1.max + d1;
  } else {
    t0 = pt0.max + K[H_]*d0*d0 + PURES[H_]*d0*(node_L->area[node_L->ca].capac);
    t1 = pt1.max + K[H_]*d1*d1 + PURES[H_]*d1*(node_R->area[node_R->ca].capac);
  }

  if ( !equal(t0,t1) ) {
    printf("t0 = %.9e, t1 = %.9e \n", t0,t1);
    printf("node->area[node->ca].L_EdgeLen = %.9e, node->area[node->ca].R_EdgeLen = %.9e \n", d0,d1);
    assert(0);             
  }
}

/********************************************************************/
      /* check sampling_segment sampling_set[i] */
/********************************************************************/
void check_a_sampling_segment(AreaType *area, TrrType *ms) {
PointType p0, p1, v0, v1;

  ms2line(ms, &p0, &p1);
  calc_BS_located(&p0,area,&v0,&v1);
  calc_BS_located(&p1,area,&v0,&v1);
}
/****************************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void check_tmparea(AreaType *tmparea, int n) {
int i, wrong;
double t0, t1;

  wrong=NO;

  for (i=0;i< min (1,n-1);++i) {
    if (equal(tmparea[i].dist,tmparea[i+1].dist)) {
      t0 = calc_boundary_length(&(tmparea[i]));
      t1 = calc_boundary_length(&(tmparea[i+1]));
      if (t0 < t1 + FUZZ) {
        wrong = YES;
        break;
      }
    } else {
      if (tmparea[i].dist >= tmparea[i+1].dist-FUZZ) {
        wrong = YES;
        break;
      }
      /* assert(tmparea[i].subtree_cost<tmparea[i+1].subtree_cost); */
    }
  }
  if (wrong) {
    print_area(&(tmparea[i]));
    print_area(&(tmparea[i+1]));
    /*
    exit(0);
    */
  }
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
void check_JS_MS( ) {
TrrType ms0,ms1;

  line2ms(&ms0, JS[0][0], JS[0][1]);
  line2ms(&ms1, JS[1][0], JS[1][1]);

  assert(trrContain(&L_MS, &ms0));
  assert(trrContain(&R_MS, &ms1));
}

/****************************************************************************/
/* calculate delays for node->JS                                         */
/****************************************************************************/
void check_const_delays(PointType *p1,PointType *p2) {

  if (equal(p1->max,p2->max) && equal(p1->min,p2->min)) {
    p1->max = p2->max;
    p1->min = p2->min;
  } else {
    printf("\n\"error\n");
    print_Point(stdout, *p1);
    print_Point(stdout, *p2);
    assert(0);
  }
}


/********************************************************************/
/*                                                                  */
/********************************************************************/
void create_one_ms_sub(int i, double *dd1,double *dd2) {
int il, ir;
double cap1,delay1,cap2,delay2,d,d1,d2;

  il = Node[i].L;
  ir = Node[i].R;

  cap1 = Node[il].area[Node[il].ca].capac;
  delay1 = Node[il].area[Node[il].ca].mr[0].max;
  cap2 = Node[ir].area[Node[ir].ca].capac;
  delay2 = Node[ir].area[Node[ir].ca].mr[0].max;

  Node[i].area[Node[i].ca].dist = d = ms_distance(Node[il].ms, Node[ir].ms);

  calc_merge_distance(PURES[H_], PUCAP[H_],cap1,delay1,cap2,delay2,d,&d1,&d2);

  Node[i].area[Node[i].ca].L_EdgeLen = EdgeLength[il]= *dd1 = d1;
  Node[i].area[Node[i].ca].R_EdgeLen = EdgeLength[ir]= *dd2 = d2;
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void create_one_merging_segment(int v) {
int il,ir ;
double d1, d2, t;
TrrType trra, trrb;

  int nterms = (int) gBoundedSkewTree->Nterms() ;
  if (v < nterms) {
    PointType &qt = Node[ v   ].m_stnPt ;
    Node[v].ms->MakeDiamond (Node[v].area[Node[v].ca].mr[0], 0  );
    Node[v].area[Node[v].ca].npts=1;
    Node[v].area[Node[v].ca].mr[0].max = Node[v].area[Node[v].ca].mr[0].min =
    qt.max = qt.min = 0;
  } else {
    il = Node[v].L;
    ir = Node[v].R;

    create_one_ms_sub(v,&d1,&d2);

    build_trr(Node[il].ms,d1,&trra);
    build_trr(Node[ir].ms,d2,&trrb);
    make_intersect(&trra, &trrb,Node[v].ms);
    ms2line(Node[v].ms,&(Node[v].area[Node[v].ca].mr[0] ),&(Node[v].area[Node[v].ca].mr[1]));
    Node[v].area[Node[v].ca].n_mr = Node[v].area[Node[v].ca].npts = 2;

    bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
    PointType &pt = Node[ il  ].m_stnPt ;
    if ( linear ) {
      t = pt        .max + d1;
    } else {
      t = K[H_]*d1*d1 + PURES[H_]*d1*Node[il].area[Node[il].ca].capac + pt        .max;
    }
    Node[v].area[Node[v].ca].capac = PUCAP[H_]*(d1 + d2) + Node[ir].area[Node[ir].ca].capac + Node[il].area[Node[il].ca].capac;
    PointType &ptV = Node[ v   ].m_stnPt ;
    Node[v].area[Node[v].ca].mr[0].max = Node[v].area[Node[v].ca].mr[0].min =
    Node[v].area[Node[v].ca].mr[1].max = Node[v].area[Node[v].ca].mr[1].min =
    ptV.max = ptV.min = t;

    Node[v].area[Node[v].ca].subtree_cost =  merge_cost (&(Node[v])) + Node[il].area[Node[il].ca].subtree_cost +
                            Node[ir].area[Node[ir].ca].subtree_cost;
  }
}

/*******************************************************************/
/*  check if merging region = merging segment when Skew_B==0        */
/*******************************************************************/
void check_ZST(int v) {
NodeType tmpnode;
double t1, t2;

  alloca_NodeType(&tmpnode);
  if (Node[v].area[Node[v].ca].npts > 2 )  {
    print_node_sub(stdout,&(Node[v]));
    assert(0);             
  }
  assign_NodeType(&tmpnode, &(Node[v]));
  *(tmpnode.ms) = *(Node[v].ms);
  create_one_merging_segment(v);

  if (!trrContain(Node[v].ms, tmpnode.ms)) {
    printf("old_ms xl:%f,xh:%f,yl:%f,yh:%f\n", tmpnode.ms->xlow,
        tmpnode.ms->xhi,tmpnode.ms->ylow,tmpnode.ms->yhi);
    printf("new_ms xl:%f,xh:%f,yl:%f,yh:%f\n",Node[v].ms->xlow,
            Node[v].ms->xhi,Node[v].ms->ylow,Node[v].ms->yhi);
    printf("\n\"child\n");
    print_node_sub(stdout, &(Node[Node[v].L]));
    print_node_sub(stdout, &(Node[Node[v].R]));
    printf("\n\"old\n");
    print_node_sub(stdout, &tmpnode);
    print_Js(stdout, &tmpnode);
    printf("\n\"new\n");
    print_node_sub(stdout, &(Node[v]));
    print_Js(stdout, &(Node[v]));
    assert(0);             
  }

  t1 = merge_cost(&tmpnode);
  t2 = merge_cost(&(Node[v]));
  if (!equal(t1,t2) ) {
    print_node_sub(stdout, &tmpnode);
    print_node_sub(stdout, &(Node[v]));

    print_node_sub(stdout, &(Node[Node[v].L]));
    print_node_sub(stdout, &(Node[Node[v].R]));

    printf("tmpnode.merge_cost:%f != Node[%d].merge_cost:%f\n",t1,v,t2);

    check_ZST_detour(&tmpnode, &(Node[Node[v].L]), &(Node[Node[v].R]));
    check_ZST_detour(&(Node[v]), &(Node[Node[v].L]), &(Node[Node[v].R]));
    assert(0);             
  }
  assign_NodeType(&(Node[v]), &tmpnode);
  free_NodeType(&tmpnode);
}


/****************************************************************************/
/*                                                                      */
/****************************************************************************/
void calc_n_JS() {
int i;
  for (i=0;i<2;++i) {
    if (same_Point(JS[i][0],JS[i][1])) {
      n_JR[i] = n_JS[i]=1;
    } else {
      n_JR[i] = n_JS[i]=2;
    }
  }
}

/****************************************************************************/
/*                                                                      */
/****************************************************************************/
void align_JS(int side) {
PointType p;

  if (equal(JS[side][0].y, JS[side][1].y) ) {
    if (JS[side][0].x < JS[side][1].x ) {
      p = JS[side][0];
      JS[side][0] =  JS[side][1];
      JS[side][1] = p;
    }
  } else if ( JS[side][0].y < JS[side][1].y ) {
    p = JS[side][0];
    JS[side][0] =  JS[side][1];
    JS[side][1] = p;
  }
}
/****************************************************************************/
/*                                                                      */
/****************************************************************************/
void JS_processing_sub2(AreaType  *area) {
  assert(Manhattan_arc(JS[1][0], JS[1][1]));
  if (a_segment_TRR(&L_MS)) {
    ms2line(&L_MS,  &(area->line[0][0]), &(area->line[0][1]) );
  }
  if (a_segment_TRR(&R_MS)) {
    ms2line(&R_MS,  &(area->line[1][0]), &(area->line[1][1]) );
  }
}
/****************************************************************************/
/*                                                                      */
/****************************************************************************/
void JS_processing_sub(AreaType  *area) {
int i;

  calc_n_JS();
  if (n_JS[0]==2) align_JS(0);
  if (n_JS[1]==2) align_JS(1);

  area->L_MS = L_MS;
  area->R_MS = R_MS;

  for (i=0;i<2;++i) {
    JR[i][0] = JS[i][0];
    JR[i][1] = JS[i][1];
  }
}
/****************************************************************************/
/*                                                                      */
/****************************************************************************/
void JS_processing(AreaType  *area) {
int i;
  JS_processing_sub(area);
  for (i=0;i<2;++i) {
    area->line[i][0] = JS[i][0];
    area->line[i][1] = JS[i][1];
  }
}
/********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/********************************************************************/
void remove_epsilon_err(PointType *q) {
double t;

  if (equal(q->max,q->min) ) { /* delays ok */
    if (q->min> q->max) {  /* correct epsilon error */
      t = (q->min+q->max)/2;
      q->max = q->min = t;
    }
  } else if (q->min> q->max) { /* delays not ok */
    assert(0);             
  }

  assert(equal(q->max - q->min, Skew_B));
}

/****************************************************************************/
/* construct merging region mr which are actually Merging Segments.         */
/****************************************************************************/
void construct_TRR_mr_sub(AreaType *area,TrrType *ms0,TrrType *ms1,double d0,
                     double d1) {
int i;
TrrType trr0, trr1, trr;

  build_trr(ms0, d0, &trr0);
  build_trr(ms1, d1, &trr1);

  make_intersect(&trr0, &trr1, &trr);
  double origB =  gBoundedSkewTree->Orig_Skew_B()  ;
  if ( origB > 0) make_1D_TRR(&trr, &trr);
/*
*/

  area->n_mr = area->npts = TRR2area(&trr, area->mr);
  for (i=0;i<area->npts;++i) {
    area->vertex[i] = i;
  }
  for (i=1;i<area->n_mr;++i) {
    area->mr[i].max = area->mr[0].max;
    area->mr[i].min = area->mr[0].min;
  }

  if ( origB > 0) assert(area->n_mr <= 2);
}

/****************************************************************************/
/* construct merging region mr which are actually Merging Segments.         */
/****************************************************************************/
void construct_TRR_mr(AreaType *area) {

  construct_TRR_mr_sub(area, &L_MS,&R_MS, area->L_EdgeLen, area->R_EdgeLen);
}

/********************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/********************************************************************/
int any_fms_on_JR() {
int i,j,n;

  if (n_Fms_Pt[0] > 0  || n_Fms_Pt[1] > 0)  return(YES);
  for (i=0;i<2;++i) {
    n = n_JR[i];
    for (j=0;j<n;++j) {
      if (pt_skew(JR[i][j]) <= Skew_B)
        return(YES);
    }
  }
  return(NO);
}


/****************************************************************************/
/*                                                                         */
/****************************************************************************/
static int point_compare_inc(const void  *p1, const void  *q1) {
PointType  *p, *q;

  p = (PointType  *) p1;
  q = (PointType  *) q1;
  return( (p->t > q->t) ? YES: NO);
}

/******************************************************************/
/* sort turn_pts on JR[side]                              */
/******************************************************************/
void sort_pts_on_line(PointType p[], int n) {
int i;
 
  for (i=0;i<n;++i) {
    p[i].t = Point_dist(p[i], p[0]);
  }
 
  qsort(p,n,sizeof(PointType),point_compare_inc);
  /* in increasing order of distance to p[0] */
}
 
/****************************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void Ex_DME_memory_allocation_sub() {

  unsigned npoints = gBoundedSkewTree->Npoints() ;
  NearestCenter = (int *) calloc(npoints, sizeof(int));
  TmpClusterId = (int *) calloc(npoints, sizeof(int));
  ClusterDelay = (double *) calloc(npoints, sizeof(double));

  TmpCluster = (TmpClusterType *) calloc(npoints, sizeof(TmpClusterType));
  for (unsigned i=0; i<npoints; i++) {
    TmpCluster[i].ms = new TrrType ;
    assert(TmpCluster[i].ms != NULL);
  }
  Tmp_x_Cluster.ms = new TrrType ;
  assert(Tmp_x_Cluster.ms != NULL);
  Tmp_y_Cluster.ms = new TrrType ;
  assert(Tmp_y_Cluster.ms != NULL);
}
/****************************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void Ex_DME_memory_allocation() {

  unsigned npoints = gBoundedSkewTree->Npoints() ;
  unsigned nterms = gBoundedSkewTree->Nterms() ;
  for ( unsigned  i=0; i<npoints; i++) { 
    Node.push_back ( NodeType() ) ;
  }

  for ( unsigned  i=0; i<nterms; i++) { 
     alloca_NodeType_with_n_areas(&(Node[i]), 1); 
  }
  for ( unsigned  i=nterms; i<npoints; i++) { 
    alloca_NodeType(&(Node[i])); 
  }
  for ( unsigned  i=0; i<npoints; i++) { 
    gBoundedSkewTree->AddTreeNode  (&(Node[i])); 
    gBoundedSkewTree->AddCandiRoot  (&(Node[i])); 
    gBoundedSkewTree->AddTempNode  (&(Node[i])); 
  }

  
  Ex_DME_memory_allocation_sub();


  for ( unsigned  i=0; i<npoints; i++) { 
    EdgeLength.push_back(0) ;
    StubLength.push_back(0) ;
  }

  N_neighbors = (int *) calloc(npoints, sizeof(int));
  assert(N_neighbors   != NULL);

  Cluster = (ClusterType *) calloc(npoints, sizeof(ClusterType));
  assert(Cluster   != NULL);

  Buffered = (int *) calloc(npoints, sizeof(int));
  assert(Buffered   != NULL);

  Neighbor_Cost = (double **) calloc(npoints, sizeof(double *));
  assert(Neighbor_Cost != NULL);

  for ( unsigned  i=0; i<npoints; i++) {
    Neighbor_Cost[i] = (double *) calloc(N_Neighbor, sizeof(double));
    assert(Neighbor_Cost[i] != NULL);
  }

  The_NEIghors = (int **) calloc(npoints, sizeof(int *));
  assert(The_NEIghors != NULL);

  for ( unsigned  i=0; i<npoints; i++) {
    The_NEIghors[i] = (int *) calloc(N_Neighbor, sizeof(int));
    assert(The_NEIghors[i] != NULL);
  }
  L_sampling = (TrrType *) calloc(N_Sampling, sizeof(TrrType));
  R_sampling = (TrrType *) calloc(N_Sampling, sizeof(TrrType));
}

/*************************************************************************/
/* Copyright (C) 1994/2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/*************************************************************************/
void ExG_DME_memory_allocation() {

  unsigned npoints = gBoundedSkewTree->Npoints() ;
  for ( unsigned  i=0; i<npoints; i++) { 
    CandiRoot.push_back ( NodeType ()  ) ;
    TempNode.push_back ( NodeType ()  ) ;
  }
  for ( unsigned i =0; i<npoints; i++) {
    alloca_NodeType(&(CandiRoot[i]));
    alloca_NodeType(&(TempNode[i]));
  }

  Best_Pair = (PairType *) calloc(npoints*N_Neighbor, sizeof(PairType));
  assert(Best_Pair != NULL);

  Marked = (char *) calloc(npoints, sizeof(char));
  assert(Marked != NULL);

  UnMarkedNodes = (int *) calloc(npoints, sizeof(int));
  assert(UnMarkedNodes != NULL);


  double t = sqrt((double) npoints);
  unsigned  n =  tMAX(1, (int) t);
  Bucket = (BucketType **) calloc(n, sizeof(BucketType *));
  assert(Bucket != NULL);

  for ( unsigned i =0; i<n; i++) {
    Bucket[i] = (BucketType *) calloc(n, sizeof(BucketType));
    assert(Bucket[i] != NULL);
  }
  MAX_N_Index = n;
}

/***********************************************************************/
/***********************************************************************/
NodeType* 
BstTree::AddOneNode ( unsigned i , double x, double y, 
   double cap, double delay  
) {
  NodeType* from = TreeNode(i) ; 
  assert( from->ca==0);
  PointType &pt =    from->area[0].mr[0];
  pt.x = x;
  pt.y = y;
  pt.max = pt.min = delay;
  from->m_stnPt = pt ;
  from->area[0].capac = cap ;
  
  return from ;
}


/***********************************************************************/
/* read in input     files                                             */
/***********************************************************************/
bool tReadInputHeader ( const string &fn ) {
#define MAX_STRING_LENGTH 1024
char commentChar = '#' ;
char line[ MAX_STRING_LENGTH ] ;

  ifstream inFile;
  inFile.open ( fn.c_str() ) ;

  if (!inFile ) return false ;
  unsigned nterms = 0 ;
  while ( inFile.getline (line, MAX_STRING_LENGTH ) ) {
    char buf[ MAX_STRING_LENGTH ] ;
    strcpy ( buf , line ) ;
    vector< char * > tokv ;
    tSplit ( buf, tokv, " " ) ;
    // if ( tokv.empty() ) continue ;
    char token [ MAX_STRING_LENGTH ] ;
    sscanf(line,"%s \n", token ) ;
    // char *token = tokv[0] ;
    string key = token ;
    if (  token[0] == commentChar)  {
    } else if ( key=="NumPins" ) {
      sscanf( line,"%s : %d \n", token , &nterms ) ;
      gBoundedSkewTree->SetNterms ( nterms ) ;
      // printf ( "%s : %d \n", token , nterms ) ; 

    } else if ( key=="PerUnitResistance" ) {
      sscanf( line,"%s : %lf \n", token , &(PURES[H_]) ) ;
      gBoundedSkewTree->SetPerUnitResistance ( PURES ) ;

    } else if ( key=="PerUnitCapacitance" ) {
      sscanf( line,"%s : %lf \n", token , &(PUCAP[H_]) ) ;
      gBoundedSkewTree->SetPerUnitCapacitance ( PUCAP ) ;

    } ;
  }
  set_K() ;

  Ex_DME_memory_allocation();
  ExG_DME_memory_allocation();

  inFile.close() ;
  return true ;
}
/***********************************************************************/
/* read in input     files                                             */
/***********************************************************************/
static bool 
tReadInputNodes ( const string &fn ) {
#define MAX_STRING_LENGTH 1024
char line[ MAX_STRING_LENGTH ] ;

  ifstream inFile;
  inFile.open ( fn.c_str() ) ;

  if (!inFile ) return false ;

  unsigned i = 0 ;
  unsigned id = 0;
  double x,y, cap , delay = 0 ;

  while ( inFile.getline (line, MAX_STRING_LENGTH ) ) {

    if ( strlen( line) ==0 ) continue ;
    char token [ MAX_STRING_LENGTH ] ;
    sscanf(line,"%s", token ) ;
    string key = token ;

    if (  key=="Sink") {
      if ( i ) {
        gBoundedSkewTree->AddOneNode ( id, x,y, cap, delay ) ;
        x = y = cap = delay = 0 ;
      }
      sscanf(line,"%s :  %d \n", token , &id ) ;
      i++ ;
    } else if (  key == "Coordinate" ) {
      sscanf(line,"%s :  %lf %lf \n", token , &x, &y ) ;
    } else if ( key=="Capacitive" ) {
      char token2 [ MAX_STRING_LENGTH ] ;
      sscanf(line,"%s %s : %lf \n", token, token2, &cap ) ;
      cap = cap*PUCAP_SCALE;
    } else if (  key == "Downstream_Delay" ) {
      sscanf(line,"%s :  %lf \n", token , &delay ) ;
      delay = delay*PUCAP_SCALE; // scale up 
    } else if (  key == "#" ) {
    } else if (  key == "NumPins" ) {
    } else if (  key == "PerUnitResistance" ) {
    } else if (  key == "PerUnitCapacitance" ) {
    } else {
      printf ( "Unknown keyword  %s \n", key.c_str() ) ;
    };
  }
  if ( i ) {
    gBoundedSkewTree->AddOneNode ( id, x,y, cap, delay ) ;
  }

  assert ( i <= gBoundedSkewTree->Nterms() ) ;

  if (i!=gBoundedSkewTree->Nterms()) {
    printf(" (i=%d) != (nterms=%d) \n", i, 
          gBoundedSkewTree->Nterms());
    gBoundedSkewTree->SetNterms ( i ) ;
  }
  inFile.close() ;

  return true ;

}
/***********************************************************************/
/* read in input     files                                             */
/***********************************************************************/
double get_input_format(char fn[]) {
FILE *f;
char line[100]; 
int i, input_format;

  f = fopen(fn, "r");
  assert(f != NULL);

  fgets(line, 100, f);
  i = 0; 
  input_format = 0; 
  while (line[i] != '\n') {
    if (line[i] != ' ' && (line[i+1] == ' ' || line[i+1] == '\n') ) {
      input_format++;
    } 
    i++;
  } 
  assert(input_format==3 || input_format==4);
  fclose(f);
  return(input_format);
}

/***********************************************************************/
/* read in input     files                                             */
/***********************************************************************/
void read_input_file(const string &fn ) {

  bool ok = tReadInputHeader (  fn ) && tReadInputNodes ( fn ) ;
  if ( !ok ) {
    cerr << "cannot open file " << fn << endl ;
    exit (0) ;
  }
}
/***********************************************************************/
/* read in input     files                                             */
/***********************************************************************/
void read_clustering_info(char ClusterFile[]) {
FILE *f;
int i, j;
int cluster_size[MAX_N_NODES], n;

  f = fopen(ClusterFile, "r");
  assert(f != NULL);


  i = -1;
  n = 0;
  do {
    i++;
    fscanf(f,"%d\n",&(N_Clusters[i]));
    if (i>=1) n+= N_Clusters[i];
  } while (N_Clusters[i]!=1);
  N_Buffer_Level = i;

  for (i=0;i<n;++i) { cluster_size[i] = 0; }

  for (i=0;i<N_Clusters[0];++i) {
    fscanf(f,"%*d %d \n", &j);
    (cluster_size[j])++;
    Cluster_id[i] = j;
  }
  for (i=0;i<N_Clusters[1];++i) { 
    if (cluster_size[i]<=1) {
      printf("warning: cluster_size[%d] = %d \n", i, cluster_size[i]);
    }
  }
  i = 0;
  while (fscanf(f,"%*d %d \n", &j) != EOF) {
    Hierachy_Cluster_id[i] = j;
    cluster_size[j]++;
    ++i;
  }
  for (i=0;i<n;++i) { 
    if (cluster_size[i]<=1) {
      printf("warning: cluster_size[%d] = %d \n", i, cluster_size[i]);
    }
  }
  fclose(f);
}


/***********************************************************************/
/* read in input     files                                             */
/***********************************************************************/
void read_input_topology( const string &fn ) {

  ifstream inFile;
  inFile.open ( fn.c_str() ) ;

  if (!inFile ) return ;

char line[ MAX_STRING_LENGTH ] ;

  while ( inFile.getline (line, MAX_STRING_LENGTH ) ) {
     int i, lchild, rchild, par ;
     int numTokens =
     sscanf(line, "id:%d child:%d %d par:%d\n",
          &i, &lchild, &rchild, &par) ;
     int id = line[0]=='i' &&  line[1] == 'd' ;
     if ( id ) {
        if ( numTokens == 4 ) {
          NodeType *from  = gBoundedSkewTree->TreeNode (i ) ;
          from->  L = lchild;
          from->  R = rchild;
          from->  parent = par;
        } else {
          cerr << "ERROR: cannot parse \"" << line << "\" in file" << fn << endl ; 
          exit (0) ;
        }
      }
  }

  inFile.close() ;

}


/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void calc_Merge3rees_cost(NodeType *node, int L, int R) {
int i;
NodeType tmpnode, child[3];
double cost;

  alloca_NodeType(&tmpnode);
  int nterms = (int) gBoundedSkewTree->Nterms() ;
  if (L==Node[L].root_id) {
    if (L<nterms) return;
    assign_NodeType(&(child[0]), &(Node[Node[L].L]));
    assign_NodeType(&(child[1]), &(Node[Node[L].R]));
  } else {
    assign_NodeType(&(child[0]), &(Node[L]));
    assign_NodeType(&(child[1]), &(TempNode[L]));
  }
  assign_NodeType(&(child[2]), &(CandiRoot[R]));
  for (i=1;i<3;++i) {
    tmpnode.Merge2Nodes ( &(child[i%3]), &(child[(i+1)%3]));
    cost = BME_merging_cost(&tmpnode, &(child[(i+2)%3]));
    if (cost < node->area[node->ca].subtree_cost) {
      node->area[node->ca].subtree_cost = cost;
    }
  } 
  free_NodeType(&tmpnode);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void calc_Merge4Trees_cost(NodeType *node, int L, int R) {
int i;
NodeType tmpnode1, tmpnode2, child[4];
double cost;

  alloca_NodeType(&tmpnode1);
  alloca_NodeType(&tmpnode2);
  int nterms = (int) gBoundedSkewTree->Nterms() ;
  if (L==Node[L].root_id) {
    if (L<nterms) return;
    assign_NodeType(&(child[0]), &(Node[Node[L].L]));
    assign_NodeType(&(child[1]), &(Node[Node[L].R]));
  } else {
    assign_NodeType(&(child[0]), &(Node[L]));
    assign_NodeType(&(child[1]), &(TempNode[L]));
  }
  if (R==Node[R].root_id) {
    if (R<nterms) return;
    assign_NodeType(&(child[2]), &(Node[Node[R].L]));
    assign_NodeType(&(child[3]), &(Node[Node[R].R]));
  } else {
    assign_NodeType(&(child[2]), &(Node[R]));
    assign_NodeType(&(child[3]), &(TempNode[R]));
  }
  for (i=1;i<3;++i) {
    tmpnode1.Merge2Nodes  ( &(child[i%3]), &(child[(i+1)%3]));
    tmpnode2.Merge2Nodes  ( &(child[(i+2)%3]), &(child[3]));
    cost = BME_merging_cost(&tmpnode1, &tmpnode2);
    if (cost < node->area[node->ca].subtree_cost) {
      node->area[node->ca].subtree_cost = cost;
    }
  }
  free_NodeType(&tmpnode1);
  free_NodeType(&tmpnode2);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void calc_merging_cost_sub(NodeType *node, int L, int R) {
int i,j;

  assert(BST_Mode == BME_MODE);
  i = CandiRoot[L].ca;
  j = CandiRoot[R].ca;
  assert(i==0);
  assert(j==0);
  if (CandiRoot[L].area[i].capac >  CandiRoot[R].area[j].capac) {
    calc_Merge3rees_cost(node, L,R);
  } else {
    calc_Merge3rees_cost(node, R,L);
  }
  /*
  calc_Merge4Trees_cost(node, L,R);
  */
  
}
