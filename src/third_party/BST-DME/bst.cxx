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


#include <cmath>
#include "bst_header.h"
#include "bst.h"
#include "bst_sub1.h"
#include "bst_sub3.h"
#include "IME_code.h"
#include "facility.h"
#include "stdio.h"
#include "stdlib.h"


/*
  Curr_Npoints is the current number of nodes (leaves + internal nodes + 1).
The leaves are numbered from 0 to nterms -1, and
the internal nodes from nterms to 2*nterms-1.`
  The node indexed by "nterms" is preserved for the clock source, which
may not be used yet.
  Initially, Curr_Npoints = nterms+1 , which means that
that are nterms leaves and one clock source.
Then Curr_Npoints is increased by one whenever a new tree root is produced
during bottom-up merging.
*/


static int Curr_Npoints ;

BstTree   *gBoundedSkewTree ;

vector <NodeType > Node, CandiRoot, TempNode;
//AreaType *TempArea;
//int N_TempArea;
AreaType *TempArea = (AreaType*)calloc(1000, sizeof(AreaType)); //Modified
int N_TempArea=1000; //jianchao : value assigned for IME //Modified

TrrType *L_sampling, *R_sampling;
int N_Sampling = 7, n_L_sampling, n_R_sampling; 
char *Marked;
int *UnMarkedNodes;
int CHECK = NO; 
int Max_n_regions = 0; 
int Max_irredundant_regions = 0; 
int N_Area_Per_Node  = 1; 
int k_Parameter = -1;
int  N_Neighbor  = 1;  
double Cost_Function = 1; 
BucketType **Bucket;
int Local_Opt = NO;
int BST_Mode = BME_MODE;
int Dynamic_Selection = NO;   /* no dynamic topology change  */
int N_Index, MAX_N_Index; 
double Start_Tcost = 0, Start_Skew_B = -2, Skew_B = 0, Skew_B_CLS[MAX_N_SINKS];
double Last_Time, Gamma=1;
double PURES[2], PUCAP[2]; /*per unit resistance and capacitance */
double PURES_V_SCALE = 1;
double PUCAP_V_SCALE = 1;
double K[2];   
       /* K[i] = 0.5*PURES[i]*PUCAP[i], quardratic terms in Elmore delay */
PairType *Best_Pair;
int  Read_Delay_Skew_File = NO;
PointType  Fms_Pt[2][2];   /* feasible merging section on JR */
int n_Fms_Pt[2];
PointType  JR_corner[2];
int N_Bal_Pt[2];
PointType Bal_Pt[2][2];  
int  n_JR[2];
PointType  JR[2][MAX_TURN_PTS];
int  n_JS[2];
PointType  JS[2][MAX_TURN_PTS];
TrrType  L_MS, R_MS;
vector<double> EdgeLength, StubLength;
int *N_neighbors;
int   **The_NEIghors;
double **Neighbor_Cost;
ClusterType *Cluster;
/* ============================= */
double MAX_x, MAX_y, MIN_x, MIN_y;
double Split_Factor = 10.0;
int *NearestCenter;
int *TmpClusterId;
TmpClusterType *TmpCluster, Tmp_x_Cluster, Tmp_y_Cluster;
/* =============================*/
int N_Buffer_Level = 1, N_Clusters[MAX_BUFFER_LEVEL], Total_CL =0;
int TreeRoots[MAX_N_SINKS];  /* nodes which are the roots */
int Cluster_id[MAX_N_NODES], *Buffered;
int R_buffer= 100;
double C_buffer= 50.0e-15*PUCAP_SCALE;
double Delay_buffer = 100e-12*PUCAP_SCALE;
int R_buffer_size[N_BUFFER_SIZE];


int All_Top = NO;

int Expand_Size = 3;
double Weight  = 0.0;
int Cmode = 0;

PointType  *Points;
int    *TmpMarked;
double *Capac;

int Hierachy_Cluster_id[MAX_N_NODES];

int N_Obstacle = 0;            /* Number of obstacles */


double MaxClusterDelay, *ClusterDelay;
int TmpCondition = NO, N_Top_Embed = 0;


double calc_merging_cost(int , int );

extern double calc_buffered_node_delay(int v);
extern int  JR_corner_exist(int i);
extern void set_K();
extern void assign_NodeType(NodeType *node1, NodeType *node2);
extern int detour_Node(int v);
extern void sort_pts_on_line(PointType JR[], int n);

extern double Point_dist(PointType p1, PointType p2);
extern double pt2linedist(PointType , PointType , PointType , PointType *ans);
extern double ms_distance(TrrType *ms1,TrrType *ms2);
extern double min4(double x1, double x2, double x3,double x4);
extern double max4(double x1, double x2, double x3,double x4); 
extern void kohck_Irredundant(AreaType *areas, int *n_areas); 
extern void tsao_Irredundant(AreaType *areas, int *n_areas); 
extern void Irredundant(AreaSetType *stair); 
extern int equal(double x,double y); 
extern int equivalent(double x,double y, double fuzz);
extern int pt_on_line_segment(PointType pt,PointType pt1,PointType pt2); 
extern int PT_on_line_segment(PointType *pt,PointType pt1,PointType pt2); 
extern int same_Point(PointType p1, PointType p2);
extern int Same_Point_delay(PointType *p, PointType *q);
extern int JS_line_type(NodeType *node); 
extern int  Manhattan_arc_JS(NodeType *node); 
extern int area_Manhattan_arc_JS(AreaType *area);
extern double pt_skew(PointType pt);
extern double merge_cost(NodeType *node);
extern double area_merge_cost(AreaType *area);
extern int Manhattan_arc(PointType p1,PointType p2);
extern int merging_segment_area(AreaType  *area);
extern int TRR_area(AreaType *area);

extern int case_Manhattan_arc();
extern void ms_to_line(TrrType *ms,double *x1,double *y1,double *x2,double *y2);
extern void ms2line(TrrType *ms, PointType *p1, PointType *p2);
extern void line_to_ms(TrrType *ms,double x1,double y1,double x2,double y2);
extern void line2ms(TrrType *ms, PointType p1, PointType p2);
extern void pts2TRR(PointType pts[], int n, TrrType *trr);

extern int trrContain(TrrType *t1,TrrType *t2);
extern int in_bbox(double x,double y,double x1,double y1,double x2,double y2);
extern void core_mid_point(TrrType *trr, PointType *p);
extern void make_intersect( TrrType *trr1, TrrType *trr2, TrrType *t );
extern void make_intersect_sub( TrrType *trr1, TrrType *trr2, TrrType *t );
extern void make_core(TrrType *trr,TrrType *core);
extern void make_1D_TRR(TrrType *trr,TrrType *core);
extern double radius(TrrType *trr);
extern void build_trr(TrrType *ms,double d,TrrType *trr);
extern double pt2ms_distance(PointType *pt, TrrType *ms);
extern double pt2TRR_distance_sub(PointType *pt, TrrType *trr);
extern double pt2TRR_distance(PointType *pt, PointType pts[], int n);
extern int bbox_overlap(double x1, double y1, double x2, double y2, 
                 double x3, double y3, double x4, double y4);
extern int L_intersect(double *x, double *y, double x1, double y1,double x2,
           double y2, double x3, double y3, double x4, double y4);
extern int lineIntersect(PointType *p, PointType p1, PointType p2, PointType p3,
              PointType p4);
extern double linedist(PointType lpt0,PointType lpt1, PointType lpt2, 
                       PointType lpt3, PointType ans[2]);
extern int parallel_line(PointType p1,PointType p2,PointType p3,PointType p4);

extern void check_Point_delay(PointType *pt);
extern void check_Point(PointType *pt);
extern void check_JS_line(NodeType *node,NodeType *node_L, NodeType *node_R);
extern void check_mss(AreaType *area,AreaType *area_L, AreaType *area_R);
extern void check_x(AreaType *, AreaType *, AreaType *, double *x);
extern void check_fms(AreaType *area_L, AreaType *area_R, int side);
extern void check_ZST_detour(NodeType *node,NodeType *node_L,NodeType *node_R);
extern void check_trr(TrrType *t) ;
extern void check_ms(TrrType *ms); 
extern void check_a_sampling_segment(AreaType *area, TrrType *ms);
extern void check_tmparea(AreaType *tmparea, int n);
extern void check_const_delays(PointType *p1,PointType *p2);

extern void get_all_areas(int v, int i);
extern void store_n_areas(AreaType tmparea[], int n, NodeType *node);
extern void store_n_areas_IME(NodeType *node,AreaSetType *result);
extern void store_last_n_areas_IME(NodeType *node,AreaSetType *result);
extern void store_area_for_sink(int i);
extern void build_NodeTRR(NodeType *node);
extern double minskew(NodeType *node, int mode); 
extern double maxskew(NodeType *node, int mode);

extern void print_IME_areas(NodeType *node,NodeType *node_L, 
       NodeType *node_R,int ,int );
extern void print_double_array(double *a, int  n);
extern void print_JR_sub(FILE *f, NodeType *node);
extern void print_JS_sub(FILE *f, NodeType *node);
extern void print_merging_region(FILE *f, NodeType *node);
extern void print_node(NodeType *node) ;
extern void print_MR(NodeType *node);
extern void print_child_region(NodeType *node);
extern void print_max_npts();

extern void print_overlapped_regions();
extern void print_max_n_mr();
extern void print_n_region_type();
extern void print_tree_of_merging_segments(char fn[]);
extern void print_Bal_Pt(NodeType *node);
extern void print_Fms_Pt(NodeType *node);
extern void print_node_info(NodeType *node);
extern void print_node_informatio(NodeType *node,NodeType *node_L, 
                                  NodeType *node_R);
extern void JS_processing(AreaType *area);
extern void JS_processing_sub(AreaType *area);
extern void JS_processing_sub2(AreaType *area);
extern void recalculate_JS();

extern void remove_epsilon_err(PointType *q) ;
extern int same_line(PointType p0,PointType p1,PointType q0,PointType q1);
extern void construct_TRR_mr(AreaType *area);
extern int any_fms_on_JR();
extern void trace();


extern double calc_JR_area_sub(PointType p0,PointType p1,PointType p2,PointType p3);
extern int calc_line_type(PointType pt1,PointType pt2);
extern int calc_side_loc(int side);

extern void calc_merge_distance(double r, double c, double cap1,double delay1,
        double cap2, double delay2, double d,double *d1,double *d2);

extern void new_calc_merge_distance(PointType pt1, PointType pt2, int delay_id,
                         double *d1,double *d2);
extern double calc_delay_increase(double p, double cap, double x, double y);
extern double pt_delay_increase(double p,double cap,PointType *q0,PointType *q1); 
extern double _pt_delay_increase(double pat, double leng,double cap,
       PointType *q0,PointType *q1);
extern void calc_pt_coor_on_a_line(PointType q0,PointType q1, double d0,double d1, 
                              PointType *pts);
extern void calc_Bal_of_2pt(PointType *pt0, PointType *pt1, int delay_id, 
       int bal_pt_id, double *d0, double *d1, PointType *ans);
extern void check_calc_Bal_of_2pt(PointType *pt0, PointType *pt1, int delay_id,
                           PointType *ans, double d0, double d1);


extern void calc_vertices(AreaType *area);
extern void calc_pt_delays(AreaType *area, PointType *q1,PointType q0,PointType
q2);
extern void calc_BS_located(PointType *pt,AreaType *area, PointType *p1,
			PointType *p2);
extern void calc_JS_delay(AreaType *area, AreaType *area_L,AreaType *area_R);
extern int calc_TreeCost(int v, double *Tcost, double *Tdist);
extern void print_cluster_cost(double cost);
extern void calc_BST_delay(int v); 

extern void set_SuperRoot();


extern void alloca_NodeType(NodeType *node);
extern void free_NodeType(NodeType *node);
extern void ExG_DME_memory_allocation();
extern void Ex_DME_memory_allocation();
extern void read_clustering_info(char ClusterFile[]);

extern void init_marked(int v); 
extern void init_turn_pt_on_JS(int side,PointType  *pt);
extern void calc_merging_cost_sub(NodeType *node, int L, int R); 
extern double path_between_JSline(AreaType *area, PointType line[2][2],
                                  PointType path[],int *n); 
extern int unblocked_segment(PointType *p0, PointType *p1);
extern double path_finder(PointType p1, PointType p2, PointType path[], int *n);
extern double calc_pathlength(PointType path[], int n, int mode);
extern void modify_blocked_areas(AreaType area[], int *n, int b1, int b2);

extern int calc_all_cluster_neighbors(PairType pair[], int n_clusters,int size);

extern void init_re_embed_top(int v);
extern void init_re_embed_top_sub(int v);
extern void draw_a_TRR(TrrType *trr);

void check_JR_linear_delay_skew(int side) ;
/********************************************************************/
/* find solution of equation: Ax*x+Bx+c=0                           */
/********************************************************************/
static 
double sol_equation(double A, double B, double C) {
double x, y;

  y = B*B-4.0*A*C;
  assert(y>=0);
  x = (-B + sqrt(y))/(2.0*A);
  assert(x > -FUZZ);
  x = tMAX(0,x);
  return(x);
}

/***********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
static 
double calc_x(AreaType  *area) {
double x;
double r,c;

  r = PURES[H_];
  c = PUCAP[H_];
  x = r*(area->unbuf_capac) + area->R_buffer*c; 
  return(x);
}
/***********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void calc_B0_B1(AreaType *area, double *B0, double *B1) {
double r,c;

  r = PURES[H_];
  c = PUCAP[H_];
  *B0= r*(area->area_L->unbuf_capac) + area->area_L->R_buffer*c;
  *B1= r*(area->area_R->unbuf_capac) + area->area_R->R_buffer*c;
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
double calc_merge_pt_delay_sub(AreaType *area, double d0, double d1) {
double B0, B1, x, t0, t1, r, c, cap0, cap1;

  r = PURES[H_];
  c = PUCAP[H_];
  cap0 = area->area_L->capac;
  cap1 = area->area_R->capac;

  calc_B0_B1(area, &B0, &B1);
  x = area->L_StubLen;
  t0 = r*d0*(c*d0/2+cap0) + JS[0][0].max + K[H_]*x*x+B0*x;
  x = area->R_StubLen;
  t1 = r*d1*(c*d1/2+cap1) + JS[1][0].max + K[H_]*x*x+B1*x;
  
  assert(equal(t0, t1));
  return(t0);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void calc_merge_pt_delay(AreaType *area, double d0, double d1) {
  area->mr[0].max = calc_merge_pt_delay_sub(area, d0, d1);
  area->mr[0].min = area->mr[0].max - Skew_B;
  check_Point_delay(&(area->mr[0]));
}
/***********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
double set_StubLen_by_ClusterDelay(AreaType *area,double delay) {
double ans, x;

  double origB =  gBoundedSkewTree->Orig_Skew_B()  ;
  if (area->R_buffer>0) {
    if (origB==0) { assert(area->ClusterDelay >= delay - FUZZ); }
    area->ClusterDelay = tMAX(area->ClusterDelay, delay);
    x = calc_x(area);
    ans = sol_equation(K[H_], x, delay - area->ClusterDelay);
  } else {
    ans = 0;
  }
  return(ans);
}
/***********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void set_area_StubLen2(AreaType  *area, double *d0, double *d1) {
double x,y,z,len0,len1, d, B0, B1;
double  Lmax, Rmax;

  Lmax = JS[0][0].max;  
  Rmax = JS[1][0].max;  
  calc_B0_B1(area, &B0, &B1);
  d = area->dist;

  x = B1*d + Rmax - Lmax  + K[H_]*d*d;
  y = B0 + B1 + 2*d*K[H_];
  z = x/y;
  if (z   <0) {
    len1 = sol_equation(K[H_], B1, Rmax - Lmax); 
    len0 = 0  ;
  } else if (z   > d) {
    len0 = sol_equation(K[H_], B0, Lmax - Rmax); 
    len1 = 0;
  } else {
    len0 = z; 
    len1 = d - z  ; 
  }
  if (area->area_L->R_buffer > 0) {
    area->L_StubLen = len0;
    *d0 = 0;
  } else {
    area->L_StubLen = 0  ;
    *d0 = len0;
  }
  if (area->area_R->R_buffer > 0) {
    area->R_StubLen = len1; 
    *d1 = 0;
  } else {
    area->R_StubLen = 0  ;
    *d1 = len1;
  }
  area->L_EdgeLen = len0;
  area->R_EdgeLen = len1;
}
/***********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void calc_area_EdgeLen(AreaType *area, double *d0, double *d1) {
double cap0, cap1, tL, tR, diff;
double B0, B1, x;

  cap0 = area->area_L->capac;
  cap1 = area->area_R->capac;

  calc_B0_B1(area, &B0, &B1);
  x = area->L_StubLen;
  tL = JS[0][0].max + K[H_]*x*x+B0*x;
  x = area->R_StubLen;
  tR = JS[1][0].max + K[H_]*x*x+B1*x;

  diff = tMAX(0, area->dist - (area->L_StubLen+area->R_StubLen));
  calc_merge_distance(PURES[H_], PUCAP[H_],cap0,tL, cap1,tR,diff,d0,d1);

  /* just for check */
  calc_merge_pt_delay_sub(area, *d0, *d1);
  
  area->L_EdgeLen = *d0 + area->L_StubLen;
  area->R_EdgeLen = *d1 + area->R_StubLen;
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void set_ClusterDelay_CASE1(AreaType *area0, AreaType *area1, double L, 
                              double delay) {
double r,c,A,B,C,x, t;

  r = PURES[H_];
  c = PUCAP[H_];
  A = r*c;
  B = area0->R_buffer*c+r*area0->unbuf_capac-r*c*L-r*C_buffer;
  C = K[H_]*L*L+r*C_buffer*L+ delay - area1->ClusterDelay;
  x = sol_equation(A,B,C);
  if (x>= L) {
    area0->ClusterDelay = area1->ClusterDelay;
  } else {
    t = delay + area0->R_buffer*c*x+ r*x*(c*x/2+area0->unbuf_capac);
    assert( t >= area0->ClusterDelay  - FUZZ);
    area0->ClusterDelay = t;
  }
}
/****************************************************************************/
/* area_L is unbuffered, area_R is buffered.                                */
/****************************************************************************/
void set_ClusterDelay_CASE2(AreaType *area_L,AreaType *area_R, double x,
        double t0, double t1) {
double t, r, c;

  assert(area_L->R_buffer == 0);
  assert(area_R->R_buffer >  0);
  r = PURES[H_];
  c = PUCAP[H_];
  t = t0 + r*x*(c*x/2+area_L->capac);
  double origB =  gBoundedSkewTree->Orig_Skew_B()  ;
  if (1 ||  origB == 0) assert(t<=area_R->ClusterDelay + FUZZ);

  area_R->ClusterDelay = tMAX(t1, t);

  assert(area_L->ClusterDelay == 0);
  assert(area_L->capac == area_L->unbuf_capac);
  assert(area_R->ClusterDelay > 0);
}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
void print_current_time() {
long current_time;

  time(&current_time);
  printf("\nCurrent Time: %s \n",ctime(&current_time));
  fflush(stdout);
}

/******************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/******************************************************************/
void rm_same_JR_turn_pts(int side) {

  unsigned n = n_JR[side]; 
  unsigned j=1;
  for (unsigned i=1;i<n;++i) {
    PointType pt1 = JR[side][j-1]; 
    PointType pt2 = JR[side][i]; 
    if (!same_Point(pt1,pt2)) {
      JR[side][j++] = pt2; 
    }
  }
  n_JR[side] = j; 
}

/***********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void rm_same_pts_on_mr(AreaType *area) {
int i,j,n;
PointType pt, pre_pt;

  n = area->n_mr;
  for (i=1, j = 1;i<n;++i) { /* remove same turn points on mr */
    pt = area->mr[i];
    pre_pt = area->mr[j-1];
    if (!same_Point(pt,pre_pt) ) {
      area->mr[j++] = pt ; 
    }
  }
  if (j>1 && same_Point(area->mr[0],area->mr[j-1]) ) j--; 
  area->n_mr = j; 
}

/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
int rm_same_pts_on_sorted_array(PointType q[], int n) {
int i,j;
PointType pt, pre_pt;

  for (i=1, j = 1;i<n;++i) { /* remove same turn points on mr */
    pt = q[i];
    pre_pt = q[j-1];
    if (!same_Point(pt,pre_pt) ) {
      q[j++] = pt ;
    }
  }
/*
*/
  if (j>1 && same_Point(q[0],q[j-1]) ) j--; 
  return(j);
}

/****************************************************************************/
/*                                                                         */
/****************************************************************************/
static int point_compare_dec(const void  *p, const void  *q) {
PointType *pp, *qq;

/*
  if (p->t < q->t) {
    return(1);
  } else if (p->t > q->t ) {
    return(-1);
  } else {
    return(0);
  }
*/
  pp = (PointType *) p;
  qq = (PointType *) q;
  return( (pp->t < qq->t) ? YES: NO);
}
/******************************************************************/
/*                                                                */
/******************************************************************/
void check_JR_turn_pts_coor(AreaType *area, int side) {
int i,n, type;
PointType p,p0,p1,q0,q1; 

  n = n_JR[side]; 
  assert(n>=2);
  p0 = JR[side][0];
  p1 = JR[side][n-1];
  q0 = area->line[side][0];
  q1 = area->line[side][1];
  assert(same_Point(p0,q0) && same_Point(p1,q1));

  type = areaJS_line_type(area); 
  for (i=1;i<n;++i) {
    p = JR[side][i]; 
    if (type == VERTICAL ) {
      assert(equal(p.x, p0.x));
      JR[side][i].x = p0.x; 
    } else if (type == HORIZONTAL ) {
      assert(equal(p.y, p0.y));
      JR[side][i].y = p0.y; 
    } else if (type == TILT || type == FLAT) {
      assert(i==n-1 || pt_on_line_segment(p,p0,p1));
    } else {
      assert(0);
    }
  }
}
/******************************************************************/
/* initialize JR[side][i]  with turning pt child->mr[j] */
/* for the case when area->line{0] and area->line{1] are */
/* parallel  hori./vert. lines                                     */
/******************************************************************/
void add_JS_pts(AreaType *area,int side, AreaType *child) {
int i, n; 

  assert(!same_Point(JS[side][0],JS[side][1])); 
  
  n=2; 
  for (i=0;i<child->n_mr;++i) {
    if ( pt_on_line_segment(child->mr[i],JS[side][0],JS[side][1]) 
         && !same_Point(child->mr[i],JS[side][0]) 
         && !same_Point(child->mr[i],JS[side][1]) ) {
      JS[side][n]  = child->mr[i]; 
      n++;
    }
  }
  n_JS[side] = n;
  assert( n <= MAX_TURN_PTS); 

  sort_pts_on_line(JS[side], n_JS[side]);
}

/******************************************************************/
/* check if pt = any point  in q[] */
/******************************************************************/
int repeated_pts(PointType q[], int n, PointType pt) {
int i; 

  for (i=0;i<n;++i) {
    if (same_Point(pt,q[i])) {
      return(i);
    }
  }
  return(NIL);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void add_LINEAR_turn_point_sub(int side,double d0,double d1,
			  PointType p0,PointType p1) {
int n;
double d;
PointType pt, q;

  if (d0<=0 || d1<=0 ) return;
  d = d0+d1;
  n = n_JR[side];
  pt.x = (p0.x*d1 + p1.x*d0)/d;
  pt.y = (p0.y*d1 + p1.y*d0)/d;
  if (n==3) {
    q = JR[side][2];
    if (same_Point(pt,q)) {
      return; 
    }
  }
  pt.max = tMAX(p0.max - d0, p1.max-d1);
  pt.min = tMIN(p0.min + d0, p1.min+d1);
  JR[side][n] = pt;
  n_JR[side] = n+1;
  
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void add_LINEAR_turn_point(AreaType *area) {
int side;
PointType p0,p1;
double d,t ;

  for (side=0;side < 2; ++side) {
    p0 = JR[side][0];
    p1 = JR[side][1];
    d = Point_dist(p0,p1);
    t = p0.max-p1.max;
    add_LINEAR_turn_point_sub(side,(d+t)/2, (d-t)/2,p0,p1);
    t = p0.min-p1.min;
    add_LINEAR_turn_point_sub(side,(d-t)/2, (d+t)/2,p0,p1);
    sort_pts_on_line(JR[side], n_JR[side]);
/*
    rm_same_JR_turn_pts(side);
*/
    n_JR[side] = rm_same_pts_on_sorted_array( JR[side], n_JR[side]);
    if (CHECK==1) {
      check_JR_turn_pts_coor(area, side); 
    }
  }
}

/******************************************************************/
/*                                                                */
/******************************************************************/
double delay_pt_JS(int JS_side, int x, int side, int i, double t_from[]) {
double ans;

  if (i==0) { /* max-delay */
    ans = JS[side][x].max;
  } else {   /* min_delay */
    ans = JS[side][x].min;
  }

  if (JS_side != side) {
    ans += t_from[side]; 
  }
  return(ans); 
  
}


/***********************************************************************/
/*                                                                     */
/***********************************************************************/
double calc_A(PointType *p0, PointType *p1) {
double A;

  if (equal(p0->x, p1->x)) {     /* JS is vertical */
    assert( !equal(p0->y, p1->y) );
    A = K[V_];
  } else {     /* JS is horizontal */
    assert( equal(p0->y, p1->y) );
    A = K[H_];
  }
  return(A);
}

/***********************************************************************/
/* add one turn point on JR due to the changing slopes of max- or min- */
/* delay (depending on the value of y)  between JR[side][x] and  */
/* JR[side][x+1]                                                 */
/***********************************************************************/
void add_turn_point(int side,int x,int y, double t_from[]) {
PointType pt1, pt2, new_pt;
int i,j; 
double d,d1,d2, B[2][2], t2,t1,  tmp[2][2]; 
double A;

  pt1 = JR[side][x]; 
  pt2 = JR[side][x+1]; 
  A = calc_A(&pt1, &pt2);
  d = Point_dist(JR[side][x],JR[side][x+1]); 
  assert(!equal(d,0));

  for (i=0;i<2;++i) {
    for (j=0;j<2;++j) {
      t1 = delay_pt_JS(side,x,  i,j, t_from);
      t2 = delay_pt_JS(side,x+1,i,j, t_from);
      B[i][j] = (t2-t1)/d-A*d; 
    }
  }

  d1 = (delay_pt_JS(side,x,0,y, t_from) - delay_pt_JS(side,x,1,y, t_from) )
       /(B[1][y]-B[0][y]); 

  assert(d1 > 0 && d1 < d);

  d2 = d - d1;

  /* calc coordinate of new_pt */
  new_pt = pt2;
  new_pt.x = (pt1.x*d2 + pt2.x*d1)/d;
  new_pt.y = (pt1.y*d2 + pt2.y*d1)/d;

  /* calc delays of new_pt */
  for (i=0;i<2;++i) {
    for (j=0;j<2;++j)  {
      tmp[i][j] = delay_pt_JS(side,x,  i,j, t_from) + A*d1*d1+B[i][j]*d1;
    }
  }
  new_pt.max = tMAX(tmp[0][0],tmp[1][0]);
  new_pt.min = tMIN(tmp[0][1],tmp[1][1]);

  assert(equal(tmp[0][y],tmp[1][y]));

  JR[side][(n_JR[side])++] = new_pt;
  assert(n_JR[side] <= MAX_TURN_PTS); 
}

  
/******************************************************************/
/* add turn points on JR[side] due to changing slopes of delays */
/******************************************************************/
void add_more_JR_pts(AreaType *area, int side, double t_from[]) {
int i, n; 
double t; 

  n = n_JR[side];

  for (i=0;i<n-1;++i) { /* for each point */
    t = (JS[side][i].max - JS[1-side][i].max - t_from[1-side])*
	(JS[side][i+1].max - JS[1-side][i+1].max - t_from[1-side]);
    if ( t  < - FUZZ ) {
      add_turn_point(side,i,0, t_from); 
    }
    t = (JS[side][i].min - JS[1-side][i].min - t_from[1-side])*
        (JS[side][i+1].min - JS[1-side][i+1].min - t_from[1-side]);
    if ( t  < - FUZZ ) {
      add_turn_point(side,i,1, t_from); 
    }
  }
  sort_pts_on_line(JR[side], n_JR[side]);
/*
  rm_same_JR_turn_pts(side);
*/
  n_JR[side] = rm_same_pts_on_sorted_array( JR[side], n_JR[side]);
  if (CHECK==1) {
    check_JR_turn_pts_coor(area, side); 
  }

}
/******************************************************************/
/*                                                                */
/******************************************************************/
void print_JR_slopes(int n, double *skew_rate, 
                double *maxD_slope, double *minD_slope) {
  printf("maxD_slope: ");
  print_double_array(maxD_slope, n-1);
  printf("minD_slope: ");
  print_double_array(minD_slope, n-1);
  printf("skew_rate: ");
  print_double_array(skew_rate, n-1);

  assert(0);
}
/******************************************************************/
/* check turn_pts on JR[side]                              */
/******************************************************************/
void check_Elmore_delay_skew(PointType q[], int n, double delta) {
int i; 
double d1,d2, t, t1,t2, skew_rate[MAX_TURN_PTS-1];
double maxD_slope[MAX_TURN_PTS-1], minD_slope[MAX_TURN_PTS-1];
double A;
PointType p0,p1,p2; 

  assert(n>=2);
  p0 = q[0];
  A = calc_A(&(q[0]), &(q[1]));
  for (i=0;i<n-1;++i) {
    p1 = q[i];
    p2 = q[i+1];
    d1 = Point_dist(p1,p0);
    d2 = Point_dist(p2,p0);
    t1 = d2 - d1;
    t2 = d2 + d1;
    assert(t1 > FUZZ);
    skew_rate[i] = (pt_skew(p2) - pt_skew(p1) ) / t1;
    maxD_slope[i] = (p2.max-p1.max)/t1 -A*t2;       
    minD_slope[i] = (p2.min-p1.min)/t1 -A*t2;       
  }
  /* slopes of max-delays must be monotone increasing */
  for (i=0;i<n-2;++i) { 
    t1 = maxD_slope[i]; 
    t2 = maxD_slope[i+1]; 
    if ( t1 > t2+100.0*FUZZ) {
      print_JR_slopes(n, skew_rate, maxD_slope, minD_slope);
    }
  }
  /* slopes of min-delays must be monotone decreasing */
  for (i=0;i<n-2;++i) {
    t1 = minD_slope[i]; 
    t2 = minD_slope[i+1]; 
    if ( t1 < t2-100.0*FUZZ) {
      print_JR_slopes(n, skew_rate, maxD_slope, minD_slope);
    }
  }

  /* slopes of skew       must be strictly monotone increasing */
  for (i=0;i<n-2;++i) {
    t  = skew_rate[i+1] - skew_rate[i];
    if ( t < -100.0*delta ) {
      print_JR_slopes(n, skew_rate, maxD_slope, minD_slope);
    }
  }

}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void set_pt_coord_case2(int side_loc, PointType *pt, double d) {
  if (side_loc==LEFT) { 
    pt->x += d;
  } else if (side_loc==RIGHT) { 
    pt->x -= d;
  } else if (side_loc==BOTTOM) {
    pt->y += d;
  } else { 
    pt->y -= d; 
  }
}
/******************************************************************/
/*                                                              */
/******************************************************************/
void calc_JS_pt_delays(int side, PointType *pt) {
int i; 
 
  for (i=0;i<n_JS[side] -1 ;++i) {
    if ( pt_on_line_segment(*pt,JS[side][i], JS[side][i+1]) ) {
      break;
    }
  }
  assert( i < n_JS[side] -1 );
  calc_pt_delays(NULL, pt, JS[side][i], JS[side][i+1]);
}

/******************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/* add turn points on JS[o_side] to  JR[side] */
/******************************************************************/
void add_more_JS_pts(int side, double dist, int n)  {
int i, j, k, o_side; 

  o_side = 1- side;

  k = n_JS[side];
  for (i=0;i<n ;++i) {
    JS[side][k] = JS[o_side][i]; 
    j = calc_side_loc(o_side); 
    set_pt_coord_case2(j, &(JS[side][k]),dist);
    j = repeated_pts(JS[side], n_JS[side], JS[side][k]); 
    if (j <0) {
      assert( k < MAX_TURN_PTS); 
      calc_JS_pt_delays(side, &(JS[side][k]));
      k++;
    }
  }
  n_JS[side] = k; 

}

/******************************************************************/
/*                                                               */
/******************************************************************/
void calc_new_delay(AreaType *area, int side, PointType *pt) {
double tL, tR;

  tL = pt_delay_increase(Gamma, area->area_L->capac,&(area->line[0][side]), pt);
  tR = pt_delay_increase(Gamma, area->area_R->capac,&(area->line[1][side]), pt);
  pt->max = tMAX(area->line[0][side].max + tL, area->line[1][side].max + tR);
  pt->min = tMIN(area->line[0][side].min + tL, area->line[1][side].min + tR);
}

/******************************************************************/
/* find the corner points of Joining Box.                       */
/******************************************************************/
void calc_JR_corner_sub2(int i, double x0,double y0,double x1,double y1) {

  if ( (x0-x1)*(y0-y1)  < 0 ) {
    if (i==0) {
      JR_corner[i].x = tMAX(x0,x1);
      JR_corner[i].y = tMAX(y0,y1);
    } else {
      JR_corner[i].x = tMIN(x0,x1);
      JR_corner[i].y = tMIN(y0,y1);
    } 
  } else {
    if (i==0) {
      JR_corner[i].x = tMIN(x0,x1);
      JR_corner[i].y = tMAX(y0,y1);
    } else {
      JR_corner[i].x = tMAX(x0,x1);
      JR_corner[i].y = tMIN(y0,y1);
    }
  }
}
/******************************************************************/
/* find the corner points of Joining Box.                       */
/******************************************************************/
void calc_JR_corner_sub1(AreaType *area, int i) {
double x0,y0, x1,y1;

  x0 = JS[0][i].x;
  y0 = JS[0][i].y;
  x1 = JS[1][i].x;
  y1 = JS[1][i].y;
  if (JR_corner_exist(i)==YES) {
    calc_JR_corner_sub2(i,x0, y0, x1,y1);
    calc_new_delay(area, i, &(JR_corner[i]));
    assert(equal(JR_corner[i].x, tMAX(x0,x1))  ||
           equal(JR_corner[i].x, tMIN(x0,x1))  );
    assert(equal(JR_corner[i].y, tMAX(y0,y1))  ||
           equal(JR_corner[i].y, tMIN(y0,y1))  );
  }

}
/******************************************************************/
/* find the corner points of Joining Box.                       */
/******************************************************************/
void calc_JR_corner(AreaType *area) {
int i;

  if (!equal(JS[0][0].y,JS[0][1].y)) {
    assert(JS[0][0].y > JS[0][1].y);
  }
  if (!equal(JS[1][0].y,JS[1][1].y)) {
    assert(JS[1][0].y > JS[1][1].y);
  }
  if (area_Manhattan_arc_JS(area) && !equal(area->dist,0)) {
    for (i=0;i<2;++i) {
      calc_JR_corner_sub1(area, i);
    }
  }
}

/******************************************************************/
/*  calc all the turn_pts JR[side][i], side=0,1, i =0,...,n  
   when JS are PARA_MANHATTAN_ARC; 
*/
/******************************************************************/
void calc_JR_endpoints(AreaType *area) {

  assert(Same_Point_delay(&(JS[0][0]), &(area->line[0][0])));
  assert(Same_Point_delay(&(JS[0][1]), &(area->line[0][1])));
  assert(Same_Point_delay(&(JS[1][0]), &(area->line[1][0])));
  assert(Same_Point_delay(&(JS[1][1]), &(area->line[1][1])));

  JR[0][0].x = JS[0][0].x;
  JR[0][1].x = JS[0][1].x;
  JR[1][0].x = JS[1][0].x;
  JR[1][1].x = JS[1][1].x;
  JR[0][0].y = JS[0][0].y;
  JR[0][1].y = JS[0][1].y;
  JR[1][0].y = JS[1][0].y;
  JR[1][1].y = JS[1][1].y;

  calc_new_delay(area, 0, &(JR[0][0]));
  calc_new_delay(area, 0, &(JR[1][0]));
  calc_new_delay(area, 1, &(JR[0][1]));
  calc_new_delay(area, 1, &(JR[1][1]));
}

/***********************************************************************/
/* remove redundant  turn points :  JR[side][i]              */
/***********************************************************************/
void rm_redundant_JR_pt(int side) {
int i,k,n;
PointType p1,p2, pt[MAX_TURN_PTS];
double d; 

  n = n_JR[side];
  assert(n>=2);
  for (i=0;i<n-1;++i) {
    p1 = JR[side][i]; 
    p2 = JR[side][i+1]; 
    d = Point_dist(p1,p2);
    assert(d >= FUZZ);
    JR[side][i].t= (pt_skew(p2)- pt_skew(p1) )/d;
  }
  pt[0]=JR[side][0];
  k=1;
  for (i=1;i<n-1;++i) {
    p1 = pt[k-1]; 
    p2 = JR[side][i];
    if (equal(p1.t,p2.t) ) { /* p2 is a redundant turn point. */
    } else if (p1.t > p2.t) {
      /* skew slope must be strictly monotone increasing, */
      /* i.e., skew curve  must be convex */ 
      check_Elmore_delay_skew(JR[side], n_JR[side], FUZZ);
      assert(0);            
    } else { 
      pt[k++] = p2; 
    }
  }
  pt[k++] = JR[side][n-1];
  for (i=0;i<k;++i) {
    JR[side][i] = pt[i];
    assert(pt[i].max > -FUZZ);
    assert(pt[i].min > -FUZZ);
  }
  n_JR[side] = k;

}

/******************************************************************/
/*                                                               */
/******************************************************************/
void calc_new_rect_JS_sub(AreaType *area,int side, double dist, double delay) {
int i, n, loc;

  n = n_JS[side];
  loc = calc_side_loc(side);
  for (i=0;i<n ;++i) {
    set_pt_coord_case2(loc, &(JS[side][i]),dist);
    JS[side][i].max += delay;
    JS[side][i].min += delay;
  }
  for (i=0;i<2;++i) {
    set_pt_coord_case2(loc, &(area->line[side][i]),dist);
    area->line[side][i].max += delay;
    area->line[side][i].min += delay;
  }
}

/******************************************************************/
/*                                                               */
/******************************************************************/
double calc_t_inc(AreaType *area, double d0) {
double t0, r, c;

  r = PURES[H_];
  c = PUCAP[H_];
  t0 = d0*r*(d0*c/2+area->unbuf_capac) + area->R_buffer*c*d0;
  return(t0);
}
/******************************************************************/
/*                                                               */
/******************************************************************/
void calc_new_rect_JS(AreaType *area, AreaType *area_L, AreaType *area_R) {
double d, d0, d1, t0, t1;

  t0 = calc_t_inc(area_L, area->L_StubLen);
  t1 = calc_t_inc(area_R, area->R_StubLen);

  d0 = area->L_StubLen;
  d1 = area->R_StubLen;
  d = area->dist ;
  area->dist = tMAX(0,d-d0-d1);
  if (d0+d1 >= d ) {
    if (d0>=d/2 && d1 >= d/2) {
      d0 = d1 = d/2;
    } else if (d0 <= d1 ) {
      d1 = d - d0;
    } else {
      d0 = d - d1;
    }
    area->L_EdgeLen = area->L_StubLen;
    area->R_EdgeLen = area->R_StubLen;
  } else {
    area->L_EdgeLen =  area->R_EdgeLen = NIL;
  }
  calc_new_rect_JS_sub(area, 0, d0, t0);
  calc_new_rect_JS_sub(area, 1, d1, t1);
}

/******************************************************************/
/*  calc all the turn_pts JR[side][i], side=0,1, i =0,...,n  */
/* when JS are parallel vertical / horizontal lines              */
/******************************************************************/
void calc_JR_case2(AreaType *area, AreaType *area_L,AreaType *area_R) {
double t_from[2] ;
int i, j;

  add_JS_pts(area, 0, area_L);
  add_JS_pts(area, 1, area_R);

  i = n_JS[0];
  j = n_JS[1];
  add_more_JS_pts(0, area->dist, j); 
  add_more_JS_pts(1, area->dist, i);

  sort_pts_on_line(JS[0], n_JS[0]);
  sort_pts_on_line(JS[1], n_JS[1]);
  assert(n_JS[0]==n_JS[1]);
  if (CHECK) check_Elmore_delay_skew(JS[0], n_JS[0], FUZZ);
  if (CHECK) check_Elmore_delay_skew(JS[1], n_JS[1], FUZZ);

  t_from[0]=pt_delay_increase(Gamma, area_L->capac, &(JS[0][0]), &(JS[1][0]) );
  t_from[1]=pt_delay_increase(Gamma, area_R->capac, &(JS[1][0]), &(JS[0][0]) );

  for (i=0;i<2;++i) {
    n_JR[i] = n_JS[i];
    for (j=0;j<n_JR[i];++j) {
      JR[i][j].x = JS[i][j].x;
      JR[i][j].y = JS[i][j].y;
      JR[i][j].max = tMAX(JS[i][j].max, JS[1-i][j].max+t_from[1-i]); 
      JR[i][j].min = tMIN(JS[i][j].min, JS[1-i][j].min+t_from[1-i]); 
    }
  }
  if (CHECK) check_Elmore_delay_skew(JR[0], n_JR[0], FUZZ);
  if (CHECK) check_Elmore_delay_skew(JR[1], n_JR[1], FUZZ);
/*
  rm_same_JR_turn_pts(0);
  rm_same_JR_turn_pts(1);
*/
  n_JR[0] = rm_same_pts_on_sorted_array( JR[0], n_JR[0]);
  n_JR[1] = rm_same_pts_on_sorted_array( JR[1], n_JR[1]);
  if (CHECK==1) {
    check_JR_turn_pts_coor(area, 0); 
    check_JR_turn_pts_coor(area, 1); 
  }

  add_more_JR_pts(area, 0, t_from); 
  add_more_JR_pts(area, 1, t_from); 

  rm_redundant_JR_pt(0); 
  rm_redundant_JR_pt(1);
}

/******************************************************************/
/*                                                                */
/******************************************************************/
void check_calc_JR_case2(AreaType *area) {
  check_JR_turn_pts_coor(area,0);
  check_JR_turn_pts_coor(area,1);
  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
    check_JR_linear_delay_skew(0);
    check_JR_linear_delay_skew(1);
  } else {
    check_Elmore_delay_skew(JR[0], n_JR[0], 0);
    check_Elmore_delay_skew(JR[1], n_JR[1], 0);
  }
}
/******************************************************************/
/*   add the endpoints of fms of JR[side][i]                */
/******************************************************************/
void add_fms_of_JR_sub(PointType pt0,PointType pt1, PointType *pt) {
double d0,d1,d;

  d = Point_dist(pt0,pt1);
  assert(d > FUZZ);
  d0 = (Skew_B-pt_skew(pt0))*d/(pt_skew(pt1)-pt_skew(pt0) );  
  d1 = d-d0;
  *pt = pt0; 
  pt->x = (pt0.x*d1+pt1.x*d0)/d; 
  pt->y = (pt0.y*d1+pt1.y*d0)/d; 

  calc_pt_delays(NULL, pt, pt0,pt1);
  assert(equal( pt_skew(*pt), Skew_B)); 
}
/******************************************************************/
/*   add the endpoints of fms of JR[side][i]                */
/******************************************************************/
void add_fms_of_JR(int side) {
PointType pt, pt0, pt1, new_JR[MAX_TURN_PTS];
int i, j, n; 
double t1, t2;

  j = 0; 
  n = n_JR[side]; 
  for (i=0;i<n-1;++i) { 
    new_JR[j++] = pt0 = JR[side][i];
    pt1 = JR[side][i+1];
    t1 = pt_skew(pt0) - Skew_B;
    t2 = pt_skew(pt1) - Skew_B;
    if ( t1*t2 < 0 && !equal(t1,0) && !equal(t2, 0) ) { 
		  /* add one point with skew==Skew_B */
      add_fms_of_JR_sub(pt0,pt1,&pt); 
      new_JR[j++] = pt; 
    }
  }
  new_JR[j++] = pt1;

  if (j>n) { /* If there are any points with skew == Skew_B*/ 
    n_JR[side] = j; 
    if (j>MAX_TURN_PTS) { printf("%d JR turn points \n",j); }
    assert(j<=MAX_TURN_PTS);
    for (i=0;i<j;++i) JR[side][i] = new_JR[i];
  }
}

/******************************************************************/
/*                                                               */
/******************************************************************/
void check_new_JS(AreaType *area) {
  double origB =  gBoundedSkewTree->Orig_Skew_B()  ;
  if (area->area_L->R_buffer>0 ) {
    if ( origB == 0) {
      assert(equal(JS[0][0].max, area->area_L->ClusterDelay));
    }
  } else {
    assert(area->L_StubLen==0);
  }
  if (area->area_R->R_buffer>0 ) {
    if ( origB == 0) {
      assert(equal(JS[1][0].max, area->area_R->ClusterDelay));
    }
  } else {
    assert(area->R_StubLen==0);
  }
}
/******************************************************************/
/* calculate new JS due to StubLength of non-zero length         */
/******************************************************************/
void calc_new_JS(AreaType *area) {
int i;
double b[2], t, x[2];
TrrType trrL, trrR, trr, msL, msR;

  /* first, calculate new L_MS and R_MS */
  if (area->L_StubLen+area->R_StubLen >= area->dist) {
    /* Case I: detour wiring happens; */
    build_trr(&L_MS, area->L_StubLen, &trrL);
    build_trr(&R_MS, area->R_StubLen, &trrR);
    make_intersect(&trrL, &trrR, &trr); 
    make_1D_TRR(&trr, &msL);
    msR = msL;
    area->L_EdgeLen = area->L_StubLen;
    area->R_EdgeLen = area->R_StubLen;
  } else {
    build_trr(&L_MS, area->L_StubLen, &trrL);
    build_trr(&R_MS, area->dist - area->L_StubLen, &trrR);
    make_intersect(&trrL, &trrR, &msL); 
    build_trr(&L_MS, area->dist - area->R_StubLen, &trrL);
    build_trr(&R_MS, area->R_StubLen, &trrR);
    make_intersect(&trrL, &trrR, &msR); 
    area->L_EdgeLen = area->R_EdgeLen = NIL;
  }
  ms2line(&msL, &(JS[0][1]), &(JS[0][0]));
  ms2line(&msR, &(JS[1][1]), &(JS[1][0]));
  area->dist = ms_distance(&msL, &msR);
  b[0] = calc_x(area->area_L);                          
  b[1] = calc_x(area->area_R);                          
  x[0] = area->L_StubLen;
  x[1] = area->R_StubLen;
  for (i=0;i<2;++i) {
    t = b[i]*x[i] + K[H_]*x[i]*x[i];
    JS[i][1].max = JS[i][0].max = t  + JS[i][0].max;
    JS[i][1].min = JS[i][0].min = t  + JS[i][0].min;
  }
  JS_processing(area);
  check_new_JS(area);
}
/******************************************************************/
/*  calc all the turn_pts JR[side][i], side=0,1, i =0,...,n  */
/******************************************************************/
void calc_JR(AreaType *area, AreaType *area_L,AreaType *area_R) {

  if (area_Manhattan_arc_JS(area)) { /* JS's are para Manhattan arcs */
    calc_JR_endpoints(area); 
  } else {  /* JS's are para rectilinear lines */
    bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
    if ( linear ) {
      calc_JR_endpoints(area); 
      add_LINEAR_turn_point(area); 
    } else {
      calc_JR_case2(area, area_L,area_R ); 
    }
    if (CHECK==1) check_calc_JR_case2(area); 
    add_fms_of_JR(0);
    add_fms_of_JR(1);
  }
}

/******************************************************************/
/*                                                                */
/******************************************************************/
double calc_JR_minskew(int i) {
int j;
double min_skew = DBL_MAX;

  for (j=0;j<n_JR[i] ;++j) { 
    min_skew = tMIN(pt_skew(JR[i][j]),min_skew);
  }
  return(min_skew);
}
/********************************************************************/
/* set node->area[node->ca].mr[x] = minimum skew section of JR[side][x]      */
/********************************************************************/
static int
calc_mss ( AreaType *area ) {
int i,j,n_pt,n ;
double min_skew;

  min_skew = DBL_MAX;
  area->n_mr = 0;

  /* fix for bug reported by Nate on 8-02-02  */
  /* The test case is in BENCHMARKS/bug_nate_8_02_02 */
  /* The bug happens only when area->dist==0, or equivalently
     calc_JR_minskew(0 == calc_JR_minskew(1) 
     Because of this bug, I added the detour at the
     incorrect child nodes. 
  */
/*  
  if (equal(area->dist,0)) { n_sides = 1; } else { n_sides = 2; }

  if (calc_JR_minskew(0) < calc_JR_minskew(1)) {
    i = 0;
  } else {
    i = 1;
  }
*/

  if (equal(area->dist,0)) { 
    /* break tie  */
    i = (area->line[0][0].max > area->line[1][0].max)?0:1 ;
  } else {
    i = (calc_JR_minskew(0) < calc_JR_minskew(1))? 0: 1;
  }
  min_skew = calc_JR_minskew(i);
  n = n_JR[i]; 
  n_pt =0;
  for (j=0;j<n;++j) { /* for each monotone section of each side */
    PointType pt = JR[i][j];
    if ( equal(pt_skew(pt),min_skew) ) {
      area->mr[n_pt++] =pt;
    }
  }
  area->n_mr = n_pt;
  return i ;
}

/******************************************************************/
/*                                                                */
/******************************************************************/
void check_set_detour_EdgeLen(AreaType *area, PointType pt0, PointType pt1) {
int line0type;
double tL, tR, skew;

  line0type = calc_line_type(pt0,pt1);
  if (line0type != MANHATTAN_ARC) return;

  if (area->L_EdgeLen == 0 ) {
    tL = 0;
    tR =  _pt_delay_increase(Gamma, area->R_EdgeLen, pt1.t, &pt1, &pt0);
  } else {
    tL =  _pt_delay_increase(Gamma, area->L_EdgeLen, pt0.t, &pt0, &pt1);
    tR = 0;
  }
  skew = tMAX(pt0.max+tL, pt1.max+tR) - tMIN(pt0.min+tL, pt1.min+tR);
  assert(equal(skew, Skew_B));
}
/********************************************************************/
/* set edgelengths for node's children  when detour wireing needed     */
/********************************************************************/
void set_detour_EdgeLen(AreaType *area, const int side ) {
PointType pt0, pt1, tmpPT;
double d0, d1, t, h, v ;

  pt0 = area->line[0][0];
  pt1 = area->line[1][0];
  pt0.t = area->area_L->capac;
  pt1.t = area->area_R->capac;
  t = pt_skew(area->mr[0]) - Skew_B;
  assert(t > 0);
  h = ABS(pt0.x-pt1.x);
  v = ABS(pt0.y-pt1.y);

  /* fix for bug reported by nate on 7-27-02 */ 
/*
  if (area->line[0][0].max > area->line[1][0].max) {
*/
  if ( side ==0 ) { /* detour at R-sided child */
    pt1.max = pt0.max - t - calc_delay_increase(Gamma, pt1.t,h,v);
    calc_Bal_of_2pt(&pt0,&pt1,0,0, &d0, &d1, &tmpPT);

    /* fix the bug reported by Nate on 8-14-02 */
    /*  d0 may not be identical to 0 on SUN/Solairs platform */
    assert( equal(d0,0) );
    area->L_EdgeLen = 0;
    area->R_EdgeLen = tMAX(area->dist, d1);
  } else { /* detour at L-sided child */
    pt0.max = pt1.max - t - calc_delay_increase(Gamma, pt0.t,h,v);
    calc_Bal_of_2pt(&pt0,&pt1,0,0, &d0, &d1, &tmpPT);
    /* fix the bug reported by Nate on 8-14-02 */
    /*  d1 may not be identical to 0 on SUN/Solairs platform */
    assert( equal(d1,0) );
    area->L_EdgeLen = tMAX(area->dist, d0);
    area->R_EdgeLen = 0;
  }
  if (0) check_set_detour_EdgeLen(area, pt0, pt1);
}

/********************************************************************/
/* set min-delay of node->area[node->ca].mr(i) */
/********************************************************************/
void set_delay_for_mss(AreaType *area) {
int i;

  for (i=0;i<area->n_mr;++i) {
    area->mr[i].min = area->mr[i].max - Skew_B;
  }
}

/******************************************************************/
/*                                                                */
/******************************************************************/
void add_Bal_Pt(int side,PointType p) {
int n;

  n = N_Bal_Pt[side];
  if (n==0 || !same_Point(p,Bal_Pt[side][0]))  {
    Bal_Pt[side][n++] = p;
  }
  N_Bal_Pt[side] = n; 
}
/********************************************************************/
/*  calculate the points with d0 to area->line[0] and d1 to line[1] */
/********************************************************************/
void calc_coordinate(AreaType *area,int side, double d0,double d1,PointType *pts) {
PointType tmp, q0,q1;
double d; 

  q0 = area->line[0][side];
  q1 = area->line[1][side];
  if (equal(q0.x,q1.x) || equal(q0.y,q1.y) ) {
    assert(JR_corner_exist(side)==NO); 
    calc_pt_coor_on_a_line(q0,q1,d0,d1,pts);
  } else {
    assert(JR_corner_exist(side)==YES); 
    tmp = JR_corner[side];
    d = Point_dist(tmp,q0); 
    if (d >= d0) {
      calc_pt_coor_on_a_line(q0,tmp,d0,d-d0,pts);
    } else {
      calc_pt_coor_on_a_line(tmp,q1,d0-d,d1,pts);
    }
  }
}

/******************************************************************/
/*                                                                */
/******************************************************************/
void new_add_Bal_Pt(int side,PointType *pt, double d0, double d1) {
int n;

  if ( equal(d0,0) || equal(d1,0) ) {
    return;
  }
  n = N_Bal_Pt[side];
  if (n==0 || !same_Point(*pt,Bal_Pt[side][0]))  {
    Bal_Pt[side][n++] = *pt;
  }
  N_Bal_Pt[side] = n;
}

/******************************************************************/
/*                                                               */
/******************************************************************/
void check_cal_Bal_Pt_sub(AreaType *area, PointType *q,int side) {
PointType q0, q1 ;
double max_x, max_y, min_x, min_y; 

  q0 =  area->line[0][side];
  q1 =  area->line[1][side];
  max_x = max4(area->line[0][0].x,area->line[0][1].x,
               area->line[1][0].x,area->line[1][1].x);
  max_y = max4(area->line[0][0].y,area->line[0][1].y,
               area->line[1][0].y,area->line[1][1].y);
  min_x = min4(area->line[0][0].x,area->line[0][1].x,
               area->line[1][0].x,area->line[1][1].x);
  min_y = min4(area->line[0][0].y,area->line[0][1].y,
               area->line[1][0].y,area->line[1][1].y);
  if (equal(q0.x, q1.x)) {
    assert(equal(q->x, q0.x));
  } else if (equal(q0.y, q1.y)) {
    assert(equal(q->y, q0.y));
  } else if ((q0.x-q1.x)*(q0.y-q1.y) < 0 ) {
    if (side==0) {
      assert(equal(q->x,max_x) || equal(q->y, max_y));
    } else {
      assert(equal(q->x,min_x) || equal(q->y, min_y));
    }
  } else {
    if (side==0) {
      assert(equal(q->x,min_x) || equal(q->y, max_y));
    } else {
      assert(equal(q->x,max_x) || equal(q->y, min_y));
    }
  }

  assert(pt_skew(*q) < Skew_B + FUZZ); 
}
/******************************************************************/
/*                                                               */
/******************************************************************/
void cal_Bal_Pt_sub(AreaType *area, int side) {
PointType q0, q1, pt0, pt1;
double d0, d1;
int bal_delay_id, i;

  q0 =  area->line[0][side];
  q1 =  area->line[1][side];
  q0.t = area->area_L->capac;
  q1.t = area->area_R->capac;
  if ((q0.x-q1.x)*(q0.y-q1.y) < 0 ) {
    bal_delay_id = 1 - side;
  } else {
    bal_delay_id = side;
  }
  calc_Bal_of_2pt(&q0, &q1, 0, bal_delay_id,&d0,&d1,&pt0);
  new_add_Bal_Pt(side, &pt0, d0, d1);
  calc_Bal_of_2pt(&q0, &q1, 1, bal_delay_id,&d0,&d1,&pt1);
  new_add_Bal_Pt(side, &pt1, d0, d1);

  for (i=0;i<N_Bal_Pt[side];++i) {
    calc_new_delay(area, side, &(Bal_Pt[side][i]));
    check_Point(&(Bal_Pt[side][i]));
    check_cal_Bal_Pt_sub(area, &(Bal_Pt[side][i]), side);
  }
}
/******************************************************************/
/*                                                               */
/******************************************************************/
void cal_Bal_Pt(AreaType *area) {

  N_Bal_Pt[0] = 0;
  N_Bal_Pt[1] = 0;
  if (!equal(area->dist, 0)) {
    cal_Bal_Pt_sub(area, 0);
    cal_Bal_Pt_sub(area, 1);
  }

}

/******************************************************************/
/*                                                               */
/******************************************************************/
void add_fms_of_line(int side,PointType  q[5], int *n) {
int i, m; 

  m = n_Fms_Pt[side];
  for  (i=0;i<m;++i) {
    q[(*n)++] = Fms_Pt[side][i];
  }
}

/******************************************************************/
/*                                                                */
/******************************************************************/
void add_Fms_Pt(int side,PointType *p) {
int n;

  n = n_Fms_Pt[side];
  assert(n<2);
  if (n==0 || !same_Point(*p,Fms_Pt[side][0]))  {
    Fms_Pt[side][n++] = *p;
  }
  n_Fms_Pt[side] = n;
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
void calc_Fms_of_line_sub(PointType *pt,PointType *q, PointType *ans) {
double d, d1, sk0, sk1;

  sk0 = pt_skew(*pt);
  sk1 = pt_skew(*q);
  d = Point_dist(*pt, *q);
  d1 = d*(Skew_B-sk1)/(sk0-sk1);
  assert( d > FUZZ);
  assert( Skew_B-sk1 >= 0);
  assert( sk0-sk1 > FUZZ);
  calc_pt_coor_on_a_line(*pt,*q,d-d1,d1,ans);

}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int calc_Fms_of_line(AreaType *area, PointType pt, PointType q, int side) {
PointType tmpPt, p1, p2;
double sk;
int i, ans = NO;

  sk = pt_skew(pt);
  if ( equal(sk, Skew_B) || sk <=Skew_B) {
    add_Fms_Pt(side, &pt);
    ans = YES;
  } else {
    for (i=0;i<N_Bal_Pt[side];++i) {
      if (Point_dist(pt, (Bal_Pt[side][i]))  < Point_dist(pt,q) ) {
        check_Point(&(Bal_Pt[side][i]));
        q = Bal_Pt[side][i];
      }
    }
    sk = pt_skew(q);
    if ( equal(sk,Skew_B) ) {
      add_Fms_Pt(side, &q);
      ans = YES;
    } else if ( sk  < Skew_B) {
      calc_Fms_of_line_sub(&pt,&q, &tmpPt);  
      calc_new_delay(area, side, &tmpPt);
      if (!equal(pt_skew(tmpPt), Skew_B)) {
        p1 = pt;
        p2 = q;
        calc_new_delay(area, side, &p1);
        calc_new_delay(area, side, &p2);
        assert(equal(pt_skew(tmpPt), Skew_B)); 
      }
      add_Fms_Pt(side, &tmpPt);
      ans = YES;
    }
  }
  return(ans);
}

/********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/********************************************************************/
void cal_Fms_Pt_sub(AreaType *area,int side){
int n0, n1, ans;
PointType tmp[2];
  
  n_Fms_Pt[side] = 0; 
  if (side==0) {
    tmp[0]=JR[0][0];
    tmp[1]=JR[1][0];
  } else {
    n0 = n_JR[0];
    n1 = n_JR[1];
    tmp[0]=JR[0][n0-1];
    tmp[1]=JR[1][n1-1];
  }
  if (JR_corner_exist(side)) {
    ans = calc_Fms_of_line(area, tmp[0], JR_corner[side], side);
    if (ans==YES) {
      ans = calc_Fms_of_line(area, tmp[1], JR_corner[side], side);
      if (ans==NO) {
         ans = calc_Fms_of_line(area, JR_corner[side], tmp[0], side);
         assert(ans == YES);
      }
    }  else {
      ans = calc_Fms_of_line(area, JR_corner[side], tmp[1], side);
      if (ans==YES) {
        ans = calc_Fms_of_line(area, tmp[1], JR_corner[side], side);
        assert(ans==YES);
      }
    }
  } else {
    ans = calc_Fms_of_line(area, tmp[0], tmp[1], side);
    if (ans) calc_Fms_of_line(area, tmp[1], tmp[0], side);
  }
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
void cal_Fms_Pt(AreaType *area, AreaType *area_L, AreaType *area_R) {
int i; 

  for (i=0;i<2;++i) {
    cal_Fms_Pt_sub(area, i); 
    assert(N_Bal_Pt[i]==0 || n_Fms_Pt[i] > 0); 
  }
}

/******************************************************************/
/* find the max- min-delay balance points on the line            */
/******************************************************************/
void add_balance_pt(int side, PointType q[5], int *n) {
int i, m; 

  m = N_Bal_Pt[side];
  for  (i=0;i<m;++i) {
    q[(*n)++] = Bal_Pt[side][i];
  }
}

/********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/********************************************************************/
void check_pts(AreaType *area, PointType q[], int n) {
int i, type;

  for (i=0;i<n-1;++i) {
    type = calc_line_type(q[i],q[i+1]);
    if (type!=VERTICAL && type != HORIZONTAL){
      for (i=0;i<n;++i) { print_Point(stdout, q[i]); }
      printf("\"corners\n");
      if (JR_corner_exist(0)) {
	print_Point(stdout, JR_corner[0]);
      }
      if (JR_corner_exist(1))  {
	print_Point(stdout, JR_corner[1]);
      }
      exit(0); 
    }
  }
}
/********************************************************************/
/* determine turn pts on the first or last line connecting JS[0]/JS[1]  */
/********************************************************************/
void mr_between_JS(AreaType *area,int side, double cap[2]) {
int i,j, n=0;
PointType q[2], pt[5]; 
  
  q[0]=area->line[0][side];
  q[1]=area->line[1][side];
   
  n=0; 
  add_balance_pt(side, pt, &n); 
  add_fms_of_line(side, pt, &n);
  if (JR_corner_exist(side)) {
    if (pt_skew((JR_corner[side])) <= Skew_B + FUZZ) {
      pt[n++] = JR_corner[side];
    }
  }

  if (n==0) return;

  for (i=0;i<n;++i) { /* calculate the distance to JS[side] */
    pt[i].t=Point_dist(pt[i],q[side]);  
  }

  qsort(pt, n, sizeof(PointType), point_compare_dec);
  /* in decreasing order of distance to area->line[side] */
  

  j = 1;
  for (i=1;i<n;++i) {
    if (equal(pt[j-1].t, pt[i].t)) {
      /* redundant points */
    } else if (pt[j-1].t > pt[i].t) {
      pt[j++] = pt[i];
    } else {
      assert(0);
    }
  }
  if (CHECK==1 && area_Manhattan_arc_JS(area)) 
     check_pts(area, pt, j);

  n = area->n_mr;
  for (i=0;i<j;++i) {
    area->mr[i+n] = pt[i];
  }
  area->n_mr += j;

}
/********************************************************************/
/* determine the necessary turn pts JR[side][i]             */
/********************************************************************/
void mr_on_JS_sub(int side,int m[2]) {
int n,o_n, o_side,m0,m1;
PointType p0,q0;

  n = n_JR[side];
  m0 = 1;
  o_side=(side+1)%2;
  p0 = JR[side][0];
  q0 = JR[o_side][0];
  if ( n_Fms_Pt[0] ==0 && pt_skew(p0) < pt_skew(q0) ) {
    for (m0=1;m0<n-1;++m0) {
      if ( equal(pt_skew(JR[side][m0]) ,Skew_B)  ) { 
	break; }
    }
  }
  p0 = JR[side][n-1];
  o_n = n_JR[o_side];
  q0 = JR[o_side][o_n-1];
  m1 = n-2; 
  if ( n_Fms_Pt[1] ==0 && pt_skew(p0) < pt_skew(q0) ) {
    for (m1=n-2;m1>=m0;--m1) {
      if ( equal(pt_skew(JR[side][m1]) ,Skew_B) ) { 
	break; }
    }
  }
  m[0] = m0;
  m[1] = m1;
}

/*****************************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/*****************************************************************************/
void 
check_JR_linear_delay_skew(int side) {
double maxD_slope[MAX_TURN_PTS-1], minD_slope[MAX_TURN_PTS-1];
double t1,t2, d, t, skew_rate[MAX_TURN_PTS-1]; 
PointType pt1,pt2;
int i,n; 

  n = n_JR[side];
  for (i=0;i<n-1;++i) {
    pt1 = JR[side][i];
    pt2 = JR[side][i+1];
    d = Point_dist(pt1,pt2);
    maxD_slope[i] = (pt2.max  - pt1.max ) / d;
    minD_slope[i] = (pt2.min  - pt1.min ) / d;
    skew_rate[i] = (pt_skew(pt2)- pt_skew(pt1) ) / d; 
  }

  for (i=0;i<n-1;++i) {
    t = ABS(skew_rate[i]);
    t1 = ABS(maxD_slope[i]);
    t2 = ABS(minD_slope[i]);
    if ( (!equal(t,2.0) && !equal(t,0.0)) || !equal(t1,1.0) || !equal(t2,1.0)){
      print_JR_slopes(n_JR[side],skew_rate, maxD_slope, minD_slope);
    }
  }

  /* slopes of max-delays must be monotone increasing */
  for (i=0;i<n-2;++i) {
    t1 = maxD_slope[i+1] - maxD_slope[i];
    if ( !equal(t1,0.0) && !equal(t1,2.0) ) {
      print_JR_slopes(n_JR[side],skew_rate, maxD_slope, minD_slope);
    }
  }
  /* slopes of min-delays must be monotone decreasing */
  for (i=0;i<n-2;++i) {
    t1 = minD_slope[i+1] - minD_slope[i];
    if ( !equal(t1,0.0) && !equal(t1,-2.0) ) {
      print_JR_slopes(n_JR[side],skew_rate, maxD_slope, minD_slope);
    }
  }

  /* slopes of skew       must be strictly monotone increasing */
  for (i=0;i<n-2;++i) {
    t1  = skew_rate[i+1] - skew_rate[i];
    if ( !equal(t1,2.0) && !equal(t1,4.0) ) {
      print_JR_slopes(n_JR[side],skew_rate, maxD_slope, minD_slope);
    }
  }

}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
double set_Fms_Pt_min_delay(AreaType *area,int side, PointType *p0,
                            PointType *p1) {
double x,y, t;

  x = ABS(p0->x - p1->x);
  y = ABS(p0->y - p1->y);
  assert(equal(x,0) || equal(y,0) );
  if (side==0) {
    t = calc_delay_increase(Gamma, area->area_L->capac, x,y);
  } else {
    t = calc_delay_increase(Gamma, area->area_R->capac, x,y);
  }
  return(t);
}


/***********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
double calc_skew_slope(AreaType *area) {
double x0,y0,x1,y1,cap0, cap1, t;

  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
    return(2.0);
  } else { /* Elmore delay model */
    x0 = area->line[0][0].x;
    y0 = area->line[0][0].y;
    x1 = area->line[1][0].x;
    y1 = area->line[1][0].y;
    cap0 = area->area_L->capac;
    cap1 = area->area_R->capac;
   
    if (equal(x0,x1)) {
      t = PURES[V_]*(cap0+ cap1+area->dist*PUCAP[V_]);
    } else if (equal(y0, y1)) {
      t = PURES[H_]*(cap0+ cap1+area->dist*PUCAP[H_]);
    } else {
      assert(0);
    }
    return(t);
  }
}

/********************************************************************/
/* determine if feasible merging section of a line connecting JR(L) */
/* and JR(R) is empty         */
/********************************************************************/
void fms_of_line_exist(AreaType *area, AreaType *area_L, AreaType *area_R, 
		  int side, int x) {
double d, t;
PointType pt;
int nn, side_loc;

  nn = area->n_mr; 
  pt = JR[side][x]; 
  t = calc_skew_slope(area);
  d = (pt_skew(pt) -Skew_B)/t;
  if (d <= 0 ) { 
    area->mr[nn] = pt; 
    area->n_mr = nn + 1;
  } else if (d <= area->dist) {
    side_loc = calc_side_loc(side); 
    set_pt_coord_case2(side_loc, &pt,d);
    pt.min += set_Fms_Pt_min_delay(area, side, &(JR[side][x]), &pt);
    pt.max = Skew_B + pt.min;
    area->mr[nn] = pt; 
    area->n_mr = nn + 1;
  }
  
  assert(nn <= MAX_mr_PTS);
}
 
/******************************************************************/
/* construct left part or right part of merging region node->area[node->ca].mr  */
/******************************************************************/
void mr_on_JS(AreaType *area,int side, AreaType *area_L,AreaType *area_R) {
int i, m[2]; 

  
  assert(n_JR[side] >=2);
  mr_on_JS_sub(side, m);
  if (side==0) {
    for (i=m[0];i<=m[1];++i) {
      fms_of_line_exist(area, area_L, area_R, side, i); 
    }
  } else {
    for (i=m[1];i>=m[0];--i) {
      fms_of_line_exist(area, area_L, area_R, side, i); 
    }
  }
}

/******************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/******************************************************************/
int JR_a_line() {
PointType p0,q0,p1,q1; 
double min_x, max_x, min_y, max_y; 

  p0 = JS[0][0];
  p1 = JS[0][1];
  q0 = JS[1][0];
  q1 = JS[1][1];
  min_x = min4(p0.x,p1.x,q0.x,q1.x); 
  max_x = max4(p0.x,p1.x,q0.x,q1.x);
  min_y = min4(p0.y,p1.y,q0.y,q1.y);
  max_y = max4(p0.y,p1.y,q0.y,q1.y);
  if (equal(max_x,min_x) || equal(max_y,min_y)) {
    return(YES);
  } else {
    return(NO);
  }
}
/******************************************************************/
/* construct merging region node                     */
/******************************************************************/
void construct_mr_sub1(AreaType *area, AreaType *area_L,AreaType *area_R) {
double cap[2]; 

  cap[0] = area_L->capac;
  cap[1] = area_R->capac;
  if (area_Manhattan_arc_JS(area) ) { /*para Manhattan Arc*/
    mr_between_JS(area,0, cap);
    if (area->mr >0 && !JR_a_line()) mr_between_JS(area,1, cap);
  } else { /* parallel  hori. or vert. lines  */
    mr_between_JS(area,0, cap);
    mr_on_JS(area,0, area_L,area_R);
    mr_between_JS(area,1, cap);
    if (area->dist > FUZZ) { /* non-overlapping hori. or vert. lines  */
      mr_on_JS(area,1, area_L,area_R);
    }
  }
}

/******************************************************************/
/* calculate the detoru wiring                       */
/******************************************************************/
void construct_mr_sub2(AreaType *area, AreaType *area_L,AreaType *area_R) {

  /* fix a bug reported by Nate on testcase test7_3 with topology topo_out*/
  /* fix for bug reported by nate on 7-27-02 */

  int side = calc_mss(area);
  check_mss(area,area_L,area_R);
  set_detour_EdgeLen(area, side );   

  set_delay_for_mss(area); 

  if (area_L->R_buffer == 0) assert(area->L_StubLen==0);
  if (area_R->R_buffer == 0) assert(area->R_StubLen==0);
  area->L_EdgeLen += area->L_StubLen;
  area->R_EdgeLen += area->R_StubLen;

}
/***********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void check_LINEAR_mr_sub(PointType p1,PointType p2) {
int i,j,n;
PointType p;

  print_Point(stdout,p1);
  print_Point(stdout,p2);
  printf("Fms_Pt \n");
  for (i=0;i<2;++i) {
    n = n_Fms_Pt[i];
    for (j=0;j<n;++j) { 
      p = Fms_Pt[i][j];
      print_Point(stdout,p);
      if (same_Point(p1,p)) { printf("p1 is fms \n"); }
      if (same_Point(p2,p)) { printf("p2 is fms \n"); }
    }
  }
  printf("Bal_Pt \n");
  for (i=0;i<2;++i) {
    n = N_Bal_Pt[i];
    for (j=0;j<n;++j) { 
      p = Bal_Pt[i][j];
      print_Point(stdout,p);
      if (same_Point(p1,p)) { printf("p1 is bal \n"); }
      if (same_Point(p2,p)) { printf("p2 is bal \n"); }
    }
  }
}
/***********************************************************************/
/*  check if mr is an octilinear octagon in the linear delay model.    */
/***********************************************************************/
void check_LINEAR_mr(AreaType *area) {
PointType p1,p2;
double a,b;
int i,n; 

  n = area->n_mr; 
  for (i=0; i<n; i++) {
    p1 = area->mr[i]; 
    p2 = area->mr[(i+1)%n]; 
    a = ABS(p2.y-p1.y);
    b = ABS(p2.x-p1.x);
    if (!equal(a,0) && !equal(b,0) && !equal(a,b)) {
      printf("a=%f, b= %f \n", a,b); 
      print_area(area); 
      check_LINEAR_mr_sub(p1,p2);
    }
    assert(equal(a,0) || equal(b,0) || equal(a,b));
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void check_mr_Point(AreaType *area) {
PointType pt;
int i; 
  
  for (i=0;i<area->n_mr;++i) {
    pt = area->mr[i]; 
    if (pt.min <0 || pt.min > pt.max || pt_skew(pt) >= Skew_B + FUZZ) {
      print_Point(stdout, pt);
      print_area(area);
      assert(0);
    }
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void check_mr(AreaType *area) {
int i,j,n;

  n = area->n_mr; 
  check_mr_Point(area);
  for (i=0;i<n;++i) {
    j = (i+1)%n;
    if (Manhattan_arc(area->mr[i],area->mr[j])) {
      check_const_delays(&(area->mr[i]),&(area->mr[j])); 
    }
  }
  if (area_Manhattan_arc_JS(area)           ) {
    for (i=0;i<n;++i) {
      j = calc_line_type(area->mr[i], area->mr[(i+1)%n]);
      assert(j==MANHATTAN_ARC || j==VERTICAL || j==HORIZONTAL);
    }
  }
  if (Skew_B == 0) assert(area->n_mr <=2);
  assert( n_JR[0] <= MAX_TURN_PTS);
  assert( n_JR[1] <= MAX_TURN_PTS);

  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
    if (area->n_mr > 16) { 
      print_area(area); 
      assert( area->n_mr <= 16); 
    }
    check_LINEAR_mr(area);
  } else {
    assert( area->n_mr <= MAX_mr_PTS); 
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void check_mr_array(AreaType area[], int n) {
int i;
  for (i=0;i<n;++i) {
    check_mr(&(area[i]));
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void check_mr2(AreaType *area) {

  if (area->npts > area->n_mr) {
    printf("area->npts = %d, area->n_mr = %d \n",area->npts,area->n_mr); 
    assert(0);               
  }

  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
    assert(area->npts <= 8);
  } else {
    assert(area->npts <= MAXPTS);
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void check_mr3_sub(AreaType *area) {
int i,j;
  print_area(area);
  for (i=0;i<2;++i) {
    printf("\"Fms_Pt[%d]:\n", i);
    for (j=0;j<n_Fms_Pt[i];++j) { print_Point(stdout, Fms_Pt[i][j]); } 
  }
  for (i=0;i<2;++i) {
    printf("\"Bal_Pt[%d]:\n", i);
    for (j=0;j<N_Bal_Pt[i];++j) { print_Point(stdout, Bal_Pt[i][j]); } 
  }
  for (i=0;i<2;++i) {
    printf("\"JR[%d]:\n", i);
    for (j=0;j<n_JR[i];++j) { print_Point(stdout, JR[i][j]); } 
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void check_mr3(AreaType *area) {
int i,j,n;
PointType p1, p2;

  n = area->n_mr;
  for (i=0;i<n;++i) {
    j = (i+1)%n; 
    p1 = area->mr[i];
    p2 = area->mr[j];
    if (!equal(p1.x, p2.x) && !equal(p1.y, p2.y) ) {
      /* not rectilinear boundary segment */
/*
      if (!equal(pt_skew(p1),Skew_B) || !equal(pt_skew(p2),Skew_B)) {
        printf("warning: 1792\n");
        print_area(area);
	check_mr3_sub(area);
        print_Point(stdout, p1);
        print_Point(stdout, p2);
      }
*/
      assert(equal(pt_skew(p1),pt_skew(p2))); 
    }
  }
}
/******************************************************************/
/* construct merging region mr for node .                     */
/******************************************************************/
void construct_mr(AreaType *area, AreaType *area_L,AreaType *area_R, int mode){

  area->n_mr = 0;
  calc_JR(area, area_L,area_R); 
  calc_JR_corner(area);
  cal_Bal_Pt(area); 
  cal_Fms_Pt(area, area_L, area_R); 
  if (any_fms_on_JR()) {  /* non-empty feasible merging region */
    if (mode==FAST) return; 
    construct_mr_sub1(area, area_L,area_R); 
  } else {  /* empty feasible merging region: need detour wire */
    construct_mr_sub2(area, area_L,area_R); 
    if (mode==FAST) return; 
  }
  if (area_Manhattan_arc_JS(area) && area->L_EdgeLen >=0 ) {
    assert(area->R_EdgeLen >=0 );
    construct_TRR_mr(area);
  }

  assert(area->n_mr <= MAX_mr_PTS);

  if (CHECK==1) check_mr(area); 
  area->n_mr = rm_same_pts_on_sorted_array(area->mr, area->n_mr);
  calc_vertices(area);
  if (CHECK==1) check_mr(area); 
  check_mr2(area);
  if (BST_Mode == BME_MODE) check_mr3(area);

}
/****************************************************************************/
/*  update root_id of all nodes in tree rooted at v.                     */
/****************************************************************************/
void updateRootId(int root_id,int v) {
  if (v<0)  return;

  NodeType *node = gBoundedSkewTree->TreeNode(v ) ;
  node->root_id = root_id;
  Neighbor_Cost[v][0] = DBL_MAX;
  
  updateRootId(root_id, node->L);
  updateRootId(root_id, node->R);
}

/****************************************************************************/
/* calculate the other child of node q which have one child p               */
/****************************************************************************/
int sibling(int p,int q) {
  if (p==Node[q].L)
    return(Node[q].R);
  if (p==Node[q].R)
    return(Node[q].L);
  assert(0);
}

/****************************************************************************/
/* set node q = the other child of node p which have one child o            */
/****************************************************************************/
void set_sibling(int o,int p,int q) {
  if (Node[p].L==o) {
    Node[p].L = q;
  } else if (Node[p].R==o) {
    Node[p].R = q; 
  } else {
    assert(0);
  }
}
/****************************************************************************/
/*   count_tree_nodes();                                     */
/****************************************************************************/
void count_tree_nodes(int root, int v, int *n) {
  if (v<0) return;

  (*n)++;
  assert(Node[v].root_id == root); 
  count_tree_nodes(root, Node[v].L,n);
  count_tree_nodes(root, Node[v].R,n);
}
/****************************************************************************/
/*   check the the root_id                                                  */
/****************************************************************************/
void check_root_id(int root) {
int n = 0;
   count_tree_nodes(root, root, &n);
}
/****************************************************************************/
/*   Print the first message.                                               */
/****************************************************************************/
void print_BST_Mode() {
  if (BST_Mode == IME_MODE) {
    printf("IME_MODE: (N_Area_Per_Node = %d, N_Sampling = %d) \n",
     N_Area_Per_Node, N_Sampling);
  } else if (BST_Mode == HYBRID_MODE) {
    printf("HYBRID_MODE: (N_Area_Per_Node = %d, N_Sampling = %d) \n",
     N_Area_Per_Node, N_Sampling);
  } else {
    printf("BME_MODE");
  }

}
/****************************************************************************/
/*   Print the first message.                                               */
/****************************************************************************/
void print_header( bool fixed_top) {

  printf("\n--------------------------------------------------------\n");

  unsigned nterms = gBoundedSkewTree->Nterms() ;
  unsigned npoints = gBoundedSkewTree->Npoints() ;

  string fn = gBoundedSkewTree->SinksFileName() ;
  printf("bst %s (nterms=%d, Npoints=%d) ", fn.c_str() , nterms, npoints); 
  printf("N_Obstalces = %d", N_Obstacle);
  printf("\n");
  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
    printf("Linear delay model ");
  } else {
    printf("Elmore delay model ");
  }
  if (Skew_B == DBL_MAX) { printf("Skew_B=DBL_MAX  ");
  } else { printf(" Skew_B = %.2f ", Skew_B); }
  if (!fixed_top) {
    printf("k_Parameter: %d CostFx=%.1f", k_Parameter, Cost_Function); 
  }
  printf("\n"); 
  print_BST_Mode();

  print_current_time();
  printf("--------------------------------------------------------\n");
  printf("\n"); 
  fflush(stdout);
}
/****************************************************************************/
/*    check compare neighbor                                                */
/****************************************************************************/
void check_compare_neighbors(int i,int j) {
int k1,k2; 
  /* printf("i=%d, j=%d \n", i,j);  */
  assert(j>=0);
  k1 = Node[i].root_id;
  k2 = Node[j].root_id;
  // unsigned npoints = gBoundedSkewTree->Npoints() ;
  int npoints = (int) gBoundedSkewTree->Npoints() ;
  
  assert(k1>=0 && k2 >= 0 && k1 <= npoints &&  k2 <= npoints);
  assert(k1 != k2 && !Marked[i] && !Marked[j]);

}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void update_neighbors_sub(int x,int y, double cost) {
int i,j,n;

  n = N_neighbors[x];
  for (j=0;j<n;++j) {
    if ( Neighbor_Cost[x][j] > cost) {
      break;
    }
  }
  if (j>=N_Neighbor) return; 
  N_neighbors[x]= n = tMIN(n+1, N_Neighbor);
  
  for (i=n-2;i>=j;--i) {
    The_NEIghors[x][i+1] = The_NEIghors[x][i];
    Neighbor_Cost[x][i+1] = Neighbor_Cost[x][i];
  }
  The_NEIghors[x][j] = y;
  Neighbor_Cost[x][j] = cost;
}

/****************************************************************************/
/*  update the nearest neighbors of mr(i) with newly added neighbor mr(j) */
/****************************************************************************/
void update_neighbors(int x,int y, double cost1) {
int cluster1, cluster2, i,j,n;
double cost2;

  n = N_neighbors[x];
  cluster1 = Node[x].root_id;
  for (i=0;i<n;++i) {
    j = The_NEIghors[x][i];
    cluster2 = Node[j].root_id;
    if (cluster1==cluster2) {
      cost2 = Neighbor_Cost[x][i]; 
      if (cost1 < cost2) {
	The_NEIghors[x][i] = y;
	Neighbor_Cost[x][i] = cost1;
      }
      return; 
    }
  }
   
  update_neighbors_sub(x,y,cost1);
}
/****************************************************************************/
/*  check if areas of a node are sorted by the decreasing order of capac.   */
/****************************************************************************/
int calc_best_area(int v) {
int min_i, i;
double x,y;

  min_i = 0;
  y = Node[v].area[min_i].capac;
  for (i=0;i<Node[v].n_area-1;++i) {
    x = Node[v].area[i].capac;
    if ( !equal(x,y) && x < y ) {
      min_i = i; 
    }
  }

  return(min_i);
  assert(min_i == 0); 
}
/****************************************************************************/
/*  compute the merging cost of mr(i) with its possible neighbor mr(j), and */
/*  update the neighborhood of mr(i) if mr(j) is indeed a neighbor of mr(i) */
/****************************************************************************/
void compare_neighbors(int i,int j) {
int k1,k2, n1, n2 ; 
double cost_inc, old_cost, cost;

  if (CHECK==1) check_compare_neighbors(i,j); 
  
  PointType &pt = Node[i].m_stnPt ;
  pt.t += 1.0;  /* number of comparisons for mr[i]; */ 

  cost = calc_merging_cost(i, j); 
  /* cost = calc_merging_cost_sub(&tmpnode,i,j); */
  k1 = Node[i].root_id;
  k2 = Node[j].root_id;
  if (BST_Mode == BME_MODE) {
    assert(Node[k1].ca==0);
    assert(Node[k2].ca==0);
    old_cost = Node[k1].area[0].subtree_cost + Node[k2].area[0].subtree_cost;
  } else {
    n1 = calc_best_area(k1);
    n2 = calc_best_area(k2);
    old_cost = Node[k1].area[n1].subtree_cost + 
               Node[k2].area[n2].subtree_cost;
  }
  cost_inc = cost - old_cost;
  update_neighbors(i,j,cost_inc);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void 
BstTree::BuildTrr ( unsigned n ) {
  m_ctrr.Initialize() ;
  unsigned j=0;
  for (unsigned i=0;i<n;++i) {
    if (!Marked[i]     ) {
      m_ctrr.Enclose ( *CandiRoot[i].ms );
      j++;
    }
  }
  double t = sqrt((double) j) ;
  N_Index = tMAX(1,(int) t);

  assert(N_Index <= MAX_N_Index);

}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
int xlow_index(NodeType *node) {
int i; 
double t;
 
  const TrrType & ctrr = gBoundedSkewTree->Ctrr() ; 
  t = (node->ms->xlow-ctrr.xlow)*N_Index/ctrr.Width(X);
  i = tMIN( (int) t, N_Index -1); 
  assert (i>=0 && i < N_Index);
  return(i); 

}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
int xhi_index(NodeType *node) {
int i;
double t;

  const TrrType & ctrr = gBoundedSkewTree->Ctrr() ; 
  t = (node->ms->xhi -ctrr.xlow)*N_Index/ctrr.Width(X);
  i = tMIN( (int) t, N_Index -1); 
  assert (i>=0 && i < N_Index);
  return(i); 
}
/****************************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
int ylow_index(NodeType *node) {
int i; 
double t;
 
  const TrrType & ctrr = gBoundedSkewTree->Ctrr() ; 
  t = (node->ms->ylow-ctrr.ylow)*N_Index/ctrr.Width(Y) ;         
  i = tMIN( (int) t, N_Index-1); 
  assert (i>=0 && i < N_Index);
  return(i); 

}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
int yhi_index(NodeType *node) {
int i;
double t;

  const TrrType & ctrr = gBoundedSkewTree->Ctrr() ; 
  t = (node->ms->yhi -ctrr.ylow)*N_Index/ctrr.Width(Y) ;         
  i = tMIN( (int) t, N_Index -1); 
  assert (i>=0 && i < N_Index);
  return(i); 
}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void bucket_partitioning_sub(int v) {
int j,j1,j2,k,k1,k2,n;

  j1 = xlow_index(&(CandiRoot[v]));
  j2 = xhi_index(&(CandiRoot[v]));
  k1 = ylow_index(&(CandiRoot[v]));
  k2 = yhi_index(&(CandiRoot[v]));

  for (j=j1;j<=j2;++j) {
    for (k=k1;k<=k2;++k) {
      n = Bucket[j][k].num;
      Bucket[j][k].element[n] = v;
      Bucket[j][k].num = n + 1;
      if (Bucket[j][k].num > BUCKET_CAPACITY) {
	printf("Bucket[%d][%d].num = %d > %d \n", 
		j,k, Bucket[j][k].num, BUCKET_CAPACITY);
      }
      assert(Bucket[j][k].num <= BUCKET_CAPACITY);
    }
  }
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void check_Bucket() {
int i,j,k,s1,s2,n_bucket,max_size,min_size;
double t1,t2;

  s1=s2=0;
  max_size = min_size = Bucket[0][0].num;
  for (i=0;i<N_Index;++i) {
    for (j=0;j<N_Index;++j) {
      k = Bucket[i][j].num;
      s1 += k;
      max_size = tMAX(max_size, k); 
      min_size = tMIN(min_size, k); 
    }
  }
  unsigned npoints = gBoundedSkewTree->Npoints() ;
  for (unsigned i=0;i<npoints;++i) {
    if (!Marked[i]     )
      s2++;
  }

  n_bucket = N_Index*N_Index; 
  t1 = (double) n_bucket; 
  t2 = (double) s1/ (double) s2; 
  printf("Nodes/Bucket:ave:%f(max:%d,min:%d) --> Redundancy:%f \n", 
	 s1/t1, max_size, min_size, t2); 
  printf("n_Bucket:%d    n_unmarked_CandiRoot:%d \n", n_bucket, s2); 
  if (s1 < s2)  {
    printf("Buckets contains %d elements \n", s1);
    printf("There are %d elements \n", s2);
    assert(s1>=s2);
  }
}
/****************************************************************************/
/*  partition n merging Nodes in buckets                                    */
/****************************************************************************/
void bucket_partitioning(int n) {
int i, j;

  gBoundedSkewTree->BuildTrr(n);
  for (i=0;i<N_Index;++i) { for (j=0;j<N_Index;++j) { Bucket[i][j].num = 0; } }

  for (i=0;i<n;++i) {
    if (!Marked[i]     ) {
      bucket_partitioning_sub(i);
    }
  }
  if (CHECK==1) check_Bucket();
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void do_compare_neighbors(int i,int j) {
int k1, k2; 

  k1 = Node[i].root_id;
  k2 = Node[j].root_id;
  if (k1 != k2)  {
    compare_neighbors(i,j);
  }
}

/****************************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void compare_neighbors_in_bucket(int v,int x,int y) {
int j;

  assert(v!=NIL);
  assert(x>=0 && y>=0 && x < N_Index && y < N_Index);
  for (j=0;j<Bucket[x][y].num;++j) {
    do_compare_neighbors(v,Bucket[x][y].element[j]);
  }
}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void calc_nearest_neighbor_sub(int v,int out_xlow,int out_xhi,int out_ylow,
    int out_yhi) {
int  n1, n2,j;

  n1 = tMAX(0,out_ylow);
  n2 = tMIN(N_Index-1,out_yhi);
  for (j=n1;j<=n2;++j) {
    if (out_xlow>=0)       compare_neighbors_in_bucket(v,out_xlow,j);
    if (out_xhi < N_Index) compare_neighbors_in_bucket(v,out_xhi,j);
  }
  n1 = tMAX(0,out_xlow+1);
  n2 = tMIN(N_Index-1,out_xhi-1);
  for (j=n1;j<=n2;++j) {
    if (out_ylow>=0) {compare_neighbors_in_bucket(v,j, out_ylow); }
    if (out_yhi < N_Index) compare_neighbors_in_bucket(v,j, out_yhi);
  }
}
/****************************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/*   calculate the nearest neighbor of merging region  mr(v).               */
/****************************************************************************/
int calc_nearest_neighbor(int v,int inc) {
int  x1,x2,y1,y2;

  x1 = xlow_index(&(CandiRoot[v])) - inc;
  x2 = xhi_index(&(CandiRoot[v]))  + inc;
  y1 = ylow_index(&(CandiRoot[v])) - inc;
  y2 = yhi_index(&(CandiRoot[v]))  + inc;

  if (x1>=0 || y1>=0 || x2 <N_Index || y2 < N_Index) {
    calc_nearest_neighbor_sub(v,x1,x2,y1,y2);
    return(YES);
  } else {
    return(NO);
  }
  /* assert(x1>=0 || y1>=0 || x2 <N_Index || y2 < N_Index); */
}
/****************************************************************************/
/*                                                                          */  
/****************************************************************************/
void init_nearest_neighbor(int v) {
int x1,x2,y1,y2,i,j;

  PointType &pt = Node[v].m_stnPt ;
         pt.t=0.0;   /* number of comparisons */ 
  N_neighbors[v]= 0;

  x1 = xlow_index(&(CandiRoot[v]));
  x2 = xhi_index(&(CandiRoot[v]));
  y1 = ylow_index(&(CandiRoot[v]));
  y2 = yhi_index(&(CandiRoot[v]));
  for (i=x1;i<=x2;++i) {
    for (j=y1;j<=y2;++j) {
      compare_neighbors_in_bucket(v, i,j);
    }
  }
}

/****************************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
int construct_NNG(int n) {
int i,j, inc, k1;
int graphSize = 0; // control search range for nearest neighbors

  assert(n>1);
  inc=0;
  do {
    ++inc;
    for (i=k1=0;i<n;++i) {
      j = UnMarkedNodes[i];
      calc_nearest_neighbor(j, inc);
      if (N_neighbors[j]>0) { k1++;}
    }
    if (CHECK) assert(inc <= N_Index);
  } while (k1  == 0 || inc < graphSize);
  assert(k1>0);
  return(k1);
}
/****************************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void show_NNG_infomation(int n, int k1 ) {
int i, j,k2,k3;
double t;

  for (i=0,t=0,k2=k3=0;i<n;++i) {
    j = UnMarkedNodes[i];
    if (N_neighbors[j]>0) {
      k2 += N_neighbors[j];
      k3 = tMAX(k3, N_neighbors[j]);
    }
    PointType &pt = Node[j].m_stnPt ;
    t +=        pt.t;
  }
  assert(t > 0);    
  printf("comparisons/nodes = %d/%d = %.1f \n", (int) t,n,  t/n);
  printf("nodes_with_neighbor/nodes = %d/%d = %.2f\n", k1,n,(double)k1/n);
  printf("neighbors/nodes = %d/%d = %.2f\n", k2,n,(double)k2/n);
  printf("max_neighbors = %d \n", k3);
  iShowTime();
  printf("\n\n");
  fflush(stdout);
}
/****************************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void build_nearest_neighbor_graph(int n, int show_info) {
int i, j,k1;

  bucket_partitioning(n);
  for (i=0;i<n;++i) {
    if (!Marked[i]     ) {
      init_nearest_neighbor(i);
    }
  }

  for (i=j=0;i<n;++i) {
    if (!Marked[i] ) { 
      UnMarkedNodes[j++] = i; 
    }
  }

  k1 = construct_NNG(j);

  if (show_info) {
    show_NNG_infomation(j, k1);
  }
}
/****************************************************************************/
/*   select the cloest pair of merging ares.                               */
/****************************************************************************/
int pair_compare_inc_sub1(PairType  *p, PairType  *q) {
  return( (p->cost > q->cost) ? YES: NO);
}

/****************************************************************************/
/* merging cost increase of cluster i  ==  Cluster[i].y - Cluster[i].x */
/****************************************************************************/
double cls_merging_cost_inc(int cid) {
  return(Cluster[cid].y - Cluster[cid].x);
}
/****************************************************************************/
/*   select the cloest pair of merging ares.                               */
/****************************************************************************/
int pair_compare_inc_sub2(PairType  *p, PairType  *q) {
int i,j;
double c1, c2;


  i = Node[p->x].root_id;
  j = Node[p->y].root_id;
  c1 = cls_merging_cost_inc(i) + cls_merging_cost_inc(j);
  i = Node[q->x].root_id;
  j = Node[q->y].root_id;
  c2 = cls_merging_cost_inc(i) + cls_merging_cost_inc(j);
  if ( equal(c1,c2) ) {
    return( (p->cost > q->cost) ? YES: NO);
  } else {
    return( (c1 < c2) ? YES: NO);
  }

}

/****************************************************************************/
/*   select the cloest pair of merging ares.                               */
/****************************************************************************/
int pair_compare_inc(const void  *p1, const void  *q1) {
PairType  *p, *q;
int ans;

  p = (PairType  *)p1;
  q = (PairType  *)q1;

  if (Cost_Function==0) {
    ans = pair_compare_inc_sub1(p,q);
  } else {
    ans = pair_compare_inc_sub2(p,q);
  }
  return(ans);
}

/****************************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
int Area_compare_inc(const void  *p1, const void  *q1) {
AreaType  *p, *q;
double a1, a2;

  p = (AreaType  *) p1;
  q = (AreaType  *) q1;
  a1 = calc_boundary_length(p);
  a2 = calc_boundary_length(q);
  if (equal(p->subtree_cost, q->subtree_cost)) {
    return( (a1 < a2) ? YES: NO);
  } else {
    return( (p->subtree_cost > q->subtree_cost) ? YES: NO);
  }
  /*
  if (equal(p->dist, q->dist)) {
    return( (a1 < a2) ? YES: NO);
  } else {
    return( (p->dist > q->dist) ? YES: NO);
  }
  */
}


/****************************************************************************/
/*   initialize the merging pairs                                           */
/****************************************************************************/
int count_merge_pairs(int n_nodes) { 
int i, j, k, n;

  for (i=0, n=0; i<n_nodes; i++) {
    k = N_neighbors[i];
    if (!Marked[i]      && k >0 ) {
      for (j=0;j<k;++j) {
        Best_Pair[n].x = i;
        Best_Pair[n].y=The_NEIghors[i][j];
        Best_Pair[n].cost=Neighbor_Cost[i][j];
        n++;
      }
    }
  }
  assert(n>1);
  return(n);
}
/****************************************************************************/
/*   initialize the merging pairs                                           */
/****************************************************************************/
static
void update_Cluster(int n_nodes) {
int i,j;
 
  /* init min merging cost in current Nearest Neighbor Graph */

  for (i=0;i<n_nodes;++i) {
    j = Node[i].root_id;
    if ( j >= 0 ) {
      Cluster[j].y = DBL_MAX;
    } else { /* root_id can be -1 */
      // printf  ("error: j=%d\n", j ) ;
    }
  }
  for (i=0;i<n_nodes;++i) {
    if (N_neighbors[i]>0) {
      j = Node[i].root_id;
      Cluster[j].x = tMIN(Cluster[j].x, Neighbor_Cost[i][0]);
      Cluster[j].y = tMIN(Cluster[j].y, Neighbor_Cost[i][0]);
      assert(Cluster[i].y >= Cluster[i].x);
    }
  }
  for (i=0;i<n_nodes;++i) {
    j = Node[i].root_id;
  }
}
/****************************************************************************/
/*   initialize the merging pairs                                           */
/****************************************************************************/
int init_pairs_to_merge(int n_nodes) { 
int i, n;

  n = count_merge_pairs(n_nodes);
  if (Cost_Function ==1) update_Cluster(n_nodes);

  qsort(Best_Pair, n, sizeof(PairType), pair_compare_inc);
  if (k_Parameter < 0) {
    i = n/ABS(k_Parameter);
  } else {
    i = k_Parameter;
  }
  n = tMIN(n, tMAX(1,i) ); 

  return(n);
}
/****************************************************************************/
/*   select merging pairs from Nearest-Neighbor graph                       */
/****************************************************************************/
int pairs_to_merge(int n_nodes) { 
int i, j, n, besta1, besta2, i1, i2;
char    *Flag;

  n = init_pairs_to_merge(n_nodes); 
  unsigned npoints = gBoundedSkewTree->Npoints() ;
  Flag = (char *) calloc(npoints, sizeof(char));
  assert(Flag != NULL);

  for (i=0; i<n_nodes; i++) { Flag[i] = NO; }

  for (i=0, j=0; i<n; i++) {
    besta1 = Best_Pair[i].x;
    besta2 = Best_Pair[i].y;
    assert(besta1 >=0 && besta2>=0);
    i1 = Node[besta1].root_id;
    i2 = Node[besta2].root_id;
    if (!Flag[i1] && !Flag[i2]) {
      Flag[i1] = Flag[i2] = YES;
      Best_Pair[j].x=besta1;
      Best_Pair[j].y=besta2;
      j++;
      /* Best_Pair[j++]=Best_Pair[i];  */
    }
  }
  assert(j>0); 
  assert(n_nodes+j <= (int) npoints); 
  free(Flag);
  return(j);
}

/***********************************************************************/
/***********************************************************************/
double
BstTree::TotalLength ( void ) const {
double Tcost = 0;
double Tdist=0 ;
  
  int root = gBoundedSkewTree->RootNodeIndex () ;
  calc_TreeCost( root, &Tcost, &Tdist);
  return Tcost ;
}

/***********************************************************************/
/***********************************************************************/
double 
BST_DME::TotalLength ( void ) const {
  return m_bstdme->TotalLength() ;
}
/***********************************************************************/
/*  print the TreeCost and cost reduction over MST.                     */
/***********************************************************************/
static double 
print_answer(const char fn[],int v, bool fixedTopology ) {
PointType p;
double Tcost = 0, Tdist=0, t, cap; 
int n_detour_nodes = calc_TreeCost(v, &Tcost, &Tdist); 
double sCost = Node[v].area[Node[v].ca].subtree_cost ;

  print_cluster_cost(Tcost);
  assert( equivalent(Tcost, sCost, 1.0));
  printf("\n%s:WL=%10.0f B=", fn, Tcost);
  if (Skew_B == DBL_MAX) { printf("inf ");} else {printf("%.0f", Skew_B);}


  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( !linear ) {
    printf("ps "); }
  t = calc_buffered_node_delay(v);
  cap = Node[v].area[Node[v].ca].unbuf_capac;
  PointType &pt = Node[v].m_stnPt ;
  printf("t:%.1f (%.1f)(%.3f)", t,        pt.max, cap);
  iShowTime();
  if (Start_Skew_B == DBL_MAX) {printf("(infty)");
  } else {printf("(%.0f)", Start_Skew_B);}
  t = 1E15/PUCAP_SCALE;
  printf("\n\n\n");
  printf("Total wirelength: %f\n", Tcost);
  printf("Total Delay: %f ps\n",calc_buffered_node_delay(v));
  if (R_buffer>0) {
    printf("R_buffer =  %f Oh\n", (double) R_buffer);
  } else {
    printf("%d types of Buffers with R = [%d,%d]\n",N_BUFFER_SIZE,
            R_buffer_size[0], R_buffer_size[N_BUFFER_SIZE-1]);
  }
  printf("C_buffer =  %f FF\n", C_buffer*t);
  printf("Delay_buffer =  %f ns\n", Delay_buffer*1E9/PUCAP_SCALE);
  printf("R[H], R[V]: %f Oh  %f Oh\n", PURES[H_], PURES[V_]);
  printf("C[H], C[V]: %f FF  %f FF \n", PUCAP[H_]*t, PUCAP[V_]*t);
  printf("Routing pattern: %.1f (1:HV, 0:VH) \n", Gamma);
  printf("\n");
  
  printf("k=%d, Cost_fx=%.1f,N_Neighbor=%d \n", 
          k_Parameter, Cost_Function, N_Neighbor);

  unsigned nterms = gBoundedSkewTree->Nterms() ;
  if (Tdist != Tcost) {
    t = (Tcost- Tdist)*100.0/Tdist;
    printf("Treewire = %.1f (detour:%.1f%%) ***", Tdist,t); 
    t = (double) (n_detour_nodes)*100.0/nterms;
    printf("n_detour_nodes = %d (%f%%) ***\n", n_detour_nodes,t); 
  }
  fflush(stdout);

  p =        pt;
  printf("skew(%d)=%f=(%f-%f) \n", v,pt_skew(p),p.max,p.min);
  assert(pt_skew(p) <= Skew_B + FUZZ);
  print_BST_Mode();
  printf("\n\n"); 

  printf("Max_n_regions (irredundant) = %d (%d)\n", Max_n_regions, 
                                Max_irredundant_regions);
  print_overlapped_regions();
  print_max_n_mr();
  print_max_npts();
  print_n_region_type();



  if (BST_Mode==IME_MODE) {
    unsigned npoints = gBoundedSkewTree->Npoints() ;
    for (unsigned i=nterms+1;i<npoints;++i) {
      assert(JS_line_type(&(Node[i])) == MANHATTAN_ARC);
    }
  }
  if (!fixedTopology) {
    print_topology(fn, v, Tcost, Tdist); 
  }
    print_bst (fn, v, Tcost, Tdist);
    print_merging_tree(fn, v); 

  return(Tcost);

}
/******************************************************************/
/*  Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao     */
/******************************************************************/
void check_update_JS(AreaType *area, PointType pt[4], int line0type) {
int parallel;
double d;
PointType tmp_pt[2],q0,q1,q2,q3;


  parallel = parallel_line(pt[0],pt[1],pt[2],pt[3]);

  if (parallel) {
    if (line0type == FLAT) { /* not consider non-rectilinear  parallel lines */
      assert(0);                                     
    } else if (line0type == TILT) {
      assert(0);                                     
    }
  }

  q0 = JS[0][0];
  q1 = JS[0][1];
  q2 = JS[1][0];
  q3 = JS[1][1];

  d=linedist(q0,q1,q2,q3,tmp_pt);
  if (CHECK==1) {
    assert(same_Point(q0,q1) || same_Point(q2,q3) || 
           parallel_line(q0,q1,q2,q3));
    assert(equal(d,area->dist)) ;
    assert(pt_on_line_segment(q0,pt[0],pt[1]));
    assert(pt_on_line_segment(q1,pt[0],pt[1]));
    assert(pt_on_line_segment(q2,pt[2],pt[3]));
    assert(pt_on_line_segment(q3,pt[2],pt[3]));
  }

}
/*****************************************************************/
/*                                                                */
/******************************************************************/
void calc_coor(PointType *pt,PointType p0,PointType p1,double z, int linetype) {
double d0,d1;
  
  
  if (linetype==HORIZONTAL || linetype==FLAT) {
    d0 = ABS(p0.x - z);
    d1 = ABS(p1.x - z);
  } else {
    d0 = ABS(p0.y - z);
    d1 = ABS(p1.y - z);
  }
  pt->x = (p0.x*d1+p1.x*d0)/(d0+d1);
  pt->y = (p0.y*d1+p1.y*d0)/(d0+d1);
}


/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void update_JS_case1(AreaType *area, PointType pt[4], PointType pts[2],
                     int line0type, int line1type) {
TrrType ms0,ms1,trr0,trr1,t0,t1;

  if (line0type==MANHATTAN_ARC ) { 
    line2ms(&ms0,pt[0],pt[1]); 
  } else { 
    ms0.MakeDiamond   (pts[0],0   ); 
  }
  if (line1type==MANHATTAN_ARC ) { 
    line2ms(&ms1,pt[2],pt[3]); 
  } else { 
    ms1.MakeDiamond   (pts[1],0   ); 
  } 
  assert(equal(ms_distance(&ms0,&ms1), area->dist)); 
  L_MS = ms0;
  R_MS = ms1;
  build_trr(&ms0, area->dist, &trr0);
  build_trr(&ms1, area->dist, &trr1);
  make_intersect(&trr1,&ms0,&t0);
  make_intersect(&trr0,&ms1,&t1);
  ms2line(&t0, &(JS[0][0]), &(JS[0][1]) );
  ms2line(&t1, &(JS[1][0]), &(JS[1][1]) );
}

/***************************************************************************/
/* return the Joining Segments of node_L and node_R  for node              */
/* when the closest portion of Node[L] and Node[R] are parallel segments   */
/***************************************************************************/
void update_JS(AreaType *area, PointType pt[4], PointType pts[2]) {
int line0type, line1type;
double max_x,max_y,min_x,min_y, t; 
TrrType ms0,ms1,trr0,trr1,t0,t1;

  line0type = calc_line_type(pt[0],pt[1]);
  line1type = calc_line_type(pt[2],pt[3]);

  if (line0type==MANHATTAN_ARC ) { line2ms(&ms0,pt[0],pt[1]); }
  if (line1type==MANHATTAN_ARC ) { line2ms(&ms1,pt[2],pt[3]); }
  if (line0type!=MANHATTAN_ARC && line1type==MANHATTAN_ARC ) {
    ms0.MakeDiamond   (pts[0],0   );
  }
  if (line0type==MANHATTAN_ARC && line1type!=MANHATTAN_ARC ) {
    ms1.MakeDiamond   (pts[1],0   );
  } 
    

  JS[0][0] = JS[0][1] = pts[0];
  JS[1][0] = JS[1][1] = pts[1];
  if ( (line0type==MANHATTAN_ARC || line1type==MANHATTAN_ARC) ) {
    t = ms_distance(&ms0,&ms1);
    assert(equivalent(t, area->dist, 1E-6)); 
    area->dist = t;
    L_MS = ms0;
    R_MS = ms1;
    build_trr(&ms0, area->dist, &trr0);
    build_trr(&ms1, area->dist, &trr1);
    make_intersect(&trr1,&ms0,&t0);
    make_intersect(&trr0,&ms1,&t1);
    ms2line(&t0, &(JS[0][0]), &(JS[0][1]) );
    ms2line(&t1, &(JS[1][0]), &(JS[1][1]) );
  } else if ( parallel_line(pt[0],pt[1],pt[2],pt[3])) { /* parallel  lines */ 
    min_y = tMAX( tMIN(pt[0].y,pt[1].y),  tMIN(pt[2].y,pt[3].y) ); 
    max_y = tMIN( tMAX(pt[0].y,pt[1].y),  tMAX(pt[2].y,pt[3].y) ); 
    min_x = tMAX( tMIN(pt[0].x,pt[1].x),  tMIN(pt[2].x,pt[3].x) ); 
    max_x = tMIN( tMAX(pt[0].x,pt[1].x),  tMAX(pt[2].x,pt[3].x) ); 
    if ( (line0type==VERTICAL || line0type==TILT) && max_y >= min_y) {
      calc_coor(&(JS[0][0]),pt[0],pt[1],min_y,line0type);
      calc_coor(&(JS[0][1]),pt[0],pt[1],max_y,line0type);
      calc_coor(&(JS[1][0]),pt[2],pt[3],min_y,line0type);
      calc_coor(&(JS[1][1]),pt[2],pt[3],max_y,line0type);
    } else if ( (line0type==HORIZONTAL || line0type==FLAT)  && max_x >=min_x) {
      calc_coor(&(JS[0][0]),pt[0],pt[1],min_x,line0type);
      calc_coor(&(JS[0][1]),pt[0],pt[1],max_x,line0type);
      calc_coor(&(JS[1][0]),pt[2],pt[3],min_x,line0type);
      calc_coor(&(JS[1][1]),pt[2],pt[3],max_x,line0type);
    }
  } else {
  }
  if (calc_line_type(JS[0][0],JS[0][1]) == MANHATTAN_ARC 
      && line0type != MANHATTAN_ARC && line1type != MANHATTAN_ARC) {
    L_MS.MakeDiamond   (pts[0],  0   );
    R_MS.MakeDiamond   (pts[1],  0   );
  }
  if (CHECK==1) check_update_JS(area,pt,line0type);
}

/****************************************************************************/
/* calculate the joining segment for node's children                    */
/****************************************************************************/
void calc_JS_sub2(AreaType *area, PointType pt[4]) {
double ldist, old_dist, old_area, new_area;
PointType pts[2],  old_JS[2][2]; 
TrrType old_L_MS,  old_R_MS; 



  ldist = linedist(pt[0],pt[1],pt[2],pt[3], pts);
  if (equal(ldist,area->dist)) {
    old_JS[0][0] = JS[0][0];
    old_JS[0][1] = JS[0][1];
    old_JS[1][0] = JS[1][0];
    old_JS[1][1] = JS[1][1];
    old_dist = area->dist;
    if (calc_line_type(JS[0][0],JS[0][1]) == MANHATTAN_ARC) {
      old_L_MS = L_MS;
      old_R_MS = R_MS;
    }
    area->dist = ldist;
    update_JS(area, pt, pts);
    old_area = calc_JR_area_sub(old_JS[0][0],old_JS[0][1],
                             old_JS[1][0],old_JS[1][1]);
    new_area = calc_JR_area_sub(JS[0][0], JS[0][1], JS[1][0], JS[1][1]);
    if (old_area >= new_area ) {  /* original JS is better */
      JS[0][0] = old_JS[0][0];
      JS[0][1] = old_JS[0][1];
      JS[1][0] = old_JS[1][0];
      JS[1][1] = old_JS[1][1];
      if (calc_line_type(JS[0][0],JS[0][1]) == MANHATTAN_ARC) {
        L_MS = old_L_MS;
        R_MS = old_R_MS;
      }
      area->dist = old_dist;
    }
  } else if (ldist < area->dist )  {
    area->dist = ldist;
    update_JS(area, pt, pts);
  }
  if (calc_line_type(JS[0][0],JS[0][1]) == MANHATTAN_ARC ) {
    check_JS_MS();
  }
}
/****************************************************************************/
/* calculate the joining segment for node's children                    */
/****************************************************************************/
void calc_JS_sub1(AreaType *area, AreaType *area_L,AreaType *area_R) {
int n1, n2;
int i,j,mn1,mn2, k0, k1, k2, k3; 
PointType pt[4], q[2];

  mn1 = n1 = area_L->npts;
  mn2 = n2 = area_R->npts;
  if (n1==2) n1 =1; 
  if (n2==2) n2 =1; 
  for (i=0; i<n1; i++) {
    k0 = area_L->vertex[i];
    k1 = area_L->vertex[(i+1)%mn1]; 
    pt[0] = area_L->mr[k0];
    pt[1] = area_L->mr[k1];
    for (j=0; j<n2; j++) {
      k2 = area_R->vertex[j];
      k3 = area_R->vertex[(j+1)%mn2];
      pt[2] = area_R->mr[k2];
      pt[3] = area_R->mr[k3];              
      linedist(pt[0],pt[1],pt[2],pt[3], q);
      calc_JS_sub2(area, pt); 
    }
  }
  calc_JS_delay(area, area_L,area_R);
}

/****************************************************************************/
/* calculate the joining segment for node's children                    */
/****************************************************************************/
void do_calc_JS(AreaType *area, AreaType *area_L,AreaType *area_R, int v1, int v2) {
int n1,n2;
int j0,j1,j2,j3;  
PointType pt[4];

  area->dist = DBL_MAX;
  n1 = area_L->npts;
  n2 = area_R->npts;
  if (n1 ==1  && n2 == 1) {
    JS[0][0] = JS[0][1] = area_L->mr[0];  
    JS[1][0] = JS[1][1] = area_R->mr[0];  
    L_MS.MakeDiamond   (area_L->mr[0], 0    );
    R_MS.MakeDiamond   (area_R->mr[0], 0    );
    area->dist = Point_dist(area_L->mr[0], area_R->mr[0]) ; 
  } else {
    if (v1<0 && v2 <0) {
      calc_JS_sub1(area, area_L,area_R); 
    } else {
      j0 = area_L->vertex[v1]; 
      j1 = area_L->vertex[(v1+1)%n1];
      j2 = area_R->vertex[v2];
      j3 = area_R->vertex[(v2+1)%n2];
      pt[0] = area_L->mr[j0]; 
      pt[1] = area_L->mr[j1]; 
      pt[2] = area_R->mr[j2];   
      pt[3] = area_R->mr[j3];           
      calc_JS_sub2(area, pt); 
    }
    calc_JS_delay(area, area_L,area_R);
  }
  if (calc_line_type(JS[0][0],JS[0][1]) == MANHATTAN_ARC) {
    check_JS_MS();
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void recalculate_MS(AreaType *area, TrrType *trr) {
  if (area->n_mr ==4 && TRR_area(area)) {
    pts2TRR(area->mr,area->n_mr, trr);
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void debug_calc_JS(AreaType *area) {
TrrType t;

  make_intersect_sub( &L_MS, &R_MS, &t);
  if (t.xlow <= t.xhi && t.ylow <= t.yhi) {
    draw_a_TRR(&L_MS);
    draw_a_TRR(&R_MS);
    print_area(area->area_L);
    print_area(area->area_R);
    printf("here for debug \n");
    area->dist = 0;
  }
}
/****************************************************************************/
/* calculate the joining segment for node's children                    */
/****************************************************************************/
void calc_JS(AreaType *area,AreaType *area_L,AreaType *area_R,int v1,int v2) {

  do_calc_JS(area,area_L,area_R,v1,v2);

  if (Skew_B ==0 && TmpCondition) {
    recalculate_MS(area_L, &L_MS);
    recalculate_MS(area_R, &R_MS);
    if (0) debug_calc_JS(area);
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void check_sampling_segment(AreaType *area,PointType *p0, PointType *p1) {
PointType  v0, v1;

  calc_BS_located(p1,area,&v0,&v1);
  calc_pt_delays(area, p1, v0,v1);
  check_Point(p0);
  check_Point(p1);
  check_const_delays(p0,p1);
}
/********************************************************************/
/* calculate the max-delay/min-delay of the sampleing segemnt */
/********************************************************************/
void calc_delay_of_ms(AreaType *area, PointType *p0, PointType *p1) {
PointType  v0, v1;

  calc_BS_located(p0,area,&v0,&v1);
  calc_pt_delays(area, p0, v0,v1);
  p1->max = p0->max;
  p1->min = p0->min;

  if (CHECK==1) check_sampling_segment(area,p0, p1); 

}
/********************************************************************/
/*  get the vertex of mr which is closest to a given point.         */
/********************************************************************/
int nearest_vertex(AreaType *area, PointType lpt) {
int i, min_i;
double d, min_d;

  min_d = DBL_MAX; 
  for (i=0;i<area->n_mr;++i) {
    d = Point_dist(lpt,area->mr[i]);
    if (d < min_d) {
      min_i = i;
      min_d = d;
    }
  }
  return(min_i);
}
/********************************************************************/
/*  get the Boundary Segments which are Manhattan Arcs.              */
/********************************************************************/
void get_Manhattan_BS(AreaType *area, int side, TrrType *ms) {
int v1, v2, k1, k2, n;
int linetype1, linetype2;


  v1 = v2 = nearest_vertex(area, area->line[side][0]);

  n = area->n_mr;
  k1 = (v1+1)%n;
  k2 = (v1+n-1)%n;
  if (Manhattan_arc(area->mr[v1], area->mr[k1])) {
    v2 = k1;
  } else if (Manhattan_arc(area->mr[v1], area->mr[k2])) {
    v2 = k2;
  }

  if (v1==v2 && area->n_mr >= 2 ) {
    linetype1 = calc_line_type(area->mr[v1], area->mr[k1]);
    linetype2 = calc_line_type(area->mr[v1], area->mr[k2]);
    assert(linetype1==VERTICAL || linetype1==HORIZONTAL);
    assert(linetype2==VERTICAL || linetype2==HORIZONTAL);
  }

  line2ms(ms,area->mr[v1], area->mr[v2]);
}

/********************************************************************/
/*  calculate the i_th sampling Manhattan arc of node              */
/********************************************************************/
void get_a_sampling_segment(int i, TrrType ms[], double d) { 
double d0,d1;
TrrType trr0, trr1;

  assert(i<N_Sampling-1 );
  assert(i > 0 );

  d0 = i*d/(N_Sampling-1);
  d1 = d - d0;

  build_trr(&(ms[0]),d0,&trr0);
  build_trr(&(ms[N_Sampling-1]),d1,&trr1);
  make_intersect(&trr0,&trr1, &(ms[i]));
}
/********************************************************************/
/*  calculate the i_th sampling Manhattan arc of node              */
/********************************************************************/
int calc_sampling_set(AreaType *area, TrrType *sampling_set) {
int i, n;
double d;

  if (!area_Manhattan_arc_JS(area)) {
    n=0;
  } else if (area->n_mr == 1) {
    n = 1;
    line2ms(&(sampling_set[0]),area->mr[0], area->mr[0]);
  } else if (area->n_mr == 2 && Manhattan_arc(area->mr[0], area->mr[1]) ) {
    n = 1;
    line2ms(&(sampling_set[0]),area->mr[0], area->mr[1]);
  } else {
    n = N_Sampling;
    get_Manhattan_BS(area, 0, &(sampling_set[0]));
    get_Manhattan_BS(area, 1, &(sampling_set[n-1]));
    d = ms_distance( &(sampling_set[0]), &(sampling_set[n-1]));
    assert( d > FUZZ && area->dist >= d - FUZZ);
    for (i=1;i<N_Sampling-1;++i) {
      get_a_sampling_segment(i, sampling_set, d);
      check_a_sampling_segment(area, &(sampling_set[i]));
    }
  }
  return(n);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void ms2line_delays(AreaType *area, TrrType *ms, PointType *p0,
      PointType *p1) {

  if (area->n_mr==1) {
    *p0=*p1 = area->mr[0];
  } else if (area->n_mr == 2 && Manhattan_arc(area->mr[0], area->mr[1]) ) {
    *p0 = area->mr[0];
    *p1 = area->mr[1];
  } else {
    ms2line(ms, p0, p1); 
    calc_delay_of_ms(area, p0, p1);       
  }
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void MergeArea_sub(AreaType  *area) {
double merge_cost, x, t ;

  merge_cost = area_merge_cost(area);

  area->subtree_cost = merge_cost + area->area_L->subtree_cost + 
                       area->area_R->subtree_cost;
  bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
  if ( linear ) {
    area->capac =  area->subtree_cost; 
  } else {
    x = merge_cost - area->L_StubLen - area->R_StubLen;
    area->capac =  area->area_L->capac + area->area_R->capac + x*PUCAP[H_];
/*
    h = ABS(JS[0][0].x - JS[1][0].x);
    v = x - h;
    area->capac = area_L->capac + area_R->capac+h*PUCAP[H_] + v*PUCAP[V_];
*/
  }
  t = area->L_EdgeLen + area->R_EdgeLen;
  assert(t  >= area->dist - FUZZ || t==-2.0);

  area->R_buffer = 0;
  area->unbuf_capac = area->capac;
}
/****************************************************************************/
/*            return the merging area of Node1 and Node2                    */
/* Case I: Construction of merging area = Construction of merging segments  */
/****************************************************************************/
void MergeManhattanArc(AreaType  *area, AreaType  *area_L, AreaType  *area_R,
                       int mode) {
double d0,d1;

  area->dist = ms_distance(&L_MS, &R_MS);
  
  
    assert(area->L_StubLen == 0);
    assert(area->R_StubLen == 0);
    calc_area_EdgeLen(area, &d0, &d1);
  
  assert(d0>=0);
  assert(d1>=0);

  calc_merge_pt_delay(area, d0, d1);
  if (area_L->R_buffer > 0  && area_R->R_buffer > 0  && 
      area->L_EdgeLen + area->R_EdgeLen > area->dist + FUZZ &&
      area->L_StubLen + area->R_StubLen < area->dist) {
    assert(equal(d0,0) || equal(d1,0));
  }

  if (mode==FAST) return;
                       
  construct_TRR_mr(area);

}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void update_JS_case_ms(double dist) {
TrrType ms0,ms1,t0,t1, trr0,trr1;

  assert(calc_line_type(JS[0][0],JS[0][1]) == MANHATTAN_ARC);

  line2ms(&ms0,JS[0][0],JS[0][1]);
  line2ms(&ms1,JS[1][0],JS[1][1]);
  assert(equal(ms_distance(&ms0,&ms1), dist));

  build_trr(&ms0, dist, &trr0);
  build_trr(&ms1, dist, &trr1);
  make_intersect(&trr1,&ms0,&t0);
  make_intersect(&trr0,&ms1,&t1);

  ms2line(&t0, &(JS[0][0]), &(JS[0][1]) );
  ms2line(&t1, &(JS[1][0]), &(JS[1][1]) );
  check_JS_MS();
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int redundant_area(AreaType tmparea[], int n, double newcap, double newskew ) {
int i;
double cap, skew;

  for (i=0;i<n;++i) {
    cap = tmparea[i].capac;
    skew = area_minskew(&(tmparea[i])); 
    if (equal(newcap, cap) && equal(newskew, skew)) {
      return(YES);
    } else if (newcap >= cap - FUZZ && newskew >= skew - FUZZ ) {
      return(YES);
    }
  }
  return(NO);
}

/*****************************************************************************/
/* uses the closest pair of boundary segments of children's regions */
/* for constructing new merging regions.                                  */
/****************************************************************************/
void IME_mergeNode_BS2(AreaType *area_L, AreaType *area_R,int *n,int i,int j,
     int b1, int b2) {

  TempArea[*n].L_area = i;
  TempArea[*n].R_area = j;

  if (b1) {assert(area_L->R_buffer > 0);} else {area_L->R_buffer = 0;}
  if (b2) {assert(area_R->R_buffer > 0);} else {area_R->R_buffer = 0;}
  calc_JS(&(TempArea[*n]), area_L, area_R, -1,-1);
  MergeArea(&(TempArea[*n]), area_L, area_R, NORMAL);
  (*n)++;
  /* 
  if (*n>1) tsao_Irredundant(TempArea, n);
  if (*n >10) kohck_Irredundant(TempArea, n);
  */
}
/*****************************************************************************/
/*                                                                          */
/****************************************************************************/
void IME_mergeNode_sub4(AreaType *a1, AreaType *area_R, int *n_tmparea, 
    int k1, int k2,  int b1, int b2) {
int i;
AreaType a2;

  a2 = *area_R;
  a2.vertex[0] = 0; 
  a2.vertex[1] = 1; 
  for (i=0;i<n_R_sampling;++i) {
    R_MS = R_sampling[i];
    ms2line_delays(area_R,&R_MS,&(a2.mr[0]),&(a2.mr[1]));
    if (same_Point(a2.mr[0],a2.mr[1])) { a2.n_mr=1;
    } else { a2.n_mr=2; }
    a2.npts = a2.n_mr;
    IME_mergeNode_BS2(a1, &a2,n_tmparea, k1, k2, b1, b2);
  }
  if (n_R_sampling!=1 && BST_Mode == HYBRID_MODE) {
    IME_mergeNode_BS2(a1, area_R,n_tmparea, k1, k2, b1, b2);
  }
}
/*****************************************************************************/
/*                                                                          */
/****************************************************************************/
void IME_mergeNode_sub3(AreaType *area_L, AreaType *area_R, int *n_tmparea, 
     int k1, int k2, int b1, int b2) {
int i;
AreaType a1;

  a1 = *area_L;
  a1.vertex[0] = 0; 
  a1.vertex[1] = 1; 
  for (i=0;i<n_L_sampling;++i) {
    L_MS = L_sampling[i];
    ms2line_delays(area_L,&L_MS,&(a1.mr[0]),&(a1.mr[1]));
    if (same_Point(a1.mr[0],a1.mr[1])) { a1.n_mr=1; 
    } else { a1.n_mr=2; }
    a1.npts = a1.n_mr;
    IME_mergeNode_sub4(&a1, area_R, n_tmparea, k1, k2, b1, b2);
  }
  if (n_L_sampling!=1 && BST_Mode == HYBRID_MODE) {
    IME_mergeNode_sub4(area_L, area_R, n_tmparea, k1, k2, b1, b2);
  }
  assert(*n_tmparea>0);
}
/*****************************************************************************/
/* IME & HYBRID MODE uses both Boundary Segments(BS) & Internal Sampling */
/* Segments(IS)*/
/*****************************************************************************/
int IME_mergeNode_sub1(NodeType *node_L, NodeType *node_R, int b1, int b2) {
int i,j,n_tmparea=0;

  n_tmparea = 0;
  assert(node_L->n_area > 0);
  assert(node_R->n_area > 0);
  for (i=0;i<node_L->n_area;++i) {
    for (j=0;j<node_R->n_area;++j) {
      IME_mergeNode_BS2(&(node_L->area[i]), &(node_R->area[j]), &n_tmparea,
                        i, j, b1, b2);
      assert(n_tmparea>0);
    }
  }
/* kohck_Irredundant(TempArea, &n_tmparea); */

  return(n_tmparea);
}

/****************************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void check_sorted_areas(AreaType TempArea[], int n_tmparea) {
int i;

  for (i=0;i<n_tmparea-1;++i) {
    assert(TempArea[i].subtree_cost <= TempArea[i+1].subtree_cost + FUZZ ); 
    assert(TempArea[i].n_mr <= MAX_mr_PTS); 
  }
}
/****************************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void new_IME_mergeNode(NodeType *node, NodeType *node_L, NodeType *node_R, 
     int b1, int b2) {
int i, n_tmparea=0;

  n_tmparea= IME_mergeNode_sub1(node_L, node_R, b1, b2);
  assert(n_tmparea <= N_TempArea);
  Max_irredundant_regions =  tMAX(Max_irredundant_regions, n_tmparea);

  qsort(TempArea, n_tmparea, sizeof(AreaType), Area_compare_inc);
  n_tmparea =  tMIN(n_tmparea, N_Area_Per_Node);
  
  modify_blocked_areas(TempArea,&n_tmparea, b1, b2);
  assert(n_tmparea <= N_TempArea);
  Max_n_regions = tMAX(Max_n_regions, n_tmparea);

  qsort(TempArea, n_tmparea, sizeof(AreaType), Area_compare_inc);
  check_sorted_areas(TempArea,n_tmparea);
  node->n_area =  tMIN(n_tmparea, N_Area_Per_Node);
  for (i=0;i<node->n_area; ++i) {
    node->area[i] = TempArea[i];
  }
  node->ca = 0;
}
/****************************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/****************************************************************************/
void IME_mergeNode(NodeType *node, NodeType *node_L, NodeType *node_R, int b1, int b2) {
AreaSetType stair, result;
int i, n_areas; 

  stair.npoly =  n_areas = IME_mergeNode_sub1(node_L, node_R, b1, b2);

  assert(stair.npoly<= N_TempArea && stair.npoly> 0); 
  assert(Skew_B > 0 || stair.npoly==1); 

  /* select n best merging regions */

  
  result.npoly =  tMIN(stair.npoly, N_Area_Per_Node);
  stair.freg  = (AreaType *) calloc(stair.npoly,  sizeof(AreaType));
  result.freg = (AreaType *) calloc(N_Area_Per_Node,  sizeof(AreaType));
  assert(  stair.freg != NULL );
  assert(  result.freg != NULL );
  for (i=0;i<stair.npoly;++i) { 
    stair.freg[i] = TempArea[i]; 
  }

  Max_n_regions = tMAX(Max_n_regions, stair.npoly); 
  Irredundant(&stair);  
  Max_irredundant_regions = tMAX(Max_irredundant_regions, stair.npoly);
  if (Dynamic_Selection) {
    KStepStair(&stair, N_Area_Per_Node, &result);
    store_n_areas_IME(node, &result);
  } else { /* greedy selection */
/*
    store_last_n_areas_IME(node, &stair);
*/
    for (i=0;i<stair.npoly;++i) { TempArea[i] = stair.freg[i]; }
    qsort(TempArea, stair.npoly, sizeof(AreaType), Area_compare_inc);
    if (CHECK==1) check_tmparea(TempArea, stair.npoly); 
    store_n_areas(TempArea, result.npoly, node);
  }
  if (CHECK) print_IME_areas(node, node_L, node_R, stair.npoly, n_areas);

  node->ca = node->n_area-1;
  free(stair.freg);
  free(result.freg);
}

/****************************************************************************/
/*      mr(v) = the merging region between merging region mr(L) and mr(R)   */
/****************************************************************************/
void
NodeType::Merge2Nodes( NodeType *node_L,NodeType *node_R) {

  ca = 0;
  AreaType *area   = &(this->area[0]);
  AreaType *area_L = &(node_L->area[node_L->ca]);
  AreaType *area_R = &(node_R->area[node_R->ca]);

  if (BST_Mode == BME_MODE) {
    calc_JS(area, area_L, area_R,  -1,-1);
    MergeArea(area, area_L, area_R,  NORMAL);
  } else {   /* IME_Mode || HYBRID_MODE */
    new_IME_mergeNode(this, node_L, node_R, area_L->R_buffer, area_R->R_buffer);
  }
}



/****************************************************************************/
/*                                                                          */
/****************************************************************************/
double BME_merging_cost(NodeType *node_L, NodeType *node_R) {
AreaType area, *area_L, *area_R;

  area.n_mr = 0; /* XXXnate */


  area_L = &(node_L->area[node_L->ca]);
  area_R = &(node_R->area[node_R->ca]);

  calc_JS(&area, area_L,area_R, -1,-1);
  MergeArea(&area, area_L, area_R,  FAST); 
  return(area.subtree_cost);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
double IME_merging_cost(NodeType *node_L, NodeType *node_R) {
int i,j, n1,n2;
double min_cost, cost; 

  n1 = node_L->n_area;
  n2 = node_R->n_area;
  min_cost = DBL_MAX;
  for (i=0;i<n1;++i) {
    node_L->ca = i;
    for (j=0;j<n2;++j) {
      node_R->ca = j;
      cost = BME_merging_cost(node_L, node_R); 
      min_cost =  tMIN(min_cost, cost);
    }
  }

  return(min_cost);
}
/****************************************************************************/
/*      mr(v) = the merging region between merging region mr(L) and mr(R)   */
/****************************************************************************/
double calc_merging_cost(int i, int j) {
double cost; 

  if (BST_Mode == BME_MODE) {
    cost = BME_merging_cost(&(CandiRoot[i]), &(CandiRoot[j])); 
  } else { /* IME_Mode */
    cost = IME_merging_cost(&(CandiRoot[i]), &(CandiRoot[j]));  
  }
  return(cost); 
}
/****************************************************************************/
/*      mr(v) = the merging region between merging region mr(L) and mr(R)   */
/****************************************************************************/
void Merge2Trees_sub3(int v, int L, int R) {
  Node[L].parent = v;
  Node[R].parent = v;
  Node[v].L = L;
  Node[v].R = R;
  Node[v].Merge2Nodes ( &(Node[L]), &(Node[R])); 
  build_NodeTRR(&(Node[v]));
}

/****************************************************************************/
/*      mr(v) = the merging region between merging region mr(L) and mr(R)   */
/****************************************************************************/
void opt_Merge3Trees_sub1(int v, int L, int R) {
int i, min_i, u[3], id[3];
double min_cost, cost, old_cost;
NodeType tmpnode;

  alloca_NodeType(&tmpnode);
  tmpnode.ca=0;
  tmpnode.n_area = 1;
  id[0] = Node[L].L;
  id[1] = Node[L].R;
  id[2] = R;
  for (i=0;i<3;++i) {
    assert(id[i] != L);
    if (id[i]<0) 
      return;
  }
  min_i = 0;
  min_cost = old_cost = Node[v].area[Node[v].ca].subtree_cost;
  for (i=1;i<3;++i) {
    u[0] = id[i%3];
    u[1] = id[(i+1)%3];
    u[2] = id[(i+2)%3];
    tmpnode.Merge2Nodes (&(Node[u[0]]), &(Node[u[1]])); 
    cost = BME_merging_cost(&tmpnode, &(Node[u[2]]));
    if (cost < min_cost) {
      min_cost = cost;
      min_i = i;
    }
  } 
  if (min_i > 0) {
    u[0] = id[min_i%3];
    u[1] = id[(min_i+1)%3];
    u[2] = id[(min_i+2)%3];
    Merge2Trees_sub3(L, u[0],u[1]);
    Merge2Trees_sub3(v,L, u[2]);
    printf(" *** opt_Merge3Trees (V=%d, L=%d, R=%d) (%d,%d,%d)\n", 
           v,L,R, u[0],u[1],u[2]);
    printf(" cost : %f -> %f (impr: %.0f%%) \n", old_cost, min_cost, 
            (old_cost - min_cost)*100.0/old_cost );
    assert(equal(min_cost, Node[v].area[Node[v].ca].subtree_cost));
  }
  free_NodeType(&tmpnode);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void opt_Merge4Trees(int v, int L, int R) {
int i, min_i, u[4], id[4];
double min_cost, cost, old_cost;
NodeType tmpnode1, tmpnode2;

  alloca_NodeType(&tmpnode1);
  alloca_NodeType(&tmpnode2);
  tmpnode1.ca=0;
  tmpnode1.n_area = 1;
  tmpnode2.ca=0;
  tmpnode2.n_area = 1;
  id[0] = Node[L].L;
  id[1] = Node[L].R;
  id[2] = Node[R].L;
  id[3] = Node[R].R;
  for (i=0;i<4;++i) {
    assert(id[i] != L);
    assert(id[i] != R);
    if (id[i]<0) 
      return;
  }
  min_i = 0;
  min_cost = old_cost = Node[v].area[Node[v].ca].subtree_cost;
  u[3] = id[3];
  for (i=1;i<3;++i) {
    u[0] = id[i%3];
    u[1] = id[(i+1)%3];
    u[2] = id[(i+2)%3];
    tmpnode1.Merge2Nodes (&(Node[u[0]]), &(Node[u[1]]));
    tmpnode2.Merge2Nodes (&(Node[u[2]]), &(Node[u[3]]));
    cost = BME_merging_cost(&tmpnode1, &tmpnode2);
    if (cost < min_cost) {
      min_cost = cost;
      min_i = i;
    }
  }
  if (min_i > 0) {
    u[0] = id[min_i%3];
    u[1] = id[(min_i+1)%3];
    u[2] = id[(min_i+2)%3];
    Merge2Trees_sub3(L, u[0],u[1]);
    Merge2Trees_sub3(R, u[2],u[3]);
    Merge2Trees_sub3(v, L, R);
    printf(" *** opt_Merge4Trees (V=%d, L=%d, R=%d) (%d,%d,%d,%d)\n", 
            v,L,R, u[0],u[1],u[2],u[3]);
    printf(" cost : %f -> %f (impr: %.0f%%) \n", old_cost, min_cost, 
            (old_cost - min_cost)*100.0/old_cost );
    assert(equal(min_cost, Node[v].area[Node[v].ca].subtree_cost));
  }
  free_NodeType(&tmpnode1);
  free_NodeType(&tmpnode2);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void opt_Merge3Trees(int v, int L, int R) {

  if (Node[L].area[Node[R].ca].capac >  Node[R].area[Node[R].ca].capac) {
    opt_Merge3Trees_sub1(v,L,R);
  } else {
    opt_Merge3Trees_sub1(v,R,L);
  }
}
/****************************************************************************/
/*      mr(v) = the merging region between merging region mr(L) and mr(R)   */
/****************************************************************************/
void Merge2Trees(int v, int L, int R) {
double old_cost;

  Merge2Trees_sub3(v,L,R);
/* if (BST_Mode == BME_MODE && detour_Node(v)) { */
  if (BST_Mode == BME_MODE && Local_Opt) {
    do {
      old_cost = Node[v].area[Node[v].ca].subtree_cost;
      opt_Merge3Trees(v,Node[v].L,Node[v].R);
      opt_Merge4Trees(v,Node[v].L,Node[v].R);
    } while (Node[v].area[Node[v].ca].subtree_cost < old_cost - FUZZ);
  }
/*
*/
}

/******************************************************************/
/*                                                               */
/******************************************************************/
void MergeSegment(AreaType *area, int mode){
double d0, d1, h, v;
double t0, t1; 

  h = ABS(JS[0][0].x - JS[1][0].x); 
  v = ABS(JS[0][0].y - JS[1][0].y);
  area->dist = h + v;

  calc_Bal_of_2pt(&(JS[0][0]), &(JS[1][0]), 0, 0, &d0, &d1, &(area->mr[0]));
  assert(equal(area->mr[0].max, area->mr[0].min));
  check_calc_Bal_of_2pt(&(JS[0][0]), &(JS[1][0]), 0, &(area->mr[0]), d0, d1);

  if ( d0==0 || d1==0 ) { /* detour happens */
    area->n_mr = 1;
    area->L_EdgeLen = d0;
    area->R_EdgeLen = d1;
  } else {
    area->L_EdgeLen = NIL;
    area->R_EdgeLen = NIL;
    if (!equal(v,0) && !equal(h,0)) {
      area->n_mr = 2;
      calc_Bal_of_2pt(&(JS[0][0]), &(JS[1][0]), 0, 1, &d0, &d1, &(area->mr[1]));
      assert(equal(area->mr[1].max, area->mr[1].min));
      t0 = pt_delay_increase(Gamma, JS[0][0].t, &(JS[0][0]), &(area->mr[1]));
      t1 = pt_delay_increase(Gamma, JS[1][0].t, &(JS[1][0]), &(area->mr[1]));
      assert(equal(JS[0][0].max + t0, JS[1][0].max + t1));
    } else {
      area->n_mr = 1;
    }
  }
  area->vertex[0] = 0;
  area->vertex[1] = 1;
  area->npts = area->n_mr;
}


/****************************************************************************/
/*            return the merging area of Node1 and Node2                    */
/* Case II: general case where merging areas not equal to merging segments. */
/****************************************************************************/
void MergeNonManhattanArc(AreaType *area,AreaType *area_L,AreaType *area_R, int mode){

  assert(area->area_L == area_L);
  assert(area->area_R == area_R);
  if (Skew_B == 0 ) {
    JS[0][0].t = area_L->capac;
    JS[1][0].t = area_R->capac;
    MergeSegment(area, mode);
  } else {
    construct_mr(area, area_L,area_R,mode);
  }

  double origB =  gBoundedSkewTree->Orig_Skew_B()  ;
  if (area->n_mr==4 && origB >0 ) assert(!TRR_area(area));
}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void cal_ms_merging_cost(AreaType  *area,AreaType  *area_L,AreaType  *area_R) {
PointType path[100];
double cost;
int n;

  cost = path_between_JSline(NULL, area->line, path, &n);
  assert(n<100);
  area->subtree_cost = cost + area_L->subtree_cost + area_R->subtree_cost;
}
/***********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void check_MergeArea1_sub(AreaType  *area, PointType *pt) {
  if (area->R_buffer > 0) {
    assert(equal(area->capac,C_buffer));
    assert(MaxClusterDelay >= pt->max);
  } else {
    assert(equal(area->capac,area->unbuf_capac));
  }
  assert(area->npts > 0);
  assert(area->n_mr > 0);
  assert(area->npts <= MAXPTS);
  assert(area->n_mr <= MAX_mr_PTS);
}
/***********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/***********************************************************************/
void check_MergeArea1(AreaType  *area) {

  check_MergeArea1_sub(area->area_L, &(JS[0][0]));
  check_MergeArea1_sub(area->area_R, &(JS[1][0]));
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void check_MergeArea2(AreaType  *area,AreaType  *area_L,AreaType  *area_R,
               int mode) {
  if (mode == NORMAL) {
    if ( (BST_Mode==BME_MODE && Skew_B  == DBL_MAX  && area->npts >4) || 
         (BST_Mode==IME_MODE && area->npts >6)) {
      print_area_info(area);
      print_area(area_L);
      print_area(area_R);
      assert(0);
    }
  }
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void MergeArea(AreaType  *area,AreaType  *area_L,AreaType  *area_R,int mode) {
double ori_dist;
PointType orig_line[2][2];

  area->area_L = area_L;
  area->area_R = area_R;
  area->L_EdgeLen = area->R_EdgeLen = NIL;
  area->L_StubLen = area->R_StubLen = 0;
  area->R_buffer = 0;
  area->ClusterDelay = 0;
  ori_dist = area->dist;
  JS_processing(area); 
  orig_line[0][0] = area->line[0][0];
  orig_line[0][1] = area->line[0][1];
  orig_line[1][0] = area->line[1][0];
  orig_line[1][1] = area->line[1][1];

  check_MergeArea1(area);
  if ( 0 && mode==FAST) {
    cal_ms_merging_cost(area,area_L, area_R);
  } else {
    if (case_Manhattan_arc() && PURES[H_]==PURES[V_] && PUCAP[H_]==PUCAP[V_]) {
      MergeManhattanArc(area,area_L, area_R, mode); 
      double origB =  gBoundedSkewTree->Orig_Skew_B()  ;
      if (area->n_mr==4 && origB >0 ) assert(!TRR_area(area));
    } else {
      MergeNonManhattanArc(area,area_L,area_R, mode);
    }
    check_MergeArea2(area, area_L, area_R, mode);
  }

  if (Manhattan_arc(JS[0][0], JS[0][1]) ) {
    assert(Manhattan_arc(JS[1][0], JS[1][1]));
    JS_processing_sub2(area); 
  } else {
    area->line[0][0] = orig_line[0][0];
    area->line[0][1] = orig_line[0][1];
    area->line[1][0] = orig_line[1][0];
    area->line[1][1] = orig_line[1][1];
  }
  area->dist = ori_dist;

  MergeArea_sub(area);
}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void embedding_sub2(int p, int v, PointType pt1, PointType pt2) {
TrrType trr1, trr2,ms;

  PointType &pt = Node[p].m_stnPt ;
  trr1.MakeDiamond(pt,EdgeLength[v] );
  line2ms(&ms, pt1, pt2);
  make_intersect(&ms, &trr1,&trr2);
  PointType &qt = Node[v].m_stnPt ;
  core_mid_point(&trr2, &qt);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
double find_new_embedding_pt_sub2(int p,int v,PointType pt1,PointType pt2) {
int n;
PointType path[100], line[2][2];
double dist;

  line[0][0] = pt1;
  line[0][1] = pt2;
  PointType &pt = Node[p].m_stnPt ;
  line[1][0] = line[1][1] = pt       ;
  dist = path_between_JSline(NULL, line, path, &n);
  Node[v].m_stnPt = path[0];
  assert(pt_on_line_segment(path[0], pt1,pt2));

  return(dist);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
double find_new_embedding_pt_sub1(int p,int v,PointType pt1,PointType pt2) {
int n1, n2;
PointType path1[100], path2[100];
double cost1, cost2;


  PointType &pt = Node[p].m_stnPt ;
   cost1 = path_finder(pt1, pt,        path1, &n1);
   assert(n1<=100);
   cost2 = path_finder(pt2, pt,        path2, &n2);
   assert(n2<=100);

   if (cost1<=cost2) {
     Node[v].m_stnPt = path1[0];
   } else {
     Node[v].m_stnPt = path2[0];
   }
  return( tMIN(cost1, cost2));
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
double find_new_embedding_pt(int p,int v,PointType pt1,PointType pt2) {
double cost;

  if (p==221 && v==219) {
    cost = find_new_embedding_pt_sub1(p,v,pt1, pt2);
  } else {
    cost = find_new_embedding_pt_sub2(p,v,pt1, pt2);
  }
  return(cost);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void embedding_sub(int p, int v, PointType pt1, PointType pt2, double edgelen,
                   AreaType *area) {
double dist,a,b, old_dist;
int unblocked; 
TrrType trr1, trr2,ms;

  assert(p>=0 );

    PointType &pt = Node[p].m_stnPt ;
    PointType &qt = Node[v].m_stnPt ;
  if (area->n_mr == 4 && TRR_area(area)) {
    pts2TRR(area->mr, area->n_mr, &trr2);
    dist = pt2TRR_distance_sub(&pt, &trr2);
    trr1.MakeDiamond (pt,dist); 
    make_intersect(&trr1, &trr2, &ms);
    core_mid_point(&ms, &qt         );
  } else {
    a = ABS(pt1.x-pt2.x);
    b = ABS(pt1.y-pt2.y);
    if ( equal(a,0) && equal(b,0) ) {  /* pt1 and pt2 are smae points */
      qt       = pt1;  
    } else if (equal(a,0)) {  /* vertical line */
      qt       .x = pt1.x;
      qt       .y = pt       .y;
    } else if (equal(b,0)) {  /* horizontal line */
      qt       .x = pt       .x;
      qt       .y = pt1.y;
    } else {  /* other types of line segment */
      pt2linedist(pt       ,pt1, pt2, &(qt       ));
    }
    assert(pt_on_line_segment(qt       , pt1,pt2));
  }

  old_dist = dist = Point_dist(pt, qt              );
  unblocked = unblocked_segment(&pt,          &qt         );

  if (equal(dist,0) || unblocked ) {
    /* o.k. */
  } else {
    dist = find_new_embedding_pt(p, v, pt1, pt2);
  }

  if ( edgelen == NIL ) {
    assert(Skew_B != 0 );
    EdgeLength[v] =  dist;
  } else {  /* detour wiring required */
    if (edgelen < dist - FUZZ) {
      printf("warning: p=%d:v= %d, edgelen = %.9f, dist = %.9f\n", 
              p,v, edgelen,dist);
    }
    assert(edgelen >= dist -  100*FUZZ);
    EdgeLength[v] =  tMAX(edgelen,dist);
  }
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void check_embedding(PointType *p0, PointType *p1, AreaType *area) {
PointType tmp_pt0, tmp_pt1;

  if (BST_Mode == BME_MODE) {  /* check embedded in MR */
    calc_BS_located(p0, area, &tmp_pt0, &tmp_pt1);
    calc_BS_located(p1, area, &tmp_pt0, &tmp_pt1);
  }
}
/*************************************************************************/
/*             top-down embedding the tree of merging regions            */
/*************************************************************************/
void embedding(int p, int child) {
PointType p0, p1;
AreaType *area;
double t, edgelen;
int v;

  if (child==0) { v = Node[p].L; } else { v = Node[p].R; }
  area = &(Node[p].area[Node[p].ca]);
  if (child==0) {
    edgelen = area->L_EdgeLen;
    StubLength[v] = area->L_StubLen; 
    p0 = area->line[0][0];
    p1 = area->line[0][1];
  } else {
    edgelen = area->R_EdgeLen;
    StubLength[v] = area->R_StubLen; 
    p0 = area->line[1][0];
    p1 = area->line[1][1];
  }

  area = &(Node[v].area[Node[v].ca]);

  check_embedding(&p0, &p1, area);

  t = drand48();
  if ( t < Gamma) {
    Node[v].pattern = 1;
  } else {
    Node[v].pattern = 0;
  }
  embedding_sub(p, v, p0, p1, edgelen, area);
  assert(StubLength[v] < EdgeLength[v] + FUZZ);

  int nterms = (int) gBoundedSkewTree->Nterms() ;
  if (v>=nterms) embedding(v,0);
  if (v>=nterms) embedding(v,1);
}

/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void calc_area_center(AreaType *area, PointType *pt) {
double x,y;
int i, j, n;

  x=y=0.0;
  n = area->npts;
  for (i=0;i<n;++i) {
    j = area->vertex[i];
    x += area->mr[j].x;
    y += area->mr[j].y;
  }
  pt->x = x/n;
  pt->y = y/n;
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void TopDown_Embedding(int v) {

  if ( BST_Mode != BME_MODE ) {
    int i = calc_best_area(v);
    get_all_areas(v,i); 
  }
/*
  assert(Node[v].ca == Node[v].n_area-1);
*/
  PointType &pt = Node[v].m_stnPt ;
  NodeType *from  = gBoundedSkewTree->SuperRootNode () ;
  int superRoot = gBoundedSkewTree->SuperRootNodeIndex () ;

  PointType &qt = from->m_stnPt ;
  AreaType * area = &(Node[v].area[Node[v].ca]);
  calc_area_center(area, &(pt       ));
  qt               .x = pt       .x;
  qt               .y = pt       .y;
  StubLength[superRoot] = EdgeLength[superRoot] = 0;

  int nterms = (int) gBoundedSkewTree->Nterms() ;
  if (v>=nterms) embedding(v,0);
  if (v>=nterms) embedding(v,1);
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
double skewchecking(int v) {
double t, cost, cost2;

  TopDown_Embedding(v);
  calc_BST_delay(v);
  
  int root = gBoundedSkewTree->RootNodeIndex () ;
  if (v==root) set_SuperRoot();

  calc_TreeCost(v, &cost, &t);
  cost2 = Node[v].area[Node[v].ca].subtree_cost;
  if (!equal(cost, cost2)) {
    assert(equivalent(cost, cost2, 1E-5));
  }

  PointType &pt = Node[v].m_stnPt ;
  t = pt_skew( pt      )-Skew_B;

  if (t  >  tMAX(Skew_B/10.0,FUZZ*10)  ) {
    printf("skew[%d] - Skew_Bound  = %f ", v, t); 
    bool linear = gBoundedSkewTree->LinearDelayModel ()  ;
    if ( !linear ) {
       printf(" (time unit: ps)"); }
    printf("\n\n"); 
    print_Point(stdout, pt      );
    print_node(&(Node[v])); 
    assert(0);
  }
  return(cost);
}

/****************************************************************************/
/*  reconstruct tree of merging Node rooted at v                            */
/****************************************************************************/
void embed_topology(int v) {
int L,R;

  assert(v>=0); 
  unsigned nterms = gBoundedSkewTree->Nterms() ;
  assert( (unsigned) v>=nterms || Node[v].area[Node[v].ca].npts == 1); 
  assert(Node[v].area[Node[v].ca].npts <=0);

  L=Node[v].L;
  R=Node[v].R;
  unsigned npoints = gBoundedSkewTree->Npoints() ;
  assert( L>=0 && R>=0 && L <= (int) npoints && R <= (int) npoints);
  if (Node[L].area[Node[L].ca].npts<=0) {
    embed_topology(L);
  }
  if (Node[R].area[Node[R].ca].npts<=0) {
    embed_topology(R);
  }
  Node[v].Merge2Nodes ( &(Node[L]), &(Node[R]));
  skewchecking(v);
}

/****************************************************************************/
/*  change the topology of the subtree to be rooted at p. */
/*  return the node id of root and the associated merging Node.             */
/****************************************************************************/
int change_topology(int p) {
int root,o,q,s, old_p = p, old_q;
int tree_size1 = 0, tree_size2 = 0;

  assert( p>=0 && !Marked[p]) ;

  /* defualt root id and its merging Node */
  root = Node[p].root_id;
  assert(Node[root].parent == NIL); 

  count_tree_nodes(root, root, &tree_size1);
  old_q = q = Node[p].parent;
  if (q<0) return(root);
  s = Node[q].parent;
  if (s<0) return(root);

  o=p;
  p = q;
  q = s;
  s = Node[q].parent;
  while (s > 0) {
    set_sibling(o,p,q);
    Node[p].area[Node[p].ca].npts = 0;
    Node[q].parent = p;
    o=p;
    p = q;
    q = s;
    s = Node[q].parent;
  }
  /* q == root */
  q = sibling(p,q);
  set_sibling(o,p,q);
  Node[p].area[Node[p].ca].npts = 0;
  Node[q].parent = p;

  Node[root].area[Node[root].ca].npts = 0;
  Node[root].L = old_p;
  Node[root].R = old_q;
  Node[old_p].parent = Node[old_q].parent = root;

  embed_topology(root);

  count_tree_nodes(root, root,  &tree_size2);
  assert(tree_size1 == tree_size2);
  assert(root == Node[old_p].root_id);
  assert(Node[root].parent < 0 );
  return(root);
}

/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
static 
int  Ex_DME () {
  int root = gBoundedSkewTree->RootNodeIndex () ;

  string  fn = gBoundedSkewTree->TopologyFileName ()  ;
  read_input_topology( fn );
  embed_topology( root );
  return root ;
}
/****************************************************************************/
/*                                                                          */
/****************************************************************************/
void calloc_TempArea() {
int n;
  n = N_Area_Per_Node*(N_Sampling+1);

  N_TempArea = 1000;
  printf("\n\nN_TempArea = %d \n\n", N_TempArea);
  TempArea = (AreaType *) calloc(N_TempArea,  sizeof(AreaType));
  assert(  TempArea != NULL ); 
}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
void set_R_buffer_size( int  t) {
int i;
  
  for (i=0;i<N_BUFFER_SIZE;++i) {
    R_buffer_size[i] = 100 + t*i;
  }
}
/****************************************************************************/
/*   parse arguments                                                        */
/****************************************************************************/
void parse_argument(int argc, char *argv[], BstTree &tree ) {
int i, j;
char fn[20]; 
int BRESinc = 20;

  N_Clusters[1] = 1;
  N_Clusters[2] = 0;

  Hierachy_Cluster_id[0] = 0;

  for (i=1;i<argc;i++) {
    if (strcmp("-G",argv[i]) == 0)  {
      i++;
      tree.SetTopologyFileName (  argv[i] ) ;
    } else if (strcmp("-s",argv[i]) == 0)  {
      i++;
      sscanf(argv[i],"%d",&N_Sampling);
    } else if (strcmp("-O",argv[i]) == 0)  { // not complete yet
      i++;
      sscanf(argv[i],"%s",fn);
      tree.SetObstructionFileName ( fn ) ;
    } else if (strcmp("-n",argv[i]) == 0)  { // for IME method
      i++;
      sscanf(argv[i],"%d",&N_Area_Per_Node);
      assert(N_Area_Per_Node>1);
      BST_Mode = IME_MODE; 
      calloc_TempArea();
    } else if (strcmp("-DS",argv[i]) == 0)  { // for IME method
      Dynamic_Selection = YES;
    } else if (strcmp("-H",argv[i]) == 0)  { // for IME method
      BST_Mode = HYBRID_MODE;
    } else if (strcmp("-D",argv[i]) == 0)  { // for delay model
      i++;
      unsigned k ;
      sscanf(argv[i],"%d",&k );
      tree.SetDelayModel ( (BST_DME::DelayModelType) k ) ;
    } else if (strcmp("-BRES",argv[i]) == 0)  { // for buffer linear model
      i++;
      sscanf(argv[i],"%d",&R_buffer);
    } else if (strcmp("-BRESinc",argv[i]) == 0)  { // for buffer linear model
      i++;
      sscanf(argv[i],"%d",&BRESinc);
    } else if (strcmp("-BCAP",argv[i]) == 0)  { // for buffer linear model
      i++;
      sscanf(argv[i],"%lf",&C_buffer);
      C_buffer *= PUCAP_SCALE;
    } else if (strcmp("-RES",argv[i]) == 0)  { // for non-uniform RC model
      i++;
      sscanf(argv[i],"%lf",&PURES_V_SCALE);
    } else if (strcmp("-CAP",argv[i]) == 0)  { // for non-uniform RC model
      i++;
      sscanf(argv[i],"%lf",&PUCAP_V_SCALE);
    } else if (strcmp("-B",argv[i]) == 0)  { // for skew bound
      i++;
      double skew =0 ;
      sscanf(argv[i],"%lf",&skew);
      if (skew < 0) { 
        skew = DBL_MAX; 
      } 
      tree.SetSkewBound ( skew ) ;
      
    } else if (strcmp("-NCL",argv[i]) == 0)  {
      j = 0;
      do {
        i++;
        j++;
        sscanf(argv[i],"%d",&(N_Clusters[j]));
      } while (N_Clusters[j]!=1);
      N_Buffer_Level = j;
    } else {
      printf("Argument %d incorrect\n",i);
      exit(0);
    }
  }

  set_R_buffer_size(BRESinc);

  if ( tree.SinksFileName().empty() ) {
     printf ("usage: bst -i inputFileName -B number (pico-seconds)\n" ) ;
     exit (0 ) ;
  } 
}

/****************************************************************************/
/*  update v's merging Node as if v were the root of tree root_id.   */
/*  i.e., update the variable  CandiRoot[v]                             */
/****************************************************************************/
void update_CandiRoot(int v, int root_id, NodeType *node_R) {
int par, sib;

  if (v<0 )  return;
  par = Node[v].parent;
  if (v==root_id) {
    assign_NodeType(&(CandiRoot[v]), &(Node[root_id]));
    Marked[v] = NO;
  } else if (par == root_id) {
    assign_NodeType(&(CandiRoot[v]), &(Node[root_id]));
    sib = sibling(v, par);
    assign_NodeType(&(TempNode[v]), &(Node[sib]));
    Marked[v]      = YES;
  } else {
    sib = sibling(v, par);
    TempNode[v].Merge2Nodes ( &(Node[sib]),node_R);
    CandiRoot[v].Merge2Nodes (&(Node[v]), &(TempNode[v]));
    if (All_Top == YES ||
        minskew(&(CandiRoot[v]), BST_Mode) <= Skew_B - 0.001 ) {
      Marked[v]=NO;
    } else {
      Marked[v]=YES;
    }
/*
    Marked[v]=(minskew(&(CandiRoot[v]),BST_Mode) >  Skew_B-0.001 ) ? YES: NO; 
*/
  }
  build_NodeTRR(&(CandiRoot[v]));

/*
  if ( minskew(&(CandiRoot[v]), BST_Mode) <= Skew_B - 0.001 ) { 
*/
  if ( (Buffered[v]==NO) && 
       (All_Top == YES || 
        minskew(&(CandiRoot[v]), BST_Mode) <= Skew_B - 0.001) )  {
    update_CandiRoot(Node[v].L, root_id,  &(TempNode[v])); 
    update_CandiRoot(Node[v].R, root_id,  &(TempNode[v])); 
  } else {
    init_marked(Node[v].L); 
    init_marked(Node[v].R); 
  }
}
/******************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/******************************************************************/
int ExG_DME (int n_trees, int v, int show_info) {
int i, j, k, n;

  while (n_trees > 1) {
    build_nearest_neighbor_graph(v, show_info);
    n = pairs_to_merge(v);
    if (show_info) {
      printf("\n#_trees=%d, #_nodes=%d, #_mering_pairs=%d\n",n_trees, v, n);
    }
    n_trees -= n;
    for (k=0;k<n;k++,v++) {
      i = Best_Pair[k].x;
      j = Best_Pair[k].y;
      /* printf("node %d == merge-pair (%d,%d)\n", v,  i, j); */
      i = change_topology(i); // change top. of subtree rooted at i
      j = change_topology(j); // change top. of subtree rooted at j
      Merge2Trees(v, i,  j);
      updateRootId(v,v);
      assert(Cluster_id[i] == Cluster_id[j]);
      Cluster_id[v] = Cluster_id[i];
      if (CHECK) skewchecking(v); 
      update_CandiRoot(v,v, NULL);
    }
  }
  return(v);
}
/****************************************************************************/
/*   Initialization                                                         */
/****************************************************************************/
void init_a_Node(int i) {
  
  Node[i].id = i;
  Node[i].ca = 0;
  Node[i].n_area = 1;
  Node[i].L=Node[i].R=Node[i].parent=-1;
  Node[i].area[0].L_EdgeLen = Node[i].area[0].R_EdgeLen = EdgeLength[i]=0; 
  Node[i].area[0].R_buffer = 0;
  StubLength[i] = Node[i].area[0].L_StubLen = Node[i].area[0].R_StubLen = 0; 
  N_neighbors[i]= 0;
  Cluster[i].x = DBL_MAX;   /* min merging cost since this cluster exists */
  Buffered[i] = NO; 
  Marked[i] = YES;
  int nterms = (int) gBoundedSkewTree->Nterms() ;
  if (i<nterms) {
    Node[i].root_id = i;
    Node[i].area[0].n_mr = Node[i].area[0].npts = 1; 
    PointType &pt = Node[i].m_stnPt ;
    pt        = Node[i].area[0].mr[0]; 
    Node[i].area[0].subtree_cost = 0;
    build_NodeTRR(&(Node[i]));
    Node[i].area[0].unbuf_capac = Node[i].area[0].capac;
  } else {
    Node[i].area[0].n_mr = Node[i].area[0].npts = Node[i].root_id = NIL;
    Node[i].area[0].mr[0].min = Node[i].area[0].mr[0].max = NIL;
    Node[i].area[0].unbuf_capac = Node[i].area[0].capac = NIL;
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void init_all_Nodes() {
  unsigned npoints = gBoundedSkewTree->Npoints() ;
  for (unsigned  i=0;i<npoints;++i) {
    init_a_Node(i);
  }

}
/****************************************************************************/
/*   Initialization                                                         */
/* to be used in skew_allocation() in bst_sub2.c */
/****************************************************************************/
void init_skew_allocation() {

  unsigned nterms = gBoundedSkewTree->Nterms() ;
  Points = (PointType *) calloc(nterms, sizeof(PointType));
  TmpMarked = (int *) calloc(nterms, sizeof(int));
  Capac = (double *) calloc(nterms, sizeof(double));

}
/****************************************************************************/
/*   Initialization                                                         */
/****************************************************************************/
void init() {
 
  int nterms = (int) gBoundedSkewTree->Nterms() ;
  N_Clusters[0] = nterms;
  if (N_Clusters[1]<=0) {
    unsigned i = ABS(N_Clusters[1]);
    N_Clusters[1] = (int) sqrt( (double) nterms);
    N_Clusters[1] =  N_Clusters[1]*2/i;
  }

  if (Hierachy_Cluster_id[0] == 0) {
    unsigned npoints = gBoundedSkewTree->Npoints() ;
    for (unsigned i=0; i<npoints; i++) {
      Cluster_id[i] = NIL;
    }
  }

  assert(N_Clusters[1] <= MAX_N_SINKS);

  init_skew_allocation();

}
/******************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/******************************************************************/
double calc_max_capac(int L) {
double cap;
int i, j;

  cap = 0;
  for (i=0;i<N_Clusters[L];++i) {
    j = TreeRoots[i];
    cap = tMAX(cap, Node[j].area[0].capac);
  }
  return(cap);
}

/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
void set_MaxClusterDelay_sub(int v) {
int i, j;
  for (i=0;i<Node[v].n_area;++i) {
    for (j=0;j<Node[v].area[i].n_mr; ++j) {
      MaxClusterDelay = tMAX(MaxClusterDelay, Node[v].area[i].mr[j].max);
    }
  }
}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
void set_MaxClusterDelay(int cid) {
int i;

  MaxClusterDelay = 0;
  for (i=0; i<Curr_Npoints; i++) {
    if (Cluster_id[i] == cid) {
      set_MaxClusterDelay_sub(i);
    }
  }
}
/******************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/******************************************************************/
static
void add_buffer_sub (int v) {
int i, j;

  for (i=0;i<Node[v].n_area;++i) {
    double t =  Delay_buffer+Buffered[v]*(Node[v].area[i].capac);
    for (j=0;j<Node[v].area[i].n_mr; ++j) {
      Node[v].area[i].mr[j].max += t; 
      Node[v].area[i].mr[j].min += t; 
    }
    Node[v].area[i].capac = C_buffer;
    Node[v].area[i].R_buffer = Buffered[v];
  }
}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
void set_ClusterDelay_sub(int v) {
int i;
  for (i=0;i<Node[v].n_area;++i) {
    Node[v].area[i].ClusterDelay = MaxClusterDelay;
  }
}
/******************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/******************************************************************/
static
void add_buffer (int L, int n_buffer, int buffered_Node[]) {

  MaxClusterDelay = 0;

  for (int i=0;i<n_buffer;++i) {
    int j = buffered_Node[i];
    add_buffer_sub (j);
    set_MaxClusterDelay_sub(j);
  }
  for (int i=0;i<n_buffer;++i) {
    set_ClusterDelay_sub(buffered_Node[i]);
  }
}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
void set_R_buffer_at_Node(double target_delay, int x) {
int i ;
double min_diff, t, diff;

  min_diff = DBL_MAX;
  for (i=0;i<N_BUFFER_SIZE;++i) {
    t = Node[x].area[0].capac * R_buffer_size[i];
    diff = ABS(t - target_delay);
    if ( diff < min_diff) {
      min_diff = diff;
      Buffered[x] =  R_buffer_size[i]; 
    } 
  }
}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
double calc_Buffered_sub(int L, int n_buffer, int buffered_Node[]) {

  int x = buffered_Node[0];
  double target_delay =  Node[x].area[0].capac *Buffered[x] ;

  for (int i=1;i< n_buffer;++i) {
    set_R_buffer_at_Node(target_delay, buffered_Node[i]);
  }
  double max_t = -DBL_MAX;
  double min_t =  DBL_MAX;
  for (int i=0;i<N_Clusters[L];++i) {
    int x = buffered_Node[i];
    double t = Node[x].area[0].capac * Buffered[x];
    max_t =  tMAX( max_t, t); 
    min_t =  tMIN( min_t, t); 
  }
  return(max_t - min_t);
}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
static
void selct_buffer_size_sub (int L, int n_buffer, int buffered_Node[]) {
int i, min_i, x;
double diff, min_diff;

  min_diff = DBL_MAX;
  for (i=0;i<N_BUFFER_SIZE;++i) {
    Buffered[buffered_Node[0]] = R_buffer_size[i];
    diff = calc_Buffered_sub(L, n_buffer, buffered_Node);
    if (diff < min_diff) {
      min_diff =   tMIN(diff, min_diff);
      min_i = i;
    }
  }
  x = buffered_Node[0];
  Buffered[x] = R_buffer_size[min_i];
  calc_Buffered_sub(L, n_buffer, buffered_Node);
}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
static
void selct_buffer_size (int L, int n_buffer, int buffered_Node[]) {
int i;

  if (R_buffer > 0) {
    for (i=0;i<n_buffer;++i) {
      Buffered[buffered_Node[i]] = R_buffer;
    }
  } else {
    selct_buffer_size_sub (L, n_buffer, buffered_Node);
  }
}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
void print_run(int L, int i, int n_trees) {
  printf("============================================================\n");
  printf("=  run ExG_DME on cluster (L=%d:i=%d/%d) cid=%d \n",
          L, i, N_Clusters[L], Total_CL);
  printf("=  n_trees = %d \n", n_trees);
  printf("============================================================\n");
}
/****************************************************************************/
/*   Initialization   for ExG-DME                                           */
/****************************************************************************/
static
int init_ExG_DME(int cid, int *u) {
int i, n;

  for (i=n=0; i<Curr_Npoints; i++) {
    if (Cluster_id[i] == cid) {
      *u = i;
      Marked[i] = NO; 
      assign_NodeType(&(CandiRoot[i]), &(Node[i]));
      assign_NodeType(&(TempNode[i]), &(Node[i]));
      check_root_id(i);
      n++;
    } else {
      Marked[i] = YES; 
    }
  }
  assert(n>0);
  return(n);
}
/****************************************************************************/
/*   Initialization                                                         */
/****************************************************************************/
static
void init_calc_whole_BST() {

  unsigned npoints = gBoundedSkewTree->Npoints() ;
  int nterms = (int) gBoundedSkewTree->Nterms() ;
  for (unsigned i=nterms; i<npoints; i++) { 
    Cluster_id[i] = NIL; 
  }
  init_all_Nodes(); 
}

/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
void init_BSTs_at_level() {
int i;

  int nterms = (int) gBoundedSkewTree->Nterms() ;
  for (i=0;i<nterms;++i) { TreeRoots[i] = i; }
  Curr_Npoints = nterms+1;
  Total_CL = 0; 
}
/**********************************************************************/
/* construct BSTs at level L of buffer hierachy                      */
/* build a BST for each clusters of nodes at level L.                */
/**********************************************************************/
static
void  BSTs_at_level(int show_info, int L) {
int i, u, n_trees;
int buffered_Node[MAX_N_NODES];

  int n_buffer = 0;
  for (i=0;i<N_Clusters[L];++i) {
    n_trees = init_ExG_DME(Total_CL, &u);
    if (show_info ) print_run(L,i,n_trees);
    if (n_trees >1 ) {  /* there is a forest */
      Curr_Npoints = ExG_DME (n_trees, Curr_Npoints, show_info); 
      buffered_Node[n_buffer++] = TreeRoots[i] = Curr_Npoints-1;
    } else {  /* there is only a single tree */
      TreeRoots[i] = u;
    }
    Total_CL++; 
  }
  selct_buffer_size (L, n_buffer, buffered_Node);
  add_buffer (L, n_buffer, buffered_Node);
}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
static
void calc_whole_BST(int show_info, int PostOpt, bool fixedTopology ) {
int L (0) ;

  assert ( PostOpt == NO ) ;
  init_calc_whole_BST();
  if (fixedTopology) { 
    int root = Ex_DME(); 

    int buffered_Node[] = { root} ;
    int n_buffer (1) ;
    selct_buffer_size (L, n_buffer, buffered_Node);
    add_buffer (L, n_buffer, buffered_Node);
  } else { 
    init_BSTs_at_level();
    L = 0;
    do {
      L++;
      init_clusters(L, PostOpt);
      BSTs_at_level(show_info,L);
    } while (N_Clusters[L]>1);
  }

  int root = gBoundedSkewTree->RootNodeIndex () ;

  skewchecking( root );
  if (show_info) print_answer(
    gBoundedSkewTree->SinksFileName().c_str(), root, fixedTopology );

}
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
BST_DME::BST_DME (
    const string &inputSinksFileName,
    const string &inputTopologyFileName,
    const string &inputObstructionFileName,
    double skewBound,
    DelayModelType delayModel
    ) {

     m_bstdme = new BstTree (inputSinksFileName,inputTopologyFileName,
          inputObstructionFileName,skewBound,delayModel );
} ;
/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
BST_DME::~BST_DME (
) {
  delete m_bstdme ;
}

/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
BstTree::BstTree (
    const string &inputSinksFileName,
    const string &inputTopologyFileName,
    const string &inputObstructionFileName,
    double skewBound,
    BST_DME::DelayModelType delayModel
    ):
     m_inputSinksFileName( inputSinksFileName ) ,
     m_inputTopologyFileName( inputTopologyFileName ),
     m_inputObstructionFileName ( inputObstructionFileName  ),
     m_skewBound ( skewBound ),
     m_delayModel ( delayModel ) ,
     m_nterms ( 0 ) 
{

} ;

/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
BstTree::~BstTree (
) {
}


/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
void
BST_DME::ConstructTree( ) {
  m_bstdme->ConstructTree() ;
}

/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
void
BstTree::ConstructTree( ) {
  

  N_Clusters[1] = 1;
  N_Clusters[2] = 0;

  gBoundedSkewTree = this ;

  clock();
  string obsFile = ObstructionFileName() ;

  if ( !obsFile.empty() ) {
    read_obstacle_file ( obsFile  );
    print_obstacles( obsFile );
  }
  
  bool fixTop = FixedTopology();
  Skew_B = SkewBound () ; 

  read_input_file( SinksFileName() ) ;

  init();
  print_header( fixTop );

  calc_whole_BST(YES, NO, fixTop );

  iRunTime();

  print_current_time();


}
