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
#include "bst.h"
#include "bst_sub1.h"

extern void make_core(TrrType *trr,TrrType *core); 
extern double Point_dist(PointType p1, PointType p2);
extern double pt2ms_distance(PointType *pt, TrrType *ms); 
extern void build_NodeTRR(NodeType *node);
 
extern void make_intersect( TrrType *trr1, TrrType *trr2, TrrType *t );
extern void core_mid_point(TrrType *trr, PointType *p); 

extern void init_all_Nodes(); 

extern void dtoa(double x, char s[], int *i);
extern void new_dtoa(double x, char s[]);

int TmpNpoints;
TrrType Center_ms;

double Obj_Cost;

/********************************************************************/
/*                                                                  */
/********************************************************************/
static
double calc_Obj_Cost(int n_clusters, double exponent) {
int i;
double t, sum;

  sum = 0; 
  for (i=0;i<n_clusters;++i) {
    t = TmpCluster[i].t;
    sum +=  pow(t, exponent);
  }
  return(sum);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
static
double calc_deviation(int n_clusters, double ave_t) {
int i;
double t, deviation, sum;

  sum = 0;  
  for (i=0;i<n_clusters;++i) {
    t = TmpCluster[i].t - ave_t;
    sum +=  t*t;
  }
  deviation = sqrt(sum/n_clusters);
  return(deviation);
}
/***************************************************************************/
/*   compute the center of points[L} ~ points[R]     */
/***************************************************************************/
static
double get_the_center(int L, int R,  PointType points[], TrrType *core) {
PointType cp;
TrrType tmp_trr;
double t, radius=0;
double min1, min2, max1, max2;
int i;

  max1 = -DBL_MAX; min1 = DBL_MAX;
  max2 = -DBL_MAX; min2 = DBL_MAX;


  for (i=L;i<=R;++i) {
    t = points[i].x - points[i].y;
    max1 = max (max1,t);
    min1 = min (min1,t);
    t = points[i].x + points[i].y;
    max2 = max (max2,t);
    min2 = min (min2,t);
  }


  radius= max (max1-min1, max2-min2)/2;

  tmp_trr.MakeDiamond(points[L], radius );
  for (i=L+1;i<=R;++i) {
    core->MakeDiamond(points[i], radius);
    make_intersect(core, &tmp_trr, &tmp_trr);
  }
  make_core(&tmp_trr, core);
  core_mid_point(core, &cp);

  return(radius);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
static
double cal_split(int cp) {
double t,x,y;
int i; 
 
  i = NearestCenter[cp];
  assert(i>=0 && i <TmpNpoints);
  t = Point_dist(Points[cp], Points[i]);
  x = Points[cp].x; 
  y = Points[cp].y; 

  t = min (t, (MAX_x - x)*Split_Factor);
  assert(t>=0); 
  t = min (t, (x - MIN_x)*Split_Factor);
  assert(t>=0); 
  t = min (t, (MAX_y - y)*Split_Factor);
  assert(t>=0); 
  t = min (t, (y - MIN_y)*Split_Factor);
  return(t);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
static
double calc_min_split() {
double t;
int i; 

  t= DBL_MAX;
  for (i=0;i<TmpNpoints;++i) {
    if (TmpClusterId[i] >=0) {
      t = min (t, cal_split(i));
    }
  }
  return(t); 
}
/********************************************************************/
/*  update NearestCenter[cp] due to the new center          */
/********************************************************************/
void update_NearestCenter(int cp,int center){
double dist;

  dist = Point_dist(Points[cp], Points[center]);

  if (Points[cp].t > dist) {   /* update nearest center of cp */
    Points[cp].t = dist;
    NearestCenter[cp] = center;
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void add_center(int cid, int sink) {
int i;

  assert(TmpClusterId[sink] == NIL);
  TmpClusterId[sink] = cid; 

  for (i=0;i<TmpNpoints;++i) {
    if (i != sink) {
      update_NearestCenter(i, sink);  
    }
  }
}
/********************************************************************/
/*  update nearest center of Points[cp]                             */
/********************************************************************/
void calc_NearestCenter(int cp) {
int i;
double t; 

  Points[cp].t = DBL_MAX; 
  for (i=0; i< TmpNpoints;++i) {
    if (TmpClusterId[i] >=0 && i!=cp) {
      t = Point_dist(Points[cp], Points[i]);
      if (t < Points[cp].t) {
        NearestCenter[cp] = i;
        Points[cp].t = t;
      }
    }
  }
}
/********************************************************************/
/*  remove the center cid                                           */
/********************************************************************/
void rm_center(int cid) {
int i,j; 

  for (i=0;i<TmpNpoints;++i) {
    if (TmpClusterId[i] == cid) break;
  }
  TmpClusterId[i] = NIL; 

  for (j=0;j<TmpNpoints;++j) {
    if (NearestCenter[j]==i) calc_NearestCenter(j);
  }
  
}
/********************************************************************/
/*  calculate the sink furthest away from all centers and boundary  */
/********************************************************************/
int calc_furthest_sink() {
int i, furthest=NIL;

  for (i=0;i<TmpNpoints;++i) {
    if (TmpClusterId[i] == NIL) {  /* Points[i] is not a center yet */
      if (furthest==NIL || cal_split(i) > cal_split(furthest) ) {
        furthest = i;
      } 
    }
  }
  return(furthest);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void build_k_center(int k) {
int i, j, counter;
double old_split, new_split; 

  TmpClusterId[0] = 0;
  NearestCenter[0] = NIL; 
  Points[0].t = DBL_MAX;
  for (i=1;i<TmpNpoints;++i) {
    TmpClusterId[i] = -1;
    NearestCenter[i] = 0; 
    Points[i].t = Point_dist(Points[i], Points[0]);
  }
  printf("X:(%.0f,%.0f) Y:(%.0f, %.0f)\n",MIN_x, MAX_x, MIN_y,MAX_y);
  for (i=1;i<k;++i) {
    j = calc_furthest_sink();
    add_center(i,j); 
    old_split = calc_min_split();
    printf("%d_th center: sink %d (%.1f, %.1f) --> min_split = %.0f\n", 
            i, j, Points[j].x, Points[j].y, old_split);
  }
  for (i=0;i<TmpNpoints;++i) {
    if (TmpClusterId[i] >= 0) {
      printf("dist(center %d, sink %d) =%.0f\n",
        NearestCenter[i], i, Points[i].t); 
    }
  }

  i = 0; 
  counter=0;
  old_split = calc_min_split(); 
  while (counter < k) {
    rm_center(i);
    j = calc_furthest_sink();
    add_center(i,j);
    new_split = calc_min_split();
    if (new_split > old_split + FUZZ) {
      printf("%d/%d:new_split = %.0f\n", counter, k, new_split);
      counter=0;
    } else {
      counter++;
    }
    i = (i+1)%k; 
    old_split = new_split; 
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void calc_cluster_center(int cid, TrrType *ms, PointType Points[]) {
int i;
TrrType temp;
double r;

  r = INT_MAX;
  TmpCluster[cid].n=0;
  for (i=0;i<TmpNpoints;++i) {
    if (cid == TmpClusterId[i]) {
      if (TmpCluster[cid].n==0) {
        ms->MakeDiamond(Points[i], r);
      } else {
        temp.MakeDiamond(Points[i], r);
        make_intersect(ms, &temp, ms); 
      }
      (TmpCluster[cid].n)++;
    }
  }
  make_core(ms, ms);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void print_cluster_info(int n_clusters, double ave_t, double max_t, 
                                        double min_t) {
int i;

  printf("\n\nn_sinks, cost\n"); 
  for (i=0;i<n_clusters;++i) {
    printf("TmpCluster[%d]: %d, %.1f ", i, TmpCluster[i].n, TmpCluster[i].t);
    if ((i+1)%2==0) printf("\n");
  }
}
/********************************************************************/
/*   cost = star model                                              */
/********************************************************************/
void calc_cost_statistics(int n_clusters, double *ave_t, double *max_t, 
                        double *min_t) {
int i;
double sum;

  sum    = 0;
  *max_t = -DBL_MAX;
  *min_t =  DBL_MAX;
  for (i=0;i<n_clusters;++i) {
    sum   += TmpCluster[i].t;
    *max_t = max (*max_t, TmpCluster[i].t);
    *min_t = min (*min_t, TmpCluster[i].t);
  }
  *ave_t  = sum /n_clusters;
  if (0) print_cluster_info(n_clusters, *ave_t, *max_t, *min_t); 
  printf("\n"); 
  printf("sum = %.1f, diff = %.1f (%.1f - %.1f), ave_t = %.1f\n", 
      sum, *max_t - *min_t, *max_t,*min_t,*ave_t); 
  printf("deviation = %E \n", calc_deviation(n_clusters, *ave_t)) ;
  printf("Obj_Cost = %E \n", calc_Obj_Cost(n_clusters, 5.0));
  printf("\n\n"); 
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void calc_star_center_cost(int cid) {
int i;
double dist;


  TmpCluster[cid].t = 0;
  TmpCluster[cid].capac = 0;
  
  for (i=0;i<TmpNpoints;++i) {
    if (cid== TmpClusterId[i]) {
      dist = pt2ms_distance(&(Points[i]),TmpCluster[cid].ms);
      TmpCluster[cid].t +=  PUCAP[H_]*dist + Capac[i];
      TmpCluster[cid].capac += Capac[i];
    }
  }
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
static
void calc_a_cluster_cost(int cid ) {
double dist;

  calc_cluster_center(cid, TmpCluster[cid].ms, Points);
  
  calc_star_center_cost(cid);
  

  dist = ms_distance(TmpCluster[cid].ms, &Center_ms);
  TmpCluster[cid].t += Weight*PUCAP[H_]*dist;;
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
static
void calc_clusters_cost(int n_clusters, double *ave_t, double *max_t, 
                        double *min_t) {
int i;
  for (i=0;i<n_clusters;++i) {
    calc_a_cluster_cost(i );
  }
  calc_cost_statistics(n_clusters, ave_t, max_t, min_t); 
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
int compare_cluster(const void *a, const void *b) {
int *p, *q;

   p = (int *) a;
   q = (int *) b;

   return( (TmpCluster[*p].t > TmpCluster[*q].t) ? YES: NO);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
void sort_cluster(int n_clusters, int cluster[]) {
int i;

  for (i=0;i<n_clusters;++i) {
    cluster[i] = i;
  }
  qsort(cluster, n_clusters, sizeof(int), compare_cluster);
  for (i=0;i<n_clusters-1;++i) {
    assert(TmpCluster[cluster[i]].t <= TmpCluster[cluster[i+1]].t);
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void calc_boundary(){
int i;

  MAX_x = MAX_y =  -DBL_MAX;
  MIN_x = MIN_y = DBL_MAX; 
  for (i=0;i<TmpNpoints;++i) {
    MAX_x = max (MAX_x, Points[i].x);
    MAX_y = max (MAX_y, Points[i].y);
    MIN_x = min (MIN_x, Points[i].x);
    MIN_y = min (MIN_y, Points[i].y);
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int int_min3(int x, int y, int z) {
  return( min ( min (x,y),z));
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int distinct(int cp, int cid, int k, int n,  PairType pair[]) {
int i, x;
double t, dist; 

  for (i=0;i<n;++i) {
    if (pair[i].y == k) {
      x = pair[i].x;
      dist =  Point_dist(Points[x], Points[k]);
      t= Point_dist(Points[cp],Points[k]);
      if (dist > t ) {
        pair[i].x = cp;
        pair[i].y = k;
      }
      return(NO);
    }
  }
  return(YES);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int in_small_clusters(int k, int cluster[]) {
int i, j;

  j = TmpClusterId[k];
  for (i=0;i<5;++i) {
    if ( j== cluster[i] ) return(YES);
  }
  return(NO);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int _pair_compare_inc1(const void *a, const void *b) {
PairType *p, *q;

   p = (PairType *) a;
   q = (PairType *) b;

   return( (p->cost > q->cost) ? YES: NO);
}
/********************************************************************/
/*  calc neighbors of cluster cid.                                  */
/********************************************************************/
int calc_cluster_neighbors(int cid, PairType pair[], TrrType *ms, int size,
int TmpNpoints, int TmpClusterId[], PointType Points[]) {
int i, n; 

  n = 0;
  for (i=0;i<TmpNpoints;++i) {
    if (TmpClusterId[i]!=cid) {
      pair[n].x = i;
      pair[n].y = cid;
      pair[n].cost = pt2ms_distance(&(Points[i]), ms);
      n++;
    }
  }
  assert(n>0);
  assert(n < MAX_N_NODES);
  qsort(pair, n, sizeof(PairType), _pair_compare_inc1);
  n = min (n, size);
  return(n); 
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
int _calc_all_cluster_neighbors(PairType pair[], int n_clusters) {
int i,j,m, n;
PairType temp[MAX_N_NODES];
TrrType ms;

  n = 0;
  for (i=0;i< n_clusters;++i) {
    calc_cluster_center(i,  &ms, Points);
    m = calc_cluster_neighbors(i,temp,&ms,5,TmpNpoints,TmpClusterId, Points);
    for (j=0;j<m;++j) {
      pair[n++] = temp[j];
    }
  }
  return(n);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int get_cluster_member(int cid, PointType points[]) {
int i,n=0;

  for (i=0;i<TmpNpoints;++i) {
    if (TmpClusterId[i]==cid) {
      points[n++] = Points[i];
    }
  }
  return(n);
}
/********************************************************************/
/* get the point in cluster cid which is closest to Points[k]       */
/********************************************************************/
int pt2cluster_dist(int k, int cid, double *min_dist) {
int i, min_i;
double d;

  *min_dist  = DBL_MAX;
  for (i=0;i<TmpNpoints;++i) {
    if (i!=k && TmpClusterId[i]==cid) {
      d = Point_dist(Points[k],Points[i]);
      *min_dist = min (*min_dist, d);
      min_i = i;
    }
  }
  return(min_i);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
double _pair_compare_inc2_sub(PairType *p) {
double cost, d1, d2;
PointType pt;

  pt2cluster_dist(p->x, p->y, &d1);
  pt2cluster_dist(p->x, TmpClusterId[p->x], &d2);
  core_mid_point(TmpCluster[p->y].ms, &pt);
  cost = d1-d2 + 0.9* Point_dist(Points[p->x], pt);                         

  return(cost);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int _pair_compare_inc2(const void *a, const void *b) {
double t1, t2;
PairType *p, *q;

   p = (PairType *) a;
   q = (PairType *) b;

   t1 = _pair_compare_inc2_sub(p);
   t2 = _pair_compare_inc2_sub(q);

   return( (t1 > t2) ? YES: NO);
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
void mark_cluster(int cid) {

  unsigned npoints = gBoundedSkewTree->Npoints() ;
  for (unsigned i=0;i<npoints;++i) {
    if ( TmpClusterId[i] == cid) {
      TmpMarked[i] = YES;
    }
  }
}
/********************************************************************/
/*  copy t2 to t1                                                   */
/********************************************************************/
void cp_TmpClusterType(TmpClusterType *t1, TmpClusterType *t2) {
  t1->n = t2->n;
  t1->id = t2->id;
  t1->marked = t2->marked;
  t1->capac = t2->capac;
  t1->t = t2->t;
  *(t1->ms) = *(t2->ms);
}
/*
  c1 = pt2ms_distance(&(Points[pair[0].x]), TmpCluster[cid].ms);
  for (i=1;i<n;++i) {
    c2 = pt2ms_distance(&(Points[pair[i].x]), TmpCluster[cid].ms);
    assert(c2 >= c1);
    if (c2 > 1.2*c1) {
      n = i+1;
      break;
    } 
  }
*/
/********************************************************************/
/*                                                                  */
/********************************************************************/
static
int expand_cluster(int n_clusters, int cid, double ave_t) {
int i,j,k, n, cid2; 
PairType pair[MAX_N_NODES];
double min_c, t;

  n = calc_cluster_neighbors(cid, pair, TmpCluster[cid].ms, 5,TmpNpoints,TmpClusterId, Points);
  qsort(pair, n, sizeof(PairType), _pair_compare_inc2); 

  printf("Expand cluster %d (cost=%.1f):\n", cid, TmpCluster[cid].t); 
  for (i=k=0; i < n; i++) {
    j = pair[i].x;
    cid2 = TmpClusterId[j];
    assert(cid!=cid2);
    
      cp_TmpClusterType(&Tmp_x_Cluster, &(TmpCluster[cid]));
      cp_TmpClusterType(&Tmp_y_Cluster, &(TmpCluster[cid2]));
      min_c = min (TmpCluster[cid].t, TmpCluster[cid2].t);
      TmpClusterId[j] = cid; 
      calc_a_cluster_cost(cid);
      calc_a_cluster_cost(cid2);

      t =  calc_Obj_Cost(n_clusters, 5.0);
      if ( Obj_Cost >= t) {
        printf("%d: add sink %d (from cluster %d) -> Cost=%.1f \n",
                i, j, cid2, TmpCluster[cid].t);
        k++;
        Obj_Cost = t;
      } else {
        TmpClusterId[j] = cid2; 
        cp_TmpClusterType(&(TmpCluster[cid]), &Tmp_x_Cluster); 
        cp_TmpClusterType(&(TmpCluster[cid2]), &Tmp_y_Cluster); 
      }
    
    if (k>=Expand_Size) break;
  }
  return(k);
}

/***********************************************************************/
/* print the pointset */
/***********************************************************************/
void print_clusters_sub(FILE *f, int n_clusters, PointType center[]) {
int i, j, n;
double x,y;

  fprintf(f, "\n");
  for (j=0;j<n_clusters;++j) {
    for (i=n=0;i<TmpNpoints;++i) { if (TmpClusterId[i]==j) { n++;} }
    fprintf(f, "\"%d:%d\n", j,n);
    for (i=0; i< TmpNpoints; i++) {
      if (TmpClusterId[i]==j) {
        x = Points[i].x;
        y = Points[i].y;
        fprintf(f, "move %f  %f cluster_id:%d\n", x, y, TmpClusterId[i]);
        fprintf(f, "     %f  %f \n", x, y);
      }
    }
    fprintf(f, "\n");
  }
  for (i=0;i<n_clusters;++i) {
     fprintf(f, "move %f %f sink %d \n", center[i].x, center[i].y, i);
     fprintf(f, "     %f %f \n", center[i].x, center[i].y);
  }
}

/********************************************************************/
/*                                                                  */
/********************************************************************/
void print_clusters(int n_clusters, PointType center[]) {
FILE *f;
char *a = "t";

  f = fopen(a,"w");
  assert(f != NULL);
  print_clusters_sub(f, n_clusters, center);
  fclose(f);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
int get_n_below_ave_t(int n, int cluster[], double ave_t) {
int i, cid;

  for (i=0;i<n;++i) {
    cid = cluster[i];
    if (equal(TmpCluster[cid].t, ave_t)) return(i+1);
    if (TmpCluster[cid].t > ave_t) return(i);
  }
  assert(0==1);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
static
void local_improvement(int n_clusters) {
int i,m, n, cid, counter, done, cluster[MAX_N_SINKS];
int n_move =0;
double ave_t, max_t,min_t;

  calc_clusters_cost(n_clusters, &ave_t, &max_t, &min_t);

  counter =0;
  done = NO;
  Obj_Cost = calc_Obj_Cost(n_clusters, 5.0);
  printf("starting Obj_Cost = %f \n", Obj_Cost);
  while ( !done) {
    for (i=0;i<n_clusters; ++i) { TmpCluster[i].marked = NO; }
    for (i=0;i<TmpNpoints;++i) { TmpMarked[i] = NO; }
    sort_cluster(n_clusters, cluster); 
    n =0; 
    for (i=0;i<n_clusters - 1;++i) {
      cid = cluster[i];
      
      printf("**************************************\n");
      printf("counter=%d: , i = %d (n_clusters=%d)\n", counter, i, n_clusters);
      printf("**************************************\n");
      TmpCluster[cid].marked = YES;
      m = expand_cluster(n_clusters, cid,ave_t);
      assert(m<=Expand_Size);
      if (m >0) {
        calc_clusters_cost(n_clusters, &ave_t, &max_t, &min_t);
        n = n + m;
      }
      assert(n>=m);
    }
    counter++; 
    if (n==0 || counter >= 50 ) {
      done = YES;
    }
    n_move += n;
  } 
  printf("================================================================\n");
  printf("total_n_move = %d (Expand_Size=%d)\n", n_move, Expand_Size);
  printf("================================================================\n");
  fflush(stdout);
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void check_cluster_size() {
int i, cid;

  for (i=0;i<TmpNpoints;++i) {
    TmpCluster[i].n = 0;
  }
  for (i=0;i<TmpNpoints;++i) {
    cid = TmpClusterId[i];
    (TmpCluster[cid].n)++;
  }
  for (i=0;i<TmpNpoints;++i) {
    if (TmpCluster[i].n <=1 ) {
      printf("TmpCluster[%d].n = %d \n", i, TmpCluster[i].n);
    }
    assert(TmpCluster[i].n >= 0) ;
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void k_center(int n_clusters) {
int i,j, cid;
double ave_t, max_t,min_t, t;
PointType center[MAX_N_SINKS]; 

  calc_boundary(); 
  build_k_center(n_clusters);
  for (i=0;i<TmpNpoints;++i) {
    cid = TmpClusterId[i];
    if (cid>=0) { 
      TmpCluster[cid].id = i; 
      assert(cid<n_clusters);
    }
  }

  for (i=0;i<TmpNpoints;++i) {
    cid = TmpClusterId[i];
    if (cid<0) {
      j = NearestCenter[i];
      TmpClusterId[i] = TmpClusterId[j];
      assert(TmpClusterId[i]>=0); 
    }
  }

  calc_clusters_cost(n_clusters, &ave_t, &max_t, &min_t);
  for (i=0;i<n_clusters; ++i) {
    core_mid_point(TmpCluster[i].ms, &(center[i])); 
  }
  for (i=0;i<TmpNpoints;++i) {
    Points[i].t = DBL_MAX;
    for (j=0;j<n_clusters; ++j) {
      t = Point_dist(Points[i], center[j]);
      if (Points[i].t > t) {
        Points[i].t = t;
        TmpClusterId[i] = j; 
      }
    }
  }
  check_cluster_size();
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void do_skew_allocation(int start_id, int n_clusters) {
int i;

    for (i=0;i<n_clusters;++i) {
      Skew_B_CLS[start_id + i] = Skew_B;
    }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
static
void do_clustering(int n, int start_id,int n_clusters) {
int i, j;
TrrType ms;

  if (n_clusters<=1) {
    for (i=0;i<n;++i) {
      Cluster_id[TreeRoots[i]] =  start_id;
    }
  } else {
    TmpNpoints = n;
    int nterms = (int) gBoundedSkewTree->Nterms() ;
    for (i=0;i<n;++i) {
      j = TreeRoots[i];
      build_NodeTRR(&(Node[j]));
      make_core(Node[j].ms, &ms);
      core_mid_point(&ms, &(Points[i]));
      Capac[i] = Node[j].area[0].capac;
      assert(Capac[i] > FUZZ);
      if (j > nterms) assert(equal(Capac[i],C_buffer));
    }
    get_the_center(0,TmpNpoints-1,Points, &Center_ms);
   
    k_center(n_clusters);
    if (Expand_Size > 0) {
      local_improvement(n_clusters);
    }
    for (i=0;i<n;++i) {
      Cluster_id[TreeRoots[i]] =  start_id + TmpClusterId[i];
    }
    do_skew_allocation(start_id, n_clusters);
  }
}
/****************************************************************************/
/* group n TreeRoots[] to n_clusters, with cluster ID begins with start_id  */
/****************************************************************************/
static
void assign_Given_Hierachy_Cluster_id(int n, int start_id,int n_clusters) {
int i, j, cid;

  for (i=0;i<n;++i) {
    cid = Cluster_id[TreeRoots[i]];
    Cluster_id[TreeRoots[i]] =  j = Hierachy_Cluster_id[cid];
    assert(j >= start_id);
    assert(j < start_id+n_clusters);
  }
}
/********************************************************************/
/*                                                                  */
/********************************************************************/
void init_clusters(int L, int PostOpt) {
int n = N_Clusters[L-1];
int start_id = Total_CL;
int n_clusters = N_Clusters[L];

  assert ( PostOpt == NO ) ;
  
    if (Hierachy_Cluster_id[0] > 0) {
      if ( L > 1 ) {
        assign_Given_Hierachy_Cluster_id(n, start_id, n_clusters);
      }
    } else {
      do_clustering(n, start_id, n_clusters);
    }
  
}
