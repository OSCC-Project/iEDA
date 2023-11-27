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
#include "bst_sub3.h"

#define NN   0
#define SS   1
#define EE   2
#define WW   3


int N_S = 0, N_S0=0;                   /* Number of detour points set  */
ObstacleType *Obstacle, *Tmp_Obstacle;    /* which are rectangles */
double **ObsBox;
PointType *S, *S0;   /* Set of detour points, which are obstacle vertices */ 
int *S_Par;
int *S_N_Hops;
int *In_Tree;
double *S_Cost;
TreeNodeType   *TreeNode;
int BufferArea[2];

extern int on_line_segment(double ,double ,double ,double ,double ,double );
extern int lineIntersect(PointType *,PointType,PointType,PointType,PointType);
extern int same_Point(PointType , PointType );
extern  double Point_dist(PointType ,PointType);
extern int bbox_overlap(double,double,double,double,double,double,double,double);
extern int _bbox_overlap(double,double,double,double,double,double,double,double, double);
extern void build_trr(TrrType *ms,double d,TrrType *trr); 
extern void make_intersect( TrrType *trr1, TrrType *trr2, TrrType *t );
extern void line2ms(TrrType *ms, PointType p1, PointType p2);
extern void core_mid_point(TrrType *trr, PointType *p);
extern int ms_type(TrrType *trr);
extern double ms_distance(TrrType *ms1,TrrType *ms2);
extern double pt2linedist(PointType p1, PointType p2, PointType p3, PointType *ans);

extern double linedist(PointType lpt0,PointType lpt1, PointType lpt2, 
              PointType lpt3, PointType pt[2]);
extern int Area_compare_inc(const void  *p1, const void  *q1);
extern int pt_on_line_segment(PointType pt,PointType pt1,PointType pt2);

extern int equal(double x,double y);
extern double pt_skew(PointType pt) ;
extern int calc_line_type(PointType pt1,PointType pt2);
extern void calc_vertices(AreaType *area);

extern void calc_pt_delays(AreaType *area, PointType *q1,PointType q0,
       PointType q2); 
extern void calc_BS_located(PointType *pt,AreaType *area, PointType *p1, 
       PointType *p2);
extern void calc_JS_delay(AreaType *area, AreaType *area_L,AreaType *area_R);


/********************************************************************/
/*                                                                  */
/********************************************************************/
double Point_E_dist(PointType p1, PointType p2) {
double a, b;

  a = ABS(p1.x-p1.x);
  b = ABS(p2.y-p2.y);
  return(sqrt(a*a+b*b));
}


/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int cp_PointType_array(PointType a[], int n, PointType b[]) {
int i;

  for (i=0;i<n;++i) {
    b[i] = a[i];
  }
  return(n);
}
/***********************************************************************/
/*    cp a[] to b[]                                                    */
/***********************************************************************/
int cp_area_array(AreaType a[], int n, AreaType b[]) {
int i;

  for (i=0;i<n;++i) {
    b[i] = a[i];
  }
  return(n);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void append_area_array(AreaType a[], int n, AreaType b[], int *m) {
int i;

  for (i=0;i<n;++i) {
    b[*m+i] = a[i];
  }
  *m  = *m + n; 
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void check_area(AreaType *area) {
int i, j, k1, k2, n, line0type, line1type;
PointType q0, q2, p1, p2;
double a, b;

  assert(area->n_mr <= MAX_mr_PTS);
  if (Skew_B == 0) assert(area->n_mr <=2);
  line0type = calc_line_type(area->line[0][0],area->line[0][1]);
  line1type = calc_line_type(area->line[1][0],area->line[1][1]);
  assert(line0type == line1type);

  n = area->n_mr;
  for (i=0;i<n;++i) {
    j = (i+1)%n;
    q0 = area->mr[i];
    q2 = area->mr[j];
    a = ABS(q0.x-q2.x);
    b = ABS(q0.y-q2.y);
    if ( !equal(a,0) && !equal(b,0) ) {
      /* no skew reservation */
        if (equal(pt_skew(q0),Skew_B) == NO || equal(pt_skew(q2),Skew_B)==NO){
          print_area(area);
          assert(0);
        }
    }
    if (line0type == MANHATTAN_ARC) {
      assert(equal(a,0) || equal(b,0) || equal(a,b) );
    }
  }
  n = area->npts;
  for (i=0;i<n;++i) {
    k1 = area->vertex[i];
    k2 = area->vertex[(i+1)%n];
    p1 = area->mr[k1];
    p2 = area->mr[k2];
    for (j=k1+1; j < k2; ++j) {
      assert(pt_on_line_segment(area->mr[j], p1,p2));
    }
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void check_area_array(AreaType area[], int n) {
int i;

  for (i=0;i<n;++i) {
    check_area(&(area[i]));
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int check_area_line(AreaType *area) {
int  linetype;

  linetype = calc_line_type(area->line[0][0],area->line[0][1]);

  if (linetype==VERTICAL ) {
    assert(area->line[0][0].y > area->line[0][1].y);
    assert(area->line[1][0].y > area->line[1][1].y);
  } else if (linetype==HORIZONTAL) {
    assert(area->line[0][0].x > area->line[0][1].x);
    assert(area->line[1][0].x > area->line[1][1].x);
  }
  return(linetype);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void check_all_areas_line(AreaType area[], int n) {
int i;
 
  for (i=0;i<n;++i) {
    check_area_line(&(area[i]));
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
double calc_subpath_length(PointType path[], int k1, int k2, int mode) {
int i;
double cost;

  cost = 0;
  for (i=k1;i<k2;++i) {
    if (mode==MANHATTAN) {
      cost +=  Point_dist(path[i],path[i+1]);
    } else {
      cost +=  Point_E_dist(path[i],path[i+1]);
    }
  }
  return(cost);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
double calc_pathlength(PointType path[], int n, int mode) {
double cost;

  cost = calc_subpath_length(path, 0, n-1, mode);
  return(cost);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void reverse_mr(AreaType *area) {
int i, n;
PointType tmp_Pt;

  n = area->n_mr;
  for (i=0;i<n/2;++i) {
    tmp_Pt = area->mr[i];
    area->mr[i] = area->mr[n-1-i];
    area->mr[n-1-i] = tmp_Pt;
  }
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void get_block_bbox(double bbox[], ObstacleType *block) {
int i;

  bbox[EE]=bbox[NN] = -DBL_MAX;
  bbox[WW]=bbox[SS] =  DBL_MAX;
  for (i=0;i<block->n_vertex;++i) {
    bbox[EE] = max (bbox[EE], block->vertex[i].x);
    bbox[WW] = min (bbox[WW], block->vertex[i].x); 
    bbox[NN] = max (bbox[NN], block->vertex[i].y); 
    bbox[SS] = min (bbox[SS], block->vertex[i].y); 
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void modify_ms_JS(AreaType *area) {
int i, line0type, line1type;
TrrType ms0,ms1,trr0,trr1,t0,t1;

  line0type = calc_line_type(area->line[0][0],area->line[0][1]);
  line1type = calc_line_type(area->line[1][0],area->line[1][1]);
  assert(line0type == line1type);
  assert(line0type == MANHATTAN_ARC);

  if ( same_Point(area->line[0][0],area->line[0][1]) && 
       same_Point(area->line[1][0],area->line[1][1])  )  return;

  line2ms(&ms0,area->line[0][0],area->line[0][1]);
  line2ms(&ms1,area->line[1][0],area->line[1][1]);

  assert(equal(ms_distance(&ms0,&ms1), area->dist));

  build_trr(&ms0, area->dist, &trr0);
  build_trr(&ms1, area->dist, &trr1);
  make_intersect(&trr1,&ms0,&t0);
  make_intersect(&trr0,&ms1,&t1);
  ms2line(&t0, &(area->line[0][1]), &(area->line[0][0]));
  ms2line(&t1, &(area->line[1][1]), &(area->line[1][0]));
  assert(area->line[0][0].y >= area->line[0][1].y);
  assert(area->line[1][0].y >= area->line[1][1].y);
  assert(area->n_mr <= 2);
  if (area->n_mr==2&&(area->L_EdgeLen == 0 || area->R_EdgeLen==0)) {
    assert(equal(area->mr[0].max, area->mr[1].max));
    assert(equal(area->mr[0].min, area->mr[1].min));
    if ( area->L_EdgeLen==0 ) {
      i = 0;
    } else {
      i = 1;
    }
    assert(pt_on_line_segment(area->line[i][0], area->mr[0], area->mr[1]));
    assert(pt_on_line_segment(area->line[i][1], area->mr[0], area->mr[1]));
    area->mr[0].x = area->line[i][0].x;
    area->mr[0].y = area->line[i][0].y;
    if (same_Point(area->line[i][0],area->line[i][1])) {
      area->n_mr = area->npts = 1;
    } else {
      assert(area->npts == 2);
      area->mr[1].x = area->line[i][1].x;
      area->mr[1].y = area->line[i][1].y;
    }
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void get_JR_bbox(double bbox[], AreaType *area) {
int i, j; 

  bbox[EE]=bbox[NN] = -DBL_MAX;
  bbox[WW]=bbox[SS] =  DBL_MAX;

  for (i=0;i<2;++i) {
    for (j=0;j<2;++j) {
      bbox[EE] = max (bbox[EE], area->line[i][j].x);
      bbox[WW] = min (bbox[WW], area->line[i][j].x);
      bbox[NN] = max (bbox[NN], area->line[i][j].y);
      bbox[SS] = min (bbox[SS], area->line[i][j].y);
    }
  }
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void get_mr_bbox(double bbox[], AreaType *area) {
int i;

  bbox[EE]=bbox[NN] = -DBL_MAX;
  bbox[WW]=bbox[SS] =  DBL_MAX;

  for (i=0;i<area->n_mr;++i) {
    bbox[EE] = max (bbox[EE], area->mr[i].x);
    bbox[WW] = min (bbox[WW], area->mr[i].x);
    bbox[NN] = max (bbox[NN], area->mr[i].y);
    bbox[SS] = min (bbox[SS], area->mr[i].y);
  }
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int JS_ms_type(AreaType *area) {
int i, j, line0type, line1type;
TrrType ms0,ms1;
PointType p0, p1;
double x,y;

  line0type = calc_line_type(area->line[0][0],area->line[0][1]);
  line1type = calc_line_type(area->line[1][0],area->line[1][1]);

  assert(line0type==MANHATTAN_ARC && line1type==MANHATTAN_ARC);

  line2ms(&ms0,area->line[0][0],area->line[0][1]);
  line2ms(&ms1,area->line[1][0],area->line[1][1]);

  i = ms_type(&ms0);
  j = ms_type(&ms1);
  assert(i*j ==0 || i*j == 1);
  if (i==0 && j==0) {
    core_mid_point(&ms0, &p0);
    core_mid_point(&ms1, &p1);
    x= p0.x - p1.x;
    y= p0.y - p1.y;
    if (equal(x,0) || equal(y,0)) {
      return(0);
    } else if (x*y > 0) {
      return(-1);
    } else {
      return(1);
    }
  } else if (i!=0) {
    return(i);
  } else if (j!=0) {
    return(j);
  }
  assert (0) ;
    return(0);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void calc_N_S0() {
int i;

  N_S0=0;
  for (i=0;i<N_Obstacle;++i) {
    N_S0 += Obstacle[i].n_vertex;      
  }

  
  S = (PointType *) calloc(N_S0 + 2, sizeof(PointType));
  assert(S   != NULL);

  S0 = (PointType *) calloc(N_S0 + 2, sizeof(PointType));
  assert(S0   != NULL);

  S_Par = (int *) calloc(N_S0 + 2, sizeof(int));
  assert(S_Par   != NULL);

  S_N_Hops = (int *) calloc(N_S0 + 2, sizeof(int));
  assert(S_N_Hops   != NULL);

  In_Tree = (int *) calloc(N_S0 + 2, sizeof(int));
  assert(In_Tree   != NULL);

  S_Cost = (double *) calloc(N_S0 + 2, sizeof(double));
  assert(S_Cost   != NULL);

}

/***********************************************************************/
/* S[N_S] = set of obstacle vertices as detour points             */
/***********************************************************************/
void calc_S() {
int i,j,k,n;

  calc_N_S0();
  n=0;
  for (i=0;i<N_Obstacle;++i) {
    k = Obstacle[i].n_vertex;
    for (j=0; j<k;++j) {
      S[n] = S0[n] = Obstacle[i].vertex[j];
      n++;      
    }
  }
  assert(n==N_S0);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void init_S(PointType p1, PointType p2, double range) {
int i;
double dist;

  N_S = 0;
  for (i=0;i<N_S0;++i) {
    dist = Point_dist(p1,S0[i]) + Point_dist(p2,S0[i]);
    if (dist <= range) {
      S[N_S++] = S0[i];
    }
  }
  S[N_S++] = p1;
  S[N_S++] = p2;
  assert(N_S <= N_S0 + 2);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_S() {
int i;
FILE *f;

  f = fopen("t", "w");
  assert(f != NULL);

  fprintf(f, "\"S \n");
  for (i=0;i<N_S;++i) {
    fprintf(f, "move %f %f %d \n", S[i].x, S[i].y, i);
    fprintf(f, "     %f %f \n", S[i].x, S[i].y);
  }
  fprintf(f, "\n\"p1p2 \n");
  fprintf(f, "move %f %f \n", S[N_S-2].x, S[N_S-2].y);
  fprintf(f, "     %f %f \n", S[N_S-1].x, S[N_S-1].y);
  fclose(f);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int block_compare_inc1(const void  *p1, const void  *q1) {
int  *p, *q;
double a[4], b[4];

  p = (int  *)p1;
  q = (int  *)q1;

  get_block_bbox(a, &(Obstacle[*p]));
  get_block_bbox(b, &(Obstacle[*q]));
  return( (a[WW] > b[WW])?YES:NO );
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int block_compare_inc2(const void  *p1, const void  *q1) {
int  *p, *q;
double a[4], b[4];

  p = (int  *)p1;
  q = (int  *)q1;

  get_block_bbox(a, &(Obstacle[*p]));
  get_block_bbox(b, &(Obstacle[*q]));
  return( (a[SS] > b[SS])?YES:NO );
}



/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void construct_TreeNode(int par, int *index, int obstacle[], int n_obstacle, int dir) {
int i, j, n1, n2, b[1000], cp;

  if (n_obstacle<1)  return;

  (*index)++;
  cp = *index;
  assert(cp>=0 && cp <2*N_Obstacle-1);
  TreeNode[cp].parent = par;  
  
  if (par >=0) {
    TreeNode[cp].level = TreeNode[par].level + 1;
    if (TreeNode[par].L==NIL) {
      TreeNode[par].L = cp;
    } else {
      assert(TreeNode[par].R == NIL);
      TreeNode[par].R = cp;
    }
  }
  TreeNode[cp].bbox[EE] = TreeNode[cp].bbox[NN] = -DBL_MAX;
  TreeNode[cp].bbox[WW] = TreeNode[cp].bbox[SS] =  DBL_MAX;
  for (i=0;i<n_obstacle;++i) {
/*
    get_block_bbox(obox, &(Obstacle[obstacle[i]]));
*/
    j = obstacle[i];
    TreeNode[cp].bbox[EE] =  max (TreeNode[cp].bbox[EE], ObsBox[j][EE]);
    TreeNode[cp].bbox[NN] =  max (TreeNode[cp].bbox[NN], ObsBox[j][NN]);
    TreeNode[cp].bbox[WW] =  min (TreeNode[cp].bbox[WW], ObsBox[j][WW]);
    TreeNode[cp].bbox[SS] =  min (TreeNode[cp].bbox[SS], ObsBox[j][SS]);
  }
  if (n_obstacle==1)  {
    TreeNode[cp].id = obstacle[0];
    if (CHECK) printf("TreeNode[%d].level = %d \n", cp, TreeNode[cp].level);
  } else {
    if (dir == 0) {
      qsort(obstacle, n_obstacle, sizeof(int), block_compare_inc1); 
    } else {
      qsort(obstacle, n_obstacle, sizeof(int), block_compare_inc2); 
    }
         
    n1 = n_obstacle/2;
    n2 = n_obstacle - n1;
    assert(n2 < 1000);
    for (i=0;i< n2;++i) {
      b[i] = obstacle[n1+i];
    }
    
    construct_TreeNode(cp, index, obstacle, n1, 1-dir);
    construct_TreeNode(cp, index, b, n2, 1-dir);
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void init_construct_TreeNode() {
int i;
int *a;
  
  i = -1;

  a = (int *) calloc(N_Obstacle, sizeof(int));
  assert(a != NULL);
  for (i=0;i<N_Obstacle;++i) {
    a[i] = i;
  }

  TreeNode = (TreeNodeType *) calloc(2*N_Obstacle, sizeof(TreeNodeType));
  assert(TreeNode != NULL);
 
  for (i=0;i<2*N_Obstacle;++i) {
    TreeNode[i].L = TreeNode[i].R = TreeNode[i].parent = TreeNode[i].id = NIL;
    TreeNode[i].level= 0;
  }

  i = NIL;
  construct_TreeNode(NIL, &i, a, N_Obstacle,0);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int unblocked_segment_sub1(PointType *p0, PointType *p1, int i) {
int j, k, ans ;

  k = _bbox_overlap(p0->x,p0->y,p1->x,p1->y, TreeNode[i].bbox[WW],  
      TreeNode[i].bbox[SS],  TreeNode[i].bbox[EE],  
      TreeNode[i].bbox[NN], 0);

  if (k==NO) return(YES);

  j = TreeNode[i].id;
  if (j==NIL) { /* internal node */
    ans = unblocked_segment_sub1(p0,p1,TreeNode[i].L);
    if (ans==YES) {
      ans = unblocked_segment_sub1(p0,p1,TreeNode[i].R);
    }
  } else {
    assert(TreeNode[i].L==NIL && TreeNode[i].R==NIL);
    ans = 1 - line_into_rectangle(p0, p1, &(Obstacle[j]));
  }
  return(ans);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void read_obstacle_file_sub(FILE *f) {
int i,j,n;
double x, y;

  fscanf(f, "%d \n", &N_Obstacle);

  Obstacle = (ObstacleType *) calloc(N_Obstacle, sizeof(ObstacleType));
  assert(Obstacle != NULL);

  Tmp_Obstacle = (ObstacleType *) calloc(N_Obstacle, sizeof(ObstacleType));
  assert(Tmp_Obstacle != NULL);

  ObsBox = (double **) calloc(N_Obstacle, sizeof(double *));
  assert(ObsBox != NULL);
  for (i=0; i<N_Obstacle; i++) {
    ObsBox[i]=(double *) calloc(4,sizeof(double));
    assert(ObsBox[i] != NULL);
  }


   i = 0;
   while (fscanf(f, "%d:  \n", &n) ==1 ) {
     Obstacle[i].n_vertex = n;
     for (j=0;j<n;++j) {
       assert( 2 == fscanf(f, "     %lf  %lf \n", &x,&y));
       Obstacle[i].vertex[j].x = x;
       Obstacle[i].vertex[j].y = y;
       
     }
     i++;
   }
   assert(i==N_Obstacle);


  for (i=0; i<N_Obstacle; i++) {
    get_block_bbox(ObsBox[i], &(Obstacle[i]));
  }
  init_construct_TreeNode(); 
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void read_obstacle_file( const string &fn) {

  FILE *f = fopen(fn.c_str(), "r");
  assert(f != NULL);
  read_obstacle_file_sub(f);
  calc_S();
  fclose(f);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int distinct_obstacle(int n) {
int i;
double x1,x2,x3,x4,y1,y2,y3,y4;

  x1 = Obstacle[n].vertex[0].x;
  y1 = Obstacle[n].vertex[0].y;
  x2 = Obstacle[n].vertex[2].x;
  y2 = Obstacle[n].vertex[2].y;
  for (i=0;i<n;++i) {
    x3 = Obstacle[i].vertex[0].x;
    y3 = Obstacle[i].vertex[0].y;
    x4 = Obstacle[i].vertex[2].x;
    y4 = Obstacle[i].vertex[2].y;
    if (bbox_overlap(x1,y1,x2,y2,x3,y3,x4,y4)) {
      return(NO);
    }
  }
  return(YES);
}

/***************************************************************************/
/*  produce the the blockage.                                              */
/***************************************************************************/
void generate_obstacle_file(char fn[], int n_blocks, int max_coor) {
FILE *out;
double x,y, w,h;
int i, j, max_block_size;
char a[100], *b = ".block";

  N_Obstacle = n_blocks;
  Obstacle = (ObstacleType *) calloc(N_Obstacle, sizeof(ObstacleType));
  assert(Obstacle != NULL);

  strcpy(a, fn);
  strcat(a,b);

  out = fopen(a,"w");
  assert(out != NULL);


  fprintf(out,"%d  \n", n_blocks);
  srand48(1);
  max_block_size = max_coor/20;
  i = 0; 
  while (i<n_blocks) {
    x=rand()%max_coor;
    y=rand()%max_coor;
    w= max_block_size/6 + rand()%max_block_size;
    h= max_block_size/6 + rand()%max_block_size;
    Obstacle[i].n_vertex = 4;
    Obstacle[i].vertex[0].x = x - w;
    Obstacle[i].vertex[1].x = x - w;
    Obstacle[i].vertex[2].x = x + w;
    Obstacle[i].vertex[3].x = x + w;
    Obstacle[i].vertex[0].y = y - h;
    Obstacle[i].vertex[1].y = y + h;
    Obstacle[i].vertex[2].y = y + h;
    Obstacle[i].vertex[3].y = y - h;
    if (distinct_obstacle(i)) {
      fprintf(out,"4: \n");
      for (j=0;j<4;++j) {
        x = Obstacle[i].vertex[j].x;
        y = Obstacle[i].vertex[j].y;
        fprintf(out,"    %f  %f \n",x,y);
      }
      ++i; 
    }
  }
  fclose(out);
}


/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int pt_outside_a_rectangle(PointType *pt, ObstacleType *block) {
double bbox[4];


  get_block_bbox(bbox, block);
  if (equal(pt->x,bbox[EE]) || equal(pt->x,bbox[WW]) ) {
    if (pt->y <= bbox[NN] + FUZZ && pt->y >= bbox[SS] - FUZZ ) {
      return(ON);
    } else {
      return(OUTSIDE);
    }
  }
  if (equal(pt->y,bbox[NN]) || equal(pt->y,bbox[SS]) ) {
    if (pt->x <= bbox[EE] + FUZZ && pt->x >= bbox[WW] - FUZZ ) {
      return(ON);
    } else {
      return(OUTSIDE);
    }
  }
  if (pt->x <= bbox[EE] && pt->x >= bbox[WW] && 
      pt->y <= bbox[NN] && pt->y >= bbox[SS]) {
    return(INSIDE);
  } else {
    return(OUTSIDE);
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int pt_outside_rectangles(PointType  *pt) {
int i, j;

  for (i=0;i<N_Obstacle;++i) {
    j = pt_outside_a_rectangle(pt, &(Obstacle[i]));
    if (j==0 || j==-1)  return(NO);
  }
  return(YES);
}

/***************************************************************************/
/*  produce the test cases.                                                */
/***************************************************************************/
static void 
generate_terminals(char fn[], int max_coor, int n_sinks) {
FILE *out;
int i;
PointType pt;

  generate_obstacle_file(fn, 40, max_coor);

  out = fopen(fn,"w");
  assert(out != NULL); 

  long current_time;
  time(&current_time);
  // string timestamp = U_Printtf("%s",ctime(&current_time));
  string timestamp = ctime(&current_time);

  string msg = "# UCLA clock benchmark 1.0" ; 
  msg += "# Created       : " + timestamp  ; 
  msg += "\n" ;
  msg += "# User          : tsao@cs.ucla.edu"  ; 
  msg += "\n" ;
  msg += "# PlatForm      : SunOS 5.7 sparc SUNW,Ultra-1"  ; 
  msg += "\n" ;
  msg += "# Source        : Randomly generated testcases "   ; 
  msg += "\n" ;
  msg += "# Note          : Coordinate unit can be micro-meter or anything" ;
  msg += "\n" ;


  fprintf(out,"%s  \n\n", msg.c_str() );
  fprintf(out,"NumPins : %d  \n\n", n_sinks);
  fprintf(out,"PerUnitResistance : 0.016600  Ohm \n\n" );
  fprintf(out,"PerUnitCapacitance : .27000e-17  Farad \n\n" );
  // fprintf(out,"%d  0.016600  2.7000e-17 \n", n_sinks);
  srand48(1);
  i=0; 
  while (i<n_sinks) {
    pt.x = rand()%max_coor;
    pt.y = rand()%max_coor;
    if (pt_outside_rectangles(&pt)) {
      fprintf(out,"Sink : %d  \n",i );
      fprintf(out,"     Coordinate : %d  %d \n",(int) pt.x,(int) pt.y);
      fprintf(out,"     Capacitive Load : 5.0e-13\n"     );
      fprintf(out,"     Downstream_Delay: 0.0e-12\n"     );
      // fprintf(out,"%d  %d  %d 5.0e-13\n",i, (int) pt.x,(int) pt.y);
      i++;
    }
  }
  fclose(out);
  exit(0);
}

/**********************************************************************/
/* Copyright (C) 1994-2000  by Andrew B. Kahng, C.-W. Albert Tsao    */
/**********************************************************************/
bool
BST_GenTestcase::GenerateTestcase ( unsigned i ) {
  if ( i > 0 ) {
    char fn[100] ;
    sprintf(fn,"n%d", i);
    generate_terminals(fn, 1000, i);
  }

  return i > 0 ;

}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int distinct_Point(PointType q[], int n) {
int i;

  for (i=0;i<n;++i) {
    if (same_Point(q[n],q[i])) {
      return(NO);
    }
  }
  return(YES);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int line_x_rectangle(PointType *p0, PointType *p1, PointType q[], 
                     ObstacleType *block){
PointType v0, v1;
int ans; 
int i, n, n_xing=0;

  n = block->n_vertex;
  for (i=0;i<n;++i) {
    v0 = block->vertex[i];
    v1 = block->vertex[(i+1)%n];
    ans = lineIntersect(&(q[n_xing]), *p0,*p1,v0,v1);
    if (ans == SAME_EDGE || ans == OVERLAPPING) {
      return(0);
    } else if (ans >0 && distinct_Point(q,n_xing)) {
      q[n_xing].max = p0->max;
      q[n_xing].min = p0->min;
      n_xing++;
    }
  }
  assert(n_xing <= 2);
  return(n_xing);
}
/***********************************************************************/
/*  check if line p0,p1 goes into rectangle[k]                         */
/***********************************************************************/
int line_into_rectangle(PointType  *p0, PointType *p1, ObstacleType *block) {
int ans, loc0, loc1, n_xing;
PointType q[10];

  ans = bbox_overlap(p0->x,p0->y,p1->x,p1->y,
                     block->vertex[0].x,block->vertex[0].y,
                     block->vertex[2].x,block->vertex[2].y);

  if (ans==NO) return(NO);
  loc0 = pt_outside_a_rectangle(p0, block);
  loc1 = pt_outside_a_rectangle(p1, block);

  if (loc0 == -1 || loc1 == -1) {  /* p0 or p1 is inside rectangle */
    return(YES);
  } else if (same_Point(*p0,*p1)) { /* p0, p1 either on boundary or outside */
    return(NO);
  } else { 
    n_xing = line_x_rectangle(p0,p1,q,block);
    if (n_xing==2) {
      return(YES);
    } else {
      return(NO);
    }
  }

}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int unblocked_segment_sub2(PointType *p0, PointType *p1) {
int i;
double bbox[4];


  for (i=0;i<N_Obstacle;++i) {
    get_block_bbox(bbox, &(Obstacle[i]));
    if (_bbox_overlap(p0->x,p0->y,p1->x,p1->y,
        bbox[WW], bbox[SS], bbox[EE], bbox[NN], FUZZ)) {
      if (line_into_rectangle(p0, p1, &(Obstacle[i])) ) {
        return(NO);
      }
    }
  }
  return(YES);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int unblocked_segment(PointType *p0, PointType *p1) {
int ans;

  if (N_Obstacle==0) return(YES);

/*
  ans = unblocked_segment_sub2(p0,p1);
*/
  ans = unblocked_segment_sub1(p0,p1,0);
  return(ans);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int same_test_pt(PointType p1, PointType p2, PointType p[3][2]) {
int i;

  for (i=0;i<3;++i) {
     if (same_Point(p1,p[i][0]) && same_Point(p2,p[i][1])) return(YES);
     if (same_Point(p1,p[i][1]) && same_Point(p2,p[i][0])) return(YES);
  }
  return(NO);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int legal_detour_path(PointType p1, PointType p2) {
int ans;
PointType p[3][2];

  p[0][0].x = 18;
  p[0][0].y = 334;
  p[0][1].x = 408;
  p[0][1].y = 275;
  p[1][0] = p[0][1];
  p[1][1].x = 436;
  p[1][1].y = 275;
  p[2][0] = p[1][1];
  p[2][1].x = 476;
  p[2][1].y = 344;
  if (same_Point(p1, p2)) return(NO);

  ans = unblocked_segment(&p1,  &p2);
  if (same_test_pt(p1,p2,p)) {
    assert(ans==YES);
  }

  return(ans);
}


/***********************************************************************/
/*  update the distance of a node i outside SPT to the new tree node cp*/
/***********************************************************************/
void update_free_node_info(int cp, int i) {
double cost;

  assert(In_Tree[cp]==YES); /* S[cp] is newly added into SPT */
  assert(In_Tree[i]==NO); /* S[i] is  outside SPT */

  cost = S_Cost[cp] + Point_dist(S[cp],S[i]);
  if (  (equal(cost,S_Cost[i]) && S_N_Hops[i] > S_N_Hops[cp] + 1)  ||
        (cost < S_Cost[i]) ){
    S_Par[i] = cp;
    S_Cost[i] = cost;
    S_N_Hops[i] = S_N_Hops[cp] + 1;
  }
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void init_path_finder(int source,int dest) {
int i;

  for (i=0;i<N_S0+2;++i) {
    In_Tree[i]=NO;
    S_Par[i] = NIL;
    S_Cost[i] = DBL_MAX;
    S_N_Hops[i] = 0;
  }
  In_Tree[source]=YES;
  S_Par[source] = NIL;
  S_Cost[source] = 0;

  assert(same_Point( S[source], S[dest]) == NO );
  assert(legal_detour_path(S[source],S[dest]) == NO); 
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_path(FILE *f, PointType path[], int n) {
int i;

  fprintf(f,"move %.9f %.9f  n_hop:%d\n", path[0].x, path[0].y, n);
  for (i=1;i<n;++i) {
    fprintf(f,"draw %.9f %.9f  \n", path[i].x, path[i].y);
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void shrink_an_obstacle(ObstacleType *block, double s) {
int i, n;
double cx, cy;

  n = block->n_vertex;
  cx = cy = 0; 
  for (i=0;i<n;++i) {
    cx += block->vertex[i].x;
    cy += block->vertex[i].y;
  }
  cx /= n;
  cy /= n;
  for (i=0;i<n;++i) {
    block->vertex[i].x = (cx+(s-1)*block->vertex[i].x)/s;
    block->vertex[i].y = (cy+(s-1)*block->vertex[i].y)/s;
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_an_obstacle(FILE *f, ObstacleType block, int index) {
int i, n;
PointType pt;
double bbox[4];

  if (1) shrink_an_obstacle(&block, 7);
  get_block_bbox(bbox, &block);

  n = block.n_vertex;

  pt = block.vertex[0];
  fprintf(f,"move %.9f %.9f  %d ", pt.x, pt.y, index);
  if (bbox[WW]>260 && bbox[EE] < 350 && bbox[NN] <240 && bbox[SS] > 100) {
    printf("   here");
  } 
  fprintf(f,"\n");
  for (i=0;i<n;++i) {
    pt = block.vertex[(i+1)%n];
    fprintf(f,"draw %.9f %.9f  \n", pt.x, pt.y);
  }
  for (i=0;i<n/2;++i) {
    pt = block.vertex[i];
    fprintf(f,"move %.9f %.9f  \n", pt.x, pt.y);
    pt = block.vertex[(i+n/2)%n];
    fprintf(f,"draw %.9f %.9f  \n", pt.x, pt.y);
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_obstacles_sub(FILE *f) {
int i;

  for (i=0;i<N_Obstacle;++i) {
    print_an_obstacle(f, Obstacle[i], i);
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void print_obstacles( const string &fn){

  string xfn = fn + ".xg" ;

  FILE *f = fopen( xfn.c_str(), "w");
  assert(f != NULL);
  print_obstacles_sub(f);
  fclose(f);
}



/***********************************************************************/
/*  calc the shortest path[] from the shortest path tree in v[]        */
/***********************************************************************/
int get_path_from_SPT(PointType path[]) {
int i, n;
PointType tmpPt;

  n = 0;
  i = N_S-1;  /* S[N_S-1] is the destination */
  while (i != N_S - 2 ) {  /* S[NS-2] is the source */
    path[n++] = S[i];
    i = S_Par[i];
  }
  path[n++] = S[i];
  for  (i=0;i<n/2;++i) {
    tmpPt = path[i];
    path[i] = path[n-1-i];
    path[n-1-i] = tmpPt;
  }

  return(n);
}


/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int path_finder_sub2(int cp) {
int i, min_i;
double min_cost;

  In_Tree[cp] = YES;    /* add S[cp] into SPT */
  for (i=0;i<N_S;++i) {
    if (In_Tree[i] == NO && legal_detour_path(S[cp],S[i]) ) {
      update_free_node_info(cp,i);
    }
  }
  min_cost = DBL_MAX;
  min_i = NIL;           /* S[min_i] is next node to be added into SPT */
  for (i=0;i<N_S;++i) {
    if (In_Tree[i] == NO && S_Cost[i] < DBL_MAX) {
      if (equal(min_cost,S_Cost[i])) {
        if ( S_N_Hops[i] < S_N_Hops[min_i]) {
          min_cost = S_Cost[i];
          min_i = i;
        }
      } else if ( S_Cost[i] < min_cost ) {
        min_cost = S_Cost[i];
        min_i = i;
      }
    }
  }
  if (N_S>=N_S0+2) assert(min_i != NIL);
  return(min_i); /* expand the node in S with minimum cost */
}

/***********************************************************************/
/*   find a planar shortest path connecting source and dest            */
/***********************************************************************/
double path_finder_sub(int source,int dest,PointType path[], int *n_hops){
int cp;
double cost;

  if (0==1) {
    printf("*************************************************\n");
    printf("source = S[%d](%f, %f), dest = S[%d](%f, %f) \n", 
       source, S[source].x, S[source].y, dest, S[dest].x, S[dest].y);
    printf("*************************************************\n");
  }
  *n_hops = 0;
  init_path_finder(source,dest);
  cp = source; 
  while (cp != dest) { /* S[dest] not in Shortest Path Tree yet */
    cp = path_finder_sub2(cp);
    if (cp==NIL) return(DBL_MAX);
  }
  In_Tree[cp] = YES;
  *n_hops = get_path_from_SPT(path);
  cost = calc_pathlength(path,*n_hops, MANHATTAN);
  assert(equal(S_Cost[dest],cost ));
  return(cost);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
double path_finder(PointType p1, PointType p2, PointType path[], int *n_hops) {
double cost, range, dist, t;

  range = cost = dist = Point_dist(p1,p2);
  
  if (N_Obstacle==0 || same_Point(p1,p2) || legal_detour_path(p1, p2) ) {
    path[0] = p1;
    path[1] = p2;
    *n_hops = 2;
  } else {
    cost = DBL_MAX;
    do {
      if (cost==DBL_MAX) {
        range = range+dist;
      } else {
        range = cost;
      }
      init_S(p1,p2, range);
      t= path_finder_sub(N_S-2, N_S-1, path, n_hops);
      cost = t; 
      if (N_S >= N_S0 + 2 && *n_hops <= 2) {
        print_S();
        assert(0);
      }
    } while ( cost > range + FUZZ);
  }
  
  assert(*n_hops>=2);
  assert(*n_hops<=100);
  assert(cost <= INT_MAX);
  return(cost);
}


/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void test_path_finder_sub(FILE *f, PointType p0, PointType p1) {
PointType path[100];
int n;
double cost; 

  cost = path_finder(p0,p1, path, &n);
  fprintf(f, "\"path(%.0f) \n", cost);
  print_path(f,path,n);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void test_path_finder(FILE *f) {
PointType p0, p1;

  p0.x = 448;
  p0.y = 334;
  p1.x = 408;
  p1.y = 343;
  test_path_finder_sub(f, p0, p1);
  p0.x = 408;
  p0.y = 275;
  p1.x = 436;
  p1.y = 275;
  test_path_finder_sub(f, p0, p1);
  p0.x = 436;
  p0.y = 275;
  p1.x = 476;
  p1.y = 344;
  test_path_finder_sub(f, p0, p1);
  p0.x = 18;
  p0.y = 334;
  p1.x = 476;
  p1.y = 344;
  test_path_finder_sub(f, p0, p1);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void calc_n_vertex(AreaType *area) {
  area->vertex[0] = 0;
  area->vertex[1] = 1;
  if (same_Point(area->mr[0], area->mr[1])) {
    area->npts = area->n_mr = 1;
  } else {
    area->npts = area->n_mr = 2;
  }
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void get_mr_for_line_sub(AreaType *area,AreaType *new_area,int begin,int end) {
int i, k;

  k = new_area->n_mr;
  for (i=begin;i<=end;++i) {
    new_area->mr[k++] = area->mr[i]; 
  }
  new_area->n_mr = k;
  new_area->vertex[0] =0;

  for (i=0;i<area->n_mr-1;++i) {
    assert(!same_Point(area->mr[i], area->mr[i+1]));
  }

  if (new_area->n_mr>1) {
    new_area->npts = 2;
    new_area->vertex[1] = new_area->n_mr - 1;
  } else {
    new_area->npts = 1;
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void get_mr_for_line(PointType pt, AreaType *area, AreaType *new_area, int dir) {
int i;

  for (i=0;i<area->n_mr;++i) {
    if (same_Point(pt,area->mr[i])) {
      new_area->n_mr = 0; 
      if (dir==0) {
        get_mr_for_line_sub(area, new_area, 0, i);
      } else {
        get_mr_for_line_sub(area, new_area, i, area->n_mr-1);
      }
      assert(new_area->n_mr<=4);
      return;
    }
  }
  for (i=0;i<area->n_mr - 1;++i) {
    if ( pt_on_line_segment(pt,area->mr[i],area->mr[i+1]) ) {
      calc_pt_delays(area, &pt, area->mr[i],area->mr[i+1]);
      if (dir==0) {
        new_area->n_mr = 0;
        get_mr_for_line_sub(area, new_area, 0, i);
        new_area->mr[new_area->n_mr] = pt;
        new_area->vertex[1] = new_area->n_mr;
        new_area->n_mr += 1; 
        new_area->npts = 2;
      } else {
        new_area->mr[0] = pt;
        new_area->n_mr = 1; 
        get_mr_for_line_sub(area, new_area, i+1, area->n_mr-1);
      }
      assert(new_area->n_mr<=4);
      return;
    }
  }
  assert(0==1);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int modify_area_sub3(AreaType *area, AreaType tmpArea[], int *n,  
                     ObstacleType *block){
PointType p0, p1, q[10];
int n_xing=0;

  p0 = area->mr[0];
  p1 = area->mr[area->n_mr-1];
  /* p0 and p1 are either outside or on the boundary of obstacle */

  n_xing = line_x_rectangle(&p0,&p1,q,block);
  if (n_xing == 2) {
    if (Point_dist(p0,q[0]) < Point_dist(p0,q[1]) ) {
      get_mr_for_line(q[0], area,  &(tmpArea[*n]), 0);
      get_mr_for_line(q[1], area,  &(tmpArea[*n+1]), 1);
    } else {
      get_mr_for_line(q[1], area,  &(tmpArea[*n]), 0);
      get_mr_for_line(q[0], area,  &(tmpArea[*n+1]), 1);
    }
    *n += 2;
    if (0) check_area_array(tmpArea, *n);
  } else {
    tmpArea[(*n)++] = *area;
  }
  return(n_xing);

}
/***********************************************************************/
/*  check if line p0,p1 goes into rectangle[k]                         */
/***********************************************************************/
void modify_area_sub2(AreaType *area, AreaType tmpArea[], int *n, 
                    ObstacleType *block){
int i,n_mr,loc0, loc1, n_xing;
PointType p0, p1, q[10];


  n_mr = area->n_mr;
  loc0 = pt_outside_a_rectangle(&(area->mr[0]), block);
  loc1 = pt_outside_a_rectangle(&(area->mr[n_mr-1]), block);
  if (loc0 > loc1) {
    reverse_mr(area);
    i = loc0;
    loc0 = loc1;
    loc1 = i;
  }
  p0 = area->mr[0];
  p1 = area->mr[n_mr-1];
  assert(!same_Point(p0,p1));

  if (loc0==INSIDE) {
    if (loc1==ON) {
      tmpArea[*n].npts = tmpArea[*n].n_mr = 1; 
      tmpArea[*n].mr[0] = p1; 
      (*n)++;
    } else if (loc1 == OUTSIDE ) {
      n_xing = line_x_rectangle(&p0,&p1,q,block);
      assert(n_xing==1);
      get_mr_for_line(q[0], area, &(tmpArea[*n]), 1);
      (*n)++;
    }
  } else  {
    /* p0 and p1 are either outside or on the boundary of obstacle */
    modify_area_sub3(area,tmpArea,n, block);
  }
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void modify_area_sub1(AreaType *area, AreaType tmpArea[],int *n,
                     ObstacleType *block){
int type; 
  
  if (0) check_area(area);
  if (area->n_mr==1) {
    type = pt_outside_a_rectangle( &(area->mr[0]), block);
    if (type==1 || type == 0) { /* add one area */
      tmpArea[(*n)++] = *area;
    }
  } else {  /* area->n_mr==2 */
    tmpArea[*n] = tmpArea[*n+1] = *area;
    modify_area_sub2(area,tmpArea,n, block);    
  } 
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void expand_a_dimention_sub(int index, double z, int dir) {
double bbox[4];

  assert(index < N_Obstacle);
  get_block_bbox(bbox, &(Tmp_Obstacle[index]));
  if (dir==NN || dir == EE) {
    bbox[dir] = max (z+1E8, bbox[dir]);
  } else {
    bbox[dir] = min (z-1E8, bbox[dir]);     
  }
  Tmp_Obstacle[index].vertex[0].x = bbox[WW];    
  Tmp_Obstacle[index].vertex[1].x = bbox[WW];    
  Tmp_Obstacle[index].vertex[2].x = bbox[EE];    
  Tmp_Obstacle[index].vertex[3].x = bbox[EE];    
  Tmp_Obstacle[index].vertex[0].y = bbox[SS];    
  Tmp_Obstacle[index].vertex[1].y = bbox[NN];    
  Tmp_Obstacle[index].vertex[2].y = bbox[NN];    
  Tmp_Obstacle[index].vertex[3].y = bbox[SS];    
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int bblock_overlap(ObstacleType *b1,ObstacleType *b2) {
double bbox1[4], bbox2[4];

  get_block_bbox(bbox1, b1);
  get_block_bbox(bbox2, b2);

  if (bbox1[SS]>=bbox2[NN]) return(NO);
  if (bbox2[SS]>=bbox1[NN]) return(NO);
  if (bbox1[WW]>=bbox2[EE]) return(NO);
  if (bbox2[WW]>=bbox1[EE]) return(NO);
  return(YES);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int expand_a_dimention_sub2(double z,int dir,int *n_unexp,int unexp[],int expanded[]){
int i, j, x, y, change = NO, n_expanded, n;

  n = *n_unexp;
  n_expanded = N_Obstacle - n;
  for (i=0;i<n;++i) {
    x = unexp[i];
    for (j=0;j<n_expanded;++j) {
      y = expanded[j];
      if (bblock_overlap(&(Tmp_Obstacle[x]),&(Tmp_Obstacle[y]))) {
        change = YES;
        expand_a_dimention_sub(x, z, dir);
        unexp[i] = unexp[*n_unexp-1];
        (*n_unexp)--;
        expanded[n_expanded++] = x;
        return(YES);
      }
    }
  }
  return(change);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void expand_a_dimention(int index, double z, int dir) {
int i, unexp[1000], expanded[1000];
int n_unexp, change;

  expand_a_dimention_sub(index, z, dir);

  for (i=0;i<N_Obstacle;++i) {
    unexp[i] = i; 
  }
  unexp[index] = N_Obstacle-1;
  n_unexp = N_Obstacle-1;
  expanded[0] = index;
  
  do {
    change = expand_a_dimention_sub2(z, dir, &n_unexp, unexp, expanded);
  } while (change==YES);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int expand_obstacle(AreaType *area, int index,double bbox[4],int type){
int i, j, k, ans, D0, D1; 
PointType p0, p1; 

  if (type == 0 ) {
    if (same_Point(area->line[0][0],area->line[1][0])) {
      return(NO);
    } else {
      ans = line_into_rectangle(&(area->line[0][0]),&(area->line[1][0]),
            &(Tmp_Obstacle[index]));
      return(ans);
    }
  } else {
    if (type==1) { D0 = EE; D1 = WW; } else { D0 = WW; D1 = EE; }
    p0.x = bbox[D0];
    p0.y = bbox[NN];
    p1.x = bbox[D1];
    p1.y = bbox[SS];
    i = line_into_rectangle(&p0, &(area->line[0][0]), 
            &(Tmp_Obstacle[index]));
    j = line_into_rectangle(&p0, &(area->line[1][0]), 
            &(Tmp_Obstacle[index]));
    if (i  || j ) {
      expand_a_dimention(index, bbox[NN], NN); 
      expand_a_dimention(index, bbox[D0], D0); 
      k=1;
    } else {
      k=0;
    }
    i = line_into_rectangle(&p1, &(area->line[0][1]), 
            &(Tmp_Obstacle[index]));
    j = line_into_rectangle(&p1, &(area->line[1][1]), 
            &(Tmp_Obstacle[index]));
    if (i  || j ) {
      if (k==1) return(YES);
      expand_a_dimention(index, bbox[SS], SS); 
      expand_a_dimention(index, bbox[D1], D1); 
    }
    return(NO);
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void modify_area_sub0(AreaType area_out[], int *n_out){
int n_in, i, j;
AreaType area_in[10], area;

  area = area_out[0]; 
  *n_out=1;
  for (i=0;i<N_Obstacle;++i) {
    n_in = cp_area_array(area_out, *n_out, area_in);
    *n_out = 0;
    for (j=0;j<n_in;++j) {
      modify_area_sub1(&(area_in[j]), area_out, n_out, &(Tmp_Obstacle[i]));
    }
    if (*n_out ==0) return;
  }
  assert(*n_out<10);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
double path_between_JSline_sub(AreaType *area, PointType line[2][2], PointType path[100], int *n, PointType pt[]) {
int done = NO, count = 0, min_n;
PointType q0, q1, min_path[100];
double cost, min_cost = DBL_MAX;

  assert(pt_on_line_segment(pt[0], line[0][0],line[0][1]));
  assert(pt_on_line_segment(pt[1], line[1][0],line[1][1]));

  done = NO;
  min_cost = DBL_MAX;
  while (!done) {
    count++;
    if (area != NULL) assert(area->n_mr<= MAX_mr_PTS);
    cost = path_finder(pt[0], pt[1], path, n);
    if (area != NULL) assert(area->n_mr<= MAX_mr_PTS);
    assert(*n<=100);
    pt2linedist(path[1],line[0][0],line[0][1], &q0);
    pt2linedist(path[*n-2],line[1][0],line[1][1], &q1);
    if (!same_Point(pt[0],q0) || !same_Point(pt[1],q1)) {
      pt[0] = q0;
      pt[1] = q1;
    } else {
      done = YES;
    }
    if ( cost <  min_cost )  {
      min_cost = cost;
      min_n = cp_PointType_array(path, *n , min_path);
    } else {
      done = YES;
    }
  }
  if (area != NULL) assert(area->n_mr<= MAX_mr_PTS);
  *n = cp_PointType_array(min_path, min_n, path);
  assert(pt_on_line_segment(path[0], line[0][0],line[0][1]));
  assert(pt_on_line_segment(path[*n-1], line[1][0],line[1][1]));
  return(min_cost);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void center_of_segment(PointType *q, PointType p1, PointType p2) {
  q->x = (p1.x+p2.x)/2;
  q->y = (p1.y+p2.y)/2;
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
double path_between_JSline(AreaType *area, PointType line[2][2],PointType path[100],int *n){

int i, j, m, num[2], min_n;
PointType q[2][9], pt[2], min_path[100];
double cost, min_cost;

  linedist(line[0][0],line[0][1],line[1][0],line[1][1], pt);
 
  for (i=0;i<2;++i) {
    m = 0;
    q[i][m++] = pt[i];
    if (!same_Point(line[i][0], line[i][1])) {
      q[i][m++] = line[i][0]; 
      q[i][m++] = line[i][1]; 
      center_of_segment(&(q[i][m++]), line[i][0], line[i][1]);
    }
    num[i] =m;
  }

  num[0] = num[1] = 1;
  min_cost = DBL_MAX;
  for (i=0;i<num[0];++i) {
    pt[0] = q[0][i];
    for (j=0;j<num[1];++j) {
      pt[1] = q[1][j];
      cost = path_between_JSline_sub(area, line, path, n, pt);
      assert(*n<=100);
      if ( cost <  min_cost ) {
        min_cost = cost;
        min_n = cp_PointType_array(path, *n , min_path);
      }
    }
  }
  *n = cp_PointType_array(min_path, min_n, path);
  return(min_cost);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int get_boundary_segment(AreaType *area,AreaType new_area[]) {
int j, k, m, n, npts;

  n = 0;
  if (0) calc_vertices(area);
  npts = area->npts;
  if (npts >2 ) {
    for (j=0;j<npts;++j) {
      new_area[n] =  *area ;
      if (0) check_area(&(new_area[n]));
      new_area[n].npts = 2;
      m = 0;
      k=area->vertex[j] - 1;
      do {
        k=(k+1)%area->n_mr;
        new_area[n].mr[m++] = area->mr[k];
      } while (k!=area->vertex[(j+1)%npts]);
      new_area[n].n_mr = m;
      new_area[n].vertex[0] = 0;
      new_area[n].vertex[1] = m-1;
      calc_vertices(&(new_area[n]));
      assert(new_area[n].npts==2);
      if (0) check_area(&(new_area[n]));
      n++;
    }
    assert(m<=4);
  } else {
    new_area[n] = *area;
    modify_ms_JS(&(new_area[n]));
    n++;
  }
  return(n);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int check_lines(AreaType *area) {

  assert(area->line[0][0].y >= area->line[0][1].y);
  assert(area->line[1][0].y >= area->line[1][1].y);
  if (same_Point(area->line[0][0], area->line[1][0]) &&
      same_Point(area->line[0][1], area->line[1][1]) ) {
    return(NO);
  } else {
    return(YES);
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int modify_area_case1(AreaType *area, AreaType out_area[]){
int i,tmp_n, out_n, in_n, containing, type;
AreaType in_area[6], tmp_area[100];
double bbox[4];

  in_n = get_boundary_segment(area, in_area);
  assert(in_n <= 6);

  get_JR_bbox(bbox, &(in_area[0]));
  type = JS_ms_type(&(in_area[0]));
  check_lines(&(in_area[0]));

  for (i=0;i<N_Obstacle;++i) {
    Tmp_Obstacle[i] = Obstacle[i];
  }
  for (i=0;i<N_Obstacle;++i) {
    containing = expand_obstacle(&(in_area[0]), i, bbox, type);
    if (containing) {
      return(0);
    }
  }
  out_n = 0;
  for (i=0;i<in_n;++i) {
    tmp_area[0] = in_area[i];
    modify_area_sub0(tmp_area, &tmp_n);
    append_area_array(tmp_area, tmp_n, out_area, &out_n);
    assert(out_n <= N_TempArea);
  }
  return(out_n);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void modify_rec_JS_line(AreaType *area) {
int linetype;
double bbox[4];

  get_mr_bbox(bbox, area);
  linetype = calc_line_type(area->line[0][0],area->line[0][1]);
  if (linetype==VERTICAL) {
    assert(area->line[0][0].y > area->line[0][1].y);
    assert(area->line[1][0].y > area->line[1][1].y);
    area->line[0][0].y = area->line[1][0].y = bbox[NN];
    area->line[0][1].y = area->line[1][1].y = bbox[SS];
  } else if (linetype==HORIZONTAL) {
    assert(area->line[0][0].x > area->line[0][1].x);
    assert(area->line[1][0].x > area->line[1][1].x);
    area->line[0][0].x = area->line[1][0].x = bbox[EE];
    area->line[0][1].x = area->line[1][1].x = bbox[WW];
  } else {
    assert(0==1);
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void check_rectilinear_line(AreaType  *area) {
int linetype, linetype2;

  if (same_Point(area->line[0][0],area->line[0][1])) {
    assert(same_Point(area->line[1][0],area->line[1][1]));
    assert(equal(area->line[0][0].x,area->line[1][0].x) ||
           equal(area->line[0][0].y,area->line[1][0].y) );
  } else {
    linetype = calc_line_type(area->line[0][0],area->line[0][1]);
    linetype2 = calc_line_type(area->line[1][0],area->line[1][1]);
    assert(linetype==VERTICAL || linetype==HORIZONTAL);
    assert(linetype == linetype2);

    if (linetype==VERTICAL) {
      assert(area->line[0][0].y > area->line[0][1].y);
      assert(area->line[1][0].y > area->line[1][1].y);
    } else {
      assert(area->line[0][0].x > area->line[0][1].x);
      assert(area->line[1][0].x > area->line[1][1].x);
    }
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void modify_rectilinear_line(AreaType *area, int linetype, int i, double z) {
  if (linetype==VERTICAL) {
    area->line[0][i].y = area->line[1][i].y = z;
  } else {
    area->line[0][i].x = area->line[1][i].x = z;
  }
  check_rectilinear_line(area);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void modify_area_case2_sub3(AreaType *area, AreaType area_out[], int *n_out, 
         double bbox[], double obox[], int d1, int d2, int linetype) {

  area_out[*n_out] = *area;
  check_rectilinear_line(area);

  if ( obox[d1] < bbox[d1] + FUZZ && obox[d2] > bbox[d2]-FUZZ) {
    modify_rectilinear_line(&(area_out[*n_out]), linetype, 1, obox[d1]);
    (*n_out)++;
    area_out[*n_out] = *area;
    modify_rectilinear_line(&(area_out[*n_out]), linetype, 0, obox[d2]);
    (*n_out)++;
  } else if (obox[d1] < bbox[d1] + FUZZ) {
    assert( obox[d2] <= bbox[d2]-FUZZ);
    modify_rectilinear_line(&(area_out[*n_out]), linetype, 1, obox[d1]);
    (*n_out)++;
  } else if (obox[d2] > bbox[d2]-FUZZ) {
    assert( obox[d1] >= bbox[d1] + FUZZ );
    modify_rectilinear_line(&(area_out[*n_out]), linetype, 0, obox[d2]);
    (*n_out)++;
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void modify_area_case2_sub2(AreaType *area, AreaType area_out[],
                           int *n_out, double bbox[], double obox[]) {
int  linetype, d1, d2;

  linetype = check_area_line(area);

  if (linetype==VERTICAL) {
    d1 = NN; d2 = SS;
  } else {
    d1 = EE; d2 = WW;
    assert(linetype==HORIZONTAL);
  }
  if (obox[d1] >= bbox[d1] + FUZZ && obox[d2]<=bbox[d2]-FUZZ ) {
  } else {
    modify_area_case2_sub3(area, area_out, n_out, bbox, obox, d1, d2, linetype);
  }
  if (0) check_area_array(area_out, *n_out);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int modify_area_case2_sub(AreaType *area, AreaType area_out[],
                          int *n_out, double obox[]) {
int intersect ;
double bbox[4];

  get_JR_bbox(bbox, area);
  
  intersect = _bbox_overlap(bbox[WW], bbox[SS],bbox[EE],bbox[NN],
                   obox[WW], obox[SS],obox[EE],obox[NN], 0);
  if (intersect==NO) {
    check_area_line(area);
    area_out[*n_out] = *area;
    check_area_line(&(area_out[*n_out]));
    (*n_out)++;
  } else {
    modify_area_case2_sub2(area, area_out, n_out, bbox, obox);
  }
  return(intersect);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void modify_mr_case2_sub(AreaType *area) {
int i, intersect;
double bbox[4] ;

  get_JR_bbox(bbox, area);
  for (i=0;i<N_Obstacle;++i) {
/*
    get_block_bbox(obox, &(Obstacle[i]));
*/
    intersect = _bbox_overlap(bbox[WW], bbox[SS],bbox[EE],bbox[NN],
                 ObsBox[i][WW], ObsBox[i][SS],ObsBox[i][EE],ObsBox[i][NN], -FUZZ);
    assert(intersect==NO);
  }
  JS[0][0] = area->line[0][0];
  JS[0][1] = area->line[0][1];
  JS[1][0] = area->line[1][0];
  JS[1][1] = area->line[1][1];
  calc_JS_delay(area, area->area_L, area->area_R);

  if (Manhattan_arc(JS[0][0], JS[0][1]) ) {
    assert(same_Point(JS[0][0], JS[0][1]) );
    assert(same_Point(JS[1][0], JS[1][1]) );
    L_MS.MakeDiamond (JS[0][0], 0    );
    R_MS.MakeDiamond (JS[1][0], 0    );
  }
  MergeArea(area, area->area_L, area->area_R, NORMAL );
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void modify_mr_case2(int n, AreaType area[]) {
int i;

  for (i=0;i<n;++i) {
    modify_mr_case2_sub(&(area[i]));
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int modify_area_case2(AreaType *area, AreaType area_out[]){
int n_in, n_out, i, j, change = NO, intersect;
AreaType area_in[1000];

  area_out[0]= *area;
  n_out=1;
  modify_rec_JS_line(&(area_out[0]));

  for (i=0;i<N_Obstacle;++i) {
    n_in = cp_area_array(area_out, n_out, area_in);
    if (0) check_area_array(area_out, n_out);
    if (0) check_area_array(area_in, n_in);
    n_out = 0;
/*
    block = Obstacle[i];
    get_block_bbox(obox, &block);
*/
    for (j=0;j<n_in;++j) {
      intersect = modify_area_case2_sub(&(area_in[j]), area_out,
                                        &n_out, ObsBox[i]);
       assert(n_out<50);
      if (intersect==YES) change=YES;
    }
    if (n_out ==0) return(0);
    assert(n_out<50);
  }
  if (change) {
    modify_mr_case2(n_out,area_out);
  } else {
    assert(n_out==1);
  }
  return(n_out);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int calc_path_balance_pt(AreaType *area, PointType path[], int n,
double delay[2][100], double cap[2][100]) {
double x, r,c ;
int i;

  delay[0][0]   = path[0].max;
  delay[1][n-1] = path[n-1].max;
  cap[0][0] = area->area_L->capac;
  cap[1][n-1] = area->area_R->capac;
  r = PURES[H_];
  c = PUCAP[H_];
   
  for (i=1;i<n;++i) {
    x = Point_dist(path[i],path[i-1]);
    delay[0][i] = delay[0][i-1] + r*x*(c*x/2 + cap[0][i-1]);
    cap[0][i] = cap[0][i-1] + c*x;
  }
  for (i=n-2;i>=0;--i) {
    x = Point_dist(path[i],path[i+1]);
    delay[1][i] = delay[1][i+1] + r*x*(c*x/2 + cap[1][i+1]);
    cap[1][i] = cap[1][i+1] + c*x;
  }
  for (i=0;i<n-2;++i) {
    if (delay[0][i+1] >= delay[1][i+1]) break;
  }
  return(i);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void calc_new_path_delay(AreaType *area, PointType path[], int n) {
int i;
PointType p0, p1;
double x0,x1,t0, t1, r,c;

  r = PURES[H_];
  c = PUCAP[H_];

  p0 = path[0];
  p1 = path[n-1];
  for (i=0;i<n;++i) {
    x0 = calc_subpath_length(path,0,i,MANHATTAN);
    x1 = calc_subpath_length(path,i,n-1,MANHATTAN);
    t0 = r*x0*(c*x0/2 + area->area_L->capac);
    t1 = r*x1*(c*x1/2 + area->area_R->capac);
    path[i].max = max ( p0.max + t0, p1.max + t1 );   
    path[i].min = min ( p0.min + t0, p1.min + t1 );   
  }
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void calc_feasible_subpath(PointType path[], int n, int k, int m[2]) {
int i,j;
double leng, dist;

   i = k;
   j = k+1;
   leng = dist = Point_dist(path[k],path[k+1]);
   while (i > 0 ) {
     dist =  Point_dist(path[i-1], path[j]);
     leng += Point_dist(path[i-1], path[i]);
     if (!equal(leng,dist)) { break; }
     i--;
   }
   while (j < n-1 ) {
     dist =  Point_dist(path[i], path[j+1]);
     leng += Point_dist(path[j], path[j+1]);
     if (!equal(leng,dist)) { break; }
     j++;
   }
   m[0] = i;
   m[1] = j;
   assert(i<j);
   assert(i>=0);
   assert(j<n);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void new_area_by_detour_sub(AreaType *area, PointType path[], int n, int k,
double delay[2][100], double cap[2][100]) {
double r,c, d0, d1, t0, t1;
int i;

  double x = Point_dist(path[k],path[k+1]);
  calc_merge_distance(r,c, cap[0][k],delay[0][k],cap[1][k+1],delay[1][k+1],
                      x,&d0,&d1);
  t0 = delay[0][k]   + r*d0*(c*d0/2+ cap[0][k]);
  t1 = delay[1][k+1] + r*d1*(c*d1/2+ cap[1][k+1]);
  assert(equal(t0,t1));
  if (equal(d0,0)) {
    area->mr[1] = area->mr[0] = path[k];
  } else if (equal(d1,0)) {
    area->mr[1] = area->mr[0] = path[k+1];
  } else {
    area->mr[0].x = (path[k].x*d1+path[k+1].x*d0)/(d0+d1);
    area->mr[0].y = (path[k].y*d1+path[k+1].y*d0)/(d0+d1);
    area->mr[1] = area->mr[0]; 
  }

  area->L_EdgeLen = d0;
  for (i=0;i<k;++i) {
    area->L_EdgeLen += Point_dist(path[i],path[i+1]);
  }
  area->R_EdgeLen = d1;
  for (i=k+1;i<n-1;++i) {
    area->R_EdgeLen += Point_dist(path[i],path[i+1]);
  }

  x = area->L_EdgeLen;
  t0 = r*x*(c*x/2 + area->area_L->capac);
  x = area->R_EdgeLen;
  t1 = r*x*(c*x/2 + area->area_R->capac);
  assert(equal(path[0].max + t0, path[n-1].max + t1));

  calc_n_vertex(area);
  for (i=0;i<area->n_mr;++i) {
    area->mr[i].max =  path[0].max + t0 ;
    area->mr[i].min = min (path[0].min + t0, path[n-1].min + t1);
    assert(pt_skew(area->mr[i]) <= Skew_B + FUZZ);
  }
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int  new_area_by_detour_sub2(AreaType *area, PointType path[], int n, int k,
             AreaType out_area[]) {
int i, m[2], n_area;
double x0,x1,t0,t1;
PointType pt0, pt1;
AreaType tmp_area, tmp_area_L, tmp_area_R;

  calc_feasible_subpath(path, n, k, m);
  pt0 = path[m[0]];
  pt1 = path[m[1]];
  x0 = calc_subpath_length(path, 0,m[0],MANHATTAN);
  x1 = calc_subpath_length(path, m[1],n-1,MANHATTAN);
  t0 = PURES[H_]*x0*(PUCAP[H_]*x0/2+area->area_L->capac);
  t1 = PURES[H_]*x1*(PUCAP[H_]*x1/2+area->area_R->capac);
  pt0.max = path[0].max + t0;
  pt0.min = path[0].min + t0;
  pt1.max = path[n-1].max + t1;
  pt1.min = path[n-1].min + t1;
  tmp_area_L = tmp_area_R = Node[0].area[0];
  tmp_area_L.capac = area->area_L->capac + x0*PUCAP[H_];
  tmp_area_R.capac = area->area_R->capac + x1*PUCAP[H_];
  tmp_area_L.mr[0] = pt0;
  tmp_area_R.mr[0] = pt1;
  JS[0][0] = JS[0][1] = pt0;
  JS[1][0] = JS[1][1] = pt1;
  L_MS.MakeDiamond (JS[0][0], 0    );
  R_MS.MakeDiamond (JS[1][0], 0    );

  MergeArea(&tmp_area, &tmp_area_L, &tmp_area_R, NORMAL );


  n_area = modify_blocked_areas_no_detour(&tmp_area, out_area);
  for (i=0;i<n_area;++i) {
    out_area[i].line[0][1] = out_area[i].line[0][0] = area->line[0][0];
    out_area[i].line[1][1] = out_area[i].line[1][0] = area->line[1][0];
    out_area[i].area_L = area->area_L;
    out_area[i].area_R = area->area_R;
    if (out_area[i].L_EdgeLen >=0) {
      assert (out_area[i].R_EdgeLen >=0);
      out_area[i].L_EdgeLen += x0;
      out_area[i].R_EdgeLen += x1;
    }
    MergeArea_sub(&(out_area[i]));
  }
  return(n_area);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int modify_blocked_areas_by_detour(AreaType *area, AreaType out_area[]) {
int i, n, n_area = 1;
PointType path[100], p1, p2, pt[2];
double delay[2][100], cap[2][100];
double cost;

  linedist(area->line[0][0],area->line[0][1],area->line[1][0],
	   area->line[1][1], pt);
  cost = path_between_JSline_sub(area, area->line, path, &n, pt);
  assert(n>2 && n<=100);

  calc_BS_located(&(path[0]),area->area_L, &p1, &p2);
  calc_pt_delays(area->area_L, &(path[0]),p1,p2);

  calc_BS_located(&(path[n-1]),area->area_R, &p1, &p2);
  calc_pt_delays(area->area_R, &(path[n-1]),p1,p2);

  area->line[0][1] = area->line[0][0] = path[0];
  area->line[1][1] = area->line[1][0] = path[n-1];

  i = calc_path_balance_pt(area, path,n, delay, cap);
  new_area_by_detour_sub(area, path, n,i, delay, cap);
  MergeArea_sub(area );
  out_area[0] = *area;
/*
  n_area = new_area_by_detour_sub2(area, path, n,i, out_area);
*/
  return(n_area);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int modify_blocked_areas_no_detour(AreaType *area, AreaType new_area[]) {
int linetype, n;

  linetype = calc_line_type(area->line[0][0],area->line[0][1]);
  if (linetype == MANHATTAN_ARC) {
    if (equal(area->dist, 0)) {
      n=1;
      new_area[0] = *area;
    } else {
      n = modify_area_case1(area, new_area);
    }
  } else {
    assert(linetype==VERTICAL || linetype==HORIZONTAL);
    n = modify_area_case2(area, new_area);
  }

  return(n);
}

/***********************************************************************/
/*                                                                     */
/***********************************************************************/
int modify_blocked_areas_sub(AreaType area[], int n, AreaType area2[], 
    int type) {
int i, n1, n2;
AreaType area1[1000];

  n2 = 0;
  for (i=0;i<n;++i) {
    if (type == 0 ) {
      n1 = modify_blocked_areas_no_detour(&(area[i]), area1);
      if ( 0 && n1==0) {
        n1 = modify_blocked_areas_by_detour(&(area[i]), area1);
      }
    } else {
      n1 = modify_blocked_areas_by_detour(&(area[i]), area1);
    }
    assert(n1<1000);
    append_area_array(area1, n1, area2, &n2);
    assert(n2<1000);
  }
  return(n2);
}
/***********************************************************************/
/*                                                                     */
/***********************************************************************/
void modify_blocked_areas(AreaType area[], int *n, int b1, int b2) {
AreaType tmp_area[1000];

  if (N_Obstacle==0)  return;

  Buffered[0] = b1;
  Buffered[1] = b2;

  int n2 = modify_blocked_areas_sub(area, *n, tmp_area, 0);
  if (n2==0) {
    *n = min (*n, N_Area_Per_Node);
    n2 = modify_blocked_areas_sub(area, *n, tmp_area,1);
  }
  *n = cp_area_array(tmp_area, n2, area);
}
