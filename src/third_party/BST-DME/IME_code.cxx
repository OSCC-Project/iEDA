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
#include "bst_sub1.h"

extern double area_minskew(AreaType *area); 
extern int   equal(double, double);

/*****************************************************************************/
/*   Copy object AreaType from stair to result.                              */
/*****************************************************************************/
static void CopyRegionFrom(AreaType *result, AreaType *stair) {
  *result = *stair; 
}
/*****************************************************************************/
/*   Copy  AreaSetType  from stair to result.                               */
/*****************************************************************************/
static void CopyPolygonFrom(AreaSetType *result, AreaSetType *stair) {

  result->npoly = stair->npoly; 
  for (int i=0;i< stair->npoly; ++i) {
    CopyRegionFrom(&(result->freg[i]), &(stair->freg[i]));
  }
}

/*****************************************************************************/
/*                                                                           */
/*****************************************************************************/
int CompareArea(const void *p, const void *q) {
AreaType *a, *b;
double a_minskew, b_minskew;
double a_cost, b_cost;
   
   a = (AreaType *) p;
   b = (AreaType *) q;
   a_minskew = area_minskew(a);
   b_minskew = area_minskew(b);
   a_cost = a->capac;
   b_cost = b->capac;
   if (a_minskew > b_minskew) { return 1;
   } else if (a_minskew < b_minskew) { return -1;
   } else if (a_cost   > b_cost  ) { return 1;
   } else if (a_cost   < b_cost  ) { return -1;
   } else return 0;
}


/*****************************************************************************/
/*  check                                                                    */
/*****************************************************************************/
static void check_region_qsort_sub(AreaSetType *stair, int i) {
double sk1, sk2; 
  sk1 = area_minskew(&(stair->freg[i]));
  sk2 = area_minskew(&(stair->freg[i+1]));
  if (sk1 > sk2) {
    printf("sk1[%d] = %f, sk2[%d] = %f \n", i,
         area_minskew(&(stair->freg[i])), i+1,
         area_minskew(&(stair->freg[i+1])));
    printf("sk1 = %f, sk2 = %f \n", sk1, sk2);
  }
  assert(sk1 <= sk2);
}
/*****************************************************************************/
/*  check                                                                    */
/*****************************************************************************/
static void check_region_qsort(AreaSetType *stair) {
int i;

  for (i=0;i<stair->npoly-1;++i) {
    check_region_qsort_sub(stair,i);
  }
}
/*****************************************************************************/
/* remove redundant regions                                                  */
/*  return number of regions that was removed                                */
/*****************************************************************************/
void Irredundant(AreaSetType *stair) {
int i, j;
double sk1, sk2, cap1, cap2,area1, area2;

   if (stair->npoly <= 1) {
      return;
   }

/* qsort(stair->freg, stair->npoly, sizeof(region), CompareRegion);
*/
   qsort(stair->freg, stair->npoly, sizeof(AreaType), CompareArea);
   check_region_qsort(stair);
   for (i = 0, j = 0; i < stair->npoly - 1; i++) {
      sk1 = area_minskew(&(stair->freg[j]));
      sk2 = area_minskew(&(stair->freg[i+1]));
      cap1 = stair->freg[j].capac; 
      cap2 = stair->freg[i+1].capac;
      area1 = calc_boundary_length(&(stair->freg[j]));
      area2 = calc_boundary_length(&(stair->freg[i+1]));
/* if (sk1 == sk2) {   */
      if (equal(sk1,sk2)) {  
         if (cap1 >= cap2+FUZZ || (equal(cap1,cap2) && area1<area2) ) {
            CopyRegionFrom(&(stair->freg[j]), &(stair->freg[i+1]));
         }
      } else if (sk1 < sk2) {
         if (cap1 >= cap2 + FUZZ ||  (equal(cap1,cap2) && area1<area2) ) {
            j++;
            if (j < i+1) {
               CopyRegionFrom(&(stair->freg[j]), &(stair->freg[i+1]));
            }
         }
      } else {
         printf("sorting error:n=%d, skew(%d):%f skew(%d):%f\n",
            stair->npoly, j,sk1,i+1,sk2);
         fprintf(stderr, "sorting error\n");
      }
   }
   stair->npoly = j+1;
}



/*****************************************************************************/
/* Region selectin using Dynamic Programming (by Koh,Cheng-Kok)
*/
/*****************************************************************************/
void KStepStair(AreaSetType *stair, int step, AreaSetType *result) {
int i, j, k, l, **best;
double **waste;
double **init;
double skew1, skew2, cap1, cap2;

/* No trimming necessary */

   if (stair->npoly <= step) { CopyPolygonFrom(result, stair); return; }

/* Initialization */

   waste = (double **)malloc((step)*sizeof(double *));
   for (i = 0; i < step; i++) {
      waste[i] = (double *)malloc((stair->npoly)*sizeof(double));
      for (j = 0; j < stair->npoly; j++) { waste[i][j] = 0.0; } }

   best = (int **)malloc(step*sizeof(int *));
   for (i = 0; i < step; i++) {
      best[i] = (int *)malloc((stair->npoly)*sizeof(int));
      for (j = 0; j < stair->npoly; j++) { best[i][j] = -1; } }

   for (i = 1; i < step; i++) {
      best[i][stair->npoly-2-i] = stair->npoly - 1 - i; }

   init = (double **)malloc((stair->npoly)*sizeof(double *));
   for (i = 0; i < stair->npoly; i++) {
      init[i] = (double *)malloc((stair->npoly)*sizeof(double));
      for (j = 0; j < stair->npoly; j++) { init[i][j] = 0.0; } }

   for (i = 0; i < stair->npoly; i++) {
      for (j = i+2; j < stair->npoly; j++) {
	 skew1 = area_minskew(&(stair->freg[j]));
	 skew2 = area_minskew(&(stair->freg[j-1]));
	 cap1 = stair->freg[i].capac;
	 cap2 = stair->freg[j-1].capac;
         init[i][j] = (skew1 - skew2)*(cap1 - cap2) + init[i][j-1]; } }

/* Done Initialization */

/* Initialization for Dynamic Programming */

   for (i = 0; i < stair->npoly; i++) {
      waste[0][i] = init[i][stair->npoly-1];
   }

   for (k = 1; k < step - 1; k++) {
      for (i = 0; i < stair->npoly-k-1; i++) {
         l = i + 1;
         waste[k][i] = init[i][l] + waste[k-1][l];
         best[k][i] = l;
         for (l++; l < stair->npoly-k; l++) {
            double temp = init[i][l] + waste[k-1][l];
            if (temp < waste[k][i]) {
               waste[k][i] = temp;
               best[k][i] = l;
            }
         }
      }
   }

   result->npoly = step;
   CopyRegionFrom(&(result->freg[0]), &(stair->freg[0]));
   for (i = 1, j = step-2, l = 0; i < step - 1; i++, j--) {
      l = best[j][l];
      CopyRegionFrom(&(result->freg[i]), &(stair->freg[l]));
   }
   CopyRegionFrom(&(result->freg[i]), &(stair->freg[stair->npoly-1]));

/* Free space */

   for (i = 0; i < stair->npoly; i++) { free((double *)init[i]); }
   free((double **)init);
   for (i = 0; i < step; i++) { free((double *)waste[i]); }
   free((double **)waste);
   for (i = 0; i < step; i++) { free((int *)best[i]); }
   free((int **)best);
}


