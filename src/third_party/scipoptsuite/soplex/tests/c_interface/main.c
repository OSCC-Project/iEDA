#include <stdio.h>
#include <soplex_interface.h>
#include <assert.h>
#include <string.h>

void test_real(void)
{
   /* create LP via columns */

   void *soplex = SoPlex_create();
   double infty = 10e+20;
   double colentries1[] = {-1.0};
   double colentries2[] = {1.0};
   double lhs[] = {-10.0};
   double primal[] = {0.0,0.0};

   /* minimize */
   SoPlex_setIntParam(soplex, 0, -1);

   /* add columns */
   SoPlex_addColReal(soplex, colentries1, 1, 1, 1.0, 0.0, infty);
   SoPlex_addColReal(soplex, colentries2, 1, 1, 1.0, -infty, infty);
   assert(SoPlex_numRows(soplex) == 1);
   assert(SoPlex_numCols(soplex) == 2);

   /* set lhs of constraint */
   SoPlex_changeLhsReal(soplex, lhs, 1);

   /* optimize and get solution and objective value */
   int result = SoPlex_optimize(soplex);
   assert(result == 1);
   SoPlex_getPrimalReal(soplex, primal, 2);
   assert(primal[0] == 0.0 && primal[1] == -10.0);
   assert(SoPlex_objValueReal(soplex) == -10.0);

	SoPlex_free(soplex);

   /* create LP via rows */

   void *soplex2 = SoPlex_create();
   double rowentries1[] = {-1.0, 1.0};
   double lb[] = {0.0, -infty};
   double ub[] = {infty, infty};
   double obj[] = {1.0, 1.0};

   /* minimize */
   SoPlex_setIntParam(soplex2, 0, -1);

   /* add row */
   SoPlex_addRowReal(soplex2, rowentries1, 2, 2, -10.0, infty);

   /* add variable bounds */
   SoPlex_changeBoundsReal(soplex2, lb, ub, 2);
   assert(SoPlex_numRows(soplex2) == 1);
   assert(SoPlex_numCols(soplex2) == 2);

   /* add objective */
   SoPlex_changeObjReal(soplex2, obj, 2);

   /* optimize and get solution and objective value */
   result = SoPlex_optimize(soplex2);
   assert(result == 1);
   SoPlex_getPrimalReal(soplex2, primal, 2);
   assert(primal[0] == 0.0 && primal[1] == -10.0);
   assert(SoPlex_objValueReal(soplex2) == -10.0);

	SoPlex_free(soplex2);
}

#ifdef SOPLEX_WITH_GMP
void test_rational(void)
{
   /* create LP via rows */

   void *soplex = SoPlex_create();
   long infty = 1000000;
   long rownums[] = {-1, 1};
   long rowdenoms[] = {1, 1};
   long objnums[] = {1, 1};
   long objdenoms[] = {1, 1};
   double primal[] = {0.0,0.0};

   /* use rational solver */
   SoPlex_setRational(soplex);

   /* minimize */
   SoPlex_setIntParam(soplex, 0, -1);

   /* add row and set objective function */
   SoPlex_addRowRational(soplex, rownums, rowdenoms, 2, 2, 1, 5, infty, 1);
   SoPlex_changeObjRational(soplex, objnums, objdenoms, 2);

   /* optimize and check rational solution and objective value */
   int result = SoPlex_optimize(soplex);
   assert(result == 1);
   assert(strcmp(SoPlex_getPrimalRationalString(soplex, 2), "0 1/5 ") == 0);
   assert(strcmp(SoPlex_objValueRationalString(soplex), "1/5") == 0);

   SoPlex_free(soplex);

   /* create LP via columns */
   void *soplex2 = SoPlex_create();
   long colnums1[] = {-1};
   long coldenoms1[] = {1};
   long colnums2[] = {1};
   long coldenoms2[] = {1};
   long lhsnums[] = {-1};
   long lhsdenoms[] = {5};

   /* use rational solver */
   SoPlex_setRational(soplex2);

   /* minimize */
   SoPlex_setIntParam(soplex2, 0, -1);

   /* add cols */
   SoPlex_addColRational(soplex2, colnums1, coldenoms1, 1, 1, 1, 5, 0, 1, infty, 1);
   SoPlex_addColRational(soplex2, colnums2, coldenoms2, 1, 1, 1, 5, -infty, 1, infty, 1);

   /* add bounds to constraint */
   SoPlex_changeLhsRational(soplex2, lhsnums, lhsdenoms, 1);

   /* optimize and check rational solution and objective value */
   result = SoPlex_optimize(soplex2);
   assert(result == 1);
   assert(strcmp(SoPlex_getPrimalRationalString(soplex2, 2), "0 -1/5 ") == 0);
   assert(strcmp(SoPlex_objValueRationalString(soplex2), "-1/25") == 0);

   SoPlex_free(soplex2);
}
#endif

int main(void)
{
   printf("testing real... \n");
   test_real();

   #ifdef SOPLEX_WITH_GMP
   printf("\n");
   printf("testing rational... \n");
   test_rational();
   #endif
}
