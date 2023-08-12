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

/**@file   sepa_basis.c
 * @brief  basis separator
 * @author Jonas Witt
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

/*#define SCIP_DEBUG*/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "sepa_basis.h"
#include "sepa_master.h"
#include "gcg.h"
#include "relax_gcg.h"
#include "pricer_gcg.h"
#include "pub_gcgvar.h"


#ifdef WITH_GSL
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#endif

#define SEPA_NAME              "basis"
#define SEPA_DESC              "separator calculates a basis of the orig problem to generate cuts, which cut off the master lp sol"
#define SEPA_PRIORITY                100
#define SEPA_FREQ                     0
#define SEPA_MAXBOUNDDIST           1.0
#define SEPA_USESSUBSCIP           FALSE /**< does the separator use a secondary SCIP instance? */
#define SEPA_DELAY                 TRUE  /**< should separation method be delayed, if other separators found cuts? */

#define STARTMAXCUTS 50       /**< maximal cuts used at the beginning */


/*
 * Data structures
 */

/** separator data */
struct SCIP_SepaData
{
   SCIP_ROW**            mastercuts;         /**< cuts in the master problem */
   SCIP_ROW**            origcuts;           /**< cuts in the original problem */
   int                   norigcuts;          /**< number of cuts in the original problem */
   int                   nmastercuts;        /**< number of cuts in the master problem */
   int                   maxcuts;            /**< maximal number of allowed cuts */
   SCIP_ROW**            newcuts;            /**< new cuts to tighten original problem */
   int                   nnewcuts;           /**< number of new cuts */
   int                   maxnewcuts;         /**< maximal number of allowed new cuts */
   int                   round;              /**< number of separation round in probing LP of current node */
   int                   currentnodenr;      /**< number of current node */
   SCIP_ROW*             objrow;             /**< row with obj coefficients */
   SCIP_Bool             enable;             /**< parameter returns if basis separator is enabled */
   SCIP_Bool             enableobj;          /**< parameter returns if objective constraint is enabled */
   SCIP_Bool             enableobjround;     /**< parameter returns if rhs/lhs of objective constraint is rounded, when obj is int */
   SCIP_Bool             enableppcuts;       /**< parameter returns if cuts generated during pricing are added to newconss array */
   SCIP_Bool             enableppobjconss;   /**< parameter returns if objective constraint for each redcost of pp is enabled */
   SCIP_Bool             enableppobjcg;      /**< parameter returns if objective constraint for each redcost of pp is enabled during pricing */
   int                   separationsetting;  /**< parameter returns which parameter setting is used for separation */
   SCIP_Bool             chgobj;             /**< parameter returns if basis is searched with different objective */
   SCIP_Bool             chgobjallways;      /**< parameter returns if obj is not only changed in first iteration */
   SCIP_Bool             genobjconvex;       /**< parameter returns if objconvex is generated dynamically */
   SCIP_Bool             enableposslack;     /**< parameter returns if positive slack should influence the probing objective function */
   SCIP_Bool             forcecuts;          /**< parameter returns if cuts are forced to enter the LP */
   int                   posslackexp;        /**< parameter returns exponent of usage of positive slack */
   SCIP_Bool             posslackexpgen;     /**< parameter returns if exponent should be automatically generated */
   SCIP_Real             posslackexpgenfactor; /**< parameter returns factor for automatically generated exponent */
   int                   maxrounds;          /**< parameter returns maximum number of separation rounds in probing LP (-1 if unlimited) */
   int                   maxroundsroot;      /**< parameter returns maximum number of separation rounds in probing LP in root node (-1 if unlimited) */
   int                   mincuts;            /**< parameter returns number of minimum cuts needed to return *result = SCIP_Separated */
   SCIP_Real             objconvex;          /**< parameter return convex combination factor */
};

/*
 * Local methods
 */

/** allocates enough memory to hold more cuts */
static
SCIP_RETCODE ensureSizeCuts(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< separator data data structure */
   int                   size                /**< new size of cut arrays */
   )
{
   assert(scip != NULL);
   assert(sepadata != NULL);
   assert(sepadata->mastercuts != NULL);
   assert(sepadata->origcuts != NULL);
   assert(sepadata->norigcuts <= sepadata->maxcuts);
   assert(sepadata->norigcuts >= 0);
   assert(sepadata->nmastercuts <= sepadata->maxcuts);
   assert(sepadata->nmastercuts >= 0);

   if( sepadata->maxcuts < size )
   {
      int newmaxcuts = SCIPcalcMemGrowSize(scip, size);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(sepadata->mastercuts), sepadata->maxcuts, newmaxcuts) );
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(sepadata->origcuts), sepadata->maxcuts, newmaxcuts) );
      sepadata->maxcuts = newmaxcuts;
   }
   assert(sepadata->maxcuts >= size);

   return SCIP_OKAY;
}

/** allocates enough memory to hold more new cuts */
static
SCIP_RETCODE ensureSizeNewCuts(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_SEPADATA*        sepadata,           /**< separator data data structure */
   int                   size                /**< new size of cut arrays */
   )
{
   assert(scip != NULL);
   assert(sepadata != NULL);
   assert(sepadata->newcuts != NULL);
   assert(sepadata->nnewcuts <= sepadata->maxnewcuts);
   assert(sepadata->nnewcuts >= 0);

   if( sepadata->maxnewcuts < size )
   {
      int newmaxnewcuts = SCIPcalcMemGrowSize(scip, size);
      SCIP_CALL( SCIPreallocBlockMemoryArray(scip, &(sepadata->newcuts), sepadata->maxnewcuts, newmaxnewcuts) );
      sepadata->maxnewcuts = newmaxnewcuts;
   }
   assert(sepadata->maxnewcuts >= size);

   return SCIP_OKAY;
}

/** returns the result of the exponentiation for given exponent and basis (basis^exponent) */
static
SCIP_Real exponentiate(
   SCIP_Real            basis,               /**< basis for exponentiation */
   int                  exponent            /**< exponent for exponentiation */
   )
{
   SCIP_Real result;
   int i;

   assert(exponent >= 0);

   result = 1.0;
   for( i = 0; i < exponent; ++i )
   {
      result *= basis;
   }

   return result;
}

/**< Initialize probing objective coefficient for each variable with original objective. */
static
SCIP_RETCODE initProbingObjWithOrigObj(
   SCIP*                origscip,           /**< orig scip problem */
   SCIP_Bool            enableobj,          /**< returns if objective row was added to the lp */
   SCIP_Real            objfactor           /**< factor, the objective is multiplied with */
)
{
   SCIP_VAR** origvars;
   int norigvars;
   SCIP_VAR* origvar;

   SCIP_Real newobj;
   int i;

   assert(SCIPinProbing(origscip));

   origvars = SCIPgetVars(origscip);
   norigvars = SCIPgetNVars(origscip);

   /* loop over original variables */
   for( i = 0; i < norigvars; ++i )
   {
      /* get variable information */
      origvar = origvars[i];
      newobj = 0.0;

      /* if objective row is enabled consider also the original objective value */
      if( enableobj )
         newobj = objfactor * SCIPvarGetObj(origvar);

      SCIP_CALL( SCIPchgVarObjProbing(origscip, origvar, newobj) );
   }
   return SCIP_OKAY;
}

/**< Change probing objective coefficient for each variable by adding original objective
 *   to the probing objective.
 */
static
SCIP_RETCODE chgProbingObjAddingOrigObj(
   SCIP*                origscip,           /**< orig scip problem */
   SCIP_Real            objfactor,          /**< factor the additional part of the objective is multiplied with */
   SCIP_Real            objdivisor          /**< factor the additional part of the objective is divided with */
)
{
   SCIP_VAR** origvars;
   int norigvars;
   SCIP_VAR* origvar;

   SCIP_Real newobj;
   int i;

   assert(SCIPinProbing(origscip));

   origvars = SCIPgetVars(origscip);
   norigvars = SCIPgetNVars(origscip);

   /* loop over original variables */
   for( i = 0; i < norigvars; ++i )
   {
      /* get variable information */
      origvar = origvars[i];

      newobj = SCIPgetVarObjProbing(origscip, origvar) + (objfactor * SCIPvarGetObj(origvar))/ objdivisor ;

      SCIP_CALL( SCIPchgVarObjProbing(origscip, origvar, newobj) );
   }
   return SCIP_OKAY;
}

/**< Initialize probing objective coefficient for each variable depending on the current origsol.
 *
 *   If variable is at upper bound set objective to -1, if variable is at lower bound set obj to 1,
 *   else set obj to 0.
 *   Additionally, add original objective to the probing objective if this is enabled.
 */
static
SCIP_RETCODE initProbingObjUsingVarBounds(
   SCIP*                origscip,           /**< orig scip problem */
   SCIP_SEPADATA*       sepadata,           /**< separator specific data */
   SCIP_SOL*            origsol,            /**< orig solution */
   SCIP_Bool            enableobj,          /**< returns if objective row was added to the lp */
   SCIP_Real            objfactor           /**< factor the objective is multiplied with */
)
{
   SCIP_Bool enableposslack;
   int posslackexp;

   SCIP_VAR** origvars;
   int norigvars;
   SCIP_VAR* origvar;

   SCIP_Real lb;
   SCIP_Real ub;
   SCIP_Real solval;
   SCIP_Real newobj;
   SCIP_Real distance;

   int i;

   origvars = SCIPgetVars(origscip);
   norigvars = SCIPgetNVars(origscip);

   enableposslack = sepadata->enableposslack;
   posslackexp = sepadata->posslackexp;

   /* loop over original variables */
   for( i = 0; i < norigvars; ++i )
   {
      /* get variable information */
      origvar = origvars[i];
      lb = SCIPvarGetLbLocal(origvar);
      ub = SCIPvarGetUbLocal(origvar);
      solval = SCIPgetSolVal(origscip, origsol, origvar);

      assert(SCIPisFeasLE(origscip, solval, ub));
      assert(SCIPisFeasGE(origscip, solval, lb));

      /* if solution value of variable is at ub or lb initialize objective value of the variable
       * such that the difference to this bound is minimized
       */
      if( SCIPisFeasEQ(origscip, lb, ub) )
      {
         newobj = 0.0;
      }
      else if( SCIPisLT(origscip, ub, SCIPinfinity(origscip)) && SCIPisFeasLE(origscip, ub, solval) )
      {
         newobj = -1.0;
      }
      else if( SCIPisGT(origscip, lb, -SCIPinfinity(origscip)) && SCIPisFeasGE(origscip, lb, solval) )
      {
         newobj = 1.0;
      }
      else if( enableposslack )
      {
         /* compute distance from solution to variable bound */
         distance = MIN(solval - lb, ub - solval);

         assert(SCIPisFeasPositive(origscip, distance));

         /* check if distance is lower than 1 and compute factor */
         if( SCIPisLT(origscip, distance, 1.0) )
         {
            newobj = exponentiate(MAX(0.0, 1.0 - distance), posslackexp);

            /* check if algebraic sign has to be changed */
            if( SCIPisLT(origscip, distance, solval - lb) )
               newobj = -newobj;
         }
         else
         {
            newobj = 0.0;
         }
      }
      else
      {
         newobj = 0.0;
      }

      /* if objective row is enabled consider also the original objective value */
      if( enableobj )
         newobj = newobj + SCIPvarGetObj(origvar);

      SCIP_CALL( SCIPchgVarObjProbing(origscip, origvar, objfactor*newobj) );
   }

   return SCIP_OKAY;
}

/**< Change probing objective depending on the current origsol.
 *
 * Loop over all constraints lhs <= sum a_i*x_i <= rhs. If lhs == sum a_i*x_i^* add a_i to objective
 * of variable i and if rhs == sum a_i*x_i^* add -a_i to objective of variable i.
 */
static
SCIP_RETCODE chgProbingObjUsingRows(
   SCIP*                origscip,           /**< orig scip problem */
   SCIP_SEPADATA*       sepadata,           /**< separator data */
   SCIP_SOL*            origsol,            /**< orig solution */
   SCIP_Real            objfactor,          /**< factor the objective is multiplied with */
   SCIP_Real            objdivisor          /**< factor the objective is divided with */
)
{
   SCIP_Bool enableposslack;
   int posslackexp;

   SCIP_ROW** rows;
   int nrows;
   SCIP_ROW* row;
   SCIP_Real* vals;
   SCIP_VAR** vars;
   SCIP_COL** cols;
   int nvars;

   SCIP_Real lhs;
   SCIP_Real rhs;
   SCIP_Real* solvals;
   SCIP_Real activity;
   SCIP_Real factor;
   SCIP_Real objadd;
   SCIP_Real obj;
   SCIP_Real norm;
   SCIP_Real distance;

   int i;
   int j;

   rows = SCIPgetLPRows(origscip);
   nrows = SCIPgetNLPRows(origscip);

   enableposslack = sepadata->enableposslack;
   posslackexp = sepadata->posslackexp;

   assert(SCIPinProbing(origscip));

   SCIP_CALL( SCIPallocBufferArray(origscip, &solvals, SCIPgetNVars(origscip)) );
   SCIP_CALL( SCIPallocBufferArray(origscip, &vars, SCIPgetNVars(origscip)) );

   /* loop over constraint and check activity */
   for( i = 0; i < nrows; ++i )
   {
      row = rows[i];
      lhs = SCIProwGetLhs(row);
      rhs = SCIProwGetRhs(row);

      nvars = SCIProwGetNNonz(row);
      if( nvars == 0 || (sepadata->objrow != NULL && strcmp(SCIProwGetName(row),SCIProwGetName(sepadata->objrow)) == 0 ) )
         continue;

      /* get values, variables and solution values */
      vals = SCIProwGetVals(row);
      cols = SCIProwGetCols(row);
      for( j = 0; j < nvars; ++j )
      {
         vars[j] = SCIPcolGetVar(cols[j]);
      }

      activity = SCIPgetRowSolActivity(origscip, row, origsol);

      if( SCIPisFeasEQ(origscip, rhs, lhs) )
      {
         continue;
      }
      if( SCIPisLT(origscip, rhs, SCIPinfinity(origscip)) && SCIPisFeasLE(origscip, rhs, activity) )
      {
         factor = -1.0;
      }
      else if( SCIPisGT(origscip, lhs, -SCIPinfinity(origscip)) && SCIPisFeasGE(origscip, lhs, activity) )
      {
         factor = 1.0;
      }
      else if( enableposslack )
      {
         assert(!(SCIPisInfinity(origscip, rhs) && SCIPisInfinity(origscip, lhs)));
         assert(!(SCIPisInfinity(origscip, activity) && SCIPisInfinity(origscip, -activity)));

         /* compute distance from solution to row */
         if( SCIPisInfinity(origscip, rhs) && SCIPisGT(origscip, lhs, -SCIPinfinity(origscip)) )
            distance = activity - lhs;
         else if( SCIPisInfinity(origscip, lhs) && SCIPisLT(origscip, rhs, SCIPinfinity(origscip)) )
            distance = rhs - activity;
         else
            distance = MIN(activity - lhs, rhs - activity);

         assert(SCIPisFeasPositive(origscip, distance) || !SCIPisCutEfficacious(origscip, origsol, row));

         /* check if distance is lower than 1 and compute factor */
         if( SCIPisLT(origscip, distance, 1.0) )
         {
            factor = exponentiate(MAX(0.0, 1.0 - distance), posslackexp);

            /* check if algebraic sign has to be changed */
            if( SCIPisLT(origscip, distance, activity - lhs) )
               factor = -1.0*factor;
         }
         else
         {
            continue;
         }
      }
      else
      {
         continue;
      }

      norm = SCIProwGetNorm(row);

      /* loop over variables of the constraint and change objective */
      for( j = 0; j < nvars; ++j )
      {
         obj = SCIPgetVarObjProbing(origscip, vars[j]);
         objadd = (factor * vals[j]) / norm;

         SCIP_CALL( SCIPchgVarObjProbing(origscip, vars[j], obj + (objfactor * objadd) / objdivisor) );
      }
   }

   SCIPfreeBufferArray(origscip, &solvals);
   SCIPfreeBufferArray(origscip, &vars);

   return SCIP_OKAY;
}

#ifdef WITH_GSL
/** Get matrix (including nrows and ncols) of rows that are satisfied with equality by sol */
static
SCIP_RETCODE getEqualityMatrixGsl(
   SCIP*                scip,               /**< SCIP data structure */
   SCIP_SOL*            sol,                /**< solution */
   gsl_matrix**         matrix,             /**< pointer to store equality matrix */
   int*                 nrows,              /**< pointer to store number of rows */
   int*                 ncols,              /**< pointer to store number of columns */
   int*                 prerank             /**< pointer to store preprocessed rank */
)
{
   int* var2col;
   int* delvars;

   int nvar2col;
   int ndelvars;

   SCIP_ROW** lprows;
   int nlprows;

   SCIP_COL** lpcols;
   int nlpcols;

   int i;
   int j;

   *ncols = SCIPgetNLPCols(scip);
   nlprows = SCIPgetNLPRows(scip);
   lprows = SCIPgetLPRows(scip);
   nlpcols = SCIPgetNLPCols(scip);
   lpcols = SCIPgetLPCols(scip);

   *nrows = 0;

   ndelvars = 0;
   nvar2col = 0;

   SCIP_CALL( SCIPallocBufferArray(scip, &var2col, nlpcols) );
   SCIP_CALL( SCIPallocBufferArray(scip, &delvars, nlpcols) );

   /* loop over lp cols and check if it is at one of its bounds */
   for( i = 0; i < nlpcols; ++i )
   {
      SCIP_COL* lpcol;
      SCIP_VAR* lpvar;

      lpcol = lpcols[i];

      lpvar = SCIPcolGetVar(lpcol);

      if( SCIPisEQ(scip, SCIPgetSolVal(scip, sol, lpvar), SCIPcolGetUb(lpcol) )
         || SCIPisEQ(scip, SCIPgetSolVal(scip, sol, lpvar), SCIPcolGetLb(lpcol)) )
      {
         int ind;

         ind = SCIPcolGetIndex(lpcol);

         delvars[ndelvars] = ind;

         ++ndelvars;

         var2col[ind] = -1;
      }
      else
      {
         int ind;

         ind = SCIPcolGetIndex(lpcol);

         var2col[ind] = nvar2col;

         ++nvar2col;
      }
   }

   SCIPsortInt(delvars, ndelvars);

   *matrix = gsl_matrix_calloc(nlprows, nvar2col);

   *ncols = nvar2col;

   /* loop over lp rows and check if solution feasibility is equal to zero */
   for( i = 0; i < nlprows; ++i )
   {
      SCIP_ROW* lprow;

      lprow = lprows[i];

      /* if solution feasiblity is equal to zero, add row to matrix */
      if( SCIPisEQ(scip, SCIPgetRowSolFeasibility(scip, lprow, sol), 0.0) )
      {
         SCIP_COL** cols;
         SCIP_Real* vals;
         int nnonz;

         cols = SCIProwGetCols(lprow);
         vals = SCIProwGetVals(lprow);
         nnonz = SCIProwGetNNonz(lprow);

         /* get nonzero coefficients of row */
         for( j = 0; j < nnonz; ++j )
         {
            int ind;
            int pos;

            ind = SCIPcolGetIndex(cols[j]);
            assert(ind >= 0 && ind < nlpcols);

            if( !SCIPsortedvecFindInt(delvars, ind, ndelvars, &pos) )
            {
               gsl_matrix_set(*matrix, *nrows, var2col[ind], vals[j]);
            }
         }
         ++(*nrows);
      }
   }
   *nrows = nlprows;
   *prerank = ndelvars;

   SCIPfreeBufferArray(scip, &delvars);
   SCIPfreeBufferArray(scip, &var2col);

   return SCIP_OKAY;
}

/** get the rank of a given matrix */
static
SCIP_RETCODE getRank(
   SCIP*                scip,
   gsl_matrix*          matrix,
   int                  nrows,
   int                  ncols,
   int*                 rank
)
{
   gsl_matrix* matrixq;
   gsl_matrix* matrixr;

   gsl_vector* tau;
   gsl_vector* norm;
   gsl_permutation* permutation;

   int ranktmp;
   int signum;

   int i;

   matrixq = gsl_matrix_alloc(nrows, nrows);
   matrixr = gsl_matrix_alloc(nrows, ncols);

   norm = gsl_vector_alloc(ncols);
   tau = gsl_vector_alloc(MIN(nrows, ncols));

   permutation = gsl_permutation_alloc(ncols);

   gsl_linalg_QRPT_decomp(matrix, tau, permutation, &signum, norm);

   gsl_linalg_QR_unpack(matrix, tau, matrixq, matrixr);

   ranktmp = 0;

   for( i = 0; i < MIN(nrows, ncols); ++i )
   {
      SCIP_Real val;

      val = gsl_matrix_get(matrixr, i, i);

      if( SCIPisZero(scip, val) )
      {
         break;
      }
      ++(ranktmp);
   }

   *rank = ranktmp;

   gsl_matrix_free(matrixq);
   gsl_matrix_free(matrixr);

   gsl_vector_free(tau);
   gsl_vector_free(norm);

   gsl_permutation_free(permutation);

   return SCIP_OKAY;
}

/** Get rank (number of linear independent rows) of rows that are satisfied
 *   with equality by solution sol */
static
SCIP_RETCODE getEqualityRankGsl(
   SCIP*                scip,               /**< SCIP data structure */
   SCIP_SOL*            sol,                /**< solution */
   int*                 equalityrank        /**< pointer to store rank of rows with equality */
   )
{
   gsl_matrix* matrix;
   int nrows;
   int ncols;

   int prerank;
   int rowrank;

   SCIP_CALL( getEqualityMatrixGsl(scip, sol, &matrix, &nrows, &ncols, &prerank) );

   SCIP_CALL( getRank(scip, matrix, nrows, ncols, &rowrank) );

   gsl_matrix_free(matrix);

   *equalityrank = rowrank + prerank;

   return SCIP_OKAY;
}
#endif

/** add cuts based on the last objective function of the pricing problems, which did not yield any new columns
 *  (stating that the reduced cost are non-negative) */
static
SCIP_RETCODE addPPObjConss(
   SCIP*                scip,               /**< SCIP data structure */
   SCIP_SEPA*           sepa,               /**< separator basis */
   int                  ppnumber,           /**< number of pricing problem */
   SCIP_Real            dualsolconv,        /**< dual solution corresponding to convexity constraint */
   SCIP_Bool            newcuts,            /**< add cut to newcuts in sepadata? (otherwise add it just to the cutpool) */
   SCIP_Bool            probing             /**< add cut to probing LP? */
)
{
   SCIP_SEPADATA* sepadata;

   SCIP* pricingscip;

   SCIP_VAR** pricingvars;
   SCIP_VAR* var;

   int npricingvars;
   int nvars;

   char name[SCIP_MAXSTRLEN];

   int j;
   int k;

   SCIP_Real lhs;
   SCIP_Real rhs;

   sepadata = SCIPsepaGetData(sepa);

   nvars = 0;
   pricingscip = GCGgetPricingprob(scip, ppnumber);
   pricingvars = SCIPgetOrigVars(pricingscip);
   npricingvars = SCIPgetNOrigVars(pricingscip);

   if( !GCGisPricingprobRelevant(scip, ppnumber) || pricingscip == NULL )
      return SCIP_OKAY;

   lhs = dualsolconv;
   rhs = SCIPinfinity(scip);

   for( k = 0; k < GCGgetNIdenticalBlocks(scip, ppnumber); ++k )
   {
      SCIP_ROW* origcut;

      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "newconstraint_%d_%d_%d", SCIPsepaGetNCalls(sepa), ppnumber, k);

      SCIP_CALL( SCIPcreateEmptyRowUnspec(scip, &origcut, name, lhs, rhs, FALSE, FALSE, TRUE) );

      nvars = 0;

      for( j = 0; j < npricingvars ; ++j )
      {
         assert(GCGvarIsPricing(pricingvars[j]));

         if( !SCIPisEQ(scip, SCIPvarGetObj(pricingvars[j]), 0.0) )
         {
            var = GCGpricingVarGetOrigvars(pricingvars[j])[k];
            assert(var != NULL);
            SCIP_CALL( SCIPaddVarToRow(scip, origcut, var, SCIPvarGetObj(pricingvars[j])) );
            ++nvars;
         }
      }

      if( nvars > 0 )
      {
         if( newcuts )
         {
            SCIP_CALL( ensureSizeNewCuts(scip, sepadata, sepadata->nnewcuts + 1) );

            sepadata->newcuts[sepadata->nnewcuts] = origcut;
            SCIP_CALL( SCIPcaptureRow(scip, sepadata->newcuts[sepadata->nnewcuts]) );
            ++(sepadata->nnewcuts);

            SCIPdebugMessage("cut added to new cuts in relaxdata\n");
         }
         else
         {
            SCIP_CALL( SCIPaddPoolCut(scip, origcut) );
            SCIPdebugMessage("cut added to orig cut pool\n");
         }

         if( probing )
         {
            SCIP_CALL( SCIPaddRowProbing(scip, origcut) );
            SCIPdebugMessage("cut added to probing\n");
         }


      }
      SCIP_CALL( SCIPreleaseRow(scip, &origcut) );
   }

   return SCIP_OKAY;
}
/*
 * Callback methods of separator
 */

/** copy method for separator plugins (called when SCIP copies plugins) */
#define sepaCopyBasis NULL

/** destructor of separator to free user data (called when SCIP is exiting) */
static
SCIP_DECL_SEPAFREE(sepaFreeBasis)
{  /*lint --e{715}*/
   SCIP_SEPADATA* sepadata;

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   SCIPfreeBlockMemory(scip, &sepadata);

   return SCIP_OKAY;
}

/** initialization method of separator (called after problem was transformed) */
static
SCIP_DECL_SEPAINIT(sepaInitBasis)
{  /*lint --e{715}*/
   SCIP*   origscip;
   SCIP_SEPADATA* sepadata;

   SCIP_VAR** origvars;
   int norigvars;

   char name[SCIP_MAXSTRLEN];

   SCIP_Real obj;
   int i;

   SCIP_Bool enable;
   SCIP_Bool enableobj;

   assert(scip != NULL);

   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   origvars = SCIPgetVars(origscip);
   norigvars = SCIPgetNVars(origscip);

   SCIPdebugMessage("sepaInitBasis\n");

   enable = sepadata->enable;
   enableobj = sepadata->enableobj;

   sepadata->maxcuts = SCIPcalcMemGrowSize(scip, STARTMAXCUTS);
   sepadata->norigcuts = 0;
   sepadata->nmastercuts = 0;
   sepadata->maxnewcuts = SCIPcalcMemGrowSize(scip, STARTMAXCUTS);
   sepadata->nnewcuts = 0;
   sepadata->objrow = NULL;
   /* if separator is disabled do nothing */
   if( !enable )
   {
      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(sepadata->origcuts), sepadata->maxcuts) ); /*lint !e506*/
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(sepadata->mastercuts), sepadata->maxcuts) ); /*lint !e506*/
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(sepadata->newcuts), sepadata->maxnewcuts) ); /*lint !e506*/

   /* if objective row is enabled create row with objective coefficients */
   if( enableobj )
   {
      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "objrow");
      SCIP_CALL( SCIPcreateEmptyRowUnspec(origscip, &(sepadata->objrow), name, -SCIPinfinity(origscip), SCIPinfinity(origscip), TRUE, FALSE, TRUE) );

      for( i = 0; i < norigvars; ++i )
      {
         obj = SCIPvarGetObj(origvars[i]);
         SCIP_CALL( SCIPaddVarToRow(origscip, sepadata->objrow, origvars[i], obj) );
      }
   }

   return SCIP_OKAY;
}


/** deinitialization method of separator (called before transformed problem is freed) */
static
SCIP_DECL_SEPAEXIT(sepaExitBasis)
{  /*lint --e{715}*/
   SCIP* origscip;
   SCIP_SEPADATA* sepadata;
   SCIP_Bool enableobj;

   int i;

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);
   enableobj = sepadata->enableobj;
   assert(sepadata->nmastercuts == sepadata->norigcuts);

   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   for( i = 0; i < sepadata->norigcuts; i++ )
   {
      SCIP_CALL( SCIPreleaseRow(origscip, &(sepadata->origcuts[i])) );
   }

   for( i = 0; i < sepadata->nnewcuts; ++i )
   {
      if( sepadata->newcuts[i] != NULL )
         SCIP_CALL( SCIPreleaseRow(origscip, &(sepadata->newcuts[i])) );
   }

   if( enableobj )
      SCIP_CALL( SCIPreleaseRow(origscip, &(sepadata->objrow)) );

   SCIPfreeBlockMemoryArrayNull(scip, &(sepadata->origcuts), sepadata->maxcuts);
   SCIPfreeBlockMemoryArrayNull(scip, &(sepadata->mastercuts), sepadata->maxcuts);
   SCIPfreeBlockMemoryArrayNull(scip, &(sepadata->newcuts), sepadata->maxnewcuts);

   return SCIP_OKAY;
}

/** solving process initialization method of separator (called when branch and bound process is about to begin) */
static
SCIP_DECL_SEPAINITSOL(sepaInitsolBasis)
{  /*lint --e{715}*/
   SCIP_SEPADATA* sepadata;

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   sepadata->nmastercuts = 0;

   return SCIP_OKAY;
}


/** solving process deinitialization method of separator (called before branch and bound process data is freed) */
static
SCIP_DECL_SEPAEXITSOL(sepaExitsolBasis)
{  /*lint --e{715}*/
   SCIP_SEPADATA* sepadata;
   int i;

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);
   assert(sepadata->nmastercuts == sepadata->norigcuts);

   assert(GCGmasterGetOrigprob(scip) != NULL);

   for( i = 0; i < sepadata->nmastercuts; i++ )
   {
      SCIP_CALL( SCIPreleaseRow(scip, &(sepadata->mastercuts[i])) );
   }

   return SCIP_OKAY;
}


/** Initialize generic convex combination coefficient */
static
SCIP_RETCODE initGenconv(
   SCIP*                origscip,           /**< original SCIP data structure */
   SCIP_SEPADATA*       sepadata,           /**< separator data structure */
   SCIP_SOL*            origsol,            /**< current original solution */
   int                  nbasis,             /**< rank of constraint matrix */
   SCIP_Real*           convex              /**< pointer to store convex combination coefficient */
)
{  /*lint --e{715}*/
#ifdef WITH_GSL
   int rank;


   SCIP_CALL( getEqualityRankGsl(origscip, origsol, &rank) );

   *convex = 1.0* rank/nbasis;

   SCIPdebugMessage("use generic coefficient %d/%d = %f\n", rank, nbasis, *convex);

#else
   SCIPwarningMessage(origscip, "Gnu Scientific Library is not enabled! \n"
      "either set sepa/basis/genobjconvex = FALSE sepa/basis/posslackexpgen = FALSE \n"
      "or compile with GSL=true and include Gnu Scientific Library\n");
   *convex = sepadata->objconvex;
#endif

   return SCIP_OKAY;
}


/** Initialize objective as convex combination of so-called face objective function and original objective function */
static
SCIP_RETCODE initConvObj(
   SCIP*                origscip,           /**< original SCIP data structure */
   SCIP_SEPADATA*       sepadata,           /**< separator data structure */
   SCIP_SOL*            origsol,            /**< current original solution */
   SCIP_Real            convex,             /**< convex coefficient to initialize objective */
   SCIP_Bool            genericconv         /**< was convex coefficient calculated generically? */
)
{
   SCIP_Real objnormnull;
   SCIP_Real objnormcurrent;

   objnormnull = 1.0;
   objnormcurrent = 1.0;

   /* if coefficient is zero, only use original objective function */
   if( SCIPisEQ(origscip, convex, 0.0) )
   {
      SCIP_CALL( initProbingObjWithOrigObj(origscip, TRUE, 1.0) );
   }
   /* if coefficient is between zero and one, calculate convex combination */
   else if( SCIPisLT(origscip, convex, 1.0) )
   {
      SCIP_CALL( initProbingObjWithOrigObj(origscip, TRUE, 1.0) );
      objnormnull = SCIPgetObjNorm(origscip);

      SCIP_CALL( initProbingObjUsingVarBounds(origscip, sepadata, origsol, FALSE, convex) );
      SCIP_CALL( chgProbingObjUsingRows(origscip, sepadata, origsol, convex, 1.0) );

      objnormcurrent = SCIPgetObjNorm(origscip)/(convex);

      if( SCIPisEQ(origscip, objnormcurrent, 0.0) )
         SCIP_CALL( initProbingObjWithOrigObj(origscip, TRUE, 1.0) );
      else if( SCIPisGT(origscip, objnormnull, 0.0) )
         SCIP_CALL( chgProbingObjAddingOrigObj(origscip, (1.0 - convex) * objnormcurrent, objnormnull) );
   }
   /* if coefficient is one, only use so-called face objective function (based on activity of rows and variables) */
   else if( SCIPisEQ(origscip, convex, 1.0) )
   {
      SCIP_CALL( initProbingObjUsingVarBounds(origscip, sepadata, origsol, !genericconv && sepadata->enableobj, 1.0) );
      SCIP_CALL( chgProbingObjUsingRows(origscip, sepadata, origsol, 1.0, 1.0) );
   }

   return SCIP_OKAY;
}

/** LP solution separation method of separator */
static
SCIP_DECL_SEPAEXECLP(sepaExeclpBasis)
{  /*lint --e{715}*/

   SCIP* origscip;
   SCIP_SEPADATA* sepadata;

   SCIP_ROW** cuts;
   SCIP_ROW* mastercut;
   SCIP_ROW* origcut;
   SCIP_COL** cols;
   SCIP_VAR** roworigvars;
   SCIP_VAR** mastervars;
   SCIP_Real* mastervals;
   int ncols;
   int ncuts;
   SCIP_Real* vals;
   int nmastervars;

   SCIP_SOL* origsol;
   SCIP_Bool lperror;
   SCIP_Bool delayed;
   SCIP_Bool cutoff;
   SCIP_Bool infeasible;
   SCIP_Real obj;

   SCIP_Bool enable;
   SCIP_Bool enableobj;
   SCIP_Bool enableobjround;
   SCIP_Bool enableppobjconss;

   char name[SCIP_MAXSTRLEN];

   int i;
   int j;
   int iteration;
   int nbasis;
   int nlprowsstart;
   int nlprows;
   int maxrounds;
   SCIP_ROW** lprows;
   int nviolatedcuts;

   SCIP_Bool isroot;

   SCIP_RESULT resultdummy;

   SCIP_Bool enoughcuts;
   int maxcuts;

   int maxnsepastallrounds;

   SCIP_Real objreldiff;
   int nfracs;
   SCIP_Real stalllpobjval;
   SCIP_Real lpobjval;
   SCIP_Bool stalling;
   int nsepastallrounds;
   int stallnfracs;
   SCIP_LPSOLSTAT stalllpsolstat;


   assert(scip != NULL);
   assert(result != NULL);

   origscip = GCGmasterGetOrigprob(scip);
   assert(origscip != NULL);

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   SCIPdebugMessage("calling sepaExeclpBasis\n");

   *result = SCIP_DIDNOTFIND;

   enable = sepadata->enable;
   enableobj = sepadata->enableobj;
   enableobjround = sepadata->enableobjround;
   enableppobjconss = sepadata->enableppobjconss;

   /* if separator is disabled do nothing */
   if( !enable )
   {
      SCIPdebugMessage("separator is not enabled\n");
      *result = SCIP_DIDNOTRUN;
      return SCIP_OKAY;
   }

   /* ensure master LP is solved to optimality */
   if( SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL )
   {
      SCIPdebugMessage("master LP not solved to optimality, do no separation!\n");
      *result = SCIP_DIDNOTRUN;
      return SCIP_OKAY;
   }

   /* ensure pricing problems were not aggregated */
   if( GCGgetNRelPricingprobs(origscip) < GCGgetNPricingprobs(origscip) )
   {
      SCIPdebugMessage("aggregated pricing problems, do no separation!\n");
      *result = SCIP_DIDNOTRUN;
      return SCIP_OKAY;
   }

   /* ensure to separate current sol */
   SCIP_CALL( GCGrelaxUpdateCurrentSol(origscip) );

   /* do not separate if current solution is feasible */
   if( GCGrelaxIsOrigSolFeasible(origscip) )
   {
      SCIPdebugMessage("Current solution is feasible, no separation necessary!\n");
      *result = SCIP_DIDNOTRUN;
      return SCIP_OKAY;
   }

   /* reset information on separation rounds in probing LP at current node */
   if( sepadata->currentnodenr != SCIPnodeGetNumber(SCIPgetCurrentNode(scip)) )
   {
      sepadata->currentnodenr = SCIPnodeGetNumber(SCIPgetCurrentNode(scip));
      sepadata->round = 0;
   }

   isroot = SCIPgetCurrentNode(scip) == SCIPgetRootNode(scip);

   /* set maximum number of rounds at current node */
   maxrounds = (isroot ? sepadata->maxroundsroot : sepadata->maxrounds);

   /* if no limit on number of rounds, set maxrounds to INT_MAX */
   if( maxrounds == -1 )
      maxrounds = INT_MAX;

   /* get current original solution */
   origsol = GCGrelaxGetCurrentOrigSol(origscip);

   /* get trans objective value */
   obj = SCIPgetSolTransObj(origscip, origsol);

   /* get number of linearly independent rows needed for basis */
   nbasis = SCIPgetNLPCols(origscip);

   *result = SCIP_DIDNOTFIND;

   /* init iteration count of current sepa call */
   iteration = 0;

   /* set parameter setting for separation */
   SCIP_CALL( SCIPsetSeparating(origscip, (SCIP_PARAMSETTING) sepadata->separationsetting, TRUE) );

   /* disable rapid learning because it does not generate cuts */
   SCIP_CALL( SCIPsetIntParam(origscip, "separating/rapidlearning/freq", -1) );

   /* start probing */
   SCIP_CALL( SCIPstartProbing(origscip) );
   SCIP_CALL( SCIPnewProbingNode(origscip) );
   SCIP_CALL( SCIPconstructLP(origscip, &cutoff) );

   /* add origcuts to probing lp */
   for( i = 0; i < GCGsepaGetNCuts(scip); ++i )
   {
      if( SCIProwGetLPPos(GCGsepaGetOrigcuts(scip)[i]) == -1 )
         SCIP_CALL( SCIPaddRowProbing(origscip, GCGsepaGetOrigcuts(scip)[i]) );
   }

   /* add new cuts which did not cut off master sol to probing lp */
   for( i = 0; i < sepadata->nnewcuts; ++i )
   {
      if( SCIProwGetLPPos(sepadata->newcuts[i]) == -1 )
         SCIP_CALL( SCIPaddRowProbing(origscip, sepadata->newcuts[i]) );
   }

   /* store number of lp rows in the beginning */
   nlprowsstart = SCIPgetNLPRows(origscip);

   nsepastallrounds = 0;
   stalllpobjval = SCIP_REAL_MIN;
   stallnfracs = INT_MAX;
   stalling = FALSE;

   maxcuts = 0;
   if( isroot )
      SCIP_CALL( SCIPgetIntParam(origscip, "separating/maxcutsroot", &maxcuts) );
   else
      SCIP_CALL( SCIPgetIntParam(origscip, "separating/maxcuts", &maxcuts) );

   maxnsepastallrounds = 0;
   if( isroot )
      SCIP_CALL( SCIPgetIntParam(origscip, "separating/maxstallroundsroot", &maxnsepastallrounds) );
   else
      SCIP_CALL( SCIPgetIntParam(origscip, "separating/maxstallrounds", &maxnsepastallrounds) );

   if( maxnsepastallrounds == -1 )
      maxnsepastallrounds = INT_MAX;

   stalllpsolstat = SCIP_LPSOLSTAT_NOTSOLVED;


   /* while the counter is smaller than the number of allowed rounds,
    * try to separate origsol via probing lp sol */
   while( sepadata->round < maxrounds && nsepastallrounds < maxnsepastallrounds )
   {
      SCIPdebugMessage("round %d of at most %d rounds\n", sepadata->round + 1, maxrounds);

      SCIP_CALL( SCIPapplyCutsProbing(origscip, &cutoff) );

      /* add new constraints if this is enabled */
      if( enableppobjconss && iteration == 0 )
      {
         SCIP_Real* dualsolconv;

         SCIPdebugMessage("add reduced cost cut for relevant pricing problems\n");

         SCIP_CALL( SCIPallocBufferArray(scip, &dualsolconv, GCGgetNPricingprobs(origscip)) );
         SCIP_CALL( GCGsetPricingObjs(scip, dualsolconv) );

         for( i = 0; i < GCGgetNPricingprobs(origscip); ++i )
         {
            SCIP_CALL( addPPObjConss(origscip, sepa, i, dualsolconv[i], FALSE, TRUE) );
         }

         SCIPfreeBufferArray(scip, &dualsolconv);
      }

      /* initialize objective of probing LP */
      if( sepadata->chgobj && (iteration == 0 || sepadata->chgobjallways) )
      {
         SCIPdebugMessage("initialize objective function\n");
         if( sepadata->genobjconvex )
         {
            SCIP_Real genconvex;

            SCIP_CALL( initGenconv(origscip, sepadata, origsol, nbasis, &genconvex) );

            SCIP_CALL( initConvObj(origscip, sepadata, origsol, genconvex, TRUE) );
         }
         else
         {
            SCIPdebugMessage("use given coefficient %g\n", sepadata->objconvex);

            if( sepadata->enableposslack && sepadata->posslackexpgen )
            {
               SCIP_Real genconvex;
               SCIP_Real factor;

               factor = sepadata->posslackexpgenfactor;

               SCIP_CALL( initGenconv(origscip, sepadata, origsol, nbasis, &genconvex) );

               sepadata->posslackexp = (int) (SCIPceil(origscip, factor/(1.0 - genconvex)) + 0.5);

               SCIPdebugMessage("exponent = %d\n", sepadata->posslackexp);

            }
            SCIP_CALL( initConvObj(origscip, sepadata, origsol, sepadata->objconvex, FALSE) );
         }
      }

      /* update rhs/lhs of objective constraint and add it to probing LP, if it exists (only in first iteration) */
      if( enableobj && iteration == 0 )
      {
         SCIPdebugMessage("initialize original objective cut\n");

         /* round rhs/lhs of objective constraint, if it exists, obj is integral and this is enabled */
         if( SCIPisObjIntegral(origscip) && enableobjround )
         {
            SCIPdebugMessage("round lhs up\n");
            obj = SCIPceil(origscip, obj);
         }

         SCIP_CALL( SCIPchgRowLhs(origscip, sepadata->objrow, obj) );
         SCIP_CALL( SCIPchgRowRhs(origscip, sepadata->objrow, SCIPinfinity(origscip)) );

         SCIPdebugMessage("add original objective cut to probing LP\n");

         /* add row to probing lp */
         SCIP_CALL( SCIPaddRowProbing(origscip, sepadata->objrow) );
      }

      SCIPdebugMessage("solve probing LP\n");

      /* solve probing lp */
      SCIP_CALL( SCIPsolveProbingLP(origscip, -1, &lperror, &cutoff) );

      assert(!lperror);

      /* check if we are stalling
       * We are stalling if
       *   the LP value did not improve and
       *   the number of fractional variables did not decrease.
       */
      if( SCIPgetLPSolstat(origscip) == SCIP_LPSOLSTAT_OPTIMAL )
      {
         SCIP_CALL( SCIPgetLPBranchCands(origscip, NULL, NULL, NULL, &nfracs, NULL, NULL) );
         lpobjval = SCIPgetLPObjval(origscip);

         objreldiff = SCIPrelDiff(lpobjval, stalllpobjval);
         SCIPdebugMessage(" -> LP bound moved from %g to %g (reldiff: %g)\n",
            stalllpobjval, lpobjval, objreldiff);

         stalling = (objreldiff <= 1e-04 &&
             nfracs >= (0.9 - 0.1 * nsepastallrounds) * stallnfracs);

         stalllpobjval = lpobjval;
         stallnfracs = nfracs;
      }
      else
      {
         stalling = (stalllpsolstat == SCIPgetLPSolstat(origscip));
      }

      if( !stalling )
      {
         nsepastallrounds = 0;
      }
      else
      {
         nsepastallrounds++;
      }
      stalllpsolstat = SCIPgetLPSolstat(origscip);

      /* separate cuts in cutpool */
      SCIPdebugMessage("separate current LP sol in cutpool\n");
      SCIP_CALL( SCIPseparateSolCutpool(origscip, SCIPgetGlobalCutpool(origscip), NULL, isroot, &resultdummy) );

      enoughcuts = (SCIPgetNCuts(origscip) >= 2 * (SCIP_Longint)maxcuts) || (resultdummy == SCIP_NEWROUND);

      if( !enoughcuts )
      {
         /* separate current probing lp sol of origscip */
         SCIPdebugMessage("separate current LP solution\n");
         SCIP_CALL( SCIPseparateSol(origscip, NULL, isroot, isroot, FALSE, &delayed, &cutoff) );

         enoughcuts = enoughcuts || (SCIPgetNCuts(origscip) >= 2 * (SCIP_Longint)maxcuts) || (resultdummy == SCIP_NEWROUND);

         /* if we are close to the stall round limit, also call the delayed separators */
         if( !enoughcuts && delayed && !cutoff && nsepastallrounds >= maxnsepastallrounds-1)
         {
            SCIPdebugMessage("call delayed separators\n");
            SCIP_CALL( SCIPseparateSol(origscip, NULL, isroot, isroot, TRUE, &delayed, &cutoff) );
         }
      }

      if( !enoughcuts && !cutoff )
      {
         /* separate cuts in cutpool */
         SCIPdebugMessage("separate current LP sol in cutpool\n");
         SCIP_CALL( SCIPseparateSolCutpool(origscip, SCIPgetGlobalCutpool(origscip), NULL, isroot, &resultdummy) );

         enoughcuts = enoughcuts || (SCIPgetNCuts(origscip) >= 2 * (SCIP_Longint)maxcuts) || (resultdummy == SCIP_NEWROUND);
      }

      if( SCIPgetNCuts(origscip) == 0 && !cutoff )
      {
         /* separate cuts in delayed cutpool */
         SCIPdebugMessage("separate current LP sol in delayed cutpool\n");
         SCIP_CALL( SCIPseparateSolCutpool(origscip, SCIPgetDelayedGlobalCutpool(origscip), NULL, isroot, &resultdummy) );

         enoughcuts = enoughcuts || (SCIPgetNCuts(origscip) >= 2 * (SCIP_Longint)maxcuts) || (resultdummy == SCIP_NEWROUND);
      }

      /* if cut off is detected set result pointer and return SCIP_OKAY */
      if( cutoff )
      {
         *result = SCIP_CUTOFF;
         SCIP_CALL( SCIPendProbing(origscip) );

         /* disable separating again */
         SCIP_CALL( SCIPsetSeparating(origscip, SCIP_PARAMSETTING_OFF, TRUE) );

         return SCIP_OKAY;
      }

      /* separate cuts in cutpool */
      SCIP_CALL( SCIPseparateSolCutpool(origscip, SCIPgetGlobalCutpool(origscip), origsol, isroot, &resultdummy) );
      SCIP_CALL( SCIPseparateSolCutpool(origscip, SCIPgetDelayedGlobalCutpool(origscip), origsol, isroot, &resultdummy) );

      assert(sepadata->norigcuts == sepadata->nmastercuts);

      SCIPdebugMessage("%d cuts are in the original sepastore!\n", SCIPgetNCuts(origscip));

      /* get separated cuts */
      cuts = SCIPgetCuts(origscip);
      ncuts = SCIPgetNCuts(origscip);

      SCIP_CALL( ensureSizeCuts(scip, sepadata, sepadata->norigcuts + ncuts) );

      mastervars = SCIPgetVars(scip);
      nmastervars = SCIPgetNVars(scip);
      SCIP_CALL( SCIPallocBufferArray(scip, &mastervals, nmastervars) );

      /* using nviolated cuts is a workaround for a SCIP issue:
       * it might happen that non-violated cuts are added to the sepastore,
       * which could lead to an infinite loop
       */
      /* initialize nviolated counting the number of cuts in the sepastore
       * that are violated by the current LP solution */
      nviolatedcuts = 0;

      /* loop over cuts and transform cut to master problem (and safe cuts) if it seperates origsol */
      for( i = 0; i < ncuts; i++ )
      {
         SCIP_Bool colvarused;
         SCIP_Real shift;

         colvarused = FALSE;
         origcut = cuts[i];

         /* if cut is violated by LP solution, increase nviolatedcuts */
         if( SCIPisCutEfficacious(origscip, NULL, origcut) )
         {
            ++nviolatedcuts;
         }

         /* get columns and vals of the cut */
         ncols = SCIProwGetNNonz(origcut);
         cols = SCIProwGetCols(origcut);
         vals = SCIProwGetVals(origcut);

         /* get the variables corresponding to the columns in the cut */
         SCIP_CALL( SCIPallocBufferArray(scip, &roworigvars, ncols) );
         for( j = 0; j < ncols; j++ )
         {
            roworigvars[j] = SCIPcolGetVar(cols[j]);
            assert(roworigvars[j] != NULL);
            if( !GCGvarIsOriginal(roworigvars[j]) )
            {
               colvarused = TRUE;
               break;
            }
         }

         if( colvarused )
         {
            SCIPwarningMessage(origscip, "colvar used in original cut %s\n", SCIProwGetName(origcut));
            SCIPfreeBufferArray(scip, &roworigvars);
            continue;
         }

         if( !SCIPisCutEfficacious(origscip, origsol, origcut) )
         {
            if( !SCIProwIsLocal(origcut) )
               SCIP_CALL( SCIPaddPoolCut(origscip, origcut) );

            SCIPfreeBufferArray(scip, &roworigvars);

            continue;
         }

         /* add the cut to the original cut storage */
         sepadata->origcuts[sepadata->norigcuts] = origcut;
         SCIP_CALL( SCIPcaptureRow(origscip, sepadata->origcuts[sepadata->norigcuts]) );
         sepadata->norigcuts++;

         /* transform the original variables to master variables */
         shift = GCGtransformOrigvalsToMastervals(origscip, roworigvars, vals, ncols, mastervars, mastervals, nmastervars);

         /* create new cut in the master problem */
         (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "mc_basis_%s", SCIProwGetName(origcut));
         SCIP_CALL( SCIPcreateEmptyRowSepa(scip, &mastercut, sepa, name,
            ( SCIPisInfinity(scip, -SCIProwGetLhs(origcut)) ?
              SCIProwGetLhs(origcut) : SCIProwGetLhs(origcut) - SCIProwGetConstant(origcut) - shift),
            ( SCIPisInfinity(scip, SCIProwGetRhs(origcut)) ?
              SCIProwGetRhs(origcut) : SCIProwGetRhs(origcut) - SCIProwGetConstant(origcut) - shift),
            SCIProwIsLocal(origcut), TRUE, FALSE) );

         /* add master variables to the cut */
         SCIP_CALL( SCIPaddVarsToRow(scip, mastercut, nmastervars, mastervars, mastervals) );

         /* add the cut to the master problem and to the master cut storage */
         SCIP_CALL( SCIPaddRow(scip, mastercut, sepadata->forcecuts, &infeasible) );
         sepadata->mastercuts[sepadata->nmastercuts] = mastercut;
         SCIP_CALL( SCIPcaptureRow(scip, sepadata->mastercuts[sepadata->nmastercuts]) );
         sepadata->nmastercuts++;
         SCIP_CALL( GCGsepaAddMastercuts(scip, origcut, mastercut) );

         SCIP_CALL( SCIPreleaseRow(scip, &mastercut) );
         SCIPfreeBufferArray(scip, &roworigvars);
      }

      SCIPdebugMessage("%d cuts are in the master sepastore!\n", SCIPgetNCuts(scip));

      ++sepadata->round;
      ++iteration;

      if( SCIPgetNCuts(scip) >= sepadata->mincuts )
      {
         *result = SCIP_SEPARATED;

         SCIPfreeBufferArray(scip, &mastervals);
         break;
      }
      /* use nviolated cuts instead of number of cuts in sepastore,
       * because non-violated might be added to the sepastore */
      else if( nviolatedcuts == 0 )
      {
         SCIPfreeBufferArray(scip, &mastervals);
         break;
      }

      SCIPfreeBufferArray(scip, &mastervals);

      assert(sepadata->norigcuts == sepadata->nmastercuts );
   }

   SCIP_CALL( SCIPclearCuts(origscip) );

   lprows = SCIPgetLPRows(origscip);
   nlprows = SCIPgetNLPRows(origscip);

   assert(nlprowsstart <= nlprows);

   SCIP_CALL( ensureSizeNewCuts(scip, sepadata, sepadata->nnewcuts + nlprows - nlprowsstart) );

   for( i = nlprowsstart; i < nlprows; ++i )
   {
      if( SCIProwGetOrigintype(lprows[i]) == SCIP_ROWORIGINTYPE_SEPA )
	   {
         sepadata->newcuts[sepadata->nnewcuts] = lprows[i];
	      SCIP_CALL( SCIPcaptureRow(origscip, sepadata->newcuts[sepadata->nnewcuts]) );
	      ++(sepadata->nnewcuts);
	   }
   }

   /* end probing */
   SCIP_CALL( SCIPendProbing(origscip) );

   if( SCIPgetNCuts(scip) > 0 )
   {
      *result = SCIP_SEPARATED;
   }

   /* disable separating again */
   SCIP_CALL( SCIPsetSeparating(origscip, SCIP_PARAMSETTING_OFF, TRUE) );

   SCIPdebugMessage("exiting sepaExeclpBasis\n");

   return SCIP_OKAY;
}

/** arbitrary primal solution separation method of separator */
#define sepaExecsolBasis NULL

/*
 * separator specific interface methods
 */

/** creates the basis separator and includes it in SCIP */
SCIP_RETCODE SCIPincludeSepaBasis(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_SEPADATA* sepadata;

   /* create master separator data */
   SCIP_CALL( SCIPallocBlockMemory(scip, &sepadata) );

   sepadata->mastercuts = NULL;
   sepadata->origcuts = NULL;
   sepadata->norigcuts = 0;
   sepadata->nmastercuts = 0;
   sepadata->maxcuts = 0;
   sepadata->newcuts = NULL;
   sepadata->nnewcuts = 0;
   sepadata->maxnewcuts = 0;
   sepadata->objrow = NULL;
   sepadata->round = 0;
   sepadata->currentnodenr = -1;

   /* include separator */
   SCIP_CALL( SCIPincludeSepa(scip, SEPA_NAME, SEPA_DESC, SEPA_PRIORITY, SEPA_FREQ, SEPA_MAXBOUNDDIST,
         SEPA_USESSUBSCIP, SEPA_DELAY,
         sepaCopyBasis, sepaFreeBasis, sepaInitBasis, sepaExitBasis, sepaInitsolBasis, sepaExitsolBasis, sepaExeclpBasis, sepaExecsolBasis,
         sepadata) );

   /* add basis separator parameters */
   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/enable", "is basis separator enabled?",
         &(sepadata->enable), FALSE, TRUE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/enableobj", "is objective constraint of separator enabled?",
         &(sepadata->enableobj), FALSE, FALSE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/enableobjround", "round obj rhs/lhs of obj constraint if obj is int?",
         &(sepadata->enableobjround), FALSE, FALSE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/enableppcuts", "add cuts generated during pricing to newconss array?",
         &(sepadata->enableppcuts), FALSE, FALSE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/enableppobjconss", "is objective constraint for redcost of each pp of "
      "separator enabled?", &(sepadata->enableppobjconss), FALSE, FALSE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/enableppobjcg", "is objective constraint for redcost of each pp during "
      "pricing of separator enabled?", &(sepadata->enableppobjcg), FALSE, FALSE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/genobjconvex", "generated obj convex dynamically",
         &(sepadata->genobjconvex), FALSE, FALSE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/enableposslack", "should positive slack influence the probing objective "
      "function?", &(sepadata->enableposslack), FALSE, FALSE, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/posslackexp", "exponent of positive slack usage",
         &(sepadata->posslackexp), FALSE, 1, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/posslackexpgen", "automatically generated exponent?",
            &(sepadata->posslackexpgen), FALSE, FALSE, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/posslackexpgenfactor", "factor for automatically generated exponent",
            &(sepadata->posslackexpgenfactor), FALSE, 0.1, SCIPepsilon(GCGmasterGetOrigprob(scip)),
            SCIPinfinity(GCGmasterGetOrigprob(scip)), NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/objconvex", "convex combination factor (= 0.0, use original objective; = 1.0, use face objective)",
         &(sepadata->objconvex), FALSE, 0.0, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/paramsetting", "parameter returns which parameter setting is used for "
      "separation (default = 0, aggressive = 1, fast = 2", &(sepadata->separationsetting), FALSE, 0, 0, 2, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/chgobj", "parameter returns if basis is searched with different objective",
      &(sepadata->chgobj), FALSE, TRUE, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/maxrounds", "parameter returns maximum number of separation rounds in probing LP (-1 if unlimited)",
      &(sepadata->maxrounds), FALSE, -1, -1, INT_MAX , NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/maxroundsroot", "parameter returns maximum number of separation rounds in probing LP in root node (-1 if unlimited)",
      &(sepadata->maxroundsroot), FALSE, -1, -1, INT_MAX , NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/mincuts", "parameter returns number of minimum cuts needed to "
      "return *result = SCIP_Separated", &(sepadata->mincuts), FALSE, 50, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/chgobjallways", "parameter returns if obj is changed not only in the "
      "first round", &(sepadata->chgobjallways), FALSE, FALSE, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(GCGmasterGetOrigprob(scip), "sepa/" SEPA_NAME "/forcecuts", "parameter returns if cuts are forced to enter the LP ",
      &(sepadata->forcecuts), FALSE, FALSE, NULL, NULL) );

   return SCIP_OKAY;
}


/** returns the array of original cuts saved in the separator data */
SCIP_ROW** GCGsepaBasisGetOrigcuts(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_SEPA* sepa;
   SCIP_SEPADATA* sepadata;

   assert(scip != NULL);

   sepa = SCIPfindSepa(scip, SEPA_NAME);
   assert(sepa != NULL);

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   return sepadata->origcuts;
}

/** returns the number of original cuts saved in the separator data */
int GCGsepaBasisGetNOrigcuts(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_SEPA* sepa;
   SCIP_SEPADATA* sepadata;

   assert(scip != NULL);

   sepa = SCIPfindSepa(scip, SEPA_NAME);
   assert(sepa != NULL);

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   return sepadata->norigcuts;
}

/** returns the array of master cuts saved in the separator data */
SCIP_ROW** GCGsepaBasisGetMastercuts(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_SEPA* sepa;
   SCIP_SEPADATA* sepadata;

   assert(scip != NULL);

   sepa = SCIPfindSepa(scip, SEPA_NAME);
   assert(sepa != NULL);

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   return sepadata->mastercuts;
}

/** returns the number of master cuts saved in the separator data */
int GCGsepaBasisGetNMastercuts(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_SEPA* sepa;
   SCIP_SEPADATA* sepadata;

   assert(scip != NULL);

   sepa = SCIPfindSepa(scip, SEPA_NAME);
   assert(sepa != NULL);

   sepadata = SCIPsepaGetData(sepa);
   assert(sepadata != NULL);

   return sepadata->nmastercuts;
}

/** transforms cut in pricing variables to cut in original variables and adds it to newcuts array */
SCIP_RETCODE GCGsepaBasisAddPricingCut(
   SCIP*                scip,
   int                  ppnumber,
   SCIP_ROW*            cut
   )
{
   SCIP* origscip;
   SCIP_SEPA* sepa;
   SCIP_SEPADATA* sepadata;

   SCIP* pricingprob;
   SCIP_Real* vals;
   SCIP_COL** cols;
   SCIP_VAR** pricingvars;
   int nvars;

   int i;
   int j;
   int k;

   char name[SCIP_MAXSTRLEN];

   assert(GCGisMaster(scip));

   sepa = SCIPfindSepa(scip, SEPA_NAME);

   if( sepa == NULL )
   {
      SCIPerrorMessage("sepa basis not found\n");
      return SCIP_OKAY;
   }

   sepadata = SCIPsepaGetData(sepa);
   origscip = GCGmasterGetOrigprob(scip);
   pricingprob = GCGgetPricingprob(origscip, ppnumber);

   if( !sepadata->enableppcuts )
   {
      return SCIP_OKAY;
   }

   assert(!SCIProwIsLocal(cut));

   nvars = SCIProwGetNNonz(cut);
   cols = SCIProwGetCols(cut);
   vals = SCIProwGetVals(cut);

   if( nvars == 0 )
   {
      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPallocBufferArray(scip, &pricingvars, nvars) );

   for( i = 0; i < nvars; ++i )
   {
      pricingvars[i] = SCIPcolGetVar(cols[i]);
      assert(pricingvars[i] != NULL);
   }

   for( k = 0; k < GCGgetNIdenticalBlocks(origscip, ppnumber); ++k )
   {
      SCIP_ROW* origcut;

      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "ppcut_%d_%d_%d", SCIPsepaGetNCalls(sepa), ppnumber, k);

      SCIP_CALL( SCIPcreateEmptyRowUnspec(origscip, &origcut, name,
         ( SCIPisInfinity(pricingprob, -SCIProwGetLhs(cut)) ?
            -SCIPinfinity(origscip) : SCIProwGetLhs(cut) - SCIProwGetConstant(cut)),
         ( SCIPisInfinity(pricingprob, SCIProwGetRhs(cut)) ?
            SCIPinfinity(origscip) : SCIProwGetRhs(cut) - SCIProwGetConstant(cut)),
             FALSE, FALSE, TRUE) );

      for( j = 0; j < nvars ; ++j )
      {
         SCIP_VAR* var;

         if( !GCGvarIsPricing(pricingvars[j]) )
         {
            nvars = 0;
            break;
         }
         assert(GCGvarIsPricing(pricingvars[j]));

         var = GCGpricingVarGetOrigvars(pricingvars[j])[k];
         assert(var != NULL);

         SCIP_CALL( SCIPaddVarToRow(origscip, origcut, var, vals[j]) );
      }

      if( nvars > 0 )
      {
         SCIP_CALL( ensureSizeNewCuts(scip, sepadata, sepadata->nnewcuts + 1) );

         sepadata->newcuts[sepadata->nnewcuts] = origcut;
         SCIP_CALL( SCIPcaptureRow(scip, sepadata->newcuts[sepadata->nnewcuts]) );
         ++(sepadata->nnewcuts);

         SCIPdebugMessage("cut added to orig cut pool\n");
      }
      SCIP_CALL( SCIPreleaseRow(origscip, &origcut) );
   }

   SCIPfreeBufferArray(scip, &pricingvars);

   return SCIP_OKAY;
}

/** add cuts which are due to the latest objective function of the pricing problems
 *  (reduced cost non-negative) */
SCIP_RETCODE SCIPsepaBasisAddPPObjConss(
   SCIP*                scip,               /**< SCIP data structure */
   int                  ppnumber,           /**< number of pricing problem */
   SCIP_Real            dualsolconv,        /**< dual solution corresponding to convexity constraint */
   SCIP_Bool            newcuts             /**< add cut to newcuts in sepadata? (otherwise add it just to the cutpool) */
)
{
   SCIP_SEPA* sepa;

   assert(GCGisMaster(scip));

   sepa = SCIPfindSepa(scip, SEPA_NAME);

   if( sepa == NULL )
   {
      SCIPerrorMessage("sepa basis not found\n");
      return SCIP_OKAY;
   }

   SCIP_CALL( addPPObjConss(GCGmasterGetOrigprob(scip), sepa, ppnumber, dualsolconv, newcuts, FALSE) );

   return SCIP_OKAY;
}
