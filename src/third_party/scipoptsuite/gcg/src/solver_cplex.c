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

/**@file solver_cplex.c
 * @brief  cplex solver for pricing problems
 * @author Gerald Gamrath
 * @author Alexander Gross
 * @author Hanna Franzen
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/
/* #define DEBUG_PRICING_ALL_OUTPUT */

#include <string.h>

#include "scip/scip.h"
#include "gcg.h"
#include "solver_cplex.h"
#include "pub_solver.h"
#include "pub_gcgcol.h"

#include "pricer_gcg.h"
#include "scip_misc.h"

#include "cplex.h"

#define CHECK_ZERO(x) { int _restat_;                                   \
      if( (_restat_ = (x)) != 0 )                                       \
      {                                                                 \
         SCIPerrorMessage("Error in pricing solver: CPLEX returned %d\n", _restat_); \
         retval = SCIP_INVALIDRESULT;                                   \
         goto TERMINATE;                                                \
      }                                                                 \
   }

#define CPX_LONG_MAX     9223372036800000000LL


#define SOLVER_NAME          "cplex"
#define SOLVER_DESC          "cplex solver for pricing problems"
#define SOLVER_PRIORITY       100
#define SOLVER_HEURENABLED   TRUE            /**< indicates whether the heuristic solving method of the solver should be enabled */
#define SOLVER_EXACTENABLED  TRUE            /**< indicates whether the exact solving method of the solver should be enabled */

#define DEFAULT_CHECKSOLS     TRUE   /**< should solutions of the pricing MIPs be checked for duplicity? */
#define DEFAULT_THREADS       1      /**< number of threads the CPLEX pricing solver is allowed to use (0: automatic) */
#define DEFAULT_STARTNODELIMIT       1000LL  /**< start node limit for heuristic pricing */
#define DEFAULT_STARTGAPLIMIT        0.2     /**< start gap limit for heuristic pricing */
#define DEFAULT_STARTSOLLIMIT        10LL    /**< start solution limit for heuristic pricing */
#define DEFAULT_NODELIMITFAC         1.25    /**< factor by which to increase node limit for heuristic pricing (1.0: add start limit) */
#define DEFAULT_STALLNODELIMITFAC    1.25    /**< factor by which to increase stalling node limit for heuristic pricing */
#define DEFAULT_GAPLIMITFAC          0.8     /**< factor by which to decrease gap limit for heuristic pricing (1.0: subtract start limit) */
#define DEFAULT_SOLLIMITFAC          1.5     /**< factor by which to increase solution limit for heuristic pricing (1.0: add start limit) */


/** pricing solver data */
struct GCG_SolverData
{
   SCIP*                 origprob;           /**< original problem SCIP instance */
   SCIP*                 masterprob;         /**< master problem SCIP instance */
   SCIP**                pricingprobs;       /**< array storing the SCIP instances for all pricing problems */
   int                   npricingprobs;      /**< number of pricing problems */
   CPXENVptr*            cpxenv;             /**< array of CPLEX environments for the pricing problems */
   CPXLPptr*             lp;                 /**< array of CPLEX problems for the pricing problems */
   int*                  nupdates;           /**< array storing the number of updates for all of the pricing problems */
   SCIP_Longint*         curnodelimit;       /**< current node limit per pricing problem */
   SCIP_Real*            curgaplimit;        /**< current gap limit per pricing problem */
   SCIP_Longint*         cursollimit;        /**< current solution limit per pricing problem */
   /**
    *  information about the basic pricing problem (without potential branching constraints)
    */
   SCIP_VAR***           pricingvars;        /**< array storing the variables of the pricing problems */
   SCIP_CONS***          pricingconss;       /**< array storing the constraints of the pricing problems */
   int*                  npricingvars;       /**< array storing the number of variables of the pricing problems */
   int*                  nbasicpricingconss; /**< array storing the basic number of constraints of the pricing problems */
   /**
    *  parameters
    */
   SCIP_Bool             checksols;          /**< should solutions of the pricing MIPs be checked for duplicity? */
   int                   threads;            /**< number of threads the CPLEX pricing solver is allowed to use (0: automatic) */
   SCIP_Longint          startnodelimit;     /**< start node limit for heuristic pricing */
   SCIP_Real             startgaplimit;      /**< start gap limit for heuristic pricing */
   SCIP_Longint          startsollimit;      /**< start solution limit for heuristic pricing */
   SCIP_Real             nodelimitfac;       /**< factor by which to increase node limit for heuristic pricing (1.0: add start limit) */
   SCIP_Real             gaplimitfac;        /**< factor by which to decrease gap limit for heuristic pricing (1.0: subtract start limit) */
   SCIP_Real             sollimitfac;        /**< factor by which to increase solution limit for heuristic pricing (1.0: add start limit) */
};

/*
 * local methods
 */

/** creates a CPLEX environment and builds the pricing problem */
static
SCIP_RETCODE buildProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_SOLVERDATA*       solverdata,         /**< solver data structure */
   SCIP*                 pricingprob,        /**< pricing problem */
   int                   probnr              /**< problem number */
   )
{
   SCIP_CONS** conss;
   SCIP_VAR** vars;
   SCIP_VAR** consvars;
   SCIP_Real* consvals;
   SCIP_VAR* var;
   double* varobj;
   double* varlb;
   double* varub;
   char* vartype;
   char** varnames = NULL;
   char** consnames = NULL;
   double* rhss;
   double* ranges;
   char* senses;
   double* coefs = NULL;
   int* rowidx;
   int* colidx;
   SCIP_Real lhs;
   SCIP_Real rhs;
   SCIP_VARTYPE type;
   SCIP_RETCODE retval;
   int nconss = 0;
   int nvars = 0;
   int status;
   int varidx;
   int nconsvars;
   int nnonzeros;
   int idx;
   int c;
   int i;
   int v;

   /* open CPLEX environment and create problem */
   solverdata->cpxenv[probnr] = CPXopenCPLEX(&status);
   solverdata->lp[probnr] = CPXcreateprob(solverdata->cpxenv[probnr], &status, SCIPgetProbName(pricingprob));
   solverdata->pricingprobs[probnr] = pricingprob;

   retval = SCIP_OKAY;

   /* set parameters */
   CHECK_ZERO( CPXsetdblparam(solverdata->cpxenv[probnr], CPX_PARAM_EPGAP, 0.0) );
   CHECK_ZERO( CPXsetdblparam(solverdata->cpxenv[probnr], CPX_PARAM_EPAGAP, 0.0) );
   CHECK_ZERO( CPXsetdblparam(solverdata->cpxenv[probnr], CPX_PARAM_EPRHS, SCIPfeastol(pricingprob)) );
   CHECK_ZERO( CPXsetdblparam(solverdata->cpxenv[probnr], CPX_PARAM_EPINT, SCIPfeastol(pricingprob)) );
   CHECK_ZERO( CPXsetintparam(solverdata->cpxenv[probnr], CPX_PARAM_THREADS, solverdata->threads) );
#ifdef DEBUG_PRICING_ALL_OUTPUT
   CHECK_ZERO( CPXsetintparam(solverdata->cpxenv[probnr], CPX_PARAM_SCRIND, CPX_ON) );
#endif

   /* set objective sense */
   assert(SCIPgetObjsense(pricingprob) == SCIP_OBJSENSE_MINIMIZE);
#if (CPX_VERSION_VERSION == 12 && CPX_VERSION_RELEASE >= 5) || CPX_VERSION_VERSION > 12
   CHECK_ZERO( CPXchgobjsen(solverdata->cpxenv[probnr], solverdata->lp[probnr], CPX_MIN) );
#else
   CPXchgobjsen(solverdata->cpxenv[probnr], solverdata->lp[probnr], CPX_MIN);
#endif

   conss = SCIPgetOrigConss(pricingprob);
   nconss = SCIPgetNOrigConss(pricingprob);
   vars = SCIPgetOrigVars(pricingprob);
   nvars = SCIPgetNOrigVars(pricingprob);

   /* arrays for storing the basic constraints and variables */
   solverdata->npricingvars[probnr] = nvars;
   solverdata->nbasicpricingconss[probnr] = nconss;

   SCIP_CALL( SCIPallocMemoryArray(scip, &solverdata->pricingvars[probnr], nvars) ); /*lint !e866*/
   SCIP_CALL( SCIPallocMemoryArray(scip, &solverdata->pricingconss[probnr], nconss) ); /*lint !e866*/

   /* temporary memory for storing all data about the variables */
   SCIP_CALL( SCIPallocBufferArray(scip, &varobj, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &vartype, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &varlb, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &varub, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &varnames, nvars) );

   /* temporary memory for storing data about the constraints */
   SCIP_CALL( SCIPallocBufferArray(scip, &rhss, nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &senses, nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &ranges, nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &consnames, nconss) );

   BMSclearMemoryArray(consnames, nconss);
   BMSclearMemoryArray(varnames, nvars);

   /* collect information about variables: bounds, objective function, name, type */
   for( i = 0; i < nvars; i++ )
   {
      var = vars[i];
      varidx = SCIPvarGetIndex(var);
      assert(0 <= varidx);
      assert(varidx < nvars);
      solverdata->pricingvars[probnr][varidx] = var;
      SCIP_CALL( SCIPcaptureVar(pricingprob, var) );

      varlb[varidx] = SCIPvarGetLbLocal(var);
      varub[varidx] = SCIPvarGetUbLocal(var);
      varobj[varidx] = SCIPvarGetObj(var);
      SCIP_CALL( SCIPduplicateBufferArray(scip, &varnames[varidx], SCIPvarGetName(var),
            (int)(strlen(SCIPvarGetName(var))+1)) ); /*lint !e866*/

      type = SCIPvarGetType(var);

      switch( type )
      {
      case SCIP_VARTYPE_BINARY:
         vartype[varidx] = 'B';
         break;
      case SCIP_VARTYPE_CONTINUOUS:
         vartype[varidx] = 'C';
         break;
      case SCIP_VARTYPE_INTEGER:
      case SCIP_VARTYPE_IMPLINT:
         vartype[varidx] = 'I';
         break;
      default:
         SCIPerrorMessage("invalid variable type\n");
         return SCIP_INVALIDDATA;
      }
   }

   /* collect right hand sides and ranges of the constraints, count total number of nonzeros */
   nnonzeros = 0;
   for( c = 0; c < nconss; ++c )
   {
      solverdata->pricingconss[probnr][c] = conss[c];
      SCIP_CALL( SCIPcaptureCons(pricingprob, conss[c]) );

      nnonzeros += GCGconsGetNVars(scip, conss[c]);
      lhs = GCGconsGetLhs(pricingprob, conss[c]);
      rhs = GCGconsGetRhs(pricingprob, conss[c]);
      SCIP_CALL( SCIPduplicateBufferArray(scip, &consnames[c], SCIPconsGetName(conss[c]),
            (int)(strlen(SCIPconsGetName(conss[c]))+1)) ); /*lint !e866*/

      if( SCIPisInfinity(scip, -lhs) )
      {
         senses[c] = 'L';
         rhss[c] = (double) rhs;
         ranges[c] = 0.0;
      }
      else if( SCIPisInfinity(scip, rhs) )
      {
         senses[c] = 'G';
         rhss[c] = (double) lhs;
         ranges[c] = 0.0;
      }
      else if( SCIPisEQ(scip, lhs, rhs) )
      {
         senses[c] = 'E';
         rhss[c] = (double) lhs;
         ranges[c] = 0.0;
      }
      else
      {
         assert(SCIPisLT(scip, lhs, rhs));
         senses[c] = 'R';
         rhss[c] = (double) lhs;
         ranges[c] = (double) (rhs - lhs);
      }
   }

   /* temporary memory for storing data about coefficients in the constraints */
   SCIP_CALL( SCIPallocBufferArray(scip, &consvars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &consvals, nvars) );

   /* temporary memory for storing nonzeros */
   SCIP_CALL( SCIPallocBufferArray(scip, &rowidx, nnonzeros) );
   SCIP_CALL( SCIPallocBufferArray(scip, &colidx, nnonzeros) );
   SCIP_CALL( SCIPallocBufferArray(scip, &coefs, nnonzeros) );

   /* collect nonzeros */
   for( c = 0, idx = 0; c < nconss; ++c )
   {
      nconsvars = GCGconsGetNVars(scip, conss[c]);
      SCIP_CALL( GCGconsGetVals(pricingprob, conss[c], consvals, nvars) );
      SCIP_CALL( GCGconsGetVars(pricingprob, conss[c], consvars, nvars) );

      /* get coefficients */
      for( v = 0; v < nconsvars; ++v )
      {
         rowidx[idx] = c;
         colidx[idx] = SCIPvarGetIndex(consvars[v]);
         coefs[idx] = (double)consvals[v];
         idx++;
      }
   }
   assert(idx == nnonzeros);

   /* add variables to CPLEX problem */
   CHECK_ZERO( CPXnewcols(solverdata->cpxenv[probnr], solverdata->lp[probnr], nvars, varobj, varlb, varub, vartype, varnames) );

   /* add empty constraints to CPLEX problem */
   CHECK_ZERO( CPXnewrows(solverdata->cpxenv[probnr], solverdata->lp[probnr], nconss, rhss, senses, ranges, consnames) );

   /* add variables to constraints in CPLEX problem */
   CHECK_ZERO( CPXchgcoeflist(solverdata->cpxenv[probnr], solverdata->lp[probnr], nnonzeros, rowidx, colidx, coefs) );

#ifdef WRITEPROBLEMS
   {
      char filename[SCIP_MAXSTRLEN];
      (void) SCIPsnprintf(filename, SCIP_MAXSTRLEN, "cplex-%s.lp", SCIPgetProbName(pricingprob));
      SCIPinfoMessage(pricingprob, NULL, "print pricing problem to %s\n", filename);
      CHECK_ZERO( CPXwriteprob(solverdata->cpxenv[probnr], solverdata->lp[probnr], filename, "lp") );
   }
#endif

 TERMINATE:
   /* free temporary memory */
   if( coefs != NULL )
   {
      SCIPfreeBufferArray(scip, &coefs);
      SCIPfreeBufferArray(scip, &colidx);
      SCIPfreeBufferArray(scip, &rowidx);

      SCIPfreeBufferArray(scip, &consvals);
      SCIPfreeBufferArray(scip, &consvars);

      for( v = nvars - 1; v >= 0; --v )
      {
         assert(varnames != NULL);
         SCIPfreeBufferArrayNull(scip, &varnames[v]);
      }

      for( c = nconss - 1; c >= 0; --c )
      {
         assert(consnames != NULL);
         SCIPfreeBufferArrayNull(scip, &consnames[c]);
      }

      SCIPfreeBufferArray(scip, &consnames);
      SCIPfreeBufferArray(scip, &ranges);
      SCIPfreeBufferArray(scip, &senses);
      SCIPfreeBufferArray(scip, &rhss);

      SCIPfreeBufferArray(scip, &varnames);
      SCIPfreeBufferArray(scip, &varub);
      SCIPfreeBufferArray(scip, &varlb);
      SCIPfreeBufferArray(scip, &vartype);
      SCIPfreeBufferArray(scip, &varobj);
   }

   return retval;
}


/** updates bounds and objective coefficients of variables in the given pricing problem */
static
SCIP_RETCODE updateVars(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_SOLVERDATA*       solverdata,         /**< solver data structure */
   SCIP*                 pricingprob,        /**< pricing problem */
   int                   probnr,             /**< problem number */
   SCIP_Bool             varobjschanged,     /**< have the objective coefficients changed? */
   SCIP_Bool             varbndschanged      /**< have the lower and upper bounds changed? */
   )
{
   SCIP_VAR** vars;
   double* varobj;
   double* bounds;
   int* objidx;
   int* updatevaridx;
   char* boundtypes;
   int nvars;
   int npricingvars;

   SCIP_RETCODE retval;
   int i;

   vars = SCIPgetOrigVars(pricingprob);
   nvars = SCIPgetNOrigVars(pricingprob);
   npricingvars = solverdata->npricingvars[probnr];

   assert(npricingvars == nvars);
   assert(npricingvars == CPXgetnumcols(solverdata->cpxenv[probnr], solverdata->lp[probnr]));

   retval = SCIP_OKAY;

   if( varobjschanged )
   {
      SCIP_CALL( SCIPallocBufferArray(scip, &objidx, npricingvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &varobj, npricingvars) );
   }

   if( varbndschanged)
   {
      SCIP_CALL( SCIPallocBufferArray(scip, &updatevaridx, 2 * npricingvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &boundtypes, 2 * npricingvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &bounds, 2 * npricingvars) );
   }

   /* get new bounds and objective coefficients of variables */
   for( i = 0; i < nvars; i++ )
   {
      SCIP_VAR* origvar;
      SCIP_VAR* var;
      int varidx;

      origvar = vars[i];
      varidx = SCIPvarGetIndex(origvar);
      assert(0 <= varidx);
      assert(varidx < npricingvars);

      if( SCIPgetStage(pricingprob) >= SCIP_STAGE_TRANSFORMED )
         var = SCIPvarGetTransVar(origvar);
      else
         var = origvar;

      updatevaridx[2 * (size_t)varidx] = varidx;
      updatevaridx[2 * (size_t)varidx + 1] = varidx;

      if( varbndschanged )
      {
         boundtypes[2 * (size_t)varidx] = 'L';
         boundtypes[2 * (size_t)varidx + 1] = 'U';
         bounds[2 * (size_t)varidx] = (double) SCIPvarGetLbGlobal(var);
         bounds[2 * (size_t)varidx + 1] = (double) SCIPvarGetUbGlobal(var);
      }

      if( varobjschanged )
      {
         objidx[varidx] = varidx;
         varobj[varidx] = SCIPvarGetObj(origvar);
      }
   }

   /* update bounds and objective coefficient of basic variables */
   if( varbndschanged )
   {
      CHECK_ZERO( CPXchgbds(solverdata->cpxenv[probnr], solverdata->lp[probnr], 2 * nvars, updatevaridx, boundtypes, bounds) );
   }
   if( varobjschanged )
   {
      CHECK_ZERO( CPXchgobj(solverdata->cpxenv[probnr], solverdata->lp[probnr], nvars, objidx, varobj) );
   }

TERMINATE:
   if( varbndschanged )
   {
      SCIPfreeBufferArray(scip, &bounds);
      SCIPfreeBufferArray(scip, &boundtypes);
      SCIPfreeBufferArray(scip, &updatevaridx);
   }
   if( varobjschanged)
   {
      SCIPfreeBufferArray(scip, &varobj);
      SCIPfreeBufferArray(scip, &objidx);
   }

   return retval;
}

/** updates branching constraints in the given pricing problem */
static
SCIP_RETCODE updateBranchingConss(
   SCIP*                 scip,               /**< SCIP data structure */
   GCG_SOLVERDATA*       solverdata,         /**< solver data structure */
   SCIP*                 pricingprob,        /**< pricing problem */
   int                   probnr              /**< problem number */
   )
{
   SCIP_CONS** conss;
   SCIP_VAR** consvars;
   SCIP_Real* consvals;
   char** newconsnames = NULL;
   char* newsenses;
   double* newrhss;
   double* newranges;
   double* newcoefs;
   int* newrowidx;
   int* newcolidx;
   int nconss;
   int nbasicpricingconss;
   int ncpxrows;
   int nnewconss = 0;
   int nnonzeros;
   int nvars;

   SCIP_RETCODE retval;
   int idx;
   int c;

   conss = SCIPgetOrigConss(pricingprob);
   nconss = SCIPgetNOrigConss(pricingprob);
   nbasicpricingconss = solverdata->nbasicpricingconss[probnr];

   nvars = SCIPgetNOrigVars(pricingprob);

   retval = SCIP_OKAY;

   ncpxrows = CPXgetnumrows(solverdata->cpxenv[probnr], solverdata->lp[probnr]);

   if( nbasicpricingconss < ncpxrows )
   {
      CHECK_ZERO( CPXdelrows(solverdata->cpxenv[probnr], solverdata->lp[probnr], nbasicpricingconss, ncpxrows - 1) );
   }

   nnewconss = nconss - nbasicpricingconss;

   if( nnewconss == 0 )
      return retval;

   /* temporary arrays for storing data about new constraints */
   SCIP_CALL( SCIPallocBufferArray(scip, &newsenses, nnewconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &newrhss, nnewconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &newranges, nnewconss) );
   SCIP_CALL( SCIPallocClearBufferArray(scip, &newconsnames, nnewconss) );

   /* get information about new constraints */
   nnonzeros = 0;

   for( c = 0; c < nconss; ++c )
   {
      SCIP_Real lhs;
      SCIP_Real rhs;
      int nconsvars;
      int considx;

      /* we assume that nothing changed about the basic constraints */
      if( c < nbasicpricingconss )
      {
         assert(conss[c] == solverdata->pricingconss[probnr][c]);
         continue;
      }

      considx = c - nbasicpricingconss;
      assert(considx >= 0);

      nconsvars = GCGconsGetNVars(scip, conss[c]);
      lhs = GCGconsGetLhs(pricingprob, conss[c]);
      rhs = GCGconsGetRhs(pricingprob, conss[c]);
      SCIP_CALL( SCIPduplicateBufferArray(scip, &newconsnames[considx], SCIPconsGetName(conss[c]),
            (int)(strlen(SCIPconsGetName(conss[c]))+1)) ); /*lint !e866*/

      if( SCIPisInfinity(scip, -lhs) )
      {
         newsenses[considx] = 'L';
         newrhss[considx] = (double) rhs;
         newranges[considx] = 0.0;
      }
      else if( SCIPisInfinity(scip, rhs) )
      {
         newsenses[considx] = 'G';
         newrhss[considx] = (double) lhs;
         newranges[considx] = 0.0;
      }
      else if( SCIPisEQ(scip, lhs, rhs) )
      {
         newsenses[considx] = 'E';
         newrhss[considx] = (double) lhs;
         newranges[considx] = 0.0;
      }
      else
      {
         assert(SCIPisLT(scip, lhs, rhs));
         newsenses[considx] = 'R';
         newrhss[considx] = (double) lhs;
         newranges[considx] = (double) (rhs - lhs);
      }

      nnonzeros += nconsvars;
   }

   /* temporary arrays for getting variables and coefficients in new constraints */
   SCIP_CALL( SCIPallocBufferArray(scip, &consvars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &consvals, nvars) );

   /* temporary arrays for storing data about new constraints */
   SCIP_CALL( SCIPallocBufferArray(scip, &newrowidx, nnonzeros) );
   SCIP_CALL( SCIPallocBufferArray(scip, &newcolidx, nnonzeros) );
   SCIP_CALL( SCIPallocBufferArray(scip, &newcoefs, nnonzeros) );

   /* collect coefficients in new constriants */
   for( c = 0, idx = 0; c < nconss; ++c )
   {
      int nconsvars;

      int v;

      /* we assume that nothing changed about the basic constraints */
      if( c < nbasicpricingconss )
      {
         assert(conss[c] == solverdata->pricingconss[probnr][c]);
         continue;
      }

      nconsvars = GCGconsGetNVars(scip, conss[c]);
      SCIP_CALL( GCGconsGetVars(pricingprob, conss[c], consvars, nvars) );
      SCIP_CALL( GCGconsGetVals(pricingprob, conss[c], consvals, nvars) );

      /* get coefficients */
      for( v = 0; v < nconsvars; ++v )
      {
         newrowidx[idx] = c;
         newcolidx[idx] = SCIPvarGetIndex(consvars[v]);
         newcoefs[idx] = (double)consvals[v];
         idx++;
      }
   }
   assert(idx == nnonzeros);

   /* add new constraints */
   CHECK_ZERO( CPXnewrows(solverdata->cpxenv[probnr], solverdata->lp[probnr], nnewconss, newrhss, newsenses, newranges, newconsnames) );
   CHECK_ZERO( CPXchgcoeflist(solverdata->cpxenv[probnr], solverdata->lp[probnr], nnonzeros, newrowidx, newcolidx, newcoefs) );

TERMINATE:
   if( nnewconss > 0 )
   {
      SCIPfreeBufferArray(scip, &newcoefs);
      SCIPfreeBufferArray(scip, &newcolidx);
      SCIPfreeBufferArray(scip, &newrowidx);

      SCIPfreeBufferArray(scip, &consvals);
      SCIPfreeBufferArray(scip, &consvars);

      for( c = nnewconss - 1; c >= 0; --c )
      {
         assert(newconsnames != NULL);
         SCIPfreeBufferArrayNull(scip, &newconsnames[c]);
      }

      SCIPfreeBufferArray(scip, &newconsnames);
      SCIPfreeBufferArray(scip, &newranges);
      SCIPfreeBufferArray(scip, &newrhss);
      SCIPfreeBufferArray(scip, &newsenses);
   }

   return retval;
}


/** solves the pricing problem with CPLEX */
static
SCIP_RETCODE solveCplex(
   SCIP*                 scip,               /**< SCIP data structure (master problem) */
   GCG_SOLVERDATA*       solverdata,         /**< solver data structure */
   SCIP*                 pricingprob,        /**< pricing problem */
   int                   probnr,             /**< problem number */
   SCIP_Real             dualsolconv,        /**< dual solution value of the corresponding convexity constraint */
   SCIP_Real*            lowerbound,         /**< pointer to store lower bound */
   int*                  ncols,              /**< pointer to store number of columns */
   GCG_PRICINGSTATUS*    status              /**< pointer to store the pricing status */
   )
{ /*lint -e715*/
   GCG_COL* col;
   SCIP_RETCODE retval;
   SCIP_Bool predisabled = FALSE;
   double* cplexsolvals;
   double objective;
   double upperbound;
   double* primsol = NULL;
   int curpreind = -1;
   int curadvind = -1;
   int nsolscplex;
   int numcols;
   int cpxstatus;
   int s;

   *ncols = 0;
   *status = GCG_PRICINGSTATUS_UNKNOWN;
   upperbound = SCIPinfinity(pricingprob);

   retval = SCIP_OKAY;

   numcols = CPXgetnumcols(solverdata->cpxenv[probnr], solverdata->lp[probnr]);
   assert(numcols == SCIPgetNOrigVars(pricingprob));

   SCIP_CALL( SCIPallocBufferArray(scip, &cplexsolvals, numcols) );
 SOLVEAGAIN:
   /* the optimization call */
   if( predisabled )
   {
      CHECK_ZERO( CPXprimopt(solverdata->cpxenv[probnr], solverdata->lp[probnr]) );
   }
   else
   {
      CHECK_ZERO( CPXmipopt(solverdata->cpxenv[probnr], solverdata->lp[probnr]) );
   }

   /* get number of solutions */
   nsolscplex = CPXgetsolnpoolnumsolns(solverdata->cpxenv[probnr], solverdata->lp[probnr]);

   /* get solution status */
   cpxstatus = CPXgetstat(solverdata->cpxenv[probnr], solverdata->lp[probnr]);

   /* handle CPLEX solution status */
   switch( cpxstatus )
   {
   /* pricing problem was solved to optimality */
   case CPXMIP_OPTIMAL: /* 101 */
      assert(nsolscplex > 0);
      *status = GCG_PRICINGSTATUS_OPTIMAL;
      CHECK_ZERO( CPXgetobjval(solverdata->cpxenv[probnr], solverdata->lp[probnr], &upperbound) );
      break;

   /* pricing problem was proven to be infeasible */
   case CPXMIP_INFEASIBLE: /* 103 */
      assert(nsolscplex == 0);
      *status = GCG_PRICINGSTATUS_INFEASIBLE;
      break;

   /* pricing problem is possibly unbounded */
   case CPXMIP_UNBOUNDED: /* 118 */
   case CPXMIP_INForUNBD: /* 119 */
   {
      int cpxretval;
      int dummy;

      SCIP_CALL( SCIPallocBufferArray(scip, &primsol, numcols) );

      CHECK_ZERO( CPXsolution(solverdata->cpxenv[probnr], solverdata->lp[probnr], &dummy, NULL, primsol, NULL, NULL, NULL) );

      assert(dummy == cpxstatus);

      cpxretval = CPXgetray(solverdata->cpxenv[probnr], solverdata->lp[probnr], cplexsolvals);

      if( cpxretval == 1254 )
      {
         assert(!predisabled);

         SCIPdebugMessage("   -> disable presolving in CPLEX to get primal ray\n");

         CHECK_ZERO( CPXgetintparam(solverdata->cpxenv[probnr], CPX_PARAM_PREIND, &curpreind) );
         CHECK_ZERO( CPXgetintparam(solverdata->cpxenv[probnr], CPX_PARAM_ADVIND, &curadvind) );

         CHECK_ZERO( CPXchgprobtype(solverdata->cpxenv[probnr], solverdata->lp[probnr], CPXPROB_FIXEDMILP) );

         CHECK_ZERO( CPXsetintparam(solverdata->cpxenv[probnr], CPX_PARAM_ADVIND, 0) );
         CHECK_ZERO( CPXsetintparam(solverdata->cpxenv[probnr], CPX_PARAM_PREIND, 0) );

         predisabled = TRUE;

         goto SOLVEAGAIN;
      }
      else
      {
         CHECK_ZERO( cpxretval );
      }

      SCIP_CALL( GCGcreateGcgCol(pricingprob, &col, probnr, solverdata->pricingvars[probnr], cplexsolvals, numcols, TRUE, SCIPinfinity(pricingprob)) );
      SCIP_CALL( GCGpricerAddCol(scip, col) );
      ++(*ncols);

      *status = GCG_PRICINGSTATUS_UNBOUNDED;

      SCIPfreeBufferArray(scip, &primsol);

      goto TERMINATE;
   }

   /* a heuristic pricing limit was reached and may be increased in the next round */
   /* @todo: CPXMIP_OPTIMAL_TOL = 102 seems to occur even if gaplimit is set to 0 */
   case CPXMIP_OPTIMAL_TOL: /* 102 */
      assert(nsolscplex > 0);
      if( solverdata->gaplimitfac < 1.0 )
         solverdata->curgaplimit[probnr] *= solverdata->gaplimitfac;
      else
         solverdata->curgaplimit[probnr] = MAX(solverdata->curgaplimit[probnr] - solverdata->startgaplimit, 0.0);
      SCIPdebugMessage("   -> gap limit reached, decreasing to %g\n", solverdata->curgaplimit[probnr]);
      *status = GCG_PRICINGSTATUS_SOLVERLIMIT;
      CHECK_ZERO( CPXgetobjval(solverdata->cpxenv[probnr], solverdata->lp[probnr], &upperbound) );
      break;

   case CPXMIP_NODE_LIM_FEAS: /* 105 */
      assert(nsolscplex > 0);
      if( solverdata->nodelimitfac > 1.0 )
         solverdata->curnodelimit[probnr] = (SCIP_Longint) (solverdata->curnodelimit[probnr] * solverdata->nodelimitfac);
      else
         solverdata->curnodelimit[probnr] += solverdata->startnodelimit;
      SCIPdebugMessage("   -> node limit reached, increasing to %"SCIP_LONGINT_FORMAT"\n", solverdata->curnodelimit[probnr]);
      *status = GCG_PRICINGSTATUS_SOLVERLIMIT;
      CHECK_ZERO( CPXgetobjval(solverdata->cpxenv[probnr], solverdata->lp[probnr], &upperbound) );
      break;

   case CPXMIP_SOL_LIM: /* 104 */
      assert(nsolscplex > 0);
      if( solverdata->sollimitfac > 1.0 )
         solverdata->cursollimit[probnr] = (SCIP_Longint) (solverdata->cursollimit[probnr] * solverdata->sollimitfac);
      else
         solverdata->cursollimit[probnr] += solverdata->startsollimit;
      SCIPdebugMessage("   -> solution limit reached, increasing to %"SCIP_LONGINT_FORMAT"\n", solverdata->cursollimit[probnr]);
      *status = GCG_PRICINGSTATUS_SOLVERLIMIT;
      CHECK_ZERO( CPXgetobjval(solverdata->cpxenv[probnr], solverdata->lp[probnr], &upperbound) );
      break;

   /* no feasible solution was found, not clear whether one exists or the problem is infeasible */
   case CPXMIP_NODE_LIM_INFEAS: /* 106 */
   case CPXMIP_TIME_LIM_INFEAS: /* 108 */
   case CPXMIP_MEM_LIM_INFEAS: /* 111 */
   case CPXMIP_ABORT_INFEAS: /* 114 */
      assert(nsolscplex ==  0);
      *status = GCG_PRICINGSTATUS_UNKNOWN;
      break;

   /* a feasible solution exists */
   case CPXMIP_TIME_LIM_FEAS: /* 107 */
   case CPXMIP_MEM_LIM_FEAS: /* 112 */
   case CPXMIP_ABORT_FEAS: /* 113 */
   case CPXMIP_FEASIBLE: /* 127 */
      assert(nsolscplex > 0);
      CHECK_ZERO( CPXgetobjval(solverdata->cpxenv[probnr], solverdata->lp[probnr], &upperbound) );
      *status = GCG_PRICINGSTATUS_UNKNOWN;
      break;

   default:
      *status = GCG_PRICINGSTATUS_UNKNOWN;
      goto TERMINATE;
   }

   CHECK_ZERO( CPXgetbestobjval(solverdata->cpxenv[probnr], solverdata->lp[probnr], lowerbound) );

   /* In case of optimality, it might happen that the "lower bound" returned by CPLEX exceeds the optimal solution value;
    * probably, this is not a bug but a feature
    */
   *lowerbound = MIN(*lowerbound, upperbound);

   SCIPdebugMessage("   -> pricing problem %d solved: cpxstatus=%d, nsols=%d, lowerbound=%g, upperbound=%g\n", probnr, cpxstatus, nsolscplex, *lowerbound, upperbound);

   assert(SCIPisFeasEQ(scip, *lowerbound, upperbound) || cpxstatus != CPXMIP_OPTIMAL);

#ifndef NDEBUG
   /* in debug mode, we check that the first solution in the solution pool is the incumbent */
   if( nsolscplex > 0 )
   {
      double oldobjective;

      CHECK_ZERO( CPXgetsolnpoolobjval(solverdata->cpxenv[probnr], solverdata->lp[probnr], 0, &oldobjective) );

      for( s = 1; s < nsolscplex; ++s )
      {
         CHECK_ZERO( CPXgetsolnpoolobjval(solverdata->cpxenv[probnr], solverdata->lp[probnr], s, &objective) );
         assert(SCIPisFeasGE(scip, objective, oldobjective));
      }
   }
#endif

   /* iterate over all CPLEX solutions and check for negative reduced costs; the first solution should always be the
    * incumbent, all other solutions are unsorted
    */
   for( s = 0; s < nsolscplex; ++s )
   {
      SCIP_SOL* sol;
      SCIP_Bool feasible;

      CHECK_ZERO( CPXgetsolnpoolobjval(solverdata->cpxenv[probnr], solverdata->lp[probnr], s, &objective) );

      CHECK_ZERO( CPXgetsolnpoolx(solverdata->cpxenv[probnr], solverdata->lp[probnr], s, cplexsolvals, 0, numcols - 1) );

      SCIP_CALL( SCIPcreateOrigSol(pricingprob, &sol, NULL) );
      SCIP_CALL( SCIPsetSolVals(pricingprob, sol, numcols, solverdata->pricingvars[probnr], cplexsolvals) );

      feasible = FALSE;

      /* check whether the solution is feasible */
      if( !solverdata->checksols )
      {
         SCIP_CALL( SCIPcheckSolOrig(pricingprob, sol, &feasible, FALSE, FALSE) );

         /* if the optimal solution is not feasible, we return GCG_PRICINGSTATUS_UNKNOWN as status */
         if( !feasible && s == 0 )
         {
            *status = GCG_PRICINGSTATUS_UNKNOWN;
         }
      }
      else
      {
         feasible = TRUE;
      }

      if( feasible )
      {
         SCIP_CALL( GCGcreateGcgColFromSol(pricingprob, &col, probnr, sol, FALSE, SCIPinfinity(pricingprob)) );
         SCIP_CALL( GCGpricerAddCol(scip, col) );
         ++(*ncols);
      }

      SCIP_CALL( SCIPfreeSol(pricingprob, &sol) );
   }

   assert(*status != GCG_PRICINGSTATUS_OPTIMAL || *ncols > 0);
 TERMINATE:
   if( predisabled )
   {
      assert(curpreind >= 0);
      assert(curadvind >= 0);
      CHECK_ZERO( CPXsetintparam(solverdata->cpxenv[probnr], CPX_PARAM_PREIND, curpreind) );
      CHECK_ZERO( CPXsetintparam(solverdata->cpxenv[probnr], CPX_PARAM_ADVIND, curadvind) );

      CHECK_ZERO( CPXchgprobtype(solverdata->cpxenv[probnr], solverdata->lp[probnr], CPXPROB_MILP) );
   }

   if( primsol != NULL )
      SCIPfreeBufferArray(scip, &primsol);

   SCIPfreeBufferArray(scip, &cplexsolvals);

   return retval;
}


/** destructor of pricing solver to free user data (called when SCIP is exiting) */
static
GCG_DECL_SOLVERFREE(solverFreeCplex)
{
   GCG_SOLVERDATA* solverdata;

   assert(scip != NULL);
   assert(solver != NULL);

   solverdata = GCGsolverGetData(solver);
   assert(solverdata != NULL);

   SCIPfreeMemory(scip, &solverdata);
   GCGsolverSetData(solver, NULL);

   return SCIP_OKAY;
}

/** solving process initialization method of pricing solver (called when branch and bound process is about to begin) */
static
GCG_DECL_SOLVERINITSOL(solverInitsolCplex)
{
   GCG_SOLVERDATA* solverdata;
   int npricingprobs;

   int i;

   assert(scip != NULL);
   assert(solver != NULL);

   solverdata = GCGsolverGetData(solver);
   assert(solverdata != NULL);

   solverdata->npricingprobs = GCGgetNPricingprobs(solverdata->origprob);
   npricingprobs = solverdata->npricingprobs;

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(solverdata->cpxenv), npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(solverdata->lp), npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(solverdata->nupdates), npricingprobs) );
   BMSclearMemoryArray(solverdata->nupdates, npricingprobs);

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(solverdata->pricingprobs), npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(solverdata->pricingvars), npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(solverdata->pricingconss), npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(solverdata->npricingvars), npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &(solverdata->nbasicpricingconss), npricingprobs) );
   BMSclearMemoryArray(solverdata->npricingvars, npricingprobs);
   BMSclearMemoryArray(solverdata->nbasicpricingconss, npricingprobs);

   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &solverdata->curnodelimit, npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &solverdata->curgaplimit, npricingprobs) );
   SCIP_CALL( SCIPallocBlockMemoryArray(scip, &solverdata->cursollimit, npricingprobs) );

   for( i = 0; i < npricingprobs; ++i )
   {
      if( GCGisPricingprobRelevant(solverdata->origprob, i) )
      {
         SCIP_CALL( buildProblem(solverdata->masterprob, solverdata, GCGgetPricingprob(solverdata->origprob, i), i) );
      }

      solverdata->curnodelimit[i] = solverdata->startnodelimit;
      solverdata->curgaplimit[i] = solverdata->startgaplimit;
      solverdata->cursollimit[i] = solverdata->startsollimit;
   }

   return SCIP_OKAY;
}

/** solving process deinitialization method of pricing solver (called before branch and bound process data is freed) */
static
GCG_DECL_SOLVEREXITSOL(solverExitsolCplex)
{
   GCG_SOLVERDATA* solverdata;
   SCIP_RETCODE retval;
   int npricingprobs;
   int i;
   int j;

   assert(scip != NULL);
   assert(solver != NULL);

   solverdata = GCGsolverGetData(solver);
   assert(solverdata != NULL);

   retval = SCIP_OKAY;

   npricingprobs = GCGgetNPricingprobs(solverdata->origprob);

   /* free pricing problems */
   for( i = 0; i < npricingprobs; ++i )
   {
      if( GCGisPricingprobRelevant(solverdata->origprob, i) )
      {
         /* free LP */
         CHECK_ZERO( CPXfreeprob(solverdata->cpxenv[i], &solverdata->lp[i]) );

         /* free environment */
         CHECK_ZERO( CPXcloseCPLEX(&solverdata->cpxenv[i]) );

         if( solverdata->nbasicpricingconss[i] > 0 )
         {
            /* release stored constraints */
            for( j = 0; j < solverdata->nbasicpricingconss[i]; ++j )
            {
               SCIP_CALL( SCIPreleaseCons(solverdata->pricingprobs[i], &solverdata->pricingconss[i][j]) );
            }
            SCIPfreeMemoryArray(scip, &(solverdata->pricingconss[i]));
         }

         if( solverdata->npricingvars[i] > 0 )
         {
            /* release stored constraints */
            for( j = 0; j < solverdata->npricingvars[i]; ++j )
            {
               SCIP_CALL( SCIPreleaseVar(solverdata->pricingprobs[i], &solverdata->pricingvars[i][j]) );
            }
            SCIPfreeMemoryArray(scip, &(solverdata->pricingvars[i]));
         }
      }
   }

 TERMINATE:
   SCIPfreeBlockMemoryArray(scip, &solverdata->cursollimit, solverdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &solverdata->curgaplimit, solverdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &solverdata->curnodelimit, solverdata->npricingprobs);

   SCIPfreeBlockMemoryArray(scip, &(solverdata->nbasicpricingconss), solverdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &(solverdata->npricingvars), solverdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &(solverdata->pricingconss), solverdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &(solverdata->pricingvars), solverdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &(solverdata->pricingprobs), solverdata->npricingprobs);

   SCIPfreeBlockMemoryArray(scip, &(solverdata->nupdates), solverdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &(solverdata->lp), solverdata->npricingprobs);
   SCIPfreeBlockMemoryArray(scip, &(solverdata->cpxenv), solverdata->npricingprobs);

   return retval;
}

#define solverInitCplex NULL
#define solverExitCplex NULL

/** update method for pricing solver, used to update solver specific pricing problem data */
static
GCG_DECL_SOLVERUPDATE(solverUpdateCplex)
{
   GCG_SOLVERDATA* solverdata;

   solverdata = GCGsolverGetData(solver);
   assert(solverdata != NULL);

   SCIPdebugMessage("CPLEX solver -- update data for problem %d: varobjschanged = %u, varbndschanged = %u, consschanged = %u\n",
      probnr, varobjschanged, varbndschanged, consschanged);

   /* update pricing problem information */
   SCIP_CALL( updateVars(solverdata->masterprob, solverdata, pricingprob, probnr, varobjschanged, varbndschanged) );
   if( consschanged )
   {
      SCIP_CALL( updateBranchingConss(solverdata->masterprob, solverdata, pricingprob, probnr) );
   }

   /* reset heuristic pricing limits */
   solverdata->curnodelimit[probnr] = solverdata->startnodelimit;
   solverdata->curgaplimit[probnr] = solverdata->startgaplimit;
   solverdata->cursollimit[probnr] = solverdata->startsollimit;

#ifdef WRITEPROBLEMS
   /* Print the pricing problem after updating:
    *  * after checking variable bounds, because they change in particular when a new generic branching subproblem is considered
    *  * but not after adding new branching constraints, since objectives will be set afterwards before solving
    */
   if( varbndschanged && !consschanged )
   {
      char filename[SCIP_MAXSTRLEN];

      ++(solverdata->nupdates[probnr]);

      (void) SCIPsnprintf(filename, SCIP_MAXSTRLEN, "cplex-%s-%d-%d.lp", SCIPgetProbName(pricingprob), SCIPgetNNodes(scip), solverdata->nupdates[probnr]);
      SCIPinfoMessage(pricingprob, NULL, "print pricing problem to %s\n", filename);
      CHECK_ZERO( CPXwriteprob(solverdata->cpxenv[probnr], solverdata->lp[probnr], filename, "lp") );
   }
#endif

   return SCIP_OKAY;
}

/** heuristic solving method of CPLEX solver */
static
GCG_DECL_SOLVERSOLVEHEUR(solverSolveHeurCplex)
{
   GCG_SOLVERDATA* solverdata;
   int ncols;
   SCIP_RETCODE retval;

   solverdata = GCGsolverGetData(solver);
   assert(solverdata != NULL);

   SCIPdebugMessage("calling heuristic pricing with CPLEX for pricing problem %d\n", probnr);

   retval = SCIP_OKAY;

   /* set heuristic limits */
   CHECK_ZERO( CPXsetlongparam(solverdata->cpxenv[probnr], CPX_PARAM_NODELIM, (long long) solverdata->curnodelimit[probnr]) );
   CHECK_ZERO( CPXsetdblparam(solverdata->cpxenv[probnr], CPX_PARAM_EPGAP, (double) solverdata->curgaplimit[probnr]) );
   CHECK_ZERO( CPXsetlongparam(solverdata->cpxenv[probnr], CPX_PARAM_INTSOLLIM, (long long) solverdata->cursollimit[probnr]) );

   /* solve the pricing problem and evaluate solution */
   SCIP_CALL( solveCplex(solverdata->masterprob, solverdata, pricingprob, probnr, dualsolconv, lowerbound, &ncols, status) );
   assert(*status != GCG_PRICINGSTATUS_OPTIMAL || ncols > 0);

 TERMINATE:
   return retval;
}

/** solving method for pricing solver which solves the pricing problem to optimality */
static
GCG_DECL_SOLVERSOLVE(solverSolveCplex)
{
   GCG_SOLVERDATA* solverdata;
   int ncols;
   SCIP_RETCODE retval;

   assert(solver != NULL);

   solverdata = GCGsolverGetData(solver);
   assert(solverdata != NULL);

   SCIPdebugMessage("calling exact pricing with CPLEX for pricing problem %d\n", probnr);

   retval = SCIP_OKAY;

   /* set limits to (infinite/zero) default values */
   CHECK_ZERO( CPXsetlongparam(solverdata->cpxenv[probnr], CPX_PARAM_NODELIM, CPX_LONG_MAX) );
   CHECK_ZERO( CPXsetdblparam(solverdata->cpxenv[probnr], CPX_PARAM_EPGAP, 0.0) );
   CHECK_ZERO( CPXsetlongparam(solverdata->cpxenv[probnr], CPX_PARAM_INTSOLLIM, CPX_LONG_MAX) );

   /* solve the pricing problem and evaluate solution */
   SCIP_CALL( solveCplex(solverdata->masterprob, solverdata, pricingprob, probnr, dualsolconv, lowerbound, &ncols, status) );
   assert(*status != GCG_PRICINGSTATUS_OPTIMAL || ncols > 0);

 TERMINATE:
   return retval;
}

/** creates the CPLEX pricing solver and includes it in GCG */
SCIP_RETCODE GCGincludeSolverCplex(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   GCG_SOLVERDATA* solverdata;

   SCIP_CALL( SCIPallocMemory(scip, &solverdata) );
   solverdata->origprob = GCGmasterGetOrigprob(scip);
   solverdata->masterprob = scip;

   SCIP_CALL( GCGpricerIncludeSolver(scip, SOLVER_NAME, SOLVER_DESC, SOLVER_PRIORITY,
         SOLVER_HEURENABLED, SOLVER_EXACTENABLED,
         solverUpdateCplex, solverSolveCplex, solverSolveHeurCplex, solverFreeCplex, solverInitCplex,
         solverExitCplex, solverInitsolCplex, solverExitsolCplex, solverdata));

   SCIP_CALL( SCIPaddBoolParam(solverdata->origprob, "pricingsolver/cplex/checksols",
         "should solutions of the pricing MIPs be checked for duplicity?",
         &solverdata->checksols, TRUE, DEFAULT_CHECKSOLS, NULL, NULL));

   SCIP_CALL( SCIPaddIntParam(solverdata->origprob, "pricingsolver/cplex/threads",
         "number of threads the CPLEX pricing solver is allowed to use (0: automatic)",
         &solverdata->threads, TRUE, DEFAULT_THREADS, 0, INT_MAX, NULL, NULL));

   SCIP_CALL( SCIPaddLongintParam(solverdata->origprob, "pricingsolver/cplex/startnodelimit",
         "start node limit for heuristic pricing",
         &solverdata->startnodelimit, TRUE, DEFAULT_STARTNODELIMIT, 0, CPX_LONG_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(solverdata->origprob, "pricingsolver/cplex/startgaplimit",
         "start gap limit for heuristic pricing",
         &solverdata->startgaplimit, TRUE, DEFAULT_STARTGAPLIMIT, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddLongintParam(solverdata->origprob, "pricingsolver/cplex/startsollimit",
         "start solution limit for heuristic pricing",
         &solverdata->startsollimit, TRUE, DEFAULT_STARTSOLLIMIT, 0, CPX_LONG_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(solverdata->origprob, "pricingsolver/cplex/nodelimitfac",
         "factor by which to increase node limit for heuristic pricing (1.0: add start limit)",
         &solverdata->nodelimitfac, TRUE, DEFAULT_NODELIMITFAC, 1.0, SCIPinfinity(solverdata->origprob), NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(solverdata->origprob, "pricingsolver/cplex/gaplimitfac",
         "factor by which to decrease gap limit for heuristic pricing (1.0: subtract start limit)",
         &solverdata->gaplimitfac, TRUE, DEFAULT_GAPLIMITFAC, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(solverdata->origprob, "pricingsolver/cplex/sollimitfac",
         "factor by which to increase solution limit for heuristic pricing (1.0: add start limit)",
         &solverdata->sollimitfac, TRUE, DEFAULT_SOLLIMITFAC, 1.0, SCIPinfinity(solverdata->origprob), NULL, NULL) );
   return SCIP_OKAY;
}
