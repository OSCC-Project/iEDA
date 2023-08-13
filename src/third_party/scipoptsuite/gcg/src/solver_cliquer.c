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

/**@file   solver_cliquer.c
 * @brief  heuristic solver for pricing problems that solves independent set problems with cliquer
 * @author Henri Lotze
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "solver_cliquer.h"
#include "scip/cons_linear.h"
#include "scip/cons_varbound.h"
#include "pub_solver.h"
#include "pricer_gcg.h"
#include "relax_gcg.h"
#include "pub_gcgcol.h"
#include "pub_gcgvar.h"

#include "cliquer/cliquer.h"


#define SOLVER_NAME          "cliquer"
#define SOLVER_DESC          "heuristic solver for pricing problems that solves independent set problems with cliquer"
#define SOLVER_PRIORITY      150

#define SOLVER_HEURENABLED   TRUE            /**< indicates whether the solver should be enabled */
#define SOLVER_EXACTENABLED  FALSE           /**< indicates whether the solver should be enabled */

#define DEFAULT_DENSITY      0.00

struct GCG_SolverData 
{
   SCIP_Real             density;            /**< graph density threshold above which to use solver */
}; 


/*
 * Local methods
 */

/** Returns whether the given var is linked in some way with other variables */
static
SCIP_Bool isVarLinked(
   SCIP_VAR** linkedvars,                         /**< Array of variables that are linked by eq-constraints */
   int        nlinkedvars,                        /**< Index of linkedvars array */
   SCIP_VAR*  var                                 /**< Variable whose membership in the linkedvars array is to be checked */
   )
{
   SCIP_Bool islinked;
   int       i;

   islinked = FALSE;
   for( i = 0; i < nlinkedvars; ++i )
   {
      if( linkedvars[i] == var )
      {
         islinked = TRUE;
      }
   }
   return islinked;
}

/** Returns whether 2 variables are linked, either simply or in a transitive way in respect to a given linkmatrix matrix.
  * Use of the wrapper function areVarsLinked(..) is recommended
  */
static
SCIP_Bool areVarsLinkedRec(
   int**          linkmatrix,                     /**< Matrix indicating which variables are linked by a node */
   int            vindex1,                        /**< Problem index of the first variable in the pair that is to be checked */
   int            vindex2,                        /**< Problem index of the second variable in the pair that is to be checked */
   int*           vartrace,                       /**< Array to keep track of which nodes have already been visited during recursion */
   int            traceindex,                     /**< Index to keep track of the number of visited nodes during recursion */
   SCIP_VAR**     linkedvars,                     /**< Array of variables that are linked by eq-constraints */
   int            nlinkedvars                     /**< Index of linkedvars array */
   )
{
   SCIP_Bool varintrace;
   int       i,j;

   varintrace = FALSE;
   /* Simple, direct link? (Matrix is symmetric) */
   if( linkmatrix[vindex1][vindex2] )
   {
      return TRUE;
   }
   /* More complex link by transitivity? */
   else
   {
      for( i = 0; i < nlinkedvars; ++i )
      {
         if( linkmatrix[vindex1][SCIPvarGetProbindex(linkedvars[i])] )
         {
            /* To ensure termination, we have to keep track of the visited vars */
            for( j = 0; j < traceindex; ++j )
            {
               if( vartrace[j] == SCIPvarGetProbindex(linkedvars[i]) )
               {
                  varintrace = TRUE;
               }
            }
            if( !varintrace )
            {
               vartrace[traceindex] = vindex1;
               ++traceindex;
               return areVarsLinkedRec(linkmatrix,SCIPvarGetProbindex(linkedvars[i]),vindex2,vartrace,traceindex,linkedvars,nlinkedvars);
            }
         }
      }
   }
   return FALSE;
}

/** Wrapper function for areVarsLinkedRec, mallocs and cleans up the necessary memory and passes through the result */
static
SCIP_Bool areVarsLinked(
   SCIP*          scip,                           /**< The problem instance */
   int**          linkmatrix,                     /**< Matrix indicating which variables are linked by a node */
   SCIP_VAR*      var1,                           /**< The first variable in the pair that is to be checked */
   SCIP_VAR*      var2,                           /**< The second variable in the pair that is to be checked */
   SCIP_VAR**     linkedvars,                     /**< Array of variables that are linked by eq-constraints */
   int            nlinkedvars                     /**< Index of linkedvars array */
   )
{
   int*      vartrace;
   int       traceindex;
   int       i;
   int       vindex1;
   int       vindex2;
   SCIP_Bool varslinked;

   vindex1 = SCIPvarGetProbindex(var1);
   vindex2 = SCIPvarGetProbindex(var2);

   /* We can save effort if a direct link is present */
   if( linkmatrix[vindex1][vindex2] )
   {
      return TRUE;
   }

   SCIP_CALL( SCIPallocBufferArray(scip,&vartrace,nlinkedvars) );
   traceindex = 0;
   for( i = 0; i < nlinkedvars; ++i )
   {
      vartrace[i] = -1;
   }

   varslinked = areVarsLinkedRec(linkmatrix,vindex1,vindex2,vartrace,traceindex,linkedvars,nlinkedvars);

   SCIPfreeBufferArray(scip,&vartrace);

   return varslinked;
}

/** Update transitivity in the linkmatrix matrix between 2 variables that are to be linked and all linked variables */
static
void updateVarLinks(
   SCIP*      scip,                                /**< The Problem instance */
   int**      linkmatrix,                          /**< Matrix indicating which variables are linked by a node */
   SCIP_VAR*  var1,                                /**< The first variable in the pair that is to be checked */
   SCIP_VAR*  var2,                                /**< The second variable in the pair that is to be checked */
   SCIP_VAR** linkedvars,                          /**< Array of variables that are linked by eq-constraints */
   int*       nlinkedvars                          /**< Index of linkedvars array */
   )
{
   int        varindex1,varindex2;
   int        i;
   SCIP_Bool  newvar1;
   SCIP_Bool  newvar2;

   newvar1 = TRUE;
   newvar2 = TRUE;

   /* Check if the variables are part of a link already, add them elsewise to the linkedvars array */
   for( i = 0; i < *nlinkedvars; ++i )
   {
      if( linkedvars[i] == var1 )
      {
         newvar1 = FALSE;
      }
      else if( linkedvars[i] == var2 )
      {
         newvar2 = FALSE;
      }
   }
   if( newvar1 )
   {
      linkedvars[*nlinkedvars] = var1;
      ++(*nlinkedvars);
   }
   if( newvar2 )
   {
      linkedvars[*nlinkedvars] = var2;
      ++(*nlinkedvars);
   }

   varindex1 = SCIPvarGetProbindex(var1);
   varindex2 = SCIPvarGetProbindex(var2);

   /* Variables may have not been simply linked before */
   linkmatrix[varindex1][varindex2] = 1;
   linkmatrix[varindex2][varindex1] = 1;

   for( i = 0; i < *nlinkedvars; ++i )
   {
      /* It is sufficient to check the links between var1 and all other vars, since var1 and var2 are linked */
      if( varindex1 != SCIPvarGetProbindex(linkedvars[i]) )
      {
         if( areVarsLinked(scip,linkmatrix,var1,linkedvars[i],linkedvars,*nlinkedvars) )
         {
            /* Add links to both var1 and var2 */
            linkmatrix[varindex1][SCIPvarGetProbindex(linkedvars[i])] = 1;
            linkmatrix[SCIPvarGetProbindex(linkedvars[i])][varindex1] = 1;
            linkmatrix[varindex2][SCIPvarGetProbindex(linkedvars[i])] = 1;
            linkmatrix[SCIPvarGetProbindex(linkedvars[i])][varindex2] = 1;
         }
      }
   }
}

/** Get the node index of a given variable in the bijection if mapped, else return -1 */
static 
int getNodeIndex(
   SCIP_VAR*      var,                            /**< Variable for which the node index is to be determined */
   SCIP_VAR**     indsetvars,                     /**< Array of variables that are mapped to a node of the graph */
   int            indexcount                      /**< Number of variables that are mapped in the graph */
   )
{
   for( int i = 0; i < indexcount; ++i )
   {
      if( var == indsetvars[i] )
      {
         return i;
      }
   }
   return -1;
}

/** Returns the node index of a given variable in the bijection or that of a linked variable, if any. */
static 
int getLinkedNodeIndex(
   SCIP*          scip,                           /**< The problem instance */
   SCIP_VAR*      var,                            /**< Variable for which the node index is to be determined */
   SCIP_VAR**     indsetvars,                     /**< Array of variables that are mapped to a node of the graph */
   int            indexcount,                     /**< Number of variables that are mapped in the graph */
   int**          linkmatrix,                     /**< Matrix indicating which variables are linked by a node */
   SCIP_VAR**     linkedvars,                     /**< Array of variables that are linked by eq-constraints */
   int            nlinkedvars                     /**< Index of linkedvars array */
   )
{
   int        nodeindex;
   int        i;

   nodeindex = getNodeIndex(var,indsetvars,indexcount);
   if( nodeindex == -1 && isVarLinked(linkedvars,nlinkedvars,var) )
   {
      for( i = 0; i < nlinkedvars; ++i )
      {
         if( linkedvars[i] != var )
         {
            if( areVarsLinked(scip,linkmatrix,var,linkedvars[i],linkedvars,nlinkedvars) )
            {
               nodeindex = getNodeIndex(linkedvars[i],indsetvars,indexcount);
               if( nodeindex != -1 )
               {
                  return nodeindex;
               }
            }
         }
      }
   }
   else
   {
      return nodeindex;
   }
   return -1;
}

/** Add a variable to the bijection graph g and indsetvars array. Returns the index of the corresponding node in the graph. */
static
int addVarToGraph(
   SCIP*          scip,                           /**< The problem instance */
   graph_t*       g,                              /**< Graph into which to insert the new variable as a node */
   SCIP_VAR*      consvar,                        /**< The variable that is to be assigned a node in the graph */
   int*           indexcount,                     /**< Pointer to Index of the next unassigned node in the graph */
   SCIP_Real      scalingfactor,                  /**< Factor for scaling the weight of newly mapped nodes */
   SCIP_VAR**     indsetvars,                     /**< Array that keeps track of variables that are part of the graph */
   int**          linkmatrix,                     /**< Matrix indicating which variables are linked by a node */
   SCIP_VAR**     linkedvars,                     /**< Array of variables that are linked by eq-constraints */
   int            nlinkedvars                     /**< Index of linkedvars array */
   )
{
   int nodeindex;
   if( isVarLinked(linkedvars,nlinkedvars,consvar) )
   {
      nodeindex = getLinkedNodeIndex(scip,consvar,indsetvars,*indexcount,linkmatrix,linkedvars,nlinkedvars);
   }
   else
   {
      nodeindex = getNodeIndex(consvar,indsetvars,*indexcount);
   }
   if( nodeindex == -1 )
   {
      /* Var not yet part of graph, add it with its corresponding weight */
      indsetvars[*indexcount] = consvar;
      if( !(SCIPvarGetObj(indsetvars[*indexcount]) > 0) )
      {
         g->weights[*indexcount] = 1 + abs((int) (scalingfactor * SCIPvarGetObj(indsetvars[*indexcount])));
      }
      else
      {
         g->weights[*indexcount] = 1;
      }
      nodeindex = *indexcount;
      ++(*indexcount);
   }
   return nodeindex;
}

/** Set the solvals of a variable and of all its linked variables, if any */
static
void setLinkedSolvals(
   SCIP*       scip,                              /**< The problem instance */
   SCIP_Real*  solvals,                           /**< Array holding the current solution values of all variables of the problem */
   int**       linkmatrix,                        /**< Matrix indicating which variables are linked by a node */
   SCIP_VAR**  linkedvars,                        /**< Array of variables that are linked by eq-constraints */
   int         nlinkedvars,                       /**< Index of linkedvars array */
   SCIP_VAR*   var,                               /**< Var that may be linked that itself should be set to the value val */
   SCIP_Real   val                                /**< Value to which all linked vars are supposed to be set to */
   )
{
   int i;

   solvals[SCIPvarGetProbindex(var)] = val;

   for( i = 0; i < nlinkedvars; ++i )
   {
      if( var != linkedvars[i] )
      {
         if( areVarsLinked(scip,linkmatrix,var,linkedvars[i],linkedvars,nlinkedvars) )
         {
            solvals[SCIPvarGetProbindex(linkedvars[i])] = val;
         }
      }
   }
}

/** Check if the objective coefficients of the variables are already Integral */
static 
SCIP_Bool areObjectivesIntegral(
   SCIP*       scip                               /**< The problem instance */
   )
{
   SCIP_Real  objval;
   SCIP_Real  nvars;
   SCIP_VAR** vars;
   int        i;

   nvars = SCIPgetNVars(scip);
   vars = SCIPgetVars(scip);

   for( i = 0; i < nvars; ++i )
   {
      objval = SCIPvarGetObj(vars[i]);
      if( !SCIPisZero(scip,objval-((int)objval)) )
      {
         return FALSE;
      }
   }
   return TRUE;
}

/** Scale the objective coefficients of the variables maximally s.t. they become integral and the sum of values does not exceed INT_MAX */
static 
SCIP_Real scaleRelativeToMax(
   SCIP*       scip                               /**< The problem instance */
   )
{
   SCIP_Real  scalingfactor;
   SCIP_Real  varval;
   SCIP_Real  biggestobj;
   SCIP_Real  nvars;
   SCIP_VAR** vars;
   int        i;

   nvars = SCIPgetNVars(scip);
   vars = SCIPgetVars(scip);

   scalingfactor = (INT_MAX / nvars) - nvars;

   /* Check for the biggest objective value to safely adjust the scalingfactor */
   biggestobj = 0.0;
   for( i = 0; i < nvars; ++i )
   {
      varval = SCIPvarGetObj(vars[i]);
      if( SCIPisLT(scip,varval,biggestobj) )
      {
         biggestobj = varval;
      }
   }
   if( SCIPisLT(scip,biggestobj,-1.0) )
   {
      /* Ensure that INT_MAX is never reached by the sum of all scaled weights */
      scalingfactor = fabs(scalingfactor / biggestobj);
   }
   return scalingfactor;
}

/* Basic idea of the heuristic solver: The biggest independent set in a graph corresponds to the biggest clique
 * of the complement graph, for which we use the cliquer library to find it. We therefore transform the variables 
 * into graph nodes and delete the edge between two nodes if there is an independent set constraint involving both. 
 * By doing this, they cannot both be part of the maximum clique and thus not be both part of the independent set.
 * The correspondence between variables and graph nodes is done by a bijection using the indsetvars array:
 * The variable indsetvars[i] is the i-th node of the graph, indexcount keeps track of the next unmapped graph node.
 * There is also the possibility that two variables x and y are linked with an equality constraint x-y = 0 due to
 * Ryan-Foster-Branching. In this case, all linked variables are mapped to the same node. There are functions
 * to get the corresponding node index.
 *
 * Since we want to add a column with the best reduced cost, we take the objective coefficient of variables into
 * account by giving their graph nodes corresponding weights and searching for a weight-maximal clique.
 *
 * If you would like to add the handling of more types of constraints, please note that the current code 
 * assumes that at no point edges are added to the graph, except during initialisation.
 *
 * This solver is currently able to handle the following type of constraints:
 * IS-Constraints, i.e. c*x + d*y <= 1*e
 * Coupling-Constraints, i.e. v + w + x -c*y <= 0
 * Clique-Constraints, i.e. v + w + x + y <= 1
 * Same-Constraints, i.e. x - y = 0 for varbound-constraints.
 * Vbd-constraints of type x - c*y <= 0 for c <= -1
 */

/** Solve the pricing problem as an independent set problem, in an approximate way. */
static
SCIP_RETCODE solveCliquer(
   SCIP_Bool             exactly,            /**< should the pricing problem be solved to optimality or heuristically? */
   SCIP*                 scip,               /**< master problem SCIP data structure */
   SCIP*                 pricingprob,        /**< pricing problem SCIP data structure */
   GCG_SOLVERDATA*       solver,             /**< solver data structure */
   int                   probnr,             /**< problem number */
   SCIP_Real*            lowerbound,         /**< pointer to store lower bound */
   GCG_PRICINGSTATUS*    status              /**< pointer to store pricing problem status */
   )
{ /*lint -e715 */
   SCIP_CONS**    constraints;
   SCIP_CONS**    markedconstraints;
   SCIP_CONSHDLR* conshdlr;
   SCIP_VAR**     lconsvars;
   SCIP_VAR**     vconsvars;
   SCIP_VAR**     indsetvars;
   SCIP_VAR**     pricingprobvars;
   SCIP_VAR**     linkedvars;
   SCIP_Real*     solvals;
   SCIP_Real*     consvals;
   SCIP_Real      scalingfactor;
   SCIP_Bool      retcode;
   set_t          clique;
   graph_t*       g;
   clique_options cl_opts;
   int**          linkmatrix;
   int            nlinkedvars;
   int            npricingprobvars;
   int            nvars;
   int            nconss;
   int            nedges;
   int            markedcount;
   int            indexcount;
   int            nodeindex0;
   int            nodeindex1;
   int            coefindex;
   GCG_COL*       col;

   int            i;
   int            j;
   int            k;

   assert(scip != NULL);
   assert(pricingprob != NULL);
   assert(solver != NULL);
   assert(lowerbound != NULL);
   assert(status != NULL);

   pricingprobvars = SCIPgetVars(pricingprob);
   npricingprobvars = SCIPgetNVars(pricingprob);

   constraints = SCIPgetConss(pricingprob);
   nconss = SCIPgetNConss(pricingprob);

   /* All variables of the problem are expected to be binary */
   if( SCIPgetNBinVars(pricingprob) < npricingprobvars )
   {
      SCIPdebugMessage("Exit: Nonbinary variables.\n");
      *status = GCG_PRICINGSTATUS_NOTAPPLICABLE;
      return SCIP_OKAY;
   }

   /* Cliquer library explicitly demands the node weights to be positive integers.
    * Additionally, the sum of node weights needs to be smaller than INT_MAX.
    * We restrict our scaling factor to always honor this constraint.
    */
   if( !areObjectivesIntegral(pricingprob) )
   {
      scalingfactor = scaleRelativeToMax(pricingprob);
   }
   else
   {
      scalingfactor = 1.0;
   }

   SCIP_CALL( SCIPallocBufferArray(pricingprob,&markedconstraints,nconss) );
   SCIP_CALL( SCIPallocBufferArray(pricingprob,&indsetvars,npricingprobvars) );
   SCIP_CALL( SCIPallocBufferArray(pricingprob,&solvals,npricingprobvars) );
   SCIP_CALL( SCIPallocBufferArray(pricingprob,&vconsvars,2) );
   SCIP_CALL( SCIPallocBufferArray(pricingprob,&linkedvars,npricingprobvars) );
   SCIP_CALL( SCIPallocBufferArray(pricingprob,&linkmatrix,npricingprobvars) );
   for( i = 0; i < npricingprobvars; ++i )
   {
      SCIP_CALL( SCIPallocBufferArray(pricingprob,&linkmatrix[i],npricingprobvars) );
   }

   /* Used to keep track of node indizes for bijection while building the graph */
   indexcount = 0;

   /* Use to handle a rare combination of IS and varbound constraints */
   markedcount = 0;

   /* Used to keep track of the index of the linkedvars array */
   nlinkedvars = 0;

   /* Build complementary graph by first creating a complete graph and then deleting edges of IS constraints. */
   /* Size is first chosen to be maximal and then later cropped down to the actual number of nodes. */
   /* Initialize the linkmatrix array. */
   /* Initialize the solvals array. */
   g = graph_new(npricingprobvars);
   for( i = 0; i < npricingprobvars; ++i )
   {
      for( j = 0; j < npricingprobvars; ++j )
      {
         if( i != j )
         {
            GRAPH_ADD_EDGE(g,i,j);
         }
         linkmatrix[i][j] = 0;
      }
      solvals[i] = -1.0; /* To later determine whether a variable was constrained */
   }

   /* Check for "same"-constraints present in Ryan-Foster-Branching and save the links between the variables. */
   for( i = 0; i < nconss; ++i )
   {
      assert(constraints[i] != NULL);
      conshdlr = SCIPconsGetHdlr(constraints[i]);
      assert(conshdlr != NULL);
      if( strcmp(SCIPconshdlrGetName(conshdlr), "varbound") == 0 )
      {
         /* Varbound constraint of type x + cy == rhs */
         if( SCIPisEQ(pricingprob, SCIPgetLhsVarbound(pricingprob,constraints[i]), SCIPgetRhsVarbound(pricingprob,constraints[i])) )
         {
            /* c == -1, thus variables have to become both 0 or both 1 */ 
            if( (SCIPgetRhsVarbound(pricingprob,constraints[i]) == 0) && (SCIPgetVbdcoefVarbound(pricingprob,constraints[i]) == -1) )
            {
               vconsvars[0] = SCIPgetVarVarbound(pricingprob,constraints[i]);
               vconsvars[1] = SCIPgetVbdvarVarbound(pricingprob,constraints[i]);
               updateVarLinks(pricingprob,linkmatrix,vconsvars[0],vconsvars[1],linkedvars,&nlinkedvars);
               /* Since the vars may not be part of the graph, we have to be able to set their solval later, thus we save the constraint */
               markedconstraints[markedcount] = constraints[i];
               ++markedcount;
            }
         }
      }
   }

   /* All links have to be established first before we can add nodes to the graph, else pairs (a,b) and (c,d) would be mapped to different nodes */
   /* if link (b,c) is present but later in the list. We have to run through the constraints again as the linked variables need to be assigned to nodes */
   /* in order for the rest of the logic to work out (node indizes are fetched during runtime) */
   for( i = 0; i < markedcount; ++i )
   {
      /* Varbound constraint of type x + cy == rhs */
      if( SCIPisEQ(pricingprob, SCIPgetLhsVarbound(pricingprob,markedconstraints[i]), SCIPgetRhsVarbound(pricingprob,markedconstraints[i])) )
      {
         /* c == -1, thus variables have to become both 0 or both 1 */ 
         if( (SCIPgetRhsVarbound(pricingprob,markedconstraints[i]) == 0) && (SCIPgetVbdcoefVarbound(pricingprob,markedconstraints[i]) == -1) )
         {
            vconsvars[0] = SCIPgetVarVarbound(pricingprob,markedconstraints[i]);
            nodeindex0 = addVarToGraph(pricingprob,g,vconsvars[0],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);
         }
      }
   }

   /* Main loop to check the nature of each constraint */
   for( i = 0; i < nconss; ++i )
   {
      assert(constraints[i] != NULL);
      conshdlr = SCIPconsGetHdlr(constraints[i]);
      assert(conshdlr != NULL);

      /* The constraint may not be of type 'linear' */
      if( strcmp(SCIPconshdlrGetName(conshdlr),"linear") == 0 )
      {
         lconsvars = SCIPgetVarsLinear(pricingprob,constraints[i]);
         consvals = SCIPgetValsLinear(pricingprob,constraints[i]);
         if( !SCIPisEQ(pricingprob,SCIPgetLhsLinear(pricingprob,constraints[i]),SCIPgetRhsLinear(pricingprob,constraints[i])) )
         {
            /* Check if we have an IS constraint */
            if( SCIPgetNVarsLinear(pricingprob,constraints[i]) == 2 &&
                SCIPisEQ(pricingprob,SCIPgetRhsLinear(pricingprob,constraints[i]),1) ) 
            {
               /* Preprocessing: Constraint is only relevant for pricing if one of the variables has an objective value < 0 */
               if( SCIPisLT(pricingprob,SCIPvarGetObj(lconsvars[0]),0) || SCIPisLT(pricingprob,SCIPvarGetObj(lconsvars[1]),0) 
                  || (getLinkedNodeIndex(pricingprob,lconsvars[0],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars) != -1 && 
                     getLinkedNodeIndex(pricingprob,lconsvars[1],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars) != -1) )
               {
                  if( SCIPisLT(pricingprob,SCIPvarGetObj(lconsvars[0]),0) )
                  {
                     nodeindex0 = addVarToGraph(pricingprob,g,lconsvars[0],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);
                  }
                  else
                  {
                     nodeindex0 = getLinkedNodeIndex(pricingprob,lconsvars[0],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars);
                  }

                  if( SCIPisLT(pricingprob,SCIPvarGetObj(lconsvars[1]),0) )
                  {
                     nodeindex1 = addVarToGraph(pricingprob,g,lconsvars[1],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);
                  }
                  else
                  {
                     nodeindex1 = getLinkedNodeIndex(pricingprob,lconsvars[1],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars);
                  }

                  if( nodeindex0 >= 0 && nodeindex1 >= 0 )
                  {
                     if( GRAPH_IS_EDGE(g,nodeindex0,nodeindex1) )
                     {
                        GRAPH_DEL_EDGE(g,nodeindex0,nodeindex1);
                     }
                     else if( nodeindex0 == nodeindex1 )
                     {
                        /* nodeindex0 and nodeindex1 are linked, thus calling the setter for one is sufficient */
                        setLinkedSolvals(pricingprob,solvals,linkmatrix,linkedvars,nlinkedvars,lconsvars[0],0.0);
                     }
                  }
               }
            }
            /* Handle other constraints that behave like IS constraints, i.e. cx+dy<=rhs with c+d>rhs, c>0, d>0 */
            else if( SCIPgetNVarsLinear(pricingprob,constraints[i]) == 2 && consvals[0] > 0 && consvals[1] > 0 
                  && SCIPisLT(pricingprob, SCIPgetRhsLinear(pricingprob,constraints[i]),consvals[0] + consvals[1]) 
                  && !SCIPisLT(pricingprob, SCIPgetRhsLinear(pricingprob,constraints[i]),consvals[0])
                  && !SCIPisLT(pricingprob, SCIPgetRhsLinear(pricingprob,constraints[i]),consvals[1]))
            { 
               /* As before, the constraint is only regarded if it is relevant for pricing */
               if( SCIPisLT(pricingprob,SCIPvarGetObj(lconsvars[0]),0) )
               {
                  nodeindex0 = addVarToGraph(pricingprob,g,lconsvars[0],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);
               }
               else
               {
                  nodeindex0 = getLinkedNodeIndex(pricingprob,lconsvars[0],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars);
               }

               if( SCIPisLT(pricingprob,SCIPvarGetObj(lconsvars[1]),0) )
               {
                  nodeindex1 = addVarToGraph(pricingprob,g,lconsvars[1],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);
               }
               else
               {
                  nodeindex1 = getLinkedNodeIndex(pricingprob,lconsvars[1],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars);
               }
               if( nodeindex0 >= 0 && nodeindex1 >= 0 )
               {
                  if( GRAPH_IS_EDGE(g,nodeindex0,nodeindex1) )
                  {
                     GRAPH_DEL_EDGE(g,nodeindex0,nodeindex1);
                  }
                  else if( nodeindex0 == nodeindex1 )
                  {
                     /* nodeindex0 and nodeindex1 are linked, thus calling the setter for one is sufficient */
                     setLinkedSolvals(pricingprob,solvals,linkmatrix,linkedvars,nlinkedvars,lconsvars[0],0.0);
                  }
               }
            }
            else
            {
               /* The current constraint is no linear IS constraint */
               SCIPgetConsNVars(pricingprob,constraints[i],&nvars,&retcode);
               coefindex = -1;

               /* Check the coefficients of the variables in the constraint */
               for( j = 0; j < nvars; ++j )
               {
                  if( consvals[j] != 1 && (coefindex == -1) )
                  {
                     coefindex = j;
                  }
                  else if( consvals[j] != 1 && coefindex != -1 )
                  {
                     /* More than one variable has a coefficient unequal to 1 */
                     SCIPdebugMessage("Exit: More than one coefficient unequal 1.\n");
                     *status = GCG_PRICINGSTATUS_NOTAPPLICABLE;
                     goto TERMINATE;
                  }
               }
               /* Check if we have a clique constraint (rhs 1 and coefficients 1) */
               if( !(coefindex == -1) && SCIPisEQ(pricingprob,SCIPgetRhsLinear(pricingprob,constraints[i]),1) )
               {
                  /* Delete the edges between all the variables of the constraint. 
                     This way, at most one can be part of the maximum clique */
                  for( j = 0; j < nvars; ++j )
                  {
                     /* We are only interested in vars potentially relevant for pricing (obj < 0) */
                     if( SCIPisLT(pricingprob,SCIPvarGetObj(lconsvars[j]),0) || getLinkedNodeIndex(pricingprob,lconsvars[j],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars) != -1 )
                     {
                        nodeindex0 = addVarToGraph(pricingprob,g,lconsvars[j],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);

                        for( k = j + 1; k < nvars; ++k )
                        {
                           if( (SCIPisLT(pricingprob,SCIPvarGetObj(lconsvars[k]),0) 
                              || getLinkedNodeIndex(pricingprob,lconsvars[k],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars) != -1))
                           {
                              nodeindex1 = addVarToGraph(pricingprob,g,lconsvars[k],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);

                              if( nodeindex0 != nodeindex1 )
                              {
                                 if( GRAPH_IS_EDGE(g,nodeindex0,nodeindex1) )
                                 {
                                    GRAPH_DEL_EDGE(g,nodeindex0,nodeindex1);
                                 }
                              }
                           }
                        }
                     }
                  }
               }
               /* Check if we have a coupling constraint (rhs 0) */
               else if( !(coefindex == -1) && SCIPisEQ(pricingprob,SCIPgetRhsLinear(pricingprob,constraints[i]), 0.0) )
               {
                  /* Special case: The coupling constraint is purely decorative (coefficient + 1 of coupling var >= #vars)*/
                  if( abs(consvals[coefindex]) + 1 >= nvars )
                  {
                     /* We cannot guarantee that there is no constraint of the form x+CouplingVar <= 1 */
                     /* If the node is part of the maximum clique, it is safe to set it to one, so we simply add it to the graph */
                     nodeindex0 = addVarToGraph(pricingprob,g,lconsvars[coefindex],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);

                     /* We additionally have to mark the variable to later set it to one */ 
                     solvals[SCIPvarGetProbindex(lconsvars[coefindex])] = -2.0;
                  }
                  /* Special case: The coefficient is -1, we treat the case like a clique constraint. */
                  else if( abs(consvals[coefindex]) == 1 )
                  {
                     /* We cannot guarantee that there is no constraint of the form x+CouplingVar <= 1 */
                     /* If the node is part of the maximum clique, it is safe to set it to one, so we simply add it to the graph */
                     /* We additionally have to mark the variable to later set it to one */ 
                     nodeindex0 = addVarToGraph(pricingprob,g,lconsvars[coefindex],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);

                     /* We additionally have to mark the variable to later set it to one */ 
                     solvals[SCIPvarGetProbindex(lconsvars[coefindex])] = -2.0;

                     /* Delete the edges between all the variables of the constraint that are not the coupling variable.
                        This way, at most one can be part of the maximum clique */
                     for( j = 0; j < nvars; ++j )
                     {
                        /* We are only interested in vars potentially relevant for pricing (obj < 0) */
                        if( j != coefindex && (SCIPisLT(pricingprob,SCIPvarGetObj(lconsvars[j]),0) 
                           || getLinkedNodeIndex(pricingprob,lconsvars[j],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars) != -1) )
                        {
                           /* Determine nodeindex0 */
                           nodeindex0 = addVarToGraph(pricingprob,g,lconsvars[j],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);
                           /* Determine nodeindex1 */
                           for( k = j + 1; k < nvars; ++k )
                           {
                              if( k != coefindex && (SCIPisLT(pricingprob,SCIPvarGetObj(lconsvars[k]),0) 
                                 || getLinkedNodeIndex(pricingprob,lconsvars[k],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars) != -1))
                              {
                                 nodeindex1 = addVarToGraph(pricingprob,g,lconsvars[k],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);
                                 if( nodeindex0 != nodeindex1 )
                                 {
                                    if( GRAPH_IS_EDGE(g,nodeindex0,nodeindex1) )
                                    {
                                       GRAPH_DEL_EDGE(g,nodeindex0,nodeindex1);
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
                  else
                  {
                     /* Coupling coefficient is between 1 and npricingprobvars. */
                     SCIPdebugMessage("Exit: Coupling coefficient unhandled, coef: %g.\n",consvals[coefindex]);
                     *status = GCG_PRICINGSTATUS_NOTAPPLICABLE;
                     goto TERMINATE;
                  }
               }
               else
               {
                  /* Constraint is neither a coupling nor a clique constraint */ 
                  SCIPdebugMessage("Exit: Unhandled linear constraint.\n");
                  *status = GCG_PRICINGSTATUS_NOTAPPLICABLE;
                  goto TERMINATE;
               }
            }
         }
         else
         {
            /* Constraint is a linear equality constraint */ 
            SCIPdebugMessage("Exit: Unhandled linear constraint: Equality constraint.\n");
            *status = GCG_PRICINGSTATUS_NOTAPPLICABLE;
            goto TERMINATE;
         }
      }
      /* Constraint may be of type varbound: lhs <= x + c*y <= rhs */
      else if( strcmp(SCIPconshdlrGetName(conshdlr), "varbound") == 0 )
      {
         vconsvars[0] = SCIPgetVarVarbound(pricingprob,constraints[i]);
         vconsvars[1] = SCIPgetVbdvarVarbound(pricingprob,constraints[i]);
         /* Check value of rhs to be 0 and of c to be <= -1 */ 
         if ( SCIPisInfinity(pricingprob, -SCIPgetLhsVarbound(pricingprob,constraints[i])) )
         {
            if( SCIPisEQ(pricingprob,SCIPgetRhsVarbound(pricingprob,constraints[i]),0) )
            {
               if( SCIPisLT(pricingprob,SCIPgetVbdcoefVarbound(pricingprob,constraints[i]),-1) || SCIPisEQ(pricingprob,SCIPgetVbdcoefVarbound(pricingprob,constraints[i]),-1) ) 
               {
                  /* if x may be relevant, add both x and y to graph */
                  if( SCIPisLT(pricingprob,SCIPvarGetObj(vconsvars[0]),0) || getLinkedNodeIndex(pricingprob,vconsvars[0],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars) != -1 )
                  {
                     nodeindex0 = addVarToGraph(pricingprob,g,vconsvars[0],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);
                     nodeindex1 = addVarToGraph(pricingprob,g,vconsvars[1],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);
                     /* It may be the case, that both the constraints x - y <= 0 and x + y <= 1 are part of the problem */
                     /* Although rare, we later ensure that we do not set x to 1 while y is set to 0 */
                     markedconstraints[markedcount] = constraints[i];
                     ++markedcount;
                  }
                  /* If only y may be relevant, add only y to the graph */
                  else if( SCIPisLT(pricingprob,SCIPvarGetObj(vconsvars[1]),0) )
                  {
                     if( SCIPisLT(pricingprob,SCIPvarGetObj(vconsvars[1]),0) )
                     {
                        nodeindex1 = addVarToGraph(pricingprob,g,vconsvars[1],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);
                     }
                  }
                  /* If none of the nodes are relevant, force x to be zero, since the constraint would be violated if x = 1 and y = 0 */
                  setLinkedSolvals(pricingprob,solvals,linkmatrix,linkedvars,nlinkedvars,vconsvars[0],0.0);
               }
               else
               {
                  /* Coefficient c of varbound is > -1 and we do not have an IS constraint*/ 
                  SCIPdebugMessage("Exit: Coefficient of Varbound unhandled Rhs: %g, Coeff: %g.\n",SCIPgetRhsVarbound(pricingprob,constraints[i]),SCIPgetVbdcoefVarbound(pricingprob,constraints[i]));
                  *status = GCG_PRICINGSTATUS_NOTAPPLICABLE;
                  goto TERMINATE;
               }
            }
            /* Rhs of varbound unequal to 0.
             * It may still be the case that we have an IS constraint with a non-linear handler.
             * The constraint may also be of the form c + 1 > rhs and c < rhs, i.e. a non-standard IS-constraint.
             * We treat these cases like a regular IS constraint.
             */
            else if( (SCIPisEQ(pricingprob,SCIPgetRhsVarbound(pricingprob,constraints[i]),1) && SCIPisEQ(pricingprob,SCIPgetVbdcoefVarbound(pricingprob,constraints[i]),1))
                  || (SCIPisLT(pricingprob,SCIPgetRhsVarbound(pricingprob,constraints[i]),SCIPgetVbdcoefVarbound(pricingprob,constraints[i]) + 1) 
                  && SCIPisLT(pricingprob,SCIPgetVbdcoefVarbound(pricingprob,constraints[i]), SCIPgetRhsVarbound(pricingprob,constraints[i]))) )
            {
               /* Preprocessing: Constraint is only relevant for pricing if one of the variables has an objective value < 0 */
               if( SCIPisLT(pricingprob,SCIPvarGetObj(vconsvars[0]),0) || SCIPisLT(pricingprob,SCIPvarGetObj(vconsvars[1]),0)
                  || (getLinkedNodeIndex(pricingprob,vconsvars[0],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars) != -1 && 
                      getLinkedNodeIndex(pricingprob,vconsvars[1],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars) != -1) )
               {
                  if( SCIPisLT(pricingprob,SCIPvarGetObj(vconsvars[0]),0) )
                  {
                     nodeindex0 = addVarToGraph(pricingprob,g,vconsvars[0],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);
                  }
                  else
                  {
                     nodeindex0 = getLinkedNodeIndex(pricingprob,vconsvars[0],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars);
                  }

                  if( SCIPisLT(pricingprob,SCIPvarGetObj(vconsvars[1]),0) )
                  {
                     nodeindex1 = addVarToGraph(pricingprob,g,vconsvars[1],&indexcount,scalingfactor,indsetvars,linkmatrix,linkedvars,nlinkedvars);
                  }
                  else
                  {
                     nodeindex1 = getLinkedNodeIndex(pricingprob,vconsvars[1],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars);
                  }

                  if( nodeindex0 >= 0 && nodeindex1 >= 0 )
                  {
                     if( GRAPH_IS_EDGE(g,nodeindex0,nodeindex1) )
                     {
                        GRAPH_DEL_EDGE(g,nodeindex0,nodeindex1);
                     }
                     else if( nodeindex0 == nodeindex1 )
                     {
                        /* nodeindex0 and nodeindex1 are linked, thus calling the setter for one is sufficient */
                        setLinkedSolvals(pricingprob,solvals,linkmatrix,linkedvars,nlinkedvars,vconsvars[0],0.0);
                     }
                  }
               }
            }
            else
            {
               /* Rhs of varbound unequal to 0 and no IS constraint*/ 
               SCIPdebugMessage("Exit: Rhs of Varbound unhandled, Rhs: %g, Coeff:%g.\n",SCIPgetRhsVarbound(pricingprob,constraints[i]),SCIPgetVbdcoefVarbound(pricingprob,constraints[i]));
               *status = GCG_PRICINGSTATUS_NOTAPPLICABLE;
               goto TERMINATE;
            }
         }
         /* We may have a varbound constraint of type x + cy == rhs */
         else if( SCIPisEQ(pricingprob, SCIPgetLhsVarbound(pricingprob,constraints[i]), SCIPgetRhsVarbound(pricingprob,constraints[i])) )
         {
            /* If the rhs is 0 and c == -1, both variables have to be set to 0 or to 1 */ 
            if( !((SCIPgetRhsVarbound(pricingprob,constraints[i]) == 0) && (SCIPgetVbdcoefVarbound(pricingprob,constraints[i]) == -1)) )
            {
               /* RHS is unequal 0 and unequal 1 */
               SCIPdebugMessage("Exit: Unhandled equality constraint, c: %g, rhs: %g.\n", SCIPgetVbdcoefVarbound(pricingprob,constraints[i]), SCIPgetRhsVarbound(pricingprob,constraints[i]));
               *status = GCG_PRICINGSTATUS_NOTAPPLICABLE;
               goto TERMINATE;
            }
         }
         else
         {
            /* We have a varbound of type lhs <= x + c*y */
            SCIPdebugMessage("Exit: Varbound of type lhs <= x+c*y, c: %g, rhs: %g.\n", SCIPgetVbdcoefVarbound(pricingprob,constraints[i]), SCIPgetRhsVarbound(pricingprob,constraints[i]));
            SCIPdebugMessage("Constraint handler: %s\n", SCIPconshdlrGetName(conshdlr));
            *status = GCG_PRICINGSTATUS_NOTAPPLICABLE;
            goto TERMINATE; 
         }
      }
      else
      {
         /* Constraint handler neither linear nor varbound */
         SCIPdebugMessage("Exit: Unhandled constraint handler: %s \n", SCIPconshdlrGetName(conshdlr));
         *status = GCG_PRICINGSTATUS_NOTAPPLICABLE;
         goto TERMINATE;        
      }
   }


   /* Assert that the graph was built in a proper way */ 
   ASSERT(graph_test(g,NULL));

   /* Determine number of edges for graph density calculation */
   nedges = 0;
   for ( i = 0; i < g->n; i++ )
   {
      for ( j = 0; j < g->n; j++ )
      {
         if( SET_CONTAINS_FAST(g->edges[i],j) )
         {
            nedges++;
         }
      }
   }
   nedges /= 2;

   /* Test if the density criteria is met */
   if( SCIPisLT(pricingprob, (float)nedges/((float)(g->n - 1) * (g->n) / 2), solver->density) )
   {
      SCIPdebugMessage("Exit: Density criteria not met,density: %g.\n",(float)nedges / ((float)(g->n - 1) * (g->n) / 2));
      *status = GCG_PRICINGSTATUS_NOTAPPLICABLE;
      goto TERMINATE;
   }

   SCIPdebugMessage("Graph size: %d.\n", indexcount);
   ASSERT( indexcount <= npricingprobvars );

   /* indexcount now holds the actual number of unique IS variables, thus we truncate the graph */
   if( indexcount > 0 )
   {
      graph_resize(g,indexcount);
   }

   /* Clean up the graph. If a variable's solval has been set to 0, it should not be part of the max clique */
   /* We enforce this by isolating the node and setting its weight to 1 as nodes cannot be deleted */
   for( i = 0; i < npricingprobvars; ++i )
   {
      if( solvals[SCIPvarGetProbindex(pricingprobvars[i])] == 0 )
      {
         nodeindex0 = getLinkedNodeIndex(pricingprob,pricingprobvars[i],indsetvars,indexcount,linkmatrix,linkedvars,nlinkedvars);
         /* The var is part of the graph if its index is unequal to -1 */
         if( nodeindex0 != -1 )
         {
            for( j = 0; j < indexcount; ++j )
            {
               if( GRAPH_IS_EDGE(g,nodeindex0,j) )
               {
                  GRAPH_DEL_EDGE(g,nodeindex0,j);
               }
            }
            g->weights[nodeindex0] = 1;
         }
      }
   }

   /* Set cliquer options */
   cl_opts.reorder_function = reorder_by_default; /* default: reorder_by_default */
   cl_opts.reorder_map = NULL;
   cl_opts.time_function = NULL; /* default: clique_print_time */
   cl_opts.output = NULL;
   cl_opts.user_function = NULL;
   cl_opts.user_data = NULL;
   cl_opts.clique_list = NULL;
   cl_opts.clique_list_length = 0;

   /* Find maximum weight clique using the cliquer library */
   clique = clique_find_single(g,0,0,FALSE,&cl_opts);

   /* Set all members of the maximum clique with objective coefficient < 0 to 1 */
   for( i = 0; i < indexcount; ++i )
   {
      /* Coupling variables were pre-set to -2.0, if they are part of the maximum clique, we enable them. 
       * If we have already set a variable to 0, this was intended and should not be reverted.
       */
      if( SET_CONTAINS(clique,i) && (SCIPisLT(pricingprob,SCIPvarGetObj(indsetvars[i]),0) || solvals[SCIPvarGetProbindex(indsetvars[i])] == -2.0) 
         && solvals[SCIPvarGetProbindex(indsetvars[i])] != 0.0 )
      {
         /* Set all linked variables, if any */
         setLinkedSolvals(pricingprob,solvals,linkmatrix,linkedvars,nlinkedvars,indsetvars[i],1.0);
      }
      else
      {
         /* We may have set some variables manually already, e.g. coupling variables */
         if( solvals[SCIPvarGetProbindex(indsetvars[i])] != 1.0)
         {
            setLinkedSolvals(pricingprob,solvals,linkmatrix,linkedvars,nlinkedvars,indsetvars[i],0.0);
         }
      }
   }

   for( i = 0; i < markedcount; ++i )
   {
      vconsvars[0] = SCIPgetVarVarbound(pricingprob,markedconstraints[i]);
      vconsvars[1] = SCIPgetVbdvarVarbound(pricingprob,markedconstraints[i]);

      /* Handle the case of marked inequality constraints of type x - y <= 0 in combination with x + y <= 1 -Constraints */
      if( SCIPisEQ(pricingprob,SCIPgetRhsVarbound(pricingprob,markedconstraints[i]),0) 
         && (SCIPisLT(pricingprob,SCIPgetVbdcoefVarbound(pricingprob,markedconstraints[i]),-1) 
            || SCIPisEQ(pricingprob,SCIPgetVbdcoefVarbound(pricingprob,markedconstraints[i]),-1)) )
      {
         /* Check if a violating assignment was made and correct it */
         if( (solvals[SCIPvarGetProbindex(vconsvars[0])] == 1) && (solvals[SCIPvarGetProbindex(vconsvars[1])] == 0) )
         {
            setLinkedSolvals(pricingprob,solvals,linkmatrix,linkedvars,nlinkedvars,vconsvars[0],0.0);
         }
      }

      /* Handle the case that there are still solvals of equality constraints that do not agree.
       * This may occur if one is unset (solval:-1) and the other one is already set (solval 0 or 1)
       */
      if( solvals[SCIPvarGetProbindex(vconsvars[0])] != solvals[SCIPvarGetProbindex(vconsvars[1])] 
         && SCIPisEQ(pricingprob, SCIPgetLhsVarbound(pricingprob,markedconstraints[i]), SCIPgetRhsVarbound(pricingprob,markedconstraints[i])) )
      {
         if( solvals[SCIPvarGetProbindex(vconsvars[0])] == 0 || solvals[SCIPvarGetProbindex(vconsvars[1])] == 0 )
         {
            setLinkedSolvals(pricingprob,solvals,linkmatrix,linkedvars,nlinkedvars,vconsvars[0],0.0);
         }
         else
         {
            /* One or both of the vars are unset and the other one, if not -1, is forced to be 1, thus we can set both to 1 */
            setLinkedSolvals(pricingprob,solvals,linkmatrix,linkedvars,nlinkedvars,vconsvars[0],1.0);
         }
      }
   }

   /* There may be variables left which are unconstrained. We set these to 1 manually if they have an objective value < 0*/
   for( i = 0; i < npricingprobvars; ++i )
   {
      if( solvals[i] == -1.0 )
      {
         if( SCIPisLT(pricingprob,SCIPvarGetObj(pricingprobvars[i]),0) )
         {
            solvals[i] = 1.0;
         }
         else
         {
            solvals[i] = 0.0;
         }
      }
   }

   /* Create a column corresponding to our clique result */
   SCIP_CALL( GCGcreateGcgCol(pricingprob, &col, probnr, pricingprobvars, solvals, npricingprobvars, FALSE, SCIPinfinity(pricingprob)) );
   SCIP_CALL( GCGpricerAddCol(scip, col) );
   *status = GCG_PRICINGSTATUS_UNKNOWN;
   set_free(clique); /* clique can only be freed if non-empty */ 

 TERMINATE:
   for( i = 0; i < npricingprobvars; ++i )
   {
      SCIPfreeBufferArray(pricingprob,&linkmatrix[i]);
   }
   SCIPfreeBufferArray(pricingprob,&linkmatrix);
   SCIPfreeBufferArray(pricingprob,&linkedvars);
   SCIPfreeBufferArray(pricingprob,&vconsvars);
   SCIPfreeBufferArray(pricingprob,&solvals);
   SCIPfreeBufferArray(pricingprob,&indsetvars);
   SCIPfreeBufferArray(pricingprob,&markedconstraints);
   graph_free(g);

   return SCIP_OKAY;
}

/*
 * Callback methods for pricing problem solver
 */

/** destructor of pricing solver to free user data (called when SCIP is exiting) */
static
GCG_DECL_SOLVERFREE(solverFreeCliquer)
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

#define solverInitsolCliquer NULL
#define solverExitsolCliquer NULL
#define solverInitCliquer NULL
#define solverExitCliquer NULL
#define solverUpdateCliquer NULL
#define solverSolveCliquer NULL

/** heuristic solving method of independent set solver */
static
GCG_DECL_SOLVERSOLVEHEUR(solverSolveHeurCliquer)
{  /*lint --e{715}*/
   GCG_SOLVERDATA* solverdata;

   solverdata = GCGsolverGetData(solver);
   assert(solverdata != NULL);

   /* solve the independent set problem approximately */
   SCIP_CALL( solveCliquer(FALSE, scip, pricingprob, solverdata, probnr, lowerbound, status) );

   return SCIP_OKAY;
}

/** creates the cliquer solver for pricing problems and includes it in GCG */
SCIP_RETCODE GCGincludeSolverCliquer(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP* origprob;
   GCG_SOLVERDATA* solverdata;

   origprob = GCGmasterGetOrigprob(scip);
   assert(origprob != NULL);

   SCIP_CALL( SCIPallocMemory(scip, &solverdata) );

   SCIP_CALL( GCGpricerIncludeSolver(scip, SOLVER_NAME, SOLVER_DESC, SOLVER_PRIORITY,
         SOLVER_HEURENABLED, SOLVER_EXACTENABLED,
         solverUpdateCliquer, solverSolveCliquer, solverSolveHeurCliquer,
         solverFreeCliquer, solverInitCliquer, solverExitCliquer,
         solverInitsolCliquer, solverExitsolCliquer, solverdata) );

   SCIP_CALL( SCIPaddRealParam(origprob, "pricingsolver/cliquer/density",
         "graph density threshold above which to use solver",
         &solverdata->density, TRUE, DEFAULT_DENSITY, 0.0, 1.0, NULL, NULL) );

   return SCIP_OKAY;
}
