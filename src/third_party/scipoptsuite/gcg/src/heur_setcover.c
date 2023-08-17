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

/**@file   heur_setcover.c
 * @brief  set covering primal heuristic according to Caprara, Fischetti, and Toth (1999)
 * @author Tobias Oelschlaegel
 * @author Christian Puchert
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>
#include "gcg.h"
#include "pricer_gcg.h"
#include "scip/clock.h"
#include "scip/cons_linear.h"
#include "heur_setcover.h"


#define HEUR_NAME             "setcover"
#define HEUR_DESC             "set covering primal heuristic according to Caprara, Fischetti, and Toth (1999)"
#define HEUR_DISPCHAR         'S'
#define HEUR_PRIORITY         0
#define HEUR_FREQ             0
#define HEUR_FREQOFS          0
#define HEUR_MAXDEPTH         -1
#define HEUR_TIMING           SCIP_HEURTIMING_AFTERNODE
#define HEUR_USESSUBSCIP      FALSE       /**< does the heuristic use a secondary SCIP instance? */

#define DEF_CORE_TENT_SIZE      10           /**< number of columns covering each row that are added to the tentative core at the beginning           */
#define DEF_LAMBDA_ADJUSTMENTS  TRUE         /**< adjust step size during the subgradient phase                                                       */
#define DEF_LAMBDA_P            50           /**< number of iterations after which lambda is adjusted                                                 */
#define DEF_LAMBDA              0.1          /**< initial step size for the subgradient phase                                                         */
#define DEF_STOP_CRIT_ITER      300          /**< number of iterations of the subgradient phase after which the stopping criterion is tested again    */
#define DEF_STOP_CRIT_DIFF      1.0          /**< stop if absolute difference between best lower and upper bound is less than SCP_STOP_CRIT_DIFF, and */
#define DEF_STOP_CRIT_RATIO     0.99         /**< the relative ratio between best lower and upper bound is less than DEF_STOP_CRIT_RATIO          */
#define DEF_PI_MIN              0.3          /**< percentage of rows to be removed after fixing columns                                               */
#define DEF_PI_ALPHA            1.1          /**< increase of pi when no improvement was made, i.e. more columns will be fixed                        */
#define DEF_BETA                1.025        /**< allowed gap between lowerbound and upper bound during the subgradient phase                         */
#define DEF_MAX_ITER            2            /**< maximum number of iterations of three-phase                                                         */
#define DEF_MAX_ITER_NO_IMP     1            /**< stop of no improvements during the last X iterations of three-phase                                 */
#define DEF_THREEPHASE_MAX_ITER 10           /**< maximum number of iterations inside three-phase                                                     */
#define DEF_GREEDY_MAX_ITER     100          /**< number of multipliers that are used for computing greedy solutions during each iteration            */
#define DEF_MIN_PROB_SIZE       1000         /**< minimum number of variables the problem has to contain for the heuristic to start                   */
#define DEFAULT_RANDSEED        19           /**< initial random seed                                                                                 */

/*
 * Data structures
 */

/** primal heuristic data */

typedef struct
{
   int                   size;               /**< actual number of elements stored in the queue    */
   int                   reserved;           /**< number of elements for which memory is allocated */
   SCIP_Real*            keys;               /**< array of keys                                    */
   int*                  data;               /**< array of values                                  */
   int**                 positions;          /**< array of pointers to save positions of elements  */
} PQUEUE;

typedef struct
{
   SCIP_Bool*            varsfixed;          /**< boolean array that indicates which variables are fixed                      */
   int                   nvarsfixed;         /**< number of fixed variables                                                   */
   SCIP_Bool*            rowscovered;        /**< boolean array that indicates which rows are covered by the fixed variables  */
   int                   nrowscovered;       /**< number of rows that are covered by the fixed variables                      */
   SCIP_Real             costsfixed;         /**< total costs of variables that are fixed                                     */
} SCP_INSTANCE;

typedef struct
{
   SCIP_Bool*            varincore;          /**< boolean array that indicates whether a variable is in the core */
   int*                  listcorevariables;  /**< array of indices of core variables */
   int                   ncorevariables;     /**< number of core variables */
   SCIP_HASHMAP*         mapvariables;       /**< maps variables-indices to [0, nvariables) in array 'variables' */
   SCIP_VAR**            variables;          /**< all variables of the problem */
   int*                  nvarconss;          /**< array that contains for each variable the number of constraints it is part of */
   int                   nvariables;         /**< total number of variables  */
   SCIP_Bool             columnsavailable;   /**< if set then 'columns' contains the columns for all core variables */
   int**                 columns;            /**< columns of core variables, NULL if not a core variable */
   SCIP_Bool             rowsavailable;      /**< if set then 'rows' contains all rows reduced to core variables */
   int**                 rows;               /**< rows that only contain core variables */
   int*                  nrowvars;           /**< number of core variables a row contains */
   int                   nconss;             /**< total number of constraints (including inactive ones) */
   int                   nactiveconss;       /**< total number of active constraints for which the variables can be retrieved */
   int                   maxconstraintvariables; /**< greatest number of variables some constraint contains */
   SCIP_CONS**           conss;              /**< all constraints of the problem */
   int*                  constraintid;       /**< unique id for each constraint */
   SCIP_Real*            delta;              /**< delta values of variables */
   int*                  delta_pos;
   int*                  solgreedy;
   int                   nsolgreedy;
} SCP_CORE;

typedef struct
{
   SCIP_Bool*            xgreedylocal;       /**< contains variables that are part of a greedy solution, this is not necessarily a global solution */
   SCIP_Real*            u;                  /**< lagrange multipliers for the rows                                                                */
   SCIP_Real*            subgradient;
   SCIP_Real*            lagrangiancostslocal;  /**< lagrangian costs (for a certain instance) when only uncovered rows are considered                */
   SCIP_Real*            lagrangiancostsglobal; /**< lagrangian costs for the whole instance when all rows and columns are considered                 */
   SCIP_Real             ubgreedylocal;      /**< bound computed by the greedy set cover algorithm for the restricted instance                     */
   SCIP_Real             lblagrangelocal;    /**< lower bound by lagrange relaxation for the restricted instance                                   */
   SCIP_Real             lblagrangeglobal;   /**< lower bound by lagrange relaxation for the unrestricted instance                                 */
} SCP_Lagrange_Sol;

struct SCIP_HeurData
{
   int                   param_core_tent_size; /**< number of columns covering each row that are added to the tentative core at the beginning           */
   SCIP_Bool             param_lambda_adjustments; /**< adjust step size during the subgradient phase                                                       */
   int                   param_lambda_p;     /**< number of iterations after which lambda is adjusted                                                 */
   SCIP_Real             param_lambda;       /**< initial step size for the subgradient phase                                                         */
   int                   param_stop_crit_iter; /**< number of iterations of the subgradient phase after which the stopping criterion is tested again    */
   SCIP_Real             param_stop_crit_diff; /**< stop if absolute difference between best lower and upper bound is less than SCP_STOP_CRIT_DIFF, and */
   SCIP_Real             param_stop_crit_ratio; /**<     the relative gap between best lower and upper bound is less than (1 - SCP_STOP_CRIT_PER         */
   SCIP_Real             param_pi_min;       /**< percentage of rows to be removed after fixing columns                                               */
   SCIP_Real             param_pi_alpha;     /**< increase of pi when no improvement was made, i.e. more columns will be fixed                        */
   SCIP_Real             param_beta;         /**< allowed a gap between lowerbound and upper bound during the subgradient phase                       */
   int                   param_max_iter;     /**< maximum number of iterations of three-phase                                                         */
   int                   param_max_iter_no_imp; /**< stop of no improvements during the last X iterations of three-phase                                 */
   int                   param_threephase_max_iter; /**< maximum number of iterations inside three-phase                                                     */
   int                   param_greedy_max_iter; /**< number of multipliers that are used for computing greedy solutions during each iteration            */
   int                   param_min_prob_size; /**< minimum number of variables the master problem needs to contain before the heuristic starts at all  */

   SCP_CORE              core;               /**< core (subcollection of columns) of the problem covering all rows */
   SCP_INSTANCE          inst;               /**< reduced instance where some variables may be fixed and some rows be covered */
   SCP_INSTANCE          subinst;            /**< reduced instance of 'inst', used during the three-phase */

   SCP_Lagrange_Sol      multbestlbtotal;    /**< lagrange multiplier that gives the best lower bound for the complete problem */
   SCP_Lagrange_Sol      multbestlbinst;     /**< best multiplier for instance 'inst' */
   SCP_Lagrange_Sol      multbestlbsubinst;  /**< best multiplier for instance 'subinst' */

   SCIP_Real             bestub;             /**< best upper bound that could be obtained so far */
   SCIP_Bool*            bestubsol;          /**< actual solution that gives the best upper bound */
   SCIP_Real             bestubinst;         /**< best upper bound for the reduced instance 'inst' (including fixed costs) */
   SCIP_Bool*            bestubinst_sol;     /**< actual solution for the instance 'inst' (including fixed variables) */
   SCIP_Real             bestubsubinst;      /**< best upper bound for the reduced instance 'subinst' (including fixed costs) */
   SCIP_Bool*            bestubsubinstsol;   /**< actual solution for the instance 'subinst' (including fixed variables) */

   /* local data that is used by (most) local procedures */
   SCIP_VAR**            vars;               /**< used to iterate through the variables of a constraint */

   /* memory that is used locally by 'threephase' */
   SCP_Lagrange_Sol      tpmultlbsubinst;    /**< lagrange multiplier */
   SCIP_Bool             useinitialmultiplier; /**< compute an own initial lagrange multiplier instead of using the best known */

   /* memory that is used locally by 'greedysetcover' */
   PQUEUE                greedyqueue;        /**< priority queue used by the greedy algorithm */
   int*                  greedycolpos;       /**< stores position of a variable within the priority queue */
   int*                  greedycolmu;        /**< value mu for each variable */
   SCIP_Real*            greedycolgamma;     /**< value gamma for each variable */
   SCIP_Real*            greedycolscore;     /**< score of each variable */
   SCP_INSTANCE          greedyinst;         /**< instance that is used to hold covered rows */

   /* memory that is used locally by redefineCore */
   int*                  rccols;             /**< stores columns covering some row sorted by increasing costs */
   SCIP_Real*            rccoldelta;         /**< delta values of the columns stored in 'rccols' */

   /* memory that is used locally by subgradientOptimization */
   SCIP_Real*            sglastlb;           /**< stores lower bounds of the last iterations */
   SCIP_RANDNUMGEN*      randnumgen;         /**< random number generator */
};

/*
 * Local methods
 */

/** initializes a priority queue */
static
SCIP_RETCODE pqueue_init(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   PQUEUE*               queue               /**< priority queue instance                                         */
   )
{
   queue->size = 0;
   queue->reserved = 32;

   SCIP_CALL( SCIPallocBufferArray(scip, &queue->keys, queue->reserved) );
   SCIP_CALL( SCIPallocBufferArray(scip, &queue->data, queue->reserved) );
   SCIP_CALL( SCIPallocBufferArray(scip, &queue->positions, queue->reserved) );

   return SCIP_OKAY;
}

/** removes all elements from a priority queue, but does not release any memory */
static
void pqueue_clear(
   SCIP*                 scip,               /**< master SCIP data structure                                       */
   PQUEUE*               queue               /**< priority queue instance                                          */
   )
{
   queue->size = 0;
}

/** releases all memory that is used by the priority queue */
static
void pqueue_destroy(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   PQUEUE*               queue               /**< priority queue instance                                         */
   )
{
   SCIPfreeBufferArrayNull(scip, &queue->positions);
   SCIPfreeBufferArrayNull(scip, &queue->data);
   SCIPfreeBufferArrayNull(scip, &queue->keys);
}

/** inserts an element with key 'key' and value 'elem' into the queue.
 *  If 'position' is not NULL, the referenced memory location will always contain the internal position of the element
 * */
static
SCIP_RETCODE pqueue_insert(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   PQUEUE*               queue,              /**< priority queue instance                                         */
   SCIP_Real             key,                /**< key of the new item                                             */
   int                   elem,               /**< value of the new item                                           */
   int*                  position            /**< address where the items position within the queue is stored     */
   )
{
   int pos;
   int parent = 0;

   if( queue->size == queue->reserved )
   {
      queue->reserved += 128;
      SCIP_CALL( SCIPreallocBufferArray(scip, &queue->keys, queue->reserved) );
      SCIP_CALL( SCIPreallocBufferArray(scip, &queue->data, queue->reserved) );
      SCIP_CALL( SCIPreallocBufferArray(scip, &queue->positions, queue->reserved) );
   }

   pos = queue->size++;
   queue->keys[pos] = key;
   queue->data[pos] = elem;

   if( pos > 0 )
      parent = (pos - 1) / 2;

   while( (pos > 0) && (key < queue->keys[parent]) )
   {
      /* swap element with parent */
      queue->keys[pos] = queue->keys[parent];
      queue->data[pos] = queue->data[parent];
      queue->positions[pos] = queue->positions[parent];
      /*queue->data[parent] = elem;*/

      if( queue->positions[pos] != NULL )
         *queue->positions[pos] = pos;

      pos = (pos - 1) / 2;
      parent = (pos - 1) / 2;
   }

   queue->keys[pos] = key;
   queue->data[pos] = elem;
   queue->positions[pos] = position;

   if( position != NULL )
      *position = pos;

   return SCIP_OKAY;
}

/** decreases the key to 'key' of the element that is currently at position 'pos' */
static
void pqueue_decrease_key(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   PQUEUE*               queue,              /**< priority queue instance                                         */
   int                   pos,                /**< position of the item within the queue that is to be changed     */
   SCIP_Real             key                 /**< new key of the item, needs to be greater than the old key       */
   )
{
   int parent;
   int* positionptr;
   int elem;

   if( pos >= queue->size )
      return;

   parent = (pos - 1) / 2;
   positionptr = queue->positions[pos];
   elem = queue->data[pos];

   while( (pos > 0) && (key < queue->keys[parent]) )
   {
      /* swap element with parent */
      queue->keys[pos] = queue->keys[parent];
      queue->data[pos] = queue->data[parent];
      queue->positions[pos] = queue->positions[parent];
      queue->data[parent] = elem;

      if( queue->positions[pos] != NULL )
         *queue->positions[pos] = pos;

      pos = (pos - 1) / 2;
      parent = (pos - 1) / 2;
   }

   queue->keys[pos] = key;
   queue->data[pos] = elem;
   queue->positions[pos] = positionptr;

   if( positionptr != NULL )
      *positionptr = pos;
}

/** increases the key to 'key' of the element that is currently at position 'pos' */
static
void pqueue_increase_key(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   PQUEUE*               queue,              /**< priority queue instance                                         */
   int                   pos,                /**< position of the item within the queue that is to be changed     */
   SCIP_Real             key                 /**< new key of the item, needs to be smaller than the old one       */
   )
{
   int elem;
   int* positionptr;
   int left;
   int right;

   left = 2 * pos + 1;
   right = 2 * pos + 2;

   if( pos >= queue->size )
      return;

   elem = queue->data[pos];
   positionptr = queue->positions[pos];

   while( left < queue->size )
   {
      if( right < queue->size )
      {
         /* both children exist, so swap with smallest child */
         if( key <= queue->keys[left] )
         {
            /* left child is not smaller than element */
            if( key <= queue->keys[right] )
               break;
            else
            {
               /* swap with right child */
               queue->keys[pos] = queue->keys[right];
               queue->data[pos] = queue->data[right];
               queue->positions[pos] = queue->positions[right];

               if( queue->positions[pos] != NULL )
                  *(queue->positions[pos]) = pos;

               pos = right;
            }
         }
         else
         {
            /* left child is smaller than the element */
            if( queue->keys[left] <= queue->keys[right] )
            {
               /* left child is smaller than right child and smaller than the element */
               queue->keys[pos] = queue->keys[left];
               queue->data[pos] = queue->data[left];
               queue->positions[pos] = queue->positions[left];

               if( queue->positions[pos] != NULL )
                  *(queue->positions[pos]) = pos;

               pos = left;
            }
            else
            {
               /* right child is smallest element */
               queue->keys[pos] = queue->keys[right];
               queue->data[pos] = queue->data[right];
               queue->positions[pos] = queue->positions[right];

               if( queue->positions[pos] != NULL )
                  *(queue->positions[pos]) = pos;

               pos = right;
            }
         }
      }
      else
      {
         /* only left child exists */
         if( key > queue->keys[left] )
         {
            /* left child is smaller than element, so swap them */
            queue->keys[pos] = queue->keys[left];
            queue->data[pos] = queue->data[left];
            queue->positions[pos] = queue->positions[left];

            if( queue->positions[pos] != NULL )
               *(queue->positions[pos]) = pos;

            pos = left;
         }
         else
            break;
      }

      left = 2 * pos + 1;
      right = 2 * pos + 2;
   }

   queue->keys[pos] = key;
   queue->data[pos] = elem;
   queue->positions[pos] = positionptr;

   if( positionptr != NULL )
      *positionptr = pos;
}

/** returns the value of a minimum element in 'elem'. Sets 'elem' to -1 if the queue is empty */
static
void pqueue_get_min(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   PQUEUE*               queue,              /**< priority queue instance                                         */
   int*                  elem                /**< address where the value of a minimum key item will be stored    */
   )
{
   *elem = -1;

   if( queue->size > 0 )
   {
      *elem = queue->data[0];

      queue->size--;

      /* copy last element to front and repair heap structure */
      if( queue->size > 0 )
      {
         queue->keys[0] = queue->keys[queue->size];
         queue->data[0] = queue->data[queue->size];
         queue->positions[0] = queue->positions[queue->size];

         if( queue->positions[0] != NULL )
            *(queue->positions[0]) = 0;

         pqueue_increase_key(scip, queue, 0, queue->keys[0]);
      }
   }
}

/** allocates memory for a lagrange multiplier and a set covering solution */
static
SCIP_RETCODE allocateMemoryForSolution(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< core data structure                                             */
   SCP_Lagrange_Sol*     mult                /**< uninitialized solution data structure                           */
   )
{
   int i;

   SCIP_CALL( SCIPallocBufferArray(scip, &mult->u, core->nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &mult->subgradient, core->nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &mult->lagrangiancostslocal, core->nvariables) );
   SCIP_CALL( SCIPallocBufferArray(scip, &mult->lagrangiancostsglobal, core->nvariables) );
   SCIP_CALL( SCIPallocClearBufferArray(scip, &mult->xgreedylocal, SCIPgetNVars(scip)) );

   for( i = 0; i < core->nvariables; i++ )
   {
      mult->lagrangiancostslocal[i] = 0.0;
      mult->lagrangiancostsglobal[i] = 0.0;
   }

   for( i = 0; i < core->nconss; i++ )
   {
      mult->u[i] = 0.0;
      mult->subgradient[i] = 0.0;
   }

   mult->lblagrangeglobal = SCIP_REAL_MIN;
   mult->lblagrangelocal = SCIP_REAL_MIN;
   mult->ubgreedylocal = SCIP_REAL_MAX;

   return SCIP_OKAY;
}

/** checks if 'var' is a core variable */
static
SCIP_Bool isCoreVariable(
   SCP_CORE*             core,                /**< initialized core data structure                                 */
   SCIP_VAR*             var                  /**< SCIP variable                                                   */
   )
{
   int varidx;

   assert(core != NULL);
   assert(var != NULL);

   varidx = SCIPvarGetProbindex(var);
   assert(varidx != -1);

   return core->varincore[varidx];
}

/** checks if 'var' is fixed within instance 'inst' */
static
SCIP_Bool isFixedVariable(
   SCP_INSTANCE*         inst,               /**< instance of the set covering problem                            */
   SCIP_VAR*             var                 /**< SCIP variable                                                   */
   )
{
   int varidx;

   assert(inst != NULL);
   assert(var != NULL);

   varidx = SCIPvarGetProbindex(var);
   assert(varidx != -1);

   return inst->varsfixed[varidx];
}

/** checks if 'variable' is part of 'solution' */
static
SCIP_Bool isVarInSolution(
   SCIP_Bool*            solution,           /**< SCIP hashtable that contains SCIP variables                     */
   SCIP_VAR*             var                 /**< SCIP variable                                                   */
   )
{
   int varidx;

   assert(solution != NULL);
   assert(var != NULL);

   varidx = SCIPvarGetProbindex(var);
   assert(varidx != -1);

   return solution[varidx];
}

/** adds a variable to the core */
static
void addVarToCore(
   SCP_CORE*             core,               /**< initialized core data structure                                 */
   SCIP_VAR*             var                 /**< SCIP variable                                                   */
   )
{
   int varidx;

   assert(core != NULL);
   assert(var != NULL);

   varidx = SCIPvarGetProbindex(var);
   assert(varidx != -1);

   if( !core->varincore[varidx] )
      ++core->ncorevariables;
   core->varincore[varidx] = TRUE;
}

/** fixes 'var' within instance 'inst' */
static
void fixVariable(
   SCP_INSTANCE*         inst,               /**< SCP instance where 'variable' is to be fixed                    */
   SCIP_VAR*             var                 /**< SCIP variable                                                   */
   )
{
   int varidx;

   assert(inst != NULL);
   assert(var != NULL);

   varidx = SCIPvarGetProbindex(var);
   assert(varidx != -1);

   if( !inst->varsfixed[varidx] )
      ++inst->nvarsfixed;
   inst->varsfixed[varidx] = TRUE;
}

/** adds variable 'var' to 'solution' */
static
void addVarToSolution(
   SCIP_Bool*            solution,           /**< solution represented as a boolean array                         */
   SCIP_VAR*             var                 /**< SCIP variable                                                   */
   )
{
   int varidx;

   assert(solution != NULL);
   assert(var != NULL);

   varidx = SCIPvarGetProbindex(var);
   assert(varidx != -1);

   /* no test whether the variable is already part of the solution */
   solution[varidx] = TRUE;
}

/** removes all variables from the core */
static
void removeVarsFromCore(
   SCIP*                 scip,               /**< SCIP data structure                                             */
   SCP_CORE*             core                /**< initialized core data structure                                 */
   )
{
   int i;

   assert(core != NULL);

   for( i = 0; i < SCIPgetNVars(scip); ++i )
      core->varincore[i] = FALSE;
   core->ncorevariables = 0;
}

/** removes a 'var' from 'solution' */
static
void removeVarFromSolution(
   SCIP_Bool*            solution,           /**< solution represented as a boolean array                         */
   SCIP_VAR*             var                 /**< SCIP variable                                                   */
   )
{
   int varidx;

   assert(solution != NULL);
   assert(var != NULL);

   varidx = SCIPvarGetProbindex(var);
   assert(varidx != -1);

   /* no test whether the variable is already part of the solution */
   solution[varidx] = FALSE;
}

/** clears a solution */
static
void clearSolution(
   SCIP*                 scip,               /**< SCIP data structure                                             */
   SCIP_Bool*            solution            /**< solution represented as a boolean array                         */
   )
{
   int i;

   assert(scip != NULL);
   assert(solution != NULL);

   for( i = 0; i < SCIPgetNVars(scip); ++i )
      solution[i] = FALSE;
}

/** checks if the row at position 'rowpos' is covered by fixed variables of 'inst' */
static
SCIP_Bool isRowCovered(
   SCP_INSTANCE*         inst,               /**< SCP instance                                                    */
   int                   rowpos              /**< position of the row within 'core'                               */
   )
{
   assert(inst != NULL);

   return inst->rowscovered[rowpos];
}

/** marks the row at position 'rowpos' as covered within instance 'inst' */
static
void markRowAsCovered(
   SCP_INSTANCE*         inst,               /**< SCP instance                                                    */
   int                   rowpos              /**< position of the row within 'core'                               */
   )
{
   assert(inst != NULL);

   if( !inst->rowscovered[rowpos] )
      ++inst->nrowscovered;
   inst->rowscovered[rowpos] = TRUE;
}

/** returns the position of 'variable' within the array core->variables */
static
SCIP_RETCODE getVarIndex(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCIP_VAR*             variable,           /**< SCIP variable                                                   */
   int*                  pos                 /**< address where the position is to be stored                      */
   )
{
   int varidx;

   assert(scip != NULL);
   assert(core != NULL);
   assert(variable != NULL);
   assert(pos != NULL);

   varidx = SCIPvarGetIndex(variable);
   *pos = (int) (size_t) SCIPhashmapGetImage(core->mapvariables, (void *) ((size_t) (varidx + 1))); /*lint !e507*/

   return SCIP_OKAY;
}

/** get all variables that are part of the constraint at position 'pos' (within SCIP) and saves them into 'vars' */
static
SCIP_RETCODE getConsVars(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   int                   pos,                /**< position of the constraint within SCIP                          */
   SCIP_VAR**            vars,               /**< allocated memory that can hold all variables                    */
   int*                  nvars,              /**< address where the number of variables will be stored            */
   SCIP_Bool*            success             /**< address where the result of the operation will be stored        */
   )
{
   *success = FALSE;

   if( SCIPconsIsActive(core->conss[pos]) == FALSE )
      return SCIP_OKAY;

   SCIP_CALL( SCIPgetConsNVars(scip, core->conss[pos], nvars, success) );

   if( *success == FALSE )
      return SCIP_OKAY;

   SCIP_CALL( SCIPgetConsVars(scip, core->conss[pos], vars, core->maxconstraintvariables, success) );

   return SCIP_OKAY;
}

/** releases all memory of a lagrange multiplier */
static
void freeMemoryForSolution(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_Lagrange_Sol*     mult                /**< initialized SCP lagrange multiplier that is to be freed         */
   )
{
   SCIPfreeBufferArray(scip, &mult->xgreedylocal);
   SCIPfreeBufferArray(scip, &mult->lagrangiancostsglobal);
   SCIPfreeBufferArray(scip, &mult->lagrangiancostslocal);
   SCIPfreeBufferArray(scip, &mult->subgradient);
   SCIPfreeBufferArray(scip, &mult->u);
}

/** creates a set covering solution. Adds all fixed variables of 'inst' and all variables of 'source' to 'dest'.
 *  'destcosts' contains the total costs of the solution.
 */
static
void copySetCoverSolution(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_INSTANCE*         inst,               /**< SCP instance                                                    */
   SCIP_Bool*            dest,               /**< array where variables belonging to the solution are inserted    */
   SCIP_Real*            destcosts,          /**< address where the total costs of the solution will be stored    */
   SCIP_Bool*            source              /**< SCP solution to the (reduced) instance 'inst'                   */
   )
{
   int i;
   SCIP_Real costs = 0.0;

   /* remove all entries from dest */
   clearSolution(scip, dest);

   /* iterate through all variables of the problem */
   for( i = 0; i < core->nvariables; i++ )
   {
      if( isVarInSolution(source, core->variables[i]) || isFixedVariable(inst, core->variables[i]) )
      {
         /* add variable 'i' if it is in the solution 'source' or fixed in instance 'inst' */
         costs += SCIPvarGetObj(core->variables[i]);
         addVarToSolution(dest, core->variables[i]);
      }
   }

   /* save total costs if required */
   if( destcosts != NULL )
      *destcosts = costs;
}

/** copies all data of the lagrange multiplier 'source' to the lagrange multiplier 'dest' */
static
void copySolution(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_Lagrange_Sol*     dest,               /**< SCP lagrange multiplier where data will be copied to            */
   SCP_Lagrange_Sol*     source              /**< SCP lagrange multiplier where data will be copied from          */
   )
{
   int i;

   clearSolution(scip, dest->xgreedylocal);

   memcpy(dest->lagrangiancostslocal, source->lagrangiancostslocal, sizeof(*dest->lagrangiancostslocal) * core->nvariables);
   memcpy(dest->lagrangiancostsglobal, source->lagrangiancostsglobal, sizeof(*dest->lagrangiancostsglobal) * core->nvariables);
   memcpy(dest->u, source->u, sizeof(*dest->u) * core->nconss);
   memcpy(dest->subgradient, source->subgradient, sizeof(*dest->subgradient) * core->nconss);

   for( i = 0; i < core->nvariables; i++ )
   {
      if( isVarInSolution(source->xgreedylocal, core->variables[i]) )
         addVarToSolution(dest->xgreedylocal, core->variables[i]);
   }

   dest->lblagrangeglobal = source->lblagrangeglobal;
   dest->lblagrangelocal = source->lblagrangelocal;
   dest->ubgreedylocal = source->ubgreedylocal;
}

/** initializes an instance where no variables are fixed and no rows are covered */
static
SCIP_RETCODE initInstance(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_INSTANCE*         inst                /**< unitialized SCP instance                                        */
   )
{
   SCIP_CALL( SCIPallocClearBufferArray(scip, &inst->varsfixed, SCIPgetNVars(scip)) );
   SCIP_CALL( SCIPallocClearBufferArray(scip, &inst->rowscovered, SCIPgetNConss(scip)) );
   inst->nvarsfixed = 0;
   inst->nrowscovered = 0;
   inst->costsfixed = 0.0;

   return SCIP_OKAY;
}

/** copies the fixed variables from 'source' to 'dest', but not the set of covered rows. this must be done separately. */
static
void copyInstance(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_INSTANCE*         dest,               /**< initialized SCP instance                                        */
   SCP_INSTANCE*         source              /**< initialized SCP instance that is to be copied                   */
   )
{
   int i;

   for( i = 0; i < SCIPgetNVars(scip); ++i )
      dest->varsfixed[i] = FALSE;
   dest->nvarsfixed = 0;
   for( i = 0; i < SCIPgetNConss(scip); ++i )
      dest->rowscovered[i] = FALSE;
   dest->nrowscovered = 0;
   dest->costsfixed = 0.0;

   for( i = 0; i < core->nvariables; i++ )
   {
      if( isFixedVariable(source, core->variables[i]) == TRUE )
      {
         fixVariable(dest, core->variables[i]);
         dest->costsfixed += SCIPvarGetObj(core->variables[i]);
      }
   }
}

/** releases all memory used by an instance */
static
void freeInstance(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_INSTANCE*         inst                /**< initialized SCP instance that is to be freed                    */
   )
{
   SCIPfreeBufferArray(scip, &inst->rowscovered);
   SCIPfreeBufferArray(scip, &inst->varsfixed);
}

/** initializes a tentative core: for each row the first few columns covering this row are added to the core */
static
SCIP_RETCODE initTentativeCore(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCIP_HEURDATA*        heurdata            /**< pointer to SCIP's heurdata                                      */
   )
{
   SCIP_VAR** vars;
   SCIP_Bool success;
   int nvars;
   int i;
   int j;

   assert(scip != NULL);
   assert(core != NULL);

   /* mapvariables is a mapping: SCIPvarGetIndex(core->variables[i]) -> i */
   SCIP_CALL( SCIPhashmapCreate(&core->mapvariables, SCIPblkmem(scip), SCIPgetNVars(scip)) );
   SCIP_CALL( SCIPallocClearBufferArray(scip, &core->varincore, SCIPgetNVars(scip)) );

   core->rowsavailable = FALSE;
   core->rows = NULL;
   core->nrowvars = NULL;
   core->columnsavailable = FALSE;
   core->columns = NULL;
   core->nvariables = SCIPgetNVars(scip);
   core->variables = SCIPgetVars(scip);

   SCIP_CALL( SCIPallocBufferArray(scip, &core->delta, core->nvariables) );
   SCIP_CALL( SCIPallocBufferArray(scip, &core->delta_pos, core->nvariables) );
   SCIP_CALL( SCIPallocBufferArray(scip, &core->solgreedy, core->nvariables) );
   core->nsolgreedy = 0;

   SCIP_CALL( SCIPallocBufferArray(scip, &core->nvarconss, core->nvariables) );
   for( i = 0; i< core->nvariables; i++ )
      core->nvarconss[i] = 0;

   /* construct mapping of variable-indices to array 'variables' */
   for( i = 0; i < core->nvariables; i++ )
   {
      int varidx = SCIPvarGetIndex(core->variables[i]);
      SCIP_CALL( SCIPhashmapInsert(core->mapvariables, (void *) ((size_t) (varidx + 1)), (void *) (size_t) i) );
   }

   core->nactiveconss = 0;
   core->maxconstraintvariables = 0;
   core->nconss = SCIPgetNConss(scip);
   core->conss = SCIPgetConss(scip);
   core->ncorevariables = 0;
   core->listcorevariables = NULL;
   SCIP_CALL( SCIPallocBufferArray(scip, &core->constraintid, core->nconss) );

   SCIP_CALL( SCIPallocBufferArray(scip, &vars, core->nvariables) );

   for( i = 0; i < core->nconss; i++ )
   {
      core->constraintid[i] = i;
      if( SCIPconsIsActive(core->conss[i]) == FALSE )
      {
         SCIPdebugMessage("constraint %i (%s) is inactive\n", i, SCIPconsGetName(core->conss[i]));
         continue;
      }

      /* get all variables that are part of this constraint */
      SCIP_CALL( SCIPgetConsNVars(scip, core->conss[i], &nvars, &success) );
      if( success == FALSE )
      {
         SCIPdebugMessage("constraint %i (%s): can't get number of variables\n", i, SCIPconsGetName(core->conss[i]));
         continue;
      }

      SCIP_CALL( SCIPgetConsVars(scip, core->conss[i], vars, core->nvariables, &success) );
      if( success == FALSE )
      {
         SCIPdebugMessage("constraint %i (%s): can't get variables\n", i, SCIPconsGetName(core->conss[i]));
         continue;
      }

      if( nvars > core->maxconstraintvariables )
         core->maxconstraintvariables = nvars;

      for( j = 0; j < nvars; j++ )
      {
         int varpos;

         SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );
         if( j < heurdata->param_core_tent_size )
         {
            /* add this variable to the core if it's not already in there */
            if( !isCoreVariable(core, core->variables[varpos]) )
               addVarToCore(core, core->variables[varpos]);
         }

         /* increase the number of constraints this variable is part of */
         core->nvarconss[varpos]++;
      }

      core->nactiveconss++;
   }

   SCIPfreeBufferArray(scip, &vars);

   /* create list of core variables, so it is easy to traverse them */
   j = 0;
   SCIP_CALL( SCIPallocBufferArray(scip, &core->listcorevariables, core->ncorevariables) );
   for( i = 0; i < core->nvariables; i++ )
   {
      if( isCoreVariable(core, core->variables[i]) )
         core->listcorevariables[j++] = i;
   }

   SCIPdebugMessage("%d variables in the tentative core\n", core->ncorevariables);

   return SCIP_OKAY;
}

/** adds all fixed variables of 'inst' to a set covering solution 'solution' */
static
void extendSolution(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_INSTANCE*         inst,               /**< (reduced) SCP instance                                          */
   SCIP_Bool*            solution            /**< SCP solution (a hashtable that contains SCIP variables)         */
   )
{
   int i;

   for( i = 0; i < core->nvariables; i++ )
   {
      if( !isFixedVariable(inst, core->variables[i]) )
         continue;

      /* add fixed variable if it is not already part of the solution */
      if( !isVarInSolution(solution, core->variables[i]) )
         addVarToSolution(solution, core->variables[i]);
   }
}

/** constructs rows of all constraints, but only includes core variables */
static
SCIP_RETCODE computeCoreRows(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCIP_HEURDATA*        heurdata            /**< pointer to the heuristics's data                                */
   )
{
   SCIP_Bool success;
   SCIP_VAR** vars;
   int i;
   int j;
   int ncorevars;
   int nvars;

   assert(scip != NULL);
   assert(core != NULL);
   assert(heurdata != NULL);

   /* don't compute again if already computed */
   if( core->rowsavailable )
      return SCIP_OKAY;

   assert(core->rows == NULL);

   SCIP_CALL( SCIPallocBufferArray(scip, &core->rows, core->nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &core->nrowvars, core->nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &vars, core->maxconstraintvariables) );

   /* iterate through list of constraints */
   for( i = 0; i < core->nconss; i++ )
   {
      core->rows[i] = NULL;
      core->nrowvars[i] = 0;

      SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
      if( success == FALSE )
         continue;

      /* count number of core variables this constraint contains */
      ncorevars = 0;
      for( j = 0; j < nvars; j++ )
      {
         int varpos;

         SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );

         if( isCoreVariable(core, core->variables[varpos]) )
            ncorevars++;
      }

      if( ncorevars > 0 )
      {
         /* create array of core variables of this constraint */
         core->nrowvars[i] = ncorevars;
         SCIP_CALL( SCIPallocBufferArray(scip, &(core->rows[i]), core->nrowvars[i]) ); /*lint !e866*/
         for( j = 0; j < nvars; j++ )
         {
            int varpos;

            SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );

            if( isCoreVariable(core, core->variables[varpos]) )
            {
               ncorevars--;
               core->rows[i][ncorevars] = varpos;
            }
         }
      }
   }

   SCIPfreeBufferArray(scip, &vars);
   core->rowsavailable = TRUE;
   return SCIP_OKAY;
}

/** constructs columns of core variables to provide faster access */
static
SCIP_RETCODE computeCoreColumns(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core                /**< SCP core data structure                                         */
   )
{
   SCIP_Bool success;
   SCIP_VAR** vars;
   int i;
   int j;
   int k;
   int nvars;

   assert(scip != NULL);
   assert(core != NULL);

   /* don't compute columns again if already computed */
   if( core->columnsavailable )
      return SCIP_OKAY;

   SCIP_CALL( SCIPallocBufferArray(scip, &core->columns, core->nvariables) );
   for( i = 0; i < core->nvariables; i++ )
   {
      /* columns[i] = NULL for all non-core variables */
      core->columns[i] = NULL;

      /* check if variable is a core variable */
      if( !isCoreVariable(core, core->variables[i]) )
         continue;

      /* only allocate memory of it is part of any constraints at all (this should always be the case for core variables) */
      if( core->nvarconss[i] == 0 )
         continue;

      SCIP_CALL( SCIPallocBufferArray(scip, &(core->columns[i]), core->nvarconss[i]) ); /*lint !e866*/

      for( j = 0; j < core->nvarconss[i]; j++ )
         core->columns[i][j] = -1;
   }

   SCIP_CALL( SCIPallocBufferArray(scip, &vars, core->maxconstraintvariables) );

   for( i = 0; i < core->nconss; i++ )
   {
      SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
      if( success == FALSE )
         continue;

      for( j = 0; j < nvars; j++ )
      {
         int varpos;

         SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );

         if( !isCoreVariable(core, core->variables[varpos]) )
            continue;

         /* add this constraint to the column of the variable */
         for( k = 0; k < core->nvarconss[varpos]; k++ )
         {
            /* find first position that is not used yet in column 'varpos' */
            if( core->columns[varpos][k] == -1 )
            {
               core->columns[varpos][k] = i;
               break;
            }
         }
      }
   }

   SCIPfreeBufferArray(scip, &vars);
   core->columnsavailable = TRUE;
   return SCIP_OKAY;
}

/** releases memory that is used by the core, including rows and columns, if they were computed */
static
void freeCore(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core                /**< SCP core data structure that is to be freed                     */
   )
{
   assert(core != NULL);
   assert(core->varincore != NULL);
   assert(core->mapvariables != NULL);
   assert(core->nvarconss != NULL);

   if( core->rowsavailable )
   {
      int i;

      for( i = 0; i < core->nconss; i++ )
         if( core->rows[i] )
            SCIPfreeBufferArray(scip, &core->rows[i]);

      SCIPfreeBufferArray(scip, &core->nrowvars);
      SCIPfreeBufferArray(scip, &core->rows);
   }

   if( core->columnsavailable )
   {
      int i;

      for( i = 0; i < core->nvariables; i++ )
         if( core->columns[i] )
            SCIPfreeBufferArray(scip, &core->columns[i]);

      SCIPfreeBufferArray(scip, &core->columns);
   }

   SCIPfreeBufferArrayNull(scip, &core->listcorevariables);
   SCIPfreeBufferArray(scip, &core->constraintid);
   SCIPfreeBufferArray(scip, &core->nvarconss);
   SCIPfreeBufferArray(scip, &core->solgreedy);
   SCIPfreeBufferArray(scip, &core->delta_pos);
   SCIPfreeBufferArray(scip, &core->delta);
   SCIPfreeBufferArray(scip, &core->varincore);
   SCIPhashmapFree(&core->mapvariables);
}

/** computes a new core based on the delta values of variables, see formula (9) in the paper */
static
SCIP_RETCODE redefineCore(
   SCIP*                 scip,               /**< master SCIP data structure                                    */
   SCIP_HEURDATA*        heurdata            /**< pointer to the heuristic's data; this contains the SCP core     */
   )
{
   SCIP_Bool recomputecolumns = FALSE;
   SCIP_Bool recomputerows = FALSE;
   SCP_CORE* core;
   int* deltaperm;
   int nvars;
   SCIP_VAR** vars;

   int i;
   int j;

   /* assumption: delta values were already computed and are sorted in increasing order, this happens in computeDelta */

   core = &heurdata->core;
   vars = heurdata->vars;

   SCIP_CALL( SCIPallocBufferArray(scip, &deltaperm, core->nvariables) );

   /* construct mapping deltaperm: variables -> delta_pos */
   for( i = 0; i < core->nvariables; i++ )
   {
      deltaperm[core->delta_pos[i]] = i;
   }

   /* release memory that is used for the core rows and columns. it will be allocated again later on */
   removeVarsFromCore(scip, core);
   if( core->columnsavailable )
   {
      recomputecolumns = TRUE;

      for( i = 0; i < core->nvariables; i++ )
      {
         if( core->columns[i] != NULL )
            SCIPfreeBufferArray(scip, &core->columns[i]);
      }

      SCIPfreeBufferArray(scip, &core->columns);
      core->columns = NULL;
      core->columnsavailable = FALSE;
   }

   if( core->rowsavailable )
   {
      recomputerows = TRUE;

      for( i = 0; i < core->nconss; i++ )
      {
         if( core->rows[i] )
            SCIPfreeBufferArray(scip, &core->rows[i]);
      }

      SCIPfreeBufferArray(scip, &core->rows);
      SCIPfreeBufferArray(scip, &core->nrowvars);

      core->rows = NULL;
      core->nrowvars = NULL;
      core->rowsavailable = FALSE;
   }

   if( core->listcorevariables != NULL )
   {
      SCIPfreeBufferArray(scip, &core->listcorevariables);
      core->listcorevariables = NULL;
   }

   /* pick the first 'SCP_CORE_TENT_SIZE'*m columns with lowest delta values to be in the core */
   for( i = 0; i < core->nvariables; i++ )
   {
      if( i >= heurdata->param_core_tent_size * core->nactiveconss )
         break;

      addVarToCore(core, core->variables[core->delta_pos[i]]);
   }

   /* then add the first 'SCP_CORE_TENT_SIZE' columns covering each row in increasing order of their delta values */
   for( i = 0; i < core->nconss; i++ )
   {
      SCIP_Bool success = FALSE;
      int *cols = heurdata->rccols; /*[SCP_CORE_TENT_SIZE];*/
      SCIP_Real *coldelta = heurdata->rccoldelta; /*[SCP_CORE_TENT_SIZE];*/

      SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
      if( success == FALSE )
         continue;

      for( j = 0; j < heurdata->param_core_tent_size; j++ )
      {
         cols[j] = 0;
         coldelta[j] = SCIP_REAL_MAX;
      }

      /* iterate through list of variables that are part of this constraint, and find the 'param_core_tent_size' variables with lowest delta values */
      for( j = 0; j < nvars; j++ )
      {
         int varpos;
         int k;
         SCIP_Real value;

         SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );

         value = core->delta[deltaperm[varpos]];

         if( j < heurdata->param_core_tent_size )
            k = j;
         else
            k = heurdata->param_core_tent_size - 1;

         if( (j < heurdata->param_core_tent_size) || (coldelta[k] > value) )
         {
            cols[k] = varpos;
            coldelta[k] = value;

            /* insertion sort */
            while( (k > 0) && (coldelta[k] < coldelta[k - 1]) )
            {
               int tmp1 = cols[k - 1];
               SCIP_Real tmp2 = coldelta[k - 1];

               cols[k - 1] = cols[k];
               coldelta[k - 1] = coldelta[k];
               cols[k] = tmp1;
               coldelta[k] = tmp2;
               k--;
            }
         }
      }

      /* add the variables with lowest delta values to the core */
      for( j = 0; j < nvars; j++ )
      {
         if( j >= heurdata->param_core_tent_size )
            break;

         if( !isCoreVariable(core, core->variables[cols[j]]) )
            addVarToCore(core, core->variables[cols[j]]);
      }
   }

   /* construct list of core variables */
   SCIP_CALL( SCIPallocBufferArray(scip, &core->listcorevariables, core->ncorevariables) );
   j = 0;
   for( i = 0; i < core->nvariables; i++ )
   {
      if( isCoreVariable(core, core->variables[i]) )
         core->listcorevariables[j++] = i;
   }
   assert(j == core->ncorevariables);

   /* recompute core rows and columns if they were available before */
   if( recomputecolumns )
      SCIP_CALL( computeCoreColumns(scip, core) );

   if( recomputerows )
      SCIP_CALL( computeCoreRows(scip, core, heurdata) );

   SCIPfreeBufferArray(scip, &deltaperm);

   SCIPdebugMessage("%d variables are in the refined core\n", core->ncorevariables);
   return SCIP_OKAY;
}

/** for all rows that are covered by the variables in inst->varsfixed, this function adds their indices to inst->rowscovered */
static
SCIP_RETCODE markRowsCoveredByFixedVariables(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_INSTANCE*         inst,               /**< initialized SCP instance where some variables may be fixed      */
   SCIP_HEURDATA*        heurdata            /**< pointer to the heuristic's data                                 */
   )
{
   int i;
   int j;

   for( i = 0; i < SCIPgetNConss(scip); ++i )
      inst->rowscovered[i] = FALSE;
   inst->nrowscovered = 0;

   if( core->columnsavailable == TRUE )
   {
      /* iterate through list of core variables */
      for( i = 0; i < core->ncorevariables; i++ )
      {
         int corevar = core->listcorevariables[i];

         if( !isFixedVariable(inst, core->variables[corevar]) )
            continue;

         /* this variable is fixed, so mark its rows as covered */
         for( j = 0; j < core->nvarconss[corevar]; j++ )
         {
            int rowidx = core->columns[corevar][j];

            if( !isRowCovered(inst, rowidx) )
               markRowAsCovered(inst, rowidx);
         }
      }
   }
   else
   {
      SCIP_VAR **vars;
      int nvars;

      vars = heurdata->vars;

      /* if the core columns are not available, iterate through list of constraints and test which conss are covered */
      for( i = 0; i < core->nconss; i++ )
      {
         SCIP_Bool success = FALSE;

         SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
         if( success == FALSE )
            continue;

         for( j = 0; j < nvars; j++ )
         {
            if( isFixedVariable(inst, vars[j]) )
            {
               if( !isRowCovered(inst, i) )
                  markRowAsCovered(inst, i);

               break;
            }
         }
      }
   }

   return SCIP_OKAY;
}

/** checks whether the given 'solution' is actually a set cover */
static
SCIP_RETCODE checkSetCover(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_INSTANCE*         inst,               /**< SCP instance                                                    */
   SCIP_Bool*            solution,           /**< possible SCP solution                                           */
   SCIP_Bool*            issetcover,         /**< address where TRUE will be stored if 'solution' is a set cover  */
   SCIP_HEURDATA*        heurdata            /**< pointer to the heuristic's data                                 */
   )
{
   SCIP_Bool success;
   SCIP_VAR** vars;
   int nvars;
   int i;
   int j;
   int varpos;

   assert(scip != NULL);
   assert(core != NULL);
   assert(inst != NULL);
   assert(solution != NULL);
   assert(issetcover != NULL);
   assert(heurdata != NULL);

   *issetcover = TRUE;
   vars = heurdata->vars;

   /* iterate through all constraints and check whether each of them contains a variable that is part of the cover */
   for( i = 0; i < core->nconss; i++ )
   {
      SCIP_Bool rowcovered = FALSE;

      SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
      if( success == FALSE )
         continue;

      for( j = 0; j < nvars; j++ )
      {
         SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );

         if( isVarInSolution(solution, core->variables[varpos]) )
         {
            rowcovered = TRUE;
            break;
         }
      }

      if( rowcovered == FALSE )
      {
         SCIPdebugMessage("check set cover: row %i is not covered by any column\n", i);
         *issetcover = FALSE;
         break;
      }
   }

   return SCIP_OKAY;
}

/** computes delta values for variables and fixes new variables within inst, see formula (9) of the paper */
static
SCIP_RETCODE computeDelta(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_INSTANCE*         inst,               /**< SCP instance                                                    */
   SCIP_Real*            lagrangiancosts,    /**< array with lagrangian costs of the rows                         */
   SCIP_Bool*            solution,           /**< valid solution to the global SCP instance                       */
   SCIP_Real             pi,                 /**< percentage of rows that need be covered by new fixed variables  */
   SCIP_HEURDATA*        heurdata            /**< pointer to the heuristic's data                                 */
   )
{
   SCIP_Bool success;
   SCIP_Real delta_sum;
   SCIP_Real delta_max;
   int* nvarcovering;
   SCIP_VAR** vars;
   int nvars;
   int i;
   int j;
   int* delta_pos;
   SCIP_Real* delta;

   delta = core->delta;
   delta_pos = core->delta_pos;

   SCIP_CALL( SCIPallocBufferArray(scip, &nvarcovering, core->nconss) );
   vars = heurdata->vars;

   /* compute nvarcovering[i] = 'number of columns covering row i' */
   if( core->rowsavailable )
   {
      for( i = 0; i < core->nconss; i++ )
      {
         nvarcovering[i] = 0;

         if( !SCIPconsIsActive(core->conss[i]) )
            continue;

         for( j = 0; j < core->nrowvars[i]; j++ )
         {
            int varpos = core->rows[i][j];
            if( isVarInSolution(solution, core->variables[varpos]) )
               nvarcovering[i]++;
         }
      }
   }
   else
   {
      for( i = 0; i < core->nconss; i++ )
      {
         nvarcovering[i] = 0;

         SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
         if( success == FALSE )
            continue;

         for( j = 0; j < nvars; j++ )
         {
            int varpos;

            SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );

            if( isVarInSolution(solution, core->variables[varpos]) )
               nvarcovering[i]++;
         }
      }
   }

   /* compute delta values of variables that are part of 'solution', see formula (9) */
   for( i = 0; i < core->nvariables; i++ )
   {
      delta[i] = SCIP_REAL_MAX;
      delta_pos[i] = i;

      if( !isVarInSolution(solution, core->variables[i]) )
         continue;

      delta[i] = lagrangiancosts[i];
      if( delta[i] < 0.0 )
         delta[i] = 0.0;

      for( j = 0; j < core->nvarconss[i]; j++ )
      {
         int rowpos = core->columns[i][j];

         if( isRowCovered(inst, rowpos) )
            continue;

         delta[i] += lagrangiancosts[rowpos] * (nvarcovering[rowpos] - 1) / ((SCIP_Real) nvarcovering[rowpos]);
      }
   }

   /* sort the variables in increasing order of their delta values, delta_pos is used to track the variables within the array */
   SCIPsortRealInt(delta, delta_pos, core->nvariables);

   /* fix variables as long as the sum of their delta values is not greater than delta_max.
    * We assume that variables with small delta values are likely part of an optimal solution. */
   delta_sum = 0.0;
   delta_max = core->nactiveconss * pi;

   for( i = 0; i < SCIPgetNVars(scip); ++i )
      inst->varsfixed[i] = FALSE;
   inst->nvarsfixed = 0;

   inst->costsfixed = 0.0;

   for( i = 0; i < core->nvariables; i++ )
   {
      int varpos = delta_pos[i];

      if( !isVarInSolution(solution, core->variables[varpos]) )
         break;

      inst->costsfixed += SCIPvarGetObj(core->variables[varpos]);
      delta_sum += delta[i];

      /* fix variable delta_pos[i] */
      fixVariable(inst, core->variables[varpos]);

      if( delta_sum >= delta_max )
         break;
   }

   SCIPfreeBufferArray(scip, &nvarcovering);
   return SCIP_OKAY;
}

/* removes redundant columns from 'solution' by removing them one by one (in the order of decreasing costs),
 *  and checking whether all rows are still covered */
static
SCIP_RETCODE removeRedundantColumns(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCIP_HEURDATA*        heurdata,           /**< pointer to the heuristic's data                                 */
   SCIP_Bool*            solution,           /**< solution to the global SCP instance                             */
   SCIP_Real*            solcosts            /**< pointer to original costs of 'solution'; contains updated costs */
   )
{
   SCP_CORE* core;
   SCIP_Real* costs;
   int* varpos;
   int* nvarcovering;
   SCIP_VAR** vars;
   int nvars;
   SCIP_Bool success;
   int solsize;

   int i;
   int j;

   core = &heurdata->core;

   if( core->columnsavailable == FALSE )
   {
      SCIPdebugMessage("can only remove redundant columns if all core columns are available\n");
      return SCIP_OKAY;
   }

   SCIP_CALL( SCIPallocBufferArray(scip, &costs, core->nvariables) );
   SCIP_CALL( SCIPallocBufferArray(scip, &varpos, core->nvariables) );
   SCIP_CALL( SCIPallocBufferArray(scip, &nvarcovering, core->nconss) );

   vars = heurdata->vars;

   /* compute nvarcovering[i] = 'number of columns covering row i' */
   if( core->rowsavailable )
   {
      for( i = 0; i < core->nconss; i++ )
      {
         nvarcovering[i] = 0;

         if( SCIPconsIsActive(core->conss[i]) == FALSE )
            continue;

         if( core->nrowvars[i] == 0 )
            continue;

         for( j = 0; j < core->nrowvars[i]; j++ )
         {
            int vpos = core->rows[i][j];

            if( isVarInSolution(solution, core->variables[vpos]) )
               nvarcovering[i]++;
         }
      }
   }
   else
   {
      /* iterate through the constraints if the core rows are not available */
      for( i = 0; i < core->nconss; i++ )
      {
         nvarcovering[i] = 0;

         SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
         if( success == FALSE )
            continue;

         for( j = 0; j < nvars; j++ )
         {
            int vpos;

            SCIP_CALL( getVarIndex(scip, core, vars[j], &vpos) );

            if( isVarInSolution(solution, core->variables[vpos]) )
               nvarcovering[i]++;
         }
      }
   }

   /* construct array of costs of variables that are part of the solution.
    * If v_1, ..., v_solsize are the variables of the solution, then
    *    costs[i] = -costs(v_i), and varpos[i] = i.
    * We use negative costs because we only sort them in increasing order and want to remove the ones with highest costs first.
    * The 'varpos' array is used to track which variable is where within the sorted array (it is permutated in the same way as 'costs') */
   solsize = 0;
   for( i = 0; i < core->nvariables; i++ )
   {
      if( isVarInSolution(solution, core->variables[i]) )
      {
         costs[solsize] = -SCIPvarGetObj(core->variables[i]);
         varpos[solsize] = i;
         solsize++;
      }
   }

   /* sort variables by decreasing costs */
   SCIPsortRealInt(costs, varpos, solsize);

   /* for each variable of the solution: try to remove it and check whether it is still a set cover */
   for( i = 0; i < solsize; i++ )
   {
      SCIP_Bool canremove = TRUE;
      int vpos = varpos[i];

      /* we can only remove this variable if every column it covers is covered by another variable */
      for( j = 0; j < core->nvarconss[vpos]; j++ )
      {
         if( nvarcovering[core->columns[vpos][j]] == 1 )
         {
            canremove = FALSE;
            break;
         }
      }

      if( canremove == TRUE )
      {
         removeVarFromSolution(solution, core->variables[vpos]);
         *solcosts -= SCIPvarGetObj(core->variables[vpos]);

         for( j = 0; j < core->nvarconss[vpos]; j++ )
         {
            nvarcovering[core->columns[vpos][j]]--;
         }
      }
   }

   SCIPfreeBufferArray(scip, &nvarcovering);
   SCIPfreeBufferArray(scip, &varpos);
   SCIPfreeBufferArray(scip, &costs);

   return SCIP_OKAY;
}

/** greedy set cover algorithm that uses lagrangian costs instead of original costs. */
static
SCIP_RETCODE greedySetCover(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_INSTANCE*         inst,               /**< (reduced) SCP instance                                          */
   SCP_Lagrange_Sol*     mult,               /**< lagrange multiplier that is used to compute lagrangian costs    */
   SCIP_HEURDATA*        heurdata            /**< pointer to the heuristic's data                                 */
   )
{
   SCP_INSTANCE* greedyinst;
   int nrowsuncovered = 0;
   SCIP_Bool success = FALSE;
   int nvars;
   PQUEUE* prioqueue;
   int* colpos;
   int* colmu;
   SCIP_Real* colgamma;
   SCIP_Real* colscore;
   SCIP_VAR** vars;

   int i;
   int j;
   int k;

   core->nsolgreedy = 0;
   mult->ubgreedylocal = 0.0;
   clearSolution(scip, mult->xgreedylocal);

   greedyinst = &heurdata->greedyinst;
   for( i = 0; i < SCIPgetNConss(scip); ++i )
      greedyinst->rowscovered[i] = FALSE;
   greedyinst->nrowscovered = 0;

   /* count number of uncovered rows and add covered rows to 'greedyinst' */
   for( i = 0; i < core->nconss; i++ )
   {
      if( SCIPconsIsActive(core->conss[i]) == FALSE )
         continue;

      /* this is actually necessary because there exist constraints where this fails, and we simply need to ignore them */
      SCIP_CALL( SCIPgetConsNVars(scip, core->conss[i], &nvars, &success) );
      if( success == FALSE )
         continue;

      if( isRowCovered(inst, i) )
         markRowAsCovered(greedyinst, i);
      else
         nrowsuncovered++;
   }

   if( core->columnsavailable == FALSE )
   {
      SCIPerrorMessage("greedy algorithm requires core columns to be available\n");
      SCIPABORT();
   }

   /* compute scores of all (unfixed) variables and add them to the priority queue */
   colpos = heurdata->greedycolpos;       /* this array contains the position of the element within the queue and is needed to update the costs */
   colmu = heurdata->greedycolmu;         /* for each column this contains the number of uncovered rows that it covers */
   colgamma = heurdata->greedycolgamma;   /* for each column this contains the lagrangian costs of uncovered rows that it covers */
   colscore = heurdata->greedycolscore;   /* for each column this contains its score */
   vars = heurdata->vars;

   prioqueue = &heurdata->greedyqueue;
   pqueue_clear(scip, prioqueue);

   /* compute initial scores for all unfixed columns, see "3. Heuristic Phase" of the paper, and add them to the priority queue */
   for( i = 0; i < core->nvariables; i++ )
   {
      int varpos = i;
      int mu = 0;
      SCIP_Real gamma = SCIPvarGetObj(core->variables[varpos]);

      colmu[i] = 0;
      colgamma[i] = 0;
      colpos[i] = 0;
      colscore[i] = 0.0;

      if( !isCoreVariable(core, core->variables[varpos]) )
          continue;
      else if( isFixedVariable(inst, core->variables[varpos]) )
         continue;

      for( j = 0; j < core->nvarconss[varpos]; j++ )
      {
         if( !isRowCovered(greedyinst, core->columns[varpos][j]) )
         {
            gamma -= mult->u[core->columns[varpos][j]];
            mu++;
         }
      }

      /* skip columns that do not cover anything */
      if( mu > 0 )
      {
         SCIP_Real score = (gamma > 0) ? (gamma / ((SCIP_Real) mu)) : (gamma * mu);
         colmu[i] = mu;
         colgamma[i] = gamma;
         colscore[i] = score;

         SCIP_CALL( pqueue_insert(scip, prioqueue, colscore[i], i, &(colpos[i])) );
      }
   }

   /* main loop: add variables to the solution as long as there are uncovered rows */
   while( nrowsuncovered > 0 )
   {
      int mincolumn;

      /* get variable with minimum score */
      pqueue_get_min(scip, prioqueue, &mincolumn);

      /* add variable 'variables[mincolumn]' to the set cover */
      addVarToSolution(mult->xgreedylocal, core->variables[mincolumn]);
      mult->ubgreedylocal += SCIPvarGetObj(core->variables[mincolumn]);
      core->solgreedy[core->nsolgreedy++] = mincolumn;

      /* mark this variable, so its score won't be updated in the future */
      colmu[mincolumn] = 0;

      /* mark all rows covered by this variable as covered and update scores of variables that are affected by this */
      for( j = 0; j < core->nvarconss[mincolumn]; j++ )
      {
         int columnpos = core->columns[mincolumn][j];

         if( !isRowCovered(greedyinst, columnpos) )
         {
            markRowAsCovered(greedyinst, columnpos);
            nrowsuncovered--;

            /* update scores of columns covering this row */
            SCIP_CALL( getConsVars(scip, core, columnpos, vars, &nvars, &success) );
            if( success == FALSE )
               continue;

            /* for each variable: subtract u[i] from the variable's costs */
            for( k = 0; k < nvars; k++ )
            {
               SCIP_Real score;
               int varpos;

               SCIP_CALL( getVarIndex(scip, core, vars[k], &varpos) );

               /* skip non-core variables */
               if( !isCoreVariable(core, core->variables[varpos]) )
                  continue;

               if( colmu[varpos] == 0 )
                  continue;

               score = colscore[varpos];

               /* compute a new score for this variable */
               colscore[varpos] = SCIP_REAL_MAX;
               colmu[varpos]--;
               colgamma[varpos] += mult->u[columnpos];

               if( colmu[varpos] > 0 )
                  colscore[varpos] = (colgamma[varpos] > 0) ? (colgamma[varpos] / ((SCIP_Real) colmu[varpos])) : (colgamma[varpos] * colmu[varpos]);

               /* either decrease or increase the key, depending on whether it decreased or increased */
               if( score > colscore[varpos] )
                  pqueue_decrease_key(scip, prioqueue, colpos[varpos], colscore[varpos]);
               else
                  pqueue_increase_key(scip, prioqueue, colpos[varpos], colscore[varpos]);
            }
         }
      }
   }

   return SCIP_OKAY;
}

/** computes lagrangian costs for all columns, only considering rows that are uncovered by fixed variables in 'inst' */
static
SCIP_RETCODE computeLocalLagrangianCosts(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_INSTANCE*         inst,               /**< (reduced) SCP instance                                          */
   SCP_Lagrange_Sol*     mult,               /**< lagrange multiplier that is used to compute the lagrangian costs*/
   SCIP_HEURDATA*        heurdata            /**< pointer to the heuristic's data                                 */
   )
{
   SCIP_VAR** vars;
   int nvars;

   int i;
   int j;

   /* set all lagrangian costs to objective values */
   for( i = 0; i < core->nvariables; i++ )
      mult->lagrangiancostslocal[i] = SCIPvarGetObj(core->variables[i]);

   vars = heurdata->vars;

   if( core->rowsavailable )
   {
      for( i = 0; i < core->nconss; i++ )
      {
         if( SCIPconsIsActive(core->conss[i]) == FALSE )
            continue;

         if( isRowCovered(inst, i) )
            continue;

         for( j = 0; j < core->nrowvars[i]; j++ )
         {
            int varpos = core->rows[i][j];
            mult->lagrangiancostslocal[varpos] -= mult->u[i];
         }
      }
   }
   else
   {
      for( i = 0; i < core->nconss; i++ )
      {
         SCIP_Bool success;

         /* skip rows that are not part of the reduced instance */
         if( isRowCovered(inst, i) )
            continue;

         SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
         if( success == FALSE )
            continue;

         /* for each core variable: subtract u[i] from the variable's costs */
         for( j = 0; j < nvars; j++ )
         {
            int varpos;

            SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );

            mult->lagrangiancostslocal[varpos] -= mult->u[i];
         }
      }
   }

   return SCIP_OKAY;
}

/** computes lagrangian costs for all columns of the unrestricted instance and a global lower bound. */
static
SCIP_RETCODE computeGlobalLagrangianCosts(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_Lagrange_Sol*     mult,               /**< lagrange multiplier that is used to compute the lagrangian costs*/
   SCIP_HEURDATA*        heurdata            /**< pointer to the heuristic's data                                 */
   )
{
   SCIP_VAR** vars;
   int nvars;
   int i;
   int j;

   /* set all lagrangian costs to objective values */
   for( i = 0; i < core->nvariables; i++ )
      mult->lagrangiancostsglobal[i] = SCIPvarGetObj(core->variables[i]);

   vars = heurdata->vars;

   for( i = 0; i < core->nconss; i++ )
   {
      SCIP_Bool success = FALSE;

      SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
      if( success == FALSE )
         continue;

      /* for each core variable: subtract u[i] from the variable's costs */
      for( j = 0; j < nvars; j++ )
      {
         int varpos;

         SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );

         mult->lagrangiancostsglobal[varpos] -= mult->u[i];
      }

      mult->lblagrangeglobal += mult->u[i];
   }

   /* compute global lower bound */
   for( i = 0; i < core->nvariables; i++ )
   {
      if( mult->lagrangiancostsglobal[i] < 0.0 )
         mult->lblagrangeglobal += mult->lagrangiancostsglobal[i];
   }

   return SCIP_OKAY;
}

/** computes an optimal solution to the lagrangian relaxation, see formulae (4), (5) in the paper */
static
SCIP_RETCODE computeOptimalSolution(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_INSTANCE*         inst,               /**< (reduced) SCP instance                                          */
   SCP_Lagrange_Sol*     mult,               /**< lagrange multiplier                                             */
   SCIP_HEURDATA*        heurdata            /**< pointer to the heuristic's data                                 */
   )
{
   SCIP_VAR** vars;
   SCIP_Bool success;
   int nvars;

   int i;
   int j;

   mult->lblagrangelocal = 0.0;
   mult->lblagrangeglobal = 0.0;

   /* compute lagrangian costs for the reduced instance */
   SCIP_CALL( computeLocalLagrangianCosts(scip, core, inst, mult, heurdata) );

   /* also, compute costs for the unrestricted instance. This even computes mult->lblagrangeglobal */
   SCIP_CALL( computeGlobalLagrangianCosts(scip, core, mult, heurdata) );

   /* compute an optimal solution for 'inst', but only save its costs in lblagrangelocal */
   for( i = 0; i < core->ncorevariables; i++ )
   {
      int varpos = core->listcorevariables[i];

      if( mult->lagrangiancostslocal[varpos] < 0.0 )
      {
         if( !isFixedVariable(inst, core->variables[varpos]) )
            mult->lblagrangelocal = mult->lblagrangelocal + mult->lagrangiancostslocal[varpos];
      }
   }

   /* the subgradient can be computed faster when the core rows are available */
   if( core->rowsavailable )
   {
      /* compute the subgradient */
      for( i = 0; i < core->nconss; i++ )
      {
         mult->subgradient[i] = 0.0;

         if( SCIPconsIsActive(core->conss[i]) == FALSE )
            continue;

         if( isRowCovered(inst, i) )
            continue;

         if( core->nrowvars[i] == 0 )
            continue;

         mult->subgradient[i] = 1.0;

         for( j = 0; j < core->nrowvars[i]; j++ )
         {
            int varpos = core->rows[i][j];

            if( mult->lagrangiancostslocal[varpos] < 0.0 )
               mult->subgradient[i] -= 1.0;
         }

         /* to get a lower bound from the multiplier we still need to subtract sum_i u[i] from lblagrangelocal */
         mult->lblagrangelocal = mult->lblagrangelocal + mult->u[i];
      }
   }
   else
   {
      /* the core rows are NOT available, so we need to iterate through the constraints */
      vars = heurdata->vars;

      for( i = 0; i < core->nconss; i++ )
      {
         mult->subgradient[i] = 0.0;

         if( isRowCovered(inst, i) )
            continue;

         SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
         if( success == FALSE )
            continue;

         mult->subgradient[i] = 1.0;

         for( j = 0; j < nvars; j++ )
         {
            int varpos;

            SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );

            if( !isCoreVariable(core, core->variables[varpos]) )
               continue;

            if( mult->lagrangiancostslocal[varpos] < 0.0 )
               mult->subgradient[i] -= 1.0;
         }

         mult->lblagrangelocal = mult->lblagrangelocal + mult->u[i];
      }
   }

   return SCIP_OKAY;
}

/** subgradient method with Held-Karp update of subgradients */
static
SCIP_RETCODE subgradientOptimization(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_INSTANCE*         inst,               /**< (reduced) SCP instance                                          */
   SCP_Lagrange_Sol*     best_mult_lb,       /**< lagrange multiplier that gives the best lower bound for 'inst'  */
   SCIP_Real             bestub,             /**< costs of the best known solution for 'inst'                     */
   SCIP_HEURDATA*        heurdata            /**< pointer the heuristic's data                                    */
   )
{
   int max_iter;
   SCP_Lagrange_Sol last_mult;
   SCP_Lagrange_Sol next_mult;
   SCP_Lagrange_Sol tmp_mult;
   SCIP_Real norm;
   SCIP_Real lambda = heurdata->param_lambda;
   SCIP_Bool lambda_adjustments = heurdata->param_lambda_adjustments;
   SCIP_Real* last_lb = heurdata->sglastlb;
   SCIP_Real stop_crit_lb = 0.0;
   int iter;
   int last_data_pos = 0;
   int i;

   SCIP_CALL( allocateMemoryForSolution(scip, core, &next_mult) );
   SCIP_CALL( allocateMemoryForSolution(scip, core, &last_mult) );

   /* save data from best lower bound multiplier in last_mult */
   copySolution(scip, core, &last_mult, best_mult_lb);

   /* permutate best u by multiplying each entry with a uniformly random value in the range [0.9, 1.1] */
   for( i = 0; i < core->nconss; i++ )
   {
      if( !isRowCovered(inst, i) )
         last_mult.u[i] = SCIPrandomGetReal(heurdata->randnumgen, 0.9, 1.1) * last_mult.u[i];
      else
         last_mult.u[i] = 0.0;
   }

   /* bestub is an upper bound for the restricted instance, without the fixed costs */
   bestub -= inst->costsfixed;

   /* maximum number of subgradient iterations */
   max_iter = 10 * core->nconss;

   for( iter = 0; iter < max_iter; iter++ )
   {
      /* compute norm of the subgradient */
      norm = 0.0;
      for( i = 0; i < core->nconss; i++ )
         norm = norm + last_mult.subgradient[i] * last_mult.subgradient[i]; /* we have subgradient[i] = 0.0 if row i is not to be considered */

      /* the norm is zero if an optimal solution (to the restricted instance) was found */
      if( SCIPisZero(scip, norm) )
         break;

      /* Held-Karp update */
      for( i = 0; i < core->nconss; i++ )
      {
         SCIP_Real hk = last_mult.u[i] + lambda * (bestub - last_mult.lblagrangelocal) * last_mult.subgradient[i] / norm;  /*lint !e795*/
         next_mult.u[i] = 0.0;

         if( SCIPisPositive(scip, 0.0) )
            next_mult.u[i] = hk;
      }

      /* for each multiplier compute an optimal solution to get a lower bound */
      SCIP_CALL( computeOptimalSolution(scip, core, inst, &next_mult, heurdata) );

      /* save the best incumbent */
      if( SCIPisGT(scip, next_mult.lblagrangelocal, best_mult_lb->lblagrangelocal) )
         copySolution(scip, core, best_mult_lb, &next_mult);

      if( SCIPisGT(scip, next_mult.lblagrangeglobal, heurdata->multbestlbtotal.lblagrangeglobal) )
      {
         copySolution(scip, core, &heurdata->multbestlbtotal, &next_mult);
      }

      /* adjust lambda, the step size, if necessary */
      if( lambda_adjustments )
      {
         /* save last 'p' lower and upper bounds */
         last_lb[last_data_pos++] = next_mult.lblagrangelocal;

         /* lambda is adjusted after param_lambda_p iterations */
         if( last_data_pos >= heurdata->param_lambda_p )
         {
            SCIP_Real max_lb = last_lb[0];
            SCIP_Real min_lb = last_lb[0];

            /* find best and worst bounds that occurred within the last lambda_p iterations */
            for( i = 1; i < heurdata->param_lambda_p; i++ )
            {
               if( SCIPisGT(scip, last_lb[i], max_lb) )
                  max_lb = last_lb[i];
               if( SCIPisLT(scip, last_lb[i], min_lb) )
                  min_lb = last_lb[i];
            }

            /* if min_lb and max_lb differ by more than 1%, lambda is halved */
            if( SCIPisGT(scip, max_lb - min_lb, 0.01) )
               lambda = lambda / 2;

            /* if min_lb and max_lb differ by less than 0.1%, lambda is multiplied by 1.5 */
            if( SCIPisLT(scip, max_lb - min_lb, 0.001) )
               lambda = lambda * 1.5;
            last_data_pos = 0;
         }
      }

      /* swap next_mult and last_mult */
      tmp_mult = last_mult;
      last_mult = next_mult;
      next_mult = tmp_mult;

      /* check stopping criteria after stop_crit_iter iterations */
      if( iter % heurdata->param_stop_crit_iter == 0 )
      {
         /* we test how the current best lower bound compares to the best lower bound that was known 'stop_crit_iter' iterations ago:
          *    if the absolute difference is smaller than param_stop_crit_diff and
          *       the ratio lb_now / lb_old is at least stop_crit_ration
          *    then we stop */
         if( (iter > 0) && (best_mult_lb->lblagrangelocal - stop_crit_lb <= heurdata->param_stop_crit_diff )
            && (stop_crit_lb / best_mult_lb->lblagrangelocal >= heurdata->param_stop_crit_ratio) )
         {
            break;
         }

         stop_crit_lb = best_mult_lb->lblagrangelocal;
      }
   }

   freeMemoryForSolution(scip, &last_mult);
   freeMemoryForSolution(scip, &next_mult);

   return SCIP_OKAY;
}

/** computes an initial lagrange multiplier, see formula (8) in the paper */
static
SCIP_RETCODE computeInitialLagrangeMultiplier(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_CORE*             core,               /**< SCP core data structure                                         */
   SCP_INSTANCE*         inst,               /**< (reduced) SCP instance                                          */
   SCP_Lagrange_Sol*     mult,               /**< initialized lagrange multiplier were the new one will be stored */
   SCIP_HEURDATA*        heurdata            /**< pointer to the heuristic's data                                 */
   )
{
   int i;
   int j;

   if( core->columnsavailable == TRUE )
   {
      for( i = 0; i < core->nconss; i++ )
         mult->u[i] = SCIP_REAL_MAX;

      for( i = 0; i < core->nvariables; i++ )
      {
         int nuncovered = 0;

         if( !isCoreVariable(core, core->variables[i]) )
            continue;

         if( isFixedVariable(inst, core->variables[i]) )
            continue;

         /* count how many uncovered, active rows this column covers */
         for( j = 0; j < core->nvarconss[i]; j++ )
         {
            int colpos = core->columns[i][j];

            /* skip inactive constraints */
            if( SCIPconsIsActive(core->conss[colpos]) == FALSE )
               continue;

            /* skip covered rows */
            if( isRowCovered(inst, colpos) )
               continue;

            nuncovered++;
         }

         /* if this column covers any rows, update their cost if necessary */
         if( nuncovered > 0 )
         {
            SCIP_Real costs = SCIPvarGetObj(core->variables[i]) / ((SCIP_Real) nuncovered);
            int nvars;

            for( j = 0; j < core->nvarconss[i]; j++ )
            {
               int colpos = core->columns[i][j];
               SCIP_Bool success = FALSE;

               if( isRowCovered(inst, colpos) )
                  continue;

               if( SCIPconsIsActive(core->conss[colpos]) == FALSE )
                  continue;

               SCIP_CALL( SCIPgetConsNVars(scip, core->conss[colpos], &nvars, &success) );
               if( success == FALSE )
                  continue;

               if( costs < mult->u[colpos] )
                  mult->u[colpos] = costs;
            }
         }
      }
   }
   else
   {
      SCIP_VAR **vars;
      int nvars;
      int *nuncoveredactive;

      vars = heurdata->vars;
      SCIP_CALL( SCIPallocBufferArray(scip, &nuncoveredactive, core->nvariables) );

      for( i = 0; i < core->nvariables; i++ )
         nuncoveredactive[i] = 0;

      for( i = 0; i < core->nconss; i++ )
      {
         SCIP_Bool success = FALSE;

         if( isRowCovered(inst, i) )
            continue;

         SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
         if( success == FALSE )
            continue;

         for( j = 0; j < nvars; j++ )
         {
            int varpos;

            SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );

            if( !isCoreVariable(core, vars[j]) )
               continue;
            else
               nuncoveredactive[varpos]++;
         }
      }

      for( i = 0; i < core->nconss; i++ )
      {
         SCIP_Bool found = FALSE;
         SCIP_Bool success = FALSE;

         if( isRowCovered(inst, i) )
         {
            mult->u[i] = 0.0;
            continue;
         }

         SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
         if( success == FALSE )
            continue;

         for( j = 0; j < nvars; j++ )
         {
            if( !isCoreVariable(core, vars[j]) )
               continue;
            else
            {
               int varpos;
               SCIP_Real costs = SCIP_REAL_MAX;

               SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );

               if( nuncoveredactive[varpos] > 0 )
                  costs = SCIPvarGetObj(core->variables[varpos]) / ((SCIP_Real) nuncoveredactive[varpos]);

               if( (found == FALSE) || (costs < mult->u[i]) )
               {
                  found = TRUE;
                  mult->u[i] = costs;
               }
            }
         }
      }

      SCIPfreeBufferArray(scip, &nuncoveredactive);
   }

   return SCIP_OKAY;
}

/* Explores a neighborhood of lagrange multipliers around 'startmult',
 * and calls the greedy algorithm for each multiplier. The neighborhood is
 * simply computed by doing the Held-Karp subgradient updates. The best solutions
 * are extended to solutions of the unrestricted instance.
 */
static
SCIP_RETCODE exploreNeighborhood(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_Lagrange_Sol*     startmult,          /**< lagrange multiplier where the search is started                 */
   SCIP_HEURDATA*        heurdata            /**< pointer to the heuristic's data                                 */
   )
{
   SCP_CORE* core;
   SCP_INSTANCE* subinst;
   SCP_Lagrange_Sol mult;
   SCIP_VAR** vars;
   SCIP_Bool success;
   SCIP_Real bestub;
   SCIP_Real norm;
   SCIP_Real costs;
   int nvars;
   int iter;
   int i;
   int j;

   core = &heurdata->core;
   subinst = &heurdata->subinst;
   vars = heurdata->vars;
   bestub = heurdata->bestub - subinst->costsfixed;

   SCIP_CALL( allocateMemoryForSolution(scip, &heurdata->core, &mult) );
   copySolution(scip, core, &mult, startmult);

   /* perform 'param_greedy_max_iter' subgradient iterations */
   for( iter = 0; iter < heurdata->param_greedy_max_iter; iter++ )
   {
      /* compute subgradient for 'mult' */
      if( core->rowsavailable )
      {
         for( i = 0; i < core->nconss; i++ )
         {
            mult.subgradient[i] = 0.0;

            if( SCIPconsIsActive(core->conss[i]) == FALSE )
               continue;

            if( isRowCovered(subinst, i) )
               continue;

            if( core->nrowvars[i] == 0 )
               continue;

            mult.subgradient[i] = 1.0;

            for( j = 0; j < core->nrowvars[i]; j++ )
            {
               int varpos = core->rows[i][j];

               if( mult.lagrangiancostslocal[varpos] < 0.0 )
                  mult.subgradient[i] -= 1.0;
            }
         }
      }
      else
      {
         for( i = 0; i < core->nconss; i++ )
         {
            mult.subgradient[i] = 0.0;

            if( isRowCovered(subinst, i) )
               continue;

            SCIP_CALL( getConsVars(scip, core, i, vars, &nvars, &success) );
            if( success == FALSE )
               continue;

            mult.subgradient[i] = 1.0;

            for( j = 0; j < nvars; j++ )
            {
               int varpos;

               SCIP_CALL( getVarIndex(scip, core, vars[j], &varpos) );

               if( !isCoreVariable(core, core->variables[varpos]) )
                  continue;

               if( mult.lagrangiancostslocal[varpos] < 0.0 )
                  mult.subgradient[i] -= 1.0;
            }
         }
      }

      /* compute norm of subgradient */
      norm = 0.0;
      for( i = 0; i < core->nconss; i++ )
         norm = norm + mult.subgradient[i] * mult.subgradient[i];

      /* the norm is zero if an optimal solution (to the restricted instance) was found */
      if( SCIPisZero(scip, norm) )
         break;

      /* Held-Karp update */
      for( i = 0; i < core->nconss; i++ )
      {
         SCIP_Real hk = mult.u[i] + heurdata->param_lambda * (bestub - mult.lblagrangelocal) * mult.subgradient[i] / norm; /*lint !e795*/
         mult.u[i] = 0.0;

         if( hk > 0.0 )
            mult.u[i] = hk;
      }

      /* first, compute lagrangian costs */
      SCIP_CALL( computeOptimalSolution(scip, core, subinst, &mult, heurdata) );

      /* save this multiplier if it gives a better local lower bound than the one known */
      if( mult.lblagrangelocal > heurdata->multbestlbsubinst.lblagrangelocal )
         copySolution(scip, core, &heurdata->multbestlbsubinst, &mult);

      /* save this multiplier if it gives a better global lower bound */
      if( mult.lblagrangeglobal > heurdata->multbestlbtotal.lblagrangeglobal )
         copySolution(scip, core, &heurdata->multbestlbtotal, &mult);

      /* apply the greedy algorithm and extend it to a solution of the unrestricted instance by adding fixed variables */
      SCIP_CALL( greedySetCover(scip, core, subinst, &mult, heurdata) );
      extendSolution(scip, core, subinst, mult.xgreedylocal);

      costs = subinst->costsfixed + mult.ubgreedylocal;
      SCIP_CALL( removeRedundantColumns(scip, heurdata, mult.xgreedylocal, &costs) );

      /* save solution if it is better */
      if( costs < heurdata->bestubsubinst )
         copySetCoverSolution(scip, core, subinst, heurdata->bestubsubinstsol, &heurdata->bestubsubinst, mult.xgreedylocal);
   }

   freeMemoryForSolution(scip, &mult);

   return SCIP_OKAY;
}

/* reports a solution to SCIP */
static
SCIP_RETCODE reportSolution(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCP_INSTANCE*         inst,               /**< (reduced) SCP instance                                          */
   SCIP_Bool*            solution,           /**< solution for 'inst'; will be extended to global solution        */
   SCIP_HEUR*            heur                /**< pointer to the heuristic's data                                 */
   )
{
   SCIP_VAR** solvars;
   SCIP_Real* solvals;
   int nsolvars;
   int i;
   SCIP_SOL* newsol;
   SCIP_Bool success = FALSE;
   SCIP_Bool foundsol = FALSE;
   SCIP_CONS** conss;
   int nconss;

   SCIP_CALL( SCIPgetVarsData(scip, &solvars, &nsolvars, NULL, NULL, NULL, NULL) );
   SCIP_CALL( SCIPcreateSol(scip, &newsol, heur) );
   SCIP_CALL( SCIPallocBufferArray(scip, &solvals, nsolvars) );

   for( i = 0; i < nsolvars; i++ )
   {
      if( isVarInSolution(solution, solvars[i]) )
         solvals[i] = 1.0;
      else if( (isFixedVariable(inst, solvars[i]) == TRUE) && (SCIPisZero(scip, SCIPvarGetObj(solvars[i])) == FALSE) )
         solvals[i] = 1.0;
      else
         solvals[i] = 0.0;
   }

   SCIP_CALL( SCIPsetSolVals(scip, newsol, nsolvars, solvars, solvals) );

   /* test all constraints and check if the activity is correct, adjust some free variable if necessary */
   conss = SCIPgetConss(scip);
   nconss = SCIPgetNConss(scip);
   for( i = 0; (i < nconss) && (foundsol == FALSE); i++ )
   {
      SCIP_Real lhs = 0.0, activity = 0.0;
      SCIP_Real *vals = NULL;
      SCIP_Bool valuesallones = FALSE;

      if( !strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(conss[i])), "linear") )
      {
         lhs = SCIPgetLhsLinear(scip, conss[i]);
         activity = SCIPgetActivityLinear(scip, conss[i], newsol);
         vals = SCIPgetValsLinear(scip, conss[i]);
      }
      else if( !strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(conss[i])), "logicor") )
      {
         valuesallones = TRUE;
      }
      else if( !strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(conss[i])), "masterbranch") )
      {
         /* do nothing */
         SCIPdebugMessage("constraint %i is a masterbranch\n", SCIPconsGetPos(conss[i]));
         continue;
      }
      else
      {
         SCIPerrorMessage("constraint is '%s', can't handle this\n", SCIPconshdlrGetName(SCIPconsGetHdlr(conss[i])));
         SCIPABORT();
      }

      if( lhs > activity )
      {
         int j;
         int nvars;
         SCIP_Bool changed = FALSE;
         SCIP_VAR **vars;


         /*SCIPdebugMessage("constraint %i: left hand side is violated by %f\n", i, lhs - activity);*/
         SCIP_CALL( SCIPgetConsNVars(scip, conss[i], &nvars, &success) );
         SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
         SCIP_CALL( SCIPgetConsVars(scip, conss[i], vars, nvars, &success) );

         for( j = 0; (changed == FALSE) && (j < nvars); j++ )
         {
            SCIP_Bool costszero = SCIPisZero(scip, SCIPvarGetObj(vars[j]));
            if( costszero == TRUE )
            {
               assert(valuesallones || vals != NULL);

               /* constraint is "logicor" */
               if( valuesallones )
                  SCIP_CALL( SCIPincSolVal(scip, newsol, vars[j], lhs - activity) );
               /* constraint is "linear" */
               else
               {
                  assert(vals != NULL);
                  if( !SCIPisZero(scip, vals[j]) )
                     SCIP_CALL( SCIPincSolVal(scip, newsol, vars[j], (lhs - activity) / vals[j]) );
                  else
                     SCIPdebugMessage("could not adjust activity\n");
               }

               SCIP_CALL( SCIPtrySol(scip, newsol, FALSE, FALSE, TRUE, TRUE, TRUE, &foundsol) );
               changed = TRUE;
            }
         }

         if( changed == FALSE )
            SCIPdebugMessage("could not find variable with zero costs\n");

         SCIPfreeBufferArray(scip, &vars);
      }

      if( !strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(conss[i])), "linear") )
      {
         if( SCIPgetLhsLinear(scip, conss[i]) > SCIPgetActivityLinear(scip, conss[i], newsol) )
            SCIPdebugMessage("activity is still smaller than lhs\n");
      }
      /* take care of the case rhs < activity. Question: can this occur?
       * YES it can, but only the convexity constraint can be violated (is this really true?).
       * We simply ignore this issue. SCIP will not accept the solution if it is not good enough.
       */
   }

   SCIP_CALL( SCIPtrySolFree(scip, &newsol, FALSE, FALSE, TRUE, TRUE, TRUE, &success) );
   SCIPfreeBufferArray(scip, &solvals);

   if( success == TRUE )
      SCIPdebugMessage("new solution found by set covering heuristic\n");

   return SCIP_OKAY;
}

/** the three-phase procedure from the paper. It tries to find near-optimal lagrange multipliers and
 * reduces the size of an instance by fixing variables that are likely to be in an optimal solution. */
static
SCIP_RETCODE threePhase(
   SCIP*                 scip,               /**< master SCIP data structure                                      */
   SCIP_HEUR*            heur,               /**< master SCIP heur data structure                                 */
   SCIP_HEURDATA*        heurdata            /**< pointer to the heuristic's data                                 */
   )
{
   SCP_Lagrange_Sol* mult_lb_subinst;
   int i;
   SCIP_Bool success;
   SCP_CORE* core;
   SCP_INSTANCE* inst;
   SCP_INSTANCE* subinst;
   int niter;

   core = &heurdata->core;
   inst = &heurdata->inst;
   subinst = &heurdata->subinst;
   mult_lb_subinst = &heurdata->tpmultlbsubinst;

   /* we first create our own copy of the instance, as we need to mark variables as fixed until all variables are fixed */
   copyInstance(scip, core, subinst, inst);
   SCIP_CALL( markRowsCoveredByFixedVariables(scip, core, subinst, heurdata) );

   /* if this function is called for the first time, no multiplier is available and we need to compute one to start with */
   if( heurdata->useinitialmultiplier )
   {
      /* next, compute initial lagrange multipliers and find a first lower bound */
      SCIP_CALL( computeInitialLagrangeMultiplier(scip, core, subinst, mult_lb_subinst, heurdata) );
      heurdata->useinitialmultiplier = FALSE;
   }
   else
   {
      /* take the best lagrange multiplier of the unrestricted instance as starting point */
      copySolution(scip, core, mult_lb_subinst, &heurdata->multbestlbtotal);
   }

   /* computeOptimalSolution computes an optimal solution to the lagrange relaxation. Also, it computes the subgradient */
   SCIP_CALL( computeOptimalSolution(scip, core, subinst, mult_lb_subinst, heurdata) );
   SCIP_CALL( greedySetCover(scip, core, subinst, mult_lb_subinst, heurdata) );

   /* we now have a lower and upper bound in mult_lb_local for the instance 'inst' and take these as our starting values */
   copySetCoverSolution(scip, core, inst, heurdata->bestubinst_sol, &heurdata->bestubinst, mult_lb_subinst->xgreedylocal);
   copySetCoverSolution(scip, core, subinst, heurdata->bestubsubinstsol, &heurdata->bestubsubinst, mult_lb_subinst->xgreedylocal);
   copySolution(scip, core, &heurdata->multbestlbinst, mult_lb_subinst);

   /* check whether 'bestubinst_sol' is a solution of the reduced instance 'subinst' */
   SCIP_CALL( checkSetCover(scip, core, subinst, heurdata->bestubinst_sol, &success, heurdata) );
   if( !success )
   {
      SCIPerrorMessage("three-phase: initial solution is not a valid set cover\n");
      SCIPABORT();
   }

   /* in the first call of this proceduce there is no best upper bound for the unrestricted instance */
   if( heurdata->bestubinst < heurdata->bestub )
   {
      copySetCoverSolution(scip, core, inst, heurdata->bestubsol, &heurdata->bestub, heurdata->bestubinst_sol);
      SCIPdebugMessage("new upper bound: %f\n", heurdata->bestub);
      SCIP_CALL( reportSolution(scip, inst, heurdata->bestubsol, heur) );
   }

   if( core->nactiveconss <= subinst->nrowscovered )
   {
      SCIPdebugMessage("threephase: all rows are already covered\n");
   }

   /* The next loop works as follows:
    * 1. Derive the subinstance 'subinst' by marking all rows as covered that are covered by fixed variables of 'subinst'. If all rows are covered by fixed variables, stop.
    * 2. Perform subgradient optimization to find a lagrange multiplier that gives a better lower bound for 'subinst'.
    * 3. Explore the neighborhood of this best multiplier by doing additional subgradient steps and calling the greedy algorithm with each multiplier as argument.
    * 4. Check whether any of these local solutions can be extended to better global solutions.
    * 5. Fix several variables to reduce the size of 'subinst'.
    */

   /* stop if all rows are covered by fixed variables or a maximum number of iterations has been reached */
   for( niter = 0; niter < heurdata->param_threephase_max_iter && core->nactiveconss > subinst->nrowscovered; ++niter )
   {
      /* we mark all rows covered by fixed variables, in addition to the ones that were already covered */
      SCIP_CALL( markRowsCoveredByFixedVariables(scip, core, subinst, heurdata) );
      SCIP_CALL( subgradientOptimization(scip, core, subinst, mult_lb_subinst, heurdata->bestubsubinst, heurdata) );

      /* explore neighborhood of the best lower bound multiplier by applying the held-karp update again.
       * this updates all upper bounds (and sometimes also lower bounds) in 'heurdata' */
      SCIP_CALL( exploreNeighborhood(scip, mult_lb_subinst, heurdata) );

      /* extend a solution of 'subinst' to one of 'inst' if it gives a better upper bound */
      if( heurdata->bestubsubinst < heurdata->bestubinst )
      {
         copySetCoverSolution(scip, core, subinst, heurdata->bestubinst_sol, &heurdata->bestubinst, heurdata->bestubsubinstsol);

         /* extend a solution of 'inst' to a solution of the whole instance and check whether this gives a better global upper bound */
         if( heurdata->bestubinst < heurdata->bestub )
         {
            copySetCoverSolution(scip, core, inst, heurdata->bestubsol, &heurdata->bestub, heurdata->bestubinst_sol);

            SCIP_CALL( removeRedundantColumns(scip, heurdata, heurdata->bestubsol, &heurdata->bestub) );
            SCIPdebugMessage("new upper bound: %f\n", heurdata->bestub);
            SCIP_CALL( reportSolution(scip, inst, heurdata->bestubsol, heur) );
         }
      }

      /* stop if any solution can not be better than the best known solution */
      if( subinst->costsfixed + mult_lb_subinst->lblagrangelocal >= heurdata->bestub )
         break;

      /* At each step we fix new variables: fix the first max(1, nconstraints / 200) variables
       * that were picked by the greedy solution (in the order they were picked) */
      for( i = 0; i < core->nsolgreedy && i <= core->nconss / 200; i++ )
      {
         fixVariable(subinst, core->variables[core->solgreedy[i]]);
         subinst->costsfixed += SCIPvarGetObj(core->variables[core->solgreedy[i]]);
      }
   }

   SCIP_CALL( checkSetCover(scip, core, inst, heurdata->bestubinst_sol, &success, heurdata) );
   if( !success )
   {
      SCIPdebugMessage("three-phase: final solution is not a valid set cover\n");
   }

   return SCIP_OKAY;
}

/** driver for the three-phase procedure */
static
SCIP_RETCODE setCoveringHeuristic(
   SCIP*                 scip,               /**< master SCIP data structure                                                 */
   SCIP_HEUR*            heur                /**< original SCIP heuristic data structure                                     */
   )
{
   SCIP_HEURDATA* heurdata;
   SCIP_Bool stopcft = FALSE;
   int niter = 0;                            /* total number of iterations of the three-phase loop                           */
   int nitercore = 0;                        /* total number of iterations on the current core                               */
   int niternoimp = 0;                       /* number of iterations for how long no improvement on the upper bound was made */
   int coret = 10;                           /* compute new core after this number of iterations                             */
   SCIP_Real pi;                             /* percentage of rows that need to be covered in each fixing phase              */
   SCIP_Real corelb = 0.0;
   SCP_CORE* core;
   SCP_INSTANCE* inst;
   SCIP_Bool success;

   int i;

   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* allocate memory that is used locally by redefineCore */
   SCIP_CALL( SCIPallocBufferArray(scip, &heurdata->rccols, heurdata->param_core_tent_size) );
   SCIP_CALL( SCIPallocBufferArray(scip, &heurdata->rccoldelta, heurdata->param_core_tent_size) );

   /* allocate memory that is used locally by subgradientOptimization */
   SCIP_CALL( SCIPallocBufferArray(scip, &heurdata->sglastlb, heurdata->param_lambda_p) );

   /* we always consider core variables (columns) only to cover rows, so we need to define the first core.
    * This is done by choosing for each row the first five columns covering it and adding them to the core. */
   SCIP_CALL( initTentativeCore(scip, &heurdata->core, heurdata) );

   /* allocate memory that is used to get all variables that are part of some constraint.
    * 'maxconstraintvariables' is the maximum number of variables any constraint of the problem contains. */
   SCIP_CALL( SCIPallocBufferArray(scip, &heurdata->vars, heurdata->core.maxconstraintvariables) );

   /* after defining the core, we compute all core columns and core rows */
   SCIP_CALL( computeCoreColumns(scip, &heurdata->core) );
   SCIP_CALL( computeCoreRows(scip, &heurdata->core, heurdata) );

   /* set up basic instance. so far no variables are fixed */
   SCIP_CALL( initInstance(scip, &heurdata->inst) );
   SCIP_CALL( initInstance(scip, &heurdata->subinst) );

   core = &heurdata->core;
   inst = &heurdata->inst;
   pi = heurdata->param_pi_min;

   /* allocate memory that is used by the greedy algorithm locally */
   SCIP_CALL( pqueue_init(scip, &heurdata->greedyqueue) );
   SCIP_CALL( SCIPallocBufferArray(scip, &heurdata->greedycolpos, core->nvariables) );
   SCIP_CALL( SCIPallocBufferArray(scip, &heurdata->greedycolmu, core->nvariables) );
   SCIP_CALL( SCIPallocBufferArray(scip, &heurdata->greedycolgamma, core->nvariables) );
   SCIP_CALL( SCIPallocBufferArray(scip, &heurdata->greedycolscore, core->nvariables) );
   SCIP_CALL( initInstance(scip, &heurdata->greedyinst) );

   SCIP_CALL( allocateMemoryForSolution(scip, &heurdata->core, &heurdata->multbestlbinst) );
   SCIP_CALL( allocateMemoryForSolution(scip, &heurdata->core, &heurdata->multbestlbsubinst) );
   SCIP_CALL( allocateMemoryForSolution(scip, &heurdata->core, &heurdata->multbestlbtotal) );
   SCIP_CALL( allocateMemoryForSolution(scip, &heurdata->core, &heurdata->tpmultlbsubinst) );

   SCIP_CALL( SCIPallocClearBufferArray(scip, &heurdata->bestubinst_sol, SCIPgetNVars(scip)) );
   SCIP_CALL( SCIPallocClearBufferArray(scip, &heurdata->bestubsubinstsol, SCIPgetNVars(scip)) );
   SCIP_CALL( SCIPallocClearBufferArray(scip, &heurdata->bestubsol, SCIPgetNVars(scip)) );

   heurdata->multbestlbtotal.lblagrangeglobal = 0;
   heurdata->bestub = SCIP_REAL_MAX;
   heurdata->useinitialmultiplier = TRUE;

   while( stopcft == FALSE )
   {
      SCIP_Bool redefine = FALSE;
      SCIP_Real currentub = heurdata->bestub;


      /* derive the reduced subinstance by marking rows covered by fixed variables */
      for( i = 0; i < SCIPgetNConss(scip); ++i )
         inst->rowscovered[i] = FALSE;
      inst->nrowscovered = 0;
      SCIP_CALL( markRowsCoveredByFixedVariables(scip, core, inst, heurdata) );

      /* call three-phase as long as 'inst' contains uncovered rows */
      if( core->nactiveconss > inst->nrowscovered )
      {
         SCIPdebugMessage("%d variables are fixed, %d rows are covered\n", inst->nvarsfixed, inst->nrowscovered);

         /* apply procedure three-phase to find an near optimal lagrange multiplier */
         SCIP_CALL( threePhase(scip, heur, heurdata) );
      }
      else
      {
         /* compute new core when all of its rows are covered by fixed variables */
         redefine = TRUE;
      }

      /* stop if maximum number of iterations is reached */
      if( niter++ >= heurdata->param_max_iter )
         stopcft = TRUE;

      /* set pi to PI_MIN if the current best solution was found in this iteration */
      if( heurdata->bestub < currentub )
      {
         niternoimp = 0;
         pi = heurdata->param_pi_min;
      }
      else
      {
         /* increase pi if no better solution was found, i.e. fix more variables in order to cover more rows */
         pi = pi * heurdata->param_pi_alpha;
         niternoimp++;
      }

      /* stop if there was no improvement during the last 'SCP_MAX_ITER_NO_IMP' iterations */
      if( niternoimp >= heurdata->param_max_iter_no_imp )
         stopcft = TRUE;

      /* stop if UB <= beta * LB */
      if( heurdata->multbestlbtotal.lblagrangeglobal * heurdata->param_beta >= heurdata->bestub )
         stopcft = TRUE;

      /* redefine core if the current core was worked on for 'coret' iterations */
      if( nitercore++ == coret )
         redefine = TRUE;

      if( stopcft )
         break;

      /* compute delta fixes variables of 'bestubsol' within the core that are likely to be part of an optimal solution */
      SCIP_CALL( markRowsCoveredByFixedVariables(scip, core, inst, heurdata) );
      SCIP_CALL( computeDelta(scip, core, inst, heurdata->multbestlbtotal.lagrangiancostsglobal, heurdata->bestubsol, pi, heurdata) );

      if( redefine == TRUE )
      {
         /* stop if the last core did not lead to any improvements */
         if( SCIPisEQ(scip, corelb, heurdata->multbestlbtotal.lblagrangeglobal) == TRUE )
            stopcft = TRUE;
         else
         {
            SCIP_CALL( redefineCore(scip, heurdata) );
            for( i = 0; i < SCIPgetNVars(scip); ++i )
               inst->varsfixed[i] = FALSE;
            inst->nvarsfixed = 0;
            inst->costsfixed = 0.0;
            pi = heurdata->param_pi_min;
            nitercore = 0;
            corelb = heurdata->multbestlbtotal.lblagrangeglobal;
         }
      }

      SCIPdebugMessage("iteration %i: best lower bound: %f, best upper bound: %f\n", niter, heurdata->multbestlbtotal.lblagrangeglobal, heurdata->bestub);
   }

   for( i = 0; i < SCIPgetNVars(scip); ++i )
      inst->varsfixed[i] = FALSE;
   inst->nvarsfixed = 0;

   SCIP_CALL( checkSetCover(scip, core, inst, heurdata->bestubsol, &success, heurdata) );
   if( success == FALSE )
      SCIPdebugMessage("final solution is not a valid set cover\n");
   else
      SCIPdebugMessage("final solution has costs %f\n", heurdata->bestub);

   /* report the final solution to SCIP */
   SCIP_CALL( reportSolution(scip, inst, heurdata->bestubsol, heur) );

   /* release all memory that was allocated before */
   SCIPfreeBufferArray(scip, &heurdata->bestubsol);
   SCIPfreeBufferArray(scip, &heurdata->bestubsubinstsol);
   SCIPfreeBufferArray(scip, &heurdata->bestubinst_sol);

   freeMemoryForSolution(scip, &heurdata->tpmultlbsubinst);
   freeMemoryForSolution(scip, &heurdata->multbestlbtotal);
   freeMemoryForSolution(scip, &heurdata->multbestlbsubinst);
   freeMemoryForSolution(scip, &heurdata->multbestlbinst);

   freeInstance(scip, &heurdata->greedyinst);
   SCIPfreeBufferArray(scip, &heurdata->greedycolscore);
   SCIPfreeBufferArray(scip, &heurdata->greedycolgamma);
   SCIPfreeBufferArray(scip, &heurdata->greedycolmu);
   SCIPfreeBufferArray(scip, &heurdata->greedycolpos);

   pqueue_destroy(scip, &heurdata->greedyqueue);

   freeInstance(scip, &heurdata->subinst);
   freeInstance(scip, &heurdata->inst);

   SCIPfreeBufferArray(scip, &heurdata->vars);

   freeCore(scip, &heurdata->core);

   SCIPfreeBufferArray(scip, &heurdata->sglastlb);

   SCIPfreeBufferArray(scip, &heurdata->rccoldelta);
   SCIPfreeBufferArray(scip, &heurdata->rccols);

   return SCIP_OKAY;
}

/*
 * Callback methods of primal heuristic
 */

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeSetcover)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* free heuristic data */
   SCIPfreeMemory(scip, &heurdata);
   SCIPheurSetData(heur, NULL);

   return SCIP_OKAY;
}

/** initialization method of primal heuristic (called after problem was transformed) */
static
SCIP_DECL_HEURINIT(heurInitSetcover)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* create random number generator */
   SCIP_CALL( SCIPcreateRandom(scip, &heurdata->randnumgen,
         SCIPinitializeRandomSeed(scip, DEFAULT_RANDSEED), TRUE) );

   return SCIP_OKAY;
}

/** deinitialization method of primal heuristic (called before transformed problem is freed) */
static
SCIP_DECL_HEUREXIT(heurExitSetcover)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* free random number generator */
   SCIPfreeRandom(scip, &heurdata->randnumgen);

   return SCIP_OKAY;
}


/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecSetcover)
{ /*lint --e{715}*/
   SCIP* origprob;
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(scip != NULL);
   assert(result != NULL);

   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   origprob = GCGmasterGetOrigprob(scip);
   assert(origprob != NULL);

   *result = SCIP_DIDNOTRUN;

   /* the heuristic is only executed on problems with many variables */
   if( SCIPgetNVars(scip) < heurdata->param_min_prob_size )
   {
      SCIPdebugMessage("not running set covering heuristic because instance is too small (only %i variables)\n", SCIPgetNVars(scip));
      return SCIP_OKAY;
   }

   /* only run on set covering problems */
   if( GCGisMasterSetCovering(origprob) == FALSE )
      return SCIP_OKAY;

   if( SCIPgetNVars(scip) == 0 )
      return SCIP_OKAY;

   SCIP_CALL( setCoveringHeuristic(scip, heur) );

   *result = SCIP_FOUNDSOL;
   return SCIP_OKAY;
}


/*
 * primal heuristic specific interface methods
 */

/** creates the setcover primal heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurSetcover(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEURDATA* heurdata = NULL;
   SCIP_HEUR* heur;

   heur = NULL;
   SCIP_CALL( SCIPallocMemory(scip, &heurdata) );

   /* include primal heuristic */

   SCIP_CALL( SCIPincludeHeurBasic(scip, &heur,
         HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP, heurExecSetcover, heurdata) );

   assert(heur != NULL);

   /* set non fundamental callbacks via setter functions */
   SCIP_CALL( SCIPsetHeurFree(scip, heur, heurFreeSetcover) );
   SCIP_CALL( SCIPsetHeurInit(scip, heur, heurInitSetcover) );
   SCIP_CALL( SCIPsetHeurExit(scip, heur, heurExitSetcover) );

   /* add setcover primal heuristic parameters */

   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/"HEUR_NAME"/tentcoresize",
      "how many columns are added to the tentative core for each row",
      &heurdata->param_core_tent_size, FALSE, DEF_CORE_TENT_SIZE, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/"HEUR_NAME"/lambdaadjustments",
      "should the step size during the subgradient phase be adjusted?",
      &heurdata->param_lambda_adjustments, FALSE, DEF_LAMBDA_ADJUSTMENTS, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/lambda",
      "default step size",
      &heurdata->param_lambda, FALSE, DEF_LAMBDA, 0.001, SCIP_REAL_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/"HEUR_NAME"/lambdap",
      "number of iterations after which lambda is adjusted",
      &heurdata->param_lambda_p, FALSE, DEF_LAMBDA_P, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/beta",
      "greatest gap between lower and upper bounds that is allowed",
      &heurdata->param_beta, FALSE, DEF_BETA, 1.0, SCIP_REAL_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/"HEUR_NAME"/sciter",
      "number of subgradient iterations after which the stopping criterion is tested again",
      &heurdata->param_stop_crit_iter, FALSE, DEF_STOP_CRIT_ITER, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/scdiff",
      "maximum absolute difference between lower and upper bound that is allowed for the stopping criterion",
      &heurdata->param_stop_crit_diff, FALSE, DEF_STOP_CRIT_DIFF, 0.0, SCIP_REAL_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/scratio",
      "minimum ratio between lower and upper bound that is allowed that is allowed for the stopping criterion",
      &heurdata->param_stop_crit_ratio, FALSE, DEF_STOP_CRIT_RATIO, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/pimin",
      "percentage of rows to be removed after column fixing",
      &heurdata->param_pi_min, FALSE, DEF_PI_MIN, 0.0, 1.0, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/"HEUR_NAME"/pialpha",
      "increase of pi when no improvement was made",
      &heurdata->param_pi_alpha, FALSE, DEF_PI_ALPHA, 0.0, SCIP_REAL_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/"HEUR_NAME"/tpmaxiter",
      "maximum iterations of the three-phase procedure",
      &heurdata->param_max_iter, FALSE, DEF_MAX_ITER, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/"HEUR_NAME"/tpmaxiternoimp",
      "maximum allowed number of iterations of the three-phase procedure without improvements",
      &heurdata->param_max_iter_no_imp, FALSE, DEF_MAX_ITER_NO_IMP, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/"HEUR_NAME"/threephasemaxiter",
      "maximum number of iterations inside three-phase",
      &heurdata->param_threephase_max_iter, FALSE, DEF_THREEPHASE_MAX_ITER, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/"HEUR_NAME"/greedymaxiter",
      "maximum number of iterations of the greedy algorithm",
      &heurdata->param_greedy_max_iter, FALSE, DEF_GREEDY_MAX_ITER, 1, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/"HEUR_NAME"/minprobsize",
         "minimum number of variables in the problem for the heuristic to start",
         &heurdata->param_min_prob_size, FALSE, DEF_MIN_PROB_SIZE, 1, INT_MAX, NULL, NULL) );

   return SCIP_OKAY;
}
