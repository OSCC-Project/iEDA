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

/**@file   struct_vardata.h
 * @ingroup DATASTRUCTURES
 * @brief  data structures for GCG variable data
 * @author Gerald Gamrath
 * @author Martin Bergner
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef GCG_STRUCT_VARDATA_H__
#define GCG_STRUCT_VARDATA_H__

#ifdef __cplusplus
extern "C" {
#endif

/** type of the variable */
enum GCG_Vartype
{
   GCG_VARTYPE_ORIGINAL = 0,                 /**< variable belongs to original problem */
   GCG_VARTYPE_PRICING = 1,                 /**< variable belongs to a pricing problem */
   GCG_VARTYPE_MASTER = 2                  /**< variable belongs to the master problem */
};
typedef enum GCG_Vartype GCG_VARTYPE;

/** additional data for linking variables */
struct GCG_LinkingVarData
{
   SCIP_VAR**            pricingvars;        /**< array of corresponding variables in the pricing programs (NULL if variable is not linking this block)*/
   SCIP_CONS**           linkconss;          /**< array of constraints in the master problem that ensure that all copies have the same values */
   int                   nblocks;            /**< number of blocks that this variable is linking */
};
typedef struct GCG_LinkingVarData GCG_LINKINGVARDATA;


/** data for original variables */
struct GCG_OrigVarData
{
   SCIP_VAR*             pricingvar;         /**< corresponding variable in the pricing program */
   SCIP_CONS**           masterconss;        /**< master constraints of the original program in which the variable has a nonzero entry */
   SCIP_Real*            coefs;              /**< coefficients in the linking constraints of the original program */
   int                   ncoefs;             /**< number of coefficients */
   SCIP_VAR**            mastervars;         /**< variables in the master problem that contain the variable */
   SCIP_Real*            mastervals;         /**< value of this variable in the master problem variables */
   int                   nmastervars;        /**< number of corresponding master variables */
   int                   maxmastervars;      /**< length of arrays mastervars and vals */
   GCG_LINKINGVARDATA*   linkingvardata;     /**< additional data for linking variables */
};
typedef struct GCG_OrigVarData GCG_ORIGVARDATA;

/** data for pricing variables */
struct GCG_PricingVarData
{
   SCIP_VAR**            origvars;           /**< corresponding variables in the original program */
   int                   norigvars;          /**< number of corresponding variables in the original program */
   int                   maxorigvars;        /**< length of origvars array */
};
typedef struct GCG_PricingVarData GCG_PRICINGVARDATA;


/** data for master variables */
struct GCG_MasterVarData
{
   int                   norigvars;          /**< number of variables in the original program corresponding to  the current variable */
   int                   maxorigvars;        /**< capacity of origvars and origvals */
   SCIP_VAR**            origvars;           /**< variables in the original program corresponding to the current variable */
   SCIP_Real*            origvals;           /**< this variable represents vals[i] times the variable origvars[i] in the
                                              *   original program */
   SCIP_Bool             isray;              /**< does this variable represent a ray or an extreme point? */
   SCIP_Bool             isartificial;       /**< is variable artificial? */
   SCIP_HASHMAP*         origvar2val;        /**< hash map that stores the fraction of original variables the master variable is contained in */
   int                   index;              /**< index of the master variable if stored in GCG's pricedvars array, -1 otherwise */
};
typedef struct GCG_MasterVarData GCG_MASTERVARDATA;

/** variable data structure */
struct SCIP_VarData
{
   union
   {
      GCG_ORIGVARDATA    origvardata;        /**< data for original variables */
      GCG_PRICINGVARDATA pricingvardata;     /**< data for pricing variables */
      GCG_MASTERVARDATA  mastervardata;      /**< data for variable of the master problem */
   } data;
   GCG_VARTYPE           vartype;            /**< type of variable */
   int                   blocknr;            /**< number of the block and pricing problem, the variable belongs to,
                                              *   or -1 if variable is directly transferred to the master problem,
                                              *   or -2 if variable is a linking variable */
   SCIP_Longint          creationnode;       /**< node where the variable is created */
   SCIP_Longint          rootredcostcall;    /**< pricing reduced cost call when the variable is created
                                              *   (-1 if variable was not created at the root node or was created in Farkas pricing) */
   SCIP_Real             creationtime;       /**< time when the variable is created */
   SCIP_Longint          iteration;          /**< iteration when the variable is created */
   SCIP_Real             gap;                /**< gap when the variable was created */
   SCIP_Real             redcost;            /**< reduced cost of the variable  */
};

#ifdef __cplusplus
}
#endif

#endif
