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

/**@file    bliss.cpp
 * @brief   helper functions for automorphism detection
 *
 * @author  Martin Bergner
 * @author  Daniel Peters
 * @author  Jonas Witt
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "bliss/graph.hh"
#include "pub_bliss.h"
#include "pub_gcgvar.h"
#include "scip_misc.h"
#include <cstring>

static
int getSign(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val                 /**< value */
   )
{
   if( SCIPisNegative(scip, val) )
      return -1;
   if( SCIPisPositive(scip, val) )
      return 1;
   else
      return 0;
}

/** compare two values of two scips */
static
int comp(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real             val1,               /**< value 1 to compare */
   SCIP_Real             val2,               /**< value 2 to compare */
   SCIP_Bool             onlysign            /**< use sign of values instead of values? */
   )
{
   SCIP_Real compval1;
   SCIP_Real compval2;

   if( onlysign )
   {
      compval1 = getSign(scip, val1);
      compval2 = getSign(scip, val2);
   }
   else
   {
      compval1 = val1;
      compval2 = val2;
   }


   if( SCIPisLT(scip, compval1, compval2) )
      return -1;
   if( SCIPisGT(scip, compval1, compval2) )
      return 1;
   else
      return 0;
}

/** compare two constraints of two scips */
static
int comp(
   SCIP*                 scip,               /**< SCIP data structure */
   AUT_CONS*             cons1,              /**< constraint 1 to compare */
   AUT_CONS*             cons2,              /**< constraint 2 to compare */
   SCIP_Bool             onlysign            /**< use sign of values instead of values? */
   )
{
   if( comp(scip, GCGconsGetRhs(scip, cons1->getCons()), GCGconsGetRhs(scip, cons2->getCons()), onlysign) != 0 )
      return comp(scip, GCGconsGetRhs(scip, cons1->getCons()), GCGconsGetRhs(scip, cons2->getCons()), onlysign);
   assert(SCIPisEQ(scip, GCGconsGetRhs(scip, cons1->getCons()), GCGconsGetRhs(scip, cons2->getCons())) || onlysign);

   if( comp(scip, GCGconsGetLhs(scip, cons1->getCons()), GCGconsGetLhs(scip, cons2->getCons()), onlysign) != 0 )
      return comp(scip, GCGconsGetLhs(scip, cons1->getCons()), GCGconsGetLhs(scip, cons2->getCons()), onlysign);
   assert(SCIPisEQ(scip, GCGconsGetLhs(scip, cons1->getCons()), GCGconsGetLhs(scip, cons2->getCons())) || onlysign);

   if( comp(scip, GCGconsGetNVars(scip, cons1->getCons()), GCGconsGetNVars(scip, cons2->getCons()), FALSE) != 0 )
      return comp(scip, GCGconsGetNVars(scip, cons1->getCons()), GCGconsGetNVars(scip, cons2->getCons()), FALSE);
   assert(SCIPisEQ(scip, GCGconsGetNVars(scip, cons1->getCons()), GCGconsGetNVars(scip, cons2->getCons())));

   return strcmp(SCIPconshdlrGetName(SCIPconsGetHdlr(cons1->getCons())), SCIPconshdlrGetName(SCIPconsGetHdlr(cons2->getCons())));
}


/** compare two variables of two scips */
static
int comp(
   SCIP*                 scip,               /**< SCIP data structure */
   AUT_VAR*              var1,               /**< variable 1 to compare */
   AUT_VAR*              var2,               /**< variable 2 to compare */
   SCIP_Bool             onlysign            /**< use sign of values instead of values? */
   )
{
   SCIP_VAR* origvar1;
   SCIP_VAR* origvar2;

   if( GCGvarIsPricing(var1->getVar()) )
         origvar1 = GCGpricingVarGetOriginalVar(var1->getVar());
      else
         origvar1 = var1->getVar();

   if( GCGvarIsPricing(var2->getVar()) )
         origvar2 = GCGpricingVarGetOriginalVar(var2->getVar());
      else
         origvar2 = var2->getVar();

   if( comp(scip, SCIPvarGetUbGlobal(origvar1), SCIPvarGetUbGlobal(origvar2), onlysign) != 0 )
      return comp(scip, SCIPvarGetUbGlobal(origvar1), SCIPvarGetUbGlobal(origvar2), onlysign);
   assert(SCIPisEQ(scip, SCIPvarGetUbGlobal(origvar1), SCIPvarGetUbGlobal(origvar2)) || onlysign);

   if( comp(scip, SCIPvarGetLbGlobal(origvar1), SCIPvarGetLbGlobal(origvar2), onlysign) != 0 )
      return comp(scip, SCIPvarGetLbGlobal(origvar1), SCIPvarGetLbGlobal(origvar2), onlysign);
   assert(SCIPisEQ(scip, SCIPvarGetLbGlobal(origvar1), SCIPvarGetLbGlobal(origvar2)) || onlysign);

   if( comp(scip, SCIPvarGetObj((origvar1)), SCIPvarGetObj(origvar2), onlysign) != 0 )
      return comp(scip, SCIPvarGetObj(origvar1), SCIPvarGetObj(origvar2), onlysign);
   assert(SCIPisEQ(scip, SCIPvarGetObj(origvar1), SCIPvarGetObj(origvar2)) || onlysign);

   if( SCIPvarGetType(origvar1) < SCIPvarGetType(origvar2) )
      return -1;
   if( SCIPvarGetType(origvar1) > SCIPvarGetType(origvar2) )
      return 1;
   return 0;
}

/** SCIP interface method for sorting the constraints */
static
SCIP_DECL_SORTPTRCOMP(sortptrcons)
{
   AUT_CONS* aut1 = (AUT_CONS*) elem1;
   AUT_CONS* aut2 = (AUT_CONS*) elem2;
   return comp(aut1->getScip(), aut1, aut2, FALSE);
}

/** SCIP interface method for sorting the constraints */
static
SCIP_DECL_SORTPTRCOMP(sortptrconssign)
{
   AUT_CONS* aut1 = (AUT_CONS*) elem1;
   AUT_CONS* aut2 = (AUT_CONS*) elem2;
   return comp(aut1->getScip(), aut1, aut2, TRUE);
}

/** SCIP interface method for sorting the variables */
static
SCIP_DECL_SORTPTRCOMP(sortptrvar)
{
   AUT_VAR* aut1 = (AUT_VAR*) elem1;
   AUT_VAR* aut2 = (AUT_VAR*) elem2;
   return comp(aut1->getScip(), aut1, aut2, FALSE);
}

/** SCIP interface method for sorting the variables */
static
SCIP_DECL_SORTPTRCOMP(sortptrvarsign)
{
   AUT_VAR* aut1 = (AUT_VAR*) elem1;
   AUT_VAR* aut2 = (AUT_VAR*) elem2;
   return comp(aut1->getScip(), aut1, aut2, TRUE);
}

/** SCIP interface method for sorting the constraint coefficients*/
static
SCIP_DECL_SORTPTRCOMP(sortptrval)
{
   AUT_COEF* aut1 = (AUT_COEF*) elem1;
   AUT_COEF* aut2 = (AUT_COEF*) elem2;
   return comp(aut1->getScip(), aut1->getVal(), aut2->getVal(), FALSE); /*lint !e864*/
}

/** SCIP interface method for sorting the constraint coefficients*/
static
SCIP_DECL_SORTPTRCOMP(sortptrvalsign)
{
   AUT_COEF* aut1 = (AUT_COEF*) elem1;
   AUT_COEF* aut2 = (AUT_COEF*) elem2;
   return comp(aut1->getScip(), aut1->getVal(), aut2->getVal(), TRUE); /*lint !e864*/
}


/** default constructor */
struct_colorinformation::struct_colorinformation()
 : color(0), lenconssarray(0), lenvarsarray(0), lencoefsarray(0), alloccoefsarray(0),
ptrarraycoefs(NULL), ptrarrayvars(NULL), ptrarrayconss(NULL), onlysign(FALSE)
{

}

/** inserts a variable to the pointer array of colorinformation */
SCIP_RETCODE struct_colorinformation::insert(
   AUT_VAR*              svar,               /**< variable which is to add */
   SCIP_Bool*            added               /**< true if a var was added */
   )
{
   int pos;

   if( !onlysign )
   {
      if( !SCIPsortedvecFindPtr(ptrarrayvars, sortptrvar, svar, lenvarsarray, &pos) )
      {
         SCIPsortedvecInsertPtr(ptrarrayvars, sortptrvar, svar, &lenvarsarray, NULL);
         *added = TRUE;
         color++;
      }
      else
         *added = FALSE;
   }
   else
   {
      if( !SCIPsortedvecFindPtr(ptrarrayvars, sortptrvarsign, svar, lenvarsarray, &pos) )
      {
         SCIPsortedvecInsertPtr(ptrarrayvars, sortptrvarsign, svar, &lenvarsarray, NULL);
         *added = TRUE;
         color++;
      }
      else
         *added = FALSE;
   }


   return SCIP_OKAY;
}

/** inserts a constraint to the pointer array of colorinformation */
SCIP_RETCODE struct_colorinformation::insert(
   AUT_CONS*             scons,              /**< constraint which is to add */
   SCIP_Bool*            added               /**< true if a constraint was added */
   )
{
   int pos;

   if( !onlysign )
   {
      if( !SCIPsortedvecFindPtr(ptrarrayconss, sortptrcons, scons,
            lenconssarray, &pos) )
      {
         SCIPsortedvecInsertPtr(ptrarrayconss, sortptrcons, scons,
               &lenconssarray, NULL);
         *added = TRUE;
         color++;
      }
      else
         *added = FALSE;
   }
   else
   {
      if( !SCIPsortedvecFindPtr(ptrarrayconss, sortptrconssign, scons,
            lenconssarray, &pos) )
      {
         SCIPsortedvecInsertPtr(ptrarrayconss, sortptrconssign, scons,
               &lenconssarray, NULL);
         *added = TRUE;
         color++;
      }
      else
         *added = FALSE;
   }

   return SCIP_OKAY;
}

/** inserts a coefficient to the pointer array of colorinformation */
SCIP_RETCODE struct_colorinformation::insert(
   AUT_COEF*             scoef,              /**< coefficient which is to add */
   SCIP_Bool*            added               /**< true if a coefficient was added */
   )
{
   int pos;

   if( !onlysign )
   {
      if( !SCIPsortedvecFindPtr(ptrarraycoefs, sortptrval, scoef, lencoefsarray, &pos) )
      {
         if( alloccoefsarray == 0 || alloccoefsarray < lencoefsarray + 1 )
         {
            int size = SCIPcalcMemGrowSize(scoef->getScip(), alloccoefsarray+1);
            SCIP_CALL( SCIPreallocMemoryArray(scip, &ptrarraycoefs, size) );
            alloccoefsarray = size;
         }

         SCIPsortedvecInsertPtr(ptrarraycoefs, sortptrval, scoef, &lencoefsarray, NULL);
         *added = TRUE;
         color++;
      }
      else
         *added = FALSE;
   }
   else
   {
      if( !SCIPsortedvecFindPtr(ptrarraycoefs, sortptrvalsign, scoef, lencoefsarray, &pos) )
      {
         if( alloccoefsarray == 0 || alloccoefsarray < lencoefsarray + 1 )
         {
            int size = SCIPcalcMemGrowSize(scoef->getScip(), alloccoefsarray+1);
            SCIP_CALL( SCIPreallocMemoryArray(scip, &ptrarraycoefs, size) );
            alloccoefsarray = size;
         }

         SCIPsortedvecInsertPtr(ptrarraycoefs, sortptrvalsign, scoef, &lencoefsarray, NULL);
         *added = TRUE;
         color++;
      }
      else
         *added = FALSE;
   }

   return SCIP_OKAY;
}

int struct_colorinformation::get(
   AUT_VAR               svar                /**< variable whose pointer you want */
   )
{
   int pos;
   SCIP_Bool found;
   if( !onlysign )
      found = SCIPsortedvecFindPtr(ptrarrayvars, sortptrvar, &svar, lenvarsarray, &pos);
   else
      found = SCIPsortedvecFindPtr(ptrarrayvars, sortptrvarsign, &svar, lenvarsarray, &pos);
   return found ? pos : -1;
}

int struct_colorinformation::get(
   AUT_CONS              scons               /**< constraint whose pointer you want */
   )
{
   int pos;
   SCIP_Bool found;
   if( !onlysign )
      found = SCIPsortedvecFindPtr(ptrarrayconss, sortptrcons, &scons, lenconssarray, &pos);
   else
      found = SCIPsortedvecFindPtr(ptrarrayconss, sortptrconssign, &scons, lenconssarray, &pos);
   return found ? pos : -1;
}

int struct_colorinformation::get(
   AUT_COEF              scoef               /**< coefficient whose pointer you want */
   )
{
   int pos;
   SCIP_Bool found;
   if( !onlysign )
      found = SCIPsortedvecFindPtr(ptrarraycoefs, sortptrval, &scoef, lencoefsarray, &pos);
   else
      found = SCIPsortedvecFindPtr(ptrarraycoefs, sortptrvalsign, &scoef, lencoefsarray, &pos);
   return found ? pos : -1;
}

SCIP_RETCODE struct_colorinformation::setOnlySign(
   SCIP_Bool            onlysign_            /**< new value for onlysign bool */
   )
{
   onlysign = onlysign_;

   return SCIP_OKAY;
}


SCIP_Bool struct_colorinformation::getOnlySign()
{
   return onlysign;
}


int struct_colorinformation::getLenVar()
{
   return lenvarsarray;
}

int struct_colorinformation::getLenCons()
{
   return lenconssarray;
}

SCIP_CONS* struct_cons::getCons()
{
   return cons;
}

SCIP* struct_cons::getScip()
{
   return scip;
}

SCIP_VAR* struct_var::getVar ()
{
   return var;
}

SCIP* struct_var::getScip()
{
   return scip;
}

SCIP* struct_coef::getScip()
{
   return scip;
}

SCIP_Real struct_coef::getVal()
{
   return val;
}

/** constructor of the variable struct */
struct_var::struct_var(
   SCIP*                 scip_,              /**< SCIP data structure */
   SCIP_VAR*             svar                /**< SCIP variable */
   )
{
   scip = scip_;
   var = svar;
}

/** constructor of the constraint struct */
struct_cons::struct_cons(
   SCIP*                 scip_,              /**< SCIP data structure */
   SCIP_CONS*            scons               /**< SCIP constraint */
   )
{
   scip = scip_;
   cons = scons;
}

/** constructor of the coefficient struct */
struct_coef::struct_coef(
   SCIP*                 scip_,              /**< SCIP data structure */
   SCIP_Real             val_                /**< SCIP value */
   )
{
   scip = scip_;
   val = val_;
}

/** returns bliss version */
extern "C"
void GCGgetBlissName(char* buffer, int len)
{
#ifdef BLISS_PATCH_PRESENT
   SCIPsnprintf(buffer, len, "bliss %sp", bliss::version);
#else
   SCIPsnprintf(buffer, len, "bliss %s", bliss::version);
#endif
}
