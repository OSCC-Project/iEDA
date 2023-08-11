%{
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: mmlparse2.y                                                   */
/*   Name....: MML Parser                                                    */
/*   Author..: Thorsten Koch                                                 */
/*   Copyright by Author, All rights reserved                                */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*
 * Copyright (C) 2001-2022 by Thorsten Koch <koch@zib.de>
 * 
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wunused-macros"
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"
#pragma clang diagnostic ignored "-Wunreachable-code"
#endif
   
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC   diagnostic ignored "-Wstrict-prototypes"
#endif
   
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
   
#include "zimpl/lint.h"
#include "zimpl/attribute.h"
#include "zimpl/mshell.h"
#include "zimpl/ratlptypes.h"
#include "zimpl/numb.h"
#include "zimpl/elem.h"
#include "zimpl/tuple.h"
#include "zimpl/mme.h"
#include "zimpl/set.h"
#include "zimpl/symbol.h"
#include "zimpl/entry.h"
#include "zimpl/idxset.h"
#include "zimpl/rdefpar.h"
#include "zimpl/bound.h"
#include "zimpl/define.h"
#include "zimpl/mono.h"
#include "zimpl/term.h"
#include "zimpl/list.h"
#include "zimpl/stmt.h"
#include "zimpl/local.h"
#include "zimpl/code.h"
#include "zimpl/inst.h"
        
#define YYERROR_VERBOSE 1

/* the function is actually getting a YYSTYPE* as argument, but the
 * type isn't available here, so it is decalred to accept any number of
 * arguments, i.e. yylex() and not yylex(void).
 */
extern int yylex();

/*lint -sem(yyerror, 1p, r_no) */ 
extern void yyerror(const char* s) is_NORETURN;
 
%}
%define api.pure
%expect 12

%union
{
   unsigned int bits;
   Numb*        numb;
   const char*  strg;
   const char*  name;
   Symbol*      sym;
   Define*      def;
   CodeNode*    code;
};

%token DECLSET DECLPAR DECLVAR DECLMIN DECLMAX DECLSUB DECLSOS
%token DEFNUMB DEFSTRG DEFBOOL DEFSET PRINT CHECK
%token BINARY INTEGER REAL IMPLICIT
%token ASGN DO WITH IN TO UNTIL BY FORALL EXISTS
%token PRIORITY STARTVAL DEFAULT
%token CMP_LE CMP_GE CMP_EQ CMP_LT CMP_GT CMP_NE INFTY
%token AND OR XOR NOT
%token SUM MIN MAX ARGMIN ARGMAX PROD
%token IF THEN ELSE END
%token INTER UNION CROSS SYMDIFF WITHOUT PROJ
%token MOD DIV POW FAC
%token CARD ROUND FLOOR CEIL RANDOM ORD
%token ABS SGN LOG LN EXP SQRT SIN COS TAN ASIN ACOS ATAN POWER SGNPOW
%token READ AS SKIP USE COMMENT MATCH
%token SUBSETS INDEXSET POWERSET
%token VIF VABS
%token TYPE1 TYPE2
%token LENGTH SUBSTR
%token <sym> NUMBSYM STRGSYM VARSYM SETSYM
%token <def> NUMBDEF STRGDEF BOOLDEF SETDEF DEFNAME
%token <name> NAME
%token <strg> STRG
%token <numb> NUMB
%token <bits> SCALE SEPARATE CHECKONLY INDICATOR QUBO PENALTY1 PENALTY2 PENALTY3 PENALTY4 PENALTY5 PENALTY6

%type <code> stmt decl_set decl_par decl_var decl_obj decl_sub decl_sos
%type <code> def_numb def_strg def_bool def_set
%type <code> exec_do command
%type <code> constraint_list constraint vbool
%type <code> cexpr cexpr_list cfactor cproduct cexpo cval
%type <code> symidx tuple tuple_list sexpr sunion sproduct sval lexpr read read_par
%type <code> idxset pure_idxset vval vproduct vfactor vexpo vexpr name_list
%type <code> cexpr_entry cexpr_entry_list set_entry set_entry_list
%type <code> par_singleton par_default
%type <code> var_type con_type lower upper priority startval 
%type <code> matrix_body matrix_head
%type <code> soset sos_type
%type <bits> con_attr con_attr_list

%right ASGN
/*%left  ','*/
%left  OR XOR 
/*%left  EXISTS*/
%left  AND
/*%left  CMP_EQ CMP_NE CMP_LE CMP_LT CMP_GE CMP_GT */
/*%left  IN*/
%left  NOT
%left  UNION WITHOUT SYMDIFF
/*left  INTER CROSS*/
/*%left  SUM */
%left  '+' '-' 
/*%left  MIN MAX*/
/*%left  PROD*/
/*%left  '*' '/' MOD DIV */
%left '*'
/*%right POW*/
/*%left  FAC*/
%%
stmt
   : decl_set   { code_set_root($1); }
   | decl_par   { code_set_root($1); }
   | decl_var   { code_set_root($1); }
   | decl_obj   { code_set_root($1); }
   | decl_sub   { code_set_root($1); }
   | decl_sos   { code_set_root($1); }
   | def_numb   { code_set_root($1); }
   | def_strg   { code_set_root($1); }
   | def_bool   { code_set_root($1); }
   | def_set    { code_set_root($1); }
   | exec_do    { code_set_root($1); }
   ;

/* ----------------------------------------------------------------------------
 * --- Set Declaration
 * ----------------------------------------------------------------------------
 */
decl_set
   : DECLSET NAME ASGN sexpr ';' {
         $$ = code_new_inst(i_newsym_set1, 3,
            code_new_name($2),                                       /* Name */
            code_new_inst(i_idxset_pseudo_new, 1,               /* index set */
               code_new_inst(i_bool_true, 0)),              
            $4);                                              /* initial set */
      }
   | DECLSET NAME '[' idxset ']' ASGN sexpr ';' {
         $$ = code_new_inst(i_newsym_set1, 3,
            code_new_name($2),                                       /* Name */
            $4,                                                 /* index set */
            $7);                                                      /* set */
      }
   | DECLSET NAME '[' idxset ']' ASGN set_entry_list ';' {
         $$ = code_new_inst(i_newsym_set2, 3,
            code_new_name($2),                                       /* Name */
            $4,                                                 /* index set */
            $7);                                   /* initial set_entry_list */
      }
   | DECLSET NAME '[' ']' ASGN set_entry_list ';' {
         $$ = code_new_inst(i_newsym_set2, 3,
            code_new_name($2),                                       /* Name */
            code_new_inst(i_idxset_pseudo_new, 1,               /* index set */
               code_new_inst(i_bool_true, 0)),              
            $6);                                   /* initial set_entry_list */
      }
   ;

set_entry_list
   : set_entry             { $$ = code_new_inst(i_entry_list_new, 1, $1); }
   | set_entry_list ',' set_entry  {
         $$ = code_new_inst(i_entry_list_add, 2, $1, $3);
      }
   | SUBSETS '(' sexpr ',' cexpr ')' {
         $$ = code_new_inst(i_entry_list_subsets, 3, $3, $5, code_new_numb(numb_new_integer(-1)));
      }
   | SUBSETS '(' sexpr ',' cexpr ',' cexpr ')' {
         $$ = code_new_inst(i_entry_list_subsets, 3, $3, $5, $7);
      }
   | POWERSET '(' sexpr ')' {
         $$ = code_new_inst(i_entry_list_powerset, 1, $3);
      }
   ;

set_entry
   : tuple sexpr           { $$ = code_new_inst(i_entry, 2, $1, $2); }
   ;


/* ----------------------------------------------------------------------------
 * --- Define Declaration
 * ----------------------------------------------------------------------------
 */
def_numb
   : DEFNUMB DEFNAME '(' name_list ')' ASGN cexpr ';' {
         $$ = code_new_inst(i_newdef, 3,
            code_new_define($2),
            code_new_inst(i_tuple_new, 1, $4),
            $7);
      }
   ;

def_strg
   : DEFSTRG DEFNAME '(' name_list ')' ASGN cexpr ';' {
         $$ = code_new_inst(i_newdef, 3,
            code_new_define($2),
            code_new_inst(i_tuple_new, 1, $4),
            $7);
      }
   ;

def_bool
   : DEFBOOL DEFNAME '(' name_list ')' ASGN lexpr ';' {
         $$ = code_new_inst(i_newdef, 3,
            code_new_define($2),
            code_new_inst(i_tuple_new, 1, $4),
            $7);
      }
   ;

def_set
   : DEFSET DEFNAME '(' name_list ')' ASGN sexpr ';' {
         $$ = code_new_inst(i_newdef, 3,
            code_new_define($2),
            code_new_inst(i_tuple_new, 1, $4),
            $7);
      }
   ;

name_list
   : NAME {
         $$ = code_new_inst(i_elem_list_new, 1, code_new_name($1));
      }
   | name_list ',' NAME {
         $$ = code_new_inst(i_elem_list_add, 2, $1, code_new_name($3));
      }
   ;

/* ----------------------------------------------------------------------------
 * --- Param Declaration
 * ----------------------------------------------------------------------------
 */
decl_par
   : DECLPAR NAME '[' idxset ']' ASGN cexpr_entry_list par_default ';' {
         $$ = code_new_inst(i_newsym_para1, 4, code_new_name($2), $4, $7, $8);
      }
   | DECLPAR NAME '[' idxset ']' ASGN cexpr ';' {
         $$ = code_new_inst(i_newsym_para2, 4, code_new_name($2), $4, $7, code_new_inst(i_nop, 0));
      }
   | DECLPAR NAME '[' idxset ']' ASGN DEFAULT cexpr ';' {
         $$ = code_new_inst(i_newsym_para2, 4, code_new_name($2), $4, $8, code_new_inst(i_nop, 0));
      }
   | DECLPAR NAME ASGN par_singleton ';' {
         $$ = code_new_inst(i_newsym_para1, 4,
            code_new_name($2),
            code_new_inst(i_idxset_pseudo_new, 1, code_new_inst(i_bool_true, 0)),
            $4,
            code_new_inst(i_nop, 0));
      }
   | DECLPAR { $$ = code_new_inst(i_nop, 0); } /* Happens when a parameter is redeclared */
   ;

par_singleton
   : cexpr_entry_list { $$ = $1; }
   | cexpr            {
         $$ = code_new_inst(i_entry_list_new, 1,
            code_new_inst(i_entry, 2, code_new_inst(i_tuple_empty, 0), $1));
      }
   ;

par_default
   : /* empty */    { $$ = code_new_inst(i_nop, 0); }
   | DEFAULT cexpr  { $$ = code_new_inst(i_entry, 2, code_new_inst(i_tuple_empty, 0), $2); }
   ;

/* ----------------------------------------------------------------------------
 * --- Var Declaration
 * ----------------------------------------------------------------------------
 */
decl_var
   : DECLVAR NAME '[' idxset ']' var_type lower upper ';' {
         $$ = code_new_inst(i_newsym_var, 7,
            code_new_name($2),
            $4, $6, $7, $8,
            code_new_numb(numb_copy(numb_unknown())),
            code_new_numb(numb_copy(numb_unknown())));
      }
   | DECLVAR NAME var_type lower upper ';' {
         $$ = code_new_inst(i_newsym_var, 7,
            code_new_name($2),
            code_new_inst(i_idxset_pseudo_new, 1,
               code_new_inst(i_bool_true, 0)),              
            $3, $4, $5,
            code_new_numb(numb_copy(numb_unknown())),
            code_new_numb(numb_copy(numb_unknown())));
      }
   | DECLVAR NAME '[' idxset ']' IMPLICIT BINARY ';' {
         $$ = code_new_inst(i_newsym_var, 7,
            code_new_name($2),
            $4,
            code_new_varclass(VAR_IMP),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(0))),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(1))),
            code_new_numb(numb_copy(numb_unknown())),
            code_new_numb(numb_copy(numb_unknown())));
      }
   | DECLVAR NAME IMPLICIT BINARY ';' {
         $$ = code_new_inst(i_newsym_var, 7,
            code_new_name($2),
            code_new_inst(i_idxset_pseudo_new, 1,
               code_new_inst(i_bool_true, 0)),              
            code_new_varclass(VAR_IMP),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(0))),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(1))),
            code_new_numb(numb_copy(numb_unknown())),
            code_new_numb(numb_copy(numb_unknown())));
      }
   | DECLVAR NAME '[' idxset ']' BINARY priority startval ';' {
         $$ = code_new_inst(i_newsym_var, 7,
            code_new_name($2),
            $4,
            code_new_varclass(VAR_INT),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(0))),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(1))),
            $7, $8);
      }
   | DECLVAR NAME BINARY priority startval ';' {
         $$ = code_new_inst(i_newsym_var, 7,
            code_new_name($2),
            code_new_inst(i_idxset_pseudo_new, 1,
               code_new_inst(i_bool_true, 0)),              
            code_new_varclass(VAR_INT),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(0))),
            code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(1))),
            $4, $5);
      }
   | DECLVAR NAME '[' idxset ']' INTEGER lower upper priority startval ';' {
         $$ = code_new_inst(i_newsym_var, 7,
            code_new_name($2), $4, code_new_varclass(VAR_INT), $7, $8, $9, $10);
      }
   | DECLVAR NAME INTEGER lower upper priority startval ';' {
         $$ = code_new_inst(i_newsym_var, 7,
            code_new_name($2),
            code_new_inst(i_idxset_pseudo_new, 1,
               code_new_inst(i_bool_true, 0)),              
            code_new_varclass(VAR_INT), $4, $5, $6, $7);
      }
   ;

var_type
   : /* empty */      { $$ = code_new_varclass(VAR_CON); }
   | REAL             { $$ = code_new_varclass(VAR_CON); }
   | IMPLICIT INTEGER { $$ = code_new_varclass(VAR_IMP); }
   ;

lower
   : /* empty */      {
         $$ = code_new_inst(i_bound_new, 1, code_new_numb(numb_new_integer(0)));
      }
   | CMP_GE cexpr      { $$ = code_new_inst(i_bound_new, 1, $2); }
   | CMP_GE '-' INFTY  { $$ = code_new_bound(BOUND_MINUS_INFTY); }
   | CMP_GE IF lexpr THEN cexpr ELSE '-' INFTY END {
         $$ = code_new_inst(i_expr_if_else, 3, $3,
            code_new_inst(i_bound_new, 1, $5),
            code_new_bound(BOUND_MINUS_INFTY));
      }
   | CMP_GE IF lexpr THEN '-' INFTY ELSE cexpr END {
         $$ = code_new_inst(i_expr_if_else, 3, $3,
            code_new_bound(BOUND_MINUS_INFTY),
            code_new_inst(i_bound_new, 1, $8));
      }
   ;

upper
   : /* empty */       { $$ = code_new_bound(BOUND_INFTY); }
   | CMP_LE cexpr      { $$ = code_new_inst(i_bound_new, 1, $2); }
   | CMP_LE INFTY      { $$ = code_new_bound(BOUND_INFTY); }
   | CMP_LE IF lexpr THEN cexpr ELSE INFTY END {
         $$ = code_new_inst(i_expr_if_else, 3, $3,
            code_new_inst(i_bound_new, 1, $5),
            code_new_bound(BOUND_INFTY));
      }
   | CMP_LE IF lexpr THEN INFTY ELSE cexpr END {
         $$ = code_new_inst(i_expr_if_else, 3, $3,
            code_new_bound(BOUND_INFTY),
            code_new_inst(i_bound_new, 1, $7));
      }
   ;

priority
   : /* empty */       { $$ = code_new_numb(numb_new_integer(0)); }
   | PRIORITY cexpr    { $$ = $2; }
   ;

startval
   : /* empty */       { $$ = code_new_numb(numb_copy(numb_unknown())); }
   | STARTVAL cexpr    { $$ = $2; }
   ;

/* ----------------------------------------------------------------------------
 * --- DATA 
 * ----------------------------------------------------------------------------
 */
cexpr_entry_list
   : cexpr_entry              { $$ = code_new_inst(i_entry_list_new, 1, $1); }
   | cexpr_entry_list ',' cexpr_entry  {
         $$ = code_new_inst(i_entry_list_add, 2, $1, $3);
      }
   | read                    { $$ = code_new_inst(i_read, 1, $1); }
   | matrix_head matrix_body { $$ = code_new_inst(i_list_matrix, 2, $1, $2); }
   ;

cexpr_entry
   : tuple cexpr            { $$ = code_new_inst(i_entry, 2, $1, $2); }
   ;

matrix_head
   : WITH cexpr_list WITH { $$ = $2; }
   ;

matrix_body
   : matrix_head cexpr_list WITH {
         $$ = code_new_inst(i_matrix_list_new, 2, $1, $2);
      }
   | matrix_body matrix_head cexpr_list WITH {
         $$ = code_new_inst(i_matrix_list_add, 3, $1, $2, $3);
      }
   ;


/* ----------------------------------------------------------------------------
 * --- Objective Declaration
 * ----------------------------------------------------------------------------
 */

decl_obj
   : DECLMIN NAME DO vexpr ';' {
         $$ = code_new_inst(i_object_min, 2, code_new_name($2), $4);
      }
   | DECLMAX NAME DO vexpr ';' {
         $$ = code_new_inst(i_object_max, 2, code_new_name($2), $4);
      }
   ;

/* ----------------------------------------------------------------------------
 * --- Subto Declaration
 * ----------------------------------------------------------------------------
 */
decl_sub
   : DECLSUB NAME DO constraint_list ';' {
        $$ = code_new_inst(i_subto, 2, code_new_name($2), $4);
     }
   ;

constraint_list
   : constraint {
        $$ = code_new_inst(i_constraint_list, 2, $1, code_new_inst(i_nop, 0));
     }
   | constraint_list AND constraint {
        $$ = code_new_inst(i_constraint_list, 2, $1, $3);
     }
   | FORALL idxset DO constraint_list {
        $$ = code_new_inst(i_constraint_list, 2, 
           code_new_inst(i_forall, 2, $2, $4),
           code_new_inst(i_nop, 0));
     }
   | IF lexpr THEN constraint_list END {
        $$ = code_new_inst(i_constraint_list, 2, 
           code_new_inst(i_expr_if_else, 3, $2, $4, code_new_inst(i_nop, 0)),
           code_new_inst(i_nop, 0));
      }
   | IF lexpr THEN constraint_list ELSE constraint_list END {
        $$ = code_new_inst(i_constraint_list, 2, 
           code_new_inst(i_expr_if_else, 3, $2, $4, $6),
           code_new_inst(i_nop, 0));
      }
   ;

constraint
   : vexpr con_type vexpr con_attr_list {
        $$ = code_new_inst(i_constraint, 4, $1, $2, $3, code_new_bits($4));
     }
   | vexpr con_type cexpr con_attr_list {
        $$ = code_new_inst(i_constraint, 4, $1, $2,
           code_new_inst(i_term_expr, 1, $3),
           code_new_bits($4));
     }
   | cexpr con_type vexpr con_attr_list {
        $$ = code_new_inst(i_constraint, 4,
           code_new_inst(i_term_expr, 1, $1),
           $2, $3, code_new_bits($4));
     }
   | cexpr con_type cexpr con_attr_list { 
        $$ = code_new_inst(i_constraint, 4,
           code_new_inst(i_term_expr, 1, $1),
           $2,
           code_new_inst(i_term_expr, 1, $3),
           code_new_bits($4));
     }
   | cexpr con_type vexpr CMP_LE cexpr con_attr_list {
        $$ = code_new_inst(i_rangeconst, 6, $1, $3, $5, $2,
           code_new_contype(CON_RHS), code_new_bits($6)); 
     }
   | cexpr con_type cexpr CMP_LE cexpr con_attr_list {
        $$ = code_new_inst(i_rangeconst, 6, $1,
           code_new_inst(i_term_expr, 1, $3), $5, $2,
           code_new_contype(CON_RHS), code_new_bits($6)); 
     }
   | cexpr con_type vexpr CMP_GE cexpr con_attr_list {
        $$ = code_new_inst(i_rangeconst, 6, $5, $3, $1, $2,
           code_new_contype(CON_LHS), code_new_bits($6)); 
     }
   | cexpr con_type cexpr CMP_GE cexpr con_attr_list {
        $$ = code_new_inst(i_rangeconst, 6, $5,
           code_new_inst(i_term_expr, 1, $3),
           $1, $2,
           code_new_contype(CON_LHS), code_new_bits($6)); 
     }
   | VIF vbool THEN vexpr con_type vexpr ELSE vexpr con_type vexpr END con_attr_list {
         $$ = code_new_inst(i_vif_else, 8, $2, $4, $5, $6, $8, $9, $10, code_new_bits($12));
      }
   | VIF vbool THEN cexpr con_type vexpr ELSE vexpr con_type vexpr END con_attr_list {
         $$ = code_new_inst(i_vif_else, 8, $2,
            code_new_inst(i_term_expr, 1, $4),
            $5, $6, $8, $9, $10, code_new_bits($12));
      }
   | VIF vbool THEN vexpr con_type cexpr ELSE vexpr con_type vexpr END con_attr_list { 
         $$ = code_new_inst(i_vif_else, 8, $2, $4, $5,
            code_new_inst(i_term_expr, 1, $6),
            $8, $9, $10, code_new_bits($12));
      }
   | VIF vbool THEN vexpr con_type vexpr ELSE cexpr con_type vexpr END con_attr_list { 
         $$ = code_new_inst(i_vif_else, 8, $2, $4, $5, $6,
            code_new_inst(i_term_expr, 1, $8),
            $9, $10, code_new_bits($12));
      }
   | VIF vbool THEN vexpr con_type vexpr ELSE vexpr con_type cexpr END con_attr_list { 
         $$ = code_new_inst(i_vif_else, 8, $2, $4, $5, $6, $8, $9,
            code_new_inst(i_term_expr, 1, $10), code_new_bits($12));
      }
   | VIF vbool THEN cexpr con_type cexpr ELSE vexpr con_type vexpr END con_attr_list { /* ??? This is an error */
         $$ = code_new_inst(i_vif_else, 8, $2,
            code_new_inst(i_term_expr, 1, $4),
            $5,
            code_new_inst(i_term_expr, 1, $6),
            $8, $9, $10, code_new_bits($12));
      }
   | VIF vbool THEN cexpr con_type vexpr ELSE cexpr con_type vexpr END con_attr_list { 
         $$ = code_new_inst(i_vif_else, 8, $2,
            code_new_inst(i_term_expr, 1, $4),
            $5, $6,
            code_new_inst(i_term_expr, 1, $8),
            $9, $10, code_new_bits($12));
      }
   | VIF vbool THEN cexpr con_type vexpr ELSE vexpr con_type cexpr END con_attr_list { 
         $$ = code_new_inst(i_vif_else, 8, $2,
            code_new_inst(i_term_expr, 1, $4),
            $5, $6, $8, $9,
            code_new_inst(i_term_expr, 1, $10), code_new_bits($12));
      }
   | VIF vbool THEN vexpr con_type cexpr ELSE cexpr con_type vexpr END con_attr_list { 
         $$ = code_new_inst(i_vif_else, 8, $2, $4, $5,
            code_new_inst(i_term_expr, 1, $6),
            code_new_inst(i_term_expr, 1, $8),
            $9, $10, code_new_bits($12));
      }
   | VIF vbool THEN vexpr con_type cexpr ELSE vexpr con_type cexpr END con_attr_list { 
         $$ = code_new_inst(i_vif_else, 8, $2, $4, $5,
            code_new_inst(i_term_expr, 1, $6),
            $8, $9,
            code_new_inst(i_term_expr, 1, $10), code_new_bits($12));
      }
   | VIF vbool THEN vexpr con_type vexpr ELSE cexpr con_type cexpr END con_attr_list { /* ??? This is an error */
         $$ = code_new_inst(i_vif_else, 8, $2, $4, $5, $6,
            code_new_inst(i_term_expr, 1, $8), $9,
            code_new_inst(i_term_expr, 1, $10), code_new_bits($12));
      }
   | VIF vbool THEN cexpr con_type cexpr ELSE cexpr con_type vexpr END con_attr_list { /* ??? This is an error */
         $$ = code_new_inst(i_vif_else, 8, $2,
            code_new_inst(i_term_expr, 1, $4),
            $5,
            code_new_inst(i_term_expr, 1, $6),
            code_new_inst(i_term_expr, 1, $8),
            $9, $10, code_new_bits($12));
      }
   | VIF vbool THEN cexpr con_type cexpr ELSE vexpr con_type cexpr END con_attr_list { /* ??? This is an error */
         $$ = code_new_inst(i_vif_else, 8, $2,
            code_new_inst(i_term_expr, 1, $4),
            $5,
            code_new_inst(i_term_expr, 1, $6),
            $8, $9,
            code_new_inst(i_term_expr, 1, $10), 
            code_new_bits($12));
      }
   | VIF vbool THEN cexpr con_type vexpr ELSE cexpr con_type cexpr END con_attr_list { /* ??? This is an error */
         $$ = code_new_inst(i_vif_else, 8, $2,
            code_new_inst(i_term_expr, 1, $4),
            $5, $6,
            code_new_inst(i_term_expr, 1, $8),
            $9,
            code_new_inst(i_term_expr, 1, $10), 
            code_new_bits($12));
      }
   | VIF vbool THEN vexpr con_type cexpr ELSE cexpr con_type cexpr END con_attr_list { /* ??? This is an error */
         $$ = code_new_inst(i_vif_else, 8, $2, $4, $5,
            code_new_inst(i_term_expr, 1, $6),
            code_new_inst(i_term_expr, 1, $8),
            $9,
            code_new_inst(i_term_expr, 1, $10), 
            code_new_bits($12));
      }
   | VIF vbool THEN cexpr con_type cexpr ELSE cexpr con_type cexpr END con_attr_list { /* ??? This is an error */
         $$ = code_new_inst(i_vif_else, 8, $2,
            code_new_inst(i_term_expr, 1, $4),
            $5,
            code_new_inst(i_term_expr, 1, $6),
            code_new_inst(i_term_expr, 1, $8),
            $9,
            code_new_inst(i_term_expr, 1, $10), 
            code_new_bits($12));
      }

   | VIF vbool THEN vexpr con_type vexpr END con_attr_list {
         $$ = code_new_inst(i_vif, 5, $2, $4, $5, $6, code_new_bits($8));
      }
   | VIF vbool THEN cexpr con_type vexpr END con_attr_list {
         $$ = code_new_inst(i_vif, 5, $2, 
            code_new_inst(i_term_expr, 1, $4), $5, $6, code_new_bits($8));
      }
   | VIF vbool THEN vexpr con_type cexpr END con_attr_list {
         $$ = code_new_inst(i_vif, 5, $2, 
            $4, $5, code_new_inst(i_term_expr, 1, $6), 
            code_new_bits($8));
      }
   | VIF vbool THEN cexpr con_type cexpr END con_attr_list { /* ??? This is an error */
         $$ = code_new_inst(i_vif, 5, $2,
            code_new_inst(i_term_expr, 1, $4), $5, 
            code_new_inst(i_term_expr, 1, $6), code_new_bits($8));
      }
   ;

vbool
   : vexpr CMP_NE vexpr { $$ = code_new_inst(i_vbool_ne, 2, $1, $3); }
   | cexpr CMP_NE vexpr {
         $$ = code_new_inst(i_vbool_ne, 2, code_new_inst(i_term_expr, 1, $1), $3);
      }
   | vexpr CMP_NE cexpr {
         $$ = code_new_inst(i_vbool_ne, 2, $1, code_new_inst(i_term_expr, 1, $3));
      }
   | vexpr CMP_EQ vexpr { $$ = code_new_inst(i_vbool_eq, 2, $1, $3); }
   | cexpr CMP_EQ vexpr {
         $$ = code_new_inst(i_vbool_eq, 2, code_new_inst(i_term_expr, 1, $1), $3);
      }
   | vexpr CMP_EQ cexpr {
         $$ = code_new_inst(i_vbool_eq, 2, $1, code_new_inst(i_term_expr, 1, $3));
      }
   | vexpr CMP_LE vexpr { $$ = code_new_inst(i_vbool_le, 2, $1, $3); }
   | cexpr CMP_LE vexpr {
         $$ = code_new_inst(i_vbool_le, 2, code_new_inst(i_term_expr, 1, $1), $3);
      }
   | vexpr CMP_LE cexpr {
         $$ = code_new_inst(i_vbool_le, 2, $1, code_new_inst(i_term_expr, 1, $3));
      }
   | vexpr CMP_GE vexpr { $$ = code_new_inst(i_vbool_ge, 2, $1, $3); }
   | cexpr CMP_GE vexpr {
         $$ = code_new_inst(i_vbool_ge, 2, code_new_inst(i_term_expr, 1, $1), $3);
      }
   | vexpr CMP_GE cexpr {
         $$ = code_new_inst(i_vbool_ge, 2, $1, code_new_inst(i_term_expr, 1, $3));
      }
   | vexpr CMP_LT vexpr { $$ = code_new_inst(i_vbool_lt, 2, $1, $3); }
   | cexpr CMP_LT vexpr {
         $$ = code_new_inst(i_vbool_lt, 2, code_new_inst(i_term_expr, 1, $1), $3);
      }
   | vexpr CMP_LT cexpr {
         $$ = code_new_inst(i_vbool_lt, 2, $1, code_new_inst(i_term_expr, 1, $3));
      }
   | vexpr CMP_GT vexpr { $$ = code_new_inst(i_vbool_gt, 2, $1, $3); }
   | cexpr CMP_GT vexpr {
         $$ = code_new_inst(i_vbool_gt, 2, code_new_inst(i_term_expr, 1, $1), $3);
      }
   | vexpr CMP_GT cexpr {
         $$ = code_new_inst(i_vbool_gt, 2, $1, code_new_inst(i_term_expr, 1, $3));
      }
   | vbool AND vbool  { $$ = code_new_inst(i_vbool_and, 2, $1, $3); }
   | vbool OR  vbool  { $$ = code_new_inst(i_vbool_or,  2, $1, $3); }
   | vbool XOR vbool  { $$ = code_new_inst(i_vbool_xor, 2, $1, $3); }
   | NOT vbool        { $$ = code_new_inst(i_vbool_not, 1, $2); }
   | '(' vbool ')'    { $$ = $2; }
   ;

con_attr_list
   : /* empty */                { $$ = 0; }
   | con_attr_list ',' con_attr { $$ = $1 | $3; }
   ;

con_attr
   : SCALE     { $$ = LP_FLAG_CON_SCALE; }
   | SEPARATE  { $$ = LP_FLAG_CON_SEPAR; }
   | CHECKONLY { $$ = LP_FLAG_CON_CHECK; }
   | INDICATOR { $$ = LP_FLAG_CON_INDIC; }
   | QUBO      { $$ = LP_FLAG_CON_QUBO;  }
   | PENALTY1  { $$ = LP_FLAG_CON_PENALTY1; }
   | PENALTY2  { $$ = LP_FLAG_CON_PENALTY2; }
   | PENALTY3  { $$ = LP_FLAG_CON_PENALTY3; }
   | PENALTY4  { $$ = LP_FLAG_CON_PENALTY4; }
   | PENALTY5  { $$ = LP_FLAG_CON_PENALTY5; }
   | PENALTY6  { $$ = LP_FLAG_CON_PENALTY6; }
   ;

con_type
   : CMP_LE  { $$ = code_new_contype(CON_RHS); }
   | CMP_GE  { $$ = code_new_contype(CON_LHS); }
   | CMP_EQ  { $$ = code_new_contype(CON_EQUAL); }
   ;

vexpr
   : vproduct { $$ = $1; }
   | vexpr '+' vproduct { $$ = code_new_inst(i_term_add, 2, $1, $3); }
   | vexpr '-' vproduct { $$ = code_new_inst(i_term_sub, 2, $1, $3); }
   | vexpr '+' cproduct { $$ = code_new_inst(i_term_const, 2, $1, $3); } 
   | vexpr '-' cproduct {
         $$ = code_new_inst(i_term_sub, 2, $1, code_new_inst(i_term_expr, 1, $3));
      }
   | cexpr '+' vproduct { $$ = code_new_inst(i_term_const, 2, $3, $1); } 
   | cexpr '-' vproduct {
         $$ = code_new_inst(i_term_sub, 2,
            code_new_inst(i_term_expr, 1, $1),
            $3);
      }
   ;

vproduct
   : vfactor                 { $$ = $1; }
   | vproduct '*' cfactor    { $$ = code_new_inst(i_term_coeff, 2, $1, $3);  }
   | vproduct '/' cfactor    {
         $$ = code_new_inst(i_term_coeff, 2, $1,
            code_new_inst(i_expr_div, 2, code_new_numb(numb_new_integer(1)), $3));
      }
   | cproduct '*' vfactor    { $$ = code_new_inst(i_term_coeff, 2, $3, $1); }
   | vproduct '*' vfactor    { $$ = code_new_inst(i_term_mul, 2, $1, $3); } 
   ;

vfactor
   : vexpo
   | '+' vfactor              { $$ = $2; }
   | '-' vfactor              { 
         $$ = code_new_inst(i_term_coeff, 2, $2, code_new_numb(numb_new_integer(-1)));
      } 
   ;   

vexpo
  : vval                   { $$ = $1; }
  | vval POW cfactor       { 
         $$ = code_new_inst(i_term_power, 2, $1, $3);
      }
  | SUM idxset DO vproduct {
         $$ = code_new_inst(i_term_sum, 2, $2, $4);
      }
  ;

vval
   : VARSYM symidx          {
         $$ = code_new_inst(i_symbol_deref, 2, code_new_symbol($1), $2);
      } 
   | VABS '(' vexpr ')' { $$ = code_new_inst(i_vabs, 1, $3); }
   | SQRT '(' vexpr ')' { $$ = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(-2)), $3); }
   | LOG  '(' vexpr ')' { $$ = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(3)), $3); }
   | EXP  '(' vexpr ')' { $$ = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(4)), $3); }
   | LN   '(' vexpr ')' { $$ = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(5)), $3); }
   | SIN  '(' vexpr ')' { $$ = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(6)), $3); }
   | COS  '(' vexpr ')' { $$ = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(7)), $3); }
   | TAN  '(' vexpr ')' { $$ = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(8)), $3); }
   | ABS  '(' vexpr ')' { $$ = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(9)), $3); }
   | SGN  '(' vexpr ')' { $$ = code_new_inst(i_vexpr_fun, 2, code_new_numb(numb_new_integer(10)), $3); }
   | POWER  '(' vexpr ',' cexpr ')' {
         $$ = code_new_inst(i_vexpr_fun, 3, code_new_numb(numb_new_integer(11)), $3, $5);
      }
   | SGNPOW '(' vexpr ',' cexpr')' {
         $$ = code_new_inst(i_vexpr_fun, 3, code_new_numb(numb_new_integer(12)), $3, $5);
      }  
   | IF lexpr THEN vexpr ELSE vexpr END {
         $$ = code_new_inst(i_expr_if_else, 3, $2, $4, $6);
      }
   | '(' vexpr ')'      { $$ = $2; }
   ;   

/* ----------------------------------------------------------------------------
 * --- SOS Declaration
 * ----------------------------------------------------------------------------
 */
decl_sos
   : DECLSOS NAME DO soset ';' {
        $$ = code_new_inst(i_sos, 2, code_new_name($2), $4);
     }
   ;

soset
   : sos_type priority DO vexpr {
        $$ = code_new_inst(i_soset, 3, $4, $1, $2);
     }
   | FORALL idxset DO soset {
         $$ = code_new_inst(i_forall, 2, $2, $4);
      }
   ;

sos_type
   : /* empty */ { $$ = code_new_numb(numb_new_integer(1)); }
   | TYPE1       { $$ = code_new_numb(numb_new_integer(1)); }
   | TYPE2       { $$ = code_new_numb(numb_new_integer(2)); }
   ;

/* ----------------------------------------------------------------------------
 * --- Do Statement
 * ----------------------------------------------------------------------------
 */
exec_do
   : DO command ';' { $$ = $2; }
   ;

command
   : PRINT cexpr_list { $$ = code_new_inst(i_print, 1, $2); }
   | PRINT tuple      { $$ = code_new_inst(i_print, 1, $2); }
   | PRINT sexpr      { $$ = code_new_inst(i_print, 1, $2); }
   | PRINT lexpr      { $$ = code_new_inst(i_print, 1, $2); }
   | PRINT VARSYM     { $$ = code_new_inst(i_print, 1, code_new_symbol($2)); }
   | CHECK lexpr      { $$ = code_new_inst(i_check, 1, $2); }
   | FORALL idxset DO command {
        $$ = code_new_inst(i_forall, 2, $2, $4);
     }
   ;

/* ----------------------------------------------------------------------------
 * --- 
 * ----------------------------------------------------------------------------
 */
idxset 
   : pure_idxset { $$ = $1; }
   | sexpr {
         $$ = code_new_inst(i_idxset_new, 3,
            code_new_inst(i_tuple_empty, 0), $1, code_new_inst(i_bool_true, 0));
      }
   ;

pure_idxset
   : tuple IN sexpr WITH lexpr {
         $$ = code_new_inst(i_idxset_new, 3, $1, $3, $5);
      }
   | tuple IN sexpr {
         $$ = code_new_inst(i_idxset_new, 3, $1, $3, code_new_inst(i_bool_true, 0));
      }
   ;

sexpr
   : sunion
   | sexpr UNION sunion  { $$ = code_new_inst(i_set_union, 2, $1, $3); }
   | sexpr '+' sunion {
         $$ = code_new_inst(i_set_union, 2, $1, $3);
      }
   | sexpr SYMDIFF sunion  { $$ = code_new_inst(i_set_sdiff, 2, $1, $3); }
   | sexpr '-' sunion  {
         $$ = code_new_inst(i_set_minus, 2, $1, $3);
      }
   | sexpr WITHOUT sunion     { $$ = code_new_inst(i_set_minus, 2, $1, $3); }
   | sexpr INTER sunion       { $$ = code_new_inst(i_set_inter, 2, $1, $3); }
   ;

sunion : sproduct
   | UNION idxset DO sproduct { $$ = code_new_inst(i_set_union2, 2, $2, $4); }
   ;

sproduct
   : sval
   | sproduct CROSS sval %prec '*' { $$ = code_new_inst(i_set_cross, 2, $1, $3); }
   | sproduct '*' sval {
         $$ = code_new_inst(i_set_cross, 2, $1, $3);
      }
   | INTER idxset DO sval      { $$ = code_new_inst(i_set_inter2, 2, $2, $4); }
   ;

sval
   : SETSYM symidx  {
         $$ = code_new_inst(i_symbol_deref, 2, code_new_symbol($1), $2);
      }
   | SETDEF '(' cexpr_list ')' {
         $$ = code_new_inst(i_define_deref, 2,
            code_new_define($1),
            code_new_inst(i_tuple_new, 1, $3));
      }
   | '{' '}' { $$ = code_new_inst(i_set_empty, 1, code_new_size(0)); }
   | '{' cexpr TO cexpr BY cexpr '}' {
         $$ = code_new_inst(i_set_range2, 3, $2, $4, $6);
      }
   | '{' cexpr TO cexpr '}' {
         $$ = code_new_inst(i_set_range2, 3, $2, $4, code_new_numb(numb_new_integer(1)));
      }
   | '{' cexpr UNTIL cexpr BY cexpr '}' {
         $$ = code_new_inst(i_set_range, 3, $2, $4, $6);
      }
   | '{' cexpr UNTIL cexpr '}' {
         $$ = code_new_inst(i_set_range, 3, $2, $4, code_new_numb(numb_new_integer(1)));
      }
   | ARGMIN idxset DO cexpr %prec UNION {
         $$ = code_new_inst(i_set_argmin, 3, code_new_numb(numb_new_integer(1)), $2, $4);
      }
   | ARGMIN '(' cexpr ')' idxset DO cexpr %prec UNION {
         $$ = code_new_inst(i_set_argmin, 3, $3, $5, $7);
      }
   | ARGMAX idxset DO cexpr %prec UNION {
         $$ = code_new_inst(i_set_argmax, 3, code_new_numb(numb_new_integer(1)), $2, $4);
      }
   | ARGMAX '(' cexpr ')' idxset DO cexpr %prec UNION {
         $$ = code_new_inst(i_set_argmax, 3, $3, $5, $7);
      }
   | '(' sexpr ')'           { $$ = $2; }
   | '{' tuple_list '}'      { $$ = code_new_inst(i_set_new_tuple, 1, $2); }
   | '{' cexpr_list '}'      { $$ = code_new_inst(i_set_new_elem, 1, $2); }
   | '{' idxset '}'          { $$ = code_new_inst(i_set_idxset, 1, $2); }
   | '{' idxset DO cexpr '}' { $$ = code_new_inst(i_set_expr, 2, $2, $4); }
   | '{' idxset DO tuple '}' { $$ = code_new_inst(i_set_expr, 2, $2, $4); }
   | PROJ '(' sexpr ',' tuple ')' {
         $$ = code_new_inst(i_set_proj, 2, $3, $5);
       }
   | INDEXSET '(' SETSYM ')' {
          $$ = code_new_inst(i_set_indexset, 1, code_new_symbol($3));
       }
   | IF lexpr THEN sexpr ELSE sexpr END {
         $$ = code_new_inst(i_expr_if_else, 3, $2, $4, $6);
      }
   ;

read
   : READ cexpr AS cexpr { $$ = code_new_inst(i_read_new, 2, $2, $4); }
   | read read_par       { $$ = code_new_inst(i_read_param, 2, $1, $2); }
   ;

read_par
   : SKIP cexpr    { $$ = code_new_inst(i_read_skip, 1, $2); }
   | USE cexpr     { $$ = code_new_inst(i_read_use, 1, $2); }
   | COMMENT cexpr { $$ = code_new_inst(i_read_comment, 1, $2); }
   | MATCH cexpr   { $$ = code_new_inst(i_read_match, 1, $2); }
   ;

tuple_list
   : tuple {
         $$ = code_new_inst(i_tuple_list_new, 1, $1);
      }
   | tuple_list ',' tuple  {
         $$ = code_new_inst(i_tuple_list_add, 2, $1, $3);
      }
   | read { $$ = code_new_inst(i_read, 1, $1); }
   ;

lexpr
   : cexpr CMP_EQ cexpr   { $$ = code_new_inst(i_bool_eq, 2, $1, $3); }
   | cexpr CMP_NE cexpr   { $$ = code_new_inst(i_bool_ne, 2, $1, $3); }
   | cexpr CMP_GT cexpr   { $$ = code_new_inst(i_bool_gt, 2, $1, $3); }
   | cexpr CMP_GE cexpr   { $$ = code_new_inst(i_bool_ge, 2, $1, $3); }
   | cexpr CMP_LT cexpr   { $$ = code_new_inst(i_bool_lt, 2, $1, $3); }
   | cexpr CMP_LE cexpr   { $$ = code_new_inst(i_bool_le, 2, $1, $3); }
   | sexpr CMP_EQ sexpr { $$ = code_new_inst(i_bool_seq, 2, $1, $3); }
   | sexpr CMP_NE sexpr { $$ = code_new_inst(i_bool_sneq, 2, $1, $3); }
   | sexpr CMP_GT sexpr { $$ = code_new_inst(i_bool_subs, 2, $3, $1); }
   | sexpr CMP_GE sexpr { $$ = code_new_inst(i_bool_sseq, 2, $3, $1); }
   | sexpr CMP_LT sexpr { $$ = code_new_inst(i_bool_subs, 2, $1, $3); }
   | sexpr CMP_LE sexpr { $$ = code_new_inst(i_bool_sseq, 2, $1, $3); }
   | lexpr AND lexpr    { $$ = code_new_inst(i_bool_and, 2, $1, $3); }
   | lexpr OR lexpr     { $$ = code_new_inst(i_bool_or,  2, $1, $3); }
   | lexpr XOR lexpr    { $$ = code_new_inst(i_bool_xor, 2, $1, $3); }
   | NOT lexpr          { $$ = code_new_inst(i_bool_not, 1, $2); }
   | '(' lexpr ')'      { $$ = $2; }
   | tuple IN sexpr     { $$ = code_new_inst(i_bool_is_elem, 2, $1, $3); } 
   | EXISTS '(' idxset ')' { $$ = code_new_inst(i_bool_exists, 1, $3); } 
   | BOOLDEF '(' cexpr_list ')' {
         $$ = code_new_inst(i_define_deref, 2,
            code_new_define($1),
            code_new_inst(i_tuple_new, 1, $3));
      }
   | IF lexpr THEN lexpr ELSE lexpr END {
        $$ = code_new_inst(i_expr_if_else, 3, $2, $4, $6);
     }
   ;
 
tuple
   : CMP_LT CMP_GT            { $$ = code_new_inst(i_tuple_empty, 0); }
   | CMP_LT cexpr_list CMP_GT { $$ = code_new_inst(i_tuple_new, 1, $2);  }
   ;

symidx
   : /* empty */  {
         $$ = code_new_inst(i_tuple_empty, 0);
      }
   | '[' cexpr_list ']' {
         $$ = code_new_inst(i_tuple_new, 1, $2);
      }
   ;

cexpr_list
   : cexpr {
         $$ = code_new_inst(i_elem_list_new, 1, $1);
      }
   | cexpr_list ',' cexpr {
         $$ = code_new_inst(i_elem_list_add, 2, $1, $3);
      }
   ;

cexpr
   : cproduct              { $$ = $1; }
   | cexpr '+' cproduct    { $$ = code_new_inst(i_expr_add, 2, $1, $3); }
   | cexpr '-' cproduct    { $$ = code_new_inst(i_expr_sub, 2, $1, $3); }
   ;

cproduct
   : cfactor               { $$ = $1; }
   | cproduct '*' cfactor  { $$ = code_new_inst(i_expr_mul, 2, $1, $3); }
   | cproduct '/' cfactor  { $$ = code_new_inst(i_expr_div, 2, $1, $3); }
   | cproduct MOD cfactor  { $$ = code_new_inst(i_expr_mod, 2, $1, $3); }
   | cproduct DIV cfactor  { $$ = code_new_inst(i_expr_intdiv, 2, $1, $3); }
   | PROD idxset DO cfactor {
         $$ = code_new_inst(i_expr_prod, 2, $2, $4);
      }
   ;

cfactor
   : cexpo
   | '+' cexpo  { $$ = $2; }
   | '-' cexpo  { $$ = code_new_inst(i_expr_neg, 1, $2); }
   ;

cexpo
   : cval
   | cval POW cfactor     { $$ = code_new_inst(i_expr_pow, 2, $1, $3); }
   | SUM idxset DO cproduct {
         $$ = code_new_inst(i_expr_sum, 2, $2, $4);
      }
   | MIN pure_idxset DO cfactor {
         $$ = code_new_inst(i_expr_min, 2, $2, $4);
      }
   | MAX pure_idxset DO cfactor {
         $$ = code_new_inst(i_expr_max, 2, $2, $4);
      }
   | MIN '(' idxset ')' {
         $$ = code_new_inst(i_expr_sglmin, 1, $3);
         }
   | MAX '(' idxset ')' {
         $$ = code_new_inst(i_expr_sglmax, 1, $3);
      }
   ;

cval
   : NUMB       { $$ = code_new_numb($1); }
   | STRG       { $$ = code_new_strg($1);  }
   | NAME       {
         $$ = code_new_inst(i_local_deref, 1, code_new_name($1));
      }
   | NUMBSYM symidx { 
         $$ = code_new_inst(i_symbol_deref, 2, code_new_symbol($1), $2);
      }
   | STRGSYM symidx { 
         $$ = code_new_inst(i_symbol_deref, 2, code_new_symbol($1), $2);
      }
   | NUMBDEF '(' cexpr_list ')' {
         $$ = code_new_inst(i_define_deref, 2,
            code_new_define($1),
            code_new_inst(i_tuple_new, 1, $3));
      }
   | STRGDEF '(' cexpr_list ')' {
         $$ = code_new_inst(i_define_deref, 2,
            code_new_define($1),
            code_new_inst(i_tuple_new, 1, $3));
      }
   | cval FAC               { $$ = code_new_inst(i_expr_fac, 1, $1); } 
   | CARD '(' sexpr ')'     { $$ = code_new_inst(i_expr_card, 1, $3); }
   | ABS '(' cexpr ')'      { $$ = code_new_inst(i_expr_abs, 1, $3); }
   | SGN '(' cexpr ')'      { $$ = code_new_inst(i_expr_sgn, 1, $3); }
   | ROUND '(' cexpr ')'    { $$ = code_new_inst(i_expr_round, 1, $3); }
   | FLOOR '(' cexpr ')'    { $$ = code_new_inst(i_expr_floor, 1, $3); }
   | CEIL '(' cexpr ')'     { $$ = code_new_inst(i_expr_ceil, 1, $3); }
   | LOG '(' cexpr ')'      { $$ = code_new_inst(i_expr_log, 1, $3); }
   | LN '(' cexpr ')'       { $$ = code_new_inst(i_expr_ln, 1, $3); }
   | EXP '(' cexpr ')'      { $$ = code_new_inst(i_expr_exp, 1, $3); }
   | SQRT '(' cexpr ')'     { $$ = code_new_inst(i_expr_sqrt, 1, $3); }
   | SIN '(' cexpr ')'      { $$ = code_new_inst(i_expr_sin, 1, $3); }
   | COS '(' cexpr ')'      { $$ = code_new_inst(i_expr_cos, 1, $3); }
   | TAN '(' cexpr ')'      { $$ = code_new_inst(i_expr_tan, 1, $3); }
   | ASIN '(' cexpr ')'     { $$ = code_new_inst(i_expr_asin, 1, $3); }
   | ACOS '(' cexpr ')'     { $$ = code_new_inst(i_expr_acos, 1, $3); }
   | ATAN '(' cexpr ')'     { $$ = code_new_inst(i_expr_atan, 1, $3); }

   | '(' cexpr ')'          { $$ = $2; }
   | LENGTH '(' cexpr ')'   { $$ = code_new_inst(i_expr_length, 1, $3); }
   | SUBSTR '(' cexpr ',' cexpr ',' cexpr ')' {
         $$ = code_new_inst(i_expr_substr, 3, $3, $5, $7);
      }
   | RANDOM '(' cexpr ',' cexpr ')' {
         $$ = code_new_inst(i_expr_rand, 2, $3, $5);
      }
   | IF lexpr THEN cexpr ELSE cexpr END {
         $$ = code_new_inst(i_expr_if_else, 3, $2, $4, $6);
      }
   | ORD '(' sexpr ',' cexpr ',' cexpr ')' {
         $$ = code_new_inst(i_expr_ord, 3, $3, $5, $7);
      }
   | MIN '(' cexpr_list ')' {
         $$ = code_new_inst(i_expr_min2, 1, $3);
      }
   | MAX '(' cexpr_list ')' {
         $$ = code_new_inst(i_expr_max2, 1, $3);
      }
   ;
