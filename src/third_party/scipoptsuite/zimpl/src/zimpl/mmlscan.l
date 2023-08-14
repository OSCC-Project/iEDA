/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*   File....: mmlscan.l                                                     */
/*   Name....: MML Lexcal Analyser                                           */
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
%{   

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wunused-macros"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#pragma clang diagnostic ignored "-Wunreachable-code"
#pragma clang diagnostic ignored "-Wextra-semi-stmt"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#endif

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#pragma GCC   diagnostic ignored "-Wmissing-prototypes"
#pragma GCC   diagnostic ignored "-Wsign-compare"
#pragma GCC   diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC   diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif

#include <sys/types.h>
//#include <stdlib.h>
#include <ctype.h>
//#include <string.h>
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
#include "zimpl/define.h"
#include "zimpl/stmt.h"
#include "zimpl/strstore.h"
#include "zimpl/mmlparse2.h"

/*lint -sem(yyerror, 1p, r_no) */ 
extern void yyerror(const char* s) is_NORETURN;

#define YY_DECL int yylex(YYSTYPE* lval)

YY_DECL;

static int        yycolumn  = 0;
static int        yydecl    = 0;

%}
%option noyylineno
%option never-interactive
%option noyywrap
%option noyymore
%option noreject
%option nounput
%pointer
%%
"**"|"^"                  { yycolumn += yyleng; return POW; }
":="                      { yycolumn += yyleng; return ASGN; }
"=="                      { yycolumn += yyleng; return CMP_EQ; }
"<="                      { yycolumn += yyleng; return CMP_LE; }
">="                      { yycolumn += yyleng; return CMP_GE; }
"<"                       { yycolumn += yyleng; return CMP_LT; }
">"                       { yycolumn += yyleng; return CMP_GT; }
"!="                      { yycolumn += yyleng; return CMP_NE; }
"!"                       { yycolumn += yyleng; return FAC; }
"not"                     { yycolumn += yyleng; return NOT; }
"and"                     { yycolumn += yyleng; return AND; }
"or"                      { yycolumn += yyleng; return OR; }
"xor"                     { yycolumn += yyleng; return XOR; }
"set"                     { yycolumn += yyleng; yydecl = DECLSET; return DECLSET; }
"param"                   { yycolumn += yyleng; yydecl = DECLPAR; return DECLPAR; }
"var"                     { yycolumn += yyleng; yydecl = DECLVAR; return DECLVAR; }
"maximize"                { yycolumn += yyleng; yydecl = DECLMAX; return DECLMAX; }
"minimize"                { yycolumn += yyleng; yydecl = DECLMIN; return DECLMIN; }
"subto"                   { yycolumn += yyleng; yydecl = DECLSUB; return DECLSUB; }
"sos"                     { yycolumn += yyleng; yydecl = DECLSOS; return DECLSOS; }
"defnumb"                 { yycolumn += yyleng; yydecl = DEFNUMB; return DEFNUMB; }
"defstrg"                 { yycolumn += yyleng; yydecl = DEFSTRG; return DEFSTRG; }
"defbool"                 { yycolumn += yyleng; yydecl = DEFBOOL; return DEFBOOL; }
"defset"                  { yycolumn += yyleng; yydecl = DEFSET;  return DEFSET; }
"in"                      { yycolumn += yyleng; return IN; }
"with"|"|"                { yycolumn += yyleng; return WITH; }
"do"|":"                  { yycolumn += yyleng; return DO; }
"binary"                  { yycolumn += yyleng; return BINARY; }
"integer"                 { yycolumn += yyleng; return INTEGER; }
"real"                    { yycolumn += yyleng; return REAL; }
"sum"                     { yycolumn += yyleng; return SUM; }
"prod"                    { yycolumn += yyleng; return PROD; }
"forall"                  { yycolumn += yyleng; return FORALL; }
"exists"                  { yycolumn += yyleng; return EXISTS; }
"vif"                     { yycolumn += yyleng; return VIF; }
"if"                      { yycolumn += yyleng; return IF; }
"then"                    { yycolumn += yyleng; return THEN; }
"else"                    { yycolumn += yyleng; return ELSE; }
"end"                     { yycolumn += yyleng; return END; }
"to"                      { yycolumn += yyleng; return TO; }
".."                      { yycolumn += yyleng; return UNTIL; }
"by"                      { yycolumn += yyleng; return BY; }
"union"                   { yycolumn += yyleng; return UNION; }
"inter"                   { yycolumn += yyleng; return INTER; }
"symdiff"                 { yycolumn += yyleng; return SYMDIFF; }
"cross"                   { yycolumn += yyleng; return CROSS; }
"proj"                    { yycolumn += yyleng; return PROJ; }
"without"|"\\"            { yycolumn += yyleng; return WITHOUT; }
"mod"|"modulo"            { yycolumn += yyleng; return MOD; }
"div"                     { yycolumn += yyleng; return DIV; }
"min"                     { yycolumn += yyleng; return MIN; }
"max"                     { yycolumn += yyleng; return MAX; }
"argmin"                  { yycolumn += yyleng; return ARGMIN; }
"argmax"                  { yycolumn += yyleng; return ARGMAX; }
"read"                    { yycolumn += yyleng; return READ; }
"as"                      { yycolumn += yyleng; return AS; }
"skip"                    { yycolumn += yyleng; return SKIP; }
"use"                     { yycolumn += yyleng; return USE; }
"comment"                 { yycolumn += yyleng; return COMMENT; }
"scale"                   { yycolumn += yyleng; return SCALE; }
"separate"                { yycolumn += yyleng; return SEPARATE; }
"checkonly"               { yycolumn += yyleng; return CHECKONLY; }
"indicator"               { yycolumn += yyleng; return INDICATOR; }
"qubo"                    { yycolumn += yyleng; return QUBO; }
"penalty1"                { yycolumn += yyleng; return PENALTY1; }
"penalty2"                { yycolumn += yyleng; return PENALTY2; }
"penalty3"                { yycolumn += yyleng; return PENALTY3; }
"penalty4"                { yycolumn += yyleng; return PENALTY4; }
"penalty5"                { yycolumn += yyleng; return PENALTY5; }
"penalty6"                { yycolumn += yyleng; return PENALTY6; }
"card"                    { yycolumn += yyleng; return CARD; }
"abs"                     { yycolumn += yyleng; return ABS; }
"vabs"                    { yycolumn += yyleng; return VABS; }
"sgn"                     { yycolumn += yyleng; return SGN; }
"round"                   { yycolumn += yyleng; return ROUND; }
"floor"                   { yycolumn += yyleng; return FLOOR; }
"ceil"                    { yycolumn += yyleng; return CEIL; }
"log"                     { yycolumn += yyleng; return LOG; }
"ln"                      { yycolumn += yyleng; return LN; }
"exp"                     { yycolumn += yyleng; return EXP; }
"sqrt"                    { yycolumn += yyleng; return SQRT; }
"sin"                     { yycolumn += yyleng; return SIN; }
"cos"                     { yycolumn += yyleng; return COS; }
"tan"                     { yycolumn += yyleng; return TAN; }
"asin"                    { yycolumn += yyleng; return ASIN; }
"acos"                    { yycolumn += yyleng; return ACOS; }
"atan"                    { yycolumn += yyleng; return ATAN; }
"pow"                     { yycolumn += yyleng; return POWER; }
"sgnpow"                  { yycolumn += yyleng; return SGNPOW; }
"priority"                { yycolumn += yyleng; return PRIORITY; }
"startval"                { yycolumn += yyleng; return STARTVAL; }
"default"                 { yycolumn += yyleng; return DEFAULT; }
"subsets"                 { yycolumn += yyleng; return SUBSETS; }
"powerset"                { yycolumn += yyleng; return POWERSET; }
"indexset"                { yycolumn += yyleng; return INDEXSET; }
"print"                   { yycolumn += yyleng; return PRINT; }
"check"                   { yycolumn += yyleng; return CHECK; }
"infinity"                { yycolumn += yyleng; return INFTY; }
"random"                  { yycolumn += yyleng; return RANDOM; }
"ord"                     { yycolumn += yyleng; return ORD; }
"type1"                   { yycolumn += yyleng; return TYPE1; }
"type2"                   { yycolumn += yyleng; return TYPE2; }
"implicit"                { yycolumn += yyleng; return IMPLICIT; }
"length"                  { yycolumn += yyleng; return LENGTH; }
"substr"                  { yycolumn += yyleng; return SUBSTR; }
"match"                   { yycolumn += yyleng; return MATCH; }

[0-9]+("."[0-9]+)?([eEdD][-+]?[0-9]+)?  { 
      yycolumn += yyleng; 
      lval->numb = numb_new_ascii(yytext);
/*      sscanf(yytext, "%lf", &lval->numb);*/
      return NUMB; 
   }
"."[0-9]+([eEdD][-+]-?[0-9]+)?  { 
      yycolumn += yyleng; 
      lval->numb = numb_new_ascii(yytext);
/*      sscanf(yytext, "%lf", &lval->numb);*/
      return NUMB; 
   }

[A-Za-z_][A-Za-z0-9_]* {

      Symbol* sym;
      Define* def;
      char    errmsg[256];
      int     what;

      yycolumn += yyleng; 

      /* If it is sure that this is the name for a constraint or 
       * objective, or sos, we do not need to lookup the name.
       */
      if (yydecl == DECLSUB || yydecl == DECLMIN || yydecl == DECLMAX 
       || yydecl == DECLSOS)
         yydecl     = 0;

      /* If it is sure that this is the name for a set, param or var
       * we check only if the name is unused and otherwise give
       * an appropriate error message.
       */
      else if (yydecl == DECLSET || yydecl == DECLPAR || yydecl == DECLVAR 
	    || yydecl == DEFNUMB || yydecl == DEFSTRG || yydecl == DEFBOOL 
       || yydecl == DEFSET)
      {
         if (NULL != symbol_lookup(yytext) || NULL != define_lookup(yytext))
         {
            /* Allow to declare parameter on the command line and ignore the
             * redeclarations in the zimpl code later on 
             */
            if (yydecl == DECLPAR)
            {
               extern void yywarning(int no, const char* s);

               sprintf(errmsg, "Redefinition of parameter %s ignored\n", yytext);
               yywarning(252, errmsg);
               yydecl = 0;
               return EOF;
            }
            sprintf(errmsg, "Name \"%s\" already in use\n", yytext);
            yyerror(errmsg);
         }
         what   = yydecl;
         yydecl = 0;

         /* if this is a define, we have to install it here,
          * so that recursive call succseed.
          */
         switch(what)
         {
         case DEFNUMB :
            lval->def = define_new(str_new(yytext), DEF_NUMB);
            return DEFNAME;
         case DEFSTRG :
            lval->def = define_new(str_new(yytext), DEF_STRG);
            return DEFNAME;
         case DEFBOOL :
            lval->def = define_new(str_new(yytext), DEF_BOOL);
            return DEFNAME;
         case DEFSET :
            lval->def = define_new(str_new(yytext), DEF_SET);
            return DEFNAME;
         default :
            break;
         }
      }
      else
      {
         assert(yydecl == 0);

         if (NULL != (sym = symbol_lookup(yytext)))
         {
            lval->sym = sym;

            switch(symbol_get_type(sym))
            {
            case SYM_NUMB :
               return NUMBSYM;
            case SYM_STRG :
               return STRGSYM;
            case SYM_VAR :
               return VARSYM;
            case SYM_SET :
               return SETSYM;
            case SYM_ERR : /* should not happen */
               sprintf(errmsg, "Symbol \"%s\" not initialised\n", yytext);
               yyerror(errmsg);
            default :
               abort();
            }
         }
         if (NULL != (def = define_lookup(yytext)))
         {
            lval->def = def;

            switch(define_get_type(def))
            {
            case DEF_NUMB :
               return NUMBDEF;
            case DEF_STRG :
               return STRGDEF;
            case DEF_BOOL :
               return BOOLDEF;
            case DEF_SET :
               return SETDEF;
            case DEF_ERR : /* should not happen */
               sprintf(errmsg, "Define \"%s\" not initialised\n", yytext);
               yyerror(errmsg);
            default :
               abort();
            }
         }
      }
      lval->name = str_new(yytext);

      return NAME;
   }

\"[^\"]*\" { 
      yycolumn += yyleng; 
      yytext[strlen(yytext) - 1] = '\0';
      lval->strg = str_new(yytext + 1); 
      return STRG; 
   }
\'[^\']*\' { 
      yycolumn += yyleng; 
      yytext[strlen(yytext) - 1] = '\0';
      lval->strg = str_new(yytext + 1); 
      return STRG; 
   }

[[:space:]]+              { yycolumn += yyleng; }

.                         { yycolumn += yyleng; return *yytext; }

<<EOF>>                   { yycolumn += yyleng; return EOF; }

%%
static const Stmt* yystmt = NULL;

void yyerror(const char* s)
{
   fprintf(stderr, "*** Error 800: File %s Line %d : %s\n", 
      stmt_get_filename(yystmt), stmt_get_lineno(yystmt), s);
   show_source(stderr, stmt_get_text(yystmt), yycolumn);
   zpl_exit(EXIT_FAILURE);
}

void yywarning(int no, const char* s)
{
   if (stmt_trigger_warning(no))
      fprintf(stderr, 
         "*** Warning %d: File %s Line %d : %s\n", 
         no, stmt_get_filename(yystmt), stmt_get_lineno(yystmt), s);
}

const Stmt* scan_get_stmt(void)
{
   return yystmt;
}

int scan_get_column(void)
{
   return yycolumn;
}

void parse_stmt(const Stmt* stmt)
{
   YY_BUFFER_STATE state;

   yy_delete_buffer(YY_CURRENT_BUFFER);

   yystmt   = stmt;
   yycolumn = 0;
   state    = yy_scan_string(stmt_get_text(stmt));

   if (yyparse() != 0)
   {
      fprintf(stderr, "*** Error 801: Parser failed\n");
      zpl_exit(EXIT_FAILURE);
   }
   yy_delete_buffer(state);
}


