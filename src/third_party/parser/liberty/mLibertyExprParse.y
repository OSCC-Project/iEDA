%code requires {

// OpenSTA, Static Timing Analyzer
// Copyright (c) 2021, Parallax Software, Inc.
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// Liberty function expression parser.
#include "log/Log.hh"
#include "mLibertyExpr.hh"

using namespace ista;

#define YYDEBUG 1

typedef void* yyscan_t;

}

%code provides {
#undef  YY_DECL
#define YY_DECL int lib_expr_lex(LIB_EXPR_STYPE *yylval_param, yyscan_t yyscanner, ista::LibertyExprBuilder *lib_expr_builder)
YY_DECL;

void yyerror(yyscan_t scanner, ista::LibertyExprBuilder *lib_expr_builder, const char *str);
}

%union {
  int int_val;
  const char *string;
  void *expr;
}

%define api.pure full
%define api.prefix {lib_expr_}
%parse-param { yyscan_t yyscanner }
%parse-param { ista::LibertyExprBuilder *lib_expr_builder }
%lex-param   { yyscan_t yyscanner }
%lex-param   { ista::LibertyExprBuilder *lib_expr_builder }

%token PORT
%left '+' '|'
%left '*' '&'
%left '^'
%left '!' '\''

%type <string> PORT
%type <expr> expr terminal terminal_expr implicit_and

%{
%}

%%

result_expr:
    expr { lib_expr_builder->set_result_expr(static_cast<LibertyExpr*>($1)); }
|  expr ';'{ lib_expr_builder->set_result_expr(static_cast<LibertyExpr*>($1)); }
;

expr:
    terminal_expr
|  implicit_and
|  expr '+' expr  { $$ = lib_expr_builder->makePlusExpr(static_cast<LibertyExpr*>($1), static_cast<LibertyExpr*>($3)); }
|  expr '|' expr  { $$ = lib_expr_builder->makeOrExpr(static_cast<LibertyExpr*>($1), static_cast<LibertyExpr*>($3)); }
|  expr '*' expr   { $$ = lib_expr_builder->makeMultExpr(static_cast<LibertyExpr*>($1), static_cast<LibertyExpr*>($3)); }
|  expr '&' expr   { $$ = lib_expr_builder->makeAndExpr(static_cast<LibertyExpr*>($1), static_cast<LibertyExpr*>($3)); }
|  expr '^' expr  { $$ = lib_expr_builder->makeXorExpr(static_cast<LibertyExpr*>($1), static_cast<LibertyExpr*>($3)); }
;

terminal_expr:
    terminal
|  '!' terminal  { $$ = lib_expr_builder->makeNotExpr(static_cast<LibertyExpr*>($2)); }
|  terminal '\''  { $$ = nullptr; }
;

implicit_and:
    terminal_expr terminal_expr
    { $$ = lib_expr_builder->makeAndExpr(static_cast<LibertyExpr*>($1), static_cast<LibertyExpr*>($2)); }
|  implicit_and terminal_expr
    { $$ = lib_expr_builder->makeAndExpr(static_cast<LibertyExpr*>($1), static_cast<LibertyExpr*>($2)); }
;

terminal:
    PORT    { $$ = lib_expr_builder->makeBufferExpr(static_cast<const char*>($1)); lib_expr_builder->stringDelete(static_cast<const char*>($1));}
|  '0'    { $$ = lib_expr_builder->makeZeroExpr(); }
|  '1'    { $$ = lib_expr_builder->makeOneExpr(); }
|  '(' expr ')'  { $$ = $2; }
;

%%

void lib_expr_error(yyscan_t scanner, ista::LibertyExprBuilder *lib_expr_builder, const char *str)
{
   LOG_ERROR << "Error found" << str;
}
