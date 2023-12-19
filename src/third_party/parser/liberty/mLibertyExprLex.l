%{
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

#include "string/Str.hh"
#include "mLibertyExprParse.hh"
#include "mLibertyExpr.hh"

using namespace ista;

#undef  YY_DECL
#define YY_DECL int lib_expr_lex(LIB_EXPR_STYPE *yylval_param, yyscan_t yyscanner, ista::LibertyExprBuilder *lib_expr_builder)
#define YYSTYPE LIB_EXPR_STYPE

%}

/* %option debug */
%option prefix="lib_expr_"
%option reentrant
%option noyywrap
%option bison-bridge
%option nounput
%option never-interactive

%x ESCAPED_STRING

PORT	[A-Za-z_]([A-Za-z0-9_\[\]])*
OPERATION	"'"|"!"|"^"|"*"|"&"|"+"|"|"|1|0
PARENTHESES	"("|")"
BLANK	[ \t\r]
ESCAPE	\\
QUOTE	\"
END_OF_LINE	\r?\n

%%

{OPERATION}|{PARENTHESES} { return ((int) yytext[0]); }

{ESCAPE}{END_OF_LINE} { /* I doubt that escaped returns get thru the parser */ }

{ESCAPE}{QUOTE}	{ BEGIN(ESCAPED_STRING); lib_expr_builder->clearRecordStr(); }

<ESCAPED_STRING>. {     
    lib_expr_builder->recordStr("\\");
    lib_expr_builder->recordStr(yytext);
    }

<ESCAPED_STRING>{ESCAPE}{QUOTE} {
    BEGIN(INITIAL);
    yylval->string = lib_expr_builder->stringCopy(lib_expr_builder->get_record_str());
    return PORT;
    }

{PORT} {
    yylval->string = lib_expr_builder->stringCopy(yytext);
    return PORT;
    }

{BLANK} {}

    /* Send out of bound characters to parser. */
. { return (int) yytext[0]; }

%%

   void LibertyExprBuilder::parseBegin()
   {
      lib_expr_lex_init(&_scanner);
      yy_scan_string(_expr_str.c_str(), _scanner);
   }

   int LibertyExprBuilder::parse()
   {
      return  lib_expr_parse(_scanner, this);
   }

   void LibertyExprBuilder::parseEnd()
   {
      if (_scanner != nullptr) {
        lib_expr_lex_destroy(_scanner);
      }
         
      _scanner = nullptr;
   }
