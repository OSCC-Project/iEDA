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

#include <ctype.h>
#include <string>

#include "string/Str.hh"
#include "mLibertyParse.hh"



#undef  YY_DECL
#define YY_DECL int lib_lex(LIB_STYPE *yylval_param, yyscan_t yyscanner, ista::LibertyReader *lib_reader)
#define YYSTYPE LIB_STYPE


%}

/* %option debug */
%option prefix="lib_"
%option reentrant
%option noyywrap
%option bison-bridge
%option nounput
%option never-interactive

%x comment
%x qstring

DIGIT_BIT [0-9]
ALPHABET [a-zA-Z]
FLOAT [-+]?(({DIGIT_BIT}+\.?{DIGIT_BIT}*)|(\.{DIGIT_BIT}+))([Ee][-+]?{DIGIT_BIT}+)?
BLANK [ \t]
BUS_LEFT_FLAG [\[<]
BUS_RIGHT_FLAG [\]>]
BUS_SUB {BUS_LEFT_FLAG}{DIGIT_BIT}+{BUS_RIGHT_FLAG}
BUS_RANGE {BUS_LEFT_FLAG}{DIGIT_BIT}+:{DIGIT_BIT}+{BUS_RIGHT_FLAG}
PIN_NAME ({ALPHABET}|_)({ALPHABET}|{DIGIT_BIT}|_)*
BUS_NAME {PIN_NAME}({BUS_SUB}|{BUS_RANGE})
BUS_NAME2 {PIN_NAME}{BUS_SUB}({BUS_SUB}|{BUS_RANGE})
MIXED_NAME {BUS_NAME}_{PIN_NAME}
HIERARCHICAL_NAME ({PIN_NAME}|{BUS_NAME}|{MIXED_NAME})([\/.]({PIN_NAME}|{BUS_NAME}|{MIXED_NAME}))+
/* ocv_table_template(2D_ocv_template) */
/* leakage_power_unit             : 1pW; */
/* default_operating_conditions : slow_100_3.00 ; */
/* revision : 1.0.17; */
/* default_wire_load : xc2v250-5_avg; */
TOKEN ({ALPHABET}|{DIGIT_BIT}|_)({ALPHABET}|{DIGIT_BIT}|[._\-])*
/* bus_naming_style : %s[%d] ; */
BUS_STYLE "%s"{BUS_LEFT_FLAG}"%d"{BUS_RIGHT_FLAG}
PUNCTUATION [,\:;|(){}+*&!'=]
TOKEN_END {PUNCTUATION}|[ \t\r\n]
END_OF_LINE \r?\n
%%

%{
yylval = yylval_param;
%}

{PUNCTUATION} { return ((int) yytext[0]); }

{FLOAT}{TOKEN_END} {
    /* Push back the TOKEN_END character. */
    yyless(yyleng - 1);
    yylval->number = static_cast<double>(strtod(yytext,
                                                        NULL));
    return FLOAT;
    }

{ALPHABET}({ALPHABET}|_|{DIGIT_BIT})*{TOKEN_END} {
    /* Push back the TOKEN_END character. */
    yyless(yyleng - 1);
    yylval->string = lib_reader->stringCopy(yytext);
    return KEYWORD;
    }

{PIN_NAME}{TOKEN_END} |
{BUS_NAME}{TOKEN_END} |
{BUS_NAME2}{TOKEN_END} |
{MIXED_NAME}{TOKEN_END} |
{HIERARCHICAL_NAME}{TOKEN_END} |
{BUS_STYLE}{TOKEN_END} |
{TOKEN}{TOKEN_END} {
    /* Push back the TOKEN_END character. */
    yyless(yyleng - 1);
    yylval->string = lib_reader->stringCopy(yytext);
    return STRING;
    }

\\?{END_OF_LINE} { lib_reader->incrLineNo(); }

"include_file"[ \t]*"(".+")"[ \t]*";"? {

    /*Support the include file TODO*/

}

"/*"    BEGIN(comment);

    /* Straight out of the flex man page. */
<comment>[^*\r\n]*      /* eat anything that's not a '*' */
<comment>"*"+[^*/\r\n]*     /* eat up '*'s not followed by '/'s */
<comment>{END_OF_LINE}  lib_reader->incrLineNo();
<comment>"*"+"/" BEGIN(INITIAL);

\"  {
    lib_reader->clearRecordStr();
    BEGIN(qstring);
    }

<qstring>\" {
    BEGIN(INITIAL);
    yylval->string = lib_reader->stringCopy(lib_reader->get_record_str());
    return STRING;
    }

<qstring>{END_OF_LINE} {
    BEGIN(INITIAL);
    yylval->string = lib_reader->stringCopy(lib_reader->get_record_str());;
    return STRING;
    }

<qstring>\\{END_OF_LINE} {
    /* Line continuation. */
    lib_reader->incrLineNo();
    }

<qstring>\\. {
    /* Escaped character. */
    lib_reader->recordStr("\\");
    lib_reader->recordStr(yytext);
    }

<qstring>[^\\\r\n\"]+ {
    /* Anything but escape, return or double quote */
    lib_reader->recordStr(yytext);
    }

<qstring><<EOF>> {
    BEGIN(INITIAL);
    yyterminate();
    }

{BLANK}* {}
    /* Send out of bound characters to parser. */
.   { return (int) yytext[0]; }

<<EOF>> {

    yyterminate();
}

%%
namespace  ista {

   void LibertyReader::parseBegin(FILE *fp)
   {
      lib_lex_init(&_scanner);
      lib_restart(fp, _scanner);
   }

   int LibertyReader::parse()
   {
      return  lib_parse(_scanner, this);
   }

   void LibertyReader::parseEnd(FILE *fp)
   {
      if (_scanner != nullptr) {
        lib_lex_destroy(_scanner);
      }
         
      _scanner = nullptr;
   }

}