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
#include "log/Log.hh"
#include "Config.hh"

#include "mVerilogParse.hh"

using ieda::Str;

#ifdef ZLIB_FOUND
extern ista::VerilogReader* gVerilogReader;
#define YY_INPUT(buf,result,max_size) gVerilogReader->getChars(buf, result, max_size)
#endif

#undef  YY_DECL
#define YY_DECL int verilog_lex(VERILOG_STYPE *yylval_param, yyscan_t yyscanner, ista::VerilogReader *verilog_reader)
#define YYSTYPE VERILOG_STYPE

%}

/*  %option debug */
%option prefix="verilog_"
%option reentrant
%option noyywrap
%option bison-bridge
%option nounput
%option never-interactive

%x COMMENT
%x ATTRIBUTE
%x QSTRING

SIGN_BIT	"+"|"-"
UNSIGNED_NUMBER [0-9][0-9_]*
BLANK	[ \t\r]
END_OF_LINE 	\r?\n
ID_ESCAPED_TOKEN \\[^ \t\r\n]+[\r\n\t ]
ID_ALPHABET_TOKEN [A-Za-z_][A-Za-z0-9_$]*
ID_TOKEN {ID_ESCAPED_TOKEN}|{ID_ALPHABET_TOKEN}

%%

%{
yylval = yylval_param;
%}

^[ \t]*`.*{END_OF_LINE} { /* Macro definition. */
    verilog_reader->incrLineNo();
    }

"//"[^\n]*{END_OF_LINE} { /* Single line comment. */
    verilog_reader->incrLineNo();
    }

"/*"	{ BEGIN COMMENT; }
<COMMENT>{
.

{END_OF_LINE}	{ verilog_reader->incrLineNo(); }

"*/"	{ BEGIN INITIAL; }

<<EOF>> {
    LOG_ERROR << "unterminated comment";
    BEGIN(INITIAL);
    yyterminate();
    }
}

"(*"	{ BEGIN ATTRIBUTE; }
<ATTRIBUTE>{
.

{END_OF_LINE}	{ verilog_reader->incrLineNo(); }

"*)"	{ BEGIN INITIAL; }

<<EOF>> {
    LOG_ERROR << "unterminated attribute";
    BEGIN(INITIAL);
    yyterminate();
    }
}

{SIGN_BIT}?{UNSIGNED_NUMBER}?"'"[bB][01_xz]+ {
  yylval->constant = Str::copy(yytext);
  return CONSTANT;
}

{SIGN_BIT}?{UNSIGNED_NUMBER}?"'"[oO][0-7_xz]+ {
  yylval->constant = Str::copy(yytext);
  return CONSTANT;
}

{SIGN_BIT}?{UNSIGNED_NUMBER}?"'"[dD][0-9_]+ {
  yylval->constant = Str::copy(yytext);
  return CONSTANT;
}

{SIGN_BIT}?{UNSIGNED_NUMBER}?"'"[hH][0-9a-fA-F_xz]+ {
  yylval->constant = Str::copy(yytext);
  return CONSTANT;
}

{SIGN_BIT}?[0-9]+ {
  yylval->integer = Str::toInt(yytext);
  return INT;
}

":"|"."|"{"|"}"|"["|"]"|","|"*"|";"|"="|"-"|"+"|"|"|"("|")"|"#" {
  return ((int) yytext[0]);
}

assign { return ASSIGN; }
endmodule { return ENDMODULE; }
inout { return INOUT; }
input { return INPUT; }
module { return MODULE; }
output { return OUTPUT; }
parameter { return PARAMETER; }
defparam { return DEFPARAM; }
reg { return REG; }
supply0 { return SUPPLY0; }
supply1 { return SUPPLY1; }
tri { return TRI; }
wand { return WAND; }
wire { return WIRE; }
wor { return WOR; }

{ID_TOKEN}("."{ID_TOKEN})* {
    yylval->string = Str::copy(Str::trimmed(yytext));
    return ID;
}

{END_OF_LINE}	{ verilog_reader->incrLineNo(); }

{BLANK}	{ /* ignore blanks */ }

\"	{
    verilog_reader->clearRecordStr();
    BEGIN(QSTRING);
    }

<QSTRING>\" {
    BEGIN(INITIAL);
    yylval->string = Str::copy(verilog_reader->get_record_str());
    return STRING;
    }

<QSTRING>{END_OF_LINE} {
    LOG_ERROR  << "unterminated string constant";
    BEGIN(INITIAL);
    yylval->string = Str::copy(verilog_reader->get_record_str());
    return STRING;
    }

<QSTRING>\\{END_OF_LINE} {
    /* Line continuation. */
    verilog_reader->incrLineNo();
    }

<QSTRING>[^\r\n\"]+ {
    /* Anything return or double quote */
    verilog_reader->recordStr(yytext);
    }

<QSTRING><<EOF>> {
    LOG_ERROR  << "unterminated string constant";
    BEGIN(INITIAL);
    yyterminate();
    }

    /* Send out of bound characters to parser. */
.   { return (int) yytext[0]; }

%%

namespace  ista {

#ifdef ZLIB_FOUND
   void VerilogReader::parseBegin(gzFile fp)
   {
      verilog_lex_init(&_scanner);
       _verilog_in = fp;
   }
#else
  void VerilogReader::parseBegin(FILE *fp)
   {
      verilog_lex_init(&_scanner);
      verilog_restart(fp, _scanner);
   }
#endif

   int VerilogReader::parse()
   {
      return  verilog_parse(_scanner, this);
   }


#ifdef ZLIB_FOUND
   void VerilogReader::parseEnd(gzFile fp)
   {
      if (_scanner != nullptr) {
        verilog_lex_destroy(_scanner);
      }

      gzclose(fp);
         
      _scanner = nullptr;
   }
#else
      void VerilogReader::parseEnd(FILE *fp)
   {
      if (_scanner != nullptr) {
        verilog_lex_destroy(_scanner);
      }
         
      _scanner = nullptr;
   }
#endif
}