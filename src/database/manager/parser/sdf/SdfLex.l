%{

#include "string/Str.hh"
#include "log/Log.hh"
#include "SdfParse.hh"
#include <string>
#include "SdfReader.hh"

using namespace ista;

#define YY_NO_INPUT

static std::string sdf_token;

int g_sdf_line = 0;
extern SdfReader* g_sdf_reader;

void
sdfFlushBuffer()
{
  YY_FLUSH_BUFFER;
}

%}

/* %option debug */
%option noyywrap
%option nounput
%option never-interactive

%x COMMENT
%x QUOTE
%x COND_EXPR

ID   ([A-Za-z_]|\\.)([A-Za-z0-9_\[\]]|\\.)*
HIERARCHICAL_CHAR "."|"/"
BLANK   [ \n\t\r\b]
END_OF_LINE   \r?\n

%%

"/*"      { BEGIN COMMENT; }
<COMMENT>{

"*/"   { BEGIN INITIAL; }

.
{END_OF_LINE}   { g_sdf_line++; }

<<EOF>> {
   BEGIN(INITIAL);
   yyterminate();
   }
}

"\""   { BEGIN QUOTE; sdf_token.erase(); }
<QUOTE>{

"\\".   { sdf_token += yytext[1]; }

"\""    {
   BEGIN INITIAL;
   Sdf_lval.string = Str::copy(sdf_token.c_str());
   return QSTRING;
   }

.  { sdf_token += yytext[0]; }

<<EOF>> {
   LOG_FATAL << "unterminated quoted string";
   BEGIN(INITIAL);
   yyterminate();
   }
}

"//"[^\n]*{END_OF_LINE} { g_sdf_line++; }

("-"|"+")?([0-9]*)("."[0-9]+)?([eE]("-"|"+")?[0-9]+)? {
   Sdf_lval.number = static_cast<float>(atof(yytext));
   return NUMBER;
   }

":"|"{"|"}"|"["|"]"|","|"*"|";"|"="|"-"|"+"|"|"|"("|")"|{HIERARCHICAL_CHAR}   {
   return ((int) yytext[0]);
   }

ABSOLUTE { return ABSOLUTE; }
CELL   { return CELL; }
CELLTYPE { return CELLTYPE; }
DATE   { return DATE; }
DELAY   { return DELAY; }
DELAYFILE { return DELAYFILE; }
DESIGN   { return DESIGN; }
DEVICE   { return DEVICE; }
DIVIDER   { return DIVIDER; }
HOLD   { return HOLD; }
INCREMENTAL|INCREMENT { return INCREMENTAL; }
INSTANCE { return INSTANCE; }
INTERCONNECT { return INTERCONNECT; }
IOPATH   { return IOPATH; }
NOCHANGE { return NOCHANGE; }
PERIOD   { return PERIOD; }
PORT   { return PORT; }
PROCESS   { return PROCESS; }
PROGRAM   { return PROGRAM; }
RECOVERY { return RECOVERY; }
RECREM { return RECREM; }
REMOVAL { return REMOVAL; }
RETAIN { return RETAIN; }
SDFVERSION { return SDFVERSION; }
SETUP   { return SETUP; }
SETUPHOLD { return SETUPHOLD; }
SKEW   { return SKEW; }
TEMPERATURE { return TEMPERATURE; }
TIMESCALE { return TIMESCALE; }
TIMINGCHECK { return TIMINGCHECK; }
VENDOR   { return VENDOR; }
VERSION { return PVERSION; }
VOLTAGE   { return VOLTAGE; }
WIDTH   { return WIDTH; }
negedge { return NEGEDGE; }
posedge { return POSEDGE; }

CONDELSE { return CONDELSE; }

COND   {
   BEGIN COND_EXPR;
   sdf_token.erase();
   return COND;
}

<COND_EXPR>"("{BLANK}*IOPATH {
   BEGIN INITIAL;
   Sdf_lval.string = Str::copy(sdf_token.c_str());
   return EXPR_OPEN_IOPATH;
}

<COND_EXPR>"(" {
   /* Timing check conditions don't allow parens,
    * so use the paren as a marker for the end of the expr.
    */
   if (g_sdf_reader->isParseTimingCheck()) {
    BEGIN INITIAL;
     Sdf_lval.string= Str::copy(sdf_token.c_str());
     g_sdf_reader->set_parse_timing_check(false);
     return EXPR_OPEN;
   }
   else {
     sdf_token += yytext[0];
   }
     
}

<COND_EXPR>{BLANK}+{ID}{BLANK}*")" {
    /* (COND expr port) */
   if (g_sdf_reader->isParseTimingCheck()) {
    BEGIN INITIAL;

    /* remove trailing ")" */
    yytext[strlen(yytext)-1] = '\0';
    sdf_token += yytext;

    Sdf_lval.string= Str::copy(sdf_token.c_str());

    g_sdf_reader->set_parse_timing_check(false);

    /* No way to pass expr and id separately, so pass them together. */
    return EXPR_ID_CLOSE;
   }
   else {
    sdf_token += yytext;
   }
          
}

<COND_EXPR>{BLANK} {}

<COND_EXPR>.   { sdf_token += yytext[0]; }

{ID}   {
   Sdf_lval.string = Str::copy(yytext);
   return ID;
}

{ID}({HIERARCHICAL_CHAR}{ID})* {
   Sdf_lval.string = Str::copy(yytext);
   return PATH;
}

{END_OF_LINE}   { g_sdf_line++; }

{BLANK}   { /* Ignore blanks. */ }

   /* Send out of bound characters to parser. */
.   { return ((int) yytext[0]); }

%%
