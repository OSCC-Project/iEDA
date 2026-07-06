%{
#include <cstdlib>
#include <cstring>

#include "spef_parser.hpp"

static char* spef_strdup_text(const char* text)
{
  const auto len = std::strlen(text);
  auto* result = static_cast<char*>(std::malloc(len + 1));
  if (result != nullptr) {
    std::memcpy(result, text, len + 1);
  }
  return result;
}
%}

%option prefix="spef_"
%option noyywrap nounput noinput yylineno

DIGIT      [0-9]
NUMBER     [-+]?(({DIGIT}+\.?{DIGIT}*)|(\.{DIGIT}+))([Ee][-+]?{DIGIT}+)?
NAME_REF   \*{DIGIT}+(:[^ \t\r\n]+)?
NAME       [^ \t\r\n]+

%x BLOCK_COMMENT
%x SKIP_LINE

%%

"/*"                          { BEGIN(BLOCK_COMMENT); }
<BLOCK_COMMENT>"*/"           { BEGIN(INITIAL); }
<BLOCK_COMMENT>\n             { }
<BLOCK_COMMENT>.              { }
<SKIP_LINE>\n+                 { BEGIN(INITIAL); return EOL; }
<SKIP_LINE>[^\n]+              { }

"//"[^\n]*                    { }
[ \t\r]+                      { }
\n+                            { return EOL; }

"*D_NET"                      { return K_D_NET; }
"*NAME_MAP"                   { return K_NAME_MAP; }
"*PORTS"                      { return K_PORTS; }
"*CONN"                       { return K_CONN; }
"*CAP"                        { return K_CAP; }
"*RES"                        { return K_RES; }
"*END"                        { return K_END; }

"*SPEF"|"*DESIGN_FLOW"|"*DESIGN"|"*DATE"|"*VENDOR"|"*PROGRAM"|"*VERSION"|"*DIVIDER"|"*DELIMITER"|"*BUS_DELIMITER"|"*T_UNIT"|"*C_UNIT"|"*R_UNIT"|"*L_UNIT" {
                                spef_lval.str = spef_strdup_text(yytext);
                                return HEADER_KEY;
                              }
"*POWER_NETS"|"*GROUND_NETS" {
                                BEGIN(SKIP_LINE);
                              }
"*DEFINE"|"*PDEFINE"|"*VARIATION"|"*S_UNIT"|"*Q_UNIT" {
                                BEGIN(SKIP_LINE);
                              }

"*LL"                         { return K_LL; }
"*UR"                         { return K_UR; }
"*LAYER"                      { return K_LAYER; }
"*C"                          { return K_COORD; }
"*L"                          { return K_LOAD; }
"*D"                          { return K_DRIVE; }
"*S"|"*V"|"*PRES"|"*I"        {
                                if (std::strcmp(yytext, "*I") == 0) {
                                  spef_lval.str = spef_strdup_text(yytext);
                                  return CONN_TYPE;
                                }
                                return K_IGNORE_ATTR;
                              }
"*P"|"*N"                     {
                                spef_lval.str = spef_strdup_text(yytext);
                                return CONN_TYPE;
                              }

{NAME_REF}                    {
                                spef_lval.str = spef_strdup_text(yytext);
                                return NAME_REF;
                              }
{NUMBER}                      {
                                spef_lval.str = spef_strdup_text(yytext);
                                return NUMBER;
                              }
\"([^\"\\]|\\.)*\"            {
                                spef_lval.str = spef_strdup_text(yytext);
                                return QUOTED_STRING;
                              }
"I"|"O"|"B"                   {
                                spef_lval.str = spef_strdup_text(yytext);
                                return DIRECTION;
                              }
{NAME}                        {
                                spef_lval.str = spef_strdup_text(yytext);
                                return SPEF_NAME;
                              }

%%
