/* ***************************************************************************************
 * Copyright (c) 2023-2025 Peng Cheng Laboratory
 * Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
 * Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
 *
 * iEDA is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 * http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
 * ***************************************************************************************/
%{

#include <iostream>

#include "itf_parser.hpp"

#undef YY_DECL
#define YY_DECL int itf_lex(ITF_STYPE* yylval_param, ITF_LTYPE* yylloc_param)
#define YYSTYPE ITF_STYPE


#define YY_USER_ACTION                                    \
  yylloc_param->first_line = yylloc_param->last_line;     \
  yylloc_param->first_column = yylloc_param->last_column; \
  if (yylloc_param->last_line == yylineno)                \
    yylloc_param->last_column += yyleng;                  \
  else {                                                  \
    yylloc_param->last_line = yylineno;                   \
    yylloc_param->last_column = yytext + yyleng - strrchr(yytext, '\n'); \
  }


%}

%option prefix = "itf_"
%option noyywrap nounput yylineno

DIGIT [0-9]
ALPHA [a-zA-Z]
FLOAT [-+]?(({DIGIT}+\.?{DIGIT}*)|(\.{DIGIT}+))([Ee][-+]?{DIGIT}+)?
PUNCTUATION [={}(),]
COMMENT \$.*
KEYWORD {ALPHA}({ALPHA}|{DIGIT}|_)*
PROCESS_NAME {ALPHA}({ALPHA}|{DIGIT}|_|\.|\+|-)+
SKIP [ \r\n\t]

%%

"TECHNOLOGY"                    { return K_TECHNOLOGY; }
"PROCESS_FOUNDRY"               { return K_PROCESS_FOUNDRY; }
"PROCESS_NODE"                  { return K_PROCESS_NODE; }
"PROCESS_TYPE"                  { return K_PROCESS_TYPE; }
"PROCESS_VERSION"               { return K_PROCESS_VERSION; }
"PROCESS_CORNER"                { return K_PROCESS_CORNER; }
"REFERENCE_DIRECTION"           { return K_REFERENCE_DIRECTION; }
"GLOBAL_TEMPERATURE"            { return K_GLOBAL_TEMPERATURE; }
"BACKGROUND_ER"                 { return K_BACKGROUND_ER; }
"HALF_NODE_SCALE_FACTOR"        { return K_HALF_NODE_SCALE_FACTOR; }
"USE_SI_DENSITY"                { return K_USE_SI_DENSITY; }
"YES"                           { return K_YES; }
"NO"                            { return K_NO; }
"DROP_FACTOR_LATERAL_SPACING"   { return K_DROP_FACTOR_LATERAL_SPACING; }
"DIELECTRIC"                    { return K_DIELECTRIC; }
"DIELECTRIC_LAYER"              { return K_DIELECTRIC_LAYER; }
"ER"                            { return K_ER; }
"THICKNESS"                     { return K_THICKNESS; }
"MEASURED_FROM"                 { return K_MEASURED_FROM; }
"TOP_OF_CHIP"                   { return K_TOP_OF_CHIP; }
"SW_T"                          { return K_SW_T; }
"TW_T"                          { return K_TW_T; }
"ASSOCIATED_CONDUCTOR"          { return K_ASSOCIATED_CONDUCTOR; }
"IS_CONFORMAL"                  { return K_IS_CONFORMAL; }
"DAMAGE_THICKNESS"              { return K_DAMAGE_THICKNESS; }
"DAMAGE_ER"                     { return K_DAMAGE_ER; }
"CONDUCTOR"                     { return K_CONDUCTOR; }
"IS_PLANAR"                     { return K_IS_PLANAR; }
"WMIN"                          { return K_WMIN; }
"SMIN"                          { return K_SMIN; }
"AIR_GAP_VS_SPACING"            { return K_AIR_GAP_VS_SPACING; }
"SPACINGS"                      { return K_SPACINGS; }
"AIR_GAP_WIDTHS"                { return K_AIR_GAP_WIDTHS; }
"AIR_GAP_THICKNESSES"           { return K_AIR_GAP_THICKNESSES; }
"AIR_GAP_BOTTOM_HEIGHTS"        { return K_AIR_GAP_BOTTOM_HEIGHTS; }
"BOTTOM_DIELECTRIC_THICKNESS"   { return K_BOTTOM_DIELECTRIC_THICKNESS; }
"BOTTOM_DIELECTRIC_ER"          { return K_BOTTOM_DIELECTRIC_ER; }
"BOTTOM_THICKNESS_VS_SI_WIDTH"  { return K_BOTTOM_THICKNESS_VS_SI_WIDTH; }
"RESISTIVE_ONLY"                { return K_RESISTIVE_ONLY; }
"CAPACITIVE_ONLY"               { return K_CAPACITIVE_ONLY; }
"T0"                            { return K_T0; }
"CRT1"                          { return K_CRT1; }
"CRT2"                          { return K_CRT2; }
"DROP_FACTOR"                   { return K_DROP_FACTOR; }
"ETCH"                          { return K_ETCH; }
"CAPACITIVE_ONLY_ETCH"          { return K_CAPACITIVE_ONLY_ETCH; }
"RESISTIVE_ONLY_ETCH"           { return K_RESISTIVE_ONLY_ETCH; }
"ETCH_VS_WIDTH_AND_SPACING"     { return K_ETCH_VS_WIDTH_AND_SPACING; }
"WIDTHS"                        { return K_WIDTHS; }
"VALUES"                        { return K_VALUES; }
"ETCH_FROM_TOP"                 { return K_ETCH_FROM_TOP; }
"FILL_RATIO"                    { return K_FILL_RATIO; }
"FILL_SPACING"                  { return K_FILL_SPACING; }
"FILL_WIDTH"                    { return K_FILL_WIDTH; }
"FILL_TYPE"                     { return K_FILL_TYPE; }
"GROUNDED"                      { return K_GROUNDED; }
"FLOATING"                      { return K_FLOATING; }
"GATE_TO_CONTACT_SMIN"          { return K_GATE_TO_CONTACT_SMIN; }
"GATE_TO_DIFFUSION_CAP"         { return K_GATE_TO_DIFFUSION_CAP; }
"NUMBER_OF_TABLES"              { return K_NUMBER_OF_TABLES; }
"CONTACT_TO_CONTACT_SPACINGS"   { return K_CONTACT_TO_CONTACT_SPACINGS; }
"GATE_TO_CONTACT_SPACINGS"      { return K_GATE_TO_CONTACT_SPACINGS; }
"CAPS_PER_MICRON"               { return K_CAPS_PER_MICRON; }
"THICKNESS_CHANGES"             { return K_THICKNESS_CHANGES; }
"LAYER_TYPE"                    { return K_LAYER_TYPE; }
"DENSITY_POLYNOMIAL_ORDERS"     { return K_DENSITY_POLYNOMIAL_ORDERS; }
"WIDTH_POLYNOMIAL_ORDERS"       { return K_WIDTH_POLYNOMIAL_ORDERS; }
"WIDTH_RANGES"                  { return K_WIDTH_RANGES; }
"POLYNOMIAL_COEFFICIENTS"       { return K_POLYNOMIAL_COEFFICIENTS; }
"RPSQ"                          { return K_RPSQ; }
"RHO"                           { return K_RHO; }
"RPSQ_VS_SI_WIDTH"              { return K_RPSQ_VS_SI_WIDTH; }
"RPSQ_VS_WIDTH_AND_SPACING"     { return K_RPSQ_VS_WIDTH_AND_SPACING; }
"WIDTH"                         { return K_WIDTH; }
"SIDE_TANGENT"                  { return K_SIDE_TANGENT; }
"THICKNESS_VS_DENSITY"          { return K_THICKNESS_VS_DENSITY; }
"VIA"                           { return K_VIA; }
"FROM"                          { return K_FROM; }
"TO"                            { return K_TO; }
"CRT_VS_AREA"                   { return K_CRT_VS_AREA; }
"RPV"                           { return K_RPV; }
"AREA"                          { return K_AREA; }
"RPV_VS_AREA"                   { return K_RPV_VS_AREA; }
"ETCH_VS_WIDTH_AND_LENGTH"      { return K_ETCH_VS_WIDTH_AND_LENGTH; }
"VARIATION_PARAMETERS"          { return K_VARIATION_PARAMETERS; }
"DENSITY_BOX_WEIGHTING_FACTOR"  { return K_DENSITY_BOX_WEIGHTING_FACTOR; }
"ILD_VS_WIDTH_AND_SPACING"      { return K_ILD_VS_WIDTH_AND_SPACING; }
"DELTAPD"                       { return K_DELTAPD; }
"LENGTHS"                       { return K_LENGTHS; }
"RHO_VS_SI_WIDTH_AND_THICKNESS" { return K_RHO_VS_SI_WIDTH_AND_THICKNESS; }
"RHO_VS_WIDTH_AND_SPACING"      { return K_RHO_VS_WIDTH_AND_SPACING; }
"CRT_VS_SI_WIDTH"               { return K_CRT_VS_SI_WIDTH; }
"POLYNOMIAL_BASED_THICKNESS_VARIATION"  { return K_POLYNOMIAL_BASED_THICKNESS_VARIATION; }
"THICKNESS_VS_WIDTH_AND_SPACING"        { return K_THICKNESS_VS_WIDTH_AND_SPACING; }
"TVF_ADJUSTMENT_TABLES"                 { return K_TVF_ADJUSTMENT_TABLES; }
"BOTTOM_THICKNESS_VS_WIDTH_AND_SPACING" { return K_BOTTOM_THICKNESS_VS_WIDTH_AND_SPACING; }
"BOTTOM_THICKNESS_VS_WIDTH_AND_DELTAPD" { return K_BOTTOM_THICKNESS_VS_WIDTH_AND_DELTAPD; }
"ETCH_VS_CONTACT_AND_GATE_SPACINGS"     { return K_ETCH_VS_CONTACT_AND_GATE_SPACINGS; }
"DENSITY_BOUNDS_VS_WIDTH"               { return K_DENSITY_BOUNDS_VS_WIDTH; }
"THICKNESS_BOUNDS"                      { return K_THICKNESS_BOUNDS; }
"DEVICE_TYPE"                           { return K_DEVICE_TYPE; }
"PARALLEL_TO_REFERENCE"                 { return K_PARALLEL_TO_REFERENCE; }
"PERPENDICULAR_TO_REFERENCE"            { return K_PERPENDICULAR_TO_REFERENCE; }
"PARALLEL_TO_GATE"                      { return K_PARALLEL_TO_GATE; }
"BW_T"                                  { return K_BW_T; }
"LINKED_TO"                             { return K_LINKED_TO; }
"EXTENSIONMIN"                          { return K_EXTENSIONMIN; }
"RAISED_DIFFUSION_THICKNESS"            { return K_RAISED_DIFFUSION_THICKNESS; }
"RAISED_DIFFUSION_TO_GATE_SMIN"         { return K_RAISED_DIFFUSION_TO_GATE_SMIN; }
"MULTIGATE"                             { return K_MULTIGATE; }
"FIN_SPACING"                           { return K_FIN_SPACING; }
"FIN_WIDTH"                             { return K_FIN_WIDTH; }
"FIN_LENGTH"                            { return K_FIN_LENGTH; }
"FIN_THICKNESS"                         { return K_FIN_THICKNESS; }
"GATE_OXIDE_TOP_T"                      { return K_GATE_OXIDE_TOP_T; }
"GATE_OXIDE_SIDE_T"                     { return K_GATE_OXIDE_SIDE_T; }
"GATE_OXIDE_ER"                         { return K_GATE_OXIDE_ER; }
"GATE_POLY_TOP_T"                       { return K_GATE_POLY_TOP_T; }
"GATE_POLY_SIDE_T"                      { return K_GATE_POLY_SIDE_T; }
"CHANNEL_ER"                            { return K_CHANNEL_ER; }
"RAISED_DIFFUSION_GROWTH"               { return K_RAISED_DIFFUSION_GROWTH; }
"GATE_DIFFUSION_LAYER_PAIR"             { return K_GATE_DIFFUSION_LAYER_PAIR; }
"RPV_VS_WIDTH_AND_LENGTH"               { return K_RPV_VS_WIDTH_AND_LENGTH; }
"RAISED_DIFFUSION_ETCH"                 { return K_RAISED_DIFFUSION_ETCH; }
"RAISED_DIFFUSION_GATE_SIDE_CONFORMAL_ER" { return K_RAISED_DIFFUSION_GATE_SIDE_CONFORMAL_ER; }

{COMMENT}       { }
{PUNCTUATION}   { return yytext[0]; }
{KEYWORD}       { yylval_param->string = yytext; return KEYWORD; }
{PROCESS_NAME}  { yylval_param->string = yytext; return PROCESS_NAME; }
{FLOAT}         { yylval_param->dval = std::atof(yytext); return NUMBER; }
{SKIP}          { }

%%

void itf_error(const char* s) {
    printf("error: %s\n yytext =%s", s, yytext);
}
