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
%code requires {

#include <string.h>

#include <iostream>
#include <string>
#include <vector>

#include "itfrData.hpp"
#include "itfMarco.h"
#include "itfrSettings.hpp"
#include "itfrCallBacks.hpp"

}

%code provides {

#undef YY_DECL
#define YY_DECL int itf_lex(ITF_STYPE* yylval_param, ITF_LTYPE* yylloc_param)
YY_DECL;

void itf_error(ITF_LTYPE*, const char*);

}

%union {
  double  dval;
  char*   string;
  void*   ent;
}

%define api.pure full
%define api.prefix {itf_}
%locations

%{

namespace itf {

#define CALLBACK(func, typ, data) \
  if (func) { \
    (*func) (typ, data, itfSettings->user_data); \
  }

extern itfrData* itfData;
// var
char* v_layer_name = nullptr;
char* v_measured_from = nullptr;
char* v_model_name = nullptr;
char* v_table_name = nullptr;
char* v_etch_effect_type = nullptr;
float v_thickness;
float v_sw_t;
float v_tw_t;
float v_t0;
float v_crt1;
float v_crt2;
float v_rho;
int v_number_of_tables;
std::vector<int> v_int_list;
std::vector<float> v_float_list;
std::vector<std::pair<float, float>> v_float_pair_list;
itf2DLUT<float, float, float> v_lut;
itf2DLUT<float, float, std::pair<float, float>> v_etch_wlv_lut;
itfiVPT v_vpt;
unsigned v_is_lut_working = 0;
unsigned v_flag_dielectric = 0;
%}

%token K_TECHNOLOGY K_PROCESS_FOUNDRY K_GLOBAL_TEMPERATURE K_BACKGROUND_ER K_HALF_NODE_SCALE_FACTOR
%token K_PROCESS_NODE K_PROCESS_TYPE K_PROCESS_VERSION K_PROCESS_CORNER K_REFERENCE_DIRECTION
%token K_USE_SI_DENSITY K_YES K_NO K_DROP_FACTOR_LATERAL_SPACING K_DIELECTRIC 
%token K_DIELECTRIC_LAYER K_ER K_THICKNESS K_MEASURED_FROM K_TOP_OF_CHIP K_SW_T K_TW_T 
%token K_ASSOCIATED_CONDUCTOR K_IS_CONFORMAL K_DAMAGE_THICKNESS K_DAMAGE_ER
%token K_CONDUCTOR K_IS_PLANAR K_WMIN K_SMIN K_AIR_GAP_VS_SPACING K_SPACINGS 
%token K_AIR_GAP_WIDTHS K_AIR_GAP_THICKNESSES K_AIR_GAP_BOTTOM_HEIGHTS
%token K_BOTTOM_DIELECTRIC_THICKNESS K_BOTTOM_DIELECTRIC_ER 
%token K_BOTTOM_THICKNESS_VS_SI_WIDTH K_RESISTIVE_ONLY K_CAPACITIVE_ONLY K_T0
%token K_CRT1 K_CRT2 K_DROP_FACTOR K_ETCH K_CAPACITIVE_ONLY_ETCH K_RESISTIVE_ONLY_ETCH
%token K_ETCH_VS_WIDTH_AND_SPACING K_WIDTHS K_VALUES K_ETCH_FROM_TOP K_FILL_RATIO
%token K_FILL_SPACING K_FILL_WIDTH K_FILL_TYPE K_GROUNDED K_FLOATING K_GATE_TO_CONTACT_SMIN
%token K_GATE_TO_DIFFUSION_CAP K_NUMBER_OF_TABLES K_CONTACT_TO_CONTACT_SPACINGS
%token K_GATE_TO_CONTACT_SPACINGS K_CAPS_PER_MICRON K_THICKNESS_CHANGES K_LAYER_TYPE 
%token K_POLYNOMIAL_BASED_THICKNESS_VARIATION K_DENSITY_POLYNOMIAL_ORDERS
%token K_WIDTH_POLYNOMIAL_ORDERS K_WIDTH_RANGES K_POLYNOMIAL_COEFFICIENTS
%token K_RPSQ K_RHO K_RPSQ_VS_SI_WIDTH K_RPSQ_VS_WIDTH_AND_SPACING
%token K_WIDTH K_SIDE_TANGENT K_THICKNESS_VS_DENSITY K_THICKNESS_VS_WIDTH_AND_SPACING
%token K_TVF_ADJUSTMENT_TABLES K_BOTTOM_THICKNESS_VS_WIDTH_AND_SPACING 
%token K_BOTTOM_THICKNESS_VS_WIDTH_AND_DELTAPD K_VIA K_FROM K_TO K_CRT_VS_AREA
%token K_RPV K_AREA K_RPV_VS_AREA K_ETCH_VS_WIDTH_AND_LENGTH K_VARIATION_PARAMETERS
%token K_DENSITY_BOX_WEIGHTING_FACTOR K_ILD_VS_WIDTH_AND_SPACING K_DELTAPD K_LENGTHS
%token K_RHO_VS_SI_WIDTH_AND_THICKNESS K_RHO_VS_WIDTH_AND_SPACING K_ETCH_VS_CONTACT_AND_GATE_SPACINGS
%token K_CRT_VS_SI_WIDTH K_DENSITY_BOUNDS_VS_WIDTH K_THICKNESS_BOUNDS
%token K_DEVICE_TYPE K_PARALLEL_TO_REFERENCE K_PERPENDICULAR_TO_REFERENCE K_PARALLEL_TO_GATE
%token K_BW_T K_LINKED_TO K_EXTENSIONMIN K_RAISED_DIFFUSION_THICKNESS K_RAISED_DIFFUSION_TO_GATE_SMIN
%token K_MULTIGATE K_FIN_SPACING K_FIN_WIDTH K_FIN_LENGTH K_FIN_THICKNESS K_GATE_OXIDE_TOP_T 
%token K_GATE_OXIDE_SIDE_T K_GATE_OXIDE_ER K_GATE_POLY_TOP_T K_GATE_POLY_SIDE_T K_CHANNEL_ER 
%token K_RAISED_DIFFUSION_GROWTH K_GATE_DIFFUSION_LAYER_PAIR K_RPV_VS_WIDTH_AND_LENGTH
%token K_RAISED_DIFFUSION_ETCH K_RAISED_DIFFUSION_GATE_SIDE_CONFORMAL_ER

%token <string> KEYWORD PROCESS_NAME
%token <dval> NUMBER

%start itf_file

%%

itf_file :
  file_optionals
  layers
  variation_params
;

file_optionals :
  file_optionals file_optional
|
;

file_optional :
  tech
  {
    CALLBACK(itfCallbacks->technology_cb, itfCallBackType::kTechnologyCbType, itfData->process_name);
  }
| global_temperature
  {
    CALLBACK(itfCallbacks->global_temperature_cb, itfCallBackType::kGlobalTemperatureCbType, itfData->global_temperature);
  }
| background_er
  {
    CALLBACK(itfCallbacks->background_er_cb, itfCallBackType::kBackgroundErCbType, itfData->background_er);
  }
| half_node_scale_factor
  {
    CALLBACK(itfCallbacks->half_node_scale_factor_cb, itfCallBackType::kHalfNodeScaleFactorCbType, itfData->half_node_scale_factor);
  }
| use_si_density
  {
    CALLBACK(itfCallbacks->use_si_density_cb, itfCallBackType::kUseSiDensityCbType, itfData->use_si_density);
  }
| drop_factor_lateral_spacing
  {
    CALLBACK(itfCallbacks->drop_factor_lateral_spacing_cb, itfCallBackType::kDropFactorLateralSpacingCbType, itfData->drop_factor_lateral_spacing);
  }
| process_foundry
  {
    CALLBACK(itfCallbacks->process_foundry_cb, itfCallBackType::kProcessFoundryCbType, itfData->process_foundry);
  }
| process_node
  {
    CALLBACK(itfCallbacks->process_node_cb, itfCallBackType::kProcessNodeCbType, itfData->process_node);
  }
| process_type
  {
    CALLBACK(itfCallbacks->process_type_cb, itfCallBackType::kProcessTypeCbType, itfData->process_type);
  }
| process_version
  {
    CALLBACK(itfCallbacks->process_version_cb, itfCallBackType::kProcessVersionCbType, itfData->process_version);
  }
| process_corner
  {
    CALLBACK(itfCallbacks->process_corner_cb, itfCallBackType::kProcessCornerCbType, itfData->process_corner);
  }
| reference_direction
  {
    CALLBACK(itfCallbacks->reference_direction_cb, itfCallBackType::kReferenceDirectionCbType, itfData->reference_direction);
  }
;

tech :
  K_TECHNOLOGY '=' PROCESS_NAME
  {
    ITF_STR_CPY(itfData->process_name, $3);
  }
| K_TECHNOLOGY '=' KEYWORD
  {
    ITF_STR_CPY(itfData->process_name, $3);
  }
;

global_temperature :
  K_GLOBAL_TEMPERATURE '=' NUMBER
  {
    itfData->has_global_temperature = 1;
    itfData->global_temperature = $3;
  }
;

background_er :
  K_BACKGROUND_ER '=' NUMBER
  {
    itfData->has_background_er = 1;
    itfData->background_er = $3;
  }
;

half_node_scale_factor :
  K_HALF_NODE_SCALE_FACTOR '=' NUMBER
  {
    itfData->has_half_node_scale_factor = 1;
    itfData->half_node_scale_factor = $3;
  }
;

use_si_density :
  K_USE_SI_DENSITY '=' use_si_density_value
  {
    itfData->has_use_si_density = 1;
  }
;

use_si_density_value :
  K_YES { itfData->use_si_density = 1; }
| K_NO  { itfData->use_si_density = 0; }
;

drop_factor_lateral_spacing :
  K_DROP_FACTOR_LATERAL_SPACING '=' NUMBER
  {
    itfData->has_drop_factor_lateral_spacing = 1;
    itfData->drop_factor_lateral_spacing = $3;
  }
;

process_foundry :
  K_PROCESS_FOUNDRY '=' KEYWORD
;

process_node :
  K_PROCESS_NODE '=' NUMBER
;

process_type :
  K_PROCESS_TYPE '=' KEYWORD
;

process_version :
  K_PROCESS_VERSION '=' NUMBER
;

process_corner :
  K_PROCESS_CORNER '=' KEYWORD
;

reference_direction :
  K_REFERENCE_DIRECTION '=' KEYWORD
;

layers:
  x_eol
  vias
;

x_eol :
  x_eol dielectric
  {
    CALLBACK(itfCallbacks->dielectric_cb, itfCallBackType::kDielectricCbType, &itfData->dielectric);
    itfData->dielectric.clear();
  }
| x_eol conductor
  {
    CALLBACK(itfCallbacks->conductor_cb, itfCallBackType::kConductorCbType, &itfData->conductor);
    itfData->conductor.clear();
  }
| x_eol multigate
|
;

dielectric :
  K_DIELECTRIC
  {
    v_flag_dielectric = 1;
  }
  layer_name
  {
    itfData->dielectric.set_dielectric_name(v_layer_name);
  }
  '{' 
    dielectric_properties
  '}'
  {
    v_flag_dielectric = 0;
  }
;

layer_name :
  KEYWORD { ITF_STR_CPY(v_layer_name, $1); }
;

dielectric_properties :
  dielectric_properties dielectric_property
| 
;

dielectric_property:
  er
| thickness             { itfData->dielectric.set_thickness(v_thickness); }
| diel_measured_from    
| associated_conductor
  {
    itfData->dielectric.set_associated_conductor(v_layer_name);
  }
| damage
| K_IS_CONFORMAL
  {
    itfData->dielectric.set_is_conformal();
  }
| bw_t
;

er :
  K_ER '=' NUMBER
  { itfData->dielectric.set_er($3); }
;

thickness :
  K_THICKNESS '=' NUMBER
  { v_thickness = $3; }
;

diel_measured_from :
  measured_from { itfData->dielectric.set_measured_from(v_measured_from); }
| sw_t          { itfData->dielectric.set_sw_t(v_sw_t); }
| tw_t          { itfData->dielectric.set_tw_t(v_tw_t); }
;

measured_from :
  K_MEASURED_FROM '=' measured_from_value
;

measured_from_value :
  layer_name    { ITF_STR_CPY(v_measured_from, v_layer_name); }
| K_TOP_OF_CHIP { ITF_STR_CPY(v_measured_from, "TOP_OF_CHIP"); }
;

sw_t :
  K_SW_T '=' NUMBER { v_sw_t = $3; }
;

tw_t :
  K_TW_T '=' NUMBER { v_tw_t = $3; }
;

associated_conductor :
  K_ASSOCIATED_CONDUCTOR '=' layer_name
;

damage :
  K_DAMAGE_THICKNESS '=' NUMBER
  K_DAMAGE_ER '=' NUMBER
  {
    itfData->dielectric.set_damage_thickness($3);
    itfData->dielectric.set_damage_er($6);
  }
;

bw_t :
  K_BW_T '=' NUMBER
  {

  }
;

conductor :
  K_CONDUCTOR layer_name
  {
    itfData->conductor.set_conductor_name(v_layer_name);
  }
  '{'
    conductor_properties
  '}'
;

conductor_properties :
  conductor_properties conductor_property
|
;

conductor_property :
  wmin
| smin
| thickness { itfData->conductor.set_thickness(v_thickness); }
| air_gap_vs_spacing
| bottom_dielectric_property
| bottom_thickness_vs_si_width
| t0 { itfData->conductor.set_t0(v_t0); }
| crt_stmt
| density_box_weight_factor
| drop_factor
| etch_stmt
| etch_vs_width_and_spacing
  {
    itfData->conductor.add_etch_vws(v_etch_effect_type, v_lut);
    ITF_FREE(v_etch_effect_type);
    v_lut.clear();
    v_is_lut_working = 0;
  }
| fill_stmt
| gate_to_contact_smin
| gate_to_diffusion_cap
| ild_vs_width_and_spacing
| K_IS_PLANAR { itfData->conductor.set_is_planar(); }
| layer_type
| measured_from { itfData->conductor.set_measured_from(v_measured_from); }
| polynomial_based_thickness_variation
| rpsq_stmt
| side_tangent
| thickness_vs_density
| thickness_vs_width_and_spacing
| tvf_adjustment_tables
| device_type
| associated_conductor { }
| linked_to
| extensionmin
| raised_diffusion_thickness
| raised_diffusion_to_gate_smin
| raised_diffusion_etch
| raised_diffusion_gate_side_conformal_er
;

wmin :
  K_WMIN '=' NUMBER
  {
    itfData->conductor.set_wmin($3);
  }
;

smin :
  K_SMIN '=' NUMBER
  {
    itfData->conductor.set_smin($3);
  }
;

air_gap_vs_spacing :
  K_AIR_GAP_VS_SPACING '{'
    spacings
    air_gap_widths
    air_gap_thicknesses
    air_gap_bottom_heights
  '}'
;

spacings :
  K_SPACINGS '{' float_list '}'
  {
    if (v_is_lut_working) {
      v_lut.set_data_list<float>("SPACINGS", v_float_list);
    } else {
      itfData->conductor.set_air_gap_spacings(v_float_list);
    }
    v_float_list.clear();
  }
;

float_list :
  float_list NUMBER
  {
    v_float_list.push_back($2);
  }
| 
;

air_gap_widths :
  K_AIR_GAP_WIDTHS '{' float_list '}'
  {
    itfData->conductor.set_air_gap_widths(v_float_list);
    v_float_list.clear();
  }
;

air_gap_thicknesses :
  K_AIR_GAP_THICKNESSES '{' float_list '}'
  {
    itfData->conductor.set_air_gap_thicknesses(v_float_list);
    v_float_list.clear();
  }
;

air_gap_bottom_heights :
  K_AIR_GAP_BOTTOM_HEIGHTS '{' float_list '}' 
  {
    itfData->conductor.set_air_gap_bottom_heights(v_float_list);
    v_float_list.clear();
  }
;

bottom_dielectric_property :
  K_BOTTOM_DIELECTRIC_THICKNESS '=' NUMBER
  K_BOTTOM_DIELECTRIC_ER '=' NUMBER
  {
    itfData->conductor.set_bottom_dielectric_thickness($3);
    itfData->conductor.set_bottom_dielectric_er($6);
  }
;

bottom_thickness_vs_si_width :
  K_BOTTOM_THICKNESS_VS_SI_WIDTH bottom_thickness_vs_si_width_type
  '{' float_pair_list '}'
  {
    itfData->conductor.get_bottom_thickness_vs_si_width().set_sr_list(v_float_pair_list);
    v_float_pair_list.clear();
  }
;

bottom_thickness_vs_si_width_type :
  K_RESISTIVE_ONLY
  {
    itfData->conductor.get_bottom_thickness_vs_si_width().set_type("RESISTIVE_ONLY");
  }
| K_CAPACITIVE_ONLY
  {
    itfData->conductor.get_bottom_thickness_vs_si_width().set_type("CAPACITIVE_ONLY");
  }
;

float_pair_list :
  float_pair_list '(' NUMBER ',' NUMBER ')'
  { v_float_pair_list.push_back({$3, $5}); }
| float_pair_list '(' NUMBER     NUMBER ')'
  { v_float_pair_list.push_back({$3, $4}); }
|
;

t0 :
  K_T0 '=' NUMBER
  {
    v_t0 = $3;
  }
;

crt_stmt :
  crt1  { itfData->conductor.set_crt1(v_crt1); }
| crt2  { itfData->conductor.set_crt2(v_crt2); }
| crt_vs_si_width
;

crt1 :
  K_CRT1 '=' NUMBER
  {
    v_crt1 = $3;
  }
;

crt2 :
  K_CRT2 '=' NUMBER
  {
    v_crt2 = $3;
  }
;

crt_vs_si_width :
  K_CRT_VS_SI_WIDTH
  '{'  
    siw_crt1_crt2_list
  '}'
;

siw_crt1_crt2_list :
  siw_crt1_crt2_list '(' NUMBER ',' NUMBER ',' NUMBER ')'
  {
    itfData->conductor.add_siw_crt1_crt2($3, $5, $7);
  }
| siw_crt1_crt2_list '(' NUMBER NUMBER NUMBER ')'
  {
    itfData->conductor.add_siw_crt1_crt2($3, $4, $5);
  }
|
;

density_box_weight_factor : 
  K_DENSITY_BOX_WEIGHTING_FACTOR '{'
    s_w_list
  '}'
;

s_w_list :
  s_w_list '(' NUMBER ',' NUMBER ')'
  {
    itfData->conductor.add_density_box_weight(int($3), $5);
  }
| s_w_list '(' NUMBER NUMBER ')'
  {
    itfData->conductor.add_density_box_weight(int($3), $4);
  }
|
;

drop_factor :
  K_DROP_FACTOR '=' NUMBER
  {
    itfData->conductor.set_drop_factor($3);
  }
;

etch_stmt :
  K_ETCH '=' NUMBER
  { itfData->conductor.set_etch($3); }
| K_CAPACITIVE_ONLY_ETCH '=' NUMBER
  { itfData->conductor.set_capacitive_only_etch($3); }
| K_RESISTIVE_ONLY_ETCH '=' NUMBER
  { itfData->conductor.set_resistive_only_etch($3); }
;

etch_vs_width_and_spacing :
  K_ETCH_VS_WIDTH_AND_SPACING 
  {
    v_lut.set_names("WIDTHS", "SPACINGS", "VALUES");
    v_is_lut_working = 1;
  }
  etch_effect_type
  apply_etch_to
  '{'
    spacings
    widths
    values
  '}'
;

etch_effect_type :
  K_RESISTIVE_ONLY  { ITF_STR_CPY(v_etch_effect_type, "RESISTIVE_ONLY"); }
| K_CAPACITIVE_ONLY { ITF_STR_CPY(v_etch_effect_type, "CAPACITIVE_ONLY"); }
| K_ETCH_FROM_TOP   { ITF_STR_CPY(v_etch_effect_type, "ETCH_FROM_TOP"); }
|
;

apply_etch_to :
  K_PARALLEL_TO_REFERENCE
| K_PERPENDICULAR_TO_REFERENCE
| K_PARALLEL_TO_GATE
|
;

widths :
  K_WIDTHS   '{' float_list '}'
  {
    v_lut.set_data_list<float>("WIDTHS", v_float_list);
    v_float_list.clear();
  }
;

values :
  K_VALUES   '{' float_list '}'
  {
    v_lut.set_data_list<float>("VALUES", v_float_list);
    v_float_list.clear();
  }
;

fill_stmt :
  K_FILL_RATIO    '=' NUMBER
  { itfData->conductor.set_fill_ratio($3); }
| K_FILL_SPACING  '=' NUMBER
  { itfData->conductor.set_fill_spacing($3); }
| K_FILL_WIDTH    '=' NUMBER
  { itfData->conductor.set_fill_width($3); }
| K_FILL_TYPE     '=' fill_type
;

fill_type :
  K_GROUNDED
  { itfData->conductor.set_fill_type("GROUNDED"); }
| K_FLOATING
  { itfData->conductor.set_fill_type("FLOATING"); }
;

gate_to_contact_smin :
  K_GATE_TO_CONTACT_SMIN '=' NUMBER
  {
    itfData->conductor.set_gate_to_contact_smin($3);
  }
;

gate_to_diffusion_cap :
  K_GATE_TO_DIFFUSION_CAP '{'
    number_of_tables
    model_list
  '}'
  {
    itfData->conductor.get_gate_to_diffusion_cap().set_number_of_tables(v_number_of_tables);
  }
;

number_of_tables :
  K_NUMBER_OF_TABLES '=' NUMBER
  {
    v_number_of_tables = int($3);
  }
;

model_list :
  model_list model
|
;

model :
  model_name
  {
    v_lut.set_names("GATE_TO_CONTACT_SPACINGS", "CONTACT_TO_CONTACT_SPACINGS", "CAPS_PER_MICRON");
    v_is_lut_working = 1;
  }
  '{'
    contact_to_contact_spacings    
    gate_to_contact_spacings
    caps_per_micron
  '}'
  {
    itfData->conductor.get_gate_to_diffusion_cap().add_model(v_model_name, v_lut);
    v_lut.clear();
    v_is_lut_working = 0;
  }
;

model_name :
  KEYWORD   { ITF_STR_CPY(v_model_name, $1); }
;

contact_to_contact_spacings :
  K_CONTACT_TO_CONTACT_SPACINGS '{' float_list '}'
  {
    v_lut.set_data_list<float>("CONTACT_TO_CONTACT_SPACINGS", v_float_list);
    v_float_list.clear();
  }
;

gate_to_contact_spacings :
  K_GATE_TO_CONTACT_SPACINGS '{' float_list '}'
  {
    v_lut.set_data_list<float>("GATE_TO_CONTACT_SPACINGS", v_float_list);
    v_float_list.clear();
  }
;

caps_per_micron :
  K_CAPS_PER_MICRON '{' float_list '}'
  {
    v_lut.set_data_list<float>("CAPS_PER_MICRON", v_float_list);
    v_float_list.clear();
  }
;

ild_vs_width_and_spacing :
  K_ILD_VS_WIDTH_AND_SPACING
  {
    v_lut.set_names("WIDTHS", "SPACINGS", "THICKNESS_CHANGES");
    v_is_lut_working = 1;
  }
  '{'
    ild_diel_layer_token '=' layer_name
    widths
    spacings
    thickness_changes
  '}'
  {
    itfData->conductor.set_ild_vws_title(v_layer_name);
    itfData->conductor.set_ild_vws_lut(v_lut);
    v_lut.clear();
    v_is_lut_working = 0;
  }
;

ild_diel_layer_token :
  K_DIELECTRIC
| K_DIELECTRIC_LAYER
;

thickness_changes :
  K_THICKNESS_CHANGES '{' float_list '}'
  {
    v_lut.set_data_list<float>("THICKNESS_CHANGES", v_float_list);
    v_float_list.clear();
  }
;

layer_type :
  K_LAYER_TYPE '=' layer_type_value
;

layer_type_value :
  KEYWORD { itfData->conductor.set_layer_type($1); }
;

polynomial_based_thickness_variation :
  K_POLYNOMIAL_BASED_THICKNESS_VARIATION '{'
    density_polynomial_orders
    width_polynomial_orders
    width_ranges
    polynomial_valueicients_lists
    polynomial_based_thickness_variation_options
  '}'
;

polynomial_based_thickness_variation_options :
  polynomial_based_thickness_variation_options
  polynomial_based_thickness_variation_option
|
;

polynomial_based_thickness_variation_option :
  density_bounds_vs_width
| thickness_bounds
;

density_polynomial_orders :
  K_DENSITY_POLYNOMIAL_ORDERS equal_op '{' int_comma_list   '}'
  {
    itfData->conductor.get_PBTV().set_density_polynomial_order(v_int_list);
    v_int_list.clear();
  }
;

equal_op :
  '='
|
;

width_polynomial_orders :
  K_WIDTH_POLYNOMIAL_ORDERS equal_op '{' int_comma_list   '}'
  {
    itfData->conductor.get_PBTV().set_width_polynomial_order(v_int_list);
    v_int_list.clear();
  }
;

width_ranges :
  K_WIDTH_RANGES equal_op '{' float_comma_list '}'
  {
    itfData->conductor.get_PBTV().set_width_range(v_float_list);
    v_float_list.clear();
  }
;

int_comma_list :
  int_comma_list NUMBER ','  { v_int_list.push_back(int($2)); }
| int_comma_list NUMBER      { v_int_list.push_back(int($2)); }
| 
;

float_comma_list :
  float_comma_list NUMBER ',' { v_float_list.push_back($2); }
| float_comma_list NUMBER     { v_float_list.push_back($2); }
|
;

polynomial_valueicients_lists :
  polynomial_valueicients_lists
  K_POLYNOMIAL_COEFFICIENTS equal_op '{'
    float_comma_list
  '}'
  {
    itfData->conductor.get_PBTV().add_polynomial_coefficients(v_float_list);
    v_float_list.clear();
  }
| 
;

density_bounds_vs_width :
  K_DENSITY_BOUNDS_VS_WIDTH '{'
    tuple_3_list
  '}'
;

tuple_3_list :
  tuple_3_list
  '(' NUMBER NUMBER NUMBER ')'
  {

  }
|
;

thickness_bounds :
  K_THICKNESS_BOUNDS '{' NUMBER NUMBER '}'
  {

  }
;

rpsq_stmt :
  K_RPSQ '=' NUMBER { itfData->conductor.set_rpsq($3); }
| rho               { itfData->conductor.set_rho(v_rho); }
| rpsq_vs_si_width
| rpsq_vs_width_and_spacing
| rho_vs_si_width_and_thickness
| rho_vs_width_and_spacing
;

rho :
  K_RHO '=' NUMBER { v_rho = $3; }
;

rpsq_vs_si_width :
  K_RPSQ_VS_SI_WIDTH '{'
    float_pair_list
  '}'
  {
    itfData->conductor.set_rpsq_vs_si_width(v_float_pair_list);
    v_float_pair_list.clear();
  }
;

rpsq_vs_width_and_spacing :
  K_RPSQ_VS_WIDTH_AND_SPACING
  {
    v_lut.set_names("WIDTHS", "SPACINGS", "VALUES");
    v_is_lut_working = 1;
  }
  '{'
    spacings
    widths
    values
  '}'
  {
    itfData->conductor.set_rpsq_vws(v_lut);
    v_lut.clear();
    v_is_lut_working = 0;
  }
;

rho_vs_si_width_and_thickness :
  K_RHO_VS_SI_WIDTH_AND_THICKNESS
  {
    v_lut.set_names("THICKNESS", "WIDTH", "VALUES");
    v_is_lut_working = 1;
  }
  '{'
    width_list
    thickness_list
    values
  '}'
  {
    itfData->conductor.set_rho_v_siw_t(v_lut);
    v_lut.clear();
    v_is_lut_working = 0;
  }
;

width_list :
  K_WIDTH '{' float_list '}'
  {
    v_lut.set_data_list<float>("WIDTH", v_float_list);
    v_float_list.clear();
  }
;

thickness_list :
  K_THICKNESS '{' float_list '}'
  {
    v_lut.set_data_list<float>("THICKNESS", v_float_list);
    v_float_list.clear();
  }
;

rho_vs_width_and_spacing :
  K_RHO_VS_WIDTH_AND_SPACING
  {
    v_lut.set_names("WIDTHS", "SPACINGS", "VALUES");
    v_is_lut_working = 1;
  }
  '{'
    spacings
    widths
    values
  '}'
  {
    itfData->conductor.set_rho_vws(v_lut);
    v_lut.clear();
    v_is_lut_working = 0;
  }
;

side_tangent :
  K_SIDE_TANGENT '=' NUMBER
  { itfData->conductor.set_side_tangent($3); }
;

thickness_vs_density :
  K_THICKNESS_VS_DENSITY thickness_vs_density_type
  '{' float_pair_list '}'
  {
    itfData->conductor.get_thickness_vs_density().set_dr_list(v_float_pair_list);
    v_float_pair_list.clear();
  }
;

thickness_vs_density_type :
  K_RESISTIVE_ONLY 
  { itfData->conductor.get_thickness_vs_density().set_type("RESISTIVE_ONLY"); }
| K_CAPACITIVE_ONLY
  { itfData->conductor.get_thickness_vs_density().set_type("CAPACITIVE_ONLY"); }
|
;

thickness_vs_width_and_spacing :
  K_THICKNESS_VS_WIDTH_AND_SPACING
  {
    v_lut.set_names("WIDTHS", "SPACINGS", "VALUES");
    v_is_lut_working = 1;
  }
  thickness_vs_width_and_spacing_type 
  '{'
    spacings
    widths
    values
  '}'
  {
    itfData->conductor.set_thickness_vws_lut(v_lut);
    v_lut.clear();
    v_is_lut_working = 0;
  }
;

thickness_vs_width_and_spacing_type :
  K_RESISTIVE_ONLY
  { itfData->conductor.set_thickness_vws_title("RESISTIVE_ONLY"); }
| K_CAPACITIVE_ONLY
  { itfData->conductor.set_thickness_vws_title("CAPACITIVE_ONLY"); }
|
;

tvf_adjustment_tables :
  K_TVF_ADJUSTMENT_TABLES '{'
    bottom_thickness_vs_width_and_others
  '}'
;

bottom_thickness_vs_width_and_others :
  bottom_thickness_vs_width_and_others bottom_thickness_vs_width_and_other
|
;

bottom_thickness_vs_width_and_other :
  bottom_thickness_vs_width_and_spacing
| bottom_thickness_vs_width_and_deltapd
;

bottom_thickness_vs_width_and_spacing :
  K_BOTTOM_THICKNESS_VS_WIDTH_AND_SPACING
  {
    v_lut.set_names("WIDTHS", "SPACINGS", "VALUES");
    v_is_lut_working = 1;
  }
  '{'
    spacings
    widths
    values
  '}'
  {
    itfData->conductor.set_tvf_bt_vws(v_lut);
    v_lut.clear();
    v_is_lut_working = 0;
  }
;

bottom_thickness_vs_width_and_deltapd :
  K_BOTTOM_THICKNESS_VS_WIDTH_AND_DELTAPD
  {
    v_lut.set_names("WIDTHS", "DELTAPD", "VALUES");
    v_is_lut_working = 1;
  }
  '{'
    deltapd
    widths
    values
  '}'
  {
    itfData->conductor.set_tvf_bt_vwd(v_lut);
    v_lut.clear();
    v_is_lut_working = 0;
  }
;

deltapd :
  K_DELTAPD '{' float_list '}'
  {
    v_lut.set_data_list<float>("DELTAPD", v_float_list);
    v_float_list.clear();
  }
;

device_type :
  K_DEVICE_TYPE '{'
    keyword_list
  '}'
;

keyword_list :
  keyword_list
  KEYWORD
  {

  }
|
;

linked_to :
  K_LINKED_TO '=' layer_name
  {

  }
;

extensionmin :
  K_EXTENSIONMIN '=' NUMBER
;

raised_diffusion_thickness :
  K_RAISED_DIFFUSION_THICKNESS '=' NUMBER
  {

  }
;

raised_diffusion_to_gate_smin :
  K_RAISED_DIFFUSION_TO_GATE_SMIN '=' NUMBER
  {

  }
;

raised_diffusion_etch :
  K_RAISED_DIFFUSION_ETCH '=' NUMBER
;

raised_diffusion_gate_side_conformal_er :
  K_RAISED_DIFFUSION_GATE_SIDE_CONFORMAL_ER '=' NUMBER
;

multigate :
  K_MULTIGATE KEYWORD
  '{'
    multigate_properties
  '}'
;

multigate_properties :
  multigate_properties
  multigate_property
|
;

multigate_property :
  fin_spacing
| fin_width
| fin_length
| fin_thickness
| gate_oxide_top_t
| gate_oxide_side_t
| gate_oxide_er
| gate_poly_top_t
| gate_poly_side_t
| channel_er
| raised_diffusion_growth
| gate_diffusion_layer_pair
;

fin_spacing :
  K_FIN_SPACING '=' NUMBER
;

fin_width :
  K_FIN_WIDTH '=' NUMBER
;

fin_length :
  K_FIN_LENGTH '=' NUMBER
;

fin_thickness :
  K_FIN_THICKNESS '=' NUMBER
;

gate_oxide_top_t :
  K_GATE_OXIDE_TOP_T '=' NUMBER
;

gate_oxide_side_t :
  K_GATE_OXIDE_SIDE_T '=' NUMBER
;

gate_oxide_er :
  K_GATE_OXIDE_ER '=' NUMBER
;

gate_poly_top_t :
  K_GATE_POLY_TOP_T '=' NUMBER
;

gate_poly_side_t :
  K_GATE_POLY_SIDE_T '=' NUMBER
;

channel_er :
  K_CHANNEL_ER '=' NUMBER
;

raised_diffusion_growth :
  K_RAISED_DIFFUSION_GROWTH '=' NUMBER
;

gate_diffusion_layer_pair :
  K_GATE_DIFFUSION_LAYER_PAIR '{'
    gate_diffusion_layer_pair_list
  '}'
;

gate_diffusion_layer_pair_list :
  gate_diffusion_layer_pair_list
  '(' keyword_list ')'
|
;

vias :
  vias via
  {
    CALLBACK(itfCallbacks->via_cb, itfCallBackType::kViaCbType, &itfData->via);
    itfData->via.clear();
  }
|
;

via :
  K_VIA layer_name
  {
    itfData->via.set_via_name(v_layer_name);
  }
  '{'
    via_properties
  '}'
;

via_properties :
  via_properties via_property
|
;

via_property :
  from
| to
| crt_property
| rho_property
| etch_property
| device_type
| rpv_vs_width_and_length
;

from :
  K_FROM '=' layer_name
  {
    itfData->via.set_from(v_layer_name);
  }
; 

to :
  K_TO '=' layer_name
  {
    itfData->via.set_to(v_layer_name);
  }
;

crt_property :
  crt1
  {
    itfData->via.set_crt1(v_crt1);
  }
| crt2        { itfData->via.set_crt2(v_crt2); }
| crt_vs_area
| t0          { itfData->via.set_t0(v_t0); }
;

crt_vs_area :
  K_CRT_VS_AREA '{'
    area_crt1_crt2_list
  '}'
;

area_crt1_crt2_list :
  area_crt1_crt2_list '(' NUMBER ',' NUMBER ',' NUMBER ')'
  {
    itfData->via.add_area_crt1_ct2($3, $5, $7);
  }
|
;

rho_property :
  rho         { itfData->via.set_rho(v_rho); }
| rpv
| area
| rpv_vs_area
;

rpv :
  K_RPV '=' NUMBER
  {
    itfData->via.set_rpv($3);
  }
;

area :
  K_AREA '=' NUMBER
  {
    itfData->via.set_area($3);
  }
;

rpv_vs_area :
  K_RPV_VS_AREA '{'
    area_rpv_list
  '}'
;

area_rpv_list :
  area_rpv_list '(' NUMBER ',' NUMBER ')'
  { itfData->via.add_area_rpv($3, $5); }
|
;

etch_property :
  etch_vs_contact_and_gate_spacings
| etch_vs_width_and_spacing
  {
    itfData->via.set_etch_vws(v_etch_effect_type, v_lut);
    ITF_FREE(v_etch_effect_type);
    v_lut.clear();
    v_is_lut_working = 0;
  }
| etch_vs_width_and_length
| K_CAPACITIVE_ONLY_ETCH '=' NUMBER
  { itfData->via.set_capacitive_only_etch($3); }
;

etch_vs_contact_and_gate_spacings :
  K_ETCH_VS_CONTACT_AND_GATE_SPACINGS
  {
    v_lut.set_names("GATE_TO_CONTACT_SPACINGS", "CONTACT_TO_CONTACT_SPACINGS", "VALUES");
    v_is_lut_working = 1;
  }
  K_CAPACITIVE_ONLY '{'
    etch_vs_contact_and_gate_spacings_property
  '}'
  {
    v_is_lut_working = 0;
  }
;

etch_vs_contact_and_gate_spacings_property :
  table
  {
    itfData->via.get_etch_cg().add_table("", v_lut);
    v_lut.clear();
  }
| number_of_tables name_tables
  {
    itfData->via.get_etch_cg().set_number_of_tables(v_number_of_tables);
    v_lut.clear();
  }
;

table :
  contact_to_contact_spacings
  gate_to_contact_spacings
  values
;

name_tables :
  name_tables
  table_name '{'
    table
  '}'
  {
    itfData->via.get_etch_cg().add_table(v_table_name, v_lut);
    v_lut.clear();

    v_lut.set_names("GATE_TO_CONTACT_SPACINGS", "CONTACT_TO_CONTACT_SPACINGS", "VALUES");
  }
|
;

table_name :
  KEYWORD { ITF_STR_CPY(v_table_name, $1); }
;

etch_vs_width_and_length :
  K_ETCH_VS_WIDTH_AND_LENGTH
  {
    v_etch_wlv_lut.set_names("WIDTHS", "LENGTHS", "VALUES");
  }
  etch_width_length_effect_type '{'
    K_LENGTHS '{' float_list '}'
    {
      v_etch_wlv_lut.set_data_list<float>("LENGTHS", v_float_list);
      v_float_list.clear();
    } 
    K_WIDTHS '{' float_list '}'
    {
      v_etch_wlv_lut.set_data_list<float>("WIDTHS", v_float_list);
      v_float_list.clear();
    }
    K_VALUES '{' float_pair_list '}'
    {
      v_etch_wlv_lut.set_data_list<std::pair<float, float>>("VALUES", v_float_pair_list);
      v_float_pair_list.clear();
    }
  '}'
  {
    itfData->via.set_etch_vwl(v_etch_effect_type, v_etch_wlv_lut);
    ITF_FREE(v_etch_effect_type);
    v_etch_wlv_lut.clear();
  }
;

etch_width_length_effect_type :
  K_RESISTIVE_ONLY  { ITF_STR_CPY(v_etch_effect_type, "RESISTIVE_ONLY"); }
| K_CAPACITIVE_ONLY { ITF_STR_CPY(v_etch_effect_type, "CAPACITIVE_ONLY"); }
|
  { ITF_FREE(v_etch_effect_type); }
;

rpv_vs_width_and_length :
  K_RPV_VS_WIDTH_AND_LENGTH '{'
    K_LENGTHS '{' float_list '}'
    K_WIDTHS  '{' float_list '}'
    K_VALUES  '{' float_list '}'
  '}'
;

variation_params :
  K_VARIATION_PARAMETERS '{'
    variation_params_list
  '}'
|
;

variation_params_list :
  variation_params_list variation_param
|
;

variation_param :
  variation_param_name '=' '{'
    variation_param_table
  '}'
  {
    CALLBACK(itfCallbacks->variation_cb, itfCallBackType::kVariationCbType, &itfData->variation_param);
    itfData->variation_param.clear();
  }
;

variation_param_name :
  KEYWORD
  {
    itfData->variation_param.param_name = $1;
  }
;

variation_param_table :
  variation_param_table
  variation_param_table_term
|
;

variation_param_table_term :
  '(' layer_name ',' variation_param_type ',' NUMBER ')'
  { 
    v_vpt.layer = v_layer_name;
    v_vpt.coeff = $6;
    itfData->variation_param.add_term(v_vpt);
    v_vpt.clear();
  }
;

variation_param_type :
  K_THICKNESS { v_vpt.type = "THICKNESS"; }
| K_WIDTH     { v_vpt.type = "WIDTH";     }
| K_RHO       { v_vpt.type = "RHO";       }
| K_ER        { v_vpt.type = "ER";        }
| K_RPV       { v_vpt.type = "RPV";       }
| KEYWORD     { v_vpt.type = $1;          }
;

%%

} // namespace itf

void itf_error(ITF_LTYPE* yylloc_param, const char* s) {
  std::cout << "error: " << s << ", "
            << "at line " << yylloc_param->last_line << ", "
            << "col " << yylloc_param->last_column << std::endl;
}
