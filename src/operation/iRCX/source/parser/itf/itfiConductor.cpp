// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include <iostream>
#include <string.h>

#include "itfiConductor.hpp"
#include "itfMarco.h"
#include "itfUtil.h"
namespace itf
{

itfiConductor::itfiConductor()
: _conductor_name(nullptr),
  _wmin(0),
  _smin(0),
  _thickness(0),
  _air_gap_spacings(),
  _air_gap_widths(),
  _air_gap_thicknesses(),
  _air_gap_bottom_heights(),
  _bottom_dielectric_thickness(0),
  _bottom_dielectric_er(0),
  _bottom_thickness_vs_si_width(),
  _t0(25),
  _crt1(std::nullopt),
  _crt2(std::nullopt),
  _crt_vs_si_width(),
  _density_box_weighting_factor(),
  _drop_factor(0),
  _etch(0),
  _capacitive_only_etch(0),
  _resistive_only_etch(0),
  _etch_vws_list(),
  _fill_ratio(0),
  _fill_width(0),
  _fill_spacing(0),
  _fill_type(nullptr),
  _gate_to_contact_smin(0),
  _gate_to_diffusion_cap(),
  _ild_vws(),
  _layer_type(nullptr),
  _measured_from(nullptr),
  _PBTV(),
  _rpsq(0),
  _rho(0),
  _rpsq_v_siw(),
  _rpsq_vws(),
  _rho_v_siw_t(),
  _rho_vws(),
  _side_tangent(0),
  _thickness_vs_density(),
  _thickness_vws(),
  _tvf_bt_vws(),
  _tvf_bt_vwd(),
  _has_air_gap_vs_spacing(0),
  _has_bottom_dielectric_thickness(0),
  _has_bottom_dielectric_er(0),
  _has_bottom_thickness_vs_si_width(0),
  _has_t0(0),
  _has_crt_stmt(0),
  _has_density_box_weighting_factor(0),
  _has_drop_factor(0),
  _has_etch_stmt(0),
  _has_evwas(0),
  _has_fill_stmt(0),
  _has_gate_to_contact_smin(0),
  _has_gate_to_diffusion_cap(0),
  _has_ild_vs_width_and_spacing(0),
  _is_planar(0),
  _has_layer_type(0),
  _has_measured_from(0),
  _has_polynomial_based_thickness_variation(0),
  _has_side_tangent(0),
  _has_thickness_vs_density(0),
  _has_thickness_vs_width_and_spacing(0),
  _has_tvf_adjustment_tables(0)
{ }

itfiConductor::itfiConductor(const itfiConductor& other)
: _conductor_name(nullptr)
, _fill_type(nullptr)
, _layer_type(nullptr)
, _measured_from(nullptr)
{
  *this = other;
}

itfiConductor& itfiConductor::operator=(const itfiConductor& rhs)
{
  if (this == &rhs) return *this;

  ITF_STR_CPY(_conductor_name, rhs._conductor_name);
  _wmin = rhs._wmin;
  _smin = rhs._smin;
  _thickness = rhs._thickness;
  _air_gap_spacings = rhs._air_gap_spacings;
  _air_gap_widths = rhs._air_gap_widths;
  _air_gap_thicknesses  = rhs._air_gap_thicknesses;
  _air_gap_bottom_heights = rhs._air_gap_bottom_heights;
  _bottom_dielectric_thickness = rhs._bottom_dielectric_thickness;
  _bottom_dielectric_er = rhs._bottom_dielectric_er;
  _bottom_thickness_vs_si_width = rhs._bottom_thickness_vs_si_width;
  _t0 = rhs._t0;
  _crt1 = rhs._crt1;
  _crt2 = rhs._crt2;
  _crt_vs_si_width = rhs._crt_vs_si_width;
  _density_box_weighting_factor = rhs._density_box_weighting_factor;
  _drop_factor = rhs._drop_factor;
  _etch = rhs._etch;
  _capacitive_only_etch = rhs._capacitive_only_etch;
  _resistive_only_etch = rhs._resistive_only_etch;
  _etch_vws_list = rhs._etch_vws_list;
  _fill_ratio = rhs._fill_ratio;
  _fill_width = rhs._fill_width;
  _fill_spacing = rhs._fill_spacing;
  ITF_STR_CPY(_fill_type, rhs._fill_type);
  _gate_to_contact_smin = rhs._gate_to_contact_smin;
  _gate_to_diffusion_cap = rhs._gate_to_diffusion_cap;
  _ild_vws = rhs._ild_vws;
  ITF_STR_CPY(_layer_type, rhs._layer_type);
  ITF_STR_CPY(_measured_from, rhs._measured_from);
  _PBTV = rhs._PBTV;
  _rpsq = rhs._rpsq;
  _rho = rhs._rho;
  _rpsq_v_siw = rhs._rpsq_v_siw;
  _rpsq_vws = rhs._rpsq_vws;
  _rho_v_siw_t = rhs._rho_v_siw_t;
  _rho_vws = rhs._rho_vws;
  _side_tangent = rhs._side_tangent;
  _thickness_vs_density = rhs._thickness_vs_density;
  _thickness_vws = rhs._thickness_vws;
  _tvf_bt_vws = rhs._tvf_bt_vws;
  _tvf_bt_vwd = rhs._tvf_bt_vwd;
  _has_air_gap_vs_spacing = rhs._has_air_gap_vs_spacing;
  _has_bottom_dielectric_thickness = rhs._has_bottom_dielectric_thickness;
  _has_bottom_dielectric_er = rhs._has_bottom_dielectric_er;
  _has_bottom_thickness_vs_si_width = rhs._has_bottom_thickness_vs_si_width;
  _has_t0 = rhs._has_t0;
  _has_crt_stmt = rhs._has_crt_stmt;
  _has_density_box_weighting_factor = rhs._has_density_box_weighting_factor;
  _has_drop_factor = rhs._has_drop_factor;
  _has_etch_stmt = rhs._has_etch_stmt;
  _has_evwas = rhs._has_evwas;
  _has_fill_stmt = rhs._has_fill_stmt;
  _has_gate_to_contact_smin = rhs._has_gate_to_contact_smin;
  _has_gate_to_diffusion_cap = rhs._has_gate_to_diffusion_cap;
  _has_ild_vs_width_and_spacing = rhs._has_ild_vs_width_and_spacing;
  _is_planar = rhs._is_planar;
  _has_layer_type = rhs._has_layer_type;
  _has_measured_from = rhs._has_measured_from;
  _has_polynomial_based_thickness_variation = rhs._has_polynomial_based_thickness_variation;
  _has_side_tangent = rhs._has_side_tangent;
  _has_thickness_vs_density = rhs._has_thickness_vs_density;
  _has_thickness_vs_width_and_spacing = rhs._has_thickness_vs_width_and_spacing;
  _has_tvf_adjustment_tables = rhs._has_tvf_adjustment_tables;

  return *this;
}

bool
itfiConductor::operator==(const itfiConductor& rhs) const 
{
  if (this == &rhs) return true;

  return itfStrCmp(_conductor_name, rhs._conductor_name)
    && _wmin == rhs._wmin
    && _smin == rhs._smin
    && _thickness == rhs._thickness
    && _air_gap_spacings == rhs._air_gap_spacings
    && _air_gap_widths == rhs._air_gap_widths
    && _air_gap_thicknesses == rhs._air_gap_thicknesses
    && _air_gap_bottom_heights == rhs._air_gap_bottom_heights
    && _bottom_dielectric_thickness == rhs._bottom_dielectric_thickness
    && _bottom_dielectric_er == rhs._bottom_dielectric_er
    && _bottom_thickness_vs_si_width == rhs._bottom_thickness_vs_si_width
    && _t0 == rhs._t0
    && _crt1 == rhs._crt1
    && _crt2 == rhs._crt2
    && _crt_vs_si_width == rhs._crt_vs_si_width
    && _density_box_weighting_factor == rhs._density_box_weighting_factor
    && _drop_factor == rhs._drop_factor
    && _etch == rhs._etch
    && _capacitive_only_etch == rhs._capacitive_only_etch
    && _resistive_only_etch == rhs._resistive_only_etch
    && _etch_vws_list == rhs._etch_vws_list
    && _fill_ratio == rhs._fill_ratio
    && _fill_width == rhs._fill_width
    && _fill_spacing == rhs._fill_spacing
    && itfStrCmp(_fill_type, rhs._fill_type) 
    && _gate_to_contact_smin == rhs._gate_to_contact_smin
    && _gate_to_diffusion_cap == rhs._gate_to_diffusion_cap
    && _ild_vws == rhs._ild_vws
    && itfStrCmp(_layer_type, rhs._layer_type)
    && itfStrCmp(_measured_from, rhs._measured_from)
    && _PBTV == rhs._PBTV
    && _rpsq == rhs._rpsq
    && _rho == rhs._rho
    && _rpsq_v_siw == rhs._rpsq_v_siw
    && _rpsq_vws == rhs._rpsq_vws
    && _rho_v_siw_t == rhs._rho_v_siw_t
    && _rho_vws == rhs._rho_vws
    && _side_tangent == rhs._side_tangent
    && _thickness_vs_density == rhs._thickness_vs_density
    && _thickness_vws == rhs._thickness_vws
    && _tvf_bt_vws == rhs._tvf_bt_vws
    && _tvf_bt_vwd == rhs._tvf_bt_vwd

    && _has_air_gap_vs_spacing == rhs._has_air_gap_vs_spacing
    && _has_bottom_dielectric_thickness == rhs._has_bottom_dielectric_thickness
    && _has_bottom_dielectric_er == rhs._has_bottom_dielectric_er
    && _has_bottom_thickness_vs_si_width == rhs._has_bottom_thickness_vs_si_width
    && _has_t0 == rhs._has_t0
    && _has_crt_stmt == rhs._has_crt_stmt
    && _has_density_box_weighting_factor == rhs._has_density_box_weighting_factor
    && _has_drop_factor == rhs._has_drop_factor
    && _has_etch_stmt == rhs._has_etch_stmt
    && _has_evwas == rhs._has_evwas
    && _has_fill_stmt == rhs._has_fill_stmt
    && _has_gate_to_contact_smin == rhs._has_gate_to_contact_smin
    && _has_gate_to_diffusion_cap == rhs._has_gate_to_diffusion_cap
    && _has_ild_vs_width_and_spacing == rhs._has_ild_vs_width_and_spacing
    && _is_planar == rhs._is_planar
    && _has_layer_type == rhs._has_layer_type
    && _has_measured_from == rhs._has_measured_from
    && _has_polynomial_based_thickness_variation == rhs._has_polynomial_based_thickness_variation
    && _has_side_tangent == rhs._has_side_tangent
    && _has_thickness_vs_density == rhs._has_thickness_vs_density
    && _has_thickness_vs_width_and_spacing == rhs._has_thickness_vs_width_and_spacing
    && _has_tvf_adjustment_tables == rhs._has_tvf_adjustment_tables
  ;
}

itfiConductor::~itfiConductor()
{
  clear();
}

const char*
itfiConductor::get_conductor_name() const
{
  return _conductor_name;
}

float
itfiConductor::get_wmin() const
{
  return _wmin;
}

float
itfiConductor::get_smin() const
{
  return _smin;
}

float
itfiConductor::get_thickness () const
{
  return _thickness;
}

std::vector<float>&
itfiConductor::get_air_gap_spacings()
{
  return _air_gap_spacings;
}

const std::vector<float>&
itfiConductor::get_air_gap_spacings() const
{
  return _air_gap_spacings;
}

std::vector<float>&
itfiConductor::get_air_gap_widths()
{
  return _air_gap_widths;
}

const std::vector<float>&
itfiConductor::get_air_gap_widths() const
{
  return _air_gap_widths;
}

std::vector<float>&
itfiConductor::get_air_gap_thicknesses()
{
  return _air_gap_thicknesses;
}

const std::vector<float>&
itfiConductor::get_air_gap_thicknesses() const
{
  return _air_gap_thicknesses;
}

std::vector<float>&
itfiConductor::get_air_gap_bottom_heights()
{
  return _air_gap_bottom_heights;
}

const std::vector<float>&
itfiConductor::get_air_gap_bottom_heights() const
{
  return _air_gap_bottom_heights;
}

float
itfiConductor::get_bottom_dielectric_thickness() const
{
  return _bottom_dielectric_thickness;
}

float
itfiConductor::get_bottom_dielectric_er() const
{
  return _bottom_dielectric_er;
}

itfiBTSW&
itfiConductor::get_bottom_thickness_vs_si_width()
{
  return _bottom_thickness_vs_si_width;
}

const itfiBTSW&
itfiConductor::get_bottom_thickness_vs_si_width() const
{
  return _bottom_thickness_vs_si_width;
}

float
itfiConductor::get_t0() const
{
  return _t0;
}

bool
itfiConductor::has_t0() const
{
  return _has_t0;
}

std::optional<float>
itfiConductor::get_crt1() const
{
  return _crt1;
}

std::optional<float>
itfiConductor::get_crt2() const
{
  return _crt2;
}

const std::vector<itfiWidthCrt>&
itfiConductor::get_crt_vs_si_width() const
{
  return _crt_vs_si_width;
}

const std::vector<itfiDensityBox>&
itfiConductor::get_density_box_weighting_factor() const
{ 
  if (_density_box_weighting_factor.size()) {
    return _density_box_weighting_factor;
  } else {
    static const std::vector<itfiDensityBox> kDefaultDensityBoxWeightingFactor{{50, 1}};
    return kDefaultDensityBoxWeightingFactor;
  }
}

float
itfiConductor::get_drop_factor() const
{
  return _drop_factor;
}

float
itfiConductor::get_etch() const
{
  return _etch;
}

float
itfiConductor::get_capacitive_only_etch() const
{
  return _capacitive_only_etch;
}

float
itfiConductor::get_resistive_only_etch() const
{
  return _resistive_only_etch;
}

// ETCH_VS_WIDTH_AND_SPACING
std::vector<itfiEtchVWS>&
itfiConductor::get_etch_vws_list()
{
  return _etch_vws_list;
}

const std::vector<itfiEtchVWS>&
itfiConductor::get_etch_vws_list() const
{
  return _etch_vws_list;
}

float
itfiConductor::get_fill_ratio() const
{
  return _fill_ratio;
}

float
itfiConductor::get_fill_width() const
{
  return _fill_width;
}

float
itfiConductor::get_fill_spacing() const
{
  return _fill_spacing;
}

const char*
itfiConductor::get_fill_type() const
{
  if (_fill_type) return _fill_type;
  return "GROUNDED";
}

float
itfiConductor::get_gate_to_contact_smin() const
{
  return _gate_to_contact_smin;
}

itfiG2DC&
itfiConductor::get_gate_to_diffusion_cap()
{
  return _gate_to_diffusion_cap;
}

const itfiG2DC&
itfiConductor::get_gate_to_diffusion_cap() const
{
  return _gate_to_diffusion_cap;
}

itfTitleLut<float, float, float>&
itfiConductor::get_ild_vws()
{
  return _ild_vws;
}

const itfTitleLut<float, float, float>&
itfiConductor::get_ild_vws() const
{
  return _ild_vws;
}

char*
itfiConductor::get_layer_type() const
{
  return _layer_type;
}

char*
itfiConductor::get_measured_from() const
{
  return _measured_from;
}

itfiPBTV&
itfiConductor::get_PBTV()
{
  return _PBTV;
}

const itfiPBTV&
itfiConductor::get_PBTV() const
{
  return _PBTV;
}

float
itfiConductor::get_rpsq() const
{
  return _rpsq;
}

float
itfiConductor::get_rho() const
{
  return _rho;
}

itf1DLUT<float, float>&
itfiConductor::get_rpsq_vs_si_width()
{
  return _rpsq_v_siw;
}

const itf1DLUT<float, float>&
itfiConductor::get_rpsq_vs_si_width() const
{
  return _rpsq_v_siw;
}

itf2DLUT<float, float, float>&
itfiConductor::get_rpsq_vws()
{
  return _rpsq_vws;
}

const itf2DLUT<float, float, float>&
itfiConductor::get_rpsq_vws() const
{
  return _rpsq_vws;
}

itfiRhoVWS&
itfiConductor::get_rho_v_siw_t()
{
  return _rho_v_siw_t;
}

const itfiRhoVWS&
itfiConductor::get_rho_v_siw_t() const
{
  return _rho_v_siw_t;
}

itf2DLUT<float, float, float>&
itfiConductor::get_rho_vws()
{
  return _rho_vws;
}

const itf2DLUT<float, float, float>&
itfiConductor::get_rho_vws() const
{
  return _rho_vws;
}

float
itfiConductor::get_side_tangent() const
{
  return _side_tangent;
}

itfiThicknessVsDensity&
itfiConductor::get_thickness_vs_density()
{
  return _thickness_vs_density;
}

const itfiThicknessVsDensity&
itfiConductor::get_thickness_vs_density() const
{
  return _thickness_vs_density;
}

itfTitleLut<float, float, float>&
itfiConductor::get_thickness_vws()
{
  return _thickness_vws;
}

const itfTitleLut<float, float, float>&
itfiConductor::get_thickness_vws() const
{
  return _thickness_vws;
}

itf2DLUT<float, float, float>&
itfiConductor::get_tvf_bt_vws()
{
  return _tvf_bt_vws;
}

const itf2DLUT<float, float, float>&
itfiConductor::get_tvf_bt_vws() const
{
  return _tvf_bt_vws;
}

itf2DLUT<float, float, float>&
itfiConductor::get_tvf_bt_vwd()
{
  return _tvf_bt_vwd;
}

const itf2DLUT<float, float, float>&
itfiConductor::get_tvf_bt_vwd() const
{
  return _tvf_bt_vwd;
}

void
itfiConductor::set_conductor_name(const char* name)
{
  ITF_STR_CPY(_conductor_name, name);
}

void
itfiConductor::set_wmin(float v)
{
  _wmin = v;
}

void
itfiConductor::set_smin(float v)
{
  _smin = v;
}

void
itfiConductor::set_thickness(float v)
{
  _thickness = v;
}

void
itfiConductor::set_air_gap_spacings(const std::vector<float>& v)
{
  _air_gap_spacings = v;
}

void
itfiConductor::set_air_gap_widths(const std::vector<float>& v)
{
  _air_gap_widths = v;
}

void
itfiConductor::set_air_gap_thicknesses(const std::vector<float>& v)
{
  _air_gap_thicknesses = v;
}

void
itfiConductor::set_air_gap_bottom_heights(const std::vector<float>& v)
{
  _air_gap_bottom_heights = v;
}

// @param s The size of the density box. Units: microns
// @param w The weighting factor.
void
itfiConductor::add_density_box_weight(int s, double w)
{
  if (_density_box_weighting_factor.size() >= 6) {
    std::cout << "Up to 5 entries are allowed" << std::endl;
    return;
  }

  if ((0 < s)   && (s < 500)
   && (-10 < w) && (w < 10) && (w != 0) )
  {
    _density_box_weighting_factor.emplace_back(s, w);
  } else {
    std::cout << "Data invalid, when s = " << s 
              << ", w = " << w
              << std::endl;
  }
}

void
itfiConductor::set_bottom_dielectric_thickness(float v)
{
  _bottom_dielectric_thickness = v;
}

void
itfiConductor::set_bottom_dielectric_er(float v)
{
  _bottom_dielectric_er = v;
}

void
itfiConductor::set_t0(float t)
{
  _t0 = t;
  _has_t0 = 1;
}

void
itfiConductor::set_crt1(float crt1)
{
  _crt1 = crt1;
}

void
itfiConductor::set_crt2(float crt2)
{
  _crt2 = crt2;
}

// @param siw Post-etch conductor widths. Units: microns
// @param crt1 Linear temperature coefficients
// @param crt2 Quadratic temperature coefficients
void
itfiConductor::add_siw_crt1_crt2(float siw, float crt1, float crt2)
{
  _crt_vs_si_width.emplace_back(siw, crt1, crt2);
}

void
itfiConductor::set_drop_factor(float v)
{
  _drop_factor = v;
}

void
itfiConductor::set_etch(float v)
{
  _etch = v;
}

void
itfiConductor::set_capacitive_only_etch(float v)
{
  _capacitive_only_etch = v;
}

void
itfiConductor::set_resistive_only_etch(float v)
{
  _resistive_only_etch = v;
}

void
itfiConductor::add_etch_vws(const char* title, const itf2DLUT<float, float, float>& lut)
{
  _etch_vws_list.emplace_back(title ? title : "", lut);
}

void
itfiConductor::set_fill_ratio(float ratio)
{
  _fill_ratio = ratio;
}

void
itfiConductor::set_fill_width(float width)
{
  _fill_width = width;
}

void
itfiConductor::set_fill_spacing(float spacing)
{
  _fill_spacing = spacing;
}

void
itfiConductor::set_fill_type(const char* type)
{
  if ((strncmp("GROUNDED", type, 8)  == 0)
   || (strncmp("FLOATING", type, 8) == 0) )
  {
    ITF_STR_CPY(_fill_type, type);
  } else {
    std::cout << "Invalid type: " << type << std::endl;
  }
}

void
itfiConductor::set_gate_to_contact_smin(float v)
{
  _gate_to_contact_smin = v;
}

void
itfiConductor::set_layer_type(const char* type)
{
  ITF_STR_CPY(_layer_type, type);
}

void
itfiConductor::set_side_tangent(float v)
{
  _side_tangent = v;
}

void
itfiConductor::set_is_planar()
{
  _is_planar = 1;
}

void
itfiConductor::set_ild_vws_lut(const itf2DLUT<float, float, float>& v)
{
  _ild_vws.set_lut(v);
}

void
itfiConductor::set_ild_vws_title(const char* v)
{
  _ild_vws.set_title(v);
}

void
itfiConductor::set_thickness_vws_lut(const itf2DLUT<float, float, float>& v)
{
  _thickness_vws.set_lut(v);
}

void
itfiConductor::set_thickness_vws_title(const char* v)
{
  _thickness_vws.set_title(v);
}

void
itfiConductor::set_rpsq(float v)
{
  _rpsq = v;
}

void
itfiConductor::set_rho(float v)
{
  _rho = v;
}

void
itfiConductor::set_rpsq_vs_si_width(const std::vector<std::pair<float, float>>& v)
{
  _rpsq_v_siw.set_points(v);
}

void
itfiConductor::set_rpsq_vws(const itf2DLUT<float, float, float>& v)
{
  _rpsq_vws = v;
}

void
itfiConductor::set_rho_v_siw_t(const itf2DLUT<float, float, float>& v)
{
  _rho_v_siw_t = v;
}

void
itfiConductor::set_rho_vws(const itf2DLUT<float, float, float>& v)
{
  _rho_vws = v;
}

void
itfiConductor::set_tvf_bt_vws(const itf2DLUT<float, float, float>& v)
{
  _tvf_bt_vws = v;
}

void
itfiConductor::set_tvf_bt_vwd(const itf2DLUT<float, float, float>& v)
{
  _tvf_bt_vwd = v;
}

void
itfiConductor::set_measured_from(const char* s)
{
  ITF_STR_CPY(_measured_from, s);
  _has_measured_from = 1;
}

void
itfiConductor::clear()
{
  ITF_FREE(_conductor_name);
  _wmin = 0;
  _smin = 0;
  _thickness = 0;
  _air_gap_spacings.clear();
  _air_gap_widths.clear();
  _air_gap_thicknesses.clear();
  _air_gap_bottom_heights.clear();
  _bottom_dielectric_thickness = 0;
  _bottom_dielectric_er = 0;
  _bottom_thickness_vs_si_width.clear();
  _t0 = 25;
  _crt1.reset();
  _crt2.reset();
  _crt_vs_si_width.clear();
  _density_box_weighting_factor.clear();
  _drop_factor = 0;
  _etch = 0;
  _capacitive_only_etch = 0;
  _resistive_only_etch = 0;
  _etch_vws_list.clear();
  _fill_ratio = 0;
  _fill_width = 0;
  _fill_spacing = 0;
  ITF_FREE(_fill_type);
  _gate_to_contact_smin = 0;
  _gate_to_diffusion_cap.clear();
  _ild_vws.clear();
  ITF_FREE(_layer_type);
  ITF_FREE(_measured_from);
  _PBTV.clear();
  _rpsq = 0;
  _rho = 0;
  _rpsq_v_siw.clear();
  _rpsq_vws.clear();
  _rho_v_siw_t.clear();
  _rho_vws.clear();
  _side_tangent = 0;
  _thickness_vs_density.clear();
  _thickness_vws.clear();
  _tvf_bt_vws.clear();
  _tvf_bt_vwd.clear();

  _has_air_gap_vs_spacing = 0;
  _has_bottom_dielectric_thickness = 0;
  _has_bottom_dielectric_er = 0;
  _has_bottom_thickness_vs_si_width = 0;
  _has_t0 = 0;
  _has_crt_stmt = 0;
  _has_density_box_weighting_factor = 0;
  _has_drop_factor = 0;
  _has_etch_stmt = 0;
  _has_evwas = 0;
  _has_fill_stmt = 0;
  _has_gate_to_contact_smin = 0;
  _has_gate_to_diffusion_cap = 0;
  _has_ild_vs_width_and_spacing = 0;
  _is_planar = 0;
  _has_layer_type = 0;
  _has_measured_from = 0;
  _has_polynomial_based_thickness_variation = 0;
  _has_side_tangent = 0;
  _has_thickness_vs_density = 0;
  _has_thickness_vs_width_and_spacing = 0;
  _has_tvf_adjustment_tables = 0;
}

// linear interpolation when width lies in the table,
// otherwise use boundary value.
// In other words, no extrapolation.
// @param width Conductor silicon (post-etch) widths. Units: microns.
// @param crt1 Linear temperature coefficients for the corresponding conductor widths
// @param crt2 Quadratic temperature coefficients
void
itfiConductor::query_crt(double width, double& crt1, double& crt2)
{
  if (_crt_vs_si_width.size() == 0) {
    std::cout << "table _crt_vs_si_width is empty" << std::endl;
    return;
  }

  auto it_high = std::upper_bound(_crt_vs_si_width.begin(), _crt_vs_si_width.end(),
    width, [](double v, itfiWidthCrt elem){
      return v < elem.si_width;
    });
  
  size_t idx_h = 0, idx_l = 0;
  if (it_high == _crt_vs_si_width.end()) {
    idx_h = _crt_vs_si_width.size() - 1;
    idx_l = idx_h;
  } else {
    idx_h = std::distance(_crt_vs_si_width.begin(), it_high);
    idx_l = idx_h - (idx_h == 0 ? 0 : 1);
  }

  auto& elem_h = _crt_vs_si_width.at(idx_h);
  auto& elem_l = _crt_vs_si_width.at(idx_l);
  auto t = (width - elem_l.si_width) / (elem_h.si_width - elem_l.si_width);
  crt1 = std::lerp(elem_l.crt_1, elem_h.crt_1, t);
  crt2 = std::lerp(elem_l.crt_2, elem_h.crt_2, t);
}

itfiBTSW::itfiBTSW()
: _type(nullptr),
  _sr_lut()
{

}

itfiBTSW::itfiBTSW(
  const itfiBTSW& other)
: _type(nullptr),
  _sr_lut()
{
  *this = other;
}

itfiBTSW&
itfiBTSW::operator=(const itfiBTSW& rhs)
{
  if (this == &rhs) return *this;

  ITF_STR_CPY(_type, rhs._type);
  _sr_lut = rhs._sr_lut;

  return *this;
}

bool
itfiBTSW::operator==(const itfiBTSW& rhs) const
{
  if (this == &rhs) return true;

  return itfStrCmp(_type, rhs._type)
    && _sr_lut == rhs._sr_lut
  ;
}

itfiBTSW::~itfiBTSW()
{
  clear();
}

const char*
itfiBTSW::get_type() const
{
  return _type;
}

itf1DLUT<float, float>&
itfiBTSW::get_sr_lut()
{
  return _sr_lut;
}

const itf1DLUT<float, float>&
itfiBTSW::get_sr_lut() const
{
  return _sr_lut;
}

const std::vector<std::pair<float, float>>&
itfiBTSW::get_sr_list() const
{
  return _sr_lut.get_points();
}

void
itfiBTSW::set_type(const char* type)
{
  if ((strncmp("RESISTIVE_ONLY", type, 14)  == 0)
   || (strncmp("CAPACITIVE_ONLY", type, 15) == 0) )
  {
    ITF_STR_CPY(_type, type);
  } else {
    std::cout << "Invalid type: " << type << std::endl;
  }
}

void
itfiBTSW::clear()
{
  ITF_FREE(_type);
  _sr_lut.clear();
}

void
itfiBTSW::set_sr_list(const std::vector<std::pair<float, float>>& pair_list)
{
  _sr_lut.set_points(pair_list);
}

size_t
itfiG2DC::get_number_of_tables() const
{
  return _number_of_tables;
}

void
itfiG2DC::set_number_of_tables(size_t n)
{
  _number_of_tables = n;
}

void
itfiG2DC::add_model(
  const char* title,
  const itf2DLUT<float, float, float>& lut
) {
  _model_list.emplace_back(title ? title : "", lut);
}

bool
itfiG2DC::operator==(const itfiG2DC& rhs) const
{
  if (this == &rhs) return true;

  return _number_of_tables == rhs._number_of_tables
    && _model_list == rhs._model_list
  ;
}

void
itfiG2DC::clear()
{
  _number_of_tables = 0;
  _model_list.clear();
}

const std::vector<int>&
itfiPBTV::get_density_polynomial_orders() const 
{
  return _density_polynomial_orders;
}

const std::vector<int>&
itfiPBTV::get_width_polynomial_orders() const 
{
  return _width_polynomial_orders;
}

const std::vector<float>&
itfiPBTV::get_width_ranges() const 
{
  return _width_ranges;
}

// @param id index of _polynomial_coefficients_list
const std::vector<float>&
itfiPBTV::get_polynomial_coefficients_list(size_t id) const
{
  if (id < _polynomial_coefficients_list.size()) {
    return _polynomial_coefficients_list.at(id);
  }

  static const std::vector<float> kEmptyCoefficients;
  return kEmptyCoefficients;
}

size_t
itfiPBTV::get_polynomial_coefficients_list_size() const
{
  return _polynomial_coefficients_list.size();
}

void
itfiPBTV::add_density_polynomial_order(int order)
{
  _density_polynomial_orders.emplace_back(order);
}

void
itfiPBTV::add_width_polynomial_order(int order)
{
  _width_polynomial_orders.emplace_back(order);
}

void
itfiPBTV::add_width_range(float threshold)
{
  _width_ranges.emplace_back(threshold);
}

void
itfiPBTV::set_density_polynomial_order(const std::vector<int>& list)
{
  _density_polynomial_orders = list;
}

void
itfiPBTV::set_width_polynomial_order(const std::vector<int>& list)
{
  _width_polynomial_orders = list;
}

void
itfiPBTV::set_width_range(const std::vector<float>& list)
{
  _width_ranges = list;
}

void
itfiPBTV::add_polynomial_coefficients(const std::vector<float>& list)
{
  _polynomial_coefficients_list.emplace_back(list);
}

bool
itfiPBTV::operator==(const itfiPBTV& rhs) const
{
  if (this == &rhs) return true;

  return _density_polynomial_orders == rhs. _density_polynomial_orders
    && _width_polynomial_orders == rhs. _width_polynomial_orders
    && _width_ranges == rhs. _width_ranges
    && _polynomial_coefficients_list == rhs. _polynomial_coefficients_list
  ;
}

void
itfiPBTV::clear()
{
  _density_polynomial_orders.clear();
  _width_polynomial_orders.clear();
  _width_ranges.clear();
  _polynomial_coefficients_list.clear();
}

itfiThicknessVsDensity::itfiThicknessVsDensity()
: _type(nullptr),
  _dr_lut()
{

}

itfiThicknessVsDensity::itfiThicknessVsDensity(const itfiThicknessVsDensity& other)
: _type(nullptr),
  _dr_lut()
{
  *this = other;
}

itfiThicknessVsDensity&
itfiThicknessVsDensity::operator=(const itfiThicknessVsDensity& rhs)
{
  if (this == &rhs) return *this;

  ITF_STR_CPY(_type, rhs._type);
  _dr_lut = rhs._dr_lut;

  return *this;
}

bool
itfiThicknessVsDensity::operator==(const itfiThicknessVsDensity& rhs) const
{
  if (this == &rhs) return true;

  return itfStrCmp(_type, rhs._type)
    && _dr_lut == rhs._dr_lut
  ;
}

itfiThicknessVsDensity::~itfiThicknessVsDensity()
{
  clear();
}

const char*
itfiThicknessVsDensity::get_type() const
{
  return _type;
}

itf1DLUT<float, float>&
itfiThicknessVsDensity::get_dr_lut()
{
  return _dr_lut;
}

const itf1DLUT<float, float>&
itfiThicknessVsDensity::get_dr_lut() const
{
  return _dr_lut;
}

const std::vector<std::pair<float, float>>&
itfiThicknessVsDensity::get_dr_list() const
{
  return _dr_lut.get_points();
}

void
itfiThicknessVsDensity::set_type(const char* type)
{
  if ((strncmp("RESISTIVE_ONLY", type, 14)  == 0)
   || (strncmp("CAPACITIVE_ONLY", type, 15) == 0) )
  {
    ITF_STR_CPY(_type, type);
  } else {
    std::cout << "Invalid type: " << type << std::endl;
  }  
}

// @param d density
// @param r relative change in thickness
void
itfiThicknessVsDensity::add_dr(float d, float r)
{
  _dr_lut.add_point(d, r);
}

void
itfiThicknessVsDensity::set_dr_list(const std::vector<std::pair<float, float>>& list)
{
  _dr_lut.set_points(list);
}

void
itfiThicknessVsDensity::clear()
{
  ITF_FREE(_type);
  _dr_lut.clear();
}

} // namespace itf
