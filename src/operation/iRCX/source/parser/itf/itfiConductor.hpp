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
#pragma once 

#include <vector>

#include "itf1DLUT.hpp"
#include "itf2DLUT.hpp"
namespace itf
{

// Bottom Thickness Vs Silicon Width
class itfiBTSW {
 public:
  // constructor
  itfiBTSW();
  itfiBTSW(const itfiBTSW&);
  ~itfiBTSW();

  // getter
  const char* get_type() const;
  itf1DLUT<float, float>& get_sr_lut();
  const itf1DLUT<float, float>& get_sr_lut() const;
  const std::vector<std::pair<float, float>>& get_sr_list() const;

  // setter
  void set_type(const char*);
  void set_sr_list(const std::vector<std::pair<float, float>>&);

  // operator
  itfiBTSW& operator=(const itfiBTSW&);
  bool operator==(const itfiBTSW&) const;

  // function
  void clear();

 private:
  // members
  char* _type;
  itf1DLUT<float, float> _sr_lut; // (silicon width, relative change)
};

// Gate To Diffusion Cap
// (gate_to_contact, contact_to_contact) -> caps_per_micron
class itfiG2DC {
 public:
  // constructor

  // getter
  size_t get_number_of_tables() const;

  // setter
  void set_number_of_tables(size_t);
  void add_model(const char*, const itf2DLUT<float, float, float>&);

  // operator
  bool operator==(const itfiG2DC&) const;

  // function
  void clear();

 private:
  // members
  size_t _number_of_tables;
  std::vector<itfTitleLut<float, float, float>> _model_list; // (gate_to_contact, contact_to_contact) -> caps_per_micron
};

// Polynomial Based Thickness Variation
class itfiPBTV {
 public:
  // constructor

  // getter
  const std::vector<int>& get_density_polynomial_orders() const;
  const std::vector<int>& get_width_polynomial_orders() const;
  const std::vector<float>& get_width_ranges() const;
  const std::vector<float>& get_polynomial_coefficients_list(size_t) const;
  size_t get_polynomial_coefficients_list_size() const;

  // setter
  void add_density_polynomial_order(int);
  void add_width_polynomial_order(int);
  void add_width_range(float);
  void set_density_polynomial_order(const std::vector<int>&);
  void set_width_polynomial_order(const std::vector<int>&);
  void set_width_range(const std::vector<float>&);
  void add_polynomial_coefficients(const std::vector<float>&);

  // operator
  bool operator==(const itfiPBTV&) const;

  // function
  void clear();
  
 private:
  // members
  std::vector<int> _density_polynomial_orders;
  std::vector<int> _width_polynomial_orders;
  std::vector<float> _width_ranges;
  std::vector<std::vector<float>> _polynomial_coefficients_list;
};

class itfiThicknessVsDensity {
 public:
  // constructor
  itfiThicknessVsDensity();
  itfiThicknessVsDensity(const itfiThicknessVsDensity&);
  ~itfiThicknessVsDensity();

  // getter
  const char* get_type() const;
  itf1DLUT<float, float>& get_dr_lut();
  const itf1DLUT<float, float>& get_dr_lut() const;
  const std::vector<std::pair<float, float>>& get_dr_list() const;

  // setter
  void set_type(const char*);
  void add_dr(float, float);
  void set_dr_list(const std::vector<std::pair<float, float>>&);

  // operator
  itfiThicknessVsDensity& operator=(const itfiThicknessVsDensity&);
  bool operator==(const itfiThicknessVsDensity&) const;

  // function
  void clear();

 private:
  // members
  char* _type;
  itf1DLUT<float, float> _dr_lut; // (density, relative change in thickness)
};

// Specifies CRT-based temperature derating for different conductor widths
class itfiWidthCrt {
 public:
  double si_width = 0;
  double crt_1    = 0;
  double crt_2    = 0;
 
  bool operator==(const itfiWidthCrt& rhs) const {
    return si_width == rhs.si_width && crt_1 == rhs.crt_1 && crt_2 == rhs.crt_2;
  }
};

class itfiDensityBox {
 public:
  int box_size; // unit microns
  double weight; // none

  bool operator==(const itfiDensityBox rhs) const {
    return box_size == rhs.box_size && weight == rhs.weight;
  }
};

// (widths, spacings) -> values
class itfiEtchVWS : public itfTitleLut<float, float, float>{
 public:
  // constructor
  itfiEtchVWS() = default;
  itfiEtchVWS(const char* title, const itf2DLUT<float, float, float> lut)
  : itfTitleLut(title, lut)
  { }
  
  // Units: microns
  std::vector<float> get_widths() const { return get_rows();}
  // Units: microns
  std::vector<float> get_spacings() const { return get_cols();}
};

// RHO_VS_SI_WIDTH_AND_THICKNESS
// (thickness, silicon width) -> rho value
class itfiRhoVWS : public itf2DLUT<float, float, float> {
 public:
  itfiRhoVWS() = default;
  // Units: microns.
  std::vector<float> get_thickness() const { return get_rows(); }
  // Units: microns.
  std::vector<float> get_width() const { return get_cols(); }
  // operator
  itfiRhoVWS& operator=(const itf2DLUT<float, float, float>& rhs) {
    itf2DLUT::operator=(rhs);
    return *this;
  }
  bool operator==(const itfiRhoVWS& rhs) const {
    return itf2DLUT::operator==(rhs);
  }
};

class itfiConductor {
 public:
  // constructor
  itfiConductor();
  itfiConductor(const itfiConductor&);
  ~itfiConductor();

  // getter
  const char* get_conductor_name() const;
  float get_wmin() const;
  float get_smin() const;
  float get_thickness () const;
  std::vector<float>& get_air_gap_spacings();
  const std::vector<float>& get_air_gap_spacings() const;
  std::vector<float>& get_air_gap_widths();
  const std::vector<float>& get_air_gap_widths() const;
  std::vector<float>& get_air_gap_thicknesses();
  const std::vector<float>& get_air_gap_thicknesses() const;
  std::vector<float>& get_air_gap_bottom_heights();
  const std::vector<float>& get_air_gap_bottom_heights() const;
  float get_bottom_dielectric_thickness() const;
  float get_bottom_dielectric_er() const;
  itfiBTSW& get_bottom_thickness_vs_si_width();
  const itfiBTSW& get_bottom_thickness_vs_si_width() const;
  float get_t0() const;
  bool has_t0() const;
  std::optional<float> get_crt1() const;
  std::optional<float> get_crt2() const;
  const std::vector<itfiWidthCrt>& get_crt_vs_si_width() const;
  const std::vector<itfiDensityBox>& get_density_box_weighting_factor() const;
  float get_drop_factor() const;
  float get_etch() const;
  float get_capacitive_only_etch() const;
  float get_resistive_only_etch() const;
  std::vector<itfiEtchVWS>& get_etch_vws_list();
  const std::vector<itfiEtchVWS>& get_etch_vws_list() const;
  float get_fill_ratio() const;
  float get_fill_width() const;
  float get_fill_spacing() const;
  const char* get_fill_type() const;
  float get_gate_to_contact_smin() const;
  itfiG2DC& get_gate_to_diffusion_cap();
  const itfiG2DC& get_gate_to_diffusion_cap() const;
  itfTitleLut<float, float, float>& get_ild_vws();
  const itfTitleLut<float, float, float>& get_ild_vws() const;
  char* get_layer_type() const;
  char* get_measured_from() const;
  itfiPBTV& get_PBTV();
  const itfiPBTV& get_PBTV() const;
  float get_rpsq() const;
  float get_rho() const;
  itf1DLUT<float, float>& get_rpsq_vs_si_width();
  const itf1DLUT<float, float>& get_rpsq_vs_si_width() const;
  itf2DLUT<float, float, float>& get_rpsq_vws();
  const itf2DLUT<float, float, float>& get_rpsq_vws() const;
  itfiRhoVWS& get_rho_v_siw_t();
  const itfiRhoVWS& get_rho_v_siw_t() const;
  itf2DLUT<float, float, float>& get_rho_vws();
  const itf2DLUT<float, float, float>& get_rho_vws() const;
  float get_side_tangent() const;
  itfiThicknessVsDensity& get_thickness_vs_density();
  const itfiThicknessVsDensity& get_thickness_vs_density() const;
  itfTitleLut<float, float, float>& get_thickness_vws();
  const itfTitleLut<float, float, float>& get_thickness_vws() const;
  itf2DLUT<float, float, float>& get_tvf_bt_vws();
  const itf2DLUT<float, float, float>& get_tvf_bt_vws() const;
  itf2DLUT<float, float, float>& get_tvf_bt_vwd();
  const itf2DLUT<float, float, float>& get_tvf_bt_vwd() const;
  
  // setter
  void set_conductor_name(const char*);
  void set_wmin(float);
  void set_smin(float);
  void set_thickness(float);
  void set_air_gap_spacings(const std::vector<float>&);
  void set_air_gap_widths(const std::vector<float>&);
  void set_air_gap_thicknesses(const std::vector<float>&);
  void set_air_gap_bottom_heights(const std::vector<float>&);
  void set_bottom_dielectric_thickness(float);
  void set_bottom_dielectric_er(float);
  void set_t0(float);
  void set_crt1(float crt1);
  void set_crt2(float crt2);
  void add_siw_crt1_crt2(float, float, float);
  void add_density_box_weight(int, double);
  void set_drop_factor(float);
  void set_etch(float);
  void set_capacitive_only_etch(float);
  void set_resistive_only_etch(float);
  void add_etch_vws(const char*, const itf2DLUT<float, float, float>&);
  void set_fill_ratio(float);
  void set_fill_width(float);
  void set_fill_spacing(float);
  void set_fill_type(const char*);
  void set_gate_to_contact_smin(float);
  void set_layer_type(const char*);
  void set_side_tangent(float);
  void set_is_planar();
  void set_ild_vws_lut(const itf2DLUT<float, float, float>&);
  void set_ild_vws_title(const char*);
  void set_thickness_vws_lut(const itf2DLUT<float, float, float>&);
  void set_thickness_vws_title(const char*);
  void set_rpsq(float);
  void set_rho(float);
  void set_rpsq_vs_si_width(const std::vector<std::pair<float, float>>&);
  void set_rpsq_vws(const itf2DLUT<float, float, float>&);
  void set_rho_v_siw_t(const itf2DLUT<float, float, float>&);
  void set_rho_vws(const itf2DLUT<float, float, float>&);
  void set_tvf_bt_vws(const itf2DLUT<float, float, float>&);
  void set_tvf_bt_vwd(const itf2DLUT<float, float, float>&);
  void set_measured_from(const char*);

  // operator
  itfiConductor& operator=(const itfiConductor&);
  bool operator==(const itfiConductor&) const;

  // function
  void clear();
  void query_crt(double, double&, double&);

 private:
  // members
  char* _conductor_name;
  float _wmin;  // The minimum width of the layer. Units: microns
  float _smin;  // Minimum spacing value. Units: microns
  float _thickness; // Units: microns
  std::vector<float> _air_gap_spacings; // Units: microns
  std::vector<float> _air_gap_widths; // Units: microns
  std::vector<float> _air_gap_thicknesses; // Units: microns
  std::vector<float> _air_gap_bottom_heights; // Units: microns
  float _bottom_dielectric_thickness; // Units: microns
  float _bottom_dielectric_er; // Relative permittivity of the dielectric
  itfiBTSW _bottom_thickness_vs_si_width;
  float _t0; // Nominal temperature for the layer. Units: degrees Celsius
  std::optional<float> _crt1;
  std::optional<float> _crt2;
  std::vector<itfiWidthCrt> _crt_vs_si_width;
  std::vector<itfiDensityBox> _density_box_weighting_factor;
  float _drop_factor;
  float _etch;
  float _capacitive_only_etch;
  float _resistive_only_etch;
  std::vector<itfiEtchVWS> _etch_vws_list; // (widths, spacings) -> values
  float _fill_ratio;
  float _fill_width;
  float _fill_spacing;
  char* _fill_type;
  float _gate_to_contact_smin;
  itfiG2DC _gate_to_diffusion_cap;
  itfTitleLut<float, float, float> _ild_vws; // (widths, spacings) -> thickness_changes
  char* _layer_type;
  char* _measured_from;
  itfiPBTV _PBTV; // polynomial based thickness variation
  float _rpsq{0.};
  float _rho{0.};
  itf1DLUT<float, float> _rpsq_v_siw;
  itf2DLUT<float, float, float> _rpsq_vws; // (widths , spacings) -> RPSQ value
  itfiRhoVWS _rho_v_siw_t; // (thickness, silicon width) -> rho value
  itf2DLUT<float, float, float> _rho_vws; // (widths, spacings) -> values
  float _side_tangent;
  itfiThicknessVsDensity _thickness_vs_density;
  itfTitleLut<float, float, float> _thickness_vws; // (widths, spacings) -> values
  itf2DLUT<float, float, float> _tvf_bt_vws; // bottom_thickness_vs_width_and_spacing. (widths, spacings) -> values
  itf2DLUT<float, float, float> _tvf_bt_vwd; // bottom_thickness_vs_width_and_deltapd. (widths, deltapd) -> values

  unsigned _has_air_gap_vs_spacing : 1;
  unsigned _has_bottom_dielectric_thickness : 1;
  unsigned _has_bottom_dielectric_er : 1;
  unsigned _has_bottom_thickness_vs_si_width : 1;
  unsigned _has_t0 : 1;
  unsigned _has_crt_stmt : 1;
  unsigned _has_density_box_weighting_factor : 1;
  unsigned _has_drop_factor : 1;
  unsigned _has_etch_stmt : 1;
  unsigned _has_evwas : 1;
  unsigned _has_fill_stmt : 1;
  unsigned _has_gate_to_contact_smin : 1;
  unsigned _has_gate_to_diffusion_cap : 1;
  unsigned _has_ild_vs_width_and_spacing : 1;
  unsigned _is_planar : 1;
  unsigned _has_layer_type : 1;
  unsigned _has_measured_from : 1;
  unsigned _has_polynomial_based_thickness_variation : 1;
  unsigned _has_side_tangent : 1;
  unsigned _has_thickness_vs_density : 1;
  unsigned _has_thickness_vs_width_and_spacing : 1;
  unsigned _has_tvf_adjustment_tables : 1;
};

} // namespace itf
