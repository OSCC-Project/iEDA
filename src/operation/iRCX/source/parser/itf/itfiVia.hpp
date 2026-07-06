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

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "itf2DLUT.hpp"
namespace itf
{

// Etch Vs Contact And Gate Spacings
class itfiEVCAGS {
 public:
  // constructor
  itfiEVCAGS();
  ~itfiEVCAGS();

  // getter
  int get_number_of_tables() const;

  // setter
  void set_number_of_tables(int);
  void add_table(const char*, const itf2DLUT<float, float, float>&);

  // operator
  bool operator==(const itfiEVCAGS&) const;

  // function
  void clear();
  
 private:
  // members
  int _number_of_tables;
  std::vector<itfTitleLut<float, float, float>> _tables; // (gate_to_contact, contact_to_contact) -> values
};

struct itfiAreaRpv {
  double area;
  double rpv;

  bool operator==(const itfiAreaRpv& rhs) const {
    return (area == rhs.area) && (rpv == rhs.rpv);
  }
};

struct itfiAreaCrt {
  double area;
  double crt1;
  double crt2;

  bool operator==(const itfiAreaCrt& rhs) const {
    return (area == rhs.area) && (crt1 == rhs.crt1) && (crt2 == rhs.crt2);
  }
};

class itfiVia {
 public:
  // constructor
  itfiVia();
  itfiVia(const itfiVia&);
  ~itfiVia();

  // getter
  itfiEVCAGS& get_etch_cg();
  const itfiEVCAGS& get_etch_cg() const;
  const char* get_via_name() const;
  std::optional<float> get_rho() const;
  std::optional<float> get_rpv() const;
  const std::vector<itfiAreaRpv>& get_rpv_vs_area() const;
  std::optional<float> get_area() const;
  std::optional<float> get_crt1() const;
  std::optional<float> get_crt2() const;
  const std::vector<itfiAreaCrt>& get_crt_vs_area() const;
  float get_t0() const;
  bool has_t0() const;
  const char* get_from() const;
  const char* get_to() const;
  const itf2DLUT<float, float, std::pair<float, float>>& get_etch_vwl() const;
  const std::string& get_etch_vwl_type() const;
  bool has_etch_vwl() const;
  bool use_etch_vwl_for_resistance() const;
  bool use_etch_vwl_for_capacitance() const;

  // setter
  void set_via_name(const char*);
  void set_from(const char*);
  void set_to(const char*);
  void set_crt1(float);
  void set_crt2(float);
  void add_area_crt1_ct2(float, float, float);
  void set_t0(float);
  void set_rho(float);
  void set_rpv(float);
  void set_area(float);
  void add_area_rpv(float, float);
  void set_etch_vws(const char*, const itf2DLUT<float, float, float>&);
  void set_etch_vwl(const char*, const itf2DLUT<float, float, std::pair<float, float>>&);
  void set_capacitive_only_etch(float);

  // operator
  itfiVia& operator=(const itfiVia&);
  bool operator==(const itfiVia&) const;

  // function
  void clear();

 private:
  // members
  char* _via_name;
  char* _from;
  char* _to;
  std::optional<float> _crt1;
  std::optional<float> _crt2;
  std::vector<itfiAreaCrt> _crt_vs_area;
  float _t0;
  bool _has_t0;
  std::optional<float> _rho; // resistivity. Units: ohms-micron
  std::optional<float> _rpv;
  std::optional<float> _area;  // Area of default via. Units: square microns
  std::vector<itfiAreaRpv> _rpv_vs_area; // (RPV, area). Units: (ohms, square microns)
  itfiEVCAGS _etch_cg; //etch_vs_contact_and_gate_spacings
  itfTitleLut<float, float, float> _etch_vws; // etch_vs_width_and_spacing;
  itf2DLUT<float, float, std::pair<float, float>> _etch_vwl; // etch_vs_width_and_length; (widths, lengths) -> values)
  std::string _etch_vwl_type; // empty: capacitance and resistance; CAPACITIVE_ONLY; RESISTIVE_ONLY
  float _capacitive_only_etch;
};
  
} // namespace itf
