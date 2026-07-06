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
#include "itfiVia.hpp"
#include "itfMarco.h"
#include "itfUtil.h"
namespace itf
{

itfiEVCAGS::itfiEVCAGS()
: _number_of_tables(0),
  _tables()
{

}

itfiEVCAGS::~itfiEVCAGS()
{
  clear();
}

int
itfiEVCAGS::get_number_of_tables() const
{
  return _number_of_tables;
}

void
itfiEVCAGS::set_number_of_tables(int n)
{ 
  _number_of_tables = n;
}

void
itfiEVCAGS::add_table(
  const char* title,
  const itf2DLUT<float, float, float>& lut
) {
  _tables.push_back(itfTitleLut<float, float, float>(title, lut));
}

bool
itfiEVCAGS::operator==(const itfiEVCAGS& rhs) const
{
  if (this == &rhs) return true;

  return _number_of_tables == rhs._number_of_tables
    && _tables == _tables
  ;
}

void
itfiEVCAGS::clear()
{
  _tables.clear();
  _number_of_tables = 0;
}

itfiVia::itfiVia()
: _via_name(nullptr),
  _from(nullptr),
  _to(nullptr),
  _crt1(std::nullopt),
  _crt2(std::nullopt),
  _crt_vs_area(),
  _t0(0),
  _has_t0(false),
  _rho(std::nullopt),
  _rpv(std::nullopt),
  _area(std::nullopt),
  _rpv_vs_area(),
  _etch_cg(),
  _etch_vws(),
  _etch_vwl(),
  _etch_vwl_type(),
  _capacitive_only_etch(0)
{ }

itfiVia::itfiVia(const itfiVia& other)
: _via_name(nullptr)
, _from(nullptr)
, _to(nullptr)
{
  *this = other;
}

itfiVia& itfiVia::operator=(const itfiVia& rhs)
{
  if (this == &rhs) return *this;

  ITF_STR_CPY(_via_name, rhs._via_name);
  ITF_STR_CPY(_from, rhs._from);
  ITF_STR_CPY(_to, rhs._to);
  _crt1 = rhs._crt1;
  _crt2 = rhs._crt2;
  _crt_vs_area = rhs._crt_vs_area;
  _t0 = rhs._t0;
  _has_t0 = rhs._has_t0;
  _rho = rhs._rho;
  _rpv = rhs._rpv;
  _area = rhs._area;
  _rpv_vs_area = rhs._rpv_vs_area;
  _etch_cg = rhs._etch_cg;
  _etch_vws = rhs._etch_vws;
  _etch_vwl = rhs._etch_vwl;
  _etch_vwl_type = rhs._etch_vwl_type;
  _capacitive_only_etch = rhs._capacitive_only_etch;
  
  return *this;
}

bool
itfiVia::operator==(const itfiVia& rhs) const
{
  if (this == &rhs) return true;

  return itfStrCmp( _via_name, rhs._via_name)
    && itfStrCmp(_from, rhs._from)
    && itfStrCmp(_to, rhs._to)
    && _crt1 == rhs._crt1
    && _crt2 == rhs._crt2
    && _crt_vs_area == rhs._crt_vs_area
    && _t0 == rhs._t0
    && _has_t0 == rhs._has_t0
    && _rho == rhs._rho
    && _rpv == rhs._rpv
    && _area == rhs._area
    && _rpv_vs_area == rhs._rpv_vs_area
    && _etch_cg == rhs._etch_cg
    && _etch_vws == rhs._etch_vws
    && _etch_vwl == rhs._etch_vwl
    && _etch_vwl_type == rhs._etch_vwl_type
    && _capacitive_only_etch == rhs._capacitive_only_etch
  ;
}

itfiVia::~itfiVia() {
  clear();
}

void
itfiVia::clear() {
  ITF_FREE(_via_name);
  ITF_FREE(_from);
  ITF_FREE(_to);
  _crt1.reset();
  _crt2.reset();
  _crt_vs_area.clear();
  _t0 = 0;
  _has_t0 = false;
  _rho.reset();
  _rpv.reset();
  _area.reset();
  _rpv_vs_area.clear();
  _etch_cg.clear();
  _etch_vws.clear();
  _etch_vwl.clear();
  _etch_vwl_type.clear();
  _capacitive_only_etch = 0;
}

itfiEVCAGS&
itfiVia::get_etch_cg()
{
  return _etch_cg;
}

const itfiEVCAGS&
itfiVia::get_etch_cg() const
{
  return _etch_cg;
}

const char*
itfiVia::get_via_name() const
{
  return _via_name;
}

std::optional<float>
itfiVia::get_rho() const
{
  return _rho;
}

std::optional<float>
itfiVia::get_rpv() const
{ 
  return _rpv;
}

const std::vector<itfiAreaRpv>&
itfiVia::get_rpv_vs_area() const
{
  return _rpv_vs_area;
}

std::optional<float> 
itfiVia::get_area() const
{
  return _area;
}

std::optional<float>
itfiVia::get_crt1() const
{
  return _crt1;
}

std::optional<float>
itfiVia::get_crt2() const
{
  return _crt2;
}

const std::vector<itfiAreaCrt>&
itfiVia::get_crt_vs_area() const
{
  return _crt_vs_area;
}

float
itfiVia::get_t0() const
{
  return _t0;
}

bool
itfiVia::has_t0() const
{
  return _has_t0;
}

const char*
itfiVia::get_from() const
{
  return _from;
}

const char*
itfiVia::get_to() const
{
  return _to;
}

const itf2DLUT<float, float, std::pair<float, float>>&
itfiVia::get_etch_vwl() const
{
  return _etch_vwl;
}

const std::string&
itfiVia::get_etch_vwl_type() const
{
  return _etch_vwl_type;
}

bool
itfiVia::has_etch_vwl() const
{
  return !_etch_vwl.get_rows().empty() && !_etch_vwl.get_cols().empty();
}

bool
itfiVia::use_etch_vwl_for_resistance() const
{
  return has_etch_vwl() && _etch_vwl_type != "CAPACITIVE_ONLY";
}

bool
itfiVia::use_etch_vwl_for_capacitance() const
{
  return has_etch_vwl() && _etch_vwl_type != "RESISTIVE_ONLY";
}

void
itfiVia::set_via_name(const char* name)
{
  ITF_STR_CPY(_via_name, name);
}

void
itfiVia::set_from(const char* from)
{
  ITF_STR_CPY(_from, from);
}

void
itfiVia::set_to(const char* to)
{
  ITF_STR_CPY(_to, to);
}

void
itfiVia::set_crt1(float v)
{
  _crt1 = v;
}

void
itfiVia::set_crt2(float v)
{
  _crt2 = v;
}

void
itfiVia::add_area_crt1_ct2(float area, float crt1, float crt2)
{
  _crt_vs_area.emplace_back(area, crt1, crt2);
}

void
itfiVia::set_t0(float t)
{
  _t0 = t;
  _has_t0 = true;
}

void
itfiVia::set_rho(float r)
{
  _rho = r;
}

void
itfiVia::set_rpv(float v)
{
  _rpv = v;
}

void
itfiVia::set_area(float v)
{
  _area = v;
}

void
itfiVia::add_area_rpv(float area, float rpv)
{
  _rpv_vs_area.emplace_back(area, rpv);
}

void
itfiVia::set_etch_vws(const char* title, const itf2DLUT<float, float, float>& lut)
{
  _etch_vws.set_title(title);
  _etch_vws.set_lut(lut);
}

void
itfiVia::set_etch_vwl(
  const char* type,
  const itf2DLUT<float, float, std::pair<float, float>>& lut)
{
  _etch_vwl_type = type ? type : "";
  _etch_vwl = lut;
}

void
itfiVia::set_capacitive_only_etch(float v)
{
  _capacitive_only_etch = v;
}

} // namespace itf
