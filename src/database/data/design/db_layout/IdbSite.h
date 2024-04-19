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
/**
 * @project		iDB
 * @file		IdbSite.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Site information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../IdbEnum.h"

using std::map;
using std::string;
using std::vector;

namespace idb {

class IdbSite
{
 public:
  IdbSite();
  ~IdbSite();

  // getter
  const string get_name() const { return _name; }
  const bool is_overlap() const { return _b_overlap; }
  const IdbSiteClass get_site_class() const { return _site_class; }
  const IdbSymmetry get_symmetry() const { return _symmetry; }
  const int32_t get_width() const { return _width; }
  const int32_t get_height() const { return _heigtht; }
  const IdbOrient get_orient() { return _orient; }

  bool is_core_site() { return _type == IdbSiteType::kCore ? true : false; }
  bool is_pad_site() { return _type == IdbSiteType::kPad ? true : false; }
  bool is_corner_site() { return _type == IdbSiteType::kCorner ? true : false; }

  IdbSite* clone();

  // setter
  void set_name(string name) { _name = name; }
  void set_class(IdbSiteClass site_class) { _site_class = site_class; }
  void set_class(string site_class);
  void set_occupied(bool b_overlap) { _b_overlap = b_overlap; }
  void set_symmetry(IdbSymmetry symmetry) { _symmetry = symmetry; }
  void set_orient(IdbOrient orient) { _orient = orient; }
  void set_orient_by_enum(int32_t lef_orient);
  void set_width(int32_t width) { _width = width; }
  void set_height(int32_t height) { _heigtht = height; }

  void set_type_core() { _type = IdbSiteType::kCore; }
  void set_type_pad() { _type = IdbSiteType::kPad; }
  void set_type_corner() { _type = IdbSiteType::kCorner; }

 private:
  string _name;
  int32_t _width;
  int32_t _heigtht;

  bool _b_overlap;
  IdbSiteClass _site_class;
  IdbSymmetry _symmetry;
  IdbOrient _orient;

  IdbSiteType _type = IdbSiteType::kNone;

  // RowPattern预留
};

class IdbSites
{
 public:
  IdbSites();
  ~IdbSites();

  // getter
  vector<IdbSite*>& get_site_list() { return _site_list; }
  const uint32_t get_sites_num() const { return _site_num; }
  IdbSite* find_site(string site_name);
  IdbSite* get_io_site() { return _io_site; }
  IdbSite* get_corner_site() { return _corner_site == nullptr ? _io_site : _corner_site; }
  IdbSite* get_core_site() { return _core_site; }
  // setter
  void set_io_site(IdbSite* site)
  {
    site->set_type_pad();
    _io_site = site;
  }
  void set_corener_site(IdbSite* site)
  {
    site->set_type_corner();
    _corner_site = site;
  }
  void set_core_site(IdbSite* site)
  {
    site->set_type_core();
    _core_site = site;
  }
  void set_sites_number(uint32_t number) { _site_num = number; }
  IdbSite* add_site_list(IdbSite* site = nullptr);
  IdbSite* add_site_list(string site_name);
  void reset();

 private:
  uint32_t _site_num;
  vector<IdbSite*> _site_list;
  IdbSite* _io_site;
  IdbSite* _corner_site;
  IdbSite* _core_site;
};

}  // namespace idb
