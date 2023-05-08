/**
 * iEDA
 * Copyright (C) 2021  PCL
 *
 * This program is free software;
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @project		iDB
 * @file		IdbSite.h
 * @copyright	(c) 2021 All Rights Reserved.
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Site information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IdbSite.h"

#include <algorithm>

namespace idb {

  IdbSite::IdbSite() {
    _b_overlap  = false;
    _site_class = IdbSiteClass::kNone;
    _symmetry   = IdbSymmetry::kNone;
    _orient     = IdbOrient::kNone;
    _width      = -1;
    _heigtht    = -1;
  }

  IdbSite::~IdbSite() { }

  IdbSite* IdbSite::clone() {
    IdbSite* idb_site     = new IdbSite();
    idb_site->_name       = _name;
    idb_site->_b_overlap  = _b_overlap;
    idb_site->_site_class = _site_class;
    idb_site->_symmetry   = _symmetry;
    idb_site->_orient     = _orient;
    idb_site->_width      = _width;
    idb_site->_heigtht    = _heigtht;

    return idb_site;
  }

  void IdbSite::set_class(string site_class) {
    _site_class = IdbEnum::GetInstance()->get_site_property()->get_class_type(site_class);
  }

  void IdbSite::set_orient_by_enum(int32_t lef_orient) {
    _orient = IdbEnum::GetInstance()->get_site_property()->get_orient_idb_value(lef_orient);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  IdbSites::IdbSites() {
    _site_num = 0;
    _site_list.clear();
  }

  IdbSites::~IdbSites() { reset(); }

  IdbSite* IdbSites::find_site(string site_name) {
    for (auto& site : _site_list) {
      if (site->get_name() == site_name) {
        return site;
      }
    }

    return nullptr;
  }

  IdbSite* IdbSites::add_site_list(IdbSite* site) {
    IdbSite* pSite = site;
    if (pSite == nullptr) {
      pSite = new IdbSite();
    }
    _site_list.emplace_back(pSite);
    _site_num++;

    return pSite;
  }

  IdbSite* IdbSites::add_site_list(string site_name) {
    IdbSite* pSite = find_site(site_name);
    if (pSite == nullptr) {
      pSite = new IdbSite();
      pSite->set_name(site_name);
      _site_list.emplace_back(pSite);
      _site_num++;
    }

    return pSite;
  }

  void IdbSites::reset() {
    for (auto& site : _site_list) {
      if (nullptr != site) {
        delete site;
        site = nullptr;
      }
    }
    _site_list.clear();

    _site_num = 0;
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace idb
