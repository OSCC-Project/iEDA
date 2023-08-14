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
/**
 * @project		iDB
 * @file		IdbRegion.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Region information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "IdbRegion.h"

using namespace std;
namespace idb {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbRegion::IdbRegion()
{
  _name = "";
  _type = IdbRegionType::kNone;
}

IdbRegion::~IdbRegion()
{
  clear_boundary();
}

void IdbRegion::set_type(string type)
{
  set_type(IdbEnum::GetInstance()->get_region_property()->get_type(type));
}

IdbRect* IdbRegion::add_boundary(int32_t ll_x, int32_t ll_y, int32_t ur_x, int32_t ur_y)
{
  IdbRect* rect = new IdbRect(ll_x, ll_y, ur_x, ur_y);

  _boudary_list.emplace_back(rect);

  return rect;
}

void IdbRegion::clear_boundary()
{
  for (IdbRect* boudary : _boudary_list) {
    if (boudary != nullptr) {
      delete boudary;
      boudary = nullptr;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbRegionList::IdbRegionList()
{
}

IdbRegionList::~IdbRegionList()
{
  for (IdbRegion* region : _region_list) {
    if (region) {
      delete region;
      region = nullptr;
    }
  }

  _region_list.clear();
  std::vector<IdbRegion*>().swap(_region_list);
}

IdbRegion* IdbRegionList::find_region(string name)
{
  for (IdbRegion* region : _region_list) {
    if (region->get_name() == name) {
      return region;
    }
  }

  return nullptr;
}

IdbRegion* IdbRegionList::add_region(IdbRegion* region)
{
  IdbRegion* pRegion = region;
  if (pRegion == nullptr) {
    pRegion = new IdbRegion();
  }

  _region_list.emplace_back(pRegion);

  return pRegion;
}

IdbRegion* IdbRegionList::add_region(string name)
{
  IdbRegion* pRegion = find_region(name);
  if (pRegion == nullptr) {
    pRegion = new IdbRegion();
    pRegion->set_name(name);
    _region_list.emplace_back(pRegion);
  }

  return pRegion;
}

}  // namespace idb
