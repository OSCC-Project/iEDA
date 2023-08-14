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
 * @file		IdbLayer.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Tech Layer information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <algorithm>

#include "IdbLayer.h"
#include "IdbRoutingLayerLef58Property.h"
#include "IdbTrackGrid.h"

namespace idb {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe  Routing Layer.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Routing Layer direction.
 *
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbLayerSpacing::IdbLayerSpacing()
{
  _min_spacing = -1;
  _min_width = -1;
  _max_width = -1;
}
IdbLayerSpacing::~IdbLayerSpacing()
{
}

bool IdbLayerSpacing::checkSpacing(int32_t spacing, int32_t width)
{
  bool result = false;
  if (_spacing_type == IdbLayerSpacingType::kSpacingDefault) {
    result = spacing >= _min_spacing ? true : false;
  } else if (_spacing_type == IdbLayerSpacingType::kSpacingRange) {
    if (width != -1 && width >= _min_width && width <= _max_width) {
      result = spacing >= _min_spacing ? true : false;
    } else {
      result = false;
    }
  } else {
    //!----------tbd----------------
    return result = false;
  }

  return result;
}

IdbLayerSpacingList::IdbLayerSpacingList()
{
  _spacing_list_num = 0;
  _spacing_list.clear();
}

IdbLayerSpacingList::~IdbLayerSpacingList()
{
  reset();
}

void IdbLayerSpacingList::add_spacing(IdbLayerSpacing* spacing)
{
  _spacing_list.emplace_back(spacing);
  _spacing_list_num++;
}

void IdbLayerSpacingList::reset()
{
  // vector<IdbLayerSpacing*>::iterator it = _spacing_list.begin();
  // for(; it != _spacing_list.end(); ++it)
  for (auto& spacing : _spacing_list) {
    if (spacing != nullptr) {
      delete spacing;
      spacing = nullptr;
    }
  }
  _spacing_list.clear();

  _spacing_list_num = 0;
}
// true : spacing is legal
// false: spacing is illeagal
bool IdbLayerSpacingList::checkSpacing(int32_t spacing, int32_t width)
{
  for (auto it : _spacing_list) {
    IdbLayerSpacing* layer_spacing = it;
    // if _spacing_list contains spacing and return true, return
    if (layer_spacing->checkSpacing(spacing, width)) {
      return true;
    }
  }

  return false;
}

int32_t IdbLayerSpacingList::get_spacing(int32_t width)
{
  int32_t spacing = 0;
  for (IdbLayerSpacing* spacing_info : _spacing_list) {
    if (spacing_info->get_spacing_type() == IdbLayerSpacingType::kSpacingDefault) {
      spacing = spacing_info->get_min_spacing();
    }
    if (width >= spacing_info->get_min_width() && width <= spacing_info->get_max_width()) {
      spacing = spacing_info->get_min_spacing();
      break;
    }
  }

  return spacing;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbMinEncloseAreaList::IdbMinEncloseAreaList()
{
  _area_list_num = 0;
  _area_list.clear();
}

IdbMinEncloseAreaList::~IdbMinEncloseAreaList()
{
  reset();
}

void IdbMinEncloseAreaList::add_min_area(int32_t area, int32_t width)
{
  IdbMinEncloseArea min_area = {area, width};
  _area_list.emplace_back(min_area);
  _area_list_num++;
}

void IdbMinEncloseAreaList::reset()
{
  _area_list.clear();
  _area_list_num = 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
IdbLayerRouting::IdbLayerRouting()
{
  _width = -1;
  _min_width = -1;
  _max_width = -1;
  _pitch.type = IdbLayerOrientType::kNone;
  _pitch.orient_x = -1;
  _pitch.orient_y = -1;
  _offset.type = IdbLayerOrientType::kNone;
  _offset.orient_x = -1;
  _offset.orient_y = -1;
  _direction = IdbLayerDirection::kNone;
  _wire_extension = -1;
  _thickness = -1;
  _resistance = -1;
  _capacitance = -1;
  _area = -1;
  _min_cut_num = -1;
  _min_cut_width = -1;

  _spacing_list = new IdbLayerSpacingList();
  _min_enclose_area_list = new IdbMinEncloseAreaList();

  set_type(IdbLayerType::kLayerRouting);

  _power_segment_width = 0;

  _spacing_table = std::make_shared<IdbLayerSpacingTable>();
}

IdbLayerRouting::~IdbLayerRouting()
{
  if (_track_grid_list.size() > 0) {
    for (IdbTrackGrid* track_grid : _track_grid_list) {
      if (track_grid) {
        track_grid = nullptr;
      }
    }
  }
  _track_grid_list.clear();
  std::vector<IdbTrackGrid*>().swap(_track_grid_list);

  if (_spacing_list) {
    delete _spacing_list;
    _spacing_list = nullptr;
  }

  if (_min_enclose_area_list) {
    delete _min_enclose_area_list;
    _min_enclose_area_list = nullptr;
  }
}

void IdbLayerRouting::set_direction(string direction_str)
{
  _direction = IdbEnum::GetInstance()->get_layer_property()->get_direction(direction_str);
}

void IdbLayerRouting::set_pitch(IdbLayerOrientValue pitch)
{
  _pitch.type = pitch.type;
  _pitch.orient_x = pitch.orient_x;
  _pitch.orient_y = pitch.orient_y;
}

void IdbLayerRouting::set_offset(IdbLayerOrientValue offset)
{
  _offset.type = offset.type;
  _offset.orient_x = offset.orient_x;
  _offset.orient_y = offset.orient_y;
}

IdbTrackGrid* IdbLayerRouting::get_prefer_track_grid()
{
  if (_direction == IdbLayerDirection::kHorizontal) {
    for (IdbTrackGrid* track_grid : _track_grid_list) {
      if (IdbTrackDirection::kDirectionY == track_grid->get_track()->get_direction()) {
        return track_grid;
      }
    }
  } else if (_direction == IdbLayerDirection::kVertical) {
    for (IdbTrackGrid* track_grid : _track_grid_list) {
      if (IdbTrackDirection::kDirectionX == track_grid->get_track()->get_direction()) {
        return track_grid;
      }
    }
  } else  // do not support other direction
  {
    return nullptr;
  }

  return nullptr;
}

IdbTrackGrid* IdbLayerRouting::get_nonprefer_track_grid()
{
  if (_direction == IdbLayerDirection::kVertical) {
    for (IdbTrackGrid* track_grid : _track_grid_list) {
      if (IdbTrackDirection::kDirectionY == track_grid->get_track()->get_direction()) {
        return track_grid;
      }
    }
  } else if (_direction == IdbLayerDirection::kHorizontal) {
    for (IdbTrackGrid* track_grid : _track_grid_list) {
      if (IdbTrackDirection::kDirectionX == track_grid->get_track()->get_direction()) {
        return track_grid;
      }
    }
  } else  // do not support other direction
  {
    return nullptr;
  }

  return nullptr;
}

int32_t IdbLayerRouting::get_spacing(int32_t width, int32_t par_length)
{
  if (this->get_spacing_table()->is_parallel()) {
    return this->get_spacing_table()->get_parallel_spacing(width, par_length);
  }
  return this->get_spacing_list()->get_spacing(width);
}

static IdbLayerSpacingTable* convertSpacingList(IdbLayerSpacingList* spacinglist)

{
  auto* tbl = new IdbLayerSpacingTable;

  int nwidth = spacinglist->get_spacing_list().size();
  auto parallel = std::make_shared<IdbParallelSpacingTable>(nwidth, 1);
  parallel->set_parallel_length(0, 0);
  auto spacings = spacinglist->get_spacing_list();
  std::sort(spacings.begin(), spacings.end(), [](IdbLayerSpacing* la, IdbLayerSpacing* lb) {
    if (la->isDefault()) {
      return true;
    }
    if (lb->isDefault()) {
      return false;
    }
    return la->get_min_width() < lb->get_min_width();
  });
  for (int i = 0; i < nwidth; ++i) {
    int w = spacings[i]->isDefault() ? 0 : spacings[i]->get_min_width();
    parallel->set_width(i, w);
    parallel->set_spacing(i, 0, spacings[i]->get_min_spacing());
  }
  tbl->set_parallel(parallel);

  return tbl;
}

std::shared_ptr<IdbLayerSpacingTable> IdbLayerRouting ::get_spacing_table()
{
  return _spacing_table;
}
std::shared_ptr<IdbLayerSpacingTable> IdbLayerRouting ::get_spacing_table_from_spacing_list()
{
  // if there is spacing_list
  // convert spacing_list to spacing table;
  if (_spacing_list) {
    return std::shared_ptr<IdbLayerSpacingTable>(convertSpacingList(_spacing_list));
  }
  return nullptr;
}

void IdbLayerRouting ::set_parallel_spacing_table(std::shared_ptr<IdbParallelSpacingTable> ptbl)
{
  _spacing_table->set_parallel(std::move(ptbl));
}

}  // namespace idb
