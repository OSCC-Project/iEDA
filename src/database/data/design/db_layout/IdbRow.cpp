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
 * @file		IdbRow.h
 * @date		25/05/2021
 * @version		0.1
* @description


        Describe Row information,.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IdbRow.h"

namespace idb {

IdbRow::IdbRow()
{
  _site = new IdbSite();
  _name = "";
  _row_num_x = -1;
  _row_num_y = -1;
  _step_x = -1;
  _step_y = -1;

  _original_coordinate = new IdbCoordinate<int32_t>();
}

IdbRow::~IdbRow()
{
  if (_site != nullptr) {
    delete _site;
    _site = nullptr;
  }

  if (_original_coordinate != nullptr) {
    delete _original_coordinate;
    _original_coordinate = nullptr;
  }
}

/**
 * @brief Specifies a repeating set of sites that create the row. You must specify one of the values as 1.
 *        If you specify 1 for numY, then the row is horizontal. If you specify 1 for numX, the row is vertical.
 *
 * @param
 * @return
 */

void IdbRow::set_row_num_x(int32_t row_num_x)
{
  _row_num_x = row_num_x;
}
/**
 * @brief Specifies a repeating set of sites that create the row. You must specify one of the values as 1.
 *        If you specify 1 for numY, then the row is horizontal. If you specify 1 for numX, the row is vertical.
 *
 * @param
 * @return
 */
void IdbRow::set_row_num_y(int32_t row_num_y)
{
  _row_num_y = row_num_y;
}

bool IdbRow::set_bounding_box()
{
  int32_t ll_x = _original_coordinate->get_x();
  int32_t ll_y = _original_coordinate->get_y();
  int32_t ur_x = ll_x + _row_num_x * _step_x;
  int32_t ur_y = ll_y + _site->get_height();
  return IdbObject::set_bounding_box(ll_x, ll_y, ur_x, ur_y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IdbRows::IdbRows()
{
  _row_num = 0;
  _row_list.clear();
}

IdbRows::~IdbRows()
{
  reset();
}

IdbRow* IdbRows::find_row(string row_name)
{
  for (IdbRow* row : _row_list) {
    if (row->get_name() == row_name) {
      return row;
    }
  }

  return nullptr;
}

IdbRow* IdbRows::add_row_list(IdbRow* row)
{
  IdbRow* pRow = row;
  if (pRow == nullptr) {
    pRow = new IdbRow();
  }
  _row_list.emplace_back(pRow);
  _row_num++;

  return pRow;
}

IdbRow* IdbRows::add_row_list(string row_name)
{
  IdbRow* pRow = find_row(row_name);
  if (pRow == nullptr) {
    pRow = new IdbRow();
    pRow->set_name(row_name);
    _row_list.emplace_back(pRow);
    _row_num++;
  }

  return pRow;
}

IdbRow* IdbRows::createRow(string row_name, IdbSite* site, int32_t orig_x, int32_t orig_y, IdbOrient site_orient, int32_t num_x,
                           int32_t num_y, int32_t step_x, int32_t step_y)
{
  if (site == nullptr) {
    return nullptr;
  }

  IdbRow* row = add_row_list(row_name);
  IdbSite* row_site = site->clone();
  row->set_site(row_site);

  row_site->set_orient(site_orient);
  row->set_orient(site_orient);

  row->set_original_coordinate(orig_x, orig_y);

  row->set_row_num_x(num_x);
  row->set_row_num_y(num_y);

  row->set_step_x(step_x);
  row->set_step_y(step_y);

  row->set_bounding_box();

  return row;
}

void IdbRows::reset()
{
  for (auto& row : _row_list) {
    if (nullptr != row) {
      delete row;
      row = nullptr;
    }
  }
  _row_list.clear();

  _row_num = 0;
}

}  // namespace idb
