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
/*
 * @Author: S.J Chen
 * @Date: 2022-04-01 11:49:58
 * @LastEditTime: 2022-04-01 12:04:45
 * @LastEditors: S.J Chen
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/operator/pre_placer/center_place/CenterPlace.hh
 * Contact : https://github.com/sjchanson
 */
#ifndef IPL_OPERATOR_PP_CENTER_PLACE_H
#define IPL_OPERATOR_PP_CENTER_PLACE_H

#include "Log.hh"
#include "PlacerDB.hh"

namespace ipl {

class CenterPlace
{
 public:
  CenterPlace() = delete;
  explicit CenterPlace(PlacerDB* placer_db);
  CenterPlace(const CenterPlace&) = delete;
  CenterPlace(CenterPlace&&) = delete;
  ~CenterPlace() = default;

  CenterPlace& operator=(const CenterPlace&) = delete;
  CenterPlace& operator=(CenterPlace&&) = delete;

  void runCenterPlace();

 private:
  PlacerDB* _placer_db;
};
inline CenterPlace::CenterPlace(PlacerDB* placer_db) : _placer_db(placer_db)
{
}

}  // namespace ipl

#endif