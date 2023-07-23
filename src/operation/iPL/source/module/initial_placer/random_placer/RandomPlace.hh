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

#ifndef IPL_RANDOM_PLACE_H
#define IPL_RANDOM_PLACE_H

#include "Log.hh"
#include "PlacerDB.hh"

namespace ipl {

class RandomPlace
{
 public:
  RandomPlace() = delete;
  explicit RandomPlace(PlacerDB* placer_db);
  RandomPlace(const RandomPlace&) = delete;
  RandomPlace(RandomPlace&&) = delete;
  ~RandomPlace() = default;

  RandomPlace& operator=(const RandomPlace&) = delete;
  RandomPlace& operator=(RandomPlace&&) = delete;

  void runRandomPlace();

 private:
  PlacerDB* _placer_db;
};
inline RandomPlace::RandomPlace(PlacerDB* placer_db) : _placer_db(placer_db)
{
}

}  // namespace ipl

#endif