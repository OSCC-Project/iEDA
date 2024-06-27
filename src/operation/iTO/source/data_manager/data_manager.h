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

#include <set>

#include "../data/Inst.h"
#include "../data/Layout.h"
#include "../data/Master.h"
#include "../data/Rects.h"
#include "../data/RowSpacing.h"

namespace ito {

const float kInf = 1E+30F;
#define REPORT_TO_TXT ;

#define toDmInst ToDataManager::get_instance()

class ToDataManager
{
 public:
  static ToDataManager* get_instance();
  static void destroy_instance();

  ito::Rectangle get_core() { return _core; }
  int get_dbu() { return _dbu; }
  int get_site_width();
  int get_site_height();

  void increDesignArea(float delta) { _design_area += delta; }

  bool reachMaxArea();

  int get_net_num() { return _num_insert_net; }
  int get_buffer_num() { return _num_insert_buf; }
  int get_resize_num() { return _number_resized_instance; }

  int add_net_num(int num = 1)
  {
    _num_insert_net += num;
    return _num_insert_net;
  }
  int add_buffer_num(int num = 1)
  {
    _num_insert_buf += num;
    return _num_insert_buf;
  }

  int add_resize_instance_num(int num = 1)
  {
    _number_resized_instance += num;
    return _number_resized_instance;
  }

  /// init operation
  std::vector<RowSpacing*> init_placer();

  /// calculate
  double calculateDesignArea(Layout* layout, int dbu);
  double calculateCoreArea(ito::Rectangle core, int dbu);
  double calcMasterArea(Master* master, int dbu);

 private:
  static ToDataManager* _instance;

  int _dbu;
  Rectangle _core;
  double _design_area = 0;
  Layout* _layout = nullptr;

  int _num_insert_net = 0;           /// number of net insert
  int _num_insert_buf = 0;           /// number of buffer insert
  int _number_resized_instance = 0;  // number of instance resize

  ToDataManager() {}
  ~ToDataManager() {}

  void initData();
};
}  // namespace ito
