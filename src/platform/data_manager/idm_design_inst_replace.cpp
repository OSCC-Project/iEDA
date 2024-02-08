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
 * @File Name: dm_design_inst.cpp
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <fstream>

#include "idm.h"

namespace idm {

struct Macro
{
  int id;
  std::string name;
  double center_x;
  double center_y;
  IdbOrient orient;
  int width;
  int height;

  Macro(int d, string n, double cx, double cy, IdbOrient o, int w, int h)
      : id(d), name(n), center_x(cx), center_y(cy), orient(o), width(w), height(h)
  {
  }
};

void DataManager::place_macro_generate_tcl(std::string directory, std::string tcl_name, int number)
{
  // 写出摆放macro的脚本
  int index = 0;
  int iterate_index = 0;

  while (index < number) {
    std::string tcl_path = directory + "/" + tcl_name + "_" + std::to_string(index) + ".tcl";
    if (true == place_macro_loc_rand(tcl_path)) {
      ++index;
    }

    std::cout << " iterate_index = " << iterate_index++ << std::endl;
  }
}

bool DataManager::place_macro_loc_rand(std::string tcl_path)
{
  auto compareByArea
      = [](const Macro& macro1, const Macro& macro2) { return (macro1.width * macro1.height) > (macro2.width * macro2.height); };


  auto isOverlap = [](const Macro& macro1, const Macro& macro2) {
    // 判断两个矩形是否重叠
    double left1 = macro1.center_x - macro1.width / 2;
    double right1 = macro1.center_x + macro1.width / 2;
    double top1 = macro1.center_y - macro1.height / 2;
    double bottom1 = macro1.center_y + macro1.height / 2;

    double left2 = macro2.center_x - macro2.width / 2;
    double right2 = macro2.center_x + macro2.width / 2;
    double top2 = macro2.center_y - macro2.height / 2;
    double bottom2 = macro2.center_y + macro2.height / 2;

    return (left1 < right2 && right1 > left2 && top1 < bottom2 && bottom1 > top2);
  };

  int size_x = 200, size_y = 200;
  std::vector<std::vector<bool>> mask_map(size_x, std::vector<bool>(size_y, false));

  double grid_width = _layout->get_die()->get_bounding_box()->get_width() / size_x;
  double grid_height = _layout->get_die()->get_bounding_box()->get_height() / size_y;
  int llx = _layout->get_die()->get_bounding_box()->get_low_x();
  int lly = _layout->get_die()->get_bounding_box()->get_low_y();
  int urx = _layout->get_die()->get_bounding_box()->get_high_x();
  int ury = _layout->get_die()->get_bounding_box()->get_high_y();
  //double Avaliable_area = (double) (urx - llx) * (double) (ury - lly);
  // std::cout << " grid_width = " << grid_width << " grid_height = " << grid_height << std::endl;
  // std::cout << " llx = " << llx << " lly = " << lly << " urx = " << urx << " ury = " << ury << std::endl;

  double total_area = 0.0;
  // Macro包含center_coor, orient, width, height(宽高可包含halo)
  std::vector<Macro> Avaliable_macro;
  // 将所有macro按大小排序存入Avalible_macro;
  IdbCore* idb_core = _layout->get_core();
  for (auto* instance : _design->get_instance_list()->get_instance_list()) {
    // 判断是否为宏单元
    // 先判断instance中点是否在core内，再判断长宽比比row height大
    if (idb_core->get_bounding_box()->containPoint(instance->get_bounding_box()->get_middle_point())) {
      if (instance->get_bounding_box()->get_height() > _layout->get_rows()->get_row_height()) {  // 这个判断macro的方式可能有问题
        int inst_id = instance->get_id();
        string inst_name = instance->get_name();
        Macro macro(inst_id, inst_name, 0.0, 0.0, instance->get_orient(), instance->get_bounding_box()->get_width(),
                    instance->get_bounding_box()->get_height());
        Avaliable_macro.push_back(macro);
        total_area += (double) instance->get_bounding_box()->get_width() * (double) instance->get_bounding_box()->get_height();
      }
    }
  }
  std::sort(Avaliable_macro.begin(), Avaliable_macro.end(), compareByArea);
  // std::cout << " Avaliable_macro.size() = " << Avaliable_macro.size() << " total_area = " << total_area
  //           << " Avaliable_area = " << Avaliable_area << std::endl;

  string orientations[] = {"R0", "MX", "R180", "MY"};
  for (size_t i = 0; i < Avaliable_macro.size(); i++) {
    // 随机一个角度，如果是90或270，交换宽高
    if (orientations[int(std::rand() % 4)] == "R0")
      Avaliable_macro[i].orient = IdbOrient::kN_R0;
    else if (orientations[int(std::rand() % 4)] == "MX")
      Avaliable_macro[i].orient = IdbOrient::kFS_MX;
    else if (orientations[int(std::rand() % 4)] == "R180")
      Avaliable_macro[i].orient = IdbOrient::kS_R180;
    else if (orientations[int(std::rand() % 4)] == "MY")
      Avaliable_macro[i].orient = IdbOrient::kFN_MY;
    else
      Avaliable_macro[i].orient = IdbOrient::kN_R0;
    if (Avaliable_macro[i].orient == IdbOrient::kE_R270 || Avaliable_macro[i].orient == IdbOrient::kW_R90) {
      int temp = Avaliable_macro[i].width;
      Avaliable_macro[i].width = Avaliable_macro[i].height;
      Avaliable_macro[i].height = temp;
    }

    // 将边界不能放的区域标记为true
    double bound_x = std::floor(Avaliable_macro[i].width / 2 / grid_width);
    double re_x = Avaliable_macro[i].width / 2 - bound_x * grid_width;
    for (int j = 0; j < size_y; j++) {
      for (int k = 0; k < bound_x + ((re_x <= grid_width / 2) ? 0 : 1); k++) {
        // if (k < size_x && j < size_y) {
        mask_map[k][j] = true;
        mask_map[size_x - k - 1][j] = true;
        // }
      }
    }
    double bound_y = std::floor(Avaliable_macro[i].height / 2 / grid_height);
    double re_y = Avaliable_macro[i].height / 2 - bound_y * grid_height;
    for (int j = 0; j < size_x; j++) {
      for (int k = 0; k < bound_y + ((re_y <= grid_height / 2) ? 0 : 1); k++) {
        // if (k < size_y && j < size_x) {
        mask_map[j][k] = true;
        mask_map[j][size_y - k - 1] = true;
        // }
      }
    }

    // 跟已有单元计算重叠
    for (size_t j = 0; j < i; j++) {
      int left_bound = static_cast<int>(std::ceil(
          ((Avaliable_macro[j].center_x - Avaliable_macro[j].width / 2 - Avaliable_macro[i].width / 2) - grid_width / 2) / grid_width));
      int right_bound = static_cast<int>(std::floor(
          ((Avaliable_macro[j].center_x + Avaliable_macro[j].width / 2 + Avaliable_macro[i].width / 2) + grid_width / 2) / grid_width));
      int low_bound = static_cast<int>(std::ceil(
          ((Avaliable_macro[j].center_y - Avaliable_macro[j].height / 2 - Avaliable_macro[i].height / 2) - grid_height / 2) / grid_height));
      int up_bound = static_cast<int>(std::floor(
          ((Avaliable_macro[j].center_y + Avaliable_macro[j].height / 2 + Avaliable_macro[i].height / 2) + grid_height / 2) / grid_height));
      for (int x_mask = left_bound; x_mask < right_bound; x_mask++) {
        for (int y_mask = low_bound; y_mask < up_bound; y_mask++) {
          if (x_mask < size_x && x_mask > 0 && y_mask < size_y && y_mask > 0) {
            mask_map[x_mask][y_mask] = true;
          }
        }
      }
    }
    std::vector<std::pair<int, int>> falseGridPoints;  // 存储为false的格点坐标
    for (size_t ii = 0; ii < mask_map.size(); ii++) {
      for (size_t jj = 0; jj < mask_map[ii].size(); jj++) {
        if (!mask_map[ii][jj]) {
          falseGridPoints.push_back(std::make_pair(ii, jj));
        }
      }
    }
    // std::cout << " falseGridPoints.size() = " << falseGridPoints.size() << std::endl;
    if (!falseGridPoints.empty()) {
      int random_number = rand() % (falseGridPoints.size() + 1);
      std::pair<int, int> selectedPoint = falseGridPoints[random_number];
      Avaliable_macro[i].center_x = selectedPoint.first * grid_width + grid_width / 2;
      Avaliable_macro[i].center_y = selectedPoint.second * grid_height + grid_height / 2;
    } else {
      std::cout << "没有可选的false格点"
                << " i = " << i << std::endl;
      return false;
    }
  }

  for (size_t i = 0; i < Avaliable_macro.size(); i++) {
    Avaliable_macro[i].center_x += llx;
    Avaliable_macro[i].center_y += lly;
  }

  for (size_t i = 0; i < Avaliable_macro.size(); i++) {
    if (Avaliable_macro[i].center_x < 0.1 || Avaliable_macro[i].center_y < 0.1) {
      std::cout << " error: exist unplaced macro" << std::endl;
      return false;
    }
  }

  for (size_t i = 0; i < Avaliable_macro.size(); i++) {
    if (Avaliable_macro[i].center_x - Avaliable_macro[i].width / 2 < llx
        || Avaliable_macro[i].center_y - Avaliable_macro[i].height / 2 < lly
        || Avaliable_macro[i].center_x + Avaliable_macro[i].width / 2 > urx
        || Avaliable_macro[i].center_y + Avaliable_macro[i].height / 2 > ury) {
      std::cout << " error: out of bound" << std::endl;
      return false;
    }
  }

  for (auto& macro1 : Avaliable_macro) {
    for (auto& macro2 : Avaliable_macro) {
      if (macro1.name.compare(macro2.name) == 1) {
        if (isOverlap(macro1, macro2)) {
          std::cout << "error, is overlap" << std::endl;
          return false;
        }
      }
    }
  }

  // 写出摆放macro的脚本
  std::ofstream outputFile(tcl_path);
  for (const Macro& macro : Avaliable_macro) {
    // 构造写入的字符串
    string ori;
    if (macro.orient == IdbOrient::kN_R0)
      ori = "R0";
    else if (macro.orient == IdbOrient::kFS_MX)
      ori = "MX";
    else if (macro.orient == IdbOrient::kS_R180)
      ori = "R180";
    else if (macro.orient == IdbOrient::kFN_MY)
      ori = "MY";

    std::string outputLine
        = "placeInstance " + macro.name + " {" + std::to_string((macro.center_x - macro.width / 2) / _layout->get_units()->get_micron_dbu())
          + " " + std::to_string((macro.center_y - macro.height / 2) / _layout->get_units()->get_micron_dbu()) + "} " + ori + " -fixed\n";

    // 写入字符串到文件
    outputFile << outputLine;
  }
  outputFile.close();

  return true;
}

void DataManager::scale_macro_loc()
{
  auto isOverlap = [](const Macro& macro1, const Macro& macro2) {
    // 判断两个矩形是否重叠
    double left1 = macro1.center_x - macro1.width / 2;
    double right1 = macro1.center_x + macro1.width / 2;
    double top1 = macro1.center_y - macro1.height / 2;
    double bottom1 = macro1.center_y + macro1.height / 2;

    double left2 = macro2.center_x - macro2.width / 2;
    double right2 = macro2.center_x + macro2.width / 2;
    double top2 = macro2.center_y - macro2.height / 2;
    double bottom2 = macro2.center_y + macro2.height / 2;

    return (left1 < right2 && right1 > left2 && top1 < bottom2 && bottom1 > top2);
  };

  double cur_width = _layout->get_die()->get_width();
  double cur_height = _layout->get_die()->get_height();
  double tar_width;
  double tar_height;

  // 计算平面的缩放比例
  double scale_x = tar_width / cur_width;
  double scale_y = tar_height / cur_height;

  std::vector<Macro> Avaliable_macro;
  // 将所有macro按大小排序存入Avalible_macro;
  for (auto* instance : _design->get_instance_list()->get_instance_list()) {
    if (instance->get_bounding_box()->get_height() > _layout->get_rows()->get_row_height()) {  // 这个判断macro的方式可能有问题
      int inst_id = instance->get_id();
      string inst_name = instance->get_name();
      auto orient = instance->get_orient();
      int width, height;
      if (orient == IdbOrient::kN_R0 || orient == IdbOrient::kS_R180 || orient == IdbOrient::kFN_MY || orient == IdbOrient::kFS_MX) {
        width = instance->get_bounding_box()->get_width();
        height = instance->get_bounding_box()->get_height();
      } else if (orient == IdbOrient::kW_R90 || orient == IdbOrient::kE_R270 || orient == IdbOrient::kFE_MY90
                 || orient == IdbOrient::kFW_MX90) {
        width = instance->get_bounding_box()->get_height();
        height = instance->get_bounding_box()->get_width();
      }
      double cx = instance->get_coordinate()->get_x() + width / 2;
      double cy = instance->get_coordinate()->get_y() + height / 2;
      Macro macro(inst_id, inst_name, cx, cy, IdbOrient::kN_R0, width, height);
      Avaliable_macro.push_back(macro);
    }
  }
  // std::vector<Rectangle> rectangles;

  // // 填充原始矩形列表，包括矩形的初始中心坐标、宽度和高度
  // rectangles.emplace_back();

  for (auto& macro : Avaliable_macro) {
    // 计算新中心坐标
    double new_center_x = macro.center_x * scale_x;
    double new_center_y = macro.center_y * scale_y;

    // 更新矩形的中心坐标
    macro.center_x = new_center_x;
    macro.center_y = new_center_y;

    if (scale_x < 1 && scale_y < 1) {
      for (auto& macro1 : Avaliable_macro) {
        for (auto& macro2 : Avaliable_macro) {
          if (abs(macro1.center_x - macro2.center_x) > 1e-3 || abs(macro1.center_y - macro2.center_y) > 1e-3) {
            if (isOverlap(macro1, macro2)) {
              std::cout << "error, is overlap" << std::endl;
              return;
            }
          }
        }
      }
    }
  }

  // 写出摆放macro的脚本
  std::ofstream outputFile("random_macro_loc.tcl");
  for (const Macro& macro : Avaliable_macro) {
    // 构造写入的字符串
    auto instance = _design->get_instance_list()->find_instance(macro.id);
    string ori;
    if (instance->get_orient() == IdbOrient::kN_R0)
      ori = "R0";
    else if (instance->get_orient() == IdbOrient::kW_R90)
      ori = "R90";
    else if (instance->get_orient() == IdbOrient::kS_R180)
      ori = "R180";
    else if (instance->get_orient() == IdbOrient::kE_R270)
      ori = "R270";
    else if (instance->get_orient() == IdbOrient::kFS_MX)
      ori = "MX";
    else if (instance->get_orient() == IdbOrient::kFN_MY)
      ori = "MY";
    else if (instance->get_orient() == IdbOrient::kFW_MX90)
      ori = "MX90";
    else if (instance->get_orient() == IdbOrient::kFE_MY90)
      ori = "MY90";

    std::string outputLine = "placeInstance " + macro.name + " {" + std::to_string(macro.center_x - macro.width / 2) + " "
                             + std::to_string(macro.center_y - macro.height / 2) + "} " + ori + " -fixed\n";

    // 写入字符串到文件
    outputFile << outputLine;
  }
  outputFile.close();
}

}  // namespace idm
