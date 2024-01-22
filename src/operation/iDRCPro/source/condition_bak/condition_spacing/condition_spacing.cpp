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

#include "condition_spacing.h"

#include "IdbLayer.h"
#include "condition.h"
#include "omp.h"
#include "rule_enum.h"

namespace idrc {

bool DrcRuleConditionSpacing::checkFastMode()
{
  bool b_result = true;

  auto* engine = get_engine();
  auto* engine_manager = engine->get_engine_manager();

  int number = 0;
  int max_number = 0;
  int overlap_number = 0;
  int min_spacing_number = 0;
  std::vector<std::pair<int, int>> vio_overlaps;
  std::vector<std::pair<int, int>> vio_min_spacings;
  //   for (auto& [type, scanline_map] : engine_manager->get_scanline_matrix()) {
  auto& scanline_map = engine_manager->get_engine_scanlines(LayoutType::kRouting);
  for (auto& [layer, scanline_engine] : scanline_map) {
    if (layer->get_id() == 0) {
      continue;
    }
    auto* scanline_dm = scanline_engine->get_data_manager();
    auto& basic_pts = scanline_dm->get_basic_points();
    max_number += basic_pts.size();
#pragma omp parallel for
    for (int i = 0; i < (int) basic_pts.size(); ++i) {
      auto& basic_point = basic_pts[i];

#pragma omp parallel sections
      {
#pragma omp section
        {
          /// only check direction of up and right
          auto* neighbour_up = basic_point->get_neighbour(DrcDirection::kUp);
          if (neighbour_up != nullptr && neighbour_up->is_overlap()) {
#pragma omp critical
            {
              overlap_number++;
              vio_overlaps.push_back(std::make_pair<int, int>(basic_point->get_x(), basic_point->get_y()));
              vio_overlaps.push_back(std::make_pair<int, int>(neighbour_up->get_point()->get_x(), neighbour_up->get_point()->get_y()));
            }
          }

          if (neighbour_up != nullptr && neighbour_up->is_spacing()) {
            int spacing = neighbour_up->get_point()->get_y() - basic_point->get_y();
            if (spacing < 100 && !neighbour_up->is_overlap()) {
#pragma omp critical
              {
                min_spacing_number++;
                vio_min_spacings.push_back(std::make_pair<int, int>(basic_point->get_x(), basic_point->get_y()));
                vio_min_spacings.push_back(
                    std::make_pair<int, int>(neighbour_up->get_point()->get_x(), neighbour_up->get_point()->get_y()));
              }
            }
            if (spacing >= 100 && spacing <= 1000) {
#pragma omp critical
              {
                number++;
              }
            }
          }
        }

#pragma omp section
        {
          auto* neighbour_right = basic_point->get_neighbour(DrcDirection::kRight);
          if (neighbour_right != nullptr && neighbour_right->is_overlap()) {
#pragma omp critical
            {
              overlap_number++;
              vio_overlaps.push_back(std::make_pair<int, int>(basic_point->get_x(), basic_point->get_y()));
              vio_overlaps.push_back(
                  std::make_pair<int, int>(neighbour_right->get_point()->get_x(), neighbour_right->get_point()->get_y()));
            }
          }

          if (neighbour_right != nullptr && neighbour_right->is_spacing()) {
            int spacing = neighbour_right->get_point()->get_x() - basic_point->get_x();
            if (spacing < 100 && !neighbour_right->is_overlap()) {
#pragma omp critical
              {
                min_spacing_number++;
                vio_min_spacings.push_back(std::make_pair<int, int>(basic_point->get_x(), basic_point->get_y()));
                vio_min_spacings.push_back(
                    std::make_pair<int, int>(neighbour_right->get_point()->get_x(), neighbour_right->get_point()->get_y()));
              }
            }
            if (spacing >= 100 && spacing <= 1000) {
#pragma omp critical
              {
                number++;
              }
            }
          }
        }
      }
    }
  }

  if (overlap_number > 0) {
    int a = 0;
  }

  if (min_spacing_number > 0) {
    int a = 0;
  }

  return b_result;
}

bool DrcRuleConditionSpacing::checkCompleteMode()
{
  bool b_result = true;

  auto* engine = get_engine();

  return b_result;
}

}  // namespace idrc