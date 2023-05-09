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
#include "MPEvaluation.hh"

#include "time.h"

using namespace std;

namespace ipl::imp {
float MPEvaluation::evaluate()
{
  float cost = 0;
  float hpwl = evalHPWL();
  float e_area = evalEArea();
  float guidance_penalty = evalLocationPenalty();
  cost += _weight_wl * hpwl / _norm_wl;
  cost += _weight_area * _solution->get_total_area() / _norm_area;
  cost += _weight_e_area * e_area / _norm_e_area;
  cost += _weight_guidance * guidance_penalty / _norm_guidance;
  return cost;
}

void MPEvaluation::showMassage()
{
  float hpwl = evalHPWL();
  float e_area = evalEArea();
  float area = _solution->get_total_area();
  float guidance = evalLocationPenalty();
  LOG_INFO << "wl: " << hpwl << " area: " << area << " e_area: " << e_area << " guidance: " << guidance << " cost: " << evaluate();
}

float MPEvaluation::evalHPWL()
{
  _evl_wl_count++;
  clock_t start = clock();
  float hpwl = 0;
  int32_t min_x, min_y, max_x, max_y, pin_x, pin_y;
  for (FPNet* net : _net_list) {
    vector<FPPin*> pin_list = net->get_pin_list();
    min_x = INT32_MAX;
    min_y = INT32_MAX;
    max_x = INT32_MIN;
    max_y = INT32_MIN;
    for (FPPin* pin : pin_list) {
      pin_x = pin->get_x();
      pin_y = pin->get_y();
      min_x = min(pin_x, min_x);
      min_y = min(pin_y, min_y);
      max_x = max(pin_x, max_x);
      max_y = max(pin_y, max_y);
    }
    hpwl += (max_x - min_x) + (max_y - min_y);
  }
  _evl_wl_time += double(clock() - start) / CLOCKS_PER_SEC;
  return hpwl;
}

float MPEvaluation::evalEArea()
{
  float e_area = 0;
  uint32_t placement_width = _solution->get_total_width();
  uint32_t placement_height = _solution->get_total_height();
  const float max_width = max(_core_width, placement_width);
  const float max_height = max(_core_height, placement_height);
  e_area = max(e_area, max_width * max_height - float(_core_width) * float(_core_height));
  return e_area;
}

float MPEvaluation::evalBlockagePenalty()
{
  float total_overflow = 0;
  float zero = 0;
  for (FPRect* blockage : _blockage_list) {
    float ux_one = blockage->get_x() + blockage->get_width();
    float uy_one = blockage->get_y() + blockage->get_height();
    float lx_one = blockage->get_x();
    float ly_one = blockage->get_y();
    for (FPInst* macro : _macro_list) {
      float ux_two = macro->get_x() + macro->get_width();
      float uy_two = macro->get_y() + macro->get_height();
      float lx_two = macro->get_x();
      float ly_two = macro->get_y();
      float lx = max(lx_one, lx_two);
      float ly = max(ly_one, ly_two);
      float ux = min(ux_one, ux_two);
      float uy = min(uy_one, uy_two);
      float overflow_x = ux - lx;
      float overflow_y = uy - ly;
      overflow_x = max(overflow_x, zero);
      overflow_y = max(overflow_y, zero);
      total_overflow += overflow_x * overflow_y;
    }
  }
  return total_overflow;
}

float MPEvaluation::evalBoundaryPenalty()
{
  float boundary_penalty = 0.0;
  for (FPInst* macro : _macro_list) {
    if (macro->isMacro()) {
      float lx = macro->get_x();
      float ly = macro->get_y();
      float ux = lx + macro->get_width();
      float uy = ly + macro->get_height();

      lx = min(lx, abs(_core_width - ux));
      ly = min(ly, abs(_core_height - uy));
      lx = min(lx, ly);
      boundary_penalty += lx * lx;
    }
  }
  return boundary_penalty;
}

float MPEvaluation::evalLocationPenalty()
{
  float location_penalty = 0.0;
  if (0 == _guidance_to_macro_map.size()) {
    return location_penalty;
  }

  for (auto map_iter = _guidance_to_macro_map.begin(); map_iter != _guidance_to_macro_map.end(); map_iter++) {
    const float location_width = map_iter->first->get_width();
    const float location_height = map_iter->first->get_height();
    const float location_x = map_iter->first->get_x() + 0.5 * location_width;
    const float location_y = map_iter->first->get_y() + 0.5 * location_height;

    const float macro_width = map_iter->second->get_width();
    const float macro_height = map_iter->second->get_height();
    const float macro_x = map_iter->second->get_height() + macro_width / 2.0;
    const float macro_y = map_iter->second->get_height() + macro_height / 2.0;

    float x_dist = abs(macro_x - location_x);
    float y_dist = abs(macro_y - location_y);

    const float width = (macro_width + location_width) / 2.0;
    const float height = (macro_height + location_height) / 2.0;
    x_dist = x_dist - width > 0.0 ? x_dist - width : 0.0;
    y_dist = y_dist - height > 0.0 ? y_dist - height : 0.0;
    if (x_dist >= 0.0 && y_dist >= 0.0) {
      location_penalty += max(x_dist, y_dist);
    }
  }
  return location_penalty;
}

float MPEvaluation::evalNotchPenalty()
{
  float notch_penalty = 0.0;
  uint32_t placement_width = _solution->get_total_width();
  uint32_t placement_height = _solution->get_total_height();
  if (placement_width > _core_width || placement_height > _core_height) {
    const float area = max(placement_width, _core_width) * max(placement_height, _core_height);
    notch_penalty = sqrt(area / (_core_width * _core_height));
    return notch_penalty;
  }
  alignMacro();
  vector<float> x_vec;
  vector<float> y_vec;
  for (FPInst* macro : _macro_list) {
    if (macro->isMacro()) {
      const float lx = macro->get_x();
      const float ly = macro->get_y();
      const float ux = lx + macro->get_width();
      const float uy = ly + macro->get_height();
      x_vec.emplace_back(lx);
      x_vec.emplace_back(ux);
      y_vec.emplace_back(ly);
      y_vec.emplace_back(uy);
    }
  }

  x_vec.emplace_back(0.0);
  y_vec.emplace_back(0.0);

  x_vec.emplace_back(_core_width);
  y_vec.emplace_back(_core_height);

  sort(x_vec.begin(), x_vec.end());
  sort(y_vec.begin(), y_vec.end());

  vector<float> x_grid;
  vector<float> y_grid;

  float temp_x = 0.0;
  x_grid.emplace_back(x_vec[0]);
  temp_x = x_vec[0];
  for (size_t i = 1; i < x_vec.size(); i++) {
    if (x_vec[i] - temp_x > 0.0) {
      temp_x = x_vec[i];
      x_grid.emplace_back(x_vec[i]);
    }
  }

  float temp_y = 0.0;
  y_grid.emplace_back(y_vec[0]);
  temp_y = y_vec[0];
  for (size_t i = 1; i < y_vec.size(); i++) {
    if (y_vec[i] - temp_y > 0.0) {
      temp_y = y_vec[i];
      y_grid.emplace_back(y_vec[i]);
    }
  }

  const int num_x = x_grid.size() - 1;
  const int num_y = y_grid.size() - 1;
  vector<vector<bool>> grid(num_x);
  for (int i = 0; i < num_x; i++) {
    grid[i].resize(num_y);
  }

  for (int i = 0; i < num_x; i++) {
    for (int j = 0; j < num_y; j++) {
      grid[i][j] = false;
    }
  }

  for (FPInst* macro : _macro_list) {
    if (macro->isMacro()) {
      const float lx = macro->get_x();
      const float ly = macro->get_y();
      const float ux = lx + macro->get_width();
      const float uy = ly + macro->get_height();

      int x_start = 0;
      int x_end = 0;
      int y_start = 0;
      int y_end = 0;
      for (int j = 0; j < num_x; j++) {
        if ((x_grid[j] <= lx) && (lx < x_grid[j + 1]))
          x_start = j;

        if ((x_grid[j] < ux) && (ux <= x_grid[j + 1]))
          x_end = j;
      }

      for (int j = 0; j < num_y; j++) {
        if ((y_grid[j] <= ly) && (ly < y_grid[j + 1]))
          y_start = j;

        if ((y_grid[j] < uy) && (uy <= y_grid[j + 1]))
          y_end = j;
      }

      for (int k = x_start; k <= x_end; k++) {
        for (int l = y_start; l <= y_end; l++) {
          grid[k][l] = true;
        }
      }
    }
  }
  // we define the notch threshold
  const float threshold_h = _core_width / 4.0;
  const float threshold_v = _core_height / 4.0;

  for (int i = 0; i < num_x; i++) {
    for (int j = 0; j < num_y; j++) {
      bool is_notch = false;
      if (grid[i][j] == true) {
        continue;
      } else {
        if (i == 0 && j == 0) {
          if (grid[i + 1][j] == true || grid[i][j + 1] == true)
            is_notch = true;
        } else if (i == num_x - 1 && j == 0) {
          if (grid[i - 1][j] == true || grid[i][j + 1] == true)
            is_notch = true;
        } else if (i == 0 && j == num_y - 1) {
          if (grid[i + 1][j] == true || grid[i][j - 1] == true)
            is_notch = true;
        } else if (i == num_x - 1 && j == num_y - 1) {
          if (grid[i - 1][j] == true || grid[i][j - 1] == true)
            is_notch = true;
        } else if (j == 0) {
          int result = 0 + grid[i - 1][j] + grid[i + 1][j] + grid[i][j + 1];
          if (result >= 2)
            is_notch = true;
        } else if (j == num_y - 1) {
          int result = 0 + grid[i - 1][j] + grid[i + 1][j] + grid[i][j - 1];
          if (result >= 2)
            is_notch = true;
        } else if (i == 0) {
          int result = 0 + grid[i][j - 1] + grid[i][j + 1] + grid[i + 1][j];
          if (result >= 2)
            is_notch = true;
        } else if (i == num_x - 1) {
          int result = 0 + grid[i][j - 1] + grid[i][j + 1] + grid[i - 1][j];
          if (result >= 2)
            is_notch = true;
        } else {
          int result = grid[i][j + 1] + grid[i][j - 1] + grid[i - 1][j] + grid[i + 1][j];
          if (result >= 3)
            is_notch = true;
        }

        if (is_notch == true) {
          const float width = x_grid[i + 1] - x_grid[i];
          const float height = y_grid[j + 1] - y_grid[j];
          if (width <= threshold_h || height <= threshold_v) {
            notch_penalty += sqrt(width * height / (_core_width * _core_height));
          }
        }
      }
    }
  }
  return notch_penalty;
}

void MPEvaluation::init_norm(SAParam* param)
{
  int perturb_per_step = param->get_perturb_per_step();
  vector<float> area_list;
  vector<float> wl_list;
  vector<float> e_area_list;
  vector<float> guidance_list;
  _norm_area = 0;
  _norm_wl = 0;
  _norm_e_area = 0;
  _norm_guidance = 0;
  float area, wl, e_area, guidance;

  for (int i = 0; i < perturb_per_step; ++i) {
    _solution->perturb();
    area = _solution->get_total_area();
    wl = evalHPWL();
    e_area = evalEArea();
    guidance = evalLocationPenalty();
    area_list.emplace_back(area);
    wl_list.emplace_back(wl);
    e_area_list.emplace_back(e_area);
    guidance_list.emplace_back(guidance);
    _norm_area += area;
    _norm_wl += wl;
    _norm_e_area += e_area;
    _norm_guidance += guidance;
  }

  _norm_area = _norm_area / perturb_per_step;
  _norm_wl = _norm_wl / perturb_per_step;
  _norm_e_area = _norm_e_area / perturb_per_step;
  _norm_guidance = _norm_guidance / perturb_per_step;

  vector<float> cost_list;
  for (size_t i = 0; i < area_list.size(); ++i) {
    float cost = 0;
    cost += _weight_area * area_list[i] / _norm_area;
    cost += _weight_wl * wl_list[i] / _norm_wl;
    cost += _weight_e_area * e_area_list[i] / _norm_e_area;
    cost += _weight_guidance * guidance_list[i] / _norm_guidance;

    cost_list.emplace_back(cost);
  }

  float delta_cost = 0.0;
  for (size_t i = 1; i < cost_list.size(); ++i) {
    delta_cost += abs(cost_list[i] - cost_list[i - 1]);
  }
  LOG_INFO << "delta_cost: " << delta_cost;

  float init_temperature = (-1.0) * (delta_cost / (perturb_per_step - 1)) / log(_init_prob);
  param->set_init_temperature(init_temperature);
  LOG_INFO << "start_time: " << init_temperature;
  LOG_INFO << "norm_e_area: " << _norm_e_area << " norm_hpwl: " << _norm_wl;
}

void MPEvaluation::alignMacro()
{
  // horizontal threshold, we use 10% as the threshold value
  float threshold_h = _core_width / 10.0;
  // vertical threshold, we use 10% as the threshold value
  float threshold_v = _core_height / 10.0;
  for (size_t i = 0; i < _macro_list.size(); i++) {
    const int weight = _macro_list[i]->isMacro() ? 1 : 0;
    if (weight > 0) {
      const float width = _macro_list[i]->get_width();
      const float height = _macro_list[i]->get_height();
      threshold_h = min(threshold_h, width);
      threshold_v = min(threshold_v, height);
    }
  }

  // Alignment macros to boundaries
  for (size_t i = 0; i < _macro_list.size(); i++) {
    const int weight = _macro_list[i]->isMacro() ? 1 : 0;
    if (weight > 0) {
      const float lx = _macro_list[i]->get_x();
      const float ly = _macro_list[i]->get_y();
      const float ux = lx + _macro_list[i]->get_width();
      const float uy = ly + _macro_list[i]->get_height();

      if (lx < threshold_h)
        _macro_list[i]->set_x(0.0);
      else if (ux < _core_width && _core_width - ux < threshold_h)
        _macro_list[i]->set_x(_core_width - _macro_list[i]->get_width());

      if (ly < threshold_v)
        _macro_list[i]->set_y(0.0);
      else if (uy < _core_height && _core_height - uy < threshold_v)
        _macro_list[i]->set_y(_core_height - _macro_list[i]->get_height());
    }
  }

  vector<int> macro_id_list;
  queue<int> macro_queue;  // seeds for alignment

  // Align macros according to X
  // left alignment
  for (size_t i = 0; i < _macro_list.size(); i++) {
    const int weight = _macro_list[i]->isMacro() ? 1 : 0;
    if (weight > 0) {
      macro_id_list.emplace_back(i);
      if (_macro_list[i]->get_x() == 0.0) {
        macro_queue.push(i);
        _macro_list[i]->set_align_flag(true);
      } else if (_macro_list[i]->get_x() + _macro_list[i]->get_width() >= _core_width)
        _macro_list[i]->set_align_flag(true);
    }
  }

  while (!macro_queue.empty()) {
    const int src = macro_queue.front();
    const float lx = _macro_list[src]->get_x();
    const float ux = _macro_list[src]->get_width() + lx;
    const float ly = _macro_list[src]->get_y();
    const float uy = _macro_list[src]->get_height() + ly;
    macro_queue.pop();
    for (auto macro_id : macro_id_list)
      if (_macro_list[macro_id]->isAlign() == false) {
        const float lx_b = _macro_list[macro_id]->get_x();
        const float ly_b = _macro_list[macro_id]->get_y();
        const float uy_b = ly_b + _macro_list[macro_id]->get_height();
        const bool y_flag = abs(ly - ly_b) <= threshold_v || abs(uy - uy_b) <= threshold_v || abs(ly - uy_b) <= threshold_v
                            || abs(uy - ly_b) <= threshold_v;
        if (y_flag == false)
          continue;
        bool x_flag = false;
        if (lx_b >= lx && lx_b <= lx + threshold_h) {
          _macro_list[macro_id]->set_x(lx);
          x_flag = true;
        } else if (lx_b >= ux && lx_b <= ux + threshold_h) {
          _macro_list[macro_id]->set_x(ux);
          x_flag = true;
        }

        if (x_flag == true) {
          if (isOverlap() == true)
            _macro_list[macro_id]->set_x(lx_b);
          else {
            macro_queue.push(macro_id);
            _macro_list[macro_id]->set_align_flag(true);
          }
        }
      }
  }

  // right alignment
  for (auto macro_id : macro_id_list) {
    _macro_list[macro_id]->set_align_flag(false);
    if (_macro_list[macro_id]->get_x() + _macro_list[macro_id]->get_width() >= _core_width) {
      _macro_list[macro_id]->set_align_flag(true);
      macro_queue.push(macro_id);
    } else if (_macro_list[macro_id]->get_x() == 0.0)
      _macro_list[macro_id]->set_align_flag(true);
  }

  while (!macro_queue.empty()) {
    const int src = macro_queue.front();
    const float lx = _macro_list[src]->get_x();
    const float ux = _macro_list[src]->get_width() + lx;
    const float ly = _macro_list[src]->get_y();
    const float uy = _macro_list[src]->get_height() + ly;
    macro_queue.pop();
    for (auto macro_id : macro_id_list)
      if (_macro_list[macro_id]->isAlign() == false) {
        const float lx_b = _macro_list[macro_id]->get_x();
        const float ly_b = _macro_list[macro_id]->get_y();
        const float ux_b = lx_b + _macro_list[macro_id]->get_width();
        const float uy_b = ly_b + _macro_list[macro_id]->get_height();
        const bool y_flag = abs(ly - ly_b) <= threshold_v || abs(uy - uy_b) <= threshold_v || abs(ly - uy_b) <= threshold_v
                            || abs(uy - ly_b) <= threshold_v;
        if (y_flag == false)
          continue;
        bool x_flag = false;
        if (ux_b <= ux && ux_b >= ux - threshold_h) {
          _macro_list[macro_id]->set_x(ux - (ux_b - lx_b));
          x_flag = true;
        } else if (ux_b <= lx && ux_b >= lx - threshold_h) {
          _macro_list[macro_id]->set_x(lx - (ux_b - lx_b));
          x_flag = true;
        }
        if (x_flag == true) {
          if (isOverlap() == true)
            _macro_list[macro_id]->set_x(lx_b);
          else {
            macro_queue.push(macro_id);
            _macro_list[macro_id]->set_align_flag(true);
          }
        }
      }
  }

  // bottom alignment
  for (auto macro_id : macro_id_list) {
    _macro_list[macro_id]->set_align_flag(false);
    if (_macro_list[macro_id]->get_y() == 0.0) {
      _macro_list[macro_id]->set_align_flag(true);
      macro_queue.push(macro_id);
    } else if (_macro_list[macro_id]->get_y() + _macro_list[macro_id]->get_height() >= _core_height)
      _macro_list[macro_id]->set_align_flag(true);
  }

  while (!macro_queue.empty()) {
    const int src = macro_queue.front();
    const float lx = _macro_list[src]->get_x();
    const float ux = _macro_list[src]->get_width() + lx;
    const float ly = _macro_list[src]->get_y();
    const float uy = _macro_list[src]->get_height() + ly;
    macro_queue.pop();
    for (auto macro_id : macro_id_list)
      if (_macro_list[macro_id]->isAlign() == false) {
        const float lx_b = _macro_list[macro_id]->get_x();
        const float ly_b = _macro_list[macro_id]->get_y();
        const float ux_b = lx_b + _macro_list[macro_id]->get_width();
        const bool x_flag = abs(lx - lx_b) <= threshold_h || abs(ux - ux_b) <= threshold_h || abs(lx - ux_b) <= threshold_h
                            || abs(ux - lx_b) <= threshold_h;
        if (x_flag == false)
          continue;
        bool y_flag = false;
        if (ly_b >= ly && ly_b <= ly + threshold_v) {
          _macro_list[macro_id]->set_y(ly);
          y_flag = true;
        } else if (ly_b >= uy && ly_b <= uy + threshold_v) {
          _macro_list[macro_id]->set_y(uy);
          y_flag = true;
        }
        if (y_flag == true) {
          if (isOverlap() == true)
            _macro_list[macro_id]->set_y(ly_b);
          else {
            macro_queue.push(macro_id);
            _macro_list[macro_id]->set_align_flag(true);
          }
        }
      }
  }

  // top alignment
  for (auto macro_id : macro_id_list) {
    _macro_list[macro_id]->set_align_flag(false);
    if (_macro_list[macro_id]->get_y() + _macro_list[macro_id]->get_height() >= _core_height) {
      _macro_list[macro_id]->set_align_flag(true);
      macro_queue.push(macro_id);
    } else if (_macro_list[macro_id]->get_y() == 0.0)
      _macro_list[macro_id]->set_align_flag(true);
  }

  while (!macro_queue.empty()) {
    const int src = macro_queue.front();
    const float lx = _macro_list[src]->get_x();
    const float ux = _macro_list[src]->get_width() + lx;
    const float ly = _macro_list[src]->get_y();
    const float uy = _macro_list[src]->get_height() + ly;
    macro_queue.pop();
    for (auto macro_id : macro_id_list)
      if (_macro_list[macro_id]->isAlign() == false) {
        const float lx_b = _macro_list[macro_id]->get_x();
        const float ly_b = _macro_list[macro_id]->get_y();
        const float ux_b = lx_b + _macro_list[macro_id]->get_width();
        const float uy_b = ly_b + _macro_list[macro_id]->get_height();
        const bool x_flag = abs(lx - lx_b) <= threshold_h || abs(ux - ux_b) <= threshold_h || abs(lx - ux_b) <= threshold_h
                            || abs(ux - lx_b) <= threshold_h;
        if (x_flag == false)
          continue;
        bool y_flag = false;
        if (uy_b <= uy && uy_b >= uy - threshold_v) {
          _macro_list[macro_id]->set_y(uy - (uy_b - ly_b));
          y_flag = true;
        } else if (uy_b <= ly && uy_b >= ly - threshold_v) {
          _macro_list[macro_id]->set_y(ly - (uy_b - ly_b));
          y_flag = true;
        }
        if (y_flag == true) {
          if (isOverlap() == true)
            _macro_list[macro_id]->set_y(ly_b);
          else {
            macro_queue.push(macro_id);
            _macro_list[macro_id]->set_align_flag(true);
          }
        }
      }
  }
}

bool MPEvaluation::isOverlap()
{
  vector<pair<float, float>> macro_block_x_list;
  vector<pair<float, float>> macro_block_y_list;

  for (size_t i = 0; i < _macro_list.size(); i++)
    if (_macro_list[i]->isMacro()) {
      const float lx = _macro_list[i]->get_x();
      const float ux = lx + _macro_list[i]->get_width();
      const float ly = _macro_list[i]->get_y();
      const float uy = ly + _macro_list[i]->get_height();
      macro_block_x_list.emplace_back(pair<float, float>(lx, ux));
      macro_block_y_list.emplace_back(pair<float, float>(ly, uy));
    }

  float overlap = 0.0;
  for (size_t i = 0; i < macro_block_x_list.size(); i++)
    for (size_t j = i + 1; j < macro_block_x_list.size(); j++) {
      const float x1 = max(macro_block_x_list[i].first, macro_block_x_list[j].first);
      const float x2 = min(macro_block_x_list[i].second, macro_block_x_list[j].second);
      const float y1 = max(macro_block_y_list[i].first, macro_block_y_list[j].first);
      const float y2 = min(macro_block_y_list[i].second, macro_block_y_list[j].second);
      const float x = 0.0;
      const float y = 0.0;
      overlap += max(x2 - x1, x) * max(y2 - y1, y);
    }

  return overlap > 0.0;
}

}  // namespace ipl::imp