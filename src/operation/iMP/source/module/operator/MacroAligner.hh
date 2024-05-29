#pragma once
#include <numeric>
#include <queue>
#include <set>

#include "Block.hh"
#include "Logger.hpp"

namespace imp {

// refine macro positions and orient
template <typename T>
struct MacroAligner
{
 public:
  MacroAligner(float notch_v_ratio = 1.5, float notch_h_ratio = 1.5) : _notch_v_ratio(notch_v_ratio), _notch_h_ratio(notch_h_ratio) {}
  ~MacroAligner() = default;

  void operator()(Block& blk)
  {
    auto sa_start = std::chrono::high_resolution_clock::now();
    alignMacrosGlobal(blk);
    auto sa_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = std::chrono::duration<float>(sa_end - sa_start);
    INFO("Macro Refinement time:", elapsed.count(), "s");
  }

  void flipBoundaryMacros(Block& blk, const std::map<std::string, std::vector<bool>>& macro_pin_locs)
  {
    std::vector<std::shared_ptr<imp::Instance>> macros = get_macros(blk);

    T outline_lx = blk.get_min_corner().x();
    T outline_ly = blk.get_min_corner().y();
    T outline_ux = outline_lx + blk.get_shape_curve().get_width();
    T outline_uy = outline_ly + blk.get_shape_curve().get_height();

    for (auto&& macro : macros) {
      const auto lx = macro->get_halo_lx();
      const auto ly = macro->get_halo_ly();
      const auto ux = macro->get_halo_ux();
      const auto uy = macro->get_halo_uy();
      auto name = macro->get_name();
      if (abs(lx - outline_lx) <= 0.2 * (_notch_h_th) && macro_pin_locs.at(name)[0] == true && macro_pin_locs.at(name)[1] == false
          || abs(ux - outline_uy) <= 0.2 * (_notch_h_th) && macro_pin_locs.at(name)[1] == true && macro_pin_locs.at(name)[1] == false) {
        macro->set_orient(flipHorizontal(macro->get_orient()));
      }
    }
  }

  // void nearBound(std::shared_ptr<imp::Instance>, const imp::geo::box<T>& outline) { if (std::abs(inst)) }
  std::vector<std::shared_ptr<imp::Instance>> get_macros(Block& blk)
  {
    std::vector<std::shared_ptr<imp::Instance>> macros;
    std::set<std::shared_ptr<imp::Instance>> macro_set = blk.get_macros();
    for (auto& macro : macro_set) {
      macros.emplace_back(macro);
    }
    return macros;
  }

  imp::geo::box<T> get_outline(Block& blk)
  {
    T outline_lx = blk.get_min_corner().x();
    T outline_ly = blk.get_min_corner().y();
    T outline_ux = outline_lx + blk.get_shape_curve().get_width();
    T outline_uy = outline_ly + blk.get_shape_curve().get_height();
    return geo::make_box(outline_lx, outline_ly, outline_ux, outline_uy);
  }

  void refineMacros(Block& blk, const std::map<std::string, std::vector<bool>>& macro_pin_locs)
  {
    std::vector<std::shared_ptr<imp::Instance>> macros = get_macros(blk);
    T outline_mid_x = (blk.get_min_corner().x() * 2 + blk.get_shape_curve().get_width()) / 2;
    T outline_mid_y = (blk.get_min_corner().y() * 2 + blk.get_shape_curve().get_height()) / 2;

    // T outline_lx = blk.get_min_corner().x();
    // T outline_ly = blk.get_min_corner().y();
    // T outline_ux = outline_lx + blk.get_shape_curve().get_width();
    // T outline_uy = outline_ly + blk.get_shape_curve().get_height();
    const imp::geo::box<T> outline = get_outline(blk);

    std::map<T, std::vector<size_t>> macro_left_y_bucket;  // macro with same y coord in same bucket;
    std::map<T, std::vector<size_t>> macro_right_y_bucket;
    std::vector<bool> fixed(macros.size(), false);
    for (size_t macro_id = 0; macro_id < macros.size(); ++macro_id) {
      int position = get_position(outline_mid_x, outline_mid_y, macros[macro_id]);
      const auto& macro = macros[macro_id];
      switch (position) {
        case 0:
          macro_left_y_bucket[macro->get_ly()].push_back(macro_id);
          break;
        case 1:
          macro_left_y_bucket[macro->get_uy()].push_back(macro_id);
          break;
        case 2:
          macro_right_y_bucket[macro->get_ly()].push_back(macro_id);
          break;
        case 3:
          macro_right_y_bucket[macro->get_uy()].push_back(macro_id);
          break;
      }
    }
    // try to move left;
    for (auto iter = macro_left_y_bucket.begin(); iter != macro_left_y_bucket.end(); ++iter) {
      if (iter->second.size() <= 1) {
        continue;
      }
      // move macros with same y coords
      auto& macro_ids = iter->second;
      // sort x ascending
      std::sort(macro_ids.begin(), macro_ids.end(),
                [&macros](size_t macro_id1, size_t macro_id2) { return macros[macro_id1]->get_lx() < macros[macro_id2]->get_lx(); });
      fixed[macro_ids[0]] = true;  // fixed Left-most macro
      for (size_t i = 1; i < macro_ids.size(); ++i) {
        if (fixed[macro_ids[i]]) {
          continue;
        }
        const auto left_macro = macros[macro_ids[i - 1]];
        const auto current_macro = macros[macro_ids[i]];
        if (left_macro->get_halo_ux() == current_macro->get_halo_lx() && macro_pin_locs.at(left_macro->get_name())[1] == false
            && macro_pin_locs.at(current_macro->get_name())[0] == false) {
          bool success = moveHor(macro_ids[i], left_macro->get_ux(), macros, outline,
                                 false);  // try to move current-macro left, not considering halo
        }
        fixed[macro_ids[i]] = true;  // fix current macro
      }
    }
    // try to move right;
    for (auto iter = macro_right_y_bucket.begin(); iter != macro_right_y_bucket.end(); ++iter) {
      if (iter->second.size() <= 1) {
        continue;
      }
      // move macros with same y coords
      auto& macro_ids = iter->second;
      // sort x descending
      std::sort(macro_ids.begin(), macro_ids.end(),
                [&macros](size_t macro_id1, size_t macro_id2) { return macros[macro_id1]->get_ux() > macros[macro_id2]->get_ux(); });
      fixed[macro_ids[0]] = true;  // fixed Right-most macro
      for (size_t i = 1; i < macro_ids.size(); ++i) {
        if (fixed[macro_ids[i]]) {
          continue;
        }
        const auto right_macro = macros[macro_ids[i - 1]];
        const auto current_macro = macros[macro_ids[i]];
        if (right_macro->get_halo_lx() == current_macro->get_halo_ux() && macro_pin_locs.at(right_macro->get_name())[0] == false
            && macro_pin_locs.at(current_macro->get_name())[1] == false) {
          moveHor(macro_ids[i], right_macro->get_lx() - current_macro->get_width(), macros, outline,
                  false);  // try to move current-macro left, not considering halo
        }
        fixed[macro_ids[i]] = true;  // fix current macro
      }
    }
  }

  int get_position(T outline_mid_x, T outline_mid_y, std::shared_ptr<imp::Instance> inst)
  {
    const auto inst_mid_x = (inst->get_lx() + inst->get_ux()) / 2;
    const auto inst_mid_y = (inst->get_ly() + inst->get_uy()) / 2;
    if (inst_mid_x < outline_mid_x) {
      if (inst_mid_y < outline_mid_y) {
        return 0;  // bottom left
      } else {
        return 1;  // top left
      }
    } else {
      if (inst_mid_y < outline_mid_y) {
        return 2;  // bottom right
      } else {
        return 3;  // top right
      }
    }
  }

  void alignMacrosGlobal(Block& blk)
  {
    // macro alginment based on OpenRoad-mpl2
    std::vector<std::shared_ptr<imp::Instance>> macros = get_macros(blk);
    T outline_lx = blk.get_min_corner().x();
    T outline_ly = blk.get_min_corner().y();
    T outline_ux = outline_lx + blk.get_shape_curve().get_width();
    T outline_uy = outline_ly + blk.get_shape_curve().get_height();
    const imp::geo::box<T> outline = geo::make_box(outline_lx, outline_ly, outline_ux, outline_uy);

    T boundary_v_th = std::numeric_limits<T>::max();
    T boundary_h_th = std::numeric_limits<T>::max();
    for (auto& macro : macros) {
      boundary_h_th = std::min(boundary_h_th, static_cast<T>(macro->get_width()));
      boundary_v_th = std::min(boundary_v_th, static_cast<T>(macro->get_height()));
    }
    const T notch_v_th = boundary_v_th * _notch_v_ratio;
    const T notch_h_th = boundary_h_th * _notch_h_ratio;
    _notch_v_th = notch_v_th;
    _notch_h_th = notch_h_th;

    // Align macros with the corresponding boundaries
    // follow the order of left, top, right, bottom
    for (size_t i = 0; i < macros.size(); ++i) {
      if (std::abs(macros[i]->get_halo_lx() - outline_lx) < boundary_h_th) {
        moveHor(i, outline_lx, macros, outline);
      }
    }
    for (size_t i = 0; i < macros.size(); ++i) {
      if (std::abs(macros[i]->get_halo_uy() - outline_ux) < boundary_v_th) {
        moveVer(i, outline_uy - macros[i]->get_halo_height(), macros, outline);
      }
    }
    for (size_t i = 0; i < macros.size(); ++i) {
      if (std::abs(macros[i]->get_halo_ux() - outline_ux) < boundary_h_th) {
        moveHor(i, outline_ux - macros[i]->get_halo_width(), macros, outline);
      }
    }
    for (size_t i = 0; i < macros.size(); ++i) {
      if (std::abs(macros[i]->get_halo_uy() - outline_ly) < boundary_v_th) {
        moveVer(i, outline_ly, macros, outline);
      }
    }

    // Comparator function to sort pairs according to second value

    std::queue<size_t> macro_queue;
    std::vector<size_t> macro_list;
    std::vector<bool> flags(macros.size(), false);
    // align to the left
    std::vector<std::pair<size_t, std::pair<T, T>>> macro_lx_map;
    for (size_t i = 0; i < macros.size(); ++i) {
      macro_lx_map.emplace_back(i, std::pair<T, T>(macros[i]->get_halo_lx(), macros[i]->get_halo_ly()));
    }
    std::sort(macro_lx_map.begin(), macro_lx_map.end(), LessOrEqualX);
    for (auto& pair : macro_lx_map) {
      if (pair.second.first <= outline_lx + boundary_h_th) {
        flags[pair.first] = true;      // fix this
        macro_queue.push(pair.first);  // use this as an anchor
      } else if (macros[pair.first]->get_halo_ux() >= outline_ux - boundary_h_th) {
        flags[pair.first] = true;  // fix this
      } else if (macros[pair.first]->get_halo_ux() <= outline_ux / 2) {
        macro_list.push_back(pair.first);
      }
    }
    while (!macro_queue.empty()) {
      const size_t macro_id = macro_queue.front();
      macro_queue.pop();
      const auto lx = macros[macro_id]->get_halo_lx();
      const auto ly = macros[macro_id]->get_halo_ly();
      const auto ux = macros[macro_id]->get_halo_ux();
      const auto uy = macros[macro_id]->get_halo_uy();
      for (auto i : macro_list) {
        if (flags[i] == true) {
          continue;
        }
        const auto lx_b = macros[i]->get_halo_lx();
        const auto ly_b = macros[i]->get_halo_ly();
        const auto ux_b = macros[i]->get_halo_ux();
        const auto uy_b = macros[i]->get_halo_uy();
        // check if adjacent
        const bool y_flag = std::abs(ly - ly_b) < notch_v_th || std::abs(ly - uy_b) < notch_v_th || std::abs(uy - ly_b) < notch_v_th
                            || std::abs(uy - uy_b) < notch_v_th;
        if (y_flag == false) {
          continue;
        }
        // try to move horizontally
        if (lx_b >= lx && lx_b <= lx + notch_h_th && lx_b < ux) {
          flags[i] = moveHor(i, lx, macros, outline);  // 距离左边界近, 左边界对齐target macro
        } else if (ux_b >= lx && ux_b <= ux && ux_b >= ux - notch_h_th) {
          flags[i] = moveHor(i, ux - macros[i]->get_halo_width(), macros, outline);  // 距离右边界近，右边界对齐target macro
        } else if (lx_b >= ux && lx_b <= ux + notch_h_th) {  // macro左边界距离target右边界近，向左贴紧target右边界。
          flags[i] = moveHor(i, ux, macros, outline);
        }
        // check if moved correctly
        if (flags[i] == true) {
          macro_queue.push(i);
        }
      }
    }

    // align to the top
    macro_list.clear();
    std::fill(flags.begin(), flags.end(), false);
    std::vector<std::pair<size_t, std::pair<T, T>>> macro_uy_map;
    for (size_t i = 0; i < macros.size(); ++i) {
      macro_uy_map.emplace_back(i, std::pair<T, T>(macros[i]->get_halo_ux(), macros[i]->get_halo_uy()));
    }
    std::sort(macro_uy_map.begin(), macro_uy_map.end(), LargeOrEqualY);
    for (auto& pair : macro_uy_map) {
      if (macros[pair.first]->get_halo_ly() <= outline_ly + boundary_v_th) {
        flags[pair.first] = true;  // fix this
      } else if (macros[pair.first]->get_halo_uy() >= outline_uy - boundary_v_th) {
        flags[pair.first] = true;      // fix this
        macro_queue.push(pair.first);  // use this as an anchor
      } else if (macros[pair.first]->get_halo_ly() >= outline_uy / 2) {
        macro_list.push_back(pair.first);
      }
    }

    while (!macro_queue.empty()) {
      const size_t macro_id = macro_queue.front();
      macro_queue.pop();
      const auto lx = macros[macro_id]->get_halo_lx();
      const auto ly = macros[macro_id]->get_halo_ly();
      const auto ux = macros[macro_id]->get_halo_ux();
      const auto uy = macros[macro_id]->get_halo_uy();
      for (auto i : macro_list) {
        if (flags[i] == true) {
          continue;
        }
        const auto lx_b = macros[i]->get_halo_lx();
        const auto ly_b = macros[i]->get_halo_ly();
        const auto ux_b = macros[i]->get_halo_ux();
        const auto uy_b = macros[i]->get_halo_uy();
        // check if adjacent
        const bool x_flag = std::abs(lx - lx_b) < notch_h_th || std::abs(lx - ux_b) < notch_h_th || std::abs(ux - lx_b) < notch_h_th
                            || std::abs(ux - ux_b) < notch_h_th;
        if (x_flag == false) {
          continue;
        }
        // try to move vertically
        if (uy_b < uy && uy_b >= uy - notch_v_th && uy_b > ly) {
          flags[i] = moveVer(i, uy - macros[i]->get_halo_height(), macros, outline);
        } else if (ly_b >= ly && ly_b <= uy && ly_b <= ly + notch_v_th) {
          flags[i] = moveVer(i, ly, macros, outline);
        } else if (uy_b <= ly && uy_b >= ly - notch_v_th) {
          flags[i] = moveVer(i, ly - macros[i]->get_halo_height(), macros, outline);
        }
        // check if moved correctly
        if (flags[i] == true) {
          macro_queue.push(i);
        }
      }
    }

    // align to the right
    macro_list.clear();
    std::fill(flags.begin(), flags.end(), false);
    std::vector<std::pair<size_t, std::pair<T, T>>> macro_ux_map;
    for (size_t i = 0; i < macros.size(); ++i) {
      macro_ux_map.emplace_back(i, std::pair<T, T>(macros[i]->get_halo_ux(), macros[i]->get_halo_uy()));
    }
    std::sort(macro_ux_map.begin(), macro_ux_map.end(), LargeOrEqualX);
    for (auto& pair : macro_ux_map) {
      if (macros[pair.first]->get_halo_lx() <= outline_lx + boundary_h_th) {
        flags[pair.first] = true;  // fix this
      } else if (macros[pair.first]->get_halo_ux() >= outline_ux - boundary_h_th) {
        flags[pair.first] = true;      // fix this
        macro_queue.push(pair.first);  // use this as an anchor
      } else if (macros[pair.first]->get_halo_ux() >= outline_ux / 2) {
        macro_list.push_back(pair.first);
      }
    }
    while (!macro_queue.empty()) {
      const size_t macro_id = macro_queue.front();
      macro_queue.pop();
      const auto lx = macros[macro_id]->get_halo_lx();
      const auto ly = macros[macro_id]->get_halo_ly();
      const auto ux = macros[macro_id]->get_halo_ux();
      const auto uy = macros[macro_id]->get_halo_uy();
      for (auto i : macro_list) {
        if (flags[i] == true) {
          continue;
        }
        const auto lx_b = macros[i]->get_halo_lx();
        const auto ly_b = macros[i]->get_halo_ly();
        const auto ux_b = macros[i]->get_halo_ux();
        const auto uy_b = macros[i]->get_halo_uy();
        // check if adjacent
        const bool y_flag = std::abs(ly - ly_b) < notch_v_th || std::abs(ly - uy_b) < notch_v_th || std::abs(uy - ly_b) < notch_v_th
                            || std::abs(uy - uy_b) < notch_v_th;
        if (y_flag == false) {
          continue;
        }
        // try to move horizontally
        if (ux_b < ux && ux_b >= ux - notch_h_th && ux_b > lx) {
          flags[i] = moveHor(i, ux - macros[i]->get_halo_width(), macros, outline);
        } else if (lx_b >= lx && lx_b <= ux && lx_b <= lx + notch_h_th) {
          flags[i] = moveHor(i, lx, macros, outline);
        } else if (ux_b <= lx && ux_b >= lx - notch_h_th) {
          flags[i] = moveHor(i, lx - macros[i]->get_halo_width(), macros, outline);
        }
        // check if moved correctly
        if (flags[i] == true) {
          macro_queue.push(i);
        }
      }
    }

    // align to the bottom
    macro_list.clear();
    std::fill(flags.begin(), flags.end(), false);
    std::vector<std::pair<size_t, std::pair<T, T>>> macro_ly_map;
    for (size_t i = 0; i < macros.size(); ++i) {
      macro_ly_map.emplace_back(i, std::pair<T, T>(macros[i]->get_halo_lx(), macros[i]->get_halo_ly()));
    }
    std::sort(macro_ly_map.begin(), macro_ly_map.end(), LessOrEqualY);
    for (auto& pair : macro_ly_map) {
      if (macros[pair.first]->get_halo_ly() <= outline_ly + boundary_v_th) {
        flags[pair.first] = true;      // fix this
        macro_queue.push(pair.first);  // use this as an anchor
      } else if (macros[pair.first]->get_halo_uy() >= outline_uy - boundary_v_th) {
        flags[pair.first] = true;  // fix this
      } else if (macros[pair.first]->get_halo_uy() <= outline_uy / 2) {
        macro_list.push_back(pair.first);
      }
    }
    while (!macro_queue.empty()) {
      const size_t macro_id = macro_queue.front();
      macro_queue.pop();
      const auto lx = macros[macro_id]->get_halo_lx();
      const auto ly = macros[macro_id]->get_halo_ly();
      const auto ux = macros[macro_id]->get_halo_ux();
      const auto uy = macros[macro_id]->get_halo_uy();
      for (auto i : macro_list) {
        if (flags[i] == true) {
          continue;
        }
        const auto lx_b = macros[i]->get_halo_lx();
        const auto ly_b = macros[i]->get_halo_ly();
        const auto ux_b = macros[i]->get_halo_ux();
        const auto uy_b = macros[i]->get_halo_uy();
        // check if adjacent
        const bool x_flag = std::abs(lx - lx_b) < notch_h_th || std::abs(lx - ux_b) < notch_h_th || std::abs(ux - lx_b) < notch_h_th
                            || std::abs(uy - ux_b) < notch_h_th;
        if (x_flag == false) {
          continue;
        }
        // try to move vertically
        if (ly_b >= ly && ly_b < ly + notch_v_th && ly_b < uy) {
          flags[i] = moveVer(i, ly, macros, outline);
        } else if (uy_b >= ly && uy_b <= uy && uy_b >= uy - notch_v_th) {
          flags[i] = moveVer(i, uy - macros[i]->get_halo_height(), macros, outline);
        } else if (ly_b >= uy && ly_b <= uy + notch_v_th) {
          flags[i] = moveVer(i, uy, macros, outline);
        }
        // check if moved correctly
        if (flags[i] == true) {
          macro_queue.push(i);
        }
      }
    }
  }

 private:
  float _notch_v_ratio;
  float _notch_h_ratio;
  float _notch_v_th;
  float _notch_h_th;

  bool isValidMove(size_t macro_id, const std::vector<std::shared_ptr<imp::Instance>>& macros, const imp::geo::box<T>& outline,
                   bool consider_halo = true) const
  {
    T macro_lx, macro_ly, macro_ux, macro_uy;
    if (consider_halo) {
      macro_lx = macros[macro_id]->get_halo_lx();
      macro_ly = macros[macro_id]->get_halo_ly();
      macro_ux = macros[macro_id]->get_halo_ux();
      macro_uy = macros[macro_id]->get_halo_uy();
    } else {
      macro_lx = macros[macro_id]->get_lx();
      macro_ly = macros[macro_id]->get_ly();
      macro_ux = macros[macro_id]->get_ux();
      macro_uy = macros[macro_id]->get_uy();
    }

    if (macro_lx < outline.min_corner().x() || macro_ly < outline.min_corner().y() || macro_ux > outline.max_corner().x()
        || macro_uy > outline.max_corner().y()) {
      return false;
    }
    for (size_t i = 0; i < macros.size(); ++i) {
      if (i == macro_id) {
        continue;
      }
      T lx, ly, ux, uy;
      if (consider_halo) {
        lx = macros[i]->get_halo_lx();
        ly = macros[i]->get_halo_ly();
        ux = macros[i]->get_halo_ux();
        uy = macros[i]->get_halo_uy();
      } else {
        lx = macros[i]->get_lx();
        ly = macros[i]->get_ly();
        ux = macros[i]->get_ux();
        uy = macros[i]->get_uy();
      }
      if (macro_lx >= ux || macro_ly >= uy || macro_ux <= lx || macro_uy <= ly) {
        continue;
      }
      return false;
    }
    return true;
  }

  bool moveHor(size_t macro_id, T x_new, std::vector<std::shared_ptr<imp::Instance>>& macros, const imp::geo::box<T>& outline,
               bool consider_halo = true)
  {
    T x_old, y_old;
    if (consider_halo == true) {
      x_old = macros[macro_id]->get_halo_lx();
      y_old = macros[macro_id]->get_halo_ly();
      macros[macro_id]->set_halo_min_corner(x_new, y_old);
    } else {
      x_old = macros[macro_id]->get_lx();
      y_old = macros[macro_id]->get_ly();
      macros[macro_id]->set_min_corner(x_new, y_old);
    }

    if (!isValidMove(macro_id, macros, outline, consider_halo)) {
      if (consider_halo == true) {
        macros[macro_id]->set_halo_min_corner(x_old, y_old);
      } else {
        macros[macro_id]->set_min_corner(x_old, y_old);
      }
      return false;
    }
    return true;
  }

  bool moveVer(size_t macro_id, T y_new, std::vector<std::shared_ptr<imp::Instance>>& macros, const imp::geo::box<T>& outline,
               bool consider_halo = true)
  {
    T x_old, y_old;
    if (consider_halo == true) {
      x_old = macros[macro_id]->get_halo_lx();
      y_old = macros[macro_id]->get_halo_ly();
      macros[macro_id]->set_halo_min_corner(x_old, y_new);
    } else {
      x_old = macros[macro_id]->get_lx();
      y_old = macros[macro_id]->get_ly();
      macros[macro_id]->set_min_corner(x_old, y_new);
    }

    if (!isValidMove(macro_id, macros, outline, consider_halo)) {
      if (consider_halo == true) {
        macros[macro_id]->set_halo_min_corner(x_old, y_old);
      } else {
        macros[macro_id]->set_min_corner(x_old, y_old);
      }
      return false;
    }
    return true;
  }

  static bool LessOrEqualX(std::pair<size_t, std::pair<T, T>>& a, std::pair<size_t, std::pair<T, T>>& b)
  {
    if (a.second.first < b.second.first) {
      return true;
    }
    if (a.second.first == b.second.first) {
      return a.second.second < b.second.second;
    }
    return false;
  }

  static bool LargeOrEqualX(std::pair<size_t, std::pair<T, T>>& a, std::pair<size_t, std::pair<T, T>>& b)
  {
    if (a.second.first > b.second.first) {
      return true;
    }
    if (a.second.first == b.second.first) {
      return a.second.second > b.second.second;
    }
    return false;
  }

  static bool LessOrEqualY(std::pair<size_t, std::pair<T, T>>& a, std::pair<size_t, std::pair<T, T>>& b)
  {
    if (a.second.second < b.second.second) {
      return true;
    }
    if (a.second.second == b.second.second) {
      return a.second.first > b.second.first;
    }
    return false;
  }

  static bool LargeOrEqualY(std::pair<size_t, std::pair<T, T>>& a, std::pair<size_t, std::pair<T, T>>& b)
  {
    if (a.second.second > b.second.second) {
      return true;
    }
    if (a.second.second == b.second.second) {
      return a.second.first < b.second.first;
    }
    return false;
  }

  Orient flipHorizontal(Orient orient)
  {
    switch (orient) {
      case Orient::kN_R0:
        return Orient::kFN_MY;
      case Orient::kFN_MY:
        return Orient::kN_R0;
      case Orient::kS_R180:
        return Orient::kFS_MX;
      case Orient::kFS_MX:
        return Orient::kS_R180;
      case Orient::kW_R90:
        return Orient::kFW_MX90;
      case Orient::kFW_MX90:
        return Orient::kW_R90;
      case Orient::kE_R270:
        return Orient::kFE_MY90;
      case Orient::kFE_MY90:
        return Orient::kE_R270;
      default:
        return Orient::kNone;
    }
  }
};

}  // namespace imp