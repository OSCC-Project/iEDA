#pragma once
#include <numeric>
#include <queue>
#include <set>

#include "Block.hh"
#include "Logger.hpp"
#include "SAPlacer.hh"
namespace imp {

template <typename T>
class MacroAligner
{
 public:
  MacroAligner(float notch_v_ratio = 1.25, float notch_h_ratio = 1.25) : _notch_v_ratio(notch_v_ratio), _notch_h_ratio(notch_h_ratio) {}
  ~MacroAligner() = default;
  void alignMacrosGlobal(Block& blk)
  {
    // macro alginment based on OpenRoad-mpl2
    std::vector<std::shared_ptr<imp::Instance>> macros = get_macros(blk);
    T outline_lx = blk.get_min_corner().x();
    T outline_ly = blk.get_min_corner().y();
    T outline_ux = outline_lx + blk.get_shape_curve().get_width();
    T outline_uy = outline_ly + blk.get_shape_curve().get_height();
    imp::geo::box<T> outline = geo::make_box(outline_lx, outline_ly, outline_ux, outline_uy);

    T boundary_v_th = std::numeric_limits<T>::max();
    T boundary_h_th = std::numeric_limits<T>::max();
    for (auto& macro : macros) {
      boundary_h_th = std::min(boundary_h_th, static_cast<T>(macro->get_width()));
      boundary_v_th = std::min(boundary_v_th, static_cast<T>(macro->get_height()));
    }
    const T notch_v_th = boundary_v_th * _notch_v_ratio;
    const T notch_h_th = boundary_h_th * _notch_h_ratio;

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
      } else if (macros[pair.first]->get_halo_uy() <= outline_ux / 2) {
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
          flags[i] = moveHor(i, lx, macros, outline);
        } else if (ux_b >= lx && ux_b <= ux && ux_b >= ux - notch_h_th) {
          flags[i] = moveHor(i, ux - macros[i]->get_halo_width(), macros, outline);
        } else if (lx_b >= ux && lx_b <= ux + notch_h_th) {
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
  std::vector<std::shared_ptr<imp::Instance>> get_macros(Block& blk)
  {
    std::vector<std::shared_ptr<imp::Instance>> macros;
    preorder_get_macros(blk, macros);
    return macros;
  }

  void preorder_get_macros(Block& blk, std::vector<std::shared_ptr<imp::Instance>>& macros)
  {
    for (auto&& i : blk.netlist().vRange()) {
      auto sub_obj = i.property();
      if (sub_obj->isInstance()) {  // add direct instance child area
        auto sub_inst = std::static_pointer_cast<Instance, Object>(sub_obj);
        if (sub_inst->get_cell_master().isMacro()) {
          macros.push_back(sub_inst);
        }
      } else {  // add block children's instance area
        auto sub_block = std::static_pointer_cast<Block, Object>(sub_obj);
        preorder_get_macros(*sub_block, macros);
      }
    }
  }
  bool isValidMove(size_t macro_id, const std::vector<std::shared_ptr<imp::Instance>>& macros, const imp::geo::box<T>& outline) const
  {
    const auto macro_lx = macros[macro_id]->get_halo_lx();
    const auto macro_ly = macros[macro_id]->get_halo_ly();
    const auto macro_ux = macros[macro_id]->get_halo_ux();
    const auto macro_uy = macros[macro_id]->get_halo_uy();
    if (macro_lx < outline.min_corner().x() || macro_ly < outline.min_corner().y() || macro_ux > outline.max_corner().x()
        || macro_uy > outline.max_corner().y()) {
      return false;
    }
    for (size_t i = 0; i < macros.size(); ++i) {
      if (i == macro_id) {
        continue;
      }
      if (macro_lx >= macros[i]->get_halo_ux() || macro_ly >= macros[i]->get_halo_uy() || macro_ux <= macros[i]->get_halo_lx()
          || macro_uy <= macros[i]->get_halo_ly()) {
        continue;
      }
      return false;
    }
    return true;
  }

  bool moveHor(size_t macro_id, T x_new, std::vector<std::shared_ptr<imp::Instance>>& macros, const imp::geo::box<T>& outline)
  {
    const auto x_old = macros[macro_id]->get_halo_lx();
    const auto y_old = macros[macro_id]->get_halo_ly();
    macros[macro_id]->set_halo_min_corner(x_new, y_old);
    if (!isValidMove(macro_id, macros, outline)) {
      macros[macro_id]->set_halo_min_corner(x_old, y_old);
      return false;
    }
    return true;
  }

  bool moveVer(size_t macro_id, T y_new, std::vector<std::shared_ptr<imp::Instance>>& macros, const imp::geo::box<T>& outline)
  {
    const auto x_old = macros[macro_id]->get_halo_lx();
    const auto y_old = macros[macro_id]->get_halo_ly();
    macros[macro_id]->set_halo_min_corner(x_old, y_new);
    if (!isValidMove(macro_id, macros, outline)) {
      macros[macro_id]->set_halo_min_corner(x_old, y_old);
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
};

}  // namespace imp