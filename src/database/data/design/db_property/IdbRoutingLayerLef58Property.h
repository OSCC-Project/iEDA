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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
namespace idb::routinglayer {

// LEF58AREA
class Lef58Area
{
 public:
  class ExceptMinSize
  {
   public:
    ExceptMinSize() = default;
    ExceptMinSize(int32_t min_width, int32_t min_length) : _min_width(min_width), _min_length(min_length) {}

    [[nodiscard]] int32_t get_min_width() const { return _min_width; }
    [[nodiscard]] int32_t get_min_length() const { return _min_length; }
    void set_min_width(int32_t min_width) { _min_width = min_width; }
    void set_min_length(int32_t min_length) { _min_length = min_length; }

   private:
    int32_t _min_width;
    int32_t _min_length;
  };

  class ExceptEdgeLength
  {
   public:
    [[nodiscard]] std::optional<int32_t> get_min_edge_length() const { return _min_edge_length; }
    [[nodiscard]] int32_t get_max_edge_length() const { return _max_edge_length; }
    void set_min_edge_length(int32_t min_edge_length) { _min_edge_length = min_edge_length; }
    void set_max_edge_length(int32_t max_edge_length) { _max_edge_length = max_edge_length; }

   private:
    std::optional<int32_t> _min_edge_length;
    int32_t _max_edge_length;
  };

  Lef58Area() = default;
  explicit Lef58Area(int32_t min_area) : _min_area(min_area) {}
  [[nodiscard]] int32_t get_min_area() const { return _min_area; }
  [[nodiscard]] std::shared_ptr<ExceptEdgeLength> get_except_edge_length() const { return _except_edge_length; }
  [[nodiscard]] const std::vector<ExceptMinSize>& get_except_min_size() const { return _except_min_size; }

  void set_min_area(int32_t min_area) { _min_area = min_area; }
  void set_except_edge_length(std::shared_ptr<ExceptEdgeLength> excpet_edge_length)
  {
    _except_edge_length = std::move(excpet_edge_length);
  };

  void add_except_min_size(ExceptMinSize except_min_size) { _except_min_size.push_back(except_min_size); }

 private:
  int32_t _min_area;
  std::shared_ptr<ExceptEdgeLength> _except_edge_length;
  std::vector<ExceptMinSize> _except_min_size;
};

// LEF58CORNERFILLSPACING
class Lef58CornerFillSpacing
{
  /*
    [PROPERTY LEF58_CORNERFILLSPACING
        "CORNERFILLSPACING spacing EDGELENGTH length1 length2
        ADJACENTEOL eolWidth
      ; " ;]
  */
 public:
  [[nodiscard]] int32_t get_spacing() const { return _spacing; }
  [[nodiscard]] int32_t get_edge_length1() const { return _edge_length1; }
  [[nodiscard]] int32_t get_edge_length2() const { return _edge_length2; }
  [[nodiscard]] int32_t get_eol_width() const { return _eol_width; }
  void set_spacing(int32_t spacing) { _spacing = spacing; }
  void set_length1(int32_t length1) { _edge_length1 = length1; }
  void set_length2(int32_t length2) { _edge_length2 = length2; }
  void set_eol_width(int32_t eol_width) { _eol_width = eol_width; }

 private:
  int32_t _spacing;
  int32_t _edge_length1;
  int32_t _edge_length2;
  int32_t _eol_width;
};

// LEF58_MINIMUMCUT
class Lef58MinimumCut
{
 public:
  enum class Orient
  {
    kNone,
    kFromAbove,
    kFromBelow,
  };
  class CutClass
  {
   public:
    CutClass(std::string&& name, int32_t num_cuts) : _class_name(std::move(name)), _num_cuts(num_cuts) {}
    CutClass() = default;
    [[nodiscard]] const std::string& get_class_name() const { return _class_name; }
    [[nodiscard]] int32_t get_num_cuts() const { return _num_cuts; }
    void set_class_name(std::string&& name) { _class_name = std::move(name); }
    void set_num_cuts(int32_t num_cuts) { _num_cuts = num_cuts; }

   private:
    std::string _class_name;
    int32_t _num_cuts;
  };
  class Length
  {
   public:
    Length() = default;
    Length(int32_t length, int32_t distance) : _length(length), _distance(distance) {}
    [[nodiscard]] int32_t get_length() const { return _length; }
    [[nodiscard]] int32_t get_distance() const { return _distance; }
    void set_length(int32_t length) { _length = length; }
    void set_distance(int32_t distance) { _distance = distance; }

   private:
    int32_t _length;
    int32_t _distance;
  };

  class Area
  {
   public:
    Area() = default;
    explicit Area(int32_t area, std::optional<int32_t> distance = std::nullopt) : _area(area), _within_distance(distance) {}
    [[nodiscard]] int32_t get_area() const { return _area; }
    [[nodiscard]] std::optional<int32_t> get_within_distance() const { return _within_distance; }
    void set_area(int32_t area) { _area = area; }
    void set_within_distance(int32_t distance) { _within_distance = distance; }

   private:
    int32_t _area;
    std::optional<int32_t> _within_distance;
  };
  [[nodiscard]] std::optional<int32_t> get_num_cuts() const { return _num_cuts; }
  [[nodiscard]] const std::vector<CutClass>& get_cut_classes() const { return _cut_classes; }
  [[nodiscard]] int32_t get_width() const { return _width; }
  [[nodiscard]] std::optional<int32_t> get_within_cut_distance() const { return _within_cut_distance; }
  [[nodiscard]] Orient get_orient() const { return _orient; }
  [[nodiscard]] std::optional<Length> get_length() const { return _length; }
  [[nodiscard]] std::optional<Area> get_area() const { return _area; }
  [[nodiscard]] bool is_same_metal_overlap() const { return _same_metal_overlap; }
  [[nodiscard]] bool is_fully_enclosed() const { return _fully_enclosed; }
  void set_num_cuts(int32_t num_cuts) { _num_cuts = num_cuts; }
  void add_cutclass(CutClass&& cut_class) { _cut_classes.push_back(std::move(cut_class)); }
  void set_width(int32_t width) { _width = width; }
  void set_within_cut_distance(int32_t distance) { _within_cut_distance = distance; }
  void set_orient(Orient orient) { _orient = orient; }
  void set_orient(const std::string& orient)
  {
    if (orient == "FROMABOVE") {
      set_orient(Orient::kFromAbove);
    } else if (orient == "FROMBELOW") {
      set_orient(Orient::kFromBelow);
    } else {
      set_orient(Orient::kNone);
    }
  }
  void set_length(Length length) { _length = length; }
  void set_area(Area area) { _area = area; }
  void set_same_metal_overlap(bool same_metal_overlap) { _same_metal_overlap = same_metal_overlap; }
  void set_fully_enclosed(bool fully_enclosed) { _fully_enclosed = fully_enclosed; }

 private:
  std::optional<int32_t> _num_cuts;
  std::vector<CutClass> _cut_classes;
  int32_t _width;
  std::optional<int32_t> _within_cut_distance;
  Orient _orient;
  std::optional<Length> _length;
  std::optional<Area> _area;
  bool _same_metal_overlap;
  bool _fully_enclosed;
};

// LEF58_MINSTEP
// TODO(incomplete property)
class Lef58MinStep
{
 public:
  class MinAdjacentLength
  {
   public:
    explicit MinAdjacentLength(int32_t min_adj_length = 0) : _min_adj_length(min_adj_length) {}
    [[nodiscard]] int32_t get_min_adj_length() const { return _min_adj_length; }
    [[nodiscard]] bool is_convex_corner() const { return _convex_corner; }
    [[nodiscard]] std::optional<int32_t> get_except_within() const { return _except_within; }
    void set_min_adj_length(int32_t min_adj_length) { _min_adj_length = min_adj_length; }
    void set_convex_corner(bool convex_corner) { _convex_corner = convex_corner; }
    void set_except_within(int32_t except_within) { _except_within = except_within; }

   private:
    int32_t _min_adj_length;
    bool _convex_corner;
    std::optional<int32_t> _except_within;
  };

  explicit Lef58MinStep(int32_t min_step_length = 0) : _min_step_length(min_step_length) {}
  [[nodiscard]] int32_t get_min_step_length() const { return _min_step_length; }
  [[nodiscard]] std::optional<int32_t> get_max_edges() const { return _max_edges; }
  [[nodiscard]] std::optional<MinAdjacentLength> get_min_adjacent_length() const { return _min_adjacent_length; }
  void set_min_step_length(int32_t min_step_length) { _min_step_length = min_step_length; }
  void set_max_edges(int32_t max_edges) { _max_edges = max_edges; }
  void set_min_adjacent_length(MinAdjacentLength& min_adjacent_length) { _min_adjacent_length = min_adjacent_length; }

 private:
  int32_t _min_step_length;
  std::optional<int32_t> _max_edges;
  std::optional<MinAdjacentLength> _min_adjacent_length;
};

// LEF58_SPACING "SPACING minSpacing NOTCHLENGTH ..."
class Lef58SpacingNotchlength
{
 public:
  Lef58SpacingNotchlength() = default;
  Lef58SpacingNotchlength(int32_t min_spacing, int32_t min_notch_length) : _min_spacing(min_spacing), _min_notch_length(min_notch_length) {}
  [[nodiscard]] int32_t get_min_spacing() const { return _min_spacing; }
  [[nodiscard]] int32_t get_min_notch_length() const { return _min_notch_length; }
  [[nodiscard]] std::optional<int32_t> get_concave_ends_side_of_notch_width() const { return _concave_ends_side_of_notch_width; }
  void set_min_spacing(int32_t min_spacing) { _min_spacing = min_spacing; }
  void set_min_notch_length(int32_t min_notch_length) { _min_notch_length = min_notch_length; }
  void set_concave_ends_side_of_notch_width(int32_t width) { _concave_ends_side_of_notch_width = width; }

 private:
  int32_t _min_spacing;
  int32_t _min_notch_length;
  std::optional<int32_t> _concave_ends_side_of_notch_width;
};

// LEF58SPACINGEOL
class Lef58SpacingEol
{
 public:
  enum class Direction
  {
    kNone,
    kBelow,
    kAbove,
  };
  class EndToEnd
  {
   public:
    [[nodiscard]] int32_t get_end_to_end_space() const { return _end_to_end_space; }
    [[nodiscard]] std::optional<int32_t> get_one_cut_space() const { return _one_cut_space; }
    [[nodiscard]] std::optional<int32_t> get_two_cut_space() const { return _two_cut_space; }
    [[nodiscard]] std::optional<int32_t> get_extionsion() const { return _extionsion; }
    [[nodiscard]] std::optional<int32_t> get_wrong_dir_extionsion() const { return _wrong_dir_extension; }
    [[nodiscard]] std::optional<int32_t> get_other_end_width() const { return _other_end_width; }

    void set_end_to_end_space(int32_t space) { _end_to_end_space = space; }
    void set_one_cut_space(int32_t space) { _one_cut_space = space; }
    void set_two_cut_space(int32_t space) { _two_cut_space = space; }
    void set_extionsion(int32_t extionsion) { _extionsion = extionsion; }
    void set_wrong_dir_extionsion(int32_t wrong_dir_extionsion) { _wrong_dir_extension = wrong_dir_extionsion; }
    void set_other_end_width(int32_t other_end_width) { _other_end_width = other_end_width; }

   private:
    int32_t _end_to_end_space;
    std::optional<int32_t> _one_cut_space;
    std::optional<int32_t> _two_cut_space;
    std::optional<int32_t> _extionsion;
    std::optional<int32_t> _wrong_dir_extension;
    std::optional<int32_t> _other_end_width;
  };

  class AdjEdgeLength
  {
   public:
    [[nodiscard]] std::optional<int32_t> get_max_length() const { return _max_length; }
    [[nodiscard]] std::optional<int32_t> get_min_length() const { return _min_length; }
    [[nodiscard]] bool is_two_sides() const { return _two_sides; }
    void set_max_length(int32_t max_length) { _max_length = max_length; }
    void set_min_length(int32_t min_length) { _min_length = min_length; }
    void set_two_sides(bool two_sides) { _two_sides = two_sides; }

   private:
    // [MAXLENGTH maxLength | MINLENGTH minLength [TWOSIDES]
    std::optional<int32_t> _max_length;
    std::optional<int32_t> _min_length;
    bool _two_sides;
  };
  class ParallelEdge
  {
   public:
    [[nodiscard]] int32_t get_par_space() const { return _par_space; }
    [[nodiscard]] bool is_subtract_eol_width() const { return _subtract_eol_width; }
    [[nodiscard]] int32_t get_par_within() const { return _par_within; }
    [[nodiscard]] std::optional<int32_t> get_prl() const { return _prl; }
    [[nodiscard]] std::optional<int32_t> get_min_length() const { return _min_length; }
    [[nodiscard]] bool is_two_edges() const { return _two_edges; }
    [[nodiscard]] bool is_same_metal() const { return _same_metal; }
    [[nodiscard]] bool is_non_eol_corner_only() const { return _non_eol_corner_only; }
    [[nodiscard]] bool is_parallel_same_mask() const { return _parallel_same_mask; }

    void set_par_space(int32_t par_space) { _par_space = par_space; }
    void set_subtract_eol_width(bool subtract_eol_width) { _subtract_eol_width = subtract_eol_width; }
    void set_par_within(int32_t par_within) { _par_within = par_within; }
    void set_prl(int32_t prl) { _prl = prl; }
    void set_min_length(int32_t min_length) { _min_length = min_length; }
    void set_two_edges(bool two_edges) { _two_edges = two_edges; }
    void set_same_metal(bool same_metal) { _same_metal = same_metal; }
    void set_non_eol_corner_only(bool non_eol_corner_only) { _non_eol_corner_only = non_eol_corner_only; }
    void set_parallel_same_mask(bool parallel_same_mask) { _parallel_same_mask = parallel_same_mask; }

   private:
    int32_t _par_space;
    bool _subtract_eol_width;
    int32_t _par_within;
    std::optional<int32_t> _prl;
    std::optional<int32_t> _min_length;
    bool _two_edges;
    bool _same_metal;
    bool _non_eol_corner_only;
    bool _parallel_same_mask;
  };

  class EncloseCut
  {
   public:
    [[nodiscard]] Direction get_direction() const { return _direction; }
    [[nodiscard]] int32_t get_enclose_dist() const { return _enclose_dist; }
    [[nodiscard]] int32_t get_cut_to_metal_space() const { return _cut_to_metal_space; }
    [[nodiscard]] bool is_all_cuts() const { return _all_cuts; }

    void set_direction(Direction direction) { _direction = direction; }
    void set_direction(const std::string& direction)
    {
      if (direction == "ABOVE") {
        set_direction(Direction::kAbove);
      } else if (direction == "BELOW") {
        set_direction(Direction::kBelow);
      } else {
        set_direction(Direction::kNone);
      }
    }
    void set_enclose_dist(int32_t dist) { _enclose_dist = dist; }
    void set_cut_to_metal_space(int32_t space) { _cut_to_metal_space = space; }
    void set_all_cuts(bool all_cuts) { _all_cuts = all_cuts; }

   private:
    Direction _direction;
    int32_t _enclose_dist;
    int32_t _cut_to_metal_space;
    bool _all_cuts;
  };

  [[nodiscard]] int32_t get_eol_space() const { return _eol_space; }
  [[nodiscard]] int32_t get_eol_width() const { return _eol_width; }
  [[nodiscard]] std::optional<int32_t> get_eol_within() const { return _eol_within; }
  [[nodiscard]] std::optional<EndToEnd> get_end_to_end() const { return _end_to_end; }
  [[nodiscard]] std::optional<AdjEdgeLength> get_adj_edge_length() const { return _adj_edge_length; }
  [[nodiscard]] std::optional<ParallelEdge> get_parallel_edge() const { return _parallel_edge; }
  [[nodiscard]] std::optional<EncloseCut> get_enclose_cut() const { return _enclose_cut; }
  void set_eol_space(int32_t eol_space) { _eol_space = eol_space; }
  void set_eol_width(int32_t eol_width) { _eol_width = eol_width; }
  void set_eol_within(int32_t eol_within) { _eol_within = eol_within; }
  void set_end_to_end(EndToEnd end_to_end) { _end_to_end = end_to_end; }
  void set_adj_edge_length(AdjEdgeLength adj_edge_length) { _adj_edge_length = adj_edge_length; }
  void set_parallel_edge(ParallelEdge parallel_edge) { _parallel_edge = parallel_edge; }
  void set_enclose_cut(EncloseCut enclose_cut) { _enclose_cut = enclose_cut; }

 private:
  int32_t _eol_space;
  int32_t _eol_width;
  std::optional<int32_t> _eol_within;
  std::optional<EndToEnd> _end_to_end;
  std::optional<AdjEdgeLength> _adj_edge_length;
  std::optional<ParallelEdge> _parallel_edge;
  std::optional<EncloseCut> _enclose_cut;
};

// LEF58_SPACINGTABLE "SPACINGTABLE JOGTOJOGSPACING ..."
class Lef58SpacingTableJogToJog
{
 public:
  class Width
  {
   public:
    Width() = default;
    Width(int32_t width, int32_t par_len, int32_t par_within, int32_t long_jog_spacing)
        : _width(width), _par_length(par_len), _par_within(par_within), _long_jog_spacing(long_jog_spacing)
    {
    }
    [[nodiscard]] int32_t get_width() const { return _width; }
    [[nodiscard]] int32_t get_par_length() const { return _par_length; }
    [[nodiscard]] int32_t get_par_within() const { return _par_within; }
    [[nodiscard]] int32_t get_long_jog_spacing() const { return _long_jog_spacing; }
    void set_width(int32_t width) { _width = width; }
    void set_par_length(int32_t par_length) { _par_length = par_length; }
    void set_par_within(int32_t par_within) { _par_within = par_within; }
    void set_long_jog_spacing(int32_t long_jog_spacing) { _long_jog_spacing = long_jog_spacing; }

   private:
    int32_t _width;
    int32_t _par_length;
    int32_t _par_within;
    int32_t _long_jog_spacing;
  };

  Lef58SpacingTableJogToJog() = default;
  Lef58SpacingTableJogToJog(int32_t jog_to_jog_spacing, int32_t jog_width, int32_t short_jog_spacing)
      : _jog_to_jog_spacing(jog_to_jog_spacing), _jog_width(jog_width), _short_jog_spacing(short_jog_spacing)
  {
  }
  [[nodiscard]] int32_t get_jog_to_jog_spacing() const { return _jog_to_jog_spacing; }
  [[nodiscard]] int32_t get_jog_width() const { return _jog_width; }
  [[nodiscard]] int32_t get_short_jog_spacing() const { return _short_jog_spacing; }
  [[nodiscard]] std::vector<Width>& get_width_list() { return _width_list; }
  void set_jog_to_jog_spacing(int32_t jog_to_jog_spacing) { _jog_to_jog_spacing = jog_to_jog_spacing; }
  void set_jog_width(int32_t jog_width) { _jog_width = jog_width; }
  void set_short_jog_spacing(int32_t short_jog_spacing) { _short_jog_spacing = short_jog_spacing; }
  template <typename... Args>
  void add_width(Args&&... args)
  {
    _width_list.emplace_back(std::forward<Args>(args)...);
  }

 private:
  int32_t _jog_to_jog_spacing;
  int32_t _jog_width;
  int32_t _short_jog_spacing;
  std::vector<Width> _width_list;
};
}  // namespace idb::routinglayer