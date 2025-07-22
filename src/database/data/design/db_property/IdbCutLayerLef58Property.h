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
#include <variant>
#include <vector>

namespace idb::cutlayer {
/*
LEF58_CUTCLASS Definition:
[PROPERTY LEF58_CUTCLASS
        "CUTCLASS className WIDTH viaWidth [LENGTH viaLength] [CUTS numCut]
                [ORIENT {HORIZONTAL | VERTICAL}]
; " ;]
*/
class Lef58Cutclass
{
 public:
  enum Orient
  {
    kNone,
    kHorizontal,
    kVertical
  };

  [[nodiscard]] const std::string& get_class_name() const { return _class_name; }
  [[nodiscard]] int32_t get_via_width() const { return _via_width; }
  [[nodiscard]] std::optional<int32_t> get_via_length() const { return _via_length; }
  [[nodiscard]] std::optional<int32_t> get_num_cut() const { return _num_cut; }
  [[nodiscard]] Orient get_orient() { return _orient; }

  void set_class_name(std::string&& classname) { _class_name = std::move(classname); }
  void set_via_width(int32_t width) { _via_width = width; }
  void set_via_length(int32_t length) { _via_length = length; }
  void set_num_cut(int32_t cuts) { _num_cut = cuts; }
  void set_orient(Orient orinet) { _orient = orinet; }
  void set_orient(const std::string& orient)
  {
    if (orient == "HORIZONTAL") {
      _orient = kHorizontal;
    } else if (orient == "VERTICAL") {
      _orient = kVertical;
    } else {
      _orient = kNone;
    }
  }

 private:
  std::string _class_name;
  int32_t _via_width;
  std::optional<int32_t> _via_length;
  std::optional<int32_t> _num_cut;
  Orient _orient;
};

/*
        [PROPERTY LEF58_ENCLOSURE
                "ENCLOSURE [CUTCLASS className][ABOVE | BELOW]
                [MINCORNER]
                {EOL eolWidth [MINLENGTH minLength]
                        [EOLONLY] [SHORTEDGEONEOL] eolOverhang otherOverhang
                        [SIDESPACING spacing EXTENSION backwardExt forwardExt
                        |ENDSPACING spacing EXTENSION extension
                        ]
                |{overhang1 overhang2
                        |[OFFCENTERLINE] END overhang1 SIDE overhang2
                        |HORIZONTAL overhang1 VERTICAL overhang2}
                        [JOGLENGTHONLY length [INCLUDELSHAPE] ]
                        [HOLLOW {HORIZONTAL|VERTICAL} length]
                        [ WIDTH minWidth
                                [INCLUDEABUTTED]
                                [EXCEPTEXTRACUT cutWithin
                                        [PRL | NOSHAREDEDGE | EXACTPRL prl]]
                        | LENGTH minLength
                        | EXTRACUT [EXTRAONLY [PRL prl]]
                        | REDUNDANTCUT cutWithin
                        | PARALLEL parLength [parLength2] WITHIN parWithin [parWithin2]
                                [BELOWENCLOSURE belowEnclosure
                                        [ALLSIDES enclosure1 enclosure2]
                                |ABOVEENCLOSURE aboveEnclosure]
                        | CONCAVECORNERS numCorner
                        | OTHERWITHINWIDTH width WITHIN within]
                }
        ;..." ;]
 */
// TODO(Incomplete)
class Lef58Enclosure
{
 public:
  [[nodiscard]] const std::string& get_class_name() const { return _class_name; }
  [[nodiscard]] std::optional<int32_t> get_overhang1() const { return _overhang1; }
  [[nodiscard]] std::optional<int32_t> get_overhang2() const { return _overhang2; }
  [[nodiscard]] std::optional<int32_t> get_end_overhang1() const { return _end_overhang1; }
  [[nodiscard]] std::optional<int32_t> get_side_overhang2() const { return _side_overhang2; }

  void set_class_name(std::string&& name) { _class_name = std::move(name); }
  void set_overhang1(int32_t overhang1) { _overhang1 = overhang1; }
  void set_overhang2(int32_t overhang2) { _overhang2 = overhang2; }
  void set_end_overhang1(int32_t overhang1) { _end_overhang1 = overhang1; }
  void set_side_overhang2(int32_t overhang2) { _side_overhang2 = overhang2; }

 private:
  std::string _class_name;
  std::optional<int32_t> _overhang1;
  std::optional<int32_t> _overhang2;
  std::optional<int32_t> _end_overhang1;
  std::optional<int32_t> _side_overhang2;
};

/*
        [PROPERTY LEF58_ENCLOSUREEDGE
                "ENCLOSUREEDGE [CUTCLASS className][ABOVE | BELOW] overhang
                        {OPPOSITE
                                {[EXCEPTEOL eolWidth] [NOCONCAVECORNER within]
                                        [CUTTOBELOWSPACING spacing
                                                [ABOVEMETAL extension]]
                                | WRONGDIRECTION
                                }
                        |[INCLUDECORNER]
                        {WIDTH [BOTHWIRE] minWidth [maxWidth]
                        |SPANLENGTH minSpanLength [maxSpanLength] }
                                PARALLEL parLength
                                {WITHIN parWithin | WITHIN minWithin maxWithin}
                                [EXCEPTEXTRACUT [cutWithin]]
                                [EXCEPTTWOEDGES [exceptWithin]]
                        |CONVEXCORNERS convexLength adjacentLength
                                PARALLEL parWithin LENGTH length
                }
        ;..." ;]
*/
// TODO(incomplete)
class Lef58EnclosureEdge
{
 public:
  // Property Definitions
  enum Direction
  {
    kNone,
    kAbove,
    kBelow
  };
  class ConvexCorners
  {
   public:
    [[nodiscard]] int32_t get_convex_length() const { return _convex_length; }
    [[nodiscard]] int32_t get_adjacent_length() const { return _adjacent_length; }
    [[nodiscard]] int32_t get_par_within() const { return _par_within; }
    [[nodiscard]] int32_t get_length() const { return _length; }
    void set_convex_length(int32_t convex_length) { _convex_length = convex_length; }
    void set_adjacent_length(int32_t adjacent_length) { _adjacent_length = adjacent_length; }
    void set_par_within(int32_t par_within) { _par_within = par_within; }
    void set_length(int32_t length) { _length = length; }

   private:
    int32_t _convex_length;
    int32_t _adjacent_length;
    int32_t _par_within;
    int32_t _length;
  };

  // getter
  [[nodiscard]] const std::string& get_class_name() const { return _class_name; }
  [[nodiscard]] Direction get_direction() const { return _direction; }
  [[nodiscard]] int32_t get_overhang() const { return _overhang; }
  [[nodiscard]] std::optional<int32_t> get_min_width() const { return _min_width; }
  [[nodiscard]] std::optional<int32_t> get_max_width() const { return _max_width; }
  [[nodiscard]] std::optional<int32_t> get_par_length() const { return _par_length; }
  [[nodiscard]] std::optional<int32_t> get_par_within() const { return _par_within; }
  [[nodiscard]] bool has_except_extracut() const { return _except_extracut; }
  [[nodiscard]] std::optional<int32_t> get_extracut_within() const { return _cut_within; }
  [[nodiscard]] bool has_except_twoedges() const { return _except_two_edges; }
  [[nodiscard]] std::optional<int32_t> get_except_within() const { return _except_within; }
  [[nodiscard]] std::optional<ConvexCorners> get_convex_corners() const { return _convex_corners; }

  // setter
  void set_class_name(std::string&& name) { _class_name = std::move(name); }
  void set_direction(Direction direction) { _direction = direction; }
  void set_direction(const std::string& direction)
  {
    if (direction == "ABOVE") {
      _direction = kAbove;
    } else if (direction == "BELOW") {
      _direction = kBelow;
    } else {
      _direction = kNone;
    }
  }
  void set_overhang(int32_t overhang) { _overhang = overhang; }
  void set_min_width(int32_t width) { _min_width = width; }
  void set_max_width(int32_t width) { _max_width = width; }
  void set_par_length(int32_t par_length) { _par_length = par_length; }
  void set_par_within(int32_t par_within) { _par_within = par_within; }
  void set_except_extracut(bool has_except_extracut) { _except_extracut = has_except_extracut; }
  void set_except_extracut_cutwithin(int32_t cut_within) { _cut_within = cut_within; }
  void set_except_twoedges(bool has_except_twoedges) { _except_two_edges = has_except_twoedges; }
  void set_except_within(int32_t except_within) { _except_within = except_within; }
  void set_convex_corners(ConvexCorners convex_corners) { _convex_corners = convex_corners; }

 private:
  std::string _class_name;
  Direction _direction;
  int32_t _overhang;
  std::optional<int32_t> _min_width;
  std::optional<int32_t> _max_width;
  std::optional<int32_t> _par_length;
  std::optional<int32_t> _par_within;
  bool _except_extracut;
  std::optional<int32_t> _cut_within;
  bool _except_two_edges;
  std::optional<int32_t> _except_within;
  std::optional<ConvexCorners> _convex_corners;
};

class Lef58EolEnclosure
{
 public:
  enum class EdgeDirection
  {
    kNone,
    kHorizontal,
    kVertical,
  };
  enum class Direction
  {
    kNone,
    kAbove,
    kBelow
  };

  enum class ApplicationType
  {
    kNone,
    kLongEdgeOnly,
    kShortEdgeOnly,
  };
  class Extension
  {
   public:
    [[nodiscard]] int32_t get_backward_ext() const { return _backward_ext; };
    [[nodiscard]] int32_t get_forward_ext() const { return _forward_ext; }
    void set_backward_ext(int32_t backward_ext) { _backward_ext = backward_ext; }
    void set_forward_ext(int32_t forward_ext) { _forward_ext = forward_ext; }

   private:
    int32_t _backward_ext;
    int32_t _forward_ext;
  };

  // getter & setter
  [[nodiscard]] int32_t get_eol_width() const { return _eol_width; }
  [[nodiscard]] std::optional<int32_t> get_min_eol_width() const { return _min_eol_width; }
  [[nodiscard]] EdgeDirection get_edge_direction() { return _edge_direction; }
  [[nodiscard]] bool is_equal_rect_width() const { return _equal_rect_width; }
  [[nodiscard]] const std::string& get_class_name() const { return _class_name; }
  [[nodiscard]] Direction get_direction() const { return _direction; }
  [[nodiscard]] ApplicationType get_application_type() const { return _application_type; }
  [[nodiscard]] int32_t get_overhang() const { return _overhang; }
  [[nodiscard]] std::optional<int32_t> get_extract_overhang() const { return _extract_overhang; }
  [[nodiscard]] std::optional<int32_t> get_par_space() const { return _par_space; }
  [[nodiscard]] std::optional<Extension> get_extension() const { return _extension; }
  [[nodiscard]] std::optional<int32_t> get_min_length() const { return _min_length; }
  [[nodiscard]] bool is_all_sides() const { return _all_sides; }

  void set_eol_width(int32_t eol_width) { _eol_width = eol_width; }
  void set_min_eol_width(int32_t min_eol_width) { _min_eol_width = min_eol_width; }
  void set_edge_direction(EdgeDirection edge_direction) { _edge_direction = edge_direction; }
  void set_edge_direction(const std::string& edge_direction)
  {
    if (edge_direction == "HORIZONTAL") {
      set_edge_direction(EdgeDirection::kHorizontal);
    } else if (edge_direction == "VERTICAL") {
      set_edge_direction(EdgeDirection::kVertical);
    } else {
      set_edge_direction(EdgeDirection::kNone);
    }
  }
  void set_equal_rect_width(bool equal_rect_width) { _equal_rect_width = equal_rect_width; }
  void set_class_name(std::string&& class_name) { _class_name = std::move(class_name); }
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
  void set_application_type(ApplicationType application_type) { _application_type = application_type; }
  void set_application_type(const std::string& application_type)
  {
    if (application_type == "LONGEDGEONLY") {
      set_application_type(ApplicationType::kLongEdgeOnly);
    } else if (application_type == "SHORTEDGEONLY") {
      set_application_type(ApplicationType::kShortEdgeOnly);
    } else {
      set_application_type(ApplicationType::kNone);
    }
  }

  void set_overhang(int32_t overhang) { _overhang = overhang; }
  void set_extract_overhang(int32_t extract_overhang) { _extract_overhang = extract_overhang; }
  void set_par_space(int32_t par_space) { _par_space = par_space; }
  void set_extension(Extension extension) { _extension = extension; }
  void set_min_length(int32_t min_length) { _min_length = min_length; }
  void set_all_sides(bool all_sides) { _all_sides = all_sides; }

 private:
  int32_t _eol_width;
  std::optional<int32_t> _min_eol_width;
  EdgeDirection _edge_direction;  // VERTICAL | HORIZONTAL
  bool _equal_rect_width;
  std::string _class_name;
  Direction _direction;  // ABOVE | BELOW

  ApplicationType _application_type;  // LONGEDGEONLY | SHORTEDGEONLY
  int32_t _overhang;
  std::optional<int32_t> _extract_overhang;
  std::optional<int32_t> _par_space;
  std::optional<Extension> _extension;
  std::optional<int32_t> _min_length;
  bool _all_sides;
};

/*
        [PROPERTY LEF58_EOLSPACING
                "EOLSPACING cutSpacing1 cutSpacing2
                        [CUTCLASS className1 [{TO className2 cutSpacing1 cutSpacing2}...]]
                        ENDWIDTH eolWidth PRL prl
                        ENCLOSURE smallerOverhang equalOverhang
                        EXTENSION sideExt backwardExt SPANLENGTH spanLength
        ; " ;]
*/
class Lef58EolSpacing
{
 public:
  class ToClass
  {
   public:
    [[nodiscard]] const std::string& get_class_name() const { return _class_name; }
    [[nodiscard]] int32_t get_cut_spacing1() const { return _cut_spacing1; }
    [[nodiscard]] int32_t get_cut_spacing2() const { return _cut_spacing2; }
    void set_class_name(std::string&& name) { _class_name = std::move(name); }
    void set_cut_spacing1(int32_t cut_spacing1) { _cut_spacing1 = cut_spacing1; }
    void set_cut_spacing2(int32_t cut_spacing2) { _cut_spacing2 = cut_spacing2; }

   private:
    std::string _class_name;
    int32_t _cut_spacing1;
    int32_t _cut_spacing2;
  };

  [[nodiscard]] int32_t get_cut_spacing1() const { return _cut_spacing1; }
  [[nodiscard]] int32_t get_cut_spacing2() const { return _cut_spacing2; }
  [[nodiscard]] const std::string& get_class_name1() const { return _class_name1; }
  [[nodiscard]] const std::vector<ToClass>& get_to_classes() const { return _to_classes; }
  [[nodiscard]] int32_t get_eol_width() const { return _eol_width; }
  [[nodiscard]] int32_t get_prl() const { return _prl; }
  [[nodiscard]] int32_t get_smaller_overhang() const { return _small_overhang; }
  [[nodiscard]] int32_t get_equal_overhang() const { return _equal_overhang; }
  [[nodiscard]] int32_t get_side_ext() const { return _side_ext; }
  [[nodiscard]] int32_t get_backward_ext() const { return _backward_ext; }
  [[nodiscard]] int32_t get_span_length() const { return _span_length; }

  void set_cut_spacing1(int32_t cut_spacing1) { _cut_spacing1 = cut_spacing1; }
  void set_cut_spacing2(int32_t cut_spacing2) { _cut_spacing2 = cut_spacing2; }
  void set_class_name1(std::string&& name1) { _class_name1 = std::move(name1); }
  void add_to_class(ToClass&& to_class) { _to_classes.emplace_back(std::move(to_class)); }
  void set_eol_width(int32_t eol_width) { _eol_width = eol_width; }
  void set_prl(int32_t prl) { _prl = prl; }
  void set_small_overhang(int32_t small_overhang) { _small_overhang = small_overhang; }
  void set_equal_overhang(int32_t equal_overhang) { _equal_overhang = equal_overhang; }
  void set_side_ext(int32_t side_ext) { _side_ext = side_ext; }
  void set_backward_ext(int32_t backward_ext) { _backward_ext = backward_ext; }
  void set_span_length(int32_t span_length) { _span_length = span_length; }

 private:
  int32_t _cut_spacing1;
  int32_t _cut_spacing2;
  std::string _class_name1;
  std::vector<ToClass> _to_classes;
  int32_t _eol_width;
  int32_t _prl;
  int32_t _small_overhang;
  int32_t _equal_overhang;
  int32_t _side_ext;
  int32_t _backward_ext;
  int32_t _span_length;
};

class Lef58SpacingTable
{
 public:
  class Layer
  {
   public:
    [[nodiscard]] const std::string& get_second_layer_name() const { return _second_layer_name; }
    void set_second_layer_name(std::string&& name) { _second_layer_name = std::move(name); }

   private:
    std::string _second_layer_name;
  };
  class Prl
  {
   public:
    [[nodiscard]] int32_t get_prl() const { return _prl; }
    [[nodiscard]] bool is_maxxy() const { return _maxxy; }
    void set_prl(int32_t prl) { _prl = prl; }
    void set_maxxy(bool maxxy) { _maxxy = maxxy; }

   private:
    int32_t _prl;
    bool _maxxy;
  };
  class ClassName
  {
   public:
    [[nodiscard]] const std::string& get_class_name() const { return _class_name; }
    void set_class_name(std::string&& name) { _class_name = std::move(name); }
    explicit ClassName(std::string&& name) :_class_name(std::move(name)) {}
   private:
    std::string _class_name;
    // TODO([SIDE|END])
  };
  class CutSpacing
  {
   public:
    [[nodiscard]] std::optional<int32_t> get_cut_spacing1() const { return _cut_spacing1; }
    [[nodiscard]] std::optional<int32_t> get_cut_spacing2() const { return _cut_spacing2; }
    void set_cut_spacing1(int32_t cut_spacing1) { _cut_spacing1 = cut_spacing1; }
    void set_cut_spacing2(int32_t cut_spacing2) { _cut_spacing2 = cut_spacing2; }
    explicit CutSpacing(std::optional<int> cut_spacing1 = std::nullopt, std::optional<int> cut_spacing2 = std::nullopt)
        : _cut_spacing1(cut_spacing1), _cut_spacing2(cut_spacing2) {};

   private:
    std::optional<int32_t> _cut_spacing1;
    std::optional<int32_t> _cut_spacing2;
  };
  class CutClass{
    public:
     [[nodiscard]] const std::vector<ClassName>& get_class_name1_list() const { return _class_name1_list; }
     [[nodiscard]] const std::vector<ClassName>& get_class_name2_list() const { return _class_name2_list; }
     [[nodiscard]] CutSpacing get_cut_spacing(int name1_index, int name2_index) const { return _cut_spacing[name2_index][name1_index]; }
     void add_class_name1(ClassName&& class_name1) { _class_name1_list.emplace_back(std::move(class_name1)); }
     void add_class_name2(ClassName&& class_name2) { _class_name2_list.emplace_back(std::move(class_name2)); }
     void add_cut_spacing_row(std::vector<CutSpacing>&& cut_spacings) { _cut_spacing.emplace_back(std::move(cut_spacings)); }

    private:
     std::vector<ClassName> _class_name1_list;
     std::vector<ClassName> _class_name2_list;
     std::vector<std::vector<CutSpacing>> _cut_spacing;
  };

  [[nodiscard]] std::optional<Layer> get_second_layer() const { return _layer; }
  [[nodiscard]] std::optional<Prl> get_prl() const { return _prl; }
  [[nodiscard]] const CutClass& get_cutclass() const { return _cut_class; }
  void set_layer(Layer&& layer) { _layer = std::move(layer); }
  void set_prl(const Prl& prl){_prl = prl;}
  void set_cutclass(CutClass&& cut_class){_cut_class = std::move(cut_class);}
 private:
  std::optional<Layer> _layer;
  std::optional<Prl> _prl;
  CutClass _cut_class;
};

}  // namespace idb::cutlayer