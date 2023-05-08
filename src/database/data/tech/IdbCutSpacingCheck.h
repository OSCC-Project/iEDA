#ifndef IDB_CUT_SPACING_CHECK
#define IDB_CUT_SPACING_CHECK
#include <string>
#include <vector>

namespace idb {

  class IdbCutSpacingCheck {
   public:
    IdbCutSpacingCheck()
        : _cut_spacing(-1),
          _center_to_center(false),
          _same_net(false),
          _stack(false),
          _except_same_pg_net(false),
          _parallel_overlap(false),
          _second_layer_name(""),
          _ajacent_cuts(-1),
          _cut_within(-1),
          _cut_area(-1) { }
    explicit IdbCutSpacingCheck(int spacing)
        : _cut_spacing(spacing),
          _center_to_center(false),
          _same_net(false),
          _stack(false),
          _except_same_pg_net(false),
          _parallel_overlap(false),
          _second_layer_name(""),
          _ajacent_cuts(-1),
          _cut_within(-1),
          _cut_area(-1) { }
    ~IdbCutSpacingCheck() { }
    // getters
    int get_cut_spacing() const { return _cut_spacing; }
    bool get_center_to_center() const { return _center_to_center; }
    bool get_same_net() const { return _same_net; }
    bool get_stack() const { return _stack; }
    bool get_except_same_pg_net() const { return _except_same_pg_net; }
    bool get_parallel_overlap() const { return _parallel_overlap; }
    const std::string &get_second_layer_name() const { return _second_layer_name; }
    int get_ajacent_cuts() const { return _ajacent_cuts; }
    int get_cut_within() const { return _cut_within; }
    int get_cut_area() const { return _cut_area; }
    // setters
    void set_cut_spacing(int cut_spacing) { _cut_spacing = cut_spacing; }
    void set_center_to_center(bool center_to_center) { _center_to_center = center_to_center; }
    void set_same_net(bool same_net) { _same_net = same_net; }
    void set_stack(bool stack) { _stack = stack; }
    void set_except_same_pg_net(bool except_same_pg_net) { _except_same_pg_net = except_same_pg_net; }
    void set_parallel_overlap(bool parallel_overlap) { _parallel_overlap = parallel_overlap; }
    void set_second_layer_name(const std::string &second_layer_name) { _second_layer_name = second_layer_name; }
    void set_ajacent_cuts(int ajacent_cuts) { _ajacent_cuts = ajacent_cuts; }
    void set_cut_within(int cut_within) { _cut_within = cut_within; }
    void set_cut_area(int cut_area) { _cut_area = cut_area; }
    // others

   private:
    int _cut_spacing;
    bool _center_to_center;
    bool _same_net;
    bool _stack;
    bool _except_same_pg_net;
    bool _parallel_overlap;
    std::string _second_layer_name;
    int _ajacent_cuts;
    int _cut_within;
    int _cut_area;
  };

  class IdbCutSpacingCheckList {
   public:
    IdbCutSpacingCheckList() { }
    ~IdbCutSpacingCheckList() { }
    void addCutSpacingCheck(std::unique_ptr<IdbCutSpacingCheck> &check) { _cut_spacing_checks.push_back(std::move(check)); }
    IdbCutSpacingCheck *getFirstCutSpacing() { return (_cut_spacing_checks.begin())->get(); }

   private:
    std::vector<std::unique_ptr<IdbCutSpacingCheck>> _cut_spacing_checks;
  };

  class IdbLef58CutSpacingCheck {
   public:
    IdbLef58CutSpacingCheck()
        : _cut_spacing(-1),
          _same_mask(false),
          _max_xy(false),
          _center_to_center(false),
          _same_net(false),
          _same_metal(false),
          _same_via(false),
          _second_layer_name(""),
          _second_layer_num(-1),
          _stack(false),
          _orthogonal_spacing(-1),
          _cut_class_name(""),
          _cut_class_id(-1),
          _short_edge_only(false),
          _prl(-1),
          _concave_corner(false),
          _width(-1),
          _enclosure(-1),
          _edge_length(-1),
          _par_length(-1),
          _par_within(-1),
          _edge_enclosure(-1),
          _adjacent_enclosure(-1),
          _extension(-1),
          _eol_width(-1),
          _min_length(-1),
          _mask_overlap(false),
          _wrong_direction(false),
          _adjacent_cuts(-1),
          _exact_aligned_cut(-1),
          _two_cuts(-1),
          _two_cuts_spacing(-1),
          _same_cut(false),
          _cut_within1(-1),
          _cut_within2(-1),
          _except_same_power_ground_net(false),
          _except_all_within(-1),
          _above(false),
          _below(false),
          _to_all(false),
          _no_prl(false),
          _side_parallel_overlap(false),
          _parallel_overlap(false),
          _except_same_net(false),
          _except_same_metal(false),
          _except_same_metal_overlap(false),
          _except_same_via(false),
          _parallel_within(-1),
          _long_edge_only(false),
          _except_two_edges(false),
          _num_cut(-1),
          _cut_area(-1) { }
    ~IdbLef58CutSpacingCheck() { }
    // getters
    // is == what rules have; has == what derived from values
    int get_cut_spacing() const { return _cut_spacing; }
    bool is_same_mask() const { return _same_mask; }
    bool is_max_xy() const { return _max_xy; }
    bool is_center_to_center() const { return _center_to_center; }
    bool is_same_net() const { return _same_net; }
    bool is_same_metal() const { return _same_metal; }
    bool is_same_via() const { return _same_via; }
    std::string get_second_layer_name() const { return _second_layer_name; }
    bool has_second_layer() const { return (_second_layer_num != -1 || _second_layer_name != std::string("")); }
    int get_second_layer_num() const { return _second_layer_num; }
    bool is_stack() const { return _stack; }
    bool has_orthogonal_spacing() const { return (_orthogonal_spacing != -1); }
    std::string get_cut_class_name() const { return _cut_class_name; }
    bool has_cut_class() const { return (_cut_class_id != -1 || _cut_class_name != std::string("")); }
    int get_cut_class_id() const { return _cut_class_id; }
    bool is_short_edge_only() const { return _short_edge_only; }
    bool has_prl() const { return (_prl != -1); }
    int get_prl() const { return _prl; }
    bool is_concave_corner() const { return _concave_corner; }
    bool has_width() const { return (_width != -1); }
    int get_width() const { return _width; }
    bool has_enclosure() const { return (_enclosure != -1); }
    int get_enclosure() const { return _enclosure; }
    bool has_edge_length() const { return (_edge_length != -1); }
    int get_edge_length() const { return _edge_length; }
    bool has_par_length() const { return (_par_length != -1); }
    int get_par_length() const { return _par_length; }
    int get_par_within() const { return _par_within; }
    int get_edge_enclosure() const { return _edge_enclosure; }
    int get_adjacent_enclosure() const { return _adjacent_enclosure; }
    bool has_extension() const { return (_extension != -1); }
    int get_extension() const { return _extension; }
    bool has_non_eol_convex_corner() const { return (_eol_width != -1); }
    int get_eol_width() const { return _eol_width; }
    bool has_min_length() const { return (_min_length != -1); }
    int get_min_length() const { return _min_length; }
    bool has_above_width() const { return (_width != -1); }
    bool is_mask_overlap() const { return _mask_overlap; }
    bool is_wrong_direction() const { return _wrong_direction; }
    bool has_adjacent_cuts() const { return (_adjacent_cuts != -1); }
    int get_adjacent_cuts() const { return _adjacent_cuts; }
    bool has_exact_aligned_cut() const { return (_exact_aligned_cut != -1); }
    int get_exact_aligned_cut() const { return _exact_aligned_cut; }
    bool has_two_cuts() const { return (_two_cuts != -1); }
    int get_two_cuts() const { return _two_cuts; }
    bool has_two_cuts_spacing() const { return (_two_cuts_spacing != -1); }
    int get_two_cuts_spacing() const { return _two_cuts_spacing; }
    bool is_same_cut() const { return _same_cut; }
    // cutWithin2 is always used as upper bound
    bool has_cut_within1() const { return (_cut_within1 != -1); }
    int get_cut_within() const { return _cut_within2; }
    int get_cut_within1() const { return _cut_within1; }
    int get_cut_within2() const { return _cut_within2; }
    bool is_except_same_power_ground_net() const { return _except_same_power_ground_net; }
    bool has_except_all_within() const { return (_except_all_within != -1); }
    int get_except_all_within() const { return _except_all_within; }
    bool is_above() const { return _above; }
    bool is_below() const { return _below; }
    bool is_to_all() const { return _to_all; }
    bool is_no_prl() const { return _no_prl; }
    bool is_side_parallel_overlap() const { return _side_parallel_overlap; }
    bool is_parallel_overlap() const { return _parallel_overlap; }
    bool is_except_same_net() const { return _except_same_net; }
    bool is_except_same_metal() const { return _except_same_metal; }
    bool is_except_same_metal_overlap() const { return _except_same_metal_overlap; }
    bool is_except_same_via() const { return _except_same_via; }
    bool has_parallel_within() const { return (_parallel_within != -1); }
    int get_parallel_within() const { return _parallel_within; }
    bool is_long_edge_only() const { return _long_edge_only; }
    bool has_same_metal_shared_edge() const { return (_par_within != -1); }
    bool isExceptTwoEdges() const { return _except_two_edges; }
    bool hasExceptSameVia() const { return (_num_cut != -1); }
    bool hasArea() const { return (_cut_area != -1); }
    int getCutArea() const { return _cut_area; }
    // setters
    void set_cut_spacing(int in) { _cut_spacing = in; }
    void set_same_mask(bool in) { _same_mask = in; }
    void set_max_xy(bool in) { _max_xy = in; }
    void set_center_to_center(bool in) { _center_to_center = in; }
    void set_same_net(bool in) { _same_net = in; }
    void set_same_metal(bool in) { _same_metal = in; }
    void set_same_via(bool in) { _same_via = in; }
    void set_second_layer_name(const std::string &in) { _second_layer_name = in; }
    void set_second_layer_num(int in) { _second_layer_num = in; }
    void set_stack(bool in) { _stack = in; }
    void set_orthogonal_spacing(int in) { _orthogonal_spacing = in; }
    void set_cut_class_name(const std::string &in) { _cut_class_name = in; }
    void set_cut_class_id(int in) { _cut_class_id = in; }
    void set_short_edge_only(bool in) { _short_edge_only = in; }
    void set_prl(int in) { _prl = in; }
    void set_concave_corner(bool in) { _concave_corner = in; }
    void set_width(int in) { _width = in; }
    void set_enclosure(int in) { _enclosure = in; }
    void set_edge_length(int in) { _edge_length = in; }
    void set_par_length(int in) { _par_length = in; }
    void set_par_within(int in) { _par_within = in; }
    void set_edge_enclosure(int in) { _edge_enclosure = in; }
    void set_adjacent_enclosure(int in) { _adjacent_enclosure = in; }
    void set_extension(int in) { _extension = in; }
    void set_eol_width(int in) { _eol_width = in; }
    void set_min_length(int in) { _min_length = in; }
    void set_mask_overlap(bool in) { _mask_overlap = in; }
    void set_wrong_direction(bool in) { _wrong_direction = in; }
    void set_adjacent_cuts(int in) { _adjacent_cuts = in; }
    void set_exact_aligned_cut(int in) { _exact_aligned_cut = in; }
    void set_two_cuts(int in) { _two_cuts = in; }
    void set_two_cuts_spacing(int in) { _two_cuts_spacing = in; }
    void set_same_cut(bool in) { _same_cut = in; }
    void set_cut_within(int in) { _cut_within2 = in; }
    void set_cut_within1(int in) { _cut_within1 = in; }
    void set_cut_within2(int in) { _cut_within2 = in; }
    void set_except_same_power_ground_net(bool in) { _except_same_power_ground_net = in; }
    void set_except_all_within(int in) { _except_all_within = in; }
    void set_above(bool in) { _above = in; }
    void set_below(bool in) { _below = in; }
    void set_to_all(bool in) { _to_all = in; }
    void set_no_prl(bool in) { _no_prl = in; }
    void set_side_parallel_overlap(bool in) { _side_parallel_overlap = in; }
    void set_except_same_net(bool in) { _except_same_net = in; }
    void set_except_same_metal(bool in) { _except_same_metal = in; }
    void set_except_same_metal_overlap(bool in) { _except_same_metal_overlap = in; }
    void set_except_same_via(bool in) { _except_same_via = in; }
    void set_parallel_within(int in) { _parallel_within = in; }
    void set_long_edge_only(bool in) { _long_edge_only = in; }
    void set_except_two_edges(bool in) { _except_two_edges = in; }
    void set_num_cut(int in) { _num_cut = in; }
    void set_cut_area(int in) { _cut_area = in; }
    // others

   private:
    int _cut_spacing;
    bool _same_mask;
    bool _max_xy;
    bool _center_to_center;
    bool _same_net;
    bool _same_metal;
    bool _same_via;
    std::string _second_layer_name;
    int _second_layer_num;
    bool _stack;
    int _orthogonal_spacing;
    std::string _cut_class_name;
    int _cut_class_id;
    bool _short_edge_only;
    int _prl;
    bool _concave_corner;
    int _width;
    int _enclosure;
    int _edge_length;
    int _par_length;
    int _par_within;
    int _edge_enclosure;
    int _adjacent_enclosure;
    int _extension;
    int _eol_width;
    int _min_length;
    bool _mask_overlap;
    bool _wrong_direction;
    int _adjacent_cuts;
    int _exact_aligned_cut;
    int _two_cuts;
    int _two_cuts_spacing;
    bool _same_cut;
    int _cut_within1;
    int _cut_within2;
    bool _except_same_power_ground_net;
    int _except_all_within;
    bool _above;
    bool _below;
    bool _to_all;
    bool _no_prl;
    bool _side_parallel_overlap;
    bool _parallel_overlap;
    bool _except_same_net;
    bool _except_same_metal;
    bool _except_same_metal_overlap;
    bool _except_same_via;
    int _parallel_within;
    bool _long_edge_only;
    bool _except_two_edges;
    int _num_cut;
    int _cut_area;
  };
}  // namespace idb

#endif
