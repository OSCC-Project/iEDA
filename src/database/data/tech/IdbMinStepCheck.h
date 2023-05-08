#ifndef IDB_MIN_STEP_CHECK_H
#define IDB_MIN_STEP_CHECK_H
#include "IdbTechEnum.h"

namespace idb {
  class IdbLef58MinStepCheck {
   public:
    IdbLef58MinStepCheck()
        : _min_step_length(-1),
          _inside_corner(false),
          _outside_corner(false),
          _step(false),
          _max_length(-1),
          _max_edges(-1),
          _min_adjacent_length(-1),
          _convex_corner(false),
          _except_within(-1),
          _concave_corner(false),
          _three_concave_corners(false),
          _width(-1),
          _min_adjacent_length2(-1),
          _min_between_length(-1),
          _except_same_corners(false),
          _eol_width(-1),
          _concave_corners(false) { }

    ~IdbLef58MinStepCheck() { }

    // getter
    int get_min_step_length() const { return _min_step_length; }
    bool has_max_edges() const { return (_max_edges != -1); }
    int get_max_edges() const { return _max_edges; }
    bool has_min_adjacent_length() const { return (_min_adjacent_length != -1); }
    int get_min_adjacent_length() const { return _min_adjacent_length; }
    bool has_eol_width() const { return (_eol_width != -1); }
    int get_eol_width() const { return _eol_width; }
    // setter
    void set_min_step_length(int min_step_length) { _min_step_length = min_step_length; }
    void set_max_edges(int max_edges) { _max_edges = max_edges; }
    void set_min_adjacent_length(int min_adjacent_length) { _min_adjacent_length = min_adjacent_length; }
    void set_eol_width(int eol_width) { _eol_width = eol_width; }
    // other

   private:
    int _min_step_length;
    bool _inside_corner;
    bool _outside_corner;
    bool _step;
    int _max_length;
    int _max_edges;
    int _min_adjacent_length;
    bool _convex_corner;
    int _except_within;
    bool _concave_corner;
    bool _three_concave_corners;
    int _width;
    int _min_adjacent_length2;
    int _min_between_length;
    bool _except_same_corners;
    int _eol_width;
    bool _concave_corners;
  };

  class IdbMinStepCheck {
   public:
    IdbMinStepCheck()
        : _min_step_length(-1),
          _min_step_type(MinstepTypeEnum::kUNKNOWN),
          _max_length(-1),
          _inside_corner(false),
          _outside_corner(false),
          _step(false),
          _max_edges(-1) { }
    ~IdbMinStepCheck() { }
    // getter
    int get_min_step_length() const { return _min_step_length; }
    bool has_max_length() const { return (_max_length != -1); }
    int get_max_length() const { return _max_length; }
    bool has_min_step_type() const { return _min_step_type != MinstepTypeEnum::kUNKNOWN; }
    MinstepTypeEnum get_min_step_type() const { return _min_step_type; }
    bool has_inside_corner() const { return _inside_corner; }
    bool has_outside_corner() const { return _outside_corner; }
    bool has_step() const { return _step; }
    bool has_max_edges() const { return (_max_edges != -1); }
    int get_max_edges() const { return _max_edges; }
    // setter
    void set_min_step_type(MinstepTypeEnum min_step_type) { _min_step_type = min_step_type; }
    void set_inside_corner(bool inside_corner) { _inside_corner = inside_corner; }
    void set_outside_corner(bool outside_corner) { _outside_corner = outside_corner; }
    void set_step(bool step) { _step = step; }
    void set_min_step_length(int min_step_length) { _min_step_length = min_step_length; }
    void set_max_length(int max_length) { _max_length = max_length; }
    void set_max_edges(int max_edges) { _max_edges = max_edges; }

    // others

   private:
    int _min_step_length;
    MinstepTypeEnum _min_step_type;
    int _max_length;
    bool _inside_corner;
    bool _outside_corner;
    bool _step;
    int _max_edges;
  };
}  // namespace idb

#endif
