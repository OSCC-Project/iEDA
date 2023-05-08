#ifndef IDB_ARRAY_SPACING_CHECK
#define IDB_ARRAY_SPACING_CHECK
#include <map>

namespace idb {
  class IdbArraySpacingCheck {
   public:
    IdbArraySpacingCheck() : _cut_spacing(0), _long_array(false), _has_via_width(false), _via_width(0) { }
    ~IdbArraySpacingCheck() { }
    // getter
    int get_cut_spacing() { return _cut_spacing; }
    bool get_long_array() { return _long_array; }
    bool get_has_via_width() { return _has_via_width; }
    int get_via_width() { return _via_width; }

    // setter
    void set_cut_spacing(int spacing) { }
    void set_long_array(bool in) { _long_array = in; }
    void set_has_via_width(bool in) { _has_via_width = in; }
    void set_via_width(int width) { _via_width = width; }

    void add_array_spacing(int arrayCuts, int arraySpacing) { _arraycuts_to_arrayspacing_map[arrayCuts] = arraySpacing; }
    // others
    void print_array_cuts() { }
    int getArrayCuts() { return (_arraycuts_to_arrayspacing_map.begin())->first; }
    int getArraySpacing() { return (_arraycuts_to_arrayspacing_map.begin())->second; }

   private:
    int _cut_spacing;
    bool _long_array;
    bool _has_via_width;
    int _via_width;
    std::map<int, int> _arraycuts_to_arrayspacing_map;
  };

}  // namespace idb

#endif
