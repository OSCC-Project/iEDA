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
#ifndef IDB_DENSITY_CHECK
#define IDB_DENSITY_CHECK
#include <vector>

namespace idb {
  class IdbDccurrentDensityCheck {
   public:
    IdbDccurrentDensityCheck() : _has_one_entry(false), _value(-1) { }
    ~IdbDccurrentDensityCheck() { }
    // getter
    bool get_has_one_entry() { return _has_one_entry; }
    double get_value() { return _value; }
    // setter
    void set_has_one_entry(bool in) { _has_one_entry = in; }
    void set_value(double in) { _value = in; }
    // others

   private:
    bool _has_one_entry;  // oneEntry or TableEntries,oneEntry for any width in
                          // routing layer,in cut layer for any area
    double _value;
  };

  class IdbDccurrentDensityCheckList {
   public:
    IdbDccurrentDensityCheckList() { }
    ~IdbDccurrentDensityCheckList() { }
    void addDccurrentDensityCheck(std::unique_ptr<IdbDccurrentDensityCheck> &check) {
      _dccurrent_density_checks.push_back(std::move(check));
    }

   private:
    std::vector<std::unique_ptr<IdbDccurrentDensityCheck>> _dccurrent_density_checks;
  };

  class IdbDensityCheck {
   public:
    IdbDensityCheck()
        : _max_density(-1),
          _min_density(-1),
          _density_check_length(-1),
          _density_check_width(-1),
          _density_check_step(-1) { }
    ~IdbDensityCheck() { }
    // getter
    double get_max_density() { return _max_density; }
    double get_min_density() { return _min_density; }
    int get_density_check_length() { return _density_check_length; }
    int get_density_check_width() { return _density_check_width; }
    int get_density_check_step() { return _density_check_step; }
    // setter
    void set_max_density(double in) { _max_density = in; }
    void set_min_density(double in) { _min_density = in; }
    void set_density_check_length(int in) { _density_check_length = in; }
    void set_density_check_width(int in) { _density_check_width = in; }
    void set_density_check_step(int in) { _density_check_step = in; }
    // other

   private:
    double _max_density;
    double _min_density;
    int _density_check_length;
    int _density_check_width;
    int _density_check_step;
  };
}  // namespace idb

#endif
