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
#ifndef IDB_SPACING_CHECK_H
#define IDB_SPACING_CHECK_H
#include <vector>

namespace idb {
  class IdbSpacingCheck {
   public:
    IdbSpacingCheck() : _min_spacing(0) { }
    explicit IdbSpacingCheck(int minSpacing) : _min_spacing(minSpacing) { }
    virtual ~IdbSpacingCheck() { }
    // getter
    int get_min_spacing() { return _min_spacing; }
    // setter
    void set_min_spacing(int min_spacing) { _min_spacing = min_spacing; }
    // other

   private:
    int _min_spacing;
  };
  class IdbSpacingCheckList {
   public:
    IdbSpacingCheckList() { }
    ~IdbSpacingCheckList() { }

    void addSpacingCheck(std::unique_ptr<IdbSpacingCheck> &check) { _spacing_checks.push_back(std::move(check)); }

   private:
    std::vector<std::unique_ptr<IdbSpacingCheck>> _spacing_checks;
  };

  class IdbSpacingEolCheck : public IdbSpacingCheck {
   public:
    IdbSpacingEolCheck()
        : IdbSpacingCheck(),
          _eol_width(-1),
          _eol_within(-1),
          _par_space(-1),
          _par_within(-1),
          _is_parallel_type(false),
          _is_two_edges(false) { }
    ~IdbSpacingEolCheck() { }

    // getter
    int get_eol_width() const { return _eol_width; }
    int get_eol_within() const { return _eol_within; }
    int get_par_space() const { return _par_space; }
    int get_par_within() const { return _par_within; }
    bool get_is_two_edges() const { return _is_two_edges; }
    bool get_is_parallel_type() const { return _is_parallel_type; }
    // setter
    void set_eol_width(int eol_width) { _eol_width = eol_width; }
    void set_eol_within(int eol_within) { _eol_within = eol_within; }
    void set_par_space(int par_space) { _par_space = par_space; }
    void set_par_within(int par_within) { _par_within = par_within; }
    void set_is_parallel_type(bool type) { _is_parallel_type = type; }
    void set_is_two_edges(int is_two_edges) { _is_two_edges = is_two_edges; }

    // other

   private:
    int _eol_width;
    int _eol_within;
    int _par_space;
    int _par_within;
    bool _is_parallel_type;
    bool _is_two_edges;
  };
  class IdbSpacingEolCheckList {
   public:
    IdbSpacingEolCheckList() { }
    ~IdbSpacingEolCheckList() { }
    void addSpacingEolCheck(std::unique_ptr<IdbSpacingEolCheck> &check) { _spacing_eol_checks.push_back(std::move(check)); }

   private:
    std::vector<std::unique_ptr<IdbSpacingEolCheck>> _spacing_eol_checks;
  };

  class IdbSpacingSamenetCheck : public IdbSpacingCheck {
   public:
    IdbSpacingSamenetCheck() : IdbSpacingCheck(), _pg_only(false) { }
    IdbSpacingSamenetCheck(int minSpacing, bool pgOnly) : IdbSpacingCheck(minSpacing), _pg_only(pgOnly) { }
    ~IdbSpacingSamenetCheck() { }
    // getter
    bool get_pg_only() { return _pg_only; }
    // setter
    void set_pg_only(bool pgOnly) { _pg_only = pgOnly; }
    // other

   private:
    bool _pg_only;  // the same net is power or ground net
  };

  class IdbSpacingSamenetCheckList {
   public:
    IdbSpacingSamenetCheckList() { }
    ~IdbSpacingSamenetCheckList() { }

    void addSpacingSamenetCheck(std::unique_ptr<IdbSpacingSamenetCheck> &check) {
      _spacing_samenet_checks.push_back(std::move(check));
    }

   private:
    std::vector<std::unique_ptr<IdbSpacingSamenetCheck>> _spacing_samenet_checks;
  };

  class IdbSpacingRangeCheck : public IdbSpacingCheck {
   public:
    IdbSpacingRangeCheck() : _min_width(0), _max_width(0) { }
    ~IdbSpacingRangeCheck() { }
    // getter
    int get_min_width() { return _min_width; }
    int get_max_width() { return _max_width; }
    // setter
    void set_min_width(int minWidth) { _min_width = minWidth; }
    void set_max_width(int maxWidth) { _max_width = maxWidth; }
    // others

   private:
    int _min_width;
    int _max_width;
  };

  class IdbSpacingRangeCheckList {
   public:
    IdbSpacingRangeCheckList() { }
    ~IdbSpacingRangeCheckList() { }
    void addSpacingRangeCheck(std::unique_ptr<IdbSpacingRangeCheck> &check) {
      _spacing_range_checks.push_back(std::move(check));
    }
    std::vector<IdbSpacingRangeCheck *> getIdbSpacingRangeChecks() {
      std::vector<IdbSpacingRangeCheck *> vec;
      for (auto &check : _spacing_range_checks) {
        vec.push_back(check.get());
      }
      return vec;
    }

   private:
    std::vector<std::unique_ptr<IdbSpacingRangeCheck>> _spacing_range_checks;
  };

}  // namespace idb

#endif
