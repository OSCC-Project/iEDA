#pragma once

#include <time.h>

#include <string>
#include <vector>

#include "GdsAref.hpp"
#include "GdsBoundary.hpp"
#include "GdsBox.hpp"
#include "GdsElement.hpp"
#include "GdsNode.hpp"
#include "GdsPath.hpp"
#include "GdsSref.hpp"
#include "GdsText.hpp"
#include "GdsTimestamp.hpp"

namespace idb {

class GdsStruct
{
 public:
  // constructor
  GdsStruct();
  GdsStruct(const GdsStruct&);
  explicit GdsStruct(const std::string&);
  ~GdsStruct();

  // getter
  time_t get_bgn_str() const { return _ts.beg; }
  time_t get_last_mod() const { return _ts.last; }
  std::string get_name() const { return _name; }
  const std::vector<GdsElemBase*>& get_element_list() const { return _element_list; }

  // setter
  void set_name(const std::string& s) { _name = s; }
  void add_element(GdsElemBase*);
  void add_element(const GdsElement&);
  void add_element(const GdsBoundary&);
  void add_element(const GdsPath&);
  void add_element(const GdsSref&);
  void add_element(const GdsAref&);
  void add_element(const GdsText&);
  void add_element(const GdsNode&);
  void add_element(const GdsBox&);

  // operator
  GdsStruct& operator=(const GdsStruct&);

  // function
  void clear();
  void clear_element_list();

 private:
  // members
  GdsTimestamp _ts;   // structure timestamp
  std::string _name;  // structure name
  std::vector<GdsElemBase*> _element_list;
};

}  // namespace idb