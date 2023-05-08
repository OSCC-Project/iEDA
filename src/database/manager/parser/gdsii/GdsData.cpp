#include "GdsData.hpp"

namespace idb {

GdsData::GdsData() : _header(), _ts(), _name(), _ref_libs(), _fonts(), _attrtable(), _generations(3), _format(), _unit()
{
  _struct_list.reserve(max_num);
}

GdsData::~GdsData()
{
  clear_struct_list();
  delete _top_struct;
  _top_struct = nullptr;
}

void GdsData::clear_struct_list()
{
  for (auto s : _struct_list) {
    delete s;
  }
  _struct_list.clear();
}

}  // namespace idb