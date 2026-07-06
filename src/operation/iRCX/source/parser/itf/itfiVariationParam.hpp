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

#include <string>
#include <vector>
namespace itf
{

// variation parameter term 
class itfiVPT {
 public:
  // operator
  bool operator==(const itfiVPT&) const;

  // function
  void clear();

  // members
  std::string layer;
  std::string type;
  float coeff;
};

class itfiVariationParam{
 public:
  // operator
  bool operator==(const itfiVariationParam&) const;
  
  // function
  void clear();

  void add_term(const itfiVPT&);
  void add_term(const char*, const char*, float);

  // members
  std::string param_name;
  std::vector<itfiVPT> term_list;
};

///////////// inline ////////////////

inline bool
itfiVPT::operator==(const itfiVPT& rhs) const
{
  if (this == &rhs) return true;

  return layer == rhs.layer
    && type == rhs.type
    && coeff == rhs.coeff
  ;
}

inline void
itfiVPT::clear()
{
  layer.clear();
  type.clear();
  coeff = 0;
}

inline bool
itfiVariationParam::operator==(const itfiVariationParam& rhs) const
{
  if (this == &rhs) return true;

  return param_name == rhs.param_name
    && term_list == rhs.term_list
  ;
}

inline void
itfiVariationParam::clear()
{
  param_name.clear();
  term_list.clear();
}

inline void
itfiVariationParam::add_term(const itfiVPT& t)
{
  term_list.push_back(t);
}

inline void
itfiVariationParam::add_term(
  const char* layer, const char* type, float coeff
) {
  itfiVPT t;
  t.layer = layer ? layer : "";
  t.type = type ? type : "";
  t.coeff = coeff;
  add_term(t);
}

} // namespace itf
