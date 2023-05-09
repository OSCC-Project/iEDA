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
/**
 * @file masterslicelayer_property_parser.h
 * @author pengming
 * @brief
 * @version 0.1
 * @date 2022-10-22
 */

#pragma once

#include <boost/spirit/include/qi.hpp>
#include <string>

namespace idb::masterslicelayer_property {
namespace qi = boost::spirit::qi;

template <typename Iterator>
bool parse_lef58_type(Iterator beg, Iterator end, std::string& type)
{
  const static qi::rule<Iterator, std::string(), qi::ascii::space_type> value_string = qi::lexeme[+(qi::char_ - qi::char_(" ;\n"))];
  const static qi::rule<Iterator, std::string(), qi::ascii::space_type> type_rule = qi::lit("TYPE") >> value_string >> qi::lit(";");
  bool ok = qi::phrase_parse(beg, end, type_rule, qi::ascii::space, type);
  if (not ok || beg != end) {
    std::cout << "Parse \"" << std::string(beg, end) << "\" failed" << std::endl;
    return false;
  }
  return true;
}
}  // namespace idb::masterslicelayer_property
