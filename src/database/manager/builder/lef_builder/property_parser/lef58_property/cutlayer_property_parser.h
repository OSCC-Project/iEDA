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
 * @file cutlayer_property_parser.h
 * @author pengming
 * @brief parser functions for cutlayer property
 * @version 0.1
 * @date 2022-10-14
 */

#pragma once
#include "cutlayer_property.h"

namespace idb::cutlayer_property {

template <typename Iterator>
bool parse_lef58_cutclass(Iterator beg, Iterator end, std::vector<lef58_cutclass>& vec);

template <typename Iterator>
bool parse_lef58_enclosure(Iterator beg, Iterator end, std::vector<lef58_enclosure>& vec);

template <typename Iterator>
bool parse_lef58_enclosureedge(Iterator beg, Iterator end, std::vector<lef58_enclosureedge>& vec);

template <typename Iterator>
bool parse_lef58_eolenclosure(Iterator beg, Iterator end, lef58_eolenclosure& eol_enclosure);

template <typename Iterator>
bool parse_lef58_eolspacing(Iterator beg, Iterator end, lef58_eolspacing& eolspacing);

template <typename Iterator>
bool parse_lef58_spacingtable(Iterator beg, Iterator end, lef58_spacingtable& spacingtable);
}  // namespace idb::cutlayer_property

namespace idb::cutlayer_property {

template <typename Iterator>
bool parse_lef58_cutclass(Iterator beg, Iterator end, std::vector<lef58_cutclass>& vec)
{
  const static qi::rule<Iterator, std::string(), ascii::space_type> value_string = lexeme[+(char_ - char_(" ;\n"))];
  const static qi::rule<Iterator, std::string(), space_type> class_name_rule = lit("CUTCLASS") >> value_string;
  const static qi::rule<Iterator, cutlayer_property::lef58_cutclass(), space_type> cutclass_rule
      = lit("CUTCLASS") >> value_string >> lit("WIDTH") >> double_ >> -(lit("LENGTH") >> double_) >> -(lit("CUTS") >> int_)
        >> -(lit("ORIENT") >> qi::string("HORIZONTAL") | qi::string("VERTICAL")) >> -(lit(";"));
  const static qi::rule<Iterator, std::vector<cutlayer_property::lef58_cutclass>(), space_type> rule = cutclass_rule % qi::eps;
  bool ok = qi::phrase_parse(beg, end, rule, space, vec);
  if (!ok || beg != end) {
    std::cout << "Parse property \"" << std::string(beg, end) << "\" failed" << std::endl;
    return false;
  }
  return true;
}

template <typename Iterator>
bool parse_lef58_enclosure(Iterator beg, Iterator end, std::vector<lef58_enclosure>& vec)
{
  const static qi::rule<Iterator, std::string(), ascii::space_type> value_string = lexeme[+(char_ - char_(" ;\n"))];
  const static qi::rule<Iterator, std::string(), space_type> class_name_rule = lit("CUTCLASS") >> value_string;
  const static qi::rule<Iterator, std::string(), space_type> direction_rule = qi::string("ABOVE") | qi::string("BELOW");
  const static qi::rule<Iterator, double(), space_type> width_rule = lit("WIDTH") >> double_;
  const static qi::rule<Iterator, double(), space_type> length_rule = lit("LENGTH") >> double_;
  const static qi::rule<Iterator, double(), space_type> redundantcut_rule = lit("REDUNDANTCUT") >> double_;
  static qi::rule<Iterator, lef58_enclosure(), space_type> enclosure_rule
      = lit("ENCLOSURE") >> -class_name_rule >> (-direction_rule) >> -double_ >> -double_ >> -(lit("END") >> double_)
        >> -(lit("SIDE") >> double_) >> -(lit("EXCEPTEXTRACUT") >> double_ >> -(qi::string("PRL") | qi::string("NOSHAREDEDGE")))
        >> -width_rule >> -length_rule >> -redundantcut_rule >> lit(";");

  bool ok = qi::phrase_parse(beg, end, (enclosure_rule % qi::eps), space, vec);
  if (!ok || beg != end) {
    std::cout << "Parse property \"" << std::string(beg, end) << "\" failed" << std::endl;
    return false;
  }
  return true;
}
template <typename Iterator>
bool parse_lef58_enclosureedge(Iterator beg, Iterator end, std::vector<lef58_enclosureedge>& vec)
{
  const static qi::rule<Iterator, std::string(), ascii::space_type> value_string = lexeme[+(char_ - char_(" ;\n"))];
  const static qi::rule<Iterator, std::string(), space_type> class_name_rule = lit("CUTCLASS") >> value_string;
  const static qi::rule<Iterator, std::string(), space_type> direction_rule = qi::string("ABOVE") | qi::string("BELOW");
  const static qi::rule<Iterator, double(), space_type> overhang_rule = double_;
  const static qi::rule<Iterator, double(), space_type> width_rule = lit("WIDTH") >> double_;
  const static qi::rule<Iterator, double(), space_type> parallel_rule = lit("PARALLEL") >> double_;
  const static qi::rule<Iterator, double(), space_type> within_rule = lit("WITHIN") >> double_;

  const static qi::rule<Iterator, cutlayer_property::lef58_enclosureedge_width(), space_type> width_struct_rule
      = width_rule >> parallel_rule >> within_rule >> -qi::string("EXCEPTEXTRACUT") >> -double_ >> -qi::string("EXCEPTTWOEDGES")
        >> -double_;
  const static qi::rule<Iterator, cutlayer_property::lef58_enclosureedge_convexcorners(), space_type> convexconers_rule
      = lit("CONVEXCORNERS") >> double_ >> double_ >> lit("PARALLEL") >> double_ >> lit("LENGTH") >> double_;

  const static qi::rule<Iterator, cutlayer_property::lef58_enclosureedge, space_type> rule
      = lit("ENCLOSUREEDGE") >> -class_name_rule >> -direction_rule >> overhang_rule >> (width_struct_rule | convexconers_rule) >> lit(";");

  bool ok = qi::phrase_parse(beg, end, rule % qi::eps, space, vec);
  if (!ok || beg != end) {
    std::cout << "Parse property \n\"" << std::string(beg, end) << "\"\n failed" << std::endl;
    return false;
  }
  return true;
}

template <typename Iterator>
bool parse_lef58_eolenclosure(Iterator beg, Iterator end, lef58_eolenclosure& eol_enclosure)
{
  const static qi::rule<Iterator, std::string(), ascii::space_type> value_string = lexeme[+(char_ - char_(" ;\n"))];
  const static qi::rule<Iterator, lef58_eolenclosure_edgeoverhang(), space_type> edge_overhang_rule
      = (qi::string("LONGEDGEONLY") | qi::string("SHORTEDGEONLY")) >> double_;
  const static qi::rule<Iterator, lef58_eolenclosure_overhang(), space_type> overhang_rule
      = double_ >> -double_ >> -(lit("PARALLELEDGE") >> double_) >> -(lit("EXTENSION") >> double_) >> -double_
        >> -(lit("MINLENGTH") >> double_) >> -qi::string("ALLSIDES");
  const static qi::rule<Iterator, lef58_eolenclosure(), space_type> eol_enclosure_rule
      = lit("EOLENCLOSURE") >> double_ >> -(lit("MINEOLWIDTH") >> double_) >> -(qi::string("HORIZONTAL") | qi::string("VERTICAL"))
        >> -(qi::string("EQUALRECTWIDTH")) >> -(qi::lit("CUTCLASS") >> value_string) >> -(qi::string("ABOVE") | qi::string("BELOW"))
        >> (edge_overhang_rule | overhang_rule) >> lit(";");

  bool ok = qi::phrase_parse(beg, end, eol_enclosure_rule, space, eol_enclosure);
  if (!ok || beg != end) {
    std::cout << "Parse property \n\"" << std::string(beg, end) << "\"\nfailed" << std::endl;
    return false;
  }
  return true;
}

template <typename Iterator>
bool parse_lef58_eolspacing(Iterator beg, Iterator end, lef58_eolspacing& eolspacing)
{
  const static qi::rule<Iterator, std::string(), ascii::space_type> value_string = lexeme[+(char_ - char_(" ;\n"))];
  const static qi::rule<Iterator, cutlayer_property::lef58_eolspacing_toclass, space_type> toclass_rule
      = lit("TO") >> value_string >> double_ >> double_;
  const static qi::rule<Iterator, cutlayer_property::lef58_eolspacing, space_type> eolspacing_rule
      = lit("EOLSPACING") >> double_ >> double_ >> -(lit("CUTCLASS") >> value_string) >> -(toclass_rule % qi::eps) >> lit("ENDWIDTH")
        >> double_ >> lit("PRL") >> double_ >> lit("ENCLOSURE") >> double_ >> double_ >> lit("EXTENSION") >> double_ >> double_
        >> lit("SPANLENGTH") >> double_ >> lit(";");

  bool ok = qi::phrase_parse(beg, end, eolspacing_rule, space, eolspacing);
  if (!ok || beg != end) {
    std::cout << "Parse property \n\"" << std::string(beg, end) << "\"\nfailed" << std::endl;
    return false;
  }
  return true;
}

template <typename Iterator>
bool parse_lef58_spacingtable(Iterator beg, Iterator end, lef58_spacingtable& spacingtable)
{
  const static qi::rule<Iterator, std::string(), ascii::space_type> value_string = lexeme[+(char_ - char_(" ;\n"))];
  const static qi::rule<Iterator, cutlayer_property::lef58_spacingtable_layer(), space_type> layer_rule = lit("LAYER") >> value_string;
  const static qi::rule<Iterator, lef58_spacingtable_prl(), space_type> prl_rule
      = lit("PRL") >> double_ >> -(qi::string("HORIZONTAL") | qi::string("VERTICAL")) >> -qi::string("MAXXY");
  const static qi::rule<Iterator, lef58_spacingtable_classname()> classname_rule
      = (+qi::alnum) >> -(*lit(' ') >> *lit('\t') >> (qi::string("END") | qi::string("SIDE")));
  const static qi::rule<Iterator, cutlayer_property::lef58_spacingtable_cutspacing(), space_type> cutspacing_rule
      = (double_ | lit("-")) >> (double_ | lit("-"));
  const static qi::rule<Iterator, lef58_spacingtable_cutspacings(), space_type> cutspacings_rule
      = classname_rule >> (cutspacing_rule % qi::eps);
  const static qi::rule<Iterator, lef58_spacingtable_cutclass(), space_type> cutclass_rule
      = lit("CUTCLASS") >> lexeme[(classname_rule) % +char_("\t ")] >> (cutspacings_rule % qi::eps);
  const static qi::rule<Iterator, lef58_spacingtable(), space_type> spacingtable_rule
      = lit("SPACINGTABLE") >> -layer_rule >> -prl_rule >> cutclass_rule >> lit(";");

  bool ok = qi::phrase_parse(beg, end, spacingtable_rule, space, spacingtable);
  if (!ok || beg != end) {
    std::cout << "Parse property \n\"" << std::string(beg, end) << "\"\nfailed" << std::endl;
    return false;
  }
  return true;
}
}  // namespace idb::cutlayer_property