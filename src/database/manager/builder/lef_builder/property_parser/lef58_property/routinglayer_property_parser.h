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
#include <boost/spirit/home/support/common_terminals.hpp>
#include "routinglayer_property.h"

namespace idb::routinglayer_property {
  template<typename Iterator>
  bool parse_lef58_area(Iterator beg, Iterator end, std::vector<lef58_area>& areas);

  template<typename Iterator>
  bool parse_lef58_conerfillspacing(Iterator beg, Iterator end, lef58_cornerfillspacing& spacing);

  template<typename Iterator>
  bool parse_lef58_minimumcut(Iterator beg, Iterator end, std::vector<lef58_minimumcut>& cuts);

  template<typename Iterator>
  bool parse_lef58_minstep(Iterator beg, Iterator end, std::vector<lef58_minstep>& minsteps);

}  // namespace idb::routinglayer_property

namespace idb::routinglayer_property {
  template<typename Iterator>
  bool parse_lef58_area(Iterator beg, Iterator end, std::vector<lef58_area>& areas) {
    const static qi::rule<Iterator, std::string(), ascii::space_type> value_string = lexeme[+(char_ - char_(" ;\n"))];
    const static qi::rule<Iterator, double_pair(), space_type> dpair_rule          = double_ >> double_;
    const static qi::rule<Iterator, lef58_area_exceptedgelength(), space_type> exceptedgelength_rule
        = lit("EXCEPTEDGELENGTH") >> double_ >> -double_;
    const static qi::rule<Iterator, std::vector<double_pair>(), space_type> except_min_size_rule =
        (lit("EXCEPTMINSIZE") >> (double_ >> double_) % qi::eps);
    const static qi::rule<Iterator, double_pair(), space_type> except_step_rule = lit("EXCEPTSTEP") >> dpair_rule;
    const static qi::rule<Iterator, lef58_area, space_type> area_rule = lit("AREA") >> double_ >> -(lit("MASK") >> int_)
                                                                        >> -(lit("EXCEPTMINWIDTH") >> double_) >> -exceptedgelength_rule >> -except_min_size_rule >> - except_step_rule
                                                                        >> -(lit("RECTWIDTH") >> double_) >> -qi::string("EXCEPTRECTANGLE")
                                                                        >> -(lit("LAYER") >> value_string) >> -(lit("OVERLAP") >> int_);

    bool ok = qi::phrase_parse(beg, end, area_rule % lit(";") >> -lit(";"), space, areas);

    if (!ok || beg != end) {
      std::cout << "Parse \"" << std::string(beg, end) << "\" failed" << std::endl;
      return false;
    }
    return true;
  }

  template<typename Iterator>
  bool parse_lef58_conerfillspacing(Iterator beg, Iterator end, lef58_cornerfillspacing& spacing) {
    const static qi::rule<Iterator, lef58_cornerfillspacing(), space_type> corner_spacing_rule =
        lit("CORNERFILLSPACING") >> double_ >> lit("EDGELENGTH") >> double_ >> double_ >> lit("ADJACENTEOL") >> double_ >>
        lit(";");

    bool ok = qi::phrase_parse(beg, end, corner_spacing_rule, space, spacing);
    if (!ok || beg != end) {
      std::cout << "Parse \"" << std::string(beg, end) << "\" failed" << std::endl;
      return false;
    }
    return true;
  }

  template<typename Iterator>
  bool parse_lef58_minimumcut(Iterator beg, Iterator end, std::vector<lef58_minimumcut>& cuts){
    const static qi::rule<Iterator, std::string(), ascii::space_type> value_string = lexeme[+(char_ - char_(" ;\n"))];
    qi::rule<Iterator, name_cuts(), space_type> namecuts_rule = 
        lit("CUTCLASS") >> value_string >> int_ ;

    qi::rule<Iterator,  lef58_minimumcut(), space_type> minimumcut_rule = 
    lit("MINIMUMCUT") >> -int_  >> -(namecuts_rule % qi::eps)
    >> lit("WIDTH") >> double_ >> -(lit("WITHIN")>> double_)
    >> -(qi::string("FROMABOVE") | qi::string("FROMBELOW"))
    >> -(lit("LENGTH")>> double_) >> -(lit("WITHIN") >> double_)
    >> -(lit("AREA") >> double_) >> -(lit("WITHIN") >> double_)
    >> -qi::string("SAMEMETALOVERLAP")
    >> -qi::string("FULLYENCLOSED");

    bool ok = qi::phrase_parse(beg, end, minimumcut_rule % lit(";") >> lit(";"), space, cuts);
    if (!ok || beg != end) {
      std::cout << "Parse \"" << std::string(beg, end) << "\" failed" << std::endl;
      return false;
    }
    return true;
  }

  template<typename Iterator>
  bool parse_lef58_minstep(Iterator beg, Iterator end, std::vector<lef58_minstep>& minsteps){
    const static qi::rule<Iterator, std::string(), ascii::space_type> value_string = lexeme[+(char_ - char_(" ;\n"))];
    const static qi::rule<Iterator, lef58_minstep(), space_type> minstep_rule = 
        lit("MINSTEP") >> double_ 
        >> -(qi::string("INSIDECORNER") | qi::string("OUTSIDECORNER") | qi::string("STEP"))
        >> -(lit("LENGTHSUM") >> double_ )
        >> -(lit("MAXEDGES") >> int_ )
        >> -lit("MINADJACENTLENGTH") >> -double_ >> -double_
        >> -qi::string("CONVEXCORNER") >> -(lit("EXCEPTWITHIN")>> double_)
        >> -qi::string("CONCAVECORNER")
        >> -qi::string("THREECONCAVECORNERS") >> -(lit("CENTERWIDTH") >> double_)
        >> -(lit("MINBETWEENLENGTH") >> double_) >> - qi::string("EXCEPTSAMECORNERS")
        >> -(lit("NOADJACENTEOL") >> double_)
        >> -(lit("EXCEPTADJACENTLENGTH") >> double_)
        >> -(lit("MINADJACENTLENGTH")>> double_)
        >> -qi::string("CONCAVECORNERS")
        >> -(lit("NOBETWEENEOL") >> double_) 
        ;

    bool ok = qi::phrase_parse(beg, end, minstep_rule % lit(";") >> lit(";"), space, minsteps);
    if (!ok || beg != end) {
      std::cout << "Parse \"" << std::string(beg, end) << "\" failed" << std::endl;
      return false;
    }
    return true;
  }

  template<typename Iterator>
  bool parse_lef58_spacing_notchlength(Iterator beg, Iterator end, lef58_spacing_notchlength& spacing_notchlen){
    const static qi::rule<Iterator, lef58_spacing_notchlength(), space_type> spacing_rule = 
      lit("SPACING") >> double_ >> lit("NOTCHLENGTH") >> double_
      >> -lit("EXCEPTWITHIN") >> -double_ >> -double_
      >> -(lit("WITHIN") >> double_ ) >> -(lit("SPANLENGTH") >> double_ )
      >> -(qi::string("WIDTH") | qi::string("CONCAVEENDS")) >> -double_
      >> -(lit("NOTCHWIDTH") >> double_ ) >> lit(";");
    bool ok = qi::phrase_parse(beg, end, spacing_rule, space, spacing_notchlen);
    if (!ok || beg != end) {
      std::cout << "Parse \"" << std::string(beg, end) << "\" failed" << std::endl;
      return false;
    }
    return true;
  }

  template<typename Iterator>
  bool parse_lef58_spacing_eol(Iterator beg, Iterator end, std::vector<lef58_spacing_eol>& spacings){
    const static qi::rule<Iterator, std::string(), ascii::space_type> value_string = lexeme[+(char_ - char_(" ;\n"))];
    const static qi::rule<Iterator, double_pair(), space_type> except_exact_width_rule = 
      lit("EXCEPTEXACTWIDTH") >> double_ >> double_;
    const static qi::rule<Iterator, lef58_spacing_eol_withcut(), space_type> withcut_rule = 
      lit("WITHCUT") >> -(lit("CUTCLASS") >> value_string)
      >> -qi::string("ABOVE")
      >> double_
      >> -(lit("ENCLOSUREEND") >> double_)
      >> -(lit("WITHIN") >> double_ );
    const static qi::rule<Iterator, lef58_spacing_eol_endprlspacing(), space_type> endprlspacing_rule = 
      lit("ENDPRLSPACING") >> double_ >> lit("PRL") >> double_;

    const static qi::rule<Iterator, lef58_spacing_eol_endtoend(), space_type> endtoend_rule = 
      lit("ENDTOEND") >> double_ >> -double_ >> -double_
      >>-lit("EXTENSION") >> -double_ >> -double_
      >>-(lit("OTHERENDWIDTH") >> double_) ;

    const static qi::rule<Iterator, lef58_spacing_eol_paralleledge(), space_type> parallel_edge_rule = 
      lit("PARALLELEDGE") >> -qi::string("SUBTRACTEOLWIDTH") >> double_
      >> lit("WITHIN") >> double_ >> -(lit("PRL") >> double_ )
      >> -(lit("MINLENGTH") >> double_ ) >> -qi::string("TWOEDGES")
      >> -qi::string("SAMEMETAL") >> -qi::string("NONEOLCORNERONLY")
      >> -qi::string("PARALLELSAMEMASK");

    const static qi::rule<Iterator, lef58_spacing_eol_enclosecut(), space_type> enclosecut_rule =
      lit("ENCLOSECUT") >> -(qi::string("BELOW") | qi::string("ABOVE")) >> double_
      >> lit("CUTSPACING") >> double_ >> -qi::string("ALLCUTS") ;

    const static qi::rule<Iterator, lef58_spacing_eol_toconcavecorner(), space_type> toconcavecorner_rule = 
      lit("TOCONCAVECORNER") >> -(lit("MINLENGTH") >> double_ )
      >> -lit("MINADJACENTLENGTH") >> -double_ >> -double_ ;

    static qi::rule<Iterator, lef58_spacing_eol(), space_type> spacing_eol_rule = 
      lit("SPACING") >> double_ >> lit("ENDOFLINE") >> double_
      >> -qi::string("EXACTWIDTH")
      >> -(lit("WRONGDIRSPACING") >> double_)
      >> -(lit("OPPOSITEWIDTH") >> double_)
      >> -lit("WITHIN") >> -double_ >> -double_
      >> -qi::string("SAMEMASK")
      >> -except_exact_width_rule
      >> -(lit("FILLCONCAVECORNER") >> double_)
      >> -withcut_rule
      >> -endprlspacing_rule
      >> -endtoend_rule
      >> -(lit("MAXLENGTH") >> double_ ) >> -(lit("MINLENGTH") >> double_) >> -qi::string("TWOSIDES") 
      >> -qi::string("EQUALRECTWIDTH")
      >> -parallel_edge_rule
      >> -enclosecut_rule
      >> -toconcavecorner_rule
      >> -(lit("TONOTCHLENGTH") >> double_ )
    ;
    bool ok = qi::phrase_parse(beg, end, spacing_eol_rule % lit(";") >> lit(";"), space, spacings);
    if (!ok || beg != end) {
      std::cout << "Parse \"" << std::string(beg, end) << "\" failed" << std::endl;
      return false;
    }
    return true;
  }

  template<typename Iterator>
  bool parse_lef58_spacingtable_jogtojog(Iterator beg, Iterator end, lef58_spacingtable_jogtojog& spacingtable){
    const static qi::rule<Iterator, lef58_spacingtable_jogtojog_width(), space_type> width_rule = 
      lit("WIDTH") >> double_ >> lit("PARALLEL") >> double_
      >> lit("WITHIN") >> double_ 
      >> -lit("EXCEPTWITHIN") >> -double_ >> -double_
      >> lit("LONGJOGSPACING") >> double_
      >> -lit("SHORTJOGSPACING") >> -double_ ;
    const static qi::rule<Iterator, lef58_spacingtable_jogtojog(), space_type> spacingtable_jog_rule = 
      lit("SPACINGTABLE") >> lit("JOGTOJOGSPACING") >> double_
      >> lit("JOGWIDTH") >> double_ >> lit("SHORTJOGSPACING") >> double_
      >> (width_rule % qi::eps)
      >> lit(";")
      ;
    bool ok = qi::phrase_parse(beg, end, spacingtable_jog_rule, space, spacingtable);
    if (!ok || beg != end) {
      std::cout << "Parse \"" << std::string(beg, end) << "\" failed" << std::endl;
      return false;
    }
    return true;
  }
}  // namespace idb::routinglayer_property