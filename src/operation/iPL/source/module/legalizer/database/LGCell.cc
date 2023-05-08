#include "LGCell.hh"

namespace ipl{

LGCell::LGCell(std::string name): _name(name),_type(LGCELL_TYPE::kNone),_width(0),_height(0){}

LGCell::~LGCell()
{
    
}

}