#include "LGRow.hh"

namespace ipl{

LGSite::LGSite(std::string name): _name(name), _width(0), _height(0){}

LGSite::~LGSite()
{

}

LGRow::LGRow(std::string row_name, LGSite* site, int32_t site_num): _name(row_name), _site(site), _site_num(site_num){}

LGRow::~LGRow()
{
    delete _site;
    _site = nullptr;
}

}