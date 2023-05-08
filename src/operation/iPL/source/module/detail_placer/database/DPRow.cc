#include "DPRow.hh"

namespace ipl{

DPSite::DPSite(std::string name): _name(name), _width(INT32_MIN),_height(INT32_MIN){}

DPSite::~DPSite(){

}

DPRow::DPRow(std::string row_name, DPSite* site, int32_t site_num): _name(row_name), _site(site), _site_num(site_num)
{

}

DPRow::~DPRow()
{

}

}