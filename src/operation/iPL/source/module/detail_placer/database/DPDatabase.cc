#include "DPDatabase.hh"

namespace ipl{

DPDatabase::DPDatabase(): _placer_db(nullptr), _shift_x(0), _shift_y(0), _outside_wl(0), _design(nullptr), _layout(nullptr)
{

}

DPDatabase::~DPDatabase()
{
    delete _design;
    delete _layout;
}

}