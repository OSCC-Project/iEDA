#include "DPPin.hh"

namespace ipl{

DPPin::DPPin(std::string name): _name(name), _x_coordi(INT32_MIN), _y_coordi(INT32_MIN), _offset_x(INT32_MIN), _offset_y(INT32_MIN), _internal_id(INT32_MIN), _net(nullptr), _instance(nullptr)
{

}

DPPin::~DPPin()
{

}

}