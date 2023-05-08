#include "LGInstance.hh"

namespace ipl{

LGInstance::LGInstance(std::string name): _name(name), _master(nullptr), _orient(Orient::kNone), _state(LGINSTANCE_STATE::kNone), _belong_cluster(nullptr), _belong_region(nullptr), _weight(1.0){}

LGInstance::~LGInstance()
{

}

void LGInstance::updateCoordi(int32_t llx, int32_t lly){
    _shape = Rectangle<int32_t>(llx, lly, llx + _shape.get_width(), lly + _shape.get_height());
}

}
