#include "masterslicelayer_parser.h"
#include "lef58_property/masterslicelayer_property_parser.h"

namespace idb {

bool MastersliceLayerParser::parse(const std::string& name, const std::string& value, IdbLayerMasterslice* data)
{
    if(name == "LEF58_TYPE"){
        return parse_lef58_type(value, data);
    }
    std::cout << "Unhandled PROPERTY: " << name << " \"" << value << "\"" << std::endl;
    return false;

}

bool MastersliceLayerParser::parse_lef58_type(const std::string& value, IdbLayerMasterslice* data)
{
    std::string type;
    bool ok = masterslicelayer_property::parse_lef58_type(value.begin(), value.end(), type);
    if(not ok){
        return false;
    }
    data->set_lef58_type(std::move(type));
    return true;

}
}  // namespace idb