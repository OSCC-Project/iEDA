#include "IdbLayer.h"
#include "property_parser.h"

namespace idb {
class MastersliceLayerParser : public PropertyBaseParser<IdbLayerMasterslice>
{
 public:
  explicit MastersliceLayerParser(IdbLefService* lefservice) : PropertyBaseParser(lefservice) {}
  ~MastersliceLayerParser() override = default;

  // operator
  bool parse(const std::string& name, const std::string& value, IdbLayerMasterslice* data) override;

 private:
  bool parse_lef58_type(const std::string& value, IdbLayerMasterslice* data);
};
}  // namespace idb
