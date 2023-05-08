#include "property_parser.h"

namespace idb {
class RoutingLayerParser : public PropertyBaseParser<IdbLayerRouting>
{
 public:
  explicit RoutingLayerParser(IdbLefService* lefservice) : PropertyBaseParser(lefservice) {}
  ~RoutingLayerParser() override = default;

  // operator
  bool parse(const std::string& name, const std::string& value, IdbLayerRouting* data) override;

 private:
  bool parse_lef58_area(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_conerfillspacing(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_minimuncut(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_minstep(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_spacing(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_spacingtable(const std::string& value, IdbLayerRouting* data);

  bool parse_lef58_spacingtable_jogtojog(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_spacing_eol(const std::string& value, IdbLayerRouting* data);
  bool parse_lef58_spacing_notchlength(const std::string& value, IdbLayerRouting* data);
};
}  // namespace idb
