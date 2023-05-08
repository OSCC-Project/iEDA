#pragma once
#include "property_parser.h"

namespace idb {
class CutLayerParser : public PropertyBaseParser<IdbLayerCut>
{
 public:
  explicit CutLayerParser(IdbLefService* lef_service) : PropertyBaseParser(lef_service) {}
  ~CutLayerParser() override = default;

  /// operator
  bool parse(const std::string& name, const std::string& value, IdbLayerCut* data) override;

 private:
  bool parse_lef58_cutclass(const std::string& value, IdbLayerCut* data);
  bool parse_lef58_enclosure(const std::string& value, IdbLayerCut* data);
  bool parse_lef58_enclosureedge(const std::string& value, IdbLayerCut* data);
  bool parse_lef58_eolenclosure(const std::string& value, IdbLayerCut* data);
  bool parse_lef58_eolspacing(const std::string& value, IdbLayerCut* data);
  bool parse_lef58_spacingtable(const std::string& value, IdbLayerCut* data);
};

}  // namespace idb
