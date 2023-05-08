#ifndef IDB_TECH_VIA_RULE_H
#define IDB_TECH_VIA_RULE_H

#include <string>

#include "IdbTechShape.h"

namespace idb {

  class IdbTechViaRule {
   public:
    IdbTechViaRule() = default;
    explicit IdbTechViaRule(const std::string &name)
        : _name(name),
          _is_default(),
          _bottom_enclosure(),
          _top_enclosure(),
          _cut_spacing_x(-1),
          _cut_spacing_y(-1),
          _cut_rect() { }

    ~IdbTechViaRule() { }
    // getter
    const std::string &get_name() const { return _name; }
    IdbTechRoutingLayer *get_bottom_layer() const { return _bottom_layer; }
    IdbTechCutLayer *get_cut_layer() const { return _cut_layer; }
    IdbTechRoutingLayer *get_top_layer() const { return _top_layer; }
    bool get_is_default() const { return _is_default; }
    const Enclosure &get_bottom_enclosure() const { return _bottom_enclosure; }
    const Enclosure &get_top_enclosure() const { return _top_enclosure; }
    int get_cut_spacing_x() const { return _cut_spacing_x; }
    int get_cut_spacing_y() const { return _cut_spacing_y; }
    const IdbTechRect &get_cut_rect() const { return _cut_rect; }
    // setter
    void set_name(std::string name) { _name = name; }
    void set_bottom_layer(IdbTechRoutingLayer *layer) { _bottom_layer = layer; }
    void set_cut_layer(IdbTechCutLayer *layer) { _cut_layer = layer; }
    void set_top_layer(IdbTechRoutingLayer *layer) { _top_layer = layer; }
    void set_is_default(bool b) { _is_default = b; }
    void set_bottom_enclosure(int x, int y) { _bottom_enclosure.setOverhang(x, y); }
    void set_top_enclosure(int x, int y) { _top_enclosure.setOverhang(x, y); }
    void setCutSpacing(int x, int y) {
      _cut_spacing_x = x;
      _cut_spacing_y = y;
    }
    void set_cut_rect(const IdbTechRect &rect) { _cut_rect = rect; }
    // others
    void setCutRect(int llx, int lly, int urx, int ury) { _cut_rect.setRectPoint(llx, lly, urx, ury); }
    int get_cut_layer_id() { return _cut_layer->get_layer_id(); }
    int getCutLowerLeftX() { return _cut_rect.getLowerLeftX(); }
    int getCutLowerLeftY() { return _cut_rect.getLowerLeftY(); }
    int getCutUpperRightX() { return _cut_rect.getUpperRightX(); }
    int getCutUpperRightY() { return _cut_rect.getUpperRightY(); }
    int getBottomEnclosureX() { return _bottom_enclosure.get_overhang1(); }
    int getBottomEnclosureY() { return _bottom_enclosure.get_overhang2(); }
    int getTopEnclosureX() { return _top_enclosure.get_overhang1(); }
    int getTopEnclosureY() { return _top_enclosure.get_overhang2(); }

   private:
    std::string _name;
    IdbTechRoutingLayer *_bottom_layer;
    IdbTechCutLayer *_cut_layer;
    IdbTechRoutingLayer *_top_layer;
    bool _is_default;
    Enclosure _bottom_enclosure;
    Enclosure _top_enclosure;
    int _cut_spacing_x;
    int _cut_spacing_y;
    IdbTechRect _cut_rect;
  };

  class IdbTechViaRuleList {
   public:
    IdbTechViaRuleList() { }
    ~IdbTechViaRuleList() { }

    void addViaRule(std::unique_ptr<IdbTechViaRule> &viaRule) { _tech_via_rules.push_back(std::move(viaRule)); }
    void printViaRule();
    IdbTechViaRule *getTechViaRule(int cutLayerId);

   private:
    std::vector<std::unique_ptr<IdbTechViaRule>> _tech_via_rules;
  };

}  // namespace idb

#endif
