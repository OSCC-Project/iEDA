/**
 * @file GuiAttribute.h
 * @author Yell

 * @brief
 * @version 0.1
 * @date 2021-11-29(V0.1)
 *
 * @copyright Copyright (c) 2021 PCNL
 *
 */
#ifndef GUI_ATTRIBUTE
#define GUI_ATTRIBUTE

#include <QColor>

#define GUI_INSTANCE_GRID_ROW  100
#define GUI_INSTANCE_GRID_COL  200
#define GUI_VIA_GRID_ROW       50
#define GUI_VIA_GRID_COL       50
#define GUI_PDN_GRID_ROW       100
#define GUI_PDN_GRID_COL       100
#define GUI_NET_GRID_PREFER    50
#define GUI_NET_GRID_NONPREFER 100

#define GUI_ITEM_WIRE_MAX     300
#define GUI_ITEM_VIA_MAX      1000
#define GUI_ITEM_INSTANCE_MAX 500
#define GUI_GRID_MAX          500

#define GUI_GCELL_GRID_COLOR QColor(244, 164, 96)

#define attributeInst GuiAttribute::getInstance()

class GuiAttribute {
 public:
  static GuiAttribute* getInstance() {
    if (!_instance) {
      _instance = new GuiAttribute;
    }
    return _instance;
  }

  /// getter
  QColor getLayerColor(std::string layer);
  QColor getLayerColor(int32_t z_oder);

  /// operator
  void resetColor();
  void addLayer(std::string layer);
  void updateColorByLayers(std::vector<std::string> layer_list);

 private:
  static GuiAttribute* _instance;
  std::vector<QColor> _color_list;
  std::vector<std::pair<std::string, QColor>> _default_color_list;  /// default color that match each layer

  GuiAttribute();
  ~GuiAttribute() = default;

  void initDefaultColorList();
  void initColorList();
};

#endif  // GUI_ATTRIBUTE
