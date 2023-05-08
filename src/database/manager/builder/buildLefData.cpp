#include "builder.h"

namespace idb {
void IdbBuilder::updateLefData()
{
  updateMacro();
}

void IdbBuilder::updateMacro()
{
  auto layout = get_lef_service()->get_layout();
  if (layout == nullptr) {
    return;
  }

  auto cellmaster_list = layout->get_cell_master_list();

  for (auto cellmaster : cellmaster_list->get_cell_master()) {
    if (cellmaster->get_origin_x() == 0 && cellmaster->get_origin_y() == 0) {
      continue;
    }
    /// pin
    for (auto term : cellmaster->get_term_list()) {
      for (auto port : term->get_port_list()) {
        for (auto shape : port->get_layer_shape()) {
          if (shape != nullptr) {
            for (auto rect : shape->get_rect_list()) {
              rect->set_low_x(rect->get_low_x() + cellmaster->get_origin_x());
              rect->set_low_y(rect->get_low_y() + cellmaster->get_origin_y());
              rect->set_high_x(rect->get_high_x() + cellmaster->get_origin_x());
              rect->set_high_y(rect->get_high_y() + cellmaster->get_origin_y());
            }
          }
        }
      }
    }

    /// obs
    for (auto obs : cellmaster->get_obs_list()) {
      for (auto layer_shape : obs->get_obs_layer_list()) {
        auto shape = layer_shape->get_shape();
        if (shape != nullptr) {
          for (auto rect : shape->get_rect_list()) {
            rect->set_low_x(rect->get_low_x() + cellmaster->get_origin_x());
            rect->set_low_y(rect->get_low_y() + cellmaster->get_origin_y());
            rect->set_high_x(rect->get_high_x() + cellmaster->get_origin_x());
            rect->set_high_y(rect->get_high_y() + cellmaster->get_origin_y());
          }
        }
      }
    }
  }
}

}  // namespace idb