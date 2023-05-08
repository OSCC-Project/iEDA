#include "file_placement.h"
#include "guiConfig.h"
#include "idbfastsetup.h"
#include "omp.h"

void IdbSpeedUpSetup::updateInstanceInFastMode(std::vector<iplf::FileInstance>& file_inst_list) {
  _gui_design->get_instance_list()->clearData();

  int row_height = _layout->get_rows()->get_row_height();

  int ignore_num = 0;
  for (auto& file_instance : file_inst_list) {
    QRectF rect = _transform.db_to_guidb_rect(file_instance.x, file_instance.y, file_instance.x + row_height,
                                              file_instance.y + row_height);
    GuiSpeedupInstance* item =
        dynamic_cast<GuiSpeedupInstance*>(_gui_design->get_instance_list()->findCurrentItem(rect.center()));
    if (item == nullptr) {
      continue;
    }

    if (item->get_rect_list().size() > 1000) {
      ignore_num++;
      continue;
    }
    item->add_rect(rect);
    item->set_type(GuiSpeedupItemType::kInstStandarCell);
  }

  _gui_design->get_instance_list()->update();

  std::cout << "GUI Instance Num = " << _gui_design->get_instance_list()->get_item_list().size()
            << " File instance number = " << file_inst_list.size() << " Ignore number = " << ignore_num << std::endl;
}
