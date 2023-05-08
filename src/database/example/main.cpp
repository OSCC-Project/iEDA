
#include <fstream>
#include <iostream>
#include <memory>

#include "IdbDesign.h"
#include "IdbLayout.h"
#include "builder.h"
// #include "data_service.h"
#include "def_read.h"
#include "def_service.h"
#include "lef_read.h"
#include "lef_service.h"

using namespace idb;

/*
TCL PATH

read_lef ""

read_def "/home/huangzengrong/Project/data/fp_pdn.def"
WRTIE: write_def "/home/huangzengrong/Project/data/FP_Test.def"
*/

int main()
{
  std::cout << "hello db... " << std::endl;

  clock_t start_time = clock();

  IdbBuilder* db_builder = new IdbBuilder();

  // string lef_file[] = {};
  // vector<string> lef_files(lef_file, lef_file + 1);

  // IdbLefService* lef_service = db_builder->buildLef(lef_files);

  // std::cout << "build lef db success... " << std::endl;

  // string defPath = "/home/huangzengrong/gitee/refactor/iEDA/scripts/110nm/100w/result/verilog/asic_top.def";
  // IdbDefService* def_service = db_builder->buildDef(defPath);

  //   IdbBuilder* db_builder_cmp     = new IdbBuilder();
  //   IdbLefService* lef_service_cmp = db_builder_cmp->buildLef(lef_files);
  //   IdbDefService* def_service_cmp = db_builder_cmp->buildDef("/home/huangzengrong/Project/syn.def");
  //   IdbNetList* net_list_org       = def_service->get_design()->get_net_list();
  //   IdbNetList* net_list_cmp       = def_service_cmp->get_design()->get_net_list();

  //   int number = 0;
  //   for (IdbNet* net : net_list_org->get_net_list()) {
  //     if (nullptr == net_list_cmp->find_net(net->get_net_name())) {
  //       std::cout << "net name = " << net->get_net_name() << std::endl;
  //       number++;
  //     }
  //   }
  //   std::cout << "total net number = " << number << std::endl;

  return 0;

/// test Via interface
#if 1
  IdbSpecialNetList* net_list = def_service->get_design()->get_special_net_list();
  IdbLayers* layer_list = def_service->get_layout()->get_layers();
  IdbLayer* layer = layer_list->find_layer("ALPA");
  net_list->initEdge(layer_list);
  vector<IdbCoordinate<int32_t>*> point_list;
  /// 1
  //   point_list.push_back(new IdbCoordinate<int32_t>(4835000, 1447440));
  //   point_list.push_back(new IdbCoordinate<int32_t>(4882000, 1447440));
  //   point_list.push_back(new IdbCoordinate<int32_t>(4882000, 1418000));
  //   point_list.push_back(new IdbCoordinate<int32_t>(4950000, 1418000));
  /// 2
  //   point_list.push_back(new IdbCoordinate<int32_t>(55000, 607440));
  //   point_list.push_back(new IdbCoordinate<int32_t>(164900, 607440));
  /// 3
  point_list.push_back(new IdbCoordinate<int32_t>(659500, 1766700));
  point_list.push_back(new IdbCoordinate<int32_t>(659500, 1945500));
  def_service->get_design()->connectIOPinToPowerStripe(point_list, layer);
  //   /// 4
  point_list.clear();
  point_list.push_back(new IdbCoordinate<int32_t>(4527500, 1766700));
  point_list.push_back(new IdbCoordinate<int32_t>(4527500, 1945500));
  def_service->get_design()->connectIOPinToPowerStripe(point_list, layer);
  //   /// 5
  point_list.clear();
  point_list.push_back(new IdbCoordinate<int32_t>(1582500, 1766700));
  point_list.push_back(new IdbCoordinate<int32_t>(1582500, 1945500));
  def_service->get_design()->connectIOPinToPowerStripe(point_list, layer);
  /// case 1
  //   point_list.push_back(new IdbCoordinate<int32_t>(2692880, 1117480+10));
  //   point_list.push_back(new IdbCoordinate<int32_t>(2792880, 1117480+10));
  //   point_list.push_back(new IdbCoordinate<int32_t>(2792880, 1891500+10));
  //   point_list.push_back(new IdbCoordinate<int32_t>(2804940, 1891500+10));
  /// case 2
  //   point_list.push_back(new IdbCoordinate<int32_t>(2675606, 1998707));
  //   point_list.push_back(new IdbCoordinate<int32_t>(2675606 + 100000, 1998707));
  //   point_list.push_back(new IdbCoordinate<int32_t>(2675606 + 100000, 1892165));
  //   point_list.push_back(new IdbCoordinate<int32_t>(2803686, 1892165));
  /// case 3
  vector<IdbCoordinate<int32_t>*> point_list1;
  //   point_list1.push_back(new IdbCoordinate<int32_t>(2675606, 1998707));
  //   point_list1.push_back(new IdbCoordinate<int32_t>(2803686, 1892165));
  //   def_service->get_design()->connectIOPinToPowerStripe(point_list1, layer);
  //   point_list1.push_back(new IdbCoordinate<int32_t>(2675606, 1998707));
  //   point_list1.push_back(new IdbCoordinate<int32_t>(2675606 + 50000, 1998707));
  //   point_list1.push_back(new IdbCoordinate<int32_t>(2675606 + 50000, 1892165));
  //   point_list1.push_back(new IdbCoordinate<int32_t>(2803686, 1892165));
  /// test
  //   point_list1.push_back(new IdbCoordinate<int32_t>(4835000, 1522860));
  //   point_list1.push_back(new IdbCoordinate<int32_t>(4875000, 1522860));
  //   def_service->get_design()->connectPowerStripe(point_list1, "VDD", "METAL7");
  //   point_list1.clear();
  //   point_list1.push_back(new IdbCoordinate<int32_t>(4835000, 1506460));
  //   point_list1.push_back(new IdbCoordinate<int32_t>(4875000, 1506460));
  //   def_service->get_design()->connectPowerStripe(point_list1, "VDD", "METAL7");
  //   point_list1.push_back(new IdbCoordinate<int32_t>(4835000, 1522860));
  //   point_list1.push_back(new IdbCoordinate<int32_t>(4875000, 1522860));
  //   def_service->get_design()->connectPowerStripe(point_list1, "VDD", "METAL7");

#endif
#if 0
  IdbLayer* layer = def_service->get_layout()->get_layers()->find_layer("ALPA");
  layer->set_power_segment_width(20000);
  layer = def_service->get_layout()->get_layers()->find_layer("METAL8");
  layer->set_power_segment_width(16400);
  layer = def_service->get_layout()->get_layers()->find_layer("METAL7");
  layer->set_power_segment_width(4100);
  layer = def_service->get_layout()->get_layers()->find_layer("METAL6");
  layer->set_power_segment_width(1640);
  layer = def_service->get_layout()->get_layers()->find_layer("METAL5");
  layer->set_power_segment_width(1640);
  layer = def_service->get_layout()->get_layers()->find_layer("METAL4");
  layer->set_power_segment_width(820);
  layer = def_service->get_layout()->get_layers()->find_layer("METAL3");
  layer->set_power_segment_width(240);
  layer = def_service->get_layout()->get_layers()->find_layer("METAL2");
  layer->set_power_segment_width(820);
  layer = def_service->get_layout()->get_layers()->find_layer("METAL1");
  layer->set_power_segment_width(240);

  vector<IdbLayer*>& layer_list = def_service->get_layout()->get_layers()->get_layers();
  for (int i = 0; i < layer_list.size(); i++) {
    IdbLayer* layer_iter = layer_list[i];
    if (layer_iter->is_cut() && layer_list[i - 1]->is_routing()) {
      int width                     = 0;
      int height                    = 0;
      IdbLayerRouting* layer_bottom = dynamic_cast<IdbLayerRouting*>(layer_list[i - 1]);
      IdbLayerRouting* layer_top    = dynamic_cast<IdbLayerRouting*>(layer_list[i + 1]);
      if (layer_bottom->is_horizontal()) {
        width  = layer_top->get_power_segment_width();
        height = layer_bottom->get_power_segment_width();
      } else {
        width  = layer_bottom->get_power_segment_width();
        height = layer_top->get_power_segment_width();
      }
      IdbLayerCut* cut_layer = dynamic_cast<IdbLayerCut*>(layer_iter);
      IdbVia* via_find       = def_service->get_design()->get_via_list()->find_via_generate(cut_layer, width, height);
    }
  }
#endif
  //   def_service->get_design()->createDefaultVias(def_service->get_layout()->get_layers());

  clock_t end_time = clock();

  std::cout << "build def db success... "
            << "use time = " << float(end_time - start_time) / 1000000 << " s" << endl;

  // Write def file
  start_time = clock();
  // string defWritePath = "/home/huangzengrong/Project/data/test.def";
  string defWritePath = "/home/huangzengrong/Project/data/test2.def";
  db_builder->saveDef(defWritePath);
  end_time = clock();
  std::cout << "write def success... "
            << "use time = " << float(end_time - start_time) / 1000000 << " s" << endl;

  // Write layout
  //   start_time = clock();
  //   string layoutWritePath = "/home/huangzengrong/Project/data/dat_files";
  //   IdbDataService* data_service_write = db_builder->buildData(def_service);
  //   db_builder->saveLayout(layoutWritePath);
  //   end_time = clock();
  //   std::cout << "write Layout Data success... "
  //             << "use time = " << float(end_time - start_time) / 1000000 << " s" << endl;

  //   // Read layout
  //   start_time = clock();
  //   string layoutReadPath = "/home/huangzengrong/Project/data/dat_files";
  //   IdbDataService* data_service_read = db_builder->buildData();
  //   db_builder->loadLayout(layoutReadPath);
  //   // end_time = clock();
  //   // std::cout << "Read Layout Data success... " << "use time = " << float(end_time - start_time) / 1000000 << " s" << endl;

  std::cout << "byebye db... " << endl;

  delete db_builder;
  db_builder = nullptr;
  // GperfTools memory profile

  return 0;
}