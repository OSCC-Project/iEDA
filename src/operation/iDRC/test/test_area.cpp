#include <string>

#include "DRC.h"
#include "DrcAPI.hpp"
#include "RoutingAreaCheck.h"
#include "idm.h"
#include "ids.hpp"

using namespace idrc;

void setRectValue(ids::DRCRect& rect, int lb_x, int lb_y, int rt_x, int rt_y, std::string name, int so_id)
{
  rect.lb_x = lb_x;
  rect.lb_y = lb_y;
  rect.rt_x = rt_x;
  rect.rt_y = rt_y;
  rect.layer_name = name;
  rect.so_id = so_id;
  rect.type = ids::RectType::kRouting;
}

// test minsize
void test1()
{
  ids::DRCRect rect1;
  setRectValue(rect1, 1000, 1000, 1300, 1140, "M1", 0);
  ids::DRCRect rect2;
  setRectValue(rect2, 1000, 1010, 1399, 1130, "M1", 0);
  // ids::DRCRect rect3;
  // setRectValue(rect3, 1000, 1000, 1320, 1280, "M1", 0);
  ids::DRCEnv env;
  // env.push_back(rect1);
  // env.push_back(rect2);
  // env.push_back(rect3);
  ids::DRCTask task;
  task.push_back(rect1);
  task.push_back(rect2);
  // task.push_back(rect3);
  if (DrcAPIInst.checkDRC(env, task)) {
    std::cout << "true" << std::endl;
  } else {
    std::cout << "false" << std::endl;
  }
}

// test minlength
void test2()
{
  ids::DRCRect rect1;
  setRectValue(rect1, 1000, 1000, 1230, 1230, "M1", 0);
  // ids::DRCRect rect2;
  // setRectValue(rect2, 1000, 1010, 1399, 1130, "M1", 0);
  // ids::DRCRect rect3;
  // setRectValue(rect3, 1000, 1000, 1320, 1280, "M1", 0);
  ids::DRCEnv env;
  // env.push_back(rect1);
  // env.push_back(rect2);
  // env.push_back(rect3);
  ids::DRCTask task;
  task.push_back(rect1);
  // task.push_back(rect2);
  // task.push_back(rect3);
  if (DrcAPIInst.checkDRC(env, task)) {
    std::cout << "true" << std::endl;
  } else {
    std::cout << "false" << std::endl;
  }
}

// test maxlength
void test3()
{
  ids::DRCRect rect1;
  setRectValue(rect1, 1000, 1000, 1600, 1090, "M1", 0);
  // ids::DRCRect rect2;
  // setRectValue(rect2, 1000, 1010, 1399, 1130, "M1", 0);
  // ids::DRCRect rect3;
  // setRectValue(rect3, 1000, 1000, 1320, 1280, "M1", 0);
  ids::DRCEnv env;
  // env.push_back(rect1);
  // env.push_back(rect2);
  // env.push_back(rect3);
  ids::DRCTask task;
  task.push_back(rect1);
  // task.push_back(rect2);
  // task.push_back(rect3);
  if (DrcAPIInst.checkDRC(env, task)) {
    std::cout << "true" << std::endl;
  } else {
    std::cout << "false" << std::endl;
  }
}

int main(int argc, char* argv[])
{
  std::string path = "";
  dmInst->get_config().set_tech_lef_path(path);
  vector<string> path_list;
  path_list.push_back(path);
  dmInst->readLef(path_list);
  std::cout << "read finish" << std::endl;

  DrcAPIInst.initDRC();
  test3();

  return 0;
}