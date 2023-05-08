/**
 * @project		iDB
 * @file		def_service.cpp
 * @author		Yell
 * @date		25/05/2021
 * @version		0.1
* @description


        This is a def db management class to provide db interface, including read and write operation.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "def_service.h"

#include <mutex>

namespace idb {

IdbDefService::IdbDefService(IdbLayout* layout)
{
  _design = std::make_unique<IdbDesign>(layout);
  _layout = layout;
}

IdbDefService::~IdbDefService()
{
}

IdbDefServiceResult IdbDefService::DefFileInit(const char* file_name)
{
  // FILE* file = fopen(file_name, "r");
  // if (file == nullptr)
  // {
  //     std::cout << "Can not open DEF file ( " << file_name  << " )"<< std::endl;

  //     return IdbDefServiceResult::kServiceFailed;
  // }
  // else
  //     std::cout << "Open DEF file success ( " << file_name  << " )"<< std::endl;

  // fclose(file);

  _def_file = file_name;

  return IdbDefServiceResult::kServiceSuccess;
}

IdbDesign* IdbDefService::get_design()
{
  if (!_design) {
    _design = std::make_unique<IdbDesign>(_layout);
  }

  return _design.get();
}

IdbDefServiceResult IdbDefService::DefFileWriteInit(const char* file_name)
{
  FILE* file = fopen(file_name, "w+");
  if (file == nullptr) {
    std::cout << "Can not create file ( " << file_name << " )" << std::endl;

    return IdbDefServiceResult::kServiceFailed;
  } else
    std::cout << "Create file success ( " << file_name << " )" << std::endl;

  fclose(file);

  _def_write_file = file_name;

  return IdbDefServiceResult::kServiceSuccess;
}

IdbDefServiceResult IdbDefService::VerilogFileInit(const char* file_name)
{
  _verilog_file = file_name;

  return IdbDefServiceResult::kServiceSuccess;
}

}  // namespace idb
