#pragma once
#include <string>
namespace python_interface {

bool placerAutoRun(const std::string& config);
bool placerRunFiller(const std::string& config);
bool placerIncrementalFlow(const std::string& config);
bool placerIncrementalLG();
bool placerCheckLegality();
bool placerReport();

bool placerInit(const std::string& config);
bool placerDestroy();
bool placerRunMP();
bool placerRunGP();
bool placerRunLG();
bool placerRunDP();

}  // namespace python_interface