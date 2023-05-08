#pragma once
/**
 * @File Name: tcl_placer.h
 * @Brief :
 * @Author : Yell (12112088@qq.com)
 * @Version : 1.0
 * @Creat Date : 2022-04-15
 *
 */
#include <iostream>
#include <string>

#include "ScriptEngine.hh"
#include "tcl_definition.h"

using ieda::TclCmd;
using ieda::TclOption;
using ieda::TclStringOption;

namespace tcl {

DEFINE_CMD_CLASS(PlacerAutoRun);
DEFINE_CMD_CLASS(PlacerFiller);
DEFINE_CMD_CLASS(PlacerIncrementalFlow);
DEFINE_CMD_CLASS(PlacerIncrementalLG);
DEFINE_CMD_CLASS(PlacerCheckLegality);
DEFINE_CMD_CLASS(PlacerReport);

DEFINE_CMD_CLASS(PlacerInit);
DEFINE_CMD_CLASS(PlacerDestroy);
DEFINE_CMD_CLASS(PlacerRunMP);
DEFINE_CMD_CLASS(PlacerRunGP);
DEFINE_CMD_CLASS(PlacerRunLG);
DEFINE_CMD_CLASS(PlacerRunDP);

}  // namespace tcl
