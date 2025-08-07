// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "tcl_notification.h"
#include "NotificationUtility.h"
#include <iostream>
#include <cstdlib>

namespace tcl {

// public

TclInitNotification::TclInitNotification(const char* cmd_name) : TclCmd(cmd_name)
{
  // This command doesn't require any parameters as it uses environment variables
  // But we can add optional parameters if needed in the future
}

unsigned TclInitNotification::exec()
{
  if (!check()) {
    return 0;
  }

  // Initialize notification utility with environment variable validation
  bool is_notify_init = ieda::NotificationUtility::getInstance().initialize(
    std::getenv("IEDA_ECOS_NOTIFICATION_URL") ? std::getenv("IEDA_ECOS_NOTIFICATION_URL") : "",
    std::getenv("ECOS_TASK_ID") ? std::getenv("ECOS_TASK_ID") : "",
    std::getenv("ECOS_PROJECT_ID") ? std::getenv("ECOS_PROJECT_ID") : "",
    std::getenv("ECOS_TASK_TYPE") ? std::getenv("ECOS_TASK_TYPE") : ""
  );

  // Log the result using the same format as the original implementation
  if (is_notify_init) {
    std::cout << "[INFO] NotificationUtility initialized successfully with : "
              << (std::getenv("IEDA_ECOS_NOTIFICATION_URL") ? std::getenv("IEDA_ECOS_NOTIFICATION_URL") : "null") << " "
              << (std::getenv("ECOS_TASK_ID") ? std::getenv("ECOS_TASK_ID") : "null") << " "
              << (std::getenv("ECOS_PROJECT_ID") ? std::getenv("ECOS_PROJECT_ID") : "null") << " "
              << (std::getenv("ECOS_TASK_TYPE") ? std::getenv("ECOS_TASK_TYPE") : "null") << std::endl;
  } else {
    std::cout << "[WARN] NotificationUtility initialization failed due to missing environment variables" << std::endl;
  }

  return 1;
}

// private

}  // namespace tcl
