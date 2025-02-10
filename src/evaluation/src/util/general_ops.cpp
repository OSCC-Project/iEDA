/*
 * @FilePath: general_ops.cpp
 * @Author: Yihang Qiu (qiuyihang23@mails.ucas.ac.cn)
 * @Date: 2024-08-24 15:37:27
 * @Description:
 */

#include "general_ops.h"

#include <sys/stat.h>
#include <unistd.h>

#include <climits>
#include <cstdlib>
#include <cstring>

#include "idm.h"

namespace ieval {

std::string getAbsoluteFilePath(std::string filename)
{
  char current_path[PATH_MAX];
  if (getcwd(current_path, sizeof(current_path)) != nullptr) {
    std::string working_dir(current_path);

    if (filename[0] == '/') {
      return filename;
    }

    return working_dir + "/" + filename;
  }

  char resolved_path[PATH_MAX];
  if (realpath(filename.c_str(), resolved_path) != nullptr) {
    return std::string(resolved_path);
  }
  return filename;
}

std::string createDirPath(std::string dir_path)
{
  const std::string base_path = dmInst->get_config().get_output_path();
  std::string full_path = base_path + dir_path;

  struct stat info;
  if (stat(full_path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR)) {
    return full_path;
  }

  if (mkdir(full_path.c_str(), 0777) == 0) {
    return full_path;
  }

  return "";
}

std::string getDefaultOutputPath()
{
  std::string base_path = dmInst->get_config().get_output_path();
  return base_path;
}

}  // namespace ieval