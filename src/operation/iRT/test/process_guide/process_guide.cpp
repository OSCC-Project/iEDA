#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Guide.hpp"
#include "Net.hpp"
#include "Util.hpp"

void readGuide(const std::string& guide_file_path, std::vector<Net>& net_list)
{
  std::ifstream* guide_file_stream = getInputFileStream(guide_file_path);

  std::string new_line;
  bool is_new_net = true;

  std::string net_name, layer_name;
  while (std::getline(*guide_file_stream, new_line)) {
    int32_t ll_x, ll_y, ur_x, ur_y;
    if (is_new_net) {
      net_name = new_line;
      is_new_net = false;
    } else if (new_line == "(") {
      std::vector<Guide> guide_list;
      while (std::getline(*guide_file_stream, new_line)) {
        if (new_line == ")") {
          is_new_net = true;

          Net net;
          net.set_net_name(net_name);
          net.set_guide_list(guide_list);
          net_list.push_back(net);

          static int32_t net_num = 0;
          net_num++;
          if (net_num % 10000 == 0) {
            std::cout << "read " << net_num << " nets" << std::endl;
          }
          break;
        }
        std::istringstream str(new_line);
        str >> ll_x >> ll_y >> ur_x >> ur_y >> layer_name;

        Guide guide;
        guide.set_ll_x(ll_x);
        guide.set_ll_y(ll_y);
        guide.set_ur_x(ur_x);
        guide.set_ur_y(ur_y);
        guide.set_layer_name(layer_name);
        guide_list.push_back(guide);
      }
    }
  }
  closeFileStream(guide_file_stream);
  std::cout << "read end" << std::endl;
}

void writeGuide(std::vector<Net>& net_list, const std::string& guide_file_path)
{
  std::ofstream* guide_file_stream = getOutputFileStream(guide_file_path);

  for (size_t i = 0; i < net_list.size(); i++) {
    Net& net = net_list[i];
    (*guide_file_stream) << net.get_net_name() << "\n(\n";
    for (Guide& guide : net.get_guide_list()) {
      (*guide_file_stream) << guide.get_ll_x() << " " << guide.get_ll_y() << " " << guide.get_ur_x() << " " << guide.get_ur_y() << " " << guide.get_layer_name()
                           << "\n";
    }
    (*guide_file_stream) << ")\n";
    if ((i + 1) % 10000 == 0) {
      std::cout << "wrote " << (i + 1) << " nets" << std::endl;
    }
  }
  closeFileStream(guide_file_stream);
  std::cout << "write end" << std::endl;
}

int32_t main()
{
  const std::string& guide_file_path_read = "";
  const std::string& guide_file_path_write = "";

  std::vector<Net> net_list;
  readGuide(guide_file_path_read, net_list);
  writeGuide(net_list, guide_file_path_write);

  std::cout << "////////////////////////////////////////////" << std::endl;
  std::cout << "guide_file_path_read: " << guide_file_path_read << std::endl;
  std::cout << "guide_file_path_write: " << guide_file_path_write << std::endl;
  std::cout << "************ file_diff_lines_num: " << countDifferentLines(guide_file_path_read, guide_file_path_write) << " ************" << std::endl;
  std::cout << "////////////////////////////////////////////" << std::endl;

  return 0;
}