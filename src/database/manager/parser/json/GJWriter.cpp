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
#include "GJWriter.hpp"
#include <unordered_map>
#include <vector>
#include <string.h>
// #include "/home/shuyuz/iEDA/src/platform/data_manager/idm.h"
#include <fstream>
#define sref_expension 0
namespace idb {

JsonTextWriter::JsonTextWriter()
{
}

JsonTextWriter::JsonTextWriter(JsonData* data, const std::string txt)
{
  _data = data;
  _stream = new std::ofstream(txt, std::ios::out);
  if (_stream != nullptr && _stream->is_open()) {
    _stream->close();
    _stream = nullptr;
  }
}

JsonTextWriter::~JsonTextWriter()
{
  _data = nullptr;
}

bool JsonTextWriter::init(std::string txt, JsonData* data)
{
  if (txt.empty() || data == nullptr) {
    return false;
  }

  if (_stream != nullptr) {
    delete _stream;
  }

  _stream = new std::ofstream(txt, std::ios::out);
  if (_stream != nullptr && !_stream->is_open()) {
    _stream->close();
    _stream = nullptr;
    return false;
  }

  if (data != _data) {
    delete _data;
    _data = data;
  }

  return true;
}

bool JsonTextWriter::close()
{
  _stream->close();

  return true;
}

void JsonTextWriter::flush()
{
  _stream->flush();

  // _data->clear_struct_list();
}

bool JsonTextWriter::begin(int i)
{
  // <stream format>
  (*_stream) <<"{"<< std::endl;
  write_header(i+1);
  write_bgnlib(i+1);
  write_libname(i+1);
  write_reflibs(i+1);
  write_fonts(i+1);
  write_attrtable(i+1);
  write_generations(i+1);
  write_format(i+1);
  write_units(i+1);

  flush();

  return true;
}

bool JsonTextWriter::finish(int i,std::vector<std::string> discard)
{
  // (*_stream) << ",\n";
  std::string indent(4*(i+1), ' ');
  // (*_stream) <<indent<< "\"data\" : [" << std::endl;
  writeStruct(i+1);

  /// @brief write top struct
  writeTopStruct(i+1);

  write_endlib(i);


  flush();

  close();

  return true;
}

// // @param data json data
// // @param txt the name of JSON-TXT file
// bool JsonTextWriter::write()
// {
//   write_struct();

//   return true;
// }

void JsonTextWriter::write_header(int i)const
{
  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "\"header\" :" << " \""<<std::to_string(_data->get_header()) << "\","<<std::endl;
}

void JsonTextWriter::write_bgnlib(int i)const
{
  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "\"lastModify\" :" << " \""<< fmt_time(_data->get_last_mod()) <<"\","<< std::endl;
  // (*_stream) <<indent<< "\"lastModify\" :" << " \""<< fmt_time(_data->get_bgn_lib()) << " " << fmt_time(_data->get_last_mod()) <<"\","<< std::endl;
}

std::string JsonTextWriter::fmt_time(time_t t) const
{
  const char* time_fmt = "%m/%d/%Y %H:%M:%S";
  char buf[64] = {0};
  struct tm bd_time;  // broken-down time
  localtime_r(&t, &bd_time);
  strftime(buf, sizeof(buf), time_fmt, &bd_time);
  return buf;
}

void JsonTextWriter::write_libname(int i)const
{
  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "\"libname\" :" << " \""<<_data->get_lib_name()<< "\","<<std::endl;
}

void JsonTextWriter::write_reflibs(int i)const {
  std::string indent(4*i, ' ');
  auto libs = _data->get_ref_libs();
  if (libs.size() == 0)
    return;

  (*_stream) << indent << "\"reflibs\" :" << "["<<std::endl;
  for (size_t i = 0; i < libs.size(); i++) {
    (*_stream) << "\"" << libs[i] << "\"";
    if (i != libs.size() - 1) {
      (*_stream) << ",";
    }
    (*_stream) << "\n";
  }
  (*_stream) << indent << "]," << std::endl;
}

void JsonTextWriter::write_fonts(int i)const
{
  auto fonts = _data->get_fonts();
  if (fonts.size() == 0)
    return;

  std::string indent(4*i, ' ');
  
  (*_stream) << indent << "\"fonts\" :" << "["<<std::endl;
  for (size_t i = 0; i < fonts.size(); i++) {
    (*_stream) << "\"" << fonts[i] << "\"";
    if (i != fonts.size() - 1) {
      (*_stream) << ",";
    }
    (*_stream) << "\n";
  }
  (*_stream) << indent << "]," << std::endl;
}

void JsonTextWriter::write_attrtable(int i)const
{
  auto attrtable = _data->get_attrtable();
  if (attrtable.size() == 0)
    return;

  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "\"attrtable\" :" << " \""<<attrtable<< "\","<<std::endl;
}

// This record contains a value to indicate
// the number of copies of deleted or back-up structures to retain.
// This numbermust be at least 2 and not more than 99.
// If the GENERATION record is omitted, a value of 3 is assumed.
void JsonTextWriter::write_generations(int i)const
{
  auto g = _data->get_generations();
  if (g == 3)
    return;

  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "\"generations\" :" << " \""<<g<< "\","<<std::endl;
}

void JsonTextWriter::write_format(int i)const
{
  auto fmt = _data->get_format();
  if (fmt.type == JsonFormatType::kJSON_Archive)
    return;
  std::string indent(4*i, ' ');

  (*_stream) <<indent<< "\"format\" : \"" << (int) fmt.type << "\",\n";

  if (!fmt.is_filtered())
    return;

  (*_stream) <<indent<< "\"mask\" : \""
             << fmt.mask << "\",\n";
}

void JsonTextWriter::write_units(int i)const
{
  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "\"units\" :" << " \""<<_data->get_unit().dbu_in_user() << " " << _data->get_unit().dbu_in_meter() << "\","<<std::endl;
  // (*_stream) << "UNITS " << _data->get_unit().dbu_in_user() << " " << _data->get_unit().dbu_in_meter() << std::endl;
}

void JsonTextWriter::write_diearea(int i)const{

  std::string indent(4*i, ' ');
  std::string indent2(4*(i+1), ' ');
  (*_stream) <<indent<< "\"diearea\" : {" << std::endl;

  auto coords = _data->get_top_struct()->get_element_list()[0]->get_xy().get_coords();
  (*_stream) <<indent2<< "\"path\" : [";
  for (size_t i = 0; i < coords.size(); i++) {
    (*_stream) << "[" << coords[i].x << "," << coords[i].y << "]";
    if (i != coords.size() - 1) 
    (*_stream) << ",";
    if (i == coords.size() - 1) 
    (*_stream) << "]"<<std::endl;
  }
  // (*_stream) <<indent2<< "\"height\" : "<< _data->get_top_struct()->
                              // get_element_list()[0]->get_xy().
                              // get_coords()[2].y<< std::endl;
  (*_stream) <<indent<< "},"<< std::endl;
}

size_t temp;
size_t temp_expension;
void JsonTextWriter::writeTopStruct(int i)
{
  JsonStruct* str = _data->get_top_struct();
  // write_bgnstr(str,i);
  // write_strname(str,i+1);
  // write_strclass( str); // Unsupported in KLayout and not used in JSON file
    auto element_list = str->get_element_list();
    temp=0;
for (size_t j = 0; j < element_list.size(); j++) {
  // if (j != 0) {
    // (*_stream) << ",\n";
  // }
  write_struct_element(element_list[j],i,temp!=0);
}
  write_endstr(i,1);

  /// @brief clear
  flush();
}

std::unordered_map<std::string, int> myHash_temp;

extern std::map<std::string, int> activated_layers;
std::vector<std::string> layer_temp;
void JsonTextWriter::writeStruct(int i)
{
  static int t = 0,diearea=0;
  std::string indent(4*i, ' ');
  std::string indent2(4*(i+1), ' ');
  std::string indent3(4*(i+2), ' ');
  size_t structListSize = _data->get_struct_list().size();
    // std::cout<<"struct nums ; "<<structListSize<<std::endl;
  
  for (size_t k = 0; t < structListSize;) {
    JsonStruct* str = _data->get_struct_list()[t++];
    myHash_temp[str->get_name()]=t-1;


  if(t==1){
    (*_stream) <<indent<< "\"version\" : \""<<dynamic_cast<JsonText*>(str->get_element_list()[0])->str<<"\"," << std::endl;
    flush();
    return;
  }
  else if(t==2){
    (*_stream) <<indent<< "\"design name\" : \""<<dynamic_cast<JsonText*>(str->get_element_list()[0])->str<<"\"," << std::endl;
    flush();
    return;
  }
  else if(t==3){
    // (*_stream) <<indent<< "\"pins\" : \"\"," << std::endl;
    flush();
    return;
  }
  else if(t==4){
    (*_stream) <<indent<< "\"data\" : [" << std::endl;
    // return;
  }
  // else if(t==4){
    // (*_stream) <<indent<< "\"fills\" : \"\"," << std::endl;
    // return;
  // }
  // else if(t==5){
  //   (*_stream) <<indent<< "\"special_net\" : \"\"," << std::endl;
  //   // (*_stream) <<indent<< "\"special_net\" : \""<<dynamic_cast<JsonText*>(str->get_element_list()[0])->str<<"\"," << std::endl;
  //   t++;
  //   return;
  // }
  // else if(t==6){
  //   // (*_stream) <<indent<< "\"net\" : \""<<dynamic_cast<JsonText*>(str->get_element_list()[0])->str<<"\"," << std::endl;
  //   (*_stream) <<indent<< "\"net\" : \"\"," << std::endl;
  //   t++;
  //   return;
  // }

  

    // <structure>
    write_bgnstr(str, i+1);
    write_strname(str, i + 2);

    auto element_list = str->get_element_list();

    temp=0;
    for (size_t j = 0; j < element_list.size(); j++) {
      // if(j!=0 )//&& activated_layers.find(layer_temp[element_list[j]->layer])!=activated_layers.end())
        // (*_stream) << ",\n";
      write_struct_element(element_list[j], i+2,temp!=0);

      // if (j != element_list.size() - 1) {
      // }
    }
    // if(element_list.size()>1)
      (*_stream) <<std::endl<<indent3<<"]";
    write_endstr(i+1,0);

    if (k != structListSize - 1) {
      (*_stream) << ",\n";
    }
  }
  /// @brief clear
  flush();
}


void JsonTextWriter::write_sref_expension(int i,int num)
{
  static int t = 0;
  std::string indent(4*i, ' ');
  std::string indent2(4*(i+1), ' ');
  std::string indent3(4*(i+2), ' ');
  size_t structListSize = _data->get_struct_list().size();

    // std::cout<<"struct nums ; "<<structListSize<<std::endl;
  
    temp_expension=0;
  for (size_t k = 0; t < structListSize;) {
    JsonStruct* str = _data->get_struct_list()[num];
  

    // <structure>
    write_bgnstr(str, i+1);
    write_strname(str, i + 2);

    auto element_list = str->get_element_list();
    for (size_t j = 0; j < element_list.size(); j++) {
      // if(j!=0)
        // (*_stream) << ",\n";
      write_struct_element(element_list[j], i+2,temp!=0);
      // if (j != element_list.size() - 1) {
      // }
    }
    temp_expension=1;
    if(element_list.size()>1)
      (*_stream) <<std::endl<<indent3<<"]";
    write_endstr(i+1,0);

    if (k != structListSize - 1) {
      (*_stream) << ",\n";
    }
  }
  /// @brief clear
  // flush();
}

void JsonTextWriter::write_endlib(int i)const
{
  std::string indent(4*i, ' ');
  (*_stream) <<std::endl;
  (*_stream) <<indent<< "}" << std::endl;
  // //check myHash_temp
  // int j=0;
  // for (const auto& pair : myHash_temp) {
  //       if (pair.second == 1) {  // 检查值是否为 1
  //           (*_stream) << j <<std::endl;
  //           (*_stream) << "Key: " << pair.first <<std::endl;
  //       }
  //     j++;
  //   }

  _data->clear_struct_list();
}

void JsonTextWriter::write_layerinfo(std::vector<std::string> layer_name,int i,int num){
  layer_temp=layer_name;
  int t=0;
  std::string indent(4*i, ' ');
  std::string indent2(4*(i+1), ' ');
  (*_stream)<<indent<<"\"layerInfo\" :["<<std::endl;
  for(int j=0 ; j<num ; j++){
    if(activated_layers.find("null")!=activated_layers.end()||activated_layers.find(layer_name[j])==activated_layers.end()){
      t++;
      (*_stream)<<indent2<<"{"<<std::endl;
      (*_stream)<<indent2<<"\"id\" : "<<j<<","<<std::endl;
      (*_stream)<<indent2<<"\"layername\" : \""<<layer_name[j]<<"\""<<std::endl<<indent2<<"}";
      if(activated_layers.find("null")!=activated_layers.end()&&j<num-1) (*_stream)<<"\,";
      else if(t<num-activated_layers.size()) (*_stream)<<"\,";
      (*_stream)<<std::endl;  
    }
  }
  (*_stream)<<indent<<"],"<<std::endl;
}

void JsonTextWriter::write_bgnstr(JsonStruct* str,int i) const
{
  if (!str)
    return;
  std::string indent(4*i, ' ');
  std::string indent2(4*(i+1), ' ');
  std::string indent3(4*(i+2), ' ');
  (*_stream) <<indent<< "{" << std::endl;
  // if(str->get_element_list().size()>1){
  (*_stream) <<indent2<< "\"type\" : \"group\"," <<std::endl;
  // }
  // else 
  //   (*_stream) <<indent2<< "\"type\" : \"not a group\"," <<std::endl;
  (*_stream) <<indent2<< "\"struct name\" :"<<"\""<<str->get_name()<<"\"";
  // myHash_temp[str->get_name()]++;
  // if(str->get_element_list().size()>1){
  (*_stream) <<"," <<std::endl<<indent2<< "\"children\":[" <<std::endl;
  // }
  // (*_stream) <<indent3<< "\"lastmodify\" : \""<< fmt_time(str->get_last_mod()) <<"\""<<std::endl;
  // (*_stream) << "\"lastmodify\" : \""<<fmt_time(str->get_bgn_str()) << " " << fmt_time(str->get_last_mod()) << std::endl;
}

void JsonTextWriter::write_strname(JsonStruct* str,int i) const
{
  if (!str)
    return;

  // std::string indent(4*i, ' ');
  // (*_stream) <<indent<< "\"strname\" :" << " \""<<str->get_name()<< "\","<<std::endl;
  // (*_stream) << "STRNAME " << str->get_name() << std::endl;
}

// Not used
// https://www.boolean.klaasholwerda.nl/interface/bnf/jsonformat.html#rec_strclass
void JsonTextWriter::write_strclass(JsonStruct* str,int i) const
{
  if (!str)
    return;
  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "STRCLASS " << std::endl;
}

void JsonTextWriter::write_endstr(int i,int t)const
{
  std::string indent(4*i, ' ');
  if(t==1){
    (*_stream) <<std::endl<<indent<<"]";
    return;
  }
  if(t==2){
    std::string indent2(4*(i+1), ' ');
    (*_stream) <<std::endl<<indent2<<"}"<<std::endl;
  }
  else
  (*_stream) <<std::endl<<indent<<"}";
  // (*_stream) <<indent<< "]";
}

void JsonTextWriter::write_struct_element(JsonElemBase* e,int i,bool out) const
{
  if (!e)
    return;

  switch (e->get_elem_type()) {
    case JsonElemType::kElement:
      write_element(dynamic_cast<JsonElement*>(e),i+1,out);
      break;
    case JsonElemType::kBoundary:
      write_boundary(dynamic_cast<JsonBoundary*>(e),i+1,out);
      break;
    case JsonElemType::kPath:
      write_path(dynamic_cast<JsonPath*>(e),i+1,out);
      break;
    // block sref printing for json
    // case JsonElemType::kSref:
    //   write_sref(dynamic_cast<JsonSref*>(e),i+1,out);
    //   break;
    case JsonElemType::kAref:
      write_aref(dynamic_cast<JsonAref*>(e),i+1,out);
      break;
    case JsonElemType::kText:
      write_text(dynamic_cast<JsonText*>(e),i+1,out);
      break;
    case JsonElemType::kNode:
      write_node(dynamic_cast<JsonNode*>(e),i+1,out);
      break;
    case JsonElemType::kBox:
      write_box(dynamic_cast<JsonBox*>(e),i+1,out);
      break;

    default:
      break;
  }
}

void JsonTextWriter::write_element(JsonElement* e,int i,bool out) const
{
  if (!e)
    return;
  if(out) (*_stream) << ",\n";
  temp++;
  write_property(e,i+1);
  write_endel(i);
}

void JsonTextWriter::write_boundary(JsonBoundary* e,int i,bool out) const
{
  if (!e)
    return;
  if(activated_layers.find("null")!=activated_layers.end()||activated_layers.find(layer_temp[e->layer])==activated_layers.end()){
    if(out) (*_stream) << ",\n";
    temp++;
    std::string indent(4*i, ' ');
    std::string indent2(4*(i+1), ' ');
    (*_stream) <<indent<< "{\n";
    (*_stream) <<indent2<< "\"boundary\" : \"boundary\",\n";
    (*_stream) <<indent2<< "\"datatype\" : \"" << e->data_type <<"\",\n";
    write_elflags(e,i+1);
    write_plex(e,i+1);
    write_layer(e->layer,i+1);
    write_property(e,i+1);
    write_xy(e,i+1);
    write_endel(i);
  }
}

void JsonTextWriter::write_path(JsonPath* e,int i,bool out) const
{
  if (!e)
    return;
  if(activated_layers.find(layer_temp[e->layer])!=activated_layers.end()){
    if(out) (*_stream) << ",\n";
    temp++;
    std::string indent(4*i, ' ');
    std::string indent2(4*(i+1), ' ');
    (*_stream) <<indent<< "{\n";
    (*_stream) <<indent2<< "\"type\" : \"path\",\n";
    (*_stream) <<indent2<< "\"datatype\" : " << e->data_type <<",\n"
              <<indent2<< "\"pathtype\" : " << (int) e->path_type <<",\n"
              <<indent2<< "\"width\" : " << e->width <<",\n";
    write_elflags(e,i+1);
    write_plex(e,i+1);
    write_layer(e->layer,i+1);
    write_property(e,i+1);
    write_xy(e,i+1);
    write_endel(i);
  }
}
// std::unordered_map<std::string, int> myHash;

void JsonTextWriter::write_sref(JsonSref* e,int i,bool out) const
{
  std::string indent(4*i, ' ');
  std::string indent2(4*(i+1), ' ');
  std::string indent3(4*(i+2), ' ');
  std::string indent4(4*(i+3), ' ');
  if (!e)
    return;
  if(out||temp_expension==1) (*_stream)<<",\n";
  temp++;
  // (*_stream) <<indent<<_data->get_struct_list().size()<< "\n";

  // test
  // auto it = myHash_temp.find(e->sname);

  //   if (it != myHash_temp.end()) {
  //       // 元素找到
  //       (*_stream)<< "元素 '" << e->sname << "' 存在于哈希表中，值为: " << it->second << std::endl;
  //   } else {
  //       // 元素未找到
  //       (*_stream) << "元素 '" << e->sname << "' 不存在于哈希表中" << std::endl;
  //   }

  (*_stream) <<indent<< "{\n";
  (*_stream) <<indent2<< "\"type\" : \"sref\",\n";
  (*_stream) <<indent2<< "\"sname\" : \"" << e->sname <<"\",\n";
  write_elflags(e,i+1);
  write_plex(e,i+1);
  write_strans(e->strans,i+1);
  write_property(e,i+1);
  write_xy(e,i+1);

/*expension sref*/
#ifdef sref_expension
  (*_stream) <<indent2<< "\"data\" : " <<"\n";

  static int t = 0;
 
  JsonStruct* str = _data->get_struct_list()[myHash_temp[e->sname]];
  write_bgnstr(str, i+2);
  write_strname(str, i + 3);
  auto element_list = str->get_element_list();

  // <structure>
  temp=0;
  temp_expension=0;
  for (size_t j = 0; j < element_list.size(); j++) {
    // if(j!=0)
      // (*_stream) << ",\n";
    write_struct_element(element_list[j], i+3,temp!=0);

    // if (j != element_list.size() - 1) {
    // }
  }
  temp_expension=1;
  if(element_list.size()>1)
    (*_stream) <<std::endl<<indent4<<"]";
  write_endstr(i+1,2);
#endif


  write_endel(i);
}
void JsonTextWriter::write_aref(JsonAref* e,int i,bool out) const
{
  if (!e)
    return;
  std::string indent(4*i, ' ');
  std::string indent2(4*(i+1), ' ');
  (*_stream) <<indent<< "{\n";
  (*_stream) <<indent2<< "\"type\" : \"aref\",\n";
  (*_stream) <<indent2<< "\"sname\" : \"" << e->sname <<"\",\n";

  write_elflags(e,i+1);
  write_plex(e,i+1);
  write_strans(e->strans,i+1);
  (*_stream) <<indent2<< "\"colrow\" : \"" << e->col << "," << e->row << "\"\n";
  write_property(e,i+1);
  write_xy(e,i+1);
  write_endel(i);
}

void JsonTextWriter::write_text(JsonText* e,int i,bool out) const
{
  if (!e)
    return;
  if(activated_layers.find("null")!=activated_layers.end()||activated_layers.find(layer_temp[e->layer])==activated_layers.end()){
    if(out) (*_stream) << ",\n";
    temp++;
    std::string indent(4*i, ' ');
    std::string indent2(4*(i+1), ' ');
    (*_stream) <<indent<<"{"<<std::endl;
    (*_stream) <<indent2<< "\"type\" : \"text\",\n";
    write_elflags(e,i+1);
    write_plex(e,i+1);
    write_layer(e->layer,i+1);
    (*_stream) <<indent2<< "\"texttype\" : \"" << e->text_type <<"\",\n"
               <<indent2<< "\"presentation\" : \"" << int(e->presentation) << "\",\n"
               <<indent2<< "\"pathtype\" : \"" << int(e->path_type) << "\",\n"
               <<indent2<< "\"width\" : \"" << e->width << "\",\n";
    write_strans(e->strans,i+1);
    (*_stream) <<indent2<< "\"string\" : \"" << e->str << "\",\n";
    write_property(e,i+1);
    write_xy(e,i+1);
    write_endel(i);
  }
}

void JsonTextWriter::write_node(JsonNode* e,int i,bool out) const
{
  if (!e)
    return;
  if(activated_layers.find("null")!=activated_layers.end()||activated_layers.find(layer_temp[e->layer])==activated_layers.end()){
    if(out) (*_stream) << ",\n";
    temp++;
    std::string indent(4*i, ' ');
    std::string indent2(4*(i+1), ' ');
    (*_stream) <<indent<<"{"<<std::endl;
    (*_stream) <<indent2<< "\"type\" : \""<< "node\","<<"\n";
    (*_stream) <<indent2<< "\"id\" : \""<<e->node_type<<"\","<<"\n";
    write_elflags(e,i+1);
    write_plex(e,i+1);
    write_layer(e->layer,i+1);
    write_property(e,i+1);
    write_xy(e,i+1);
    write_endel(i);
  }
}

void JsonTextWriter::write_box(JsonBox* e,int i,bool out) const
{
  if (!e)
    return;
  if(activated_layers.find("null")!=activated_layers.end()||activated_layers.find(layer_temp[e->layer])==activated_layers.end()){  
    if(out) (*_stream) << ",\n";
    temp++;
    std::string indent(4*i, ' ');
    std::string indent2(4*(i+1), ' ');
    (*_stream) <<indent<<"{"<<std::endl;
    (*_stream) <<indent2<< "\"type\" : \""<< "box\","<<"\n";
    (*_stream) <<indent2<< "\"id\" : \""<<e->box_type<<"\","<<"\n";
    write_elflags(e,i+1);
    write_plex(e,i+1);
    write_layer(e->layer,i+1);
    // (*_stream) << "BOXTYPE " << e->box_type << "\n";
    write_property(e,i+1);
    write_xy(e,i+1);
    write_endel(i);
  }
}

// The document recommends setting attributes from 1 to 127,
// In practice, KLayout 0.26.2 reads JSON-TXT files smoothly when setting attributes from 0 to 65535.
// Since the "PROPATTR" is a two-byte signed integer,
// those property whose PROPATTR < 0 will not be written.
void JsonTextWriter::write_property(JsonElemBase* e,int i) const
{
  if (!e)
    return;

  for (const auto& [attr, value] : e->get_property_map()) {
    if (attr < 0)
      continue;
  std::string indent(4*i, ' ');
    (*_stream) <<indent<< "\"propattr\" : " << attr <<","<<"\n"
               <<indent<< "\"propvalue\" : " << value <<""<<"\n";
  }
}

// quantity check in terms of
// https://www.boolean.klaasholwerda.nl/interface/bnf/jsonformat.html#rec_xy
void JsonTextWriter::write_xy(JsonElemBase* e,int i) const
{
  if (!e)
    return;

  int min = 0;
  int max = 0;
  switch (e->get_elem_type()) {
    case JsonElemType::kElement:
      min = 0;
      max = 0;
      return;
    case JsonElemType::kBoundary:
      min = 4;
      max = 200;
      break;
    case JsonElemType::kPath:
      min = 2;
      max = 200;
      break;
    case JsonElemType::kSref:
      min = 1;
      max = 1;
      break;
    case JsonElemType::kAref:
      min = 3;
      max = 1;
      break;
    case JsonElemType::kText:
      min = 1;
      max = 1;
      break;
    case JsonElemType::kNode:
      min = 1;
      max = 50;
      break;
    case JsonElemType::kBox:
      min = 5;
      max = 5;
      break;

    default:
      return;
  }

  int num = e->get_xy().get_nums();
  assert(num);

  if (min > num)
    std::cout << "Warn: coordinate total is less than the expected"
              << ", JsonElemType =" << (int) e->get_elem_type() << std::endl;

  if (max < num)
    std::cout << "Warn: coordinate total is more than the expected"
              << ", JsonElemType =" << (int) e->get_elem_type() << std::endl;
  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "\"path\" : [";
  auto coords = e->get_xy().get_coords();
  for (size_t i = 0; i < coords.size(); i++) {
    (*_stream) << "[" << coords[i].x << "," << coords[i].y << "]";
    if (i != coords.size() - 1) {
    (*_stream) << ",";
  }
}
  

//delete if don't expension sref
#ifdef sref_expension
  if((int)e->get_elem_type()!=3)
    (*_stream) << "]"<<std::endl;
  else 
    (*_stream) << "],"<<std::endl;
#else
  (*_stream) << "]"<<std::endl;
#endif
}

void JsonTextWriter::write_endel(int i)const
{
  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "}" ;
}

void JsonTextWriter::write_strans(const JsonStrans& strans,int i) const
{
  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "\"strans\" : " << strans.bit_flag << ",\n"
             <<indent<< "\"mag\" : " << strans.mag << ",\n"
             <<indent<< "\"angle\" : " << strans.angle << ",\n";
}

void JsonTextWriter::write_elflags(const JsonElemBase* e,int i) const
{
  if (!e)
    return;

  auto value = e->get_flags().get_value();
  if (value == 0)
    return;
  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "\"eflags\" : " << value <<","<< std::endl;
}

void JsonTextWriter::write_plex(const JsonElemBase* e,int i) const
{
  if (!e)
    return;

  auto value = e->get_plex();
  if (value == 0)
    return;
  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "\"plex\" : " << value <<","<< std::endl;
}

// The document allows setting layer-value in the range of 0 to 255.
// In practice, KLayout supports layer-value from 0 to 65535.
// So layer < 0 will be assert. 
void JsonTextWriter::write_layer(JsonLayer layer,int i) const
{
  assert(layer >= 0);
  std::string indent(4*i, ' ');
  (*_stream) <<indent<< "\"layer\" : " << layer <<","<< std::endl;
}

}  // namespace idb