#include "onnx/onnx_pb.h"
#include "onnx/proto_utils.h"
#include <fstream>
#include <iostream>
#include "base64-util.h"
#include "file-util.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h" // for stringify JSON
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/error/en.h"
#include "rapidjson/filereadstream.h"

using namespace BASE_NAMESPACE;

int main(void) {
  onnx::ModelProto model;
  std::ifstream in("../decoder1.onnx", std::ios_base::binary);
  model.ParseFromIstream(&in);
  in.close();
  std::cout << "input size:" << model.graph().input().size() << "\n";
  std::cout << "input initializer:" << model.graph().initializer().size() << "\n";
  auto &initializer = model.graph().initializer();
  rapidjson::Document d;  // 创建一个空的Document
    d.SetObject();   // 将Document设置为Object类型
    rapidjson::Document::AllocatorType &allocator = d.GetAllocator();
  for (int i = 0; i < initializer.size(); i++) {
    std::string name = initializer[i].name();
    std::cout << "initializer name:" << initializer[i].name() << " data_type:" << initializer[i].data_type() << " "
              << initializer[i].DataType_Name(initializer[i].data_type()) << "\n";
    auto &dims = initializer[i].dims();
    for (int j = 0; j < dims.size(); j++)
      std::cout << dims[j] << ",";
    std::cout << "\n";
    std::cout<<initializer[i].has_raw_data()<<" "<<initializer[i].raw_data().size() <<"\n";
    std::string path = initializer[i].name();
    std::string buff = initializer[i].raw_data();
    std::ofstream out (path, std::ios::out|std::ios::binary);
    out.write(buff.data(),buff.size());
    out.close();
    std::vector<unsigned char> ubuff(buff.begin(),buff.end());
    std::string base64_str = base64_encode(ubuff.data(),ubuff.size());
    std::cout<<"add "<<name<<" to json\n";
    d.AddMember(rapidjson::Value().SetString(name.c_str(),allocator).Move(),rapidjson::Value().SetString(base64_str.c_str(),allocator).Move(),allocator);
  }
  std::cout<<"write to json file\n";
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  d.Accept(writer);
  std::ofstream jout("param.json");
    jout<<buffer.GetString();
    jout.close();

  auto modelBuff = file_to_buff("param.json");
  rapidjson::Document din;
  rapidjson::ParseResult ok = din.Parse(modelBuff.data()); 
  if (!ok)
      std::cout<<"JSON parse error: "<<GetParseError_En(ok.Code())<<" ("<<ok.Offset()<<")\n";
  
}
