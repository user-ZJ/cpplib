#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <limits>
#include "utils/file-util.h"
#include "utils/logging.h"
#include "crypto/AESCryptoWrapper.h"



using namespace BASE_NAMESPACE;


int main(int argc, char *argv[]) {
  FLAGS_log_dir = ".";
  google::EnableLogCleaner(30);  // keep your logs for 30 days
  // gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);  //初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  if (argc < 4) {
    LOG(INFO) << "usage: aes_crypto -e/-d inputfile outputfile";
    exit(1);
  }
  std::string command(argv[1]);

  std::string hexkey = "e0080defd1d1612545c7adf21f6842a3d7589c98deb127c4100b2b83e2b9ffa5";
  std::string hexiv = "d9c9e68bfceb6ceff7c6636624a9c2ee";

  std::string msg = file_to_str(argv[2]);
  std::string out_str;

  if(command=="-e"){
    out_str = AESCipherEncrypt(msg,hexkey,hexiv);
  }else if(command=="-d"){
    out_str = AESCipherDecrypt(msg,hexkey,hexiv);
  }else{
    LOG(ERROR)<<"unsupport command:"<<command;
    return 1;
  }
  write_to_file(argv[3],out_str);

  return 0;
}