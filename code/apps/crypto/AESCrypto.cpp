#include <iostream>
#include <string>
#include "utils/file-util.h"
#include "utils/logging.h"
#include "crypto/AESCryptoWrapper.h"



using namespace BASE_NAMESPACE;


int main(int argc, char *argv[]) {
  FLAGS_log_dir = ".";
  google::EnableLogCleaner(std::chrono::hours(24 * 30));  // keep your logs for 30 days
  google::InitGoogleLogging(argv[0]);  //初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  if (argc < 4) {
    std::cerr << "usage: aes_crypto -e/-d inputfile outputfile" << std::endl;
    return 1;
  }
  std::string command(argv[1]);
  std::string msg = file_to_str(argv[2]);
  std::string out_str;

  if(command=="-e"){
    out_str = AESCipherEncrypt(msg);
  }else if(command=="-d"){
    out_str = AESCipherDecrypt(msg);
  }else{
    LOG(ERROR)<<"unsupport command:"<<command;
    return 1;
  }
  if (out_str.empty() && !msg.empty()) {
    LOG(ERROR) << command << " failed: " << argv[2];
    return 1;
  }
  writeBinaryFile(argv[3],out_str);

  return 0;
}
