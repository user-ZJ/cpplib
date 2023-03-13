#include "crypto/RSACryptoWrapper.h"
#include "crypto/AESCryptoWrapper.h"
#include "utils/logging.h"
#include "utils/flags.h"
#include <iostream>
#include <string>
#include <tuple>


using namespace BASE_NAMESPACE;
using namespace std;

int main(int argc, char *argv[]) {
  FLAGS_log_dir = ".";
  google::EnableLogCleaner(30);  // keep your logs for 30 days
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);  //初始化 glog
  google::SetStderrLogging(google::GLOG_ERROR);
  std::string msg("Test this sign message");
  std::string privKey, pubKey;
  std::tie(privKey, pubKey) = GenRSAKey();
  std::string hexsign = SignSha256(msg, pubKey,privKey);
  LOG(INFO) << "hexsign:" << hexsign;
  bool verify = VerifySha256(msg, hexsign, pubKey);
  LOG(INFO) << "verify:" << verify;

  std::string enc = RSACipherEncrypt(msg,pubKey);
  std::string dec = RSACipherDecrypt(enc,privKey);
  LOG(INFO)<<"dec:"<<dec;

  ////////////////////////////////////////
  std::string hexkey,hexiv;
  std::tie(hexkey,hexiv) = GenAESKey();
  enc = AESCipherEncrypt(msg,hexkey,hexiv);
  dec = AESCipherDecrypt(enc,hexkey,hexiv);
  LOG(INFO)<<"dec:"<<dec;
  return 0;
}
