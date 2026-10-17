#include "crypto/AESCryptoWrapper.h"
#include "utils/logging.h"
#include "utils/flags.h"
#include <iostream>
#include <string>
#include <tuple>


using namespace BASE_NAMESPACE;
using namespace std;

static int failures = 0;

static void Check(const std::string &name, bool ok) {
  std::cout << (ok ? "PASS " : "FAIL ") << name << std::endl;
  if (!ok) ++failures;
}

int main(int argc, char *argv[]) {
  FLAGS_log_dir = ".";
  google::EnableLogCleaner(30);  // keep your logs for 30 days
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);  //初始化 glog
  google::SetStderrLogging(google::GLOG_FATAL);

  std::string hexkey,hexiv;
  std::tie(hexkey,hexiv) = GenAESKey();
  Check("GenAESKey", hexkey.size() == 64 && hexiv.size() == 32);

  // 16 的整数倍（含 0）会额外填充一整块
  for (std::size_t n : {0, 1, 15, 16, 17, 32, 1000}) {
    std::string msg(n, 'x');
    auto enc = AESCipherEncrypt(msg, hexkey, hexiv);
    bool ok = enc.size() == (n / 16 + 1) * 16 && AESCipherDecrypt(enc, hexkey, hexiv) == msg;
    Check("roundtrip len=" + std::to_string(n), ok);
  }

  auto enc = AESCipherEncrypt("Test this sign message", hexkey, hexiv);
  std::string other_key = hexkey;
  other_key[0] = other_key[0] == '0' ? '1' : '0';
  Check("wrong key -> empty", AESCipherDecrypt(enc, other_key, hexiv).empty());
  Check("truncated ciphertext -> empty", AESCipherDecrypt(enc.substr(0, enc.size() - 1), hexkey, hexiv).empty());
  Check("short key -> empty", AESCipherEncrypt("hello", "abcd", hexiv).empty());
  Check("non-hex key -> empty", AESCipherEncrypt("hello", std::string(64, 'g'), hexiv).empty());

  std::string model(100000, 'm');
  auto enc_model = AESCipherEncrypt(model);
  std::vector<char> buff(enc_model.begin(), enc_model.end());
  auto dec_model = AESCipherDecrypt(buff);
  Check("default key vector roundtrip", std::string(dec_model.begin(), dec_model.end()) == model);

  return failures == 0 ? 0 : 1;
}
