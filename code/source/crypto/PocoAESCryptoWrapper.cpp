#include "AESCryptoWrapper.h"
#include "Poco/Crypto/Cipher.h"
#include "Poco/Crypto/CipherFactory.h"
#include "Poco/Crypto/CipherKey.h"
#include "utils/hex-util.h"
#include "utils/logging.h"

using namespace Poco::Crypto;

namespace BASE_NAMESPACE {

namespace {

// 内置密钥，用于模型文件加解密；修改会导致已加密的模型无法解密
const char *const kDefaultHexKey = "e0080defd1d1612545c7adf21f6842a3d7589c98deb127c4100b2b83e2b9ffa5";
const char *const kDefaultHexIv = "d9c9e68bfceb6ceff7c6636624a9c2ee";

Cipher::Ptr CreateCipher(const std::string &hexkey, const std::string &hexiv) {
  auto key = HexBinaryDecoder(hexkey);
  auto iv = HexBinaryDecoder(hexiv);
  if (key.size() != 32 || iv.size() != 16) throw Poco::InvalidArgumentException("invalid AES key or iv");
  CipherKey cipher_key("aes-256-cbc", key, iv);
  return CipherFactory::defaultFactory().createCipher(cipher_key);
}

}  // namespace

std::pair<std::string, std::string> GenAESKey() {
  try {
    CipherKey cipher_key("aes-256-cbc");
    return std::make_pair(HexBinaryEncoder(cipher_key.getKey()), HexBinaryEncoder(cipher_key.getIV()));
  } catch (const Poco::Exception &e) {
    LOG(ERROR) << "GenAESKey error: " << e.displayText();
    return {};
  }
}

std::string AESCipherEncrypt(const std::string &msg, const std::string &hexkey, const std::string &hexiv) {
  try {
    return CreateCipher(hexkey, hexiv)->encryptString(msg);
  } catch (const Poco::Exception &e) {
    LOG(ERROR) << "AES encrypt error: " << e.displayText();
    return "";
  }
}

std::string AESCipherDecrypt(const std::string &msg, const std::string &hexkey, const std::string &hexiv) {
  try {
    return CreateCipher(hexkey, hexiv)->decryptString(msg);
  } catch (const Poco::Exception &e) {
    LOG(ERROR) << "AES decrypt error (wrong key or corrupted data): " << e.displayText();
    return "";
  }
}

std::string AESCipherEncrypt(const std::string &msg) {
  return AESCipherEncrypt(msg, kDefaultHexKey, kDefaultHexIv);
}

std::string AESCipherDecrypt(const std::string &msg) {
  return AESCipherDecrypt(msg, kDefaultHexKey, kDefaultHexIv);
}

std::vector<char> AESCipherDecrypt(const std::vector<char> &buff) {
  std::string out_str = AESCipherDecrypt(std::string(buff.begin(), buff.end()));
  return std::vector<char>(out_str.begin(), out_str.end());
}

}  // namespace BASE_NAMESPACE
