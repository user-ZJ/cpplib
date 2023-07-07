#include "AESCryptoWrapper.h"
#include "Poco/Crypto/Cipher.h"
#include "Poco/Crypto/CipherFactory.h"
#include "Poco/Crypto/CipherKey.h"
#include "Poco/Crypto/RSADigestEngine.h"
#include "Poco/Crypto/X509Certificate.h"
#include "utils/hex-util.h"
#include "utils/logging.h"
#include <sstream>

using namespace Poco::Crypto;

namespace BASE_NAMESPACE {

std::pair<std::string,std::string> GenAESKey(){
	CipherKey cipher_key("aes256");
	auto key = cipher_key.getKey();
	auto iv = cipher_key.getIV();
	LOG(INFO)<<"key:"<<HexBinaryEncoder(key);
	LOG(INFO)<<"IV:"<<HexBinaryEncoder(iv);
	return std::make_pair(HexBinaryEncoder(key),HexBinaryEncoder(iv));
}

std::string AESCipherEncrypt(const std::string &msg, const std::string &hexkey, const std::string &hexiv) {
  try {
    CipherKey cipher_key("aes256", HexBinaryDecoder(hexkey), HexBinaryDecoder(hexiv));
    Cipher::Ptr pCipher = CipherFactory::defaultFactory().createCipher(cipher_key);
    return pCipher->encryptString(msg);
  } catch (...) {
    LOG(ERROR) << "AES Encrypt error";
    return "";
  }
}

std::string AESCipherDecrypt(const std::string &msg, const std::string &hexkey, const std::string &hexiv) {
  try {
    CipherKey cipher_key("aes256", HexBinaryDecoder(hexkey), HexBinaryDecoder(hexiv));
    Cipher::Ptr pCipher = CipherFactory::defaultFactory().createCipher(cipher_key);
    return pCipher->decryptString(msg);
  } catch (...) {
    LOG(ERROR) << "AES Decrypt error";
    return "";
  }
}

std::string AESCipherDecrypt(const std::string &msg) {
  try {
    std::string hexkey = "e0080defd1d1612545c7adf21f6842a3d7589c98deb127c4100b2b83e2b9ffa5";
    std::string hexiv = "d9c9e68bfceb6ceff7c6636624a9c2ee";
    CipherKey cipher_key("aes256", HexBinaryDecoder(hexkey), HexBinaryDecoder(hexiv));
    Cipher::Ptr pCipher = CipherFactory::defaultFactory().createCipher(cipher_key);
    return pCipher->decryptString(msg);
  } catch (...) {
    LOG(ERROR) << "AES Decrypt error";
    return "";
  }
}

std::vector<char> AESCipherDecrypt(const std::vector<char> &buff) {
  std::string str(buff.begin(),buff.end());
  std::string out_str = AESCipherDecrypt(str);
  return std::vector<char>(out_str.begin(),out_str.end());
}

}  // namespace BASE_NAMESPACE