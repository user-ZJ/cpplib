#include "AESCryptoWrapper.h"
#include "Poco/Crypto/Cipher.h"
#include "Poco/Crypto/CipherKey.h"
#include "Poco/Crypto/CipherFactory.h"
#include "Poco/Crypto/RSADigestEngine.h"
#include "Poco/Crypto/X509Certificate.h"
#include "utils/logging.h"
#include <sstream>
#include "utils/hex-util.h"

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

std::string AESCipherEncrypt(const std::string &msg, const std::string &hexkey,const std::string &hexiv) {
  try {
    CipherKey cipher_key("aes256",HexBinaryDecoder(hexkey),HexBinaryDecoder(hexiv));
    Cipher::Ptr pCipher = CipherFactory::defaultFactory().createCipher(cipher_key);
    return pCipher->encryptString(msg);
  }
  catch (...) {
    LOG(ERROR) << "AES Encrypt error";
    return "";
  }
}

std::string AESCipherDecrypt(const std::string &msg, const std::string &hexkey,const std::string &hexiv) {
  try {
    CipherKey cipher_key("aes256",HexBinaryDecoder(hexkey),HexBinaryDecoder(hexiv));
    Cipher::Ptr pCipher = CipherFactory::defaultFactory().createCipher(cipher_key);
    return pCipher->decryptString(msg);
  }
  catch (...) {
    LOG(ERROR) << "AES Decrypt error";
    return "";
  }
}

}  // namespace BASE_NAMESPACE