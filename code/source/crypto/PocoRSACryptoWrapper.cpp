#include "RSACryptoWrapper.h"
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

void GenRSAKeyFile() {
  RSAKey key(RSAKey::KL_2048, RSAKey::EXP_LARGE);
  std::string pubFile("key.pub");
  std::string privFile("key.priv");
  LOG(INFO)<<"save key to "<<pubFile<<" and "<<privFile;
  key.save(pubFile, privFile, "test_password");
}

std::pair<std::string, std::string> GenRSAKey() {
  RSAKey key(RSAKey::KL_2048, RSAKey::EXP_LARGE);
  std::ostringstream strPub;
  std::ostringstream strPriv;
  key.save(&strPub, &strPriv, "test_password");
  std::string pubKey = strPub.str();
  std::string privKey = strPriv.str();
  LOG(INFO) << "private key:\n" << privKey;
  LOG(INFO) << "public key:\n" << pubKey;
  return std::make_pair(privKey, pubKey);
}

std::string SignSha256(const std::string &msg, const std::string &pubKey, const std::string &privKey) {
  try {
    std::istringstream iPub(pubKey);
    std::istringstream iPriv(privKey);
    RSAKey key(&iPub, &iPriv, "test_password");
    RSADigestEngine eng(key, "SHA256");
    eng.update(msg.c_str(), static_cast<unsigned>(msg.length()));
    const Poco::DigestEngine::Digest &sig = eng.signature();
    std::string hexDig = Poco::DigestEngine::digestToHex(sig);
    return hexDig;
  }
  catch (...) {
    LOG(ERROR) << "sign sha256 error";
    return "";
  }
}

bool VerifySha256(const std::string &msg, const std::string &hexDig, const std::string &pubKey) {
  std::istringstream iPub(pubKey);
  RSAKey key(&iPub);
  RSADigestEngine eng(key, "SHA256");
  eng.update(msg.c_str(), static_cast<unsigned>(msg.length()));
  Poco::DigestEngine::Digest sig = Poco::DigestEngine::digestFromHex(hexDig);
  return eng.verify(sig);
}

std::string RSACipherEncrypt(const std::string &msg, const std::string &pubKey) {
  try {
    std::istringstream iPub(pubKey);
    RSAKey key(&iPub);
    Cipher::Ptr pCipher = CipherFactory::defaultFactory().createCipher(key);
    return pCipher->encryptString(msg);
  }
  catch (...) {
    LOG(ERROR) << "RAS Encrypt error";
    return "";
  }
}

std::string RSACipherDecrypt(const std::string &msg, const std::string &privKey) {
  try {
    std::istringstream iPriv(privKey);
    RSAKey key(0, &iPriv, "test_password");
    Cipher::Ptr pCipher = CipherFactory::defaultFactory().createCipher(key);
    return pCipher->decryptString(msg);
  }
  catch (...) {
    LOG(ERROR) << "RAS Decrypt error";
    return "";
  }
}



}  // namespace BASE_NAMESPACE