#include "PocoRSACryptoWrapper.h"
#include "Poco/Crypto/Cipher.h"
#include "Poco/Crypto/CipherFactory.h"
#include "Poco/Crypto/RSADigestEngine.h"
#include "Poco/Crypto/RSAKey.h"
#include "utils/logging.h"
#include <sstream>

using namespace Poco::Crypto;

namespace BASE_NAMESPACE {

void GenRSAKeyFile(const std::string &passphrase) {
  RSAKey key(RSAKey::KL_2048, RSAKey::EXP_LARGE);
  std::string pubFile("key.pub");
  std::string privFile("key.priv");
  LOG(INFO) << "save key to " << pubFile << " and " << privFile;
  key.save(pubFile, privFile, passphrase);
}

std::pair<std::string, std::string> GenRSAKey(const std::string &passphrase) {
  try {
    RSAKey key(RSAKey::KL_2048, RSAKey::EXP_LARGE);
    std::ostringstream strPub;
    std::ostringstream strPriv;
    key.save(&strPub, &strPriv, passphrase);
    return std::make_pair(strPriv.str(), strPub.str());
  } catch (const Poco::Exception &e) {
    LOG(ERROR) << "GenRSAKey error: " << e.displayText();
    return {};
  }
}

std::string SignSha256(const std::string &msg, const std::string &pubKey, const std::string &privKey,
                       const std::string &passphrase) {
  try {
    std::istringstream iPub(pubKey);
    std::istringstream iPriv(privKey);
    RSAKey key(&iPub, &iPriv, passphrase);
    RSADigestEngine eng(key, "SHA256");
    eng.update(msg.data(), static_cast<unsigned>(msg.length()));
    return Poco::DigestEngine::digestToHex(eng.signature());
  } catch (const Poco::Exception &e) {
    LOG(ERROR) << "sign sha256 error: " << e.displayText();
    return "";
  }
}

bool VerifySha256(const std::string &msg, const std::string &hexDig, const std::string &pubKey) {
  try {
    std::istringstream iPub(pubKey);
    RSAKey key(&iPub);
    RSADigestEngine eng(key, "SHA256");
    eng.update(msg.data(), static_cast<unsigned>(msg.length()));
    return eng.verify(Poco::DigestEngine::digestFromHex(hexDig));
  } catch (const Poco::Exception &e) {
    LOG(ERROR) << "verify sha256 error: " << e.displayText();
    return false;
  }
}

std::string RSACipherEncrypt(const std::string &msg, const std::string &pubKey) {
  try {
    std::istringstream iPub(pubKey);
    RSAKey key(&iPub);
    // PKCS#1 v1.5 加密填充存在 Bleichenbacher 攻击，使用 OAEP
    Cipher::Ptr pCipher = CipherFactory::defaultFactory().createCipher(key, RSA_PADDING_PKCS1_OAEP);
    return pCipher->encryptString(msg);
  } catch (const Poco::Exception &e) {
    LOG(ERROR) << "RSA encrypt error: " << e.displayText();
    return "";
  }
}

std::string RSACipherDecrypt(const std::string &msg, const std::string &privKey, const std::string &passphrase) {
  try {
    std::istringstream iPriv(privKey);
    RSAKey key(nullptr, &iPriv, passphrase);
    Cipher::Ptr pCipher = CipherFactory::defaultFactory().createCipher(key, RSA_PADDING_PKCS1_OAEP);
    return pCipher->decryptString(msg);
  } catch (const Poco::Exception &e) {
    LOG(ERROR) << "RSA decrypt error: " << e.displayText();
    return "";
  }
}

}  // namespace BASE_NAMESPACE
