#include "utils/hex-util.h"
#include "utils/logging.h"
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <sstream>

#define AES_KEY_SIZE 256
#define AES_IV_SIZE 128

namespace BASE_NAMESPACE {

std::pair<std::string, std::string> GenAESKey() {
  std::vector<unsigned char> key(AES_KEY_SIZE / 8);
  std::vector<unsigned char> iv(AES_IV_SIZE / 8);

  // Generate random key and initialization vector
  if (RAND_bytes(key.data(), AES_KEY_SIZE / 8) != 1) { LOG(ERROR) << "Error generating random key"; }
  if (RAND_bytes(iv.data(), AES_IV_SIZE / 8) != 1) { LOG(ERROR) << "Error generating random initialization vector"; }

  LOG(INFO) << "key:" << HexBinaryEncoder(key);
  LOG(INFO) << "IV:" << HexBinaryEncoder(iv);
  return std::make_pair(HexBinaryEncoder(key), HexBinaryEncoder(iv));
}

std::string AESCipherEncrypt(const std::string &msg, const std::string &hexkey, const std::string &hexiv) {
  try {
    LOG(INFO) << "AESCipherEncrypt start";
    std::vector<unsigned char> plaintext{msg.begin(), msg.end()};
    std::vector<unsigned char> key = HexBinaryDecoder(hexkey);
    std::vector<unsigned char> iv = HexBinaryDecoder(hexiv);
    size_t ciphertext_len = ((plaintext.size() + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE) * AES_BLOCK_SIZE;
    std::vector<unsigned char> ciphertext(ciphertext_len);
    int len;
    size_t cipher_len;
    // Create and initialize the context
    EVP_CIPHER_CTX *ctx;
    if (!(ctx = EVP_CIPHER_CTX_new())) { LOG(ERROR) << "Error creating EVP context\n"; }
    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key.data(), iv.data()) != 1) {
      LOG(ERROR) << "Error initializing EVP context\n";
    }

    // Encrypt plaintext
    if (EVP_EncryptUpdate(ctx, ciphertext.data(), &len, plaintext.data(), plaintext.size()) != 1) {
      LOG(ERROR) << "Error encrypting plaintext\n";
    }
    cipher_len = len;

    // Finalize encryption
    if (EVP_EncryptFinal_ex(ctx, ciphertext.data() + len, &len) != 1) { LOG(ERROR) << "Error finalizing encryption\n"; }
    cipher_len += len;

    // Clean up the context
    EVP_CIPHER_CTX_free(ctx);
    LOG(INFO) << "ciphertext_len:" << ciphertext_len << " cipher_len:" << cipher_len;

    LOG(INFO) << "AESCipherEncrypt end";
    return std::string(ciphertext.begin(), ciphertext.begin() + cipher_len);
  }
  catch (...) {
    LOG(ERROR) << "AES Encrypt error";
    return "";
  }
}

std::string AESCipherDecrypt(const std::string &msg, const std::string &hexkey, const std::string &hexiv) {
  try {
    LOG(INFO) << "AESCipherDecrypt start";
    EVP_CIPHER_CTX *ctx;
    int len;
    size_t decrypted_len=0;
    std::vector<unsigned char> ciphertext{msg.begin(), msg.end()};
    std::vector<unsigned char> key = HexBinaryDecoder(hexkey);
    std::vector<unsigned char> iv = HexBinaryDecoder(hexiv);
    std::vector<unsigned char> decrypted(msg.size());
    // Create and initialize the context
    if (!(ctx = EVP_CIPHER_CTX_new())) {
      LOG(ERROR)<<"Error creating EVP context\n";
    }
    if (EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key.data(), iv.data()) != 1) {
      LOG(ERROR)<<"Error initializing EVP context\n";
    }

    // Decrypt ciphertext
    if (EVP_DecryptUpdate(ctx, decrypted.data(), &len, ciphertext.data(), ciphertext.size()) != 1) {
      LOG(ERROR)<<"Error decrypting ciphertext\n";
    }
    decrypted_len = len;

    // Finalize decryption
    if (EVP_DecryptFinal_ex(ctx, decrypted.data() + len, &len) != 1) {
      LOG(ERROR)<<"Error finalizing decryption\n";
    }
    decrypted_len += len;

    // Clean up the context
    EVP_CIPHER_CTX_free(ctx);
    LOG(INFO) << "AESCipherDecrypt end";
    return std::string(decrypted.begin(), decrypted.begin()+decrypted_len);
  }
  catch (...) {
    LOG(ERROR) << "AES Decrypt error";
    return "";
  }
}

std::string AESCipherDecrypt(const std::string &msg) {
  try {
    std::string hexkey = "e0080defd1d1612545c7adf21f6842a3d7589c98deb127c4100b2b83e2b9ffa5";
    std::string hexiv = "d9c9e68bfceb6ceff7c6636624a9c2ee";
    // CipherKey cipher_key("aes256", HexBinaryDecoder(hexkey), HexBinaryDecoder(hexiv));
    // Cipher::Ptr pCipher = CipherFactory::defaultFactory().createCipher(cipher_key);
    // return pCipher->decryptString(msg);
    return AESCipherDecrypt(msg, hexkey, hexiv);
  }
  catch (...) {
    LOG(ERROR) << "AES Decrypt error";
    return "";
  }
}

std::vector<char> AESCipherDecrypt(const std::vector<char> &buff) {
  std::string str(buff.begin(), buff.end());
  std::string out_str = AESCipherDecrypt(str);
  return std::vector<char>(out_str.begin(), out_str.end());
}

}  // namespace BASE_NAMESPACE