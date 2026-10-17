#include "AESCryptoWrapper.h"
#include "utils/hex-util.h"
#include "utils/logging.h"
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <algorithm>
#include <memory>

namespace BASE_NAMESPACE {

namespace {

constexpr std::size_t kKeySize = 32;  // AES-256
constexpr std::size_t kIvSize = 16;
constexpr std::size_t kBlockSize = 16;
// EVP 接口长度参数为 int，大输入需分段
constexpr std::size_t kMaxChunk = 1 << 30;

// 内置密钥，用于模型文件加解密；修改会导致已加密的模型无法解密
const char *const kDefaultHexKey = "e0080defd1d1612545c7adf21f6842a3d7589c98deb127c4100b2b83e2b9ffa5";
const char *const kDefaultHexIv = "d9c9e68bfceb6ceff7c6636624a9c2ee";

struct CipherCtxDeleter {
  void operator()(EVP_CIPHER_CTX *ctx) const { EVP_CIPHER_CTX_free(ctx); }
};
using CipherCtxPtr = std::unique_ptr<EVP_CIPHER_CTX, CipherCtxDeleter>;

// 用完即擦除，避免密钥残留在已释放的堆内存中
struct SecretBytes {
  std::vector<unsigned char> data;
  ~SecretBytes() {
    if (!data.empty()) OPENSSL_cleanse(data.data(), data.size());
  }
};

void LogOpenSSLError(const char *what) {
  char buf[256] = "unknown error";
  unsigned long err = ERR_get_error();
  if (err != 0) ERR_error_string_n(err, buf, sizeof(buf));
  ERR_clear_error();
  LOG(ERROR) << what << ": " << buf;
}

bool DecodeHexExact(const std::string &hex, std::size_t size, const char *name, SecretBytes &out) {
  out.data = HexBinaryDecoder(hex);
  if (out.data.size() != size) {
    LOG(ERROR) << "invalid AES " << name << ": expect " << size * 2 << " hex chars";
    return false;
  }
  return true;
}

// AES-256-CBC + PKCS7；失败时清空 out 并返回 false
template <typename Out>
bool AesCbc(bool encrypt, const void *input, std::size_t input_len, const std::string &hexkey,
            const std::string &hexiv, Out &out) {
  out.clear();
  SecretBytes key, iv;
  if (!DecodeHexExact(hexkey, kKeySize, "key", key) || !DecodeHexExact(hexiv, kIvSize, "iv", iv)) return false;
  if (!encrypt && input_len % kBlockSize != 0) {
    LOG(ERROR) << "AES decrypt: ciphertext length " << input_len << " is not a multiple of " << kBlockSize;
    return false;
  }

  CipherCtxPtr ctx(EVP_CIPHER_CTX_new());
  if (!ctx) {
    LogOpenSSLError("EVP_CIPHER_CTX_new");
    return false;
  }
  if (EVP_CipherInit_ex(ctx.get(), EVP_aes_256_cbc(), nullptr, key.data.data(), iv.data.data(), encrypt ? 1 : 0)
      != 1) {
    LogOpenSSLError("EVP_CipherInit_ex");
    return false;
  }

  // PKCS7 填充最多增加一整块
  out.resize(input_len + kBlockSize);
  auto *dst = reinterpret_cast<unsigned char *>(&out[0]);
  const auto *src = static_cast<const unsigned char *>(input);
  std::size_t total = 0;
  for (std::size_t offset = 0; offset < input_len; offset += kMaxChunk) {
    int chunk = static_cast<int>(std::min(kMaxChunk, input_len - offset));
    int written = 0;
    if (EVP_CipherUpdate(ctx.get(), dst + total, &written, src + offset, chunk) != 1) {
      LogOpenSSLError(encrypt ? "AES encrypt" : "AES decrypt");
      out.clear();
      return false;
    }
    total += static_cast<std::size_t>(written);
  }
  int written = 0;
  if (EVP_CipherFinal_ex(ctx.get(), dst + total, &written) != 1) {
    // 解密时多为密钥错误或数据损坏导致填充校验失败
    LogOpenSSLError(encrypt ? "AES encrypt final" : "AES decrypt final (wrong key or corrupted data)");
    out.clear();
    return false;
  }
  total += static_cast<std::size_t>(written);
  out.resize(total);
  return true;
}

}  // namespace

std::pair<std::string, std::string> GenAESKey() {
  SecretBytes key, iv;
  key.data.resize(kKeySize);
  iv.data.resize(kIvSize);
  if (RAND_bytes(key.data.data(), static_cast<int>(kKeySize)) != 1
      || RAND_bytes(iv.data.data(), static_cast<int>(kIvSize)) != 1) {
    LogOpenSSLError("RAND_bytes");
    return {};
  }
  return std::make_pair(HexBinaryEncoder(key.data), HexBinaryEncoder(iv.data));
}

std::string AESCipherEncrypt(const std::string &msg, const std::string &hexkey, const std::string &hexiv) {
  std::string out;
  AesCbc(true, msg.data(), msg.size(), hexkey, hexiv, out);
  return out;
}

std::string AESCipherEncrypt(const std::string &msg) {
  return AESCipherEncrypt(msg, kDefaultHexKey, kDefaultHexIv);
}

std::string AESCipherDecrypt(const std::string &msg, const std::string &hexkey, const std::string &hexiv) {
  std::string out;
  AesCbc(false, msg.data(), msg.size(), hexkey, hexiv, out);
  return out;
}

std::string AESCipherDecrypt(const std::string &msg) {
  return AESCipherDecrypt(msg, kDefaultHexKey, kDefaultHexIv);
}

std::vector<char> AESCipherDecrypt(const std::vector<char> &buff) {
  std::vector<char> out;
  AesCbc(false, buff.data(), buff.size(), kDefaultHexKey, kDefaultHexIv, out);
  return out;
}

}  // namespace BASE_NAMESPACE
