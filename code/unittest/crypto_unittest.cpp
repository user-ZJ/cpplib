#include <string>
#include <tuple>
#include <vector>
#include "crypto/AESCryptoWrapper.h"
#include "gtest/gtest.h"

using namespace BASE_NAMESPACE;

TEST(AesTest, generateKey) {
  std::string hexkey;
  std::string hexiv;
  std::tie(hexkey, hexiv) = GenAESKey();
  EXPECT_EQ(64u, hexkey.size());
  EXPECT_EQ(32u, hexiv.size());
}

TEST(AesTest, roundtrip) {
  std::string hexkey;
  std::string hexiv;
  std::tie(hexkey, hexiv) = GenAESKey();
  for (std::size_t n : {0u, 1u, 15u, 16u, 17u, 32u, 1000u}) {
    std::string msg(n, 'x');
    auto enc = AESCipherEncrypt(msg, hexkey, hexiv);
    EXPECT_EQ((n / 16 + 1) * 16, enc.size()) << "len=" << n;
    EXPECT_EQ(msg, AESCipherDecrypt(enc, hexkey, hexiv)) << "len=" << n;
  }
}

TEST(AesTest, rejectBadInput) {
  std::string hexkey;
  std::string hexiv;
  std::tie(hexkey, hexiv) = GenAESKey();
  auto enc = AESCipherEncrypt("Test this sign message", hexkey, hexiv);
  ASSERT_FALSE(enc.empty());

  std::string other_key = hexkey;
  other_key[0] = other_key[0] == '0' ? '1' : '0';
  EXPECT_NE("Test this sign message", AESCipherDecrypt(enc, other_key, hexiv));
  EXPECT_TRUE(AESCipherDecrypt(enc.substr(0, enc.size() - 1), hexkey, hexiv).empty());
  EXPECT_TRUE(AESCipherEncrypt("hello", "abcd", hexiv).empty());
  EXPECT_TRUE(AESCipherEncrypt("hello", std::string(64, 'g'), hexiv).empty());
}

TEST(AesTest, defaultKeyVectorRoundtrip) {
  std::string model(100000, 'm');
  auto enc_model = AESCipherEncrypt(model);
  ASSERT_FALSE(enc_model.empty());
  std::vector<char> buff(enc_model.begin(), enc_model.end());
  auto dec_model = AESCipherDecrypt(buff);
  EXPECT_EQ(model, std::string(dec_model.begin(), dec_model.end()));
}
