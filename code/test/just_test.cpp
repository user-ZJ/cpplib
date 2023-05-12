#include <iostream>
#include <cstring>
#include <openssl/rsa.h>
#include <openssl/pem.h>

int main() {
    // Generate RSA key pair
    RSA *rsa = RSA_generate_key(2048, RSA_F4, NULL, NULL);
    if (rsa == NULL) {
        std::cerr << "Failed to generate RSA key pair" << std::endl;
        return -1;
    }

    // Get public key
    BIO *bio = BIO_new(BIO_s_mem());
    if (bio == NULL) {
        std::cerr << "Failed to create BIO" << std::endl;
        RSA_free(rsa);
        return -1;
    }

    if (PEM_write_bio_RSAPublicKey(bio, rsa) != 1) {
        std::cerr << "Failed to write public key to BIO" << std::endl;
        BIO_free(bio);
        RSA_free(rsa);
        return -1;
    }

    char *public_key;
    long public_key_len = BIO_get_mem_data(bio, &public_key);
    std::cout << "Public key: " << std::endl << public_key << std::endl;

    // Get private key
    bio = BIO_new(BIO_s_mem());
    if (bio == NULL) {
        std::cerr << "Failed to create BIO" << std::endl;
        RSA_free(rsa);
        return -1;
    }

    if (PEM_write_bio_RSAPrivateKey(bio, rsa, NULL, NULL, 0, NULL, NULL) != 1) {
        std::cerr << "Failed to write private key to BIO" << std::endl;
        BIO_free(bio);
        RSA_free(rsa);
        return -1;
    }

    char *private_key;
    long private_key_len = BIO_get_mem_data(bio, &private_key);
    std::cout << "Private key: " << std::endl << private_key << std::endl;

    // Encrypt message with public key
    const char *message = "Hello, world!";
    unsigned char ciphertext[RSA_size(rsa)];
    int ciphertext_len = RSA_public_encrypt(strlen(message) + 1, (unsigned char *)message, ciphertext, rsa, RSA_PKCS1_OAEP_PADDING);
    if (ciphertext_len == -1) {
        std::cerr << "Failed to encrypt message" << std::endl;
        BIO_free(bio);
        RSA_free(rsa);
        return -1;
    }

    std::cout << "Ciphertext: ";
    for (int i = 0; i < ciphertext_len; i++) {
        printf("%02x", ciphertext[i]);
    }
    std::cout << std::endl;

    // Decrypt message with private key
    unsigned char plaintext[ciphertext_len];
    int plaintext_len = RSA_private_decrypt(ciphertext_len, ciphertext, plaintext, rsa, RSA_PKCS1_OAEP_PADDING);
    if (plaintext_len == -1) {
        std::cerr << "Failed to decrypt message" << std::endl;
        BIO_free(bio);
        RSA_free(rsa);
        return -1;
    }

    std::cout << "Plaintext: " << plaintext << std::endl;

    // Clean up
    BIO_free(bio);
    RSA_free(rsa);

    return 0;
}