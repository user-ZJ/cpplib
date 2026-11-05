# Apply only to the pinned Poco sources; accept an already-applied patch.
# Fail on unexpected contents so a dependency upgrade cannot silently skip it.
function(poco_replace_checked relative_path before after)
    set(path "${POCO_SOURCE_DIR}/${relative_path}")
    file(READ "${path}" contents)
    string(FIND "${contents}" "${before}" original_position)
    string(FIND "${contents}" "${after}" patched_position)
    if(NOT original_position EQUAL -1)
        string(REPLACE "${before}" "${after}" patched "${contents}")
        file(WRITE "${path}" "${patched}")
    elseif(patched_position EQUAL -1)
        message(FATAL_ERROR "Poco compatibility patch no longer matches ${path}; review the dependency version and patch.")
    endif()
endfunction()

poco_replace_checked(Crypto/src/PKCS12Container.cpp
    "char* pBuffer = reinterpret_cast<char*>(X509_alias_get0(pCert, &length));"
    "const char* pBuffer = reinterpret_cast<const char*>(X509_alias_get0(pCert, &length));")
poco_replace_checked(Crypto/src/X509Certificate.cpp
    "_X509_NAME_oneline_utf8(X509_NAME *name)"
    "_X509_NAME_oneline_utf8(const X509_NAME *name)")
poco_replace_checked(Crypto/src/X509Certificate.cpp
    "if (X509_NAME* issuer" "if (const X509_NAME* issuer")
poco_replace_checked(Crypto/src/X509Certificate.cpp
    "if (X509_NAME* subj" "if (const X509_NAME* subj")
poco_replace_checked(Crypto/src/X509Certificate.cpp
    "reinterpret_cast<char*>(certTime->data)"
    "reinterpret_cast<const char*>(ASN1_STRING_get0_data(certTime)), ASN1_STRING_length(certTime)")
poco_replace_checked(Crypto/src/X509Certificate.cpp
    "certTime->type" "ASN1_STRING_type(certTime)")
