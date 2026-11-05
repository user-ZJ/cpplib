#include <iostream>
#include <string>
#include "flatbuffers/idl.h"

int main()
{
    std::string input_json_data = "{first_name: \"somename\",last_name: \"someothername\",age: 21}";

    std::string schemafile;
    std::string jsonfile;
    bool ok = flatbuffers::LoadFile("sample.fbs", false, &schemafile);
    if (!ok) {
        std::cout << "load file failed!" << std::endl;
        return -1;
    }
    std::cout<<"schemafile"<<schemafile<<std::endl;

    flatbuffers::Parser parser;
    parser.Parse(schemafile.c_str());
    if (!parser.Parse(input_json_data.c_str())) {
        std::cout << "flatbuffers parser failed with error : " << parser.error_ << std::endl;
        return -1;
    }

    std::string jsongen;
    if (GenText(parser, parser.builder_.GetBufferPointer(), &jsongen)) {
        std::cout << "Couldn't serialize parsed data to JSON!" << std::endl;
        return -1;
    }

    std::cout << "intput json" << std::endl
            << input_json_data << std::endl
            << std::endl
            << "output json" << std::endl
            << jsongen << std::endl;

    return 0;
}