# protobuf使用笔记

## 1. 安装

### linux安装

```shell
sudo apt-get install autoconf automake libtool curl make g++ unzip
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git submodule update --init --recursive
./autogen.sh
# 编译C++版本protocol runtime和protocolbuf编译器（protoc）
./configure #--prefix=/usr
make -j$(nproc) # $(nproc) ensures it uses all cores for compilation
make check
sudo make install
sudo ldconfig # refresh shared library cache.
```

## 2. 第一个示例

### 创建addressbook.proto文件

每个元素上的“= 1”、“= 2”标记标识该字段在二进制编码中使用的唯一“标签”。标签编号 1-15 比更高的编号需要少一个字节来编码，因此作为一种优化，您可以决定将这些标签用于常用或重复的元素，而将标签 16 和更高的标签用于不太常用的可选元素。重复字段中的每个元素都需要重新编码标签号，因此重复字段特别适合这种优化。

每个字段都必须使用以下修饰符之一进行注释：

- `optional`: 该字段可以设置也可以不设置。如果未设置可选字段值，则使用默认值。对于简单类型，您可以指定自己的默认值，就像我们在示例中为电话号码所做的那样`type`。否则，使用系统默认值：数字类型为零，字符串为空字符串，布尔值为 false。对于嵌入式消息，默认值始终是消息的“默认实例”或“原型”，没有设置任何字段。调用访问器以获取未显式设置的可选（或必需）字段的值始终返回该字段的默认值。
- `repeated`：该字段可以重复任意次数（包括零次）。重复值的顺序将保存在协议缓冲区中。将重复字段视为动态大小的数组。
- `required`：必须提供该字段的值，否则该消息将被视为“未初始化”。如果`libprotobuf`在调试模式下编译，序列化未初始化的消息将导致断言失败。在优化的构建中，会跳过检查并且无论如何都会写入消息。但是，解析未初始化的消息总是会失败（通过`false`从 parse 方法返回）。除此之外，必填字段的行为与可选字段完全相同。

```protobuf
syntax = "proto2";

// 包名，可防止命名冲突，在C++中会被编译成命名空间名
package tutorial;

message Person {
  optional string name = 1;
  optional int32 id = 2;
  optional string email = 3;

  enum PhoneType {
    MOBILE = 0;
    HOME = 1;
    WORK = 2;
  }

  message PhoneNumber {
    optional string number = 1;
    optional PhoneType type = 2 [default = HOME];
  }

  repeated PhoneNumber phones = 4;
}

message AddressBook {
  repeated Person people = 1;
}
```

### 编译proto文件

生成protobuf文件（需要先安装protobuf），生成addressbook.pb.h和addressbook.pb.cc

```shell
# SRC_DIR 表示应用程序的源代码所在的位置——如果不提供值，则使用当前目录
# DST_DIR 希望生成的代码所在的位置
# --cpp_out指生成C++类
protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/addressbook.proto
```

### 写信息

```cpp
#include <iostream>
#include <fstream>
#include <string>
#include "addressbook.pb.h"
using namespace std;

// This function fills in a Person message based on user input.
void PromptForAddress(tutorial::Person* person) {
  cout << "Enter person ID number: ";
  int id;
  cin >> id;
  person->set_id(id);
  cin.ignore(256, '\n');

  cout << "Enter name: ";
  getline(cin, *person->mutable_name());

  cout << "Enter email address (blank for none): ";
  string email;
  getline(cin, email);
  if (!email.empty()) {
    person->set_email(email);
  }

  while (true) {
    cout << "Enter a phone number (or leave blank to finish): ";
    string number;
    getline(cin, number);
    if (number.empty()) {
      break;
    }

    tutorial::Person::PhoneNumber* phone_number = person->add_phones();
    phone_number->set_number(number);

    cout << "Is this a mobile, home, or work phone? ";
    string type;
    getline(cin, type);
    if (type == "mobile") {
      phone_number->set_type(tutorial::Person::MOBILE);
    } else if (type == "home") {
      phone_number->set_type(tutorial::Person::HOME);
    } else if (type == "work") {
      phone_number->set_type(tutorial::Person::WORK);
    } else {
      cout << "Unknown phone type.  Using default." << endl;
    }
  }
}

// Main function:  Reads the entire address book from a file,
//   adds one person based on user input, then writes it back out to the same
//   file.
int main(int argc, char* argv[]) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 2) {
    cerr << "Usage:  " << argv[0] << " ADDRESS_BOOK_FILE" << endl;
    return -1;
  }

  tutorial::AddressBook address_book;

  {
    // Read the existing address book.
    fstream input(argv[1], ios::in | ios::binary);
    if (!input) {
      cout << argv[1] << ": File not found.  Creating a new file." << endl;
    } else if (!address_book.ParseFromIstream(&input)) {
      cerr << "Failed to parse address book." << endl;
      return -1;
    }
  }

  // Add an address.
  PromptForAddress(address_book.add_people());

  {
    // Write the new address book back to disk.
    fstream output(argv[1], ios::out | ios::trunc | ios::binary);
    if (!address_book.SerializeToOstream(&output)) {
      cerr << "Failed to write address book." << endl;
      return -1;
    }
  }

  // Optional:  Delete all global objects allocated by libprotobuf.
  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}
```

### 读信息

```cpp
#include <iostream>
#include <fstream>
#include <string>
#include "addressbook.pb.h"
using namespace std;

// Iterates though all people in the AddressBook and prints info about them.
void ListPeople(const tutorial::AddressBook& address_book) {
  for (int i = 0; i < address_book.people_size(); i++) {
    const tutorial::Person& person = address_book.people(i);

    cout << "Person ID: " << person.id() << endl;
    cout << "  Name: " << person.name() << endl;
    if (person.has_email()) {
      cout << "  E-mail address: " << person.email() << endl;
    }

    for (int j = 0; j < person.phones_size(); j++) {
      const tutorial::Person::PhoneNumber& phone_number = person.phones(j);

      switch (phone_number.type()) {
        case tutorial::Person::MOBILE:
          cout << "  Mobile phone #: ";
          break;
        case tutorial::Person::HOME:
          cout << "  Home phone #: ";
          break;
        case tutorial::Person::WORK:
          cout << "  Work phone #: ";
          break;
      }
      cout << phone_number.number() << endl;
    }
  }
}

// Main function:  Reads the entire address book from a file and prints all
//   the information inside.
int main(int argc, char* argv[]) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  if (argc != 2) {
    cerr << "Usage:  " << argv[0] << " ADDRESS_BOOK_FILE" << endl;
    return -1;
  }

  tutorial::AddressBook address_book;

  {
    // Read the existing address book.
    fstream input(argv[1], ios::in | ios::binary);
    if (!address_book.ParseFromIstream(&input)) {
      cerr << "Failed to parse address book." << endl;
      return -1;
    }
  }

  ListPeople(address_book);

  // Optional:  Delete all global objects allocated by libprotobuf.
  google::protobuf::ShutdownProtobufLibrary();

  return 0;
}
```

## 3. 扩展兼容

在你发布使用你的协议缓冲区的代码之后，迟早你肯定会想要“改进”协议缓冲区的定义。如果您希望您的新缓冲区向后兼容，并且您的旧缓冲区向前兼容——您几乎肯定希望这样做——那么您需要遵循一些规则。在新版本的协议缓冲区中：

- *不得*更改任何现有字段的标签号。
- *不得*添加或删除任何必填字段。
- *可以*删除可选或重复的字段。
- *可以*添加新的可选或重复字段，但您必须使用新的标记号（即，从未在此协议缓冲区中使用过的标记号，即使已删除的字段也不使用）。

遵循以上规则，旧代码可以直接使用新的proto信息，删除的可选字段将使用默认值，删除的可重复字段为空；

但是新增的可选字段不会出现在旧的消息中，需要使用`has_`函数来判断是否存在，或者在`.proto`文件中提供默认值。