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
syntax = "proto3";

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

## 4. 从其他文件中导入模块

```protobuf
import "myproject/other_protos.proto";
```

protobuf编译器会从`-I/--proto_path`指定的路劲中查找import的文件，如果没有指定，则默认在当前编译目录下查找

## 5. oneof

一个信息结构体中有多个字段，所有字段共享内存，每次最多只能设置一个字段，当一个字段被设置时，其他字段会被清除。

使用方法为，在结构体命名后添加`_oneof`后缀。

```cpp
message SampleMessage {
  oneof test_oneof {
    string name = 4;
    SubMessage sub_message = 9;
  }
}
```

oneof的字段中不能有map好repeat

## 6. maps

```protobuf
map<key_type, value_type> map_field = N;
```

key_type是标量（整形，浮点型，字符串），但不可以是枚举类型

value_type是处理map类型以外的所有类型

map类型不能是repeated

```protobuf
map<string, Project> projects = 3;
```

## 7. 定义rpc服务

定义一个请求参数为SearchRequest，返回参数为SearchResponse的服务

```protobuf
service SearchService {
  rpc Search(SearchRequest) returns (SearchResponse);
}
```

## 8. json支持

protobuf支持json编码规范，可以更方便的在各个系统中传输数据。protobuf和json类型对应关系如下：

如果 JSON 编码的数据中缺少某个值，或者它的值为 null，则在解析到协议缓冲区时，它将被解释为适当的默认值。 如果某个字段在协议缓冲区中具有默认值，则在 JSON 编码的数据中默认将其省略以节省空间。 实现可以提供选项以在 JSON 编码的输出中发出具有默认值的字段。

| proto3                 | json          | json example                              | notes                                                        |
| ---------------------- | ------------- | ----------------------------------------- | ------------------------------------------------------------ |
| message                | object        | {"fooBar": v, "g": null, …}               | 生成 JSON 对象。 消息字段名称映射到 lowerCamelCase 并成为 JSON 对象键。 如果指定了 json_name 字段选项，则指定的值将用作键。 解析器接受 lowerCamelCase 名称（或由 json_name 选项指定的名称）和原始 proto 字段名称。 null 是所有字段类型的可接受值，并被视为相应字段类型的默认值。 |
| enum                   | string        | `"FOO_BAR"`                               | 使用 proto 中指定的枚举值的名称。 解析器接受枚举名称和整数值。 |
| map<K,V>               | object        | `{"k": v, …}`                             | 所有的key都转换为字符类型                                    |
| repeated V             | array         | `[v, …]`                                  | null被转换为空列表                                           |
| bool                   | true, false   | `true, false`                             |                                                              |
| string                 | string        | `"Hello World!"`                          |                                                              |
| bytes                  | base64 string | `"YWJjMTIzIT8kKiYoKSctPUB+"`              | JSON 值将是使用带有填充的标准 base64 编码编码为字符串的数据。 接受带有/不带有填充的标准或 URL 安全的 base64 编码。 |
| int32, fixed32, uint32 | number        | `1, -10, 0`                               | JSON 值将是一个十进制数。 接受数字或字符串。                 |
| int64, fixed64, uint64 | string        | `"1", "-10"`                              | JSON 值将是一个十进制字符串。 接受数字或字符串。             |
| float, double          | number        | `1.1, -10.0, 0, "NaN", "Infinity"`        | JSON 值将是一个数字或特殊字符串值“NaN”、“Infinity”和“-Infinity”之一。 接受数字或字符串。 也接受指数符号。 -0 被认为等同于 0。 |
| Any                    | `object`      | `{"@type": "url", "f": v, … }`            | 如果 Any 包含一个具有特殊 JSON 映射的值，则将其转换为：{"@type": xxx, "value": yyy}。 否则，该值将被转换为 JSON 对象，并插入“@type”字段以指示实际数据类型。 |
| Timestamp              | string        | `"1972-01-01T10:00:20.021Z"`              | 使用 RFC 3339，其中生成的输出将始终进行 Z 归一化，并使用 0、3、6 或 9 位小数。 也接受除“Z”之外的偏移量。 |
| Duration               | string        | `"1.000340012s", "1s"`                    | 生成的输出始终包含 0、3、6 或 9 个小数位数，具体取决于所需的精度，后跟后缀“s”。 接受任何小数位（也可以没有），只要它们符合纳秒精度并且需要后缀“s”。 |
| Struct                 | `object`      | `{ … }`                                   | Any JSON object. See `struct.proto`.                         |
| Wrapper types          | various types | `2, "2", "foo", true, "true", null, 0, …` | 包装器在 JSON 中使用与包装的原始类型相同的表示形式，除了在数据转换和传输期间允许并保留“null”。 |
| FieldMask              | string        | `"f.fooBar,h"`                            | See `field_mask.proto`.                                      |
| ListValue              | array         | `[foo, bar, …]`                           |                                                              |
| Value                  | value         |                                           | Any JSON value. Check [google.protobuf.Value](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#google.protobuf.Value) for details. |
| NullValue              | null          |                                           | JSON null                                                    |
| Empty                  | object        | `{}`                                      | An empty JSON object                                         |

## 9. 生成代码选项

`optimize_for`（文件选项）：可以设置为`SPEED`、`CODE_SIZE`或`LITE_RUNTIME`。这会通过以下方式影响 C++ 和 Java 代码生成器（可能还有第三方生成器）：

- `SPEED`（默认）：protocol buffer 编译器将生成用于对消息类型进行序列化、解析和执行其他常见操作的代码。这段代码是高度优化的。
- `CODE_SIZE`：协议缓冲区编译器将生成最少的类，并将依赖共享的、基于反射的代码来实现序列化、解析和各种其他操作。因此生成的代码将比 with 小得多`SPEED`，但操作会更慢。类仍将实现与模式中完全相同的公共 API `SPEED`。此模式在包含大量`.proto`文件且不需要所有文件都非常快的应用程序中最有用。
- `LITE_RUNTIME`：protocol buffer 编译器将生成仅依赖于“lite”运行时库的类（`libprotobuf-lite`而不是`libprotobuf`）。lite 运行时比完整库小得多（大约小一个数量级），但省略了描述符和反射等某些功能。这对于在手机等受限平台上运行的应用程序特别有用。编译器仍将生成所有方法的快速实现，就像它在`SPEED`模式中所做的那样。生成的类只会实现`MessageLite`每种语言的接口，它只提供完整`Message`接口方法的子集。

```protobuf
option optimize_for = CODE_SIZE;
```

`cc_enable_arenas`（文件选项）：为 C++ 生成的代码启用[竞技场分配。](https://developers.google.com/protocol-buffers/docs/reference/arenas)

`deprecated`（字段选项）：如果设置为`true`，则表示该字段已弃用，不应被新代码使用。在大多数语言中，这没有实际效果。在 Java 中，这成为`@Deprecated`注解。将来，其他特定于语言的代码生成器可能会在字段的访问器上生成弃用注释，这反过来会导致在编译尝试使用该字段的代码时发出警告。如果该字段未被任何人使用并且您希望阻止新用户使用它，请考虑将字段声明替换为[保留](https://developers.google.com/protocol-buffers/docs/proto3#reserved)语句。

```protobuf
int32 old_field = 6 [deprecated = true];
```



## 参考

https://developers.google.com/protocol-buffers/docs/proto3