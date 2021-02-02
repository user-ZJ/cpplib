## char* 转 string

```cpp
//方法1
std::string str(buffer,buffer+length);
//或
std::string str(buffer,buffer+length,'\0');
//如果字符串已存在
str.assign(buffer,buffer+length);
```

## string 转stream

```cpp
stringstream ss(str);
```

## stringstream转string

```cpp
stringstream ss;
ss<< "asdfg";
string str = ss.str();
```

## char * 转stringstream

```cpp
stringstream iss(reinterpret_cast<char *>(buff),length);
```

## stringstream转char *

```cpp
stringstring oss;
const char* oss.str().c_str();
```

