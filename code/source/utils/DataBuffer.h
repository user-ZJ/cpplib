#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

class DataBuffer {
 public:
  DataBuffer(const std::string &key, int *data, unsigned int length);
  DataBuffer(const DataBuffer &db);
  DataBuffer &operator=(const DataBuffer &db);
  DataBuffer(DataBuffer&& db) noexcept;
  DataBuffer& operator = (DataBuffer&& db) noexcept;
  ~DataBuffer();

 private:
  std::string m_key;
  unsigned int m_length;
  int *m_data;
};

DataBuffer::DataBuffer(const std::string &key, int *data, unsigned int length) : m_key(key), m_length(length) {
  if (data != nullptr && m_length > 0) {
    m_data = new int[m_length];
    memcpy(m_data, data, sizeof(int) * m_length);
  } else {
    m_length = 0;
  }
}

DataBuffer::DataBuffer(const DataBuffer &db) : m_key(db.m_key), m_length(db.m_length) {
  if (db.m_data != nullptr && db.m_length > 0) {
    m_data = new int[db.m_length];
    memcpy(m_data, db.m_data, sizeof(int) * m_length);
  }
}
DataBuffer &DataBuffer::operator=(const DataBuffer &db) {
  if (this == &db) return *this;
  // copy & swap
  DataBuffer tmp{db};
  std::swap(m_key, tmp.m_key);
  std::swap(m_data, tmp.m_data);
  std::swap(m_length, tmp.m_length);
  return *this;
}
DataBuffer::DataBuffer(DataBuffer&& db) noexcept
  : m_key(std::move(db.m_key))
  , m_length(db.m_length)
  , m_data(db.m_data) 
{
    db.m_data = nullptr;
    db.m_length=0;
}

DataBuffer& DataBuffer::operator=(DataBuffer&& db) noexcept
{
    if (this == &db) return *this;
   
    DataBuffer tmp{std::move(db)};
    std::swap(tmp.m_key, m_key);
    std::swap(tmp.m_length, m_length);
    std::swap(tmp.m_data, m_data);
    return *this;
}
DataBuffer::~DataBuffer() {
  // 删除空指针是安全的
  delete[] m_data;
}

