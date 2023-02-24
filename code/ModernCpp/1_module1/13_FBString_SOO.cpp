/*
small: 0~23  本地缓存Local Buffer

medium: 24~254 存储在堆上，执行贪婪拷贝（eager copy）

large: >=255 存储堆上，但使用共享计数进行Copy-on-Write 

*/
#include<cstddef>
using namespace std;

template <class Char>
class fbstring_core {
 public:
  fbstring_core() noexcept { reset(); }

  //拷贝构造
  fbstring_core(const fbstring_core& rhs) {
    assert(&rhs != this);
    switch (rhs.category()) {
      case Category::isSmall:
        copySmall(rhs);
        break;
      case Category::isMedium:
        copyMedium(rhs);
        break;
      case Category::isLarge:
        copyLarge(rhs);
        break;
      default:
        folly::assume_unreachable();
    }
    assert(size() == rhs.size());
    assert(memcmp(data(), rhs.data(), size() * sizeof(Char)) == 0);
  }

  fbstring_core& operator=(const fbstring_core& rhs) = delete;

  fbstring_core(fbstring_core&& goner) noexcept {
    ml_ = goner.ml_;
    goner.reset();
  }

  //含参构造
  fbstring_core(
      const Char* const data,
      const size_t size,
      bool disableSSO = FBSTRING_DISABLE_SSO) {
    if (!disableSSO && size <= maxSmallSize) {
      initSmall(data, size);
    } else if (size <= maxMediumSize) {
      initMedium(data, size);
    } else {
      initLarge(data, size);
    }
    assert(this->size() == size);
    assert(size == 0 || memcmp(this->data(), data, size * sizeof(Char)) == 0);
  }

  ~fbstring_core() noexcept {
    if (category() == Category::isSmall) {
      return;
    }
    destroyMediumLarge();
  }

  // Snatches a previously mallocated string. The parameter "size"
  // is the size of the string, and the parameter "allocatedSize"
  // is the size of the mallocated block.  The string must be
  // \0-terminated, so allocatedSize >= size + 1 and data[size] == '\0'.
  //
  // So if you want a 2-character string, pass malloc(3) as "data",
  // pass 2 as "size", and pass 3 as "allocatedSize".
  fbstring_core(
      Char* const data,
      const size_t size,
      const size_t allocatedSize,
      AcquireMallocatedString) {
    if (size > 0) {
      assert(allocatedSize >= size + 1);
      assert(data[size] == '\0');
      // Use the medium string storage
      ml_.data_ = data;
      ml_.size_ = size;
      // Don't forget about null terminator
      ml_.setCapacity(allocatedSize - 1, Category::isMedium);
    } else {
      // No need for the memory
      free(data);
      reset();
    }
  }

  // swap below doesn't test whether &rhs == this (and instead
  // potentially does extra work) on the premise that the rarity of
  // that situation actually makes the check more expensive than is
  // worth.
  void swap(fbstring_core& rhs) {
    auto const t = ml_;
    ml_ = rhs.ml_;
    rhs.ml_ = t;
  }

  // In C++11 data() and c_str() are 100% equivalent.
  const Char* data() const { return c_str(); }

  Char* data() { return c_str(); }

  Char* mutableData() {
    switch (category()) {
      case Category::isSmall:
        return small_;
      case Category::isMedium:
        return ml_.data_;
      case Category::isLarge:
        return mutableDataLarge();
    }
    folly::assume_unreachable();
  }

  const Char* c_str() const {
    const Char* ptr = ml_.data_;
    // With this syntax, GCC and Clang generate a CMOV instead of a branch.
    ptr = (category() == Category::isSmall) ? small_ : ptr;
    return ptr;
  }

  void shrink(const size_t delta) {
    if (category() == Category::isSmall) {
      shrinkSmall(delta);
    } else if (
        category() == Category::isMedium || RefCounted::refs(ml_.data_) == 1) {
      shrinkMedium(delta);
    } else {
      shrinkLarge(delta);
    }
  }

  FOLLY_NOINLINE
  void reserve(size_t minCapacity, bool disableSSO = FBSTRING_DISABLE_SSO) {
    FOLLY_PUSH_WARNING
    FOLLY_CLANG_DISABLE_WARNING("-Wcovered-switch-default")
    switch (category()) {
      case Category::isSmall:
        reserveSmall(minCapacity, disableSSO);
        break;
      case Category::isMedium:
        reserveMedium(minCapacity);
        break;
      case Category::isLarge:
        reserveLarge(minCapacity);
        break;
      default:
        folly::assume_unreachable();
    }
    FOLLY_POP_WARNING
    assert(capacity() >= minCapacity);
  }

  Char* expandNoinit(
      const size_t delta,
      bool expGrowth = false,
      bool disableSSO = FBSTRING_DISABLE_SSO);

  void push_back(Char c) { *expandNoinit(1, /* expGrowth = */ true) = c; }

  size_t size() const {
    size_t ret = ml_.size_;
    if /* constexpr */ (kIsLittleEndian) {
      // We can save a couple instructions, because the category is
      // small iff the last char, as unsigned, is <= maxSmallSize.
      typedef typename std::make_unsigned<Char>::type UChar;
      auto maybeSmallSize = size_t(maxSmallSize) -
          size_t(static_cast<UChar>(small_[maxSmallSize]));
      // With this syntax, GCC and Clang generate a CMOV instead of a branch.
      ret = (static_cast<ssize_t>(maybeSmallSize) >= 0) ? maybeSmallSize : ret;
    } else {
      ret = (category() == Category::isSmall) ? smallSize() : ret;
    }
    return ret;
  }

  size_t capacity() const {
    FOLLY_PUSH_WARNING
    FOLLY_CLANG_DISABLE_WARNING("-Wcovered-switch-default")
    switch (category()) {
      case Category::isSmall:
        return maxSmallSize;
      case Category::isLarge:
        // For large-sized strings, a multi-referenced chunk has no
        // available capacity. This is because any attempt to append
        // data would trigger a new allocation.
        if (RefCounted::refs(ml_.data_) > 1) {
          return ml_.size_;
        }
        break;
      case Category::isMedium:
      default:
        break;
    }
    FOLLY_POP_WARNING
    return ml_.capacity();
  }

  //是否copy-on-write 共享
  bool isShared() const {
    return category() == Category::isLarge && RefCounted::refs(ml_.data_) > 1;
  }

 private:
  Char* c_str() {
    Char* ptr = ml_.data_;
    // With this syntax, GCC and Clang generate a CMOV instead of a branch.
    ptr = (category() == Category::isSmall) ? small_ : ptr;
    return ptr;
  }

  void reset() { setSmallSize(0); }

  //medium或large字符串的销毁
  FOLLY_NOINLINE void destroyMediumLarge() noexcept {
    auto const c = category();
    assert(c != Category::isSmall);
    if (c == Category::isMedium) {
      free(ml_.data_);
    } else {
      RefCounted::decrementRefs(ml_.data_);
    }
  }

  //引用计数管理
  struct RefCounted {
    std::atomic<size_t> refCount_;
    Char data_[1];

    constexpr static size_t getDataOffset() {
      return offsetof(RefCounted, data_);
    }

    static RefCounted* fromData(Char* p) {
      return static_cast<RefCounted*>(static_cast<void*>(
          static_cast<unsigned char*>(static_cast<void*>(p)) -
          getDataOffset()));
    }

    static size_t refs(Char* p) {
      return fromData(p)->refCount_.load(std::memory_order_acquire);
    }

    static void incrementRefs(Char* p) {
      fromData(p)->refCount_.fetch_add(1, std::memory_order_acq_rel);
    }

    static void decrementRefs(Char* p) {
      auto const dis = fromData(p);
      size_t oldcnt = dis->refCount_.fetch_sub(1, std::memory_order_acq_rel);
      assert(oldcnt > 0);
      if (oldcnt == 1) {
        free(dis);
      }
    }

    static RefCounted* create(size_t* size) {
      size_t capacityBytes;
      if (!folly::checked_add(&capacityBytes, *size, size_t(1))) {
        throw_exception(std::length_error(""));
      }
      if (!folly::checked_muladd(
              &capacityBytes, capacityBytes, sizeof(Char), getDataOffset())) {
        throw_exception(std::length_error(""));
      }
      const size_t allocSize = goodMallocSize(capacityBytes);
      auto result = static_cast<RefCounted*>(checkedMalloc(allocSize));
      result->refCount_.store(1, std::memory_order_release);
      *size = (allocSize - getDataOffset()) / sizeof(Char) - 1;
      return result;
    }

    static RefCounted* create(const Char* data, size_t* size) {
      const size_t effectiveSize = *size;
      auto result = create(size);
      if (FOLLY_LIKELY(effectiveSize > 0)) {
        fbstring_detail::podCopy(data, data + effectiveSize, result->data_);
      }
      return result;
    }

    static RefCounted* reallocate(
        Char* const data,
        const size_t currentSize,
        const size_t currentCapacity,
        size_t* newCapacity) {
      assert(*newCapacity > 0 && *newCapacity > currentSize);
      size_t capacityBytes;
      if (!folly::checked_add(&capacityBytes, *newCapacity, size_t(1))) {
        throw_exception(std::length_error(""));
      }
      if (!folly::checked_muladd(
              &capacityBytes, capacityBytes, sizeof(Char), getDataOffset())) {
        throw_exception(std::length_error(""));
      }
      const size_t allocNewCapacity = goodMallocSize(capacityBytes);
      auto const dis = fromData(data);
      assert(dis->refCount_.load(std::memory_order_acquire) == 1);
      auto result = static_cast<RefCounted*>(smartRealloc(
          dis,
          getDataOffset() + (currentSize + 1) * sizeof(Char),
          getDataOffset() + (currentCapacity + 1) * sizeof(Char),
          allocNewCapacity));
      assert(result->refCount_.load(std::memory_order_acquire) == 1);
      *newCapacity = (allocNewCapacity - getDataOffset()) / sizeof(Char) - 1;
      return result;
    }
  };

  typedef uint8_t category_type;

  enum class Category : category_type {
    isSmall = 0,
    isMedium = kIsLittleEndian ? 0x80 : 0x2,
    isLarge = kIsLittleEndian ? 0x40 : 0x1,
  };

  Category category() const {
    // works for both big-endian and little-endian
    return static_cast<Category>(bytes_[lastChar] & categoryExtractMask);
  }

  //medium 或large字符串结构
  struct MediumLarge {
    Char* data_;
    size_t size_;
    size_t capacity_;

    size_t capacity() const {
      return kIsLittleEndian ? capacity_ & capacityExtractMask : capacity_ >> 2;
    }

    void setCapacity(size_t cap, Category cat) {
      capacity_ = kIsLittleEndian
          ? cap | (static_cast<size_t>(cat) << kCategoryShift)
          : (cap << 2) | static_cast<size_t>(cat);
    }
  };

  //联合存储结构
  union {
    uint8_t bytes_[sizeof(MediumLarge)]; // For accessing the last byte.
    Char small_[sizeof(MediumLarge) / sizeof(Char)];
    MediumLarge ml_;
  };

  constexpr static size_t lastChar = sizeof(MediumLarge) - 1;
  constexpr static size_t maxSmallSize = lastChar / sizeof(Char);
  constexpr static size_t maxMediumSize = 254 / sizeof(Char);
  constexpr static uint8_t categoryExtractMask = kIsLittleEndian ? 0xC0 : 0x3;
  constexpr static size_t kCategoryShift = (sizeof(size_t) - 1) * 8;
  constexpr static size_t capacityExtractMask = kIsLittleEndian
      ? ~(size_t(categoryExtractMask) << kCategoryShift)
      : 0x0 /* unused */;

  static_assert(
      !(sizeof(MediumLarge) % sizeof(Char)),
      "Corrupt memory layout for fbstring.");

  size_t smallSize() const {
    assert(category() == Category::isSmall);
    constexpr auto shift = kIsLittleEndian ? 0 : 2;
    auto smallShifted = static_cast<size_t>(small_[maxSmallSize]) >> shift;
    assert(static_cast<size_t>(maxSmallSize) >= smallShifted);
    return static_cast<size_t>(maxSmallSize) - smallShifted;
  }

  void setSmallSize(size_t s) {
    // Warning: this should work with uninitialized strings too,
    // so don't assume anything about the previous value of
    // small_[maxSmallSize].
    assert(s <= maxSmallSize);
    constexpr auto shift = kIsLittleEndian ? 0 : 2;
    small_[maxSmallSize] = char((maxSmallSize - s) << shift);
    small_[s] = '\0';
    assert(category() == Category::isSmall && size() == s);
  }

  void copySmall(const fbstring_core&);
  void copyMedium(const fbstring_core&);
  void copyLarge(const fbstring_core&);

  void initSmall(const Char* data, size_t size);
  void initMedium(const Char* data, size_t size);
  void initLarge(const Char* data, size_t size);

  void reserveSmall(size_t minCapacity, bool disableSSO);
  void reserveMedium(size_t minCapacity);
  void reserveLarge(size_t minCapacity);

  void shrinkSmall(size_t delta);
  void shrinkMedium(size_t delta);
  void shrinkLarge(size_t delta);

  void unshare(size_t minCapacity = 0);
  Char* mutableDataLarge();
};


//拷贝small 字符串
template <class Char>
inline void fbstring_core<Char>::copySmall(const fbstring_core& rhs) {
  static_assert(offsetof(MediumLarge, data_) == 0, "fbstring layout failure");
  static_assert(
      offsetof(MediumLarge, size_) == sizeof(ml_.data_),
      "fbstring layout failure");
  static_assert(
      offsetof(MediumLarge, capacity_) == 2 * sizeof(ml_.data_),
      "fbstring layout failure");
  // Just write the whole thing, don't look at details. In
  // particular we need to copy capacity anyway because we want
  // to set the size (don't forget that the last character,
  // which stores a short string's length, is shared with the
  // ml_.capacity field).
  ml_ = rhs.ml_;
  assert(category() == Category::isSmall && this->size() == rhs.size());
}

//拷贝medium 字符串
template <class Char>
FOLLY_NOINLINE void fbstring_core<Char>::copyMedium(const fbstring_core& rhs) {
  // Medium strings are copied eagerly. Don't forget to allocate
  // one extra Char for the null terminator.
  auto const allocSize = goodMallocSize((1 + rhs.ml_.size_) * sizeof(Char));
  ml_.data_ = static_cast<Char*>(checkedMalloc(allocSize));
  // Also copies terminator.
  fbstring_detail::podCopy(
      rhs.ml_.data_, rhs.ml_.data_ + rhs.ml_.size_ + 1, ml_.data_);
  ml_.size_ = rhs.ml_.size_;
  ml_.setCapacity(allocSize / sizeof(Char) - 1, Category::isMedium);
  assert(category() == Category::isMedium);
}

//拷贝large 字符串
template <class Char>
FOLLY_NOINLINE void fbstring_core<Char>::copyLarge(const fbstring_core& rhs) {
  // Large strings are just refcounted
  ml_ = rhs.ml_;
  RefCounted::incrementRefs(ml_.data_);
  assert(category() == Category::isLarge && size() == rhs.size());
}

// 初始化small 字符串
template <class Char>
inline void fbstring_core<Char>::initSmall(
    const Char* const data, const size_t size) {
  // Layout is: Char* data_, size_t size_, size_t capacity_
  static_assert(
      sizeof(*this) == sizeof(Char*) + 2 * sizeof(size_t),
      "fbstring has unexpected size");
  static_assert(
      sizeof(Char*) == sizeof(size_t), "fbstring size assumption violation");
  // sizeof(size_t) must be a power of 2
  static_assert(
      (sizeof(size_t) & (sizeof(size_t) - 1)) == 0,
      "fbstring size assumption violation");

// If data is aligned, use fast word-wise copying. Otherwise,
// use conservative memcpy.
// The word-wise path reads bytes which are outside the range of
// the string, and makes ASan unhappy, so we disable it when
// compiling with ASan.
#ifndef FOLLY_SANITIZE_ADDRESS
  if ((reinterpret_cast<size_t>(data) & (sizeof(size_t) - 1)) == 0) {
    const size_t byteSize = size * sizeof(Char);
    constexpr size_t wordWidth = sizeof(size_t);
    switch ((byteSize + wordWidth - 1) / wordWidth) { // Number of words.
      case 3:
        ml_.capacity_ = reinterpret_cast<const size_t*>(data)[2];
        FOLLY_FALLTHROUGH;
      case 2:
        ml_.size_ = reinterpret_cast<const size_t*>(data)[1];
        FOLLY_FALLTHROUGH;
      case 1:
        ml_.data_ = *reinterpret_cast<Char**>(const_cast<Char*>(data));
        FOLLY_FALLTHROUGH;
      case 0:
        break;
    }
  } else
#endif
  {
    if (size != 0) {
      fbstring_detail::podCopy(data, data + size, small_);
    }
  }
  setSmallSize(size);
}

//初始化medium字符串
template <class Char>
FOLLY_NOINLINE void fbstring_core<Char>::initMedium(
    const Char* const data, const size_t size) {
  // Medium strings are allocated normally. Don't forget to
  // allocate one extra Char for the terminating null.
  auto const allocSize = goodMallocSize((1 + size) * sizeof(Char));
  ml_.data_ = static_cast<Char*>(checkedMalloc(allocSize));
  if (FOLLY_LIKELY(size > 0)) {
    fbstring_detail::podCopy(data, data + size, ml_.data_);
  }
  ml_.size_ = size;
  ml_.setCapacity(allocSize / sizeof(Char) - 1, Category::isMedium);
  ml_.data_[size] = '\0';
}


//初始化large字符串
template <class Char>
FOLLY_NOINLINE void fbstring_core<Char>::initLarge(
    const Char* const data, const size_t size) {
  // Large strings are allocated differently
  size_t effectiveCapacity = size;
  auto const newRC = RefCounted::create(data, &effectiveCapacity);
  ml_.data_ = newRC->data_;
  ml_.size_ = size;
  ml_.setCapacity(effectiveCapacity, Category::isLarge);
  ml_.data_[size] = '\0';
}


//copy-on-write 变更时更改
template <class Char>
FOLLY_NOINLINE void fbstring_core<Char>::unshare(size_t minCapacity) {
  assert(category() == Category::isLarge);
  size_t effectiveCapacity = std::max(minCapacity, ml_.capacity());
  auto const newRC = RefCounted::create(&effectiveCapacity);
  // If this fails, someone placed the wrong capacity in an
  // fbstring.
  assert(effectiveCapacity >= ml_.capacity());
  // Also copies terminator.
  fbstring_detail::podCopy(ml_.data_, ml_.data_ + ml_.size_ + 1, newRC->data_);
  RefCounted::decrementRefs(ml_.data_);
  ml_.data_ = newRC->data_;
  ml_.setCapacity(effectiveCapacity, Category::isLarge);
  // size_ remains unchanged.
}


//对于large字符串，进行copy-on-write
template <class Char>
inline Char* fbstring_core<Char>::mutableDataLarge() {
  assert(category() == Category::isLarge);
  if (RefCounted::refs(ml_.data_) > 1) { // Ensure unique.
    unshare();
  }
  return ml_.data_;
}

//保留large字符串
template <class Char>
FOLLY_NOINLINE void fbstring_core<Char>::reserveLarge(size_t minCapacity) {
  assert(category() == Category::isLarge);
  if (RefCounted::refs(ml_.data_) > 1) { // Ensure unique
    // We must make it unique regardless; in-place reallocation is
    // useless if the string is shared. In order to not surprise
    // people, reserve the new block at current capacity or
    // more. That way, a string's capacity never shrinks after a
    // call to reserve.
    unshare(minCapacity);
  } else {
    // String is not shared, so let's try to realloc (if needed)
    if (minCapacity > ml_.capacity()) {
      // Asking for more memory
      auto const newRC = RefCounted::reallocate(
          ml_.data_, ml_.size_, ml_.capacity(), &minCapacity);
      ml_.data_ = newRC->data_;
      ml_.setCapacity(minCapacity, Category::isLarge);
    }
    assert(capacity() >= minCapacity);
  }
}

template <class Char>
FOLLY_NOINLINE void fbstring_core<Char>::reserveMedium(
    const size_t minCapacity) {
  assert(category() == Category::isMedium);
  // String is not shared
  if (minCapacity <= ml_.capacity()) {
    return; // nothing to do, there's enough room
  }
  if (minCapacity <= maxMediumSize) {
    // Keep the string at medium size. Don't forget to allocate
    // one extra Char for the terminating null.
    size_t capacityBytes = goodMallocSize((1 + minCapacity) * sizeof(Char));
    // Also copies terminator.
    ml_.data_ = static_cast<Char*>(smartRealloc(
        ml_.data_,
        (ml_.size_ + 1) * sizeof(Char),
        (ml_.capacity() + 1) * sizeof(Char),
        capacityBytes));
    ml_.setCapacity(capacityBytes / sizeof(Char) - 1, Category::isMedium);
  } else {
    // Conversion from medium to large string
    fbstring_core nascent;
    // Will recurse to another branch of this function
    nascent.reserve(minCapacity);
    nascent.ml_.size_ = ml_.size_;
    // Also copies terminator.
    fbstring_detail::podCopy(
        ml_.data_, ml_.data_ + ml_.size_ + 1, nascent.ml_.data_);
    nascent.swap(*this);
    assert(capacity() >= minCapacity);
  }
}

template <class Char>
FOLLY_NOINLINE void fbstring_core<Char>::reserveSmall(
    size_t minCapacity, const bool disableSSO) {
  assert(category() == Category::isSmall);
  if (!disableSSO && minCapacity <= maxSmallSize) {
    // small
    // Nothing to do, everything stays put
  } else if (minCapacity <= maxMediumSize) {
    // medium
    // Don't forget to allocate one extra Char for the terminating null
    auto const allocSizeBytes =
        goodMallocSize((1 + minCapacity) * sizeof(Char));
    auto const pData = static_cast<Char*>(checkedMalloc(allocSizeBytes));
    auto const size = smallSize();
    // Also copies terminator.
    fbstring_detail::podCopy(small_, small_ + size + 1, pData);
    ml_.data_ = pData;
    ml_.size_ = size;
    ml_.setCapacity(allocSizeBytes / sizeof(Char) - 1, Category::isMedium);
  } else {
    // large
    auto const newRC = RefCounted::create(&minCapacity);
    auto const size = smallSize();
    // Also copies terminator.
    fbstring_detail::podCopy(small_, small_ + size + 1, newRC->data_);
    ml_.data_ = newRC->data_;
    ml_.size_ = size;
    ml_.setCapacity(minCapacity, Category::isLarge);
    assert(capacity() >= minCapacity);
  }
}

template <class Char>
inline Char* fbstring_core<Char>::expandNoinit(
    const size_t delta,
    bool expGrowth, /* = false */
    bool disableSSO /* = FBSTRING_DISABLE_SSO */) {
  // Strategy is simple: make room, then change size
  assert(capacity() >= size());
  size_t sz, newSz;
  if (category() == Category::isSmall) {
    sz = smallSize();
    newSz = sz + delta;
    if (!disableSSO && FOLLY_LIKELY(newSz <= maxSmallSize)) {
      setSmallSize(newSz);
      return small_ + sz;
    }
    reserveSmall(
        expGrowth ? std::max(newSz, 2 * maxSmallSize) : newSz, disableSSO);
  } else {
    sz = ml_.size_;
    newSz = sz + delta;
    if (FOLLY_UNLIKELY(newSz > capacity())) {
      // ensures not shared
      reserve(expGrowth ? std::max(newSz, 1 + capacity() * 3 / 2) : newSz);
    }
  }
  assert(capacity() >= newSz);
  // Category can't be small - we took care of that above
  assert(category() == Category::isMedium || category() == Category::isLarge);
  ml_.size_ = newSz;
  ml_.data_[newSz] = '\0';
  assert(size() == newSz);
  return ml_.data_ + sz;
}

template <class Char>
inline void fbstring_core<Char>::shrinkSmall(const size_t delta) {
  // Check for underflow
  assert(delta <= smallSize());
  setSmallSize(smallSize() - delta);
}

template <class Char>
inline void fbstring_core<Char>::shrinkMedium(const size_t delta) {
  // Medium strings and unique large strings need no special
  // handling.
  assert(ml_.size_ >= delta);
  ml_.size_ -= delta;
  ml_.data_[ml_.size_] = '\0';
}

template <class Char>
inline void fbstring_core<Char>::shrinkLarge(const size_t delta) {
  assert(ml_.size_ >= delta);
  // Shared large string, must make unique. This is because of the
  // durn terminator must be written, which may trample the shared
  // data.
  if (delta) {
    fbstring_core(ml_.data_, ml_.size_ - delta).swap(*this);
  }
  // No need to write the terminator.
}


