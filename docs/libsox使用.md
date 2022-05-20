# libsox使用笔记

## 数据结构

`sox_sample_t`

```cpp
typedef sox_int32_t sox_sample_t;
```

`sox_format_t`

语音文件对应的数据结构

```cpp
struct sox_format_t {
  char             * filename;      /**< File name */

  /**语音信号格式，包括:采样率，通道数，位宽等**/
  sox_signalinfo_t signal;

  /**  语音信号编码格式，包括:编码格式（wav,ulaw等），采样精度，压缩率，大小端等 **/
  sox_encodinginfo_t encoding;

  char             * filetype;      /**< Type of file, as determined by header inspection or libmagic. */
  sox_oob_t        oob;             /**< comments, instrument info, loop info (out-of-band data) */
  sox_bool         seekable;        /**< Can seek on this file */
  char             mode;            /**< Read or write mode ('r' or 'w') */
  sox_uint64_t     olength;         /**< Samples * chans written to file */
  sox_uint64_t     clips;           /**< Incremented if clipping occurs */
  int              sox_errno;       /**< Failure error code */
  char             sox_errstr[256]; /**< Failure error text */
  void             * fp;            /**< File stream pointer */
  lsx_io_type      io_type;         /**< Stores whether this is a file, pipe or URL */
  sox_uint64_t     tell_off;        /**< Current offset within file */
  sox_uint64_t     data_start;      /**< Offset at which headers end and sound data begins (set by lsx_check_read_params) */
  sox_format_handler_t handler;     /**< Format handler for this file */
  void             * priv;          /**< Format handler's private data area */
};
```

`sox_signalinfo_t`

语音信号对应的信息

```cpp
typedef struct sox_signalinfo_t {                                                                                                                                                  
  sox_rate_t       rate;         /**采样率, 0 if unknown */
  unsigned         channels;     /**通道数, 0 if unknown */
  unsigned         precision;    /**采样点位宽, 0 if unknown */
  sox_uint64_t     length;       /**音频总采样点数, 0 if unknown, -1 if unspecified */
  double           * mult;       /**< Effects headroom multiplier; may be null */
} sox_signalinfo_t;
```

`sox_encodinginfo_t`

```cpp
typedef struct sox_encodinginfo_t {
  sox_encoding_t encoding; /**< format of sample numbers */
  unsigned bits_per_sample;/**< 0 if unknown or variable; uncompressed value if lossless; compressed value if lossy */
  double compression;      /**< compression factor (where applicable) */

  sox_option_t reverse_bytes;
  sox_option_t reverse_nibbles;
  sox_option_t reverse_bits;
  sox_bool opposite_endian;
} sox_encodinginfo_t;
```

`sox_effect_t`

effect表示一次语音转换，其中handler表示语音处理函数的封装。

```cpp
struct sox_effect_t { 
  sox_effects_globals_t    * global_info; /**< global effect parameters */
  sox_signalinfo_t         in_signal;     /**< Information about the incoming data stream */
  sox_signalinfo_t         out_signal;    /**< Information about the outgoing data stream */
  sox_encodinginfo_t       const * in_encoding;  /**< Information about the incoming data encoding */
  sox_encodinginfo_t       const * out_encoding; /**< Information about the outgoing data encoding */
  sox_effect_handler_t     handler;   /**< The handler for this effect */
  sox_uint64_t         clips;         /**< increment if clipping occurs */
  size_t               flows;         /**< 1 if MCHAN, number of chans otherwise */
  size_t               flow;          /**< flow number */
  void                 * priv;        /**< Effect's private data area (each flow has a separate copy) */
  /* The following items are private to the libSoX effects chain functions. */
  sox_sample_t             * obuf;    /**< output buffer */
  size_t                   obeg;      /**< output buffer: start of valid data section */
  size_t                   oend;      /**< output buffer: one past valid data section (oend-obeg is length of current content) */
  size_t               imin;          /**< minimum input buffer content required for calling this effect's flow function; set via lsx_effect_set_imin() */
};
```

`sox_effects_chain_t`

chain表示语音处理的一条链路，包含多个effect(效果转换)

```cpp
typedef struct sox_effects_chain_t {                                                                                                                                               
  sox_effect_t **effects;                  /**< Table of effects to be applied to a stream */
  size_t length;                           /**< Number of effects to be applied */
  sox_effects_globals_t global_info;       /**< Copy of global effects settings */
  sox_encodinginfo_t const * in_enc;       /**< Input encoding */
  sox_encodinginfo_t const * out_enc;      /**< Output encoding */
  /* The following items are private to the libSoX effects chain functions. */
  size_t table_size;                       /**< Size of effects table (including unused entries) */
  sox_sample_t *il_buf;                    /**< Channel interleave buffer */
} sox_effects_chain_t;
```

`sox_effect_handler_t`

```cpp
struct sox_effect_handler_t {                                                                                                                                                      
  char const * name;  /**< Effect name */
  char const * usage; /**< Short explanation of parameters accepted by effect */
  unsigned int flags; /**< Combination of SOX_EFF_* flags */
  sox_effect_handler_getopts getopts; /**< Called to parse command-line arguments (called once per effect). */
  sox_effect_handler_start start;     /**< Called to initialize effect (called once per flow). */
  sox_effect_handler_flow flow;       /**< Called to process samples. */
  sox_effect_handler_drain drain;     /**< Called to finish getting output after input is complete. */
  sox_effect_handler_stop stop;       /**< Called to shut down effect (called once per flow). */
  sox_effect_handler_kill kill;       /**< Called to shut down effect (called once per effect). */
  size_t       priv_size;             /**< Size of private data SoX should pre-allocate for effect */
};
```

## 函数说明

`sox_open_read`

```cpp
sox_format_t *
sox_open_read(                                                                                                                                                                
     LSX_PARAM_IN_Z   char               const * path,      /**文件路径*/
     LSX_PARAM_IN_OPT sox_signalinfo_t   const * signal,    /**语音信息，包括采样率等, or NULL if none. */
     LSX_PARAM_IN_OPT sox_encodinginfo_t const * encoding,  /**音频数据编码格式, 自动识别时填NULL. */
     LSX_PARAM_IN_OPT_Z char             const * filetype   /**文件类型, 自动识别时填NULL. */
     );
```

`sox_open_mem_read`

```cpp
sox_format_t *
sox_open_mem_read(
    LSX_PARAM_IN_BYTECOUNT(buffer_size) void  * buffer,     /**< Pointer to audio data buffer (required). */
    size_t                                      buffer_size,/**< Number of bytes to read from audio data buffer. */
    LSX_PARAM_IN_OPT sox_signalinfo_t   const * signal,     /**< Information already known about audio stream, or NULL if none. */
    LSX_PARAM_IN_OPT sox_encodinginfo_t const * encoding,   /**< Information already known about sample encoding, or NULL if none. */
    LSX_PARAM_IN_OPT_Z char             const * filetype    /**< Previously-determined file type, or NULL to auto-detect. */
    );
```

`sox_read`

```cpp
// 返回读取到的sample数，0表示读取到文件结尾
size_t sox_read( 
    LSX_PARAM_INOUT sox_format_t * ft, /**< Format pointer. */
    LSX_PARAM_OUT_CAP_POST_COUNT(len,return) sox_sample_t *buf, /**< Buffer from which to read samples. */
    size_t len /**< Number of samples available in buf. */
    );
```

`sox_open_write`

```cpp
sox_format_t *
sox_open_write(
    LSX_PARAM_IN_Z     char               const * path,     /**文件路径 */
    LSX_PARAM_IN       sox_signalinfo_t   const * signal,   /**输出语音信息 (required). */
    LSX_PARAM_IN_OPT   sox_encodinginfo_t const * encoding, /**输出语音编码格式, or NULL to use defaults. */
    LSX_PARAM_IN_OPT_Z char               const * filetype, /**< Previously-determined file type, or NULL to auto-detect. */
    LSX_PARAM_IN_OPT   sox_oob_t          const * oob,      /**< Out-of-band data to add to file, or NULL if none. */
    LSX_PARAM_IN_OPT   sox_bool           (LSX_CALLBACK * overwrite_permitted)(LSX_PARAM_IN_Z char const * filename) 
    );
```

`sox_open_mem_write`

```cpp
sox_format_t *
sox_open_mem_write(
    LSX_PARAM_OUT_BYTECAP(buffer_size) void                     * buffer,      /**< Pointer to audio data buffer that receives data (required). */
    LSX_PARAM_IN                       size_t                     buffer_size, /**< Maximum number of bytes to write to audio data buffer. */
    LSX_PARAM_IN                       sox_signalinfo_t   const * signal,      /**< Information about desired audio stream (required). */
    LSX_PARAM_IN_OPT                   sox_encodinginfo_t const * encoding,    /**< Information about desired sample encoding, or NULL to use defaults.*/
    LSX_PARAM_IN_OPT_Z                 char               const * filetype,    /**< Previously-determined file type, or NULL to auto-detect. */
    LSX_PARAM_IN_OPT                   sox_oob_t          const * oob          /**< Out-of-band data to add to file, or NULL if none. */
    );
```

`sox_open_memstream_write`

```cpp
sox_format_t *
sox_open_memstream_write(
    LSX_PARAM_OUT      char                     * * buffer_ptr,    /**< Receives pointer to audio data buffer that receives data (required). */
    LSX_PARAM_OUT      size_t                   * buffer_size_ptr, /**< Receives size of data written to audio data buffer (required). */
    LSX_PARAM_IN       sox_signalinfo_t   const * signal,          /**< Information about desired audio stream (required). */
    LSX_PARAM_IN_OPT   sox_encodinginfo_t const * encoding,        /**< Information about desired sample encoding, or NULL to use defaults. */
    LSX_PARAM_IN_OPT_Z char               const * filetype,        /**< Previously-determined file type, or NULL to auto-detect. */
    LSX_PARAM_IN_OPT   sox_oob_t          const * oob              /**< Out-of-band data to add to file, or NULL if none. */
    );
```

`sox_write`

```cpp
size_t
sox_write(
    LSX_PARAM_INOUT sox_format_t * ft, /**< Format pointer. */
    LSX_PARAM_IN_COUNT(len) sox_sample_t const * buf, /**< Buffer from which to read samples. */
    size_t len /**< Number of samples available in buf. */
    );
```

`sox_create_effects_chain`

```cpp
sox_effects_chain_t *
LSX_API
sox_create_effects_chain(
    LSX_PARAM_IN sox_encodinginfo_t const * in_enc, /**< Input encoding. */
    LSX_PARAM_IN sox_encodinginfo_t const * out_enc /**< Output encoding. */
    );
```

## 使用示例

`参考https://github.com/dmkrepo/libsox/blob/master/src/example6.c`

### 1. 读取音频文件

```cpp
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "usage: audio_test audio_path" << std::endl;
    exit(1);
  }
  sox_format_t *in;

  /* All libSoX applications must start by initialising the SoX library */
  assert(sox_init() == SOX_SUCCESS);
  assert(in = sox_open_read(argv[1], NULL, NULL, NULL));
  int sample_rate = in->signal.rate;
  int channels = in->signal.channels;
  int64_t sample_num = in->signal.length;
  std::cout<<"sample_rate:"<<sample_rate<<" channels:"<<channels<<" sample_num:"<<sample_num<<std::endl;
  std::vector<sox_sample_t> audio_data(sample_num);
  std::vector<int16_t>  audio_int16(sample_num);
  assert(sox_read(in, audio_data.data(), sample_num)==sample_num);
  for(size_t i=0;i<audio_data.size();i++){
      //std::cout<<SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i],)<<" ";
      //std::cout<<SOX_SAMPLE_TO_SIGNED_32BIT(audio_data[i],)<<" ";
      int16_t d = SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i],)*(std::numeric_limits<int16_t>::max()+1.0);
      audio_int16[i] = d;
  }
  sox_close(in);
  sox_quit();
}
```

### 2. 数组保存为wav

```cpp
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "usage: audio_test audio_path" << std::endl;
    exit(1);
  }
  sox_format_t *in,*out;

  /* All libSoX applications must start by initialising the SoX library */
  assert(sox_init() == SOX_SUCCESS);
  assert(in = sox_open_read(argv[1], NULL, NULL, NULL));
  int sample_rate = in->signal.rate;
  int channels = in->signal.channels;
  int64_t sample_num = in->signal.length;
  std::cout<<"sample_rate:"<<sample_rate<<" channels:"<<channels<<" sample_num:"<<sample_num<<std::endl;
  std::vector<sox_sample_t> audio_data(sample_num);
  std::vector<int16_t>  audio_int16(sample_num);
  assert(sox_read(in, audio_data.data(), sample_num)==sample_num);
  for(size_t i=0;i<audio_data.size();i++){
      //std::cout<<SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i],)<<" ";
      //std::cout<<SOX_SAMPLE_TO_SIGNED_32BIT(audio_data[i],)<<" ";
      int16_t d = SOX_SAMPLE_TO_FLOAT_64BIT(audio_data[i],)*(std::numeric_limits<int16_t>::max()+1.0);
      audio_int16[i] = d;
  }
  // 将数组内容写入到wav
  sox_signalinfo_t out_signal = {
    in->signal.rate,
    in->signal.channels,
    16,
    0,
    NULL
  };

  assert(out = sox_open_write("out.wav", &out_signal, NULL, NULL, NULL, NULL));
  for(size_t i=0;i<audio_data.size();i++){
    audio_data[i] = SOX_SIGNED_16BIT_TO_SAMPLE(audio_int16[i],);
  }
  assert(sox_write(out, audio_data.data(), audio_data.size()) == audio_data.size());
  sox_close(out);
  sox_close(in);
  sox_quit();

  return 0;
}
```

### 3. 从buff中读取数据，修改采样率后写入文件

```cpp
#define buffer_size 12345678
static char buffer[buffer_size];

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "usage: audio_test audio_path" << std::endl;
    exit(1);
  }
  static sox_format_t * in, * in1,* out,*out1; /* input and output files */
  sox_effects_chain_t * chain;
  sox_effect_t * e;
  char * args[10];
  sox_signalinfo_t interm_signal; /* @ intermediate points in the chain. */
  sox_encodinginfo_t out_encoding = {
    SOX_ENCODING_SIGN2,
    16,
    0,
    sox_option_default,
    sox_option_default,
    sox_option_default,
    sox_false
  };
  sox_signalinfo_t out_signal = {
    8000,
    1,
    0,
    0,
    NULL
  };
  assert(sox_init() == SOX_SUCCESS);
  assert(in = sox_open_read(argv[1], NULL, NULL, NULL));
  std::vector<sox_sample_t> audio_data(in->signal.length);
  assert(sox_read(in, audio_data.data(), in->signal.length)==in->signal.length);
  assert(out1 = sox_open_mem_write(buffer, buffer_size, &in->signal, NULL, "sox", NULL));
  assert(sox_write(out1, audio_data.data(), audio_data.size()) == audio_data.size());
  sox_close(in);
  sox_close(out1);

  assert(in1 = sox_open_mem_read(buffer, buffer_size, NULL, NULL, NULL));

  assert(out = sox_open_write("out.wav", &out_signal, &out_encoding, NULL, NULL, NULL));

  chain = sox_create_effects_chain(&in->encoding, &out->encoding);

  interm_signal = in1->signal; /* NB: deep copy */

  e = sox_create_effect(sox_find_effect("input"));
  args[0] = (char *)in1, assert(sox_effect_options(e, 1, args) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &interm_signal, &in1->signal) == SOX_SUCCESS);
  free(e);

  if (in1->signal.rate != out->signal.rate) {
    e = sox_create_effect(sox_find_effect("rate"));
    assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &out->signal) == SOX_SUCCESS);
    free(e);
  }

  if (in1->signal.channels != out->signal.channels) {
    e = sox_create_effect(sox_find_effect("channels"));
    assert(sox_effect_options(e, 0, NULL) == SOX_SUCCESS);
    assert(sox_add_effect(chain, e, &interm_signal, &out->signal) == SOX_SUCCESS);
    free(e);
  }

  e = sox_create_effect(sox_find_effect("output"));
  args[0] = (char *)out, assert(sox_effect_options(e, 1, args) == SOX_SUCCESS);
  assert(sox_add_effect(chain, e, &interm_signal, &out->signal) == SOX_SUCCESS);
  free(e);

  sox_flow_effects(chain, NULL, NULL);

  sox_delete_effects_chain(chain);
  sox_close(out);
  sox_close(in1);
  sox_quit();

  return 0;
}
```