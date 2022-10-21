opus使用笔记
=======================
Opus 是一个完全开放、免版税、用途广泛的音频编解码器。Opus 在互联网上的交互式语音和音乐传输方面无可匹敌，但也适用于存储和流媒体应用。
它由 Internet 工程任务组 (IETF) 标准化为RFC 6716 ，它结合了 Skype 的 SILK 编解码器和 Xiph.Org 的 CELT 编解码器的技术。


编译
----------------
.. code-block:: shell

    git clone https://gitlab.xiph.org/xiph/opus.git
    cd opus
    mkdir build
    cd build
    cmake -DOPUS_BUILD_TESTING=ON -DOPUS_BUILD_PROGRAMS=ON ..
    make 

opus_demo使用
---------------------
::

    ./opus_demo --help
    Usage: ./opus_demo [-e] <application> <采样率> <通道数> <比特率>  [options] <input> <output>
        ./opus_demo -d <采样率> <通道数> [options] <input> <output>

    application: voip | audio | restricted-lowdelay
    options:
    -e                   : only runs the encoder (output the bit-stream)
    -d                   : only runs the decoder (reads the bit-stream as input)
    -cbr                 : 启用恒定比特率; 默认值：可变比特率
    -cvbr                : 启用约束变量比特率; 默认值：不受约束
    -delayed-decision    : 使用前瞻语音/音乐检测（仅限专家）; 默认值：禁用
    -bandwidth <NB|MB|WB|SWB|FB> : 音频带宽（从窄带到全频带）; 默认值：采样率
    -framesize <2.5|5|10|20|40|60|80|100|120> : 帧长，单位为毫秒; 默认值：20
    -max_payload <bytes> : 最大有效负载大小（字节），默认值：1024
    -complexity <comp>   : 复杂度，0（最低）... 10（最高）; 默认值：10
    -inbandfec           : 启用SILK内置FEC
    -forcemono           : 强制单声道编码，即使是立体声输入
    -dtx                 : 启用SILK DTX
    -loss <perc>         : 模拟丢包，百分比（0-100）; 默认值：0

.. code-block:: shell

    # 编码示例
    ./opus_demo -e voip 48000 2 128000 xxx.pcm xxx.opus
    # 解码示例
    ./opus_demo -d 48000 2 xxx.opus xxx.pcm


opus接口说明
-------------------
参考：https://opus-codec.org/docs/opus_api-1.3.1/index.html

编码器
`````````````````
OpusEncoder
::::::::::::::::::
Opus编码器状态

opus_encode
:::::::::::::::::::
.. code-block:: cpp
    
    opus_int32 opus_encode	(	OpusEncoder * 	st,
        const opus_int16 * 	pcm,  
        int 	frame_size, 
        unsigned char * 	data,
        opus_int32 	max_data_bytes 
        )	
    // pcm:输入信号(双通道交错存放)，长度为frame_size*channels*sizeof(opus_int16)
    // frame_size:输入信号中每个通道的样本数.这必须是编码器采样率的 Opus 帧大小。
    //            例如,在48kHz时，允许的值为 120(2.5ms)、240、480、960(20ms)、1920 和 2880(120ms)。
    //            传入少于 10 ms 的持续时间（48 kHz 时的 480 个样本）将阻止编码器使用 LPC 或混合模式.
    //            建议使用20ms时的值。
    // data:编码输出地址，必须包含至少max_data_bytes的存储空间
    // max_data_bytes:编码输出内存空间大小
    // 返回值：成功时编码数据包的长度（以字节为单位）或失败时的负错误代码

opus_encode_float
:::::::::::::::::::::::
编码的float版本，用法和opus_encode一样

.. code-block:: cpp

    
    opus_int32 opus_encode_float(OpusEncoder *st,
            const float *pcm,
            int 	frame_size,
            unsigned char * 	data,
            opus_int32 	max_data_bytes 
            )	

opus_encoder_create
::::::::::::::::::::::::::
申请encoder state空间，并初始化encoder state

.. code-block:: cpp

    OpusEncoder* opus_encoder_create(opus_int32 Fs,
        int 	channels,
        int 	application,
        int * 	error 
        )
    // Fs:采样率，支持8000, 12000, 16000, 24000,48000	
    // channels: 通道数，支持1，2
    // application:编码模式，支持OPUS_APPLICATION_VOIP/OPUS_APPLICATION_AUDIO/OPUS_APPLICATION_RESTRICTED_LOWDELAY
    // error: 错误码
    // 返回值：编码器状态指针

opus_encoder_ctl
:::::::::::::::::::::::
对编码器进行设置

.. code-block:: cpp

    int opus_encoder_ctl(OpusEncoder * 	st,
        int 	request,
            ... 
        )
    // st: 编码器状态
    // request:此参数和所有剩余参数应替换为通用 CTL或编码器相关 CTL中的便利宏之一。

opus_encoder_destroy
::::::::::::::::::::::::::::::
释放opus_encoder_create申请的内存

.. code-block:: cpp

    void opus_encoder_destroy(OpusEncoder * st)	
    // st:编码器状态

opus_encoder_get_size
::::::::::::::::::::::::::::::::
获取OpusEncoder结构体大小

.. code-block:: cpp

    int opus_encoder_get_size(int channels)
    // channels:通道数，支持0，1
    // 返回值：编码器状态需要的字节数

opus_encoder_init
:::::::::::::::::::::::::::::
使用已经申请的内存，初始化编码器状态，已经申请的内存大小必须大于opus_encoder_get_size返回的大小。
使用与自主管理内存申请和释放的应用

.. code-block:: cpp

    int opus_encoder_init(OpusEncoder * st,
        opus_int32 	Fs,
        int 	channels,
        int 	application)	
    // Fs:采样率，支持8000, 12000, 16000, 24000,48000	
    // channels: 通道数，支持1，2
    // application:编码模式，支持OPUS_APPLICATION_VOIP/OPUS_APPLICATION_AUDIO/OPUS_APPLICATION_RESTRICTED_LOWDELAY
    // error: 错误码
    // 返回值：成功返回OPUS_OK,失败返回错误码


解码器
``````````````````````
OpusDecoder
:::::::::::::::::::
解码器状态

opus_decode
:::::::::::::::::::::
解码opus数据包

.. code-block:: cpp

    int opus_decode(OpusDecoder *st,
        const unsigned char *data,
        opus_int32 	len,
        opus_int16 *pcm,
        int frame_size,
        int decode_fec)
    // st:解码器状态
    // data:数据包地址，如果为null，表示数据包丢失
    // pcm:音频数据(双通道时交替存储)，长度为frame_size*channels*sizeof(opus_int16)
    // frame_size:pcm数据中每个通道的样本数。如果这小于最大数据包持续时间（120毫秒；48kHz为5760），
    //            此函数将无法解码某些数据包。在 PLC (data==NULL) 或 FEC (decode_fec=1) 的情况下，
    //            frame_size 需要正好是丢失音频的持续时间，否则解码器将不会处于解码下一个传入数据包的最佳状态。
    //            对于 PLC 和 FEC 情况，frame_size必须是 2.5 ms 的倍数。
    // decode_fec:FLAG（0 或 1），0:单帧解码；1：使用带内前向纠错数据进行解码
    // 返回值：解码出来的音频个数

opus_decode_float
:::::::::::::::::::::::::::
同opus_decode，只是解码出来的是float数据

.. code-block:: cpp
 
    int opus_decode_float(OpusDecoder *st,
        const unsigned char *data,
        opus_int32 	len,
        float *pcm,
        int 	frame_size,
        int 	decode_fec)


opus_decoder_create
:::::::::::::::::::::::::::::::
申请内存，并初始化解码状态

.. code-block:: cpp
   
    OpusDecoder* opus_decoder_create(opus_int32 Fs,
        int 	channels,
        int * 	error)
    // Fs:采样率，支持8000, 12000, 16000, 24000,48000
    // channels:通道数，支持1，2
    // error :成功设置OPUS_OK ，失败返回错误码
    // 返回值：初始化后的解码状态指针

opus_decoder_ctl
::::::::::::::::::::::::::
获取/设置解码状态

.. code-block:: cpp
  
    int opus_decoder_ctl(OpusDecoder *st,
        int 	request,
 	    ...)
    // st:解码状态
    // request:此参数和所有剩余参数应替换为通用 CTL或编码器相关 CTL中的便利宏之一。

opus_decoder_destroy
:::::::::::::::::::::::::::::::
释放解码器状态的内存

.. code-block:: cpp

    void opus_decoder_destroy(OpusDecoder *st)
    
opus_decoder_get_nb_samples
::::::::::::::::::::::::::::::::::::::
获取opus数据包中样本数

.. code-block:: cpp

    int opus_decoder_get_nb_samples(const OpusDecoder *dec,
        const unsigned char 	packet[],
        opus_int32 	len)
    // dec:解码状态
    // packet:char* opus数据包
    // len:数据包长度
    // 返回值：数据包中样本数

opus_decoder_get_size
::::::::::::::::::::::::::::
获取解码状态需要的内存空间大小

.. code-block:: cpp

    int opus_decoder_get_size(int channels)
    // channels：通道数，支持1，2
    // 返回值：解码器需要的字节数

opus_decoder_init
:::::::::::::::::::::::::::::
使用已申请的内存初始化解码器，已申请的内存必须大于opus_decoder_get_size返回的大小。
适用于需要自己管理内存的应用

.. code-block:: cpp
  
    int opus_decoder_init(OpusDecoder *st,
        opus_int32 	Fs,
        int 	channels)
    // st:解码器状态
    // Fs:采样率，支持8000, 12000, 16000, 24000,48000
    // channels：通道数，支持1,2
    // 返回值：成功返回OPUS_OK，失败返回错误码

opus_packet_get_bandwidth
::::::::::::::::::::::::::::::::
获取opus数据包的带宽

.. code-block:: cpp

    int opus_packet_get_bandwidth(const unsigned char *data)
    // data:opus数据包地址
    // 返回值：OPUS_BANDWIDTH_NARROWBAND	Narrowband (4kHz bandpass)
    //        OPUS_BANDWIDTH_MEDIUMBAND	Mediumband (6kHz bandpass)
    //        OPUS_BANDWIDTH_WIDEBAND	Wideband (8kHz bandpass)
    //        OPUS_BANDWIDTH_SUPERWIDEBAND	Superwideband (12kHz bandpass)
    //        OPUS_BANDWIDTH_FULLBAND	Fullband (20kHz bandpass)
    //        OPUS_INVALID_PACKET	传递的压缩数据已损坏或类型不受支持

opus_packet_get_nb_channels
::::::::::::::::::::::::::::::::::::
获取opus数据包的通道数

.. code-block:: cpp

    int opus_packet_get_nb_channels(const unsigned char *data)
    // data:opus数据包地址
    // 返回值：通道数；OPUS_INVALID_PACKET	传递的压缩数据已损坏或类型不受支持

opus_packet_get_nb_samples
::::::::::::::::::::::::::::::::::
获取 Opus 数据包的样本数。

.. code-block:: cpp
    
    int opus_packet_get_nb_samples(const unsigned char packet[],
        opus_int32 	len,
        opus_int32 	Fs)
    // packet:(char *)  opus数据包地址
    // len:opus数据包长度
    // Fs:采样率，必须是400的倍数，否则返回的结果不准确
    // 返回值：数据包中的样本数

opus_packet_get_samples_per_frame
::::::::::::::::::::::::::::::::::::::::::::::
获取opus数据包中每帧的样本数

.. code-block:: cpp

    int opus_packet_get_samples_per_frame(const unsigned char *data,
        opus_int32 	Fs)
    // data:opus数据包地址
    // Fs:采样率，必须是400的倍数，否则返回的结果不准确
    // 返回值：每帧的样本数

opus_packet_parse
::::::::::::::::::::::::::::
将 opus 数据包解析为一个或多个帧。
Opus_decode 将在内部执行此操作，因此大多数应用程序不需要使用此函数。此函数不复制帧，返回的指针是指向输入数据包的指针。

.. code-block:: cpp

    int opus_packet_parse(const unsigned char *data,
        opus_int32 	len,
        unsigned char * 	out_toc,
        const unsigned char * 	frames[48],
        opus_int16 	size[48],
        int * 	payload_offset)
    // data: 待解析的opus数据包	
    // len:data的长度
    // out_toc:目录指针
    // frames:封装帧
    // size:封装帧的大小
    // payload_offset:返回数据包中有效负载的位置（以字节为单位）
    // 返回值：帧数

opus_pcm_soft_clip
::::::::::::::::::::::::::::::
应用软削波将浮点信号置于 [-1,1] 范围内。
如果信号已经在该范围内，则什么也不做。
如果存在 [-1,1] 之外的值，则信号会被尽可能平滑地削波，以适应范围并避免在处理过程中产生过度失真。

.. code-block:: cpp

    void opus_pcm_soft_clip(float *pcm,
        int 	frame_size,
        int 	channels,
        float * 	softclip_mem)
    // pcm:输入的pcm数据，平滑过程中会修改该数据
    // frame_size:每个通道要处理的样本数
    // channels:通道数
    // softclip_mem:用于软削波过程的状态存储器（每个通道一个浮点数，初始化为零）






