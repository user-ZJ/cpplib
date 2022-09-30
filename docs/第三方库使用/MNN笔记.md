# MNN笔记

### Pipeline

一个session中可能包含多个pipeline，每个pipeline包含多个计算单元

* ErrorCode encode(bool isStatic = false, bool supportDebug = false);

  1. 计算所有op的输入shape和输出shape
  2. 几何变换
  3. 拷贝op和输入/输出向量信息到mBuffer

* ErrorCode allocMemory();

  创建执行图，并为每个op分配内存

* ErrorCode execute();

  执行pipeline

* ErrorCode executeCallBack(const TensorCallBackWithInfo& before, const TensorCallBackWithInfo& after);

  在执行前后做一些其他处理

* std::vector<Schedule::PipelineInfo>& getPipelineInfo();

  获取pipeline的详细信息

* void cloneExecution(const std::map<const Op*, std::shared_ptr<Execution>>& cache);

  从cache中拷贝执行图

* const std::map<const Op*, std::shared_ptr<Execution>>& getCache()

  获取执行图的缓存

### Schedule

执行图信息，包括Type，PipelineInfo，ScheduleInfo，

```cpp
class MNN_PUBLIC Schedule {
public:
    enum Type {
        // Size can be compute seperately
        SEPERATE = 0,
        // When size is fixed, the content is fixed
        CONSTANT = 1,
        // Size can't be compute seperately
        NOT_SEPERATE
    };
    /** pipeline info */
    struct PipelineInfo {
        /** op */
        const Op* op;
        /** input tensors */
        std::vector<Tensor*> inputs;
        /** output tensors */
        std::vector<Tensor*> outputs;
        /** schedule type*/
        Schedule::Type type = Schedule::Type::SEPERATE;
    };

    /** schedule info */
    struct ScheduleInfo {
        /** pipelines with backend info */
        std::vector<std::pair<Backend::Info, std::vector<PipelineInfo>>> pipelineInfo;
        /** input tensors map */
        std::map<std::string, Tensor*> inputTensors;
        /** output tensors map */
        std::map<std::string, Tensor*> outputTensor;
        /** all tensors map */
        std::vector<std::pair<int, std::shared_ptr<Tensor>>> allTensors;
        /** input valid for resize*/
        bool validForResize;
    };

    /**
     * @breif schedule net ops to pipeline with configuration.
     * @param net       given net.
     * @param config    given configuration.
     * @return schedule info.
     */
    static ScheduleInfo schedule(const Net* net, const std::vector<ScheduleConfig>& config);
    static MNNForwardType getApprociateType(const ScheduleConfig& config);
};
```

### MNNForwardType 

MNN推理使用的后端

```cpp
typedef enum {
    MNN_FORWARD_CPU = 0,

    /*
     Firtly find the first available backends not equal to CPU
     If no other backends, use cpu
     */
    MNN_FORWARD_AUTO = 4,

    /*Hand write metal*/
    MNN_FORWARD_METAL = 1,

    /*NVIDIA GPU API*/
    MNN_FORWARD_CUDA = 2,

    /*Android / Common Device GPU API*/
    MNN_FORWARD_OPENCL = 3,
    MNN_FORWARD_OPENGL = 6,
    MNN_FORWARD_VULKAN = 7,

    /*Android 8.1's NNAPI, Not Support yet*/
    MNN_FORWARD_NN = 5,

    /*User can use API from Backend.hpp to add or search Backend*/
    MNN_FORWARD_USER_0 = 8,
    MNN_FORWARD_USER_1 = 9,
    MNN_FORWARD_USER_2 = 10,
    MNN_FORWARD_USER_3 = 11,

    MNN_FORWARD_ALL,

    /* Apply arm extension instruction set to accelerate some Ops, this forward type
       is only used in MNN internal, and will be active automatically when user set forward type
       to be MNN_FORWARD_CPU and extension instruction set is valid on hardware.
    */
    MNN_FORWARD_CPU_EXTENSION

} MNNForwardType;
```

### CommandBuffer

```cpp
struct Command {
    const Op* op;    //操作
    std::vector<Tensor*> inputs;   //输入的tensor
    std::vector<Tensor*> outputs;  //输出的tensor
    std::vector<uint8_t> buffer; // storage for op
    std::string name;    //command名称
};
struct CommandBuffer {
    std::vector<Command> command;   
    std::vector<std::shared_ptr<Tensor>> extras;    
};
```

### Op

算子/操作

* const flatbuffers::String *name() const

  算子的名称

* OpType type() const 

  算子的类型

  ```cpp
  enum OpType {
    OpType_AbsVal = 0,
    OpType_QuantizedAdd = 1,
    OpType_ArgMax = 2,
    OpType_AsString = 3,
    OpType_InstanceNorm = 4,
    OpType_BatchToSpaceND = 5,
    OpType_Bias = 6,
    OpType_BinaryOp = 7,
    OpType_Bnll = 8,
    OpType_Cast = 9,
    OpType_Concat = 10,
    OpType_Const = 11,
    OpType_Convolution = 12,
    OpType_ConvolutionDepthwise = 13,
    OpType_Crop = 14,
    OpType_CropAndResize = 15,
    OpType_Cubic = 16,
    OpType_Deconvolution = 17,
    OpType_DeconvolutionDepthwise = 18,
    OpType_Dequantize = 19,
    OpType_DetectionOutput = 20,
    OpType_Dropout = 21,
    OpType_Eltwise = 22,
    OpType_ELU = 23,
    OpType_Embed = 24,
    OpType_Exp = 25,
    OpType_ExpandDims = 26,
    OpType_Fill = 27,
    OpType_Flatten = 28,
    OpType_FloorMod = 29,
    OpType_Gather = 30,
    OpType_GatherV2 = 31,
    OpType_Im2Seq = 32,
    OpType_InnerProduct = 33,
    OpType_Input = 34,
    OpType_Interp = 35,
    OpType_Log = 36,
    OpType_LRN = 37,
    OpType_LSTM = 38,
    OpType_MatMul = 39,
    OpType_MVN = 40,
    OpType_NonMaxSuppression = 41,
    OpType_NonMaxSuppressionV2 = 42,
    OpType_Normalize = 43,
    OpType_Pack = 44,
    OpType_Padding = 45,
    OpType_Permute = 46,
    OpType_Pooling = 47,
    OpType_Power = 48,
    OpType_PReLU = 49,
    OpType_PriorBox = 50,
    OpType_Proposal = 51,
    OpType_QuantizedAvgPool = 52,
    OpType_QuantizedBiasAdd = 53,
    OpType_QuantizedConcat = 54,
    OpType_QuantizedDepthwiseConv2D = 55,
    OpType_QuantizedLogistic = 56,
    OpType_QuantizedMatMul = 57,
    OpType_QuantizedMaxPool = 58,
    OpType_QuantizedRelu = 59,
    OpType_QuantizedRelu6 = 60,
    OpType_QuantizedReshape = 61,
    OpType_QuantizedSoftmax = 62,
    OpType_QuantizeMaxMin = 63,
    OpType_QuantizeV2 = 64,
    OpType_Range = 65,
    OpType_Rank = 66,
    OpType_ReduceJoin = 67,
    OpType_Reduction = 68,
    OpType_ReLU = 69,
    OpType_ReLU6 = 70,
    OpType_RequantizationRange = 71,
    OpType_Requantize = 72,
    OpType_Reshape = 73,
    OpType_Resize = 74,
    OpType_RNN = 75,
    OpType_ROIPooling = 76,
    OpType_Scale = 77,
    OpType_Selu = 78,
    OpType_Seq2Out = 79,
    OpType_Shape = 80,
    OpType_Sigmoid = 81,
    OpType_Size = 82,
    OpType_Slice = 83,
    OpType_SliceTf = 84,
    OpType_Softmax = 85,
    OpType_SpaceToBatchND = 86,
    OpType_SpatialProduct = 87,
    OpType_Split = 88,
    OpType_SPP = 89,
    OpType_Squeeze = 90,
    OpType_StridedSlice = 91,
    OpType_StringJoin = 92,
    OpType_StringSplit = 93,
    OpType_StringToNumber = 94,
    OpType_TanH = 95,
    OpType_TfQuantizedConv2D = 96,
    OpType_Threshold = 97,
    OpType_Tile = 98,
    OpType_TopKV2 = 99,
    OpType_Transpose = 100,
    OpType_UnaryOp = 101,
    OpType_Unpack = 102,
    OpType_Where = 103,
    OpType_Moments = 104,
    OpType_RNNSequenceGRU = 105,
    OpType_BatchMatMul = 106,
    OpType_Unsqueeze = 107,
    OpType_CosineSimilarity = 108,
    OpType_DepthToSpace = 109,
    OpType_SpaceToDepth = 110,
    OpType_ReverseSequence = 111,
    OpType_Pooling3D = 112,
    OpType_Convolution3D = 113,
    OpType_MatrixBandPart = 114,
    OpType_GatherND = 115,
    OpType_DetectionPostProcess = 116,
    OpType_UnravelIndex = 117,
    OpType_ScatterNd = 118,
    OpType_OneHot = 119,
    OpType_BroadcastTo = 120,
    OpType_Dilation2D = 121,
    OpType_Raster = 128,
    OpType_ConvertTensor = 129,
    OpType_ArgMin = 130,
    OpType_LinSpace = 131,
    OpType_RandomUniform = 132,
    OpType_TensorArray = 133,
    OpType_TensorArraySize = 134,
    OpType_TensorArrayRead = 135,
    OpType_TensorArrayWrite = 136,
    OpType_TensorArrayGather = 137,
    OpType_TensorArrayScatter = 138,
    OpType_TensorArraySplit = 139,
    OpType_TensorArrayConcat = 140,
    OpType_LSTMBlockCell = 141,
    OpType_Reverse = 142,
    OpType_Plugin = 256,
    OpType_Select = 257,
    OpType_ZerosLike = 258,
    OpType_Broastcast = 259,
    OpType_SetDiff1D = 260,
    OpType_ReluGrad = 261,
    OpType_Relu6Grad = 262,
    OpType_PoolGrad = 263,
    OpType_SoftmaxGrad = 264,
    OpType_Conv2DBackPropFilter = 265,
    OpType_TrainableParam = 266,
    OpType_BatchNorm = 267,
    OpType_ZeroGrad = 268,
    OpType_Extra = 512,
    OpType_ConvInt8 = 513,
    OpType_Int8ToFloat = 514,
    OpType_DepthwiseConvInt8 = 515,
    OpType_PoolInt8 = 516,
    OpType_FloatToInt8 = 517,
    OpType_EltwiseInt8 = 518,
    OpType_While = 600,
    OpType_If = 601,
    OpType_LayerNorm = 603,
    OpType_GridSample = 604,
    OpType_MIN = OpType_AbsVal,
    OpType_MAX = OpType_GridSample
  };
  ```

### GeometryComputer

节点计算类

MNN_PUBLIC static const GeometryComputer* search(int type);

根据算子类型获取计算过程

MNN_PUBLIC bool compute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,Context& context, CommandBuffer& cmd) const;

计算单个节点



### ScheduleConfig

session调度的配置

常用配置有：

MNNForwardType type = MNN_FORWARD_CPU;  后端类型

Path path;   推理路径包括由`**path**`的`**inputs**`到`**outputs**`途径的所有算子，在不指定时，会根据模型结构自动识别。为了节约内存，MNN会复用`**outputs**`之外的tensor内存。如果需要保留中间tensor的结果，可以使用`**saveTensors**`保留tensor结果，避免内存复用。

### backendConfig

后端配置

```cpp
struct BackendConfig {
    enum MemoryMode { Memory_Normal = 0, Memory_High, Memory_Low };

    MemoryMode memory = Memory_Normal;

    enum PowerMode { Power_Normal = 0, Power_High, Power_Low };

    PowerMode power = Power_Normal;

    //precision 为 Low 时，使用 fp16 存储与计算，计算结果与CPU计算结果有少量误差，实时性最好
    //precision 为 Normal 时，使用 fp16存储，计算时将fp16转为fp32计算，计算结果与CPU计算结果相近，实时性也较好
    //precision 为 High 时，使用 fp32 存储与计算，实时性下降，但与CPU计算结果保持一致。
    //后端 CPU  precision 为 Low 时，根据设备情况开启 FP16 或 BF16 计算
    enum PrecisionMode { Precision_Normal = 0, Precision_High, Precision_Low };

    PrecisionMode precision = Precision_Normal;

    /** user defined context */
    union {
        //sharedContext用于自定义后端，用户可以根据自身需要赋值。
        void* sharedContext = nullptr;
        size_t flags; // Valid for CPU Backend
    };
};
```

### RuntimeInfo

运行时资源管理

默认情况下，在createSession时对应create单独一个 Runtime。对于串行的一系列模型，可以先单独创建Runtime ，然后在各 Session 创建时传入，使各模型用共享同样的运行时资源（对CPU而言为线程池、内存池，对GPU而言Kernel池等）。

```cpp
//typedef std::pair<std::map<MNNForwardType, std::shared_ptr<Runtime>>, std::shared_ptr<Runtime>> RuntimeInfo;
ScheduleConfig config;
config.numberThread = 4;
auto runtimeInfo = Interpreter::createRuntime({config});
```

### SessionMode

```cpp
enum SessionMode {
        /** About CallBack, Default Session_Debug*/
        /** runSessionWithCallBack is allowed and can get internal op info*/
        Session_Debug = 0,
        /** runSessionWithCallBack is not valid and can't get any info of op in session*/
        Session_Release = 1,

        /** About input tenosr, Default Session_Input_Inside*/
        /** The input tensor is alloced by session, input data after session resized*/
        Session_Input_Inside = 2,
        /** The input tensor is alloced by user, set input data before session resize*/
        Session_Input_User = 3,
};
```

### Interpreter

解释器:是模型数据的持有者

在创建完`Session`，且不再创建`Session`或更新训练模型数据时，`Interpreter`可以通过`releaseModel`函数释放模型数据，以节省内存。

Interpreter中保存有Content* mNet指针,即Interpreter中保存网络结构及装载网络的sessions

```cpp
struct Content {
    AutoStorage<uint8_t> buffer;
    const Net* net = nullptr;
    std::vector<std::unique_ptr<Session>> sessions;
    std::map<const Tensor*, const Session*> tensorMap;
    Interpreter::SessionMode callBackMode = Interpreter::Session_Debug;
    Interpreter::SessionMode inputMode    = Interpreter::Session_Input_Inside;
    AutoStorage<uint8_t> cacheBuffer;
    size_t cacheOffset = 0;
    std::string cacheFile;
    std::mutex lock;
};
```



### Session

会话：推理数据的持有者

`Session`通过`Interpreter`创建；多个推理可以共用同一个模型，即，多个`Session`可以共用一个`Interpreter`。

### Tensor

向量，表示模型输入，输出以及中间结果

```cpp
	/** dimension type used to create tensor */
    enum DimensionType {
        /** for tensorflow net type. uses NHWC as data format. */
        TENSORFLOW,
        /** for caffe net type. uses NCHW as data format. */
        CAFFE,
        /** for caffe net type. uses NC4HW4 as data format. */
        CAFFE_C4
    };

    /** handle type */
    enum HandleDataType {
        /** default handle type */
        HANDLE_NONE = 0,
        /** string handle type */
        HANDLE_STRING = 1
    };
```

```cpp
//根据shape创建tensor
//Tensor *t = createDevice<float>({1,512,20})
template <typename T>
static Tensor* createDevice(const std::vector<int>& shape, DimensionType dimType = TENSORFLOW) {
        return createDevice(shape, halide_type_of<T>(), dimType);
    }
//通过指定数据创建tensor
template <typename T>
static Tensor* create(const std::vector<int>& shape, void* data = NULL, DimensionType dimType = TENSORFLOW) {
        return create(shape, halide_type_of<T>(), data, dimType);
    }
//通过deviceTensor创建hostTensor
static Tensor* createHostTensorFromDevice(const Tensor* deviceTensor, bool copyData = true);
//获取tensor中的数据
const halide_buffer_t& buffer() const {
        return mBuffer;
}
halide_buffer_t& buffer() {
        return mBuffer;
}
//获取维度类型TENSORFLOW，CAFFE，CAFFE_C4
DimensionType getDimensionType() const;
//获取handle type  HANDLE_NONE,HANDLE_STRING
HandleDataType getHandleDataType() const;
//获取数据类型
inline halide_type_t getType() const;
//获取tensor中的数据，并转化为对应类型的指针，如t->host<float>()
template <typename T>
T* host() const;
//获取后端类型
uint64_t deviceId() const;
//获取tensor的维度数，如1x512x20,tensor是三维的，则返回3
int dimensions() const;
//获取tensor的维度
std::vector<int> shape() const;
//获取buffer char类型大小
int size() const;
//获取buffer中元素个数
inline int elementSize() const;
//调试使用，打印出tensor中的所有数据
void print() const;
//打印出tensor的shape
void printShape() const;
```



### VAPR,Variable,Expr

**Expr**   节点，创建graph的基本单元，保存当前节点的数据和输入输出等信息。

**Variable**  变量，类似于tensor，维护数据的维度，数据存储在Expr中

**VAPR**  给Variable添加INPUT，CONSTANT，TRAINABLE标签

```cpp
typedef std::shared_ptr<Expr> EXPRP;
typedef std::weak_ptr<Expr> WeakEXPRP;
typedef std::vector<int> INTS;
typedef std::vector<VARP> VARPS;
```

```cpp
class MNN_PUBLIC VARP {
public:
    VARP() {
        // Do nothing
    }
    VARP(std::shared_ptr<Variable> c) {
        mContent = std::move(c);
    }
    VARP(Variable* c) {
        mContent.reset(c);
    }
    Variable* get() const  {
        return mContent.get();
    }
    ~ VARP() {
        // Do nothing
    }
    VARP(const VARP& var) {
        mContent = var.mContent;
    }
    VARP(VARP&& var) {
        mContent = std::move(var.mContent);
    }
    VARP operator+(VARP var) const;
    VARP operator-(VARP var) const;
    VARP operator*(VARP var) const;
    VARP operator/(VARP var) const;
    VARP mean(INTS dims) const;
    VARP sum(INTS dims) const;

    bool operator==(const VARP& var) const {
        return var.mContent == mContent;
    }
    bool operator<(const VARP& var) const {
        return mContent < var.mContent;
    }
    bool operator<=(const VARP& var) const {
        return mContent <= var.mContent;
    }
    VARP& operator=(const VARP& var) {
        mContent = var.mContent;
        return *this;
    }
    VARP& operator=(Variable* var) {
        mContent.reset(var);
        return *this;
    }
    Variable* operator->() const  {
        return mContent.get();
    }
    enum InputType {
        INPUT = 0,
        CONSTANT = 1,
        TRAINABLE = 2,
    };
    bool fix(InputType type) const;
private:
    friend class Variable;  //可以访问Variable方法
    std::shared_ptr<Variable> mContent; //Variable对象
};

class MNN_PUBLIC Variable {
public:
    struct Info {
        Dimensionformat order = NHWC;  //数据格式
        INTS dim;  //vector 数据维度
        halide_type_t type;  //数据类型
        int size;  //元素个数
        void syncSize();  //更新size大小
    };
    const std::string& name() const;   //tensor的名称
    void setName(const std::string& name);  //设置tensor名称
    std::pair<EXPRP, int> expr() const {
        return std::make_pair(mFrom, mFromIndex);
    }
    // If compute info error, return nullptr
    const Info* getInfo();  //获取tensor的描述信息，如维度，数据类型等
    bool resize(INTS dims); // 修改tensor的维度
    template <typename T>
    const T* readMap() {    // tensor的数据的起始位置
        return (const T*)readInternal();  
    }

    template <typename T>
    T* writeMap() {  // tensor的数据的起始位置
        return (T*)writeInternal();
    }

    //Depecerate
    void unMap();

    bool input(VARP src);
    static void replace(VARP dst, VARP src);  //相当于“=”

    static VARP create(EXPRP expr, int index = 0);  

    static std::vector<VARP> load(const char* fileName);
    static std::map<std::string, VARP> loadMap(const char* fileName);
    static std::vector<VARP> load(const uint8_t* buffer, size_t length);
    static std::map<std::string, VARP> loadMap(const uint8_t* buffer, size_t length);
    static std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>> getInputAndOutput(const std::map<std::string, VARP>& allVariable);
    static std::vector<VARP> mapToSequence(const std::map<std::string, VARP>& source);
    static std::vector<EXPRP> getExecuteOrder(const std::vector<VARP>& output);
    static void save(const std::vector<VARP>& vars, const char* fileName);
    static void save(const std::vector<VARP>& vars, NetT* dest);
    
    // Pack a few Variable to compute in one pipeline
    static void prepareCompute(const std::vector<VARP>& vars, bool forceCPU = false);
    static void compute(const std::vector<VARP>& vars, bool forceCPU = false);

    size_t linkNumber() const;
    const std::vector<WeakEXPRP>& toExprs() const;
    void setExpr(EXPRP expr, int index) {
        mFrom = expr;
        mFromIndex = index;
    }
private:
    Variable(EXPRP expr, int index) {
        mFrom      = expr;
        mFromIndex = index;
    }

    void* readInternal(bool forShape = false);
    void* writeInternal(bool inform=true);
    void informDirty();

    friend class Expr;  //可以访问Expr方法
    EXPRP mFrom;   //EXPR对象
    int mFromIndex;  //tensor在graph的编号
};
```

```cpp
class MNN_PUBLIC Expr {
public:
    struct Inside;
    enum MemoryType {
        COPY,
        MOVE,
        REF
    };
    static EXPRP create(Tensor* tensor, bool own = false);

    static EXPRP create(Variable::Info&& info, const void* ptr, VARP::InputType type, MemoryType copy = COPY);
    static EXPRP create(const OpT* op, std::vector<VARP> inputs, int outputSize = 1);
    static EXPRP create(std::shared_ptr<BufferStorage> extra, std::vector<VARP>&& inputs, int outputSize = 1);
    static EXPRP create(std::unique_ptr<OpT>&& op, std::vector<VARP> inputs, int outputSize = 1) {
        return create(op.get(), inputs, outputSize);
    }
    void setName(const std::string& name);

    const Op* get() const {
        return mOp;
    }
    const std::vector<VARP>& inputs() const {
        return mInputs;
    }
    int outputSize() const {
        return (int)mOutputNames.size();
    }
    static void replace(EXPRP oldExpr, EXPRP newExpr);
    bool requireInfo();
    void visitOutputs(const std::function<bool(EXPRP, int)>& visit);
    static void visit(EXPRP expr, const std::function<bool(EXPRP)>& before, const std::function<bool(EXPRP)>& after);

    const std::vector<WeakEXPRP>& outputs() const {
        return mTo;
    }
    ~Expr();

    bool visited() const {
        return mVisited;
    }
    void setVisited(bool visited) {
        mVisited = visited;
    }
    const std::string& name() const {
        return mName;
    }
    const std::string& outputName(int index) {
        return mOutputNames[index];
    }

    VARP::InputType inputType() const {return mType;}
    Variable::Info* outputInfo(int index) const;
    std::shared_ptr<BufferStorage> extra() const {
        return mStorage;
    }
    bool setInfoDirty();
    std::shared_ptr<Inside> inside() const {
        return mInside;
    }
    bool valid() const {
        return mValid;
    }

private:
    static void _addLinkForInputs(EXPRP expr);

    Expr(int outputSize);
    Expr(Tensor* tensor, bool own = false);

    friend class Variable;  //可以访问Variable方法
    friend class VARP;     //可以访问VAPR方法
    VARP::InputType mType;  //节点类型，INPUT，CONSTANT，TRAINABLE
    const Op* mOp;  //操作
    std::vector<VARP> mInputs; //操作的所有输入
    std::vector<std::string> mOutputNames;  //输出节点名称

    bool mValid = true;
    std::shared_ptr<BufferStorage> mStorage;   //参数
    std::string mName;  //节点名
    std::shared_ptr<Inside> mInside = nullptr;
    bool mVisited                   = false;
    std::vector<WeakEXPRP> mTo;

};
```





### NetT

NetT是保存网络的结构

netT->oplists  模型中所有算子

### OpT

OpT表示算子

```cpp
struct OpT : public flatbuffers::NativeTable {                                                                                   
    typedef Op TableType;
    std::vector<int32_t> inputIndexes;  //输入维度
    OpParameterUnion main;    
    std::string name;    //算子命名
    std::vector<int32_t> outputIndexes;  //输出维度
    OpType type;       //算子类型，如Conv2d
    MNN_DATA_FORMAT defaultDimentionFormat;  //数据类型NHWC等
    OpT()
        : type(OpType_AbsVal),
          defaultDimentionFormat(MNN_DATA_FORMAT_NHWC) {
    }
  };
```



## ONNX转换

```cpp
auto opConverter = onnxOpConverterSuit::get()->search(opType);
```

onnxOpConverterSuit是算子转换器的包装类，包含mConverterContainer，mConverterContainer中是所有算子转换起的集合，如果算子不在该集合中，则MNN会尝试使用默认转换器DefaultonnxOpConverter进行转换，DefaultonnxOpConverter只会对int，float和tensor进行转换

```cpp
std::map<std::string, onnxOpConverter*> mConverterContainer;
```

所有算子都是通过onnxOpConverterRegister添加到mConverterContainer中的

```cpp
template <typename T>
class onnxOpConverterRegister {
public:
    onnxOpConverterRegister(const char* name) {
        T* opConverter                 = new T;
        onnxOpConverterSuit* container = onnxOpConverterSuit::get();
        container->insert(opConverter, name);
    }
    ~onnxOpConverterRegister() {
    }

private:
    onnxOpConverterRegister();
};
```

```cpp
MNN::OpT* MNNOp  = new MNN::OpT;
MNNOp->name      = name;
MNNOp->type      = opConverter->opType();  //算子类型，如Conv2d
MNNOp->main.type = opConverter->type();  //算子参数类型，在schema中定义的算子参数类型
mnnNodesMap.insert(std::make_pair(name, MNNOp));
```

MNN会将算子中的已知参数转换为一个const算子

```cpp
// convert initializer to be Constant node(op)
for (int k = 0; k < onnxNode.input_size(); ++k) {
    const auto& inputName = onnxNode.input(k);
    const auto it         = initializers.find(inputName);
    if (it != initializers.end() && tensorsName.find(it->first) == tensorsName.end()) {
                // Create const Op
        MNN::OpT* constOp   = new MNN::OpT;
        constOp->type       = MNN::OpType_Const;
        constOp->main.type  = MNN::OpParameter_Blob;
        constOp->main.value = onnxOpConverter::convertTensorToBlob(it->second);
        mnnNodesMap.insert(std::make_pair(inputName, constOp));
        auto outputIndex = (int)netT->tensorName.size();
        constOp->name    = it->first;
        constOp->outputIndexes.push_back(outputIndex);
        tensorsName.insert(std::make_pair(it->first, outputIndex));
        netT->tensorName.emplace_back(constOp->name);
        netT->oplists.emplace_back(constOp);
   }
}
```

对算子进行转换

```cpp
opConverter->run(MNNOp, &onnxNode, opInitializers);
```

### 优化计算图

计算图优化主要是做一些算子转换和融合

```cpp
std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT, modelPath.forTraining, modelPath);
```

```cpp
postConvertPass = {
        // Seperate Tensor for inplace op
        "RemoveInplace",

        // Remove Unuseful Op such as NoOp, Identity, Seq2Out,
        "RemoveUnusefulOp",

        // Remove Dropout, if `forTraining` flag is set, Dropout will be reserved
        "RemoveDropout",

        // Remove Dup op
        "FuseDupOp",

        // Turn InnerProduct from Caffe / Onnx to Convolution
        "TransformInnerProduct",

        // Turn Im2Seq from Caffe to Reshape
        "TransformIm2Seq",

        // Turn Caffe's ShuffleChannel to compose op
        "TransformShuffleChannel",

        // Turn Onnx's Pad to Tensorflow's Pad
        "TransformOnnxPad",
};

std::vector<std::string> afterProgramConvert = {
        // Turn BatchNormal to Scale When inference, if `forTraining` flag is set, BN will be reserved
        "TransformBatchNormal",

        // expand ShapeN to N Shapes
        "ResolveTfShapeN",

        // WARNNING: should merge BN and Scale before Relu and Relu6

        // Merge BN info Convolution, if `forTraining` flag is set, BN will be reserved
        "MergeBNToConvolution",

        // Merge Scale info Convolution
        "MergeScaleToConvolution",

        // Merge Relu Convolution
        "MergeReluToConvolution",

        // Merge Relu6 Convolution
        "MergeRelu6ToConvolution",
};
afterProgramConvert = {
        // Add tensor dimension format convert for NC4HW4 - NHWC / NC4HW4 - NCHW
        "AddTensorFormatConverter",

        // Turn group convolution to Slice - Convolution - Concat
        "TransformGroupConvolution",

        // Remove output tensor convert
        "RemoveOutputTensorConvert",
};
```

算子融合算法先通过代码注册到对应的map

```cpp
static PostConverterRegister<MergeBNToConvolution> __l("MergeBNToConvolution");
```

```cpp
PostConverterRegister(const char* claim) {
        T* instance = new T;
        PostConverter::add(std::shared_ptr<PostConverter>(instance), claim);
}
```

```cpp
void PostConverter::add(std::shared_ptr<PostConverter> converter, std::string key) {
    auto gConverter = getConvertMap();
    gConverter->insert(std::make_pair(key, converter));
}
```

进行融合时，再根据需要融合的算子调用注册的方法

```cpp
void RunNetPass(const std::vector<std::string>& passes, std::unique_ptr<MNN::NetT>& originNet) {
    for (auto pass : passes) {
        auto convert = PostConverter::get(pass);
        if (nullptr == convert) {
            LOG(INFO) << "Can't find pass of " << pass << "\n";
            continue;
        }
        bool valid = convert->onExecute(originNet);
        if (!valid) {
            LOG(INFO) << "Run " << pass << "Error\n";
        }
    }
}
```

```cpp
class MergeToConvolution : public PostConverter {
public:
    virtual bool merge2Convolution(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) const = 0;

    virtual bool merge2Convolution3D(const MNN::OpT* inplaceOp, MNN::OpT* convolutionOp) const = 0;

    virtual bool onExecute(std::unique_ptr<MNN::NetT>& net) const override {
        // Merge Layer
        std::vector<MNN::OpT*> readyToDelete;
        for (auto iter = net->oplists.begin(); iter != net->oplists.end(); iter++) {
            MNN::OpT& currentOp = *(iter->get());
            if (currentOp.type != MNN::OpType_Convolution
                && currentOp.type != MNN::OpType_Deconvolution
                && currentOp.type != MNN::OpType_ConvolutionDepthwise
                && currentOp.type != MNN::OpType_Convolution3D) {
                continue;
            }
            DCHECK(currentOp.outputIndexes.size() == 1) << "Conv output ERROR!";

            // merge Batchnorm/Relu/Relu6 to Convolution
            std::vector<MNN::OpT*> nextOp = PostTreatUtils::_findOpByInputIndex(currentOp.outputIndexes[0], net.get());
            while (1) {
                if (nextOp.size() != 1) {
                    break;
                }
                const int nextOutputIndex = nextOp[0]->outputIndexes[0];
                bool succ;
                if (currentOp.type == MNN::OpType_Convolution3D) {
                    succ = merge2Convolution3D(nextOp[0], &currentOp);
                } else {
                    succ = merge2Convolution(nextOp[0], &currentOp);
                }
                if (PostTreatUtils::_isSingleInputOutput(nextOp[0]) && succ) {
                    // LOG(INFO) << "Merge " << nextOp[0]->name.c_str()<< " into convolution: " <<
                    // currentOp.name.c_str();
                    currentOp.outputIndexes[0] = nextOp[0]->outputIndexes[0];
                    readyToDelete.push_back(nextOp[0]);
                    nextOp = PostTreatUtils::_findOpByInputIndex(nextOutputIndex, net.get());
                } else {
                    break;
                }
            }
        }
        for (auto op : readyToDelete) {
            PostTreatUtils::_removeOpInNet(op, net.get());
        }
        return true;
    }
};
```



