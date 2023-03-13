#ifndef WEB_BASE_PARAM_
#define WEB_BASE_PARAM_
#include <string>

namespace BASE_NAMESPACE{

class BaseForm{
  public:
    BaseForm(const BaseForm&)=default;
    BaseForm &operator=(const BaseForm&)=default;
    BaseForm(BaseForm&&)=default;
    BaseForm &operator=(BaseForm&&)=default
    virtual ~BaseForm()=default;
    std::string task_id;
    BaseForm():task_id(""){}
};

class BaseResponse{
public:
    BaseResponse(const BaseResponse&)=default;
    BaseResponse &operator=(const BaseResponse&)=default;
    BaseResponse(BaseResponse&&)=default;
    BaseResponse &operator=(BaseResponse&&)=default
    virtual ~BaseResponse()=default;
    int code;
    std::string message;
    std::string task_id;
    int timecost;
    BaseResponse():code(9999),message("INIT"),task_id(""),timecost(0){}
};


}

#endif