

template<class _Tp>
class  shared_ptr
{
    template <class _Yp, class _OrigPtr>
    typename enable_if<is_convertible<_OrigPtr*,const enable_shared_from_this<_Yp>*>::value,void>::type
     __enable_weak_this(const enable_shared_from_this<_Yp>* __e,_OrigPtr* __ptr) _NOEXCEPT
    {
        typedef typename remove_cv<_Yp>::type _RawYp;
        if (__e && __e->__weak_this_.expired())
        {
            __e->__weak_this_ = shared_ptr<_RawYp>(*this,
                const_cast<_RawYp*>(static_cast<const _Yp*>(__ptr)));
        }
    }

    void __enable_weak_this(...) _NOEXCEPT {}
};

