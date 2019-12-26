#ifndef _LOSS_H_
#define _LOSS_H_

#include <xtensor/xarray.hpp>

enum loss_type {
    mean_squared_error,
    mean_absolute_error,
    log_error
};


class loss {
public:    
    friend class loss_factory;

    loss(const loss&) = default;
    ~loss() = default;

    virtual xt::xarray<double> derivative(const xt::xarray<double>& p, const xt::xarray<double>& y) = 0;
    virtual xt::xarray<double> cost(const xt::xarray<double>& p, const xt::xarray<double>& y) = 0;

protected:
    loss() = default;
};

class mse_loss : public loss {
public:
    friend class loss_factory;

    mse_loss(const mse_loss&) = default;
    ~mse_loss() = default;

    xt::xarray<double> derivative(const xt::xarray<double>& p, const xt::xarray<double>& y) override {
        return (p - y);
    }

    xt::xarray<double> cost(const xt::xarray<double>& p, const xt::xarray<double>& y) override {
        return xt::sum(0.5 * xt::square(derivative(p, y)), {0}) / y.shape()[0];
    }

protected:
    mse_loss() = default;
};

class mae_loss : public loss {
public:
    friend class loss_factory;

    mae_loss(const mae_loss&) = default;
    ~mae_loss() = default;

    xt::xarray<double> derivative(const xt::xarray<double>& p, const xt::xarray<double>& y) override {
        return (p - y);
    }

    xt::xarray<double> cost(const xt::xarray<double>& p, const xt::xarray<double>& y) override {
        return xt::sum(xt::abs(derivative(p, y)), {0}) / y.shape()[0];
    }

protected:
    mae_loss() = default;        
};

class log_loss : public loss {
public:
    friend class loss_factory;

    log_loss(const log_loss&) = default;
    ~log_loss() = default;

    xt::xarray<double> derivative(const xt::xarray<double>& p, const xt::xarray<double>& y) override {
        return -y / (1.0 + xt::exp(p * y));
    }

    xt::xarray<double> cost(const xt::xarray<double>& p, const xt::xarray<double>& y) override {
        return xt::sum(xt::log(1 + xt::exp(-p * y)), {0}) / y.shape()[0];
    }

protected:
    log_loss() = default;    
};



class loss_factory {
public:
    loss_factory() = delete;
    loss_factory(const loss_factory&) = delete;
    ~loss_factory() = delete;


    static std::shared_ptr<loss> create(loss_type type) {
        switch (type) {
            case mean_squared_error:
                return std::allocate_shared<mse_loss>(LossAlloc<mse_loss>());
            case mean_absolute_error:
                return std::allocate_shared<mae_loss>(LossAlloc<mae_loss>());
            case log_error:
                return std::allocate_shared<log_loss>(LossAlloc<log_loss>());
            default:
                return nullptr;
        };
    }

protected:
    template<typename T>
    struct LossAlloc : std::allocator<T> {
        template<typename U, typename... A>
        void construct(U* ptr, A&&... args) {
            new(ptr) U(std::forward<A>(args)...);
        }
        template<typename U>
        void destroy(U* ptr) {
            ptr->~U();
        }
    }; 
};


#endif