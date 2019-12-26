#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_


#include <xtensor/xarray.hpp>
#include <functional>

enum activation_type {
    linear_t,
    sigmoid_t,
    tanh_t
};

class activation {
public:    
    friend class activation_factory;

    activation(const activation&) = default;
    ~activation() = default;

    virtual xt::xarray<double> apply(const xt::xarray<double>& X) const = 0;
    virtual xt::xarray<double> revert(const xt::xarray<double>& X, const xt::xarray<double>& g) const = 0;

protected:
    activation() = default;    
};

class linear_activation : public activation {
public:    
    friend class activation_factory;

    linear_activation(const linear_activation&) = default;
    ~linear_activation() = default;

    xt::xarray<double> apply(const xt::xarray<double>& X) const override {
        return X;
    }

    xt::xarray<double> revert(const xt::xarray<double>& X, const xt::xarray<double>& g) const override
    {
        return 1.0;
    }

protected:
    linear_activation() = default;    
};

class sigmoid_activation : public activation {
public:    
    friend class activation_factory;

    sigmoid_activation(const sigmoid_activation&) = default;
    ~sigmoid_activation() = default;

    xt::xarray<double> apply(const xt::xarray<double>& X) const override {
        return 1.0 / (1.0 + xt::exp(-X));
    }

    xt::xarray<double> revert(const xt::xarray<double>& X, const xt::xarray<double>& g) const override
    {
        return g * (1.0 - g);
    }

protected:
    sigmoid_activation() = default;    
};

class tanh_activation : public activation {
public:    
    friend class activation_factory;

    tanh_activation(const tanh_activation&) = default;
    ~tanh_activation() = default;

    xt::xarray<double> apply(const xt::xarray<double>& X) const override {
        return xt::tanh(X);
    }

    xt::xarray<double> revert(const xt::xarray<double>& X, const xt::xarray<double>& g) const override
    {
        return 1.0 - g * g;
    }

protected:
    tanh_activation() = default;    
};


class activation_factory {
public:
    activation_factory() = delete;
    activation_factory(const activation_factory&) = delete;
    ~activation_factory() = delete;


    static std::shared_ptr<activation> create(activation_type type) {
        switch (type) {
            case linear_t:
                return std::allocate_shared<linear_activation>(ActivationAlloc<linear_activation>());
            case sigmoid_t:
                return std::allocate_shared<sigmoid_activation>(ActivationAlloc<sigmoid_activation>());
            case tanh_t:
                return std::allocate_shared<tanh_activation>(ActivationAlloc<tanh_activation>());
            default:
                return nullptr;
        };
    }

protected:
    template<typename T>
    struct ActivationAlloc : std::allocator<T> {
        template<typename U, typename... A>
        void construct(U* ptr, A&&... args) {
            new(ptr) U(std::apply<A>(args)...);
        }
        template<typename U>
        void destroy(U* ptr) {
            ptr->~U();
        }
    }; 
};

#endif