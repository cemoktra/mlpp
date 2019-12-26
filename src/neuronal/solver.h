#ifndef _SOLVER_H_
#define _SOLVER_H_

#include <xtensor/xarray.hpp>

// forward declaration
class dense_layer;

// solver type enum
enum solver_type {
    gradient_descent_solver,
    stochastic_gradient_descent_solver,
    ada_grad_decay_solver,
    adam_solver,
    adamax_solver
};

// gradient descent solver declarations
class gradient_descent
{
public:
    friend class solver_factory;
    
    gradient_descent(const gradient_descent&) = default;
    ~gradient_descent() = default;

    virtual void reset();
    virtual void do_iteration(std::shared_ptr<dense_layer> layer, const xt::xarray<double>& g);

protected:
    gradient_descent(double learning_rate = 0.1);

    size_t m_iteration;
    double m_rate;
};


class stochastic_gradient_descent : public gradient_descent
{
public:
    friend class solver_factory;

    stochastic_gradient_descent(const stochastic_gradient_descent&) = default;
    ~stochastic_gradient_descent() = default;

    void do_iteration(std::shared_ptr<dense_layer> layer, const xt::xarray<double>& g) override;

protected:
    stochastic_gradient_descent(double learning_rate = 0.1, size_t batches = 10);

    size_t m_batches;
};


class ada_grad_decay : public gradient_descent
{
public:
    friend class solver_factory;

    ada_grad_decay(const ada_grad_decay&) = default;
    ~ada_grad_decay() = default;

    void reset() override;
    void do_iteration(std::shared_ptr<dense_layer> layer, const xt::xarray<double>& g) override;

protected:
    ada_grad_decay(double learning_rate = 0.1, double decay = 0.9, double epsilon = 1e-8);

    xt::xarray<double> m_grad_history;
    double m_decay;
    double m_epsilon;
};

class adam : public gradient_descent
{
public:
    friend class solver_factory;

    adam(const adam&) = default;
    ~adam() = default;

    void reset() override;
    void do_iteration(std::shared_ptr<dense_layer> layer, const xt::xarray<double>& g) override;

protected:
    adam(double learning_rate = 0.1, double beta1 = 0.8, double beta2 = 0.999, double epsilon = 1e-8);

    xt::xarray<double> m;
    xt::xarray<double> v;

    double m_beta1;
    double m_beta2;
    double m_epsilon;
}; 

class adamax : public gradient_descent
{
public:
    friend class solver_factory;

    adamax(const adamax&) = default;
    ~adamax() = default;

    void reset() override;
    void do_iteration(std::shared_ptr<dense_layer> layer, const xt::xarray<double>& g) override;

protected:
    adamax(double learning_rate = 0.1, double beta1 = 0.9, double beta2 = 0.999);

    xt::xarray<double> m;
    xt::xarray<double> u;

    double m_beta1;
    double m_beta2;
};



class solver_factory {
public:
    solver_factory() = delete;
    solver_factory(const solver_factory&) = delete;
    ~solver_factory() = delete;

    static std::shared_ptr<gradient_descent> create(solver_type type) {
        switch (type) {
            case gradient_descent_solver:
                return std::allocate_shared<gradient_descent>(SolverAlloc<gradient_descent>());
            case stochastic_gradient_descent_solver:
                return std::allocate_shared<stochastic_gradient_descent>(SolverAlloc<stochastic_gradient_descent>());
            case ada_grad_decay_solver:
                return std::allocate_shared<ada_grad_decay>(SolverAlloc<ada_grad_decay>());
            case adam_solver:
                return std::allocate_shared<adam>(SolverAlloc<adam>());
            case adamax_solver:
                return std::allocate_shared<adamax>(SolverAlloc<adamax>());
            default:
                return nullptr;
        };
    }

protected:
    template<typename T>
    struct SolverAlloc : std::allocator<T> {
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