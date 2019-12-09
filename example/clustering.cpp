#include <clustering/kmeans.h>
#include <preprocessing/scaler.h>
#include <xtensor/xio.hpp>
#include <iostream>

int main(int argc, char** args)
{
    xt::xarray<double> X = {{2.0, 5.0},
                            {2.1, 4.8},
                            {2.2, 4.9},
                            {1.9, 5.1},
                            {2.1, 5.2},
                            {5.0, 2.0},
                            {5.1, 2.1},
                            {5.2, 2.2},
                            {5.1, 2.1},
                            {4.9, 2.0}};

    standard_scaler scaler;
    X = scaler.fit_transform(X);

    kmeans cls;
    cls.set_param("k", 2.0);

    cls.fit(X);


    return 0;
}
