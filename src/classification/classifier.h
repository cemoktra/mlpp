#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <Eigen/Dense>
#include <core/model.h>

class classifier : public model {
public:
    classifier() = default;
    classifier(const classifier&) = delete;
    ~classifier() = default;
    
    virtual void init_classes(size_t number_of_classes) = 0;    
};

#endif