#ifndef __SEMANTIC_SEGMENTATION_RV1126_HPP__
#define __SEMANTIC_SEGMENTATION_RV1126_HPP__
#include "semantic_segmentation.hpp"
#include "rknn_api.h"

class EngineInferRV1126: public EngineInfer
{
public:
    EngineInferRV1126();
    virtual ~EngineInferRV1126();

public:
    virtual StatusCode init(const std::string config);
    virtual StatusCode predict();

protected:

private:
    InferParam net_params;
    rknn_input* inputs;
    rknn_output* outputs;
    rknn_context ctx{};
    unsigned char* model;


}

#endif