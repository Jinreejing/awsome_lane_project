#ifndef __SEMANTIC_SEGMENTATION_HPP__
#define __SEMANTIC_SEGMENTATION_HPP__

#include <string>
#include <map>
#include "opencv2/opencv.hpp"
#include "../utils/common.hpp"

typedef struct INFER_PARAM_TAG{
public:
    INTER_PARAM_TAG(){
        config_path = "None";
        net_w = 0;
        net_h = 0;
    }
public:
    std::string config_path;
    int net_w;
    int net_h;
}InferParam;


class EngineInfer
{
public:
    EngineInfer(){
        m_scale_x = 0;
        m_scale_y = 0;
    }
    virtual ~EngineInfer()=default;

public:
    virtual StatusCode init(const std::string config) = 0;
    virtual StatusCode predict() = 0;
    virtual StatusCode setImage(const cv::Mat &img, bool keep_ratio=false);

    // segmentation result
    virtual StatusCode segData(cv::Mat &mask, int height, int width){
        cv::resize(m_mask(cv::Rect(0, 0, m_img_real_w, m_img_real_h)), mask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
        return 0;
    }

protected:
    int warmUp(){
        m_img = cv::Mat(m_net_h, m_net_w, CV_8UC3);
        cv::randu(m_img, cv::Scalar::all(0), cv::Scalar::all(255));
        predict();
    }

protected:
    cv::Mat m_img;
    float   m_scale_x;
    float   m_scale_y;
    int m_img_real_w;
    int m_img_real_h;
    int m_net_w;
    int m_net_h;
    cv::Mat m_mask;

}
#endif //__SEMANTIC_SEGMENTATION_HPP__