#ifndef _LANE_AF_
#define _LANE_AF_

#define RV1126
#ifdef RV1126
#include "semantic_segmentation_rv1126.hpp"
#endif
// LaneAF 方法
// 功能：1. 初始化
//      2. 输入图片数据，得到车道线结果
//      3. 释放资源
class laneAF{
public:
    laneAF(){
        config_path=_config_path;

    }
    ~laneAF(){

    }

    int init_engine(const char*_config_path);
    int get_lane_points(std::vector<cv::Point2i>lane_points, cv::Mat& img);

private:
    const char* config_path;
    EngineInfer* handle;


}

#endif