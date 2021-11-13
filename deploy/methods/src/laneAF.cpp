#include "laneAF.hpp"


laneAF::laneAF(){
    handle = nullptr;
}

laneAF::~laneAF(){}

int laneAF::init_engine(const char*_config_path){
    config_path=_config_path;
#ifdef RV1126
    handle = new EngineInferRV1126();
#else if TRT
    handle = new EngineInferTrt();
#endif


}


int laneAF::get_lane_points(std::vector<cv::Point2i>lane_points, cv::Mat& img){

    return 0;
}
