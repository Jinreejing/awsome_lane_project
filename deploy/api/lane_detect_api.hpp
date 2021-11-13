//
// Created by xiangj15 on 2021/7/13.
//

#ifndef LANE_PROJECT_LANE_DETECT_API_H
#define LANE_PROJECT_LANE_DETECT_API_H

#include <iostream>
#include <vector>


#ifdef __cplusplus
extern "C" {
#endif

    struct point2i{
        int x;
        int y;
        point2i(int _x, int _y){
            this->x = _x;
            this->y = _y;
        }
    };

    //////////////////////////////////////////////////////////////////////
    typedef void* LaneDetectHandle;
    //////////////////////////////////////////////////////////////////////
    // API For lane detect
    /*!
     * \brief Create lane detect handle
     * \param[in] config: Config file path. The config file will specify the onnx model path and the trt model path.
     * \return craft analysis handle, failed when return nullptr
     */
    LaneDetectHandle LaneDetect_API_Create(const char* path_config_detector);

    /*!
     * \brief Do lane detect with single image.
     * \param[out] output_lanes: output lane points.
     * \return success code, 0 success, others failed
     */
    int LaneDetect_API_Run(std::vector<std::vector<point2i>>& output_lanes,
                             LaneDetectHandle handle,
                             unsigned char* buffer,
                             int width, int height, int up_height, int down_height, int channel);

    /*!
     * \brief Do lane detect with single image.
     * \param[out] output_lanes: output lane points.
     * \return success code, 0 success, others failed
     */
    int LaneDetect_API_Run_Package(char* out_ptr, int out_length,
                             LaneDetectHandle handle,
                             unsigned char* buffer,
                             int width, int height, int up_height, int down_height, int channel);


    void LaneDetect_API_Release(LaneDetectHandle handle);

#ifdef __cplusplus
}
#endif
#endif //LANE_PROJECT_LANE_DETECT_API_H