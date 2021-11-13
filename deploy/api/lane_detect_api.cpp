#include "../inc/lane_detect_api.h"
#define RV1126
#ifdef RV1126
#include "../../infer/semantic_segmentation_rv1126.hpp"
#endif

// 封装车道线检测通用的方法和对外接口
// 功能：1. 初始化接口
//      2. 根据原图，车道线坐标渲染，输出渲染图
//      3. 根据标定参数，输出每条车道线的id、车道线开始和结束X坐标，以及一元三次方程的四个参数
//      4. 根据标定参数，输出渲染图
//      5. 调用模型运行


int decodeAFs(unsigned char* output_ptr, std::map<int, std::vector<cv::Point>> &out_points_pair,
        float* hm_ptr, float* vaf_ptr, float* haf_ptr, float fg_thresh,
              float err_thresh){
    int next_lane_id = 1;
    std::vector<std::vector<cv::Point>> lane_end_pts = {};
    out_points_pair.clear();

#if 0
    cv::Mat temp_mask = cv::Mat::zeros(72,208,CV_8UC1);
    for (int k = 0; k< 72*208; k++){
        *(temp_mask.ptr<uchar>() + k) = (unsigned char)(sigmoid(hm_ptr[k])*255);
    }
    cv::imwrite("hm2.png", temp_mask);
#endif

    //按行遍历，从下往上
    for (int i = fea_height - 1; i>=0 ; --i){
        float* hm_i_ptr = hm_ptr + i * fea_width;

        std::vector<int> col_vals = {};
        // 取出大于阈值的所有坐标
        for(int j = 0; j < fea_width; ++j){
            //if (sigmoid(*(hm_i_ptr+j)) > fg_thresh)
            if (*(hm_i_ptr+j) > fg_thresh){
                col_vals.emplace_back(j);
            }
        }

        // 对满足要求的列值聚类， 每个类多个点
        std::vector<std::vector<int>> clusters = {{}};
//        std::cout<<"**cluster size : "<<clusters.size()<<std::endl;
        int pre_col;
        if (col_vals.size() >0) pre_col = col_vals[0];
        for(auto col_val: col_vals) {
            // 新的类
            if (col_val - pre_col > err_thresh){
                std::vector<int> cluster = {};
                cluster.emplace_back(col_val);
                clusters.emplace_back(cluster);
                pre_col = col_val;
                continue;
            }
            // 当前列点和上一个列点距离小于阈值，并且满足两点在haf_ptr数组中都大于0，则将该点添加到当前类中
            if ((haf_ptr[i*fea_width + pre_col] >=0) && (haf_ptr[i*fea_width + col_val]>=0)){
                clusters[clusters.size()-1].emplace_back(col_val);
                pre_col = col_val;
                continue;
            }
            // 和上一个条件一致
            else if ((haf_ptr[i*fea_width + pre_col] >=0) && (haf_ptr[i*fea_width + col_val]<0)){
                clusters[clusters.size() - 1].emplace_back(col_val);
                pre_col = col_val;
            }// 如果上一个点在haf_ptr数组中值小于0，而当前点在haf_ptr数组中大于0，则新建一个聚类
            else if ((haf_ptr[i * fea_width + pre_col] < 0) && (haf_ptr[i * fea_width + col_val] >= 0)){
                std::vector < int > cluster = {};
                cluster.emplace_back(col_val);
                clusters.emplace_back(cluster);
                pre_col = col_val;
                continue;
            } // 如果当前点和上一个点在haf_ptr数组中都小于0，则将该点添加到当前类中
            else if ((haf_ptr[i*fea_width + pre_col] <0) && (haf_ptr[i*fea_width + col_val]<0)){
                clusters[clusters.size() - 1].emplace_back(col_val);
                pre_col = col_val;
                continue;
            }
        }

        //test
//        for (auto item :clusters){
//            for (auto val:item) {
//                printf("%d,", val);
//            }
//            printf("\t\t");
//        }
//        printf("\n\n");
        // 初始化assigned中每个类值为0
        std::vector<int> assigned = {};

        for(int k =0; k< clusters.size(); k++){
            assigned.emplace_back(0);
//            for(auto val: clusters[k]){
//                std::cout<<"cluster size:"<<k<<"/"<<clusters.size()<<", val("<<clusters[k].size()<<"):"<<val<<std::endl;
//            }
        }

        // parse vertically
        //assign existing lanes
        // cv::Mat C;
        // C = cv::Mat::ones(lane_end_pts.size(), clusters.size(), CV_32F) * 1000;
        std::vector<float> C;
        for (int k=0; k<lane_end_pts.size(); ++k){
            std::vector<cv::Point> pts = lane_end_pts[k];
            for(int c=0; c< clusters.size(); ++c){
                std::vector<int> cluster = clusters[c];
                float cluster_mean = std::accumulate(std::begin(cluster), std::end(cluster), 0.0)/cluster.size();
                //printf();
                std::vector<std::pair<float,float>> vafs;
                float err_sum = 0;
                for(auto pt:pts){
                    float vaf_x = vaf_ptr[pt.y * fea_width + pt.x];
                    float vaf_y = vaf_ptr[fea_height*fea_width + pt.y * fea_width + pt.x];
//                    printf("vaf: %.6f, %.6f\n", vaf_x, vaf_y);
                    float vaf_norm = sqrt(vaf_x * vaf_x + vaf_y * vaf_y);
                    vaf_x /= vaf_norm;
                    vaf_y /= vaf_norm;
                    // get predicted cluster center
                    float pt_clm = sqrt((pt.x - cluster_mean) * (pt.x - cluster_mean) + (pt.y - i)*(pt.y - i));
                    float pred_x = pt.x + vaf_x * pt_clm;
                    float pred_y = pt.y + vaf_y * pt_clm;
                    err_sum += sqrt((pred_x - cluster_mean) * (pred_x - cluster_mean) + (pred_y - i)*(pred_y - i));
                }
                C.emplace_back(err_sum / pts.size());
                // C.at<float>(k,c) = err_sum / pts.size();
            }
        }


        // 对C进行排序
        std::vector<int> idx(C.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&C](int i1, int i2) {return C[i1] < C[i2]; });

        std::map<int, std::vector<int>> cls_lane_pts;
        // assign clusters to lane( in acsending order of error)
        for(int k=0; k<idx.size(); ++k) {
            if ((C[idx[k]] >= err_thresh) || (isnan(C[idx[k]])))
                break;
            int c = idx[k] % clusters.size();
            int r = int(idx[k] / clusters.size());

            if (assigned[c] == 1) continue;
            assigned[c] = 1;
            std::vector<int> cluster = clusters[c];
            if (cls_lane_pts.find(r) == cls_lane_pts.end())
                cls_lane_pts.insert(std::make_pair(r, cluster));
            else
                cls_lane_pts[r].insert(cls_lane_pts[r].end(), cluster.begin(), cluster.end());
        }
        for (auto &idx_vals: cls_lane_pts){
            int r = idx_vals.first;
            std::vector<int> cluster = idx_vals.second;

            std::vector<cv::Point> lane_pts;
            for(auto col_val: cluster){
                output_ptr[i * fea_width + col_val] = r+1;
                lane_pts.emplace_back(cv::Point(col_val, i));
            }
            lane_end_pts[r] = lane_pts;
            float cluster_mean = std::accumulate(std::begin(cluster), std::end(cluster), 0.0)/cluster.size();
//            for(auto val: cluster){
//                printf("%d, ", val);
//            }

//            std::cout<<cluster_mean<<std::endl;
            cv::Point lane_centers = cv::Point((int)(cluster_mean * stride *resize_w_ratio), (int)(i *stride*resize_h_ratio));
            //printf("**%f**%d**%d**  ", cluster_mean,lane_centers.x, i);
            out_points_pair.at(r+1).emplace_back(lane_centers);
        }
//        printf("\n");

//        std::cout<<"cluster size : "<<clusters.size()<<std::endl;
        for(int k=0; k< clusters.size(); ++k){
            std::vector<int> cluster = clusters[k];
            if (assigned[k]==0){
                std::vector<cv::Point> cluster_coords;
                if (cluster.size() == 0){
//                    next_lane_id += 1;
                    continue;
                }
                for (auto m: cluster){
                    output_ptr[i*fea_width + m] = next_lane_id;
                    cluster_coords.emplace_back(cv::Point(m, i));
                }
//                std::cout<<"!!!!!!!!"<<cluster.size()<<std::endl;
                lane_end_pts.emplace_back(cluster_coords);
                float cluster_mean = std::accumulate(std::begin(cluster), std::end(cluster), 0.0)/cluster.size();
               // printf("%f, \t", cluster_mean);

//                std::cout<<i<<","<<cluster_mean<<std::endl;
                std::vector<cv::Point> lane_centers = {cv::Point(cluster_mean* stride*resize_w_ratio, i*stride*resize_h_ratio)};
                out_points_pair.emplace(next_lane_id, lane_centers);
                next_lane_id += 1;
            }
        }
        //printf("\n");
    }


    return 0;
}


LaneDetectHandle LaneDetect_API_Create(const char* path_config_detector){
    //1. 创建模型推理对象
#ifdef RV1126
    EngineInfer* handle = new EngineInferRV1126();
#else if TRT
    EngineInfer* handle = new EngineInferTrt();
#endif
    //2. 如果为空，则创建失败，返回nulptr
    if (nullptr == handle)
        return nullptr;

    //3. 通过config文件加载模型，并返回调用句柄
    std::string config = path_config_detector;
    if (StatusCode::INIT_SUCCESS != handle->init(config))
        return nullptr;
    return (LaneDetectHandle )(handle);
}



int LaneDetect_API_Run(std::vector<std::vector<point2i>>& output_lanes,
                       LaneDetectHandle handle,
                       unsigned char* buffer,
                       int width, int height, int up_height, int down_height, int channel){
#ifdef RV1126
    EngineInfer *oHandle = (EngineInfer *)handle;
#else if TRT
    EngineInfer *oHandle = (EngineInfer *)handle;
#endif
    if (oHandle == NULL) {
        std::cout<<"LaneDetect_STATUS_ERROR_DETECT_ENGINE"<<std::endl;
        return -1;
    }
    if (buffer == NULL || width <= 0 || height <= 0)
    {
        std::cout<<"LaneDetect_STATUS_ERROR_INPUT_IMAGE"<<std::endl;
        return -1;
    }
    int new_height = height - up_height - down_height;
    cv::Mat image;
    if (channel == 3){
        cv::Mat colorImg(new_height, width, CV_8UC3, buffer + up_height * width * channel);
        image = colorImg;
//        cv::imwrite("inputimg.jpg", image);
    }
    else{
        std::cout<<"LaneDetect_STATUS_ERROR_INPUT_IMAGE"<<std::endl;
        return -1;
    }

    int flag = oHandle->setImage(image, true);
    flag = oHandle->predict();


    // acquire results
    for (auto it: oHandle->out_points_pair){
//        printf("lane point size: %d\n", it.second.size());
        if (it.second.size() <20)
            continue;
        std::vector<point2i> lane_points;
        //printf("emplace lane point size: %d\n", it.second.size());
        for(auto point: it.second){
            lane_points.emplace_back(point2i(point.x, point.y + up_height));
        }
        output_lanes.emplace_back(lane_points);
    }
    return flag;
}


void LaneDetect_API_Release(LaneDetectHandle handle){
    LaneDetectTrt *oHandle = (LaneDetectTrt *)handle;
    if (nullptr != oHandle){
        oHandle->free();
        delete oHandle;
    }
    return;
}
