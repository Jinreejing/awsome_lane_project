#include <stdio.h>
#include <sys/time.h>


enum class StatusCode : int {
    // initial
    INIT_LOAD_PARAM_FAILED = 1,
    INIT_LOAD_BIN_FAILED = 3,
    INIT_SUCCESS = 0,

    //infer
    INFER_CHANNEL_ERROR = 5,
    INFER_SUCCESS = 2,

    //write image
    WRITE_CHANNEL_ERROR = 7,
    WRITE_IMAGE_ERROR=9,
    WRITE_SUCCESS=4,

    // get lane points from pic
    GET_LANE_POINTS_SUCCESS=6,
};

static double time_stamp() {
    struct timeval time;
    gettimeofday(&time, 0);
    return (time.tv_sec * 1000000 + time.tv_usec);
} 