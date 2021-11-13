#include "semantic_segmentation_rv1126.hpp"



EngineInferRV1126::EngineInferRV1126(){
    model = nullptr;
    m_net_input_index = 0;
    m_net_output_index = 0;
    num_net_input_node = 1;
    num_net_output_node = 1;
    inputs = new rknn_input;
    outputs = new rknn_output;
}


EngineInferRV1126::~EngineInferRV1126(){
    rknn_outputs_release(ctx, 1, outputs);
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
}

StatusCode EngineInferRV1126::init(const InferParam &param){
    //1. 拷贝参数
    net_params = std::move(param);
    //2. 打开模型
    FILE *fp = fopen(net_params.model_path, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", net_params.model_path);
        return StatusCode::INIT_LOAD_BIN_FAILED;
    }
    //3. 读取模型至model对应内存。
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", net_params.model_paths[0]);
        free(model);
        return StatusCode::INIT_LOAD_BIN_FAILED;
    }
    fclose(fp);

    //4. rknn模型加载，至ctx。
    int ret = rknn_init(&ctx, model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return StatusCode::INIT_LOAD_BIN_FAILED;
    }

#if 1
    // 5. 获取模型输入输出信息
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return StatusCode::INIT_LOAD_BIN_FAILED;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return StatusCode::INIT_LOAD_BIN_FAILED;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return StatusCode::INIT_LOAD_BIN_FAILED;
        }
        printRKNNTensor(&(output_attrs[i]));
    }
#endif

    return StatusCode::INIT_SUCCESS;
}


StatusCode EngineInferRV1126::predict(){
    // 1. 按rknn的格式设置输入参数
    memset(inputs, 0, sizeof(rknn_input));
    inputs->index = 0;
    inputs->type = RKNN_TENSOR_UINT8;
    inputs->size =net_params.net_height * net_params.net_width * 3;
    inputs->fmt = RKNN_TENSOR_NHWC;
    inputs->buf = m_img.data;

    int ret = rknn_inputs_set(ctx, num_net_input_node, inputs);
    if(ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return StatusCode::INFER_CHANNEL_ERROR;
    }
    // 2. 推理
    time_beg = time_stamp();
    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return StatusCode::INFER_CHANNEL_ERROR;
    }

    // 3. 获取输出信息
    memset(outputs, 0, sizeof(rknn_output));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
    if(ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return StatusCode::INFER_CHANNEL_ERROR;
    }
    time_end = time_stamp();
    printf("Predict rknn_run time: %.4f ms\n",(time_end-time_beg)/1000);

#if 0
    FILE *fp = fopen("./save.txt", "w+");   //  打开文件
    printf("output size: %d\n",outputs[0].size);
    for (int i = 0; i< int(outputs[0].size/sizeof(float)); i++){
        fprintf(fp, "%.5f\n", predictions[i]);           //  存储文件
    }
    fclose(fp);
#endif
    return StatusCode::INFER_SUCCESS;
}
