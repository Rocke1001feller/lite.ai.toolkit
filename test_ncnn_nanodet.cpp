//
// Test NCNN functionality with NanoDet
// export DYLD_LIBRARY_PATH=/Users/guoliufang/Documents/Pay/Sell-SDK/offline-sdk/lite.ai.toolkit/build/lite.ai.toolkit/lib:$DYLD_LIBRARY_PATH
// g++ -std=c++14 -I./build/lite.ai.toolkit/include -I./build/lite.ai.toolkit/opencv2 -I./build/lite.ai.toolkit/include/onnxruntime -I./ncnn test_ncnn_nanodet.cpp -L./build/lite.ai.toolkit/lib -llite.ai.toolkit -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lncnn -lonnxruntime -o test_ncnn_nanodet
// ./test_ncnn_nanodet

#include "lite/lite.h"

int main()
{
#ifdef ENABLE_NCNN
    std::string param_path = "/Users/guoliufang/Documents/Pay/Sell-SDK/offline-sdk/lite.ai.toolkit/hub/ncnn/cv/nanodet_m-opt.param";
    std::string bin_path = "/Users/guoliufang/Documents/Pay/Sell-SDK/offline-sdk/lite.ai.toolkit/hub/ncnn/cv/nanodet_m-opt.bin";
    std::string test_img_path = "/Users/guoliufang/Documents/Pay/Sell-SDK/offline-sdk/lite.ai.toolkit/examples/lite/resources/test_lite_detection_1.jpg";
    std::string save_img_path = "/Users/guoliufang/Documents/Pay/Sell-SDK/offline-sdk/lite.ai.toolkit/logs/test_ncnn_nanodet.jpg";

    std::cout << "Testing NCNN NanoDet..." << std::endl;

    // Test Specific Engine NCNN
    lite::ncnn::cv::detection::NanoDet *nanodet =
        new lite::ncnn::cv::detection::NanoDet(
            param_path, bin_path, 1, 320, 320);

    std::vector<lite::types::Boxf> detected_boxes;
    cv::Mat img_bgr = cv::imread(test_img_path);
    
    if (img_bgr.empty()) {
        std::cout << "Error: Could not read image: " << test_img_path << std::endl;
        delete nanodet;
        return -1;
    }

    nanodet->detect(img_bgr, detected_boxes);

    lite::utils::draw_boxes_inplace(img_bgr, detected_boxes);
    cv::imwrite(save_img_path, img_bgr);

    std::cout << "NCNN Version Detected Boxes Num: " << detected_boxes.size() << std::endl;
    std::cout << "Result saved to: " << save_img_path << std::endl;

    delete nanodet;
#else
    std::cout << "NCNN is not enabled in this build!" << std::endl;
    return -1;
#endif

    return 0;
}

// guoliufang@LiuFang lite.ai.toolkit % echo $DYLD_LIBRARY_PATH

// guoliufang@LiuFang lite.ai.toolkit % export DYLD_LIBRARY_PATH=/Users/guoliufang/Documents/Pay/Sell-SDK/offline-sdk/lite.ai.toolkit/build/lite.ai.toolkit/lib:$DYLD_LIBRARY_PATH
// guoliufang@LiuFang lite.ai.toolkit % echo $DYLD_LIBRARY_PATH                                                                                                                   
// /Users/guoliufang/Documents/Pay/Sell-SDK/offline-sdk/lite.ai.toolkit/build/lite.ai.toolkit/lib:
// guoliufang@LiuFang lite.ai.toolkit % g++ -std=c++14 -I./build/lite.ai.toolkit/include -I./build/lite.ai.toolkit/opencv2 -I./build/lite.ai.toolkit/include/onnxruntime -I./ncnn test_ncnn_nanodet.cpp -L./build/lite.ai.toolkit/lib -llite.ai.toolkit -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lncnn -lonnxruntime -o test_ncnn_nanodet
// guoliufang@LiuFang lite.ai.toolkit % ./test_ncnn_nanodet
// Testing NCNN NanoDet...
// LITENCNN_DEBUG LogId: /Users/guoliufang/Documents/Pay/Sell-SDK/offline-sdk/lite.ai.toolkit/hub/ncnn/cv/nanodet_m-opt.param
// =============== Input-Dims ==============
// Input: input.1: shape: c=0 h=0 w=0
// =============== Output-Dims ==============
// Output: cls_pred_stride_8: shape: c=0 h=0 w=0
// Output: dis_pred_stride_8: shape: c=0 h=0 w=0
// Output: cls_pred_stride_16: shape: c=0 h=0 w=0
// Output: dis_pred_stride_16: shape: c=0 h=0 w=0
// Output: cls_pred_stride_32: shape: c=0 h=0 w=0
// Output: dis_pred_stride_32: shape: c=0 h=0 w=0
// ========================================
// generate_bboxes num: 30
// NCNN Version Detected Boxes Num: 5
// Result saved to: /Users/guoliufang/Documents/Pay/Sell-SDK/offline-sdk/lite.ai.toolkit/logs/test_ncnn_nanodet.jpg
// guoliufang@LiuFang lite.ai.toolkit % 