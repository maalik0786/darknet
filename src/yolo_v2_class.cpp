#include "darknet.h"
#include "yolo_v2_class.hpp"

#include "network.h"

extern "C" {
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "option_list.h"
#include "stb_image.h"
}
//#include <sys/time.h>

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>

#define NFRAMES 3

//static Detector* detector = NULL;
static std::unique_ptr<Detector> detector;

int init(const char* configuration_filename, const char* weights_filename, const int gpu)
{
    detector.reset(new Detector(configuration_filename, weights_filename, gpu));
    return 1;
}

int detect_image(const char* filename, bbox_t_container& container)
{
    auto detection = detector->detect(filename);
    for (size_t i = 0; i < detection.size() && i < C_SHARP_MAX_OBJECTS; ++i)
        container.candidates[i] = detection[i];
    return detection.size();
}

int detect_objects(const float* data, const int width, const int height, bbox_t_container& container)
{
    // ReSharper disable once CppCStyleCast
    auto detection = detector-> detect(image{ height, width,3, (float*)data });
    for (size_t i = 0; i < detection.size() && i < C_SHARP_MAX_OBJECTS; ++i)
        container.candidates[i] = detection[i];
    return detection.size();
}

int track_objects(const float* data, const int width, const int height, bbox_t_container& container)
{
    auto tracking = detector->tracking_id(detector->detect(image{ height, width,3, (float*)data }));
    for (size_t i = 0; i < tracking.size() && i < C_SHARP_MAX_OBJECTS; ++i)
        container.candidates[i] = tracking[i];
    return tracking.size();
}

int dispose()
{
    //if (detector != NULL) delete detector;
    //detector = NULL;
    detector.reset();
    return 1;
}

int get_device_count()
{
#ifdef GPU
    auto count = 0;
    cudaGetDeviceCount(&count);
    return count;
#else
    return -1;
#endif	// GPU
}

bool built_with_cuda(){
#ifdef GPU
    return true;
#else
    return false;
#endif
}

bool built_with_cudnn(){
#ifdef CUDNN
    return true;
#else
    return false;
#endif
}

bool built_with_opencv(){
#ifdef OPENCV
    return true;
#else
    return false;
#endif
}


int get_device_name(int gpu, char* device_name) {
#ifdef GPU
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, gpu);
    std::string result = prop.name;
    std::copy(result.begin(), result.end(), device_name);
    return 1;
#else
    return -1;
#endif	// GPU
}

#ifdef GPU
void check_cuda(const cudaError_t status)
{
    if (status != cudaSuccess)
    {
        const auto s = cudaGetErrorString(status);
        printf("CUDA Error Prev: %s\n", s);
    }
}
#endif

struct detector_gpu_t
{
    network net;
    image images[NFRAMES];
    float* avg;
    float* predictions[NFRAMES];
    int demo_index;
    unsigned int* track_id;
};

LIB_API Detector::Detector(std::string cfg_filename, std::string weight_filename, int gpu_id) : cur_gpu_id(gpu_id)
{
    detector_gpu_ptr = std::make_shared<detector_gpu_t>();
    auto& detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());

    auto& net = detector_gpu.net;
    net.gpu_index = cur_gpu_id;

    const auto cfg_file = const_cast<char *>(cfg_filename.data());
    const auto weight_file = const_cast<char *>(weight_filename.data());

    net = parse_network_cfg_custom(cfg_file, 1, 0);//slow 1-2s
    load_weights(&net, weight_file);//fast <100ms
  
    set_batch_network(&net, 1);
    net.gpu_index = cur_gpu_id;
    fuse_conv_batchnorm(net);
   
    const auto l = net.layers[net.n - 1];

    
    detector_gpu.avg = static_cast<float *>(calloc(l.outputs, sizeof(float)));
    for (int j = 0; j < NFRAMES; ++j)
        detector_gpu.predictions[j] = static_cast<float*>(calloc(l.outputs, sizeof(float)));
    for (int j = 0; j < NFRAMES; ++j) detector_gpu.images[j] = make_image(1, 1, 3);
    
    detector_gpu.track_id = static_cast<unsigned int *>(calloc(l.classes, sizeof(unsigned int)));
    
    for (int j = 0; j <l.classes; ++j) detector_gpu.track_id[j] = 1;
    detector_gpu.demo_index = 1;
}

LIB_API Detector::~Detector()
{
    detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
    layer l = detector_gpu.net.layers[detector_gpu.net.n - 1];

    free(detector_gpu.track_id);
    
    free(detector_gpu.avg);
    for (int j = 0; j < NFRAMES; ++j) free(detector_gpu.predictions[j]);
    for (int j = 0; j < NFRAMES; ++j) if (detector_gpu.images[j].data) free(detector_gpu.images[j].data);

#ifdef GPU
    int old_gpu_index;
    cudaGetDevice(&old_gpu_index);
    cuda_set_device(detector_gpu.net.gpu_index);
#endif
    free_network(detector_gpu.net);
#ifdef GPU
    cudaSetDevice(old_gpu_index);
#endif
}

LIB_API int Detector::get_net_width() const
{
    detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
    return detector_gpu.net.w;
}

LIB_API int Detector::get_net_height() const
{
    detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
    return detector_gpu.net.h;
}

LIB_API int Detector::get_net_color_depth() const
{
    detector_gpu_t& detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
    return detector_gpu.net.c;
}


LIB_API std::vector<bbox_t> Detector::detect(std::string image_filename, float thresh, bool use_mean) const
{
    const std::shared_ptr<image_t> image_ptr(new image_t, [](image_t* img)
    {
        if (img->data) free(img->data);
        delete img;
    });
    *image_ptr = load_image(image_filename);
    return detect(*image_ptr, thresh, use_mean);
}

static image load_image_stb(char* filename, const int channels)
{
    int w, h, c;
    const auto data = stbi_load(filename, &w, &h, &c, channels);
    if (!data)
        throw std::runtime_error("file not found");
    if (channels) c = channels;
    const auto im = make_image(w, h, c);
    for (auto k = 0; k < c; ++k)
    {
        for (auto j = 0; j < h; ++j)
        {
            for (auto i = 0; i < w; ++i)
            {
                const auto dst_index = i + w * j + w * h * k;
                const auto src_index = k + c * i + c * w * j;
                im.data[dst_index] = (float)data[src_index] / 255.;
            }
        }
    }
    free(data);
    return im;
}

LIB_API image_t Detector::load_image(const std::string image_filename)
{
    const auto input = const_cast<char *>(image_filename.data());
    const auto im = load_image_stb(input, 3);

    image_t img{};
    img.c = im.c;
    img.data = im.data;
    img.h = im.h;
    img.w = im.w;

    return img;
}


LIB_API void Detector::free_image(image_t m)
{
    if (m.data)
       free(m.data);
}

LIB_API std::vector<bbox_t> Detector::detect(const image_t img, const float thresh, const bool use_mean) const
{
    auto& detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
    auto& net = detector_gpu.net;
#ifdef GPU
    int old_gpu_index;
    cudaGetDevice(&old_gpu_index);
    if (cur_gpu_id != old_gpu_index)
        cudaSetDevice(net.gpu_index);

    net.wait_stream = wait_stream; // 1 - wait CUDA-stream, 0 - not to wait
#endif
    //std::cout << "net.gpu_index = " << net.gpu_index << std::endl;
     
    image im;
    im.c = img.c;
    im.data = img.data;
    im.h = img.h;
    im.w = img.w;

    image sized;

    if (net.w == im.w && net.h == im.h)
    {
        //sized = im;
        /*this makes no sense, the image is already fitting perfectly, see below
        sized = make_image(im.w, im.h, im.c);
        memcpy(sized.data, im.data, im.w * im.h * im.c * sizeof(float));
        //*/
    }
    else
        sized = resize_image(im, net.w, net.h);
    auto l = net.layers[net.n - 1];

    const auto x = sized.data;

    const auto prediction = network_predict(net, x);

    if (use_mean)
    {
        memcpy(detector_gpu.predictions[detector_gpu.demo_index], prediction, l.outputs * sizeof(float));
        mean_arrays(detector_gpu.predictions, NFRAMES, l.outputs, detector_gpu.avg);
        l.output = detector_gpu.avg;
        detector_gpu.demo_index = (detector_gpu.demo_index + 1) % NFRAMES;
    }
    //get_region_boxes(l, 1, 1, thresh, detector_gpu.probs, detector_gpu.boxes, 0, 0);
    //if (nms) do_nms_sort(detector_gpu.boxes, detector_gpu.probs, l.w*l.h*l.n, l.classes, nms);

    auto nboxes = 0;
    const auto letterbox = 0;
    const float hier_thresh = 0.5;
    const auto dets = get_network_boxes(&net, sized.w, sized.h, thresh, hier_thresh, nullptr, 1, &nboxes, letterbox);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

    std::vector<bbox_t> bbox_vec;

    for (auto i = 0; i < nboxes; ++i)
    {
        const auto b = dets[i].bbox;
        auto const obj_id = max_index(dets[i].prob, l.classes);
        auto const prob = dets[i].prob[obj_id];
        if (prob > thresh)
        {
            bbox_t bbox{};
            bbox.x = std::max(static_cast<double>(0), (b.x - b.w / 2.) * sized.w);
            bbox.y = std::max(static_cast<double>(0), (b.y - b.h / 2.) * sized.h);
            bbox.w = sized.w * b.w;
            bbox.h = sized.h * b.h;
            bbox.obj_id = obj_id;
            bbox.prob = prob;
            bbox.track_id = 0;
            bbox.frames_counter = 0;
            bbox.x_3d = NAN;
            bbox.y_3d = NAN;
            bbox.z_3d = NAN;

            bbox_vec.push_back(bbox);
        }
    }

    free_detections(dets, nboxes);
    if (sized.data)
        free(sized.data);

#ifdef GPU
    if (cur_gpu_id != old_gpu_index)
        cudaSetDevice(old_gpu_index);
#endif
    return bbox_vec;
}

std::vector<bbox_t> Detector::save_bounding_boxes_into_vector(const image img, const float thresh, const struct layer l, detection* const dets, int nboxes) const
{
    std::vector<bbox_t> bbox_vec;
#ifdef OPENCV
    cv::Mat src = detector -> image_to_mat(img);
    cv::cvtColor(src, src, CV_RGB2BGR);
    cv::Mat croppedImg; 
#endif

    for (auto i = 0; i < nboxes; ++i)
    {
        const auto b = dets[i].bbox;
        auto const obj_id = max_index(dets[i].prob, l.classes);
        auto const prob = dets[i].prob[obj_id];
        auto const bbox_x = std::max(static_cast<double>(0), (b.x - b.w / 2.)* img.w);
        auto const bbox_y = std::max(static_cast<double>(0), (b.y - b.h / 2.)* img.h);
        auto const bbox_w = img.w * b.w;
        auto const bbox_h = img.h * b.h;
#ifdef OPENCV
        if (bbox_x - 10 > 0 && bbox_y - 10 > 0 && bbox_w - 30 && bbox_h - 30)
        croppedImg = src(cv::Rect(bbox_x - 10, bbox_y -10, bbox_w + 30, bbox_h + 30));
        else
            croppedImg = src(cv::Rect(bbox_x, bbox_y, bbox_w, bbox_h));
  
        auto shape = detector-> detect_shape(croppedImg);
        bbox_t bbox{};
#endif
        if (prob > thresh)
        {
            bbox.x = bbox_x;
            bbox.y = bbox_y;
            bbox.w = bbox_w;
            bbox.h = bbox_h;
            bbox.obj_id = obj_id;
            bbox.prob = prob;
            bbox.track_id = 0;
            bbox.frames_counter = 0;
            bbox.x_3d = NAN;
            bbox.y_3d = NAN;
            bbox.z_3d = NAN;
#ifdef OPENCV
            static_cast<char>(shape);
#else
            static_cast<char>(None);
#endif
        }
        bbox_vec.push_back(bbox);
    }
    return bbox_vec;
}

LIB_API std::vector<bbox_t> Detector::detect(const image img, const float thresh) const
{
    auto& detector_gpu = *static_cast<detector_gpu_t*>(detector_gpu_ptr.get());
    auto& net = detector_gpu.net;
    net.wait_stream = wait_stream;
    const auto l = net.layers[net.n - 1];
    network_predict_gpu(net, img.data);
    
    auto nboxes = 0;
    const auto dets = get_network_boxes(&net, img.w, img.h, thresh, 0.5, nullptr, 1, &nboxes, 0);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
    std::vector<bbox_t> bbox_vec = save_bounding_boxes_into_vector(img, thresh, l, dets, nboxes);
    
    free_detections(dets, nboxes);
    return bbox_vec;
}

LIB_API std::vector<bbox_t> Detector::tracking_id(std::vector<bbox_t> cur_bbox_vec, bool const change_history, int const frames_story, int const max_dist)
{
    detector_gpu_t& det_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());

    bool prev_track_id_present = false;
    unsigned int frame_id  = det_gpu.demo_index++;
    // TODO: call the shape_detector
    for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
        cur_bbox_vec[i].frames_counter = frame_id;
    // add shape into vector

    for (auto& i : prev_bbox_vec_deque)
        if (i.size() > 0) prev_track_id_present = true;

    if (!prev_track_id_present)
    {
        for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
            cur_bbox_vec[i].track_id = det_gpu.track_id[cur_bbox_vec[i].obj_id]++;
        prev_bbox_vec_deque.push_front(cur_bbox_vec);
        if (prev_bbox_vec_deque.size() > frames_story) prev_bbox_vec_deque.pop_back();
        return cur_bbox_vec;
    }

    std::vector<unsigned int> dist_vec(cur_bbox_vec.size(), std::numeric_limits<unsigned int>::max());

    for (auto& prev_bbox_vec : prev_bbox_vec_deque)
    {
        for (auto& i : prev_bbox_vec)
        {
            int cur_index = -1;
            for (size_t m = 0; m < cur_bbox_vec.size(); ++m)
            {
                bbox_t const& k = cur_bbox_vec[m];
                if (i.obj_id == k.obj_id)
                {
                    float center_x_diff = (float)(i.x + i.w / 2) - (float)(k.x + k.w / 2);
                    float center_y_diff = (float)(i.y + i.h / 2) - (float)(k.y + k.h / 2);
                    unsigned int cur_dist = sqrt(center_x_diff * center_x_diff + center_y_diff * center_y_diff);
                    if (cur_dist < max_dist && (k.track_id == 0 || dist_vec[m] > cur_dist))
                    {
                        dist_vec[m] = cur_dist;
                        cur_index = m;
                    }
                }
            }

            bool track_id_absent = !std::any_of(cur_bbox_vec.begin(), cur_bbox_vec.end(),[&i](bbox_t const& b)
            {
                return b.track_id == i.track_id && b.obj_id == i.obj_id;
            });

            if (cur_index >= 0 && track_id_absent)
            {
                cur_bbox_vec[cur_index].track_id = i.track_id;
                cur_bbox_vec[cur_index].w = (cur_bbox_vec[cur_index].w + i.w) / 2;
                cur_bbox_vec[cur_index].h = (cur_bbox_vec[cur_index].h + i.h) / 2;
            }
        }
    }

    for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
        if (cur_bbox_vec[i].track_id == 0)
            cur_bbox_vec[i].track_id = det_gpu.track_id[cur_bbox_vec[i].obj_id]++;

    if (change_history)
    {
        prev_bbox_vec_deque.push_front(cur_bbox_vec);
        if (prev_bbox_vec_deque.size() > frames_story) prev_bbox_vec_deque.pop_back();
    }
    return cur_bbox_vec;
}

#ifdef OPENCV
shape_type Detector::detect_shape(cv::Mat src)
{
    if (src.empty()) return None;

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, CV_BGR2GRAY);
   
    // Use Canny instead of threshold to catch squares with gradient shading
    cv::Mat bw;
    cv::Canny(gray, bw, 0, 50, 5);
   
    // Find contours
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> approx;
    cv::Mat dst = src.clone();

    for (int i = 0; i < contours.size(); i++)
    {
        // Approximate contour with accuracy proportional
        // to the contour perimeter
        cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) * 0.02, true);

        // Skip small or non-convex objects 
        if (std::fabs(cv::contourArea(contours[i])) < 100 || !cv::isContourConvex(approx))
           continue;

        if (approx.size() == 3)
            return Triangle;

        else if (approx.size() >= 4 && approx.size() <= 6)
        {
            // Number of vertices of polygonal curve
            int vtc = approx.size();

            // Get the cosines of all corners
            std::vector<double> cos;
            for (int j = 2; j < vtc + 1; j++)
                cos.push_back(detector -> shape_angle(approx[j % vtc], approx[j - 2], approx[j - 1]));

            // Sort ascending the cosine values
            std::sort(cos.begin(), cos.end());

            // Get the lowest and the highest cosine
            double mincos = cos.front();
            double maxcos = cos.back();

            // Use the degrees obtained above and the number of vertices
            // to determine the shape of the contour
            if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.3)
                return Rectangle;

            else if (vtc == 5 && mincos >= -0.34 && maxcos <= -0.27)
                return Penta;

            else if (vtc == 6 && mincos >= -0.55 && maxcos <= -0.45) 
                return Hexa;
        }

        else
        {
            // Detect and label circles
            double area = cv::contourArea(contours[i]);
            cv::Rect r = cv::boundingRect(contours[i]);
            int radius = r.width / 2;

            if (std::abs(1 - ((double)r.width / r.height)) <= 0.2 &&
                std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
                return Circle;
        }
    }
    return None;
}

cv::Mat Detector:: image_to_mat(image img)
{
    int channels = img.c;
    int width = img.w;
    int height = img.h;
    cv::Mat mat = cv::Mat(height, width, CV_8UC(channels));
    int step = mat.step;

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < img.c; ++c) {
                float val = img.data[c * img.h * img.w + y * img.w + x];
                mat.data[y * step + x * img.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return mat;
}

double Detector::shape_angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}
#endif

void* Detector::get_cuda_context() const
{
#ifdef GPU
    int old_gpu_index;
    cudaGetDevice(&old_gpu_index);
    if (cur_gpu_id != old_gpu_index)
        cudaSetDevice(cur_gpu_id);

    void* cuda_context = cuda_get_context();

    if (cur_gpu_id != old_gpu_index)
        cudaSetDevice(old_gpu_index);

    return cuda_context;
#else   // GPU
    return NULL;
#endif  // GPU
}
