#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#include "crop_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include "image.h"

__device__ float get_pixel_kernel(float *image, int w, int h, int x, int y, int c)
{
    if(x < 0 || x >= w || y < 0 || y >= h) return 0;
    return image[x + w*(y + c*h)];
}

__device__ float3 rgb_to_hsv_kernel(float3 rgb)
{
    auto r = rgb.x;
    auto g = rgb.y;
    auto b = rgb.z;

    float h, s;
    const auto max = (r > g) ? ( (r > b) ? r : b) : ( (g > b) ? g : b);
    const auto min = (r < g) ? ( (r < b) ? r : b) : ( (g < b) ? g : b);
    const auto delta = max - min;
    float v = max;
    if(max == 0){
        s = 0;
        h = -1;
    }else{
        s = delta/max;
        if(r == max){
            h = (g - b) / delta;
        } else if (g == max) {
            h = 2 + (b - r) / delta;
        } else {
            h = 4 + (r - g) / delta;
        }
        if (h < 0) h += 6;
    }
    return make_float3(h, s, v);
}

__device__ float3 hsv_to_rgb_kernel(float3 hsv)
{
    const auto h = hsv.x;
    const auto s = hsv.y;
    const auto v = hsv.z;

    float r, g, b;

    if (s == 0) {
        r = g = b = v;
    } else {
        const auto index = (int) floorf(h);
        float f = h - index;
        float p = v * (1 - s);
        float q = v * (1 - s * f);
        float t = v * (1 - s * (1 - f));
        if(index == 0){
            r = v; g = t; b = p;
        } else if(index == 1){
            r = q; g = v; b = p;
        } else if(index == 2){
            r = p; g = v; b = t;
        } else if(index == 3){
            r = p; g = q; b = v;
        } else if(index == 4){
            r = t; g = p; b = v;
        } else {
            r = v; g = p; b = q;
        }
    }
    r = (r < 0) ? 0 : ((r > 1) ? 1 : r);
    g = (g < 0) ? 0 : ((g > 1) ? 1 : g);
    b = (b < 0) ? 0 : ((b > 1) ? 1 : b);
    return make_float3(r, g, b);
}

__device__ float bilinear_interpolate_kernel(float *image, int w, int h, float x, float y, int c)
{
    const auto ix = (int) floorf(x);
    const auto iy = (int) floorf(y);

    const auto dx = x - ix;
    const auto dy = y - iy;

    const auto val = (1-dy) * (1-dx) * get_pixel_kernel(image, w, h, ix, iy, c) +
        dy     * (1-dx) * get_pixel_kernel(image, w, h, ix, iy+1, c) +
        (1-dy) *   dx   * get_pixel_kernel(image, w, h, ix+1, iy, c) +
        dy     *   dx   * get_pixel_kernel(image, w, h, ix+1, iy+1, c);
    return val;
}

__global__ void levels_image_kernel(float *image, float *rand, int batch, int w, int h, int train, float saturation, float exposure, float translate, float scale, float shift)
{
    auto size = batch * w * h;
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;
    id /= h;
    const auto rshift = rand[0];
    const auto gshift = rand[1];
    const auto bshift = rand[2];
    float r0 = rand[8*id + 0];
    float r1 = rand[8*id + 1];
    float r2 = rand[8*id + 2];
    float r3 = rand[8*id + 3];

    saturation = r0*(saturation - 1) + 1;
    saturation = (r1 > .5) ? 1./saturation : saturation;
    exposure = r2*(exposure - 1) + 1;
    exposure = (r3 > .5) ? 1./exposure : exposure;

    size_t offset = id * h * w * 3;
    image += offset;
    const float r = image[x + w*(y + h*0)];
    const float g = image[x + w*(y + h*1)];
    const float b = image[x + w*(y + h*2)];
    float3 rgb = make_float3(r,g,b);
    if(train){
        float3 hsv = rgb_to_hsv_kernel(rgb);
        hsv.y *= saturation;
        hsv.z *= exposure;
        rgb = hsv_to_rgb_kernel(hsv);
    } else {
        shift = 0;
    }
    image[x + w*(y + h*0)] = rgb.x*scale + translate + (rshift - .5)*shift;
    image[x + w*(y + h*1)] = rgb.y*scale + translate + (gshift - .5)*shift;
    image[x + w*(y + h*2)] = rgb.z*scale + translate + (bshift - .5)*shift;
}

__global__ void forward_crop_layer_kernel(float *input, float *rand, int size, int c, int h, int w, int crop_height, int crop_width, int train, int flip, float angle, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;

    float cx = w/2.;
    float cy = h/2.;

    int count = id;
    int j = id % crop_width;
    id /= crop_width;
    int i = id % crop_height;
    id /= crop_height;
    const int k = id % c;
    id /= c;
    int b = id;

    float r4 = rand[8*b + 4];
    float r5 = rand[8*b + 5];
    float r6 = rand[8*b + 6];
    float r7 = rand[8*b + 7];

    float dw = (w - crop_width)*r4;
    float dh = (h - crop_height)*r5;
    flip = (flip && (r6 > .5));
    angle = 2*angle*r7 - angle;
    if(!train){
        dw = (w - crop_width)/2.;
        dh = (h - crop_height)/2.;
        flip = 0;
        angle = 0;
    }

    input += w*h*c*b;

    float x = (flip) ? w - dw - j - 1 : j + dw;
    float y = i + dh;

    const float rx = cos(angle)*(x-cx) - sin(angle)*(y-cy) + cx;
    const float ry = sin(angle)*(x-cx) + cos(angle)*(y-cy) + cy;

    output[count] = bilinear_interpolate_kernel(input, w, h, rx, ry, k);
}

extern "C" void forward_crop_layer_gpu(crop_layer layer, network_state state)
{
    cuda_random(layer.rand_gpu, layer.batch*8);

    float radians = layer.angle*3.14159265/180.;

    float scale = 2;
    float translate = -1;
    if(layer.noadjust){
        scale = 1;
        translate = 0;
    }

    auto size = layer.batch * layer.w * layer.h;

    levels_image_kernel<<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >>>(state.input, layer.rand_gpu, layer.batch, layer.w, layer.h, state.train, layer.saturation, layer.exposure, translate, scale, layer.shift);
    CHECK_CUDA(cudaPeekAtLastError());

    size = layer.batch*layer.c*layer.out_w*layer.out_h;

    forward_crop_layer_kernel<<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >>>(state.input, layer.rand_gpu, size, layer.c, layer.h, layer.w, layer.out_h, layer.out_w, state.train, layer.flip, radians, layer.output_gpu);
    CHECK_CUDA(cudaPeekAtLastError());

/*
       cuda_pull_array(layer.output_gpu, layer.output, size);
       image im = float_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 0*(size/layer.batch));
       image im2 = float_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 1*(size/layer.batch));
       image im3 = float_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 2*(size/layer.batch));

       translate_image(im, -translate);
       scale_image(im, 1/scale);
       translate_image(im2, -translate);
       scale_image(im2, 1/scale);
       translate_image(im3, -translate);
       scale_image(im3, 1/scale);

       show_image(im, "cropped");
       show_image(im2, "cropped2");
       show_image(im3, "cropped3");
       cvWaitKey(0);
       */
}
