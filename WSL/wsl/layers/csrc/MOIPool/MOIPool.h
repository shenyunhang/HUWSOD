// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#pragma once
#include <torch/types.h>

namespace wsl {

std::tuple<at::Tensor, at::Tensor> MOIPool_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const at::Tensor& oh_labels,
    const at::Tensor& superpixels);

at::Tensor MOIPool_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width);

#if defined(WITH_CUDA) || defined(WITH_HIP)
std::tuple<at::Tensor, at::Tensor> MOIPool_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const at::Tensor& oh_labels,
    const at::Tensor& superpixels);

at::Tensor MOIPool_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width);
#endif

// Interface for Python
inline std::tuple<at::Tensor, at::Tensor> MOIPool_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const at::Tensor& oh_labels,
    const at::Tensor& superpixels) {
  if (input.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return MOIPool_forward_cuda(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        oh_labels,
        superpixels);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not compiled with CPU support");
  return MOIPool_forward_cpu(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      oh_labels,
      superpixels);
}

inline at::Tensor MOIPool_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width) {
  if (grad.is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return MOIPool_backward_cuda(
        grad,
        rois,
        argmax,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not compiled with CPU support");
  return MOIPool_backward_cpu(
      grad,
      rois,
      argmax,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width);
}

} // namespace wsl
