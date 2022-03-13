#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

#include "cuda_helpers.h"

template <typename T>
__global__ void MoIPoolForward(
    const int nthreads,
    const T* input,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const T* rois,
    T* output,
    int* argmax_data,
    const int* oh_labels,
    const int* superpixels,
    const int len_labels) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = round(offset_rois[1] * spatial_scale);
    int roi_start_h = round(offset_rois[2] * spatial_scale);
    int roi_end_w = round(offset_rois[3] * spatial_scale);
    int roi_end_h = round(offset_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    if (is_empty) {
      output[index] = 0;
      argmax_data[index] = -1;
      continue;
    }

    // Define an empty pooling region to be zero
    T maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    const int* offset_spp = superpixels +
        roi_batch_ind * int(height / spatial_scale * width / spatial_scale);

    for (int h = roi_start_h; h < roi_end_h; ++h) {
      for (int w = roi_start_w; w < roi_end_w; ++w) {
        int input_index = h * width + w;

        int spp_index =
            h / spatial_scale * width / spatial_scale + w / spatial_scale;
        int oh_labels_id = offset_spp[spp_index];
        if (oh_labels[n * len_labels + oh_labels_id] == 0) {
          continue;
        }

        int t_roi_height = 0;
        int t_h = 0;
        for (int hh = roi_start_h; hh < roi_end_h; ++hh) {
          int spp_index =
              hh / spatial_scale * width / spatial_scale + w / spatial_scale;
          int oh_labels_id = offset_spp[spp_index];
          if (oh_labels[n * len_labels + oh_labels_id] == 0) {
            continue;
          }
          t_roi_height++;
          if (h == hh) {
            t_h = t_roi_height;
          }
        }
        if (1.0 * t_roi_height / pooled_height * ph > t_h ||
            1.0 * t_roi_height / pooled_height * (ph + 1) < t_h) {
          continue;
        }

        int t_roi_width = 0;
        int t_w = 0;
        for (int ww = roi_start_w; ww < roi_end_w; ++ww) {
          int spp_index =
              h / spatial_scale * width / spatial_scale + ww / spatial_scale;
          int oh_labels_id = offset_spp[spp_index];
          if (oh_labels[n * len_labels + oh_labels_id] == 0) {
            continue;
          }
          t_roi_width++;
          if (w == ww) {
            t_w = t_roi_width;
          }
        }
        if (1.0 * t_roi_width / pooled_width * pw > t_w ||
            1.0 * t_roi_width / pooled_width * (pw + 1) < t_w) {
          continue;
        }

        if (offset_input[input_index] > maxval) {
          maxval = offset_input[input_index];
          maxidx = input_index;
        }
      }
    }
    output[index] = maxval;
    argmax_data[index] = maxidx;

    if (maxidx == -1) {
      output[index] = 0;
      argmax_data[index] = -1;
    }
  }
}

template <typename T>
__global__ void MoIForward(
    const int nthreads,
    const T spatial_scale,
    const int height,
    const int width,
    const int height_sp,
    const int width_sp,
    const T* rois,
    int* mois,
    const int* oh_labels,
    const int* superpixels,
    const int len_labels) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, h, w) is an element in the pooled mois
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = round(offset_rois[1] * spatial_scale);
    int roi_start_h = round(offset_rois[2] * spatial_scale);
    int roi_end_w = round(offset_rois[3] * spatial_scale);
    int roi_end_h = round(offset_rois[4] * spatial_scale);

    if (w >= roi_start_w && w <= roi_end_w && h >= roi_start_h &&
        h <= roi_end_h) {
    } else {
      mois[index] = 0;
      continue;
    }

    const int* offset_spp = superpixels + roi_batch_ind * height_sp * width_sp;

    const int* offset_oh_labels = oh_labels + n * len_labels;

    T spatial_scale_sp = 1.0 * height / height_sp;

    int hhstart = static_cast<int>(floor(static_cast<T>(h) / spatial_scale_sp));
    int wwstart = static_cast<int>(floor(static_cast<T>(w) / spatial_scale_sp));
    int hhend =
        static_cast<int>(ceil(static_cast<T>(h + 1) / spatial_scale_sp));
    int wwend =
        static_cast<int>(ceil(static_cast<T>(w + 1) / spatial_scale_sp));
    hhstart = min(max(hhstart, 0), height_sp);
    hhend = min(max(hhend, 0), height_sp);
    wwstart = min(max(wwstart, 0), width_sp);
    wwend = min(max(wwend, 0), width_sp);

    bool is_valid = false;
    int prev_id = -1;
    for (int hh = hhstart; hh < hhend; ++hh) {
      for (int ww = wwstart; ww < wwend; ++ww) {
        int spp_index = hh * width_sp + ww;
        int oh_labels_id = offset_spp[spp_index];
        if (oh_labels_id == prev_id) {
          continue;
        }
        prev_id = oh_labels_id;
        if (offset_oh_labels[oh_labels_id] == 1) {
          is_valid = true;
        }
        if (is_valid) {
          break;
        }
      }
      if (is_valid) {
        break;
      }
    }
    if (is_valid) {
      mois[index] = 1;
    } else {
      mois[index] = 0;
    }
  }
}

template <typename T>
__global__ void MoIPoolForward2(
    const int nthreads,
    const T* input,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const T* rois,
    T* output,
    int* argmax_data,
    const int* mois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = round(offset_rois[1] * spatial_scale);
    int roi_start_h = round(offset_rois[2] * spatial_scale);
    int roi_end_w = round(offset_rois[3] * spatial_scale);
    int roi_end_h = round(offset_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    T maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    const int* offset_mois = mois + n * height * width;

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int input_index = h * width + w;

        if (offset_mois[input_index] == 0) {
          continue;
        }

        if (offset_input[input_index] > maxval) {
          maxval = offset_input[input_index];
          maxidx = input_index;
        }
      }
    }
    output[index] = maxval;
    argmax_data[index] = maxidx;

    if (maxidx == -1) {
      output[index] = 0;
      argmax_data[index] = -1;
    }
  }
}

template <typename T>
__global__ void RoIPoolBackward(
    const int nthreads,
    const T* grad_output,
    const int* argmax_data,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T* grad_input,
    const T* rois,
    const int n_stride,
    const int c_stride,
    const int h_stride,
    const int w_stride) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    T* grad_input_offset =
        grad_input + ((roi_batch_ind * channels + c) * height * width);

    int output_offset = n * n_stride + c * c_stride;
    const int* argmax_data_offset =
        argmax_data + (n * channels + c) * pooled_height * pooled_width;
    int argmax = argmax_data_offset[ph * pooled_width + pw];

    if (argmax != -1) {
      atomicAdd(
          grad_input_offset + argmax,
          static_cast<T>(
              grad_output[output_offset + ph * h_stride + pw * w_stride]));
    }
  }
}

namespace wsl {

std::tuple<at::Tensor, at::Tensor> MOIPool_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const at::Tensor& oh_labels,
    const at::Tensor& superpixels) {
  AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.is_cuda(), "rois must be a CUDA tensor");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "MOIPool_forward_cuda";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  at::cuda::CUDAGuard device_guard(input.device());

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  at::Tensor argmax = at::zeros(
      {num_rois, channels, pooled_height, pooled_width},
      input.options().dtype(at::kInt));

  auto output_size = num_rois * pooled_height * pooled_width * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(output, argmax);
  }

  auto input_ = input.contiguous(), rois_ = rois.contiguous();

  auto len_labels = oh_labels.size(1);
  auto superpixels_ = superpixels.contiguous();
  auto oh_labels_ = oh_labels.contiguous();

  auto height_sp = superpixels.size(1);
  auto width_sp = superpixels.size(2);

  at::Tensor mois =
      at::zeros({num_rois, height, width}, input.options().dtype(at::kInt));

  // printf("spatial_scale_sp: %f %d %d\n", 1.0 * height / height_sp, height,
  // height_sp); printf("spatial_scale: %f\n", spatial_scale);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "MOI_forward", [&] {
    MoIForward<scalar_t><<<grid, block, 0, stream>>>(
        num_rois * height * width,
        spatial_scale,
        height,
        width,
        height_sp,
        width_sp,
        rois_.data_ptr<scalar_t>(),
        mois.data_ptr<int>(),
        oh_labels_.data_ptr<int>(),
        superpixels_.data_ptr<int>(),
        len_labels);
  });
  AT_CUDA_CHECK(cudaGetLastError());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "MOIPool_forward", [&] {
        MoIPoolForward2<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            argmax.data_ptr<int>(),
            mois.data_ptr<int>());
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(output, argmax);
}

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
    const int width) {
  // Check if input tensors are CUDA tensors
  AT_ASSERTM(grad.is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.is_cuda(), "rois must be a CUDA tensor");
  AT_ASSERTM(argmax.is_cuda(), "argmax must be a CUDA tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2},
      argmax_t{argmax, "argmax", 3};

  at::CheckedFrom c = "MOIPool_backward_cuda";
  at::checkAllSameGPU(c, {grad_t, rois_t, argmax_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  at::cuda::CUDAGuard device_guard(grad.device());

  auto num_rois = rois.size(0);

  at::Tensor grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int h_stride = grad.stride(2);
  int w_stride = grad.stride(3);

  auto argmax_ = argmax.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "MOIPool_backward", [&] {
        RoIPoolBackward<scalar_t><<<grid, block, 0, stream>>>(
            grad.numel(),
            grad.data_ptr<scalar_t>(),
            argmax_.data_ptr<int>(),
            num_rois,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<scalar_t>(),
            n_stride,
            c_stride,
            h_stride,
            w_stride);
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}

} // namespace wsl
