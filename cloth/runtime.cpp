module;

#include <cuda_runtime_api.h>

module xayah.cloth.runtime;

import std;

namespace xayah::cloth {

    Resource::Resource() : stream_(nullptr), memory_pool_(nullptr) {
        int device;
        if (const cudaError_t result = cudaGetDevice(&device); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        cudaMemPoolProps properties{};
        properties.allocType     = cudaMemAllocationTypePinned;
        properties.handleTypes   = cudaMemHandleTypeNone;
        properties.location.type = cudaMemLocationTypeDevice;
        properties.location.id   = device;

        cudaMemPool_t memory_pool;
        if (const cudaError_t result = cudaMemPoolCreate(&memory_pool, &properties); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        memory_pool_ = memory_pool;

        std::uint64_t release_threshold = std::numeric_limits<std::uint64_t>::max();
        if (const cudaError_t result = cudaMemPoolSetAttribute(memory_pool, cudaMemPoolAttrReleaseThreshold, &release_threshold); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));

        cudaStream_t stream;
        if (const cudaError_t result = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        stream_ = stream;
    }

    Resource::~Resource() noexcept {
        if (cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)) != cudaSuccess) std::terminate();
        if (cudaStreamDestroy(static_cast<cudaStream_t>(stream_)) != cudaSuccess) std::terminate();
        if (cudaMemPoolDestroy(static_cast<cudaMemPool_t>(memory_pool_)) != cudaSuccess) std::terminate();
    }

    void* Resource::allocate(const std::size_t bytes) {
        void* allocation;
        if (const cudaError_t result = cudaMallocFromPoolAsync(&allocation, bytes, static_cast<cudaMemPool_t>(memory_pool_), static_cast<cudaStream_t>(stream_)); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        return allocation;
    }

    void Resource::release(void* allocation) noexcept {
        if (cudaFreeAsync(allocation, static_cast<cudaStream_t>(stream_)) != cudaSuccess) std::terminate();
    }

    void Resource::copy_from_host(void* destination, const void* source, const std::size_t bytes) {
        if (const cudaError_t result = cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream_)); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    void Resource::copy_to_host(void* destination, const void* source, const std::size_t bytes) {
        if (const cudaError_t result = cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream_)); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    void Resource::copy_device(void* destination, const void* source, const std::size_t bytes) {
        if (const cudaError_t result = cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream_)); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    void Resource::zero(void* destination, const std::size_t bytes) {
        if (const cudaError_t result = cudaMemsetAsync(destination, 0, bytes, static_cast<cudaStream_t>(stream_)); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    void Resource::synchronize() {
        if (const cudaError_t result = cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)); result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }

    void* Resource::native_stream() const {
        return stream_;
    }

    template <BufferElement Element>
    Buffer<Element>::Buffer() noexcept : resource_(), data_(nullptr), size_(0) {}

    template <BufferElement Element>
    Buffer<Element>::Buffer(std::shared_ptr<Resource> resource, const std::size_t size) : resource_(std::move(resource)), data_(size == 0 ? nullptr : static_cast<Element*>(resource_->allocate(size * sizeof(Element)))), size_(size) {}

    template <BufferElement Element>
    Buffer<Element>::~Buffer() noexcept {
        if (data_ != nullptr) resource_->release(data_);
    }

    template <BufferElement Element>
    Buffer<Element>::Buffer(Buffer&& other) noexcept : resource_(std::move(other.resource_)), data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}

    template <BufferElement Element>
    Buffer<Element>& Buffer<Element>::operator=(Buffer&& other) noexcept {
        if (data_ != nullptr) resource_->release(data_);
        resource_ = std::move(other.resource_);
        data_     = std::exchange(other.data_, nullptr);
        size_     = std::exchange(other.size_, 0);
        return *this;
    }

    template <BufferElement Element>
    Element* Buffer<Element>::data() {
        return data_;
    }

    template <BufferElement Element>
    const Element* Buffer<Element>::data() const {
        return data_;
    }

    template <BufferElement Element>
    std::size_t Buffer<Element>::size() const {
        return size_;
    }

    template class Buffer<float>;
    template class Buffer<std::uint32_t>;

} // namespace xayah::cloth
