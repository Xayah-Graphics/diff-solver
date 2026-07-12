module;

#include <cuda_runtime_api.h>

module xayah.cloth.runtime;

import std;

namespace xayah::cloth {

    namespace {

        void check_cuda(const cudaError_t result) {
            if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
        }

        void terminate_on_cuda_error(const cudaError_t result) noexcept {
            if (result != cudaSuccess) std::terminate();
        }

    } // namespace

    Resource::Resource() : stream_(nullptr), memory_pool_(nullptr) {
        int device;
        check_cuda(cudaGetDevice(&device));
        cudaMemPoolProps properties{};
        properties.allocType     = cudaMemAllocationTypePinned;
        properties.handleTypes   = cudaMemHandleTypeNone;
        properties.location.type = cudaMemLocationTypeDevice;
        properties.location.id   = device;

        cudaMemPool_t memory_pool;
        check_cuda(cudaMemPoolCreate(&memory_pool, &properties));
        memory_pool_ = memory_pool;

        std::uint64_t release_threshold = std::numeric_limits<std::uint64_t>::max();
        check_cuda(cudaMemPoolSetAttribute(memory_pool, cudaMemPoolAttrReleaseThreshold, &release_threshold));

        cudaStream_t stream;
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        stream_ = stream;
    }

    Resource::~Resource() noexcept {
        terminate_on_cuda_error(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
        terminate_on_cuda_error(cudaStreamDestroy(static_cast<cudaStream_t>(stream_)));
        terminate_on_cuda_error(cudaMemPoolDestroy(static_cast<cudaMemPool_t>(memory_pool_)));
    }

    void* Resource::allocate(const std::size_t bytes) {
        void* allocation;
        check_cuda(cudaMallocFromPoolAsync(&allocation, bytes, static_cast<cudaMemPool_t>(memory_pool_), static_cast<cudaStream_t>(stream_)));
        return allocation;
    }

    void Resource::release(void* allocation) noexcept {
        terminate_on_cuda_error(cudaFreeAsync(allocation, static_cast<cudaStream_t>(stream_)));
    }

    void Resource::copy_from_host(void* destination, const void* source, const std::size_t bytes) {
        check_cuda(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream_)));
    }

    void Resource::copy_to_host(void* destination, const void* source, const std::size_t bytes) {
        check_cuda(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream_)));
    }

    void Resource::copy_device(void* destination, const void* source, const std::size_t bytes) {
        check_cuda(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream_)));
    }

    void Resource::zero(void* destination, const std::size_t bytes) {
        check_cuda(cudaMemsetAsync(destination, 0, bytes, static_cast<cudaStream_t>(stream_)));
    }

    void Resource::synchronize() {
        check_cuda(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
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
