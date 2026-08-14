export module xayah.cuda;

import std;

export namespace xayah::cuda {

    struct Resource {
        Resource();
        ~Resource() noexcept;

        Resource(const Resource&)            = delete;
        Resource& operator=(const Resource&) = delete;
        Resource(Resource&&)                 = delete;
        Resource& operator=(Resource&&)      = delete;

        [[nodiscard]] void* allocate(std::size_t bytes);
        void release(void* allocation) noexcept;
        void copy_from_host(void* destination, const void* source, std::size_t bytes);
        void copy_to_host(void* destination, const void* source, std::size_t bytes);
        void copy_device(void* destination, const void* source, std::size_t bytes);
        void zero(void* destination, std::size_t bytes);
        void synchronize();

        void* native_stream;

    private:
        void* memory_pool_;
    };

    template <typename Element>
        requires std::is_trivially_copyable_v<Element>
    struct Buffer {
        Buffer() noexcept : data(nullptr), size(0), resource_() {}

        Buffer(std::shared_ptr<Resource> resource, const std::size_t next_size) : data(next_size == 0u ? nullptr : static_cast<Element*>(resource->allocate(next_size * sizeof(Element)))), size(next_size), resource_(std::move(resource)) {}

        ~Buffer() noexcept {
            if (data != nullptr) resource_->release(data);
        }

        Buffer(const Buffer&)            = delete;
        Buffer& operator=(const Buffer&) = delete;

        Buffer(Buffer&& other) noexcept : data(std::exchange(other.data, nullptr)), size(std::exchange(other.size, 0u)), resource_(std::move(other.resource_)) {}

        Buffer& operator=(Buffer&& other) noexcept {
            if (data != nullptr) resource_->release(data);
            resource_ = std::move(other.resource_);
            data      = std::exchange(other.data, nullptr);
            size      = std::exchange(other.size, 0u);
            return *this;
        }

        Element* data;
        std::size_t size;

    private:
        std::shared_ptr<Resource> resource_;
    };

} // namespace xayah::cuda
