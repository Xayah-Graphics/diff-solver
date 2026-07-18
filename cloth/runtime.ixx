export module xayah.cloth.runtime;

import std;

export namespace xayah::cloth {

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
    concept BufferElement = std::same_as<Element, float> || std::same_as<Element, std::uint32_t>;

    template <BufferElement Element>
    struct Buffer {
        Buffer() noexcept;
        Buffer(std::shared_ptr<Resource> resource, std::size_t size);
        ~Buffer() noexcept;

        Buffer(const Buffer&)            = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&& other) noexcept;
        Buffer& operator=(Buffer&& other) noexcept;

        Element* data;
        std::size_t size;

    private:
        std::shared_ptr<Resource> resource_;
    };

    extern template struct Buffer<float>;
    extern template struct Buffer<std::uint32_t>;

} // namespace xayah::cloth
