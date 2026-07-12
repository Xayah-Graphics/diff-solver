export module xayah.cloth.runtime;

import std;

export namespace xayah::cloth {

    class Resource {
    public:
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

        [[nodiscard]] void* native_stream() const;

    private:
        void* stream_;
        void* memory_pool_;
    };

    template <typename Element>
    concept BufferElement = std::same_as<Element, float> || std::same_as<Element, std::uint32_t>;

    template <BufferElement Element>
    class Buffer {
    public:
        Buffer() noexcept;
        Buffer(std::shared_ptr<Resource> resource, std::size_t size);
        ~Buffer() noexcept;

        Buffer(const Buffer&)            = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&& other) noexcept;
        Buffer& operator=(Buffer&& other) noexcept;

        [[nodiscard]] Element* data();
        [[nodiscard]] const Element* data() const;
        [[nodiscard]] std::size_t size() const;

    private:
        std::shared_ptr<Resource> resource_;
        Element* data_;
        std::size_t size_;
    };

    extern template class Buffer<float>;
    extern template class Buffer<std::uint32_t>;

} // namespace xayah::cloth
