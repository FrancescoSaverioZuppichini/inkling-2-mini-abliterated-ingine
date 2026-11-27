// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Modified for fngine from cinference:src/core/arena.h @ 383e5db. See NOTICE.
#pragma once

#include "core/dtype.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <vector>

namespace ingine {

struct DeviceSpan {
    void* data        = nullptr;
    std::size_t bytes = 0;
};

// Owning device allocation for long-lived buffers. DeviceArena remains the
// suballocation primitive for workspaces; this type owns exactly one cudaMalloc.
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(std::size_t size_bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    // Completes before returning (like copy_from_host).
    void fill(int byte_value = 0);
    // Completes the upload before returning. Callers must first order any prior device
    // accesses to the destination range.
    void copy_from_host(const void* source, std::size_t count, std::size_t byte_offset = 0);
    void copy_to_host(void* destination, std::size_t count, std::size_t byte_offset = 0) const;

    // Raw access is intentional: Tensor and Weight are non-owning views.
    void* p           = nullptr;
    std::size_t bytes = 0;

private:
    void require_range(std::size_t byte_offset, std::size_t count, const char* operation) const;
};

// Device memory pressure: when the cudaMalloc of a DeviceBuffer fails, the registered release hooks free what
// they can (the Generator drops its parked conversations; each returns whether it freed anything) and the
// allocation is retried once. add_release_hook returns an id for remove_release_hook. fail_next_allocations(n)
// makes the next n DeviceBuffer allocation attempts fail as out of memory (tests).
using ReleaseHook = std::function<bool()>;
int add_release_hook(ReleaseHook hook);
void remove_release_hook(int id);
void fail_next_allocations(int n);

// [Agent] Long-lived buffers packed back to back in one virtual address range that is backed by physical memory 2 MiB
// at a time (CUDA virtual memory management). One cudaMalloc per tensor rounds every allocation of 2 MiB or more up to
// a whole number of 2 MiB pages and puts small ones in pools: on Inkling's ~1,000 weight and drafter tensors that wasted
// ~150 MiB. Here the only slack is the tail of the last page. Buffers are never freed one by one: the heap releases
// everything when it is destroyed. Kernels see ordinary device pointers, so values and speed do not change.
class DeviceHeap {
public:
    DeviceHeap() noexcept = default;
    ~DeviceHeap();
    DeviceHeap(const DeviceHeap&)            = delete;
    DeviceHeap& operator=(const DeviceHeap&) = delete;
    DeviceHeap(DeviceHeap&& other) noexcept;
    DeviceHeap& operator=(DeviceHeap&& other) noexcept;

    // Zero-filled is not promised: like cudaMalloc, contents start undefined.
    void* alloc(std::size_t bytes, std::size_t align = 256);
    std::size_t used() const noexcept { return off_; }
    std::size_t mapped() const noexcept { return mapped_; }

private:
    void release() noexcept;

    std::uintptr_t base_  = 0;  // CUdeviceptr of the reservation
    std::size_t reserved_ = 0, mapped_ = 0, off_ = 0, gran_ = 0;
    int device_           = 0;
    std::vector<unsigned long long> handles_;  // CUmemGenericAllocationHandle per mapped piece
    std::vector<std::size_t> sizes_;
};

class DeviceArena {
public:
    class Scope {
    public:
        ~Scope() noexcept;

        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        friend class DeviceArena;

        explicit Scope(DeviceArena& arena) noexcept;

        DeviceArena* arena_       = nullptr;
        std::size_t saved_offset_ = 0;
    };

    explicit DeviceArena(std::size_t capacity_bytes);
    // Non-owning arena over an already allocated device region.
    explicit DeviceArena(DeviceSpan storage);
    ~DeviceArena();

    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&& other) noexcept;
    DeviceArena& operator=(DeviceArena&& other) noexcept;

    DeviceSpan alloc_bytes(std::size_t bytes, std::size_t align = 256);
    Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t align = 256);
    [[nodiscard]] Scope scope() noexcept;
    void reset() noexcept;

    void* base() const noexcept;
    std::size_t used() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t peak_used() const noexcept;
    void reset_peak() noexcept;

private:
    void* base_       = nullptr;
    std::size_t cap_  = 0;
    std::size_t off_  = 0;
    std::size_t peak_ = 0;
    bool owns_        = true;
};

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t size_bytes);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void* data() const noexcept;
    std::size_t size() const noexcept;

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
};

using WorkspaceArena = DeviceArena;

} // namespace ingine
