//! Upload staging.
//!
//! Uploads happen at a resource boundary (create, or a world revision bump), not
//! per draw, so this is the one part of the resource layer that may allocate. It
//! still allocates from the host allocator, and the pool reuses blocks so a
//! revision bump does not turn into a churn of host allocations.

use reconl_core::alloc::{HostAlloc, HostVec};
use reconl_core::error::{Code, Error, Result};

pub struct StagingBuffer {
    bytes: HostVec<u8>,
    used: usize,
    generation: u64,
}

impl StagingBuffer {
    pub fn new(alloc: HostAlloc, capacity: usize) -> Result<Self> {
        let mut bytes = HostVec::new(alloc);
        bytes.try_reserve(capacity.max(1))?;
        Ok(Self { bytes, used: 0, generation: 0 })
    }

    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn reset(&mut self) {
        self.used = 0;
        self.generation += 1;
    }

    /// Copies `data` in and returns the offset it landed at.
    pub fn stage(&mut self, data: &[u8]) -> Result<usize> {
        let needed = self.used.checked_add(data.len()).ok_or_else(|| {
            Error::new(Code::OutOfMemory, "staging offset overflow")
        })?;
        if needed > self.bytes.capacity() {
            return Err(Error::new(Code::BudgetExceeded, "staging buffer is full"));
        }
        self.bytes.resize_with(needed, || 0)?;
        self.bytes.as_mut_slice()[self.used..needed].copy_from_slice(data);
        let at = self.used;
        self.used = needed;
        Ok(at)
    }

    pub fn staged(&self) -> &[u8] {
        &self.bytes.as_slice()[..self.used]
    }
}

/// A small pool of staging buffers, one per in-flight upload.
pub struct StagingPool {
    alloc: HostAlloc,
    buffers: Vec<StagingBuffer>,
    block_bytes: usize,
}

impl StagingPool {
    pub fn new(alloc: HostAlloc, block_bytes: usize, count: usize) -> Result<Self> {
        let mut buffers = Vec::with_capacity(count);
        for _ in 0..count {
            buffers.push(StagingBuffer::new(alloc, block_bytes)?);
        }
        Ok(Self { alloc, buffers, block_bytes })
    }

    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    pub fn take(&mut self) -> Result<&mut StagingBuffer> {
        // Round-robin by least-recently reset; simple and sufficient because the
        // caller holds at most one buffer at a time.
        if let Some(index) = self
            .buffers
            .iter()
            .enumerate()
            .min_by_key(|(_, b)| b.generation())
            .map(|(i, _)| i)
        {
            self.buffers[index].reset();
            return Ok(&mut self.buffers[index]);
        }
        self.buffers.push(StagingBuffer::new(self.alloc, self.block_bytes)?);
        let index = self.buffers.len() - 1;
        self.buffers[index].reset();
        Ok(&mut self.buffers[index])
    }

    pub fn total_capacity(&self) -> usize {
        self.buffers.iter().map(|b| b.capacity()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_copies_in_and_reports_offsets() {
        let mut buffer = StagingBuffer::new(HostAlloc::system(), 64).unwrap();
        assert_eq!(buffer.stage(b"hello ").unwrap(), 0);
        assert_eq!(buffer.stage(b"world").unwrap(), 6);
        assert_eq!(buffer.staged(), b"hello world");
        buffer.reset();
        assert_eq!(buffer.staged().len(), 0);
        assert_eq!(buffer.stage(b"again").unwrap(), 0);
    }

    #[test]
    fn a_full_staging_buffer_is_a_returned_error_not_a_crash() {
        let mut buffer = StagingBuffer::new(HostAlloc::system(), 8).unwrap();
        buffer.stage(b"12345678").unwrap();
        let err = buffer.stage(b"x").unwrap_err();
        assert_eq!(err.code, Code::BudgetExceeded);
        assert_eq!(buffer.staged(), b"12345678");
    }

    #[test]
    fn the_pool_reuses_buffers_instead_of_growing() {
        let mut pool = StagingPool::new(HostAlloc::system(), 32, 2).unwrap();
        let before = pool.total_capacity();
        for _ in 0..10 {
            let b = pool.take().unwrap();
            b.stage(b"payload").unwrap();
        }
        assert_eq!(pool.buffers.len(), 2, "the pool must not grow on reuse");
        assert_eq!(pool.total_capacity(), before);
    }
}
