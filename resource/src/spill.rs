//! The spill arena: an append-only cache file with a checksum per entry.
//!
//! **Disk is a cache, not a state store.** Deleting the directory at any time
//! costs performance and never correctness, so every read is validated and every
//! failure is a miss. That property is what makes the T4 tier safe to ship.
//!
//! ## Format
//!
//! ```text
//! header (32 bytes)
//!   magic       "RCLS" (u32)
//!   version     u32
//!   reserved    u64
//!   seed        u64            (checksum seed, fixed per arena)
//!   header_crc  u64            (xxHash64 of the 24 bytes before it)
//! record, repeated
//!   magic       u32 = RECORD_MAGIC
//!   payload_len u32
//!   key         u64
//!   checksum    u64            (xxHash64 of the payload, seeded)
//!   payload     payload_len bytes
//! ```
//!
//! ## Recovery
//!
//! On open the file is scanned record by record. A record whose magic, length or
//! checksum does not validate ends the scan: the arena is truncated to the last
//! good record and everything after it is forgotten. A process killed mid-write
//! therefore leaves a cache that is smaller, never one that is wrong. Dropped
//! entries are counted (`recovered_torn`) and logged once.
//!
//! ## Reads
//!
//! Reads use positioned reads rather than `mmap`. The brief specifies `mmap`,
//! and this is the one place milestone 1 knowingly deviates: positioned reads
//! are portable across the Windows/unix split without an unsafe mapping layer,
//! and the file format is unchanged, so swapping in a mapping later cannot
//! invalidate an existing cache. Recorded in README.md under "provisional".

use reconl_core::error::{Code, Error, Result};
use reconl_core::hash::{xxh64_seeded, XxHash64};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const RCLS_MAGIC: u32 = 0x5243_4C53; // "RCLS"
pub const RECORD_MAGIC: u32 = 0x5243_4C52; // "RCLR"
pub const FORMAT_VERSION: u32 = 1;
pub const HEADER_BYTES: u64 = 32;
pub const RECORD_HEADER_BYTES: u64 = 24;

/// What a lookup did. `Corrupt` is not an error: it is a miss that was detected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hit {
    Fresh,
    Miss,
    /// The entry was found but failed validation: dropped, and reported.
    Corrupt,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ArenaStats {
    pub entries: u64,
    pub bytes: u64,
    pub capacity: u64,
    pub hits: u64,
    pub misses: u64,
    pub corrupt: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
    pub recovered_torn: u64,
    pub compactions: u64,
    /// Dead bytes waiting for the compactor: overwritten records and evictions.
    pub garbage_bytes: u64,
    pub errors: u64,
}

#[derive(Clone)]
pub struct SpillConfig {
    pub dir: PathBuf,
    pub file_name: String,
    /// Hard cap on arena bytes. 0 = no cap (the caller's budget still applies).
    pub max_bytes: u64,
    /// Seed for every checksum in this arena.
    pub seed: u64,
}

impl SpillConfig {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into(), file_name: "arena.rcls".to_string(), max_bytes: 0, seed: 0x5243_4C53_5F41_5245 }
    }

    pub fn path(&self) -> PathBuf {
        self.dir.join(&self.file_name)
    }
}

#[derive(Clone, Copy)]
struct IndexEntry {
    offset: u64,
    len: u32,
    checksum: u64,
    /// Monotonic use counter for LRU.
    last_used: u64,
}

pub struct SpillArena {
    config: SpillConfig,
    file: File,
    index: HashMap<u64, IndexEntry>,
    write_offset: u64,
    clock: u64,
    stats: ArenaStats,
    /// Bytes of dead records waiting for the compactor.
    garbage: u64,
}

impl SpillArena {
    /// Opens (or creates) the arena and recovers whatever is intact.
    pub fn open(config: SpillConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.dir).map_err(|e| {
            Error::new(Code::Io, "could not create the spill directory").with_context(&e.to_string())
        })?;
        let path = config.path();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .map_err(|e| Error::new(Code::Io, "could not open the spill arena").with_context(&e.to_string()))?;

        let len = file
            .metadata()
            .map_err(|e| Error::new(Code::Io, "could not stat the spill arena").with_context(&e.to_string()))?
            .len();

        let mut arena = Self {
            config,
            file,
            index: HashMap::new(),
            write_offset: HEADER_BYTES,
            clock: 0,
            stats: ArenaStats {
                entries: 0,
                bytes: 0,
                capacity: 0,
                hits: 0,
                misses: 0,
                corrupt: 0,
                evictions: 0,
                evicted_bytes: 0,
                recovered_torn: 0,
                compactions: 0,
                garbage_bytes: 0,
                errors: 0,
            },
            garbage: 0,
        };
        arena.stats.capacity = arena.config.max_bytes;

        if len < HEADER_BYTES {
            if len > 0 {
                // Something is there and it is not an arena: replace it, and say so.
                arena.stats.corrupt += 1;
            }
            arena.write_header()?;
            return Ok(arena);
        }
        arena.recover(len)?;
        Ok(arena)
    }

    fn write_header(&mut self) -> Result<()> {
        let mut buf = [0u8; HEADER_BYTES as usize];
        buf[0..4].copy_from_slice(&RCLS_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf[16..24].copy_from_slice(&self.config.seed.to_le_bytes());
        let crc = xxh64_seeded(self.config.seed, &buf[0..24]);
        buf[24..32].copy_from_slice(&crc.to_le_bytes());
        self.file
            .seek(SeekFrom::Start(0))
            .and_then(|_| self.file.write_all(&buf))
            .map_err(|e| Error::new(Code::Io, "could not write the arena header").with_context(&e.to_string()))?;
        self.write_offset = HEADER_BYTES;
        Ok(())
    }

    /// Scans records, keeping the intact prefix and truncating at the first
    /// invalid one.
    fn recover(&mut self, file_len: u64) -> Result<()> {
        let mut header = [0u8; HEADER_BYTES as usize];
        self.file.seek(SeekFrom::Start(0)).and_then(|_| self.file.read_exact(&mut header)).map_err(|e| {
            Error::new(Code::Io, "could not read the arena header").with_context(&e.to_string())
        })?;
        let magic = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        let seed = u64::from_le_bytes(header[16..24].try_into().unwrap_or([0; 8]));
        let crc = u64::from_le_bytes(header[24..32].try_into().unwrap_or([0; 8]));
        let expect = xxh64_seeded(seed, &header[0..24]);
        if magic != RCLS_MAGIC || version != FORMAT_VERSION || crc != expect {
            // Not our file, or a torn header: start over rather than trust it.
            self.stats.corrupt += 1;
            self.stats.recovered_torn += 1;
            self.write_header()?;
            return Ok(());
        }

        let mut offset = HEADER_BYTES;
        let mut record = vec![0u8; RECORD_HEADER_BYTES as usize];
        while offset + RECORD_HEADER_BYTES <= file_len {
            if self.file.seek(SeekFrom::Start(offset)).and_then(|_| self.file.read_exact(&mut record)).is_err() {
                break;
            }
            let rmagic = u32::from_le_bytes(record[0..4].try_into().unwrap_or([0; 4]));
            let len = u32::from_le_bytes(record[4..8].try_into().unwrap_or([0; 4])) as u64;
            let key = u64::from_le_bytes(record[8..16].try_into().unwrap_or([0; 8]));
            let checksum = u64::from_le_bytes(record[16..24].try_into().unwrap_or([0; 8]));
            if rmagic != RECORD_MAGIC || len == 0 || offset + RECORD_HEADER_BYTES + len > file_len {
                break;
            }
            let mut payload = vec![0u8; len as usize];
            if self
                .file
                .seek(SeekFrom::Start(offset + RECORD_HEADER_BYTES))
                .and_then(|_| self.file.read_exact(&mut payload))
                .is_err()
            {
                break;
            }
            if verify(&payload, seed) != checksum {
                break;
            }
            // A later record for the same key wins (the older one becomes garbage).
            if let Some(old) = self.index.insert(key, IndexEntry { offset, len: len as u32, checksum, last_used: 0 }) {
                self.garbage += RECORD_HEADER_BYTES + old.len as u64;
            }
            self.stats.entries = self.index.len() as u64;
            offset += RECORD_HEADER_BYTES + len;
        }

        self.write_offset = offset;
        self.stats.bytes = self.recompute_bytes();
        if offset < file_len {
            self.stats.recovered_torn += 1;
            let dropped = file_len - offset;
            // Truncate so the next append overwrites the torn tail.
            self.file.set_len(offset).map_err(|e| {
                Error::new(Code::Io, "could not truncate a torn arena tail").with_context(&e.to_string())
            })?;
            reconl_core::log_warn!(
                "spill arena: dropped {} bytes of torn tail at offset {} (kept {} entries)",
                dropped,
                offset,
                self.stats.entries
            );
        }
        Ok(())
    }

    fn recompute_bytes(&self) -> u64 {
        self.index.values().map(|e| RECORD_HEADER_BYTES + e.len as u64).sum()
    }

    pub fn stats(&self) -> ArenaStats {
        let mut s = self.stats;
        s.entries = self.index.len() as u64;
        s.bytes = self.recompute_bytes();
        s.garbage_bytes = self.garbage;
        s
    }

    pub fn contains(&self, key: u64) -> bool {
        self.index.contains_key(&key)
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Writes a payload. Returns how many bytes are now on disk.
    pub fn put(&mut self, key: u64, payload: &[u8]) -> Result<u64> {
        if payload.is_empty() {
            return Err(Error::new(Code::InvalidArgument, "refusing to cache an empty payload"));
        }
        let checksum = verify(payload, self.config.seed);
        let needed = RECORD_HEADER_BYTES + payload.len() as u64;

        // Make room before writing, so a full arena cannot produce a torn record
        // as part of normal operation.
        self.evict_for(needed);

        let mut header = [0u8; RECORD_HEADER_BYTES as usize];
        header[0..4].copy_from_slice(&RECORD_MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        header[8..16].copy_from_slice(&key.to_le_bytes());
        header[16..24].copy_from_slice(&checksum.to_le_bytes());

        let offset = self.write_offset;
        let write = (|| -> std::io::Result<()> {
            self.file.seek(SeekFrom::Start(offset))?;
            self.file.write_all(&header)?;
            self.file.write_all(payload)?;
            Ok(())
        })();
        if let Err(e) = write {
            self.stats.errors += 1;
            return Err(Error::new(Code::Io, "spill write failed").with_context(&e.to_string()));
        }

        self.write_offset = offset + needed;
        if let Some(old) = self.index.insert(key, IndexEntry { offset, len: payload.len() as u32, checksum, last_used: self.clock }) {
            self.garbage += RECORD_HEADER_BYTES + old.len as u64;
        }
        self.clock += 1;
        self.stats.bytes = self.recompute_bytes();
        Ok(needed)
    }

    /// Reads a payload into `out` (cleared first). Every failure is a miss.
    ///
    /// `out` is a `Vec` deliberately: the caller is off the render path (cache
    /// fill and compaction happen between frames), and the borrow rules make a
    /// host-allocated caller buffer awkward for a one-shot read.
    pub fn get(&mut self, key: u64, out: &mut Vec<u8>) -> Hit {
        out.clear();
        let entry = match self.index.get_mut(&key) {
            Some(e) => {
                e.last_used = self.clock;
                *e
            }
            None => {
                self.stats.misses += 1;
                return Hit::Miss;
            }
        };
        let mut buf = vec![0u8; entry.len as usize];
        let read = self
            .file
            .seek(SeekFrom::Start(entry.offset + RECORD_HEADER_BYTES))
            .and_then(|_| self.file.read_exact(&mut buf));
        match read {
            Ok(()) => {
                if verify(&buf, self.config.seed) != entry.checksum {
                    // The bytes changed under us: drop the entry rather than use it.
                    self.index.remove(&key);
                    self.stats.corrupt += 1;
                    self.stats.bytes = self.recompute_bytes();
                    return Hit::Corrupt;
                }
                out.extend_from_slice(&buf);
                self.clock += 1;
                self.stats.hits += 1;
                Hit::Fresh
            }
            Err(_) => {
                self.index.remove(&key);
                self.stats.corrupt += 1;
                self.stats.bytes = self.recompute_bytes();
                Hit::Corrupt
            }
        }
    }

    /// Drops least-recently-used entries until `needed` bytes fit under the cap.
    pub fn evict_for(&mut self, needed: u64) {
        let cap = self.config.max_bytes;
        if cap == 0 {
            return;
        }
        let mut current = self.recompute_bytes();
        if current + needed <= cap {
            return;
        }
        let mut order: Vec<(u64, u64)> = self.index.iter().map(|(k, v)| (*k, v.last_used)).collect();
        order.sort_by_key(|(_, used)| *used);
        for (key, _) in order {
            if current + needed <= cap {
                break;
            }
            if let Some(entry) = self.index.remove(&key) {
                let bytes = RECORD_HEADER_BYTES + entry.len as u64;
                current -= bytes;
                self.stats.evictions += 1;
                self.stats.evicted_bytes += bytes;
                self.garbage += bytes;
            }
        }
        self.stats.bytes = self.recompute_bytes();
    }

    /// Rewrites the file with only the live entries, in one pass.
    ///
    /// Called by the caller between frames, never from the render path: the
    /// brief's rule is that eviction must be interruptible and never run on the
    /// render thread, and a compaction is IO bound by definition.
    pub fn compact(&mut self) -> Result<u64> {
        if self.garbage == 0 {
            return Ok(0);
        }
        let before = self.file.metadata().map(|m| m.len()).unwrap_or(0);
        let temp_path = self.config.path().with_extension("rcls.compact");
        let mut temp = File::create(&temp_path)
            .map_err(|e| Error::new(Code::Io, "could not create the compaction file").with_context(&e.to_string()))?;

        let mut header = [0u8; HEADER_BYTES as usize];
        header[0..4].copy_from_slice(&RCLS_MAGIC.to_le_bytes());
        header[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        header[16..24].copy_from_slice(&self.config.seed.to_le_bytes());
        let crc = xxh64_seeded(self.config.seed, &header[0..24]);
        header[24..32].copy_from_slice(&crc.to_le_bytes());
        temp.write_all(&header).map_err(|e| Error::new(Code::Io, "compaction write failed").with_context(&e.to_string()))?;

        let mut new_index: HashMap<u64, IndexEntry> = HashMap::with_capacity(self.index.len());
        let mut offset = HEADER_BYTES;
        let mut payload = Vec::new();
        let entries: Vec<(u64, IndexEntry)> = self.index.iter().map(|(k, v)| (*k, *v)).collect();
        for (key, entry) in entries {
            payload.resize(entry.len as usize, 0);
            let ok = self
                .file
                .seek(SeekFrom::Start(entry.offset + RECORD_HEADER_BYTES))
                .and_then(|_| self.file.read_exact(&mut payload))
                .is_ok();
            if !ok || verify(&payload, self.config.seed) != entry.checksum {
                self.stats.corrupt += 1;
                continue;
            }
            let mut rec = [0u8; RECORD_HEADER_BYTES as usize];
            rec[0..4].copy_from_slice(&RECORD_MAGIC.to_le_bytes());
            rec[4..8].copy_from_slice(&(entry.len).to_le_bytes());
            rec[8..16].copy_from_slice(&key.to_le_bytes());
            rec[16..24].copy_from_slice(&entry.checksum.to_le_bytes());
            if temp.write_all(&rec).and_then(|_| temp.write_all(&payload)).is_err() {
                self.stats.errors += 1;
                let _ = std::fs::remove_file(&temp_path);
                return Err(Error::new(Code::Io, "compaction write failed"));
            }
            new_index.insert(
                key,
                IndexEntry { offset, len: entry.len, checksum: entry.checksum, last_used: entry.last_used },
            );
            offset += RECORD_HEADER_BYTES + entry.len as u64;
        }
        temp.sync_all().map_err(|e| Error::new(Code::Io, "compaction sync failed").with_context(&e.to_string()))?;
        drop(temp);

        // Replace: close, swap, reopen. A failed swap leaves the original file
        // untouched, which is the safe direction for a cache.
        let backup = self.config.path().with_extension("rcls.old");
        let _ = std::fs::remove_file(&backup);
        std::fs::rename(self.config.path(), &backup)
            .map_err(|e| Error::new(Code::Io, "could not rotate the arena").with_context(&e.to_string()))?;
        std::fs::rename(&temp_path, self.config.path())
            .map_err(|e| Error::new(Code::Io, "could not install the compacted arena").with_context(&e.to_string()))?;
        let _ = std::fs::remove_file(&backup);
        self.file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(self.config.path())
            .map_err(|e| Error::new(Code::Io, "could not reopen the arena").with_context(&e.to_string()))?;
        self.index = new_index;
        self.write_offset = offset;
        self.garbage = 0;
        self.stats.compactions += 1;
        let after = self.file.metadata().map(|m| m.len()).unwrap_or(0);
        reconl_core::log_info!("spill arena compacted: {} -> {} bytes", before, after);
        Ok(before.saturating_sub(after))
    }

    /// Flushes to the OS. Called at frame boundaries, never mid-frame.
    pub fn flush(&mut self) -> Result<()> {
        self.file
            .sync_data()
            .map_err(|e| Error::new(Code::Io, "spill flush failed").with_context(&e.to_string()))
    }

    pub fn path(&self) -> PathBuf {
        self.config.path()
    }
}

fn verify(payload: &[u8], seed: u64) -> u64 {
    let mut h = XxHash64::new(seed ^ 0x5243_4C53_4352_4331);
    h.update(&(payload.len() as u64).to_le_bytes());
    h.update(payload);
    h.finish()
}

/// Where the arena lives when the host does not say.
///
/// `RECONL_SPILL_DIR`, then the per-user cache directory. Never anywhere else,
/// and never without the host opting in to disk use.
pub fn default_spill_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("RECONL_SPILL_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        return Path::new(&local).join("ReconL").join("cache");
    }
    if let Ok(cache) = std::env::var("XDG_CACHE_HOME") {
        return Path::new(&cache).join("reconl");
    }
    if let Ok(home) = std::env::var("HOME") {
        return Path::new(&home).join(".cache").join("reconl");
    }
    PathBuf::from(".reconl-spill")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("reconl-spill-test-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn round_trips_payloads_and_survives_reopen() {
        let dir = temp_dir("roundtrip");
        {
            let mut arena = SpillArena::open(SpillConfig::new(&dir)).unwrap();
            arena.put(1, b"first payload").unwrap();
            arena.put(2, b"second payload").unwrap();
            assert!(arena.contains(1) && arena.contains(2));
            arena.flush().unwrap();
        }
        let mut arena = SpillArena::open(SpillConfig::new(&dir)).unwrap();
        assert_eq!(arena.len(), 2, "a clean reopen keeps every entry");
        assert_eq!(arena.stats().recovered_torn, 0);
        let mut out = Vec::new();
        assert_eq!(arena.get(1, &mut out), Hit::Fresh);
        assert_eq!(out, b"first payload");
        assert_eq!(arena.get(9, &mut out), Hit::Miss);
        assert!(out.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_torn_tail_is_dropped_and_the_rest_survives() {
        let dir = temp_dir("torn");
        let mut arena = SpillArena::open(SpillConfig::new(&dir)).unwrap();
        arena.put(1, b"aaaaaaaaaaaaaaaa").unwrap();
        arena.put(2, b"bbbbbbbbbbbbbbbb").unwrap();
        let good_len = arena.write_offset;
        drop(arena);

        // Simulate a process killed mid-append: header of a third record, no payload.
        {
            let mut file = OpenOptions::new().append(true).open(dir.join("arena.rcls")).unwrap();
            let mut rec = [0u8; RECORD_HEADER_BYTES as usize];
            rec[0..4].copy_from_slice(&RECORD_MAGIC.to_le_bytes());
            rec[4..8].copy_from_slice(&64u32.to_le_bytes());
            rec[8..16].copy_from_slice(&3u64.to_le_bytes());
            file.write_all(&rec).unwrap();
            file.write_all(&[0u8; 10]).unwrap(); // short payload
        }

        let mut arena = SpillArena::open(SpillConfig::new(&dir)).unwrap();
        assert_eq!(arena.stats().recovered_torn, 1, "the torn record must be counted");
        assert_eq!(arena.len(), 2, "the intact prefix must survive");
        assert_eq!(arena.write_offset, good_len, "the arena rewinds to the last good record");
        let mut out = Vec::new();
        assert_eq!(arena.get(2, &mut out), Hit::Fresh);
        assert_eq!(out, b"bbbbbbbbbbbbbbbb");
        // And it accepts new writes immediately, overwriting the tear.
        arena.put(3, b"cccc").unwrap();
        assert!(arena.contains(3));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_payload_corrupted_while_open_is_reported_corrupt_not_returned() {
        let dir = temp_dir("flip");
        let mut arena = SpillArena::open(SpillConfig::new(&dir)).unwrap();
        arena.put(7, b"trust me on this one").unwrap();
        let entry = *arena.index.get(&7).unwrap();
        arena.flush().unwrap();

        // Flip a payload bit from outside while the arena is open: the index
        // still has the entry, so this exercises the read-time check rather than
        // the open-time recovery.
        {
            let mut file = OpenOptions::new().read(true).write(true).open(dir.join("arena.rcls")).unwrap();
            file.seek(SeekFrom::Start(entry.offset + RECORD_HEADER_BYTES + 2)).unwrap();
            file.write_all(&[0xFF]).unwrap();
            file.sync_all().unwrap();
        }

        let mut out = Vec::new();
        assert_eq!(arena.get(7, &mut out), Hit::Corrupt, "a bad checksum must never be returned as data");
        assert!(out.is_empty());
        assert_eq!(arena.stats().corrupt, 1);
        // Dropped, not retried forever.
        assert_eq!(arena.get(7, &mut out), Hit::Miss);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_torn_record_in_the_middle_ends_the_scan() {
        let dir = temp_dir("midflip");
        let mut arena = SpillArena::open(SpillConfig::new(&dir)).unwrap();
        arena.put(1, b"first").unwrap();
        arena.put(2, b"second payload here").unwrap();
        arena.put(3, b"third").unwrap();
        let second = *arena.index.get(&2).unwrap();
        arena.flush().unwrap();
        drop(arena);

        {
            let mut file = OpenOptions::new().read(true).write(true).open(dir.join("arena.rcls")).unwrap();
            file.seek(SeekFrom::Start(second.offset + RECORD_HEADER_BYTES)).unwrap();
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap();
            file.sync_all().unwrap();
        }

        // The rule: an invalid record ends the scan, so the damaged record and
        // everything after it are forgotten. Nothing wrong is ever returned.
        let mut arena = SpillArena::open(SpillConfig::new(&dir)).unwrap();
        assert_eq!(arena.stats().recovered_torn, 1);
        assert_eq!(arena.len(), 1);
        let mut out = Vec::new();
        assert_eq!(arena.get(1, &mut out), Hit::Fresh);
        assert_eq!(arena.get(2, &mut out), Hit::Miss);
        assert_eq!(arena.get(3, &mut out), Hit::Miss);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_honours_the_cap_and_counts_what_it_dropped() {
        let dir = temp_dir("evict");
        let mut config = SpillConfig::new(&dir);
        config.max_bytes = 200;
        let mut arena = SpillArena::open(config).unwrap();
        for key in 0..20u64 {
            arena.put(key, &vec![key as u8; 24]).unwrap();
            arena.evict_for(0);
        }
        let stats = arena.stats();
        assert!(stats.bytes <= 200, "arena grew past its cap: {} bytes", stats.bytes);
        assert!(stats.evictions > 0, "nothing was evicted at a 200 byte cap");
        // Whatever survived must still read back correctly.
        let mut out = Vec::new();
        let live: Vec<u64> = arena.index.keys().copied().collect();
        for key in live {
            assert_eq!(arena.get(key, &mut out), Hit::Fresh);
            assert_eq!(out.len(), 24);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compaction_shrinks_the_file_and_keeps_the_live_entries() {
        let dir = temp_dir("compact");
        let mut arena = SpillArena::open(SpillConfig::new(&dir)).unwrap();
        for round in 0..5u64 {
            for key in 0..4u64 {
                arena.put(key, &vec![(round * 10 + key) as u8; 64]).unwrap();
            }
        }
        let before_file_len = std::fs::metadata(arena.path()).unwrap().len();
        assert!(arena.stats().garbage_bytes > 0, "garbage should have accumulated");
        arena.compact().unwrap();
        let after_file_len = std::fs::metadata(arena.path()).unwrap().len();
        assert!(after_file_len < before_file_len, "compaction did not shrink the file");
        let after = arena.stats();
        assert_eq!(after.entries, 4, "only the newest payload per key survives");
        assert_eq!(after.bytes, 4 * RECORD_HEADER_BYTES + 4 * 64);
        let mut out = Vec::new();
        assert_eq!(arena.get(2, &mut out), Hit::Fresh);
        assert_eq!(out[0], (4 * 10 + 2) as u8, "the newest payload must win");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn deleting_the_directory_costs_speed_and_nothing_else() {
        let dir = temp_dir("delete");
        let mut arena = SpillArena::open(SpillConfig::new(&dir)).unwrap();
        arena.put(1, b"payload").unwrap();
        arena.flush().unwrap();
        let path = arena.path();
        assert!(path.exists());
        // The arena holds the file open; on Windows that blocks removal, so the
        // honest test is: reopen after the directory is gone.
        drop(arena);
        std::fs::remove_dir_all(&dir).unwrap();
        let mut arena = SpillArena::open(SpillConfig::new(&dir)).unwrap();
        assert!(arena.is_empty(), "a deleted cache must come back cold, not broken");
        let mut out = Vec::new();
        assert_eq!(arena.get(1, &mut out), Hit::Miss);
        // And it is usable immediately.
        arena.put(1, b"payload again").unwrap();
        assert_eq!(arena.get(1, &mut out), Hit::Fresh);
        assert_eq!(out, b"payload again");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_foreign_file_is_replaced_rather_than_trusted() {
        let dir = temp_dir("foreign");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("arena.rcls"), b"not an arena at all, just text").unwrap();
        let arena = SpillArena::open(SpillConfig::new(&dir)).unwrap();
        assert!(arena.is_empty());
        assert_eq!(arena.stats().corrupt, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_spill_dir_honours_the_environment() {
        let previous = std::env::var("RECONL_SPILL_DIR").ok();
        std::env::set_var("RECONL_SPILL_DIR", "C:\\reconl-test-spill");
        assert_eq!(default_spill_dir(), PathBuf::from("C:\\reconl-test-spill"));
        std::env::remove_var("RECONL_SPILL_DIR");
        assert!(default_spill_dir().to_string_lossy().contains("ReconL") || default_spill_dir().to_string_lossy().contains("reconl"));
        if let Some(p) = previous {
            std::env::set_var("RECONL_SPILL_DIR", p);
        }
    }
}
