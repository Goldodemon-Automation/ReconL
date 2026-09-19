//! Allocation-free fixed-capacity text.
//!
//! Error messages and downgrade reasons must survive inside a frame loop where
//! ReconL has promised to allocate nothing, so they cannot be `String`. `Text`
//! is a `[u8; N]` with an explicit length: `set`/`push` never allocate, never
//! panic, and truncate on a char boundary.

use core::fmt;

#[derive(Clone, Copy)]
pub struct Text<const N: usize> {
    bytes: [u8; N],
    len: usize,
}

impl<const N: usize> Default for Text<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Text<N> {
    pub const fn new() -> Self {
        Self { bytes: [0; N], len: 0 }
    }

    pub fn from_str(s: &str) -> Self {
        let mut t = Self::new();
        t.push(s);
        t
    }

    pub fn clear(&mut self) {
        self.len = 0;
        if N > 0 {
            self.bytes[0] = 0;
        }
    }

    pub fn set(&mut self, s: &str) -> &mut Self {
        self.clear();
        self.push(s)
    }

    pub fn push(&mut self, s: &str) -> &mut Self {
        // Always leave room for a NUL so the buffer can be handed to C as-is.
        let cap = N.saturating_sub(1);
        for ch in s.chars() {
            let need = ch.len_utf8();
            if self.len + need > cap {
                break;
            }
            let mut buf = [0u8; 4];
            let enc = ch.encode_utf8(&mut buf);
            self.bytes[self.len..self.len + need].copy_from_slice(enc.as_bytes());
            self.len += need;
        }
        self.bytes[self.len] = 0;
        self
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    pub fn as_str(&self) -> &str {
        core::str::from_utf8(self.as_bytes()).unwrap_or("")
    }

    /// NUL-terminated view, valid for the lifetime of `self`.
    pub fn as_ptr(&self) -> *const core::ffi::c_char {
        self.bytes.as_ptr() as *const core::ffi::c_char
    }

    pub fn as_mut_ptr(&mut self) -> *mut core::ffi::c_char {
        self.bytes.as_mut_ptr() as *mut core::ffi::c_char
    }
}

impl<const N: usize> fmt::Write for Text<N> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push(s);
        Ok(())
    }
}

impl<const N: usize> fmt::Display for Text<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<const N: usize> fmt::Debug for Text<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<const N: usize> PartialEq for Text<N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl<const N: usize> Eq for Text<N> {}

#[cfg(test)]
mod tests {
    use super::*;
    use core::fmt::Write as _;

    #[test]
    fn truncates_on_char_boundary_and_stays_nul_terminated() {
        let mut t: Text<8> = Text::new();
        write!(t, "{}", "abcdef\u{00e9}\u{00e9}").unwrap();
        assert_eq!(t.as_str(), "abcdef");
        assert_eq!(unsafe { t.as_ptr().read() } as u8, b'a');
        // valid utf8, terminated
        assert_eq!(unsafe { *t.as_ptr().add(6) }, 0);
    }

    #[test]
    fn write_fmt_never_allocates_growth() {
        let mut t: Text<16> = Text::new();
        write!(t, "tier {} -> {} ({})", 0, 4, "ram cap").unwrap();
        // 16 bytes minus the NUL: 15 characters, truncated on a char boundary.
        assert_eq!(t.as_str(), "tier 0 -> 4 (ra");
        assert_eq!(t.as_str().len(), 15);
    }

    #[test]
    fn clear_and_set() {
        let mut t: Text<32> = Text::new();
        t.set("one");
        assert_eq!(t.as_str(), "one");
        t.set("two");
        assert_eq!(t.as_str(), "two");
        t.clear();
        assert!(t.is_empty());
        assert_eq!(t.as_str(), "");
    }
}
