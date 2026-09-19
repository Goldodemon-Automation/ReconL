//! Error model.
//!
//! ReconL never unwinds across the C boundary, so an error is a value: a code,
//! a message, and the source location that produced it. Messages are fixed
//! capacity ([`Text`]) because they can be built inside a frame loop.

use crate::text::Text;
use core::fmt;
use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};

pub const MESSAGE_CAP: usize = 192;
pub const FILE_CAP: usize = 260;
pub const FUNCTION_CAP: usize = 64;

#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Code {
    Ok = 0,
    InvalidArgument = -1,
    OutOfMemory = -2,
    NotSupported = -3,
    BackendUnavailable = -4,
    BudgetExceeded = -5,
    DeviceLost = -6,
    InvalidHandle = -7,
    StructSize = -8,
    WrongStructType = -9,
    AbiVersion = -10,
    FrameInProgress = -11,
    NoFrame = -12,
    NotReady = -13,
    Io = -14,
    CorruptCache = -15,
    /// The call succeeded at a lower tier than asked for. Check the stats.
    Degraded = -16,
    /// An internal invariant broke; the safe path was taken and counted.
    Panic = -17,
    /// A frame contained no geometry while the frame policy required some. The
    /// frame is not presented; the counter says it happened.
    EmptyFrame = -18,
}

impl Code {
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    pub fn from_i32(v: i32) -> Code {
        match v {
            0 => Code::Ok,
            -1 => Code::InvalidArgument,
            -2 => Code::OutOfMemory,
            -3 => Code::NotSupported,
            -4 => Code::BackendUnavailable,
            -5 => Code::BudgetExceeded,
            -6 => Code::DeviceLost,
            -7 => Code::InvalidHandle,
            -8 => Code::StructSize,
            -9 => Code::WrongStructType,
            -10 => Code::AbiVersion,
            -11 => Code::FrameInProgress,
            -12 => Code::NoFrame,
            -13 => Code::NotReady,
            -14 => Code::Io,
            -15 => Code::CorruptCache,
            -16 => Code::Degraded,
            -18 => Code::EmptyFrame,
            _ => Code::Panic,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Code::Ok => "RECONL_OK",
            Code::InvalidArgument => "RECONL_ERR_INVALID_ARGUMENT",
            Code::OutOfMemory => "RECONL_ERR_OUT_OF_MEMORY",
            Code::NotSupported => "RECONL_ERR_NOT_SUPPORTED",
            Code::BackendUnavailable => "RECONL_ERR_BACKEND_UNAVAILABLE",
            Code::BudgetExceeded => "RECONL_ERR_BUDGET_EXCEEDED",
            Code::DeviceLost => "RECONL_ERR_DEVICE_LOST",
            Code::InvalidHandle => "RECONL_ERR_INVALID_HANDLE",
            Code::StructSize => "RECONL_ERR_STRUCT_SIZE",
            Code::WrongStructType => "RECONL_ERR_WRONG_STRUCT_TYPE",
            Code::AbiVersion => "RECONL_ERR_ABI_VERSION",
            Code::FrameInProgress => "RECONL_ERR_FRAME_IN_PROGRESS",
            Code::NoFrame => "RECONL_ERR_NO_FRAME",
            Code::NotReady => "RECONL_ERR_NOT_READY",
            Code::Io => "RECONL_ERR_IO",
            Code::CorruptCache => "RECONL_ERR_CORRUPT_CACHE",
            Code::Degraded => "RECONL_ERR_DEGRADED",
            Code::Panic => "RECONL_ERR_PANIC",
            Code::EmptyFrame => "RECONL_ERR_EMPTY_FRAME",
        }
    }

    pub const fn is_ok(self) -> bool {
        matches!(self, Code::Ok)
    }
}

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Copy)]
pub struct Error {
    pub code: Code,
    pub message: Text<MESSAGE_CAP>,
    pub file: Text<FILE_CAP>,
    pub function: Text<FUNCTION_CAP>,
    pub line: u32,
}

impl Error {
    pub fn new(code: Code, message: &str) -> Self {
        Self {
            code,
            message: Text::from_str(message),
            file: Text::new(),
            function: Text::new(),
            line: 0,
        }
    }

    pub fn at(code: Code, file: &str, line: u32, function: &str) -> Self {
        Self {
            code,
            message: Text::new(),
            file: Text::from_str(short_file(file)),
            function: Text::from_str(function),
            line,
        }
    }

    pub fn with_message(mut self, message: &str) -> Self {
        self.message.set(message);
        self
    }

    /// Builds an error from `format_args!` without allocating.
    pub fn fmt_at(code: Code, file: &str, line: u32, function: &str, args: fmt::Arguments<'_>) -> Self {
        use fmt::Write as _;
        let mut err = Self::at(code, file, line, function);
        let _ = err.message.write_fmt(args);
        err
    }

    pub fn message_str(&self) -> &str {
        self.message.as_str()
    }

    pub fn with_context(mut self, message: &str) -> Self {
        if !self.message.is_empty() {
            let existing: Text<MESSAGE_CAP> = self.message;
            self.message.clear();
            self.message.push(message);
            self.message.push(": ");
            self.message.push(existing.as_str());
        } else {
            self.message.set(message);
        }
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} ({}:{})", self.code.name(), self.message.as_str(), self.file.as_str(), self.line)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl From<Error> for Code {
    fn from(e: Error) -> Code {
        e.code
    }
}

fn short_file(path: &str) -> &str {
    // Keep the last two path components: enough to locate, short enough for the ABI.
    let mut seen = 0;
    for (idx, _) in path.char_indices().rev() {
        if path.as_bytes()[idx] == b'/' || path.as_bytes()[idx] == b'\\' {
            seen += 1;
            if seen == 2 {
                return &path[idx + 1..];
            }
        }
    }
    path
}

/// `err!(Code::X, "why {}", n)` -> `Err(Error)` with this call site recorded.
#[macro_export]
macro_rules! err {
    ($code:expr, $($arg:tt)*) => {
        Err($crate::error::Error::fmt_at(
            $code,
            file!(),
            line!(),
            $crate::function_name!(),
            format_args!($($arg)*),
        ))
    };
}

/// Best-effort function name without `std::any::type_name` silliness.
#[macro_export]
macro_rules! function_name {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            core::any::type_name::<T>()
        }
        let n = type_name_of(f);
        // strip "::f" and the trailing "::{{closure}}" if any
        n.trim_end_matches("::f").rsplit("::").next().unwrap_or("?")
    }};
}

/// Process-wide last error, used by `reconlGetLastError(NULL, ...)`.
///
/// This is a diagnostic slot, not a behaviour switch: it cannot change what
/// ReconL renders. A device keeps its own slot so that two devices on two
/// threads do not overwrite each other's diagnosis.
pub struct LastErrorSlot {
    code: AtomicI32,
    line: AtomicU32,
    message: core::cell::UnsafeCell<Text<MESSAGE_CAP>>,
    file: core::cell::UnsafeCell<Text<FILE_CAP>>,
    function: core::cell::UnsafeCell<Text<FUNCTION_CAP>>,
}

unsafe impl Sync for LastErrorSlot {}

impl LastErrorSlot {
    pub const fn new() -> Self {
        Self {
            code: AtomicI32::new(0),
            line: AtomicU32::new(0),
            message: core::cell::UnsafeCell::new(Text::new()),
            file: core::cell::UnsafeCell::new(Text::new()),
            function: core::cell::UnsafeCell::new(Text::new()),
        }
    }

    pub fn store(&self, err: &Error) {
        // SAFETY: every access is serialised by the caller holding the lock that
        // guards this slot (`LAST_ERROR_LOCK` below). The slot is never aliased.
        unsafe {
            (*self.message.get()).set(err.message.as_str());
            (*self.file.get()).set(err.file.as_str());
            (*self.function.get()).set(err.function.as_str());
        }
        self.line.store(err.line, Ordering::Relaxed);
        self.code.store(err.code.as_i32(), Ordering::Release);
    }

    pub fn store_code(&self, code: Code) {
        self.code.store(code.as_i32(), Ordering::Release);
    }

    pub fn snapshot(&self) -> Error {
        let code = Code::from_i32(self.code.load(Ordering::Acquire));
        // SAFETY: guarded by LAST_ERROR_LOCK in `global_last_error`/`record_global`.
        unsafe {
            Error {
                code,
                message: *self.message.get(),
                file: *self.file.get(),
                function: *self.function.get(),
                line: self.line.load(Ordering::Relaxed),
            }
        }
    }

    pub fn clear(&self) {
        self.code.store(0, Ordering::Release);
    }
}

static GLOBAL_LAST_ERROR: LastErrorSlot = LastErrorSlot::new();
static LAST_ERROR_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn record_global(err: &Error) {
    if let Ok(_guard) = LAST_ERROR_LOCK.lock() {
        GLOBAL_LAST_ERROR.store(err);
    }
}

pub fn record_global_code(code: Code) {
    if let Ok(_guard) = LAST_ERROR_LOCK.lock() {
        if code.is_ok() {
            GLOBAL_LAST_ERROR.clear();
        } else {
            GLOBAL_LAST_ERROR.store_code(code);
        }
    }
}

pub fn global_last_error() -> Error {
    match LAST_ERROR_LOCK.lock() {
        Ok(_guard) => GLOBAL_LAST_ERROR.snapshot(),
        Err(_) => Error::new(Code::Panic, "last-error slot poisoned by a panic elsewhere"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_file_keeps_two_components() {
        assert_eq!(short_file("core/src/error.rs"), "src/error.rs");
        assert_eq!(short_file("error.rs"), "error.rs");
    }

    #[test]
    fn context_composes_without_allocating() {
        let e = Error::new(Code::Io, "disk said no").with_context("spill arena");
        assert_eq!(e.message_str(), "spill arena: disk said no");
    }

    #[test]
    fn fmt_at_records_location() {
        let e = Error::fmt_at(Code::BudgetExceeded, "core/src/budget.rs", 42, "reserve", format_args!("{} > {}", 9u32, 4u32));
        assert_eq!(e.message_str(), "9 > 4");
        assert_eq!(e.line, 42);
        assert_eq!(e.file.as_str(), "src/budget.rs");
    }

    #[test]
    fn global_slot_round_trips() {
        let e = Error::fmt_at(Code::DeviceLost, "a/b.rs", 7, "call", format_args!("gone"));
        record_global(&e);
        let got = global_last_error();
        assert_eq!(got.code, Code::DeviceLost);
        assert_eq!(got.message_str(), "gone");
        assert_eq!(got.line, 7);
        record_global_code(Code::Ok);
        assert_eq!(global_last_error().code, Code::Ok);
    }
}
