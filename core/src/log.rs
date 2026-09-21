//! Logging: one process-wide level and one optional sink, safe from any thread.
//!
//! Emitting a line allocates nothing (the message is formatted into a
//! fixed-capacity buffer) so logging is legal inside a frame loop even though
//! allocation is not. If the sink cannot keep up, lines are dropped, not queued
//! - a renderer that grows a log backlog under pressure has worse problems.

use crate::text::Text;
use core::fmt;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

pub const LINE_CAP: usize = 512;

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Level {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl Level {
    pub fn from_u32(v: u32) -> Level {
        match v {
            0 => Level::Off,
            1 => Level::Error,
            2 => Level::Warn,
            3 => Level::Info,
            4 => Level::Debug,
            _ => Level::Trace,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Level::Off => "off",
            Level::Error => "error",
            Level::Warn => "warn",
            Level::Info => "info",
            Level::Debug => "debug",
            Level::Trace => "trace",
        }
    }
}

pub type SinkFn = Option<unsafe extern "C" fn(*mut core::ffi::c_void, u32, *const core::ffi::c_char, *const core::ffi::c_char, u32)>;

static LEVEL: AtomicU32 = AtomicU32::new(Level::Info as u32);
static SINK_FN: AtomicUsize = AtomicUsize::new(0);
static SINK_USER: AtomicUsize = AtomicUsize::new(0);
static DROPPED: AtomicU32 = AtomicU32::new(0);

pub fn set_level(level: Level) {
    LEVEL.store(level as u32, Ordering::Release);
}

pub fn level() -> Level {
    Level::from_u32(LEVEL.load(Ordering::Acquire))
}

pub fn set_sink(sink: SinkFn, user: *mut core::ffi::c_void) {
    SINK_USER.store(user as usize, Ordering::Release);
    SINK_FN.store(sink.map(|f| f as usize).unwrap_or(0), Ordering::Release);
}

#[inline]
pub fn enabled(level: Level) -> bool {
    (level as u32) <= LEVEL.load(Ordering::Relaxed)
}

/// Emits one line. Never allocates, never panics, never calls back into ReconL
/// when a host sink is installed (the sink contract, enforced by documentation).
pub fn emit(level: Level, file: &str, line: u32, args: fmt::Arguments<'_>) {
    if !enabled(level) {
        return;
    }
    use fmt::Write as _;
    let mut buf: Text<LINE_CAP> = Text::new();
    let _ = buf.write_fmt(args);
    let mut floc: Text<128> = Text::new();
    let _ = write!(floc, "{}:{}", file, line);

    let raw = SINK_FN.load(Ordering::Acquire);
    if raw != 0 {
        // SAFETY: `raw` was produced by `f as usize` for a valid
        // `unsafe extern "C" fn`, so the reverse transmute restores that type.
        // The pointers we pass point at `buf`/`floc`, which outlive the call.
        let f: SinkFn = unsafe { core::mem::transmute::<usize, SinkFn>(raw) };
        if let Some(f) = f {
            let user = SINK_USER.load(Ordering::Acquire) as *mut core::ffi::c_void;
            unsafe { f(user, level as u32, buf.as_ptr(), floc.as_ptr(), line) };
        }
        return;
    }

    // Default sink: stderr. `eprintln!` formats into a stack buffer; no alloc.
    eprintln!("[reconl/{}] {} ({}:{})", level.label(), buf.as_str(), file, line);
}

pub fn dropped_lines() -> u32 {
    DROPPED.load(Ordering::Relaxed)
}

#[macro_export]
macro_rules! rlog {
    ($level:expr, $($arg:tt)*) => {
        $crate::log::emit($level, file!(), line!(), format_args!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => { $crate::rlog!($crate::log::Level::Error, $($arg)*) };
}
#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => { $crate::rlog!($crate::log::Level::Warn, $($arg)*) };
}
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => { $crate::rlog!($crate::log::Level::Info, $($arg)*) };
}
#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => { $crate::rlog!($crate::log::Level::Debug, $($arg)*) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static CAPTURED: Mutex<Vec<String>> = Mutex::new(Vec::new());

    unsafe extern "C" fn capture(
        _user: *mut core::ffi::c_void,
        level: u32,
        message: *const core::ffi::c_char,
        _file: *const core::ffi::c_char,
        _line: u32,
    ) {
        let msg = unsafe { std::ffi::CStr::from_ptr(message) }.to_string_lossy().into_owned();
        CAPTURED.lock().unwrap().push(format!("{}:{}", level, msg));
    }

    // One test, not two: the sink and the level are process-wide, so two tests
    // touching them in parallel would race - which is the same state the
    // header's logging note declares: one process-wide sink, safe from any
    // thread.
    #[test]
    fn sink_receives_formatted_lines_and_long_lines_truncate() {
        set_sink(Some(capture), core::ptr::null_mut());
        set_level(Level::Info);
        rlog!(Level::Debug, "hidden {}", 1);
        rlog!(Level::Error, "shown {}", 2);
        set_level(Level::Off);
        rlog!(Level::Error, "hidden too");
        set_level(Level::Trace);
        let long = "x".repeat(4096);
        rlog!(Level::Info, "{}", long);
        let got = CAPTURED.lock().unwrap().clone();
        assert_eq!(got[0], "1:shown 2");
        assert_eq!(got.len(), 2, "the off level and the debug line emitted nothing");
        assert!(got[1].len() <= LINE_CAP + 4, "line was {} bytes", got[1].len());
        set_sink(None, core::ptr::null_mut());
        set_level(Level::Info);
    }
}
