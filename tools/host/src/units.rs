//! Sizes, times and rates, formatted for a terminal and parsed from a command
//! line.
//!
//! One place decides that `65536` prints as `64.0 KiB` and that `--ram-cap=64MB`
//! means 64 MiB, so the two tools that quote a budget quote it the same way and
//! a number copied out of one tool's output means the same in the other's input.

const BYTE_UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];

/// Bytes in binary units, with one decimal past the byte range.
pub fn bytes(n: u64) -> String {
    if n < 1024 {
        return format!("{n} B");
    }
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < BYTE_UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", BYTE_UNITS[unit])
}

/// A duration: `ns` below a microsecond, then `us`, then `ms`. Zero prints as
/// `0` rather than `0.000 ms`, because a zero here means "not reported" more
/// often than it means "instant" - and the two must not be confused.
pub fn ns(n: u64) -> String {
    if n == 0 {
        return "0".into();
    }
    if n < 1_000 {
        return format!("{n} ns");
    }
    if n < 1_000_000 {
        return format!("{:.1} us", n as f64 / 1e3);
    }
    if n < 1_000_000_000 {
        return format!("{:.3} ms", n as f64 / 1e6);
    }
    format!("{:.3} s", n as f64 / 1e9)
}

/// Millions of shaded pixels per second - the reference tier's honest unit of
/// work, since its cost is fill, not draw calls.
pub fn mpix_per_sec(pixels: u64, total_ns: u64) -> String {
    if total_ns == 0 {
        return "0".into();
    }
    format!("{:.2} Mpix/s", pixels as f64 / (total_ns as f64 / 1e9) / 1e6)
}

/// Frames per second from a mean frame time.
pub fn fps(mean_ns: u64) -> String {
    if mean_ns == 0 {
        return "0".into();
    }
    format!("{:.1}", 1e9 / mean_ns as f64)
}

/// Parses a size: a decimal number with an optional binary suffix.
///
/// `K`/`M`/`G`/`T` and their `B`, `KB`.., `KiB`.. spellings all mean the binary
/// multiplier (1024), so `--ram-cap=64MB` is 64 MiB and matches what [`bytes`]
/// prints back.
pub fn parse_size(text: &str) -> Result<u64, String> {
    let t = text.trim();
    let split = t
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(t.len());
    let (number, suffix) = t.split_at(split);
    if number.is_empty() {
        return Err(format!("`{text}` is not a size: expected a number, optionally followed by K/M/G/T"));
    }
    let multiplier = match suffix.trim().to_ascii_uppercase().as_str() {
        "" | "B" => 1u64,
        "K" | "KB" | "KIB" => 1 << 10,
        "M" | "MB" | "MIB" => 1 << 20,
        "G" | "GB" | "GIB" => 1 << 30,
        "T" | "TB" | "TIB" => 1 << 40,
        other => return Err(format!("`{text}`: `{other}` is not a size suffix (use B, K, M, G or T)")),
    };
    if number.contains('.') {
        let value: f64 = number
            .parse()
            .map_err(|_| format!("`{number}` is not a number"))?;
        return Ok((value * multiplier as f64) as u64);
    }
    number
        .parse::<u64>()
        .map_err(|_| format!("`{number}` is not a number"))
        .and_then(|v| v.checked_mul(multiplier).ok_or_else(|| format!("`{text}` overflows")))
}

/// Parses an unsigned integer argument, naming it in the error.
pub fn parse_u32(text: &str, what: &str) -> Result<u32, String> {
    text.parse::<u32>().map_err(|_| format!("`{text}` is not a valid {what}"))
}

/// `name=value` lookup over an argument list: `--frames=60` -> `Some("60")`.
pub fn value_of<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let prefix = format!("{flag}=");
    args.iter().find_map(|a| a.strip_prefix(prefix.as_str()))
}

/// A bare flag: `--audit` present -> true.
pub fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// Rejects any argument this tool does not know, so a typo is an error rather
/// than a silently ignored option - which is how a measurement ends up
/// mislabelled.
pub fn reject_unknown(args: &[String], known: &[&str]) -> Result<(), String> {
    for a in args {
        let bare = a.split('=').next().unwrap_or(a);
        if !bare.starts_with('-') {
            // The commonest mistake, and the one a bare "unexpected argument"
            // leaves the caller guessing at: an option's value written as a
            // second argument instead of after an `=`.
            return Err(format!(
                "unexpected argument `{a}` — options take their value with `=`, e.g. --png=FILE"
            ));
        }
        if !known.contains(&bare) {
            return Err(format!("unknown option `{bare}`"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_round_trip_through_their_own_units() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(64 << 20), "64.0 MiB");
        assert_eq!(parse_size("64MB").unwrap(), 64 << 20);
        assert_eq!(parse_size("64MiB").unwrap(), 64 << 20);
        assert_eq!(parse_size("64M").unwrap(), 64 << 20);
        assert_eq!(parse_size("1024").unwrap(), 1024);
        assert_eq!(parse_size("1.5G").unwrap(), (1.5 * (1u64 << 30) as f64) as u64);
        assert_eq!(bytes(parse_size("8MiB").unwrap()), "8.0 MiB");
    }

    #[test]
    fn bad_sizes_are_refused_with_the_offending_text() {
        for bad in ["", "MB", "12X", "1.2.3", "-4"] {
            let e = parse_size(bad).unwrap_err();
            assert!(e.contains(bad) || bad.is_empty(), "`{bad}` produced `{e}`");
        }
    }

    #[test]
    fn zero_time_prints_as_zero_not_as_a_measurement() {
        assert_eq!(ns(0), "0");
        assert_eq!(ns(999), "999 ns");
        assert_eq!(ns(1_500), "1.5 us");
        assert_eq!(ns(2_500_000), "2.500 ms");
        assert_eq!(fps(0), "0");
        assert_eq!(mpix_per_sec(1000, 0), "0");
    }

    #[test]
    fn unknown_options_are_rejected() {
        let args = vec!["--frames=60".to_string(), "--frumes=3".to_string()];
        assert_eq!(reject_unknown(&args, &["--frames"]).unwrap_err(), "unknown option `--frumes`");
        assert!(reject_unknown(&vec!["--frames=60".into()], &["--frames"]).is_ok());
        assert!(reject_unknown(&vec!["stray".into()], &[]).is_err());
    }
}
