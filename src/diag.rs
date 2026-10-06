//! Merged, explicit error diagnostics for short CLI reports
//!
//! Each [`alt`][crate::combinator::alt] branch fails at its own farthest
//! offset with its own commit level ([`ErrMode::Backtrack`] vs
//! [`ErrMode::Cut`]).  Rather than stacking every context onto a single
//! chain, [`DiagReport::merge`] explicitly merges the per-branch
//! [`BranchDiag`]s:
//!
//! - the branch that reached the real farthest offset wins
//! - branches failing at the same offset have their expectations merged
//!   while each branch's source is kept
//! - a [`Commit::Committed`] ([`ErrMode::Cut`]) failure is never reported as
//!   [`Commit::Recoverable`]
//! - error types without a [`DiagError`] implementation are reported as
//!   opaque; no expectation text is fabricated for them
//!
//! A report is built from errors you already have: it neither re-runs the
//! parser nor copies the input, keeping it cheap enough for short CLI
//! diagnostics.  [`ContextError`] itself is left unchanged.
//!
//! # Running the tests and examples
//!
//! ```console
//! $ cargo test --lib diag
//! $ cargo test --doc diag
//! ```
//!
//! # Example: two `alt` branches failing at the same offset
//!
//! ```
//! use winnow::diag::{BranchDiag, Commit, DiagReport};
//! use winnow::error::{ContextError, ErrMode, StrContext, StrContextValue};
//! use winnow::prelude::*;
//! use winnow::token::literal;
//!
//! let input = "!";
//!
//! // First `alt` branch
//! let first: ErrMode<ContextError> = literal("a")
//!     .context(StrContext::Expected(StrContextValue::CharLiteral('a')))
//!     .parse_peek(input)
//!     .map(|_| ())
//!     .unwrap_err();
//! // Second `alt` branch
//! let second: ErrMode<ContextError> = literal("b")
//!     .context(StrContext::Expected(StrContextValue::CharLiteral('b')))
//!     .parse_peek(input)
//!     .map(|_| ())
//!     .unwrap_err();
//!
//! let report = DiagReport::merge([
//!     BranchDiag::from_err_mode(&first, 0).with_source("first"),
//!     BranchDiag::from_err_mode(&second, 0).with_source("second"),
//! ]);
//!
//! // Same farthest offset: expectations merge, both sources are kept
//! assert_eq!(report.expected().collect::<Vec<_>>(), ["`a`", "`b`"]);
//! assert_eq!(
//!     report.sources().collect::<Vec<_>>(),
//!     [Some("first"), Some("second")]
//! );
//! assert_eq!(report.commit(), Commit::Recoverable);
//! assert_eq!(
//!     report.to_string(),
//!     "error at byte 0: expected `a`, `b` [from first, second]"
//! );
//! ```

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::fmt;

use crate::error::ContextError;
use crate::error::ErrMode;
use crate::error::StrContext;

/// Commit level of a parse failure, tracked separately from the farthest offset
///
/// See [`ErrMode`]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Commit {
    /// [`ErrMode::Backtrack`]: another `alt` branch may still be tried
    #[default]
    Recoverable,
    /// [`ErrMode::Cut`]: the parser committed to this branch and it must not
    /// be reported as recoverable
    Committed,
}

impl Commit {
    /// Whether the failure is [`Commit::Committed`]
    #[inline]
    pub fn is_committed(self) -> bool {
        matches!(self, Self::Committed)
    }
}

/// Extract expectation text from an error for diagnostics
///
/// The default implementation returns `None`, marking the error as opaque:
/// unsupported custom error types are rendered as `<opaque error>` and no
/// text is fabricated for them.
pub trait DiagError {
    /// Expected items at the failure offset, or `None` when the error type is
    /// unsupported (opaque)
    fn diag_expected(&self) -> Option<Vec<String>> {
        None
    }
}

impl DiagError for ContextError<StrContext> {
    fn diag_expected(&self) -> Option<Vec<String>> {
        Some(
            self.context()
                .map(|context| match context {
                    StrContext::Label(name) => alloc::format!("invalid {name}"),
                    StrContext::Expected(value) => value.to_string(),
                })
                .collect(),
        )
    }
}

/// Diagnostic for a single `alt` branch
///
/// The farthest `offset` and the [`Commit`] level are kept as separate
/// fields so merging never confuses the two.
#[derive(Clone, Debug, Default)]
pub struct BranchDiag {
    offset: usize,
    commit: Commit,
    expected: Vec<String>,
    source: Option<String>,
    opaque: bool,
}

impl BranchDiag {
    /// A branch with no expectation text
    #[inline]
    pub fn new(offset: usize, commit: Commit) -> Self {
        Self {
            offset,
            commit,
            expected: Vec::new(),
            source: None,
            opaque: false,
        }
    }

    /// A branch whose error type is unsupported
    ///
    /// No expectation text is fabricated; the report will show
    /// `<opaque error>`.
    #[inline]
    pub fn opaque(offset: usize, commit: Commit) -> Self {
        Self {
            opaque: true,
            ..Self::new(offset, commit)
        }
    }

    /// Build from an existing modal error, keeping the commit level of
    /// [`ErrMode::Cut`] separate from the farthest offset
    ///
    /// This only reads the error; it neither re-runs the parser nor copies
    /// the input.
    #[inline]
    pub fn from_err_mode<E: DiagError>(err: &ErrMode<E>, offset: usize) -> Self {
        match err {
            ErrMode::Cut(e) => Self::from_error(e, offset, Commit::Committed),
            ErrMode::Backtrack(e) => Self::from_error(e, offset, Commit::Recoverable),
            ErrMode::Incomplete(_) => Self::new(offset, Commit::Recoverable),
        }
    }

    /// Build from an error whose commit level is known out of band
    #[inline]
    pub fn from_error<E: DiagError>(err: &E, offset: usize, commit: Commit) -> Self {
        let mut diag = Self::new(offset, commit);
        match err.diag_expected() {
            Some(expected) => diag.expected = expected,
            None => diag.opaque = true,
        }
        diag
    }

    /// Identify which `alt` branch this diagnostic came from
    #[inline]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Expectations at the farthest offset
    #[inline]
    pub fn with_expected<I, S>(mut self, expected: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.expected = expected.into_iter().map(Into::into).collect();
        self
    }

    /// Farthest byte offset this branch reached before failing
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Commit level of the failure
    #[inline]
    pub fn commit(&self) -> Commit {
        self.commit
    }

    /// Expectations at the farthest offset
    #[inline]
    pub fn expected(&self) -> &[String] {
        &self.expected
    }

    /// Which `alt` branch this diagnostic came from
    #[inline]
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    /// Whether the error type is unsupported and no text was fabricated
    #[inline]
    pub fn is_opaque(&self) -> bool {
        self.opaque
    }
}

/// Explicitly merged diagnostic across `alt` branches, for short CLI reports
///
/// # Example: two `alt` branches failing at different offsets
///
/// ```
/// use winnow::diag::{BranchDiag, DiagReport};
/// use winnow::error::{ContextError, ErrMode, StrContext, StrContextValue};
/// use winnow::prelude::*;
/// use winnow::token::literal;
///
/// let input = "ab!";
///
/// // First `alt` branch: consumes `ab`, then fails at byte 2
/// let first: ErrMode<ContextError> = (
///     literal("ab"),
///     literal("cd").context(StrContext::Expected(StrContextValue::StringLiteral("cd"))),
/// )
///     .parse_peek(input)
///     .map(|_| ())
///     .unwrap_err();
///
/// // Second `alt` branch: fails immediately at byte 0
/// let second: ErrMode<ContextError> = literal("z")
///     .context(StrContext::Expected(StrContextValue::StringLiteral("z")))
///     .parse_peek(input)
///     .map(|_| ())
///     .unwrap_err();
///
/// // Merge the errors recorded while parsing; no re-parsing, no input copies
/// let report = DiagReport::merge([
///     BranchDiag::from_err_mode(&first, 2).with_source("pair"),
///     BranchDiag::from_err_mode(&second, 0).with_source("single"),
/// ]);
///
/// // The branch that reached the real farthest offset wins
/// assert_eq!(report.offset(), 2);
/// assert_eq!(report.expected().collect::<Vec<_>>(), ["`cd`"]);
/// assert_eq!(report.sources().collect::<Vec<_>>(), [Some("pair")]);
/// assert_eq!(report.to_string(), "error at byte 2: expected `cd` [from pair]");
/// ```
#[derive(Clone, Debug, Default)]
pub struct DiagReport {
    offset: usize,
    commit: Commit,
    expected: Vec<String>,
    sources: Vec<Option<String>>,
    opaque: bool,
}

impl DiagReport {
    /// Explicitly merge the diagnostics of each `alt` branch
    ///
    /// - A [`Commit::Committed`] branch cannot be backtracked: it is the real
    ///   failure, so recoverable branches are discarded and the report is
    ///   never marked recoverable.
    /// - Among the remaining branches, the real farthest offset wins.
    /// - Branches failing at that offset have their expectations merged
    ///   (deduplicated, in branch order) while every branch's source is kept.
    pub fn merge(branches: impl IntoIterator<Item = BranchDiag>) -> Self {
        let branches: Vec<BranchDiag> = branches.into_iter().collect();
        let committed = branches.iter().any(|b| b.commit.is_committed());
        let candidates = branches
            .iter()
            .filter(|b| !committed || b.commit.is_committed());
        let offset = candidates.clone().map(|b| b.offset).max().unwrap_or(0);
        let mut expected: Vec<String> = Vec::new();
        let mut sources: Vec<Option<String>> = Vec::new();
        let mut opaque = false;
        for branch in candidates.filter(|b| b.offset == offset) {
            for item in &branch.expected {
                if !expected.contains(item) {
                    expected.push(item.clone());
                }
            }
            sources.push(branch.source.clone());
            opaque |= branch.opaque;
        }
        Self {
            offset,
            commit: if committed {
                Commit::Committed
            } else {
                Commit::Recoverable
            },
            expected,
            sources,
            opaque,
        }
    }

    /// The real farthest byte offset across the merged branches
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Commit level of the merged failure
    ///
    /// A [`Commit::Committed`] failure is never reported as
    /// [`Commit::Recoverable`].
    #[inline]
    pub fn commit(&self) -> Commit {
        self.commit
    }

    /// Merged, deduplicated expectations at the farthest offset
    #[inline]
    pub fn expected(&self) -> impl Iterator<Item = &str> + '_ {
        self.expected.iter().map(String::as_str)
    }

    /// Source of each branch that reached the farthest offset
    ///
    /// Branches without [`BranchDiag::with_source`] show up as `None`.
    #[inline]
    pub fn sources(&self) -> impl Iterator<Item = Option<&str>> + '_ {
        self.sources.iter().map(|s| s.as_deref())
    }

    /// Whether any merged branch had an unsupported (opaque) error type
    #[inline]
    pub fn is_opaque(&self) -> bool {
        self.opaque
    }
}

impl fmt::Display for DiagReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error at byte {}", self.offset)?;
        let mut detailed = false;
        if !self.expected.is_empty() {
            write!(f, ": expected {}", self.expected.join(", "))?;
            detailed = true;
        }
        if self.opaque {
            if detailed {
                write!(f, ", ")?;
            } else {
                write!(f, ": ")?;
            }
            write!(f, "<opaque error>")?;
        }
        let show_sources = self.sources.len() > 1 || self.sources.iter().any(Option::is_some);
        if show_sources {
            write!(f, " [from ")?;
            for (i, source) in self.sources.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                match source {
                    Some(source) => write!(f, "{source}")?,
                    None => write!(f, "#{i}")?,
                }
            }
            write!(f, "]")?;
        }
        if self.commit.is_committed() {
            write!(f, "; unrecoverable")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::StrContextValue;

    #[test]
    fn farthest_branch_wins() {
        let report = DiagReport::merge([
            BranchDiag::new(2, Commit::Recoverable)
                .with_expected(["`cd`"])
                .with_source("pair"),
            BranchDiag::new(0, Commit::Recoverable)
                .with_expected(["`z`"])
                .with_source("single"),
        ]);
        assert_eq!(report.offset(), 2);
        assert_eq!(report.expected().collect::<Vec<_>>(), ["`cd`"]);
        assert_eq!(report.sources().collect::<Vec<_>>(), [Some("pair")]);
        assert_eq!(report.commit(), Commit::Recoverable);
    }

    #[test]
    fn same_offset_merges_expectations_and_keeps_sources() {
        let report = DiagReport::merge([
            BranchDiag::new(0, Commit::Recoverable)
                .with_expected(["`a`"])
                .with_source("first"),
            BranchDiag::new(0, Commit::Recoverable)
                .with_expected(["`b`", "`a`"])
                .with_source("second"),
        ]);
        assert_eq!(report.offset(), 0);
        // deduplicated, in branch order
        assert_eq!(report.expected().collect::<Vec<_>>(), ["`a`", "`b`"]);
        // both sources are kept
        assert_eq!(
            report.sources().collect::<Vec<_>>(),
            [Some("first"), Some("second")]
        );
    }

    #[test]
    fn committed_failure_is_not_reported_as_recoverable() {
        let report = DiagReport::merge([
            BranchDiag::new(9, Commit::Recoverable)
                .with_expected(["`z`"])
                .with_source("far"),
            BranchDiag::new(2, Commit::Committed)
                .with_expected(["`c`"])
                .with_source("cut"),
        ]);
        assert_eq!(report.commit(), Commit::Committed);
        assert_eq!(report.offset(), 2);
        assert_eq!(report.expected().collect::<Vec<_>>(), ["`c`"]);
        assert_eq!(report.sources().collect::<Vec<_>>(), [Some("cut")]);
        let rendered = report.to_string();
        assert!(rendered.contains("unrecoverable"), "{rendered}");
    }

    #[test]
    fn alt_cut_failure_stays_committed() {
        use crate::combinator::alt;
        use crate::combinator::cut_err;
        use crate::token::literal;
        use crate::Parser;

        let mut parser = alt((
            (
                literal("ab"),
                cut_err(literal("cd"))
                    .context(StrContext::Expected(StrContextValue::StringLiteral("cd"))),
            )
                .void(),
            literal("z").void(),
        ));
        let err: ErrMode<ContextError> = parser.parse_peek("ab!").map(|_| ()).unwrap_err();
        let report = DiagReport::merge([BranchDiag::from_err_mode(&err, 2).with_source("pair")]);
        assert_eq!(report.commit(), Commit::Committed);
        assert_eq!(report.expected().collect::<Vec<_>>(), ["`cd`"]);
        let rendered = report.to_string();
        assert!(rendered.contains("unrecoverable"), "{rendered}");
    }

    struct CustomError;
    // Default `DiagError` implementation: unsupported, hence opaque
    impl DiagError for CustomError {}

    #[test]
    fn opaque_error_type_fabricates_no_text() {
        let err: ErrMode<CustomError> = ErrMode::Backtrack(CustomError);
        let report = DiagReport::merge([BranchDiag::from_err_mode(&err, 3)]);
        assert!(report.is_opaque());
        assert_eq!(report.expected().count(), 0);
        assert_eq!(report.to_string(), "error at byte 3: <opaque error>");
    }

    #[test]
    fn context_error_provides_expectations() {
        let mut err = ContextError::new();
        err.push(StrContext::Expected(StrContextValue::StringLiteral("cd")));
        err.push(StrContext::Label("pair"));
        let diag = BranchDiag::from_error(&err, 2, Commit::Recoverable);
        assert!(!diag.is_opaque());
        assert_eq!(
            diag.expected(),
            ["`cd`".to_string(), "invalid pair".to_string()].as_slice()
        );
    }

    #[test]
    fn unnamed_sources_render_as_indices() {
        let report = DiagReport::merge([
            BranchDiag::new(1, Commit::Recoverable).with_expected(["`a`"]),
            BranchDiag::new(1, Commit::Recoverable).with_expected(["`b`"]),
        ]);
        assert_eq!(report.sources().count(), 2);
        assert_eq!(
            report.to_string(),
            "error at byte 1: expected `a`, `b` [from #0, #1]"
        );
    }
}
