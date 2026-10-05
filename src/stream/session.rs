//! Explicit streaming parse sessions over [`Partial`] input
//!
//! A [`PartialSession`] binds one parser and one [`SessionConfig`] into a
//! stateful, resumable parse mode:
//! - [`ErrMode::Incomplete`][crate::error::ErrMode::Incomplete] is not a final
//!   failure: the session saves the real [`Needed`] amount and the consumed
//!   position, and parsing continues once more chunks are appended with
//!   [`PartialSession::feed`].
//! - Only [`PartialSession::finalize`] treats the end of input as definitive,
//!   converting a pending `Incomplete` into a [`FinalError`] that reports the
//!   real stream position.
//! - Each result is committed exactly once; [`PartialSession::cancel`] drops
//!   the session without re-committing already accepted results.
//! - The configured buffer limit rejects a chunk *before* the input buffer is
//!   allowed to grow past it.
//! - Changing the [`SessionConfig`] rejects sessions bound to the old
//!   configuration.
//!
//! The default complete-input parsing path ([`Parser::parse`],
//! [`Parser::parse_peek`]) is unchanged; sessions are an additive, opt-in
//! mode for streaming input.
//!
//! # Example
//!
//! Three chunks carry two 4-byte tags; the first chunk cannot complete a
//! token, the second and third each commit exactly one:
//!
//! ```rust
//! use winnow::error::ContextError;
//! use winnow::error::Needed;
//! use winnow::stream::FeedOutcome;
//! use winnow::stream::FinalizeOutcome;
//! use winnow::stream::Partial;
//! use winnow::stream::PartialSession;
//! use winnow::stream::SessionConfig;
//! use winnow::token::take;
//! use winnow::ModalResult;
//! use winnow::Parser;
//!
//! fn tag(input: &mut Partial<&[u8]>) -> ModalResult<Vec<u8>, ContextError> {
//!     take(4usize).map(|s: &[u8]| s.to_vec()).parse_next(input)
//! }
//!
//! let config = SessionConfig::new(16);
//! let mut session = PartialSession::new(tag, &config);
//!
//! // Chunk 1: too short for one token, no value is produced
//! let outcome = session.feed(&config, b"ab").unwrap();
//! assert_eq!(
//!     outcome,
//!     FeedOutcome::Incomplete {
//!         needed: Needed::new(2),
//!         consumed: 0,
//!     }
//! );
//!
//! // Chunk 2: completes the first token exactly one
//! let outcome = session.feed(&config, b"cdwx").unwrap();
//! assert_eq!(
//!     outcome,
//!     FeedOutcome::Committed {
//!         output: b"abcd".to_vec(),
//!         consumed: 4,
//!     }
//! );
//!
//! // Chunk 3: completes the second token
//! let outcome = session.feed(&config, b"yz").unwrap();
//! assert_eq!(
//!     outcome,
//!     FeedOutcome::Committed {
//!         output: b"wxyz".to_vec(),
//!         consumed: 8,
//!     }
//! );
//!
//! // The input ended cleanly on a token boundary
//! let outcome = session.finalize(&config).unwrap();
//! assert_eq!(outcome, FinalizeOutcome::Complete { consumed: 8 });
//! ```
//!
//! # Local checks
//!
//! ```console
//! $ cargo test --lib stream::session
//! $ cargo test --doc stream::session
//! $ cargo run --example partial_session
//! ```

use alloc::vec::Vec;

use core::fmt;

use crate::error::ErrMode;
use crate::stream::Needed;
use crate::stream::Offset;
use crate::stream::Partial;
use crate::stream::Stream as _;
use crate::stream::StreamIsPartial as _;
use crate::Parser;

const DEFAULT_BUFFER_LIMIT: usize = 4 * 1024 * 1024;

/// Configuration bound to a [`PartialSession`]
///
/// A session captures the configuration [`generation`][Self::generation] when
/// it is created. Changing the configuration bumps the generation, and
/// sessions bound to the old configuration are rejected with
/// [`SessionError::ConfigChanged`] / [`FinalErrorKind::ConfigChanged`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionConfig {
    buffer_limit: usize,
    generation: u64,
}

impl SessionConfig {
    /// Create a configuration rejecting buffers larger than `buffer_limit` bytes
    pub fn new(buffer_limit: usize) -> Self {
        let generation = 0;
        Self {
            buffer_limit,
            generation,
        }
    }

    /// The maximum number of uncommitted bytes a session may buffer
    pub fn buffer_limit(&self) -> usize {
        self.buffer_limit
    }

    /// Change the buffer limit
    ///
    /// This bumps the configuration [`generation`][Self::generation], so
    /// sessions bound to the previous configuration are rejected.
    pub fn set_buffer_limit(&mut self, buffer_limit: usize) {
        self.buffer_limit = buffer_limit;
        self.generation += 1;
    }

    /// Revision of the configuration, bumped on every change
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self::new(DEFAULT_BUFFER_LIMIT)
    }
}

/// An explicit streaming parse session over [`Partial`] byte input
///
/// The session binds one parser and one [`SessionConfig`] (see
/// [module docs][self] for the full workflow). Chunks are appended with
/// [`feed`][Self::feed]; a temporary end of input is reported as
/// [`FeedOutcome::Incomplete`] rather than a failure, keeping the real
/// [`Needed`] amount and the consumed position until more data arrives or
/// [`finalize`][Self::finalize] converts it into a [`FinalError`].
pub struct PartialSession<P> {
    parser: P,
    generation: u64,
    buffer_limit: usize,
    buffer: Vec<u8>,
    start: usize,
    base: usize,
    consumed: usize,
    needed: Option<Needed>,
    committed: u64,
    state: SessionState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SessionState {
    Active,
    Failed,
    Closed,
}

impl<P> PartialSession<P> {
    /// Bind `parser` and `config` into a new streaming parse session
    pub fn new(parser: P, config: &SessionConfig) -> Self {
        Self {
            parser,
            generation: config.generation(),
            buffer_limit: config.buffer_limit(),
            buffer: Vec::new(),
            start: 0,
            base: 0,
            consumed: 0,
            needed: None,
            committed: 0,
            state: SessionState::Active,
        }
    }

    /// Absolute stream position up to which input has been consumed
    pub fn position(&self) -> usize {
        self.consumed
    }

    /// The real amount of data reported by the last [`FeedOutcome::Incomplete`]
    pub fn last_needed(&self) -> Option<Needed> {
        self.needed
    }

    /// Number of results committed so far
    ///
    /// Each result is committed exactly once.
    pub fn committed_count(&self) -> u64 {
        self.committed
    }

    /// Bytes currently buffered while waiting for more input
    pub fn buffered(&self) -> usize {
        self.buffer.len() - self.start
    }

    /// Append a chunk and continue parsing
    ///
    /// - A chunk that would grow the buffer past the configured limit is
    ///   rejected *before* the buffer grows.
    /// - A temporary end of input yields [`FeedOutcome::Incomplete`] with the
    ///   real [`Needed`] amount and consumed position; it is not a failure.
    /// - A completed token is committed exactly once.
    pub fn feed<O, E>(
        &mut self,
        config: &SessionConfig,
        chunk: &[u8],
    ) -> Result<FeedOutcome<O>, SessionError<E>>
    where
        for<'a> P: Parser<Partial<&'a [u8]>, O, ErrMode<E>>,
    {
        if self.state != SessionState::Active {
            return Err(SessionError::Closed);
        }
        if config.generation() != self.generation {
            return Err(SessionError::ConfigChanged {
                expected: self.generation,
                found: config.generation(),
            });
        }
        let buffered = self.buffered();
        if buffered.saturating_add(chunk.len()) > self.buffer_limit {
            // Reject the chunk before the input buffer is allowed to grow
            return Err(SessionError::BufferLimitExceeded {
                limit: self.buffer_limit,
                buffered,
                incoming: chunk.len(),
            });
        }
        self.buffer.extend_from_slice(chunk);

        let window = &self.buffer[self.start..];
        let mut input = Partial::new(window);
        let checkpoint = input.checkpoint();
        match self.parser.parse_next(&mut input) {
            Ok(output) => {
                let used = input.offset_from(&checkpoint);
                self.commit(used);
                Ok(FeedOutcome::Committed {
                    output,
                    consumed: self.consumed,
                })
            }
            Err(ErrMode::Incomplete(needed)) => {
                let advanced = input.offset_from(&checkpoint);
                self.consumed = self.base + self.start + advanced;
                self.needed = Some(needed);
                Ok(FeedOutcome::Incomplete {
                    needed,
                    consumed: self.consumed,
                })
            }
            Err(ErrMode::Backtrack(e)) | Err(ErrMode::Cut(e)) => {
                self.state = SessionState::Failed;
                Err(SessionError::Parse(e))
            }
        }
    }

    /// End the input, converting any pending `Incomplete` into a final error
    ///
    /// The remaining input is parsed with complete semantics: a temporary end
    /// of input is no longer retried. If the last [`feed`][Self::feed] paused
    /// on [`FeedOutcome::Incomplete`], the stream is truncated and the error
    /// reports the real stream position where parsing stopped.
    pub fn finalize<O, E>(
        mut self,
        config: &SessionConfig,
    ) -> Result<FinalizeOutcome<O>, FinalError<E>>
    where
        for<'a> P: Parser<Partial<&'a [u8]>, O, ErrMode<E>>,
    {
        if self.state != SessionState::Active {
            return Err(FinalError {
                offset: self.consumed,
                kind: FinalErrorKind::Closed,
            });
        }
        if config.generation() != self.generation {
            return Err(FinalError {
                offset: self.consumed,
                kind: FinalErrorKind::ConfigChanged {
                    expected: self.generation,
                    found: config.generation(),
                },
            });
        }
        self.state = SessionState::Closed;
        if self.buffer.len() == self.start {
            return Ok(FinalizeOutcome::Complete {
                consumed: self.consumed,
            });
        }
        let pending_needed = self.needed;
        let pending_consumed = self.consumed;

        let window = &self.buffer[self.start..];
        let mut input = Partial::new(window);
        let checkpoint = input.checkpoint();
        // The input is final: a temporary EOF is converted into a final result
        let _ = input.complete();
        match self.parser.parse_next(&mut input) {
            Ok(output) => {
                let used = input.offset_from(&checkpoint);
                self.commit(used);
                Ok(FinalizeOutcome::Committed {
                    output,
                    consumed: self.consumed,
                })
            }
            Err(ErrMode::Incomplete(needed)) => {
                let advanced = input.offset_from(&checkpoint);
                let offset = self.base + self.start + advanced;
                Err(FinalError {
                    offset,
                    kind: FinalErrorKind::Truncated { needed },
                })
            }
            Err(ErrMode::Backtrack(e)) | Err(ErrMode::Cut(e)) => {
                if let Some(needed) = pending_needed {
                    // The last feed paused on a temporary EOF: the stream is
                    // truncated at the real position where parsing stopped
                    Err(FinalError {
                        offset: pending_consumed,
                        kind: FinalErrorKind::Truncated { needed },
                    })
                } else {
                    let offset = self.base + self.start;
                    Err(FinalError {
                        offset,
                        kind: FinalErrorKind::Parse(e),
                    })
                }
            }
        }
    }

    /// Cancel the session, dropping any buffered input
    ///
    /// Results that were already committed stay committed exactly once;
    /// cancellation never re-commits them.
    pub fn cancel(mut self) -> Cancelled {
        self.state = SessionState::Closed;
        self.buffer.clear();
        Cancelled {
            committed: self.committed,
            consumed: self.consumed,
        }
    }

    fn commit(&mut self, used: usize) {
        let end = self.start + used;
        self.buffer.drain(..end);
        self.base += end;
        self.start = 0;
        self.consumed = self.base;
        self.needed = None;
        self.committed += 1;
    }
}

impl<P> fmt::Debug for PartialSession<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PartialSession")
            .field("generation", &self.generation)
            .field("buffer_limit", &self.buffer_limit)
            .field("buffered", &self.buffered())
            .field("consumed", &self.consumed)
            .field("needed", &self.needed)
            .field("committed", &self.committed)
            .field("state", &self.state)
            .finish()
    }
}

/// Summary of a cancelled [`PartialSession`]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cancelled {
    /// Results committed before cancellation; cancellation never re-commits them
    pub committed: u64,
    /// Absolute stream position up to which input was consumed
    pub consumed: usize,
}

/// The result of [`PartialSession::feed`]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeedOutcome<O> {
    /// More data is needed; no value was produced
    Incomplete {
        /// The real amount of data needed, as reported by the parser
        needed: Needed,
        /// Absolute stream position up to which input was consumed
        consumed: usize,
    },
    /// The parser committed a result; each result is committed exactly once
    Committed {
        /// The committed value
        output: O,
        /// Absolute stream position up to which input was consumed
        consumed: usize,
    },
}

/// The result of [`PartialSession::finalize`]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinalizeOutcome<O> {
    /// The final chunk completed one more token, committed exactly once
    Committed {
        /// The committed value
        output: O,
        /// Absolute stream position up to which input was consumed
        consumed: usize,
    },
    /// The input ended cleanly on a token boundary
    Complete {
        /// Absolute stream position up to which input was consumed
        consumed: usize,
    },
}

/// Why [`PartialSession::feed`] could not continue
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionError<E> {
    /// The session was bound to a different configuration generation
    ConfigChanged {
        /// The generation the session was bound to
        expected: u64,
        /// The generation of the configuration passed in
        found: u64,
    },
    /// The chunk was rejected before the buffer grew past the configured limit
    BufferLimitExceeded {
        /// The configured limit
        limit: usize,
        /// Bytes already buffered
        buffered: usize,
        /// Bytes the rejected chunk would have added
        incoming: usize,
    },
    /// The parser rejected the input; the session is closed
    Parse(E),
    /// The session is closed and can no longer accept input
    Closed,
}

impl<E: fmt::Display> fmt::Display for SessionError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfigChanged { expected, found } => write!(
                f,
                "session was bound to configuration generation {expected}, found {found}"
            ),
            Self::BufferLimitExceeded {
                limit,
                buffered,
                incoming,
            } => write!(
                f,
                "chunk of {incoming} bytes would grow {buffered} buffered bytes past the limit of {limit} bytes"
            ),
            Self::Parse(e) => write!(f, "parser rejected the input: {e}"),
            Self::Closed => write!(f, "session is closed"),
        }
    }
}

#[cfg(feature = "std")]
impl<E> std::error::Error for SessionError<E> where E: fmt::Debug + fmt::Display {}

/// A final error once the input has ended
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalError<E> {
    offset: usize,
    kind: FinalErrorKind<E>,
}

impl<E> FinalError<E> {
    /// Absolute stream position where parsing stopped
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// What went wrong with the final input
    pub fn kind(&self) -> &FinalErrorKind<E> {
        &self.kind
    }

    /// Unwrap into the [`FinalErrorKind`]
    pub fn into_kind(self) -> FinalErrorKind<E> {
        self.kind
    }
}

impl<E: fmt::Display> fmt::Display for FinalError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at offset {}: ", self.offset)?;
        match &self.kind {
            FinalErrorKind::Truncated { needed } => {
                write!(f, "input ended mid-token, {needed:?} more data needed")
            }
            FinalErrorKind::Parse(e) => write!(f, "parser rejected the final input: {e}"),
            FinalErrorKind::ConfigChanged { expected, found } => write!(
                f,
                "session was bound to configuration generation {expected}, found {found}"
            ),
            FinalErrorKind::Closed => write!(f, "session is closed"),
        }
    }
}

#[cfg(feature = "std")]
impl<E> std::error::Error for FinalError<E> where E: fmt::Debug + fmt::Display {}

/// Why [`PartialSession::finalize`] failed
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinalErrorKind<E> {
    /// The input ended in the middle of a token
    Truncated {
        /// The real amount of data the parser still needed
        needed: Needed,
    },
    /// The parser rejected the final input
    Parse(E),
    /// The session was bound to a different configuration generation
    ConfigChanged {
        /// The generation the session was bound to
        expected: u64,
        /// The generation of the configuration passed in
        found: u64,
    },
    /// The session was already closed
    Closed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ContextError;
    use crate::error::ModalResult;
    use crate::token::take;
    use crate::token::take_while;

    fn tag4(input: &mut Partial<&[u8]>) -> ModalResult<Vec<u8>, ContextError> {
        take(4usize).map(|s: &[u8]| s.to_vec()).parse_next(input)
    }

    fn two_tags(input: &mut Partial<&[u8]>) -> ModalResult<(Vec<u8>, Vec<u8>), ContextError> {
        (
            take(2usize).map(|s: &[u8]| s.to_vec()),
            take(4usize).map(|s: &[u8]| s.to_vec()),
        )
            .parse_next(input)
    }

    fn word(input: &mut Partial<&[u8]>) -> ModalResult<Vec<u8>, ContextError> {
        take_while(1.., |b: u8| b.is_ascii_alphabetic())
            .map(|s: &[u8]| s.to_vec())
            .parse_next(input)
    }

    fn complete_tag4(input: &mut &[u8]) -> ModalResult<Vec<u8>, ContextError> {
        take(4usize).map(|s: &[u8]| s.to_vec()).parse_next(input)
    }

    #[test]
    fn two_chunks_one_token_commits_once() {
        let config = SessionConfig::new(8);
        let mut session = PartialSession::new(tag4, &config);

        // First chunk: the token is incomplete, no value is produced
        let outcome = session.feed(&config, b"ab").unwrap();
        assert_eq!(
            outcome,
            FeedOutcome::Incomplete {
                needed: Needed::new(2),
                consumed: 0,
            }
        );
        assert_eq!(session.committed_count(), 0);
        assert_eq!(session.last_needed(), Some(Needed::new(2)));

        // Second chunk: the token completes exactly once
        let outcome = session.feed(&config, b"cd").unwrap();
        assert_eq!(
            outcome,
            FeedOutcome::Committed {
                output: b"abcd".to_vec(),
                consumed: 4,
            }
        );
        assert_eq!(session.committed_count(), 1);

        // A later chunk starts the next token; the accepted result is not re-committed
        let outcome = session.feed(&config, b"e").unwrap();
        assert_eq!(
            outcome,
            FeedOutcome::Incomplete {
                needed: Needed::new(3),
                consumed: 4,
            }
        );
        assert_eq!(session.committed_count(), 1);

        // Cancelling does not re-commit the accepted result
        let cancelled = session.cancel();
        assert_eq!(
            cancelled,
            Cancelled {
                committed: 1,
                consumed: 4,
            }
        );
    }

    #[test]
    fn finalize_truncated_reports_real_position() {
        let config = SessionConfig::new(16);
        let mut session = PartialSession::new(two_tags, &config);

        // The first tag is consumed inside the parser, the second is cut short
        let outcome = session.feed(&config, b"abcd").unwrap();
        assert_eq!(
            outcome,
            FeedOutcome::Incomplete {
                needed: Needed::new(2),
                consumed: 2,
            }
        );
        assert_eq!(session.position(), 2);
        assert_eq!(session.last_needed(), Some(Needed::new(2)));

        // Only finalize turns the temporary EOF into a final error,
        // reporting the real position where parsing stopped
        let err = session.finalize(&config).unwrap_err();
        assert_eq!(err.offset(), 2);
        assert!(matches!(
            err.kind(),
            FinalErrorKind::Truncated { needed } if *needed == Needed::new(2)
        ));
    }

    #[test]
    fn finalize_commits_unterminated_token_once() {
        let config = SessionConfig::new(16);
        let mut session = PartialSession::new(word, &config);

        // On partial input, an unterminated token cannot commit yet
        let outcome = session.feed(&config, b"ab").unwrap();
        assert!(matches!(outcome, FeedOutcome::Incomplete { .. }));
        assert_eq!(session.committed_count(), 0);

        // Finalizing lets it commit exactly once
        let outcome = session.finalize(&config).unwrap();
        assert_eq!(
            outcome,
            FinalizeOutcome::Committed {
                output: b"ab".to_vec(),
                consumed: 2,
            }
        );
    }

    #[test]
    fn config_change_rejects_session() {
        let mut config = SessionConfig::new(8);
        let mut session = PartialSession::new(tag4, &config);

        let outcome = session.feed(&config, b"ab").unwrap();
        assert!(matches!(outcome, FeedOutcome::Incomplete { .. }));

        config.set_buffer_limit(16);
        let err = session.feed(&config, b"cd").unwrap_err();
        assert!(matches!(
            err,
            SessionError::ConfigChanged {
                expected: 0,
                found: 1
            }
        ));

        let err = session.finalize(&config).unwrap_err();
        assert_eq!(err.offset(), 0);
        assert!(matches!(
            err.kind(),
            FinalErrorKind::ConfigChanged {
                expected: 0,
                found: 1
            }
        ));
    }

    #[test]
    fn buffer_limit_rejects_growth_first() {
        let config = SessionConfig::new(4);
        let mut session = PartialSession::new(tag4, &config);

        let outcome = session.feed(&config, b"abc").unwrap();
        assert!(matches!(outcome, FeedOutcome::Incomplete { .. }));

        // The chunk is rejected before the buffer grows past the limit
        let err = session.feed(&config, b"de").unwrap_err();
        assert!(matches!(
            err,
            SessionError::BufferLimitExceeded {
                limit: 4,
                buffered: 3,
                incoming: 2,
            }
        ));
        assert_eq!(session.buffered(), 3);

        // A chunk within the limit still goes through
        let outcome = session.feed(&config, b"d").unwrap();
        assert_eq!(
            outcome,
            FeedOutcome::Committed {
                output: b"abcd".to_vec(),
                consumed: 4,
            }
        );
    }

    #[test]
    fn cancel_drops_pending_input() {
        let config = SessionConfig::new(8);
        let mut session = PartialSession::new(tag4, &config);

        let outcome = session.feed(&config, b"ab").unwrap();
        assert!(matches!(outcome, FeedOutcome::Incomplete { .. }));

        let cancelled = session.cancel();
        assert_eq!(
            cancelled,
            Cancelled {
                committed: 0,
                consumed: 0,
            }
        );
    }

    #[test]
    fn complete_parse_is_unchanged() {
        // The default complete-input path is untouched by sessions
        let parsed = complete_tag4.parse(&b"abcd"[..]).unwrap();
        assert_eq!(parsed, b"abcd".to_vec());
    }
}
