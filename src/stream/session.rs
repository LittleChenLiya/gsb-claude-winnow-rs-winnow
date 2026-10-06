//! Explicit streaming parse sessions over [`Partial`] input
//!
//! A [`Session`] owns a buffer and a parser, letting callers feed input chunk-by-chunk while
//! the parser sees a [`Partial`] stream:
//! - [`ErrMode::Incomplete`] is treated as a *temporary* eof:
//!   the real [`Needed`] and the already-consumed position are saved on the session, and
//!   feeding another chunk resumes parsing instead of failing.
//! - [`Session::finalize`] marks the input complete, converting a still-pending `Incomplete`
//!   into a final error reported at the real (absolute) stream position.
//! - A session is bound to the parser and [`SessionConfig`] it was created with; changing the
//!   configuration rejects further use of the old session.
//! - [`Session::cancel`] drops the session without re-committing already accepted values.
//! - The configured buffer capacity is checked *before* a chunk is appended, so an oversized
//!   chunk is rejected without growing the buffer.
//!
//! The default, complete-input [`Parser::parse`] path is unaffected by all of this.
//!
//! # Example
//!
//! A single token split across three chunks:
//!
//! ```rust
//! use winnow::error::ContextError;
//! use winnow::error::ModalResult;
//! use winnow::stream::Partial;
//! use winnow::stream::Session;
//! use winnow::stream::SessionConfig;
//! use winnow::token::take;
//! use winnow::Parser;
//!
//! // A parser function is higher-ranked over the input lifetime, letting the session
//! // re-borrow its buffer on every feed; outputs are owned so they can outlive the buffer.
//! fn token<'i>(i: &mut Partial<&'i [u8]>) -> ModalResult<Vec<u8>, ContextError> {
//!     take(6usize).map(|s: &[u8]| s.to_vec()).parse_next(i)
//! }
//!
//! let config = SessionConfig::new(1024);
//! let mut session = Session::<_, Vec<u8>, ContextError>::new(token, &config);
//!
//! // `Incomplete`: no value is committed yet
//! let first = session.feed(&config, b"he").unwrap();
//! assert!(first.accepted.is_empty());
//! assert!(first.needed.is_some());
//!
//! // still `Incomplete`
//! let second = session.feed(&config, b"ll").unwrap();
//! assert!(second.accepted.is_empty());
//!
//! // the token completes exactly once
//! let third = session.feed(&config, b"o!").unwrap();
//! assert_eq!(third.accepted, vec![b"hello!".to_vec()]);
//! assert_eq!(third.position, 6);
//! ```
//!
//! # Local verification
//!
//! ```console
//! cargo test --lib stream::session
//! cargo test --doc stream::session
//! cargo run --example partial_session
//! ```

use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::error::ErrMode;
use crate::stream::Needed;
use crate::stream::Offset;
use crate::stream::Partial;
use crate::stream::Stream;
use crate::stream::StreamIsPartial;
use crate::Parser;

/// Configuration bound to a [`Session`]
///
/// Any mutation bumps [`SessionConfig::generation`], rejecting sessions that were created
/// from the previous configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionConfig {
    capacity: usize,
    generation: u64,
}

impl SessionConfig {
    /// Create a configuration with a buffer `capacity` in bytes
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            generation: 0,
        }
    }

    /// Maximum number of uncommitted bytes the session may buffer
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Identity of this configuration revision
    ///
    /// Sessions snapshot the generation at creation and refuse to operate under a
    /// configuration whose generation has moved on.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Change the buffer capacity, rejecting sessions bound to the previous configuration
    #[inline]
    pub fn set_capacity(&mut self, capacity: usize) {
        if capacity != self.capacity {
            self.capacity = capacity;
            self.invalidate();
        }
    }

    /// Explicitly reject all sessions bound to this configuration
    #[inline]
    pub fn invalidate(&mut self) {
        self.generation += 1;
    }
}

impl Default for SessionConfig {
    #[inline]
    fn default() -> Self {
        Self::new(4096)
    }
}

/// Values committed by a single [`Session::feed`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedReport<O> {
    /// Values accepted by this feed, in order; empty while the parser reports `Incomplete`
    pub accepted: Vec<O>,
    /// The real [`Needed`] when the parser is waiting for more data
    pub needed: Option<Needed>,
    /// Absolute offset of the unparsed remainder within the stream
    pub position: usize,
}

/// Values committed by a final [`Session::finalize`] pass
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalizeReport<O> {
    /// Values accepted while draining the completed input
    pub accepted: Vec<O>,
    /// Absolute offset of the end of the accepted stream
    pub position: usize,
}

/// Accounting of a [`Session::cancel`]led session
///
/// Values that were already accepted stay accepted; cancellation only discards the
/// uncommitted remainder and never re-commits them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionSummary {
    /// Number of values committed over the session's lifetime
    pub accepted: usize,
    /// Absolute offset up to which input was consumed
    pub consumed: usize,
    /// Number of uncommitted buffered bytes that were dropped
    pub dropped: usize,
}

/// Errors from driving a [`Session`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError<E> {
    /// Appending the chunk would exceed the configured capacity
    ///
    /// The chunk was rejected before being appended; the buffer is unchanged.
    CapacityExceeded {
        /// Configured [`SessionConfig::capacity`]
        limit: usize,
        /// Buffer size the append would have produced
        attempted: usize,
    },
    /// The session was bound to a different [`SessionConfig`] generation
    ConfigChanged {
        /// Generation the session was created from
        expected: u64,
        /// Generation of the configuration that was passed in
        found: u64,
    },
    /// The parser failed at the given absolute stream offset
    Parse {
        /// Real position of the unparseable remainder within the stream
        offset: usize,
        /// The parser's error
        error: E,
    },
    /// [`Session::finalize`]d while the parser still needed more data
    Truncated {
        /// Real position of the truncated remainder within the stream
        offset: usize,
        /// The real [`Needed`] at the point of truncation
        needed: Needed,
    },
}

/// An explicit streaming parse session over [`Partial`] input
///
/// The session owns the parser and a snapshot of the [`SessionConfig`] it was created with
/// ("mode binding"): every [`feed`][Session::feed] / [`finalize`][Session::finalize] must be
/// given that same, unmodified configuration.
///
/// `O` is the (owned) value the parser produces and `E` its error type, as in
/// `ModalResult<O, E>`.
///
/// The parser is generally a function (e.g. `fn(&mut Partial<&[u8]>) -> ModalResult<O, E>`)
/// so that it is higher-ranked over the session buffer's lifetime.
///
/// See the [module][self] documentation for an example.
pub struct Session<P, O, E> {
    parser: P,
    buffer: Vec<u8>,
    config: SessionConfig,
    accepted: usize,
    consumed: usize,
    needed: Option<Needed>,
    _output: PhantomData<fn() -> (O, E)>,
}

impl<P, O, E> Session<P, O, E> {
    /// Bind a `parser` and a snapshot of `config` into a new session
    #[inline]
    pub fn new(parser: P, config: &SessionConfig) -> Self {
        Self {
            parser,
            buffer: Vec::new(),
            config: *config,
            accepted: 0,
            consumed: 0,
            needed: None,
            _output: PhantomData,
        }
    }

    /// The configuration snapshot this session is bound to
    #[inline]
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Absolute offset of the unparsed remainder within the stream
    #[inline]
    pub fn position(&self) -> usize {
        self.consumed
    }

    /// Number of values committed so far
    #[inline]
    pub fn accepted_count(&self) -> usize {
        self.accepted
    }

    /// The real [`Needed`] saved by the most recent `Incomplete`, if any
    #[inline]
    pub fn needed(&self) -> Option<Needed> {
        self.needed
    }

    /// Number of uncommitted bytes currently buffered
    #[inline]
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// End the session without committing anything further
    ///
    /// Already accepted values are *not* re-committed: they were handed out exactly once by
    /// [`feed`][Session::feed] / [`finalize`][Session::finalize], and `cancel` neither
    /// re-runs the parser nor yields them again.
    #[inline]
    pub fn cancel(self) -> SessionSummary {
        SessionSummary {
            accepted: self.accepted,
            consumed: self.consumed,
            dropped: self.buffer.len(),
        }
    }

    fn check_config(&self, config: &SessionConfig) -> Result<(), SessionError<E>> {
        if config.generation() != self.config.generation() {
            return Err(SessionError::ConfigChanged {
                expected: self.config.generation(),
                found: config.generation(),
            });
        }
        Ok(())
    }
}

impl<P, O, E> Session<P, O, E>
where
    P: for<'i> Parser<Partial<&'i [u8]>, O, ErrMode<E>>,
{
    /// Append a `chunk` and resume parsing
    ///
    /// The chunk is rejected ([`SessionError::CapacityExceeded`]) *before* the buffer grows
    /// if it would exceed the configured capacity.
    ///
    /// A temporary eof ([`ErrMode::Incomplete`]) is not a failure: the real [`Needed`] and
    /// consumed position are saved on the session and reported in the [`FeedReport`].
    pub fn feed(
        &mut self,
        config: &SessionConfig,
        chunk: &[u8],
    ) -> Result<FeedReport<O>, SessionError<E>> {
        self.check_config(config)?;
        let attempted = self.buffer.len().saturating_add(chunk.len());
        if attempted > self.config.capacity() {
            return Err(SessionError::CapacityExceeded {
                limit: self.config.capacity(),
                attempted,
            });
        }
        self.buffer.extend_from_slice(chunk);
        let mut accepted = Vec::new();
        let needed = self.drain(true, &mut accepted)?;
        Ok(FeedReport {
            accepted,
            needed,
            position: self.consumed,
        })
    }

    /// Mark the input complete and convert any pending `Incomplete` into a final error
    ///
    /// Remaining buffered input is parsed as complete; a parser that still needs more data
    /// fails at the real stream position.
    pub fn finalize(
        mut self,
        config: &SessionConfig,
    ) -> Result<FinalizeReport<O>, SessionError<E>> {
        self.check_config(config)?;
        let mut accepted = Vec::new();
        if !self.buffer.is_empty() {
            self.drain(false, &mut accepted)?;
            if !self.buffer.is_empty() {
                return Err(SessionError::Truncated {
                    offset: self.consumed,
                    needed: self.needed.unwrap_or(Needed::Unknown),
                });
            }
        }
        Ok(FinalizeReport {
            accepted,
            position: self.consumed,
        })
    }

    /// Commit as many values as the buffer allows
    ///
    /// With `partial == false` the buffer is treated as complete input.
    fn drain(
        &mut self,
        partial: bool,
        accepted: &mut Vec<O>,
    ) -> Result<Option<Needed>, SessionError<E>> {
        loop {
            let (result, offset) = {
                let mut input = Partial::new(&self.buffer[..]);
                if !partial {
                    let _state = input.complete();
                }
                let start = input.checkpoint();
                let result = self.parser.parse_next(&mut input);
                let offset = input.offset_from(&start);
                (result, offset)
            };
            match result {
                Ok(value) => {
                    self.buffer.drain(..offset);
                    self.consumed += offset;
                    self.accepted += 1;
                    self.needed = None;
                    accepted.push(value);
                    if offset == 0 {
                        // A parser that accepts without consuming would never make progress
                        return Ok(None);
                    }
                }
                Err(ErrMode::Incomplete(needed)) => {
                    self.needed = Some(needed);
                    if partial {
                        // Temporary eof: wait for the next chunk rather than failing
                        return Ok(Some(needed));
                    }
                    return Err(SessionError::Truncated {
                        offset: self.consumed,
                        needed,
                    });
                }
                Err(mode) => {
                    let error = match mode {
                        ErrMode::Backtrack(e) | ErrMode::Cut(e) => e,
                        ErrMode::Incomplete(_) => unreachable!("`Incomplete` handled above"),
                    };
                    return Err(SessionError::Parse {
                        offset: self.consumed,
                        error,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ContextError;
    use crate::error::ModalResult;
    use crate::token::take;

    type Token = Vec<u8>;

    fn token2<'i>(i: &mut Partial<&'i [u8]>) -> ModalResult<Token, ContextError> {
        take(2usize).map(|s: &[u8]| s.to_vec()).parse_next(i)
    }

    fn token4<'i>(i: &mut Partial<&'i [u8]>) -> ModalResult<Token, ContextError> {
        take(4usize).map(|s: &[u8]| s.to_vec()).parse_next(i)
    }

    #[test]
    fn two_chunks_form_one_token() {
        let config = SessionConfig::new(64);
        let mut session = Session::<_, Token, ContextError>::new(token4, &config);

        // First chunk: `Incomplete`, no value committed, real need saved
        let first = session.feed(&config, b"ab").unwrap();
        assert_eq!(first.accepted, Vec::<Token>::new());
        assert_eq!(first.needed, Some(Needed::new(2)));
        assert_eq!(first.position, 0);
        assert_eq!(session.needed(), Some(Needed::new(2)));
        assert_eq!(session.accepted_count(), 0);

        // Second chunk completes the token, exactly once
        let second = session.feed(&config, b"cd").unwrap();
        assert_eq!(second.accepted, vec![b"abcd".to_vec()]);
        assert_eq!(second.position, 4);
        assert_eq!(session.accepted_count(), 1);

        // Feeding again must not re-commit the accepted value
        let third = session.feed(&config, b"").unwrap();
        assert_eq!(third.accepted, Vec::<Token>::new());
        assert_eq!(session.accepted_count(), 1);
    }

    #[test]
    fn finalize_reports_real_position_on_truncation() {
        let config = SessionConfig::new(64);
        let mut session = Session::<_, Token, ContextError>::new(token2, &config);

        let report = session.feed(&config, b"abcd").unwrap();
        assert_eq!(report.accepted, vec![b"ab".to_vec(), b"cd".to_vec()]);

        // Dangling byte: `Incomplete` while partial...
        let report = session.feed(&config, b"e").unwrap();
        assert_eq!(report.needed, Some(Needed::new(1)));

        // ...but finalizing converts it into a final error at the real position
        let err = session.finalize(&config).unwrap_err();
        assert_eq!(
            err,
            SessionError::Parse {
                offset: 4,
                error: ContextError::new(),
            }
        );
    }

    #[test]
    fn finalize_accepts_clean_token_boundary() {
        let config = SessionConfig::new(64);
        let mut session = Session::<_, Token, ContextError>::new(token2, &config);
        session.feed(&config, b"abcd").unwrap();
        let report = session.finalize(&config).unwrap();
        assert_eq!(report.accepted, Vec::<Token>::new());
        assert_eq!(report.position, 4);
    }

    #[test]
    fn config_change_rejects_old_session() {
        let mut config = SessionConfig::new(64);
        let mut session = Session::<_, Token, ContextError>::new(token4, &config);
        session.feed(&config, b"ab").unwrap();

        config.set_capacity(128);
        assert_eq!(
            session.feed(&config, b"cd"),
            Err(SessionError::ConfigChanged {
                expected: 0,
                found: 1,
            })
        );
        assert_eq!(
            session.finalize(&config),
            Err(SessionError::ConfigChanged {
                expected: 0,
                found: 1,
            })
        );
    }

    #[test]
    fn capacity_is_checked_before_growing_the_buffer() {
        let config = SessionConfig::new(4);
        let mut session = Session::<_, Token, ContextError>::new(token4, &config);
        session.feed(&config, b"ab").unwrap();

        // Rejected before the buffer grows...
        assert_eq!(
            session.feed(&config, b"cde"),
            Err(SessionError::CapacityExceeded {
                limit: 4,
                attempted: 5,
            })
        );
        // ...leaving the buffered input untouched
        assert_eq!(session.buffered(), 2);
        let report = session.feed(&config, b"cd").unwrap();
        assert_eq!(report.accepted, vec![b"abcd".to_vec()]);
    }

    #[test]
    fn cancel_does_not_recommit_accepted_values() {
        let config = SessionConfig::new(64);
        let mut session = Session::<_, Token, ContextError>::new(token2, &config);

        let report = session.feed(&config, b"ab").unwrap();
        assert_eq!(report.accepted, vec![b"ab".to_vec()]);
        session.feed(&config, b"c").unwrap();

        // The accepted value was handed out exactly once, above; cancel only accounts for it
        let summary = session.cancel();
        assert_eq!(
            summary,
            SessionSummary {
                accepted: 1,
                consumed: 2,
                dropped: 1,
            }
        );
    }

    #[test]
    fn complete_parse_is_unaffected() {
        fn take4<'i>(i: &mut &'i [u8]) -> ModalResult<&'i [u8], ContextError> {
            take(4usize).parse_next(i)
        }

        assert_eq!(take4.parse(&b"abcd"[..]), Ok(&b"abcd"[..]));
        assert!(take4.parse(&b"abc"[..]).is_err());
    }
}
