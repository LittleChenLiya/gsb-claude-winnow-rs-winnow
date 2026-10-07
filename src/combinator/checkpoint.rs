//! Scoped checkpointing of the input, rewound unless explicitly committed

use core::ops::Deref;
use core::ops::DerefMut;

use crate::combinator::trace;
use crate::error::ParserError;
use crate::stream::Stateful;
use crate::stream::Stream;
use crate::Parser;
use crate::Result;

/// Run a parser within a [`CheckpointGuard`] scope, keeping the input
/// advancement only if the guard is explicitly
/// [`commit`][CheckpointGuard::commit]ted
///
/// - On failure or early return (e.g. via `?`), the input is rewound to the
///   position it had when the scope was entered
/// - [`ErrMode::Cut`][crate::error::ErrMode::Cut] errors propagate outward
///   unchanged but do not secretly commit; the input is still rewound
/// - Returning `Ok` does not commit either; only
///   [`CheckpointGuard::commit`] keeps the advancement
///
/// For [`Stateful`] inputs, this uses the [`PositionOnly`] policy: the user
/// state is left untouched. To also roll the state back, select the policy
/// explicitly with [`checkpoint_guard_with`] and [`WithState`].
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ascii")] {
/// # use winnow::prelude::*;
/// use winnow::combinator::checkpoint_guard;
/// use winnow::combinator::CheckpointGuard;
/// use winnow::ascii::alpha1;
/// use winnow::token::literal;
///
/// fn terminated_word<'i>(i: &mut &'i str) -> ModalResult<&'i str> {
///     checkpoint_guard(|g: &mut CheckpointGuard<'_, &'i str>| -> ModalResult<&'i str> {
///         let word = alpha1.parse_next(&mut **g)?;
///         literal(";").parse_next(&mut **g)?;
///         g.commit();
///         Ok(word)
///     })
///     .parse_next(i)
/// }
///
/// // Committed: the input stays advanced
/// assert_eq!(terminated_word.parse_peek("abc;rest"), Ok(("rest", "abc")));
///
/// // Early return on failure: the input is rewound
/// let mut input = "abc-rest";
/// assert!(terminated_word.parse_next(&mut input).is_err());
/// assert_eq!(input, "abc-rest");
/// # }
/// ```
#[doc(alias = "transaction")]
#[doc(alias = "rewind")]
pub fn checkpoint_guard<Input, Output, Error, F>(f: F) -> impl Parser<Input, Output, Error>
where
    Input: Stream,
    Error: ParserError<Input>,
    F: FnMut(&mut CheckpointGuard<'_, Input>) -> Result<Output, Error>,
{
    checkpoint_guard_with(PositionOnly, f)
}

/// [`checkpoint_guard`] with an explicit [`GuardPolicy`] for what gets restored
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ascii")] {
/// # use winnow::prelude::*;
/// use winnow::combinator::checkpoint_guard_with;
/// use winnow::combinator::CheckpointGuard;
/// use winnow::combinator::WithState;
/// use winnow::stream::Stateful;
/// use winnow::ascii::alpha1;
///
/// #[derive(Clone, Debug, Default, PartialEq)]
/// struct State {
///     count: usize,
/// }
///
/// type Stream<'is> = Stateful<&'is str, State>;
///
/// fn counted_word<'i>(i: &mut Stream<'i>) -> ModalResult<&'i str> {
///     checkpoint_guard_with(WithState, |g: &mut CheckpointGuard<'_, Stream<'i>, WithState>| -> ModalResult<&'i str> {
///         g.state.count += 1;
///         let word = alpha1.parse_next(&mut **g)?;
///         g.commit();
///         Ok(word)
///     })
///     .parse_next(i)
/// }
///
/// // Committed: position and state are kept
/// let mut input = Stream { input: "abc;", state: State::default() };
/// assert_eq!(counted_word.parse_next(&mut input).unwrap(), "abc");
/// assert_eq!(input.state.count, 1);
///
/// // Rewound: position and state are rolled back
/// let mut input = Stream { input: "123;", state: State::default() };
/// assert!(counted_word.parse_next(&mut input).is_err());
/// assert_eq!(input.input, "123;");
/// assert_eq!(input.state.count, 0);
/// # }
/// ```
pub fn checkpoint_guard_with<Input, Policy, Output, Error, F>(
    _policy: Policy,
    mut f: F,
) -> impl Parser<Input, Output, Error>
where
    Input: Stream,
    Policy: GuardPolicy<Input>,
    Error: ParserError<Input>,
    F: FnMut(&mut CheckpointGuard<'_, Input, Policy>) -> Result<Output, Error>,
{
    trace("checkpoint_guard", move |input: &mut Input| {
        let mut guard = CheckpointGuard::<Input, Policy>::new(input);
        // On every exit path without `commit`, dropping `guard` rewinds the
        // input according to `Policy`, including `ErrMode::Cut` errors which
        // are propagated to the caller untouched.
        f(&mut guard)
    })
}

/// Scoped access to the input, rewound on drop unless [`commit`][CheckpointGuard::commit]ted
///
/// See [`checkpoint_guard`].
///
/// The guard dereferences to the input; pass `&mut **guard` to sub-parsers.
pub struct CheckpointGuard<'i, I: Stream, P: GuardPolicy<I> = PositionOnly> {
    input: &'i mut I,
    saved: Option<P::Saved>,
    committed: bool,
}

impl<'i, I: Stream, P: GuardPolicy<I>> CheckpointGuard<'i, I, P> {
    fn new(input: &'i mut I) -> Self {
        let saved = P::save(input);
        Self {
            input,
            saved: Some(saved),
            committed: false,
        }
    }

    /// Keep the input advancement made within the guard's scope
    ///
    /// Without this, dropping the guard rewinds the input according to the
    /// [`GuardPolicy`], even when the scope exited successfully.
    #[inline]
    pub fn commit(&mut self) {
        self.committed = true;
    }
}

impl<I: Stream, P: GuardPolicy<I>> Deref for CheckpointGuard<'_, I, P> {
    type Target = I;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.input
    }
}

impl<I: Stream, P: GuardPolicy<I>> DerefMut for CheckpointGuard<'_, I, P> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.input
    }
}

impl<I: Stream, P: GuardPolicy<I>> Drop for CheckpointGuard<'_, I, P> {
    fn drop(&mut self) {
        if !self.committed {
            if let Some(saved) = self.saved.take() {
                P::restore(self.input, saved);
            }
        }
    }
}

/// How a [`CheckpointGuard`] saves and restores the input when it is dropped
/// without being [`commit`][CheckpointGuard::commit]ted
///
/// [`Stream::checkpoint`] only covers the parse position; for inputs carrying
/// extra state, like [`Stateful`], restoring that state must be opted into by
/// explicitly selecting a policy like [`WithState`].
pub trait GuardPolicy<I: Stream> {
    /// Data saved when the guard's scope is entered
    type Saved;
    /// Save everything needed to later [`restore`][GuardPolicy::restore] the input
    fn save(input: &I) -> Self::Saved;
    /// Revert the input to what [`save`][GuardPolicy::save] captured
    fn restore(input: &mut I, saved: Self::Saved);
}

/// Only revert the parse position (default policy)
///
/// For [`Stateful`] inputs, the user state is left as-is.
#[derive(Debug)]
pub struct PositionOnly;

impl<I: Stream> GuardPolicy<I> for PositionOnly {
    type Saved = I::Checkpoint;

    #[inline(always)]
    fn save(input: &I) -> Self::Saved {
        input.checkpoint()
    }

    #[inline(always)]
    fn restore(input: &mut I, saved: Self::Saved) {
        input.reset(&saved);
    }
}

/// Revert both the parse position and the [`Stateful`] user state
///
/// This requires the state to be [`Clone`]able so a copy can be kept.
#[derive(Debug)]
pub struct WithState;

impl<I: Stream, S: Clone + core::fmt::Debug> GuardPolicy<Stateful<I, S>> for WithState {
    type Saved = (I::Checkpoint, S);

    #[inline(always)]
    fn save(input: &Stateful<I, S>) -> Self::Saved {
        (input.input.checkpoint(), input.state.clone())
    }

    #[inline(always)]
    fn restore(input: &mut Stateful<I, S>, saved: Self::Saved) {
        input.input.reset(&saved.0);
        input.state = saved.1;
    }
}

#[cfg(all(test, feature = "ascii"))]
mod tests {
    use super::*;

    use crate::ascii::alpha1;
    use crate::combinator::cut_err;
    use crate::combinator::opt;
    use crate::combinator::peek;
    use crate::error::ErrMode;
    use crate::token::literal;
    use crate::ModalResult;

    fn scoped_word<'i>(i: &mut &'i str) -> ModalResult<&'i str> {
        checkpoint_guard(|g: &mut CheckpointGuard<'_, &'i str>| -> ModalResult<&'i str> {
            let word = alpha1.parse_next(&mut **g)?;
            literal(";").parse_next(&mut **g)?;
            g.commit();
            Ok(word)
        })
        .parse_next(i)
    }

    #[test]
    fn commit_keeps_advancement() {
        assert_eq!(scoped_word.parse_peek("abc;rest"), Ok(("rest", "abc")));
    }

    #[test]
    fn failure_rewinds() {
        let mut input = "abc-rest";
        let err = scoped_word.parse_next(&mut input).unwrap_err();
        assert!(matches!(err, ErrMode::Backtrack(_)));
        assert_eq!(input, "abc-rest");
    }

    #[test]
    fn uncommitted_ok_rewinds() {
        fn parser<'i>(i: &mut &'i str) -> ModalResult<&'i str> {
            checkpoint_guard(|g: &mut CheckpointGuard<'_, &'i str>| -> ModalResult<&'i str> {
                // no `commit`: even `Ok` does not keep the advancement
                alpha1.parse_next(&mut **g)
            })
            .parse_next(i)
        }

        let mut input = "abc;";
        assert_eq!(parser.parse_next(&mut input).unwrap(), "abc");
        assert_eq!(input, "abc;");
    }

    #[test]
    fn cut_propagates_without_committing() {
        fn parser<'i>(i: &mut &'i str) -> ModalResult<&'i str> {
            checkpoint_guard(|g: &mut CheckpointGuard<'_, &'i str>| -> ModalResult<&'i str> {
                let word = alpha1.parse_next(&mut **g)?;
                cut_err(literal(";")).parse_next(&mut **g)?;
                g.commit();
                Ok(word)
            })
            .parse_next(i)
        }

        let mut input = "abc-rest";
        let err = parser.parse_next(&mut input).unwrap_err();
        // The Cut error propagates outward unchanged ...
        assert!(matches!(err, ErrMode::Cut(_)));
        // ... but it did not secretly commit: the position is restored
        assert_eq!(input, "abc-rest");
    }

    #[derive(Clone, Debug, Default, PartialEq)]
    struct Count {
        parses: usize,
    }

    type StatefulStream<'is> = Stateful<&'is str, Count>;

    #[test]
    fn stateful_position_only_keeps_state() {
        fn parser(i: &mut StatefulStream<'_>) -> ModalResult<()> {
            checkpoint_guard(
                |g: &mut CheckpointGuard<'_, StatefulStream<'_>>| -> ModalResult<()> {
                    g.state.parses += 1;
                    alpha1.parse_next(&mut **g)?;
                    g.commit();
                    Ok(())
                },
            )
            .parse_next(i)
        }

        let mut input = StatefulStream {
            input: "123",
            state: Count::default(),
        };
        assert!(parser.parse_next(&mut input).is_err());
        // PositionOnly: the position is rewound, the state is left as-is
        assert_eq!(input.input, "123");
        assert_eq!(input.state.parses, 1);
    }

    #[test]
    fn stateful_with_state_restores_state() {
        fn parser(i: &mut StatefulStream<'_>) -> ModalResult<()> {
            checkpoint_guard_with(
                WithState,
                |g: &mut CheckpointGuard<'_, StatefulStream<'_>, WithState>| -> ModalResult<()> {
                    g.state.parses += 1;
                    alpha1.parse_next(&mut **g)?;
                    g.commit();
                    Ok(())
                },
            )
            .parse_next(i)
        }

        // Failure: both position and state are rolled back
        let mut input = StatefulStream {
            input: "123",
            state: Count::default(),
        };
        assert!(parser.parse_next(&mut input).is_err());
        assert_eq!(input.input, "123");
        assert_eq!(input.state.parses, 0);

        // Commit: both position and state are kept
        let mut input = StatefulStream {
            input: "abc",
            state: Count::default(),
        };
        parser.parse_next(&mut input).unwrap();
        assert_eq!(input.input, "");
        assert_eq!(input.state.parses, 1);
    }

    #[test]
    fn default_interface_regression() {
        // Baseline: the existing combinators' rewind/commit semantics are
        // unchanged, so their results can be cross-checked against the guard.

        // `peek` never advances the input
        fn peek_word<'i>(i: &mut &'i str) -> ModalResult<&'i str> {
            peek(alpha1).parse_next(i)
        }
        assert_eq!(peek_word.parse_peek("abc;"), Ok(("abc;", "abc")));

        // `opt` rewinds on backtrack and produces `None`
        fn opt_word<'i>(i: &mut &'i str) -> ModalResult<Option<(&'i str, &'i str)>> {
            opt((alpha1, literal(";"))).parse_next(i)
        }
        assert_eq!(
            opt_word.parse_peek("abc;rest"),
            Ok(("rest", Some(("abc", ";"))))
        );
        assert_eq!(opt_word.parse_peek("abc-rest"), Ok(("abc-rest", None)));

        // `cut_err` propagates as Cut and keeps the partial advancement,
        // unlike the guard which rewinds unless committed
        fn cut_word<'i>(i: &mut &'i str) -> ModalResult<&'i str> {
            let word = alpha1.parse_next(i)?;
            cut_err(literal(";")).parse_next(i)?;
            Ok(word)
        }
        let mut input = "abc-rest";
        let err = cut_word.parse_next(&mut input).unwrap_err();
        assert!(matches!(err, ErrMode::Cut(_)));
        assert_eq!(input, "-rest");
    }
}
