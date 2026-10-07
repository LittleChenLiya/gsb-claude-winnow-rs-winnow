use core::ops::Deref;
use core::ops::DerefMut;

use crate::combinator::trace;
use crate::error::ParserError;
use crate::stream::Stateful;
use crate::stream::Stream;
use crate::Parser;
use crate::Result;

/// A scope over a [`Stream::checkpoint`] that rolls the input back unless explicitly
/// [committed][CheckpointGuard::commit]
///
/// On [`Drop`], the input is restored to where the guard started unless
/// [`commit`][CheckpointGuard::commit] was called.  See [`checkpoint_guard`] for running a
/// parser within a guard scope.
///
/// Note that a [`Stream::checkpoint`] only covers the input *position*.  State outside of
/// it, like [`Stateful::state`], is left untouched on rollback unless an explicit
/// [`GuardRecovery`] strategy is selected, see [`checkpoint_guard_with`].
#[derive(Debug)]
pub struct CheckpointGuard<'i, I: Stream> {
    input: &'i mut I,
    checkpoint: I::Checkpoint,
    committed: bool,
}

impl<'i, I: Stream> CheckpointGuard<'i, I> {
    /// Save the current input position, starting a guard scope
    #[inline]
    pub fn new(input: &'i mut I) -> Self {
        let checkpoint = input.checkpoint();
        Self {
            input,
            checkpoint,
            committed: false,
        }
    }

    /// Keep the input advancement made within this guard scope
    ///
    /// Without this, the input is restored to where the guard started.
    #[inline(always)]
    pub fn commit(&mut self) {
        self.committed = true;
    }

    /// Whether [`commit`][Self::commit] was called
    #[inline(always)]
    pub fn is_committed(&self) -> bool {
        self.committed
    }

    /// Restore the input to where the guard started, discarding any
    /// [`commit`][Self::commit]
    #[inline(always)]
    pub fn rollback(&mut self) {
        self.input.reset(&self.checkpoint);
        self.committed = false;
    }
}

impl<I: Stream> Drop for CheckpointGuard<'_, I> {
    #[inline(always)]
    fn drop(&mut self) {
        if !self.committed {
            self.input.reset(&self.checkpoint);
        }
    }
}

impl<I: Stream> Deref for CheckpointGuard<'_, I> {
    type Target = I;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.input
    }
}

impl<I: Stream> DerefMut for CheckpointGuard<'_, I> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.input
    }
}

/// Explicit strategy for recovering state not covered by a [`Stream::checkpoint`]
///
/// A [`Stream::checkpoint`] only covers the input position.  Inputs carrying extra state,
/// like [`Stateful`], require the caller to select whether and how that state
/// participates in a [`CheckpointGuard`] rollback.
pub trait GuardRecovery<Input: Stream> {
    /// The saved state
    type State;
    /// Save off any extra state before the guarded parser runs
    fn save(&self, input: &Input) -> Self::State;
    /// Restore the saved state when the guard scope rolls back
    fn restore(&self, input: &mut Input, state: Self::State);
}

/// Only recover the input position, leaving extra state, like [`Stateful::state`], as-is
///
/// This is the strategy used by [`checkpoint_guard`].
#[derive(Debug, Default, Clone, Copy)]
pub struct PositionOnly;

impl<Input: Stream> GuardRecovery<Input> for PositionOnly {
    type State = ();

    #[inline(always)]
    fn save(&self, _input: &Input) -> Self::State {}

    #[inline(always)]
    fn restore(&self, _input: &mut Input, _state: Self::State) {}
}

/// Recover [`Stateful::state`] by cloning it before the guard scope and writing it back
/// on rollback
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ascii")] {
/// # use winnow::prelude::*;
/// # use winnow::error::ContextError;
/// # use winnow::error::ErrMode;
/// use winnow::combinator::checkpoint_guard_with;
/// use winnow::combinator::CheckpointGuard;
/// use winnow::combinator::CloneState;
/// use winnow::stream::Stateful;
/// use winnow::ascii::alpha1;
///
/// #[derive(Clone, Debug, Default)]
/// struct Count(u32);
///
/// type Stream<'i> = Stateful<&'i str, Count>;
///
/// fn counted_word<'i>() -> impl Parser<Stream<'i>, &'i str, ErrMode<ContextError>> {
///     checkpoint_guard_with(CloneState, |guard: &mut CheckpointGuard<'_, Stream<'i>>| -> ModalResult<&'i str> {
///         guard.state.0 += 1;
///         // no commit: both the position and the count are rolled back
///         alpha1.parse_next(&mut **guard)
///     })
/// }
///
/// let mut input = Stream { input: "abc", state: Count::default() };
/// let word = counted_word().parse_next(&mut input).unwrap();
/// assert_eq!(word, "abc");
/// assert_eq!(input.input, "abc");
/// assert_eq!(input.state.0, 0);
/// # }
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct CloneState;

impl<I, S> GuardRecovery<Stateful<I, S>> for CloneState
where
    I: Stream,
    S: Clone + core::fmt::Debug,
{
    type State = S;

    #[inline(always)]
    fn save(&self, input: &Stateful<I, S>) -> Self::State {
        input.state.clone()
    }

    #[inline(always)]
    fn restore(&self, input: &mut Stateful<I, S>, state: Self::State) {
        input.state = state;
    }
}

/// Apply a parser within a [`CheckpointGuard`] scope, only advancing the input on an
/// explicit [`commit`][CheckpointGuard::commit]
///
/// - `Ok` keeps the input advancement only if the guard was
///   [committed][CheckpointGuard::commit]; otherwise the input is restored to where the
///   guard started.
/// - `Err`, whether from a normal failure or an early return, always restores the input.
///   [`ErrMode::Cut`][crate::error::ErrMode::Cut] still propagates outward but does not
///   commit the consumed input.
///
/// Only the input position is recovered; to also recover extra state, like
/// [`Stateful::state`], select a [`GuardRecovery`] strategy with
/// [`checkpoint_guard_with`].
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ascii")] {
/// # use winnow::prelude::*;
/// # use winnow::error::ContextError;
/// # use winnow::error::ErrMode;
/// use winnow::combinator::checkpoint_guard;
/// use winnow::combinator::CheckpointGuard;
/// use winnow::ascii::alpha1;
///
/// fn word<'i>(commit: bool) -> impl Parser<&'i str, &'i str, ErrMode<ContextError>> {
///     checkpoint_guard(move |guard: &mut CheckpointGuard<'_, &'i str>| -> ModalResult<&'i str> {
///         let word = alpha1.parse_next(&mut **guard)?;
///         if commit {
///             guard.commit();
///         }
///         Ok(word)
///     })
/// }
///
/// // The input only advances when the guard is committed
/// assert_eq!(word(true).parse_peek("abc123"), Ok(("123", "abc")));
/// assert_eq!(word(false).parse_peek("abc123"), Ok(("abc123", "abc")));
/// # }
/// ```
pub fn checkpoint_guard<Input, Output, Error, F>(parser: F) -> impl Parser<Input, Output, Error>
where
    Input: Stream,
    Error: ParserError<Input>,
    F: FnMut(&mut CheckpointGuard<'_, Input>) -> Result<Output, Error>,
{
    checkpoint_guard_with(PositionOnly, parser)
}

/// [`checkpoint_guard`] with an explicit [`GuardRecovery`] strategy for state outside of
/// the [`Stream::checkpoint`]
///
/// See [`CloneState`] for an example.
pub fn checkpoint_guard_with<Input, Output, Error, Recovery, F>(
    recovery: Recovery,
    mut parser: F,
) -> impl Parser<Input, Output, Error>
where
    Input: Stream,
    Error: ParserError<Input>,
    Recovery: GuardRecovery<Input>,
    F: FnMut(&mut CheckpointGuard<'_, Input>) -> Result<Output, Error>,
{
    trace("checkpoint_guard", move |input: &mut Input| {
        let state = recovery.save(input);
        let mut guard = CheckpointGuard::new(input);
        let result = parser(&mut guard);
        if result.is_err() || !guard.is_committed() {
            guard.rollback();
            recovery.restore(&mut guard, state);
        }
        result
    })
}

#[cfg(all(test, feature = "ascii"))]
mod tests {
    use super::*;
    use crate::ascii::alpha1;
    use crate::combinator::alt;
    use crate::combinator::cut_err;
    use crate::combinator::fail;
    use crate::error::ContextError;
    use crate::error::ErrMode;
    use crate::ModalResult;

    /// Guarded word parser: optionally commits, optionally fails after consuming
    fn guarded_word<'i>(
        commit: bool,
        fail_after: bool,
    ) -> impl Parser<&'i str, &'i str, ErrMode<ContextError>> {
        checkpoint_guard(
            move |guard: &mut CheckpointGuard<'_, &'i str>| -> ModalResult<&'i str> {
                let word = alpha1.parse_next(&mut **guard)?;
                if fail_after {
                    return Err(ErrMode::Backtrack(ContextError::new()));
                }
                if commit {
                    guard.commit();
                }
                Ok(word)
            },
        )
    }

    /// The same behavior, written against the original `Stream::checkpoint` /
    /// `Stream::reset` interface, as a regression reference
    fn manual_word<'i>(
        commit: bool,
        fail_after: bool,
    ) -> impl Parser<&'i str, &'i str, ErrMode<ContextError>> {
        move |input: &mut &'i str| -> ModalResult<&'i str> {
            let start = input.checkpoint();
            let word = alpha1.parse_next(input)?;
            if fail_after {
                input.reset(&start);
                return Err(ErrMode::Backtrack(ContextError::new()));
            }
            if !commit {
                input.reset(&start);
            }
            Ok(word)
        }
    }

    #[test]
    fn commit_keeps_position() {
        assert_eq!(
            guarded_word(true, false).parse_peek("abc123"),
            Ok(("123", "abc"))
        );
    }

    #[test]
    fn uncommitted_success_restores_position() {
        assert_eq!(
            guarded_word(false, false).parse_peek("abc123"),
            Ok(("abc123", "abc"))
        );
    }

    #[test]
    fn failure_restores_position() {
        let mut input = "abc";
        let result: ModalResult<&str> = guarded_word(true, true).parse_next(&mut input);
        assert!(matches!(result, Err(ErrMode::Backtrack(_))));
        assert_eq!(input, "abc");
    }

    #[test]
    fn early_return_restores_position() {
        // `?` returns early from the guarded parser; the position is still restored
        let mut input = "123";
        let result: ModalResult<&str> = guarded_word(true, false).parse_next(&mut input);
        assert!(matches!(result, Err(ErrMode::Backtrack(_))));
        assert_eq!(input, "123");
    }

    /// Guarded parser that commits and then hits a `cut_err`
    fn guarded_cut<'i>() -> impl Parser<&'i str, &'i str, ErrMode<ContextError>> {
        checkpoint_guard(
            |guard: &mut CheckpointGuard<'_, &'i str>| -> ModalResult<&'i str> {
                let word = alpha1.parse_next(&mut **guard)?;
                guard.commit();
                let _ = word;
                cut_err(fail).parse_next(&mut **guard)
            },
        )
    }

    #[test]
    fn cut_propagates_without_committing() {
        // `Cut` propagates outward but does not commit the consumed input
        let mut input = "abc";
        let result: ModalResult<&str> = guarded_cut().parse_next(&mut input);
        assert!(matches!(result, Err(ErrMode::Cut(_))));
        assert_eq!(input, "abc");

        // `Cut` still suppresses alternatives once it leaves the guard
        let mut input = "abc";
        let result: ModalResult<&str> = alt((guarded_cut(), alpha1)).parse_next(&mut input);
        assert!(matches!(result, Err(ErrMode::Cut(_))));
        assert_eq!(input, "abc");
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct Count(u32);

    type SStream<'i> = Stateful<&'i str, Count>;

    /// Guarded word parser over a `Stateful` stream, counting each attempt
    fn counted_word<'i, R>(
        recovery: R,
        commit: bool,
        fail_after: bool,
    ) -> impl Parser<SStream<'i>, &'i str, ErrMode<ContextError>>
    where
        R: GuardRecovery<SStream<'i>>,
    {
        checkpoint_guard_with(
            recovery,
            move |guard: &mut CheckpointGuard<'_, SStream<'i>>| -> ModalResult<&'i str> {
                guard.state.0 += 1;
                let word = alpha1.parse_next(&mut **guard)?;
                if fail_after {
                    return Err(ErrMode::Backtrack(ContextError::new()));
                }
                if commit {
                    guard.commit();
                }
                Ok(word)
            },
        )
    }

    #[test]
    fn stateful_clone_state_restores_counter_on_rollback() {
        let mut input = Stateful {
            input: "abc",
            state: Count::default(),
        };
        let word = counted_word(CloneState, false, false)
            .parse_next(&mut input)
            .unwrap();
        assert_eq!(word, "abc");
        assert_eq!(input.input, "abc");
        assert_eq!(input.state.0, 0);
    }

    #[test]
    fn stateful_clone_state_restores_counter_on_failure() {
        let mut input = Stateful {
            input: "abc",
            state: Count::default(),
        };
        let result: ModalResult<&str> = counted_word(CloneState, true, true).parse_next(&mut input);
        assert!(matches!(result, Err(ErrMode::Backtrack(_))));
        assert_eq!(input.input, "abc");
        assert_eq!(input.state.0, 0);
    }

    #[test]
    fn stateful_clone_state_keeps_counter_on_commit() {
        let mut input = Stateful {
            input: "abc",
            state: Count::default(),
        };
        let word = counted_word(CloneState, true, false)
            .parse_next(&mut input)
            .unwrap();
        assert_eq!(word, "abc");
        assert_eq!(input.input, "");
        assert_eq!(input.state.0, 1);
    }

    #[test]
    fn stateful_default_leaves_state_to_caller() {
        // `checkpoint_guard` uses `PositionOnly`: the position is restored but the
        // counter is not, which is why state recovery is an explicit opt-in strategy
        let mut input = Stateful {
            input: "abc",
            state: Count::default(),
        };
        let word = counted_word(PositionOnly, false, false)
            .parse_next(&mut input)
            .unwrap();
        assert_eq!(word, "abc");
        assert_eq!(input.input, "abc");
        assert_eq!(input.state.0, 1);
    }

    fn shape<T>(result: Result<T, ErrMode<ContextError>>) -> Result<T, &'static str> {
        match result {
            Ok(ok) => Ok(ok),
            Err(ErrMode::Backtrack(_)) => Err("backtrack"),
            Err(ErrMode::Cut(_)) => Err("cut"),
            Err(ErrMode::Incomplete(_)) => Err("incomplete"),
        }
    }

    #[test]
    fn matches_manual_checkpoint_reset() {
        // Regression: the guard combinator agrees with the original
        // `Stream::checkpoint` / `Stream::reset` interface
        for input in ["abc123", "123"] {
            for commit in [true, false] {
                for fail_after in [true, false] {
                    let guarded = shape(guarded_word(commit, fail_after).parse_peek(input));
                    let manual = shape(manual_word(commit, fail_after).parse_peek(input));
                    assert_eq!(
                        guarded, manual,
                        "input={input:?} commit={commit} fail_after={fail_after}"
                    );
                }
            }
        }
    }
}
