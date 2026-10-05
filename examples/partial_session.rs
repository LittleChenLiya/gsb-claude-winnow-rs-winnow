//! Streaming fixed-size tags through an explicit `Partial` parse session.
//!
//! Two 4-byte tags (`abcd`, `wxyz`) arrive in three chunks (`ab`, `cdwx`,
//! `yz`). The first chunk cannot complete a token, so nothing is committed;
//! the second and third chunks each commit exactly one token. A truncated
//! stream is only reported when the input is finalized.
//!
//! Run with:
//!
//! ```console
//! $ cargo run --example partial_session
//! ```

use winnow::error::ContextError;
use winnow::stream::FeedOutcome;
use winnow::stream::FinalErrorKind;
use winnow::stream::FinalizeOutcome;
use winnow::stream::Partial;
use winnow::stream::PartialSession;
use winnow::stream::SessionConfig;
use winnow::token::take;
use winnow::ModalResult;
use winnow::Parser;

fn tag(input: &mut Partial<&[u8]>) -> ModalResult<Vec<u8>, ContextError> {
    take(4usize).map(|s: &[u8]| s.to_vec()).parse_next(input)
}

fn main() {
    let config = SessionConfig::new(16);
    let mut session = PartialSession::new(tag, &config);

    // Three chunks carry two 4-byte tags
    for chunk in [&b"ab"[..], b"cdwx", b"yz"] {
        match session.feed(&config, chunk) {
            Ok(FeedOutcome::Incomplete { needed, consumed }) => {
                println!("chunk {chunk:?}: incomplete, needs {needed:?} more, consumed {consumed} bytes so far");
            }
            Ok(FeedOutcome::Committed { output, consumed }) => {
                println!(
                    "chunk {chunk:?}: committed {} ({} bytes consumed in total)",
                    String::from_utf8_lossy(&output),
                    consumed
                );
            }
            Err(err) => {
                println!("chunk {chunk:?}: rejected: {err}");
                break;
            }
        }
    }

    match session.finalize(&config) {
        Ok(FinalizeOutcome::Committed { output, consumed }) => {
            println!(
                "finalize: committed {} ({} bytes consumed in total)",
                String::from_utf8_lossy(&output),
                consumed
            );
        }
        Ok(FinalizeOutcome::Complete { consumed }) => {
            println!("finalize: stream ended cleanly after {consumed} bytes");
        }
        Err(err) => {
            println!("finalize: failed at offset {}: {}", err.offset(), err);
        }
    }

    // A truncated stream only fails when finalized, at the real position
    let config = SessionConfig::new(16);
    let mut session = PartialSession::new(tag, &config);
    let outcome = session.feed(&config, b"tr").unwrap();
    println!("truncated stream: first chunk is only {outcome:?}");
    match session.finalize(&config) {
        Err(err) => {
            assert_eq!(err.offset(), 0);
            assert!(matches!(err.kind(), FinalErrorKind::Truncated { .. }));
            println!("truncated stream: finalize reports {err}");
        }
        outcome => unreachable!("truncated stream must fail at finalize, got {outcome:?}"),
    }
}
