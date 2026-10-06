//! Drive a parser over a token split across three chunks using `stream::Session`.
//!
//! Run with:
//! ```console
//! cargo run --example partial_session
//! ```

use winnow::error::ContextError;
use winnow::error::ModalResult;
use winnow::stream::Partial;
use winnow::stream::Session;
use winnow::stream::SessionConfig;
use winnow::stream::SessionError;
use winnow::token::take;
use winnow::Parser;

/// A record is a 6-byte token; outputs are owned so they can outlive the session buffer
fn record<'i>(i: &mut Partial<&'i [u8]>) -> ModalResult<Vec<u8>, ContextError> {
    take(6usize).map(|s: &[u8]| s.to_vec()).parse_next(i)
}

fn main() {
    let config = SessionConfig::new(1024);
    let mut session = Session::<_, Vec<u8>, ContextError>::new(record, &config);

    // The record "hello!" arrives split across three chunks
    for chunk in [&b"he"[..], b"ll", b"o!"] {
        match session.feed(&config, chunk) {
            Ok(report) => {
                println!(
                    "fed {:?}: accepted={:?} needed={:?} position={}",
                    String::from_utf8_lossy(chunk),
                    report
                        .accepted
                        .iter()
                        .map(|v| String::from_utf8_lossy(v))
                        .collect::<Vec<_>>(),
                    report.needed,
                    report.position,
                );
            }
            Err(err) => {
                eprintln!("feed failed: {err:?}");
                std::process::exit(1);
            }
        }
    }

    // No more input is coming: mark the stream complete
    match session.finalize(&config) {
        Ok(report) => println!("finalized at position {}", report.position),
        Err(SessionError::Parse { offset, .. }) => {
            eprintln!("truncated input at offset {offset}");
            std::process::exit(1);
        }
        Err(err) => {
            eprintln!("finalize failed: {err:?}");
            std::process::exit(1);
        }
    }
}
