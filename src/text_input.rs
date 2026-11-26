//! Text input sessions: how an application tells the platform that a text
//! field is focused, and what kind of text it edits.
//!
//! The protocol follows Flutter's text input channel:
//!
//! 1. [`window::start_text_input`](crate::window::start_text_input) when a
//!    field gains focus. Desktop platforms enable the input method; mobile
//!    and web platforms show the soft keyboard configured by
//!    [`TextInputConfig`].
//! 2. [`window::set_text_input_state`](crate::window::set_text_input_state)
//!    after every local edit or selection change. Mobile and web platforms
//!    keep a native mirror of the text (Android `Editable`, iOS hidden text
//!    view, HTML `<textarea>`) that input methods edit; desktop platforms use
//!    the state for reconversion where supported.
//! 3. [`window::set_text_input_rect`](crate::window::set_text_input_rect)
//!    with the caret rectangle, so candidate windows open next to it.
//! 4. [`window::stop_text_input`](crate::window::stop_text_input) when the
//!    field loses focus.
//!
//! Events arrive through [`EventHandler`](crate::EventHandler):
//!
//! - desktop: `on_ime_preedit` (composition text, caret in UTF-16 units),
//!   `on_ime_commit` (final text), and `char_event` for keys typed without
//!   composition;
//! - Android, iOS, web: `on_ime_state_changed` with the mirror's whole state
//!   after each input-method edit. These platforms never also report the same
//!   edit as a commit or character;
//! - `text_input_action_event` for the keyboard's action key (and Enter in a
//!   single-line field);
//! - `soft_keyboard_event` when a soft keyboard appears, moves, or hides.
//!
//! All offsets are UTF-16 code units, as platform text APIs use.

/// What the field edits; selects the keyboard layout and input-method behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextInputKind {
    #[default]
    Text,
    /// Masked text. Input methods do not compose, suggest, or learn.
    Password,
    Number,
    Decimal,
    Phone,
    Email,
    Url,
    Search,
    /// Several lines: Enter inserts a newline instead of sending an action.
    Multiline,
}

/// The soft keyboard's action key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextInputAction {
    #[default]
    Unspecified,
    Done,
    Go,
    Next,
    Previous,
    Search,
    Send,
    Newline,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextInputCapitalization {
    #[default]
    None,
    Characters,
    Words,
    Sentences,
}

/// Configuration of a text input session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextInputConfig {
    pub kind: TextInputKind,
    pub action: TextInputAction,
    pub autocorrect: bool,
    pub capitalization: TextInputCapitalization,
    /// Selection only: no keyboard, no edits.
    pub read_only: bool,
}

impl Default for TextInputConfig {
    fn default() -> Self {
        Self {
            kind: TextInputKind::Text,
            action: TextInputAction::Unspecified,
            autocorrect: true,
            capitalization: TextInputCapitalization::None,
            read_only: false,
        }
    }
}

impl TextInputConfig {
    pub fn is_multiline(&self) -> bool {
        self.kind == TextInputKind::Multiline
    }

    pub fn is_secure(&self) -> bool {
        self.kind == TextInputKind::Password
    }

    /// The action key, resolving `Unspecified` to Done or Newline.
    pub fn resolved_action(&self) -> TextInputAction {
        match self.action {
            TextInputAction::Unspecified if self.is_multiline() => TextInputAction::Newline,
            TextInputAction::Unspecified => TextInputAction::Done,
            action => action,
        }
    }
}

/// The field's editing state, in UTF-16 code units.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TextInputState {
    pub text: String,
    /// Selection anchor.
    pub selection_start: i32,
    /// Selection extent (the end that moves).
    pub selection_end: i32,
    /// Composing range, or -1 for none.
    pub composing_start: i32,
    pub composing_end: i32,
}

impl TextInputState {
    pub fn has_composing(&self) -> bool {
        self.composing_start >= 0 && self.composing_end > self.composing_start
    }
}

static SESSION: std::sync::Mutex<Option<TextInputConfig>> = std::sync::Mutex::new(None);

/// The active session, readable from backends without the display lock.
#[allow(dead_code)]
pub(crate) fn session() -> Option<TextInputConfig> {
    *SESSION.lock().unwrap()
}

pub(crate) fn set_session(config: Option<TextInputConfig>) {
    *SESSION.lock().unwrap() = config;
}

/// Convert a UTF-16 offset to a byte offset into `text` (clamped).
pub fn utf16_to_byte(text: &str, utf16: usize) -> usize {
    let mut units = 0;
    for (index, ch) in text.char_indices() {
        if units >= utf16 {
            return index;
        }
        units += ch.len_utf16();
    }
    text.len()
}

/// Convert a byte offset into `text` to a UTF-16 offset (clamped).
pub fn byte_to_utf16(text: &str, byte: usize) -> usize {
    let mut end = byte.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].chars().map(char::len_utf16).sum()
}

/// Convert a char offset to a UTF-16 offset.
pub fn chars_to_utf16(text: &str, chars: usize) -> usize {
    text.chars().take(chars).map(char::len_utf16).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_convert_between_units() {
        let text = "a😀你";
        assert_eq!(byte_to_utf16(text, 5), 3);
        assert_eq!(utf16_to_byte(text, 3), 5);
        assert_eq!(chars_to_utf16(text, 2), 3);
        assert_eq!(utf16_to_byte(text, 10), text.len());
    }

    #[test]
    fn unspecified_action_follows_the_kind() {
        let mut config = TextInputConfig::default();
        assert_eq!(config.resolved_action(), TextInputAction::Done);
        config.kind = TextInputKind::Multiline;
        assert_eq!(config.resolved_action(), TextInputAction::Newline);
    }
}
