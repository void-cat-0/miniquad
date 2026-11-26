#![allow(dead_code)]

#[cfg(not(target_os = "android"))]
use std::sync::mpsc;

#[derive(Default)]
pub(crate) struct DroppedFiles {
    pub paths: Vec<std::path::PathBuf>,
    pub bytes: Vec<Vec<u8>>,
}
pub(crate) struct NativeDisplayData {
    pub screen_width: i32,
    pub screen_height: i32,
    pub screen_position: (u32, u32),
    pub text_input: Option<crate::TextInputConfig>,
    pub dpi_scale: f32,
    pub high_dpi: bool,
    pub quit_requested: bool,
    pub quit_ordered: bool,
    #[cfg(target_os = "android")]
    pub native_requests: Box<dyn Fn(Request) + Send>,
    #[cfg(not(target_os = "android"))]
    pub native_requests: mpsc::Sender<Request>,
    pub clipboard: Box<dyn Clipboard>,
    pub dropped_files: DroppedFiles,
    pub blocking_event_loop: bool,

    #[cfg(target_vendor = "apple")]
    pub view: crate::native::apple::frameworks::ObjcId,
    #[cfg(target_os = "ios")]
    pub view_ctrl: crate::native::apple::frameworks::ObjcId,
    #[cfg(target_vendor = "apple")]
    pub gfx_api: crate::conf::AppleGfxApi,
}
#[cfg(target_vendor = "apple")]
unsafe impl Send for NativeDisplayData {}
#[cfg(target_vendor = "apple")]
unsafe impl Sync for NativeDisplayData {}

impl NativeDisplayData {
    pub fn new(
        screen_width: i32,
        screen_height: i32,
        #[cfg(target_os = "android")] native_requests: Box<dyn Fn(Request) + Send>,
        #[cfg(not(target_os = "android"))] native_requests: mpsc::Sender<Request>,
        clipboard: Box<dyn Clipboard>,
    ) -> NativeDisplayData {
        NativeDisplayData {
            screen_width,
            screen_height,
            screen_position: (0, 0),
            text_input: None,
            dpi_scale: 1.,
            high_dpi: false,
            quit_requested: false,
            quit_ordered: false,
            native_requests,
            clipboard,
            dropped_files: Default::default(),
            blocking_event_loop: false,
            #[cfg(target_vendor = "apple")]
            gfx_api: crate::conf::AppleGfxApi::OpenGl,
            #[cfg(target_vendor = "apple")]
            view: std::ptr::null_mut(),
            #[cfg(target_os = "ios")]
            view_ctrl: std::ptr::null_mut(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum Request {
    ScheduleUpdate,
    SetCursorGrab(bool),
    ShowMouse(bool),
    SetMouseCursor(crate::CursorIcon),
    SetWindowSize { new_width: u32, new_height: u32 },
    SetWindowPosition { new_x: u32, new_y: u32 },
    SetFullscreen(bool),
    ShowKeyboard(bool),
    StartTextInput(crate::TextInputConfig),
    StopTextInput,
    SetTextInputState(crate::TextInputState),
    SetTextInputRect { x: f32, y: f32, w: f32, h: f32 },
}

/// The legacy combined text input update used by the mobile and web
/// backends, derived from a new-style request and the active session.
#[allow(dead_code)]
pub(crate) struct LegacyTextInput {
    pub text: String,
    pub selection_start: usize,
    pub selection_end: usize,
    pub is_password: bool,
    pub is_multiline: bool,
    /// 1 during a session, 0 to end it.
    pub element_id: u64,
    pub max_length: usize,
}

#[allow(dead_code)]
impl LegacyTextInput {
    pub fn from_request(request: &Request) -> Self {
        let config = crate::text_input::session();
        match request {
            Request::SetTextInputState(state) if config.is_some() => {
                let config = config.unwrap();
                Self {
                    text: state.text.clone(),
                    selection_start: state.selection_start.max(0) as usize,
                    selection_end: state.selection_end.max(0) as usize,
                    is_password: config.is_secure(),
                    is_multiline: config.is_multiline(),
                    element_id: 1,
                    max_length: 0,
                }
            }
            _ => Self {
                text: String::new(),
                selection_start: 0,
                selection_end: 0,
                is_password: false,
                is_multiline: false,
                element_id: 0,
                max_length: 0,
            },
        }
    }
}

pub trait Clipboard: Send + Sync {
    fn get(&mut self) -> Option<String>;
    fn set(&mut self, string: &str);
}

pub mod module;

#[cfg(target_env = "ohos")]
pub mod ohos;

#[cfg(target_env = "ohos")]
pub use ohos::*;

#[cfg(all(target_os = "linux", not(target_env = "ohos")))]
pub mod linux_x11;

#[cfg(all(target_os = "linux", not(target_env = "ohos")))]
pub mod linux_wayland;

#[cfg(target_os = "android")]
pub mod android;

#[cfg(target_os = "windows")]
pub mod windows;

#[cfg(target_os = "android")]
pub use android::*;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod apple;

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "ios")]
pub mod ios;

#[cfg(any(target_os = "android", target_os = "linux"))]
pub mod egl;

// there is no glGetProcAddr on webgl, so its impossible to make "gl" module work
// on macos.. well, there is, but way easier to just statically link to gl
#[cfg(not(target_arch = "wasm32"))]
pub mod gl;

#[cfg(target_arch = "wasm32")]
pub use wasm::webgl as gl;

pub mod query_stab;
