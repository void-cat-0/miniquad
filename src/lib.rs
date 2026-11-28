#![doc = include_str!("../README.md")]
#![allow(
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::unused_unit,
    clippy::identity_op,
    clippy::missing_safety_doc
)]

pub mod conf;
mod event;
pub mod fs;
pub mod graphics;
pub mod native;
use std::collections::HashMap;
use std::ops::{Index, IndexMut};

#[cfg(feature = "log-impl")]
pub mod log;

pub use event::*;
pub mod text_input;
pub use text_input::{
    TextInputAction, TextInputCapitalization, TextInputConfig, TextInputKind, TextInputState,
};

/// A haptic feedback pattern for [`window::haptic_feedback`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HapticFeedback {
    /// A light tap, such as a key press or a small toggle.
    Light,
    Medium,
    /// A strong tap, such as a long press.
    Heavy,
    /// A selection moving through discrete values (pickers, sliders).
    Selection,
    Success,
    Warning,
    Error,
}

pub use graphics::*;

mod default_icon;

pub use native::gl;

#[cfg(target_env = "ohos")]
use napi_derive_ohos::napi;

#[cfg(target_env = "ohos")]
use napi_ohos::{bindgen_prelude::Object, Env, Result};

#[cfg(target_env = "ohos")]
use ohos_hilog_binding::forward_stdio_to_hilog;

#[derive(Clone)]
pub(crate) struct ResourceManager<T> {
    id: usize,
    resources: HashMap<usize, T>,
}

impl<T> Default for ResourceManager<T> {
    fn default() -> Self {
        Self {
            id: 0,
            resources: HashMap::new(),
        }
    }
}

impl<T> ResourceManager<T> {
    pub fn add(&mut self, resource: T) -> usize {
        self.resources.insert(self.id, resource);
        self.id += 1;
        self.id - 1
    }

    pub fn remove(&mut self, id: usize) -> T {
        // Let it crash if the resource is not found
        self.resources.remove(&id).unwrap()
    }
}

impl<T> Index<usize> for ResourceManager<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.resources[&index]
    }
}

impl<T> IndexMut<usize> for ResourceManager<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.resources.get_mut(&index).unwrap()
    }
}

pub mod date {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn now() -> f64 {
        use std::time::SystemTime;

        let time = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|e| panic!("{}", e));
        time.as_secs_f64()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn now() -> f64 {
        use crate::native;

        unsafe { native::wasm::now() }
    }
}

pub type Context = dyn RenderingBackend;

use std::sync::{Mutex, OnceLock};

static NATIVE_DISPLAY: OnceLock<Mutex<native::NativeDisplayData>> = OnceLock::new();

fn set_display(display: native::NativeDisplayData) {
    NATIVE_DISPLAY
        .set(Mutex::new(display))
        .unwrap_or_else(|_| panic!("NATIVE_DISPLAY already set"));
}
/// This for now is Android specific since the process can continue running but the display
/// is restarted. We support reinitializing the display.
#[cfg(any(target_os = "android", target_env = "ohos"))]
fn set_or_replace_display(display: native::NativeDisplayData) {
    if let Some(m) = NATIVE_DISPLAY.get() {
        // Replace existing display
        *m.lock().unwrap() = display;
    } else {
        // First time initialization
        set_display(display);
    }
}
fn native_display() -> &'static Mutex<native::NativeDisplayData> {
    NATIVE_DISPLAY
        .get()
        .expect("Backend has not initialized NATIVE_DISPLAY yet.") //|| Mutex::new(Default::default()))
}

/// Window and associated to window rendering context related functions.
/// in macroquad <= 0.3, it was ctx.screen_size(). Now it is window::screen_size()
pub mod window {
    use super::*;

    /// The same as
    /// ```ignore
    /// if metal {
    ///    Box::new(MetalContext::new())
    /// } else {
    ///   Box::new(GlContext::new())
    /// };
    /// ```
    /// but under #[cfg] gate to avoid MetalContext on non-apple platforms
    pub fn new_rendering_backend() -> Box<dyn RenderingBackend> {
        #[cfg(target_vendor = "apple")]
        {
            if window::apple_gfx_api() == conf::AppleGfxApi::Metal {
                Box::new(MetalContext::new())
            } else {
                Box::new(GlContext::new())
            }
        }
        #[cfg(not(target_vendor = "apple"))]
        Box::new(GlContext::new())
    }

    /// The current framebuffer size in pixels
    /// NOTE: [High DPI Rendering](../conf/index.html#high-dpi-rendering)
    pub fn screen_size() -> (f32, f32) {
        let d = native_display().lock().unwrap();
        (d.screen_width as f32, d.screen_height as f32)
    }

    /// The dpi scaling factor (window pixels to framebuffer pixels)
    /// NOTE: [High DPI Rendering](../conf/index.html#high-dpi-rendering)
    pub fn dpi_scale() -> f32 {
        let d = native_display().lock().unwrap();
        d.dpi_scale
    }

    /// True when high_dpi was requested and actually running in a high-dpi scenario
    /// NOTE: [High DPI Rendering](../conf/index.html#high-dpi-rendering)
    pub fn high_dpi() -> bool {
        let d = native_display().lock().unwrap();
        d.high_dpi
    }

    pub fn blocking_event_loop() -> bool {
        let d = native_display().lock().unwrap();
        d.blocking_event_loop
    }

    /// This function simply quits the application without
    /// giving the user a chance to intervene. Usually this might
    /// be called when the user clicks the 'Ok' button in a 'Really Quit?'
    /// dialog box
    /// Window might not be actually closed right away (exit(0) might not
    /// happen in the order_quit implmentation) and execution might continue for some time after
    /// But the window is going to be inevitably closed at some point.
    pub fn order_quit() {
        let mut d = native_display().lock().unwrap();
        d.quit_ordered = true;
    }

    /// Shortcut for `order_quit`. Will add a legacy attribute at some point.
    pub fn quit() {
        order_quit()
    }

    /// Calling request_quit() will trigger "quit_requested_event" event , giving
    /// the user code a chance to intervene and cancel the pending quit process
    /// (for instance to show a 'Really Quit?' dialog box).
    /// If the event handler callback does nothing, the application will be quit as usual.
    /// To prevent this, call the function "cancel_quit()"" from inside the event handler.
    pub fn request_quit() {
        let mut d = native_display().lock().unwrap();
        d.quit_requested = true;
    }

    /// Cancels a pending quit request, either initiated
    /// by the user clicking the window close button, or programmatically
    /// by calling "request_quit()". The only place where calling this
    /// function makes sense is from inside the event handler callback when
    /// the "quit_requested_event" event has been received
    pub fn cancel_quit() {
        let mut d = native_display().lock().unwrap();
        d.quit_requested = false;
    }
    /// Capture mouse cursor to the current window
    /// On WASM this will automatically hide cursor
    /// On desktop this will bound cursor to windows border
    /// NOTICE: on desktop cursor will not be automatically released after window lost focus
    ///         so set_cursor_grab(false) on window's focus lost is recommended.
    /// TODO: implement window focus events
    pub fn set_cursor_grab(grab: bool) {
        let d = native_display().lock().unwrap();
        #[cfg(target_os = "android")]
        {
            (d.native_requests)(native::Request::SetCursorGrab(grab));
        }

        #[cfg(not(target_os = "android"))]
        {
            d.native_requests
                .send(native::Request::SetCursorGrab(grab))
                .unwrap();
        }
    }

    /// With `conf.platform.blocking_event_loop`, `schedule_update` called from an
    /// event handler makes draw()/update() functions to be called without waiting
    /// for a next event.
    ///
    /// Does nothing without `conf.platform.blocking_event_loop`.
    pub fn schedule_update() {
        #[cfg(all(target_os = "android", not(target_arch = "wasm32")))]
        {
            let d = native_display().lock().unwrap();
            (d.native_requests)(native::Request::ScheduleUpdate);
        }

        #[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
        {
            let d = native_display().lock().unwrap();
            d.native_requests
                .send(native::Request::ScheduleUpdate)
                .unwrap();
        }

        #[cfg(target_arch = "wasm32")]
        unsafe {
            native::wasm::sapp_schedule_update();
        }
    }

    /// Show or hide the mouse cursor
    pub fn show_mouse(shown: bool) {
        let d = native_display().lock().unwrap();
        #[cfg(target_os = "android")]
        {
            (d.native_requests)(native::Request::ShowMouse(shown));
        }

        #[cfg(not(target_os = "android"))]
        {
            d.native_requests
                .send(native::Request::ShowMouse(shown))
                .unwrap();
        }
    }

    /// Set the mouse cursor icon.
    pub fn set_mouse_cursor(cursor_icon: CursorIcon) {
        let d = native_display().lock().unwrap();
        #[cfg(target_os = "android")]
        {
            (d.native_requests)(native::Request::SetMouseCursor(cursor_icon));
        }

        #[cfg(not(target_os = "android"))]
        {
            d.native_requests
                .send(native::Request::SetMouseCursor(cursor_icon))
                .unwrap();
        }
    }

    /// Set the application's window size.
    pub fn set_window_size(new_width: u32, new_height: u32) {
        let d = native_display().lock().unwrap();
        #[cfg(target_os = "android")]
        {
            (d.native_requests)(native::Request::SetWindowSize {
                new_width,
                new_height,
            });
        }

        #[cfg(not(target_os = "android"))]
        {
            d.native_requests
                .send(native::Request::SetWindowSize {
                    new_width,
                    new_height,
                })
                .unwrap();
        }
    }

    pub fn set_window_position(new_x: u32, new_y: u32) {
        let d = native_display().lock().unwrap();
        #[cfg(target_os = "android")]
        {
            (d.native_requests)(native::Request::SetWindowPosition { new_x, new_y });
        }

        #[cfg(not(target_os = "android"))]
        {
            d.native_requests
                .send(native::Request::SetWindowPosition { new_x, new_y })
                .unwrap();
        }
    }

    /// Get the position of the window.
    /// TODO: implement for other platforms
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    pub fn get_window_position() -> (u32, u32) {
        let d = native_display().lock().unwrap();
        d.screen_position
    }

    pub fn set_fullscreen(fullscreen: bool) {
        let d = native_display().lock().unwrap();
        #[cfg(target_os = "android")]
        {
            (d.native_requests)(native::Request::SetFullscreen(fullscreen));
        }

        #[cfg(not(target_os = "android"))]
        {
            d.native_requests
                .send(native::Request::SetFullscreen(fullscreen))
                .unwrap();
        }
    }

    /// Get current OS clipboard value
    pub fn clipboard_get() -> Option<String> {
        let mut d = native_display().lock().unwrap();
        d.clipboard.get()
    }

    /// Save value to OS clipboard
    pub fn clipboard_set(data: &str) {
        let mut d = native_display().lock().unwrap();
        d.clipboard.set(data)
    }
    pub fn dropped_file_count() -> usize {
        let d = native_display().lock().unwrap();
        d.dropped_files.bytes.len()
    }
    pub fn dropped_file_bytes(index: usize) -> Option<Vec<u8>> {
        let d = native_display().lock().unwrap();
        d.dropped_files.bytes.get(index).cloned()
    }
    pub fn dropped_file_path(index: usize) -> Option<std::path::PathBuf> {
        let d = native_display().lock().unwrap();
        d.dropped_files.paths.get(index).cloned()
    }

    fn send_request(d: &native::NativeDisplayData, request: native::Request) {
        #[cfg(target_os = "android")]
        {
            (d.native_requests)(request);
        }

        #[cfg(not(target_os = "android"))]
        {
            let _ = d.native_requests.send(request);
        }
    }

    /// Show or hide the soft keyboard without changing the text input
    /// session. Prefer [`start_text_input`], which also configures it.
    pub fn show_keyboard(show: bool) {
        let d = native_display().lock().unwrap();
        send_request(&d, native::Request::ShowKeyboard(show));
    }

    /// Begin a text input session for a focused field: enable the input
    /// method (desktop) or show the configured soft keyboard (mobile, web).
    /// Starting again replaces the configuration. See [`crate::text_input`].
    pub fn start_text_input(config: crate::TextInputConfig) {
        let mut d = native_display().lock().unwrap();
        d.text_input = Some(config);
        crate::text_input::set_session(Some(config));
        send_request(&d, native::Request::StartTextInput(config));
    }

    /// End the text input session: disable the input method, cancel any
    /// composition, and hide the soft keyboard.
    pub fn stop_text_input() {
        let mut d = native_display().lock().unwrap();
        d.text_input = None;
        crate::text_input::set_session(None);
        send_request(&d, native::Request::StopTextInput);
    }

    /// The active session's configuration.
    pub fn text_input() -> Option<crate::TextInputConfig> {
        native_display().lock().unwrap().text_input
    }

    /// Send the field's editing state after a local edit or selection change.
    /// Offsets are UTF-16 code units. Platforms that mirror the text (Android,
    /// iOS, web) update the mirror without echoing the change back.
    pub fn set_text_input_state(state: &crate::TextInputState) {
        let d = native_display().lock().unwrap();
        send_request(&d, native::Request::SetTextInputState(state.clone()));
    }

    /// The caret (or composition) rectangle in physical window pixels, for
    /// candidate windows and soft-keyboard positioning.
    pub fn set_text_input_rect(x: f32, y: f32, width: f32, height: f32) {
        let d = native_display().lock().unwrap();
        send_request(
            &d,
            native::Request::SetTextInputRect {
                x,
                y,
                w: width,
                h: height,
            },
        );
    }

    /// Show the platform's floating text toolbar (Cut, Copy, Paste, Select
    /// all) above `x, y, width, height`, the selection's bounds in physical
    /// window pixels. The actions apply to the active text input session's
    /// selection and arrive as ordinary editing-state changes. Calling it
    /// again moves the toolbar. Android only; other platforms ignore it.
    pub fn show_text_menu(x: f32, y: f32, width: f32, height: f32) {
        let d = native_display().lock().unwrap();
        send_request(
            &d,
            native::Request::ShowTextMenu {
                x,
                y,
                w: width,
                h: height,
            },
        );
    }

    /// Hide the toolbar opened by [`show_text_menu`]. Stopping the text
    /// input session hides it too.
    pub fn hide_text_menu() {
        let d = native_display().lock().unwrap();
        send_request(&d, native::Request::HideTextMenu);
    }

    /// Play a haptic feedback pattern: Android `performHapticFeedback`, iOS
    /// feedback generators, and `navigator.vibrate` on the web. Does nothing
    /// on desktop platforms or where the user disabled haptics.
    pub fn haptic_feedback(kind: crate::HapticFeedback) {
        let d = native_display().lock().unwrap();
        send_request(&d, native::Request::HapticFeedback(kind));
    }

    /// Whether the platform keeps its own copy of the text and reports whole
    /// editing states (`on_ime_state_changed`) instead of composition and
    /// commit events.
    pub fn text_input_mirrors_state() -> bool {
        cfg!(any(
            target_os = "android",
            target_os = "ios",
            target_arch = "wasm32"
        ))
    }

    /// Legacy name for a caret-rectangle update at one point.
    #[deprecated(note = "use set_text_input_rect")]
    pub fn set_ime_position(x: i32, y: i32) {
        set_text_input_rect(x as f32, y as f32, 0., 0.);
    }

    /// Legacy toggle for a default text input session.
    #[deprecated(note = "use start_text_input / stop_text_input")]
    pub fn set_ime_enabled(enabled: bool) {
        if enabled {
            start_text_input(Default::default());
        } else {
            stop_text_input();
        }
    }

    /// Whether a text input session is active.
    pub fn is_ime_enabled() -> bool {
        native_display().lock().unwrap().text_input.is_some()
    }

    /// Legacy combined session start and state update; `element_id` 0 stops.
    #[deprecated(note = "use start_text_input and set_text_input_state")]
    pub fn update_text_input_state(
        text: String,
        selection_start: usize,
        selection_end: usize,
        is_password: bool,
        is_multiline: bool,
        element_id: u64,
        _max_length: usize,
    ) {
        if element_id == 0 {
            stop_text_input();
            return;
        }
        let kind = if is_password {
            crate::TextInputKind::Password
        } else if is_multiline {
            crate::TextInputKind::Multiline
        } else {
            crate::TextInputKind::Text
        };
        if text_input().map(|config| config.kind) != Some(kind) {
            start_text_input(crate::TextInputConfig {
                kind,
                ..Default::default()
            });
        }
        set_text_input_state(&crate::TextInputState {
            text,
            selection_start: selection_start as i32,
            selection_end: selection_end as i32,
            composing_start: -1,
            composing_end: -1,
        });
    }

    #[cfg(target_vendor = "apple")]
    pub fn apple_gfx_api() -> crate::conf::AppleGfxApi {
        let d = native_display().lock().unwrap();
        d.gfx_api
    }
    #[cfg(target_vendor = "apple")]
    pub fn apple_view() -> crate::native::apple::frameworks::ObjcId {
        let d = native_display().lock().unwrap();
        d.view
    }
    #[cfg(target_os = "ios")]
    pub fn apple_view_ctrl() -> crate::native::apple::frameworks::ObjcId {
        let d = native_display().lock().unwrap();
        d.view_ctrl
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Hash, Eq)]
pub enum CursorIcon {
    Default,
    Help,
    Pointer,
    Wait,
    Crosshair,
    Text,
    Move,
    NotAllowed,
    EWResize,
    NSResize,
    NESWResize,
    NWSEResize,
}

/// Start miniquad.
pub fn start<F>(conf: conf::Conf, f: F)
where
    F: 'static + FnOnce() -> Box<dyn EventHandler>,
{
    #[cfg(target_env = "ohos")]
    unsafe {
        native::ohos::run(conf, f);
    }

    #[cfg(all(target_os = "linux", not(target_env = "ohos")))]
    {
        let mut f = Some(f);
        let f = &mut f;
        match conf.platform.linux_backend {
            conf::LinuxBackend::X11Only => {
                native::linux_x11::run(&conf, f).expect("X11 backend failed")
            }
            conf::LinuxBackend::WaylandOnly => {
                native::linux_wayland::run(&conf, f).expect("Wayland backend failed")
            }
            conf::LinuxBackend::X11WithWaylandFallback => {
                if let Err(err) = native::linux_x11::run(&conf, f) {
                    eprintln!("{err:?}");
                    eprintln!("Failed to initialize through X11! Trying wayland instead");
                    native::linux_wayland::run(&conf, f);
                }
            }
            conf::LinuxBackend::WaylandWithX11Fallback => {
                if native::linux_wayland::run(&conf, f).is_none() {
                    eprintln!("Failed to initialize through wayland! Trying X11 instead");
                    native::linux_x11::run(&conf, f).unwrap()
                }
            }
        }
    }

    #[cfg(target_os = "android")]
    unsafe {
        native::android::run(conf, f);
    }

    #[cfg(target_arch = "wasm32")]
    {
        native::wasm::run(&conf, f);
    }

    #[cfg(target_os = "windows")]
    {
        native::windows::run(&conf, f);
    }

    #[cfg(target_os = "macos")]
    unsafe {
        native::macos::run(conf, f);
    }

    #[cfg(target_os = "ios")]
    unsafe {
        native::ios::run(conf, f);
    }
}

#[cfg(target_env = "ohos")]
unsafe extern "C" {
    fn quad_main();
}

#[cfg(target_env = "ohos")]
static mut OHOS_EXPORTS: Option<Object<'static>> = None;

#[cfg(target_env = "ohos")]
static mut OHOS_ENV: Option<Env> = None;

#[cfg(target_env = "ohos")]
#[napi(module_exports)] //ignore this error ,this is a napi bug.
pub fn init(exports: Object, env: Env) -> Result<()> {
    let _handle = forward_stdio_to_hilog();
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);
        let num_cpus = libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as usize;
        let start_cpu = num_cpus.saturating_sub(4);
        for cpu in start_cpu..num_cpus {
            libc::CPU_SET(cpu, &mut cpuset);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
        OHOS_EXPORTS = Some(std::mem::transmute(exports));
        OHOS_ENV = Some(env);
        quad_main();
    }
    Ok(())
}
