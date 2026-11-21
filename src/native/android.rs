use crate::{
    event::{EventHandler, KeyCode, KeyMods, TouchPhase},
    native::{
        egl::{self, LibEgl},
        NativeDisplayData,
    },
};

use std::{cell::RefCell, ptr::NonNull, sync::mpsc, thread, time::Duration};

#[cfg(feature = "vulkan")]
use {crate::graphics::vulkan::VulkanContext, ash::vk};

pub use crate::native::gl::{self, *};

mod keycodes;

pub use ndk_sys;

pub mod ndk_utils;

#[no_mangle]
pub unsafe extern "C" fn JNI_OnLoad(
    vm: *mut ndk_sys::JavaVM,
    _: std::ffi::c_void,
) -> ndk_sys::jint {
    VM = vm as *mut _ as _;

    ndk_sys::JNI_VERSION_1_6 as _
}

extern "C" {
    fn quad_main();
}

/// Short recap on how miniquad on Android works
/// There is a MainActivity, a normal Java activity
/// It creates a View and pass a reference to a view to rust.
/// Rust spawn a thread that render things into this view as often as
/// possible.
/// Also MainActivty collects user input events and calls native rust functions.
///
/// This long explanation was to illustrate how we ended up with evets callback
/// and drawing in the different threads.
/// Message enum is used to send data from the callbacks to the drawing thread.
#[derive(Debug)]
enum Message {
    SurfaceChanged {
        width: i32,
        height: i32,
    },
    SurfaceCreated(NativeWindow),
    SurfaceDestroyed(Option<mpsc::Sender<()>>),
    Touch {
        phase: TouchPhase,
        touch_id: u64,
        x: f32,
        y: f32,
        time: u64,
    },
    Character {
        character: u32,
    },
    KeyDown {
        keycode: KeyCode,
    },
    KeyUp {
        keycode: KeyCode,
    },
    Pause,
    Resume,
    Destroy,
    ImePreedit(String),
    ImeCommit(Option<String>),
    ImeStateChanged {
        text: String,
        selection_start: usize,
        selection_end: usize,
        composing_start: Option<usize>,
        composing_end: Option<usize>,
        element_id: u64,
    },
    ImeAction(i32),
    Request(crate::native::Request),
}
/// Owns the reference acquired by ANativeWindow_fromSurface.
#[derive(Debug)]
struct NativeWindow(NonNull<ndk_sys::ANativeWindow>);

// The Android native window reference can move from the Java UI thread to
// the rendering thread. All rendering operations stay on that thread.
unsafe impl Send for NativeWindow {}

impl NativeWindow {
    #[cfg(feature = "vulkan")]
    fn vulkan_handle(&self) -> crate::native::vulkan::VulkanWindow {
        crate::native::vulkan::VulkanWindow::Android {
            window: self.0.as_ptr() as usize,
        }
    }
}

impl Drop for NativeWindow {
    fn drop(&mut self) {
        unsafe { ndk_sys::ANativeWindow_release(self.0.as_ptr()) };
    }
}

thread_local! {
    static MESSAGES_TX: RefCell<Option<mpsc::Sender<Message>>> = RefCell::new(None);
    #[cfg(feature = "vulkan")]
    static SYNC_SURFACE_DESTROY: std::cell::Cell<bool> = std::cell::Cell::new(false);
}

fn send_message(message: Message) {
    MESSAGES_TX.with(|tx| {
        if let Some(tx) = tx.borrow().as_ref() {
            // Java lifecycle events may arrive after the render thread exits.
            let _ = tx.send(message);
        }
    });
}

pub static mut ACTIVITY: ndk_sys::jobject = std::ptr::null_mut();
static mut VM: *mut ndk_sys::JavaVM = std::ptr::null_mut();

pub unsafe fn console_debug(msg: *const ::core::ffi::c_char) {
    ndk_sys::__android_log_write(
        ndk_sys::android_LogPriority_ANDROID_LOG_DEBUG as _,
        b"SAPP\0".as_ptr() as _,
        msg,
    );
}

pub unsafe fn console_info(msg: *const ::core::ffi::c_char) {
    ndk_sys::__android_log_write(
        ndk_sys::android_LogPriority_ANDROID_LOG_INFO as _,
        b"SAPP\0".as_ptr() as _,
        msg,
    );
}

pub unsafe fn console_warn(msg: *const ::core::ffi::c_char) {
    ndk_sys::__android_log_write(
        ndk_sys::android_LogPriority_ANDROID_LOG_WARN as _,
        b"SAPP\0".as_ptr() as _,
        msg,
    );
}

pub unsafe fn console_error(msg: *const ::core::ffi::c_char) {
    ndk_sys::__android_log_write(
        ndk_sys::android_LogPriority_ANDROID_LOG_ERROR as _,
        b"SAPP\0".as_ptr() as _,
        msg,
    );
}

// fn log_info(message: &str) {
//     use std::ffi::CString;

//     let msg = CString::new(message).unwrap_or_else(|_| panic!());

//     unsafe { console_info(msg.as_ptr()) };
// }

struct EglState {
    libegl: LibEgl,
    display: egl::EGLDisplay,
    config: egl::EGLConfig,
    context: egl::EGLContext,
    surface: egl::EGLSurface,
}

impl EglState {
    unsafe fn new(conf: &crate::conf::Conf, window: &NativeWindow) -> Self {
        let mut libegl = LibEgl::try_load().expect("Can't load LibEGL");
        let (context, config, display) = egl::create_egl_context(
            &mut libegl,
            std::ptr::null_mut(), // EGL_DEFAULT_DISPLAY
            conf.platform.framebuffer_alpha,
            conf.sample_count,
        )
        .expect("Can't create EGL context");
        assert!(!display.is_null());
        assert!(!config.is_null());

        crate::native::gl::load_gl_funcs(|proc| {
            let name = std::ffi::CString::new(proc).unwrap();
            (libegl.eglGetProcAddress)(name.as_ptr() as _)
        });

        let mut state = Self {
            libegl,
            display,
            config,
            context,
            surface: std::ptr::null_mut(),
        };
        state.attach(window);
        state
    }

    unsafe fn attach(&mut self, window: &NativeWindow) {
        self.surface = (self.libegl.eglCreateWindowSurface)(
            self.display,
            self.config,
            window.0.as_ptr() as _,
            std::ptr::null_mut(),
        );
        assert!(!self.surface.is_null());
        assert_ne!(
            (self.libegl.eglMakeCurrent)(self.display, self.surface, self.surface, self.context),
            0
        );
    }

    unsafe fn detach(&mut self) {
        if !self.surface.is_null() {
            (self.libegl.eglMakeCurrent)(
                self.display,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            (self.libegl.eglDestroySurface)(self.display, self.surface);
            self.surface = std::ptr::null_mut();
        }
    }
}

impl Drop for EglState {
    fn drop(&mut self) {
        unsafe {
            self.detach();
            (self.libegl.eglDestroyContext)(self.display, self.context);
            (self.libegl.eglTerminate)(self.display);
        }
    }
}

struct MainThreadState {
    event_handler: Box<dyn EventHandler>,
    egl: Option<EglState>,
    window: Option<NativeWindow>,
    #[cfg(feature = "vulkan")]
    vulkan_surface_attached: bool,
    #[cfg(feature = "vulkan")]
    window_size: (i32, i32),
    #[cfg(feature = "vulkan")]
    paused: bool,
    quit: bool,
    fullscreen: bool,
    update_requested: bool,
    keymods: KeyMods,
}

impl MainThreadState {
    #[cfg(feature = "vulkan")]
    fn vulkan_context(&mut self) -> &mut VulkanContext {
        self.event_handler
            .android_vulkan_context()
            .expect("Android Vulkan EventHandler must return its android_vulkan_context")
    }

    #[cfg(feature = "vulkan")]
    fn suspend_vulkan_surface(&mut self) {
        self.vulkan_context().suspend_android_surface();
        self.vulkan_surface_attached = false;
        crate::native_display().lock().unwrap().vulkan_window = None;
    }

    fn detach_window(&mut self) {
        if let Some(egl) = &mut self.egl {
            unsafe { egl.detach() };
        }
        #[cfg(feature = "vulkan")]
        if self.egl.is_none() && self.vulkan_surface_attached {
            self.suspend_vulkan_surface();
        }
        self.window = None;
    }

    #[cfg(feature = "vulkan")]
    fn attach_vulkan_surface(&mut self, size: (u32, u32)) {
        if let Some(window) = &self.window {
            let handle = window.vulkan_handle();
            match self.vulkan_context().resume_android_surface(handle, size) {
                Ok(()) => {
                    self.vulkan_surface_attached = true;
                    crate::native_display().lock().unwrap().vulkan_window = Some(handle);
                }
                Err(vk::Result::ERROR_SURFACE_LOST_KHR | vk::Result::ERROR_OUT_OF_DATE_KHR) => (),
                Err(error) => panic!("Failed to attach Android Vulkan surface: {:?}", error),
            }
        }
    }

    fn waiting_for_vulkan_window(&self) -> bool {
        #[cfg(feature = "vulkan")]
        if self.egl.is_none() {
            return self.paused
                || !self.vulkan_surface_attached
                || self.window_size.0 <= 0
                || self.window_size.1 <= 0;
        }
        false
    }

    fn process_message(&mut self, msg: Message) {
        match msg {
            Message::SurfaceCreated(window) => {
                self.detach_window();
                #[cfg(feature = "vulkan")]
                {
                    self.window_size = (0, 0);
                }
                if let Some(egl) = &mut self.egl {
                    unsafe { egl.attach(&window) };
                }
                self.window = Some(window);
                self.update_requested = true;
            }
            Message::SurfaceDestroyed(done) => {
                self.detach_window();
                if let Some(done) = done {
                    let _ = done.send(());
                }
            }
            Message::SurfaceChanged { width, height } => {
                {
                    let mut d = crate::native_display().lock().unwrap();
                    d.screen_width = width as _;
                    d.screen_height = height as _;
                }
                #[cfg(feature = "vulkan")]
                if self.egl.is_none() {
                    self.window_size = (width, height);
                    if !self.vulkan_surface_attached && width > 0 && height > 0 {
                        self.attach_vulkan_surface((width as u32, height as u32));
                    }
                }
                self.update_requested = true;
                self.event_handler.resize_event(width as _, height as _);
            }
            Message::Touch {
                phase,
                touch_id,
                x,
                y,
                time,
            } => {
                self.event_handler.touch_event(phase, touch_id, x, y, time as f64 / 1000.);
            }
            Message::Character { character } => {
                if let Some(character) = char::from_u32(character) {
                    self.event_handler
                        .char_event(character, Default::default(), false);
                }
            }
            Message::KeyDown { keycode } => {
                match keycode {
                    KeyCode::LeftShift | KeyCode::RightShift => self.keymods.shift = true,
                    KeyCode::LeftControl | KeyCode::RightControl => self.keymods.ctrl = true,
                    KeyCode::LeftAlt | KeyCode::RightAlt => self.keymods.alt = true,
                    KeyCode::LeftSuper | KeyCode::RightSuper => self.keymods.logo = true,
                    _ => {}
                }
                self.event_handler
                    .key_down_event(keycode, self.keymods, false);
            }
            Message::KeyUp { keycode } => {
                match keycode {
                    KeyCode::LeftShift | KeyCode::RightShift => self.keymods.shift = false,
                    KeyCode::LeftControl | KeyCode::RightControl => self.keymods.ctrl = false,
                    KeyCode::LeftAlt | KeyCode::RightAlt => self.keymods.alt = false,
                    KeyCode::LeftSuper | KeyCode::RightSuper => self.keymods.logo = false,
                    _ => {}
                }
                self.event_handler.key_up_event(keycode, self.keymods);
            }
            Message::ImePreedit(text) => {
                let cursor_pos = text.encode_utf16().count();
                self.event_handler.on_ime_preedit(&text, cursor_pos);
            }
            Message::ImeCommit(text) => {
                self.event_handler.on_ime_commit(text.as_deref());
            }
            Message::ImeStateChanged { text, selection_start, selection_end, composing_start, composing_end, element_id } => {
                self.event_handler.on_ime_state_changed(&text, selection_start, selection_end, composing_start, composing_end, element_id);
            }
            Message::ImeAction(_action_code) => {
                self.event_handler.key_down_event(KeyCode::Enter, self.keymods, false);
                self.event_handler.key_up_event(KeyCode::Enter, self.keymods);
            }
            Message::Pause => {
                #[cfg(feature = "vulkan")]
                {
                    self.paused = true;
                }
                self.event_handler.window_minimized_event();
            }
            Message::Resume => {
                #[cfg(feature = "vulkan")]
                {
                    self.paused = false;
                    // A surface lost on pause may come back without a fresh
                    // SurfaceCreated event (the Activity kept the same view).
                    let (width, height) = self.window_size;
                    if self.egl.is_none()
                        && !self.vulkan_surface_attached
                        && width > 0
                        && height > 0
                    {
                        self.attach_vulkan_surface((width as u32, height as u32));
                    }
                }
                self.update_requested = true;
                if self.fullscreen {
                    unsafe {
                        let env = attach_jni_env();
                        set_full_screen(env, true);
                    }
                }

                self.event_handler.window_restored_event()
            }
            Message::Destroy => {
                self.quit = true;
                self.event_handler.quit_requested_event()
            }
            Message::Request(req) => self.process_request(req),
        }
    }

    /// Returns true if a live Vulkan window needs a delayed acquire retry.
    fn frame(&mut self) -> bool {
        if let Some(egl) = &mut self.egl {
            self.event_handler.update();
            if !egl.surface.is_null() {
                self.update_requested = false;
                self.event_handler.draw();
                unsafe { (egl.libegl.eglSwapBuffers)(egl.display, egl.surface) };
            }
            return false;
        }
        #[cfg(feature = "vulkan")]
        {
            if self.waiting_for_vulkan_window() {
                self.event_handler.update();
                return false;
            }
            let (width, height) = self.window_size;
            let context = self.vulkan_context();
            context.set_window_size(width as u32, height as u32);
            match context.prepare_android_frame() {
                Ok(true) => {
                    self.update_requested = false;
                    self.event_handler.update();
                    self.event_handler.draw();
                    let context = self.vulkan_context();
                    if !context.offscreen {
                        // Present an acquired frame even if the handler did not
                        // commit it. This is a no-op after commit_frame().
                        context.present();
                    }
                    if context.android_window_needs_attention() {
                        self.update_requested = true;
                        return true;
                    }
                }
                Ok(false) => return true,
                Err(vk::Result::ERROR_SURFACE_LOST_KHR) => self.suspend_vulkan_surface(),
                Err(error) => panic!("Failed to prepare Android Vulkan frame: {:?}", error),
            }
        }
        false
    }

    fn process_request(&mut self, request: crate::native::Request) {
        use crate::native::Request::*;

        match request {
            ScheduleUpdate => {
                self.update_requested = true;
            }
            SetFullscreen(fullscreen) => {
                unsafe {
                    let env = attach_jni_env();
                    set_full_screen(env, fullscreen);
                }
                self.fullscreen = fullscreen;
            }
            ShowKeyboard(show) => unsafe {
                let env = attach_jni_env();
                ndk_utils::call_void_method!(env, ACTIVITY, "showKeyboard", "(Z)V", show as i32);
            },
            UpdateTextInputState {
                text,
                selection_start,
                selection_end,
                is_password,
                is_multiline,
                element_id,
                max_length,
            } => unsafe {
                let env = attach_jni_env();
                let c_text = std::ffi::CString::new(text).unwrap_or_default();
                let text_jstring = (**env).NewStringUTF.unwrap()(env, c_text.as_ptr());
                ndk_utils::call_void_method!(
                    env,
                    ACTIVITY,
                    "updateTextInputState",
                    "(Ljava/lang/String;IIZZJI)V",
                    text_jstring,
                    selection_start as i32,
                    selection_end as i32,
                    is_password as i32,
                    is_multiline as i32,
                    element_id as i64,
                    max_length as i32
                );
                (**env).DeleteLocalRef.unwrap()(env, text_jstring);
            },
            SetImePosition { .. } => {
                // IME position control not applicable on Android
            }
            SetImeEnabled(..) => {
                // IME enable/disable not applicable on Android
            }
            _ => {}
        }
    }
}

/// Get the JNI Env by calling ndk's AttachCurrentThread
///
/// Safety note: This function is not exactly correct now, it should be fixed!
///
/// AttachCurrentThread should be called at least once for any given thread that
/// wants to use the JNI and DetachCurrentThread should be called only once, when
/// the thread stack is empty and the thread is about to stop
///
/// calling AttachCurrentThread from the same thread multiple time is very cheap
///
/// BUT! there is no DetachCurrentThread call right now, this code:
/// `thread::spawn(|| attach_jni_env());` will lead to internal jni crash :/
/// thread::spawn(|| { attach_jni_env(); loop {} }); is basically what miniquad
/// is doing. this is not correct, but works
/// TODO: the problem here -
/// TODO:   thread::spawn(|| { Attach(); .. Detach() }); will not work as well.
/// TODO: JNI will check that thread's stack is still alive and will crash.
///
/// TODO: Figure how to get into the thread destructor to correctly call Detach
/// TODO: (this should be a GH issue)
/// TODO: for reference - grep for "pthread_setspecific" in SDL2 sources, SDL fixed it!
pub unsafe fn attach_jni_env() -> *mut ndk_sys::JNIEnv {
    let mut env: *mut ndk_sys::JNIEnv = std::ptr::null_mut();
    let attach_current_thread = (**VM).AttachCurrentThread.unwrap();

    let res = attach_current_thread(VM, &mut env, std::ptr::null_mut());
    assert!(res == 0);

    env
}

pub struct AndroidClipboard {}
impl AndroidClipboard {
    pub fn new() -> AndroidClipboard {
        AndroidClipboard {}
    }
}
impl crate::native::Clipboard for AndroidClipboard {
    fn get(&mut self) -> Option<String> {
        unsafe {
            let env = attach_jni_env();

            let text = ndk_utils::call_object_method!(
                env,
                ACTIVITY,
                "getClipboardText",
                "()Ljava/lang/String;"
            );
            if text.is_null() {
                return None;
            }

            let text = ndk_utils::get_utf_str!(env, text);
            Some(text)
        }
    }

    fn set(&mut self, data: &str) {
        let data = std::ffi::CString::new(data).unwrap();
        unsafe {
            let env = attach_jni_env();

            let new_string_utf = (**env).NewStringUTF.unwrap();
            let jtext = new_string_utf(env, data.as_ptr());

            ndk_utils::call_void_method!(
                env,
                ACTIVITY,
                "setClipboardText",
                "(Ljava/lang/String;)V",
                jtext
            );
        }
    }
}

pub unsafe fn run<F>(conf: crate::conf::Conf, f: F)
where
    F: 'static + FnOnce() -> Box<dyn EventHandler>,
{
    if conf.platform.android_panic_hook {
        use std::ffi::CString;
        use std::panic;

        panic::set_hook(Box::new(|info| {
            let msg = CString::new(format!("{info}")).unwrap_or_else(|_| {
                CString::new(format!("MALFORMED ERROR MESSAGE {:?}", info.location())).unwrap()
            });
            console_error(msg.as_ptr());
        }));
    }

    if conf.fullscreen {
        let env = attach_jni_env();
        set_full_screen(env, true);
    }

    // yeah, just adding Send to outer F will do it, but it will brake the API
    // in other backends
    struct SendHack<F>(F);
    unsafe impl<F> Send for SendHack<F> {}

    let f = SendHack(f);
    let vulkan = conf.platform.rendering_backend == crate::conf::RenderingBackend::Vulkan;

    let (tx, rx) = mpsc::channel();
    let tx2 = tx.clone();
    MESSAGES_TX.with(move |messages_tx| *messages_tx.borrow_mut() = Some(tx2));
    #[cfg(feature = "vulkan")]
    SYNC_SURFACE_DESTROY.with(|sync| sync.set(vulkan));

    thread::spawn(move || {
        // Wait for a current native window and its first nonzero extent. A
        // window can be destroyed or replaced even during Activity startup.
        let mut window = None;
        let mut paused = false;
        let (screen_width, screen_height) = loop {
            match rx.recv() {
                Ok(Message::SurfaceCreated(created)) => window = Some(created),
                Ok(Message::SurfaceDestroyed(done)) => {
                    window = None;
                    if let Some(done) = done {
                        let _ = done.send(());
                    }
                }
                Ok(Message::SurfaceChanged { width, height })
                    if window.is_some() && width > 0 && height > 0 =>
                {
                    break (width, height);
                }
                Ok(Message::Pause) => paused = true,
                Ok(Message::Resume) => paused = false,
                Ok(Message::Destroy) | Err(_) => return,
                _ => {}
            }
        };
        let window = window.unwrap();

        // EGL and a Vulkan surface cannot be connected to the same window.
        let egl = if vulkan {
            None
        } else {
            Some(EglState::new(&conf, &window))
        };
        let clipboard = Box::new(AndroidClipboard::new());
        let tx_fn = Box::new(move |req| {
            let _ = tx.send(Message::Request(req));
        });
        crate::set_or_replace_display(NativeDisplayData {
            high_dpi: conf.high_dpi,
            blocking_event_loop: conf.platform.blocking_event_loop,
            rendering_backend: conf.platform.rendering_backend,
            #[cfg(feature = "vulkan")]
            vulkan_window: vulkan.then(|| window.vulkan_handle()),
            ..NativeDisplayData::new(screen_width, screen_height, tx_fn, clipboard)
        });

        let mut s = MainThreadState {
            egl,
            window: Some(window),
            event_handler: f.0(),
            #[cfg(feature = "vulkan")]
            vulkan_surface_attached: vulkan,
            #[cfg(feature = "vulkan")]
            window_size: (screen_width, screen_height),
            #[cfg(feature = "vulkan")]
            paused,
            quit: false,
            fullscreen: conf.fullscreen,
            update_requested: true,
            keymods: KeyMods::default(),
        };
        #[cfg(feature = "vulkan")]
        if vulkan {
            // Reject handlers that cannot follow Android surface changes before
            // allowing them to record anything into the window swapchain.
            s.vulkan_context();
        }
        if paused {
            s.event_handler.window_minimized_event();
        }

        let rx_timeout = conf
            .platform
            .sleep_interval_ms
            .map(|sleep| Duration::from_millis(sleep as u64));

        let mut retry_vulkan = false;
        while !s.quit {
            let waiting_for_window = s.waiting_for_vulkan_window();
            if waiting_for_window
                || retry_vulkan
                || (conf.platform.blocking_event_loop && !s.update_requested)
            {
                let timeout = if retry_vulkan && !waiting_for_window {
                    // Poll Vulkan without blocking on image acquisition while a
                    // Java lifecycle callback may be waiting for this thread.
                    Some(Duration::from_millis(8))
                } else {
                    rx_timeout
                };
                match rx_recv(&rx, timeout) {
                    Ok(msg) => s.process_message(msg),
                    Err(mpsc::RecvTimeoutError::Timeout) => s.update_requested = true,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            while !s.quit {
                let Ok(msg) = rx.try_recv() else { break };
                s.process_message(msg);
            }

            if !s.quit && (!conf.platform.blocking_event_loop || s.update_requested) {
                retry_vulkan = s.frame();
            } else {
                retry_vulkan = false;
            }
            thread::yield_now();
        }

        s.detach_window();
    });
}

/// Adds a call to Receiver as if there was a `.recv_timeout_opt(timeout)`
/// where the `timeout` arg is optional.
fn rx_recv<T>(
    rx: &mpsc::Receiver<T>,
    timeout: Option<Duration>,
) -> Result<T, mpsc::RecvTimeoutError> {
    match timeout {
        Some(timeout) => rx.recv_timeout(timeout),
        // No timeout specified so just do a normal blocking recv()
        None => rx.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected),
    }
}

#[no_mangle]
extern "C" fn jni_on_load(vm: *mut std::ffi::c_void) {
    unsafe {
        VM = vm as _;
    }
}

unsafe fn create_native_window(surface: ndk_sys::jobject) -> Option<NativeWindow> {
    let env = attach_jni_env();
    NonNull::new(ndk_sys::ANativeWindow_fromSurface(env, surface)).map(NativeWindow)
}

#[no_mangle]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_initializeContext(
    env: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    activity: ndk_sys::jobject,
) {
    let env = attach_jni_env();
    let activity_ref = (**env).NewGlobalRef.unwrap()(env, activity);
    ndk_context::initialize_android_context(std::mem::transmute(VM), activity_ref as _);

    let mut env = unsafe { jni::JNIEnv::from_raw(env as *mut jni::sys::JNIEnv).unwrap() };
    let context = unsafe { jni::objects::JObject::from_raw(activity as jni::sys::jobject) };
    let _ = rustls_platform_verifier::android::init_with_env(&mut env, context);
}

#[no_mangle]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_releaseContext(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    ndk_context::release_android_context();
}

#[no_mangle]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_activityOnCreate(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    activity: ndk_sys::jobject,
) {
    let env = attach_jni_env();
    ACTIVITY = (**env).NewGlobalRef.unwrap()(env, activity);
    quad_main();
}

#[no_mangle]
unsafe extern "C" fn Java_quad_1native_QuadNative_activityOnResume(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    send_message(Message::Resume);
}

#[no_mangle]
unsafe extern "C" fn Java_quad_1native_QuadNative_activityOnPause(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    send_message(Message::Pause);
}

#[no_mangle]
unsafe extern "C" fn Java_quad_1native_QuadNative_activityOnDestroy(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    send_message(Message::Destroy);
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnSurfaceCreated(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    surface: ndk_sys::jobject,
) {
    if let Some(window) = unsafe { create_native_window(surface) } {
        send_message(Message::SurfaceCreated(window));
    }
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnSurfaceDestroyed(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    #[cfg(feature = "vulkan")]
    if SYNC_SURFACE_DESTROY.with(|sync| sync.get()) {
        // SurfaceHolder requires the render thread to stop using its surface
        // before this callback returns. An exited render thread drops the sender.
        let (done, wait) = mpsc::channel();
        send_message(Message::SurfaceDestroyed(Some(done)));
        let _ = wait.recv();
        return;
    }
    send_message(Message::SurfaceDestroyed(None));
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnSurfaceChanged(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    _: ndk_sys::jobject,
    width: ndk_sys::jint,
    height: ndk_sys::jint,
) {
    send_message(Message::SurfaceChanged {
        width: width as _,
        height: height as _,
    });
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnTouch(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    touch_id: ndk_sys::jint,
    action: ndk_sys::jint,
    x: ndk_sys::jfloat,
    y: ndk_sys::jfloat,
    time: ndk_sys::jlong,
) {
    let phase = match action {
        0 => TouchPhase::Started,
        1 => TouchPhase::Moved,
        2 => TouchPhase::Ended,
        3 => TouchPhase::Cancelled,
        x => panic!("Unsupported touch phase: {}", x),
    };

    send_message(Message::Touch {
        phase,
        touch_id: touch_id as _,
        x: x as f32,
        y: y as f32,
        time: time as u64,
    });
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnKeyDown(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    keycode: ndk_sys::jint,
) {
    let keycode = keycodes::translate_keycode(keycode as _);

    send_message(Message::KeyDown { keycode });
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnKeyUp(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    keycode: ndk_sys::jint,
) {
    let keycode = keycodes::translate_keycode(keycode as _);

    send_message(Message::KeyUp { keycode });
}

#[no_mangle]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnCharacter(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    character: ndk_sys::jint,
) {
    send_message(Message::Character {
        character: character as u32,
    });
}

#[no_mangle]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_surfaceOnImePreedit(
    env: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    text: ndk_sys::jstring,
) {
    let s = jstring_to_string(env, text).unwrap_or_default();
    send_message(Message::ImePreedit(s));
}

#[no_mangle]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_surfaceOnImeCommit(
    env: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    text: ndk_sys::jstring,
) {
    let s = if text.is_null() {
        None
    } else {
        jstring_to_string(env, text)
    };
    send_message(Message::ImeCommit(s));
}

#[no_mangle]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_surfaceOnImeStateChanged(
    env: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jclass,
    text: ndk_sys::jstring,
    selection_start: ndk_sys::jint,
    selection_end: ndk_sys::jint,
    composing_start: ndk_sys::jint,
    composing_end: ndk_sys::jint,
    element_id: ndk_sys::jlong,
) {
    let s = jstring_to_string(env, text).unwrap_or_default();
    let sel_start = if selection_start >= 0 { selection_start as usize } else { 0 };
    let sel_end = if selection_end >= 0 { selection_end as usize } else { 0 };
    send_message(Message::ImeStateChanged {
        text: s,
        selection_start: sel_start,
        selection_end: sel_end,
        composing_start: if composing_start >= 0 { Some(composing_start as usize) } else { None },
        composing_end: if composing_end >= 0 { Some(composing_end as usize) } else { None },
        element_id: element_id as u64,
    });
}

#[no_mangle]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_surfaceOnImeAction(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jclass,
    action_code: ndk_sys::jint,
) {
    send_message(Message::ImeAction(action_code as i32));
}

unsafe fn jstring_to_string(env: *mut ndk_sys::JNIEnv, jstr: ndk_sys::jstring) -> Option<String> {
    if jstr.is_null() {
        return None;
    }
    let get_string_utf_chars = (**env).GetStringUTFChars.unwrap();
    let release_string_utf_chars = (**env).ReleaseStringUTFChars.unwrap();
    let chars = get_string_utf_chars(env, jstr, std::ptr::null_mut());
    if chars.is_null() {
        return None;
    }
    let s = std::ffi::CStr::from_ptr(chars)
        .to_str()
        .ok()
        .map(|s| s.to_string());
    release_string_utf_chars(env, jstr, chars);
    s
}

unsafe fn set_full_screen(env: *mut ndk_sys::JNIEnv, fullscreen: bool) {
    ndk_utils::call_void_method!(env, ACTIVITY, "setFullScreen", "(Z)V", fullscreen as i32);
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct android_asset {
    pub content: *mut ::core::ffi::c_char,
    pub content_length: ::core::ffi::c_int,
}

// According to documentation, AAssetManager_fromJava is as available as an
// AAssetManager_open, which was used before
// For some reason it is missing fron ndk_sys binding
extern "C" {
    pub fn AAssetManager_fromJava(
        env: *mut ndk_sys::JNIEnv,
        assetManager: ndk_sys::jobject,
    ) -> *mut ndk_sys::AAssetManager;
}

pub(crate) unsafe fn load_asset(filepath: *const ::core::ffi::c_char, out: *mut android_asset) {
    let env = attach_jni_env();

    let get_method_id = (**env).GetMethodID.unwrap();
    let get_object_class = (**env).GetObjectClass.unwrap();
    let call_object_method = (**env).CallObjectMethod.unwrap();

    let mid = (get_method_id)(
        env,
        get_object_class(env, ACTIVITY),
        b"getAssets\0".as_ptr() as _,
        b"()Landroid/content/res/AssetManager;\0".as_ptr() as _,
    );
    let asset_manager = (call_object_method)(env, ACTIVITY, mid);
    let mgr = AAssetManager_fromJava(env, asset_manager);
    let asset = ndk_sys::AAssetManager_open(mgr, filepath, ndk_sys::AASSET_MODE_BUFFER as _);
    if asset.is_null() {
        return;
    }
    let length = ndk_sys::AAsset_getLength64(asset);
    // TODO: memory leak right here! this buffer would never freed
    let buffer = libc::malloc(length as _);
    if ndk_sys::AAsset_read(asset, buffer, length as _) > 0 {
        ndk_sys::AAsset_close(asset);

        (*out).content_length = length as _;
        (*out).content = buffer as _;
    }
}
