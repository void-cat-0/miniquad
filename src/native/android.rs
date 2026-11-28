use crate::{
    event::{EventHandler, KeyCode, KeyMods, TouchPhase},
    native::{
        egl::{self, LibEgl},
        NativeDisplayData,
    },
};

use std::{cell::RefCell, sync::mpsc, thread, time::Duration};

pub use crate::native::gl::{self, *};

mod keycodes;

pub use ndk_sys;

pub mod ndk_utils;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn JNI_OnLoad(
    vm: *mut ndk_sys::JavaVM,
    _: std::ffi::c_void,
) -> ndk_sys::jint { unsafe {
    VM = vm as *mut _ as _;

    ndk_sys::JNI_VERSION_1_6 as _
}}

unsafe extern "C" {
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
    SurfaceCreated {
        window: *mut ndk_sys::ANativeWindow,
    },
    SurfaceDestroyed,
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
    /// Soft keyboard frame in surface pixels.
    Keyboard {
        visible: bool,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
    },
    Request(crate::native::Request),
}
unsafe impl Send for Message {}

thread_local! {
    static MESSAGES_TX: RefCell<Option<mpsc::Sender<Message>>> = RefCell::new(None);
}

fn send_message(message: Message) {
    MESSAGES_TX.with(|tx| {
        let mut tx = tx.borrow_mut();
        tx.as_mut().unwrap().send(message).unwrap();
    })
}

pub static mut ACTIVITY: ndk_sys::jobject = std::ptr::null_mut();
static mut VM: *mut ndk_sys::JavaVM = std::ptr::null_mut();

pub unsafe fn console_debug(msg: *const ::core::ffi::c_char) { unsafe {
    ndk_sys::__android_log_write(
        ndk_sys::android_LogPriority_ANDROID_LOG_DEBUG as _,
        b"SAPP\0".as_ptr() as _,
        msg,
    );
}}

pub unsafe fn console_info(msg: *const ::core::ffi::c_char) { unsafe {
    ndk_sys::__android_log_write(
        ndk_sys::android_LogPriority_ANDROID_LOG_INFO as _,
        b"SAPP\0".as_ptr() as _,
        msg,
    );
}}

pub unsafe fn console_warn(msg: *const ::core::ffi::c_char) { unsafe {
    ndk_sys::__android_log_write(
        ndk_sys::android_LogPriority_ANDROID_LOG_WARN as _,
        b"SAPP\0".as_ptr() as _,
        msg,
    );
}}

pub unsafe fn console_error(msg: *const ::core::ffi::c_char) { unsafe {
    ndk_sys::__android_log_write(
        ndk_sys::android_LogPriority_ANDROID_LOG_ERROR as _,
        b"SAPP\0".as_ptr() as _,
        msg,
    );
}}

// fn log_info(message: &str) {
//     use std::ffi::CString;

//     let msg = CString::new(message).unwrap_or_else(|_| panic!());

//     unsafe { console_info(msg.as_ptr()) };
// }

struct MainThreadState {
    libegl: LibEgl,
    egl_display: egl::EGLDisplay,
    egl_config: egl::EGLConfig,
    egl_context: egl::EGLContext,
    surface: egl::EGLSurface,
    window: *mut ndk_sys::ANativeWindow,
    event_handler: Box<dyn EventHandler>,
    quit: bool,
    fullscreen: bool,
    update_requested: bool,
    keymods: KeyMods,
}

impl MainThreadState {
    unsafe fn destroy_surface(&mut self) { unsafe {
        (self.libegl.eglMakeCurrent)(
            self.egl_display,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        (self.libegl.eglDestroySurface)(self.egl_display, self.surface);
        self.surface = std::ptr::null_mut();
    }}

    unsafe fn update_surface(&mut self, window: *mut ndk_sys::ANativeWindow) { unsafe {
        if !self.window.is_null() {
            ndk_sys::ANativeWindow_release(self.window);
        }
        self.window = window;
        if self.surface.is_null() == false {
            self.destroy_surface();
        }

        self.surface = (self.libegl.eglCreateWindowSurface)(
            self.egl_display,
            self.egl_config,
            window as _,
            std::ptr::null_mut(),
        );

        assert!(!self.surface.is_null());

        let res = (self.libegl.eglMakeCurrent)(
            self.egl_display,
            self.surface,
            self.surface,
            self.egl_context,
        );

        assert!(res != 0);
    }}

    fn process_message(&mut self, msg: Message) {
        match msg {
            Message::SurfaceCreated { window } => unsafe {
                self.update_surface(window);
            },
            Message::SurfaceDestroyed => unsafe {
                self.destroy_surface();
            },
            Message::SurfaceChanged { width, height } => {
                {
                    let mut d = crate::native_display().lock().unwrap();
                    d.screen_width = width as _;
                    d.screen_height = height as _;
                }
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
            Message::ImeAction(code) => {
                // android.view.inputmethod.EditorInfo action codes.
                use crate::TextInputAction as A;
                let action = match code {
                    2 => A::Go,
                    3 => A::Search,
                    4 => A::Send,
                    5 => A::Next,
                    6 => A::Done,
                    7 => A::Previous,
                    _ => crate::text_input::session()
                        .map_or(A::Done, |config| config.resolved_action()),
                };
                self.event_handler.text_input_action_event(action);
            }
            Message::Keyboard { visible, x, y, w, h } => {
                self.event_handler
                    .soft_keyboard_event(visible, x as f32, y as f32, w as f32, h as f32);
            }
            Message::Pause => self.event_handler.window_minimized_event(),
            Message::Resume => {
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

    fn frame(&mut self) {
        self.event_handler.update();

        if self.surface.is_null() == false {
            self.update_requested = false;
            self.event_handler.draw();

            unsafe {
                (self.libegl.eglSwapBuffers)(self.egl_display, self.surface);
            }
        }
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
            StartTextInput(config) => unsafe {
                let env = attach_jni_env();
                ndk_utils::call_void_method!(
                    env,
                    ACTIVITY,
                    "startTextInput",
                    "(IIZIZ)V",
                    config.kind as i32,
                    config.action as i32,
                    config.autocorrect as i32,
                    config.capitalization as i32,
                    config.read_only as i32
                );
            },
            StopTextInput => unsafe {
                let env = attach_jni_env();
                ndk_utils::call_void_method!(env, ACTIVITY, "stopTextInput", "()V");
            },
            SetTextInputState(state) => unsafe {
                let env = attach_jni_env();
                let text = new_jstring(env, &state.text);
                ndk_utils::call_void_method!(
                    env,
                    ACTIVITY,
                    "setTextInputState",
                    "(Ljava/lang/String;IIII)V",
                    text,
                    state.selection_start,
                    state.selection_end,
                    state.composing_start,
                    state.composing_end
                );
                (**env).DeleteLocalRef.unwrap()(env, text);
            },
            SetTextInputRect { .. } => {}
            HapticFeedback(kind) => unsafe {
                let env = attach_jni_env();
                ndk_utils::call_void_method!(env, ACTIVITY, "hapticFeedback", "(I)V", kind as i32);
            },
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
pub unsafe fn attach_jni_env() -> *mut ndk_sys::JNIEnv { unsafe {
    let mut env: *mut ndk_sys::JNIEnv = std::ptr::null_mut();
    let attach_current_thread = (**VM).AttachCurrentThread.unwrap();

    let res = attach_current_thread(VM, &mut env, std::ptr::null_mut());
    assert!(res == 0);

    env
}}

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

            let result = jstring_to_string(env, text);
            (**env).DeleteLocalRef.unwrap()(env, text);
            result
        }
    }

    fn set(&mut self, data: &str) {
        unsafe {
            let env = attach_jni_env();
            let jtext = new_jstring(env, data);

            ndk_utils::call_void_method!(
                env,
                ACTIVITY,
                "setClipboardText",
                "(Ljava/lang/String;)V",
                jtext
            );
            (**env).DeleteLocalRef.unwrap()(env, jtext);
        }
    }
}

pub unsafe fn run<F>(conf: crate::conf::Conf, f: F)
where
    F: 'static + FnOnce() -> Box<dyn EventHandler>,
{ unsafe {
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

    let (tx, rx) = mpsc::channel();

    let tx2 = tx.clone();
    MESSAGES_TX.with(move |messages_tx| *messages_tx.borrow_mut() = Some(tx2));

    thread::spawn(move || {
        let _ = &f;
        let mut libegl = LibEgl::try_load().expect("Cant load LibEGL");

        // skip all the messages until android will be able to actually open a window
        //
        // sometimes before launching an app android will show a permission dialog
        // it is important to create GL context only after a first SurfaceChanged
        let window = 'a: loop {
            match rx.try_recv() {
                Ok(Message::SurfaceCreated { window }) => {
                    break 'a window;
                }
                _ => {}
            }
        };
        let (screen_width, screen_height) = 'a: loop {
            match rx.try_recv() {
                Ok(Message::SurfaceChanged { width, height }) => {
                    break 'a (width as f32, height as f32);
                }
                _ => {}
            }
        };

        let (egl_context, egl_config, egl_display) = crate::native::egl::create_egl_context(
            &mut libegl,
            std::ptr::null_mut(), /* EGL_DEFAULT_DISPLAY */
            conf.platform.framebuffer_alpha,
            conf.sample_count,
        )
        .expect("Cant create EGL context");

        assert!(!egl_display.is_null());
        assert!(!egl_config.is_null());

        crate::native::gl::load_gl_funcs(|proc| {
            let name = std::ffi::CString::new(proc).unwrap();
            (libegl.eglGetProcAddress)(name.as_ptr() as _)
        });

        let surface = (libegl.eglCreateWindowSurface)(
            egl_display,
            egl_config,
            window as _,
            std::ptr::null_mut(),
        );

        if (libegl.eglMakeCurrent)(egl_display, surface, surface, egl_context) == 0 {
            panic!();
        }

        let clipboard = Box::new(AndroidClipboard::new());
        let tx_fn = Box::new(move |req| tx.send(Message::Request(req)).unwrap());
        crate::set_or_replace_display(NativeDisplayData {
            high_dpi: conf.high_dpi,
            blocking_event_loop: conf.platform.blocking_event_loop,
            ..NativeDisplayData::new(screen_width as _, screen_height as _, tx_fn, clipboard)
        });

        let event_handler = f.0();
        let mut s = MainThreadState {
            libegl,
            egl_display,
            egl_config,
            egl_context,
            surface,
            window,
            event_handler,
            quit: false,
            fullscreen: conf.fullscreen,
            update_requested: true,
            keymods: KeyMods {
                shift: false,
                ctrl: false,
                alt: false,
                logo: false,
            },
        };

        let rx_timeout = conf
            .platform
            .sleep_interval_ms
            .map(|sleep| Duration::from_millis(sleep as u64));

        while !s.quit {
            let block_on_wait = conf.platform.blocking_event_loop && !s.update_requested;

            if block_on_wait {
                // We don't need to loop here because the loop above consumes all
                // available messages. Instead we are going to block until receiving here.

                match rx_recv(&rx, rx_timeout) {
                    Ok(msg) => s.process_message(msg),
                    // Timeout so time to do periodic update()
                    Err(mpsc::RecvTimeoutError::Timeout) => s.update_requested = true,
                    Err(mpsc::RecvTimeoutError::Disconnected) => panic!(),
                }
            } else {
                // process all the messages from the main thread
                while let Ok(msg) = rx.try_recv() {
                    s.process_message(msg);
                }
            }

            if !conf.platform.blocking_event_loop || s.update_requested {
                s.frame();
            }

            // order_quit() closes the activity; Destroy then ends this loop.
            let quit_ordered = {
                let mut d = crate::native_display().lock().unwrap();
                std::mem::replace(&mut d.quit_ordered, false)
            };
            if quit_ordered {
                let env = attach_jni_env();
                ndk_utils::call_void_method!(env, ACTIVITY, "quitApp", "()V");
            }

            thread::yield_now();
        }

        (s.libegl.eglMakeCurrent)(
            s.egl_display,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        (s.libegl.eglDestroySurface)(s.egl_display, s.surface);
        (s.libegl.eglDestroyContext)(s.egl_display, s.egl_context);
        (s.libegl.eglTerminate)(s.egl_display);
    });
}}

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

#[unsafe(no_mangle)]
extern "C" fn jni_on_load(vm: *mut std::ffi::c_void) {
    unsafe {
        VM = vm as _;
    }
}

unsafe fn create_native_window(surface: ndk_sys::jobject) -> *mut ndk_sys::ANativeWindow { unsafe {
    let env = attach_jni_env();

    ndk_sys::ANativeWindow_fromSurface(env, surface)
}}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_initializeContext(
    env: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    activity: ndk_sys::jobject,
) { unsafe {
    let env = attach_jni_env();
    let activity_ref = (**env).NewGlobalRef.unwrap()(env, activity);
    ndk_context::initialize_android_context(std::mem::transmute(VM), activity_ref as _);

    let mut env = unsafe { jni::JNIEnv::from_raw(env as *mut jni::sys::JNIEnv).unwrap() };
    let context = unsafe { jni::objects::JObject::from_raw(activity as jni::sys::jobject) };
    let _ = rustls_platform_verifier::android::init_with_env(&mut env, context);
}}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_releaseContext(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) { unsafe {
    ndk_context::release_android_context();
}}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_activityOnCreate(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    activity: ndk_sys::jobject,
) { unsafe {
    let env = attach_jni_env();
    ACTIVITY = (**env).NewGlobalRef.unwrap()(env, activity);
    quad_main();
}}

#[unsafe(no_mangle)]
unsafe extern "C" fn Java_quad_1native_QuadNative_activityOnResume(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    send_message(Message::Resume);
}

#[unsafe(no_mangle)]
unsafe extern "C" fn Java_quad_1native_QuadNative_activityOnPause(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    send_message(Message::Pause);
}

#[unsafe(no_mangle)]
unsafe extern "C" fn Java_quad_1native_QuadNative_activityOnDestroy(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    send_message(Message::Destroy);
}

#[unsafe(no_mangle)]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnSurfaceCreated(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    surface: ndk_sys::jobject,
) {
    let window = unsafe { create_native_window(surface) };
    send_message(Message::SurfaceCreated { window });
}

#[unsafe(no_mangle)]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnSurfaceDestroyed(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
) {
    send_message(Message::SurfaceDestroyed);
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnKeyDown(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    keycode: ndk_sys::jint,
) {
    let keycode = keycodes::translate_keycode(keycode as _);

    send_message(Message::KeyDown { keycode });
}

#[unsafe(no_mangle)]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnKeyUp(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    keycode: ndk_sys::jint,
) {
    let keycode = keycodes::translate_keycode(keycode as _);

    send_message(Message::KeyUp { keycode });
}

#[unsafe(no_mangle)]
extern "C" fn Java_quad_1native_QuadNative_surfaceOnCharacter(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    character: ndk_sys::jint,
) {
    send_message(Message::Character {
        character: character as u32,
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_surfaceOnImePreedit(
    env: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    text: ndk_sys::jstring,
) { unsafe {
    let s = jstring_to_string(env, text).unwrap_or_default();
    send_message(Message::ImePreedit(s));
}}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_surfaceOnImeCommit(
    env: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jobject,
    text: ndk_sys::jstring,
) { unsafe {
    let s = if text.is_null() {
        None
    } else {
        jstring_to_string(env, text)
    };
    send_message(Message::ImeCommit(s));
}}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_surfaceOnImeStateChanged(
    env: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jclass,
    text: ndk_sys::jstring,
    selection_start: ndk_sys::jint,
    selection_end: ndk_sys::jint,
    composing_start: ndk_sys::jint,
    composing_end: ndk_sys::jint,
    element_id: ndk_sys::jlong,
) { unsafe {
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
}}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_surfaceOnKeyboard(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jclass,
    visible: ndk_sys::jboolean,
    x: ndk_sys::jint,
    y: ndk_sys::jint,
    w: ndk_sys::jint,
    h: ndk_sys::jint,
) {
    send_message(Message::Keyboard {
        visible: visible != 0,
        x,
        y,
        w,
        h,
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_quad_1native_QuadNative_surfaceOnImeAction(
    _: *mut ndk_sys::JNIEnv,
    _: ndk_sys::jclass,
    action_code: ndk_sys::jint,
) {
    send_message(Message::ImeAction(action_code as i32));
}

unsafe fn jstring_to_string(env: *mut ndk_sys::JNIEnv, jstr: ndk_sys::jstring) -> Option<String> { unsafe {
    if jstr.is_null() {
        return None;
    }
    let len = (**env).GetStringLength.unwrap()(env, jstr);
    let chars = (**env).GetStringChars.unwrap()(env, jstr, std::ptr::null_mut());
    if chars.is_null() {
        return None;
    }
    let units = std::slice::from_raw_parts(chars, len.max(0) as usize);
    let s = String::from_utf16_lossy(units);
    (**env).ReleaseStringChars.unwrap()(env, jstr, chars);
    Some(s)
}}

/// A new local reference to a Java string with `text`'s contents.
pub(crate) unsafe fn new_jstring(env: *mut ndk_sys::JNIEnv, text: &str) -> ndk_sys::jstring { unsafe {
    let units: Vec<u16> = text.encode_utf16().collect();
    (**env).NewString.unwrap()(env, units.as_ptr(), units.len() as ndk_sys::jsize)
}}

unsafe fn set_full_screen(env: *mut ndk_sys::JNIEnv, fullscreen: bool) { unsafe {
    ndk_utils::call_void_method!(env, ACTIVITY, "setFullScreen", "(Z)V", fullscreen as i32);
}}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct android_asset {
    pub content: *mut ::core::ffi::c_char,
    pub content_length: ::core::ffi::c_int,
}

// According to documentation, AAssetManager_fromJava is as available as an
// AAssetManager_open, which was used before
// For some reason it is missing fron ndk_sys binding
unsafe extern "C" {
    pub fn AAssetManager_fromJava(
        env: *mut ndk_sys::JNIEnv,
        assetManager: ndk_sys::jobject,
    ) -> *mut ndk_sys::AAssetManager;
}

pub(crate) unsafe fn load_asset(filepath: *const ::core::ffi::c_char, out: *mut android_asset) { unsafe {
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
}}
