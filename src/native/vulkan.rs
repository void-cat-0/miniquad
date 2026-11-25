use raw_window_handle_vulkan::*;
use std::{cell::RefCell, num::NonZeroU32};

use crate::{conf::RenderingBackend, vulkan::VulkanContext};

thread_local! {
    static PREPARED_CONTEXT: RefCell<Option<VulkanContext>> = const { RefCell::new(None) };
}

pub(crate) fn take_prepared_context() -> Option<VulkanContext> {
    PREPARED_CONTEXT.with(|slot| slot.borrow_mut().take())
}

/// Keeps an unused startup context from outliving the native window.
pub(crate) struct PreparedVulkan;

impl Drop for PreparedVulkan {
    fn drop(&mut self) {
        if let Some(mut context) = take_prepared_context() {
            context.cleanup();
        }
    }
}

// Store integer addresses so the global display data stays Send without
// asserting thread safety for raw window/display pointers.
#[derive(Clone, Copy)]
pub(crate) enum VulkanWindow {
    #[cfg(target_os = "android")]
    Android {
        window: usize,
    },
    Win32 {
        hwnd: usize,
        hinstance: usize,
    },
    Xlib {
        display: usize,
        window: u64,
        screen: i32,
    },
    Wayland {
        display: usize,
        surface: usize,
    },
}

impl VulkanWindow {
    pub(crate) fn prepare(
        self,
        requested: RenderingBackend,
        size: (u32, u32),
    ) -> Option<PreparedVulkan> {
        match requested {
            RenderingBackend::OpenGl => None,
            RenderingBackend::Vulkan => Some(PreparedVulkan),
            RenderingBackend::VulkanPreferred => {
                let (display, window) = self.handles();
                match VulkanContext::try_with_texture_size_limit(
                    display,
                    window,
                    size,
                    NonZeroU32::MAX,
                ) {
                    Ok(context) => {
                        #[cfg(target_os = "android")]
                        {
                            let properties = unsafe {
                                context
                                    .instance
                                    .get_physical_device_properties(context.physical_device)
                            };
                            let device_name = unsafe {
                                std::ffi::CStr::from_ptr(properties.device_name.as_ptr())
                            }
                            .to_string_lossy();
                            crate::native::android::log_diagnostic(
                                "INFO",
                                &format!(
                                    "Vulkan initialized: {device_name}, API 0x{:x}.",
                                    properties.api_version
                                ),
                            );
                        }
                        PREPARED_CONTEXT.with(|slot| *slot.borrow_mut() = Some(context));
                        Some(PreparedVulkan)
                    }
                    Err(error) => {
                        #[cfg(target_os = "android")]
                        super::android::warn_vulkan_fallback(&error);
                        #[cfg(not(target_os = "android"))]
                        eprintln!("Vulkan startup failed ({error}); using OpenGL");
                        None
                    }
                }
            }
        }
    }

    pub fn handles(self) -> (RawDisplayHandle, RawWindowHandle) {
        match self {
            #[cfg(target_os = "android")]
            Self::Android { window } => {
                let mut handle = AndroidNdkWindowHandle::empty();
                handle.a_native_window = window as *mut _;
                (
                    RawDisplayHandle::Android(AndroidDisplayHandle::empty()),
                    RawWindowHandle::AndroidNdk(handle),
                )
            }
            Self::Win32 { hwnd, hinstance } => {
                let mut window = Win32WindowHandle::empty();
                window.hwnd = hwnd as *mut _;
                window.hinstance = hinstance as *mut _;
                (
                    RawDisplayHandle::Windows(WindowsDisplayHandle::empty()),
                    RawWindowHandle::Win32(window),
                )
            }
            Self::Xlib {
                display,
                window,
                screen,
            } => {
                let mut d = XlibDisplayHandle::empty();
                d.display = display as *mut _;
                d.screen = screen;
                let mut w = XlibWindowHandle::empty();
                w.window = window as _;
                (RawDisplayHandle::Xlib(d), RawWindowHandle::Xlib(w))
            }
            Self::Wayland { display, surface } => {
                let mut d = WaylandDisplayHandle::empty();
                d.display = display as *mut _;
                let mut w = WaylandWindowHandle::empty();
                w.surface = surface as *mut _;
                (RawDisplayHandle::Wayland(d), RawWindowHandle::Wayland(w))
            }
        }
    }
}
