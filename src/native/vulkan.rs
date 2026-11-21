use raw_window_handle_vulkan::*;

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
