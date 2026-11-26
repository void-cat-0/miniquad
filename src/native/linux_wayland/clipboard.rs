use super::*;
use crate::{wl_request, wl_request_constructor};

use core::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;

// new clipboard content is available
// this could be fired at any time, so we just store the `data_offer` for later use
pub(super) unsafe extern "C" fn data_device_handle_selection(
    data: *mut c_void,
    data_device: *mut wl_data_device,
    data_offer: *mut wl_data_offer,
) {
    let display: &mut WaylandPayload = &mut *(data as *mut _);
    assert_eq!(data_device, display.data_device);
    display.clipboard.data_offer = (!data_offer.is_null()).then_some(data_offer);
}

/// Clipboard state owned by the display payload: the content we offer, the
/// `data_source` naming us as the selection owner, and the `data_offer` of
/// whichever client owns the selection right now.
#[derive(Debug, Default)]
pub(super) struct ClipboardState {
    content: String,
    data_source: Option<*mut wl_data_source>,
    data_offer: Option<*mut wl_data_offer>,
}

impl ClipboardState {
    unsafe fn respond_to_clipboard_request(&mut self, mime_type: &str, fd: c_int) {
        #![allow(clippy::single_match)]
        match mime_type {
            "UTF8_STRING" => {
                libc::write(fd, self.content.as_ptr() as _, self.content.len());
            }
            _ => {}
        }
        libc::close(fd);
    }
}

impl WaylandPayload {
    pub(super) unsafe fn clipboard_get(&mut self, mime_type: &str) -> Option<Vec<u8>> {
        let data_offer = self.clipboard.data_offer?;
        let mime_type = std::ffi::CString::new(mime_type).unwrap();
        self.client
            .data_offer_receive(self.display, data_offer, mime_type.as_ptr())
    }

    pub(super) unsafe fn clipboard_set(&mut self, data: &str) {
        self.clipboard.content.clear();
        self.clipboard.content.push_str(data);
        if self.data_device.is_null() {
            return;
        }
        // Wayland requires that only the window with focus can set the clipboard
        if let Some(serial) = self.keyboard_context.enter_serial {
            let data_source = self.clipboard_new_data_source();
            // only support copying utf8 strings
            let mime_type = std::ffi::CString::new("UTF8_STRING").unwrap();
            wl_request!(
                self.client,
                data_source,
                WL_DATA_SOURCE_OFFER,
                mime_type.as_ptr()
            );
            wl_request!(
                self.client,
                self.data_device,
                WL_DATA_DEVICE_SET_SELECTION,
                data_source,
                serial
            );
        }
    }

    /// Since the data_source is constructed by us, we need to dispose of it properly.
    pub(super) unsafe fn clipboard_destroy_data_source(&mut self) {
        if let Some(data_source) = self.clipboard.data_source.take() {
            wl_request!(self.client, data_source, WL_DATA_SOURCE_DESTROY);
            (self.client.wl_proxy_destroy)(data_source as _);
        }
    }

    unsafe fn clipboard_new_data_source(&mut self) -> *mut wl_data_source {
        self.clipboard_destroy_data_source();
        let data_source: *mut wl_data_source = wl_request_constructor!(
            self.client,
            self.data_device_manager,
            WL_DATA_DEVICE_MANAGER_CREATE_DATA_SOURCE,
            self.client.wl_data_source_interface,
        );
        assert!(!data_source.is_null());
        (self.client.wl_proxy_add_listener)(
            data_source as _,
            (&raw const DATA_SOURCE_LISTENER).cast_mut().cast(),
            self as *mut _ as _,
        );
        self.clipboard.data_source = Some(data_source);
        data_source
    }
}

// libwayland keeps this pointer for as long as the proxy lives, so it cannot
// be a stack local; the handlers never change, so neither does the table.
static DATA_SOURCE_LISTENER: wl_data_source_listener = wl_data_source_listener {
    send: data_source_handle_send,
    cancelled: data_source_handle_cancelled,
    ..wl_data_source_listener::dummy()
};

// some app (could be ourself) is requesting the owned clipboard
unsafe extern "C" fn data_source_handle_send(
    data: *mut c_void,
    data_source: *mut wl_data_source,
    mime_type: *const c_char,
    fd: c_int,
) {
    let display: &mut WaylandPayload = &mut *(data as *mut _);
    let mime_type = core::ffi::CStr::from_ptr(mime_type).to_str().unwrap();
    assert!(display.clipboard.data_source == Some(data_source));
    display
        .clipboard
        .respond_to_clipboard_request(mime_type, fd);
}

// the owned clipboard has been replaced by some other app
unsafe extern "C" fn data_source_handle_cancelled(
    data: *mut c_void,
    data_source: *mut wl_data_source,
) {
    let display: &mut WaylandPayload = &mut *(data as *mut _);
    assert!(display.clipboard.data_source == Some(data_source));
    display.clipboard_destroy_data_source();
}

/// Handle to the clipboard state that lives inside a [`WaylandPayload`].
///
/// The payload only exists while the Wayland event loop runs, so the pointer is
/// cleared by [`DetachOnDrop`] before the payload goes away; calls after the
/// window closes find nothing to talk to.
pub struct WaylandClipboard {
    payload: Arc<AtomicPtr<WaylandPayload>>,
}

impl WaylandClipboard {
    pub(super) fn new(payload: Arc<AtomicPtr<WaylandPayload>>) -> Self {
        Self { payload }
    }
}

impl crate::native::Clipboard for WaylandClipboard {
    fn get(&mut self) -> Option<String> {
        let payload = self.payload.load(Ordering::Acquire);
        if payload.is_null() {
            return None;
        }
        let bytes = unsafe { (*payload).clipboard_get("UTF8_STRING")? };
        Some(std::str::from_utf8(&bytes).ok()?.to_string())
    }

    fn set(&mut self, data: &str) {
        let payload = self.payload.load(Ordering::Acquire);
        if payload.is_null() {
            return;
        }
        unsafe { (*payload).clipboard_set(data) }
    }
}

/// Clears the clipboard's payload pointer when the event loop returns, before
/// the payload itself is dropped.
pub(super) struct DetachOnDrop(pub Arc<AtomicPtr<WaylandPayload>>);

impl Drop for DetachOnDrop {
    fn drop(&mut self) {
        self.0.store(std::ptr::null_mut(), Ordering::Release);
    }
}
