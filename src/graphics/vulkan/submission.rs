//! Host queue ownership for the normal window presentation path.
//!
//! Requests contain handles, not borrowed Vulkan structs. Each request owns
//! one frame slot until its host calls have returned; the slot's GPU fence
//! independently protects the resources used by the submitted commands.
//!
//! The worker touches the graphics queue and the swapchain, so the context
//! drains it before any other host access to those objects: the queue must be
//! externally synchronized, and `vkAcquireNextImageKHR` must not race the
//! worker's `vkQueuePresentKHR` on the same swapchain (see
//! `VulkanContext::begin_frame`).
//!
//! Before presenting, the worker can acquire the next image without waiting
//! and publish it to the context. Recording into that image does not access
//! the queue or swapchain, so it can overlap the current host present call.

use ash::{extensions::khr::Swapchain, vk, Device};
use std::{
    sync::mpsc::{self, Receiver, SyncSender},
    thread::{self, JoinHandle},
};

use super::MAX_FRAMES_IN_FLIGHT;

type PresentResult = Result<bool, vk::Result>;
type AcquireResult = Result<Option<PreparedImage>, vk::Result>;

#[derive(Debug, PartialEq)]
pub(super) struct PreparedImage {
    pub swapchain: vk::SwapchainKHR,
    pub image_index: u32,
    pub semaphore: vk::Semaphore,
    pub suboptimal: bool,
}

enum SubmissionEvent {
    Prepared(AcquireResult),
    Completed(usize, PresentResult),
}

pub(super) struct FrameSubmission {
    pub slot: usize,
    pub command_buffer: vk::CommandBuffer,
    pub wait: Option<vk::Semaphore>,
    pub signal: vk::Semaphore,
    pub fence: vk::Fence,
    pub swapchain: vk::SwapchainKHR,
    pub image_index: u32,
    /// A semaphore whose previous GPU wait has finished. The next frame uses
    /// it only if the worker successfully acquires another image.
    pub next_acquire: Option<vk::Semaphore>,
}

impl FrameSubmission {
    /// Queue and acquisition calls share one worker. Other callers drain it
    /// before accessing the queue or the swapchain themselves.
    pub fn execute(
        &self,
        device: &Device,
        swapchain: &Swapchain,
        queue: vk::Queue,
        prepared: impl FnOnce(AcquireResult),
    ) -> PresentResult {
        #[cfg(feature = "tracy")]
        let _span = crate::tracy::span_with_callstack(crate::tracy::span_location!(
            "Vulkan::submit_and_present"
        ));
        let command_buffers = [self.command_buffer];
        let wait_semaphores = [self.wait.unwrap_or_default()];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let wait_count = usize::from(self.wait.is_some());
        let signal_semaphores = [self.signal];
        let submit = vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .wait_semaphores(&wait_semaphores[..wait_count])
            .wait_dst_stage_mask(&wait_stages[..wait_count])
            .signal_semaphores(&signal_semaphores);
        let swapchains = [self.swapchain];
        let image_indices = [self.image_index];
        let present = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        unsafe {
            {
                #[cfg(feature = "tracy")]
                let _span = crate::tracy::span_with_callstack(crate::tracy::span_location!(
                    "Vulkan::queue_submit"
                ));
                if let Err(error) = device.queue_submit(queue, &[submit.build()], self.fence) {
                    if self.next_acquire.is_some() {
                        prepared(Err(error));
                    }
                    return Err(error);
                }
            }
            if let Some(semaphore) = self.next_acquire {
                let result = {
                    #[cfg(feature = "tracy")]
                    let _span = crate::tracy::span(crate::tracy::span_location!(
                        "Vulkan::preacquire_next_image (wall time)"
                    ));
                    match swapchain.acquire_next_image(
                        self.swapchain,
                        0,
                        semaphore,
                        vk::Fence::null(),
                    ) {
                        Ok((image_index, suboptimal)) => Ok(Some(PreparedImage {
                            swapchain: self.swapchain,
                            image_index,
                            semaphore,
                            suboptimal,
                        })),
                        // Never stall the queue worker waiting for an image
                        // which its upcoming present may need to release.
                        Err(vk::Result::NOT_READY | vk::Result::TIMEOUT) => Ok(None),
                        Err(error) => Err(error),
                    }
                };
                prepared(result);
            }
            #[cfg(feature = "tracy")]
            let _span = crate::tracy::span_with_callstack(crate::tracy::span_location!(
                "Vulkan::queue_present"
            ));
            match swapchain.queue_present(queue, &present) {
                Ok(suboptimal) => Ok(suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => Ok(true),
                Err(error) => Err(error),
            }
        }
    }
}

pub(super) struct SubmissionWorker {
    requests: Option<SyncSender<FrameSubmission>>,
    completed: Receiver<SubmissionEvent>,
    pending: [bool; MAX_FRAMES_IN_FLIGHT],
    preacquire: bool,
    acquiring: bool,
    prepared: Option<PreparedImage>,
    thread: Option<JoinHandle<()>>,
}

impl SubmissionWorker {
    pub fn new(device: &Device, swapchain: &Swapchain, queue: vk::Queue) -> Option<Self> {
        if !super::env_enabled("PRPR_VULKAN_SUBMIT_THREAD") {
            return None;
        }
        let device = device.clone();
        let swapchain = swapchain.clone();
        let mut worker =
            Self::spawn(move |frame, prepared| frame.execute(&device, &swapchain, queue, prepared));
        worker.preacquire = super::env_enabled("PRPR_VULKAN_PREACQUIRE");
        Some(worker)
    }

    fn spawn(
        mut execute: impl FnMut(FrameSubmission, &mut dyn FnMut(AcquireResult)) -> PresentResult
            + Send
            + 'static,
    ) -> Self {
        let (requests, receiver) = mpsc::sync_channel::<FrameSubmission>(MAX_FRAMES_IN_FLIGHT);
        let (completed, completion_receiver) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("vulkan-submit".into())
            .spawn(move || {
                #[cfg(feature = "tracy")]
                crate::tracy::register_thread("vulkan-submit");
                loop {
                    let received = {
                        #[cfg(feature = "tracy")]
                        let _span = crate::tracy::span_with_callstack(
                            crate::tracy::span_location!("Vulkan::submit_wait_request (wall time)"),
                        );
                        receiver.recv()
                    };
                    let Ok(frame) = received else { break };
                    let slot = frame.slot;
                    let result = execute(frame, &mut |result| {
                        let _ = completed.send(SubmissionEvent::Prepared(result));
                    });
                    // The context may be unwinding; shutdown still consumes all
                    // queued requests before releasing the device references.
                    let _ = completed.send(SubmissionEvent::Completed(slot, result));
                }
            })
            .expect("Failed to create the Vulkan submission thread");
        Self {
            requests: Some(requests),
            completed: completion_receiver,
            pending: [false; MAX_FRAMES_IN_FLIGHT],
            preacquire: true,
            acquiring: false,
            prepared: None,
            thread: Some(thread),
        }
    }

    pub fn preacquire_enabled(&self) -> bool {
        self.preacquire
    }

    pub fn submit(&mut self, frame: FrameSubmission) {
        #[cfg(feature = "tracy")]
        let _span = crate::tracy::span_with_callstack(crate::tracy::span_location!(
            "Vulkan::submit_enqueue"
        ));
        let slot = frame.slot;
        assert!(
            !self.pending[slot],
            "frame slot still owned by the submission thread"
        );
        if frame.next_acquire.is_some() {
            assert!(
                !self.acquiring && self.prepared.is_none(),
                "next image was not consumed"
            );
            self.acquiring = true;
        }
        self.requests
            .as_ref()
            .unwrap()
            .send(frame)
            .expect("Vulkan submission thread stopped");
        self.pending[slot] = true;
    }

    fn complete(&mut self, event: SubmissionEvent) -> PresentResult {
        match event {
            SubmissionEvent::Completed(slot, result) => {
                self.pending[slot] = false;
                result
            }
            SubmissionEvent::Prepared(result) => {
                self.acquiring = false;
                match result {
                    Ok(image) => {
                        self.prepared = image;
                        Ok(false)
                    }
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => Ok(true),
                    Err(error) => Err(error),
                }
            }
        }
    }

    fn wait_while(&mut self, pending: impl Fn(&Self) -> bool) -> PresentResult {
        let mut recreate = false;
        while pending(self) {
            let event = self
                .completed
                .recv()
                .expect("Vulkan submission thread stopped");
            recreate |= self.complete(event)?;
        }
        Ok(recreate)
    }

    /// Wait only for submission and nonblocking acquisition, leaving the
    /// current present call free to run while the context records next frame.
    pub fn wait_prepared(&mut self) -> Result<(Option<PreparedImage>, bool), vk::Result> {
        #[cfg(feature = "tracy")]
        let _span = crate::tracy::span(crate::tracy::span_location!(
            "Vulkan::submit_wait_prepared (wall time)"
        ));
        let recreate = self.wait_while(|worker| worker.acquiring)?;
        Ok((self.prepared.take(), recreate))
    }

    /// Called after draining, before replacing swapchain/semaphore handles.
    pub fn discard_prepared(&mut self) {
        assert!(!self.acquiring);
        self.prepared = None;
    }

    pub fn wait_slot(&mut self, slot: usize) -> PresentResult {
        #[cfg(feature = "tracy")]
        let _span = crate::tracy::span(crate::tracy::span_location!(
            "Vulkan::submit_wait_slot (wall time)"
        ));
        self.wait_while(|worker| worker.pending[slot])
    }

    /// Transfer exclusive queue access back to the calling thread. This only
    /// waits for host calls; the caller must also wait on the GPU before freeing
    /// resources or resetting commands that may still be in flight.
    pub fn drain(&mut self) -> PresentResult {
        #[cfg(feature = "tracy")]
        let _span = crate::tracy::span_with_callstack(crate::tracy::span_location!(
            "Vulkan::submit_drain (wall time)"
        ));
        let mut recreate = false;
        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            recreate |= self.wait_slot(slot)?;
        }
        Ok(recreate)
    }
}

impl Drop for SubmissionWorker {
    fn drop(&mut self) {
        // Closing the request channel wakes an idle worker and lets an active
        // worker finish before the Vulkan context destroys its device.
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
