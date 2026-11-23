use super::resource::{image_barrier, transfer_destination_layout, vk_index_type};
use super::{VulkanContext, DEPTH_FORMAT, MAX_FRAMES_IN_FLIGHT};
use crate::graphics::*;
use ash::vk;
use ash::vk::Handle;
use std::{convert::TryFrom, mem::MaybeUninit, ptr};

impl VulkanContext {
    pub(crate) fn grab_screen(&mut self, dst: usize) {
        assert!(!self.offscreen, "Window capture requires a swapchain frame");
        if !self.frame_active && !self.begin_frame() {
            return;
        }
        let image_index = self.acquired_image.unwrap();
        let src_layout = self.swapchain_image_layouts[image_index];
        if src_layout == vk::ImageLayout::UNDEFINED {
            // Before the first window pass there is nothing defined to capture.
            return;
        }
        let module = &self.textures[dst];
        let dst_blit = unsafe {
            self.instance
                .get_physical_device_format_properties(self.physical_device, module.format)
        }
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::BLIT_DST);
        assert!(
            self.swapchain_can_capture
                && dst_blit
                && module.usage.contains(vk::ImageUsageFlags::TRANSFER_DST),
            "This Vulkan swapchain or destination texture does not support screen capture"
        );
        let src_image = self.swapchain_images[image_index];
        let (dst_image, dst_layout) = (module.image, module.layout);
        let aspect = vk::ImageAspectFlags::COLOR;
        let dst_final = transfer_destination_layout(dst_layout, aspect);
        let (w, h) = (module.width as i32, module.height as i32);
        let subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        // Texture rows follow OpenGL's bottom-left convention, whereas the
        // swapchain uses top-left. Flip during the format-converting blit.
        let blit = vk::ImageBlit {
            src_subresource: subresource,
            src_offsets: [
                vk::Offset3D {
                    x: 0,
                    y: self.swapchain_extent.height as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: self.swapchain_extent.width as i32,
                    y: 0,
                    z: 1,
                },
            ],
            dst_subresource: subresource,
            dst_offsets: [vk::Offset3D::default(), vk::Offset3D { x: w, y: h, z: 1 }],
        };
        self.record_submit(|ctx, cmd| {
            ctx.barrier(
                cmd,
                src_image,
                aspect,
                src_layout,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            ctx.barrier(
                cmd,
                dst_image,
                aspect,
                dst_layout,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            unsafe {
                ctx.device.cmd_blit_image(
                    cmd,
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::NEAREST,
                );
            }
            ctx.barrier(
                cmd,
                src_image,
                aspect,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_layout,
            );
            ctx.barrier(
                cmd,
                dst_image,
                aspect,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                dst_final,
            );
        });
        self.textures[dst].layout = dst_final;
    }

    pub(crate) fn record_submit<F>(&mut self, f: F)
    where
        F: FnOnce(&VulkanContext, vk::CommandBuffer),
    {
        self.try_record_submit(f)
            .expect("Failed to submit a Vulkan transfer");
    }

    pub(super) fn try_record_submit<F>(&mut self, f: F) -> Result<(), vk::Result>
    where
        F: FnOnce(&VulkanContext, vk::CommandBuffer),
    {
        self.drain_submissions();
        self.flush_recording();
        unsafe {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);
            let command_buffer = self.device.allocate_command_buffers(&allocate_info)?[0];
            let command_buffers = [command_buffer];
            let result = (|| {
                let begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.device
                    .begin_command_buffer(command_buffer, &begin_info)?;
                f(self, command_buffer);
                self.device.end_command_buffer(command_buffer)?;
                let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
                self.device.queue_submit(
                    self.graphics_queue,
                    &[submit_info.build()],
                    vk::Fence::null(),
                )?;
                self.device.queue_wait_idle(self.graphics_queue)
            })();
            self.device
                .free_command_buffers(self.command_pool, &command_buffers);
            result?;
        }
        self.release_frame_resources();
        Ok(())
    }

    /// Submit pending draws before a synchronous transfer reads their output.
    pub(crate) fn flush_recording(&mut self) -> bool {
        if !self.recording || !self.frame_active || !self.cmd_dirty {
            return false;
        }
        self.drain_submissions();
        self.suspend_pass();
        self.flush_draw_commands();
        let command_buffer = self.frame_command_buffer();
        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end the command buffer");
            let command_buffers = [command_buffer];
            let wait_semaphores = [self.frame_wait];
            let wait_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let wait_count = usize::from(self.frame_wait_pending);
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores[..wait_count])
                .wait_dst_stage_mask(&wait_mask[..wait_count])
                .command_buffers(&command_buffers)
                .build();
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], vk::Fence::null())
                .expect("Failed to submit the recorded frame");
        }
        self.frame_wait_pending = false;
        self.cmd_dirty = false;
        self.frame_submitted = true;
        true
    }

    /// Reclaim buffers after a synchronous graphics operation. Restart the
    /// current command buffer if `flush_recording` submitted that segment.
    pub(crate) fn release_frame_resources(&mut self) {
        // Synchronous callers drain the worker, flush recorded draws, and
        // wait for this queue or their own later submission on it.
        self.release_all_retired_buffers();
        if !self.frame_submitted || !self.frame_active {
            return;
        }
        self.reset_frame_storage();
        self.restart_command_buffer(self.frame_command_buffer());
        self.frame_submitted = false;
    }

    // The caller waits for this slot's GPU work before reusing its storage.
    fn reset_frame_storage(&mut self) {
        self.free_allocated_sets();
        self.release_retired_buffers(self.current_frame);
        let uniform = &mut self.uniform_arenas[self.current_frame];
        let stream = &mut self.stream_arenas[self.current_frame];
        for (buffer, memory) in uniform.retired.drain(..).chain(stream.retired.drain(..)) {
            unsafe {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
        }
        uniform.offset = 0;
        stream.offset = 0;
        self.stream_epoch = self.stream_epoch.wrapping_add(1);
        if let Some(arena) = &mut self.texture_upload_arenas[self.current_frame] {
            arena.offset = 0;
        }
    }

    /// Primary command buffer for the current frame slot, not the swapchain
    /// image. Only append commands while this frame is actively recording.
    pub fn frame_command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffers[self.current_frame]
    }

    fn restart_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device
                .reset_command_buffer(
                    command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .expect("Failed to reset the command buffer");
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin the command buffer");
        }
    }

    /// The desktop host can prepare before calling draw, giving it a chance
    /// to service events while a window has no drawable image.
    #[cfg(not(target_os = "android"))]
    pub(crate) fn prepare_desktop_frame(&mut self) -> bool {
        self.offscreen || self.frame_active || self.begin_frame()
    }

    #[cfg(target_os = "android")]
    fn begin_frame(&mut self) -> bool {
        // The Android host pre-acquires before invoking application code.
        false
    }

    #[cfg(not(target_os = "android"))]
    fn begin_frame(&mut self) -> bool {
        if self.offscreen || self.window_size.0 == 0 || self.window_size.1 == 0 {
            return false;
        }
        #[cfg(feature = "tracy")]
        let _span =
            crate::tracy::span_with_callstack(crate::tracy::span_location!("Vulkan::begin_frame"));
        if let Some(worker) = &mut self.submission {
            let state = worker
                .poll()
                .expect("Failed to submit or present the frame");
            self.update_presentation(state);
        }
        let recreate = self.surface_lost
            || self.presentation_needs_recreate
            || self.swapchain == vk::SwapchainKHR::null();
        let slot = self.current_frame;
        if self
            .submission
            .as_ref()
            .is_some_and(|worker| !worker.frame_ready(slot, recreate))
        {
            return false;
        }
        let fences = if recreate {
            &self.in_flight[..]
        } else {
            &self.in_flight[slot..=slot]
        };
        let wait_start = std::time::Instant::now();
        let ready = fences.iter().all(|&fence| unsafe {
            self.device
                .get_fence_status(fence)
                .expect("Failed to poll the frame fence")
        });
        self.frame_wait_ns = self
            .frame_wait_ns
            .saturating_add(wait_start.elapsed().as_nanos() as u64);
        if !ready {
            return false;
        }
        if recreate {
            let result = if self.surface_lost {
                self.recreate_desktop_surface()
            } else {
                self.recreate_swapchain()
            };
            if let Err(error) = result {
                self.desktop_frame_not_ready(error);
                return false;
            }
            if self.swapchain == vk::SwapchainKHR::null() {
                return false;
            }
        }

        let prepared = self
            .submission
            .as_mut()
            .and_then(|worker| worker.take_prepared());
        let (index, slot, semaphore, suboptimal) = if let Some(image) = prepared {
            (
                image.image_index,
                image.slot,
                image.semaphore,
                image.suboptimal,
            )
        } else {
            // No worker request owns this swapchain. Do not wait indefinitely:
            // the native window thread may need to process resize/restore first.
            let slot = self.acquire_slot;
            let semaphore = self.image_available[slot];
            #[cfg(feature = "tracy")]
            let _span = crate::tracy::span_with_callstack(crate::tracy::span_location!(
                "Vulkan::acquire_next_image (wall time)"
            ));
            match unsafe {
                self.swapchain_loader.acquire_next_image(
                    self.swapchain,
                    1_000_000,
                    semaphore,
                    vk::Fence::null(),
                )
            } {
                Ok((index, suboptimal)) => (index, slot, semaphore, suboptimal),
                Err(error) => {
                    self.desktop_frame_not_ready(error);
                    return false;
                }
            }
        };
        self.acquire_slot = (slot + 1) % self.image_available.len();
        self.reset_frame_storage();
        self.start_acquired_frame(index, semaphore, suboptimal);
        true
    }

    #[cfg(not(target_os = "android"))]
    fn desktop_frame_not_ready(&mut self, error: vk::Result) {
        match error {
            vk::Result::NOT_READY | vk::Result::TIMEOUT => (),
            vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => {
                self.presentation_needs_recreate = true;
            }
            vk::Result::ERROR_SURFACE_LOST_KHR => self.surface_lost = true,
            // Another surface owner may not have released the native window
            // yet during rebind. Keep retrying the surface, not the old chain.
            vk::Result::ERROR_NATIVE_WINDOW_IN_USE_KHR if self.surface_lost => (),
            other => panic!("Failed to prepare the Vulkan window frame: {:?}", other),
        }
    }

    fn start_acquired_frame(
        &mut self,
        image_index: u32,
        acquire_semaphore: vk::Semaphore,
        suboptimal: bool,
    ) {
        // Reset only once an image was acquired and a submission will use
        // this fence; the slot stays idle on a failed or skipped acquisition.
        unsafe {
            self.device
                .reset_fences(&[self.in_flight[self.current_frame]])
                .expect("Failed to reset the frame fence");
        }
        let image_index = image_index as usize;
        self.swapchain_suboptimal = suboptimal;
        self.frame_wait = acquire_semaphore;
        self.acquired_image = Some(image_index);
        // The presentation engine owned the image until now, so whatever it
        // holds is undefined.
        self.swapchain_image_layouts[image_index] = vk::ImageLayout::UNDEFINED;
        self.restart_command_buffer(self.frame_command_buffer());

        self.frame_wait_pending = true;
        self.recording = true;
        self.frame_active = true;
        self.cmd_dirty = false;
        self.frame_submitted = false;
        self.cur_pipeline = None;
        self.cur_index_buffer = None;
        self.cur_vertex_buffers.clear();
        self.cur_target = None;
        self.pass_suspended = false;
        self.bound_textures.clear();
    }

    /// Prepare the window frame before entering application callbacks. A
    /// transient lack of a drawable image leaves the fence and frame inactive.
    #[cfg(target_os = "android")]
    pub(crate) fn prepare_android_frame(&mut self) -> Result<bool, vk::Result> {
        if self.offscreen {
            return Ok(true);
        }
        if self.surface_lost {
            return Err(vk::Result::ERROR_SURFACE_LOST_KHR);
        }
        if self.surface == vk::SurfaceKHR::null()
            || self.window_size.0 == 0
            || self.window_size.1 == 0
        {
            return Ok(false);
        }
        if self.frame_active {
            return Ok(true);
        }
        // The primary, acquire semaphore, and transient storage belong to the
        // frame slot. The other slot may still execute while we record this one.
        let slot = self.current_frame;
        let recreate =
            self.presentation_needs_recreate || self.swapchain == vk::SwapchainKHR::null();
        let fences = if recreate {
            // Both slots must retire before replacing the swapchain. Poll first
            // so lifecycle messages can be handled while GPU work is pending.
            &self.in_flight[..]
        } else {
            &self.in_flight[slot..=slot]
        };
        for &fence in fences {
            if !unsafe { self.device.get_fence_status(fence)? } {
                return Ok(false);
            }
        }
        if recreate {
            match self.recreate_swapchain() {
                Ok(()) => (),
                Err(error) => return self.android_frame_not_ready(error),
            }
            if self.swapchain == vk::SwapchainKHR::null() {
                return Ok(false);
            }
        }

        let acquire_semaphore = self.image_available[slot];
        let image = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                0, // Do not block Android's surface lifecycle messages.
                acquire_semaphore,
                vk::Fence::null(),
            )
        };
        match image {
            Ok((image_index, suboptimal)) => {
                self.reset_frame_storage();
                self.start_acquired_frame(image_index, acquire_semaphore, suboptimal);
                Ok(true)
            }
            Err(error) => self.android_frame_not_ready(error),
        }
    }

    #[cfg(target_os = "android")]
    pub(crate) fn android_window_needs_attention(&self) -> bool {
        !self.offscreen && (self.surface_lost || self.presentation_needs_recreate)
    }

    #[cfg(target_os = "android")]
    fn android_frame_not_ready(&mut self, error: vk::Result) -> Result<bool, vk::Result> {
        match error {
            vk::Result::ERROR_OUT_OF_DATE_KHR => {
                self.presentation_needs_recreate = true;
                Ok(false)
            }
            vk::Result::NOT_READY | vk::Result::TIMEOUT => Ok(false),
            vk::Result::ERROR_SURFACE_LOST_KHR => {
                self.surface_lost = true;
                Err(error)
            }
            _ => Err(error),
        }
    }

    #[cfg(target_os = "android")]
    pub(super) fn finish_android_window_frame(&mut self) {
        if self.offscreen {
            return;
        }
        if self.frame_active {
            // Flush uploads and offscreen draws to keep their resource layouts
            // valid, without trying to present to a disappearing window.
            self.flush_recording();
            let wait = [self.frame_wait];
            let stages = [vk::PipelineStageFlags::ALL_COMMANDS];
            let count = usize::from(self.frame_wait_pending);
            let submit = vk::SubmitInfo::builder()
                .wait_semaphores(&wait[..count])
                .wait_dst_stage_mask(&stages[..count]);
            // Consume an unused acquire semaphore and signal the slot fence
            // that was reset at acquisition, even for an empty frame.
            unsafe {
                self.device
                    .queue_submit(
                        self.graphics_queue,
                        &[submit.build()],
                        self.in_flight[self.current_frame],
                    )
                    .expect("Failed to retire the Android window frame");
            }
            self.last_submitted_slot = Some(self.current_frame);
        }
        self.frame_wait = vk::Semaphore::null();
        self.frame_wait_pending = false;
        self.acquired_image = None;
        self.recording = false;
        self.pass_open = false;
        self.pass_suspended = false;
        self.frame_active = false;
        self.frame_submitted = false;
        self.cmd_dirty = false;
        self.cur_target = None;
        self.cur_pipeline = None;
        self.cur_index_buffer = None;
        self.cur_vertex_buffers.clear();
        self.bound_textures.clear();
    }

    /// Switch this context to offscreen video export.
    ///
    /// In offscreen mode a frame is recorded into a command-buffer ring indexed
    /// by `current_frame`; no swap-chain image is acquired and nothing is
    /// presented.  The ring is what lets the exporter keep several frames in
    /// flight instead of blocking on a readback fence every frame.
    ///
    /// Must be called before the first `begin_offscreen_frame`. A partially
    /// recorded window frame is submitted without presenting it.
    pub fn enable_offscreen_video(&mut self) {
        if self.offscreen {
            return;
        }
        self.drain_submissions();
        // Finish any window commands (including queued texture uploads) before
        // replacing the command-buffer ring. Do not present the partial frame.
        self.flush_recording();
        if self.frame_wait_pending {
            self.wait_acquired_semaphore(self.frame_wait);
        }
        let prepared = self
            .submission
            .as_mut()
            .and_then(super::submission::SubmissionWorker::take_prepared);
        if let Some(image) = prepared {
            self.wait_acquired_semaphore(image.semaphore);
        }
        // Exporters also use the public graphics queue for conversion and
        // readback. Offscreen mode returns its ownership to the caller.
        self.submission.take();
        let command_buffers = unsafe {
            self.device.device_wait_idle().unwrap();
            for command_buffer in &self.command_buffers {
                let _ = self.device.reset_command_buffer(
                    *command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                );
            }
            let info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);
            self.device.allocate_command_buffers(&info).unwrap()
        };
        unsafe {
            if !self.command_buffers.is_empty() {
                self.device
                    .free_command_buffers(self.command_pool, &self.command_buffers);
            }
        }
        self.command_buffers = command_buffers;
        self.release_all_retired_buffers();
        // An incomplete normal frame can leave a per-frame fence reset without
        // a matching submission.  The device is idle here, so replace the ring
        // with fresh signaled fences before the offscreen ring starts using
        // them.
        let old_fences = std::mem::take(&mut self.in_flight);
        for fence in old_fences {
            unsafe { self.device.destroy_fence(fence, None) };
        }
        self.in_flight = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| super::create_fence(&self.device))
            .collect();
        self.offscreen = true;
        self.current_frame = 0;
        self.acquired_image = None;
        self.frame_wait_pending = false;
        self.frame_active = false;
        self.recording = false;
        self.pass_open = false;
        self.pass_suspended = false;
        self.cmd_dirty = false;
        self.frame_submitted = false;
    }

    /// Start recording one offscreen frame.  The returned slot is the index in
    /// the `MAX_FRAMES_IN_FLIGHT` ring.
    ///
    /// The previous use of this slot is waited on here; that is the only
    /// per-frame stall, and it is `MAX_FRAMES_IN_FLIGHT - 1` frames behind the
    /// frame currently being recorded.
    pub fn begin_offscreen_frame(&mut self) -> usize {
        assert!(self.offscreen, "enable_offscreen_video() was not called");
        assert!(
            !self.frame_active,
            "the previous offscreen frame was not finished"
        );
        let slot = self.current_frame;
        unsafe {
            let fence = self.in_flight[slot];
            match self.device.wait_for_fences(&[fence], true, 10_000_000_000) {
                Ok(()) => {}
                Err(e) => panic!("offscreen frame fence wait failed/timeout: {:?}", e),
            }
            self.reset_frame_storage();
            self.device
                .reset_fences(&[fence])
                .expect("Failed to reset the offscreen frame fence");
            self.restart_command_buffer(self.command_buffers[slot]);
        }
        self.acquired_image = Some(slot);
        self.frame_wait_pending = false;
        self.recording = true;
        self.frame_active = true;
        self.cmd_dirty = false;
        self.frame_submitted = false;
        self.cur_pipeline = None;
        self.cur_index_buffer = None;
        self.cur_vertex_buffers.clear();
        self.cur_target = None;
        self.pass_suspended = false;
        self.bound_textures.clear();
        slot
    }

    /// Record the GPU Y/UV planes into an app-owned NV12 image and copy that
    /// image into an app-owned host-visible staging buffer.
    ///
    /// The commands go into the current offscreen frame's command buffer, so
    /// they are submitted together with the scene at `finish_offscreen_frame`.
    /// The staging buffer contains tightly packed NV12 (Y plane then
    /// interleaved UV plane), which is exactly what ffmpeg's `nv12` rawvideo
    /// input expects.
    pub fn record_nv12_copy_to_staging(
        &mut self,
        y_index: usize,
        uv_index: usize,
        nv12_image_raw: u64,
        staging_buffer_raw: u64,
        width: u32,
        height: u32,
    ) {
        assert!(
            self.offscreen && self.frame_active,
            "record_nv12_copy_to_staging() requires an active offscreen frame"
        );
        assert!(
            width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0,
            "NV12 dimensions must be positive and even"
        );
        // Transfer commands cannot be recorded inside a dynamic rendering
        // instance.
        self.suspend_pass();
        let command_buffer = self.frame_command_buffer();
        self.flush_draw_commands();
        unsafe {
            let nv12 = vk::Image::from_raw(nv12_image_raw);
            let staging = vk::Buffer::from_raw(staging_buffer_raw);
            let y = self.textures[y_index].image;
            let uv = self.textures[uv_index].image;
            let y_layout = self.textures[y_index].layout;
            let uv_layout = self.textures[uv_index].layout;
            let color = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let to_transfer_src = |image, layout| {
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(layout)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(color)
                    .build()
            };
            let back_to_attachment = |image, layout| {
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(color)
                    .build()
            };
            let nv12_barrier = |old, new, src, dst| {
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(src)
                    .dst_access_mask(dst)
                    .old_layout(old)
                    .new_layout(new)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(nv12)
                    .subresource_range(color)
                    .build()
            };

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    to_transfer_src(y, y_layout),
                    to_transfer_src(uv, uv_layout),
                    nv12_barrier(
                        vk::ImageLayout::GENERAL,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                        vk::AccessFlags::TRANSFER_WRITE,
                    ),
                ],
            );

            let y_copy = vk::ImageCopy::builder()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_0,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .build();
            self.device.cmd_copy_image(
                command_buffer,
                y,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                nv12,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[y_copy],
            );
            let uv_copy = vk::ImageCopy::builder()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_1,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .extent(vk::Extent3D {
                    width: width / 2,
                    height: height / 2,
                    depth: 1,
                })
                .build();
            self.device.cmd_copy_image(
                command_buffer,
                uv,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                nv12,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[uv_copy],
            );

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[nv12_barrier(
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                )],
            );

            let nv12_regions = [
                vk::BufferImageCopy::builder()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::PLANE_0,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .build(),
                vk::BufferImageCopy::builder()
                    .buffer_offset(width as u64 * height as u64)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::PLANE_1,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: width / 2,
                        height: height / 2,
                        depth: 1,
                    })
                    .build(),
            ];
            self.device.cmd_copy_image_to_buffer(
                command_buffer,
                nv12,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging,
                &nv12_regions,
            );

            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    nv12_barrier(
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::ImageLayout::GENERAL,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                    ),
                    back_to_attachment(y, y_layout),
                    back_to_attachment(uv, uv_layout),
                ],
            );
        }
    }

    /// Copy the Y/UV render targets straight into an NV12 staging buffer.
    ///
    /// The old exporter first copied both planes into an intermediate NV12
    /// image and then copied that image to the buffer.  The intermediate copy
    /// is unnecessary: Y is R8, UV is RG8, so two `cmd_copy_image_to_buffer`
    /// calls already produce exactly the NV12 layout ffmpeg expects.
    pub fn record_yuv_copy_to_staging(
        &mut self,
        y_index: usize,
        uv_index: usize,
        staging_buffer_raw: u64,
        width: u32,
        height: u32,
    ) {
        assert!(
            self.offscreen && self.frame_active,
            "record_yuv_copy_to_staging() requires an active offscreen frame"
        );
        assert!(
            width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0,
            "NV12 dimensions must be positive and even"
        );
        self.suspend_pass();
        let command_buffer = self.frame_command_buffer();
        self.flush_draw_commands();
        unsafe {
            let staging = vk::Buffer::from_raw(staging_buffer_raw);
            let y = self.textures[y_index].image;
            let uv = self.textures[uv_index].image;
            let y_layout = self.textures[y_index].layout;
            let uv_layout = self.textures[uv_index].layout;
            let color = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let to_transfer_src = |image, layout| {
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .old_layout(layout)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(color)
                    .build()
            };
            let back_to_attachment = |image, layout| {
                vk::ImageMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(color)
                    .build()
            };
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer_src(y, y_layout), to_transfer_src(uv, uv_layout)],
            );
            let y_region = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .build();
            let uv_region = vk::BufferImageCopy::builder()
                .buffer_offset(width as u64 * height as u64)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: width / 2,
                    height: height / 2,
                    depth: 1,
                })
                .build();
            self.device.cmd_copy_image_to_buffer(
                command_buffer,
                y,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging,
                &[y_region],
            );
            self.device.cmd_copy_image_to_buffer(
                command_buffer,
                uv,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging,
                &[uv_region],
            );
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    back_to_attachment(y, y_layout),
                    back_to_attachment(uv, uv_layout),
                ],
            );
        }
    }

    /// Close and submit the offscreen frame.  The GPU is not waited on here;
    /// the fence is consumed by `begin_offscreen_frame` when this ring slot is
    /// reused.
    pub fn finish_offscreen_frame(&mut self) {
        if !self.frame_active {
            return;
        }
        self.drain_submissions();
        self.end_render_pass();
        self.flush_draw_commands();
        let slot = self.acquired_image.take().unwrap();
        let command_buffer = self.command_buffers[slot];
        unsafe {
            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end the offscreen command buffer");
            let command_buffers = [command_buffer];
            let submit = vk::SubmitInfo::builder().command_buffers(&command_buffers);
            self.device
                .queue_submit(self.graphics_queue, &[submit.build()], self.in_flight[slot])
                .expect("Failed to submit the offscreen frame");
        }
        self.last_submitted_slot = Some(slot);
        self.current_frame = (slot + 1) % MAX_FRAMES_IN_FLIGHT;
        self.recording = false;
        self.frame_active = false;
        self.pass_open = false;
        self.cmd_dirty = false;
        self.frame_submitted = true;
    }

    /// Wait for every offscreen frame still in flight (used at the end of an
    /// export so the last staging buffers can be drained).
    pub fn wait_offscreen_idle(&mut self) {
        if !self.offscreen {
            return;
        }
        let fences = self.in_flight.clone();
        unsafe {
            self.device
                .wait_for_fences(&fences, true, u64::MAX)
                .expect("Failed to wait for offscreen frames");
        }
        self.release_all_retired_buffers();
    }

    pub fn begin_pass(&mut self, pass: Option<RenderPass>, action: PassAction) {
        if self.offscreen && pass.is_none() {
            // The exporter renders into its own render targets; the swapchain
            // image is never acquired in offscreen mode, so a default pass
            // would transition an image that is not ours.
            return;
        }
        if !self.frame_active && !self.begin_frame() {
            return;
        }
        if self.render_pass_is_open(pass) && matches!(action, PassAction::Nothing) {
            return;
        }
        self.close_pass();
        self.cur_target = pass.map(|pass| pass.0);
        let area = self.open_pass(action);
        self.viewport = area;
        self.scissor = area;
        self.bound_textures.clear();
    }

    fn open_pass(&mut self, action: PassAction) -> vk::Rect2D {
        let (
            color_views,
            resolve_views,
            depth_view,
            samples,
            width,
            height,
            color_formats,
            depth_format,
        ) = match self.cur_target {
            None => {
                self.transition_swapchain_image();
                self.transition_pass_texture(
                    self.default_depth.texture as usize,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                );
                (
                    vec![self.swapchain_image_views[self.acquired_image.unwrap()]],
                    vec![vk::ImageView::null()],
                    self.textures[self.default_depth.texture as usize].view,
                    1,
                    self.swapchain_extent.width,
                    self.swapchain_extent.height,
                    vec![self.surface_format.format],
                    DEPTH_FORMAT,
                )
            }
            Some(pass) => {
                self.transition_attachments(pass);
                let module = self.passes[pass]
                    .as_ref()
                    .expect("the render pass has been deleted");
                let target = module
                    .colors
                    .first()
                    .copied()
                    .or(module.depth)
                    .expect("Render pass has no attachments");
                (
                    module
                        .colors
                        .iter()
                        .map(|color| self.textures[color.texture as usize].view)
                        .collect(),
                    module
                        .resolves
                        .iter()
                        .map(|resolve| {
                            resolve.map_or(vk::ImageView::null(), |r| {
                                self.textures[r.texture as usize].view
                            })
                        })
                        .collect::<Vec<_>>(),
                    module.depth.map_or(vk::ImageView::null(), |depth| {
                        self.textures[depth.texture as usize].view
                    }),
                    module.samples,
                    self.textures[target.texture as usize].width,
                    self.textures[target.texture as usize].height,
                    module
                        .colors
                        .iter()
                        .map(|color| self.textures[color.texture as usize].format)
                        .collect(),
                    module.depth.map_or(DEPTH_FORMAT, |depth| {
                        self.textures[depth.texture as usize].format
                    }),
                )
            }
        };
        self.cur_samples = samples;
        self.cur_color_formats = color_formats;
        self.cur_depth_format = if depth_view == vk::ImageView::null() {
            None
        } else {
            Some(depth_format)
        };

        let mut clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let mut clear_depth = vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        };
        let (color_load_op, depth_load_op) = match action {
            PassAction::Nothing => (vk::AttachmentLoadOp::LOAD, vk::AttachmentLoadOp::LOAD),
            PassAction::Clear {
                color: clear_color,
                depth,
                stencil,
            } => {
                if let Some((r, g, b, a)) = clear_color {
                    clear.color = vk::ClearColorValue {
                        float32: [r, g, b, a],
                    };
                }
                if let Some(v) = depth {
                    clear_depth.depth = v;
                }
                if let Some(v) = stencil {
                    clear_depth.stencil = v as u32;
                }
                (
                    if clear_color.is_some() {
                        vk::AttachmentLoadOp::CLEAR
                    } else {
                        vk::AttachmentLoadOp::LOAD
                    },
                    if depth.is_some() {
                        vk::AttachmentLoadOp::CLEAR
                    } else {
                        vk::AttachmentLoadOp::LOAD
                    },
                )
            }
        };

        let color_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
        let color_attachments: Vec<_> = color_views
            .iter()
            .zip(&resolve_views)
            .map(|(&color_view, &resolve_view)| {
                let has_resolve = resolve_view != vk::ImageView::null();
                vk::RenderingAttachmentInfo::builder()
                    .image_view(color_view)
                    .image_layout(color_layout)
                    .resolve_mode(if has_resolve {
                        vk::ResolveModeFlags::AVERAGE
                    } else {
                        vk::ResolveModeFlags::NONE
                    })
                    .resolve_image_view(resolve_view)
                    .resolve_image_layout(if has_resolve {
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    })
                    .load_op(color_load_op)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(clear)
                    .build()
            })
            .collect();
        let has_depth = depth_view != vk::ImageView::null();
        let depth_attachments = [vk::RenderingAttachmentInfo::builder()
            .image_view(depth_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(depth_load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: clear_depth,
            })
            .build()];

        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let rendering_info = vk::RenderingInfo {
            render_area: area,
            layer_count: 1,
            color_attachment_count: color_attachments.len() as u32,
            p_color_attachments: color_attachments.as_ptr(),
            p_depth_attachment: if has_depth {
                depth_attachments.as_ptr()
            } else {
                std::ptr::null()
            },
            p_stencil_attachment: std::ptr::null(),
            ..Default::default()
        };

        if self.recorder.enabled() {
            self.recorder.begin(
                area,
                &color_attachments,
                if has_depth {
                    Some(depth_attachments[0])
                } else {
                    None
                },
            );
        } else {
            unsafe {
                self.dynamic_rendering.begin(
                    &self.device,
                    self.frame_command_buffer(),
                    &rendering_info,
                );
            }
        }

        self.pass_open = true;
        self.pass_suspended = false;
        self.cmd_dirty = true;
        area
    }

    /// Whether the window supports transfer-source blits from its swapchain.
    /// The destination texture also needs transfer-destination and blit support.
    pub fn can_capture_window(&self) -> bool {
        !self.offscreen && self.swapchain_can_capture
    }

    /// Whether this target still owns the pass, including while its Vulkan
    /// rendering instance is suspended by a transfer.
    pub fn render_pass_is_open(&self, pass: Option<RenderPass>) -> bool {
        (self.pass_open || self.pass_suspended) && self.cur_target == pass.map(|pass| pass.0)
    }

    /// Release the given target even if a transfer already closed its rendering
    /// instance. Leave a different target alone.
    pub fn end_render_pass_if_target(&mut self, pass: Option<RenderPass>) {
        if self.render_pass_is_open(pass) {
            self.end_render_pass();
        }
    }

    /// Finish attachment writes and MSAA resolves, matching miniquad 0.4.
    pub fn end_render_pass(&mut self) {
        self.close_pass();
        self.pass_suspended = false;
        self.cur_target = None;
    }

    pub(super) fn suspend_pass(&mut self) {
        if self.pass_open {
            self.close_pass();
            self.pass_suspended = true;
        }
    }

    fn close_pass(&mut self) {
        if !self.pass_open {
            return;
        }
        if self.recorder.enabled() {
            self.recorder.batch(
                &self.cur_color_formats,
                self.cur_depth_format,
                self.cur_samples,
            );
            self.recorder.end();
        } else {
            unsafe {
                self.dynamic_rendering
                    .end(&self.device, self.frame_command_buffer());
            }
        }
        self.pass_open = false;
        self.cmd_dirty = true;
    }

    pub fn flush_draw_commands(&mut self) {
        #[cfg(feature = "tracy")]
        let _span = crate::tracy::span(crate::tracy::span_location!("Vulkan::flush_draw_commands"));
        if self.recorder.enabled() && self.frame_active {
            if self.pass_open {
                self.recorder.batch(
                    &self.cur_color_formats,
                    self.cur_depth_format,
                    self.cur_samples,
                );
            }
            let command_buffer = self.frame_command_buffer();
            self.recorder.execute(
                &self.device,
                &self.dynamic_rendering,
                command_buffer,
                self.current_frame,
            );
        }
    }

    pub(super) fn pass_image_barrier(
        &mut self,
        command: vk::CommandBuffer,
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        image: vk::ImageMemoryBarrier,
    ) {
        if self.recorder.enabled() {
            self.recorder.barrier(src, dst, image);
        } else {
            unsafe {
                self.device.cmd_pipeline_barrier(
                    command,
                    src,
                    dst,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image],
                );
            }
        }
    }

    /// `vkCmdBeginRendering` has no `initialLayout` to transition from, so the
    /// acquired swap chain image has to be moved into attachment layout
    /// explicitly. Its previous contents are never needed: the first pass of
    /// every frame clears.
    fn transition_swapchain_image(&mut self) {
        let index = self.acquired_image.unwrap();
        let image = self.swapchain_images[index];
        // Only the first pass of a frame may come from `UNDEFINED`; later passes
        // load what the earlier ones wrote, so their contents have to survive.
        let old_layout = self.swapchain_image_layouts[index];
        let command_buffer = self.frame_command_buffer();
        let (src, dst, barrier) = image_barrier(
            image,
            vk::ImageAspectFlags::COLOR,
            old_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
        // The acquire semaphore is waited at COLOR_ATTACHMENT_OUTPUT. Include
        // that stage in the source scope so the layout transition waits too.
        self.pass_image_barrier(
            command_buffer,
            src | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst,
            barrier,
        );
        self.swapchain_image_layouts[index] = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    }

    /// Move a render target's colour, resolve and depth images into attachment
    /// layouts; an earlier pass might have sampled or written them.
    fn transition_attachments(&mut self, pass: usize) {
        let module = self.passes[pass]
            .take()
            .expect("the render pass has been deleted");
        for texture in module.colors.iter().chain(module.resolves.iter().flatten()) {
            self.transition_pass_texture(
                texture.texture as usize,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }
        if let Some(depth) = module.depth {
            self.transition_pass_texture(
                depth.texture as usize,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            );
        }
        self.passes[pass] = Some(module);
    }

    pub fn clear(
        &mut self,
        color: Option<(f32, f32, f32, f32)>,
        depth: Option<f32>,
        stencil: Option<i32>,
    ) {
        if color.is_none() && depth.is_none() && stencil.is_none() {
            return;
        }
        let viewport = self.viewport;
        let scissor = self.scissor;
        let bindings = std::mem::take(&mut self.bound_textures);
        self.begin_pass(
            self.cur_target.map(RenderPass),
            PassAction::Clear {
                color,
                depth,
                stencil,
            },
        );
        self.viewport = viewport;
        self.scissor = scissor;
        if self.frame_active {
            self.bound_textures = bindings;
        }
    }

    pub fn apply_pipeline(&mut self, pipeline: &Pipeline) {
        self.cur_pipeline = Some(*pipeline);
        self.cur_params = self.pipelines[pipeline.0].params;
    }

    pub fn set_cull_face(&mut self, cull_face: CullFace) {
        self.cur_params.cull_face = cull_face;
    }

    pub fn set_color_write(&mut self, color_write: (bool, bool, bool, bool)) {
        self.cur_params.color_write = color_write;
    }

    pub fn set_blend(&mut self, color_blend: Option<BlendState>, alpha_blend: Option<BlendState>) {
        if color_blend.is_none() && alpha_blend.is_some() {
            panic!("AlphaBlend without ColorBlend");
        }
        self.cur_params.color_blend = color_blend;
        self.cur_params.alpha_blend = alpha_blend;
    }

    pub fn set_stencil(&mut self, stencil_test: Option<StencilState>) {
        self.cur_params.stencil_test = stencil_test;
    }

    pub fn apply_viewport(&mut self, x: i32, y: i32, w: i32, h: i32) {
        self.viewport = vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D {
                width: w.max(0) as u32,
                height: h.max(0) as u32,
            },
        };
    }

    pub fn apply_scissor_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        self.scissor = vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D {
                width: w.max(0) as u32,
                height: h.max(0) as u32,
            },
        };
    }

    pub(super) fn apply_bindings(
        &mut self,
        vertex_buffers: &[BufferId],
        index_buffer: BufferId,
        textures: impl Iterator<Item = usize>,
    ) {
        if !self.frame_active {
            return;
        }
        let pipeline = self
            .cur_pipeline
            .expect("apply_bindings without a bound pipeline");
        let shader = self.pipelines[pipeline.0].shader;
        let images = self.shaders[shader].images.len();

        self.cur_vertex_buffers.clear();
        self.cur_vertex_buffers
            .extend(vertex_buffers.iter().map(|buffer| Some(buffer.0)));
        self.cur_index_buffer = Some(index_buffer.0);

        self.bound_textures.clear();
        self.bound_textures.extend(textures.take(images).map(Some));
    }

    fn prepare_bound_textures(&mut self, images: usize) {
        for slot in 0..images.min(self.bound_textures.len()) {
            if let Some(index) = self.bound_textures[slot] {
                let layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                if self.textures[index].layout != layout {
                    self.suspend_pass();
                    self.transition_pass_texture(index, layout);
                }
            }
        }
    }

    pub(super) fn prepare_render_texture_for_sampling(&mut self, index: usize) {
        if !self.frame_active || !self.recording || self.frame_submitted {
            return;
        }
        assert!(
            !self.pass_open && !self.pass_suspended,
            "Finish the render pass before preparing its texture for sampling"
        );
        let texture = &self.textures[index];
        if texture.layout != vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
            || !texture.usage.contains(vk::ImageUsageFlags::SAMPLED)
        {
            return;
        }
        self.transition_pass_texture(index, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        self.cmd_dirty = true;
    }

    fn transition_pass_texture(&mut self, index: usize, to: vk::ImageLayout) {
        let module = &self.textures[index];
        // Separate rendering instances still need a dependency from the
        // previous attachment store/resolve to this one's load and writes,
        // even when there is no layout transition.
        if module.layout == to
            && !matches!(
                to,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    | vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            )
        {
            return;
        }
        let aspect = if super::resource::is_depth_format(module.format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let (src, dst, barrier) = image_barrier(module.image, aspect, module.layout, to);
        let command = self.frame_command_buffer();
        self.pass_image_barrier(command, src, dst, barrier);
        self.textures[index].layout = to;
    }

    /// # Safety
    /// `uniform_ptr` must address `size` readable bytes in the bound shader's uniform layout.
    pub unsafe fn apply_uniforms_from_bytes(&mut self, uniform_ptr: *const u8, size: usize) {
        if !self.frame_active {
            return;
        }
        let pipeline = self
            .cur_pipeline
            .expect("apply_uniforms without a bound pipeline");
        let shader = self.pipelines[pipeline.0].shader;
        let buffer_size = self.shaders[shader].uniforms_size;
        if buffer_size == 0 {
            return;
        }

        let module = &mut self.shaders[shader];
        let uniforms = &mut module.uniforms;
        uniforms
            .bytes
            .resize(buffer_size as usize, MaybeUninit::new(0));
        uniforms.offset = None;
        // Copy fields rather than the whole input so packed and std140 layouts
        // may differ and the unused std140 padding stays zeroed.
        for &(tight, std140, element, stride, count) in &module.uniform_offsets {
            assert!(
                tight as usize + count as usize * element as usize <= size,
                "Uniforms struct does not match shader uniforms layout"
            );
            for index in 0..count {
                ptr::copy_nonoverlapping(
                    uniform_ptr.add((tight + index * element) as usize).cast(),
                    uniforms
                        .bytes
                        .as_mut_ptr()
                        .add((std140 + index * stride) as usize),
                    element as usize,
                );
            }
        }
    }

    fn stage_uniforms_for_draw(&mut self, shader: usize) -> Option<vk::DeviceSize> {
        let bytes = self.shaders[shader].uniforms_size;
        if bytes == 0 {
            return None;
        }
        if let Some((epoch, offset)) = self.shaders[shader].uniforms.offset {
            if epoch == self.stream_epoch {
                return Some(offset);
            }
        }
        let offset = self.allocate_uniforms(bytes);
        let uniforms = &mut self.shaders[shader].uniforms;
        unsafe {
            let dst = self.uniform_arenas[self.current_frame]
                .mapped
                .add(offset as usize);
            if uniforms.bytes.is_empty() {
                ptr::write_bytes(dst, 0, bytes as usize);
            } else {
                ptr::copy_nonoverlapping(uniforms.bytes.as_ptr(), dst.cast(), bytes as usize);
            }
        }
        uniforms.offset = Some((self.stream_epoch, offset));
        Some(offset)
    }

    /// Reserve an aligned, immutable slice until the current command segment
    /// has been submitted and its GPU work has completed.
    fn allocate_uniforms(&mut self, size: vk::DeviceSize) -> vk::DeviceSize {
        let slot = self.current_frame;
        let alignment = self.uniform_arenas[slot].alignment;
        let offset = (self.uniform_arenas[slot].offset + alignment - 1) & !(alignment - 1);
        if offset + size > self.uniform_arenas[slot].capacity {
            self.grow_uniform_arena(slot, offset + size);
        }
        self.uniform_arenas[slot].offset = offset + size;
        offset
    }

    /// Enlarge a frame slot's arena, carrying the slices it handed out already
    /// over to the new one.
    fn grow_uniform_arena(&mut self, slot: usize, required: vk::DeviceSize) {
        unsafe {
            let mut capacity = self.uniform_arenas[slot].capacity;
            while capacity < required {
                capacity *= 2;
            }
            let used = self.uniform_arenas[slot].offset;
            let (buffer, memory, mapped) =
                self.create_mapped_buffer(capacity, vk::BufferUsageFlags::UNIFORM_BUFFER);

            let arena = &mut self.uniform_arenas[slot];
            ptr::copy_nonoverlapping(arena.mapped, mapped, used as usize);
            self.device.unmap_memory(arena.memory);
            arena.retired.push((arena.buffer, arena.memory));
            arena.buffer = buffer;
            arena.memory = memory;
            arena.mapped = mapped;
            arena.capacity = capacity;
        }
    }

    pub fn draw(&mut self, base_element: i32, num_elements: i32, num_instances: i32) {
        if !self.frame_active || num_elements == 0 || num_instances == 0 {
            return;
        }
        assert!(base_element >= 0 && num_elements >= 0 && num_instances >= 0);
        assert!(
            self.pass_open || self.pass_suspended,
            "Drawing outside a render pass"
        );
        if self.viewport.extent.width == 0 || self.viewport.extent.height == 0 {
            return;
        }
        let pipeline = self.cur_pipeline.expect("Drawing without a bound pipeline");
        let index_buffer = self.cur_index_buffer.expect("Unset index buffer");
        let shader_index = self.pipelines[pipeline.0].shader;
        self.prepare_bound_textures(self.shaders[shader_index].images.len());
        if self.pass_suspended {
            // The transfer closed only the Vulkan instance. Reload the stored
            // attachments without resetting bindings, viewport or scissor.
            self.open_pass(PassAction::Nothing);
        }
        let screen = self.cur_target.is_none();
        let mut params = self.cur_params;
        if !screen {
            // Texture targets use a positive-height Vulkan viewport to keep
            // OpenGL texture row order, which reverses framebuffer winding.
            params.front_face_order = match params.front_face_order {
                FrontFaceOrder::Clockwise => FrontFaceOrder::CounterClockwise,
                FrontFaceOrder::CounterClockwise => FrontFaceOrder::Clockwise,
            };
        }
        let vk_pipeline = self.pipeline_for_current_target(pipeline, params);
        let uniform_offset = self.stage_uniforms_for_draw(shader_index);
        let uniforms_size = self.shaders[shader_index].uniforms_size;

        self.stage_buffer_for_draw(index_buffer);
        for slot in 0..self.cur_vertex_buffers.len() {
            let buffer = self.cur_vertex_buffers[slot].expect("Unbound vertex buffer");
            self.stage_buffer_for_draw(buffer);
        }
        let shader = &self.shaders[shader_index];
        let index = &self.buffers[index_buffer];
        assert!(
            index.current_buffer != vk::Buffer::null(),
            "Buffer has been deleted"
        );
        let index_type = index.index_type.expect("Unset index buffer type");
        let (vertex_buffers, vertex_offsets) = self
            .cur_vertex_buffers
            .iter()
            .map(|slot| {
                let buffer = &self.buffers[slot.expect("Unbound vertex buffer")];
                assert!(
                    buffer.current_buffer != vk::Buffer::null(),
                    "Buffer has been deleted"
                );
                (buffer.current_buffer, buffer.current_offset)
            })
            .unzip();
        let height = self.viewport.extent.height as f32;
        let target_size = self.cur_target.map_or(
            (self.swapchain_extent.width, self.swapchain_extent.height),
            |pass| {
                let pass = self.passes[pass].as_ref().expect("Deleted render pass");
                let texture = pass.colors.first().copied().or(pass.depth).unwrap();
                let texture = &self.textures[texture.texture as usize];
                (texture.width, texture.height)
            },
        );
        let mut scissor = self.scissor;
        // OpenGL permits a scissor to extend beyond the framebuffer; Vulkan
        // requires nonnegative offsets and a rectangle inside the render area.
        let left = (scissor.offset.x as i64).clamp(0, target_size.0 as i64);
        let right =
            (scissor.offset.x as i64 + scissor.extent.width as i64).clamp(0, target_size.0 as i64);
        let bottom = (scissor.offset.y as i64).clamp(0, target_size.1 as i64);
        let top =
            (scissor.offset.y as i64 + scissor.extent.height as i64).clamp(0, target_size.1 as i64);
        scissor.offset.x = left as i32;
        scissor.offset.y = bottom as i32;
        scissor.extent.width = (right - left) as u32;
        scissor.extent.height = (top - bottom) as u32;
        if scissor.extent.width == 0 || scissor.extent.height == 0 {
            return;
        }
        if screen {
            // miniquad's scissor coordinates start at the bottom-left. The
            // negative-height screen viewport uses Vulkan's top-left origin.
            scissor.offset.y = (target_size.1 as i64 - top) as i32;
        }
        let draw = super::recording::RecordedDraw {
            pipeline: vk_pipeline,
            pipeline_layout: shader.pipeline_layout,
            descriptor_layout: shader.descriptor_set_layout,
            uniform: (uniforms_size > 0).then(|| vk::DescriptorBufferInfo {
                buffer: self.uniform_arenas[self.current_frame].buffer,
                offset: 0,
                range: uniforms_size,
            }),
            uniform_offset: uniform_offset
                .map(|offset| u32::try_from(offset).expect("Uniform offset exceeds u32")),
            images: (0..shader.images.len())
                .map(|slot| {
                    let index = self.bound_textures.get(slot).copied().flatten();
                    let texture = &self.textures[index.unwrap_or(self.dummy_texture)];
                    vk::DescriptorImageInfo {
                        sampler: texture.sampler,
                        image_view: texture.sample_view.unwrap_or(texture.view),
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    }
                })
                .collect(),
            // The screen flips Y; texture targets retain OpenGL row order.
            viewport: vk::Viewport {
                x: self.viewport.offset.x as f32,
                y: if screen {
                    target_size.1 as f32 - self.viewport.offset.y as f32
                } else {
                    self.viewport.offset.y as f32
                },
                width: self.viewport.extent.width as f32,
                height: if screen { -height } else { height },
                min_depth: 0.0,
                max_depth: 1.0,
            },
            scissor,
            index_buffer: index.current_buffer,
            index_offset: index.current_offset,
            index_type: vk_index_type(index_type),
            first_index: base_element as u32,
            index_count: num_elements as u32,
            instance_count: num_instances as u32,
            vertex_buffers,
            vertex_offsets,
        };

        if self.recorder.enabled() {
            self.recorder.draws.push(draw);
        } else {
            let set = self.allocate_descriptor_set(draw.descriptor_layout);
            draw.update_descriptors(&self.device, set);
            draw.record(&self.device, self.frame_command_buffer(), set);
        }
        self.cmd_dirty = true;
    }

    pub fn commit_frame(&mut self) {}

    pub fn present(&mut self) {
        #[cfg(feature = "tracy")]
        let _span =
            crate::tracy::span_with_callstack(crate::tracy::span_location!("Vulkan::present"));
        if !self.frame_active {
            return;
        }
        self.end_render_pass();
        self.flush_draw_commands();
        let command_buffer = self.frame_command_buffer();
        let image_index = self.acquired_image.take().unwrap();
        let swapchain_image = self.swapchain_images[image_index];
        let swapchain_layout = self.swapchain_image_layouts[image_index];

        unsafe {
            // Hand the rendered image over to the presentation engine.
            let present_barrier = vk::ImageMemoryBarrier::builder()
                .src_access_mask(
                    if swapchain_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
                        vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    } else {
                        vk::AccessFlags::empty()
                    },
                )
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(swapchain_layout)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(swapchain_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[present_barrier],
            );

            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end the command buffer");
        }

        let submission = super::submission::FrameSubmission {
            slot: self.current_frame,
            command_buffer,
            wait: self.frame_wait_pending.then_some(self.frame_wait),
            signal: self.render_finished[image_index],
            fence: self.in_flight[self.current_frame],
            swapchain: self.swapchain,
            image_index: image_index as u32,
            // begin_frame checked this slot's fence, which also retires any
            // older submission that used the next round-robin semaphore.
            next_acquire: self
                .submission
                .as_ref()
                // With fewer images, the next semaphore can still be used by
                // the other frame slot. Keep the drained acquisition path.
                .filter(|worker| {
                    worker.preacquire_enabled() && self.image_available.len() > MAX_FRAMES_IN_FLIGHT
                })
                .map(|_| (self.acquire_slot, self.image_available[self.acquire_slot])),
        };
        if let Some(worker) = &mut self.submission {
            worker.submit(submission);
        } else {
            match submission.execute(
                &self.device,
                &self.swapchain_loader,
                self.graphics_queue,
                |_| {},
            ) {
                Ok(state) => self.update_presentation(state),
                Err(error) => panic!("Failed to submit or present the frame: {:?}", error),
            }
        }
        // Android's compositor handles rotation; IDENTITY can legitimately be
        // suboptimal until the display rotates back, even after recreation.
        if !cfg!(target_os = "android") {
            self.presentation_needs_recreate |= self.swapchain_suboptimal;
        }
        self.swapchain_suboptimal = false;
        self.swapchain_image_layouts[image_index] = vk::ImageLayout::PRESENT_SRC_KHR;
        self.frame_wait_pending = false;
        self.last_submitted_slot = Some(self.current_frame);
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        self.recording = false;
        self.frame_active = false;
        // The next frame can consume its acquired image before this present
        // returns. Rebuilding still drains the worker before handles change.
    }
}
