use super::backend;
use super::resource::{image_barrier, vk_index_type};
use super::{VulkanContext, DEPTH_FORMAT, MAX_FRAMES_IN_FLIGHT};
use crate::graphics::*;
use ash::vk;
use ash::vk::Handle;
use std::{convert::TryFrom, ptr};

impl VulkanContext {
    pub(crate) fn grab_screen(&mut self, dst: usize) {
        assert!(!self.offscreen, "Window capture requires a swapchain frame");
        if !self.frame_active {
            self.begin_frame();
        }
        self.close_pass();
        let image_index = self.acquired_image.unwrap();
        let src_image = self.swapchain_images[image_index];
        let src_layout = self.swapchain_image_layouts[image_index];
        let module = &self.textures[dst];
        let (dst_image, dst_layout) = (module.image, module.layout);
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
            let aspect = vk::ImageAspectFlags::COLOR;
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
                dst_layout,
            );
        });
    }

    pub(crate) fn record_submit<F>(&mut self, f: F)
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
            let command_buffer = self
                .device
                .allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate a command buffer")[0];

            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin the command buffer");

            f(self, command_buffer);

            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to end the command buffer");

            let command_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
            self.device
                .queue_submit(
                    self.graphics_queue,
                    &[submit_info.build()],
                    vk::Fence::null(),
                )
                .expect("Failed to submit the command buffer");
            self.device
                .queue_wait_idle(self.graphics_queue)
                .expect("Failed to wait for the command buffer");

            self.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
        }
        self.release_frame_resources();
    }

    /// Submit pending draws before a synchronous transfer reads their output.
    pub(crate) fn flush_recording(&mut self) -> bool {
        if !self.recording || !self.frame_active || !self.cmd_dirty {
            return false;
        }
        self.drain_submissions();
        let image_index = self.acquired_image.unwrap();
        self.close_pass();
        self.flush_draw_commands();
        let command_buffer = self.command_buffers[image_index];
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

    /// Release the per-frame pools and start recording again, after the GPU has
    /// finished the submission `flush_recording` made.
    pub(crate) fn release_frame_resources(&mut self) {
        if !self.frame_submitted || !self.frame_active {
            return;
        }
        let image_index = self.acquired_image.unwrap();
        self.reset_frame_storage();
        self.restart_command_buffer(self.command_buffers[image_index]);
        self.frame_submitted = false;
    }

    // The caller waits for this slot's GPU work before reusing its storage.
    fn reset_frame_storage(&mut self) {
        self.free_allocated_sets();
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

    fn begin_frame(&mut self) {
        #[cfg(feature = "tracy")]
        let _span =
            crate::tracy::span_with_callstack(crate::tracy::span_location!("Vulkan::begin_frame"));
        let mut prepared = if let Some(worker) = &mut self.submission {
            let (image, recreate) = worker
                .wait_prepared()
                .expect("Failed to submit or acquire the next frame");
            self.presentation_needs_recreate |= recreate;
            image
        } else {
            None
        };
        unsafe {
            let fence = self.in_flight[self.current_frame];
            let wait_start = std::time::Instant::now();
            {
                #[cfg(feature = "tracy")]
                let _span = crate::tracy::span_with_callstack(crate::tracy::span_location!(
                    "Vulkan::wait_frame_fence (wall time)"
                ));
                self.device
                    .wait_for_fences(&[fence], true, u64::MAX)
                    .expect("Failed to wait for the frame fence");
            }
            self.frame_wait_ns = self
                .frame_wait_ns
                .saturating_add(wait_start.elapsed().as_nanos() as u64);
            self.reset_frame_storage();

            // A prepared image was acquired by the worker before it started
            // presenting. Recording into it accesses neither the queue nor
            // the swapchain. If acquisition did not find a free image, reclaim
            // host ownership before acquiring here; a GPU fence alone does
            // not synchronize this thread with the worker's present call.
            if prepared.is_none() || self.presentation_needs_recreate {
                self.drain_submissions();
                if self.presentation_needs_recreate {
                    if let Some(image) = prepared.take() {
                        self.wait_acquired_semaphore(image.semaphore);
                    }
                    self.recreate_swapchain();
                }
            }

            let (image_index, slot, suboptimal) = loop {
                let slot = self.acquire_slot;
                if let Some(image) = prepared.take() {
                    assert_eq!(
                        image.swapchain, self.swapchain,
                        "prepared image belongs to an old swapchain"
                    );
                    assert_eq!(
                        image.semaphore, self.image_available[slot],
                        "prepared image used an unexpected semaphore"
                    );
                    break (image.image_index, slot, image.suboptimal);
                }
                #[cfg(feature = "tracy")]
                let _span = crate::tracy::span_with_callstack(crate::tracy::span_location!(
                    "Vulkan::acquire_next_image (wall time)"
                ));
                match self.swapchain_loader.acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_available[slot],
                    vk::Fence::null(),
                ) {
                    Ok((image_index, suboptimal)) => break (image_index, slot, suboptimal),
                    // The swapchain no longer matches the surface; rebuild it
                    // and try again with the new one.
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        self.recreate_swapchain();
                    }
                    // With the worker drained above, `VK_NOT_READY`/`VK_TIMEOUT`
                    // are not expected: the acquisition blocks until the
                    // presentation engine returns an image. Anything else means
                    // the swapchain/device is in a state this backend cannot
                    // recover from, so fail fast instead of spinning.
                    Err(error) => panic!("Failed to acquire a swapchain image: {:?}", error),
                }
            };

            // Reset only once an image was acquired and a submission will use
            // this fence; the slot stays idle otherwise.
            self.device
                .reset_fences(&[fence])
                .expect("Failed to reset the frame fence");
            let image_index = image_index as usize;
            self.swapchain_suboptimal = suboptimal;

            self.frame_wait = self.image_available[slot];
            self.acquire_slot = (slot + 1) % self.image_available.len();
            self.acquired_image = Some(image_index);
            // The presentation engine owned the image until now, so whatever it
            // holds is undefined.
            self.swapchain_image_layouts[image_index] = vk::ImageLayout::UNDEFINED;

            self.restart_command_buffer(self.command_buffers[image_index]);

            self.frame_wait_pending = true;
            self.recording = true;
            self.frame_active = true;
            self.cmd_dirty = false;
        }

        self.cur_pipeline = None;
        self.cur_index_buffer = None;
        self.cur_vertex_buffers.clear();
        self.pending.clear();
    }

    /// Switch this context to offscreen video export.
    ///
    /// In offscreen mode a frame is recorded into a command-buffer ring indexed
    /// by `current_frame`; no swap-chain image is acquired and nothing is
    /// presented.  The ring is what lets the exporter keep several frames in
    /// flight instead of blocking on a readback fence every frame.
    ///
    /// Must be called once, before the first `begin_offscreen_frame`, while the
    /// device is idle.
    pub fn enable_offscreen_video(&mut self) {
        if self.offscreen {
            return;
        }
        self.drain_submissions();
        // Exporters also use the public graphics queue for conversion and
        // readback. Offscreen mode returns its ownership to the caller.
        self.submission.take();
        let command_buffers = unsafe {
            self.device.device_wait_idle().unwrap();
            if self.pass_open && self.acquired_image.is_some() {
                self.close_pass();
                self.flush_draw_commands();
            }
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
        self.frame_active = false;
        self.recording = false;
        self.pass_open = false;
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
        self.pending.clear();
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
        self.close_pass();
        let command_buffer = self.command_buffers[self.acquired_image.unwrap()];
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
        self.close_pass();
        let command_buffer = self.command_buffers[self.acquired_image.unwrap()];
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
        self.close_pass();
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
    }

    pub fn begin_pass(&mut self, pass: Option<RenderPass>, action: PassAction) {
        if self.offscreen && pass.is_none() {
            // The exporter renders into its own render targets; the swapchain
            // image is never acquired in offscreen mode, so a default pass
            // would transition an image that is not ours.
            return;
        }
        if !self.frame_active {
            self.begin_frame();
        }
        let target = pass.map(|pass| pass.0);
        // Reuse an open target when no clear is requested. end_render_pass
        // still closes the instance so resolves finish at the public boundary.
        if self.pass_open && self.cur_pass == target && matches!(action, PassAction::Nothing) {
            return;
        }
        if self.pass_open {
            self.close_pass();
        }
        self.transition_sampled_textures();

        let (
            color_views,
            resolve_views,
            depth_view,
            samples,
            width,
            height,
            color_formats,
            depth_format,
        ) = match pass {
            None => {
                self.transition_swapchain_image();
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
                self.transition_attachments(pass.0);
                let module = self.passes[pass.0]
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
        self.cur_target = target;

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
                self.device.cmd_begin_rendering(
                    self.command_buffers[self.acquired_image.unwrap()],
                    &rendering_info,
                );
            }
        }

        self.pass_open = true;
        self.cur_pass = pass.map(|pass| pass.0);
        self.viewport = area;
        self.scissor = area;
        self.cmd_dirty = true;
        self.pending.clear();
    }

    /// Finish attachment writes and MSAA resolves, matching miniquad 0.4.
    pub fn end_render_pass(&mut self) {
        self.close_pass();
        self.cur_target = None;
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
                self.device
                    .cmd_end_rendering(self.command_buffers[self.acquired_image.unwrap()]);
            }
        }
        self.pass_open = false;
        self.cur_pass = None;
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
            self.recorder.execute(
                &self.device,
                self.command_buffers[self.acquired_image.unwrap()],
                self.current_frame,
            );
        }
    }

    fn pass_image_barrier(
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
        if old_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            return;
        }
        let command_buffer = self.command_buffers[index];
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

    /// Move a render target's colour and resolve images into attachment layout.
    ///
    /// The other half of `transition_sampled_textures`: a target may have been
    /// sampled (or written) by an earlier pass, and `vkCmdBeginRendering` does
    /// not transition on the application's behalf.
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
        let pending = std::mem::take(&mut self.pending);
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
        self.pending = pending;
    }

    pub fn apply_pipeline(&mut self, pipeline: &Pipeline) {
        self.cur_pipeline = Some(*pipeline);
        self.cur_params = self.pipelines[pipeline.0].params;
        self.pending.uniform_offset = None;
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

    pub(crate) fn apply_bindings(&mut self, bindings: &backend::Bindings) {
        let pipeline = self
            .cur_pipeline
            .expect("apply_bindings without a bound pipeline");
        let shader = self.pipelines[pipeline.0].shader;
        let images = self.shaders[shader].images.len();

        self.cur_vertex_buffers
            .resize(bindings.vertex_buffers.len(), None);
        for (slot, buffer) in bindings.vertex_buffers.iter().enumerate() {
            self.cur_vertex_buffers[slot] = Some(buffer.0);
        }
        self.cur_index_buffer = Some(bindings.index_buffer.0);

        let sampled: Vec<usize> = bindings
            .images
            .iter()
            .take(images)
            .filter(|texture| texture.texture != 0)
            .map(|texture| texture.texture as usize)
            .collect();
        self.pending.sampled.extend(sampled.iter().copied());
        let needs_transition = sampled
            .iter()
            .any(|&index| self.textures[index].layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        if needs_transition && self.pass_open {
            let target = self.cur_target;
            let viewport = self.viewport;
            let scissor = self.scissor;
            self.close_pass();
            self.begin_pass(target.map(RenderPass), PassAction::Nothing);
            self.viewport = viewport;
            self.scissor = scissor;
        }

        self.pending.images = vec![None; images];
        for (slot, texture) in bindings.images.iter().take(images).enumerate() {
            if texture.texture == 0 {
                continue;
            }
            let module = &self.textures[texture.texture as usize];
            self.pending.images[slot] = Some(vk::DescriptorImageInfo {
                sampler: module.sampler,
                image_view: module.sample_view.unwrap_or(module.view),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            });
        }
    }

    /// Move the textures the pass just sampled into `SHADER_READ_ONLY_OPTIMAL`.
    ///
    /// Recorded after `cmdEndRendering` because Vulkan rejects image barriers
    /// inside a dynamic rendering instance.
    fn transition_sampled_textures(&mut self) {
        for index in std::mem::take(&mut self.pending.sampled) {
            self.transition_pass_texture(index, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
    }

    fn transition_pass_texture(&mut self, index: usize, to: vk::ImageLayout) {
        let module = &self.textures[index];
        if module.layout == to {
            return;
        }
        let aspect = if super::resource::is_depth_format(module.format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let (src, dst, barrier) = image_barrier(module.image, aspect, module.layout, to);
        let command = self.command_buffers[self.acquired_image.unwrap()];
        self.pass_image_barrier(command, src, dst, barrier);
        self.textures[index].layout = to;
    }

    /// # Safety
    /// `uniform_ptr` must address `size` readable bytes in the bound shader's uniform layout.
    pub unsafe fn apply_uniforms_from_bytes(&mut self, uniform_ptr: *const u8, size: usize) {
        let pipeline = self
            .cur_pipeline
            .expect("apply_uniforms without a bound pipeline");
        let shader = self.pipelines[pipeline.0].shader;
        let buffer_size = self.shaders[shader].uniforms_size;
        if buffer_size == 0 {
            return;
        }

        let offset = match self.pending.uniform_offset {
            Some(offset) => offset,
            None => {
                let offset = self.allocate_uniforms(buffer_size);
                unsafe {
                    ptr::write_bytes(
                        self.uniform_arenas[self.current_frame]
                            .mapped
                            .add(offset as usize),
                        0,
                        buffer_size as usize,
                    );
                }
                self.pending.uniform_offset = Some(offset);
                offset
            }
        };

        unsafe {
            let mapped = self.uniform_arenas[self.current_frame]
                .mapped
                .add(offset as usize);
            for (tight, std140, element, stride, count) in &self.shaders[shader].uniform_offsets {
                assert!(
                    *tight as usize + *count as usize * *element as usize <= size,
                    "Uniforms struct does not match shader uniforms layout"
                );
                for index in 0..*count {
                    ptr::copy_nonoverlapping(
                        uniform_ptr.add((*tight + index * *element) as usize),
                        mapped.add((*std140 + index * *stride) as usize),
                        *element as usize,
                    );
                }
            }
        }
    }

    /// Reserve a slice of the frame's uniform arena for one draw call.
    ///
    /// The arena is a bump allocator: slices are aligned to what the device
    /// requires of a dynamic uniform offset and are all released when the frame
    /// ends.
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
        if num_elements == 0 || num_instances == 0 {
            return;
        }
        assert!(base_element >= 0 && num_elements >= 0 && num_instances >= 0);
        assert!(self.pass_open, "Drawing outside a render pass");
        if self.viewport.extent.width == 0 || self.viewport.extent.height == 0 {
            return;
        }
        let pipeline = self.cur_pipeline.expect("Drawing without a bound pipeline");
        let index_buffer = self.cur_index_buffer.expect("Unset index buffer");
        let shader_index = self.pipelines[pipeline.0].shader;
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
        let uniforms_size = self.shaders[shader_index].uniforms_size;
        let uniform_offset = if uniforms_size == 0 {
            None
        } else {
            let offset = match self.pending.uniform_offset {
                Some(offset) => offset,
                None => {
                    let offset = self.allocate_uniforms(uniforms_size);
                    unsafe {
                        ptr::write_bytes(
                            self.uniform_arenas[self.current_frame]
                                .mapped
                                .add(offset as usize),
                            0,
                            uniforms_size as usize,
                        );
                    }
                    offset
                }
            };
            Some(offset)
        };

        let shader = &self.shaders[shader_index];
        let dummy = &self.textures[self.dummy_texture];
        let dummy_info = vk::DescriptorImageInfo {
            sampler: dummy.sampler,
            image_view: dummy.sample_view.unwrap_or(dummy.view),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
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
                .map(|slot| self.pending.images[slot].unwrap_or(dummy_info))
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
            first_index: u32::try_from(base_element).expect("Negative first index"),
            index_count: u32::try_from(num_elements).expect("Negative index count"),
            instance_count: u32::try_from(num_instances).expect("Negative instance count"),
            vertex_buffers,
            vertex_offsets,
        };

        if self.recorder.enabled() {
            self.recorder.draws.push(draw);
        } else {
            let set = self.allocate_descriptor_set(draw.descriptor_layout);
            draw.update_descriptors(&self.device, set);
            draw.record(
                &self.device,
                self.command_buffers[self.acquired_image.unwrap()],
                set,
            );
        }
        self.cmd_dirty = true;
        self.pending.uniform_offset = None;
    }

    pub fn commit_frame(&mut self) {}

    pub fn present(&mut self) {
        #[cfg(feature = "tracy")]
        let _span =
            crate::tracy::span_with_callstack(crate::tracy::span_location!("Vulkan::present"));
        if !self.frame_active {
            return;
        }
        self.close_pass();
        self.flush_draw_commands();
        let image_index = self.acquired_image.take().unwrap();
        let command_buffer = self.command_buffers[image_index];
        let swapchain_image = self.swapchain_images[image_index];
        let swapchain_layout = self.swapchain_image_layouts[image_index];

        unsafe {
            // Hand the rendered image over to the presentation engine.
            let present_barrier = vk::ImageMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
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
            // begin_frame waited on this slot's fence, which also retires any
            // older submission that used the next round-robin semaphore.
            next_acquire: self
                .submission
                .as_ref()
                // With fewer images, the next semaphore can still be used by
                // the other frame slot. Keep the drained acquisition path.
                .filter(|worker| {
                    worker.preacquire_enabled() && self.image_available.len() > MAX_FRAMES_IN_FLIGHT
                })
                .map(|_| self.image_available[self.acquire_slot]),
        };
        if let Some(worker) = &mut self.submission {
            worker.submit(submission);
        } else {
            self.presentation_needs_recreate |= submission
                .execute(
                    &self.device,
                    &self.swapchain_loader,
                    self.graphics_queue,
                    |_| {},
                )
                .expect("Failed to submit or present the frame");
        }
        self.presentation_needs_recreate |= self.swapchain_suboptimal;
        self.swapchain_suboptimal = false;
        self.swapchain_image_layouts[image_index] = vk::ImageLayout::PRESENT_SRC_KHR;
        self.frame_wait_pending = false;
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        self.recording = false;
        self.frame_active = false;
        // The next frame can consume its acquired image before this present
        // returns. Rebuilding still drains the worker before handles change.
    }
}
