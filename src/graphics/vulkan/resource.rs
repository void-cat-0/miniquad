use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    ptr,
};

use ash::{vk, Device};

use crate::graphics::*;

use super::backend::{IndexType, Texture};
use super::cstr;
use super::{
    find_memory_type_index, VulkanContext, DEPTH_FORMAT, IMAGE_BINDING_BASE, UNIFORM_BINDING,
};

pub struct ShaderModule {
    pub vertex: vk::ShaderModule,
    pub fragment: vk::ShaderModule,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub uniforms_size: vk::DeviceSize,
    /// `(tight offset, std140 offset, element size, std140 stride, element count)`
    /// per uniform in `ShaderMeta` declaration order.
    pub uniform_offsets: Vec<(u32, u32, u32, u32, u32)>,
    pub images: Vec<String>,
    pub float_vertex_locations: BTreeSet<u32>,
}

pub struct PipelineVariant {
    pub color_formats: Vec<vk::Format>,
    pub samples: u32,
    pub depth_format: Option<vk::Format>,
    pub params: PipelineParams,
    pub pipeline: vk::Pipeline,
}

pub struct PipelineModule {
    pub variants: Vec<PipelineVariant>,
    pub shader: usize,
    pub inputs: Vec<VertexInput>,
    pub params: PipelineParams,
}

pub struct BufferModule {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub current_buffer: vk::Buffer,
    pub current_offset: vk::DeviceSize,
    pub index_type: Option<IndexType>,
    pub widen: bool,
}

pub struct TextureModule {
    pub image: vk::Image,
    pub view: vk::ImageView,
    /// View descriptors bind. `RGB8` render targets are stored as four-channel
    /// images but sample with a constant alpha of one, the way OpenGL treats a
    /// three-channel texture, so they need a second view.
    pub sample_view: Option<vk::ImageView>,
    pub sampler: vk::Sampler,
    pub memory: vk::DeviceMemory,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub layout: vk::ImageLayout,
    pub mip_levels: u32,
    pub filter: FilterMode,
    pub wrap: TextureWrap,
    pub mipmap_filter: MipmapFilterMode,
    pub access: TextureAccess,
}

pub struct RenderPassModule {
    pub colors: Vec<Texture>,
    pub depth: Option<Texture>,
    /// Single-sampled target a multisampled colour attachment resolves into.
    pub resolves: Vec<Option<Texture>>,
    pub samples: u32,
}

pub(crate) fn empty_texture_module() -> TextureModule {
    TextureModule {
        image: vk::Image::null(),
        view: vk::ImageView::null(),
        sample_view: None,
        sampler: vk::Sampler::null(),
        memory: vk::DeviceMemory::null(),
        width: 0,
        height: 0,
        format: vk::Format::UNDEFINED,
        layout: vk::ImageLayout::UNDEFINED,
        mip_levels: 1,
        filter: FilterMode::Linear,
        wrap: TextureWrap::Clamp,
        mipmap_filter: MipmapFilterMode::None,
        access: TextureAccess::Static,
    }
}

// Until a resource enters the resource table, failures must release the handles
// already created for it. The table owns them after `commit`.
struct ResourceAllocation {
    device: Device,
    buffer: vk::Buffer,
    image: vk::Image,
    memory: vk::DeviceMemory,
    views: Vec<vk::ImageView>,
    sampler: vk::Sampler,
    committed: bool,
}

impl ResourceAllocation {
    fn new(device: &Device, image: vk::Image, memory: vk::DeviceMemory) -> Self {
        Self {
            device: device.clone(),
            buffer: vk::Buffer::null(),
            image,
            memory,
            views: Vec::new(),
            sampler: vk::Sampler::null(),
            committed: false,
        }
    }

    fn buffer(device: &Device, buffer: vk::Buffer, memory: vk::DeviceMemory) -> Self {
        let mut allocation = Self::new(device, vk::Image::null(), memory);
        allocation.buffer = buffer;
        allocation
    }

    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for ResourceAllocation {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
            for &view in &self.views {
                self.device.destroy_image_view(view, None);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl VulkanContext {
    pub(crate) fn set_sampler_params(
        &mut self,
        index: usize,
        min_filter: FilterMode,
        mag_filter: FilterMode,
        wrap_x: TextureWrap,
        wrap_y: TextureWrap,
        mipmap_filter: MipmapFilterMode,
    ) {
        self.synchronize_resources();
        let module = &mut self.textures[index];
        let (mode, max_lod) = match mipmap_filter {
            MipmapFilterMode::None => (vk::SamplerMipmapMode::NEAREST, 0.0),
            MipmapFilterMode::Nearest => (vk::SamplerMipmapMode::NEAREST, vk::LOD_CLAMP_NONE),
            MipmapFilterMode::Linear => (vk::SamplerMipmapMode::LINEAR, vk::LOD_CLAMP_NONE),
        };
        let info = vk::SamplerCreateInfo::builder()
            .min_filter(vk_sampler_filter(min_filter))
            .mag_filter(vk_sampler_filter(mag_filter))
            .mipmap_mode(mode)
            .max_lod(max_lod)
            .address_mode_u(vk_sampler_wrap(wrap_x))
            .address_mode_v(vk_sampler_wrap(wrap_y))
            .address_mode_w(vk_sampler_wrap(wrap_x));
        let sampler = unsafe { self.device.create_sampler(&info, None).unwrap() };
        let old = std::mem::replace(&mut module.sampler, sampler);
        module.filter = min_filter;
        module.wrap = wrap_x;
        module.mipmap_filter = mipmap_filter;
        for image in self.pending.images.iter_mut().flatten() {
            if image.sampler == old {
                image.sampler = sampler;
            }
        }
        unsafe {
            self.device.destroy_sampler(old, None);
        }
    }
    /// Allocate a buffer and bind memory of the requested properties to it.
    pub(crate) unsafe fn create_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        required: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = self
            .device
            .create_buffer(&create_info, None)
            .expect("Failed to create the buffer");

        let requirements = self.device.get_buffer_memory_requirements(buffer);
        let memory_type_index = find_memory_type_index(
            self.memory_properties,
            requirements.memory_type_bits,
            required,
        )
        .unwrap_or_else(|| {
            self.device.destroy_buffer(buffer, None);
            panic!("No memory type satisfies the buffer requirements");
        });

        let allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = self
            .device
            .allocate_memory(&allocate_info, None)
            .unwrap_or_else(|error| {
                self.device.destroy_buffer(buffer, None);
                panic!("Failed to allocate buffer memory: {:?}", error);
            });
        if let Err(error) = self.device.bind_buffer_memory(buffer, memory, 0) {
            self.device.destroy_buffer(buffer, None);
            self.device.free_memory(memory, None);
            panic!("Failed to bind buffer memory: {:?}", error);
        }

        (buffer, memory)
    }

    pub(crate) fn create_mapped_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> (vk::Buffer, vk::DeviceMemory, *mut u8) {
        unsafe {
            let (buffer, memory) = self.create_buffer(
                size,
                usage,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let mapped = self
                .device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap_or_else(|error| {
                    self.device.destroy_buffer(buffer, None);
                    self.device.free_memory(memory, None);
                    panic!("Failed to map buffer memory: {:?}", error);
                }) as *mut u8;
            (buffer, memory, mapped)
        }
    }

    /// Allocate an image and bind memory of the requested properties to it.
    pub(crate) unsafe fn create_image(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_levels: u32,
        samples: vk::SampleCountFlags,
    ) -> (vk::Image, vk::DeviceMemory) {
        let create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = self
            .device
            .create_image(&create_info, None)
            .expect("Failed to create the image");

        let requirements = self.device.get_image_memory_requirements(image);
        let memory_type_index = find_memory_type_index(
            self.memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .unwrap_or_else(|| {
            self.device.destroy_image(image, None);
            panic!("No memory type satisfies the image requirements");
        });

        let allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = self
            .device
            .allocate_memory(&allocate_info, None)
            .unwrap_or_else(|error| {
                self.device.destroy_image(image, None);
                panic!("Failed to allocate image memory: {:?}", error);
            });
        if let Err(error) = self.device.bind_image_memory(image, memory, 0) {
            self.device.destroy_image(image, None);
            self.device.free_memory(memory, None);
            panic!("Failed to bind image memory: {:?}", error);
        }

        (image, memory)
    }

    // -----------------------------------------------------------------
    // Shaders
    // -----------------------------------------------------------------

    pub fn new_shader(
        &mut self,
        vertex_source: &str,
        fragment_source: &str,
        meta: ShaderMeta,
    ) -> Result<ShaderId, ShaderError> {
        let (vertex, fragment, arrays) =
            super::shader::compile_glsl_with_layout(vertex_source, fragment_source, &meta)?;
        let (uniform_offsets, uniforms_size) = std140_layout_with_arrays(&meta.uniforms, &arrays);
        let float_vertex_locations = vertex_float_locations(&vertex);

        unsafe {
            let vertex = self
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        code_size: vertex.len() * 4,
                        p_code: vertex.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|error| {
                    ShaderError::LinkError(format!(
                        "Failed to create the vertex shader module: {:?}",
                        error
                    ))
                })?;
            let fragment = self
                .device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        code_size: fragment.len() * 4,
                        p_code: fragment.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .map_err(|error| {
                    self.device.destroy_shader_module(vertex, None);
                    ShaderError::LinkError(format!(
                        "Failed to create the fragment shader module: {:?}",
                        error
                    ))
                })?;

            let mut bindings = Vec::new();
            if uniforms_size > 0 {
                bindings.push(
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(UNIFORM_BINDING)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                        .build(),
                );
            }
            for slot in 0..meta.images.len() {
                bindings.push(
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(IMAGE_BINDING_BASE + slot as u32)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                        .build(),
                );
            }

            let descriptor_set_layout = self
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings),
                    None,
                )
                .map_err(|error| {
                    self.device.destroy_shader_module(vertex, None);
                    self.device.destroy_shader_module(fragment, None);
                    ShaderError::LinkError(format!(
                        "Failed to create the descriptor set layout: {:?}",
                        error
                    ))
                })?;

            let pipeline_layout = self
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(std::slice::from_ref(&descriptor_set_layout)),
                    None,
                )
                .map_err(|error| {
                    self.device
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    self.device.destroy_shader_module(vertex, None);
                    self.device.destroy_shader_module(fragment, None);
                    ShaderError::LinkError(format!(
                        "Failed to create the pipeline layout: {:?}",
                        error
                    ))
                })?;

            let module = ShaderModule {
                vertex,
                fragment,
                descriptor_set_layout,
                pipeline_layout,
                uniforms_size: uniforms_size as vk::DeviceSize,
                uniform_offsets,
                images: meta.images,
                float_vertex_locations,
            };
            let index = if let Some(index) = self.free_shaders.pop() {
                self.shaders[index] = module;
                index
            } else {
                self.shaders.push(module);
                self.shaders.len() - 1
            };
            Ok(ShaderId(index))
        }
    }

    // -----------------------------------------------------------------
    // Pipelines
    // -----------------------------------------------------------------

    pub fn new_pipeline(
        &mut self,
        buffer_layout: &[BufferLayout],
        attributes: &[VertexAttribute],
        shader: ShaderId,
        params: PipelineParams,
    ) -> Pipeline {
        let inputs = vertex_inputs(
            buffer_layout,
            attributes,
            &self.shaders[shader.0].float_vertex_locations,
        );

        let module = PipelineModule {
            variants: Vec::new(),
            shader: shader.0,
            inputs,
            params,
        };
        let index = if let Some(index) = self.free_pipelines.pop() {
            self.pipelines[index] = module;
            index
        } else {
            self.pipelines.push(module);
            self.pipelines.len() - 1
        };
        Pipeline(index)
    }

    /// The pipeline a target of `color_format` with `samples` samples per pixel
    /// needs. Both are baked into the `VkPipeline`, so the first draw into a new
    /// kind of target compiles the variant for it.
    pub fn pipeline_for_target(
        &mut self,
        pipeline: Pipeline,
        color_formats: &[vk::Format],
        samples: u32,
        depth_format: Option<vk::Format>,
        params: PipelineParams,
    ) -> vk::Pipeline {
        let module = &self.pipelines[pipeline.0];
        if let Some(existing) = module.variants.iter().find(|variant| {
            variant.color_formats == color_formats
                && variant.samples == samples
                && variant.depth_format == depth_format
                && variant.params == params
        }) {
            return existing.pipeline;
        }
        let shader = module.shader;
        let created = self.create_graphics_pipeline(
            shader,
            &module.inputs,
            &params,
            samples,
            color_formats,
            depth_format,
        );
        self.pipelines[pipeline.0].variants.push(PipelineVariant {
            color_formats: color_formats.to_vec(),
            samples,
            depth_format,
            params,
            pipeline: created,
        });
        created
    }

    pub(crate) fn pipeline_for_current_target(
        &mut self,
        pipeline: Pipeline,
        params: PipelineParams,
    ) -> vk::Pipeline {
        let color_formats = std::mem::take(&mut self.cur_color_formats);
        let created = self.pipeline_for_target(
            pipeline,
            &color_formats,
            self.cur_samples,
            self.cur_depth_format,
            params,
        );
        self.cur_color_formats = color_formats;
        created
    }

    fn create_graphics_pipeline(
        &self,
        shader: usize,
        inputs: &[VertexInput],
        params: &PipelineParams,
        samples: u32,
        color_formats: &[vk::Format],
        depth_format: Option<vk::Format>,
    ) -> vk::Pipeline {
        let shader = &self.shaders[shader];
        unsafe {
            let entry_name = cstr(b"main\0");
            let stages = [
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: shader.vertex,
                    p_name: entry_name.as_ptr(),
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    module: shader.fragment,
                    p_name: entry_name.as_ptr(),
                    ..Default::default()
                },
            ];

            let vertex_bindings: Vec<vk::VertexInputBindingDescription> = inputs
                .iter()
                .map(|input| input.binding)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .map(|binding| {
                    let first = inputs.iter().find(|input| input.binding == binding);
                    vk::VertexInputBindingDescription {
                        binding,
                        stride: first.map_or(0, |input| input.stride),
                        input_rate: first
                            .map_or(vk::VertexInputRate::VERTEX, |input| input.input_rate),
                    }
                })
                .collect();
            let vertex_attributes: Vec<vk::VertexInputAttributeDescription> = inputs
                .iter()
                .map(|input| vk::VertexInputAttributeDescription {
                    location: input.location,
                    binding: input.binding,
                    format: input.format,
                    offset: input.offset,
                })
                .collect();

            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(&vertex_bindings)
                .vertex_attribute_descriptions(&vertex_attributes);

            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk_primitive(params.primitive_type))
                .primitive_restart_enable(false);

            let viewport_state = vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            };

            let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
                depth_clamp_enable: 0,
                rasterizer_discard_enable: 0,
                polygon_mode: vk::PolygonMode::FILL,
                line_width: 1.0,
                cull_mode: vk_cull(params.cull_face),
                front_face: vk_front_face(params.front_face_order),
                depth_bias_enable: 0,
                ..Default::default()
            };

            let (stencil_test_enable, front, back) = match params.stencil_test {
                Some(stencil) => (
                    1,
                    vk_stencil_face(stencil.front),
                    vk_stencil_face(stencil.back),
                ),
                None => (
                    0,
                    vk::StencilOpState::default(),
                    vk::StencilOpState::default(),
                ),
            };
            let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: (depth_format.is_some() && params.depth_write) as u32,
                depth_write_enable: (depth_format.is_some() && params.depth_write) as u32,
                depth_compare_op: vk_compare(params.depth_test),
                stencil_test_enable,
                front,
                back,
                max_depth_bounds: 1.0,
                ..Default::default()
            };

            let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
                blend_enable: params.color_blend.is_some() as u32,
                src_color_blend_factor: vk_blend_factor(
                    params
                        .color_blend
                        .map_or(BlendFactor::One, |state| state.sfactor),
                ),
                dst_color_blend_factor: vk_blend_factor(
                    params
                        .color_blend
                        .map_or(BlendFactor::Zero, |state| state.dfactor),
                ),
                color_blend_op: vk_blend_op(
                    params
                        .color_blend
                        .map_or(Equation::Add, |state| state.equation),
                ),
                src_alpha_blend_factor: vk_blend_factor(
                    params
                        .alpha_blend
                        .or(params.color_blend)
                        .map_or(BlendFactor::One, |state| state.sfactor),
                ),
                dst_alpha_blend_factor: vk_blend_factor(
                    params
                        .alpha_blend
                        .or(params.color_blend)
                        .map_or(BlendFactor::Zero, |state| state.dfactor),
                ),
                alpha_blend_op: vk_blend_op(
                    params
                        .alpha_blend
                        .or(params.color_blend)
                        .map_or(Equation::Add, |state| state.equation),
                ),
                color_write_mask: color_write_mask(params.color_write),
            };

            let blend_attachments = vec![color_blend_attachment; color_formats.len()];
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(&blend_attachments)
                .blend_constants([0.0; 4]);

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

            let multisample_state = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::from_raw(samples),
                ..Default::default()
            };

            // Dynamic rendering: the pipeline declares its attachment formats
            // here instead of being tied to one `VkRenderPass`, so it has to be
            // told the format of the target it is about to draw into.

            let mut rendering =
                vk::PipelineRenderingCreateInfo::builder().color_attachment_formats(color_formats);
            if let Some(depth_format) = depth_format {
                rendering = rendering.depth_attachment_format(depth_format);
            }
            let mut rendering = rendering.build();

            let create_info = vk::GraphicsPipelineCreateInfo::builder()
                .stages(&stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .depth_stencil_state(&depth_stencil_state)
                .color_blend_state(&color_blend_state)
                .dynamic_state(&dynamic_state)
                .layout(shader.pipeline_layout)
                .push_next(&mut rendering)
                .render_pass(vk::RenderPass::null())
                .subpass(0);

            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info.build()], None)
                .map(|pipelines| pipelines[0])
                .unwrap_or_else(|(pipelines, err)| {
                    for pipeline in pipelines {
                        self.device.destroy_pipeline(pipeline, None);
                    }
                    panic!("Failed to create the graphics pipeline: {:?}", err)
                })
        }
    }

    // -----------------------------------------------------------------
    // Buffers
    // -----------------------------------------------------------------

    pub fn new_buffer(
        &mut self,
        buffer_type: BufferType,
        index_type: Option<IndexType>,
        size: usize,
        data: Option<*const u8>,
    ) -> BufferId {
        unsafe {
            let usage = match buffer_type {
                BufferType::VertexBuffer => vk::BufferUsageFlags::VERTEX_BUFFER,
                BufferType::IndexBuffer => vk::BufferUsageFlags::INDEX_BUFFER,
            };
            let capacity = if index_type == Some(IndexType::Byte) {
                size * 2
            } else {
                size
            };
            let (buffer, memory) = self.create_buffer(
                capacity.max(4) as vk::DeviceSize,
                usage,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let mut allocation = ResourceAllocation::buffer(&self.device, buffer, memory);

            if let Some(data) = data {
                if index_type == Some(IndexType::Byte) {
                    let wide = widen_indices(data, size);
                    self.write_buffer(memory, 0, wide.as_ptr(), wide.len());
                } else {
                    self.write_buffer(memory, 0, data, size);
                }
            }

            let module = BufferModule {
                buffer,
                memory,
                size: capacity as vk::DeviceSize,
                current_buffer: buffer,
                current_offset: 0,
                index_type: index_type.map(|index_type| {
                    if index_type == IndexType::Byte {
                        IndexType::Short
                    } else {
                        index_type
                    }
                }),
                widen: index_type == Some(IndexType::Byte),
            };
            let index = if let Some(index) = self.free_buffers.pop() {
                self.buffers[index] = module;
                index
            } else {
                self.buffers.push(module);
                self.buffers.len() - 1
            };
            allocation.commit();

            BufferId(index)
        }
    }

    pub(crate) unsafe fn write_buffer(
        &self,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
        data: *const u8,
        size: usize,
    ) {
        if size == 0 {
            return;
        }
        let mapped = self
            .device
            .map_memory(
                memory,
                offset,
                size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .expect("Failed to map buffer memory");
        ptr::copy_nonoverlapping(data, mapped as *mut u8, size);
        self.device.unmap_memory(memory);
    }

    pub fn update_buffer(&mut self, index: usize, data: *const u8, size: usize) {
        assert!(
            self.buffers[index].buffer != vk::Buffer::null(),
            "Buffer has been deleted"
        );
        let (buffer, offset) = if self.buffers[index].widen {
            let wide = widen_indices(data, size);
            self.append_stream(wide.as_ptr(), wide.len())
        } else {
            self.append_stream(data, size)
        };
        let module = &mut self.buffers[index];
        module.current_buffer = buffer;
        module.current_offset = offset;
    }

    pub(crate) fn append_stream(
        &mut self,
        data: *const u8,
        size: usize,
    ) -> (vk::Buffer, vk::DeviceSize) {
        if size == 0 {
            return (self.stream_arenas[self.current_frame].buffer, 0);
        }
        unsafe {
            let slot = self.current_frame;
            let alignment = 16;
            let offset = (self.stream_arenas[slot].offset + alignment - 1) & !(alignment - 1);
            if offset + size as vk::DeviceSize > self.stream_arenas[slot].capacity {
                self.grow_stream_arena(slot, offset + size as vk::DeviceSize);
            }
            let arena = &mut self.stream_arenas[slot];
            ptr::copy_nonoverlapping(data, arena.mapped.add(offset as usize), size);
            arena.offset = offset + size as vk::DeviceSize;
            (arena.buffer, offset)
        }
    }

    fn grow_stream_arena(&mut self, slot: usize, required: vk::DeviceSize) {
        unsafe {
            let mut capacity = self.stream_arenas[slot].capacity;
            while capacity < required {
                capacity *= 2;
            }
            let used = self.stream_arenas[slot].offset;
            let (buffer, memory, mapped) = self.create_mapped_buffer(
                capacity,
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
            );

            let arena = &mut self.stream_arenas[slot];
            ptr::copy_nonoverlapping(arena.mapped, mapped, used as usize);
            self.device.unmap_memory(arena.memory);
            arena.retired.push((arena.buffer, arena.memory));
            arena.buffer = buffer;
            arena.memory = memory;
            arena.mapped = mapped;
            arena.capacity = capacity;
        }
    }

    pub fn delete_buffer(&mut self, index: usize) {
        self.synchronize_resources();
        unsafe {
            if self.buffers[index].buffer == vk::Buffer::null() {
                return;
            }
            self.device.destroy_buffer(self.buffers[index].buffer, None);
            self.device.free_memory(self.buffers[index].memory, None);
            self.buffers[index].buffer = vk::Buffer::null();
            self.buffers[index].memory = vk::DeviceMemory::null();
            self.buffers[index].current_buffer = vk::Buffer::null();
            self.free_buffers.push(index);
            if self.cur_index_buffer == Some(index) {
                self.cur_index_buffer = None;
            }
            for slot in &mut self.cur_vertex_buffers {
                if *slot == Some(index) {
                    *slot = None;
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Textures
    // -----------------------------------------------------------------

    pub fn new_texture(
        &mut self,
        bytes: Option<&[u8]>,
        params: TextureParams,
        access: TextureAccess,
    ) -> usize {
        unsafe {
            let format = vk_texture_format(params.format, access);
            let depth =
                params.format == TextureFormat::Depth || params.format == TextureFormat::Depth32;
            let opaque = params.format == TextureFormat::RGB8;

            let mip_levels = mip_level_count(
                params.width,
                params.height,
                if params.allocate_mipmaps {
                    MipmapFilterMode::Linear
                } else {
                    params.mipmap_filter
                },
            );

            let usage = if depth {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
            } else {
                // `TRANSFER_SRC` so that `Texture::read_pixels` and
                // `Context::copy_texture` can pull data back out.
                let mut usage = vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC;
                if access == TextureAccess::RenderTarget {
                    usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
                }
                usage
            };

            let (image, memory) = self.create_image(
                params.width,
                params.height,
                format,
                usage,
                mip_levels,
                vk::SampleCountFlags::TYPE_1,
            );
            let mut allocation = ResourceAllocation::new(&self.device, image, memory);
            let attachment_levels = if access == TextureAccess::RenderTarget {
                1
            } else {
                mip_levels
            };
            let view = self.create_image_view(
                image,
                format,
                depth,
                attachment_levels,
                params.format,
                false,
            );
            allocation.views.push(view);
            let sample_view = if opaque || mip_levels != attachment_levels {
                let view =
                    self.create_image_view(image, format, depth, mip_levels, params.format, opaque);
                allocation.views.push(view);
                Some(view)
            } else {
                None
            };
            let sampler = create_sampler(
                &self.device,
                params.min_filter,
                params.wrap,
                params.mipmap_filter,
            );
            allocation.sampler = sampler;

            let module = TextureModule {
                image,
                view,
                sample_view,
                sampler,
                memory,
                width: params.width,
                height: params.height,
                format,
                layout: vk::ImageLayout::UNDEFINED,
                mip_levels,
                filter: params.min_filter,
                wrap: params.wrap,
                mipmap_filter: params.mipmap_filter,
                access,
            };
            let index = self.insert_texture(module);
            allocation.commit();

            match bytes {
                Some(bytes) => {
                    self.upload_texture(
                        index,
                        0,
                        0,
                        params.width as i32,
                        params.height as i32,
                        bytes,
                    );
                    if access == TextureAccess::RenderTarget && !depth {
                        let from = self.textures[index].layout;
                        self.transition_texture(
                            index,
                            from,
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        );
                    }
                }
                None if depth => self.transition_texture(
                    index,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ),
                None => self.transition_texture(
                    index,
                    vk::ImageLayout::UNDEFINED,
                    if access == TextureAccess::RenderTarget {
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    } else {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    },
                ),
            }

            index
        }
    }

    pub(crate) unsafe fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        depth: bool,
        mip_levels: u32,
        texture_format: TextureFormat,
        opaque: bool,
    ) -> vk::ImageView {
        let components = if opaque {
            vk::ComponentMapping {
                r: vk::ComponentSwizzle::R,
                g: vk::ComponentSwizzle::G,
                b: vk::ComponentSwizzle::B,
                a: vk::ComponentSwizzle::ONE,
            }
        } else {
            match texture_format {
                TextureFormat::Alpha => vk::ComponentMapping {
                    r: vk::ComponentSwizzle::ZERO,
                    g: vk::ComponentSwizzle::ZERO,
                    b: vk::ComponentSwizzle::ZERO,
                    a: vk::ComponentSwizzle::R,
                },
                _ => vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                },
            }
        };
        let create_view_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: if depth {
                    vk::ImageAspectFlags::DEPTH
                } else {
                    vk::ImageAspectFlags::COLOR
                },
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };
        self.device
            .create_image_view(&create_view_info, None)
            .expect("Failed to create the image view")
    }

    pub fn set_texture_filter(&mut self, index: usize, filter: FilterMode) {
        if self.textures[index].filter == filter {
            return;
        }
        let wrap = self.textures[index].wrap;
        self.replace_texture_sampler(index, filter, wrap);
    }

    pub fn set_texture_wrap(&mut self, index: usize, wrap: TextureWrap) {
        if self.textures[index].wrap == wrap {
            return;
        }
        let filter = self.textures[index].filter;
        self.replace_texture_sampler(index, filter, wrap);
    }

    fn replace_texture_sampler(&mut self, index: usize, filter: FilterMode, wrap: TextureWrap) {
        self.synchronize_resources();
        let module = &mut self.textures[index];
        let sampler = create_sampler(&self.device, filter, wrap, module.mipmap_filter);
        let old_sampler = std::mem::replace(&mut module.sampler, sampler);
        module.filter = filter;
        module.wrap = wrap;
        if old_sampler != vk::Sampler::null() {
            for info in self.pending.images.iter_mut().flatten() {
                if info.sampler == old_sampler {
                    info.sampler = sampler;
                }
            }
            unsafe { self.device.destroy_sampler(old_sampler, None) };
        }
    }

    pub fn resize_texture(&mut self, index: usize, params: TextureParams, bytes: Option<&[u8]>) {
        self.synchronize_resources();
        let fresh = if params.sample_count > 1 {
            self.new_multisample_texture(params, params.sample_count as u32)
        } else {
            self.new_texture(bytes, params, self.textures[index].access)
        };
        let previous = &self.textures[index];
        let replacement = &self.textures[fresh];
        for image in self.pending.images.iter_mut().flatten() {
            if image.image_view == previous.view || Some(image.image_view) == previous.sample_view {
                image.image_view = replacement.sample_view.unwrap_or(replacement.view);
                image.sampler = replacement.sampler;
            }
        }
        self.destroy_texture(index);
        let module = std::mem::replace(&mut self.textures[fresh], empty_texture_module());
        self.textures[index] = module;
        self.free_textures.push(fresh);
        for pass in self.passes.iter_mut().flatten() {
            for attachment in pass
                .colors
                .iter_mut()
                .chain(pass.depth.iter_mut())
                .chain(pass.resolves.iter_mut().flatten())
            {
                if attachment.texture as usize == index {
                    attachment.width = params.width;
                    attachment.height = params.height;
                }
            }
        }
        if self.default_depth.texture as usize == index {
            self.default_depth.width = params.width;
            self.default_depth.height = params.height;
        }
    }

    pub fn update_texture_part(
        &mut self,
        index: usize,
        x_offset: i32,
        y_offset: i32,
        width: i32,
        height: i32,
        bytes: &[u8],
        _format: TextureFormat,
    ) {
        self.upload_texture(index, x_offset, y_offset, width, height, bytes);
    }

    pub fn delete_texture(&mut self, index: usize) {
        if index >= self.textures.len() || self.textures[index].image == vk::Image::null() {
            return;
        }
        self.synchronize_resources();
        self.pending.sampled.retain(|texture| *texture != index);
        let module = &self.textures[index];
        for image in &mut self.pending.images {
            if image.as_ref().is_some_and(|image| {
                image.image_view == module.view || Some(image.image_view) == module.sample_view
            }) {
                *image = None;
            }
        }
        self.destroy_texture(index);
        self.textures[index] = empty_texture_module();
        self.free_textures.push(index);
    }

    fn insert_texture(&mut self, module: TextureModule) -> usize {
        if let Some(index) = self.free_textures.pop() {
            self.textures[index] = module;
            index
        } else {
            self.textures.push(module);
            self.textures.len() - 1
        }
    }

    pub(crate) fn destroy_texture(&mut self, index: usize) {
        unsafe {
            let Some(module) = self.textures.get(index) else {
                return;
            };
            if module.image == vk::Image::null() {
                return;
            }
            if module.sampler != vk::Sampler::null() {
                self.device.destroy_sampler(module.sampler, None);
            }
            if let Some(sample_view) = module.sample_view {
                self.device.destroy_image_view(sample_view, None);
            }
            self.device.destroy_image_view(module.view, None);
            self.device.destroy_image(module.image, None);
            self.device.free_memory(module.memory, None);
        }
    }

    /// Fill the mip chain by repeatedly halving the previous level.
    pub fn generate_mipmap(&mut self, index: usize) {
        let (image, levels, width, height) = {
            let module = &self.textures[index];
            (module.image, module.mip_levels, module.width, module.height)
        };
        if levels == 1 {
            return;
        }
        let from_layout = self.textures[index].layout;
        self.record_submit(|ctx, cmd| {
            let transition = |base, count, from, to| {
                let (src, dst, mut barrier) =
                    image_barrier(image, vk::ImageAspectFlags::COLOR, from, to);
                barrier.subresource_range.base_mip_level = base;
                barrier.subresource_range.level_count = count;
                unsafe {
                    ctx.device.cmd_pipeline_barrier(
                        cmd,
                        src,
                        dst,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                }
            };
            transition(
                0,
                levels,
                from_layout,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            for level in 1..levels {
                transition(
                    level - 1,
                    1,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                let subresource = vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: level,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let blit = vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        mip_level: level - 1,
                        ..subresource
                    },
                    src_offsets: [
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: (width >> (level - 1)).max(1) as i32,
                            y: (height >> (level - 1)).max(1) as i32,
                            z: 1,
                        },
                    ],
                    dst_subresource: subresource,
                    dst_offsets: [
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: (width >> level).max(1) as i32,
                            y: (height >> level).max(1) as i32,
                            z: 1,
                        },
                    ],
                };
                unsafe {
                    ctx.device.cmd_blit_image(
                        cmd,
                        image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[blit],
                        vk::Filter::LINEAR,
                    );
                }
            }
            transition(
                0,
                levels - 1,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                from_layout,
            );
            transition(
                levels - 1,
                1,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                from_layout,
            );
        });
    }

    fn ensure_readback(&mut self, size: usize) {
        if self.readback_capacity >= size as vk::DeviceSize {
            return;
        }
        self.drain_submissions();
        unsafe {
            self.device.device_wait_idle().unwrap();
            let (buffer, memory, mapped) = self
                .create_mapped_buffer(size as vk::DeviceSize, vk::BufferUsageFlags::TRANSFER_DST);
            if self.readback_buffer != vk::Buffer::null() {
                self.device.unmap_memory(self.readback_memory);
                self.device.destroy_buffer(self.readback_buffer, None);
                self.device.free_memory(self.readback_memory, None);
            }
            self.readback_buffer = buffer;
            self.readback_memory = memory;
            self.readback_mapped = mapped;
            self.readback_capacity = size as vk::DeviceSize;
        }
    }

    pub(crate) fn read_texture_pixels(
        &mut self,
        index: usize,
        width: u32,
        height: u32,
        format: TextureFormat,
        bytes: &mut [u8],
    ) {
        self.drain_submissions();
        unsafe {
            let image = self.textures[index].image;
            let from_layout = self.textures[index].layout;
            let aspect_mask = if is_depth_format(self.textures[index].format) {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };
            // An `RGB8` render target is stored with four channels, so it is
            // read back into a matching buffer and packed afterwards.
            let packed = self.textures[index].format == vk::Format::R8G8B8A8_UNORM
                && format == TextureFormat::RGB8;
            let mut rgba = if packed {
                vec![0; width as usize * height as usize * 4]
            } else {
                Vec::new()
            };
            let (pixels, pixel_len) = if packed {
                (rgba.as_mut_ptr(), rgba.len())
            } else {
                (bytes.as_mut_ptr(), bytes.len())
            };
            self.ensure_readback(pixel_len);
            let staging = self.readback_buffer;

            let region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            };

            self.flush_recording();
            let cmd = self.readback_cmd;
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                .expect("Failed to reset the readback command buffer");
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("Failed to begin the readback command buffer");
            self.barrier(
                cmd,
                image,
                aspect_mask,
                from_layout,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            self.device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging,
                &[region],
            );
            self.barrier(
                cmd,
                image,
                aspect_mask,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                from_layout,
            );
            self.device
                .end_command_buffer(cmd)
                .expect("Failed to end the readback command buffer");

            self.device
                .reset_fences(&[self.readback_fence])
                .expect("Failed to reset the readback fence");
            let command_buffers = [cmd];
            let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
            self.device
                .queue_submit(
                    self.graphics_queue,
                    &[submit_info.build()],
                    self.readback_fence,
                )
                .expect("Failed to submit the readback command buffer");
            self.device
                .wait_for_fences(&[self.readback_fence], true, std::u64::MAX)
                .expect("Failed to wait for the readback fence");
            self.release_frame_resources();

            // Texture targets use a positive-height viewport, so image row zero
            // already matches OpenGL's bottom row and glReadPixels row order.
            ptr::copy_nonoverlapping(self.readback_mapped, pixels, pixel_len);
            if packed {
                let out = bytes.as_mut_ptr();
                for i in 0..width as usize * height as usize {
                    ptr::copy_nonoverlapping(pixels.add(i * 4), out.add(i * 3), 3);
                }
            }
        }
    }

    fn transition_texture(&mut self, index: usize, from: vk::ImageLayout, to: vk::ImageLayout) {
        if from == to {
            return;
        }
        let image = self.textures[index].image;
        let aspect_mask = if is_depth_format(self.textures[index].format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        self.record_submit(|ctx, cmd| ctx.barrier(cmd, image, aspect_mask, from, to));
        self.textures[index].layout = to;
    }

    fn upload_texture(
        &mut self,
        index: usize,
        x_offset: i32,
        y_offset: i32,
        width: i32,
        height: i32,
        bytes: &[u8],
    ) {
        let expanded;
        let bytes = if self.textures[index].format == vk::Format::R8G8B8A8_UNORM
            && bytes.len() == width as usize * height as usize * 3
        {
            expanded = bytes
                .chunks_exact(3)
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], u8::MAX])
                .collect::<Vec<u8>>();
            &expanded[..]
        } else {
            bytes
        };
        unsafe {
            let image = self.textures[index].image;
            let from_layout = self.textures[index].layout;
            let aspect_mask = if is_depth_format(self.textures[index].format) {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };
            let final_layout = if is_depth_format(self.textures[index].format) {
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            } else {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            };

            let (staging, staging_memory) = self.create_buffer(
                bytes.len() as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let allocation = ResourceAllocation::buffer(&self.device, staging, staging_memory);
            let mapped = self
                .device
                .map_memory(
                    staging_memory,
                    0,
                    bytes.len() as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map the staging buffer");
            ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
            self.device.unmap_memory(staging_memory);

            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D {
                    x: x_offset,
                    y: y_offset,
                    z: 0,
                },
                image_extent: vk::Extent3D {
                    width: width as u32,
                    height: height as u32,
                    depth: 1,
                },
                ..Default::default()
            };

            self.record_submit(|ctx, cmd| {
                ctx.barrier(
                    cmd,
                    image,
                    aspect_mask,
                    from_layout,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                ctx.device.cmd_copy_buffer_to_image(
                    cmd,
                    staging,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
                ctx.barrier(
                    cmd,
                    image,
                    aspect_mask,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    final_layout,
                );
            });

            drop(allocation);
            self.textures[index].layout = final_layout;
        }
    }

    /// Record an image layout transition into `cmd`.
    pub fn barrier(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        aspect_mask: vk::ImageAspectFlags,
        from: vk::ImageLayout,
        to: vk::ImageLayout,
    ) {
        if from == to {
            return;
        }
        let (src_stage, dst_stage, barrier) = image_barrier(image, aspect_mask, from, to);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }

    // -----------------------------------------------------------------
    // Render targets
    // -----------------------------------------------------------------

    pub fn new_render_pass(&mut self, color: Texture, depth: Option<Texture>) -> usize {
        self.make_render_pass(color, depth, None, 1)
    }

    /// Render target whose colour attachment holds `samples` samples per pixel
    /// and resolves into `resolve` when the pass ends.
    pub fn new_multisampled_render_pass(
        &mut self,
        color: Texture,
        resolve: Texture,
        samples: u32,
    ) -> usize {
        self.make_render_pass(color, None, Some(resolve), samples)
    }

    pub(crate) fn make_render_pass(
        &mut self,
        color: Texture,
        depth: Option<Texture>,
        resolve: Option<Texture>,
        samples: u32,
    ) -> usize {
        let module = RenderPassModule {
            colors: vec![color],
            depth,
            resolves: vec![resolve],
            samples,
        };
        match self.free_passes.pop() {
            Some(index) => {
                self.passes[index] = Some(module);
                index
            }
            None => {
                self.passes.push(Some(module));
                self.passes.len() - 1
            }
        }
    }

    pub fn delete_render_pass(&mut self, index: usize) {
        self.synchronize_resources();
        if self.passes[index].take().is_some() {
            self.free_passes.push(index);
        }
        if self.cur_target == Some(index) {
            self.cur_target = None;
            self.cur_pass = None;
        }
    }

    /// Colour attachment holding `samples` samples per pixel. It is only
    /// usable through the render pass `new_multisampled_render_pass` makes.
    pub fn new_multisample_texture(&mut self, params: TextureParams, samples: u32) -> usize {
        unsafe {
            let format = vk_texture_format(params.format, TextureAccess::RenderTarget);
            let depth = is_depth_format(format);
            let usage = if depth {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            };
            let sample_count = self.texture_sample_count(params.format, samples);
            let (image, memory) =
                self.create_image(params.width, params.height, format, usage, 1, sample_count);
            let mut allocation = ResourceAllocation::new(&self.device, image, memory);
            let view = self.create_image_view(image, format, depth, 1, params.format, false);
            allocation.views.push(view);

            let module = TextureModule {
                image,
                view,
                sample_view: None,
                sampler: vk::Sampler::null(),
                memory,
                width: params.width,
                height: params.height,
                format,
                layout: vk::ImageLayout::UNDEFINED,
                mip_levels: 1,
                filter: params.min_filter,
                wrap: params.wrap,
                mipmap_filter: MipmapFilterMode::None,
                access: TextureAccess::RenderTarget,
            };
            let index = self.insert_texture(module);
            allocation.commit();
            self.transition_texture(
                index,
                vk::ImageLayout::UNDEFINED,
                if depth {
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                } else {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                },
            );
            index
        }
    }

    pub(crate) fn texture_sample_count(
        &self,
        format: TextureFormat,
        samples: u32,
    ) -> vk::SampleCountFlags {
        if samples <= 1 {
            return vk::SampleCountFlags::TYPE_1;
        }
        let format = vk_texture_format(format, TextureAccess::RenderTarget);
        let usage = if is_depth_format(format) {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        };
        let properties = unsafe {
            self.instance.get_physical_device_image_format_properties(
                self.physical_device,
                format,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                usage,
                vk::ImageCreateFlags::empty(),
            )
        }
        .expect("Texture format does not support render attachments");
        let limits = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        }
        .limits;
        vk_sample_count_flags(
            samples,
            properties.sample_counts
                & limits.framebuffer_color_sample_counts
                & limits.framebuffer_depth_sample_counts,
        )
    }

    /// Copy `src` into `dst`, scaling when the sizes differ.
    pub fn copy_texture(&mut self, src: usize, dst: usize) {
        if src == dst {
            return;
        }
        let (src_image, src_layout, src_size) = {
            let module = &self.textures[src];
            (module.image, module.layout, (module.width, module.height))
        };
        let (dst_image, dst_layout, dst_size) = {
            let module = &self.textures[dst];
            (module.image, module.layout, (module.width, module.height))
        };
        let depth = is_depth_format(self.textures[src].format);
        if depth || is_depth_format(self.textures[dst].format) {
            assert_eq!(
                self.textures[src].format, self.textures[dst].format,
                "Depth copies require matching texture formats"
            );
        }
        let aspect_mask = if depth {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let subresource = vk::ImageSubresourceLayers {
            aspect_mask,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let blit = vk::ImageBlit {
            src_subresource: subresource,
            src_offsets: [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: src_size.0 as i32,
                    y: src_size.1 as i32,
                    z: 1,
                },
            ],
            dst_subresource: subresource,
            dst_offsets: [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: dst_size.0 as i32,
                    y: dst_size.1 as i32,
                    z: 1,
                },
            ],
        };
        self.record_submit(|ctx, cmd| {
            ctx.barrier(
                cmd,
                src_image,
                aspect_mask,
                src_layout,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            ctx.barrier(
                cmd,
                dst_image,
                aspect_mask,
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
                    if depth {
                        vk::Filter::NEAREST
                    } else {
                        vk::Filter::LINEAR
                    },
                );
            }
            ctx.barrier(
                cmd,
                src_image,
                aspect_mask,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_layout,
            );
            ctx.barrier(
                cmd,
                dst_image,
                aspect_mask,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                dst_layout,
            );
        });
    }
}

// ---------------------------------------------------------------------
// Shaders and layouts
// ---------------------------------------------------------------------

pub(super) fn image_barrier(
    image: vk::Image,
    aspect_mask: vk::ImageAspectFlags,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) -> (
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
    vk::ImageMemoryBarrier,
) {
    let (src_stage, src_access) = layout_scope(from);
    let (dst_stage, dst_access) = layout_scope(to);
    let barrier = vk::ImageMemoryBarrier::builder()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .old_layout(from)
        .new_layout(to)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: 1,
        })
        .build();
    (src_stage, dst_stage, barrier)
}

fn layout_scope(layout: vk::ImageLayout) -> (vk::PipelineStageFlags, vk::AccessFlags) {
    match layout {
        vk::ImageLayout::UNDEFINED => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::empty(),
        ),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
        ),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
    }
}

fn widen_indices(data: *const u8, size: usize) -> Vec<u8> {
    let bytes = unsafe { std::slice::from_raw_parts(data, size) };
    let mut wide = Vec::with_capacity(size * 2);
    for byte in bytes {
        wide.extend_from_slice(&(*byte as u16).to_ne_bytes());
    }
    wide
}

fn uniform_alignment(uniform_type: UniformType) -> u32 {
    match uniform_type {
        UniformType::Float1 | UniformType::Int1 => 4,
        UniformType::Float2 | UniformType::Int2 => 8,
        UniformType::Float3
        | UniformType::Int3
        | UniformType::Float4
        | UniformType::Int4
        | UniformType::Mat4 => 16,
    }
}

fn align_to(align: u32, offset: u32) -> u32 {
    offset.div_ceil(align) * align
}

/// Compute the std140 layout of a `UniformBlockLayout` alongside the tight
/// offsets `apply_uniforms_from_bytes` uses to address the values.
///
/// Returns `(tight offset, std140 offset, byte size, stride, count)` per uniform in
/// declaration order, plus the total std140 block size.
pub fn std140_layout(layout: &UniformBlockLayout) -> (Vec<(u32, u32, u32, u32, u32)>, u32) {
    std140_layout_with_arrays(layout, &HashSet::new())
}

fn std140_layout_with_arrays(
    layout: &UniformBlockLayout,
    arrays: &HashSet<String>,
) -> (Vec<(u32, u32, u32, u32, u32)>, u32) {
    let mut entries = Vec::with_capacity(layout.uniforms.len());
    let mut tight = 0u32;
    let mut std140 = 0u32;

    for desc in &layout.uniforms {
        let array = desc.array_count > 1 || arrays.contains(&desc.name);
        let align = uniform_alignment(desc.uniform_type);
        let align = if array { align.max(16) } else { align };
        let size = desc.uniform_type.size() as u32;
        let stride = if array { align_to(16, size) } else { size };
        let count = desc.array_count as u32;
        let total = stride * count;
        let offset = align_to(align, std140);
        entries.push((tight, offset, size, stride, count));
        tight += size * count;
        std140 = offset + total;
    }

    (entries, align_to(16, std140).max(16))
}

// ---------------------------------------------------------------------
// State translation
// ---------------------------------------------------------------------

pub(crate) fn is_depth_format(format: vk::Format) -> bool {
    matches!(format, vk::Format::D16_UNORM | vk::Format::D32_SFLOAT)
}

pub fn vk_format(format: TextureFormat) -> vk::Format {
    match format {
        TextureFormat::RGB8 => vk::Format::R8G8B8_UNORM,
        TextureFormat::RGBA8 => vk::Format::R8G8B8A8_UNORM,
        TextureFormat::Depth => DEPTH_FORMAT,
        TextureFormat::Alpha => vk::Format::R8_UNORM,
        TextureFormat::RGBA16F => vk::Format::R16G16B16A16_SFLOAT,
        TextureFormat::Depth32 => vk::Format::D32_SFLOAT,
    }
}

/// Image format a texture is stored in. Vulkan guarantees no three-channel
/// format as a colour attachment, so an `RGB8` render target is backed by
/// `R8G8B8A8_UNORM` and sampled as opaque.
pub fn vk_texture_format(format: TextureFormat, _access: TextureAccess) -> vk::Format {
    // Three-channel optimal images are optional in Vulkan. Store RGB in RGBA
    // and pack/unpack transfers so the public format still uses three bytes.
    if format == TextureFormat::RGB8 {
        vk::Format::R8G8B8A8_UNORM
    } else {
        vk_format(format)
    }
}

pub fn vk_sampler_filter(filter: FilterMode) -> vk::Filter {
    match filter {
        FilterMode::Linear => vk::Filter::LINEAR,
        FilterMode::Nearest => vk::Filter::NEAREST,
    }
}

pub fn vk_sampler_wrap(wrap: TextureWrap) -> vk::SamplerAddressMode {
    match wrap {
        TextureWrap::Repeat => vk::SamplerAddressMode::REPEAT,
        TextureWrap::Mirror => vk::SamplerAddressMode::MIRRORED_REPEAT,
        TextureWrap::Clamp => vk::SamplerAddressMode::CLAMP_TO_EDGE,
    }
}

/// Largest colour-attachment sample count the device supports that does not
/// exceed `samples`. Falls back to `TYPE_1` when nothing above 1 is available,
/// which is the contract `RenderPass::samples` documents.
pub fn vk_sample_count_flags(
    samples: u32,
    supported: vk::SampleCountFlags,
) -> vk::SampleCountFlags {
    let mut chosen = vk::SampleCountFlags::TYPE_1;
    for candidate in [
        vk::SampleCountFlags::TYPE_64,
        vk::SampleCountFlags::TYPE_32,
        vk::SampleCountFlags::TYPE_16,
        vk::SampleCountFlags::TYPE_8,
        vk::SampleCountFlags::TYPE_4,
        vk::SampleCountFlags::TYPE_2,
    ] {
        if candidate.as_raw() <= samples && supported.contains(candidate) {
            chosen = candidate;
            break;
        }
    }
    chosen
}

pub fn vk_vertex_format(format: VertexFormat) -> vk::Format {
    match format {
        VertexFormat::Float1 => vk::Format::R32_SFLOAT,
        VertexFormat::Float2 => vk::Format::R32G32_SFLOAT,
        VertexFormat::Float3 => vk::Format::R32G32B32_SFLOAT,
        VertexFormat::Float4 => vk::Format::R32G32B32A32_SFLOAT,
        VertexFormat::Byte1 => vk::Format::R8_UINT,
        VertexFormat::Byte2 => vk::Format::R8G8_UINT,
        VertexFormat::Byte3 => vk::Format::R8G8B8_UINT,
        VertexFormat::Byte4 => vk::Format::R8G8B8A8_UINT,
        VertexFormat::Short1 => vk::Format::R16_UINT,
        VertexFormat::Short2 => vk::Format::R16G16_UINT,
        VertexFormat::Short3 => vk::Format::R16G16B16_UINT,
        VertexFormat::Short4 => vk::Format::R16G16B16A16_UINT,
        VertexFormat::Int1 => vk::Format::R32_UINT,
        VertexFormat::Int2 => vk::Format::R32G32_UINT,
        VertexFormat::Int3 => vk::Format::R32G32B32_UINT,
        VertexFormat::Int4 => vk::Format::R32G32B32A32_UINT,
        VertexFormat::Mat4 => vk::Format::R32G32B32A32_SFLOAT,
    }
}

pub fn vk_index_type(index_type: IndexType) -> vk::IndexType {
    // Vulkan core has no 8-bit index type; byte indices are widened to u16 by
    // the caller through `IndexType::for_type`.
    match index_type {
        IndexType::Byte | IndexType::Short => vk::IndexType::UINT16,
        IndexType::Int => vk::IndexType::UINT32,
    }
}

pub fn vk_stencil_op(op: StencilOp) -> vk::StencilOp {
    match op {
        StencilOp::Keep => vk::StencilOp::KEEP,
        StencilOp::Zero => vk::StencilOp::ZERO,
        StencilOp::Replace => vk::StencilOp::REPLACE,
        StencilOp::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
        StencilOp::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
        StencilOp::Invert => vk::StencilOp::INVERT,
        StencilOp::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
        StencilOp::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
    }
}

pub fn vk_stencil_compare(compare: CompareFunc) -> vk::CompareOp {
    match compare {
        CompareFunc::Always => vk::CompareOp::ALWAYS,
        CompareFunc::Never => vk::CompareOp::NEVER,
        CompareFunc::Less => vk::CompareOp::LESS,
        CompareFunc::Equal => vk::CompareOp::EQUAL,
        CompareFunc::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareFunc::Greater => vk::CompareOp::GREATER,
        CompareFunc::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareFunc::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
    }
}

pub fn vk_stencil_face(state: StencilFaceState) -> vk::StencilOpState {
    vk::StencilOpState {
        fail_op: vk_stencil_op(state.fail_op),
        pass_op: vk_stencil_op(state.pass_op),
        depth_fail_op: vk_stencil_op(state.depth_fail_op),
        compare_op: vk_stencil_compare(state.test_func),
        compare_mask: state.test_mask,
        write_mask: state.write_mask,
        reference: state.test_ref as u32,
    }
}

pub fn vk_compare(compare: Comparison) -> vk::CompareOp {
    match compare {
        Comparison::Never => vk::CompareOp::NEVER,
        Comparison::Less => vk::CompareOp::LESS,
        Comparison::Equal => vk::CompareOp::EQUAL,
        Comparison::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
        Comparison::Greater => vk::CompareOp::GREATER,
        Comparison::NotEqual => vk::CompareOp::NOT_EQUAL,
        Comparison::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
        Comparison::Always => vk::CompareOp::ALWAYS,
    }
}

pub fn vk_blend_factor(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SourceAlphaSaturate => vk::BlendFactor::SRC_ALPHA_SATURATE,
        BlendFactor::Value(value) => vk_blend_value(value, false),
        BlendFactor::OneMinusValue(value) => vk_blend_value(value, true),
    }
}

fn vk_blend_value(value: BlendValue, invert: bool) -> vk::BlendFactor {
    match (value, invert) {
        (BlendValue::SourceColor, false) => vk::BlendFactor::SRC_COLOR,
        (BlendValue::SourceColor, true) => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        (BlendValue::SourceAlpha, false) => vk::BlendFactor::SRC_ALPHA,
        (BlendValue::SourceAlpha, true) => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        (BlendValue::DestinationColor, false) => vk::BlendFactor::DST_COLOR,
        (BlendValue::DestinationColor, true) => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        (BlendValue::DestinationAlpha, false) => vk::BlendFactor::DST_ALPHA,
        (BlendValue::DestinationAlpha, true) => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
    }
}

pub fn vk_blend_op(equation: Equation) -> vk::BlendOp {
    match equation {
        Equation::Add => vk::BlendOp::ADD,
        Equation::Subtract => vk::BlendOp::SUBTRACT,
        Equation::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
    }
}

pub fn vk_primitive(primitive: PrimitiveType) -> vk::PrimitiveTopology {
    match primitive {
        PrimitiveType::Triangles => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveType::Lines => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveType::Points => vk::PrimitiveTopology::POINT_LIST,
    }
}

pub fn vk_cull(cull_face: CullFace) -> vk::CullModeFlags {
    match cull_face {
        CullFace::Nothing => vk::CullModeFlags::NONE,
        CullFace::Front => vk::CullModeFlags::FRONT,
        CullFace::Back => vk::CullModeFlags::BACK,
    }
}

/// Vulkan colour write mask for a miniquad `color_write` tuple.
pub fn color_write_mask(color_write: (bool, bool, bool, bool)) -> vk::ColorComponentFlags {
    let (r, g, b, a) = color_write;
    let mut mask = vk::ColorComponentFlags::empty();
    if r {
        mask |= vk::ColorComponentFlags::R;
    }
    if g {
        mask |= vk::ColorComponentFlags::G;
    }
    if b {
        mask |= vk::ColorComponentFlags::B;
    }
    if a {
        mask |= vk::ColorComponentFlags::A;
    }
    mask
}

pub fn vk_front_face(order: FrontFaceOrder) -> vk::FrontFace {
    match order {
        FrontFaceOrder::Clockwise => vk::FrontFace::CLOCKWISE,
        FrontFaceOrder::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
    }
}

pub fn create_sampler(
    device: &Device,
    filter: FilterMode,
    wrap: TextureWrap,
    mipmap_filter: MipmapFilterMode,
) -> vk::Sampler {
    unsafe {
        let base = vk_sampler_filter(filter);
        let (mipmap_mode, max_lod) = match mipmap_filter {
            MipmapFilterMode::None => (vk::SamplerMipmapMode::NEAREST, 0.0),
            MipmapFilterMode::Nearest => (vk::SamplerMipmapMode::NEAREST, vk::LOD_CLAMP_NONE),
            MipmapFilterMode::Linear => (vk::SamplerMipmapMode::LINEAR, vk::LOD_CLAMP_NONE),
        };
        let create_info = vk::SamplerCreateInfo {
            mag_filter: base,
            min_filter: base,
            mipmap_mode,
            address_mode_u: vk_sampler_wrap(wrap),
            address_mode_v: vk_sampler_wrap(wrap),
            address_mode_w: vk_sampler_wrap(wrap),
            mip_lod_bias: 0.0,
            anisotropy_enable: 0,
            max_anisotropy: 0.0,
            compare_enable: 0,
            compare_op: vk::CompareOp::ALWAYS,
            min_lod: 0.0,
            max_lod,
            border_color: vk::BorderColor::FLOAT_OPAQUE_BLACK,
            unnormalized_coordinates: 0,
            ..Default::default()
        };
        device
            .create_sampler(&create_info, None)
            .expect("Failed to create the sampler")
    }
}

/// Number of mip levels a texture of this size gets.
pub fn mip_level_count(width: u32, height: u32, mipmap_filter: MipmapFilterMode) -> u32 {
    if mipmap_filter == MipmapFilterMode::None {
        return 1;
    }
    let mut levels = 1;
    let (mut w, mut h) = (width, height);
    while w > 1 || h > 1 {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        levels += 1;
    }
    levels
}

// ---------------------------------------------------------------------
// Vertex input
// ---------------------------------------------------------------------

/// Reflect the float inputs from the compiled shader, including every matrix
/// column. Legacy GLSL receives unnormalized integer vertex attributes as
/// floats, while native Vulkan shaders can explicitly request integer inputs.
fn vertex_float_locations(words: &[u32]) -> BTreeSet<u32> {
    #[derive(Clone, Copy)]
    enum InputType {
        Float,
        Vector(u32),
        Matrix(u32, u32),
        Pointer(u32),
    }

    fn float_width(id: u32, types: &BTreeMap<u32, InputType>) -> Option<u32> {
        match types.get(&id)? {
            InputType::Float => Some(1),
            InputType::Vector(component) => float_width(*component, types),
            InputType::Matrix(column, count) => {
                float_width(*column, types).map(|width| width * count)
            }
            InputType::Pointer(target) => float_width(*target, types),
        }
    }

    let mut types = BTreeMap::new();
    let mut variables = BTreeMap::new();
    let mut locations = BTreeMap::new();
    let mut cursor = 5;
    while let Some(&header) = words.get(cursor) {
        let count = (header >> 16) as usize;
        let Some(instruction) = words.get(cursor..cursor.saturating_add(count)) else {
            break;
        };
        if count == 0 {
            break;
        }
        match (header & 0xffff, instruction) {
            (22, [_, id, _]) => {
                types.insert(*id, InputType::Float);
            }
            (23, [_, id, component, _]) => {
                types.insert(*id, InputType::Vector(*component));
            }
            (24, [_, id, column, count]) => {
                types.insert(*id, InputType::Matrix(*column, *count));
            }
            (32, [_, id, _, target]) => {
                types.insert(*id, InputType::Pointer(*target));
            }
            // OpVariable with Input storage class.
            (59, [_, type_id, id, 1, ..]) => {
                variables.insert(*id, *type_id);
            }
            // OpDecorate with Location decoration.
            (71, [_, id, 30, location]) => {
                locations.insert(*id, *location);
            }
            _ => {}
        }
        cursor += count;
    }

    let mut floats = BTreeSet::new();
    for (id, type_id) in variables {
        if let (Some(&location), Some(width)) = (locations.get(&id), float_width(type_id, &types)) {
            floats.extend(location..location + width);
        }
    }
    floats
}

fn vertex_format_for_input(format: VertexFormat, float_input: bool) -> vk::Format {
    if float_input {
        // USCALED preserves glVertexAttribPointer(..., GL_FALSE)'s 0..255 /
        // 0..65535 conversion. UNORM would divide by the maximum a second time
        // in shaders such as phira's color0 / 255.0.
        match format {
            VertexFormat::Byte1 => return vk::Format::R8_USCALED,
            VertexFormat::Byte2 => return vk::Format::R8G8_USCALED,
            VertexFormat::Byte3 => return vk::Format::R8G8B8_USCALED,
            VertexFormat::Byte4 => return vk::Format::R8G8B8A8_USCALED,
            VertexFormat::Short1 => return vk::Format::R16_USCALED,
            VertexFormat::Short2 => return vk::Format::R16G16_USCALED,
            VertexFormat::Short3 => return vk::Format::R16G16B16_USCALED,
            VertexFormat::Short4 => return vk::Format::R16G16B16A16_USCALED,
            _ => {}
        }
    }
    vk_vertex_format(format)
}

/// One entry of the vertex input state derived from a `VertexAttribute`.
#[derive(Clone, Copy)]
pub struct VertexInput {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
    pub stride: u32,
    pub input_rate: vk::VertexInputRate,
}

fn vertex_inputs(
    buffer_layout: &[BufferLayout],
    attributes: &[VertexAttribute],
    float_locations: &BTreeSet<u32>,
) -> Vec<VertexInput> {
    #[derive(Clone, Copy, Default)]
    struct BufferCacheData {
        stride: u32,
        offset: u32,
    }

    let mut cache = vec![BufferCacheData::default(); buffer_layout.len()];

    for VertexAttribute {
        format,
        buffer_index,
        ..
    } in attributes
    {
        let layout = &buffer_layout[*buffer_index];
        let entry = &mut cache[*buffer_index];
        if layout.stride == 0 {
            entry.stride += format.size_bytes() as u32;
        } else {
            entry.stride = layout.stride as u32;
        }
    }

    let mut inputs = Vec::with_capacity(attributes.len());
    let mut location = 0u32;

    for VertexAttribute {
        format,
        buffer_index,
        ..
    } in attributes
    {
        let layout = &buffer_layout[*buffer_index];
        let entry = &mut cache[*buffer_index];

        let mut format = *format;
        let mut count = 1u32;
        if format == VertexFormat::Mat4 {
            format = VertexFormat::Float4;
            count = 4;
        }

        let input_rate = match layout.step_func {
            VertexStep::PerVertex => vk::VertexInputRate::VERTEX,
            VertexStep::PerInstance => vk::VertexInputRate::INSTANCE,
        };

        for _ in 0..count {
            inputs.push(VertexInput {
                location,
                binding: *buffer_index as u32,
                format: vertex_format_for_input(format, float_locations.contains(&location)),
                offset: entry.offset,
                stride: entry.stride,
                input_rate,
            });
            location += 1;
            entry.offset += format.size_bytes() as u32;
        }
    }

    inputs
}
