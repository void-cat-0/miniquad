use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    mem::MaybeUninit,
    ptr,
};

use ash::{vk, Device};

use crate::graphics::*;

use super::backend::{IndexType, Texture};
use super::cstr;
use super::{
    find_memory_type_index, PassRendering, TextureUploadArena, VulkanContext, DEPTH_FORMAT,
    IMAGE_BINDING_BASE, TEXTURE_UPLOAD_ARENA_SIZE, UNIFORM_BINDING,
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
    pub(super) uniforms: super::UniformState,
    pub images: Vec<String>,
    pub float_vertex_locations: BTreeSet<u32>,
    /// Runtime GLSL inputs bind by source name; SPIR-V assets use pipeline order.
    pub(super) named_vertex_locations: Option<HashMap<String, u32>>,
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
    initial_data: bool,
    stream: StreamData,
}

/// The logical contents of an updated buffer outlive the frame that wrote
/// them. Its transient GPU copy is valid only in the epoch that staged it.
#[derive(Default)]
struct StreamData {
    // BufferSource may be a typed vertex slice with uninitialized padding.
    bytes: Vec<MaybeUninit<u8>>,
    uploaded_epoch: Option<u64>,
}

impl StreamData {
    fn update_prefix(&mut self, bytes: &[MaybeUninit<u8>]) {
        if bytes.is_empty() {
            return;
        }
        if bytes.len() > self.bytes.len() {
            self.bytes.resize(bytes.len(), MaybeUninit::uninit());
        }
        self.bytes[..bytes.len()].copy_from_slice(bytes);
        self.uploaded_epoch = None;
    }

    fn needs_upload(&self, epoch: u64) -> bool {
        !self.bytes.is_empty() && self.uploaded_epoch != Some(epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compatible_sample_count, transfer_destination_layout, vk, MaybeUninit, StreamData,
    };

    // Each byte asserted here came from one of the initialized test arrays.
    fn data(stream: &StreamData) -> Vec<u8> {
        stream
            .bytes
            .iter()
            .map(|byte| unsafe { byte.assume_init() })
            .collect()
    }

    #[test]
    fn updated_data_is_replayed_after_switching_slots_or_restarting() {
        let mut stream = StreamData::default();
        assert!(!stream.needs_upload(1));
        stream.update_prefix(&[1, 2, 3, 4].map(MaybeUninit::new));
        assert!(stream.needs_upload(1));
        stream.uploaded_epoch = Some(1);
        assert!(!stream.needs_upload(1));
        assert!(stream.needs_upload(2));
        stream.uploaded_epoch = Some(2);
        assert!(stream.needs_upload(3));
        assert_eq!(data(&stream), [1, 2, 3, 4]);
    }

    #[test]
    fn a_second_update_preserves_the_previous_suffix_and_needs_a_new_slice() {
        let mut stream = StreamData::default();
        stream.update_prefix(&[1, 2, 3, 4].map(MaybeUninit::new));
        stream.uploaded_epoch = Some(1);
        stream.update_prefix(&[]);
        assert!(!stream.needs_upload(1));
        stream.update_prefix(&[5, 6].map(MaybeUninit::new));
        assert!(stream.needs_upload(1));
        assert_eq!(data(&stream), [5, 6, 3, 4]);
    }

    #[test]
    fn transfer_into_a_new_attachment_uses_a_defined_layout() {
        let undefined = vk::ImageLayout::UNDEFINED;
        let color = vk::ImageAspectFlags::COLOR;
        let depth = vk::ImageAspectFlags::DEPTH;
        assert_eq!(
            transfer_destination_layout(undefined, color),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
        assert_eq!(
            transfer_destination_layout(undefined, depth),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
        let sampled = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        assert_eq!(transfer_destination_layout(sampled, color), sampled);
    }

    #[test]
    fn msaa_uses_the_largest_supported_count_common_to_both_attachments() {
        let color = vk::SampleCountFlags::TYPE_1
            | vk::SampleCountFlags::TYPE_4
            | vk::SampleCountFlags::TYPE_8;
        let depth = vk::SampleCountFlags::TYPE_1
            | vk::SampleCountFlags::TYPE_2
            | vk::SampleCountFlags::TYPE_4;
        assert_eq!(compatible_sample_count(8, color, None), 8);
        assert_eq!(compatible_sample_count(8, color, Some(depth)), 4);
        assert_eq!(compatible_sample_count(2, color, Some(depth)), 1);
        assert_eq!(compatible_sample_count(0, color, None), 1);
    }
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
    source_format: TextureFormat,
    pub(super) usage: vk::ImageUsageFlags,
    pub layout: vk::ImageLayout,
    pub mip_levels: u32,
    pub filter: FilterMode,
    pub mag_filter: FilterMode,
    pub wrap: TextureWrap,
    pub wrap_y: TextureWrap,
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
        source_format: TextureFormat::RGBA8,
        usage: vk::ImageUsageFlags::empty(),
        layout: vk::ImageLayout::UNDEFINED,
        mip_levels: 1,
        filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        wrap: TextureWrap::Clamp,
        wrap_y: TextureWrap::Clamp,
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
        let module = &self.textures[index];
        if module.filter == min_filter
            && module.mag_filter == mag_filter
            && module.wrap == wrap_x
            && module.wrap_y == wrap_y
            && module.mipmap_filter == mipmap_filter
        {
            return;
        }

        self.synchronize_resources();
        let sampler = create_sampler_with_params(
            &self.device,
            min_filter,
            mag_filter,
            wrap_x,
            wrap_y,
            mipmap_filter,
        );
        let module = &mut self.textures[index];
        let old = std::mem::replace(&mut module.sampler, sampler);
        module.filter = min_filter;
        module.mag_filter = mag_filter;
        module.wrap = wrap_x;
        module.wrap_y = wrap_y;
        module.mipmap_filter = mipmap_filter;
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
        self.try_create_buffer(size, usage, required)
            .expect("Failed to allocate a Vulkan buffer")
    }

    unsafe fn try_create_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        required: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), vk::Result> {
        let create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = self.device.create_buffer(&create_info, None)?;

        let requirements = self.device.get_buffer_memory_requirements(buffer);
        let Some(memory_type_index) = find_memory_type_index(
            self.memory_properties,
            requirements.memory_type_bits,
            required,
        ) else {
            self.device.destroy_buffer(buffer, None);
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        };

        let allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = self.device.allocate_memory(&allocate_info, None).map_err(|error| {
            self.device.destroy_buffer(buffer, None);
            error
        })?;
        if let Err(error) = self.device.bind_buffer_memory(buffer, memory, 0) {
            self.device.destroy_buffer(buffer, None);
            self.device.free_memory(memory, None);
            return Err(error);
        }

        Ok((buffer, memory))
    }

    pub(crate) fn create_mapped_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> (vk::Buffer, vk::DeviceMemory, *mut u8) {
        self.try_create_mapped_buffer(size, usage).expect("Failed to map a Vulkan buffer")
    }

    pub(super) fn try_create_mapped_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), vk::Result> {
        unsafe {
            let (buffer, memory) = self.try_create_buffer(
                size,
                usage,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let mapped = self.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|error| {
                    self.device.destroy_buffer(buffer, None);
                    self.device.free_memory(memory, None);
                    error
                })? as *mut u8;
            Ok((buffer, memory, mapped))
        }
    }

    /// Allocate an image and bind memory of the requested properties to it.
    unsafe fn try_create_image(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_levels: u32,
        samples: vk::SampleCountFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory), vk::Result> {
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

        let image = self.device.create_image(&create_info, None)?;

        let requirements = self.device.get_image_memory_requirements(image);
        let Some(memory_type_index) = find_memory_type_index(
            self.memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) else {
            self.device.destroy_image(image, None);
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        };

        let allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = match self.device.allocate_memory(&allocate_info, None) {
            Ok(memory) => memory,
            Err(error) => {
                self.device.destroy_image(image, None);
                return Err(error);
            }
        };
        if let Err(error) = self.device.bind_image_memory(image, memory, 0) {
            self.device.destroy_image(image, None);
            self.device.free_memory(memory, None);
            return Err(error);
        }

        Ok((image, memory))
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
        let (vertex, fragment, layout) =
            super::shader::compile_glsl_with_layout(vertex_source, fragment_source, &meta)?;
        self.new_shader_words(
            &vertex,
            &fragment,
            meta,
            &layout.uniform_arrays,
            Some(&layout.sampler_aliases),
        )
    }

    pub(super) fn new_shader_spirv(
        &mut self,
        vertex: &[u8],
        fragment: &[u8],
        meta: ShaderMeta,
        uniform_arrays: &[&str],
    ) -> Result<ShaderId, ShaderError> {
        fn decode(bytes: &[u8], stage: ShaderType) -> Result<Vec<u32>, ShaderError> {
            let invalid = |error_message| ShaderError::CompilationError {
                shader_type: stage,
                error_message,
            };
            let words = ash::util::read_spv(&mut std::io::Cursor::new(bytes))
                .map_err(|error| invalid(format!("Invalid precompiled SPIR-V: {error}")))?;
            if words.len() < 5 {
                return Err(invalid("Truncated SPIR-V header".into()));
            }
            Ok(words)
        }

        let vertex = decode(vertex, ShaderType::Vertex)?;
        let fragment = decode(fragment, ShaderType::Fragment)?;
        let arrays = uniform_arrays
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<HashSet<_>>();
        self.new_shader_words(&vertex, &fragment, meta, &arrays, None)
    }

    fn new_shader_words(
        &mut self,
        vertex: &[u32],
        fragment: &[u32],
        meta: ShaderMeta,
        arrays: &HashSet<String>,
        glsl_aliases: Option<&HashMap<String, String>>,
    ) -> Result<ShaderId, ShaderError> {
        let (uniform_offsets, uniforms_size) = std140_layout_with_arrays(&meta.uniforms, arrays);
        let (float_vertex_locations, named_vertex_locations) =
            reflect_vertex_inputs(vertex, glsl_aliases);

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
                uniforms: Default::default(),
                images: meta.images,
                float_vertex_locations,
                named_vertex_locations,
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
        let shader_module = &self.shaders[shader.0];
        let inputs = vertex_inputs(
            buffer_layout,
            attributes,
            &shader_module.float_vertex_locations,
            shader_module.named_vertex_locations.as_ref(),
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

    /// The pipeline for the given attachment formats and sample count.
    /// The first draw into a different kind of target compiles a new variant.
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
        let render_pass = match &mut self.rendering {
            PassRendering::Legacy(passes) => {
                passes.pipeline_pass(&self.device, color_formats, depth_format, samples)
            }
            PassRendering::Dynamic(_) => vk::RenderPass::null(),
        };
        let created = self.create_graphics_pipeline(
            module,
            &params,
            samples,
            color_formats,
            depth_format,
            render_pass,
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
        module: &PipelineModule,
        params: &PipelineParams,
        samples: u32,
        color_formats: &[vk::Format],
        depth_format: Option<vk::Format>,
        render_pass: vk::RenderPass,
    ) -> vk::Pipeline {
        let shader = &self.shaders[module.shader];
        let inputs = &module.inputs;
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
                .render_pass(render_pass)
                .subpass(0);
            let mut rendering = vk::PipelineRenderingCreateInfo::builder()
                .color_attachment_formats(color_formats)
                .depth_attachment_format(depth_format.unwrap_or(vk::Format::UNDEFINED));
            let create_info = if render_pass == vk::RenderPass::null() {
                create_info.push_next(&mut rendering)
            } else {
                create_info
            };

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
        let widen = index_type == Some(IndexType::Byte);
        let element_bytes = if widen { 2 } else { 1 };
        assert!(
            size <= isize::MAX as usize / element_bytes,
            "Buffer exceeds the address space"
        );
        let capacity = size * element_bytes;
        unsafe {
            let usage = match buffer_type {
                BufferType::VertexBuffer => vk::BufferUsageFlags::VERTEX_BUFFER,
                BufferType::IndexBuffer => vk::BufferUsageFlags::INDEX_BUFFER,
            };
            let (buffer, memory) = self.create_buffer(
                capacity.max(4) as vk::DeviceSize,
                usage,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let mut allocation = ResourceAllocation::buffer(&self.device, buffer, memory);

            if let Some(data) = data {
                if widen {
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
                widen,
                initial_data: data.is_some(),
                stream: StreamData::default(),
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
        let module = &self.buffers[index];
        assert!(
            module.buffer != vk::Buffer::null(),
            "Buffer has been deleted"
        );
        let logical_size = module.size as usize / if module.widen { 2 } else { 1 };
        assert!(
            size <= logical_size,
            "Buffer update is larger than the buffer"
        );
        if size == 0 {
            return;
        }
        if module.widen {
            let wide = widen_indices(data, size);
            self.update_stream_data(index, wide.as_ptr(), wide.len());
        } else {
            self.update_stream_data(index, data, size);
        }
    }

    fn update_stream_data(&mut self, index: usize, data: *const u8, len: usize) {
        let module = &mut self.buffers[index];
        let size = module.size as usize;
        if module.stream.bytes.is_empty() && module.initial_data && len < size {
            // A partial first update must preserve the initial suffix. This
            // allocation is host coherent and the GPU only ever reads it.
            module.stream.bytes.resize(size, MaybeUninit::uninit());
            unsafe {
                let mapped = self
                    .device
                    .map_memory(module.memory, 0, module.size, vk::MemoryMapFlags::empty())
                    .expect("Failed to map initial buffer data");
                ptr::copy_nonoverlapping(
                    mapped.cast::<MaybeUninit<u8>>().add(len),
                    module.stream.bytes.as_mut_ptr().add(len),
                    size - len,
                );
                self.device.unmap_memory(module.memory);
            }
        }
        let bytes = unsafe { std::slice::from_raw_parts(data.cast::<MaybeUninit<u8>>(), len) };
        module.stream.update_prefix(bytes);
    }

    pub(super) fn stage_buffer_for_draw(&mut self, index: usize) {
        // Updates may arrive in input/resize callbacks before a slot is ready.
        // Draws stage into their own epoch, never into another slot's storage.
        let stream = &self.buffers[index].stream;
        if !stream.needs_upload(self.stream_epoch) {
            return;
        }
        let size = stream.bytes.len();
        let (buffer, offset, dst) = self.reserve_stream_slice(size);
        let module = &mut self.buffers[index];
        unsafe { ptr::copy_nonoverlapping(module.stream.bytes.as_ptr(), dst.cast(), size) };
        module.current_buffer = buffer;
        module.current_offset = offset;
        module.stream.uploaded_epoch = Some(self.stream_epoch);
    }

    fn reserve_stream_slice(&mut self, size: usize) -> (vk::Buffer, vk::DeviceSize, *mut u8) {
        let slot = self.current_frame;
        let alignment = 16;
        let offset = (self.stream_arenas[slot].offset + alignment - 1) & !(alignment - 1);
        let end = offset + size as vk::DeviceSize;
        if end > self.stream_arenas[slot].capacity {
            self.grow_stream_arena(slot, end);
        }
        let arena = &mut self.stream_arenas[slot];
        arena.offset = end;
        (arena.buffer, offset, unsafe {
            arena.mapped.add(offset as usize)
        })
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

    /// Synchronously delete a buffer when using the raw Vulkan context. The
    /// rendering trait uses the managed frame fences instead.
    pub fn delete_buffer(&mut self, index: usize) {
        if let Some((buffer, memory)) = self.remove_buffer(index) {
            self.synchronize_resources();
            unsafe {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
        }
    }

    pub(super) fn retire_buffer(&mut self, index: usize) {
        let Some((buffer, memory)) = self.remove_buffer(index) else {
            return;
        };
        // A draw has already captured the raw VkBuffer, so its logical ID can
        // be reused now. After commit use the slot handed to the worker, not
        // the next (possibly already signaled) slot the CPU will record into.
        let slot = if self.frame_active {
            Some(self.current_frame)
        } else {
            self.last_submitted_slot
        };
        if let Some(slot) = slot {
            self.retired_buffers[slot].push((buffer, memory));
        } else {
            unsafe {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
        }
    }

    fn remove_buffer(&mut self, index: usize) -> Option<(vk::Buffer, vk::DeviceMemory)> {
        let module = &mut self.buffers[index];
        if module.buffer == vk::Buffer::null() {
            return None;
        }
        // Only the base allocation belongs to the buffer. Its most recent
        // draw may instead point into the frame's stream arena.
        let buffer = std::mem::take(&mut module.buffer);
        let memory = std::mem::take(&mut module.memory);
        module.current_buffer = vk::Buffer::null();
        module.stream = StreamData::default();
        self.free_buffers.push(index);
        if self.cur_index_buffer == Some(index) {
            self.cur_index_buffer = None;
        }
        for slot in &mut self.cur_vertex_buffers {
            if *slot == Some(index) {
                *slot = None;
            }
        }
        Some((buffer, memory))
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
        Self::validate_texture_extent(self.texture_size_limit, params);
        self.allocate_texture(bytes, params, access, params.wrap)
    }

    /// Allocate optional render storage without recording an initial transition.
    /// The first clear transitions the image out of UNDEFINED in the frame.
    pub(super) fn try_new_rgba8_render_texture(
        &mut self,
        width: u32,
        height: u32,
    ) -> Option<(usize, TextureParams)> {
        if !self.texture_size_limit.contains(width, height) {
            return None;
        }
        let params = TextureParams {
            kind: TextureKind::Texture2D,
            format: TextureFormat::RGBA8,
            wrap: TextureWrap::Clamp,
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::None,
            width,
            height,
            allocate_mipmaps: false,
            sample_count: 1,
        };
        let format = vk_texture_format(params.format, TextureAccess::RenderTarget);
        let usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC;
        // Optimal RGBA8 is required to support this combination of sampling,
        // linear filtering, color attachment, and transfers at one sample.
        unsafe {
            let limits = self
                .instance
                .get_physical_device_properties(self.physical_device)
                .limits;
            if width > limits.max_framebuffer_width || height > limits.max_framebuffer_height {
                return None;
            }
            let samples = vk::SampleCountFlags::TYPE_1;
            let (image, memory) = self
                .try_create_image(width, height, format, usage, 1, samples)
                .ok()?;
            let mut allocation = ResourceAllocation::new(&self.device, image, memory);
            let view = self
                .try_create_image_view(image, format, false, 1, None)
                .ok()?;
            allocation.views.push(view);
            let sampler = try_create_sampler_with_params(
                &self.device,
                params.min_filter,
                params.mag_filter,
                params.wrap,
                params.wrap,
                params.mipmap_filter,
            )
            .ok()?;
            allocation.sampler = sampler;

            let index = self.insert_texture(TextureModule {
                image,
                view,
                sample_view: None,
                sampler,
                memory,
                width,
                height,
                format,
                source_format: params.format,
                usage,
                layout: vk::ImageLayout::UNDEFINED,
                mip_levels: 1,
                filter: params.min_filter,
                mag_filter: params.mag_filter,
                wrap: params.wrap,
                wrap_y: params.wrap,
                mipmap_filter: params.mipmap_filter,
                access: TextureAccess::RenderTarget,
            });
            allocation.commit();
            Some((index, params))
        }
    }

    pub(super) fn initialize_dummy_texture(&mut self) -> Result<(), vk::Result> {
        self.dummy_texture = self.try_allocate_texture(
            Some(&[0, 0, 0, 255]),
            TextureParams { width: 1, height: 1, ..Default::default() },
            TextureAccess::Static,
            TextureWrap::Clamp,
        )?;
        Ok(())
    }

    fn allocate_texture(
        &mut self,
        bytes: Option<&[u8]>,
        params: TextureParams,
        access: TextureAccess,
        wrap_y: TextureWrap,
    ) -> usize {
        self.try_allocate_texture(bytes, params, access, wrap_y)
            .expect("Failed to allocate and initialize a Vulkan texture")
    }

    fn try_allocate_texture(
        &mut self,
        bytes: Option<&[u8]>,
        params: TextureParams,
        access: TextureAccess,
        wrap_y: TextureWrap,
    ) -> Result<usize, vk::Result> {
        let bytes = bytes.map(|bytes| {
            let len = params
                .format
                .transfer_len(params.width, params.height, bytes.len());
            &bytes[..len]
        });
        unsafe {
            let format = vk_texture_format(params.format, access);
            let depth =
                params.format == TextureFormat::Depth || params.format == TextureFormat::Depth32;

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

            let (image, memory) = self.try_create_image(
                params.width,
                params.height,
                format,
                usage,
                mip_levels,
                vk::SampleCountFlags::TYPE_1,
            )?;
            let mut allocation = ResourceAllocation::new(&self.device, image, memory);
            let attachment_levels = if depth || access == TextureAccess::RenderTarget {
                1
            } else {
                mip_levels
            };
            // Attachment views must use an identity component mapping. The
            // sample view of RGB and Alpha textures may need a different one.
            let separate_sample_view = params.format == TextureFormat::RGB8
                || (params.format == TextureFormat::Alpha && access == TextureAccess::RenderTarget)
                || mip_levels != attachment_levels;
            let view = self.try_create_image_view(
                image,
                format,
                depth,
                attachment_levels,
                (!separate_sample_view).then_some(params.format),
            )?;
            allocation.views.push(view);
            let sample_view = if separate_sample_view {
                let view =
                    self.try_create_image_view(image, format, depth, mip_levels, Some(params.format))?;
                allocation.views.push(view);
                Some(view)
            } else {
                None
            };
            let sampler = try_create_sampler_with_params(
                &self.device,
                params.min_filter,
                params.mag_filter,
                params.wrap,
                wrap_y,
                params.mipmap_filter,
            )?;
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
                source_format: params.format,
                usage,
                layout: vk::ImageLayout::UNDEFINED,
                mip_levels,
                filter: params.min_filter,
                mag_filter: params.mag_filter,
                wrap: params.wrap,
                wrap_y,
                mipmap_filter: params.mipmap_filter,
                access,
            };
            let index = self.insert_texture(module);
            allocation.commit();

            let initialized = (|| {
                if let Some(bytes) = bytes {
                    self.try_upload_texture(
                        index,
                        TextureUpdateRegion {
                            x: 0,
                            y: 0,
                            width: params.width as i32,
                            height: params.height as i32,
                            bytes,
                        },
                    )?;
                }
                let layout = if depth {
                    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                } else if access == TextureAccess::RenderTarget {
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                } else {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                };
                self.transition_texture(index, layout)
            })();
            if let Err(error) = initialized {
                self.destroy_texture(index);
                self.textures[index] = empty_texture_module();
                self.free_textures.push(index);
                return Err(error);
            }
            Ok(index)
        }
    }

    unsafe fn try_create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        depth: bool,
        mip_levels: u32,
        sampled_format: Option<TextureFormat>,
    ) -> Result<vk::ImageView, vk::Result> {
        let components = match sampled_format {
            Some(TextureFormat::RGB8) => vk::ComponentMapping {
                a: vk::ComponentSwizzle::ONE,
                ..Default::default()
            },
            Some(TextureFormat::Alpha) => vk::ComponentMapping {
                r: vk::ComponentSwizzle::ZERO,
                g: vk::ComponentSwizzle::ZERO,
                b: vk::ComponentSwizzle::ZERO,
                a: vk::ComponentSwizzle::R,
            },
            _ => vk::ComponentMapping::default(),
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
        self.device.create_image_view(&create_view_info, None)
    }

    pub fn set_texture_filter(&mut self, index: usize, filter: FilterMode) {
        let module = &self.textures[index];
        self.set_sampler_params(
            index,
            filter,
            filter,
            module.wrap,
            module.wrap_y,
            module.mipmap_filter,
        );
    }

    pub fn set_texture_wrap(&mut self, index: usize, wrap: TextureWrap) {
        let module = &self.textures[index];
        self.set_sampler_params(
            index,
            module.filter,
            module.mag_filter,
            wrap,
            wrap,
            module.mipmap_filter,
        );
    }

    pub fn resize_texture(
        &mut self,
        index: usize,
        mut params: TextureParams,
        bytes: Option<&[u8]>,
    ) {
        params.sample_count = self
            .texture_sample_count(params.format, params.sample_count.max(1) as u32)
            .as_raw() as i32;
        self.resize_texture_with_wrap_y(index, params, bytes, params.wrap);
    }

    pub(crate) fn resize_texture_with_wrap_y(
        &mut self,
        index: usize,
        params: TextureParams,
        bytes: Option<&[u8]>,
        wrap_y: TextureWrap,
    ) {
        Self::validate_texture_extent(self.texture_size_limit, params);
        self.synchronize_resources();
        let fresh = if params.sample_count > 1 {
            assert!(bytes.is_none(), "Multisampled textures cannot be uploaded");
            self.allocate_attachment_texture(
                params,
                vk::SampleCountFlags::from_raw(params.sample_count as u32),
            )
        } else {
            self.allocate_texture(bytes, params, self.textures[index].access, wrap_y)
        };
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
        self.update_texture_regions(
            index,
            &[TextureUpdateRegion {
                x: x_offset,
                y: y_offset,
                width,
                height,
                bytes,
            }],
        );
    }

    pub(super) fn update_texture_regions(
        &mut self,
        index: usize,
        regions: &[TextureUpdateRegion<'_>],
    ) {
        let texture = &self.textures[index];
        assert!(
            texture.usage.contains(vk::ImageUsageFlags::TRANSFER_DST),
            "Texture does not support transfer writes"
        );
        let bpp = texture.source_format.bytes_per_pixel();
        for region in regions {
            let TextureUpdateRegion {
                x,
                y,
                width,
                height,
                bytes,
            } = *region;
            assert!(
                x >= 0
                    && y >= 0
                    && width > 0
                    && height > 0
                    && x as u32 + width as u32 <= texture.width
                    && y as u32 + height as u32 <= texture.height,
                "texture update region is outside the texture",
            );
            let pixels = u64::from(width as u32) * u64::from(height as u32);
            // Compare pixels first; the byte count could overflow on 32-bit.
            assert!(
                pixels == (bytes.len() / bpp) as u64 && bytes.len() % bpp == 0,
                "texture update data must be tightly packed",
            );
        }
        // Regions in one copy have no ordering. Overlapping uploads must run in slice order.
        if texture.source_format == TextureFormat::RGBA8 && !regions_overlap(regions) {
            self.upload_rgba8_regions(index, regions);
        } else {
            for &region in regions {
                self.upload_texture(index, region);
            }
        }
    }

    pub fn delete_texture(&mut self, index: usize) {
        if index >= self.textures.len() || self.textures[index].image == vk::Image::null() {
            return;
        }
        self.synchronize_resources();
        for bound in &mut self.bound_textures {
            if *bound == Some(index) {
                *bound = None;
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
            if let PassRendering::Legacy(passes) = &mut self.rendering {
                passes.release_framebuffers(&self.device, &[module.view]);
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

    pub(super) fn read_texture_pixels(&mut self, index: usize, bytes: &mut [u8]) {
        let texture = &self.textures[index];
        let (width, height) = (texture.width, texture.height);
        let len = texture
            .source_format
            .transfer_len(width, height, bytes.len());
        let bytes = &mut bytes[..len];
        assert!(
            texture.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC),
            "Texture does not support transfer reads"
        );
        let from_layout = texture.layout;
        assert_ne!(
            from_layout,
            vk::ImageLayout::UNDEFINED,
            "Cannot read an uninitialized texture"
        );
        let image = texture.image;
        let aspect_mask = if is_depth_format(texture.format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        // RGB8 storage is RGBA8. Let the array allocation check that the
        // expanded size fits the address space before using it for staging.
        let packed = texture.source_format == TextureFormat::RGB8;
        let mut rgba = if packed {
            vec![[0u8; 4]; len / 3]
        } else {
            Vec::new()
        };
        let pixels = if packed {
            rgba.as_flattened_mut()
        } else {
            &mut *bytes
        };
        let pixel_len = pixels.len();
        self.drain_submissions();
        unsafe {
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
            let host_read = vk::BufferMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(staging)
                .size(pixel_len as vk::DeviceSize);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[*host_read],
                &[],
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
                .wait_for_fences(&[self.readback_fence], true, u64::MAX)
                .expect("Failed to wait for the readback fence");
            self.release_frame_resources();

            // Texture targets use a positive-height viewport, so image row zero
            // already matches OpenGL's bottom row and glReadPixels row order.
            ptr::copy_nonoverlapping(self.readback_mapped, pixels.as_mut_ptr(), pixel_len);
        }
        if packed {
            for (dst, src) in bytes.chunks_exact_mut(3).zip(rgba) {
                dst.copy_from_slice(&src[..3]);
            }
        }
    }

    fn transition_texture(&mut self, index: usize, to: vk::ImageLayout) -> Result<(), vk::Result> {
        let texture = &self.textures[index];
        let from = texture.layout;
        if from == to {
            return Ok(());
        }
        let image = texture.image;
        let aspect = if is_depth_format(texture.format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        self.try_record_submit(|ctx, cmd| ctx.barrier(cmd, image, aspect, from, to))?;
        self.textures[index].layout = to;
        Ok(())
    }

    fn upload_texture(&mut self, index: usize, region: TextureUpdateRegion<'_>) {
        self.try_upload_texture(index, region).expect("Failed to upload Vulkan texture");
    }

    fn try_upload_texture(
        &mut self,
        index: usize,
        region: TextureUpdateRegion<'_>,
    ) -> Result<(), vk::Result> {
        let TextureUpdateRegion {
            x,
            y,
            width,
            height,
            bytes,
        } = region;
        let expanded;
        let bytes = if self.textures[index].source_format == TextureFormat::RGB8 {
            expanded = bytes
                .chunks_exact(3)
                .map(|pixel| [pixel[0], pixel[1], pixel[2], u8::MAX])
                .collect::<Vec<_>>();
            expanded.as_flattened()
        } else {
            bytes
        };
        let aspect = if is_depth_format(self.textures[index].format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let region = texture_upload_copy(x, y, width, height, aspect, 0);
        self.submit_texture_upload(
            index,
            aspect,
            bytes.len(),
            &[region],
            std::iter::once(bytes),
        )
    }

    fn upload_rgba8_regions(&mut self, index: usize, regions: &[TextureUpdateRegion<'_>]) {
        let aspect = vk::ImageAspectFlags::COLOR;
        let mut size = 0;
        let mut copies = Vec::with_capacity(regions.len());
        for region in regions {
            // Each validated RGBA8 slice is a whole number of four-byte texels.
            copies.push(texture_upload_copy(
                region.x,
                region.y,
                region.width,
                region.height,
                aspect,
                size,
            ));
            size += region.bytes.len() as vk::DeviceSize;
        }
        // Rectangles are disjoint and the image's dimensions fit in i32, so
        // the RGBA8 total fits in VkDeviceSize. Mapped pointer offsets are narrower.
        assert!(
            size <= isize::MAX as vk::DeviceSize,
            "Texture staging size exceeds the address space"
        );
        if let Some((staging, mapped, base)) = self.reserve_texture_upload(index, size as usize) {
            for (copy, region) in copies.iter_mut().zip(regions) {
                copy.buffer_offset += base;
                unsafe {
                    ptr::copy_nonoverlapping(
                        region.bytes.as_ptr(),
                        mapped.add(copy.buffer_offset as usize),
                        region.bytes.len(),
                    );
                }
            }
            self.record_frame_texture_upload(index, staging, copies);
        } else {
            self.submit_texture_upload(
                index,
                aspect,
                size as usize,
                &copies,
                regions.iter().map(|r| r.bytes),
            ).expect("Failed to upload Vulkan texture regions");
        }
    }

    fn reserve_texture_upload(
        &mut self,
        index: usize,
        size: usize,
    ) -> Option<(vk::Buffer, *mut u8, vk::DeviceSize)> {
        let texture = &self.textures[index];
        if !self.frame_active
            || !self.recording
            || self.frame_submitted
            || self.offscreen
            || texture.access != TextureAccess::Static
            || texture.mip_levels != 1
            || texture.layout != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            || size as vk::DeviceSize > TEXTURE_UPLOAD_ARENA_SIZE
        {
            return None;
        }

        let slot = self.current_frame;
        if self.texture_upload_arenas[slot].is_none() {
            let (buffer, memory, mapped) = self.create_mapped_buffer(
                TEXTURE_UPLOAD_ARENA_SIZE,
                vk::BufferUsageFlags::TRANSFER_SRC,
            );
            self.texture_upload_arenas[slot] = Some(TextureUploadArena {
                buffer,
                memory,
                mapped,
                offset: 0,
            });
        }
        let arena = self.texture_upload_arenas[slot].as_mut().unwrap();
        let base = arena.offset;
        let end = base + size as vk::DeviceSize;
        if end > TEXTURE_UPLOAD_ARENA_SIZE {
            return None;
        }
        // This path stages whole RGBA8 texels; each region and batch consumes
        // a multiple of four bytes, so every VkBufferImageCopy stays aligned.
        arena.offset = end;
        Some((arena.buffer, arena.mapped, base))
    }

    fn record_frame_texture_upload(
        &mut self,
        index: usize,
        staging: vk::Buffer,
        copies: Vec<vk::BufferImageCopy>,
    ) {
        self.suspend_pass();
        let image = self.textures[index].image;
        let command = self.frame_command_buffer();
        let aspect = vk::ImageAspectFlags::COLOR;
        let shader_read = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let transfer_dst = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        let transfer = vk::PipelineStageFlags::TRANSFER;
        let transfer_access = vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE;
        let (src, dst, mut to_copy) = image_barrier(image, aspect, shader_read, transfer_dst);
        // Also order consecutive uploads of the same texels, even if there
        // was no intervening shader read.
        to_copy.src_access_mask |= transfer_access;
        self.pass_image_barrier(command, src | transfer, dst, to_copy);
        if self.recorder.enabled() {
            self.recorder.copy_buffer_to_image(staging, image, copies);
        } else {
            unsafe {
                self.device.cmd_copy_buffer_to_image(
                    command,
                    staging,
                    image,
                    transfer_dst,
                    &copies,
                );
            }
        }
        let (src, dst, mut to_sample) = image_barrier(image, aspect, transfer_dst, shader_read);
        to_sample.dst_access_mask |= transfer_access;
        self.pass_image_barrier(command, src, dst | transfer, to_sample);
        self.cmd_dirty = true;
    }

    fn submit_texture_upload<'a>(
        &mut self,
        index: usize,
        aspect: vk::ImageAspectFlags,
        size: usize,
        copies: &[vk::BufferImageCopy],
        sources: impl ExactSizeIterator<Item = &'a [u8]>,
    ) -> Result<(), vk::Result> {
        let texture = &self.textures[index];
        let image = texture.image;
        let from_layout = texture.layout;
        let final_layout = if aspect == vk::ImageAspectFlags::DEPTH {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        unsafe {
            let (staging, memory, mapped) = self.try_create_mapped_buffer(
                size as vk::DeviceSize,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;
            let allocation = ResourceAllocation::buffer(&self.device, staging, memory);
            for (copy, bytes) in copies.iter().zip(sources) {
                let offset = copy.buffer_offset as usize;
                ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.add(offset), bytes.len());
            }
            self.device.unmap_memory(memory);
            self.try_record_submit(|ctx, cmd| {
                ctx.barrier(
                    cmd,
                    image,
                    aspect,
                    from_layout,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                ctx.device.cmd_copy_buffer_to_image(
                    cmd,
                    staging,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    copies,
                );
                ctx.barrier(
                    cmd,
                    image,
                    aspect,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    final_layout,
                );
            })?;
            drop(allocation);
        }
        self.textures[index].layout = final_layout;
        Ok(())
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
        // Native render passes and framebuffers don't belong to this ID.
        // End the matching pass so it cannot be reopened with a recycled ID.
        self.end_render_pass_if_target(Some(RenderPass(index)));
        if self.passes[index].take().is_some() {
            self.free_passes.push(index);
        }
    }

    /// Colour attachment holding `samples` samples per pixel. It is only
    /// usable through the render pass `new_multisampled_render_pass` makes.
    pub fn new_multisample_texture(&mut self, params: TextureParams, samples: u32) -> usize {
        let samples = self.texture_sample_count(params.format, samples);
        self.new_attachment_texture(params, samples)
    }

    pub(super) fn new_attachment_texture(
        &mut self,
        params: TextureParams,
        samples: vk::SampleCountFlags,
    ) -> usize {
        Self::validate_texture_extent(self.texture_size_limit, params);
        self.allocate_attachment_texture(params, samples)
    }

    pub(super) fn try_new_swapchain_depth_texture(&mut self) -> Result<usize, vk::Result> {
        // Window depth follows the swapchain extent, not the managed texture
        // cap. It only needs attachment usage, not sampling or transfer.
        let vk::Extent2D { width, height } = self.swapchain_extent;
        let params = TextureParams {
            width,
            height,
            format: TextureFormat::Depth,
            ..Default::default()
        };
        if !self.device_texture_size_limit.contains(width, height) {
            return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
        }
        self.try_allocate_attachment_texture(params, vk::SampleCountFlags::TYPE_1)
    }

    fn allocate_attachment_texture(
        &mut self,
        params: TextureParams,
        samples: vk::SampleCountFlags,
    ) -> usize {
        self.try_allocate_attachment_texture(params, samples)
            .expect("Failed to allocate a Vulkan render attachment")
    }

    fn try_allocate_attachment_texture(
        &mut self,
        params: TextureParams,
        samples: vk::SampleCountFlags,
    ) -> Result<usize, vk::Result> {
        unsafe {
            let format = vk_texture_format(params.format, TextureAccess::RenderTarget);
            let depth = is_depth_format(format);
            let usage = if depth {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            };
            let (image, memory) =
                self.try_create_image(params.width, params.height, format, usage, 1, samples)?;
            let mut allocation = ResourceAllocation::new(&self.device, image, memory);
            let view = self.try_create_image_view(image, format, depth, 1, None)?;
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
                source_format: params.format,
                usage,
                layout: vk::ImageLayout::UNDEFINED,
                mip_levels: 1,
                filter: params.min_filter,
                mag_filter: params.mag_filter,
                wrap: params.wrap,
                wrap_y: params.wrap,
                mipmap_filter: MipmapFilterMode::None,
                access: TextureAccess::RenderTarget,
            };
            let index = self.insert_texture(module);
            allocation.commit();
            // The producing pass performs the first layout transition.
            Ok(index)
        }
    }

    fn validate_texture_extent(limit: TextureSizeLimit, params: TextureParams) {
        assert_eq!(
            params.kind,
            TextureKind::Texture2D,
            "Vulkan currently supports 2D textures"
        );
        assert!(
            limit.contains(params.width, params.height),
            "texture dimensions {}x{} are outside the Vulkan texture size limit of {}",
            params.width,
            params.height,
            limit.max(),
        );
    }

    pub(crate) fn texture_sample_count(
        &self,
        format: TextureFormat,
        samples: u32,
    ) -> vk::SampleCountFlags {
        if samples <= 1 {
            return vk::SampleCountFlags::TYPE_1;
        }
        vk_sample_count_flags(samples, self.attachment_sample_counts(format))
    }

    pub(super) fn render_target_sample_count(
        &self,
        color: TextureFormat,
        depth: Option<TextureFormat>,
        requested: i32,
    ) -> i32 {
        if requested <= 1 {
            return 1;
        }
        let color_counts = self.attachment_sample_counts(color);
        let depth_counts = depth.map(|format| self.attachment_sample_counts(format));
        compatible_sample_count(requested, color_counts, depth_counts)
    }

    fn attachment_sample_counts(&self, format: TextureFormat) -> vk::SampleCountFlags {
        let format = vk_texture_format(format, TextureAccess::RenderTarget);
        let depth = is_depth_format(format);
        let usage = if depth {
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
        let framebuffer_counts = if depth {
            unsafe {
                self.instance
                    .get_physical_device_properties(self.physical_device)
                    .limits
                    .framebuffer_depth_sample_counts
            }
        } else {
            self.color_sample_counts
        };
        properties.sample_counts & framebuffer_counts
    }

    pub(super) fn texture_has_color_attachment_usage(&self, index: usize) -> bool {
        self.textures.get(index).is_some_and(|texture| {
            texture
                .usage
                .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        })
    }

    /// Copy `src` into `dst`, scaling when the sizes differ.
    /// The source texture must already contain data.
    pub fn copy_texture(&mut self, src: usize, dst: usize) {
        if src == dst {
            return;
        }
        let source = &self.textures[src];
        let destination = &self.textures[dst];
        assert!(
            source.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC),
            "Source texture does not support transfer reads"
        );
        assert!(
            destination
                .usage
                .contains(vk::ImageUsageFlags::TRANSFER_DST),
            "Destination texture does not support transfer writes"
        );
        let src_layout = source.layout;
        assert_ne!(
            src_layout,
            vk::ImageLayout::UNDEFINED,
            "Cannot copy from an uninitialized texture"
        );
        let depth = is_depth_format(source.format);
        if depth || is_depth_format(destination.format) {
            assert_eq!(
                source.format, destination.format,
                "Depth copies require matching texture formats"
            );
        }
        let aspect_mask = if depth {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let src_image = source.image;
        let src_size = (source.width, source.height);
        let dst_image = destination.image;
        let dst_size = (destination.width, destination.height);
        let dst_layout = destination.layout;
        let dst_final_layout = transfer_destination_layout(dst_layout, aspect_mask);
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
                dst_final_layout,
            );
        });
        self.textures[dst].layout = dst_final_layout;
    }
}

// ---------------------------------------------------------------------
// Shaders and layouts
// ---------------------------------------------------------------------

/// Keep a transfer destination's layout when it already has one. The lazily
/// allocated render textures start UNDEFINED, which is only an old layout:
/// preserve their newly written contents in the appropriate attachment layout.
pub(super) fn transfer_destination_layout(
    layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
) -> vk::ImageLayout {
    if layout != vk::ImageLayout::UNDEFINED {
        layout
    } else if aspect.contains(vk::ImageAspectFlags::COLOR) {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    } else {
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    }
}

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
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
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
/// offsets `apply_uniforms_from_slice` uses to address the values.
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

// Each edge adds two nonnegative i32 values and fits in u32.
fn regions_overlap(regions: &[TextureUpdateRegion<'_>]) -> bool {
    regions.iter().enumerate().any(|(i, a)| {
        let (x, y) = (a.x as u32, a.y as u32);
        let (right, bottom) = (x + a.width as u32, y + a.height as u32);
        regions[..i].iter().any(|b| {
            let (bx, by) = (b.x as u32, b.y as u32);
            x < bx + b.width as u32 && bx < right && y < by + b.height as u32 && by < bottom
        })
    })
}

fn texture_upload_copy(
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    aspect_mask: vk::ImageAspectFlags,
    byte_offset: vk::DeviceSize,
) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: byte_offset,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask,
            layer_count: 1,
            ..Default::default()
        },
        image_offset: vk::Offset3D { x, y, z: 0 },
        image_extent: vk::Extent3D {
            width: width as u32,
            height: height as u32,
            depth: 1,
        },
        ..Default::default()
    }
}

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

fn compatible_sample_count(
    requested: i32,
    color: vk::SampleCountFlags,
    depth: Option<vk::SampleCountFlags>,
) -> i32 {
    let shared = depth.map_or(color, |depth| color & depth);
    vk_sample_count_flags(requested.max(1) as u32, shared).as_raw() as i32
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
    create_sampler_with_params(device, filter, filter, wrap, wrap, mipmap_filter)
}

fn create_sampler_with_params(
    device: &Device,
    min_filter: FilterMode,
    mag_filter: FilterMode,
    wrap_x: TextureWrap,
    wrap_y: TextureWrap,
    mipmap_filter: MipmapFilterMode,
) -> vk::Sampler {
    try_create_sampler_with_params(
        device,
        min_filter,
        mag_filter,
        wrap_x,
        wrap_y,
        mipmap_filter,
    )
    .expect("Failed to create the sampler")
}

fn try_create_sampler_with_params(
    device: &Device,
    min_filter: FilterMode,
    mag_filter: FilterMode,
    wrap_x: TextureWrap,
    wrap_y: TextureWrap,
    mipmap_filter: MipmapFilterMode,
) -> Result<vk::Sampler, vk::Result> {
    unsafe {
        let (mipmap_mode, max_lod) = match mipmap_filter {
            MipmapFilterMode::None => (vk::SamplerMipmapMode::NEAREST, 0.0),
            MipmapFilterMode::Nearest => (vk::SamplerMipmapMode::NEAREST, vk::LOD_CLAMP_NONE),
            MipmapFilterMode::Linear => (vk::SamplerMipmapMode::LINEAR, vk::LOD_CLAMP_NONE),
        };
        let create_info = vk::SamplerCreateInfo {
            mag_filter: vk_sampler_filter(mag_filter),
            min_filter: vk_sampler_filter(min_filter),
            mipmap_mode,
            address_mode_u: vk_sampler_wrap(wrap_x),
            address_mode_v: vk_sampler_wrap(wrap_y),
            address_mode_w: vk_sampler_wrap(wrap_x),
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
        device.create_sampler(&create_info, None)
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
fn reflect_vertex_inputs(
    words: &[u32],
    glsl_aliases: Option<&HashMap<String, String>>,
) -> (BTreeSet<u32>, Option<HashMap<String, u32>>) {
    fn literal_string(words: &[u32]) -> Option<String> {
        let mut bytes = Vec::new();
        for byte in words.iter().flat_map(|word| word.to_le_bytes()) {
            if byte == 0 {
                return String::from_utf8(bytes).ok();
            }
            bytes.push(byte);
        }
        None
    }

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
    let mut names = HashMap::new();
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
            // OpName. Only source GLSL needs names; ignore them in assets.
            (5, [_, id, encoded @ ..]) if glsl_aliases.is_some() => {
                if let Some(name) = literal_string(encoded) {
                    names.insert(*id, name);
                }
            }
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
    let mut named = glsl_aliases.map(|_| HashMap::new());
    let source_names: HashMap<&str, &str> = glsl_aliases
        .into_iter()
        .flatten()
        .map(|(name, alias)| (alias.as_str(), name.as_str()))
        .collect();
    for (id, type_id) in variables {
        let Some(&location) = locations.get(&id) else {
            continue; // built-in, not a user-provided vertex attribute
        };
        if let Some(width) = float_width(type_id, &types) {
            floats.extend(location..location + width);
        }
        if let (Some(named), Some(name)) = (&mut named, names.remove(&id)) {
            let name = source_names
                .get(name.as_str())
                .map_or(name, |name| (*name).to_owned());
            named.insert(name, location);
        }
    }
    (floats, named)
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
    named_locations: Option<&HashMap<String, u32>>,
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
    let mut ordinal = 0u32;

    for VertexAttribute {
        name,
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

        let location = match named_locations {
            Some(locations) => locations.get(*name).copied(),
            None => Some(ordinal),
        };
        for column in 0..count {
            if let Some(location) = location.map(|location| location + column) {
                inputs.push(VertexInput {
                    location,
                    binding: *buffer_index as u32,
                    format: vertex_format_for_input(format, float_locations.contains(&location)),
                    offset: entry.offset,
                    stride: entry.stride,
                    input_rate,
                });
            }
            // An attribute omitted by the shader still occupies bytes in
            // the user's vertex/instance buffer layout.
            entry.offset += format.size_bytes() as u32;
        }
        ordinal += count;
    }

    inputs
}

#[cfg(test)]
mod vertex_input_tests {
    use super::*;

    const FRAGMENT: &str = "#version 100
        precision mediump float;
        varying vec4 v_color;
        void main() { gl_FragColor = v_color; }";

    fn compile(
        vertex: &str,
        fragment: &str,
        images: &[&str],
    ) -> (Vec<u32>, super::super::shader::GlslLayout) {
        let meta = ShaderMeta {
            images: images.iter().map(|image| (*image).to_owned()).collect(),
            uniforms: UniformBlockLayout { uniforms: vec![] },
        };
        let (vertex, _, layout) =
            super::super::shader::compile_glsl_with_layout(vertex, fragment, &meta).unwrap();
        (vertex, layout)
    }

    #[test]
    fn glsl_attribute_order_does_not_change_vertex_memory_layout() {
        let (words, layout) = compile(
            "#version 100
            attribute vec4 color0;
            attribute vec3 position;
            varying vec4 v_color;
            void main() { gl_Position = vec4(position, 1.0); v_color = color0; }",
            FRAGMENT,
            &[],
        );
        let (floats, named) = reflect_vertex_inputs(&words, Some(&layout.sampler_aliases));
        let named = named.unwrap();
        assert_eq!(named.len(), 2);
        let attributes = [
            VertexAttribute::new("position", VertexFormat::Float3),
            VertexAttribute::new("texcoord", VertexFormat::Float2),
            VertexAttribute::new("color0", VertexFormat::Byte4),
        ];
        let inputs = vertex_inputs(
            &[BufferLayout::default()],
            &attributes,
            &floats,
            Some(&named),
        );
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].location, named["position"]);
        assert_eq!(inputs[1].location, named["color0"]);
        assert_eq!(inputs[1].format, vk::Format::R8G8B8A8_USCALED);
        assert_eq!(inputs[1].offset, 20);
        assert_eq!(inputs[1].stride, 24);

        let (_, unreflected) = reflect_vertex_inputs(&words, None);
        assert!(unreflected.is_none());
        let ordinal = vertex_inputs(&[BufferLayout::default()], &attributes, &floats, None);
        assert_eq!(
            ordinal.iter().map(|i| i.location).collect::<Vec<_>>(),
            [0, 1, 2]
        );
    }

    #[test]
    fn glsl_reflection_uses_explicit_locations_and_matrix_columns() {
        let (words, layout) = compile(
            "#version 330
            layout(location = 6) in mat4 transform;
            layout(location = 3) in vec3 position;
            layout(location = 0) in uvec4 color0;
            out vec4 v_color;
            void main() { gl_Position = transform * vec4(position, 1.0); v_color = vec4(color0); }",
            FRAGMENT,
            &[],
        );
        let (floats, named) = reflect_vertex_inputs(&words, Some(&layout.sampler_aliases));
        let named = named.unwrap();
        assert_eq!(named["transform"], 6);
        assert_eq!(named["position"], 3);
        assert_eq!(named["color0"], 0);
        assert_eq!(floats.iter().copied().collect::<Vec<_>>(), [3, 6, 7, 8, 9]);
        let attributes = [
            VertexAttribute::new("position", VertexFormat::Float3),
            VertexAttribute::new("unused", VertexFormat::Float2),
            VertexAttribute::new("color0", VertexFormat::Byte4),
            VertexAttribute::with_buffer("transform", VertexFormat::Mat4, 1),
        ];
        let layouts = [
            BufferLayout::default(),
            BufferLayout {
                step_func: VertexStep::PerInstance,
                ..Default::default()
            },
        ];
        let inputs = vertex_inputs(&layouts, &attributes, &floats, Some(&named));
        assert_eq!(inputs[1].format, vk::Format::R8G8B8A8_UINT);
        assert_eq!(inputs[1].offset, 20);
        for (column, input) in inputs[2..].iter().enumerate() {
            assert_eq!(input.location, 6 + column as u32);
            assert_eq!(input.offset, 16 * column as u32);
            assert_eq!(input.stride, 64);
            assert_eq!(input.binding, 1);
            assert_eq!(input.input_rate, vk::VertexInputRate::INSTANCE);
        }
        assert_eq!(inputs.len(), 6);
    }

    #[test]
    fn glsl_reflection_restores_legacy_names_rewritten_by_translator() {
        let (words, layout) = compile(
            "#version 100
            attribute vec4 texture;
            varying vec4 v_color;
            void main() { gl_Position = texture; v_color = vec4(1.0); }",
            "#version 100
            precision mediump float;
            varying vec4 v_color;
            uniform sampler2D texture;
            void main() { gl_FragColor = v_color * texture2D(texture, vec2(0.5)); }",
            &["texture"],
        );
        let (_, named) = reflect_vertex_inputs(&words, Some(&layout.sampler_aliases));
        let named = named.unwrap();
        assert_eq!(named.len(), 1);
        assert_eq!(named["texture"], 0);
    }
}
