use std::cell::RefCell;

use super::*;

// The Vulkan implementation keeps attachment metadata alongside its indices.
#[derive(Clone, Copy)]
pub struct Texture {
    pub texture: u32,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub wrap: TextureWrap,
    pub filter: FilterMode,
    pub mipmap_filter: MipmapFilterMode,
}

impl Texture {
    pub fn empty() -> Self {
        Self {
            texture: 0,
            width: 0,
            height: 0,
            format: TextureFormat::RGBA8,
            wrap: TextureWrap::Clamp,
            filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::None,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum IndexType {
    Byte,
    Short,
    Int,
}
impl IndexType {
    fn from_size(size: usize) -> Self {
        match size {
            1 => Self::Byte,
            2 => Self::Short,
            4 => Self::Int,
            _ => panic!("Index buffers require u8, u16 or u32 elements"),
        }
    }
    pub fn size(self) -> u8 {
        match self {
            Self::Byte => 1,
            Self::Short => 2,
            Self::Int => 4,
        }
    }
}

pub(crate) struct Bindings {
    pub vertex_buffers: Vec<BufferId>,
    pub index_buffer: BufferId,
    pub images: Vec<Texture>,
}

/// Vulkan implementation of miniquad 0.4's rendering interface.
/// Construct with `window::new_rendering_backend` after selecting Vulkan in `Conf`.
pub struct VulkanBackend {
    // RenderingBackend::draw takes &self. Interior mutability allows command
    // recording without changing the public trait or the other backends.
    context: RefCell<VulkanContext>,
    textures: Vec<Option<TextureParams>>,
    buffers: Vec<usize>,
    wrap_y: Vec<TextureWrap>,
    passes: Vec<Vec<TextureId>>,
}

impl VulkanBackend {
    pub fn new(display: RawDisplayHandle, window: RawWindowHandle, size: (u32, u32)) -> Self {
        let context = VulkanContext::new(display, window, size);
        let textures = vec![None; context.textures.len()];
        let wrap_y = vec![TextureWrap::Clamp; textures.len()];
        Self {
            wrap_y,
            context: RefCell::new(context),
            textures,
            buffers: Vec::new(),
            passes: Vec::new(),
        }
    }

    fn texture_index(&self, texture: TextureId) -> usize {
        match texture.0 {
            TextureIdInner::Managed(index) => {
                assert!(
                    self.textures.get(index).and_then(|p| *p).is_some(),
                    "Texture has been deleted"
                );
                index
            }
            TextureIdInner::Raw(_) => {
                panic!("Vulkan textures require managed image views, samplers and layouts")
            }
        }
    }

    fn attachment(&self, texture: TextureId) -> Texture {
        let index = self.texture_index(texture);
        let params = self.textures[index].unwrap();
        Texture {
            texture: index as u32,
            width: params.width,
            height: params.height,
            format: params.format,
            wrap: params.wrap,
            filter: params.min_filter,
            mipmap_filter: params.mipmap_filter,
        }
    }

    fn update_sampler(&mut self, index: usize) {
        let params = self.textures[index].unwrap();
        self.context.get_mut().set_sampler_params(
            index,
            params.min_filter,
            params.mag_filter,
            params.wrap,
            self.wrap_y[index],
            params.mipmap_filter,
        );
    }
}

impl RenderingBackend for VulkanBackend {
    fn vulkan_context(&mut self) -> Option<&mut VulkanContext> {
        Some(self.context.get_mut())
    }

    fn info(&self) -> ContextInfo {
        ContextInfo {
            backend: Backend::Vulkan,
            gl_version_string: String::new(),
            glsl_support: GlslSupport {
                v100: true,
                v100_ext: true,
                v130: true,
                v150: true,
                v330: true,
                v300es: true,
            },
            features: self.context.borrow().features(),
        }
    }

    fn new_shader(
        &mut self,
        source: ShaderSource,
        meta: ShaderMeta,
    ) -> Result<ShaderId, ShaderError> {
        match source {
            ShaderSource::Glsl { vertex, fragment } => {
                self.context.get_mut().new_shader(vertex, fragment, meta)
            }
            ShaderSource::Msl { .. } => Err(ShaderError::LinkError(
                "Vulkan requires GLSL shader sources".into(),
            )),
        }
    }

    fn new_texture(
        &mut self,
        access: TextureAccess,
        data: TextureSource,
        params: TextureParams,
    ) -> TextureId {
        assert_eq!(
            params.kind,
            TextureKind::Texture2D,
            "Vulkan currently supports 2D textures"
        );
        assert!(params.width > 0 && params.height > 0);
        let bytes = match data {
            TextureSource::Empty => None,
            TextureSource::Bytes(bytes) => Some(bytes),
            TextureSource::Array(levels) => {
                assert_eq!(levels.len(), 1, "2D textures have one layer");
                assert_eq!(
                    levels[0].len(),
                    1,
                    "Explicit mipmap uploads are not supported by Vulkan"
                );
                Some(levels[0][0])
            }
        };
        if let Some(bytes) = bytes {
            assert_eq!(
                bytes.len(),
                params.format.size(params.width, params.height) as usize
            );
        }
        let ctx = self.context.get_mut();
        let mut params = params;
        params.sample_count = ctx
            .texture_sample_count(params.format, params.sample_count.max(1) as u32)
            .as_raw() as i32;
        let index = if params.sample_count > 1 {
            assert_eq!(access, TextureAccess::RenderTarget);
            assert!(bytes.is_none(), "Multisampled textures cannot be uploaded");
            ctx.new_multisample_texture(params, params.sample_count as u32)
        } else {
            ctx.new_texture(bytes, params, access)
        };
        self.textures
            .resize(self.textures.len().max(index + 1), None);
        self.textures[index] = Some(params);
        self.wrap_y
            .resize(self.wrap_y.len().max(index + 1), TextureWrap::Clamp);
        self.wrap_y[index] = params.wrap;
        if params.sample_count <= 1 {
            self.update_sampler(index);
        }
        TextureId(TextureIdInner::Managed(index))
    }

    fn texture_params(&self, texture: TextureId) -> TextureParams {
        self.textures[self.texture_index(texture)].unwrap()
    }
    unsafe fn texture_raw_id(&self, texture: TextureId) -> RawId {
        RawId::Vulkan(
            self.context
                .borrow()
                .texture_image_raw(self.texture_index(texture)),
        )
    }
    fn texture_set_min_filter(
        &mut self,
        texture: TextureId,
        filter: FilterMode,
        mipmap: MipmapFilterMode,
    ) {
        let index = self.texture_index(texture);
        let params = self.textures[index].as_mut().unwrap();
        params.min_filter = filter;
        params.mipmap_filter = mipmap;
        self.update_sampler(index);
    }
    fn texture_set_mag_filter(&mut self, texture: TextureId, filter: FilterMode) {
        let index = self.texture_index(texture);
        let params = self.textures[index].as_mut().unwrap();
        params.mag_filter = filter;
        self.update_sampler(index);
    }
    fn texture_set_wrap(&mut self, texture: TextureId, wrap_x: TextureWrap, wrap_y: TextureWrap) {
        let index = self.texture_index(texture);
        self.textures[index].as_mut().unwrap().wrap = wrap_x;
        self.wrap_y[index] = wrap_y;
        self.update_sampler(index);
    }
    fn texture_generate_mipmaps(&mut self, texture: TextureId) {
        let index = self.texture_index(texture);
        self.context.get_mut().generate_mipmap(index);
    }
    fn texture_resize(
        &mut self,
        texture: TextureId,
        width: u32,
        height: u32,
        bytes: Option<&[u8]>,
    ) {
        let index = self.texture_index(texture);
        let params = self.textures[index].as_mut().unwrap();
        assert!(width > 0 && height > 0);
        if let Some(bytes) = bytes {
            assert_eq!(
                params.sample_count, 1,
                "Multisampled textures cannot be uploaded"
            );
            assert_eq!(bytes.len(), params.format.size(width, height) as usize);
        }
        params.width = width;
        params.height = height;
        let sample_count = params.sample_count;
        self.context.get_mut().resize_texture(index, *params, bytes);
        if sample_count == 1 {
            self.update_sampler(index);
        }
    }
    fn texture_grab_screen(&mut self, texture: TextureId) {
        let index = self.texture_index(texture);
        let params = self.textures[index].unwrap();
        assert_eq!(
            params.sample_count, 1,
            "Window capture requires a single-sampled texture"
        );
        assert!(
            !is_depth_format(vk_format(params.format)),
            "Window capture requires a color texture"
        );
        let size = crate::window::screen_size();
        let ctx = self.context.get_mut();
        ctx.set_window_size(size.0 as u32, size.1 as u32);
        ctx.grab_screen(index);
    }
    fn texture_read_pixels(&mut self, texture: TextureId, bytes: &mut [u8]) {
        let index = self.texture_index(texture);
        let params = self.textures[index].unwrap();
        assert_eq!(
            bytes.len(),
            params.format.size(params.width, params.height) as usize
        );
        self.context.get_mut().read_texture_pixels(
            index,
            params.width,
            params.height,
            params.format,
            bytes,
        );
    }
    fn texture_update_part(
        &mut self,
        texture: TextureId,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        bytes: &[u8],
    ) {
        let index = self.texture_index(texture);
        let params = self.textures[index].unwrap();
        assert!(x >= 0 && y >= 0 && width > 0 && height > 0);
        assert!(
            x as u32 + width as u32 <= params.width && y as u32 + height as u32 <= params.height
        );
        assert_eq!(
            bytes.len(),
            params.format.size(width as u32, height as u32) as usize
        );
        self.context.get_mut().update_texture_part(
            index,
            x,
            y,
            width,
            height,
            bytes,
            params.format,
        );
    }
    fn delete_texture(&mut self, texture: TextureId) {
        let index = self.texture_index(texture);
        self.context.get_mut().delete_texture(index);
        self.textures[index] = None;
    }

    fn new_render_pass_mrt(
        &mut self,
        colors: &[TextureId],
        resolves: Option<&[TextureId]>,
        depth: Option<TextureId>,
    ) -> RenderPass {
        assert!(
            !colors.is_empty() || depth.is_some(),
            "Render pass needs an attachment"
        );
        let attachments: Vec<_> = colors.iter().map(|id| self.attachment(*id)).collect();
        let depth_attachment = depth.map(|id| self.attachment(id));
        let target = colors.first().copied().or(depth).unwrap();
        let params = self.texture_params(target);
        let resolves = match resolves {
            Some(ids) => {
                assert_eq!(ids.len(), colors.len());
                assert!(params.sample_count > 1);
                ids.iter()
                    .zip(colors)
                    .map(|(id, color)| {
                        let r = self.texture_params(*id);
                        let color = self.texture_params(*color);
                        assert_eq!(r.sample_count, 1);
                        assert_eq!(
                            (r.width, r.height, r.format),
                            (color.width, color.height, color.format)
                        );
                        Some(self.attachment(*id))
                    })
                    .collect()
            }
            None => vec![None; colors.len()],
        };
        for id in colors.iter().copied().chain(depth) {
            let p = self.texture_params(id);
            assert_eq!(
                (p.width, p.height, p.sample_count),
                (params.width, params.height, params.sample_count)
            );
        }
        for id in colors {
            assert!(
                !is_depth_format(vk_format(self.texture_params(*id).format)),
                "Color attachments require color formats"
            );
        }
        if let Some(depth) = depth {
            assert!(
                is_depth_format(vk_format(self.texture_params(depth).format)),
                "Depth attachments require depth formats"
            );
        }
        let ctx = self.context.get_mut();
        let index = match ctx.free_passes.pop() {
            Some(index) => index,
            None => {
                ctx.passes.push(None);
                ctx.passes.len() - 1
            }
        };
        ctx.passes[index] = Some(RenderPassModule {
            colors: attachments,
            depth: depth_attachment,
            resolves,
            samples: params.sample_count.max(1) as u32,
        });
        self.passes
            .resize(self.passes.len().max(index + 1), Vec::new());
        self.passes[index] = colors.to_vec();
        RenderPass(index)
    }
    fn render_pass_color_attachments(&self, pass: RenderPass) -> &[TextureId] {
        &self.passes[pass.0]
    }
    fn render_pass_resolve_texture(&self, pass: RenderPass) -> Option<TextureId> {
        let ctx = self.context.borrow();
        let pass = ctx.passes[pass.0].as_ref().expect("Deleted render pass");
        assert_eq!(pass.resolves.len(), 1, "multiple resolve attachments");
        pass.resolves[0].map(|texture| TextureId(TextureIdInner::Managed(texture.texture as usize)))
    }
    fn delete_render_pass(&mut self, pass: RenderPass) {
        self.context.get_mut().delete_render_pass(pass.0);
        self.passes[pass.0] = Vec::new();
    }
    fn new_pipeline(
        &mut self,
        layouts: &[BufferLayout],
        attributes: &[VertexAttribute],
        shader: ShaderId,
        params: PipelineParams,
    ) -> Pipeline {
        self.context
            .get_mut()
            .new_pipeline(layouts, attributes, shader, params)
    }
    fn apply_pipeline(&mut self, pipeline: &Pipeline) {
        self.context.get_mut().apply_pipeline(pipeline);
    }
    fn delete_pipeline(&mut self, pipeline: Pipeline) {
        let ctx = self.context.get_mut();
        ctx.synchronize_resources();
        let module = &mut ctx.pipelines[pipeline.0];
        if module.shader == usize::MAX {
            return;
        }
        for variant in std::mem::take(&mut module.variants) {
            unsafe { ctx.device.destroy_pipeline(variant.pipeline, None) };
        }
        module.inputs = Vec::new();
        module.shader = usize::MAX;
        ctx.free_pipelines.push(pipeline.0);
        if ctx.cur_pipeline == Some(pipeline) {
            ctx.cur_pipeline = None;
            ctx.pending.clear();
        }
    }
    fn delete_shader(&mut self, shader: ShaderId) {
        let ctx = self.context.get_mut();
        ctx.synchronize_resources();
        let module = &mut ctx.shaders[shader.0];
        if module.vertex == vk::ShaderModule::null() {
            return;
        }
        unsafe {
            ctx.device.destroy_shader_module(module.vertex, None);
            ctx.device.destroy_shader_module(module.fragment, None);
            ctx.device
                .destroy_descriptor_set_layout(module.descriptor_set_layout, None);
            ctx.device
                .destroy_pipeline_layout(module.pipeline_layout, None);
        }
        module.vertex = vk::ShaderModule::null();
        module.fragment = vk::ShaderModule::null();
        module.descriptor_set_layout = vk::DescriptorSetLayout::null();
        module.pipeline_layout = vk::PipelineLayout::null();
        module.uniform_offsets = Vec::new();
        module.images = Vec::new();
        module.float_vertex_locations = Default::default();
        ctx.free_shaders.push(shader.0);
    }
    fn new_buffer(
        &mut self,
        type_: BufferType,
        _usage: BufferUsage,
        source: BufferSource,
    ) -> BufferId {
        let (size, element, bytes) = match source {
            BufferSource::Empty { size, element_size } => (size, element_size, None),
            BufferSource::Slice(arg) => (
                arg.size,
                arg.element_size,
                Some(unsafe { std::slice::from_raw_parts(arg.ptr as *const u8, arg.size) }),
            ),
        };
        let index_type = if type_ == BufferType::IndexBuffer {
            Some(IndexType::from_size(element))
        } else {
            None
        };
        let index =
            self.context
                .get_mut()
                .new_buffer(type_, index_type, size, bytes.map(|b| b.as_ptr()));
        self.buffers.resize(self.buffers.len().max(index.0 + 1), 0);
        self.buffers[index.0] = size;
        index
    }
    fn buffer_update(&mut self, buffer: BufferId, source: BufferSource) {
        let arg = match source {
            BufferSource::Slice(arg) => arg,
            BufferSource::Empty { .. } => panic!("buffer_update requires data"),
        };
        assert!(arg.size <= self.buffers[buffer.0]);
        self.context
            .get_mut()
            .update_buffer(buffer.0, arg.ptr as *const u8, arg.size);
    }
    fn buffer_size(&mut self, buffer: BufferId) -> usize {
        self.buffers[buffer.0]
    }
    fn delete_buffer(&mut self, buffer: BufferId) {
        self.context.get_mut().delete_buffer(buffer.0);
    }
    fn apply_viewport(&mut self, x: i32, y: i32, w: i32, h: i32) {
        self.context.get_mut().apply_viewport(x, y, w, h);
    }
    fn apply_scissor_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        self.context.get_mut().apply_scissor_rect(x, y, w, h);
    }
    fn apply_bindings_from_slice(
        &mut self,
        vertices: &[BufferId],
        indices: BufferId,
        images: &[TextureId],
    ) {
        let bindings = Bindings {
            vertex_buffers: vertices.to_vec(),
            index_buffer: indices,
            images: images.iter().map(|id| self.attachment(*id)).collect(),
        };
        self.context.get_mut().apply_bindings(&bindings);
    }
    fn apply_uniforms_from_bytes(&mut self, ptr: *const u8, size: usize) {
        unsafe { self.context.get_mut().apply_uniforms_from_bytes(ptr, size) };
    }
    fn clear(
        &mut self,
        color: Option<(f32, f32, f32, f32)>,
        depth: Option<f32>,
        stencil: Option<i32>,
    ) {
        self.context.get_mut().clear(color, depth, stencil);
    }
    fn begin_default_pass(&mut self, action: PassAction) {
        self.begin_pass(None, action);
    }
    fn begin_pass(&mut self, pass: Option<RenderPass>, action: PassAction) {
        let ctx = self.context.get_mut();
        let size = crate::window::screen_size();
        ctx.set_window_size(size.0 as u32, size.1 as u32);
        ctx.begin_pass(pass, action);
    }
    fn end_render_pass(&mut self) {
        self.context.get_mut().end_render_pass();
    }
    fn commit_frame(&mut self) {
        let ctx = self.context.get_mut();
        if ctx.offscreen {
            ctx.finish_offscreen_frame();
        } else {
            ctx.present();
        }
    }
    fn draw(&self, base: i32, count: i32, instances: i32) {
        self.context.borrow_mut().draw(base, count, instances);
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        self.context.get_mut().cleanup();
    }
}
