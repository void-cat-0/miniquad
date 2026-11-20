use std::ffi::CString;

use crate::{window, ResourceManager};

mod cache;

use super::*;
use cache::*;

/// Raw OpenGL bindings
/// Highly unsafe, some of the functions could be missing due to incompatible GL version
/// or all of them might be missing alltogether if rendering context is not a GL one.
pub mod raw_gl {
    use super::*;

    #[doc(inline)]
    pub use crate::native::gl::*;

    /// Default GL transfer tuple. Legacy storage and format support depend on the context.
    pub fn texture_format_into_gl(format: TextureFormat) -> (GLenum, GLenum, GLenum) {
        format.into()
    }
}

#[derive(Clone, Copy, Debug)]
struct Buffer {
    gl_buf: GLuint,
    buffer_type: BufferType,
    size: usize,
    // Dimension of the indices for this buffer,
    // used only as a type argument for glDrawElements and can be
    // 1, 2 or 4
    index_type: Option<u32>,
}

#[derive(Debug)]
struct ShaderUniform {
    gl_loc: UniformLocation,
    uniform_type: UniformType,
    array_count: i32,
}

struct ShaderInternal {
    program: GLuint,
    images: Vec<ShaderImage>,
    uniforms: Vec<ShaderUniform>,
}

#[derive(Clone, Copy, Debug)]
enum TextureOrRenderbuffer {
    Texture(GLuint),
    Renderbuffer(GLuint),
}
impl TextureOrRenderbuffer {
    fn texture(&self) -> Option<GLuint> {
        match self {
            TextureOrRenderbuffer::Texture(id) => Some(*id),
            _ => None,
        }
    }
    fn renderbuffer(&self) -> Option<GLuint> {
        match self {
            TextureOrRenderbuffer::Renderbuffer(id) => Some(*id),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Texture {
    raw: TextureOrRenderbuffer,
    params: TextureParams,
}

/// Converts from TextureFormat to (internal_format, format, pixel_type)
impl From<TextureFormat> for (GLenum, GLenum, GLenum) {
    fn from(format: TextureFormat) -> Self {
        match format {
            TextureFormat::RGB8 => (GL_RGB, GL_RGB, GL_UNSIGNED_BYTE),
            TextureFormat::RGBA8 => (GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE),
            TextureFormat::RGBA16F => (GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT),
            TextureFormat::Depth => (GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT),
            TextureFormat::Depth32 => (GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT),
            #[cfg(target_arch = "wasm32")]
            TextureFormat::Alpha => (GL_ALPHA, GL_ALPHA, GL_UNSIGNED_BYTE),
            #[cfg(not(target_arch = "wasm32"))]
            TextureFormat::Alpha => (GL_R8, GL_RED, GL_UNSIGNED_BYTE), // texture updates will swizzle Red -> Alpha to match WASM
        }
    }
}

impl From<TextureKind> for GLuint {
    fn from(kind: TextureKind) -> GLuint {
        match kind {
            TextureKind::Texture2D => GL_TEXTURE_2D,
            TextureKind::CubeMap => GL_TEXTURE_CUBE_MAP,
        }
    }
}
impl From<Equation> for GLenum {
    fn from(eq: Equation) -> Self {
        match eq {
            Equation::Add => GL_FUNC_ADD,
            Equation::Subtract => GL_FUNC_SUBTRACT,
            Equation::ReverseSubtract => GL_FUNC_REVERSE_SUBTRACT,
        }
    }
}

impl From<BlendFactor> for GLenum {
    fn from(factor: BlendFactor) -> GLenum {
        match factor {
            BlendFactor::Zero => GL_ZERO,
            BlendFactor::One => GL_ONE,
            BlendFactor::Value(BlendValue::SourceColor) => GL_SRC_COLOR,
            BlendFactor::Value(BlendValue::SourceAlpha) => GL_SRC_ALPHA,
            BlendFactor::Value(BlendValue::DestinationColor) => GL_DST_COLOR,
            BlendFactor::Value(BlendValue::DestinationAlpha) => GL_DST_ALPHA,
            BlendFactor::OneMinusValue(BlendValue::SourceColor) => GL_ONE_MINUS_SRC_COLOR,
            BlendFactor::OneMinusValue(BlendValue::SourceAlpha) => GL_ONE_MINUS_SRC_ALPHA,
            BlendFactor::OneMinusValue(BlendValue::DestinationColor) => GL_ONE_MINUS_DST_COLOR,
            BlendFactor::OneMinusValue(BlendValue::DestinationAlpha) => GL_ONE_MINUS_DST_ALPHA,
            BlendFactor::SourceAlphaSaturate => GL_SRC_ALPHA_SATURATE,
        }
    }
}

impl From<StencilOp> for GLenum {
    fn from(op: StencilOp) -> Self {
        match op {
            StencilOp::Keep => GL_KEEP,
            StencilOp::Zero => GL_ZERO,
            StencilOp::Replace => GL_REPLACE,
            StencilOp::IncrementClamp => GL_INCR,
            StencilOp::DecrementClamp => GL_DECR,
            StencilOp::Invert => GL_INVERT,
            StencilOp::IncrementWrap => GL_INCR_WRAP,
            StencilOp::DecrementWrap => GL_DECR_WRAP,
        }
    }
}

impl From<CompareFunc> for GLenum {
    fn from(cf: CompareFunc) -> Self {
        match cf {
            CompareFunc::Always => GL_ALWAYS,
            CompareFunc::Never => GL_NEVER,
            CompareFunc::Less => GL_LESS,
            CompareFunc::Equal => GL_EQUAL,
            CompareFunc::LessOrEqual => GL_LEQUAL,
            CompareFunc::Greater => GL_GREATER,
            CompareFunc::NotEqual => GL_NOTEQUAL,
            CompareFunc::GreaterOrEqual => GL_GEQUAL,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GlApi {
    Desktop,
    Es,
    Web,
}

struct TransferCaps {
    api: GlApi,
    split_framebuffers: bool,
    pack_subimage: bool,
    unpack_subimage: bool,
    pixel_buffers: bool,
    max_mip_level: bool,
    alpha_red: Option<GLenum>,
    alpha_swizzle: bool,
    half_float: Option<GLenum>,
    legacy_depth: bool,
    legacy_depth_upload: bool,
    sized_depth: bool,
}

impl TransferCaps {
    fn new(version: &str) -> Self {
        let (api, version) = if let Some(rest) = version.strip_prefix("OpenGL ES") {
            (GlApi::Es, rest)
        } else if let Some(rest) = version.strip_prefix("WebGL ") {
            (GlApi::Web, rest)
        } else {
            (GlApi::Desktop, version)
        };
        let version = version
            .split_ascii_whitespace()
            .find_map(|word| {
                let mut numbers = word.split('.');
                Some((
                    numbers.next()?.parse::<u32>().ok()?,
                    numbers.next()?.parse::<u32>().ok()?,
                ))
            })
            .unwrap_or_default();
        let desktop = api == GlApi::Desktop;
        let gles = api == GlApi::Es;
        let modern = version.0 >= if api == GlApi::Web { 2 } else { 3 };
        // Only native legacy contexts need extension fallbacks. Desktop GL 3.0+
        // uses indexed extension queries; GL_EXTENSIONS is invalid on core profiles.
        #[cfg(not(target_arch = "wasm32"))]
        let extensions = if (desktop && version < (3, 3)) || (gles && !modern) {
            unsafe { transfer_extensions(modern) }
        } else {
            Vec::new()
        };
        #[cfg(not(target_arch = "wasm32"))]
        let has = |ext: &str| extensions.iter().any(|name| name == ext);
        #[cfg(target_arch = "wasm32")]
        let has = |_: &str| false;
        // Red-channel storage does not require swizzling. Keep it available for
        // readback on desktop core GL and GLES2 with EXT_texture_rg, even when
        // the GPU cannot expose the stored value in the sampled alpha channel.
        let alpha_red = if (desktop || gles) && modern || (desktop && has("GL_ARB_texture_rg")) {
            Some(GL_R8)
        } else if has("GL_EXT_texture_rg") {
            Some(if gles { GL_RED } else { GL_R8 })
        } else {
            None
        };
        let alpha_swizzle = alpha_red.is_some()
            && ((gles && modern)
                || (desktop
                    && (version >= (3, 3)
                        || has("GL_ARB_texture_swizzle")
                        || has("GL_EXT_texture_swizzle"))));
        let half_float = (modern
            || (desktop && has("GL_ARB_half_float_pixel") && has("GL_ARB_texture_float")))
        .then_some(GL_HALF_FLOAT);
        #[cfg(not(target_arch = "wasm32"))]
        let half_float = half_float
            .or_else(|| (gles && has("GL_OES_texture_half_float")).then_some(GL_HALF_FLOAT_OES));
        Self {
            api,
            split_framebuffers: modern
                || (desktop
                    && (has("GL_ARB_framebuffer_object") || has("GL_EXT_framebuffer_blit"))),
            pack_subimage: desktop || modern || has("GL_NV_pack_subimage"),
            unpack_subimage: desktop || modern || has("GL_EXT_unpack_subimage"),
            pixel_buffers: modern
                || (desktop && version >= (2, 1))
                || has("GL_ARB_pixel_buffer_object")
                || has("GL_EXT_pixel_buffer_object")
                || has("GL_NV_pixel_buffer_object"),
            max_mip_level: desktop || modern || has("GL_APPLE_texture_max_level"),
            alpha_red,
            alpha_swizzle,
            half_float,
            legacy_depth: api == GlApi::Web
                || has("GL_OES_depth_texture")
                || has("GL_ANGLE_depth_texture"),
            legacy_depth_upload: has("GL_OES_depth_texture"),
            sized_depth: modern,
        }
    }

    fn upload_format(&self, format: TextureFormat, has_data: bool) -> (GLenum, GLenum, GLenum) {
        match format {
            TextureFormat::RGBA16F | TextureFormat::Depth | TextureFormat::Depth32
                if self.api == GlApi::Web && has_data =>
            {
                panic!("the WebGL host only supports byte color texture uploads");
            }
            TextureFormat::RGBA16F => {
                let pixel_type = self
                    .half_float
                    .expect("half-float textures are unsupported by this GL context");
                let internal = if pixel_type == GL_HALF_FLOAT {
                    GL_RGBA16F
                } else {
                    GL_RGBA
                };
                (internal, GL_RGBA, pixel_type)
            }
            TextureFormat::Depth => {
                let internal = if self.sized_depth || self.api == GlApi::Desktop {
                    GL_DEPTH_COMPONENT16
                } else if self.legacy_depth && (!has_data || self.legacy_depth_upload) {
                    GL_DEPTH_COMPONENT
                } else {
                    panic!("depth textures are unsupported by this GL context for this source");
                };
                (internal, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT)
            }
            TextureFormat::Depth32 => {
                let internal = if self.sized_depth {
                    GL_DEPTH_COMPONENT32F
                } else if self.api == GlApi::Desktop {
                    GL_DEPTH_COMPONENT32
                } else {
                    panic!("float depth textures require desktop GL, GLES3 or WebGL2");
                };
                (internal, GL_DEPTH_COMPONENT, GL_FLOAT)
            }
            TextureFormat::Alpha => match self.alpha_red {
                Some(internal) => (internal, GL_RED, GL_UNSIGNED_BYTE),
                None => (GL_ALPHA, GL_ALPHA, GL_UNSIGNED_BYTE),
            },
            _ => format.into(),
        }
    }

    fn renderbuffer_format(&self, format: TextureFormat) -> GLenum {
        match format {
            TextureFormat::RGB8 => GL_RGB8,
            TextureFormat::RGBA8 => GL_RGBA8,
            TextureFormat::Alpha => GL_R8,
            _ => self.upload_format(format, false).0,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn transfer_extensions(indexed: bool) -> Vec<String> {
    use std::ffi::CStr;
    if indexed {
        (0..gl_integer(GL_NUM_EXTENSIONS).max(0) as GLuint)
            .filter_map(|index| {
                let ptr = glGetStringi(GL_EXTENSIONS, index);
                (!ptr.is_null()).then(|| CStr::from_ptr(ptr.cast()).to_string_lossy().into_owned())
            })
            .collect()
    } else {
        let ptr = glGetString(GL_EXTENSIONS);
        if ptr.is_null() {
            Vec::new()
        } else {
            CStr::from_ptr(ptr.cast())
                .to_string_lossy()
                .split_ascii_whitespace()
                .map(str::to_owned)
                .collect()
        }
    }
}

// GLES can expose RGBA/FLOAT rather than RGBA/HALF_FLOAT as the readable
// pair for a floating-point color attachment. GL widens its binary16 values
// exactly; convert the finite widened values back to their original bits.
#[cfg(not(target_arch = "wasm32"))]
unsafe fn read_es_half_float(width: GLsizei, height: GLsizei, bytes: &mut [u8]) {
    let format = gl_integer(GL_IMPLEMENTATION_COLOR_READ_FORMAT) as GLenum;
    let pixel_type = gl_integer(GL_IMPLEMENTATION_COLOR_READ_TYPE) as GLenum;
    match (format, pixel_type) {
        (GL_RGBA, GL_HALF_FLOAT | GL_HALF_FLOAT_OES) => {
            glReadPixels(
                0,
                0,
                width,
                height,
                format,
                pixel_type,
                bytes.as_mut_ptr().cast(),
            );
        }
        (GL_RGBA, GL_FLOAT) => {
            let mut floats = vec![0.0f32; bytes.len() / 2];
            glReadPixels(
                0,
                0,
                width,
                height,
                format,
                pixel_type,
                floats.as_mut_ptr().cast(),
            );
            for (value, out) in floats.into_iter().zip(bytes.chunks_exact_mut(2)) {
                out.copy_from_slice(&narrow_widened_half(value).to_ne_bytes());
            }
        }
        _ => panic!("this GLES context does not support RGBA half-float readback"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn narrow_widened_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = (bits >> 23) & 0xff;
    let mantissa = bits & 0x7f_ffff;
    sign | match exponent {
        255 => {
            if mantissa == 0 {
                0x7c00
            } else {
                0x7e00
            }
        }
        143..=254 => 0x7c00,
        113..=142 => (((exponent - 112) << 10) | (mantissa >> 13)) as u16,
        103..=112 => ((mantissa | 0x80_0000) >> (126 - exponent)) as u16,
        _ => 0,
    }
}

// GL receives a pointer, not the slice length. Compare in pixels before
// multiplying by the number of bytes per pixel.
fn transfer_len(format: TextureFormat, width: GLsizei, height: GLsizei, available: usize) -> usize {
    let pixels = u64::from(width.max(0) as u32) * u64::from(height.max(0) as u32);
    let bytes_per_pixel = format.size(1, 1) as usize;
    assert!(
        pixels <= (available / bytes_per_pixel) as u64,
        "texture transfer buffer is too small"
    );
    pixels as usize * bytes_per_pixel
}

fn mip_sizes(width: GLsizei, height: GLsizei) -> impl Iterator<Item = (GLsizei, GLsizei)> {
    std::iter::successors(Some((width, height)), |&(w, h)| {
        (w > 0 && h > 0 && (w > 1 || h > 1)).then(|| ((w / 2).max(1), (h / 2).max(1)))
    })
}

unsafe fn gl_integer(param: GLenum) -> GLint {
    let mut value = 0;
    glGetIntegerv(param, &mut value);
    value
}

// Borrow the actual GL binding on the active texture unit without changing the
// active unit or Miniquad's cache. Raw GL users need not synchronize the cache
// just to make a transfer.
struct TransferTextureBinding(GLenum, GLuint);

impl TransferTextureBinding {
    unsafe fn new(kind: TextureKind, texture: GLuint) -> Self {
        let name = match kind {
            TextureKind::Texture2D => GL_TEXTURE_BINDING_2D,
            TextureKind::CubeMap => GL_TEXTURE_BINDING_CUBE_MAP,
        };
        let binding = Self(kind.into(), gl_integer(name) as GLuint);
        glBindTexture(binding.0, texture);
        binding
    }
}

impl Drop for TransferTextureBinding {
    fn drop(&mut self) {
        unsafe { glBindTexture(self.0, self.1) };
    }
}

#[derive(Clone, Copy)]
enum PixelTransfer {
    Pack,
    Unpack,
}

struct TightPixelStore {
    params: [(GLenum, GLint); 6],
    count: usize,
    buffer: Option<(GLenum, GLuint)>,
}

impl TightPixelStore {
    unsafe fn new(caps: &TransferCaps, transfer: PixelTransfer) -> Self {
        let (alignment, subimage, offsets, buffer, binding) = match transfer {
            PixelTransfer::Pack => (
                GL_PACK_ALIGNMENT,
                caps.pack_subimage,
                [GL_PACK_ROW_LENGTH, GL_PACK_SKIP_ROWS, GL_PACK_SKIP_PIXELS],
                GL_PIXEL_PACK_BUFFER,
                GL_PIXEL_PACK_BUFFER_BINDING,
            ),
            PixelTransfer::Unpack => (
                GL_UNPACK_ALIGNMENT,
                caps.unpack_subimage,
                [
                    GL_UNPACK_ROW_LENGTH,
                    GL_UNPACK_SKIP_ROWS,
                    GL_UNPACK_SKIP_PIXELS,
                ],
                GL_PIXEL_UNPACK_BUFFER,
                GL_PIXEL_UNPACK_BUFFER_BINDING,
            ),
        };
        let mut state = Self {
            params: [(0, 0); 6],
            count: 0,
            buffer: None,
        };
        state.set(alignment, 1);
        if subimage {
            for param in offsets {
                state.set(param, 0);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if caps.api == GlApi::Desktop {
            state.set(
                match transfer {
                    PixelTransfer::Pack => GL_PACK_SWAP_BYTES,
                    PixelTransfer::Unpack => GL_UNPACK_SWAP_BYTES,
                },
                0,
            );
        }
        #[cfg(target_arch = "wasm32")]
        if matches!(transfer, PixelTransfer::Unpack) {
            state.set(GL_UNPACK_FLIP_Y_WEBGL, 0);
            state.set(GL_UNPACK_PREMULTIPLY_ALPHA_WEBGL, 0);
        }
        if caps.pixel_buffers {
            let previous = gl_integer(binding) as GLuint;
            if previous != 0 {
                state.buffer = Some((buffer, previous));
                glBindBuffer(buffer, 0);
            }
        }
        state
    }

    unsafe fn set(&mut self, param: GLenum, tight: GLint) {
        let previous = gl_integer(param);
        if previous != tight {
            self.params[self.count] = (param, previous);
            self.count += 1;
            glPixelStorei(param, tight);
        }
    }
}

impl Drop for TightPixelStore {
    fn drop(&mut self) {
        unsafe {
            for &(param, previous) in &self.params[..self.count] {
                glPixelStorei(param, previous);
            }
            if let Some((buffer, previous)) = self.buffer {
                glBindBuffer(buffer, previous);
            }
        }
    }
}

struct ReadbackFramebuffer {
    fbo: GLuint,
    target: GLenum,
    previous: GLuint,
    previous_read: Option<GLuint>,
}

impl ReadbackFramebuffer {
    unsafe fn new(caps: &TransferCaps, depth: bool) -> Option<Self> {
        let target = if caps.split_framebuffers && !depth {
            GL_READ_FRAMEBUFFER
        } else {
            GL_FRAMEBUFFER
        };
        let previous = gl_integer(if target == GL_READ_FRAMEBUFFER {
            GL_READ_FRAMEBUFFER_BINDING
        } else {
            GL_FRAMEBUFFER_BINDING
        });
        let previous_read = (caps.split_framebuffers && depth)
            .then(|| gl_integer(GL_READ_FRAMEBUFFER_BINDING) as GLuint);
        let mut fbo = 0;
        glGenFramebuffers(1, &mut fbo);
        if fbo == 0 {
            return None;
        }
        glBindFramebuffer(target, fbo);
        Some(Self {
            fbo,
            target,
            previous: previous as GLuint,
            previous_read,
        })
    }
}

impl Drop for ReadbackFramebuffer {
    fn drop(&mut self) {
        unsafe {
            glBindFramebuffer(self.target, self.previous);
            if let Some(read) = self.previous_read {
                glBindFramebuffer(GL_READ_FRAMEBUFFER, read);
            }
            glDeleteFramebuffers(1, &self.fbo);
        }
    }
}

impl Texture {
    pub fn new(
        caps: &TransferCaps,
        access: TextureAccess,
        source: TextureSource,
        params: TextureParams,
    ) -> Texture {
        let (width, height) = (params.width as GLsizei, params.height as GLsizei);
        let faces = if params.kind == TextureKind::CubeMap {
            6
        } else {
            1
        };
        let target = |face: usize| {
            if params.kind == TextureKind::CubeMap {
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + face as GLenum
            } else {
                GL_TEXTURE_2D
            }
        };
        let mut params = params;
        if access != TextureAccess::RenderTarget {
            params.sample_count = 1;
        }
        if params.sample_count > 1 {
            let internal_format = caps.renderbuffer_format(params.format);
            let mut renderbuffer = 0;
            unsafe {
                let previous = gl_integer(GL_RENDERBUFFER_BINDING) as GLuint;
                glGenRenderbuffers(1, &mut renderbuffer);
                glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
                glRenderbufferStorageMultisample(
                    GL_RENDERBUFFER,
                    params.sample_count,
                    internal_format,
                    width,
                    height,
                );
                glBindRenderbuffer(GL_RENDERBUFFER, previous);
            }
            return Texture {
                raw: TextureOrRenderbuffer::Renderbuffer(renderbuffer),
                params,
            };
        }

        let (internal_format, format, pixel_type) =
            caps.upload_format(params.format, !matches!(source, TextureSource::Empty));
        match source {
            TextureSource::Empty => {}
            TextureSource::Bytes(bytes) => {
                transfer_len(params.format, width, height, bytes.len());
            }
            TextureSource::Array(array) => {
                for mipmaps in array.iter().take(faces) {
                    for ((w, h), bytes) in mip_sizes(width, height).zip(*mipmaps) {
                        transfer_len(params.format, w, h, bytes.len());
                    }
                }
            }
        }
        let mut texture = 0;
        unsafe {
            glGenTextures(1, &mut texture);
            let _binding = TransferTextureBinding::new(params.kind, texture);
            let _unpack = TightPixelStore::new(caps, PixelTransfer::Unpack);
            let gl_target = params.kind.into();
            if params.format == TextureFormat::Alpha && caps.alpha_swizzle {
                glTexParameteri(gl_target, GL_TEXTURE_SWIZZLE_A, GL_RED as GLint);
            }
            let upload = |target, level, w, h, bytes: Option<&[u8]>| {
                glTexImage2D(
                    target,
                    level,
                    internal_format as GLint,
                    w,
                    h,
                    0,
                    format,
                    pixel_type,
                    bytes.map_or(std::ptr::null(), |b| b.as_ptr().cast()),
                );
            };
            match source {
                TextureSource::Empty => {
                    for face in 0..faces {
                        upload(target(face), 0, width, height, None);
                    }
                }
                TextureSource::Bytes(bytes) => upload(target(0), 0, width, height, Some(bytes)),
                TextureSource::Array(array) => {
                    let mut levels = 0;
                    for (face, mipmaps) in array.iter().take(faces).enumerate() {
                        for (level, ((w, h), bytes)) in
                            mip_sizes(width, height).zip(*mipmaps).enumerate()
                        {
                            upload(target(face), level as GLint, w, h, Some(bytes));
                            levels = levels.max(level as GLint + 1);
                        }
                    }
                    if caps.max_mip_level && levels > 1 {
                        glTexParameteri(gl_target, GL_TEXTURE_MAX_LEVEL, levels - 1);
                    }
                }
            }

            let wrap = match params.wrap {
                TextureWrap::Repeat => GL_REPEAT,
                TextureWrap::Mirror => GL_MIRRORED_REPEAT,
                TextureWrap::Clamp => GL_CLAMP_TO_EDGE,
            };
            let min_filter = Self::gl_filter(params.min_filter, params.mipmap_filter);
            let mag_filter = match params.mag_filter {
                FilterMode::Nearest => GL_NEAREST,
                FilterMode::Linear => GL_LINEAR,
            };
            glTexParameteri(gl_target, GL_TEXTURE_WRAP_S, wrap as GLint);
            glTexParameteri(gl_target, GL_TEXTURE_WRAP_T, wrap as GLint);
            glTexParameteri(gl_target, GL_TEXTURE_MIN_FILTER, min_filter as GLint);
            glTexParameteri(gl_target, GL_TEXTURE_MAG_FILTER, mag_filter as GLint);
        }
        Texture {
            raw: TextureOrRenderbuffer::Texture(texture),
            params,
        }
    }

    pub fn resize(&mut self, caps: &TransferCaps, width: u32, height: u32, source: Option<&[u8]>) {
        let (gl_width, gl_height) = (width as GLsizei, height as GLsizei);
        if gl_width < 0 || gl_height < 0 || self.params.kind == TextureKind::CubeMap {
            return;
        }
        if let Some(bytes) = source {
            transfer_len(self.params.format, gl_width, gl_height, bytes.len());
        }
        let raw = self
            .raw
            .texture()
            .expect("Resize not supported for multisampled textures");
        let (internal_format, format, pixel_type) =
            caps.upload_format(self.params.format, source.is_some());
        unsafe {
            let _binding = TransferTextureBinding::new(self.params.kind, raw);
            let _unpack = TightPixelStore::new(caps, PixelTransfer::Unpack);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                internal_format as GLint,
                gl_width,
                gl_height,
                0,
                format,
                pixel_type,
                source.map_or(std::ptr::null(), |b| b.as_ptr().cast()),
            );
        }
        self.params.width = width;
        self.params.height = height;
    }

    pub fn update_texture_part(
        &self,
        caps: &TransferCaps,
        x_offset: i32,
        y_offset: i32,
        width: i32,
        height: i32,
        source: &[u8],
    ) {
        if width <= 0 || height <= 0 || self.params.kind == TextureKind::CubeMap {
            return;
        }
        transfer_len(self.params.format, width, height, source.len());
        let raw = self
            .raw
            .texture()
            .expect("Texture updates are not supported for multisampled textures");
        let (_, format, pixel_type) = caps.upload_format(self.params.format, true);
        unsafe {
            let _binding = TransferTextureBinding::new(self.params.kind, raw);
            let _unpack = TightPixelStore::new(caps, PixelTransfer::Unpack);
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                x_offset,
                y_offset,
                width,
                height,
                format,
                pixel_type,
                source.as_ptr().cast(),
            );
        }
    }

    /// Read texture data into CPU memory.
    pub fn read_pixels(&self, caps: &TransferCaps, bytes: &mut [u8]) {
        let (width, height) = (self.params.width as GLsizei, self.params.height as GLsizei);
        if width <= 0 || height <= 0 || self.params.kind == TextureKind::CubeMap {
            return;
        }
        let desktop = caps.api == GlApi::Desktop;
        let depth = matches!(
            self.params.format,
            TextureFormat::Depth | TextureFormat::Depth32
        );
        if (depth && !desktop)
            || (caps.api == GlApi::Web && self.params.format == TextureFormat::RGBA16F)
        {
            return;
        }
        let raw = self
            .raw
            .texture()
            .expect("Texture readback is not supported for multisampled textures");
        let len = transfer_len(self.params.format, width, height, bytes.len());

        // GLES/WebGL guarantee RGBA/UNSIGNED_BYTE for normalized color buffers,
        // not a direct RGB or RED read. Keep the public bytes tightly packed.
        let compact_channels = match self.params.format {
            TextureFormat::RGB8 if !desktop => Some(3),
            TextureFormat::Alpha if !desktop => Some(1),
            _ => None,
        };
        let mut rgba = compact_channels.map(|channels| vec![[0u8; 4]; len / channels]);
        let (_, format, pixel_type) = caps.upload_format(self.params.format, false);
        unsafe {
            let Some(fbo) = ReadbackFramebuffer::new(caps, depth) else {
                return;
            };
            let attachment = if depth {
                GL_DEPTH_ATTACHMENT
            } else {
                GL_COLOR_ATTACHMENT0
            };
            glFramebufferTexture2D(fbo.target, attachment, GL_TEXTURE_2D, raw, 0);
            #[cfg(not(target_arch = "wasm32"))]
            if depth {
                glReadBuffer(GL_NONE);
                glDrawBuffers(1, &GL_NONE);
            }
            if glCheckFramebufferStatus(fbo.target) != GL_FRAMEBUFFER_COMPLETE {
                return;
            }

            let _pack = TightPixelStore::new(caps, PixelTransfer::Pack);
            #[cfg(not(target_arch = "wasm32"))]
            if !desktop && self.params.format == TextureFormat::RGBA16F {
                read_es_half_float(width, height, &mut bytes[..len]);
                return;
            }
            if let Some(rgba) = rgba.as_mut() {
                glReadPixels(
                    0,
                    0,
                    width,
                    height,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    rgba.as_mut_ptr().cast(),
                );
            } else {
                glReadPixels(
                    0,
                    0,
                    width,
                    height,
                    format,
                    pixel_type,
                    bytes.as_mut_ptr().cast(),
                );
            }
        }
        if let (Some(rgba), Some(channels)) = (rgba, compact_channels) {
            for (src, dst) in rgba.iter().zip(bytes[..len].chunks_exact_mut(channels)) {
                dst.copy_from_slice(&src[..channels]);
            }
        }
    }

    fn gl_filter(filter: FilterMode, mipmap_filter: MipmapFilterMode) -> GLenum {
        match filter {
            FilterMode::Nearest => match mipmap_filter {
                MipmapFilterMode::None => GL_NEAREST,
                MipmapFilterMode::Nearest => GL_NEAREST_MIPMAP_NEAREST,
                MipmapFilterMode::Linear => GL_NEAREST_MIPMAP_LINEAR,
            },
            FilterMode::Linear => match mipmap_filter {
                MipmapFilterMode::None => GL_LINEAR,
                MipmapFilterMode::Nearest => GL_LINEAR_MIPMAP_NEAREST,
                MipmapFilterMode::Linear => GL_LINEAR_MIPMAP_LINEAR,
            },
        }
    }
}

pub(crate) struct PipelineInternal {
    layout: Vec<Option<VertexAttributeInternal>>,
    shader: ShaderId,
    params: PipelineParams,
}

type UniformLocation = Option<GLint>;

pub struct ShaderImage {
    gl_loc: UniformLocation,
}

fn get_uniform_location(program: GLuint, name: &str) -> Option<i32> {
    let cname = CString::new(name).unwrap_or_else(|e| panic!("{}", e));
    let location = unsafe { glGetUniformLocation(program, cname.as_ptr()) };

    if location == -1 {
        return None;
    }

    Some(location)
}

pub(crate) struct RenderPassInternal {
    gl_fb: GLuint,
    color_textures: Vec<TextureId>,
    resolves: Option<Vec<(u32, TextureId)>>,
    depth_texture: Option<TextureId>,
}

struct Textures(ResourceManager<Texture>);
impl Textures {
    fn get(&self, texture: TextureId) -> Texture {
        match texture.0 {
            TextureIdInner::Raw(RawId::OpenGl(texture)) => Texture {
                raw: TextureOrRenderbuffer::Texture(texture),
                params: Default::default(),
            },
            #[cfg(target_vendor = "apple")]
            TextureIdInner::Raw(RawId::Metal(..)) => panic!("Metal texture in OpenGL context!"),
            #[cfg(all(
                feature = "vulkan",
                any(windows, all(target_os = "linux", not(target_env = "ohos")))
            ))]
            TextureIdInner::Raw(RawId::Vulkan(_)) => panic!("Vulkan image passed to OpenGL"),
            TextureIdInner::Managed(texture) => self.0[texture],
        }
    }
}
pub struct GlContext {
    shaders: ResourceManager<ShaderInternal>,
    pipelines: ResourceManager<PipelineInternal>,
    passes: ResourceManager<RenderPassInternal>,
    buffers: ResourceManager<Buffer>,
    textures: Textures,
    default_framebuffer: GLuint,
    pub(crate) cache: GlCache,
    pub(crate) info: ContextInfo,
    transfer_caps: TransferCaps,
}

impl Default for GlContext {
    fn default() -> Self {
        Self::new()
    }
}

impl GlContext {
    pub fn new() -> GlContext {
        unsafe {
            let mut default_framebuffer: GLuint = 0;
            glGetIntegerv(
                GL_FRAMEBUFFER_BINDING,
                &mut default_framebuffer as *mut _ as *mut _,
            );
            let mut vao = 0;

            glGenVertexArrays(1, &mut vao as *mut _);
            glBindVertexArray(vao);
            let info = gl_info();
            let transfer_caps = TransferCaps::new(&info.gl_version_string);
            GlContext {
                default_framebuffer,
                shaders: ResourceManager::default(),
                pipelines: ResourceManager::default(),
                passes: ResourceManager::default(),
                buffers: ResourceManager::default(),
                textures: Textures(ResourceManager::default()),
                info,
                transfer_caps,
                cache: GlCache {
                    stored_index_buffer: 0,
                    stored_index_type: None,
                    stored_vertex_buffer: 0,
                    index_buffer: 0,
                    index_type: None,
                    vertex_buffer: 0,
                    cur_pipeline: None,
                    cur_pass: None,
                    color_blend: None,
                    alpha_blend: None,
                    stencil: None,
                    color_write: (true, true, true, true),
                    cull_face: CullFace::Nothing,
                    stored_texture: 0,
                    stored_target: 0,
                    textures: [CachedTexture {
                        target: 0,
                        texture: 0,
                    }; MAX_SHADERSTAGE_IMAGES],
                    attributes: [None; MAX_VERTEX_ATTRIBUTES],
                },
            }
        }
    }

    pub fn features(&self) -> &Features {
        &self.info.features
    }
}

fn load_shader_internal(
    vertex_shader: &str,
    fragment_shader: &str,
    meta: ShaderMeta,
) -> Result<ShaderInternal, ShaderError> {
    unsafe {
        let vertex_shader = load_shader(GL_VERTEX_SHADER, vertex_shader)?;
        let fragment_shader = load_shader(GL_FRAGMENT_SHADER, fragment_shader)?;

        let program = glCreateProgram();
        glAttachShader(program, vertex_shader);
        glAttachShader(program, fragment_shader);
        glLinkProgram(program);

        // delete no longer used shaders
        glDetachShader(program, vertex_shader);
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);

        let mut link_status = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &mut link_status as *mut _);
        if link_status == 0 {
            let mut max_length: i32 = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &mut max_length as *mut _);

            let mut error_message = vec![0u8; max_length as usize + 1];
            glGetProgramInfoLog(
                program,
                max_length,
                &mut max_length as *mut _,
                error_message.as_mut_ptr() as *mut _,
            );
            assert!(max_length >= 1);
            let error_message =
                std::string::String::from_utf8_lossy(&error_message[0..max_length as usize - 1]);
            return Err(ShaderError::LinkError(error_message.to_string()));
        }

        glUseProgram(program);

        #[rustfmt::skip]
        let images = meta.images.iter().map(|name| ShaderImage {
            gl_loc: get_uniform_location(program, name),
        }).collect();

        #[rustfmt::skip]
        let uniforms = meta.uniforms.uniforms.iter().scan(0, |offset, uniform| {
            let res = ShaderUniform {
                gl_loc: get_uniform_location(program, &uniform.name),
                uniform_type: uniform.uniform_type,
                array_count: uniform.array_count as _,
            };
            *offset += uniform.uniform_type.size() * uniform.array_count;
            Some(res)
        }).collect();

        Ok(ShaderInternal {
            program,
            images,
            uniforms,
        })
    }
}

pub fn load_shader(shader_type: GLenum, source: &str) -> Result<GLuint, ShaderError> {
    unsafe {
        let shader = glCreateShader(shader_type);
        assert!(shader != 0);

        let cstring = CString::new(source)?;
        let csource = [cstring];
        glShaderSource(shader, 1, csource.as_ptr() as *const _, std::ptr::null());
        glCompileShader(shader);

        let mut is_compiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &mut is_compiled as *mut _);
        if is_compiled == 0 {
            let mut max_length: i32 = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &mut max_length as *mut _);

            let mut error_message = vec![0u8; max_length as usize + 1];
            glGetShaderInfoLog(
                shader,
                max_length,
                &mut max_length as *mut _,
                error_message.as_mut_ptr() as *mut _,
            );

            assert!(max_length >= 1);
            let mut error_message =
                std::string::String::from_utf8_lossy(&error_message[0..max_length as usize - 1])
                    .into_owned();

            // On Wasm + Chrome, for unknown reason, string with zero-terminator is returned. On Firefox there is no zero-terminators in JavaScript string.
            if error_message.ends_with('\0') {
                error_message.pop();
            }

            return Err(ShaderError::CompilationError {
                shader_type: match shader_type {
                    GL_VERTEX_SHADER => ShaderType::Vertex,
                    GL_FRAGMENT_SHADER => ShaderType::Fragment,
                    _ => unreachable!(),
                },
                error_message,
            });
        }

        Ok(shader)
    }
}

impl GlContext {
    fn set_blend(&mut self, color_blend: Option<BlendState>, alpha_blend: Option<BlendState>) {
        if color_blend.is_none() && alpha_blend.is_some() {
            panic!("AlphaBlend without ColorBlend");
        }
        if self.cache.color_blend == color_blend && self.cache.alpha_blend == alpha_blend {
            return;
        }

        unsafe {
            if let Some(color_blend) = color_blend {
                if self.cache.color_blend.is_none() {
                    glEnable(GL_BLEND);
                }

                let BlendState {
                    equation: eq_rgb,
                    sfactor: src_rgb,
                    dfactor: dst_rgb,
                } = color_blend;

                if let Some(BlendState {
                    equation: eq_alpha,
                    sfactor: src_alpha,
                    dfactor: dst_alpha,
                }) = alpha_blend
                {
                    glBlendFuncSeparate(
                        src_rgb.into(),
                        dst_rgb.into(),
                        src_alpha.into(),
                        dst_alpha.into(),
                    );
                    glBlendEquationSeparate(eq_rgb.into(), eq_alpha.into());
                } else {
                    glBlendFunc(src_rgb.into(), dst_rgb.into());
                    glBlendEquationSeparate(eq_rgb.into(), eq_rgb.into());
                }
            } else if self.cache.color_blend.is_some() {
                glDisable(GL_BLEND);
            }
        }

        self.cache.color_blend = color_blend;
        self.cache.alpha_blend = alpha_blend;
    }

    fn set_stencil(&mut self, stencil_test: Option<StencilState>) {
        if self.cache.stencil == stencil_test {
            return;
        }
        unsafe {
            if let Some(stencil) = stencil_test {
                if self.cache.stencil.is_none() {
                    glEnable(GL_STENCIL_TEST);
                }

                let front = &stencil.front;
                glStencilOpSeparate(
                    GL_FRONT,
                    front.fail_op.into(),
                    front.depth_fail_op.into(),
                    front.pass_op.into(),
                );
                glStencilFuncSeparate(
                    GL_FRONT,
                    front.test_func.into(),
                    front.test_ref,
                    front.test_mask,
                );
                glStencilMaskSeparate(GL_FRONT, front.write_mask);

                let back = &stencil.back;
                glStencilOpSeparate(
                    GL_BACK,
                    back.fail_op.into(),
                    back.depth_fail_op.into(),
                    back.pass_op.into(),
                );
                glStencilFuncSeparate(
                    GL_BACK,
                    back.test_func.into(),
                    back.test_ref,
                    back.test_mask,
                );
                glStencilMaskSeparate(GL_BACK, back.write_mask);
            } else if self.cache.stencil.is_some() {
                glDisable(GL_STENCIL_TEST);
            }
        }

        self.cache.stencil = stencil_test;
    }

    fn set_cull_face(&mut self, cull_face: CullFace) {
        if self.cache.cull_face == cull_face {
            return;
        }

        match cull_face {
            CullFace::Nothing => unsafe {
                glDisable(GL_CULL_FACE);
            },
            CullFace::Front => unsafe {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_FRONT);
            },
            CullFace::Back => unsafe {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
            },
        }
        self.cache.cull_face = cull_face;
    }

    fn set_color_write(&mut self, color_write: ColorMask) {
        if self.cache.color_write == color_write {
            return;
        }
        let (r, g, b, a) = color_write;
        unsafe { glColorMask(r as _, g as _, b as _, a as _) }
        self.cache.color_write = color_write;
    }
}

#[allow(clippy::field_reassign_with_default)]
fn gl_info() -> ContextInfo {
    let version_string = unsafe { glGetString(super::gl::GL_VERSION) };
    let gl_version_string = unsafe { std::ffi::CStr::from_ptr(version_string as _) }
        .to_str()
        .unwrap()
        .to_string();
    //let gles2 = !gles3 && gl_version_string.contains("OpenGL ES");

    let gl2 = gl_version_string.is_empty()
        || gl_version_string.starts_with("2")
        || gl_version_string.starts_with("OpenGL ES 2");
    let webgl1 = gl_version_string == "WebGL 1.0";

    let features = Features {
        instancing: !gl2,
        resolve_attachments: !webgl1 && !gl2,
    };

    let mut glsl_support = GlslSupport::default();

    // this is not quite documented,
    // but somehow even GL2.1 usually have all the compatibility extensions to support glsl100
    // It was tested on really old windows machines, virtual machines etc. glsl100 always works!
    glsl_support.v100 = true;

    // on wasm miniquad always creates webgl1 context, with the only glsl available being version 100
    #[cfg(target_arch = "wasm32")]
    {
        // on web, miniquad always loads EXT_shader_texture_lod and OES_standard_derivatives
        glsl_support.v100_ext = true;

        let webgl2 = gl_version_string.contains("WebGL 2.0");
        if webgl2 {
            glsl_support.v300es = true;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let gles3 = gl_version_string.contains("OpenGL ES 3");

        if gles3 {
            glsl_support.v300es = true;
        }
    }

    // there is no gl3.4, so 4+ and 3.3 covers all modern OpenGL
    if gl_version_string.starts_with("3.2") {
        glsl_support.v150 = true; // MacOS is defaulting to 3.2 and GLSL 150
    } else if gl_version_string.starts_with("4") || gl_version_string.starts_with("3.3") {
        glsl_support.v330 = true;
    // gl 3.0, 3.1, 3.2 maps to 1.30, 1.40, 1.50 glsl versions
    } else if gl_version_string.starts_with("3") {
        glsl_support.v130 = true;
    }

    ContextInfo {
        backend: Backend::OpenGl,
        gl_version_string,
        glsl_support,
        features,
    }
}

impl RenderingBackend for GlContext {
    fn info(&self) -> ContextInfo {
        self.info.clone()
    }

    fn max_texture_size(&self) -> Option<u32> {
        unsafe {
            let mut size = 0;
            glGetIntegerv(GL_MAX_TEXTURE_SIZE, &mut size);
            (size > 0).then_some(size as u32)
        }
    }

    fn new_shader(
        &mut self,
        shader: ShaderSource,
        meta: ShaderMeta,
    ) -> Result<ShaderId, ShaderError> {
        let (fragment, vertex) = match shader {
            ShaderSource::Glsl { fragment, vertex } => (fragment, vertex),
            #[cfg(all(
                feature = "vulkan",
                any(windows, all(target_os = "linux", not(target_env = "ohos")))
            ))]
            ShaderSource::SpirV(_) => {
                return Err(ShaderError::LinkError(
                    "SPIR-V sources require the Vulkan backend".into(),
                ));
            }
            ShaderSource::Msl { .. } => panic!("Metal source on OpenGl context"),
        };
        let shader = load_shader_internal(vertex, fragment, meta)?;
        Ok(ShaderId(self.shaders.add(shader)))
    }

    fn new_texture(
        &mut self,
        access: TextureAccess,
        source: TextureSource,
        params: TextureParams,
    ) -> TextureId {
        let texture = Texture::new(&self.transfer_caps, access, source, params);
        TextureId(TextureIdInner::Managed(self.textures.0.add(texture)))
    }

    fn delete_texture(&mut self, texture: TextureId) {
        self.cache.clear_texture_bindings();

        let t = self.textures.get(texture);
        match &t.raw {
            TextureOrRenderbuffer::Texture(raw) => unsafe {
                glDeleteTextures(1, raw as *const _);
            },
            TextureOrRenderbuffer::Renderbuffer(raw) => unsafe {
                glDeleteRenderbuffers(1, raw as *const _);
            },
        }
        if let TextureIdInner::Managed(id) = texture.0 {
            self.textures.0.remove(id);
        }
    }

    fn delete_shader(&mut self, program: ShaderId) {
        unsafe { glDeleteProgram(self.shaders[program.0].program) };
        self.shaders.remove(program.0);
        self.cache.cur_pipeline = None;
    }

    fn delete_pipeline(&mut self, pipeline: Pipeline) {
        self.pipelines.remove(pipeline.0);
    }

    fn texture_set_wrap(&mut self, texture: TextureId, wrap_x: TextureWrap, wrap_y: TextureWrap) {
        let t = self.textures.get(texture);
        let raw = t
            .raw
            .texture()
            .expect("texture_set_wrap not yet implemented for RenderBuffer(multisampled) textures");

        self.cache.store_texture_binding(0);
        self.cache.bind_texture(0, t.params.kind.into(), raw);
        let wrap_x = match wrap_x {
            TextureWrap::Repeat => GL_REPEAT,
            TextureWrap::Mirror => GL_MIRRORED_REPEAT,
            TextureWrap::Clamp => GL_CLAMP_TO_EDGE,
        };

        let wrap_y = match wrap_y {
            TextureWrap::Repeat => GL_REPEAT,
            TextureWrap::Mirror => GL_MIRRORED_REPEAT,
            TextureWrap::Clamp => GL_CLAMP_TO_EDGE,
        };

        unsafe {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_x as i32);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_y as i32);
        }
        self.cache.restore_texture_binding(0);
    }

    fn texture_set_min_filter(
        &mut self,
        texture: TextureId,
        filter: FilterMode,
        mipmap_filter: MipmapFilterMode,
    ) {
        let t = self.textures.get(texture);
        let raw = t.raw.texture().expect(
            "texture_set_min_filter not yet implemented for RenderBuffer(multisampled) textures",
        );

        self.cache.store_texture_binding(0);
        self.cache.bind_texture(0, t.params.kind.into(), raw);

        let filter = Texture::gl_filter(filter, mipmap_filter);
        unsafe {
            glTexParameteri(t.params.kind.into(), GL_TEXTURE_MIN_FILTER, filter as i32);
        }
        self.cache.restore_texture_binding(0);
    }
    fn texture_set_mag_filter(&mut self, texture: TextureId, filter: FilterMode) {
        let t = self.textures.get(texture);
        let raw = t
            .raw
            .texture()
            .expect("texture_set_wrap not yet implemented for RenderBuffer(multisampled) textures");

        self.cache.store_texture_binding(0);
        self.cache.bind_texture(0, t.params.kind.into(), raw);

        let filter = match filter {
            FilterMode::Nearest => GL_NEAREST,
            FilterMode::Linear => GL_LINEAR,
        };
        unsafe {
            glTexParameteri(t.params.kind.into(), GL_TEXTURE_MAG_FILTER, filter as i32);
        }
        self.cache.restore_texture_binding(0);
    }
    fn texture_resize(
        &mut self,
        texture: TextureId,
        width: u32,
        height: u32,
        source: Option<&[u8]>,
    ) {
        let mut t = self.textures.get(texture);
        t.resize(&self.transfer_caps, width, height, source);
        if let TextureIdInner::Managed(tex_id) = texture.0 {
            self.textures.0[tex_id].params = t.params;
        };
    }
    fn texture_read_pixels(&mut self, texture: TextureId, bytes: &mut [u8]) {
        let t = self.textures.get(texture);
        t.read_pixels(&self.transfer_caps, bytes);
    }
    fn texture_generate_mipmaps(&mut self, texture: TextureId) {
        let t = self.textures.get(texture);
        let raw = t.raw.texture().expect(
            "texture_generate_mipmaps not yet implemented for RenderBuffer(multisampled) textures",
        );

        self.cache.store_texture_binding(0);
        self.cache.bind_texture(0, t.params.kind.into(), raw);
        unsafe {
            glGenerateMipmap(t.params.kind.into());
        }
        self.cache.restore_texture_binding(0);
    }
    fn texture_update_part(
        &mut self,
        texture: TextureId,
        x_offset: i32,
        y_offset: i32,
        width: i32,
        height: i32,
        source: &[u8],
    ) {
        let t = self.textures.get(texture);
        t.update_texture_part(
            &self.transfer_caps,
            x_offset,
            y_offset,
            width,
            height,
            source,
        );
    }
    fn texture_params(&self, texture: TextureId) -> TextureParams {
        let texture = self.textures.get(texture);
        texture.params
    }
    unsafe fn texture_raw_id(&self, texture: TextureId) -> RawId {
        let texture = self.textures.get(texture);
        let raw = texture
            .raw
            .texture()
            .expect("For multisampled texture raw_id is not supported");

        RawId::OpenGl(raw)
    }

    fn new_render_pass_mrt(
        &mut self,
        color_img: &[TextureId],
        resolve_img: Option<&[TextureId]>,
        depth_img: Option<TextureId>,
    ) -> RenderPass {
        if color_img.is_empty() && depth_img.is_none() {
            panic!("Render pass should have at least one non-none target");
        }
        let mut gl_fb = 0;

        let mut resolves = None;
        unsafe {
            glGenFramebuffers(1, &mut gl_fb as *mut _);
            glBindFramebuffer(GL_FRAMEBUFFER, gl_fb);
            for (i, color_img) in color_img.iter().enumerate() {
                let texture = self.textures.get(*color_img);
                if texture.params.sample_count > 1 {
                    let raw = texture.raw.renderbuffer().unwrap();
                    glFramebufferRenderbuffer(
                        GL_FRAMEBUFFER,
                        GL_COLOR_ATTACHMENT0 + i as u32,
                        GL_RENDERBUFFER,
                        raw,
                    );
                } else {
                    let raw = texture.raw.texture().unwrap();
                    glFramebufferTexture2D(
                        GL_FRAMEBUFFER,
                        GL_COLOR_ATTACHMENT0 + i as u32,
                        GL_TEXTURE_2D,
                        raw,
                        0,
                    );
                }
            }
            if let Some(depth_img) = depth_img {
                let texture = self.textures.get(depth_img);
                if texture.params.sample_count > 1 {
                    let raw = texture.raw.renderbuffer().unwrap();
                    glFramebufferRenderbuffer(
                        GL_FRAMEBUFFER,
                        GL_DEPTH_ATTACHMENT,
                        GL_RENDERBUFFER,
                        raw,
                    );
                } else {
                    let raw = texture.raw.texture().unwrap();
                    glFramebufferTexture2D(
                        GL_FRAMEBUFFER,
                        GL_DEPTH_ATTACHMENT,
                        GL_TEXTURE_2D,
                        raw,
                        0,
                    );
                }
            }
            let mut attachments = vec![];
            for i in 0..color_img.len() {
                attachments.push(GL_COLOR_ATTACHMENT0 + i as u32);
            }

            if color_img.len() > 1 {
                glDrawBuffers(color_img.len() as _, attachments.as_ptr() as _);
            }

            if let Some(resolve_img) = resolve_img {
                resolves = Some(vec![]);
                let resolves = resolves.as_mut().unwrap();
                for (i, resolve_img) in resolve_img.iter().enumerate() {
                    let mut resolve_fb = 0;
                    glGenFramebuffers(1, &mut resolve_fb as *mut _);
                    glBindFramebuffer(GL_FRAMEBUFFER, resolve_fb);
                    resolves.push((resolve_fb, *resolve_img));
                    let texture = self.textures.get(*resolve_img);
                    let raw = texture.raw.texture().unwrap();
                    glFramebufferTexture2D(
                        GL_FRAMEBUFFER,
                        GL_COLOR_ATTACHMENT0 + i as u32,
                        GL_TEXTURE_2D,
                        raw,
                        0,
                    );
                    let fb_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
                    assert!(fb_status != 0);
                    glDrawBuffers(1, attachments.as_ptr() as _);
                }
            }
            glBindFramebuffer(GL_FRAMEBUFFER, self.default_framebuffer);
        }
        let pass = RenderPassInternal {
            gl_fb,
            color_textures: color_img.to_vec(),
            resolves,
            depth_texture: depth_img,
        };

        RenderPass(self.passes.add(pass))
    }
    fn render_pass_color_attachments(&self, render_pass: RenderPass) -> &[TextureId] {
        &self.passes[render_pass.0].color_textures
    }
    fn render_pass_resolve_texture(&self, pass: RenderPass) -> Option<TextureId> {
        self.passes[pass.0].resolves.as_ref().map(|resolves| {
            assert_eq!(resolves.len(), 1, "multiple resolve attachments");
            resolves[0].1
        })
    }
    fn delete_render_pass(&mut self, render_pass: RenderPass) {
        let pass_id = render_pass.0;

        let render_pass = self.passes.remove(pass_id);

        unsafe { glDeleteFramebuffers(1, &render_pass.gl_fb as *const _) }

        if let Some(resolves) = render_pass.resolves {
            for (fb, _) in resolves { unsafe { glDeleteFramebuffers(1, &fb); } }
        }
    }

    fn new_pipeline(
        &mut self,
        buffer_layout: &[BufferLayout],
        attributes: &[VertexAttribute],
        shader: ShaderId,
        params: PipelineParams,
    ) -> Pipeline {
        #[derive(Clone, Copy, Default)]
        struct BufferCacheData {
            stride: i32,
            offset: i64,
        }

        let mut buffer_cache: Vec<BufferCacheData> =
            vec![BufferCacheData::default(); buffer_layout.len()];

        for VertexAttribute {
            format,
            buffer_index,
            ..
        } in attributes
        {
            let layout = buffer_layout.get(*buffer_index).unwrap_or_else(|| panic!());
            let cache = buffer_cache
                .get_mut(*buffer_index)
                .unwrap_or_else(|| panic!());

            if layout.stride == 0 {
                cache.stride += format.size_bytes();
            } else {
                cache.stride = layout.stride;
            }
            // WebGL 1 limitation
            assert!(cache.stride <= 255);
        }

        let program = self.shaders[shader.0].program;

        let attributes_len = attributes
            .iter()
            .map(|layout| match layout.format {
                VertexFormat::Mat4 => 4,
                _ => 1,
            })
            .sum();

        let mut vertex_layout: Vec<Option<VertexAttributeInternal>> = vec![None; attributes_len];

        for VertexAttribute {
            name,
            format,
            buffer_index,
            gl_pass_as_float,
        } in attributes
        {
            let buffer_data = &mut buffer_cache
                .get_mut(*buffer_index)
                .unwrap_or_else(|| panic!());
            let layout = buffer_layout.get(*buffer_index).unwrap_or_else(|| panic!());

            let cname = CString::new(*name).unwrap_or_else(|e| panic!("{}", e));
            let attr_loc = unsafe { glGetAttribLocation(program, cname.as_ptr() as *const _) };
            let attr_loc = if attr_loc == -1 { None } else { Some(attr_loc) };
            let divisor = if layout.step_func == VertexStep::PerVertex {
                0
            } else {
                layout.step_rate
            };

            let mut attributes_count: usize = 1;
            let mut format = *format;

            if format == VertexFormat::Mat4 {
                format = VertexFormat::Float4;
                attributes_count = 4;
            }
            for i in 0..attributes_count {
                if let Some(attr_loc) = attr_loc {
                    let attr_loc = attr_loc as GLuint + i as GLuint;

                    let attr = VertexAttributeInternal {
                        attr_loc,
                        size: format.components(),
                        type_: format.type_(),
                        offset: buffer_data.offset,
                        stride: buffer_data.stride,
                        buffer_index: *buffer_index,
                        divisor,
                        gl_pass_as_float: *gl_pass_as_float,
                    };

                    assert!(
                        attr_loc < vertex_layout.len() as u32,
                        "attribute: {} outside of allocated attributes array len: {}",
                        name,
                        vertex_layout.len()
                    );
                    vertex_layout[attr_loc as usize] = Some(attr);
                }
                buffer_data.offset += format.size_bytes() as i64
            }
        }

        let pipeline = PipelineInternal {
            layout: vertex_layout,
            shader,
            params,
        };

        Pipeline(self.pipelines.add(pipeline))
    }

    fn apply_pipeline(&mut self, pipeline: &Pipeline) {
        self.cache.cur_pipeline = Some(*pipeline);

        {
            let pipeline = &self.pipelines[pipeline.0];
            let shader = &self.shaders[pipeline.shader.0];
            unsafe {
                glUseProgram(shader.program);
            }

            unsafe {
                glEnable(GL_SCISSOR_TEST);
            }

            if pipeline.params.depth_write {
                unsafe {
                    glEnable(GL_DEPTH_TEST);
                    glDepthFunc(pipeline.params.depth_test.into())
                }
            } else {
                unsafe {
                    glDisable(GL_DEPTH_TEST);
                }
            }

            match pipeline.params.front_face_order {
                FrontFaceOrder::Clockwise => unsafe {
                    glFrontFace(GL_CW);
                },
                FrontFaceOrder::CounterClockwise => unsafe {
                    glFrontFace(GL_CCW);
                },
            }
        }

        self.set_cull_face(self.pipelines[pipeline.0].params.cull_face);
        self.set_blend(
            self.pipelines[pipeline.0].params.color_blend,
            self.pipelines[pipeline.0].params.alpha_blend,
        );

        self.set_stencil(self.pipelines[pipeline.0].params.stencil_test);
        self.set_color_write(self.pipelines[pipeline.0].params.color_write);
    }

    fn new_buffer(
        &mut self,
        type_: BufferType,
        usage: BufferUsage,
        data: BufferSource,
    ) -> BufferId {
        let gl_target = gl_buffer_target(&type_);
        let gl_usage = gl_usage(&usage);
        let (size, element_size) = match &data {
            BufferSource::Slice(data) => (data.size, data.element_size),
            BufferSource::Empty { size, element_size } => (*size, *element_size),
        };
        let index_type = match type_ {
            BufferType::IndexBuffer
                if element_size == 1 || element_size == 2 || element_size == 4 =>
            {
                Some(element_size as u32)
            }
            BufferType::IndexBuffer => panic!("unsupported index buffer dimension"),
            BufferType::VertexBuffer => None,
        };
        let mut gl_buf: u32 = 0;

        unsafe {
            glGenBuffers(1, &mut gl_buf as *mut _);
            self.cache.store_buffer_binding(gl_target);
            self.cache.bind_buffer(gl_target, gl_buf, index_type);

            glBufferData(gl_target, size as _, std::ptr::null() as *const _, gl_usage);
            if let BufferSource::Slice(data) = data {
                debug_assert!(data.is_slice);
                glBufferSubData(gl_target, 0, size as _, data.ptr as _);
            }
            self.cache.restore_buffer_binding(gl_target);
        }

        let buffer = Buffer {
            gl_buf,
            buffer_type: type_,
            size,
            index_type,
        };

        BufferId(self.buffers.add(buffer))
    }

    fn buffer_update(&mut self, buffer: BufferId, data: BufferSource) {
        let data = match data {
            BufferSource::Slice(data) => data,
            _ => panic!("buffer_update expects BufferSource::slice"),
        };
        debug_assert!(data.is_slice);
        let buffer = &self.buffers[buffer.0];

        if matches!(buffer.buffer_type, BufferType::IndexBuffer) {
            assert!(buffer.index_type.is_some());
            assert!(data.element_size as u32 == buffer.index_type.unwrap());
        };

        let size = data.size;

        assert!(size <= buffer.size);

        let gl_target = gl_buffer_target(&buffer.buffer_type);
        self.cache.store_buffer_binding(gl_target);
        self.cache
            .bind_buffer(gl_target, buffer.gl_buf, buffer.index_type);
        unsafe { glBufferSubData(gl_target, 0, size as _, data.ptr as _) };
        self.cache.restore_buffer_binding(gl_target);
    }

    /// Size of buffer in bytes
    fn buffer_size(&mut self, buffer: BufferId) -> usize {
        self.buffers[buffer.0].size
    }

    /// Delete GPU buffer, leaving handle unmodified.
    ///
    /// More high-level code on top of miniquad probably is going to call this in Drop implementation of some
    /// more RAII buffer object.
    ///
    /// There is no protection against using deleted textures later. However its not an UB in OpenGl and thats why
    /// this function is not marked as unsafe
    fn delete_buffer(&mut self, buffer: BufferId) {
        unsafe { glDeleteBuffers(1, &self.buffers[buffer.0].gl_buf as *const _) }
        self.cache.clear_buffer_bindings();
        self.cache.clear_vertex_attributes();
        self.buffers.remove(buffer.0);
    }

    /// Set a new viewport rectangle.
    /// Should be applied after begin_pass.
    fn apply_viewport(&mut self, x: i32, y: i32, w: i32, h: i32) {
        unsafe {
            glViewport(x, y, w, h);
        }
    }

    /// Set a new scissor rectangle.
    /// Should be applied after begin_pass.
    fn apply_scissor_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        unsafe {
            glScissor(x, y, w, h);
        }
    }

    fn apply_bindings_from_slice(
        &mut self,
        vertex_buffers: &[BufferId],
        index_buffer: BufferId,
        textures: &[TextureId],
    ) {
        let pip = &self.pipelines[self.cache.cur_pipeline.unwrap().0];
        let shader = &self.shaders[pip.shader.0];

        for (n, shader_image) in shader.images.iter().enumerate() {
            let bindings_image = textures
                .get(n)
                .unwrap_or_else(|| panic!("Image count in bindings and shader did not match!"));
            if let Some(gl_loc) = shader_image.gl_loc {
                let texture = self.textures.get(*bindings_image);
                let raw = match texture.raw {
                    TextureOrRenderbuffer::Texture(id) => id,
                    TextureOrRenderbuffer::Renderbuffer(id) => id,
                };
                unsafe {
                    self.cache.bind_texture(n, texture.params.kind.into(), raw);
                    glUniform1i(gl_loc, n as i32);
                }
            }
        }

        self.cache.bind_buffer(
            GL_ELEMENT_ARRAY_BUFFER,
            self.buffers[index_buffer.0].gl_buf,
            self.buffers[index_buffer.0].index_type,
        );

        let pip = &self.pipelines[self.cache.cur_pipeline.unwrap().0];

        for attr_index in 0..MAX_VERTEX_ATTRIBUTES {
            let cached_attr = &mut self.cache.attributes[attr_index];

            let pip_attribute = pip.layout.get(attr_index).copied();

            if let Some(Some(attribute)) = pip_attribute {
                assert!(
                    attribute.buffer_index < vertex_buffers.len(),
                    "Attribute index outside of vertex_buffers length"
                );
                let vb = vertex_buffers[attribute.buffer_index];
                let vb = self.buffers[vb.0];

                if cached_attr.map_or(true, |cached_attr| {
                    attribute != cached_attr.attribute || cached_attr.gl_vbuf != vb.gl_buf
                }) {
                    self.cache
                        .bind_buffer(GL_ARRAY_BUFFER, vb.gl_buf, vb.index_type);

                    unsafe {
                        match attribute.type_ {
                            GL_INT | GL_UNSIGNED_INT | GL_SHORT | GL_UNSIGNED_SHORT
                            | GL_UNSIGNED_BYTE | GL_BYTE
                                if !attribute.gl_pass_as_float =>
                            {
                                glVertexAttribIPointer(
                                    attr_index as GLuint,
                                    attribute.size,
                                    attribute.type_,
                                    attribute.stride,
                                    attribute.offset as *mut _,
                                )
                            }
                            _ => glVertexAttribPointer(
                                attr_index as GLuint,
                                attribute.size,
                                attribute.type_,
                                GL_FALSE as u8,
                                attribute.stride,
                                attribute.offset as *mut _,
                            ),
                        }
                        if self.info.features.instancing {
                            glVertexAttribDivisor(attr_index as GLuint, attribute.divisor as u32);
                        }
                        glEnableVertexAttribArray(attr_index as GLuint);
                    };

                    let cached_attr = &mut self.cache.attributes[attr_index];
                    *cached_attr = Some(CachedAttribute {
                        attribute,
                        gl_vbuf: vb.gl_buf,
                    });
                }
            } else if cached_attr.is_some() {
                unsafe {
                    glDisableVertexAttribArray(attr_index as GLuint);
                }
                *cached_attr = None;
            }
        }
    }

    fn apply_uniforms_from_bytes(&mut self, uniform_ptr: *const u8, size: usize) {
        let pip = &self.pipelines[self.cache.cur_pipeline.unwrap().0];
        let shader = &self.shaders[pip.shader.0];

        let mut offset = 0;

        for uniform in shader.uniforms.iter() {
            use UniformType::*;

            assert!(
                offset as i32 <= size as i32 - uniform.uniform_type.size() as i32 / 4,
                "Uniforms struct does not match shader uniforms layout"
            );

            unsafe {
                let data = (uniform_ptr as *const f32).add(offset);
                let data_int = (uniform_ptr as *const i32).add(offset);

                if let Some(gl_loc) = uniform.gl_loc {
                    match uniform.uniform_type {
                        Float1 => {
                            glUniform1fv(gl_loc, uniform.array_count, data);
                        }
                        Float2 => {
                            glUniform2fv(gl_loc, uniform.array_count, data);
                        }
                        Float3 => {
                            glUniform3fv(gl_loc, uniform.array_count, data);
                        }
                        Float4 => {
                            glUniform4fv(gl_loc, uniform.array_count, data);
                        }
                        Int1 => {
                            glUniform1iv(gl_loc, uniform.array_count, data_int);
                        }
                        Int2 => {
                            glUniform2iv(gl_loc, uniform.array_count, data_int);
                        }
                        Int3 => {
                            glUniform3iv(gl_loc, uniform.array_count, data_int);
                        }
                        Int4 => {
                            glUniform4iv(gl_loc, uniform.array_count, data_int);
                        }
                        Mat4 => {
                            glUniformMatrix4fv(gl_loc, uniform.array_count, 0, data);
                        }
                    }
                }
            }
            offset += uniform.uniform_type.size() / 4 * uniform.array_count as usize;
        }
    }

    fn clear(
        &mut self,
        color: Option<(f32, f32, f32, f32)>,
        depth: Option<f32>,
        stencil: Option<i32>,
    ) {
        let mut bits = 0;
        if let Some((r, g, b, a)) = color {
            bits |= GL_COLOR_BUFFER_BIT;
            unsafe {
                glClearColor(r, g, b, a);
            }
        }

        if let Some(v) = depth {
            bits |= GL_DEPTH_BUFFER_BIT;
            unsafe {
                glClearDepthf(v);
            }
        }

        if let Some(v) = stencil {
            bits |= GL_STENCIL_BUFFER_BIT;
            unsafe {
                glClearStencil(v);
            }
        }

        if bits != 0 {
            unsafe {
                glClear(bits);
            }
        }
    }

    fn begin_default_pass(&mut self, action: PassAction) {
        self.begin_pass(None, action);
    }

    fn begin_pass(&mut self, pass: Option<RenderPass>, action: PassAction) {
        self.cache.cur_pass = pass;
        let (framebuffer, w, h) = match pass {
            None => {
                let (screen_width, screen_height) = window::screen_size();

                (
                    self.default_framebuffer,
                    screen_width as i32,
                    screen_height as i32,
                )
            }
            Some(pass) => {
                let pass = &self.passes[pass.0];
                // new_render_pass will panic with both color and depth components none
                // so unwrap is safe here
                let texture = pass
                    .color_textures
                    .first()
                    .copied()
                    .or(pass.depth_texture)
                    .unwrap();
                (
                    pass.gl_fb,
                    self.textures.get(texture).params.width as i32,
                    self.textures.get(texture).params.height as i32,
                )
            }
        };
        unsafe {
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
            glViewport(0, 0, w, h);
            glScissor(0, 0, w, h);
        }
        match action {
            PassAction::Nothing => {}
            PassAction::Clear {
                color,
                depth,
                stencil,
            } => {
                self.clear(color, depth, stencil);
            }
        }
    }

    fn end_render_pass(&mut self) {
        unsafe {
            if let Some(pass) = self.cache.cur_pass.take() {
                let pass = &self.passes[pass.0];
                if let Some(resolves) = &pass.resolves {
                    glBindFramebuffer(GL_READ_FRAMEBUFFER, pass.gl_fb);
                    for (i, (resolve_fb, resolve_img)) in resolves.iter().enumerate() {
                        let texture = self.textures.get(*resolve_img);
                        let w = texture.params.width;
                        let h = texture.params.height;
                        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, *resolve_fb);
                        glReadBuffer(GL_COLOR_ATTACHMENT0 + i as u32);
                        glBlitFramebuffer(
                            0,
                            0,
                            w as _,
                            h as _,
                            0,
                            0,
                            w as _,
                            h as _,
                            GL_COLOR_BUFFER_BIT,
                            GL_NEAREST,
                        );
                    }
                }
            }
            glBindFramebuffer(GL_FRAMEBUFFER, self.default_framebuffer);
            self.cache.bind_buffer(GL_ARRAY_BUFFER, 0, None);
            self.cache.bind_buffer(GL_ELEMENT_ARRAY_BUFFER, 0, None);

            // On Android when resizing the screen, visual tearing may occur
            // on some devices since clearing the screen is clipped by scissor.
            // So disable it when cleaning up the render pass.
            glDisable(GL_SCISSOR_TEST);
        }
    }

    fn commit_frame(&mut self) {
        self.cache.clear_buffer_bindings();
        self.cache.clear_texture_bindings();
    }

    fn draw(&self, base_element: i32, num_elements: i32, num_instances: i32) {
        assert!(
            self.cache.cur_pipeline.is_some(),
            "Drawing without any binded pipeline"
        );

        if !self.info.features.instancing && num_instances != 1 {
            eprintln!("Instanced rendering is not supported by the GPU");
            eprintln!("Ignoring this draw call");
            return;
        }

        let pip = &self.pipelines[self.cache.cur_pipeline.unwrap().0];
        let primitive_type = pip.params.primitive_type.into();
        let index_type = self.cache.index_type.expect("Unset index buffer type");

        unsafe {
            glDrawElementsInstanced(
                primitive_type,
                num_elements,
                match index_type {
                    1 => GL_UNSIGNED_BYTE,
                    2 => GL_UNSIGNED_SHORT,
                    4 => GL_UNSIGNED_INT,
                    _ => panic!("Unsupported index buffer type!"),
                },
                (index_type as i32 * base_element) as *mut _,
                num_instances,
            );
        }
    }
}

#[cfg(test)]
mod transfer_tests {
    use super::*;

    #[test]
    fn transfer_sizes_match_the_layout() {
        assert_eq!(transfer_len(TextureFormat::RGB8, 1, 3, 16), 9);
        assert_eq!(transfer_len(TextureFormat::RGBA16F, 4, 1, 32), 32);
        assert_eq!(
            mip_sizes(5, 2).collect::<Vec<_>>(),
            [(5, 2), (2, 1), (1, 1)]
        );
    }

    #[test]
    #[should_panic(expected = "texture transfer buffer is too small")]
    fn a_large_transfer_cannot_wrap_to_a_small_buffer() {
        transfer_len(TextureFormat::RGBA16F, i32::MAX, i32::MAX, 64);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn widened_halves_keep_finite_values_and_special_classes() {
        for (float, half) in [
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (2f32.powi(-24), 1),
            (1023.0 * 2f32.powi(-24), 0x03ff),
            (65504.0, 0x7bff),
            (-0.0, 0x8000),
            (f32::NEG_INFINITY, 0xfc00),
            (f32::NAN, 0x7e00),
        ] {
            assert_eq!(narrow_widened_half(float), half);
        }
    }
}
