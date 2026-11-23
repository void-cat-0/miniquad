//! Desktop OpenGL 3.3+ transfer regression: `cargo run --example texture_readback`.

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    use miniquad::{conf, window, EventHandler};

    struct Stage;
    impl EventHandler for Stage {
        fn update(&mut self) {}
        fn draw(&mut self) {
            window::quit();
        }
    }

    let mut conf = conf::Conf::default();
    conf.window_title = "Texture transfer checks".to_owned();
    conf.platform.linux_backend = conf::LinuxBackend::X11Only;
    miniquad::start(conf, || {
        let mut ctx = miniquad::GlContext::new();
        println!(
            "Testing {}",
            miniquad::RenderingBackend::info(&ctx).gl_version_string
        );
        check_transfers(&mut ctx);
        check_resolves(&mut ctx);
        check_size_limit();
        println!("OpenGL texture allocation, transfer and resolve checks passed");
        Box::new(Stage)
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn check_transfers(ctx: &mut dyn miniquad::RenderingBackend) {
    use miniquad::{gl::*, TextureFormat, TextureId, TextureParams, TextureSource};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn integer(param: GLenum) -> GLint {
        let mut value = 0;
        unsafe { glGetIntegerv(param, &mut value) };
        value
    }

    let alignment_and_offsets = [
        (GL_PACK_ALIGNMENT, 8),
        (GL_PACK_ROW_LENGTH, 5),
        (GL_PACK_SKIP_PIXELS, 1),
        (GL_PACK_SKIP_ROWS, 2),
        (GL_PACK_SWAP_BYTES, 1),
        (GL_UNPACK_ALIGNMENT, 8),
        (GL_UNPACK_ROW_LENGTH, 5),
        (GL_UNPACK_SKIP_PIXELS, 1),
        (GL_UNPACK_SKIP_ROWS, 2),
        (GL_UNPACK_SWAP_BYTES, 1),
    ];
    let mut fbos = [0; 2];
    let mut pbos = [0; 2];
    let mut foreign_texture = 0;
    unsafe {
        glGenFramebuffers(2, fbos.as_mut_ptr());
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbos[0]);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbos[1]);
        glGenBuffers(2, pbos.as_mut_ptr());
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[0]);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbos[1]);
        glActiveTexture(GL_TEXTURE0 + 2);
        glGenTextures(1, &mut foreign_texture);
        glBindTexture(GL_TEXTURE_2D, foreign_texture);
        for (param, value) in alignment_and_offsets {
            glPixelStorei(param, value);
        }
    }
    let check_state = || {
        for (param, value) in alignment_and_offsets {
            assert_eq!(integer(param), value, "pixel store {param:#x}");
        }
        assert_eq!(integer(GL_TEXTURE_BINDING_2D), foreign_texture as GLint);
        assert_eq!(integer(GL_READ_FRAMEBUFFER_BINDING), fbos[1] as GLint);
        assert_eq!(integer(GL_DRAW_FRAMEBUFFER_BINDING), fbos[0] as GLint);
        assert_eq!(integer(GL_PIXEL_PACK_BUFFER_BINDING), pbos[0] as GLint);
        assert_eq!(integer(GL_PIXEL_UNPACK_BUFFER_BINDING), pbos[1] as GLint);
        assert_eq!(unsafe { glGetError() }, GL_NO_ERROR);
    };
    let rgb = TextureParams {
        format: TextureFormat::RGB8,
        width: 1,
        height: 2,
        ..Default::default()
    };
    let tex = ctx.new_texture_from_data_and_format(&[1, 2, 3, 4, 5, 6], rgb);
    check_state();
    ctx.texture_update_part(tex, 0, 1, 1, 1, &[7, 8, 9]);
    check_state();
    let mut bytes = [0xee; 64];
    ctx.texture_read_pixels(tex, &mut bytes);
    assert_eq!(&bytes[..6], &[1, 2, 3, 7, 8, 9]);
    assert_eq!(bytes[6..], [0xee; 58]);
    check_state();
    assert!(catch_unwind(AssertUnwindSafe(
        || ctx.texture_read_pixels(tex, &mut [0; 5])
    ))
    .is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| ctx.texture_resize(
        tex,
        2,
        2,
        Some(&[0; 11])
    )))
    .is_err());
    let device_limit = integer(GL_MAX_TEXTURE_SIZE) as u32;
    assert_eq!(ctx.max_texture_size(), Some(device_limit));
    for (width, height) in [(0, 2), (1, device_limit + 1), (u32::MAX, 1)] {
        assert!(!ctx.try_texture_resize(tex, width, height, None));
    }
    let raw = unsafe { ctx.texture_raw_id(tex) };
    let raw_texture = TextureId::from_raw_id(raw);
    assert!(!ctx.try_texture_resize(raw_texture, 1, 3, None));
    assert!(catch_unwind(AssertUnwindSafe(|| ctx.texture_params(raw_texture))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(
        || ctx.texture_read_pixels(raw_texture, &mut bytes)
    ))
    .is_err());

    // A preexisting GL error must be rejected before changing the texture.
    unsafe { glBindTexture(GL_NONE, 0) };
    assert!(!ctx.try_texture_resize(tex, 1, 3, None));
    assert_eq!(ctx.texture_size(tex), (1, 2));
    ctx.texture_read_pixels(tex, &mut bytes);
    assert_eq!(&bytes[..6], &[1, 2, 3, 7, 8, 9]);
    check_state();

    // A fresh object has no previous storage to preserve. Do not confuse an
    // older GL error with a failure of the new allocation.
    unsafe { glBindTexture(GL_NONE, 0) };
    let fresh = ctx.new_texture_from_rgba8(1, 1, &[37; 4]);
    check_state();

    let new_pixels = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    assert!(ctx.try_texture_resize(tex, 1, 3, Some(&new_pixels)));
    assert_eq!(ctx.texture_size(tex), (1, 3));
    assert_eq!(unsafe { ctx.texture_raw_id(tex) }, raw);
    ctx.texture_read_pixels(tex, &mut bytes);
    assert_eq!(&bytes[..9], &new_pixels);
    assert_eq!(bytes[9..], [0xee; 55]);
    check_state();

    let float = TextureParams {
        format: TextureFormat::RGBA16F,
        width: 1,
        height: 1,
        ..Default::default()
    };
    let half = [0x3c00u16, 0xc000, 0x3400, 0x3800]
        .map(u16::to_ne_bytes)
        .concat();
    let half_texture = ctx.new_texture_from_data_and_format(&half, float);
    let mut half_read = [0xee; 16];
    ctx.texture_read_pixels(half_texture, &mut half_read);
    assert_eq!(&half_read[..8], half.as_slice());
    assert_eq!(half_read[8..], [0xee; 8]);
    check_state();

    let depth = TextureParams {
        format: TextureFormat::Depth,
        ..float
    };
    let depth_bytes = u16::MAX.to_ne_bytes();
    let depth_texture = ctx.new_texture_from_data_and_format(&depth_bytes, depth);
    let mut depth_read = [0; 2];
    ctx.texture_read_pixels(depth_texture, &mut depth_read);
    assert_eq!(depth_read, depth_bytes);
    check_state();

    let mipmapped = ctx.new_texture(
        miniquad::TextureAccess::Static,
        TextureSource::Array(&[&[&[0x81; 16], &[0; 4]]]),
        TextureParams {
            width: 2,
            height: 2,
            ..Default::default()
        },
    );
    let mut rgba = [0; 16];
    ctx.texture_read_pixels(mipmapped, &mut rgba);
    assert_eq!(rgba, [0x81; 16]);
    check_state();

    for texture in [tex, fresh, half_texture, depth_texture, mipmapped] {
        ctx.delete_texture(texture);
    }
    unsafe {
        glDeleteTextures(1, &foreign_texture);
        glDeleteBuffers(2, pbos.as_ptr());
        glDeleteFramebuffers(2, fbos.as_ptr());
        glActiveTexture(GL_TEXTURE0);
        for (param, _) in alignment_and_offsets {
            glPixelStorei(
                param,
                if matches!(param, GL_PACK_ALIGNMENT | GL_UNPACK_ALIGNMENT) {
                    4
                } else {
                    0
                },
            );
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn check_resolves(ctx: &mut dyn miniquad::RenderingBackend) {
    use miniquad::{gl::*, PassAction, TextureFormat, TextureParams};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let params = TextureParams {
        width: 2,
        height: 2,
        ..Default::default()
    };
    if ctx.info().features.resolve_attachments {
        // Valid dimensions, but the driver's allocation call must reject too many samples.
        let mut foreign_renderbuffer = 0;
        let mut max_samples = 0;
        unsafe {
            glGenRenderbuffers(1, &mut foreign_renderbuffer);
            glBindRenderbuffer(GL_RENDERBUFFER, foreign_renderbuffer);
            glGetIntegerv(GL_MAX_SAMPLES, &mut max_samples);
        }
        assert!(catch_unwind(AssertUnwindSafe(|| {
            ctx.new_render_texture(TextureParams {
                sample_count: max_samples + 1,
                ..params
            })
        }))
        .is_err());
        unsafe {
            let mut binding = 0;
            glGetIntegerv(GL_RENDERBUFFER_BINDING, &mut binding);
            assert_eq!(binding, foreign_renderbuffer as GLint);
            assert_eq!(glGetError(), GL_NO_ERROR);
            glDeleteRenderbuffers(1, &foreign_renderbuffer);
        }
    }
    let resolved = [
        ctx.new_render_texture(params),
        ctx.new_render_texture(params),
    ];
    let pass = ctx.new_render_pass(resolved[0], None);
    assert_eq!(ctx.render_pass_resolve_texture(pass), None);
    ctx.delete_render_pass(pass);

    let sample_count = ctx.render_target_sample_count(params.format, Some(TextureFormat::Depth), 4);
    if sample_count > 1 {
        let params = TextureParams {
            sample_count,
            ..params
        };
        let colors = [
            ctx.new_render_texture(params),
            ctx.new_render_texture(params),
        ];
        let depth = ctx.new_render_texture(TextureParams {
            format: TextureFormat::Depth,
            ..params
        });
        let color_samples = ctx.texture_params(colors[0]).sample_count;
        assert!(color_samples >= sample_count);
        assert_eq!(ctx.texture_params(depth).sample_count, color_samples);

        let single = ctx.new_render_pass_mrt(&colors[..1], Some(&resolved[..1]), Some(depth));
        assert_eq!(ctx.render_pass_resolve_texture(single), Some(resolved[0]));
        ctx.begin_pass(Some(single), PassAction::clear_color(1., 0., 0., 1.));
        ctx.end_render_pass();
        ctx.delete_render_pass(single);
        let mut bytes = [0; 16];
        ctx.texture_read_pixels(resolved[0], &mut bytes);
        assert_eq!(bytes, [255, 0, 0, 255].repeat(4).as_slice());

        let multiple = ctx.new_render_pass_mrt(&colors, Some(&resolved), Some(depth));
        assert_eq!(ctx.render_pass_resolve_texture(multiple), None);
        ctx.begin_pass(Some(multiple), PassAction::Nothing);
        unsafe {
            glClearBufferfv(GL_COLOR, 0, [0., 1., 0., 1.].as_ptr());
            glClearBufferfv(GL_COLOR, 1, [0., 0., 1., 1.].as_ptr());
        }
        ctx.end_render_pass();
        ctx.delete_render_pass(multiple);
        for (&texture, pixel) in resolved.iter().zip([[0, 255, 0, 255], [0, 0, 255, 255]]) {
            ctx.texture_read_pixels(texture, &mut bytes);
            assert_eq!(bytes, pixel.repeat(4).as_slice());
        }
        for texture in [colors[0], colors[1], depth] {
            ctx.delete_texture(texture);
        }
    }
    for texture in resolved {
        ctx.delete_texture(texture);
    }
    assert_eq!(unsafe { glGetError() }, GL_NO_ERROR);
}

#[cfg(not(target_arch = "wasm32"))]
fn check_size_limit() {
    use miniquad::{gl::*, window, TextureKind, TextureParams};
    use std::num::NonZeroU32;

    let mut ctx =
        window::new_rendering_backend_with_texture_size_limit(NonZeroU32::new(4).unwrap());
    assert_eq!(ctx.max_texture_size(), Some(4));
    let params = TextureParams {
        width: 4,
        height: 4,
        ..Default::default()
    };
    for (width, height, sample_count) in [(0, 4, 1), (4, 5, 1), (5, 4, 2)] {
        assert!(ctx
            .try_new_render_texture(TextureParams {
                width,
                height,
                sample_count,
                ..params
            })
            .is_none());
    }
    let cube = TextureParams {
        kind: TextureKind::CubeMap,
        ..params
    };
    assert!(ctx
        .try_new_render_texture(TextureParams {
            width: 5,
            height: 5,
            ..cube
        })
        .is_none());
    // Both axes are in range, but GL rejects non-square cubemap storage.
    assert!(ctx
        .try_new_render_texture(TextureParams { height: 2, ..cube })
        .is_none());
    assert_eq!(unsafe { glGetError() }, GL_NO_ERROR);

    let tex = ctx.try_new_render_texture(params).unwrap();
    assert!(!ctx.try_texture_resize(tex, 5, 1, None));
    assert_eq!(ctx.texture_size(tex), (4, 4));
    ctx.texture_resize(tex, 1, 4, None);
    assert_eq!(ctx.texture_size(tex), (1, 4));
    ctx.delete_texture(tex);
}

#[cfg(target_arch = "wasm32")]
fn main() {}
