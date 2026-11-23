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
        println!("OpenGL texture transfer checks passed");
        Box::new(Stage)
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn check_transfers(ctx: &mut dyn miniquad::RenderingBackend) {
    use miniquad::{gl::*, TextureFormat, TextureParams, TextureSource};
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
    assert_eq!(ctx.texture_size(tex), (1, 2));
    check_state();
    ctx.texture_resize(tex, 1, 3, Some(&[1, 2, 3, 4, 5, 6, 7, 8, 9]));
    ctx.texture_read_pixels(tex, &mut bytes);
    assert_eq!(&bytes[..9], &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
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

    for texture in [tex, half_texture, depth_texture, mipmapped] {
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

#[cfg(target_arch = "wasm32")]
fn main() {}
