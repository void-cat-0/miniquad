//! Exercise direct Miniquad draws across sampler changes, uploads and readback.
//! `cargo run --features vulkan --example vulkan_pass_regression -- [--wayland]`

#[cfg(all(
    feature = "vulkan",
    any(windows, all(target_os = "linux", not(target_env = "ohos")))
))]
mod regression {
    use miniquad::*;

    const WIDTH: usize = 24;
    const HEIGHT: usize = 8;
    const BLACK: [u8; 4] = [0, 0, 0, 255];
    const RED: [u8; 4] = [255, 0, 0, 255];
    const GREEN: [u8; 4] = [0, 255, 0, 255];

    struct Regression(Box<dyn RenderingBackend>);

    impl EventHandler for Regression {
        fn update(&mut self) {}

        fn desktop_vulkan_context(&mut self) -> Option<&mut graphics::vulkan::VulkanContext> {
            self.0.vulkan_context()
        }

        fn draw(&mut self) {
            let ctx = self.0.as_mut();
            exercise(ctx, 1, TextureFormat::RGBA8);
            exercise(ctx, 1, TextureFormat::RGB8);
            let samples =
                ctx.render_target_sample_count(TextureFormat::RGBA8, Some(TextureFormat::Depth), 4);
            if samples > 1 {
                exercise(ctx, samples, TextureFormat::RGBA8);
            }
            ctx.commit_frame();
            eprintln!("Vulkan pass continuation checks passed (1× and requested 4× → {samples}×)");
            window::order_quit();
        }
    }

    fn exercise(ctx: &mut dyn RenderingBackend, samples: i32, source_format: TextureFormat) {
        let target_params = TextureParams {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            ..Default::default()
        };
        let target = ctx.new_render_texture(target_params);
        let color = if samples > 1 {
            ctx.new_render_texture(TextureParams {
                sample_count: samples,
                ..target_params
            })
        } else {
            target
        };
        let depth = ctx.new_render_texture(TextureParams {
            format: TextureFormat::Depth,
            sample_count: samples,
            ..target_params
        });
        let resolves = [target];
        let pass = ctx.new_render_pass_mrt(
            &[color],
            (samples > 1).then_some(resolves.as_slice()),
            Some(depth),
        );

        let red_blue = match source_format {
            TextureFormat::RGB8 => &[255, 0, 0, 0, 0, 255][..],
            _ => &[255, 0, 0, 255, 0, 0, 255, 255][..],
        };
        let green = match source_format {
            TextureFormat::RGB8 => &[0, 255, 0, 0, 255, 0][..],
            _ => &[0, 255, 0, 255, 0, 255, 0, 255][..],
        };
        let source = ctx.new_texture_from_data_and_format(
            red_blue,
            TextureParams {
                width: 2,
                height: 1,
                format: source_format,
                ..Default::default()
            },
        );

        let shader = ctx
            .new_shader(
                ShaderSource::Glsl {
                    vertex: "#version 100\n\
                        attribute vec2 position;\n\
                        uniform float z;\n\
                        void main() { gl_Position = vec4(position, z, 1.0); }",
                    fragment: "#version 100\n\
                        precision mediump float;\n\
                        uniform sampler2D tex;\n\
                        void main() { gl_FragColor = texture2D(tex, vec2(0.375, 0.5)); }",
                },
                ShaderMeta {
                    images: vec!["tex".into()],
                    uniforms: UniformBlockLayout {
                        uniforms: vec![UniformDesc::new("z", UniformType::Float1)],
                    },
                },
            )
            .unwrap();
        let pipeline = ctx.new_pipeline(
            &[BufferLayout::default()],
            &[VertexAttribute::new("position", VertexFormat::Float2)],
            shader,
            PipelineParams {
                depth_write: true,
                depth_test: Comparison::Less,
                ..Default::default()
            },
        );
        let vertices: [[f32; 2]; 16] = std::array::from_fn(|i| {
            let left = -1. + (i / 4) as f32 * 0.5;
            let right = left + 0.5;
            [[left, -1.], [right, -1.], [right, 1.], [left, 1.]][i % 4]
        });
        let indices: [u16; 24] =
            std::array::from_fn(|i| (i / 6 * 4) as u16 + [0, 1, 2, 0, 2, 3][i % 6]);
        let vertices = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&vertices),
        );
        let indices = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&indices),
        );

        ctx.begin_pass(Some(pass), PassAction::clear_color(0., 0., 0., 1.));
        ctx.apply_viewport(2, 0, 20, HEIGHT as i32);
        ctx.apply_scissor_rect(3, 2, 18, 4);
        ctx.apply_pipeline(&pipeline);
        ctx.apply_bindings_from_slice(&[vertices], indices, &[source]);
        ctx.apply_uniforms(UniformsSource::table(&-0.5f32));
        ctx.draw(0, 6, 1);

        // Replace the sampler without ending or rebinding the pass. The first
        // draw must keep its linear state; the second must use nearest.
        ctx.texture_set_filter(source, FilterMode::Nearest, MipmapFilterMode::None);
        ctx.draw(6, 6, 1);

        let mut pixels = [[0; 4]; WIDTH * HEIGHT];
        let pixel = |pixels: &[[u8; 4]; WIDTH * HEIGHT], x, y| pixels[y * WIDTH + x];
        ctx.texture_read_pixels(target, pixels.as_flattened_mut());
        let linear = pixel(&pixels, 4, 4);
        assert!(
            linear[0] > 160 && linear[2] > 40,
            "linear sample: {:?}",
            linear
        );
        assert_eq!(pixel(&pixels, 9, 4), RED, "next draw sees the new sampler");

        // A nested transfer (with either the frame upload arena or synchronous
        // RGB8 staging) cannot disturb the caller's state or overwrite earlier draws.
        ctx.texture_update(source, green);
        ctx.draw(12, 6, 1);
        ctx.texture_update(source, red_blue);
        ctx.apply_uniforms(UniformsSource::table(&0.5f32));
        ctx.draw(0, 6, 1); // Rejected by the depth from the first segment.
        ctx.texture_read_pixels(target, pixels.as_flattened_mut()); // Leave the pass suspended.
        ctx.end_render_pass();
        ctx.clear(Some((0., 0., 1., 1.)), None, None); // The window, not the ended target.
        ctx.end_render_pass();
        ctx.texture_read_pixels(target, pixels.as_flattened_mut());

        assert_eq!(
            pixel(&pixels, 4, 4),
            linear,
            "color and depth survive reopening"
        );
        assert_eq!(pixel(&pixels, 9, 4), RED, "pre-upload draw stays red");
        assert_eq!(
            pixel(&pixels, 14, 4),
            GREEN,
            "post-upload draw becomes green"
        );
        assert_eq!(pixel(&pixels, 2, 4), BLACK, "scissor survives reopening");
        assert_eq!(
            pixel(&pixels, 14, 1),
            BLACK,
            "vertical scissor survives reopening"
        );
        assert_eq!(pixel(&pixels, 19, 4), BLACK, "end releases the target");

        ctx.delete_render_pass(pass);
        ctx.delete_texture(target);
        if samples > 1 {
            ctx.delete_texture(color);
        }
        ctx.delete_texture(depth);
        ctx.delete_texture(source);
        ctx.delete_pipeline(pipeline);
        ctx.delete_shader(shader);
        ctx.delete_buffer(vertices);
        ctx.delete_buffer(indices);
    }

    pub fn run() {
        miniquad::start(
            conf::Conf {
                window_title: "Vulkan pass continuation".into(),
                window_width: 96,
                window_height: 96,
                platform: conf::Platform {
                    rendering_backend: conf::RenderingBackend::Vulkan,
                    linux_backend: if std::env::args().any(|arg| arg == "--wayland") {
                        conf::LinuxBackend::WaylandOnly
                    } else {
                        conf::LinuxBackend::X11Only
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
            || Box::new(Regression(window::new_rendering_backend())),
        );
    }
}

fn main() {
    #[cfg(all(
        feature = "vulkan",
        any(windows, all(target_os = "linux", not(target_env = "ohos")))
    ))]
    regression::run();
}
