//! Check Vulkan pass continuation, multiple render targets, depth and attachment resize.
//! `cargo run --features vulkan --example vulkan_pass_regression -- [--wayland] [--resize-window] [--expect-rendering=legacy|khr|core]`

#[cfg(all(
    feature = "vulkan",
    any(windows, all(target_os = "linux", not(target_env = "ohos")))
))]
mod regression {
    use miniquad::{graphics::vulkan::VulkanRenderingMode, *};

    const WIDTH: usize = 24;
    const HEIGHT: usize = 8;
    const BLACK: [u8; 4] = [0, 0, 0, 255];
    const RED: [u8; 4] = [255, 0, 0, 255];
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 255];
    const WHITE: [u8; 4] = [255, 255, 255, 255];

    struct Regression {
        ctx: Box<dyn RenderingBackend>,
        resize_window: bool,
        initial_extent: Option<(u32, u32)>,
    }

    impl Regression {
        fn check_window_resize(&mut self) -> bool {
            let extent = self.ctx.vulkan_context().unwrap().swapchain_extent;
            let size = (extent.width, extent.height);
            if self.initial_extent == Some(size) {
                return false;
            }
            self.ctx.begin_default_pass(PassAction::default());
            self.ctx.end_render_pass();
            self.ctx.commit_frame();
            if self.initial_extent.is_none() {
                self.initial_extent = Some(size);
                window::set_window_size(160, 128);
                false
            } else {
                eprintln!("Vulkan swapchain resize check passed: {size:?}");
                true
            }
        }
    }

    impl EventHandler for Regression {
        fn update(&mut self) {}

        fn desktop_vulkan_context(&mut self) -> Option<&mut graphics::vulkan::VulkanContext> {
            self.ctx.vulkan_context()
        }

        fn draw(&mut self) {
            if self.resize_window && !self.check_window_resize() {
                return;
            }
            let ctx = self.ctx.as_mut();
            exercise(ctx, 1, TextureFormat::RGBA8);
            exercise(ctx, 1, TextureFormat::RGB8);
            exercise_depth_only(ctx);
            exercise_mrt_resize(ctx, 1);
            let samples =
                ctx.render_target_sample_count(TextureFormat::RGBA8, Some(TextureFormat::Depth), 4);
            if samples > 1 {
                exercise(ctx, samples, TextureFormat::RGBA8);
                exercise_mrt_resize(ctx, samples);
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

    fn exercise_depth_only(ctx: &mut dyn RenderingBackend) {
        // Depth textures can also be attachments when created through the
        // generic texture API. The framebuffer must use only level zero.
        let depth = ctx.new_texture(
            TextureAccess::Static,
            TextureSource::Empty,
            TextureParams {
                width: 4,
                height: 4,
                format: TextureFormat::Depth,
                allocate_mipmaps: true,
                ..Default::default()
            },
        );
        let pass = ctx.new_render_pass_mrt(&[], None, Some(depth));
        ctx.begin_pass(
            Some(pass),
            PassAction::Clear {
                color: None,
                depth: Some(1.0),
                stencil: None,
            },
        );
        ctx.end_render_pass();
        let mut pixels = [0; 4 * 4 * 2];
        ctx.texture_read_pixels(depth, &mut pixels);
        assert_eq!(pixels, [255; 4 * 4 * 2], "depth-only clear");
        ctx.delete_render_pass(pass);
        ctx.delete_texture(depth);
    }

    fn exercise_mrt_resize(ctx: &mut dyn RenderingBackend, samples: i32) {
        let params = TextureParams {
            width: 8,
            height: 4,
            ..Default::default()
        };
        let targets = [(); 2].map(|_| ctx.new_render_texture(params));
        let color_params = TextureParams {
            sample_count: samples,
            ..params
        };
        let colors = if samples > 1 {
            [(); 2].map(|_| ctx.new_render_texture(color_params))
        } else {
            targets
        };
        let depth = ctx.new_render_texture(TextureParams {
            format: TextureFormat::Depth,
            ..color_params
        });
        let resolves = (samples > 1).then_some(targets.as_slice());
        let pass = ctx.new_render_pass_mrt(&colors, resolves, Some(depth));
        let attachments = [colors.as_slice(), resolves.unwrap_or(&[]), &[depth]].concat();

        let shader = ctx
            .new_shader(
                ShaderSource::Glsl {
                    vertex: "#version 330 core\n\
                        in vec2 position;\n\
                        uniform float z;\n\
                        void main() { gl_Position = vec4(position, z, 1.0); }",
                    fragment: "#version 330 core\n\
                        uniform float z;\n\
                        layout(location = 0) out vec4 color0;\n\
                        layout(location = 1) out vec4 color1;\n\
                        void main() {\n\
                            color0 = z < 0.0 ? vec4(1, 0, 0, 1) : vec4(0, 0, 1, 1);\n\
                            color1 = z < 0.0 ? vec4(0, 1, 0, 1) : vec4(1, 1, 1, 1);\n\
                        }",
                },
                ShaderMeta {
                    images: vec![],
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
        let vertices = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&[[-1f32, -1.], [1., -1.], [1., 1.], [-1., 1.]]),
        );
        let indices = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&[0u16, 1, 2, 0, 2, 3]),
        );

        for (width, height) in [(params.width, params.height), (14, 10)] {
            if (width, height) != (params.width, params.height) {
                // Replace the images without changing their IDs, the pass or the pipeline.
                for &texture in &attachments {
                    ctx.texture_resize(texture, width, height, None);
                }
            }
            let mut pixels = vec![[0; 4]; (width * height) as usize];
            ctx.begin_pass(Some(pass), PassAction::clear_color(0., 0., 0., 1.));
            ctx.apply_pipeline(&pipeline);
            ctx.apply_bindings_from_slice(&[vertices], indices, &[]);
            ctx.apply_scissor_rect(0, 0, width as i32 / 2, height as i32);
            ctx.apply_uniforms(UniformsSource::table(&-0.5f32));
            ctx.draw(0, 6, 1);

            // Suspend the pass. Both colors and depth must survive reopening.
            ctx.texture_read_pixels(targets[0], pixels.as_flattened_mut());
            // The far quad covers the whole target, but must fail depth on the left.
            ctx.apply_scissor_rect(0, 0, width as i32, height as i32);
            ctx.apply_uniforms(UniformsSource::table(&0.5f32));
            ctx.draw(0, 6, 1);
            ctx.end_render_pass();

            for (attachment, (target, (near, far))) in targets
                .into_iter()
                .zip([(RED, BLUE), (GREEN, WHITE)])
                .enumerate()
            {
                pixels.fill([0; 4]);
                ctx.texture_read_pixels(target, pixels.as_flattened_mut());
                for (y, row) in pixels.chunks_exact(width as usize).enumerate() {
                    for (x, &pixel) in row.iter().enumerate() {
                        let expected = if x < width as usize / 2 { near } else { far };
                        assert_eq!(
                            pixel, expected,
                            "{samples}× MRT attachment {attachment}, {width}×{height} pixel ({x}, {y})"
                        );
                    }
                }
            }
        }

        ctx.delete_render_pass(pass);
        for texture in attachments {
            ctx.delete_texture(texture);
        }
        ctx.delete_pipeline(pipeline);
        ctx.delete_shader(shader);
        ctx.delete_buffer(vertices);
        ctx.delete_buffer(indices);
    }

    pub fn run() {
        let mut platform = conf::Platform {
            rendering_backend: conf::RenderingBackend::Vulkan,
            linux_backend: conf::LinuxBackend::X11Only,
            ..Default::default()
        };
        let mut expected_rendering = None;
        let mut resize_window = false;
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--wayland" => platform.linux_backend = conf::LinuxBackend::WaylandOnly,
                "--resize-window" => resize_window = true,
                "--expect-rendering=legacy" => expected_rendering = Some("legacy"),
                "--expect-rendering=khr" => expected_rendering = Some("khr"),
                "--expect-rendering=core" => expected_rendering = Some("core"),
                _ => panic!("unknown option: {arg}"),
            }
        }
        if resize_window && platform.linux_backend == conf::LinuxBackend::WaylandOnly {
            panic!("the window resize regression requires X11 or Windows");
        }
        miniquad::start(
            conf::Conf {
                window_title: "Vulkan pass continuation".into(),
                window_width: 96,
                window_height: 96,
                platform,
                ..Default::default()
            },
            move || {
                let mut ctx = window::new_rendering_backend();
                let rendering = match ctx.vulkan_context().unwrap().rendering_mode() {
                    VulkanRenderingMode::Legacy => "legacy",
                    VulkanRenderingMode::DynamicKhr => "khr",
                    VulkanRenderingMode::DynamicCore => "core",
                };
                if let Some(expected) = expected_rendering {
                    assert_eq!(rendering, expected, "Vulkan rendering mode");
                }
                eprintln!("Vulkan rendering mode: {rendering}");
                Box::new(Regression {
                    ctx,
                    resize_window,
                    initial_extent: None,
                })
            },
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
