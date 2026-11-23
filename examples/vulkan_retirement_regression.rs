//! Draw from Vulkan buffers before and after retiring them, including slot reuse.
//! `cargo run --features vulkan --example vulkan_retirement_regression -- [--wayland] [--offscreen]`

#[cfg(all(
    feature = "vulkan",
    any(windows, all(target_os = "linux", not(target_env = "ohos")))
))]
mod regression {
    use miniquad::{graphics::vulkan::MAX_FRAMES_IN_FLIGHT, *};

    const WIDTH: usize = 16;
    const HEIGHT: usize = 8;
    const FRAMES: usize = MAX_FRAMES_IN_FLIGHT * 3;
    const BLACK: [u8; 4] = [0, 0, 0, 255];
    const RED: [u8; 4] = [255, 0, 0, 255];
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    const BLUE: [u8; 4] = [0, 0, 255, 255];
    const WHITE: [u8; 4] = [255, 255, 255, 255];

    fn frame_colors(frame: usize) -> [([u8; 4], [u8; 4]); 2] {
        if frame % 2 == 0 {
            [(RED, BLACK), (BLACK, GREEN)]
        } else {
            [(BLUE, BLACK), (BLACK, WHITE)]
        }
    }

    #[derive(Debug, PartialEq)]
    struct Mesh {
        vertices: BufferId,
        indices: BufferId,
    }

    impl Mesh {
        fn new(ctx: &mut dyn RenderingBackend, colors: ([u8; 4], [u8; 4]), right: bool) -> Self {
            let (left_color, right_color) = colors;
            let vertices: [[f32; 5]; 8] = std::array::from_fn(|i| {
                let (left, color) = if i < 4 {
                    (-1., left_color)
                } else {
                    (0., right_color)
                };
                let [x, y] = [[left, -1.], [left + 1., -1.], [left + 1., 1.], [left, 1.]][i % 4];
                let [r, g, b, _] = color.map(|c| f32::from(c) / 255.);
                [x, y, r, g, b]
            });
            let offset = if right { 4 } else { 0 };
            let indices = [0u16, 1, 2, 0, 2, 3].map(|index| index + offset);
            Self {
                vertices: ctx.new_buffer(
                    BufferType::VertexBuffer,
                    BufferUsage::Immutable,
                    BufferSource::slice(&vertices),
                ),
                indices: ctx.new_buffer(
                    BufferType::IndexBuffer,
                    BufferUsage::Immutable,
                    BufferSource::slice(&indices),
                ),
            }
        }

        fn draw(&self, ctx: &mut dyn RenderingBackend, pipeline: &Pipeline) {
            ctx.apply_pipeline(pipeline);
            ctx.apply_bindings_from_slice(&[self.vertices], self.indices, &[]);
            ctx.draw(0, 6, 1);
        }

        fn delete(&self, ctx: &mut dyn RenderingBackend) {
            // LIFO reuse returns the same IDs when the next mesh is allocated.
            ctx.delete_buffer(self.indices);
            ctx.delete_buffer(self.vertices);
        }
    }

    struct Regression {
        ctx: Box<dyn RenderingBackend>,
        pipeline: Pipeline,
        targets: [[TextureId; 2]; FRAMES],
        frame: usize,
        offscreen: bool,
    }

    impl Regression {
        fn new(offscreen: bool) -> Self {
            let mut ctx = window::new_rendering_backend();
            if offscreen {
                ctx.vulkan_context().unwrap().enable_offscreen_video();
            }
            let shader = ctx
                .new_shader(
                    ShaderSource::Glsl {
                        vertex: "#version 100\n\
                            attribute vec2 position;\n\
                            attribute vec3 color;\n\
                            varying vec3 v_color;\n\
                            void main() {\n\
                                gl_Position = vec4(position, 0., 1.);\n\
                                v_color = color;\n\
                            }",
                        fragment: "#version 100\n\
                            precision mediump float;\n\
                            varying vec3 v_color;\n\
                            void main() { gl_FragColor = vec4(v_color, 1.); }",
                    },
                    ShaderMeta {
                        images: vec![],
                        uniforms: UniformBlockLayout { uniforms: vec![] },
                    },
                )
                .unwrap();
            let pipeline = ctx.new_pipeline(
                &[BufferLayout::default()],
                &[
                    VertexAttribute::new("position", VertexFormat::Float2),
                    VertexAttribute::new("color", VertexFormat::Float3),
                ],
                shader,
                PipelineParams::default(),
            );
            // Preallocate the readback targets so their creation cannot drain
            // the graphics queue between the retirements under test.
            let targets = std::array::from_fn(|_| {
                std::array::from_fn(|_| {
                    ctx.new_render_texture(TextureParams {
                        width: WIDTH as u32,
                        height: HEIGHT as u32,
                        ..Default::default()
                    })
                })
            });
            Self {
                ctx,
                pipeline,
                targets,
                frame: 0,
                offscreen,
            }
        }

        fn check_pixels(&mut self) {
            let mut pixels = [[0; 4]; WIDTH * HEIGHT];
            for (frame, targets) in self.targets.iter().enumerate() {
                for (&target, (left, right)) in targets.iter().zip(frame_colors(frame)) {
                    self.ctx
                        .texture_read_pixels(target, pixels.as_flattened_mut());
                    assert_eq!(
                        pixels[HEIGHT / 2 * WIDTH + WIDTH / 4],
                        left,
                        "frame {frame}: left"
                    );
                    assert_eq!(
                        pixels[HEIGHT / 2 * WIDTH + WIDTH * 3 / 4],
                        right,
                        "frame {frame}: right"
                    );
                }
            }
        }
    }

    impl EventHandler for Regression {
        fn update(&mut self) {}

        fn desktop_vulkan_context(&mut self) -> Option<&mut graphics::vulkan::VulkanContext> {
            self.ctx.vulkan_context()
        }

        fn draw(&mut self) {
            let ctx = self.ctx.as_mut();
            if self.offscreen {
                ctx.vulkan_context().unwrap().begin_offscreen_frame();
            }
            let [left, right] = self.targets[self.frame];
            let [left_colors, right_colors] = frame_colors(self.frame);
            let first = Mesh::new(ctx, left_colors, false);
            let pass = ctx.new_render_pass(left, None);
            let unrelated = ctx.new_render_pass(left, None);
            let clear = || PassAction::clear_color(0., 0., 0., 1.);
            ctx.begin_pass(Some(pass), clear());
            first.draw(ctx, &self.pipeline);
            first.delete(ctx);
            assert!(
                ctx.vulkan_context().unwrap().pass_open,
                "retiring vertex/index buffers must not force a submission"
            );

            let second = Mesh::new(ctx, right_colors, true);
            assert_eq!(first, second, "exercise recycled logical buffer IDs");
            ctx.delete_render_pass(pass);
            assert!(!ctx
                .vulkan_context()
                .unwrap()
                .render_pass_is_open(Some(pass)));
            let second_pass = ctx.new_render_pass(right, None);
            assert_eq!(pass, second_pass, "exercise recycled logical pass ID");
            ctx.begin_pass(Some(second_pass), clear());
            ctx.delete_render_pass(unrelated);
            assert!(
                ctx.vulkan_context().unwrap().pass_open,
                "deleting an unrelated pass must not interrupt live rendering"
            );
            second.draw(ctx, &self.pipeline);
            ctx.end_render_pass();
            if !self.offscreen {
                ctx.begin_default_pass(clear());
                ctx.end_render_pass();
            }
            ctx.commit_frame();
            // With the submission worker, the frame can still be waiting to
            // enter the graphics queue; the next slot cannot prove completion.
            second.delete(ctx);
            ctx.delete_render_pass(second_pass);
            self.frame += 1;
            if self.frame == FRAMES {
                if self.offscreen {
                    ctx.vulkan_context().unwrap().wait_offscreen_idle();
                }
                self.check_pixels();
                println!("Vulkan retirement checks passed");
                window::order_quit();
            }
        }
    }

    pub fn run() {
        let mut platform = conf::Platform {
            rendering_backend: conf::RenderingBackend::Vulkan,
            linux_backend: conf::LinuxBackend::X11Only,
            ..Default::default()
        };
        let mut offscreen = false;
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--wayland" => platform.linux_backend = conf::LinuxBackend::WaylandOnly,
                "--offscreen" => offscreen = true,
                _ => panic!("unknown option: {}", arg),
            }
        }
        miniquad::start(
            conf::Conf {
                window_title: "Vulkan resource retirement".into(),
                window_width: 96,
                window_height: 96,
                platform,
                ..Default::default()
            },
            move || Box::new(Regression::new(offscreen)),
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
