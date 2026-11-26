use std::collections::HashMap;

use ash::{vk, Device};

use crate::graphics::PassAction;

const COLOR_LAYOUT: vk::ImageLayout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
const DEPTH_LAYOUT: vk::ImageLayout = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

#[derive(Clone, Copy)]
pub(super) struct AttachmentView {
    pub view: vk::ImageView,
    pub format: vk::Format,
}

pub(super) struct ColorAttachment {
    pub image: AttachmentView,
    pub resolve: Option<vk::ImageView>,
}

pub(super) struct PassTarget {
    pub colors: Vec<ColorAttachment>,
    pub depth: Option<AttachmentView>,
    pub extent: vk::Extent2D,
    pub samples: u32,
}

impl PassTarget {
    pub fn area(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: self.extent,
        }
    }

    pub fn dynamic_attachments(
        &self,
        clear: PassClear,
    ) -> (
        Vec<vk::RenderingAttachmentInfo>,
        Option<vk::RenderingAttachmentInfo>,
    ) {
        let colors = self
            .colors
            .iter()
            .map(|color| {
                let info = vk::RenderingAttachmentInfo::builder()
                    .image_view(color.image.view)
                    .image_layout(COLOR_LAYOUT)
                    .load_op(load_op(clear.color.is_some()))
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(clear.color_value());
                if let Some(resolve) = color.resolve {
                    info.resolve_mode(vk::ResolveModeFlags::AVERAGE)
                        .resolve_image_view(resolve)
                        .resolve_image_layout(COLOR_LAYOUT)
                        .build()
                } else {
                    info.build()
                }
            })
            .collect();
        let depth = self.depth.map(|depth| {
            vk::RenderingAttachmentInfo::builder()
                .image_view(depth.view)
                .image_layout(DEPTH_LAYOUT)
                .load_op(load_op(clear.depth.is_some()))
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear.depth_value())
                .build()
        });
        (colors, depth)
    }
}

pub(super) struct PassClear {
    color: Option<vk::ClearColorValue>,
    depth: Option<vk::ClearDepthStencilValue>,
}

impl From<PassAction> for PassClear {
    fn from(action: PassAction) -> Self {
        match action {
            PassAction::Nothing => Self {
                color: None,
                depth: None,
            },
            PassAction::Clear { color, depth, .. } => Self {
                color: color.map(|(r, g, b, a)| vk::ClearColorValue {
                    float32: [r, g, b, a],
                }),
                // Miniquad's Vulkan depth formats do not have a stencil plane.
                depth: depth.map(|depth| vk::ClearDepthStencilValue { depth, stencil: 0 }),
            },
        }
    }
}

impl PassClear {
    fn color_value(&self) -> vk::ClearValue {
        vk::ClearValue {
            color: self.color.unwrap_or_default(),
        }
    }

    fn depth_value(&self) -> vk::ClearValue {
        vk::ClearValue {
            depth_stencil: self.depth.unwrap_or_default(),
        }
    }
}

fn load_op(clear: bool) -> vk::AttachmentLoadOp {
    if clear {
        vk::AttachmentLoadOp::CLEAR
    } else {
        vk::AttachmentLoadOp::LOAD
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct PassLayout {
    // Ordered formats and whether each color has a single-sample resolve.
    colors: Vec<(vk::Format, bool)>,
    depth: Option<vk::Format>,
    samples: vk::SampleCountFlags,
}

#[derive(Hash, PartialEq, Eq)]
struct FramebufferKey {
    views: Vec<vk::ImageView>,
    extent: vk::Extent2D,
}

#[derive(Default)]
struct CompatiblePasses {
    // A framebuffer is compatible with every LOAD/CLEAR variant of its layout.
    framebuffers: HashMap<FramebufferKey, vk::Framebuffer>,
    render_passes: [Option<vk::RenderPass>; 4],
}

impl CompatiblePasses {
    fn render_pass(
        &mut self,
        device: &Device,
        layout: &PassLayout,
        clear_color: bool,
        clear_depth: bool,
    ) -> vk::RenderPass {
        let variant = usize::from(clear_color) + 2 * usize::from(clear_depth);
        *self.render_passes[variant].get_or_insert_with(|| {
            create_render_pass(device, layout, load_op(clear_color), load_op(clear_depth))
        })
    }
}

#[derive(Default)]
pub(super) struct LegacyRenderPasses {
    // The context owns native objects, independently of the recyclable Miniquad
    // pass IDs. A framebuffer survives until its views can be destroyed.
    passes: HashMap<PassLayout, CompatiblePasses>,
}

impl LegacyRenderPasses {
    pub fn begin(
        &mut self,
        device: &Device,
        command: vk::CommandBuffer,
        target: &PassTarget,
        clear: PassClear,
    ) {
        let layout = PassLayout {
            colors: target
                .colors
                .iter()
                .map(|color| (color.image.format, color.resolve.is_some()))
                .collect(),
            depth: target.depth.map(|depth| depth.format),
            samples: vk::SampleCountFlags::from_raw(target.samples),
        };
        let passes = self.passes.entry(layout.clone()).or_default();
        let render_pass = passes.render_pass(
            device,
            &layout,
            !target.colors.is_empty() && clear.color.is_some(),
            target.depth.is_some() && clear.depth.is_some(),
        );

        // Match the order in create_render_pass: colors, active resolves, depth.
        let mut views: Vec<_> = target
            .colors
            .iter()
            .map(|color| color.image.view)
            .chain(target.colors.iter().filter_map(|color| color.resolve))
            .collect();
        let mut clear_values = vec![clear.color_value(); views.len()];
        if let Some(depth) = target.depth {
            views.push(depth.view);
            clear_values.push(clear.depth_value());
        }
        let key = FramebufferKey {
            views,
            extent: target.extent,
        };
        let framebuffer = if let Some(&framebuffer) = passes.framebuffers.get(&key) {
            framebuffer
        } else {
            let info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&key.views)
                .width(key.extent.width)
                .height(key.extent.height)
                .layers(1);
            let framebuffer = unsafe { device.create_framebuffer(&info, None) }
                .expect("Failed to create a Vulkan framebuffer");
            passes.framebuffers.insert(key, framebuffer);
            framebuffer
        };
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(target.area())
            .clear_values(&clear_values);
        unsafe { device.cmd_begin_render_pass(command, &info, vk::SubpassContents::INLINE) };
    }

    pub fn pipeline_pass(
        &mut self,
        device: &Device,
        colors: &[vk::Format],
        depth: Option<vk::Format>,
        samples: u32,
    ) -> vk::RenderPass {
        // Single-subpass render-pass compatibility ignores resolve attachments.
        // Load/store ops also do not affect pipeline compatibility.
        let layout = PassLayout {
            colors: colors.iter().map(|&format| (format, false)).collect(),
            depth,
            samples: vk::SampleCountFlags::from_raw(samples),
        };
        self.passes
            .entry(layout.clone())
            .or_default()
            .render_pass(device, &layout, false, false)
    }

    // Call after work that uses the views has completed, and before destroying them.
    pub fn release_framebuffers(&mut self, device: &Device, views: &[vk::ImageView]) {
        for passes in self.passes.values_mut() {
            passes.framebuffers.retain(|key, framebuffer| {
                if key.views.iter().any(|view| views.contains(view)) {
                    unsafe { device.destroy_framebuffer(*framebuffer, None) };
                    false
                } else {
                    true
                }
            });
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        for (_, passes) in self.passes.drain() {
            for framebuffer in passes.framebuffers.into_values() {
                unsafe { device.destroy_framebuffer(framebuffer, None) };
            }
            for render_pass in passes.render_passes.into_iter().flatten() {
                unsafe { device.destroy_render_pass(render_pass, None) };
            }
        }
    }
}

fn attachment(
    format: vk::Format,
    samples: vk::SampleCountFlags,
    layout: vk::ImageLayout,
    load: vk::AttachmentLoadOp,
) -> vk::AttachmentDescription {
    // The existing explicit barriers perform all transitions and order stores
    // against later loads, just as in dynamic rendering.
    vk::AttachmentDescription::builder()
        .format(format)
        .samples(samples)
        .load_op(load)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(layout)
        .final_layout(layout)
        .build()
}

fn create_render_pass(
    device: &Device,
    layout: &PassLayout,
    color_load: vk::AttachmentLoadOp,
    depth_load: vk::AttachmentLoadOp,
) -> vk::RenderPass {
    let mut attachments: Vec<_> = layout
        .colors
        .iter()
        .map(|&(format, _)| attachment(format, layout.samples, COLOR_LAYOUT, color_load))
        .collect();
    let colors: Vec<_> = (0..attachments.len() as u32)
        .map(|attachment| vk::AttachmentReference {
            attachment,
            layout: COLOR_LAYOUT,
        })
        .collect();
    let resolves: Vec<_> = layout
        .colors
        .iter()
        .map(|&(format, resolve)| {
            let index = if resolve {
                let index = attachments.len() as u32;
                attachments.push(attachment(
                    format,
                    vk::SampleCountFlags::TYPE_1,
                    COLOR_LAYOUT,
                    vk::AttachmentLoadOp::DONT_CARE,
                ));
                index
            } else {
                vk::ATTACHMENT_UNUSED
            };
            vk::AttachmentReference {
                attachment: index,
                layout: COLOR_LAYOUT,
            }
        })
        .collect();
    let depth = layout.depth.map(|format| {
        let index = attachments.len() as u32;
        attachments.push(attachment(format, layout.samples, DEPTH_LAYOUT, depth_load));
        vk::AttachmentReference {
            attachment: index,
            layout: DEPTH_LAYOUT,
        }
    });
    let mut subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colors);
    if resolves
        .iter()
        .any(|r| r.attachment != vk::ATTACHMENT_UNUSED)
    {
        subpass = subpass.resolve_attachments(&resolves);
    }
    if let Some(depth) = &depth {
        subpass = subpass.depth_stencil_attachment(depth);
    }
    let subpasses = [subpass.build()];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses);
    unsafe { device.create_render_pass(&info, None) }
        .expect("Failed to create a Vulkan render pass")
}
