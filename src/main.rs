// #![feature(let_chains)]
use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use ron::de;
use std::{sync::Arc, time::Instant, arch::x86_64::_mm_movedup_pd, io::Cursor};
use vulkano::{
    buffer::{
        allocator::{CpuBufferAllocator, CpuBufferAllocatorCreateInfo},
        BufferUsage, CpuAccessibleBuffer, TypedBufferAccess,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned,
        QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage, swapchain, ImageDimensions, ImmutableImage, MipmapsCount},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState}, render_pass,
			color_blend::ColorBlendState
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
	sampler::{Sampler, SamplerCreateInfo, SamplerAddressMode, Filter}
};
use vulkano::impl_vertex;
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent, MouseScrollDelta::LineDelta},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    position: [f32; 3],
}
impl_vertex!(Vertex, position);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Normal {
    normal: [f32; 3],
}
impl_vertex!(Normal, normal);

pub const VERTICES: [Vertex; 9] = [
	Vertex { position: [1.000000, 1.000000, -1.000000] },
	Vertex { position: [1.000000, 1.000000, -1.000000] },
	Vertex { position: [1.000000, -1.000000, -1.000000] },
	Vertex { position: [1.000000, 1.000000, 1.000000] },
	Vertex { position: [1.000000, -1.000000, 1.000000] },
	Vertex { position: [-1.000000, 1.000000, -1.000000] },
	Vertex { position: [-1.000000, -1.000000, -1.000000] },
	Vertex { position: [-1.000000, 1.000000, 1.000000] },
	Vertex { position: [-1.000000, -1.000000, 1.000000] }
];

pub const NORMALS: [Normal; 9] = [
	Normal { normal: [1.000000, 1.000000, -1.000000] },
	Normal { normal: [1.000000, 1.000000, -1.000000] },
	Normal { normal: [1.000000, -1.000000, -1.000000] },
	Normal { normal: [1.000000, 1.000000, 1.000000] },
	Normal { normal: [1.000000, -1.000000, 1.000000] },
	Normal { normal: [-1.000000, 1.000000, -1.000000] },
	Normal { normal: [-1.000000, -1.000000, -1.000000] },
	Normal { normal: [-1.000000, 1.000000, 1.000000] },
	Normal { normal: [-1.000000, -1.000000, 1.000000] }
];

pub const INDICES: [u16; 36] = [
	5,3,1,
	3,8,4,
	7,6,8,
	2,8,6,
	1,4,2,
	5,2,6,
	5,7,3,
	3,7,8,
	7,5,6,
	2,4,8,
	1,3,4,
	5,1,2
];

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/basic_vert.glsl",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/basic_frag.glsl"
    }
}

fn main() 
{
	let library = VulkanLibrary::new().unwrap();
	let required_extensions = vulkano_win::required_extensions(&library);
	let instance = Instance::new(
		library,
		InstanceCreateInfo { 
			application_name: Some("vk test".to_string()),
			engine_name: Some("detail".to_string()), 
			enabled_extensions: required_extensions,
			enumerate_portability: true, // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
			..Default::default()
		},
	)
	.unwrap();

	let event_loop = EventLoop::new();
	let surface = WindowBuilder::new()
		.build_vk_surface(&event_loop, instance.clone())
		.unwrap();

	let device_extensions = DeviceExtensions {
		khr_swapchain: true,
		..DeviceExtensions::empty()
	};
	
	let (physical_device, queue_family_index) : (Arc<PhysicalDevice>, u32) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

	println!("Using device {} of type {:?}", 
		physical_device.properties().device_name, 
		physical_device.properties().device_type
	);
	
	let (device, mut queues) = Device::new(
		physical_device, 
		DeviceCreateInfo { 
			enabled_extensions: device_extensions,
			queue_create_infos: vec![
				QueueCreateInfo {
					queue_family_index,
					..Default::default()
				}
			],
			..Default::default()
		}
	)
	.unwrap();

	let queue = queues.next().unwrap();

	let (mut swapchain, images) = {

		let surface_capabilities = device
			.physical_device()
			.surface_capabilities(&surface, Default::default())
			.unwrap();

		let image_format = Some(
			device
				.physical_device()
				.surface_formats(&surface, Default::default())
				.unwrap()[0]
				.0
		);

		let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

		Swapchain::new(
			device.clone(),
			surface.clone(),
			SwapchainCreateInfo { 
				min_image_count: surface_capabilities.min_image_count, 
				image_format: image_format, 
				image_extent: window.inner_size().into(), 
				image_usage: ImageUsage::COLOR_ATTACHMENT, 
				composite_alpha: surface_capabilities
					.supported_composite_alpha
					.into_iter()
					.next()
					.unwrap(), 
				..Default::default()
			}
		)
		.unwrap()
	};

	let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

	// VERTEX BUFFER
	let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
		&memory_allocator, 
		BufferUsage::VERTEX_BUFFER, 
		false, 
		VERTICES
	)
	.unwrap();

	// NORMAL BUFFER
	let normals_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage::VERTEX_BUFFER,
        false,
        NORMALS,
    )
    .unwrap();

	// INDEX BUFFER
    let index_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage::INDEX_BUFFER,
        false,
        INDICES,
    )
    .unwrap();

	// UNIFORM BUFFER
	let uniform_buffer = CpuBufferAllocator::new(
		memory_allocator.clone(), 
		CpuBufferAllocatorCreateInfo { 
			buffer_usage: BufferUsage::UNIFORM_BUFFER, 
			..Default::default()
		}
	);

	let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

	let render_pass = vulkano::single_pass_renderpass!(
		device.clone(),
		attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
	)
	.unwrap();

	let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
	let rotation_start = Instant::now();

	let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
	let command_buffer_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());

	let mut uploads = AutoCommandBufferBuilder::primary(
		&command_buffer_allocator,
		queue.queue_family_index(),
		CommandBufferUsage::OneTimeSubmit
	)
	.unwrap();

	// let png_bytes = include_bytes!("./../images/woag.png").to_vec();

	
    let texture = {
		let png_bytes = include_bytes!("./../images/woag.png").to_vec();
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info();
        let dimensions = ImageDimensions::Dim2d {
            width: info.width,
            height: info.height,
            array_layers: 1,
        };
        let mut image_data = Vec::new();
        image_data.resize((info.width * info.height * 4) as usize, 0);
        reader.next_frame(&mut image_data).unwrap();

        let image = ImmutableImage::from_iter(
            &memory_allocator,
            image_data,
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            &mut uploads,
        )
        .unwrap();
		
        ImageView::new_default(image).unwrap()
    };

	let sampler = Sampler::new(
        device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::Repeat; 3],
            ..Default::default()
        },
    )
    .unwrap();

	let (mut pipeline, mut framebuffers) = window_size_dependent_setup(&memory_allocator, &vs, &fs, &images, render_pass.clone());
	let mut recreate_swapchain = false;

	// let layout = pipeline.layout().set_layouts().get(0).unwrap();
    // let texture_set = PersistentDescriptorSet::new(
    //     &descriptor_set_allocator,
    //     layout.clone(),
    //     [WriteDescriptorSet::image_view_sampler(0, texture, sampler)],
    // )
    // .unwrap();

	let mut scale_factor = 0.5f32;
	let time_start = Instant::now();

	event_loop.run(
		move |event, _, control_flow| 
		{
			match event 
			{
				Event::WindowEvent { event: WindowEvent::MouseWheel { delta, .. }, .. } => 
				{
					match delta
					{
						LineDelta(_, delta) => 
						{ 
							scale_factor += delta * 0.05; 
						}
						_ => {}
					}
					println!("Delta : {:?}", delta);
				}

				Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => 
				{
					*control_flow = ControlFlow::Exit;
				}

				Event::WindowEvent { event: WindowEvent::Resized(_), .. } => 
				{
					recreate_swapchain = true;
				}

				Event::RedrawEventsCleared =>
				{
					scale_factor = (cgmath::Angle::sin(cgmath::Rad(Instant::now().duration_since(time_start).as_secs_f32() * 2.0f32)) * 0.4f32 + 1.0f32) / 2.0f32;
					let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
					let dimensions = window.inner_size();

					if dimensions.width == 0 || dimensions.height == 0
					{
						return;
					}

					previous_frame_end.as_mut().unwrap().cleanup_finished();

					if recreate_swapchain
					{
						println!("swapchain recreation");
						let (new_swapchain, new_images) =
							match swapchain.recreate(
								SwapchainCreateInfo {
									image_extent: dimensions.into(),
									..swapchain.create_info()
								}
							)
							{
								Ok(r) => r,
								Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
								Err(e) => panic!("Failed to recreate swapchain! \nError : '{:?}'", e)
							};

						swapchain = new_swapchain;

						let (new_pipeline, new_framebuffers) = window_size_dependent_setup(
							&memory_allocator, 
							&vs, 
							&fs, 
							&new_images, 
							render_pass.clone()
						);

						pipeline = new_pipeline;
						framebuffers = new_framebuffers;
						recreate_swapchain = false;
					}

					let uniform_buffer_subbuffer = {
						let elapsed = rotation_start.elapsed();
						let rotation = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
						let rotation = Matrix3::from_angle_y(Rad(rotation as f32));

						let aspect_ratio = swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
						let projection = cgmath::perspective(
							Rad(std::f32::consts::FRAC_PI_2),
							aspect_ratio,
							0.01,
							100.0,
						);
						let view = Matrix4::look_at_rh(
							Point3::new(0.0, 1.0, 1.5),
							Point3::new(0.0, 0.0, 0.0),
							Vector3::new(0.0, -1.0, 0.0),
						);
						let scale = Matrix4::from_scale(scale_factor);
						// let scale = Matrix4::from_scale(0.5);

						let uniform_data = vs::ty::Data {
							world: Matrix4::from(rotation).into(),
							view: (view * scale).into(),
							proj: projection.into()
						};

						uniform_buffer.from_data(uniform_data).unwrap()
					};

					let layout = pipeline.layout().set_layouts().get(0).unwrap();

					let set = PersistentDescriptorSet::new(
						&descriptor_set_allocator,
						layout.clone(),
						[
							WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)
						]
					)
					.unwrap();

					let texture_set = PersistentDescriptorSet::new(
						&descriptor_set_allocator,
						layout.clone(),
						[
							WriteDescriptorSet::image_view_sampler(0, texture.clone(), sampler.clone())
						]
					)
					.unwrap();


					let (image_index, suboptimal, acquire_future) =
						match acquire_next_image(swapchain.clone(), None)
						{
							Ok(r) => r,
							Err(AcquireError::OutOfDate) =>
							{
								recreate_swapchain = true;
								return;
							}
							Err(e) => panic!("Failed to acquire next image! \nError : '{:?}'", e)
						};
					
					if suboptimal
					{
						recreate_swapchain = true;
					}

					let mut builder = AutoCommandBufferBuilder::primary(
						&command_buffer_allocator,
						queue.queue_family_index(),
						CommandBufferUsage::OneTimeSubmit
					)
					.unwrap();

					builder
						.begin_render_pass(
							RenderPassBeginInfo { 
								clear_values: vec![
									Some([0.0, 0.0, 0.0, 1.0].into()),
									Some(1f32.into())
								],
								..RenderPassBeginInfo::framebuffer(
									framebuffers[image_index as usize].clone()
								)
							}, 
							SubpassContents::Inline
						)
						.unwrap()
						.bind_pipeline_graphics(pipeline.clone())
						.bind_descriptor_sets(
							PipelineBindPoint::Graphics, 
							pipeline.layout().clone(), 
							0, 
							set.clone()
						)
						.bind_descriptor_sets(
							PipelineBindPoint::Graphics, 
							pipeline.layout().clone(), 
							0, 
							texture_set.clone()
						)
						.bind_vertex_buffers(0, (vertex_buffer.clone(), normals_buffer.clone()))
						.bind_index_buffer(index_buffer.clone())
						.draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
						.unwrap()
						.end_render_pass()
						.unwrap();
					
					let command_buffer = builder.build().unwrap();

					let future = previous_frame_end
						.take()
						.unwrap()
						.join(acquire_future)
						.then_execute(queue.clone(), command_buffer)
						.unwrap()
						.then_swapchain_present(
							queue.clone(),
							SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index)
						)
						.then_signal_fence_and_flush();

					match future 
					{
						Ok(future) =>
						{
							previous_frame_end = Some(future.boxed());
						}
						Err(FlushError::OutOfDate) => 
						{
							recreate_swapchain = true;
							previous_frame_end = Some(sync::now(device.clone()).boxed());
						}
						Err(e) => 
						{
							println!("Failed to flush future! \nError : '{:?}'", e);
							previous_frame_end = Some(sync::now(device.clone()).boxed());
						}
					}
				}
				_ => ()
			}
		}
	);

    // println!("Hello, world!");
}

fn window_size_dependent_setup(
	memory_allocator: &StandardMemoryAllocator,
	vs: &ShaderModule,
	fs: &ShaderModule,
	images: &[Arc<SwapchainImage>],
	render_pass: Arc<RenderPass>
) -> (Arc<GraphicsPipeline>, Vec<Arc<Framebuffer>>)
{
	let dimensions = images[0].dimensions().width_height();

	let depth_buffer = ImageView::new_default(
		AttachmentImage::transient(
			memory_allocator, 
			dimensions.clone(), 
			Format::D16_UNORM
		)
		.unwrap()
	)
	.unwrap();

	let framebuffers = images 
		.iter()
		.map( |image|
			{
				let view = ImageView::new_default(image.clone()).unwrap();
				Framebuffer::new(
					render_pass.clone(), 
					FramebufferCreateInfo { 
						attachments: vec![view, depth_buffer.clone()],
						..Default::default()
					}
				)
			}
			.unwrap()
		)
		.collect::<Vec<_>>();

	let subpass = Subpass::from(render_pass.clone(), 0).unwrap(); // h
	let pipeline = GraphicsPipeline::start()
		.vertex_input_state(
			BuffersDefinition::new()
				.vertex::<Vertex>()
				.vertex::<Normal>()
		)
		.vertex_shader(vs.entry_point("main").unwrap(), ())
		.input_assembly_state(InputAssemblyState::new())
		.viewport_state(
			ViewportState::viewport_fixed_scissor_irrelevant(
				[
					Viewport {
						origin: [0.0, 0.0],
						dimensions: [dimensions[0] as f32, dimensions[1] as f32],
						depth_range: 0.0..1.0
					}
				]
			)
		)
		.fragment_shader(fs.entry_point("main").unwrap(), ())
		.depth_stencil_state(DepthStencilState::simple_depth_test())
		.color_blend_state(ColorBlendState::new(subpass.num_color_attachments()).blend_alpha())
        .render_pass(subpass)
		// .render_pass(Subpass::from(render_pass, 0).unwrap())
		.build(memory_allocator.device().clone())
		.unwrap();

	(pipeline, framebuffers)
}