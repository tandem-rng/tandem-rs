//! The fill as a `wgpu` compute shader, behind the `wgpu` feature.
//!
//! [`GpuFill`] runs the specification's row fill on any adapter wgpu drives and writes stream
//! words into a storage buffer. Each fill advances the generator exactly as the CPU fills do,
//! so a CPU fill, a GPU fill and a mix of both read the same stream.
//!
//! WGSL has no 64-bit integers. The shader builds the 32x32 to 64-bit product of the mix
//! from 16-bit halves, which is exact, and carries chunk and block indices as word pairs.

use std::vec::Vec;

use wgpu::util::DeviceExt;

use crate::{Tandem, align, to_f32, to_f64};

/// The shader source. `fill` is its entry point.
pub const SHADER: &str = include_str!("tandem.wgsl");

const GROUPS_PER_WORKGROUP: u64 = 32;
const BLOCK_BYTES: u64 = 16;

/// Where a fill's values lie inside the storage buffer it wrote.
///
/// The buffer holds whole 16-byte stream blocks, so the values start at `byte_offset` and
/// run for `byte_len` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// Offset of the first value in bytes.
    pub byte_offset: u64,
    /// Length of the values in bytes.
    pub byte_len: u64,
}

/// A compiled fill pipeline on one device.
pub struct GpuFill {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    adapter: std::string::String,
}

impl GpuFill {
    /// Compile the fill for a device the caller owns.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tandem"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tandem fill"),
            layout: None,
            module: &module,
            entry_point: Some("fill"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        GpuFill {
            device,
            queue,
            pipeline,
            layout,
            adapter: std::string::String::new(),
        }
    }

    /// Request the default adapter with its largest buffer limits and compile the fill.
    /// Returns `None` when no adapter exists. `TANDEM_GPU_ADAPTER=<index>` picks another
    /// adapter from the enumeration order, for hosts with several GPUs.
    pub fn new_default() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = match std::env::var("TANDEM_GPU_ADAPTER") {
            Ok(i) => pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
                .into_iter()
                .nth(i.parse().ok()?)?,
            Err(_) => pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
            )
            .ok()?,
        };
        Self::from_adapter(&adapter)
    }

    /// Compile the fill on `adapter` with its largest buffer limits.
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Option<Self> {
        let supported = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
            max_buffer_size: supported.max_buffer_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tandem"),
            required_features: wgpu::Features::empty(),
            required_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        let mut fill = Self::new(device, queue);
        fill.adapter = adapter.get_info().name;
        Some(fill)
    }

    /// Name of the adapter, when known from [`from_adapter`](Self::from_adapter).
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// The device the fill runs on.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue the fill submits to.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Bytes a storage buffer needs for `n` values of `bits` width from the generator's
    /// aligned position: whole 16-byte blocks, at least one.
    pub fn buffer_size(rng: &Tandem, bits: u32, n: usize) -> u64 {
        let p0 = align(rng.position(), bits);
        let p1 = p0 + u64::from(bits) * n as u64;
        let blocks = (p1.div_ceil(128)).saturating_sub(p0 >> 7).max(1);
        blocks * BLOCK_BYTES
    }

    /// Write `n` stream words into `out` from the generator's 32-bit aligned position and
    /// advance it, as [`Tandem::fill_u32`] does. `out` is a `STORAGE` buffer of at least
    /// [`buffer_size`](Self::buffer_size) bytes. The work is submitted, not awaited.
    pub fn fill_words(&self, rng: &mut Tandem, out: &wgpu::Buffer, n: usize) -> Span {
        self.fill_bits(rng, 32, out, n)
    }

    fn fill_bits(&self, rng: &mut Tandem, bits: u32, out: &wgpu::Buffer, n: usize) -> Span {
        let k = u64::from(rng.chunk_length());
        let p0 = align(rng.position(), bits);
        let p1 = p0 + u64::from(bits) * n as u64;
        rng.set_position(p1);

        let block_start = p0 >> 7;
        let block_end = p1.div_ceil(128);
        let n_blocks = block_end - block_start;
        let span = Span {
            byte_offset: (p0 - (block_start << 7)) / 8,
            byte_len: (p1 - p0) / 8,
        };
        assert!(
            out.size() >= n_blocks.max(1) * BLOCK_BYTES,
            "output buffer holds {} bytes, the fill needs {}",
            out.size(),
            n_blocks.max(1) * BLOCK_BYTES
        );
        if n_blocks == 0 {
            return span;
        }

        let g0 = (block_start >> 3) / k;
        let g1 = ((block_end - 1) >> 3) / k;
        let workgroups = (g1 - g0 + 1).div_ceil(GROUPS_PER_WORKGROUP);
        assert!(
            workgroups <= 65535,
            "fill too large for one dispatch: split it by position"
        );

        let key = rng.key();
        let params: [u32; 12] = [
            key[0],
            key[1],
            key[2],
            key[3],
            g0 as u32,
            (g0 >> 32) as u32,
            block_start as u32,
            (block_start >> 32) as u32,
            n_blocks as u32,
            rng.chunk_length(),
            0,
            0,
        ];
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tandem params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tandem fill"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        span
    }

    /// Fill on the GPU and read the bytes of the values back.
    fn read_bytes(&self, rng: &mut Tandem, bits: u32, n: usize) -> Vec<u8> {
        let size = Self::buffer_size(rng, bits, n);
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tandem out"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let span = self.fill_bits(rng, bits, &out, n);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tandem staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |r| {
            r.expect("map the staging buffer")
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("wait for the fill");
        let bytes = {
            let view = staging.get_mapped_range(..).expect("mapped staging buffer");
            view[span.byte_offset as usize..(span.byte_offset + span.byte_len) as usize].to_vec()
        };
        staging.unmap();
        bytes
    }

    /// `n` 32-bit draws, the values of [`Tandem::fill_u32`].
    pub fn read_u32(&self, rng: &mut Tandem, n: usize) -> Vec<u32> {
        words(&self.read_bytes(rng, 32, n), u32::from_le_bytes)
    }

    /// `n` 64-bit draws, the values of [`Tandem::fill_u64`].
    pub fn read_u64(&self, rng: &mut Tandem, n: usize) -> Vec<u64> {
        words(&self.read_bytes(rng, 64, n), u64::from_le_bytes)
    }

    /// `n` uniform `f32` in `[0, 1)`, the values of [`Tandem::fill_f32`].
    pub fn read_f32(&self, rng: &mut Tandem, n: usize) -> Vec<f32> {
        words(&self.read_bytes(rng, 32, n), |b| {
            to_f32(u32::from_le_bytes(b))
        })
    }

    /// `n` uniform `f64` in `[0, 1)`, the values of [`Tandem::fill_f64`].
    pub fn read_f64(&self, rng: &mut Tandem, n: usize) -> Vec<f64> {
        words(&self.read_bytes(rng, 64, n), |b| {
            to_f64(u64::from_le_bytes(b))
        })
    }
}

fn words<T, const N: usize>(bytes: &[u8], from: impl Fn([u8; N]) -> T) -> Vec<T> {
    bytes.as_chunks::<N>().0.iter().map(|c| from(*c)).collect()
}
