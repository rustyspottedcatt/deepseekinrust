use cudarc::driver::*;
use cust::prelude::*;
use tch::{Kind, Tensor};
use std::error::Error;
use std::ffi::CString;

const BLOCK_SIZE: i32 = 128;

fn tensor_to_device_buffer(tensor: &Tensor) -> Result<DeviceBuffer<f32>, Box<dyn std::error::Error>> {
    let slice: Vec<f32> = Vec::<f32>::try_from(tensor).expect("wrong type of tensor");

    let device_buffer = DeviceBuffer::from_slice(&slice)?;
    Ok(device_buffer)
}


fn init_cuda() -> Result<(Device, Context), Box<dyn Error>> {
    cust::init(cust::CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let context = Context::new(device)?;

    Ok((device, context))
}

pub fn act_quant(x: &Tensor) -> Result<(Tensor, Tensor), Box<dyn Error>> {
    assert!(x.is_contiguous(), "Input tensor must be contiguous");
    assert_eq!(x.size().last().unwrap() % BLOCK_SIZE as i64, 0, "Last dim must be divisible by BLOCK_SIZE");

    let device = x.device();
    let y = Tensor::empty_like(x);
    let s = Tensor::empty(&[x.size()[0], x.size()[1] / BLOCK_SIZE as i64], (Kind::Float, device));

    let ptx_cstr = CString::new(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/cuda/act_quant_kernel.ptx")))?;
    let module = Module::load_from_string(&ptx_cstr)?;
    let function = module.get_function("act_quant_kernel")?;

    unsafe {
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let x_numel = x.numel() as u32 / BLOCK_SIZE as u32;

        let x = tensor_to_device_buffer(&x)?;
        let y = tensor_to_device_buffer(&y)?;
        let s = tensor_to_device_buffer(&s)?;

        launch!(
            function<<<(x_numel / BLOCK_SIZE as u32), 1, 1, stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                s.as_device_ptr(),
                BLOCK_SIZE
            )
        )?;
        stream.synchronize()?;
    }

    Ok((y, s))
}

pub fn weight_dequant(x: &Tensor, s: &Tensor) -> Result<Tensor, Box<dyn Error>> {
    assert!(x.is_contiguous() && s.is_contiguous(), "Input tensor must be contiguous");
    assert!(x.dim() == 2 && s.dim() == 2, "x and s must be 2D tensors.");

    let device = x.device();
    let y = Tensor::empty_like(x);

    let ptx_cstr = CString::new(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/cuda/weight_dequant_kernel.ptx")))?;
    let module = Module::load_from_string(&ptx_cstr)?;
    let function = module.get_function("weight_dequant_kernel")?;

    let (m, n) = (x.size()[0] as u32, x.size()[1] as u32);

    let x_dev = tensor_to_device_buffer(&x)?;
    let y_dev = tensor_to_device_buffer(&y)?;
    let s_dev = tensor_to_device_buffer(&s)?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    unsafe {
        launch!(
             function<<<(m + BLOCK_SIZE as u32 - 1) / BLOCK_SIZE as u32, (n + BLOCK_SIZE as u32 - 1) / BLOCK_SIZE as u32, 1, stream>>>(
                x_dev.as_device_ptr(),
                s_dev.as_device_ptr(),
                y_dev.as_device_ptr(),
                m,
                n,
                BLOCK_SIZE
            )
        )?;
    }

    stream.synchronize()?;
    Ok(y)
}

pub fn fp8_gemm(a: &Tensor, a_s: &Tensor, b: &Tensor, b_s: &Tensor) -> Result<Tensor, Box<dyn Error>> {
    assert!(a.is_contiguous() && b.is_contiguous(), "Input tensor must be contiguous");
    assert!(a_s.is_contiguous() && b.is_contiguous(), "Input tensor must be contiguous");

    let (m, k) = (a.size()[0] as u32, a.size()[1] as u32);
    let n = b.size()[0] as u32;

    let device = a.device();
    let dtype = Kind::Float;
    let c = Tensor::empty(&[m as i64, n as i64], (dtype, device));

    let ptx_cstr = CString::new(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/cuda/fp8_gemm_kernel.ptx")))?;
    let module = Module::load_from_string(&ptx_cstr)?;
    let function = module.get_function("fp8_gemm_kernel")?;

    let a_dev = tensor_to_device_buffer(&a)?;
    let a_s_dev = tensor_to_device_buffer(&a_s)?;
    let b_dev = tensor_to_device_buffer(&b)?;
    let b_s_dev = tensor_to_device_buffer(&b_s)?;
    let c_dev = tensor_to_device_buffer(&c)?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    unsafe {
        launch!(
              function<<<(m + BLOCK_SIZE as u32 - 1) / BLOCK_SIZE as u32, (n + BLOCK_SIZE as u32 - 1) / BLOCK_SIZE as u32, 1, stream>>>(
                a_dev.as_device_ptr(),
                a_s_dev.as_device_ptr(),
                b_dev.as_device_ptr(),
                b_s_dev.as_device_ptr(),
                c_dev.as_device_ptr(),
                m,
                n,
                k,
                BLOCK_SIZE
            )
        )?;
    }

    stream.synchronize()?;
    Ok(c)
}
