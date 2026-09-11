/// <reference types="@webgpu/types" />

/** Optional independent WebGPU execution of the projection displayed in the lens.
 * Autograd keeps its deterministic Float64 CPU reference. This compute path uses
 * the same recorded inputs, in Float32, and reports the measured discrepancy.
 * Call only inside the model Worker.
 */
let devicePromise: Promise<GPUDevice | null> | undefined;
async function getDevice() {
  if (!devicePromise) devicePromise = (async () => {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    return adapter ? adapter.requestDevice() : null;
  })().catch(() => null);
  return devicePromise;
}

export async function gpuProjection(
  input: ArrayLike<number>, weight: ArrayLike<number>, rows: number,
  inputDim: number, outputDim: number, expected?: ArrayLike<number>,
) {
  const device = await getDevice();
  if (!device) return {available: false, reason: 'WebGPU compute is unavailable. The worker uses the explicit CPU calculation.'};
  if (input.length !== rows * inputDim || weight.length !== outputDim * inputDim)
    throw new Error('GPU projection dimensions do not match the recorded inputs.');
  const buffers: GPUBuffer[] = [];
  const make = (size: number, usage: GPUBufferUsageFlags, data?: Float32Array | Uint32Array) => {
    const b = device.createBuffer({size: Math.max(4, size), usage}); buffers.push(b);
    if (data) device.queue.writeBuffer(b, 0, data.buffer as ArrayBuffer);
    return b;
  };
  try {
    const x = make(input.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, Float32Array.from(input));
    const w = make(weight.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, Float32Array.from(weight));
    const resultSize = rows * outputDim * 4;
    const y = make(resultSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const dims = make(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Uint32Array([rows,inputDim,outputDim,0]));
    const staging = make(resultSize, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const module = device.createShaderModule({code: `
      struct Dims { rows: u32, inputs: u32, outputs: u32, pad: u32 }
      @group(0) @binding(0) var<storage, read> x: array<f32>;
      @group(0) @binding(1) var<storage, read> w: array<f32>;
      @group(0) @binding(2) var<storage, read_write> y: array<f32>;
      @group(0) @binding(3) var<uniform> d: Dims;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= d.rows * d.outputs) { return; }
        let row = i / d.outputs;
        let out = i % d.outputs;
        var sum = 0.0;
        for (var k = 0u; k < d.inputs; k++) {
          sum += x[row * d.inputs + k] * w[out * d.inputs + k];
        }
        y[i] = sum;
      }`});
    const pipeline = await device.createComputePipelineAsync({layout:'auto', compute:{module,entryPoint:'main'}});
    const bind = device.createBindGroup({layout:pipeline.getBindGroupLayout(0), entries:[x,w,y,dims].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0,bind);
    pass.dispatchWorkgroups(Math.ceil(rows * outputDim / 64)); pass.end();
    encoder.copyBufferToBuffer(y,0,staging,0,resultSize);
    const start = performance.now();
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const values = Array.from(new Float32Array(staging.getMappedRange().slice(0)));
    staging.unmap();
    let maxAbsoluteError = 0;
    if (expected) values.forEach((v,i)=>{maxAbsoluteError=Math.max(maxAbsoluteError,Math.abs(v-expected[i]));});
    return {available:true,values,shape:[rows,outputDim],milliseconds:performance.now()-start,maxAbsoluteError,precision:'Float32'};
  } finally { buffers.forEach(b=>b.destroy()); }
}
