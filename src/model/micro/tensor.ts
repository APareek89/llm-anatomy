/** Small vector-valued reverse-mode tape. No scalar node is created per weight. */
export interface TensorView { name: string; shape: number[]; values: number[]; gradient?: number[] }
export interface RecordedStep {
  id: number; name: string; layer: number; phase: 'forward'|'backward'|'update';
  inputs: TensorView[]; outputs: TensorView[]; formula: string; explanation: string;
  arithmetic: string; matrix?: {shape: number[]; values: number[]; mask?: boolean[]};
  parameter?: string; token?: number;
}
export class Tensor {
  data: Float64Array; grad: Float64Array; shape: number[]; name: string;
  constructor(shape: number[], data?: ArrayLike<number>, name='tensor') {
    this.shape=shape; this.name=name;
    const n=shape.reduce((a,b)=>a*b,1);
    this.data=data ? Float64Array.from(data):new Float64Array(n); this.grad=new Float64Array(n);
    if(this.data.length!==n) throw new Error(`Shape mismatch ${name}: ${n} vs ${this.data.length}`);
  }
  get length(){return this.data.length;}
}
export const fmt=(n:number)=>Number.isFinite(n)?n.toPrecision(7):String(n);
export const view=(t:Tensor):TensorView=>({name:t.name,shape:[...t.shape],values:Array.from(t.data)});
export class Tape {
  nodes:{out:Tensor;inputs:Tensor[];backward:()=>void;step?:RecordedStep}[]=[];
  steps:RecordedStep[]=[]; layer=-1;
  constructor(public record=false){}
  add(out:Tensor, inputs:Tensor[], backward:()=>void, name:string, formula:string, explanation:string, arithmetic:string, matrix?:RecordedStep['matrix']) {
    let step:RecordedStep|undefined;
    if(this.record){step={id:this.steps.length,name,layer:this.layer,phase:'forward',inputs:inputs.map(view),outputs:[view(out)],formula,explanation,arithmetic,matrix};this.steps.push(step);}
    this.nodes.push({out,inputs,backward,step});return out;
  }
  backward(loss:Tensor, scale=1) {
    loss.grad[0]=scale;
    for(let i=this.nodes.length-1;i>=0;i--)this.nodes[i].backward();
    if(this.record)for(const node of this.nodes)if(node.step){node.step.outputs[0].gradient=Array.from(node.out.grad);node.step.inputs.forEach((v,i)=>v.gradient=Array.from(node.inputs[i].grad));}
  }
}
export function matmul(t:Tape,x:Tensor,w:Tensor,name='Projection') {
  const [rows,cols]=x.shape, outdim=w.shape[0], out=new Tensor([rows,outdim],undefined,name);
  for(let r=0;r<rows;r++)for(let o=0;o<outdim;o++){let a=0;for(let c=0;c<cols;c++)a+=x.data[r*cols+c]*w.data[o*cols+c];out.data[r*outdim+o]=a;}
  return t.add(out,[x,w],()=>{for(let r=0;r<rows;r++)for(let o=0;o<outdim;o++){const g=out.grad[r*outdim+o];for(let c=0;c<cols;c++){x.grad[r*cols+c]+=g*w.data[o*cols+c];w.grad[o*cols+c]+=g*x.data[r*cols+c];}}},name,'Y = X Wᵀ','Each output adds the input features multiplied by one learned row of weights.',t.record?`${Array.from({length:cols},(_,c)=>`${fmt(x.data[c])} × ${fmt(w.data[c])}`).join(' + ')} = ${fmt(out.data[0])}`:'');
}
export function add(t:Tape,a:Tensor,b:Tensor,name='Residual add') {
  const out=new Tensor(a.shape,undefined,name);for(let i=0;i<a.length;i++)out.data[i]=a.data[i]+b.data[i];
  return t.add(out,[a,b],()=>{for(let i=0;i<a.length;i++){a.grad[i]+=out.grad[i];b.grad[i]+=out.grad[i];}},name,'y = x + branch','The residual stream retains the input and adds the branch contribution.',`${fmt(a.data[0])} + ${fmt(b.data[0])} = ${fmt(out.data[0])}`);
}
export function mul(t:Tape,a:Tensor,b:Tensor,name='Elementwise gate') {
  const out=new Tensor(a.shape,undefined,name);for(let i=0;i<a.length;i++)out.data[i]=a.data[i]*b.data[i];
  return t.add(out,[a,b],()=>{for(let i=0;i<a.length;i++){a.grad[i]+=out.grad[i]*b.data[i];b.grad[i]+=out.grad[i]*a.data[i];}},name,'yᵢ = aᵢ bᵢ','Multiply matching coordinates to control how much information passes.',`${fmt(a.data[0])} × ${fmt(b.data[0])} = ${fmt(out.data[0])}`);
}
export function activation(t:Tape,x:Tensor,type:'silu'|'sigmoid') {
  const out=new Tensor(x.shape,undefined,type==='silu'?'SiLU (swish)':'Sigmoid');
  for(let i=0;i<x.length;i++){const s=1/(1+Math.exp(-x.data[i]));out.data[i]=type==='silu'?x.data[i]*s:s;}
  return t.add(out,[x],()=>{for(let i=0;i<x.length;i++){const s=1/(1+Math.exp(-x.data[i]));x.grad[i]+=out.grad[i]*(type==='silu'?s+x.data[i]*s*(1-s):s*(1-s));}},out.name,type==='silu'?'SiLU(x) = x / (1 + exp(−x))':'σ(x) = 1 / (1 + exp(−x))',type==='silu'?'A smooth gate suppresses negative inputs while allowing positive signals through.':'Turn any real input into a gate between zero and one.',type==='silu'?`${fmt(x.data[0])} / (1 + exp(−${fmt(x.data[0])})) = ${fmt(out.data[0])}`:`1 / (1 + exp(−${fmt(x.data[0])})) = ${fmt(out.data[0])}`);
}
export function rmsnorm(t:Tape,x:Tensor,w:Tensor,group:number,name='RMSNorm',zeroCentered=true) {
  const out=new Tensor(x.shape,undefined,name), rows=x.length/group, inv=new Float64Array(rows), eps=1e-6;
  for(let r=0;r<rows;r++){let sq=0;for(let d=0;d<group;d++)sq+=x.data[r*group+d]**2;inv[r]=1/Math.sqrt(sq/group+eps);for(let d=0;d<group;d++)out.data[r*group+d]=x.data[r*group+d]*inv[r]*(w.data[d]+(zeroCentered?1:0));}
  return t.add(out,[x,w],()=>{for(let r=0;r<rows;r++){let dot=0;for(let d=0;d<group;d++)dot+=out.grad[r*group+d]*(w.data[d]+(zeroCentered?1:0))*x.data[r*group+d];for(let d=0;d<group;d++){const i=r*group+d;x.grad[i]+=out.grad[i]*(w.data[d]+(zeroCentered?1:0))*inv[r]-x.data[i]*inv[r]**3*dot/group;w.grad[d]+=out.grad[i]*x.data[i]*inv[r];}}},name,zeroCentered?'yᵢ = xᵢ (1 + wᵢ) / √(mean(x²) + 10⁻⁶)':'yᵢ = xᵢ wᵢ / √(mean(x²) + 10⁻⁶)','Normalize signal magnitude, then apply a learned feature scale.',`${fmt(x.data[0])} × ${fmt(w.data[0]+(zeroCentered?1:0))} × ${fmt(inv[0])} = ${fmt(out.data[0])}`);
}
export function slice(t:Tape,x:Tensor,start:number,end:number,name='Split projection') {
  const [rows,cols]=x.shape, width=end-start, out=new Tensor([rows,width],undefined,name);
  for(let r=0;r<rows;r++)for(let c=0;c<width;c++)out.data[r*width+c]=x.data[r*cols+start+c];
  return t.add(out,[x],()=>{for(let r=0;r<rows;r++)for(let c=0;c<width;c++)x.grad[r*cols+start+c]+=out.grad[r*width+c];},name,`Y[:, :] = X[:, ${start}:${end}]`,'Select the coordinates assigned to this branch.',`X[0, ${start}] = ${fmt(x.data[start])} → Y[0, 0] = ${fmt(out.data[0])}`);
}
export function splitInterleaved(t:Tape,x:Tensor,heads:number,dim:number,part:0|1,name:string) {
  const rows=x.shape[0], out=new Tensor([rows,heads*dim],undefined,name);
  for(let r=0;r<rows;r++)for(let h=0;h<heads;h++)for(let d=0;d<dim;d++)out.data[r*heads*dim+h*dim+d]=x.data[r*heads*dim*2+h*dim*2+part*dim+d];
  return t.add(out,[x],()=>{for(let r=0;r<rows;r++)for(let h=0;h<heads;h++)for(let d=0;d<dim;d++)x.grad[r*heads*dim*2+h*dim*2+part*dim+d]+=out.grad[r*heads*dim+h*dim+d];},name,'[Q₀, gate₀, Q₁, gate₁, …] → selected head coordinates','Qwen packs each query head beside its output gate.',`X[0, ${part*dim}] = ${fmt(x.data[part*dim])} → ${fmt(out.data[0])}`);
}
export function embedding(t:Tape,ids:number[],w:Tensor) {
  const dim=w.shape[1], out=new Tensor([ids.length,dim],undefined,'Token embeddings');
  ids.forEach((id,r)=>out.data.set(w.data.subarray(id*dim,(id+1)*dim),r*dim));
  return t.add(out,[w],()=>{ids.forEach((id,r)=>{for(let d=0;d<dim;d++)w.grad[id*dim+d]+=out.grad[r*dim+d];});},'Embedding lookup','X[t, :] = embedding[token_id[t], :]','Read the learned row for each word. Only rows used by this batch receive an embedding gradient.',`embedding[${ids[0]}, 0] = ${fmt(out.data[0])}`);
}
export function softmaxValues(values:ArrayLike<number>,temperature=1) {
  const temp=Math.max(0.001,temperature), max=Math.max(...Array.from(values)), ex=Array.from(values,v=>Math.exp((v-max)/temp)), sum=ex.reduce((a,b)=>a+b,0);return ex.map(v=>v/sum);
}
export function crossEntropy(t:Tape,logits:Tensor,targets:number[]) {
  const [rows,vocab]=logits.shape, probs=new Float64Array(logits.length), out=new Tensor([1],undefined,'Mean cross-entropy loss');
  for(let r=0;r<rows;r++){const p=softmaxValues(logits.data.subarray(r*vocab,(r+1)*vocab));probs.set(p,r*vocab);out.data[0]-=Math.log(Math.max(1e-300,p[targets[r]]))/rows;}
  return t.add(out,[logits],()=>{for(let r=0;r<rows;r++)for(let v=0;v<vocab;v++)logits.grad[r*vocab+v]+=out.grad[0]*(probs[r*vocab+v]-(v===targets[r]?1:0))/rows;},'Cross-entropy loss','L = − meanₜ log softmax(logitsₜ)[targetₜ]','Penalize low probability on the correct next word. Gradients show how each weight can lower this loss.',`−log(${fmt(probs[targets[0]])}) = ${fmt(-Math.log(probs[targets[0]]))}; mean over ${rows} targets = ${fmt(out.data[0])}`,{shape:[rows,vocab],values:Array.from(probs)});
}
