import test from 'node:test';
import assert from 'node:assert/strict';
import {MicroModel,type MicroFamily} from '../src/model/micro/model';
import {Tensor,Tape,clippedSwiGLU,gelu,topKRouter,expertMixture} from '../src/model/micro/tensor';
import {attention} from '../src/model/micro/mixers';
function finiteCheck(parameters:Tensor[],fn:()=>number,count=5){let checked=0,max=0;for(const p of parameters)for(const i of new Set(Array.from({length:Math.min(count,p.length)},(_,i)=>Math.floor(i*(p.length-1)/Math.max(1,Math.min(count,p.length)-1))))){const value=p.data[i],analytic=p.grad[i],eps=1e-6;p.data[i]=value+eps;const plus=fn();p.data[i]=value-eps;const minus=fn();p.data[i]=value;const finite=(plus-minus)/(2*eps),error=Math.abs(analytic-finite)/Math.max(1,Math.abs(analytic),Math.abs(finite));max=Math.max(max,error);assert.ok(error<3e-5,`${p.name}[${i}] analytic ${analytic}, finite ${finite}, error ${error}`);checked++;}return {checked,max};}
const tensor=(shape:number[],name:string)=>new Tensor(shape,Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>Math.sin(i*.71+.31)*.25),name);
function objective(t:Tape,out:Tensor,backward:boolean){let sum=0;for(let i=0;i<out.length;i++){const c=Math.sin(i+.6);sum+=c*out.data[i];if(backward)out.grad[i]=c;}if(backward)for(let i=t.nodes.length-1;i>=0;i--)t.nodes[i].backward();return sum;}
for(const family of ['gpt-oss','llama','gemma'] as MicroFamily[])test(`${family}: complete small model finite differences cover every learned tensor`,()=>{
 const m=new MicroModel({family,hidden:4,layers:family==='gemma'?6:2,intermediate:6,queryHeads:2,kvHeads:1,headDim:4,experts:3,expertsPerToken:2,localWindow:2,seed:17}),ids=[2,...m.tokenizer.encode('the cat sat')],targets=m.tokenizer.encode('the cat sat on');m.zeroGrad();const p=m.loss(ids,targets);p.tape.backward(p.loss);const result=finiteCheck(m.parameters,()=>m.loss(ids,targets).loss.data[0]);console.log(family,result);
});
test('sliding GQA with learned attention sinks has exact query/key/value/sink adjoints',()=>{
 const q=tensor([4,16],'q'),k=tensor([4,8],'k'),v=tensor([4,8],'v'),sink=tensor([4],'sinks');const run=(backward=false)=>{const t=new Tape();return objective(t,attention(t,q,k,v,4,2,4,{window:2,sinks:sink}),backward);};run(true);finiteCheck([q,k,v,sink],()=>run(),30);
});
test('GELU and GPT clipped SwiGLU adjoints cover active and clamped coordinates',()=>{
 const g=new Tensor([1,5],[-8,-2,.5,6.8,7.2],'gate'),u=new Tensor([1,5],[-9,-6,.7,6.9,9],'up');const run=(backward=false)=>{const t=new Tape();return objective(t,gelu(t,clippedSwiGLU(t,g,u)),backward);};run(true);finiteCheck([g,u],()=>run());
});
test('top-k expert routing differentiates selected scores and actual weighted expert outputs',()=>{
 const logits=new Tensor([2,4],[1.1,.7,-.4,-1,.2,.8,-.1,1.4],'router'),experts=Array.from({length:4},(_,i)=>tensor([2,3],`expert.${i}`));experts.forEach((e,i)=>e.data.forEach((_,j)=>e.data[j]+=i*.1));const run=(backward=false)=>{const t=new Tape();return objective(t,expertMixture(t,topKRouter(t,logits,2),experts),backward);};run(true);finiteCheck([logits,...experts],()=>run(),8);assert.equal(logits.grad[2],0);assert.equal(logits.grad[3],0);assert.equal(logits.grad[4],0);assert.equal(logits.grad[6],0);
});
