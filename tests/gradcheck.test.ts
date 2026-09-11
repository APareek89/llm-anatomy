import test from 'node:test';
import assert from 'node:assert/strict';
import {MicroModel} from '../src/model/micro/model';
import {Tensor,Tape} from '../src/model/micro/tensor';
import {attention,rope,deltaRule} from '../src/model/micro/mixers';
function check(f:()=>number,parameters:Tensor[],perTensor=4){const eps=1e-5;let count=0,max=0;for(const p of parameters){const indices=new Set(Array.from({length:Math.min(perTensor,p.length)},(_,i)=>Math.floor(i*(p.length-1)/Math.max(1,Math.min(perTensor,p.length)-1))));for(const i of indices){const analytic=p.grad[i],old=p.data[i];p.data[i]=old+eps;const plus=f();p.data[i]=old-eps;const minus=f();p.data[i]=old;const numeric=(plus-minus)/(2*eps),error=Math.abs(analytic-numeric)/Math.max(1,Math.abs(analytic),Math.abs(numeric));max=Math.max(max,error);assert.ok(error<3e-5,`${p.name}[${i}]: analytic=${analytic} finite=${numeric}, error=${error}`);count++;}}return {count,max};}
test('full Micro-Qwen backward matches finite differences through all parameter tensors',()=>{
 const model=new MicroModel({hidden:4,layers:4,intermediate:6,queryHeads:2,kvHeads:1,headDim:8,deltaKeyHeads:1,deltaValueHeads:3,deltaKeyDim:2,deltaValueDim:2,seed:13}),ids=model.tokenizer.encode('the cat sat on'),targets=model.tokenizer.encode('cat sat on the');
 model.zeroGrad();const pass=model.loss(ids,targets);pass.tape.backward(pass.loss);const result=check(()=>model.loss(ids,targets).loss.data[0],model.parameters,5);assert.ok(result.count>200);console.log(`Full model: ${result.count} finite differences; max normalized error ${result.max}`);
});
test('GQA mapping with multiple KV heads and nonzero rotary positions has correct gradients',()=>{
 const make=(shape:number[],name:string)=>new Tensor(shape,Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>Math.sin(i*0.7+0.3)*0.2),name),q=make([4,32],'q'),k=make([4,16],'k'),v=make([4,16],'v');
 const run=(backward=false)=>{const t=new Tape(),qr=rope(t,q,4,8,.25,1e7),kr=rope(t,k,2,8,.25,1e7),o=attention(t,qr,kr,v,4,2,8);let sum=0;for(let i=0;i<o.length;i++){const factor=Math.sin(i+1);sum+=o.data[i]*factor;if(backward)o.grad[i]=factor;}if(backward)for(let i=t.nodes.length-1;i>=0;i--)t.nodes[i].backward();return sum;};run(true);check(()=>run(),[q,k,v],30);
});
test('Delta recurrence finite differences include multiple grouped heads and memory carry',()=>{
 const make=(shape:number[],name:string,offset=0)=>new Tensor(shape,Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>Math.sin(i*.71+.2)*.25+offset),name),q=make([4,4],'q'),k=make([4,4],'k'),v=make([4,18],'v'),a=make([4,6],'a'),b=make([4,6],'b'),A=make([6],'A'),dt=make([6],'dt',-2);
 const run=(backward=false)=>{const t=new Tape(),o=deltaRule(t,q,k,v,a,b,A,dt,2,6,2,3);let sum=0;for(let i=0;i<o.length;i++){const factor=Math.sin(i+1);sum+=o.data[i]*factor;if(backward)o.grad[i]=factor;}if(backward)for(let i=t.nodes.length-1;i>=0;i--)t.nodes[i].backward();return sum;};run(true);check(()=>run(),[q,k,v,a,b,A,dt],20);
});
