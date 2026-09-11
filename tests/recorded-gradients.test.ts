import test from 'node:test';
import assert from 'node:assert/strict';
import {MicroModel} from '../src/model/micro/model';
import {Tensor,Tape,type TensorView} from '../src/model/micro/tensor';
import {attention,deltaRule} from '../src/model/micro/mixers';
const tensor=(shape:number[],name:string,offset=0)=>new Tensor(shape,Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>Math.sin(i*.71+.3)*.2+offset),name);
function scalarObjective(tape:Tape,value:Tensor){const coefficients=Array.from({length:value.length},(_,i)=>Math.sin(i+.4)),loss=new Tensor([1],[coefficients.reduce((sum,c,i)=>sum+c*value.data[i],0)]);tape.add(loss,[value],()=>{for(let i=0;i<value.length;i++)value.grad[i]+=loss.grad[0]*coefficients[i];},'Test objective','dot','Weighted test objective','');tape.backward(loss);}
function complete(view:TensorView,where:string){assert.equal(view.gradient?.length,view.values.length,`${where}: ${view.name} gradient shape`);assert.ok(view.gradient?.every(Number.isFinite),`${where}: finite gradients`);}
test('every differentiable recorded training input and output has an actual gradient',()=>{
 const m=new MicroModel({hidden:4,layers:4,intermediate:6,queryHeads:2,headDim:8,deltaKeyDim:2,deltaValueDim:2});
 const result=m.trainStep({record:true,batchSize:2,predictions:false});
 for(const step of result.steps.filter(s=>s.phase==='forward')){
   if(step.name.startsWith('Word tokenization')){assert.equal(step.outputs[0].gradient,undefined,'Discrete token IDs have no invented gradient');continue;}
   if(step.name==='Training batch'){
     for(const output of step.outputs){if(output.name.includes('input and target IDs'))assert.equal(output.gradient,undefined);else complete(output,step.name);}
     continue;
   }
   step.inputs.forEach(v=>complete(v,step.name));step.outputs.forEach(v=>complete(v,step.name));
 }
 const summary=result.steps[0];for(let i=0;i<result.batch.length;i++){
   const example=result.batch[i],probability=summary.outputs[i*4+1],tokenLoss=summary.outputs[i*4+2],sentenceLoss=summary.outputs[i*4+3];
   for(let j=0;j<example.targetProbabilities.length;j++){assert.equal(probability.gradient![j],-1/(result.batchSize*example.targetProbabilities.length*example.targetProbabilities[j]));assert.equal(tokenLoss.gradient![j],1/(result.batchSize*example.tokenLosses.length));}
   assert.equal(sentenceLoss.gradient![0],1/result.batchSize);
 }
});
test('manual attention score and softmax records expose the same adjoints as the fused attention graph',()=>{
 const tape=new Tape(true),q=tensor([3,16],'q'),k=tensor([3,8],'k'),v=tensor([3,8],'v');
 scalarObjective(tape,attention(tape,q,k,v,4,2,4));
 const score=tape.steps.find(s=>s.name==='Q · Kᵀ, scaling and causal mask')!,softmax=tape.steps.find(s=>s.name==='Causal attention softmax')!;
 assert.deepEqual(score.inputs[0].gradient,Array.from(q.grad));assert.deepEqual(score.inputs[1].gradient,Array.from(k.grad));assert.deepEqual(softmax.inputs[0].gradient,score.outputs[0].gradient);
 for(let r=0;r<3;r++){
   let dot=0;for(let s=0;s<=r;s++)dot+=softmax.outputs[0].values[r*3+s]*softmax.outputs[0].gradient![r*3+s];
   for(let s=0;s<3;s++){const expected=s<=r?softmax.outputs[0].values[r*3+s]*(softmax.outputs[0].gradient![r*3+s]-dot):0;assert.ok(Math.abs(expected-softmax.inputs[0].gradient![r*3+s])<1e-13);}
 }
});
test('recorded Delta state input adjoints match finite differences of the displayed state update',()=>{
 const tape=new Tape(true),q=tensor([3,2],'q'),k=tensor([3,2],'k'),v=tensor([3,2],'v'),a=tensor([3,1],'a'),b=tensor([3,1],'b'),A=tensor([1],'A'),dt=tensor([1],'dt',-2);
 scalarObjective(tape,deltaRule(tape,q,k,v,a,b,A,dt,1,1,2,2));
 for(const step of tape.steps.filter(s=>s.name.startsWith('Delta rule state update'))){
   step.inputs.forEach(input=>complete(input,step.name));complete(step.outputs[0],step.name);
   const objective=()=>{const [previous,key,value,gates]=step.inputs.map(v=>v.values),[decay,beta]=gates,upstream=step.outputs[0].gradient!;let sum=0;for(let j=0;j<2;j++){let prediction=0;for(let i=0;i<2;i++)prediction+=key[i]*previous[i*2+j]*decay;const delta=beta*(value[j]-prediction);for(let i=0;i<2;i++)sum+=(previous[i*2+j]*decay+key[i]*delta)*upstream[i*2+j];}return sum;};
   for(const input of step.inputs)for(let i=0;i<input.values.length;i++){const old=input.values[i],eps=1e-6;input.values[i]=old+eps;const plus=objective();input.values[i]=old-eps;const minus=objective();input.values[i]=old;const finite=(plus-minus)/(2*eps);assert.ok(Math.abs(finite-input.gradient![i])<1e-9,`${step.name} ${input.name}[${i}]: ${finite} vs ${input.gradient![i]}`);}
 }
});
