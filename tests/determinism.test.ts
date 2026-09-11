import test from 'node:test';
import assert from 'node:assert/strict';
import {MicroModel} from '../src/model/micro/model';
test('same seed yields identical weights, batches, losses, and updates',()=>{
 const config={hidden:8,layers:4,intermediate:12,queryHeads:2,kvHeads:1,headDim:8,deltaKeyHeads:1,deltaValueHeads:3,deltaKeyDim:2,deltaValueDim:2,seed:19};const a=new MicroModel(config),b=new MicroModel(config),losses:number[]=[];
 for(let i=0;i<5;i++){const x=a.trainStep({batchSize:2,predictions:false}),y=b.trainStep({batchSize:2,predictions:false});assert.equal(x.loss,y.loss);assert.deepEqual(x.batch,y.batch);losses.push(x.loss);}
 for(let i=0;i<a.parameters.length;i++)assert.deepEqual(a.parameters[i].data,b.parameters[i].data);
 assert.deepEqual(a.infer('the cat sat on',{record:false}),b.infer('the cat sat on',{record:false}));
 a.reset();const fresh=new MicroModel(config);assert.equal(a.infer('the cat',{record:false}).logits[0],fresh.infer('the cat',{record:false}).logits[0]);console.log('Deterministic loss curve:',losses);
});
test('recorded updates equal the actual SGD update, including clipping',()=>{
 const m=new MicroModel({hidden:4,layers:4,intermediate:6,queryHeads:2,headDim:8,deltaKeyDim:2,deltaValueDim:2}),result=m.trainStep({batchSize:2,learningRate:.03,record:true,predictions:false});
 const updates=result.steps.filter(s=>s.phase==='update');assert.equal(updates.length,m.parameters.length);
 for(const s of updates){for(let i=0;i<s.outputs[0].values.length;i++){assert.equal(s.outputs[0].values[i],s.inputs[0].values[i]-.03*result.clipScale*s.inputs[1].values[i]);}}
 for(const example of result.batch){assert.ok(Math.abs(example.loss-example.tokenLosses.reduce((a,b)=>a+b,0)/example.tokenLosses.length)<1e-12);example.targetProbabilities.forEach((p,i)=>assert.ok(Math.abs(-Math.log(p)-example.tokenLosses[i])<1e-12));}
 for(const s of result.steps.filter(s=>s.phase==='forward'&&s.name!=='Training batch'&&!s.name.startsWith('Word tokenization')))assert.equal(s.outputs[0].gradient?.length,s.outputs[0].values.length,`Missing exact output gradient for ${s.name}`);
 assert.ok(result.steps.some(s=>s.name.startsWith('Delta rule state update')));assert.ok(result.steps.some(s=>s.name==='Q · Kᵀ, scaling and causal mask'));
});
