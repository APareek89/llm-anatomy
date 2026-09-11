import test from 'node:test';
import assert from 'node:assert/strict';
import {MicroModel,type MicroFamily} from '../src/model/micro/model';
for(const family of ['gpt-oss','llama','gemma'] as MicroFamily[])test(`${family}: deterministic real 300-step learning, checkpoint and family mechanisms`,{timeout:120000},()=>{
 const m=new MicroModel({family}),initial=m.evaluate(),replica=new MicroModel({family});
 for(let i=0;i<3;i++){const a=m.trainStep({predictions:false}),b=replica.trainStep({predictions:false});assert.equal(a.loss,b.loss);assert.deepEqual(a.batch,b.batch);}
 for(let i=3;i<300;i++)m.trainStep({batchSize:4,learningRate:.05,predictions:false});
 const final=m.evaluate();assert.ok(final.loss<.85,`${family} final loss ${final.loss}`);assert.ok(final.loss<initial.loss/3);
 for(const [prompt,token] of [['the cat sat on','the'],['the cat sat on the','mat'],['the dog sat on the','rug']]){const r=m.infer(prompt,{record:false});assert.equal(r.prediction,token,`${family}: ${prompt}`);assert.ok(r.top[0].probability>.5);}
 const restored=new MicroModel({family});restored.importState(m.exportState());assert.deepEqual(restored.infer('the cat sat on',{record:false}).logits,m.infer('the cat sat on',{record:false}).logits);
 const result=m.trainStep({record:true,batchSize:2,predictions:false});for(const s of result.steps.filter(s=>s.phase==='forward'&&s.name!=='Training batch'&&!s.name.startsWith('Word tokenization'))){for(const v of [...s.inputs,...s.outputs]){assert.equal(v.gradient?.length,v.values.length,`${family} ${s.name} ${v.name}`);assert.ok(v.gradient?.every(Number.isFinite));}}
 if(family==='gpt-oss'){
   const routing=result.steps.filter(s=>s.name==='Top-k expert routing');assert.equal(routing.length,m.config.layers);
   for(const s of routing){const [rows,experts]=s.outputs[0].shape;for(let r=0;r<rows;r++){const row=s.outputs[0].values.slice(r*experts,(r+1)*experts);assert.equal(row.filter(v=>v>0).length,m.config.expertsPerToken);assert.ok(Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-12);}}
   const probabilities=result.steps.filter(s=>s.name==='Causal attention softmax');assert.ok(probabilities.every(s=>s.outputs[0].shape[1]===s.outputs[0].shape[0]+1));
   assert.ok(result.steps.some(s=>s.name==='GPT-OSS clipped SwiGLU'));assert.ok(result.steps.some(s=>s.name==='Sliding attention output projection'));assert.ok(result.steps.some(s=>s.name==='Full attention output projection'));
 }else{
   assert.equal(m.paramMap.has('lm_head.weight'),false,'Tied output head must share the actual embedding tensor');assert.ok(result.steps.some(s=>s.name==='Tied embedding LM head logits'));
   const unseen=m.tokenizer.lookup.get('<unk>')!,embedding=m.w('embed_tokens.weight');assert.ok(Array.from(embedding.grad.slice(unseen*m.config.hidden,(unseen+1)*m.config.hidden)).some(g=>g!==0),'An unseen input row must still receive the tied output-head gradient');
   if(family==='gemma'){assert.equal(result.steps.filter(s=>s.name==='Post-attention RMSNorm').length,6);assert.equal(result.steps.filter(s=>s.name==='Post-feed-forward RMSNorm').length,6);assert.equal(result.steps.filter(s=>s.name==='Sliding attention output projection').length,5);assert.equal(result.steps.filter(s=>s.name==='Full attention output projection').length,1);assert.ok(result.steps.some(s=>s.name==='GELU tanh gate'));assert.ok(result.steps.some(s=>s.name==='Gemma embedding scale'));}
 }
 console.log(`${family}: loss ${initial.loss.toFixed(6)} → ${final.loss.toFixed(6)}, corpus accuracy ${(final.accuracy*100).toFixed(2)}%`);
});
