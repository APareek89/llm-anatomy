/** Produce actual independently trained starter checkpoints for the added model families. */
import {mkdir,writeFile} from 'node:fs/promises';
import {cpus,platform,release,arch} from 'node:os';
import {fileURLToPath} from 'node:url';
import {MicroModel,type MicroFamily} from '../src/model/micro/model';
const destination=fileURLToPath(new URL('../public/data/',import.meta.url));
const environment={runtime:process.version,platform:platform(),osRelease:release(),architecture:arch(),cpuModel:cpus()[0]?.model??'Unavailable',logicalCPUs:cpus().length};
await mkdir(destination,{recursive:true});
for(const family of ['gpt-oss','llama','gemma'] as MicroFamily[]){
 const settings={family,seed:42,steps:300,batchSize:4,learningRate:.05},model=new MicroModel({family,seed:settings.seed});
 const probes=()=>Object.fromEntries(['the cat sat on','the cat sat on the','the dog sat on the'].map(prompt=>{const r=model.infer(prompt,{record:false});return [prompt,{prediction:r.prediction,top:r.top}];}));
 const initial={...model.evaluate(),probes:probes()},history:{step:number;batchLoss:number;batchAccuracy:number}[]=[];
 const start=performance.now();for(let i=0;i<settings.steps;i++){const r=model.trainStep({batchSize:settings.batchSize,learningRate:settings.learningRate,predictions:false});history.push({step:r.step,batchLoss:r.loss,batchAccuracy:r.accuracy});}
 const trainingMilliseconds=performance.now()-start,final={...model.evaluate(),probes:probes()};
 const benchmark={source:`Actual separately trained ${family} micro model; zero official model weights`,createdAt:new Date().toISOString(),settings,config:model.config,parameterCount:model.parameterCount,vocabulary:model.tokenizer.vocab,environment,trainingMilliseconds,initial,final,history,notes:['Timing covers 300 actual CPU forward/backward/SGD updates, excluding evaluations and serialization.','All new micro-family inference includes the same BOS marker used during training.','GPT uses real top-k routing, expert gradients, learned sinks, alternating sliding/full attention and clipped SwiGLU.','Llama/Gemma share embedding and output-head weights; their embedding receives gradients through both paths.','Long-context RoPE frequency scaling is omitted in these 64-token teaching models; ordinary full RoPE is used.','Tiny GPT reference computes all candidate experts then combines top-k; this reproduces sparse routing numerics but not a production sparse-kernel speedup.','Checkpoint retains exact round-trippable Float64 numbers and RNG state.']};
 await writeFile(`${destination}micro-${family}-checkpoint.json`,JSON.stringify(model.exportState()));
 await writeFile(`${destination}micro-${family}-benchmark.json`,JSON.stringify(benchmark,null,2));
 console.log(JSON.stringify({family,parameters:model.parameterCount,milliseconds:trainingMilliseconds,initialLoss:initial.loss,finalLoss:final.loss,accuracy:final.accuracy,probes:Object.fromEntries(Object.entries(final.probes).map(([prompt,p])=>[prompt,p.top[0]]))},null,2));
}
