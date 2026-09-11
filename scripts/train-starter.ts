/** Reproduce the shipped [micro] checkpoint from a seeded random initialization. */
import {mkdir,writeFile} from 'node:fs/promises';
import {cpus,platform,release,arch} from 'node:os';
import {fileURLToPath} from 'node:url';
import {MicroModel} from '../src/model/micro/model';
const destination=fileURLToPath(new URL('../public/data/',import.meta.url));
const settings={seed:42,steps:300,batchSize:4,learningRate:0.05};
const model=new MicroModel({seed:settings.seed});
const prompts=['the cat sat on','the cat sat on the','the dog sat on the'];
const probes=()=>Object.fromEntries(prompts.map(prompt=>{const p=model.infer(prompt,{record:false});return[prompt,{prediction:p.prediction,top:p.top}];}));
const initial={...model.evaluate(),probes:probes()};
const history:{step:number;batchLoss:number;batchAccuracy:number}[]=[];
const start=performance.now();
for(let i=0;i<settings.steps;i++){const result=model.trainStep({batchSize:settings.batchSize,learningRate:settings.learningRate,predictions:false,record:false});history.push({step:result.step,batchLoss:result.loss,batchAccuracy:result.accuracy});if((i+1)%50===0)console.log(`Step ${i+1}/300 · batch loss ${result.loss.toFixed(6)}`);}
const trainingMilliseconds=performance.now()-start;
const final={...model.evaluate(),probes:probes()};
const checkpoint=model.exportState();
const benchmark={source:'Actual deterministic Micro-Qwen CPU training; no Qwen weights',createdAt:new Date().toISOString(),settings,config:model.config,parameterCount:model.parameterCount,vocabulary:model.tokenizer.vocab,environment:{runtime:process.version,platform:platform(),osRelease:release(),architecture:arch(),cpuModel:cpus()[0]?.model??'Unavailable',logicalCPUs:cpus().length},trainingMilliseconds,initial,final,history,notes:['Timing covers the 300 forward/backward/update steps; initial/final complete-corpus evaluations and checkpoint serialization are excluded.','history contains sampled-batch metrics; initial/final contain complete-corpus metrics.','Checkpoint stores exact Float64 values as round-trippable JSON numbers and includes RNG state.','Primary math runs in a deterministic CPU worker in the browser. This Node measurement is not a browser performance guarantee.']};
await mkdir(destination,{recursive:true});
await writeFile(`${destination}micro-checkpoint.json`,JSON.stringify(checkpoint));
await writeFile(`${destination}micro-benchmark.json`,JSON.stringify(benchmark,null,2));
console.log(JSON.stringify({trainingMilliseconds,initialLoss:initial.loss,finalLoss:final.loss,accuracy:final.accuracy,predictions:Object.fromEntries(Object.entries(final.probes).map(([k,v])=>[k,v.top[0]]))},null,2));
