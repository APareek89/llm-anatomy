/// <reference lib="webworker" />
import {MicroModel,type MicroConfig} from './model';
import {CORPUS} from './corpus';
import {gpuProjection} from '../gpu';
const scope=self as unknown as DedicatedWorkerGlobalScope;
let model=new MicroModel();let training=false,cancelled=false;
interface Request {id:string|number;type:'init'|'infer'|'train'|'reset'|'evaluate'|'stop'|'parameter'|'export'|'import'|'gpu';config?:Partial<MicroConfig>;seed?:number;prompt?:string;temperature?:number;record?:boolean;steps?:number;batchSize?:number;learningRate?:number;checkpoint?:ReturnType<MicroModel['exportState']>;checkpointUrl?:string;name?:string;input?:number[];weight?:number[];rows?:number;inputDim?:number;outputDim?:number;expected?:number[];}
function ready(id:Request['id'],type='ready'){scope.postMessage({id,type,config:model.config,vocab:model.tokenizer.vocab,parameterCount:model.parameterCount,parameters:model.metadata(),step:model.iteration,corpus:CORPUS,backend:'Deterministic Float64 CPU worker; optional WebGPU projection verification'});}
function validateSettings(m:Request){if(m.learningRate!==undefined&&(!Number.isFinite(m.learningRate)||m.learningRate<=0||m.learningRate>1))throw new Error('Learning rate must be greater than zero and at most 1.');if(m.steps!==undefined&&(!Number.isInteger(m.steps)||m.steps<1||m.steps>10000))throw new Error('Training steps must be an integer from 1 to 10000.');}
async function handle(m:Request){
  try {
    if(m.type==='stop'){cancelled=true;scope.postMessage({id:m.id,type:'stopping'});return;}
    if(m.type==='gpu'){scope.postMessage({id:m.id,type:'gpu',result:await gpuProjection(m.input??[],m.weight??[],m.rows??1,m.inputDim??1,m.outputDim??1,m.expected)});return;}
    if(training&&['init','train','reset','import'].includes(m.type))throw new Error('Training is in progress. Stop it before resetting or starting another run.');
    if(m.type==='init'){model=new MicroModel({...m.config,...(m.seed!==undefined?{seed:m.seed}:{})});if(m.checkpointUrl){const r=await fetch(m.checkpointUrl);if(!r.ok)throw new Error(`Checkpoint HTTP ${r.status}`);model.importState(await r.json());}else if(m.checkpoint)model.importState(m.checkpoint);ready(m.id);}
    else if(m.type==='infer')scope.postMessage({id:m.id,type:'inference',result:model.infer(m.prompt??'the cat sat on',{temperature:m.temperature,seed:m.seed,record:m.record??true})});
    else if(m.type==='reset'){model.reset(m.seed);ready(m.id,'reset');}
    else if(m.type==='evaluate')scope.postMessage({id:m.id,type:'evaluation',result:model.evaluate()});
    else if(m.type==='parameter'){const p=model.w(m.name??'embed_tokens.weight');scope.postMessage({id:m.id,type:'parameter',result:{name:p.name,shape:p.shape,values:Array.from(p.data),gradient:Array.from(p.grad)}});}
    else if(m.type==='export')scope.postMessage({id:m.id,type:'export',result:model.exportState()});
    else if(m.type==='import'){if(!m.checkpoint)throw new Error('Checkpoint required');model.importState(m.checkpoint);ready(m.id);}
    else if(m.type==='train'){
      validateSettings(m);training=true;cancelled=false;const count=m.steps??1,start=performance.now(),history:{step:number;loss:number;accuracy:number}[]=[];let result:ReturnType<MicroModel['trainStep']>|undefined;
      try {for(let i=0;i<count&&!cancelled;i++){const report=i===count-1||(i+1)%10===0;result=model.trainStep({batchSize:m.batchSize,learningRate:m.learningRate,record:(m.record??count===1)&&i===0,predictions:report||count===1});history.push({step:result.step,loss:result.loss,accuracy:result.accuracy});if(report||count===1)scope.postMessage({id:m.id,type:'progress',completed:i+1,total:count,milliseconds:performance.now()-start,result,history:history.slice(-10)});if(i%2===1)await new Promise<void>(resolve=>setTimeout(resolve,0));}
      scope.postMessage({id:m.id,type:'training',result,history,milliseconds:performance.now()-start,cancelled,evaluation:model.evaluate()});}finally{training=false;}
    }
  }catch(error){scope.postMessage({id:m.id,type:'error',error:error instanceof Error?error.message:String(error)});}
}
scope.addEventListener('message',event=>{void handle(event.data as Request);});
