import test from 'node:test';
import assert from 'node:assert/strict';
import {MicroModel} from '../src/model/micro/model';
test('the actual default eight-layer model learns the/mat/rug in 300 seeded SGD steps',{timeout:120000},()=>{
  const model=new MicroModel({seed:42}),initial=model.evaluate();
  const start=performance.now();
  for(let step=0;step<300;step++)model.trainStep({batchSize:4,learningRate:.05,record:false,predictions:false});
  const elapsed=performance.now()-start,final=model.evaluate();
  assert.ok(final.loss<.8,`Complete-corpus loss ${final.loss}`);
  assert.ok(final.loss<initial.loss/3,`Loss should materially decrease from ${initial.loss}`);
  assert.ok(final.accuracy>.7,`Complete-corpus accuracy ${final.accuracy}`);
  for(const [prompt,expected] of [['the cat sat on','the'],['the cat sat on the','mat'],['the dog sat on the','rug']]){
    const result=model.infer(prompt,{record:false});
    assert.equal(result.prediction,expected,`${prompt}: ${JSON.stringify(result.top.slice(0,3))}`);
    assert.ok(result.top[0].probability>.7,`${expected} needs confident learned probability`);
  }
  const restored=new MicroModel();restored.importState(model.exportState());
  assert.deepEqual(restored.infer('the dog sat on the',{record:false}).logits,model.infer('the dog sat on the',{record:false}).logits);
  const nextOriginal=model.trainStep({batchSize:4,predictions:false}),nextRestored=restored.trainStep({batchSize:4,predictions:false});
  assert.equal(nextOriginal.loss,nextRestored.loss,'Checkpoint must preserve RNG/batch progression');
  console.log(`300 true training steps: ${(elapsed/1000).toFixed(2)} s; corpus loss ${initial.loss.toFixed(6)} → ${final.loss.toFixed(6)}; accuracy ${(final.accuracy*100).toFixed(2)}%`);
});
