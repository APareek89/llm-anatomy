import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {RealTokenizer} from '../src/data/tokenizer';

const data=JSON.parse(readFileSync(new URL('../public/data/tokenizer.json',import.meta.url),'utf8'));
const fixtures=JSON.parse(readFileSync(new URL('./fixtures/tokenizer-oracle.json',import.meta.url),'utf8')) as {oracle:string;samples:{text:string;ids:number[];decoded:string}[]};
const tokenizer = new RealTokenizer(data);

test(`Real Qwen tokenizer matches ${fixtures.oracle} on multilingual, Unicode, whitespace, code and special tokens`,()=>{
  for(const sample of fixtures.samples) {
    assert.deepEqual(tokenizer.encode(sample.text),sample.ids,`Token IDs differ for ${JSON.stringify(sample.text)}`);
    assert.equal(tokenizer.decode(sample.ids),sample.decoded,`Decoded output differs for ${JSON.stringify(sample.text)}`);
  }
});
