import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const B='E:/robosuit/RLPD_论文汇报_20260903/build';
await fs.mkdir(B+'/template-inspect/source-slides',{recursive:true});await fs.mkdir(B+'/template-inspect/layouts',{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load('F:/桌面/自控ppt/汇报/基础模板.pptx'));
await fs.writeFile(B+'/template-inspect/template-inspect.ndjson',(await p.inspect({kind:'slide,shape,textbox,image,layout',maxChars:1000000})).ndjson);
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];
 await fs.writeFile(B+'/template-inspect/source-slides/slide-'+(i+1)+'.png',new Uint8Array(await (await p.export({slide:s,format:'png',scale:0.8})).arrayBuffer()));
 await fs.writeFile(B+'/template-inspect/layouts/slide-'+(i+1)+'.json',await (await s.export({format:'layout'})).text());
}
console.log('Inspected all',p.slides.items.length,'source slides');
