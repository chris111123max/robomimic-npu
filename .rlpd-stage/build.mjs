import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const ROOT='E:/robosuit/RLPD_论文汇报_20260903',B=ROOT+'/build';
const OUT=ROOT+'/RLPD_利用离线数据提升在线强化学习效率_陈元凯.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(B+'/template-starter.pptx'));
const C={g:'#006C39',mid:'#218458',pale:'#EAF5EE',gray:'#F2F4F3',ink:'#202A25',muted:'#5C6760',line:'#D2DFD6',white:'#FFFFFF'};let id=0;
function text(s,t,x,y,w,h,size=24,bold=false,color=C.ink,fill='none',align='left'){
 const o=s.shapes.add({name:`content-${++id}`,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});
 o.text=t;o.text.style={fontSize:size,typeface:'微软雅黑',bold,color,alignment:align,verticalAlignment:'top',wrap:'square',autoFit:'none',lineSpacing:1.12,insets:{left:0,right:0,top:0,bottom:0}};return o;
}
function box(s,t,x,y,w,h,size=24,fill=C.pale,color=C.g){const o=text(s,t,x,y,w,h,size,true,color,fill,'center');o.text.style={verticalAlignment:'middle',insets:{left:10,right:10,top:5,bottom:5}};return o;}
function rule(s,x,y,w){s.shapes.add({name:`content-rule-${++id}`,geometry:'rect',position:{left:x,top:y,width:w,height:1.4},fill:C.line,line:{fill:'none',width:0}});}
function arrow(s,a,b,from='right',to='left'){return s.shapes.connect(a,b,{kind:'elbow',fromSide:from,toSide:to,line:{fill:C.mid,width:2},tail:{type:'triangle',width:'sm',length:'sm'}});}
async function image(s,name,x,y,w,h){let a=await fs.readFile(B+'/assets/'+name+'.png');return s.images.add({name:'paper-'+name,blob:a.buffer.slice(a.byteOffset,a.byteOffset+a.byteLength),contentType:'image/png',alt:'RLPD论文'+name+'高清裁图；未经重画',fit:'contain',position:{left:x,top:y,width:w,height:h}});}
function header(s,n,t){s.shapes.items.find(q=>q.name==='文本框 6').text.replace('5',String(n));s.shapes.items.find(q=>q.name==='文本框 1').text.replace('实验结果',t);}
function notes(s,cite,body){text(s,'来源：RLPD.pdf，'+cite,60,622,1160,14,12,false,C.muted);s.speakerNotes.textFrame.setText(body+'\n\n[Sources]\nF:/桌面/自控ppt/论文/IL+RL/RLPD.pdf\n'+cite+'\n[/Sources]');}
function table(s,v,x,y,w,h,widths,size=22,hi=()=>false){let t=s.tables.add({rows:v.length,columns:v[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values:v});t.borders.assign({fill:C.white,width:1});t.cells.block({row:0,column:0,rowCount:v.length,columnCount:v[0].length}).assign({textStyle:{typeface:'微软雅黑',fontSize:size},margins:{left:8,right:8,top:4,bottom:4},anchor:'center'});for(let r=0;r<v.length;r++)for(let c=0;c<v[0].length;c++){let a=t.getCell(r,c);a.fill=r===0?C.g:hi(r,c)?'#D9EEDD':r%2?C.gray:C.white;a.text.style={typeface:'微软雅黑',fontSize:size,color:r===0?C.white:hi(r,c)?C.g:C.ink,bold:r===0||hi(r,c),alignment:'left',verticalAlignment:'middle'};}return t;}
const slides=p.slides.items;
// P1 — Paper cover, using Fig.1 (not reused on the result page).
{
let s=slides[0];header(s,1,'RLPD：离线数据提升在线学习效率');
text(s,'Efficient Online Reinforcement Learning with Offline Data',60,112,1160,44,33,true,C.g);
text(s,'RLPD：Reinforcement Learning with Prior Data',60,164,1160,29,23,false,C.g);
text(s,'ICML 2023｜Philip J. Ball · Laura Smith · Ilya Kostrikov · Sergey Levine',60,202,1160,30,21,false,C.muted);
const blocks=[['① 问题','在线交互成本高，稀疏奖励下难以发现成功轨迹。\n仅靠新采样数据，策略改进缓慢。'],['② 核心想法','利用已有专家示范或次优探索轨迹，\n为在线学习提供有效的先验信息。'],['③ 方法','不做离线预训练，也不加入模仿约束。\n让离策略强化学习稳定混合离线与在线数据。']];
for(let i=0;i<3;i++){let y=268+112*i;text(s,blocks[i][0],60,y,620,34,27,true,C.g);text(s,blocks[i][1],60,y+42,620,66,23);}
await image(s,'fig1',739,252,481,330);
text(s,'图1：6个蚂蚁迷宫任务的平均归一化回报',730,586,490,25,18,false,C.muted,'none','center');
notes(s,'第1页图1、摘要与引言。10个随机种子，阴影为1个标准差。','汇报人：陈元凯；导师：张金会。作者前三位为共同贡献。图1聚合所有6个D4RL AntMaze任务，RLPD由于提前收敛仅运行300K在线步。横轴是环境交互步数，纵轴是归一化回报。只以RLPD论文为内容来源。');
}
// P2 — Research question, no introductory RL/SAC teaching.
{
let s=slides[1];header(s,2,'普通离策略方法能直接用离线数据吗？');
text(s,'已有数据能帮助探索，但“把数据放进去”并不等于“稳定地利用数据”。',60,114,1160,45,28,true,C.g);
const rows=[['离线预训练 → 在线微调','需要额外训练阶段与超参数；流程和调试成本增加。\n较好的初始策略，也不一定带来持续的在线改进。'],['行为约束／模仿约束','把策略行为拉向已有数据，可能限制在线探索。\n当离线轨迹次优时，超越数据策略也可能受到限制。'],['直接 SAC ＋ 离线数据','没有额外稳定化设计时，可能出现虚假的高 Q 值。\n数据外区域的价值外推失控，使学习不稳定。']];
rows.forEach((r,i)=>{let y=202+i*109;text(s,r[0],60,y,345,37,25,true,C.g);text(s,r[1],450,y,770,78,25);if(i<2)rule(s,60,y+88,1160);});
box(s,'RLPD 的问题：只做最小改动，能否稳定利用先验数据开展在线学习？',60,550,1160,60,26);
notes(s,'第1–4页，第1节、第4.1–4.2节。','离策略强化学习可以训练来自其他策略的数据，但函数逼近对未覆盖区域的外推仍会导致不稳定。RLPD不做离线强化学习预训练，也不添加行为克隆辅助损失。这里陈述先前方法的潜在限制，不声称所有预训练方法或行为约束必然失败。');
}
// P3 — Editable redraw of Algorithm 1.
{
let s=slides[2];header(s,3,'RLPD：在线与离线数据共同训练');
text(s,'策略和价值网络随机初始化；离线数据已经存在，但不先做离线训练。',60,112,1160,39,27,true,C.g);
let env=box(s,'策略与环境交互',60,185,270,74,25);
let replay=box(s,'在线经验池 R\n初始为空，持续加入新经验',438,178,337,88,23);
let offline=box(s,'离线数据池 D\n专家示范或次优轨迹',60,354,365,88,24);
let batch=box(s,'合并训练批次\n50% 来自 R ＋ 50% 来自 D',863,265,357,92,23,C.g,C.white);
let update=box(s,'G 次价值网络更新\n随后 1 次策略网络更新',863,425,357,90,24);
arrow(s,env,replay);arrow(s,replay,batch,'right','top');arrow(s,offline,batch);arrow(s,batch,update,'bottom','top');
text(s,'在线采样 128 条',782,187,244,32,20,false,C.g);
text(s,'离线采样 128 条',455,411,320,33,20,false,C.g);
text(s,'关键组件',60,479,720,33,26,true,C.g);
text(s,'对称采样 ＋ LayerNorm\n价值网络集成 ＋ 高 UTD',60,522,730,71,27,true,C.g);
text(s,'更新后的策略继续交互，\n新经验持续补充 R。',863,548,357,58,23);
notes(s,'第5页 Algorithm 1；第18页表1。图为原算法的中文简化重绘。','每个环境步先把转移加入在线池R，再做G次价值更新；每次重采样R、D各128条。使用价值网络集成；目标从随机选取的1或2个目标网络计算，取子集大小取决于环境。更新价值网络和目标网络后，每个环境步更新一次策略（算法使用最后的混合批次）。策略目标使用集成价值平均。高UTD并非将策略也每步更新G次。');
}
// P4 — Symmetric sampling.
{
let s=slides[3];header(s,4,'对称采样：离线和在线数据各占一半');
box(s,'训练批次 = 50% 在线经验 ＋ 50% 离线数据',60,113,1160,55,29);
table(s,[['采样方式','潜在问题／权衡','RLPD 的处理'],['只用离线数据初始化经验池','小数据集可能被稀释；大数据集可能挤占在线数据','始终独立采样离线数据'],['100% 离线数据','缺少在线分布与新经验的反馈','保留在线数据'],['100% 在线数据','放弃已有示范或探索轨迹的帮助','保留离线数据'],['50/50 对称采样','平衡先验利用与在线适应','作为稳健的默认方案']],60,187,1160,232,[310,535,315],21,(r,c)=>r===4);
await image(s,'fig12',60,438,675,173);
text(s,'50% 不是唯一可行比例',775,444,445,38,27,true,C.g);
text(s,'25% 或 75% 在部分任务也有效。\n论文中，50/50 在收敛速度、\n跨种子方差和最终性能之间，\n提供了较稳健的折中。',775,491,445,115,23);
notes(s,'第3页第4.1节；第8页图10–12（配图为图12）。','图12横轴为环境步数，纵轴为归一化回报。深色虚线表示50%离线数据；另比较0%、25%、75%、100%。作者指出25%离线比例可略微提高walker2d-medium的最终性能，但会牺牲稀疏任务的方差和样本效率。对称采样不仅提高奖励密度，也可减少依赖高方差在线数据导致的不稳定。大量离线数据直接塞入回放池时，还可能导致在线数据采样不足。');
}
// P5 — LayerNorm: four large original curves.
{
let s=slides[4];header(s,5,'LayerNorm：抑制数据外的价值发散');
text(s,'先验数据覆盖有限；价值网络可能给未见过的状态—动作赋予虚假高 Q 值。',60,111,1160,42,27,true,C.g);
await image(s,'fig2',60,163,1160,286);
text(s,'图2：蓝线＝无 LayerNorm；橙线＝有 LayerNorm。每个任务分别展示 Q 值与归一化回报。',60,454,1160,28,20,false,C.muted);
text(s,'无归一化：外推误差被不断放大',60,496,555,36,26,true,C.g);
text(s,'仅做对称采样仍可能出现 Q 值发散，\n伴随回报低、跨种子波动大。',60,539,555,67,24);
text(s,'加入 LayerNorm：约束价值表示',665,496,555,36,26,true,C.g);
text(s,'归一化价值网络中间表示，抑制无约束外推。\n不把策略强行拉回离线数据附近，仍允许探索。',665,539,555,67,23);
notes(s,'第3页图2、第4页第4.2节及图3。原图纵轴含对数刻度。','论文分析归一化特征如何约束数据外Q值，界仍依赖输出层权重范数；不能理解为训练过程中Q被固定常数硬截断。图2两个任务为AntMaze Large Play和Pen Sparse，蓝线无LayerNorm、橙线有LayerNorm。归一化改变价值网络的外推性质，而不是显式约束策略分布。对某些密集奖励任务，LayerNorm优势没有困难稀疏任务显著，见附录。');
}
// P6 — Sample-efficient RL.
{
let s=slides[5];header(s,6,'价值网络集成＋高 UTD：加速学习');
text(s,'先验数据的价值，需要通过贝尔曼更新向更早的状态传播。',60,113,1160,39,28,true,C.g);
table(s,[['组件','作用'],['高 UTD','每个环境步进行更多价值更新，\n让每条数据被更充分利用'],['价值网络集成','结合多个价值估计，缓解高更新率\n下的过拟合与不稳定'],['LayerNorm','限制价值网络在数据外的\n过度外推'],['图像随机平移增强','提升像素输入任务的\n泛化能力']],60,171,725,278,[230,495],23);
await image(s,'fig6',829,173,391,363);
text(s,'更多更新需要配合稳定化设计',60,486,725,37,26,true,C.g);
text(s,'单纯提高 UTD 也可能加重过拟合。\n论文默认使用 10 个价值网络；\n状态输入实验默认 UTD=20。',60,530,725,80,22);
text(s,'图6：视觉猎豹奔跑＋专家数据。\nUTD 从 1 提高到 10，\nRLPD 的样本效率显著改善。',829,545,391,66,21,false,C.g);
notes(s,'第4页第4.3节；第7页图6、第8页图9、第18页表1。','UTD是每个环境步对应的梯度更新次数。高UTD不是免费的计算收益；本页讲样本效率，而不是更少算力。论文采用随机集成方法，并通过图9比较权重衰减、随机失活与价值网络集成，集成在困难稀疏任务中更可靠。图6只针对Cheetah Run Expert视觉任务单独提高UTD到10，不应误写所有视觉实验均采用UTD20。不同环境还需要考虑双Q最小值、价值目标熵项与网络深度，论文并非声称四个组件就消除所有环境敏感性。');
}
// P7 — Large evidence curves plus concise native summary table.
{
let s=slides[6];header(s,7,'RLPD：简单设计组合达到强性能');
text(s,'状态输入：21 个任务，10 个随机种子；曲线为均值，阴影为标准差。',60,110,1160,32,25,true,C.g);
['灵巧手操作｜稀疏奖励','蚂蚁迷宫｜稀疏奖励','行走控制｜密集奖励'].forEach((v,i)=>text(s,v,60+i*397,147,367,28,22,true,C.g,'none','center'));
await image(s,'fig4-plots',60,178,1160,161);
text(s,'横轴：环境交互步数（千步）；纵轴：归一化回报。深色虚线为 RLPD。',60,339,1160,24,17,false,C.muted);
text(s,'视觉输入：行走／猎豹奔跑／人形行走',60,365,626,29,22,true,C.g);
await image(s,'fig5',60,396,623,160);
table(s,[['实验','主要结果'],['状态输入','21 个任务总体匹配或超过既有方法'],['灵巧手开门','最高约 2.5× 归一化回报提升'],['蚂蚁迷宫','有效解决全部 6 个任务']],722,375,498,118,[135,363],18,(r,c)=>r===2);
text(s,'视觉任务中稳定优于纯在线方法，\n并在多种设置下超过行为克隆。',722,504,498,55,21);
box(s,'RLPD = 对称采样 ＋ LayerNorm ＋ 价值网络集成 ＋ 高 UTD',60,565,1160,29,23);
text(s,'核心贡献：通过关键设计，让普通离策略强化学习稳定利用先验数据。',60,598,1160,24,21,true,C.g);
notes(s,'第6页图4、第7页图5与正文。开门的2.5×是回报提升，不是成功率。','核心贡献不是复杂的新算法，而是证明普通离策略强化学习经关键设计后可以稳定利用先验数据。图4聚合21个状态输入任务：3个Sparse Adroit（转笔、开门、物体移位）；6个D4RL AntMaze；12个D4RL Locomotion。各类最强先前方法不同：前两者IQL加微调，后者Off2On；还比较SACfD。论文声称匹配或超过，是整体基准结果而非每条曲线每个时刻均胜出。Adroit归一化回报反映完成速度，不是单纯成功率；AntMaze的归一化回报则是100次评估中的成功比例。视觉图5比较中等/专家数据、行为克隆、纯在线和DrQ-v2；其归一化回报定义为单轮回报除以10。RLPD在多种视觉设置下超过行为克隆，但并非所有任务都超过专家行为克隆。');
}
await fs.mkdir(B+'/final-preview',{recursive:true});await fs.mkdir(B+'/final-layout',{recursive:true});
for(let i=0;i<slides.length;i++){
let stem='slide-'+String(i+1).padStart(2,'0');
await fs.writeFile(B+'/final-preview/'+stem+'.png',new Uint8Array(await (await p.export({slide:slides[i],format:'png',scale:1})).arrayBuffer()));
await fs.writeFile(B+'/final-layout/'+stem+'.layout.json',await (await slides[i].export({format:'layout'})).text());console.log('Rendered',i+1);
}
await fs.writeFile(B+'/final-inspect.ndjson',(await p.inspect({kind:'slide,shape,textbox,table,image',maxChars:1000000})).ndjson);
await(await PresentationFile.exportPptx(p)).save(OUT);console.log(OUT);
