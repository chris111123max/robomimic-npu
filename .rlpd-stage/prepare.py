from pathlib import Path
import json,pdfplumber
B=Path('E:/robosuit/RLPD_论文汇报_20260903/build')
(B/'assets').mkdir(parents=True,exist_ok=True)
with pdfplumber.open(r'F:\桌面\自控ppt\论文\IL+RL\RLPD.pdf') as pdf:
    (B/'paper-text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n'+(p.extract_text() or '') for i,p in enumerate(pdf.pages)),encoding='utf8')
    for n in [1,3,6,7,8]:
        pdf.pages[n-1].to_image(resolution=150).original.save(B/f'assets/page{n}.png')
        print('page',n,flush=True)
layout=json.loads((B/'template-inspect/layouts/slide-6.json').read_text(encoding='utf8'))
ids={e['id']:e['aid'] for e in layout['elements']}
roles=['paper cover','research question','training flow','sampling comparison','normalization evidence','sample efficiency components','experimental evidence and conclusion']
mapping={'sourceInventory':[{'sourceSlide':i,'type':'cover' if i==1 else 'body frame','inspected':True} for i in range(1,11)],'outputSlides':[{'outputSlide':i+1,'sourceSlide':6,'narrativeRole':role,'reuseMode':'duplicate-slide','editTargets':[{'action':'rewrite','sourceElementId':ids['7'],'reason':'Update page number; retain typography.'},{'action':'rewrite','sourceElementId':ids['2'],'reason':'Rewrite inherited title.'},{'action':'add','newPrimitiveAllowed':True,'mustNotOverlapInherited':True,'zone':{'left':58,'top':108,'width':1162,'height':528},'reason':'User approved native content inside this template body, and requests seven RLPD pages with text, tables and figures.'}]} for i,role in enumerate(roles)],'omittedSourceSlides':[{'sourceSlide':i,'reason':'Reuse slide6 body frame containing an editable heading; no separate generic cover requested.'} for i in range(1,11) if i!=6]}
(B/'template-frame-map.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf8')
(B/'template-audit.txt').write_text('All 10 source slides rendered and inspected. Source6 duplicates source2-10 body chrome but has editable title. Keep BIT logo, motto, green footer, page number block and small inherited ochre accents. Preserve layout33/master18 hierarchy. Inherited title 华文仿宋 37.33px bold; page number Century Gothic 48px bold white. New body content 微软雅黑 22-25px, charts directly cropped at 300dpi. Output slides all clone source6. No old paper content reused.',encoding='utf8')
(B/'deviation-log.txt').write_text('Explicitly authorized additions inside blank body region only. Titles shortened to fit inherited 839.55px frame; full topic appears on P1. Original curve colors retained for evidence fidelity. No unrelated topics or basic RL/SAC teaching. Fig4 and Fig5 labels/legends remain in cropped figures and are explained in Chinese. Algorithm1 redrawn in editable native shapes. Door 2.5x is normalized return, not success rate. High UTD is explained with state default and pixel ablation, not universal settings.',encoding='utf8')
(B/'source-notes.txt').write_text('Only scholarly source: supplied RLPD.pdf, Efficient Online Reinforcement Learning with Offline Data, ICML2023. P1 Fig1 and introduction; P2 Sec1/4; P3 Algorithm1 p5; P4 Sec4.1/5.1 Fig12 p8; P5 Sec4.2 Fig2 p3; P6 Sec4.3 Fig6 p7 Table1 p18; P7 Fig4 p6 and Fig5 p7. No estimates digitized from plots.',encoding='utf8')
(B/'slide-plan.txt').write_text('\n'.join(f'{i+1}. {r}' for i,r in enumerate(roles)),encoding='utf8')
