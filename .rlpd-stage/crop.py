from pathlib import Path
import json,pdfplumber
B=Path('E:/robosuit/RLPD_论文汇报_20260903/build')
jobs=[(1,'fig1',(332,171,518,316)),(3,'fig2',(309,67,543,129)),(6,'fig4',(65,65,532,166)),(6,'fig4-plots',(65,76,532,156)),(7,'fig5',(58,62,291,136)),(7,'fig6',(192,264,287,357)),(8,'fig12',(309,322,544,397))]
with pdfplumber.open(r'F:\桌面\自控ppt\论文\IL+RL\RLPD.pdf') as pdf:
    for n,name,box in jobs:
        pdf.pages[n-1].crop(box).to_image(resolution=350).original.save(B/'assets'/f'{name}.png')
        print(name,flush=True)
(B/'assets/crops.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2),encoding='utf8')
