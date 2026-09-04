import fs from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
const b='E:/robosuit/RLPD_论文汇报_20260903/build';
const source='C:/Users/asus/.codex/plugins/cache/openai-primary-runtime/presentations/26.826.12353/skills/presentations/template_following_scripts/check_template_fidelity.mjs';
let code=await fs.readFile(source,'utf8');
code=code.replace('function runCapture(command, args, options = {}) {',`function runCapture(command, args, options = {}) {
  if (command === 'unzip') {
    const py='C:/Users/asus/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
    const pycode=args[0]==='-Z1' ? "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); sys.stdout.buffer.write(chr(10).join(z.namelist()).encode('utf8'))" : "import zipfile,sys; sys.stdout.buffer.write(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]))";
    command=py; args=['-c',pycode,args[1],...(args[0]==='-p'?[args[2]]:[])];
  }`);
await fs.mkdir(b+'/qa',{recursive:true});
const target=b+'/qa/check_template_fidelity_windows.mjs';await fs.writeFile(target,code);
const r=spawnSync(process.execPath,[target,'--workspace',b,'--starter-pptx',b+'/template-starter.pptx','--final-pptx',b+'/../RLPD_利用离线数据提升在线强化学习效率_陈元凯.pptx','--map',b+'/template-frame-map.json','--starter-layout-dir',b+'/template-starter-layout','--final-layout-dir',b+'/final-layout','--edit-dir',b],{encoding:'utf8',maxBuffer:10000000});
console.log(r.stdout,r.stderr);process.exitCode=r.status;
