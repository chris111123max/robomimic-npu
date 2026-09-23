robomimic Remote-Only Development Instructions



Purpose



These instructions define the default workflow for all robomimic / robosuite / NPU tasks in this project.



Codex runs locally on Windows, but the authoritative working project is on the remote server.



Local Codex workspace:



E:\\robosuit\\robomimic-npu



Remote SSH host:



bit-robomimic



Authoritative remote project root:



/data/home/3220251075/lerobot\_workspace/robomimic



Source of truth



The remote project is the ONLY default source of truth.



Unless the user explicitly asks for a local E: operation:



do not inspect local E: source code



do not search local E: source code



do not edit local E: source code



do not test local E: source code



do not infer current implementation details from the local copy



do not silently synchronize remote code with the local copy



All normal robomimic work must be performed on bit-robomimic.



Persistent SSH terminal policy



Efficiency is important.



For each task, prefer a small number of persistent SSH terminals instead of opening a new SSH connection for every command.



When the Codex execution environment supports reusable terminal sessions / PTYs:



Open one primary persistent SSH terminal:

ssh bit-robomimic



Keep that SSH session alive for the task.



In that terminal, enter the project once:

cd /data/home/3220251075/lerobot\_workspace/robomimic



Discover and initialize the existing project runtime environment once when needed.



Reuse the SAME remote terminal for:



ls / find / rg / grep



reading files



searching code



editing files



checking configs



inspecting datasets



inspecting logs



inspecting checkpoints



running short tests



running smoke tests



debugging



inspecting stdout/stderr



rerunning validation



Do NOT repeatedly execute:



ssh bit-robomimic "<command 1>"

ssh bit-robomimic "<command 2>"

ssh bit-robomimic "<command 3>"



when an already-connected persistent remote terminal is available.



Once inside an SSH terminal, execute remote commands directly.

Do not run nested ssh bit-robomimic ... commands from inside the already-connected remote shell.



Recommended fixed terminal layout



Use as few persistent terminals as practical.



Preferred layout:



Terminal 1 — Primary remote shell



keep one persistent ssh bit-robomimic session



use for source inspection, editing, searching, short tests, and debugging



Terminal 2 — Long-running training / evaluation / logs, only when needed



use for training, evaluation, long smoke tests, tail -f, monitoring, or other blocking commands



keep the primary shell free for inspection and debugging



Terminal 3 — Optional



use only when a genuinely independent remote process is useful



do not create extra terminals without a reason



For ordinary tasks, one persistent remote terminal is enough.

For training/debugging tasks with a long-running process, two terminals are usually enough.



Fallback when persistent terminal reuse is unavailable



If the current Codex runtime cannot reliably keep or reuse an interactive SSH terminal:



do not fail the task merely because persistent SSH is unavailable



fall back to non-interactive SSH automatically



combine related remote operations into as few SSH invocations as possible



batch related reads, searches, log checks, and status checks together



avoid one SSH connection per file or per tiny command



Prefer one grouped SSH call containing several related operations over many separate SSH calls.



Example idea:



ssh bit-robomimic "bash -lc '

cd /data/home/3220251075/lerobot\_workspace/robomimic

echo === SEARCH ===

rg <pattern> . || true

echo === CONFIG ===

sed -n "1,220p" <relevant-file>

echo === STATUS ===

<relevant-read-only-check>

'"



Persistent SSH is preferred.

Batched non-interactive SSH is the fallback.



SSH connection behavior



Do NOT run a separate connectivity check before every task.



Do not routinely start with:



ssh bit-robomimic "echo connected"



The first real remote command or the creation of the persistent SSH terminal is itself the connectivity check.



Only diagnose connectivity if the first real SSH operation fails.



When diagnosis is needed, use:



ssh bit-robomimic "echo connected"



If SSH is unavailable:



stop remote work



report the connectivity problem



do not silently fall back to E:



do not pretend the remote project was inspected



do not ask the user to manually establish SSH unless user action is genuinely required



Remote inspection rules



Always inspect the actual remote implementation before making conclusions.



Before modifying code:



Read the relevant remote file.



Read enough surrounding context.



Search related definitions and call sites.



Inspect relevant configs, launch scripts, datasets, tests, metrics, logs, and checkpoints when needed.



Avoid scanning unrelated large datasets or experiment directories without a reason.



Use efficient commands such as:



rg

grep

find

sed

head

tail



Limit output where possible.

Do not dump huge logs, checkpoints, datasets, or entire large files when targeted inspection is sufficient.



Remote editing rules



When a code change is needed, modify the remote file directly.



Preferred editing methods:



controlled remote Python scripts



apply\_patch if available



precise here-documents



other deterministic text edits



Rules:



make the smallest reasonable change



avoid broad ambiguous replacements



do not rewrite entire files unnecessarily



do not overwrite unrelated user changes



do not modify unrelated files without a reason



immediately reread the modified region after editing



verify that the intended change is actually present



Runtime environment



All Python, NPU training, evaluation, tests, and project execution must happen on bit-robomimic.



Do not assume a Conda environment name unless it has been verified from the current remote installation or project configuration.



When a persistent SSH shell is available:



identify the existing runtime environment once



initialize or activate it once



reuse that environment for the remainder of the task



If Conda is required but unavailable in the shell:



inspect the existing Conda installation



locate the existing environment



initialize it correctly



do not install a second Python or Conda distribution merely to solve PATH issues



Debugging loop



For implementation, debugging, repair, or validation tasks:



Reuse the primary persistent SSH terminal when available.



Inspect the actual remote source.



Inspect related configs, callers, training logs, metrics, datasets, checkpoints, or runtime outputs.



Modify the actual remote source.



Reread the modified region.



Run the relevant remote validation or smoke test.



Capture and inspect stdout and stderr.



If it fails or diverges, diagnose the actual runtime evidence.



Modify the remote source again if needed.



Rerun validation.



Continue until the requested behavior is verified or a concrete blocker is identified.



Summarize what changed and what was actually tested.



Never claim that a fix works unless the relevant remote validation has actually completed successfully.



Training and long-running jobs



Training outputs, logs, checkpoints, datasets, metrics, and generated results must be inspected directly on the remote server.



For long-running jobs:



keep the primary SSH shell available when possible



use a second persistent SSH terminal for training, evaluation, or blocking monitoring



use remote log files when appropriate



inspect logs from the primary shell or a dedicated monitoring terminal



do not launch duplicate training jobs accidentally



do not terminate or restart an existing job unless requested or clearly necessary



prefer short smoke tests before launching expensive long training when validating code changes



Project-state safety



Do not delete or overwrite these unless the user explicitly requests it:



datasets



checkpoints



training outputs



experiment directories



environments



large project directories



generated metrics or logs needed for debugging



Do not use destructive Git commands such as:



git reset --hard

git clean -fd

git checkout -- .



unless explicitly requested.



Git synchronization with the local E: copy is not part of the default workflow.



Local E: exception



Only inspect or modify:



E:\\robosuit\\robomimic-npu



when the user explicitly requests local work.



Examples:



修改本地 E 盘代码



查看本地 E 盘文件



检查本地 Git



把远端代码同步到 E 盘



这次只处理本地项目



Without an explicit local request, ignore the local project contents.



Final default behavior



For every normal robomimic task:



REMOTE FIRST, REMOTE ONLY.



Prefer:

one persistent SSH terminal for the whole task



Use:

a second persistent SSH terminal only for long-running training, evaluation, or blocking work



Fallback:

if terminal reuse is unavailable, batch related operations into the minimum practical number of SSH calls



Read remote code through SSH.

Modify remote code through SSH.

Run remote commands through SSH.

Inspect remote results through SSH.



Do not use the local E: project unless the user explicitly asks for it.

