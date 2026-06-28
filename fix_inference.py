import re
path = "local_voice_ai/agent.py"
with open(path, 'r') as f:
    content = f.read()
new_content = content.replace(
    'def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()',
    'def prewarm(proc: JobProcess) -> None:
    proc.userdata["inference"] = False
    proc.userdata["vad"] = silero.VAD.load()'
)
with open(path, 'w') as f:
    f.write(new_content)
print("Fix applied")