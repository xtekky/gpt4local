from g4l.local import LocalEngine
import time

engine = LocalEngine(
    gpu_layers = -1,  # use all GPU layers
    cores      = 0    # use all CPU cores
)

response = engine.chat.completions.create(
    model    = 'mistral-7b-instruct-v0.2.Q4_K_S',
    messages = [{"role": "user", "content": "hey how are you today"}],
    stream   = True
)

start        = False
stream_start = time.time()
start_time   = 0
tokens       = 0

for token in response:
    if start != True:
        start_time = time.time()
    
    start = True
    
    print(token.choices[0].delta.content or "")
    tokens += 1

elapsed = time.time() - start_time

print(f'loading time = {start_time - stream_start}s')
print(f'total tokens = {tokens}')
print(f'total time   = {elapsed}s')
print(f'speed        = {tokens / elapsed}/s ')