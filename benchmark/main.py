from g4l.local import LocalEngine
import time

def benchmark_chat_completion(model, message, num_iterations=1):
    engine = LocalEngine(
        gpu_layers = -1,  # use all GPU layers
        cores      = 0        # use 8 CPU cores
    )
    total_tokens = 0
    total_time = 0
    loading_times = []

    for i in range(num_iterations):
        response = engine.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            stream=True
        )

        start = False
        stream_start = time.time()
        start_time = 0
        tokens = 0
    
        for token in response:
            if not start:
                start_time = time.time()
                start = True
            tokens += 1

        elapsed = time.time() - start_time
        loading_time = start_time - stream_start

        total_tokens += tokens
        total_time += elapsed
        loading_times.append(loading_time)

    avg_tokens       = total_tokens / num_iterations
    avg_time         = total_time / num_iterations
    avg_loading_time = sum(loading_times) / num_iterations
    avg_speed        = avg_tokens / avg_time

    print(f"Model                = {model}")
    # print(f"Message              = {message}")
    print(f"Number of iterations = {num_iterations}")
    print(f"Average loading time = {avg_loading_time:.2f}s")
    print(f"Average total tokens = {avg_tokens:.2f}")
    print(f"Average total time   = {avg_time:.2f}s")
    print(f"Average speed        = {avg_speed:.2f} t/s")

# Example usage
benchmark_chat_completion(model='mistral-7b-instruct', message="hey how are you today", num_iterations=5)