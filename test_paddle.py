import paddle
print("Paddle version:", paddle.__version__)
print("Is CUDA available:", paddle.is_compiled_with_cuda())
print("Number of GPUs:", paddle.device.get_device())
try:
    paddle.utils.run_check()
except Exception as e:
    print("Paddle check failed:", e)
