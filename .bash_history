clear
source vllm_engine/bin/activate
source vllm_engine/bin/activate
/teamspace/studios/this_studio/vllm_engine/bin/python /teamspace/studios/this_studio/chat.py
/teamspace/studios/this_studio/vllm_engine/bin/python /teamspace/studios/this_studio/chat.py
exit
source vllm_engine/bin/activate
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.3   --max-num-batched-tokens 2048   --no-enable-prefix-caching --max-model-len 2048   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001
/teamspace/studios/this_studio/vllm_engine/bin/python /teamspace/studios/this_studio/chat.py
clear
exit
exit
exit
/teamspace/studios/this_studio/vllm_engine/bin/python /teamspace/studios/this_studio/chat.py
exit
exit
source vllm_engine/bin/activate
vllm serve PaddlePaddle/PaddleOCR-VL   --trust-remote-code   --max-num-batched-tokens 16384   --no-enable-prefix-caching   --mm-processor-cache-gb 0   --port 8000
 --gpu-memory-utilization 0.5
vllm serve PaddlePaddle/PaddleOCR-VL   --trust-remote-code  --gpu-memory-utilization 0.5   --max-num-batched-tokens 16384   --no-enable-prefix-caching   --mm-processor-cache-gb 0   --port 8000
vllm serve PaddlePaddle/PaddleOCR-VL   --trust-remote-code  --gpu-memory-utilization 0.5 --max-model-len 4096   --max-num-batched-tokens 16384   --no-enable-prefix-caching   --mm-processor-cache-gb 0   --port 8000
vllm serve PaddlePaddle/PaddleOCR-VL   --trust-remote-code  --gpu-memory-utilization 0.4 --max-model-len 4096   --max-num-batched-tokens 2048   --no-enable-prefix-caching   --mm-processor-cache-gb 0   --port 8000
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.3   --max-num-batched-tokens 2048   --no-enable-prefix-caching --max-model-len 2048   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001
clear
clear
vllm serve PaddlePaddle/PaddleOCR-VL   --trust-remote-code  --gpu-memory-utilization 0.4 --max-model-len 8192   --max-num-batched-tokens 2048   --no-enable-prefix-caching   --mm-processor-cache-gb 0   --port 8000
vllm serve PaddlePaddle/PaddleOCR-VL   --trust-remote-code  --gpu-memory-utilization 0.4 --max-model-len 8192   --max-num-batched-tokens 2048   --no-enable-prefix-caching   --mm-processor-cache-gb 0   --port 8000
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.3   --max-num-batched-tokens 2048   --no-enable-prefix-caching --max-model-len 2048   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001
source vllm_engine/bin/activate
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.3   --max-num-batched-tokens 2048   --no-enable-prefix-caching --max-model-len 2048   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.2   --max-num-batched-tokens 2048   --no-enable-prefix-caching --max-model-len 2048   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001
exit
source vllm_engine/bin/activate
nvidia-smi
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.3   --max-num-batched-tokens 2048   --no-enable-prefix-caching --max-model-len 2048   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.3   --max-num-batched-tokens 2048   --no-enable-prefix-caching   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001
clear
nvidia-smi
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --max-num-batched-tokens 2048   --no-enable-prefix-caching --max-model-len 2048   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.2   --max-num-batched-tokens 512   --no-enable-prefix-caching --max-model-len 1024   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001 
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.9   --max-num-batched-tokens 512   --no-enable-prefix-caching --max-model-len 1024   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001 
clear
nvidia-smi
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.25   --max-num-batched-tokens 512   --no-enable-prefix-caching --max-model-len 1024   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001 
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.225   --max-num-batched-tokens 512   --no-enable-prefix-caching --max-model-len 1024   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001 
clear
vllm serve PaddlePaddle/PaddleOCR-VL   --trust-remote-code   --gpu-memory-utilization 0.4
  --max-model-len 8096
  --max-num-batched-tokens 2048   --no-enable-prefix-caching   --mm-processor-cache-gb 0   --port 8000
clear
source vllm_engine/bin/activate
clear
nvidia-smi
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.25   --max-num-batched-tokens 512   --no-enable-prefix-caching --max-model-len 1024   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001 
vllm serve PaddlePaddle/PaddleOCR-VL   --trust-remote-code   --gpu-memory-utilization 0.4
source vllm_engine/bin/activate
source vllm_engine/bin/activate
vllm serve PaddlePaddle/PaddleOCR-VL   --trust-remote-code   --gpu-memory-utilization 0.4
  --max-model-len 8096
  --max-num-batched-tokens 2048   --no-enable-prefix-caching   --mm-processor-cache-gb 0   --port 8000
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.25   --max-num-batched-tokens 512   --no-enable-prefix-caching --max-model-len 1024   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001
exit
exit
source vllm_engine/bin/activate
source vllm_engine/bin/activate
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.25   --max-num-batched-tokens 512   --no-enable-prefix-caching --max-model-len 1024   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001
vllm serve PaddlePaddle/PaddleOCR-VL   --trust-remote-code   --gpu-memory-utilization 0.4
  --max-model-len 8096
  --max-num-batched-tokens 2048   --no-enable-prefix-caching   --mm-processor-cache-gb 0   --port 8000
clear
clear
vllm serve PaddlePaddle/PaddleOCR-VL   --trust-remote-code   --gpu-memory-utilization 0.4
vllm serve Qwen/Qwen2.5-1.5B   --trust-remote-code   --gpu-memory-utilization 0.25   --max-num-batched-tokens 512   --no-enable-prefix-caching --max-model-len 1024   --mm-processor-cache-gb 0   --served-model-name Qwen2.5-1.5B   --port 8001
gemini
gemini
clear
gemini
exit
gemini
clear
exit
exit
gemini
git status
pwd
git init
touch .gitignore
git add .
gemini
clear
exit
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4
--max-model-len 8096
--max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
gemini
source vllm_engine/bin/activate
source vllm_engine/bin/activate
exit
exit
python app.py 
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4
--max-model-len 8096
--max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4
--max-model-len 8096
--max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
gemini
clear
python app.py 
clear
gemini
git status
git config user.email "nalanlearnings@gmail.com"
git add -f requirements.txt
git add README.md app.py .gitignore static/
git add *.py *.ipynb .idea/ .vscode/
git status
git commit -m "Initial Commit"
git branch -M main
git status
git remote add origin https://github.com/Nalan11/in-paper.git
git push -u origin main
git add .
git status
git commit -m "initial commit"
git config user.email "nalanlearnings@gmail.com"
git config user.name "Nalan11"
git commit -m "initial commit"
git config --global user.name "Your Name"
git branch -M main
git status
git push -u origin main
exit
gemini
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 1024 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 8096 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
python app.py 
git status
source vllm_engine/bin/activate
source vllm_engine/bin/activate
python app.py 
exit
pip install seaborn --dry-run
uv pip install seaborn --dry-run
uv pip install seaborn
exit
uv pip install beautifulsoup4 --dry-run
gemini
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 1024 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4
gemini
exit
gemini
clear
gemini
clea
clear
exit
gemini
source vllm_engine/bin/activate
source vllm_engine/bin/activate
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 1024 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
exit
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4
--max-model-len 8096
--max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
exit
source vllm_engine/bin/activate
source vllm_engine/bin/activate
python app2.py
python app2.py
python app2.py
gemini
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 2048 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 8096 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
python app2.py
gemini
exit
gemini
gemini
exit
gemini
source vllm_engine/bin/activate
source vllm_engine/bin/activate
python  app2.py 
source vllm_engine/bin/activate
python  app2.py 
exit
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 1024 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
clear
python app2.py
python --version
python app.py
python app2.py
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 1024 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 8096 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
python app.py
python app.py
python app2.py
python app.py
exit
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 1024 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 2048 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
python app2.py
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 8096 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
gemini
git status
git add .
git status
git commit -m "v1.2 AppV2"
git push origin main
clear
gemini
exit
gemini
gemini
clear
gemini
gemini
exit
gemini
exit
source vllm_engine/bin/activate
source vllm_engine/bin/activate
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 2048 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 8096 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
clear
clear
exit
exit
app2.py
clear
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 2048 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 4096 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
CLEAR
clear
python app2.py
clear
vllm serve Qwen/Qwen2.5-3B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 4096 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-3B --port 8001
python app2.py
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 8096 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
gemini
gemini
exit
source vllm_engine/bin/activate
start_servers.sh
ps aux | grep vllm
clear
exit
source vllm_engine/bin/activate
source vllm_engine/bin/activate
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 8096 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
python3 -m src.api.main
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 2048 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
exit
exit
python3 -m src.api.main
exit
uv pip install requirements.txt --dry-run
uv pip install requests psutil --dry-run
uv pip install psutil requests
python3 -m src.api.main
source vllm_engine/bin/activate
python3 -m src.api.main
clear
exit
python3 -m src.api.main
exit
gemini
exit
gemini
exit
python3 -m src.api.main
python3 -m src.api.main
clear
python3 -m src.api.main
clear
source vllm_engine/bin/activate
source vllm_engine/bin/activate
python3 -m src.api.main
clear
exit
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.25 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 2048 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.3 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 4096 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.3 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 4096 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 8096 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.3 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
python3 -m src.api.main
gemini
exit
python3 -m src.api.main
eixt
exit
source vllm_engine/bin/activate
source vllm_engine/bin/activate
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 8192 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
EXIT
exit
source vllm_engine/bin/activate
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.30 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
gemini
clear
python3 -m src.api.main
exit
python3 -m src.api.main
clear
gemini
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
exit
gemini
exit
source vllm_engine/bin/activate
source vllm_engine/bin/activate
python3 -m src.api.main
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
clear
exi
exit
source vllm_engine/bin/activate
source vllm_engine/bin/activate
clear
clear
python3 -m src.api.main
clear
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.30 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.30 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 30000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
c
clear
python3 -m src.api.main
clear
python3 -m src.api.main
clear
python3 -m src.api.main
clear
python3 -m src.api.main
python3 -m src.api.main
gemini
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.30 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
exit
source vllm_engine/bin/activate
source vllm_engine/bin/activate
python3 -m src.api.main
python3 -m src.api.main
exit
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.30 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
clear
vllm serve Qwen/Qwen3-4B-AWQ   --trust-remote-code   --quantization awq   --gpu-memory-utilization 0.3   --max-model-len 8000   --served-model-name Qwen3-4B-AWQ   --port 8001
vllm serve Qwen/Qwen3-4B-AWQ   --trust-remote-code   --quantization awq   --gpu-memory-utilization 0.4   --max-model-len 8000   --served-model-name Qwen3-4B-AWQ   --port 8001
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
python3 -m src.api.main
clear
python3 -m src.api.main
clear
clear
vllm chat --url http://localhost:8001/v1 --quick "What id docker containers"
source vllm_engine/bin/activate
vllm chat --url http://localhost:8001/v1 --quick "What is docker containers"
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.3 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
source vllm_engine/bin/activate
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
clear
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.30 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
clear
vllm chat --url http://localhost:8002/v1 --quick "What is docker containers"
vllm chat --url http://localhost:8002/v1 --quick "What is docker containers"
vllm chat http://localhost:8002/v1 --quick "What is docker containers"
vllm chat --url http://localhost:8002/v1 --quick "Hello"
vllm chat --url http://localhost:8002/v1 --quick "What is your role"
vllm chat --url http://localhost:8002/v1 --quick "Your name is Jackson. What is you name?"
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.30 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8002
vllm serve Qwen/Qwen3-4B-AWQ   --trust-remote-code   --quantization awq   --gpu-memory-utilization 0.4   --max-model-len 8000   --served-model-name Qwen3-4B-AWQ   --port 8001
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.30 --max-num-batched-tokens 1024 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8002
vllm chat --url http://localhost:8002/v1 --quick "Your name is Jackson. What is you name?"
vllm chat --url http://localhost:8002/v1 --quick "Your name is Jackson. What is you name?"
vllm chat --url http://localhost:8002/v1 --quick "What is the capital of America"
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.30 --max-num-batched-tokens 1024 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8002
vllm chat --url http://localhost:8001/v1 --quick "What is the capital of America"
vllm chat --url http://localhost:8000/v1 --quick "What is the capital of America"
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.30 --max-num-batched-tokens 1024 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8000
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --enforce-eager~ --gpu-memory-utilization 0.30 --max-num-batched-tokens 1024 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8000
clear
vllm chat --url http://localhost:8000/v1 --quick "What is the capital of America"
vllm chat --url http://localhost:8000/v1 --quick "Fuck you"
uv pip install "huggingface_hub[cli]" --dry-run
uv pip install "huggingface_hub[cli]"
huggingface-cli scan-cache
huggingface-cli delete-cache
huggingface-cli delete-cache
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --enforce-eager --gpu-memory-utilization 0.30 --max-num-batched-tokens 1024 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8000
clear
vllm chat --url http://localhost:8000/v1 --quick "Hi"
vllm chat --url http://localhost:8001/v1 --quick "Hi"
vllm chat --url http://localhost:8001/v1 --quick "What is docker containers?"
python3 -m src.api.main
clear
vllm chat --url http://localhost:8001/v1 --quick "What is the tallest building?"
vllm chat --url http://localhost:8001/v1 "What is the tallest building?"
vllm chat --help
python3 -m src.api.main
clear
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.30 --max-num-batched-tokens 1024 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen2.5-1.5B --port 8001
vllm chat --url http://localhost:8002/v1 --quick "What is the tallest building? Do not think"
vllm serve Qwen/Qwen3-4B-AWQ   --trust-remote-code   --gpu-memory-utilization 0.4   --max-model-len 8000   --served-model-name Qwen3-4B-AWQ   --port 8002
python3 -m src.api.main
clear
clear
gemini
python3 -m src.api.main
clear
vllm serve Qwen/Qwen3-4B-AWQ   --trust-remote-code   --gpu-memory-utilization 0.4   --max-model-len 8000   --served-model-name Qwen3-4B-AWQ   --port 8002
clear
python3 -m src.api.main
gemini
vllm serve Qwen/Qwen3-4B-AWQ   --trust-remote-code   --gpu-memory-utilization 0.4   --max-model-len 8000   --served-model-name Qwen3-4B-AWQ   --port 8001
xit
exit
source vllm_engine/bin/activate
source vllm_engine/bin/activate
vllm serve Qwen/Qwen3-4B-AWQ --trust-remote-code --gpu-memory-utilization 0.40 --max-num-batched-tokens 1024 --max-model-len 8000 --served-model-name Qwen3-4B-AWQ --port 8001
exit
source vllm_engine/bin/activate
source vllm_engine/bin/activate
vllm serve Qwen/Qwen3-4B-AWQ --trust-remote-code --gpu-memory-utilization 0.40 --max-num-batched-tokens 1024 --max-model-len 8000 --served-model-name Qwen3-4B-AWQ --port 8001
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
vllm serve Qwen/Qwen3-4B-AWQ --trust-remote-code --gpu-memory-utilization 0.40 --max-num-batched-tokens 1024 --max-model-len 8000 --served-model-name Qwen3-4B-AWQ --port 8001
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.3 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
clear
clear
/home/zeus/miniconda3/envs/cloudspace/bin/python /teamspace/studios/this_studio/chat.py
clear
python3 -m src.api.main
clear
uv pip install sse-starlette  --dry-run
python3 -m src.api.main
clear
python3 -m src.api.main
clear
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.3 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
python3 -m src.api.main
gemini
gemini
clear
exit
source vllm_engine/bin/activate
python chat.py 
clear
python chat.py 
python chat.py 
python chat.py 
clear
source vllm_engine/bin/activate
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
clear
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.4 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
clear
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.30 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
EXIT
exit
source vllm_engine/bin/activate
source vllm_engine/bin/activate
vllm serve Qwen/Qwen3-4B-AWQ --trust-remote-code --gpu-memory-utilization 0.40 --max-num-batched-tokens 1024 --max-model-len 8000 --served-model-name Qwen3-4B-AWQ --port 8001
vllm serve Qwen/Qwen3-4B-AWQ --trust-remote-code --gpu-memory-utilization 0.40 --max-model-len 8000 --served-model-name Qwen3-4B-AWQ --port 8001
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.30 --max-model-len 32768 --port 8000
clear
python chat.py 
vllm serve Qwen/Qwen3-4B-AWQ --trust-remote-code --gpu-memory-utilization 0.80 --max-model-len 8000 --served-model-name Qwen3-4B-AWQ --port 8001
vllm serve Qwen/Qwen2.5-1.5B --trust-remote-code --gpu-memory-utilization 0.4 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen3-4B-AWQ --port 8001
clear
vllm serve Qwen/Qwen3-4B-AWQ --trust-remote-code --gpu-memory-utilization 0.4 --max-num-batched-tokens 512 --no-enable-prefix-caching --max-model-len 8000 --mm-processor-cache-gb 0 --served-model-name Qwen3-4B-AWQ --port 8001
vllm serve Qwen/Qwen3-4B-AWQ   --trust-remote-code   --quantization awq   --gpu-memory-utilization 0.40   --max-model-len 8000   --served-model-name Qwen3-4B-AWQ   --port 8001
hf --help
hf delete-cache
huggingface-cli delete-cache
huggingface-cli delete-cache
python test_pipeline_llm.py 
python3 -m src.api.main
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.30 --max-model-len 32768 --port 8000
gemini
exit
gemini
clear
gemini
exit
gemini
source vllm_engine/bin/activate
source vllm_engine/bin/activate
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.30 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
clear
python3 -m src.api.main
clear
python3 -m src.api.main
clear
python3 -m src.api.main
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code --gpu-memory-utilization 0.30 --max-model-len 32768 --max-num-batched-tokens 2048 --no-enable-prefix-caching --mm-processor-cache-gb 0 --port 8000
exit
gemini
exit
source vllm_engine/bin/activate
source vllm_engine/bin/activate
git status
