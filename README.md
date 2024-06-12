# 你的表情，我猜对了吗？

文件目录存放如下：

```
.
|   
+---cache
|   +---.locks
|   |   \---models--bert-base-chinese
|   \---models--bert-base-chinese
|       +---.no_exist
|       |   \---c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f
|       |           added_tokens.json
|       |           special_tokens_map.json
|       |           
|       +---blobs
|       +---refs
|       |       main
|       |       
|       \---snapshots
|           \---c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f
|                   config.json
|                   model.safetensors
|                   tokenizer.json
|                   tokenizer_config.json
|                   vocab.txt
|                   
+---code
|   |   data_utils.py
|   |   emoji_matching.py
|   |   model.py
|   |   train.py
|   |   
|   +---cache
|   |   +---.locks
|   |   |   \---models--bert-base-chinese
|   |   \---models--bert-base-chinese
|   |       +---.no_exist
|   |       |   \---c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f
|   |       |           added_tokens.json
|   |       |           special_tokens_map.json
|   |       |           
|   |       +---blobs
|   |       +---refs
|   |       |       main
|   |       |       
|   |       \---snapshots
|   |           \---c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f
|   |                   config.json
|   |                   model.safetensors
|   |                   tokenizer.json
|   |                   tokenizer_config.json
|   |                   vocab.txt
|   |                   
|   \---__pycache__
|           data_utils.cpython-39.pyc
|           inference.cpython-39.pyc
|           model.cpython-39.pyc
|           
+---emo-visual-data
|   |   data.json
|   |   
|   \---emo
|           000317dc-9047-4d68-bb55-e40c09ed0f9a.jpg
|           0005fce3-aefd-4694-bb94-55fbe56d0793.jpg
|           000ba939-ccf5-4071-af24-09c1d8829b0d.jpg
|           ...
|           
+---img
|       1.jpg
|       2.jpg
|       3.jpg
|       4.jpg
|       
\---model
        model_weights.pth
```