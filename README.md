# QA_Independent_study
based on FengZQ's QA system

## 各檔案說明：
- BFS: 為了對BFS的預訓練方法做實驗寫的程式
- data4pretraining: further pretraining所需要的資料
- data4finetuning: finetuning所需要的資料
- pretrain_bienc: 訓練DSSM的程式碼
- saved_model: 訓練完成的模型
- bienc_model: DSSM模型定義
- data_XXX.py: 資料集定義，其中est是cmrc+drcd
- finetuning.py: 在下游任務上微調
- further_pretraining: 對LM做further pretraining的程式碼
- model_MRC.py: LM定義
- preprocess_XXX.ipynb: 把下載下來的資料集處理成json或者pkl
- replace_punc.py: 把中文標點替換成英文標點
- system_establishment.py: 在這個py檔上做過系統架構的實驗
- test_biencoder.py: 測試DSSM效能

pretrain_bienc中:
cut_pretraining_data.py: 用mT5生成QNews資料集

## 環境

> python 3.7.13
> pytorch 1.12.1

## 操作流程：

1. pretrain_bienc/cut_pretraining_data.py 生成資料集
2. pretrain_bienc/filter_data.py 資料清洗
3. pretrain_bienc/pretrain 用1的資料集預訓練DSSM
   > pretrain_bienc\pretrain.py", line 37, in <module>
    model = torch.nn.DataParallel(model, device_ids=[1,3])
    這邊因為我的 device id 是 0 所以我改掉了。
4. further_pretraining.py 進一步預訓練MRC模型
5. finetuning.py 微調MRC模型

