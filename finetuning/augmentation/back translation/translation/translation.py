from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

class contentDataset(Dataset):
    def __init__(self, file, tok, max_len, pad_index=None):
        super().__init__()
        self.tok =tok
        self.max_len = max_len
        self.content = pd.read_csv(file)
        self.len = self.content.shape[0]
        self.pad_index = self.tok.pad_token
    
    def add_padding_data(self, inputs, max_len):
        if len(inputs) < max_len:
            # pad = np.array([self.pad_index] * (max_len - len(inputs)))
            pad = np.array([0] * (max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
            return inputs
        else:
            inputs = inputs[:max_len]
            return inputs
    
    def __getitem__(self,idx):
        instance = self.content.iloc[idx]
        # text = "[CLS]" + instance['content'] + "[SEP]"
        text = instance['text']
        input_ids = self.tok.encode(text)
        
        input_ids = self.add_padding_data(input_ids, max_len=self.max_len)
        label_ids = instance['label']
        # encoder_attention_mask = input_ids.ne(0).float()
        return {"encoder_input_ids" : np.array(input_ids, dtype=np.int_),
                "label" : np.array(label_ids,dtype=np.int_)}
        
    def __len__(self):
        return self.len


def main(model, tokenizer, device, path, lang_type, query_num):

    name = path.split('/')[-1].replace('.csv','')

    setup = contentDataset(file = path ,tok = tokenizer, max_len = 128)
    
    dataloader = DataLoader(setup, batch_size= 40, shuffle=False)
    model.eval()

    translated_text = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}


        encoder_attention_mask = batch["encoder_input_ids"].ne(0).float().to(device)
        with torch.no_grad():            
            generated_tokens = model.generate(
                            batch['encoder_input_ids'],attention_mask=encoder_attention_mask,
                            forced_bos_token_id=tokenizer.lang_code_to_id[lang_type]
                            ).to(device)
        result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for text in result:
            translated_text.append(text)
        
    translated_df = pd.DataFrame(translated_text,columns=['text'])
    translated_df.to_csv(f"data_{query_num}/bt/{name}_{lang_type}.csv")

if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model.to(device)

    # korean: ko_KR
    # spanish: es_XX
    # portugess: pt_XX
    # Chinese: zh_CN

    # Arabic: ar_AR
    # france: fr_XX
    # germany: de_DE

    # japan: ja_XX
    # Italian: it_IT
    # Russian: ru_RU

    lang_type_list = ['ko_KR', 'es_XX', "pt_XX", "zh_CN", "ar_AR", "fr_XX", "de_DE", "ja_XX", "it_IT", "ru_RU"]
    data_path_list = ["business", "sports", "sci_tech", "world"]
    query_num = 5 # 25, 50, 100

    for lang_type in lang_type_list:
        # lang_type = "ko_KR"
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", max_length=128, truncation=True)
        tokenizer.src_lang = "en_XX"
        
        for data_path in data_path_list:
            path = f"data_{query_num}/{data_path}.csv"
        
            main(model, tokenizer, device, path, lang_type, query_num)