from tokenizers import CharBPETokenizer
import os
import torch
from transformers import RobertaConfig

paths = ['data/train.txt']
# Initialize a tokenizer
tokenizer = CharBPETokenizer(split_on_whitespace_only=True)
# Customize training
tokenizer.train(files=paths, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])
os.makedirs('./tokenizer/Charbpetokenizer', exist_ok=True)
tokenizer.save_model('./tokenizer/Charbpetokenizer')
tokenizer = CharBPETokenizer(
    "./tokenizer/Charbpetokenizer/vocab.json",
    "./tokenizer/Charbpetokenizer/merges.txt",
)


config = RobertaConfig(
    vocab_size=12531,
    max_position_embeddings=130,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./tokenizer/Charbpetokenizer", max_len=128)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
print(model.num_parameters())


from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/train.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
