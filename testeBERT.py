# ================================
# 1. Ler o dataset
# ================================

def ler_dados(caminho):
    frases = []
    labels = []

    frase_atual = []
    label_atual = []

    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()

            if linha == "":
                if frase_atual:
                    frases.append(frase_atual)
                    labels.append(label_atual)
                    frase_atual = []
                    label_atual = []
            else:
                palavra, tag = linha.split()
                frase_atual.append(palavra)
                label_atual.append(tag)

    return frases, labels


frases, labels = ler_dados("data.txt")

print("Frases:", frases)
print("Labels:", labels)


# ================================
# 2. Criar mapeamento de labels
# ================================

labels_unicas = list(set(label for frase in labels for label in frase))

label2id = {label: i for i, label in enumerate(labels_unicas)}
id2label = {i: label for label, i in label2id.items()}

print("Mapeamento:", label2id)


# ================================
# 3. Tokenizer
# ================================

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


# ================================
# 4. Alinhar labels com tokens
# ================================

def alinhar_labels(frases, labels, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        frases,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    all_labels = []

    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []

        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]])

            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


dados_tokenizados = alinhar_labels(frases, labels, tokenizer, label2id)

print("Dados tokenizados:", dados_tokenizados)


# ================================
# 5. Carregar modelo BERT
# ================================

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

print("Modelo carregado com sucesso!")

# ================================
# 6. Treinar modelo
# ================================

import torch
from torch.utils.data import DataLoader

# Converter dados para tensores
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {}

        for key, val in self.encodings.items():
            if isinstance(val, list):
                item[key] = torch.tensor(val[idx])
            else:
                item[key] = val[idx].clone().detach()

        return item


dataset = NERDataset(dados_tokenizados)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Otimizador
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Usar GPU se tiver
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Treinamento
model.train()

for epoch in range(3):  # 3 épocas
    print(f"\nEpoch {epoch+1}")

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        print("Loss:", loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print("\nTreinamento finalizado!")

from seqeval.metrics import classification_report

model.eval()

predictions = []
true_labels = []

with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits

        preds = torch.argmax(logits, dim=2)

        for i in range(len(preds)):
            pred_labels = []
            true = []

            for j in range(len(preds[i])):
                if batch["labels"][i][j] != -100:
                    pred_labels.append(id2label[preds[i][j].item()])
                    true.append(id2label[batch["labels"][i][j].item()])

            predictions.append(pred_labels)
            true_labels.append(true)

print("\nRelatório de avaliação:")
print(classification_report(true_labels, predictions))