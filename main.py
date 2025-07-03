import pandas as pd

# Example loading
users_df = pd.read_csv("users.csv")         # user_id, gender, department_name
items_df = pd.read_csv("items.csv")         # item_id, item_name, item_category
ratings_df = pd.read_csv("ratings.csv")     # user_id, item_id, interaction (e.g., view, search)
from sklearn.preprocessing import LabelEncoder

# Encode user fields
gender_encoder = LabelEncoder().fit(users_df["gender"])
dept_encoder = LabelEncoder().fit(users_df["department_name"])
users_df["gender_enc"] = gender_encoder.transform(users_df["gender"])
users_df["dept_enc"] = dept_encoder.transform(users_df["department_name"])

# Encode item fields
cat_encoder = LabelEncoder().fit(items_df["item_category"])
items_df["cat_enc"] = cat_encoder.transform(items_df["item_category"])
from sklearn.preprocessing import LabelEncoder

# Encode user fields
gender_encoder = LabelEncoder().fit(users_df["gender"])
dept_encoder = LabelEncoder().fit(users_df["department_name"])
users_df["gender_enc"] = gender_encoder.transform(users_df["gender"])
users_df["dept_enc"] = dept_encoder.transform(users_df["department_name"])
from sentence_transformers import SentenceTransformer
import numpy as np

text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Encode item names
items_df["item_text_emb"] = list(text_encoder.encode(items_df["item_name"].tolist(), show_progress_bar=True))

# Encode department names
users_df["dept_text_emb"] = list(text_encoder.encode(users_df["department_name"].tolist(), show_progress_bar=True))

# Encode item fields
cat_encoder = LabelEncoder().fit(items_df["item_category"])
items_df["cat_enc"] = cat_encoder.transform(items_df["item_category"])
import torch
from torch.utils.data import Dataset, DataLoader

class RecSysDataset(Dataset):
    def __init__(self, ratings, users, items):
        self.data = ratings
        self.users = users.set_index("user_id")
        self.items = items.set_index("item_id")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user = self.users.loc[row["user_id"]]
        item = self.items.loc[row["item_id"]]
        
        return {
            "user_id": row["user_id"],
            "item_id": row["item_id"],
            "gender": user["gender_enc"],
            "dept_text_emb": torch.tensor(user["dept_text_emb"], dtype=torch.float32),
            "category": item["cat_enc"],
            "item_text_emb": torch.tensor(item["item_text_emb"], dtype=torch.float32),
        }


train_dataset = RecSysDataset(ratings_df, users_df, items_df)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerWithTextModel(nn.Module):
    def __init__(self, num_genders, num_cats, text_emb_dim=384, emb_dim=32):
        super().__init__()

        self.gender_emb = nn.Embedding(num_genders, emb_dim)
        self.cat_emb = nn.Embedding(num_cats, emb_dim)

        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim + text_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim + text_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, gender, dept_text_emb, category, item_text_emb):
        gender_vec = self.gender_emb(gender)
        cat_vec = self.cat_emb(category)

        user_input = torch.cat([gender_vec, dept_text_emb], dim=1)
        item_input = torch.cat([cat_vec, item_text_emb], dim=1)

        user_emb = self.user_mlp(user_input)
        item_emb = self.item_mlp(item_input)

        return user_emb, item_emb

    def predict_score(self, user_emb, item_emb):
        return (user_emb * item_emb).sum(dim=1)

model = TwoTowerModel(
    num_genders=len(gender_encoder.classes_),
    num_depts=len(dept_encoder.classes_),
    num_cats=len(cat_encoder.classes_)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()  # binary classification

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(10):
    model.train()
    total_loss = 0

    for batch in train_loader:
        gender = batch["gender"].to(device)
        dept = batch["dept"].to(device)
        cat = batch["category"].to(device)

        user_emb, item_emb = model(gender, dept, cat)
        scores = model.predict_score(user_emb, item_emb)

        # Implicit feedback = positive (1), generate negatives randomly
        labels = torch.ones_like(scores).to(device)

        loss = criterion(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
